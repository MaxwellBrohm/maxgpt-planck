"""stream_core.py: a worker process that dies, and the prefetch window (stream_fixtures.py)."""
import json
import os
import time

import pytest

import extract
import stream_core as C
import stream_tools as T
import stream_fetch as F
import stream_work as W
import stream_fixtures as X
from test_stream_core import cfg, events


@pytest.fixture(autouse=True)
def plenty_of_disk(monkeypatch):
    monkeypatch.setattr(F, "disk_free", lambda p: 10 ** 13)


def test_a_dying_worker_fails_only_its_own_file(tmp_path, monkeypatch):
    """cccc file 1's worker dies while file 0 is mid-run in the other worker; file 0 must be
    re-run and committed. Two Gutenberg files come first so both worker processes exist before
    the cccc chunk: CPython 3.12's executor wakes its manager thread before it spawns a worker on
    demand, so the death of a just-spawned worker can go unseen until another result arrives,
    and file 0 would then finish in the first pass (still correct, but the retry is untested)."""
    root = str(tmp_path)
    X.make_world(root, gutenberg_files=2)
    monkeypatch.setenv("STREAM_TEST_SYNC", root)
    c = cfg(root, workers=2, batch_wait=5, readers={"cccc": "stream_fixtures:reader_that_dies"})
    r = C.run(c)
    chunks = {e["fid"]: e["chunk_files"] for e in events(c["out"], "file_done")}
    assert chunks["common-pile/cccc/" + X.CCCC[0]] == 2        # died together with file 1
    assert r["reason"] == "failed" and r["failed"] == ["common-pile/cccc/" + X.CCCC[1]]
    assert "worker process died" in events(c["out"], "file_failed")[0]["error"]
    assert len(events(c["out"], "file_done")) == 5
    assert X.output_ids(c["out"], c["codec"]) == X.oracle(root, c["manifest"], skip={X.CCCC[1]})


def test_prefetch_window_holds_one_raw_file_at_a_time(tmp_path, monkeypatch):
    root = str(tmp_path)
    X.make_world(root)
    seen, real_fetch, real_extract, holder = [], F.fetch, W.extract_file, {}
    real_start = F.Prefetcher.start

    def start(self):
        holder["pf"] = self
        real_start(self)

    def fetch(item, cfg, procs, stop):
        others = [x for x in holder["pf"].items if x is not item and not x["local"]
                  and x["state"] in ("run", "ready")]
        seen.append(len(others))
        return real_fetch(item, cfg, procs, stop)

    def slow_extract(job):
        time.sleep(0.2)
        return real_extract(job)

    monkeypatch.setattr(F.Prefetcher, "start", start)
    monkeypatch.setattr(F, "fetch", fetch)
    monkeypatch.setattr(W, "extract_file", slow_extract)
    c = cfg(root, dl_threads=1, window_gb=1 / F.GIB)          # a one-byte window
    assert C.run(c)["reason"] == "done"
    assert len(seen) == 5 and seen == [0] * 5
    assert X.output_ids(c["out"], c["codec"]) == X.oracle(root, c["manifest"])


def test_match_selects_files_by_path_substring(tmp_path):
    root = str(tmp_path)
    X.make_world(root)
    got = T.dry_run(cfg(root, match=["2020-10", "dolly"]))["by_source"]
    assert {s: b["files"] for s, b in got.items()} == {"dolly": 1, "cccc": 1}
    r = C.run(cfg(root, match=["2020-10"]))
    assert r["files"] == 1 and [e["fid"] for e in events(cfg(root)["out"], "file_done")] == [
        "common-pile/cccc/" + X.CCCC[2]]


@pytest.mark.parametrize("busy", [False, True])
def test_workers_are_released_before_waiting_on_a_busy_lock(tmp_path, monkeypatch, busy):
    import fcntl
    import stream_loop
    import stream_lock
    root = str(tmp_path)
    X.make_world(root)
    c = cfg(root, workers=2)
    fd = os.open(c["lock_file"], os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    assert not stream_lock.lock_free_now(c["lock_file"])
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    assert stream_lock.lock_free_now(c["lock_file"])
    made, checks, real = [], [], stream_loop._executor
    monkeypatch.setattr(stream_loop, "lock_free_now", lambda p: checks.append(p) or not busy)
    monkeypatch.setattr(stream_loop, "_executor", lambda cfg, n: made.append(n) or real(cfg, n))
    assert C.run(c)["reason"] == "done"
    assert len(checks) >= 2                   # asked before every chunk after the first
    assert len(made) == (len(checks) + 1 if busy else 1)


def test_a_code_change_under_the_run_stops_it_cleanly(tmp_path, monkeypatch):
    import extract
    root = str(tmp_path)
    X.make_world(root)
    real, calls = extract.code_hashes, {"n": 0}

    def drifting():
        calls["n"] += 1
        h = real()
        if calls["n"] > 1:                         # after run_start: one file differs
            h["stream_work.py"] = "0" * 64
        return h

    monkeypatch.setattr(extract, "code_hashes", drifting)
    c = cfg(root)
    assert C.run(c)["reason"] == "code_changed"
    assert events(c["out"], "file_done") == []
    monkeypatch.setattr(extract, "code_hashes", real)
    assert C.run(c)["reason"] == "done"


def test_cccc_needs_the_explicit_unrecorded_license_allow(tmp_path):
    root = str(tmp_path)
    X.make_world(root)
    with pytest.raises(SystemExit, match="allow-unrecorded-license cccc"):
        C.run(cfg(root, allow_unrecorded_license=[]))
    e = [x for x in json.load(open(cfg(root)["manifest"]))["files"] if x["source"] == "cccc"][0]
    job = dict(reader="readers_core:read_core", source="cccc", path=X.raw_path(root, e),
               rel="x", tmp=str(tmp_path / "t"), extra_meta={}, floor=0, sidecars={},
               allow_unrecorded=[], o=dict(extract.DEFAULTS, scratch_dir=str(tmp_path / "s")))
    res = W.extract_file(job)                      # the worker gate, on its own
    ok = W.extract_file(dict(job, allow_unrecorded=["cccc"], tmp=str(tmp_path / "t2")))
    assert res["docs"] == 0 and ok["docs"] > 0
    assert res["stats"]["dropped"]["license_unrecorded"]["docs"] == ok["docs"]
