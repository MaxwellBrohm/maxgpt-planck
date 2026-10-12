"""stream_core.py on fake files (stream_fixtures.py): end to end, resume, delete, disk guard,
ledger crash recovery, the heavy lock. Nothing here touches the network or the real lock file."""
import fcntl
import json
import os
from collections import Counter

import pytest

import stream_core as C
import stream_tools as T
import stream_fetch as F
import stream_io as SIO
import stream_ledger
import stream_work as W
import stream_fixtures as X

CODECS = [c for c in ("gz", "zst") if SIO.codec_ok(c)]
LIMIT = 3000


@pytest.fixture(autouse=True)
def plenty_of_disk(monkeypatch):
    monkeypatch.setattr(F, "disk_free", lambda p: 10 ** 13)


def cfg(root, **kw):
    c = dict(manifest=os.path.join(root, "manifest.json"), out=os.path.join(root, "out"),
             raw=os.path.join(root, "raw"), starter=os.path.join(root, "starter"),
             lock_file=os.path.join(root, "heavy.lock"), workers=1, dl_threads=2,
             shard_bytes=LIMIT, codec=CODECS[0], retry_sleep=0, batch_wait=0, blocked_grace=0,
             nice=0, lock_timeout=30, allow_unrecorded_license=["cccc"])
    c.update(kw)
    return c


def events(out, name):
    return [e for e in X.ledger(out) if e["event"] == name]


def raw_files(root):
    return sorted(os.path.relpath(os.path.join(d, f), root) for d, _, fs in
                  os.walk(os.path.join(root, "raw")) for f in fs)


@pytest.mark.parametrize("codec", CODECS)
def test_end_to_end_dedup_shards_delete_and_idempotent_restart(tmp_path, codec):
    root = str(tmp_path)
    X.make_world(root)
    c = cfg(root, codec=codec, workers=2)
    r = C.run(c)
    assert r["reason"] == "done" and r["files"] == 5
    out = c["out"]
    want = X.oracle(root, c["manifest"])
    assert X.output_ids(out, codec) == want
    blob = json.dumps(want)
    assert "cccc:dupB" not in blob and "cccc:w2" not in blob and "cccc:cross" not in blob
    assert "cccc:dupA" in blob and "cccc:w1" in blob and "gutenberg:10" in blob
    assert blob.count("cccc:same-id") == 1
    done = events(out, "file_done")
    assert len(done) == 5
    drops = Counter()
    for e in done:
        drops.update({k: v["docs"] for k, v in e["stats"]["dropped"].items()})
    assert drops["dup_exact"] == 3 and drops["dup_id"] == 1 and drops["date_gate"] == 1
    assert any(len({s["shard"] for s in e["segments"]}) > 1 for e in done)   # spans shards
    sealed = events(out, "shard_sealed")
    for s in sealed:                        # every shard is sealed at the end, hashes match
        assert SIO.sha256_file(os.path.join(out, s["shard"])) == s["sha256"]
        assert SIO.sha256_file(os.path.join(out, s["keys"])) == s["keys_sha256"]
    for src in want:
        mine = sorted((s for s in sealed if s["source"] == src), key=lambda s: s["shard"])
        assert len(mine) == len(X.shard_paths(out, src, codec))
        assert all(s["ubytes"] >= LIMIT for s in mine[:-1])
    assert raw_files(root) == []                            # every downloaded raw file deleted
    assert len(events(out, "raw_deleted")) == 4             # the starter-local file is not
    dolly = json.load(open(c["manifest"]))["files"][0]
    assert os.path.exists(X.raw_path(root, dolly, "starter"))
    before = X.tree_hashes(out)
    assert C.run(c)["files"] == 0                           # restart: nothing left to do
    assert X.tree_hashes(out) == before and len(events(out, "file_done")) == 5


def test_download_resumes_a_partial_file_and_refetches_a_bad_one(tmp_path):
    root = str(tmp_path)
    X.make_world(root)
    c = cfg(root)
    files = json.load(open(c["manifest"]))["files"]
    good, bad = files[2], files[3]                          # two cccc files
    data = open(X.raw_path(root, good), "rb").read()
    half = len(data) // 2
    for e, head in ((good, data[:half]), (bad, b"\0" * (bad["size"] // 2))):
        p = X.raw_path(root, e, "raw") + ".part"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(head)
    assert C.run(c)["reason"] == "done"
    fetch = {e["fid"].split("/", 2)[-1]: e["fetch"] for e in events(c["out"], "file_done")}
    assert fetch[good["path"]]["fetched_bytes"] == len(data) - half
    assert fetch[bad["path"]]["fetched_bytes"] == bad["size"] - bad["size"] // 2 + bad["size"]
    assert X.output_ids(c["out"], c["codec"]) == X.oracle(root, c["manifest"])


def test_failed_file_keeps_its_raw_and_a_crash_before_delete_is_finished_on_restart(
        tmp_path, monkeypatch):
    root = str(tmp_path)
    X.make_world(root, corrupt_gutenberg=True)
    c = cfg(root)
    real = W.delete_raw

    def crash(ctx, item):
        if item["local"]:
            return real(ctx, item)
        raise RuntimeError("simulated crash before delete")

    monkeypatch.setattr(W, "delete_raw", crash)
    with pytest.raises(RuntimeError):
        C.run(c)
    first = [e["fid"] for e in events(c["out"], "file_done")]
    cccc0 = json.load(open(c["manifest"]))["files"][2]
    assert first == ["databricks/databricks-dolly-15k/databricks-dolly-15k.jsonl",
                     "common-pile/cccc/" + X.CCCC[0]]
    assert os.path.exists(X.raw_path(root, cccc0, "raw"))          # committed, not yet deleted
    monkeypatch.setattr(W, "delete_raw", real)
    r = C.run(c)
    assert r["reason"] == "failed" and r["failed"] == [
        "common-pile/project_gutenberg/" + X.GUT]
    fail = events(c["out"], "file_failed")[-1]
    assert fail["stage"] == "extract" and "gzip" in fail["error"].lower()
    assert raw_files(root) == ["raw/common-pile__project_gutenberg/" + X.GUT]
    assert events(c["out"], "raw_deleted")[0]["fid"] == "common-pile/cccc/" + X.CCCC[0]
    assert len(events(c["out"], "raw_deleted")) == 3
    assert X.output_ids(c["out"], c["codec"]) == X.oracle(root, c["manifest"])


def test_disk_guard_stops_cleanly_and_resumes(tmp_path, monkeypatch):
    root = str(tmp_path)
    X.make_world(root)
    with pytest.raises(ValueError):
        C.check_config(cfg(root, min_free_gb=79))
    c = cfg(root)
    monkeypatch.setattr(F, "disk_free", lambda p: 85 * F.GIB)   # over the floor, under 90
    assert C.run(c)["reason"] == "disk_low"
    assert events(c["out"], "file_done") == [] and raw_files(root) == []
    assert events(c["out"], "run_stop")[-1]["reason"] == "disk_low"
    calls = {"n": 0}

    def later_low(p):
        calls["n"] += 1
        return 10 ** 13 if calls["n"] < 12 else 85 * F.GIB

    monkeypatch.setattr(F, "disk_free", later_low)
    assert C.run(c)["reason"] == "disk_low"
    n = len(events(c["out"], "file_done"))
    assert 0 < n < 5
    monkeypatch.setattr(F, "disk_free", lambda p: 10 ** 13)
    assert C.run(c)["reason"] == "done"
    assert X.output_ids(c["out"], c["codec"]) == X.oracle(root, c["manifest"])


@pytest.mark.parametrize("crash_at", [3, 4])    # 3: first cccc file (new shards only);
def test_crash_between_store_and_ledger_recovers_to_the_same_bytes(tmp_path, monkeypatch,
                                                                   crash_at):
    """4: second cccc file, whose first frame lands in a shard that holds a committed frame."""
    clean = str(tmp_path / "clean")
    os.makedirs(clean)
    X.make_world(clean)
    C.run(cfg(clean))
    root = str(tmp_path / "crash")
    os.makedirs(root)
    X.make_world(root)
    c = cfg(root)
    real, seen = stream_ledger.Ledger.append, {"n": 0}

    def flaky(self, event, **kw):
        if event == "file_done":
            seen["n"] += 1
            if seen["n"] == crash_at:
                raise RuntimeError("simulated crash after the store commit")
        return real(self, event, **kw)

    monkeypatch.setattr(stream_ledger.Ledger, "append", flaky)
    with pytest.raises(RuntimeError):
        C.run(c)
    monkeypatch.setattr(stream_ledger.Ledger, "append", real)
    segs = [s for e in events(c["out"], "file_done") for s in e["segments"]]
    open_shard = os.path.join(c["out"], segs[-1]["shard"])
    grew = os.path.getsize(open_shard) > segs[-1]["end"]    # the crashed file appended to it
    assert grew == (crash_at == 4)
    with open(os.path.join(c["out"], "LEDGER.jsonl"), "a") as f:
        f.write('{"event": "file_do')                       # a torn last line
    assert C.run(c)["reason"] == "done"
    start = events(c["out"], "run_start")[-1]
    assert start["recovery"]["ledger_torn_bytes"] > 0
    assert len(start["recovery"]["store_rolled_back"]) == 1
    assert X.output_ids(c["out"], c["codec"]) == X.oracle(root, c["manifest"])
    assert X.tree_hashes(c["out"]) == X.tree_hashes(os.path.join(clean, "out"))


def test_heavy_lock_is_taken_and_a_timeout_stops_cleanly(tmp_path):
    root = str(tmp_path)
    X.make_world(root)
    c = cfg(root, lock_timeout=1)
    fd = os.open(c["lock_file"], os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert C.run(c)["reason"] == "lock_timeout"
        assert events(c["out"], "file_done") == []
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert C.run(c)["reason"] == "done"
    waits = [e["lock_wait"] for e in events(c["out"], "file_done")]
    assert len(waits) == 5 and all(w < 1 for w in waits)


def test_unready_source_is_refused_unless_skipped(tmp_path):
    root = str(tmp_path)
    fake = dict(source="nosuchsource", dataset="x/y", revision="r", path="a.jsonl.gz", size=1,
                sha256="0" * 64, tier=1, role="core", url="file:///nonexistent")
    X.make_world(root, extra_entries=[fake])
    with pytest.raises(SystemExit, match="nosuchsource"):
        C.run(cfg(root))
    assert C.run(cfg(root, skip_unready=True))["reason"] == "done"
    assert T.dry_run(cfg(root, skip_unready=True))["done"] == 5
