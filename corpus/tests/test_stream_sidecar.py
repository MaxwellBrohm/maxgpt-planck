"""stream_core.py per-document sidecars (--sidecar): alignment with the surviving documents, the
.bin -> .npy step at seal time, crash recovery, and the hand-off to neardedup_lsh.py."""
import hashlib
import json
import os

import numpy as np
import pytest

import stream_core as C
import stream_tools as T
import stream_fetch as F
import stream_io as SIO
import stream_ledger
import stream_recover as SR
import stream_fixtures as X
from test_stream_core import cfg, events

HOOK = {"t": "stream_fixtures:len_row"}


@pytest.fixture(autouse=True)
def plenty_of_disk(monkeypatch):
    monkeypatch.setattr(F, "disk_free", lambda p: 10 ** 13)


def check_sidecars(out, codec):
    """Every shard is sealed, has <stem>.t.npy and no .bin, and row i describes line i."""
    n = 0
    for src in ("dolly", "gutenberg", "cccc"):
        if not os.path.isdir(os.path.join(out, src)):
            continue
        for p in X.shard_paths(out, src, codec):
            stem = p.rsplit(".jsonl.", 1)[0]
            assert not os.path.exists(stem + ".t.bin")
            rows = np.load(stem + ".t.npy")
            docs = list(SIO.iter_docs(p))
            assert len(rows) == len(docs) > 0
            for d, r in zip(docs, rows):
                b = d["text"].encode("utf-8")
                assert int(r["n"]) == len(b)
                assert int(r["k"]) == int.from_bytes(hashlib.sha1(b).digest()[:8], "big")
            n += len(docs)
    return n


def test_sidecar_rows_follow_the_surviving_documents(tmp_path):
    root = str(tmp_path)
    X.make_world(root)
    c = cfg(root, workers=2, sidecars=HOOK)
    assert C.run(c)["reason"] == "done"
    assert check_sidecars(c["out"], c["codec"]) == sum(
        e["stats"]["docs_kept"] for e in events(c["out"], "file_done"))
    assert all("t" in s["sidecars"] for s in events(c["out"], "shard_sealed"))
    with pytest.raises(SystemExit, match="sidecar"):
        C.run(dict(c, sidecars={}))                 # a changed sidecar set is refused


@pytest.mark.parametrize("crash_at", [3, 4])
def test_crash_recovery_with_a_sidecar_gives_the_same_bytes(tmp_path, monkeypatch, crash_at):
    clean, root = str(tmp_path / "clean"), str(tmp_path / "crash")
    for r in (clean, root):
        os.makedirs(r)
        X.make_world(r)
    C.run(cfg(clean, sidecars=HOOK))
    real, seen = stream_ledger.Ledger.append, {"n": 0}

    def flaky(self, event, **kw):
        if event == "file_done":
            seen["n"] += 1
            if seen["n"] == crash_at:
                raise RuntimeError("simulated crash after the store commit")
        return real(self, event, **kw)

    monkeypatch.setattr(stream_ledger.Ledger, "append", flaky)
    with pytest.raises(RuntimeError):
        C.run(cfg(root, sidecars=HOOK))
    monkeypatch.setattr(stream_ledger.Ledger, "append", real)
    assert C.run(cfg(root, sidecars=HOOK))["reason"] == "done"
    check_sidecars(cfg(root)["out"], cfg(root)["codec"])
    assert X.tree_hashes(cfg(root)["out"]) == X.tree_hashes(cfg(clean)["out"])


def test_a_crash_before_the_npy_step_is_finished_on_restart(tmp_path, monkeypatch):
    root = str(tmp_path)
    X.make_world(root)
    c = cfg(root, sidecars=HOOK)
    real = SR.finalize_sidecars

    def crash(*a, **kw):
        raise RuntimeError("simulated crash before the .npy step")

    monkeypatch.setattr(SR, "finalize_sidecars", crash)
    with pytest.raises(RuntimeError):
        C.run(c)
    sealed = events(c["out"], "shard_sealed")
    assert sealed and os.path.exists(
        os.path.join(c["out"], sealed[0]["shard"].rsplit(".jsonl.", 1)[0] + ".t.bin"))
    monkeypatch.setattr(SR, "finalize_sidecars", real)
    assert C.run(c)["reason"] == "done"
    check_sidecars(c["out"], c["codec"])


def test_neardedup_pass_reads_the_runner_minhash_sidecars(tmp_path):
    import neardedup_lsh
    root = str(tmp_path)
    X.make_world(root)
    c = cfg(root, workers=2, sidecars={"mh": "stream_work:mh_row"})
    assert C.run(c)["reason"] == "done"
    spec = T.neardedup_spec(c["out"])
    assert [s["label"] for s in spec] == sorted(
        [s["label"] for s in spec], key=["dolly", "gutenberg", "cccc"].index)
    neardedup_lsh.find_near_dups(spec, str(tmp_path / "nd"))
    dropped = []
    for k, s in enumerate(spec):
        res = neardedup_lsh.load_result(s["stem"])
        docs = list(SIO.iter_docs(s["stem"] + ".jsonl." + c["codec"]))
        assert len(res) == len(docs)
        for d, r in zip(docs, res):
            if not r["keep"]:
                lead = list(SIO.iter_docs(spec[r["lshard"]]["stem"] + ".jsonl." + c["codec"]))
                dropped.append((d["id"], lead[r["lrow"]]["id"]))
    assert [x for x in dropped if "cccc:dupA" in x or "cccc:near" in x] == [
        ("cccc:dupA", "cccc:near")]                  # the earlier date wins, not file order
    json.dumps(spec)                                 # the CLI writes it as JSON


def test_seal_open_closes_a_source_whose_files_are_not_all_done(tmp_path):
    root = str(tmp_path)
    X.make_world(root)
    c = cfg(root, sidecars=HOOK)
    assert C.run(dict(c, match=["2019-04"]))["reason"] == "done"        # cccc files 0 and 1
    open_before = {s["shard"] for s in events(c["out"], "shard_sealed")}
    sealed = T.seal_open(dict(c, sources=["cccc"]))
    assert len(sealed) == 1 and sealed[0].startswith("cccc/") and sealed[0] not in open_before
    stem = os.path.join(c["out"], sealed[0].rsplit(".jsonl.", 1)[0])
    assert os.path.exists(stem + ".t.npy") and not os.path.exists(stem + ".t.bin")
    assert T.seal_open(dict(c, sources=["cccc"])) == []                  # nothing open now
    assert C.run(dict(c, match=["2020-10"]))["reason"] == "done"        # file 2: a new shard
    check_sidecars(c["out"], c["codec"])
    assert X.output_ids(c["out"], c["codec"]) == X.oracle(
        root, c["manifest"], skip={"databricks-dolly-15k.jsonl", X.GUT})   # cccc files only
