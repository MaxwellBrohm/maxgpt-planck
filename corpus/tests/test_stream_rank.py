"""Exact-dedup priority when a file fails and is retried in a later run (stream_rank.py): the late
file's documents still win over lower-ranked copies committed in the meantime, those copies are
dropped at finalize (core_displaced.py), and the final output equals an uninterrupted run's.

World: stream_fixtures.py plus two CCCC files of a 2019-09 snapshot (ranked between the 2019-04
and 2020-10 files) that share RANK_DUP, the lower-ranked copy (file 1) dated earlier, so
near-dedup's date rule would keep it over the late copy if it took part, and the id cccc:sid with
two different texts (a dup_id pair). Run 1 of the late world fails Gutenberg, cccc 2019-04 file 0
and 2019-09 file 0; run 2 commits them after the files ranked below them."""
import gzip
import hashlib
import json
import os
import sqlite3

import pytest

import core_finalize as CF
import stream_core as C
import stream_fetch as F
import stream_io as SIO
import stream_ledger
import stream_rank as R
import stream_fixtures as X
from fixtures import cp, prose
from test_stream_core import cfg, events

RANK_DUP = "RANK_DUP " + " ".join(f"a {w} of" for w in reversed(X.RARE))
SNAP = ["CC-MAIN-2019-09/cccc-CC-MAIN-2019-09-0000.json.gz",
        "CC-MAIN-2019-09/cccc-CC-MAIN-2019-09-0001.json.gz"]
BREAK = [X.GUT, X.CCCC[0], SNAP[0]]            # the files that fail in run 1 of the late world


@pytest.fixture(autouse=True)
def plenty_of_disk(monkeypatch):
    monkeypatch.setattr(F, "disk_free", lambda p: 10 ** 13)


def make_world(root):
    extra = []
    for k, (path, day) in enumerate(zip(SNAP, ("2019-02-20", "2019-02-10"))):
        docs = [cp(f"rk{k}", RANK_DUP, day, url=f"example.org/rk{k}"),
                cp(f"f{k}", prose(f"snap-{k}", 6), day, url=f"example.org/f{k}"),
                cp("sid", f"SID_{k} " + prose(f"sid-{k}", 6), day, url=f"example.org/s{k}")]
        data = gzip.compress("".join(json.dumps(d) + "\n" for d in docs).encode(), mtime=0)
        p = os.path.join(root, "remote", "common-pile__cccc", path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(data)
        extra.append(dict(source="cccc", dataset="common-pile/cccc", revision="rev0", path=path,
                          size=len(data), sha256=hashlib.sha256(data).hexdigest(), tier=5,
                          role="core", starter_local=False, url="file://" + p))
    return X.make_world(root, extra_entries=extra)


def scfg(root, **kw):
    return cfg(root, sidecars={"mh": "stream_work:mh_row"}, curl_attempts=1, **kw)


def finalize(root):
    for t in (1, 2, 5):
        r = CF.finalize(dict(stage1=os.path.join(root, "out"), final=os.path.join(root, "fin"),
                             tier=t, workers=1, nice=0, lock_timeout=30, level=3,
                             lock_file=os.path.join(root, "heavy.lock")))
        assert r["reason"] in ("done", "up_to_date")


def final_docs(fin):
    out = {}
    for src in sorted(os.listdir(fin)):
        d = os.path.join(fin, src)
        if not src.startswith("_") and os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if ".jsonl." in f:
                    out.setdefault(src, []).extend(
                        (x["id"], x["text"]) for x in SIO.iter_docs(os.path.join(d, f)))
    return {k: sorted(v) for k, v in out.items()}


def late_world(root, broken=BREAK):
    """Run 1 with the `broken` files failing their checksum, run 2 with them repaired."""
    make_world(root)
    m = [e for e in json.load(open(os.path.join(root, "manifest.json")))["files"]
         if e["path"] in broken]
    good = {e["path"]: open(X.raw_path(root, e), "rb").read() for e in m}
    for e in m:
        open(X.raw_path(root, e), "wb").write(b"\0" * e["size"])
    r1 = C.run(scfg(root))
    assert r1["reason"] == "failed" and len(r1["failed"]) == len(m)
    for e in m:
        open(X.raw_path(root, e), "wb").write(good[e["path"]])
    return m


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(F, "disk_free", lambda p: 10 ** 13)
        clean = str(tmp_path_factory.mktemp("clean"))
        make_world(clean)
        assert C.run(scfg(clean))["reason"] == "done"
        late = str(tmp_path_factory.mktemp("late"))
        late_world(late)
        assert C.run(scfg(late))["reason"] == "done"
        for r in (clean, late):
            finalize(r)
    return clean, late


def test_stage1_keeps_the_late_copies_and_names_the_displaced_ones(runs):
    clean, late = runs
    a, b = X.output_ids(f"{clean}/out", "gz"), X.output_ids(f"{late}/out", "gz")
    assert "gutenberg:10" in a["gutenberg"] and "gutenberg:10" in b["gutenberg"]
    assert sorted(set(b["cccc"]) - set(a["cccc"])) == ["cccc:cross", "cccc:dupB", "cccc:rk1"]
    assert b["cccc"].count("cccc:sid") == 2 and a["cccc"].count("cccc:sid") == 1
    disp = {d["fid"].rsplit("/", 1)[1]: (len(d["keys"]), len(d["ids"])) for e in
            events(f"{late}/out", "file_done") for d in e.get("displaced") or ()}
    assert disp == {X.CCCC[1].rsplit("/", 1)[1]: (1, 0), X.CCCC[2].rsplit("/", 1)[1]: (1, 0),
                    SNAP[1].rsplit("/", 1)[1]: (1, 1)}
    assert not any(e.get("displaced") for e in events(f"{clean}/out", "file_done"))


def test_final_output_equals_the_uninterrupted_run(runs):
    clean, late = runs
    a, b = final_docs(f"{clean}/fin"), final_docs(f"{late}/fin")
    assert a == b
    ids = [i for i, _ in b["cccc"]]
    assert "cccc:rk0" in ids and "cccc:rk1" not in ids and "cccc:cross" not in ids
    assert [t[:5] for i, t in b["cccc"] if i == "cccc:sid"] == ["SID_0"]
    rec, ref = (json.load(open(CF.tier_json(f"{r}/fin", 5)))["sources"]["cccc"]
                for r in (late, clean))
    assert rec["dropped"].pop("dup_exact_rank")["docs"] == 4          # cross, dupB, rk1, sid
    assert rec["dropped"] == ref["dropped"] and rec["near_dup_lead"] == ref["near_dup_lead"]


def test_a_crash_after_the_store_commit_of_a_late_file_rolls_its_overrides_back(
        tmp_path, runs, monkeypatch):
    late = runs[1]
    root = str(tmp_path)
    late_world(root)
    real, seen = stream_ledger.Ledger.append, {"n": 0}

    def flaky(self, event, **kw):
        if event == "file_done" and kw.get("displaced"):
            seen["n"] += 1
            if seen["n"] == 2:                     # the second late file (cccc 2019-04 file 0)
                raise RuntimeError("simulated crash after the store commit")
        return real(self, event, **kw)

    monkeypatch.setattr(stream_ledger.Ledger, "append", flaky)
    with pytest.raises(RuntimeError):
        C.run(scfg(root))
    db = sqlite3.connect(f"{root}/out/_state/exact.sqlite")
    assert db.execute("SELECT count(*) FROM overrides").fetchone()[0] == 2
    db.close()
    monkeypatch.setattr(stream_ledger.Ledger, "append", real)
    assert C.run(scfg(root))["reason"] == "done"
    assert len(events(f"{root}/out", "run_start")[-1]["recovery"]["store_rolled_back"]) == 1
    stage1 = [{k: v for k, v in X.tree_hashes(f"{r}/out").items() if not k.endswith(".nd.npy")}
              for r in (root, late)]                    # late/out also holds finalize's .nd.npy
    assert stage1[0] == stage1[1]
    rows = []
    for r in (root, late):
        db = sqlite3.connect(f"{r}/out/_state/exact.sqlite")
        rows.append(db.execute("SELECT o.t, o.k, f.fid FROM overrides o JOIN files f "
                               "ON o.f = f.f ORDER BY 1, 2, 3").fetchall())
        db.close()
    assert rows[0] == rows[1] and len(rows[0]) == 4


def test_keystore_owners_overrides_and_rollback(tmp_path):
    ks = stream_ledger.KeyStore(str(tmp_path / "s.sqlite"))
    ks.commit_file("w", [5, 6], [50])
    ks.commit_file("late", [7], [70], over={"keys": [5], "ids": [50]})
    got = sorted(ks.owners("keys", [5, 6, 7, 8]))
    assert [(k, ks.fid(f)) for k, f in got] == [(5, "w"), (5, "late"), (6, "w"), (7, "late")]
    rank = {"late": 0, "w": 1}
    assert R.best_owner(ks, "keys", [5, 6], rank) == {5: (0, "late"), 6: (1, "w")}
    assert ks.rollback_uncommitted({"w"}) == ["late"]
    assert sorted((k, ks.fid(f)) for k, f in ks.owners("keys", [5, 7])) == [(5, "w")]
    assert ks.owners("ids", [50]) and ks.n_over == 0
    ks.close()


def test_pass_a_follows_the_finalize_priority_within_a_tier(tmp_path):
    root = str(tmp_path)
    oasst = dict(source="oasst2", dataset="OpenAssistant/oasst2", revision="r",
                 path="2023-11-05_oasst2_all.trees.jsonl.gz", size=1, sha256="0" * 64, tier=1,
                 role="core", url="file:///nonexistent")
    X.make_world(root, extra_entries=[oasst])
    man = json.load(open(os.path.join(root, "manifest.json")))
    assert [e["source"] for e in man["files"]][:2] == ["dolly", "oasst2"]       # alphabetical
    items, _ = C.select(C.check_config(scfg(root)), man, set())
    assert [it["source"] for it in items][:3] == ["oasst2", "dolly", "gutenberg"]
    assert CF.PRIORITY is R.PRIORITY
    ranks = R.manifest_ranks(man)
    assert [it["fid"] for it in items] == sorted(ranks, key=ranks.get)


def test_a_displaced_key_missing_from_its_owner_stops_finalize(runs):
    import core_displaced as DR
    late = f"{runs[1]}/out"
    ev = [json.loads(x) for x in open(f"{late}/LEDGER.jsonl")]
    shards = {s["shard"] for e in ev if e["event"] == "file_done" for s in e["segments"]}
    got = DR.displaced_rows(late, ev, shards)
    assert sum(len(r) for r in got.values()) == 4
    d = next(x for e in ev if e["event"] == "file_done" for x in e.get("displaced") or ())
    d["keys"].append(12345)                                   # a key its owner never held
    with pytest.raises(SystemExit, match="1 displaced keys and 0 ids not found"):
        DR.displaced_rows(late, ev, shards)


def test_a_tier_finalized_around_failed_files_says_so_and_is_rebuilt_after_the_retry(
        tmp_path, runs):
    import stream_tools as T
    root = str(tmp_path)
    late_world(root)                                   # run 1: three files failed
    T.seal_open(scfg(root))
    fin = lambda t: CF.finalize(dict(stage1=f"{root}/out", final=f"{root}/fin", tier=t,  # noqa
                                     workers=1, nice=0, lock_timeout=30, level=3,
                                     lock_file=f"{root}/heavy.lock"))
    for t in (1, 5):
        assert fin(t)["reason"] == "done"
    rec = json.load(open(CF.tier_json(f"{root}/fin", 5)))
    cc = "common-pile/cccc/"
    assert rec["complete"] is False and rec["coverage"] == {"cccc": {
        "expected": 5, "done": 3, "failed": [cc + X.CCCC[0], cc + SNAP[0]], "missing": []}}
    assert json.load(open(f"{root}/fin/MANIFEST.json"))["tiers"]["5"]["complete"] is False
    assert json.load(open(CF.tier_json(f"{root}/fin", 1)))["complete"] is True
    assert C.run(scfg(root))["reason"] == "done"
    finalize(root)
    rec = json.load(open(CF.tier_json(f"{root}/fin", 5)))
    assert rec["complete"] is True and rec["coverage"]["cccc"]["done"] == 5
    assert final_docs(f"{root}/fin") == final_docs(f"{runs[0]}/fin")     # the clean run's


def test_late_commit_audit_lists_the_retried_files(runs, capsys):
    import stream_tools as T
    clean, late = runs
    assert T.late_commits(f"{clean}/out", f"{clean}/manifest.json") == []
    rows = T.late_commits(f"{late}/out", f"{late}/manifest.json")
    assert [(r["fid"].rsplit("/", 1)[1], r["displaced"], r["rank_aware"]) for r in rows] == [
        (X.GUT.rsplit("/", 1)[1], 1, True), (X.CCCC[0].rsplit("/", 1)[1], 1, True),
        (SNAP[0].rsplit("/", 1)[1], 2, True)]
    assert rows[0]["late_after"] == {"rank": 3, "manifest": 3}    # 3 cccc files went first
    assert T.main(["late-commits", "--out", f"{late}/out", "--manifest",
                   f"{late}/manifest.json"]) == 0
    assert "3 late commits; 0 by code before stream_rank.py" in capsys.readouterr().out
