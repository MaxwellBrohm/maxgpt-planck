"""core_finalize.py end to end on the core_fixtures world: stream_core.py (pass A, MinHash
sidecars) then finalize tiers 1, 2 and 5 (near-dedup across tiers, boilerplate, final shards,
pass D). Every planted case in core_fixtures.py is checked; each one fails when its stage is
disabled (see the module doc there)."""
import hashlib
import json
import os

import numpy as np
import pytest

import core_final_work as FW
import core_finalize as CF
import stream_core as C
import stream_fetch as F
import stream_io as SIO
import core_fixtures as X

CODEC = SIO.best_codec()


@pytest.fixture(autouse=True)
def plenty_of_disk(monkeypatch):
    monkeypatch.setattr(F, "disk_free", lambda p: 10 ** 13)


def scfg(root, **kw):
    c = dict(manifest=os.path.join(root, "manifest.json"), out=os.path.join(root, "s1"),
             raw=os.path.join(root, "raw"), lock_file=os.path.join(root, "heavy.lock"),
             workers=2, dl_threads=2, shard_bytes=6000, codec=CODEC, retry_sleep=0,
             batch_wait=0, blocked_grace=0, nice=0, lock_timeout=30,
             allow_unrecorded_license=["cccc"], sidecars={"mh": "stream_work:mh_row"})
    c.update(kw)
    return c


def fcfg(root, tier, **kw):
    c = dict(stage1=os.path.join(root, "s1"), final=os.path.join(root, "fin"), tier=tier,
             workers=2, nice=0, lock_file=os.path.join(root, "heavy.lock"), lock_timeout=30,
             level=3)
    c.update(kw)
    return c


def final_docs(fin):
    out = {}
    for src in sorted(os.listdir(fin)):
        d = os.path.join(fin, src)
        if src.startswith("_") or not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".jsonl." + CODEC):
                p = os.path.join(d, f)
                docs = list(SIO.iter_docs(p))
                keys = SIO.read_keys(p.replace(".jsonl." + CODEC, ".keys.u64"))
                assert len(keys) == len(docs)
                for doc, k in zip(docs, keys):
                    h = hashlib.sha1(doc["text"].encode("utf-8")).digest()
                    assert doc["meta"]["sha1"] == h.hex()
                    assert int(k) == int.from_bytes(h[:8], "big")
                out.setdefault(src, []).extend(docs)
    return out


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("core"))
    X.make_world(root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(F, "disk_free", lambda p: 10 ** 13)
        assert C.run(scfg(root))["reason"] == "done"
    for t in (1, 2, 5):
        assert CF.finalize(fcfg(root, t))["reason"] == "done"
    fin = os.path.join(root, "fin")
    return root, final_docs(fin), {t: json.load(open(CF.tier_json(fin, t))) for t in (1, 2, 5)}


def with_marker(docs, m):
    return [d for d in docs if m in d["text"]]


def test_near_dedup_runs_before_boilerplate(world):
    _, docs, rec = world
    body = with_marker(docs["cccc"], "BODY_MARKER")
    assert [d["id"] for d in body] == ["cccc:copy0"]                 # earliest date wins
    assert all(line in body[0]["text"] for line in X.BODY)           # its body was not stripped
    assert body[0]["meta"]["nd_csize"] == 12
    lead = rec[5]["sources"]["cccc"]["near_dup_lead"]
    assert lead["same_file"] == 5 and lead["same_group"] == 6        # copies 1-5 file 0, 6-11 file 1


def test_boilerplate_lines_stripped_by_url_group(world):
    _, docs, rec = world
    assert not with_marker(docs["cccc"], "NAV_MARKER")
    assert len(with_marker(docs["cccc"], "NEARBOILER_MARKER")) == 11  # 9 URL groups < 10
    assert not with_marker(docs["cccc"], "SHORT_MARKER")
    st = rec[5]["sources"]["cccc"]
    assert st["dropped"]["boilerplate_short"]["docs"] == 1
    assert st["boilerplate"]["lines"] == 1 + 5 + 5                     # NAV, ABLOCK, BBLOCK
    assert all(d["meta"].get("boilerplate_bytes", 0) > 0 for d in docs["cccc"]
               if d["id"].startswith(("cccc:base", "cccc:qa", "cccc:qb")))


def test_pass_d_drops_texts_made_identical_by_stripping(world):
    _, docs, rec = world
    assert [d["id"] for d in with_marker(docs["cccc"], "POSTCLEAN_MARKER")] == ["cccc:p1"]
    assert [d["id"] for d in with_marker(docs["gutenberg"] + docs["cccc"], "EARLYCLEAN_MARKER")
            ] == ["gutenberg:8"]                                     # equal to a tier-2 text
    assert rec[5]["pass_d"]["dropped"] == 2
    assert rec[5]["sources"]["cccc"]["dropped"]["dup_exact_post_clean"]["docs"] == 2


def test_earlier_tiers_win_across_sources_and_chat_anchors_first(world):
    _, docs, rec = world
    assert with_marker(docs["gutenberg"], "GUTBOOK_MARKER")
    assert not with_marker(docs["cccc"], "GUTBOOK_MARKER")
    assert rec[5]["sources"]["cccc"]["near_dup_lead"]["other_source:gutenberg"] == 1
    assert with_marker(docs["dolly"], "DOLLY_MARKER")
    assert not with_marker(docs.get("foodista", []), "DOLLY_MARKER")   # dated, but priority 8
    assert rec[1]["sources"]["foodista"]["near_dup_lead"]["other_source:dolly"] == 1


def test_accounting_masks_manifest_and_meta(world):
    root, docs, rec = world
    fin = os.path.join(root, "fin")
    for r in rec.values():
        for src, st in r["sources"].items():
            assert st["docs_in"] == st["docs_kept"] + sum(d["docs"] for d in st["dropped"].values())
            assert st["docs_kept"] == len(docs[src])
        for s in r["shards"]:
            m = np.load(CF.mask_path(fin, s["stage1"]))
            info = np.load(os.path.join(fin, s["shard"].rsplit(".jsonl.", 1)[0] + ".info.npy"))
            assert int(m.sum()) == s["docs"] == len(info)
            assert SIO.sha256_file(os.path.join(fin, s["shard"])) == s["sha256"]
    man = json.load(open(os.path.join(fin, "MANIFEST.json")))
    assert man["totals"]["docs"] == sum(len(v) for v in docs.values())
    assert all(d["meta"]["decon"] == "pending" and d["meta"]["nd_csize"] >= 1
               for v in docs.values() for d in v)


def test_rerun_is_a_no_op_and_an_earlier_change_rebuilds(world):
    root = world[0]
    fin = os.path.join(root, "fin")
    before = open(CF.tier_json(fin, 5)).read()
    assert CF.finalize(fcfg(root, 5))["reason"] == "up_to_date"
    assert CF.finalize(fcfg(root, 2))["reason"] == "up_to_date"
    assert open(CF.tier_json(fin, 5)).read() == before
    assert CF.finalize(fcfg(root, 2, level=4))["reason"] == "done"     # tier 2 record changes
    assert CF.finalize(fcfg(root, 5))["reason"] == "done"              # so tier 5 rebuilds
    assert final_docs(fin)["cccc"] == world[1]["cccc"]


def test_a_later_tier_needs_the_earlier_records(tmp_path):
    root = str(tmp_path)
    X.make_world(root)
    assert C.run(scfg(root, tiers=[2, 5]))["reason"] == "done"
    with pytest.raises(SystemExit, match="finalize it first"):
        CF.finalize(fcfg(root, 5))


def test_url_rewrite_fetches_from_the_mirror(tmp_path):
    root = str(tmp_path)
    X.make_world(root, url_prefix="https://mirror.invalid/x")
    c = scfg(root, tiers=[1], curl_attempts=1)
    assert C.run(c)["reason"] == "failed"                               # the canonical URL fails
    rw = {"https://mirror.invalid/x": "file://" + os.path.join(root, "remote")}
    assert C.run(dict(c, url_rewrite=rw))["reason"] == "done"
    done = [json.loads(x) for x in open(os.path.join(c["out"], "LEDGER.jsonl"))]
    assert {e["fid"].split("/")[0] for e in done if e["event"] == "file_done"} == {
        "databricks", "common-pile"}


def test_count_bad_and_strip_lines(tmp_path):
    a, b = FW.group_hash("http://www.x.org/p/", "00" * 20), FW.group_hash("x.org/p", "11" * 20)
    assert a == b != FW.group_hash(None, "00" * 20)
    lh = np.array([5, 5, 5, 7, 7, 9], dtype=np.uint64) << np.uint64(40)
    grp = np.array([1, 2, 2, 1, 2, 3], dtype=np.uint64)
    part = FW.part_of(lh, 4)
    o = np.argsort(part, kind="stable")
    pre = str(tmp_path / "x")
    np.save(pre + ".pairs.npy", np.stack([lh[o], grp[o]], axis=1))
    np.save(pre + ".offs.npy", np.searchsorted(part[o], np.arange(5)))
    assert FW.count_bad([pre], 4, 2).tolist() == [5 << 40, 7 << 40]
    assert FW.count_bad([pre], 4, 3).tolist() == []
    from boilerplate import line_hash
    bad = np.sort(np.array([line_hash("drop me")], dtype=np.uint64))
    assert FW.strip_lines("keep this\n  drop me \nand this", bad) == "keep this\nand this"
    assert FW.strip_lines("nothing here", bad) == "nothing here"


def test_prefetch_downloads_only_later_tiers_and_the_run_reuses_them(tmp_path):
    import fcntl
    import core_prefetch as P
    root = str(tmp_path)
    X.make_world(root)
    c = scfg(root)
    pc = {k: c[k] for k in ("manifest", "out", "raw", "retry_sleep")}
    assert P.run(dict(pc, tiers=[2, 5], window_gb=1, dl_threads=2), poll=0.05) == "all_fetched"
    got = sorted(os.path.relpath(os.path.join(d, f), c["raw"]) for d, _, fs in
                 os.walk(c["raw"]) for f in fs)
    assert got and all(g.startswith(("common-pile__cccc/", "common-pile__project_gutenberg/"))
                       for g in got) and len(got) == 4
    with open(os.path.join(root, "prefetch.lock"), "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert P.run(dict(pc, tiers=[5], window_gb=1)) == "busy"
    assert C.run(c)["reason"] == "done"
    fetch = {e["fid"].split("/")[1]: e["fetch"] for e in
             map(json.loads, open(os.path.join(c["out"], "LEDGER.jsonl")))
             if e["event"] == "file_done"}
    assert fetch["cccc"]["cached"] and fetch["project_gutenberg"]["cached"]
    assert not fetch["foodista"]["cached"]
