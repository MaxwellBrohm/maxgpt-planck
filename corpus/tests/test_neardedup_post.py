"""neardedup_post.py (second near-dedup pass over the final core v0 shards) and neardedup_post_tok.py,
on the fake FINAL of pd_fixtures.py."""
import hashlib
import json
import os

import numpy as np
import pytest

import neardedup as ND
import neardedup_post as NP
import neardedup_post_tok as NT
import pd_fixtures as F
import stream_io as SIO
from nd_fixtures import jaccard


def lines(path):
    with SIO.open_shard(path) as f:
        return list(f)


def ids(path):
    return [json.loads(x)["id"] for x in lines(path)]


def run(tmp_path, final=None, **kw):
    final = final or F.make_final(str(tmp_path / "final"))
    out = str(tmp_path / "out")
    m = NP.run(dict(final=final, out=out, workers=1, nice=0, **kw), log=lambda s: None)
    return final, out, m


def snapshot(root):
    got = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            got[p] = (SIO.sha256_file(p), os.stat(p).st_mtime_ns)
    return got


def test_planted_pairs_are_what_the_fixture_says():
    t = F.texts()
    assert jaccard(t["stage1_biz_old"], t["stage1_biz_new"]) < 0.6       # pass B kept both
    for a, b in F.DROPPED.items():
        assert jaccard(t[a], t[b]) >= 0.85, (a, b)
    assert 0.3 <= jaccard(t["far0"], t["far1"]) <= 0.55
    sig = np.stack([ND.signature(t["vf0"]), ND.signature(t["vf1"])])
    bands = ND.band_hashes(sig)
    assert (bands[:, 0] == bands[:, 1]).any()                             # an LSH candidate
    assert ND.est_jaccard(sig[0], sig[1]) < ND.VERIFY - 0.05              # that verify rejects


def test_drops_exactly_the_planted_near_duplicates(tmp_path):
    final, out, m = run(tmp_path)
    before = {i for s in m["shards"] for i in ids(os.path.join(final, s["shard"]))}
    after = {i for s in m["shards"] for i in ids(os.path.join(out, s["shard"]))}
    assert before - after == set(F.DROPPED)
    assert after == before - set(F.DROPPED)


def test_lead_follows_the_core_v0_keep_rule(tmp_path):
    final, out, m = run(tmp_path)
    rep = json.load(open(os.path.join(out, "_pd", "neardedup_report.json")))
    where = {}
    for k, stem in enumerate(rep["shards"]):
        shard = os.path.relpath(stem, os.path.join(os.path.realpath(out), "_pd", "sig"))
        rows = ids(os.path.join(final, shard + ".jsonl." + SIO.best_codec()))
        for r, i in enumerate(rows):
            where[i] = (k, r)
    for drop, lead in F.DROPPED.items():
        k, r = where[drop]
        nd = np.load(rep["shards"][k] + ND.ND_SUFFIX)
        assert nd["keep"][r] == 0
        assert (int(nd["lshard"][r]), int(nd["lrow"][r])) == where[lead], drop


def test_kept_lines_byte_identical_and_sidecars_filtered(tmp_path):
    final, out, m = run(tmp_path)
    for s in m["shards"]:
        src, dst = lines(os.path.join(final, s["shard"])), lines(os.path.join(out, s["shard"]))
        keep = [x for x in src if json.loads(x)["id"] not in F.DROPPED]
        assert dst == keep
        stem0 = os.path.join(final, s["shard"]).rsplit(".jsonl.", 1)[0]
        stem1 = os.path.join(out, s["shard"]).rsplit(".jsonl.", 1)[0]
        keys = SIO.read_keys(stem1 + ".keys.u64")
        want = [int(hashlib.sha1(json.loads(x)["text"].encode()).hexdigest()[:16], 16)
                for x in dst]
        assert keys.tolist() == want
        info0, info1 = np.load(stem0 + ".info.npy"), np.load(stem1 + ".info.npy")
        pos = [src.index(x) for x in dst]
        assert info1.tolist() == info0[pos].tolist()


def test_shard_without_drops_is_a_byte_copy(tmp_path):
    final, out, m = run(tmp_path)
    for s in m["shards"]:
        if s["source"] in ("dolly", "gutenberg"):
            a, b = (open(os.path.join(r, s["shard"]), "rb").read() for r in (final, out))
            assert a == b and s["dropped_docs"] == 0


def test_manifest_hashes_counts_and_per_source_stats(tmp_path):
    final, out, m = run(tmp_path)
    man = json.load(open(os.path.join(out, "MANIFEST.json")))
    assert man["shards"] == m["shards"] and len(man["shards"]) == len(F.docs())
    texts, want = F.texts(), {}
    for s in man["shards"]:
        p = os.path.join(out, s["shard"])
        assert SIO.sha256_file(p) == s["sha256"]
        assert SIO.sha256_file(os.path.join(out, s["keys"])) == s["keys_sha256"]
        assert len(lines(p)) == s["docs"] == s["v0_docs"] - s["dropped_docs"]
        assert sum(map(len, lines(p))) == s["ubytes"]
        w = want.setdefault(s["source"], dict.fromkeys(
            ("dropped_docs", "dropped_bytes", "docs", "text_bytes"), 0))
        for i in ids(os.path.join(final, s["shard"])):
            if i in F.DROPPED:
                w["dropped_docs"] += 1
                w["dropped_bytes"] += len(texts[i].encode("utf-8"))
            else:
                w["docs"] += 1
                w["text_bytes"] += len(texts[i].encode("utf-8"))
    for src, w in want.items():
        got = man["sources"][src]
        assert {k: got[k] for k in w} == w, src
        assert got["text_bytes_in"] - got["dropped_bytes"] == got["text_bytes"]
        assert got["tier"] == F.TIERS[src]
    assert man["totals"]["dropped_docs"] == len(F.DROPPED)
    assert man["totals"]["docs"] == sum(w["docs"] for w in want.values())


def test_final_is_never_modified(tmp_path):
    final = F.make_final(str(tmp_path / "final"))
    before = snapshot(final)
    run(tmp_path, final)
    assert snapshot(final) == before


def test_rerun_is_a_no_op_and_resumes_a_lost_shard(tmp_path):
    final, out, m = run(tmp_path)
    s = next(x for x in m["shards"] if x["shard"].startswith("cccc/cccc-00000"))
    p = os.path.join(out, s["shard"])
    t0 = os.stat(p).st_mtime_ns
    _, _, m2 = run(tmp_path, final)
    assert os.stat(p).st_mtime_ns == t0 and m2["shards"] == m["shards"]
    os.remove(p)
    os.remove(os.path.join(out, "_pd", "shards", s["shard"].rsplit(".jsonl.", 1)[0] + ".json"))
    _, _, m3 = run(tmp_path, final)
    assert SIO.sha256_file(p) == s["sha256"] and m3["shards"] == m["shards"]


def test_input_that_differs_from_its_manifest_is_refused(tmp_path):
    final = F.make_final(str(tmp_path / "final"))
    man = json.load(open(os.path.join(final, "MANIFEST.json")))
    p = os.path.join(final, next(s["shard"] for s in man["shards"] if s["source"] == "cccc"))
    with open(p, "ab") as f:
        f.write(SIO.Frame(SIO.best_codec(), 3).end())
    with pytest.raises(RuntimeError, match="sha256"):
        run(tmp_path, final)


def test_input_changed_after_signatures_is_refused_at_write(tmp_path):
    final, out, m = run(tmp_path)
    s = next(x for x in m["shards"] if x["dropped_docs"])
    os.remove(os.path.join(out, "_pd", "shards", s["shard"].rsplit(".jsonl.", 1)[0] + ".json"))
    with open(os.path.join(final, s["shard"]), "ab") as f:
        f.write(SIO.Frame(SIO.best_codec(), 3).end())
    with pytest.raises(RuntimeError, match="sha256"):
        run(tmp_path, final)


def test_out_inside_final_is_refused(tmp_path):
    final = F.make_final(str(tmp_path / "final"))
    with pytest.raises(SystemExit):
        NP.run(dict(final=final, out=os.path.join(final, "pd"), workers=1, nice=0))


def test_sidecars_are_neardedup_rows_of_the_final_text(tmp_path):
    final, out, m = run(tmp_path)
    for s in m["shards"]:
        ds = [json.loads(x) for x in lines(os.path.join(final, s["shard"]))]
        stem = os.path.join(out, "_pd", "sig", s["shard"].rsplit(".jsonl.", 1)[0])
        want = ND.sketch([d["text"] for d in ds],
                         [d["meta"].get("date") or d["meta"].get("created") for d in ds])
        got = ND.load_sidecar(stem, mmap=False)
        assert np.array_equal(got["sig"], want["sig"]) and np.array_equal(got["day"], want["day"])


def test_token_estimates_by_the_final_stats_method(tmp_path):
    final, out, m = run(tmp_path)
    count = lambda s: len(s.split())                                          # noqa: E731
    res = NT.run(out, count=count, log=lambda s: None)
    man = json.load(open(os.path.join(out, "MANIFEST.json")))
    for src, b in man["sources"].items():
        ts = [json.loads(x)["text"] for s in man["shards"] if s["source"] == src
              for x in lines(os.path.join(out, s["shard"]))]
        r = res["sources"][src]
        assert r["sample_docs"] == len(ts) == b["docs"]                       # all docs sampled
        assert r["tokens_est_8k"] == sum(map(count, ts))
        assert r["bytes_per_token_8k"] == round(b["text_bytes"] / sum(map(count, ts)), 4)
        lo, hi = r["tokens_est_8k_ci95"]
        assert lo <= r["tokens_est_8k"] <= hi
        assert r["dropped_tokens_est_8k"] == round(b["dropped_bytes"] / r["bytes_per_token_8k"])
    assert res["totals"]["tokens_est_8k"] == sum(r["tokens_est_8k"] for r in res["sources"].values())


def test_sample_plan_is_uniform_per_source_and_capped():
    shards = [{"source": "cccc", "docs": 3000}, {"source": "loc", "docs": 500},
              {"source": "cccc", "docs": 1000}, {"source": "loc", "docs": 500}]
    sel = NT.sample_plan(shards)
    assert len(sel[0]) + len(sel[2]) == 2000 and len(sel[1]) + len(sel[3]) == 400
    assert all(0 <= r < shards[i]["docs"] for i, rows in sel.items() for r in rows)
    assert 1300 <= len(sel[0]) <= 1700                                     # 3/4 of 2000, +-
    assert sel == NT.sample_plan(shards)


def test_worker_pool_gives_the_serial_result(tmp_path):
    final, out, m = run(tmp_path)
    m3 = NP.run(dict(final=final, out=str(tmp_path / "out3"), workers=3, nice=0),
                log=lambda s: None)
    assert m3["shards"] == m["shards"] and m3["sources"] == m["sources"]
