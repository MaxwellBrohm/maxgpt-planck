"""MinHash near-dedup (neardedup.py pass A, neardedup_lsh.py pass B) on synthetic documents."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

import neardedup as ND
import neardedup_lsh as NL
from nd_fixtures import Gen, jaccard, write_shard

CORPUS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINNED = "The quick brown fox jumps over the lazy dog near the river bank."


def keep_of(stem):
    return NL.load_result(stem)["keep"].astype(bool)


def test_normalization_ignores_case_punctuation_accents_and_digits():
    a = "The Café opened in 2019, and (honestly) it's GREAT!"
    b = "the cafe opened in 2020 and honestly it s great"
    assert ND.words(a) == ND.words(b)
    assert np.array_equal(ND.signature(a), ND.signature(b))
    assert not np.array_equal(ND.shingles(a), ND.shingles(b.replace("opened", "closed")))


def test_signature_is_pinned_and_the_same_in_a_fresh_process():
    s = ND.signature(PINNED)
    assert s.dtype == np.uint32 and s.shape == (126,)
    assert s[:6].tolist() == [159682933, 254653186, 980578921, 48270057, 358731926, 132137691]
    assert int(s.sum()) == 52759213505
    text = Gen(3).text(300)
    code = (f"import sys; sys.path.insert(0, {CORPUS!r}); import neardedup as ND; "
            f"print(ND.signature({text!r}).tolist())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env=dict(os.environ, PYTHONHASHSEED="12345")).stdout
    assert json.loads(out) == ND.signature(text).tolist()


def test_day_number():
    d = ND.day_number
    assert d("2019-03-04T05:33:58.000Z") == d("2019-03-04") == __import__("datetime").date(
        2019, 3, 4).toordinal()
    assert d("2019") == d(2019) == d("2019-01-01")
    assert d("1850") < d("2019") < ND.UNDATED
    for bad in (None, "", "sometime in 2019", "2019-13-01", True, "0000"):
        assert d(bad) == ND.UNDATED, bad


def test_wordless_texts_match_only_identical_texts():
    a, b = "!!! ??? ... --- *** " * 20, "### $$$ %%% ^^^ " * 25
    assert ND.shingles(a).size == 0
    assert ND.est_jaccard(ND.signature(a), ND.signature(b)) < 0.05
    assert np.array_equal(ND.signature(a), ND.signature(a))


def test_band_hashes_change_only_in_the_band_of_a_changed_value():
    s = ND.signature(PINNED)
    t = np.repeat(s[None], ND.NPERM, axis=0)
    t[np.arange(ND.NPERM), np.arange(ND.NPERM)] ^= 1           # row i changes position i
    bs, bt = ND.band_hashes(s[None])[:, 0], ND.band_hashes(t)
    assert bs.shape == (14,) and bt.shape == (14, ND.NPERM)
    changed = bt != bs[:, None]                               # (band, position)
    assert (changed.sum(axis=0) == 1).all()                   # every position is in one band
    assert (changed.argmax(axis=0) == np.arange(ND.NPERM) // ND.ROWS).all()


def test_long_documents_are_hashed_in_blocks_without_loss():
    sh = ND.shingles(Gen(6).text(3 * ND.BLOCK + 123))
    assert sh.size > 3 * ND.BLOCK
    direct = ((sh[:, None] * ND._A + ND._B) >> np.uint64(32)).min(axis=0).astype(np.uint32)
    assert np.array_equal(ND.minhash(sh), direct)


def test_signature_estimates_jaccard():
    g, err = Gen(5), []
    for target in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3):
        for _ in range(20):
            ws = g.words(g.r.randint(300, 700))
            a, b = " ".join(ws), " ".join(g.near_copy(ws, target))
            err.append(float(ND.est_jaccard(ND.signature(a), ND.signature(b))) - jaccard(a, b))
    err = np.array(err)                    # sd of one estimate is at most 0.045 (126 perms)
    assert abs(err.mean()) < 0.015 and np.sqrt((err ** 2).mean()) < 0.05 and abs(err).max() < 0.16


@pytest.fixture(scope="module")
def levels(tmp_path_factory):
    """One edited copy per base doc (no chains): bases in shard s0 (they lead), copies in s1."""
    d, g = tmp_path_factory.mktemp("levels"), Gen(11)
    base, copy, jac = [], [], []
    for target in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.4):
        for _ in range(60):
            ws = g.words(g.r.randint(250, 700))
            base.append(" ".join(ws))
            copy.append(" ".join(g.near_copy(ws, target)))
            jac.append(jaccard(base[-1], copy[-1]))
    return d, base, copy, np.array(jac)


def detected(d, base, copy, verify, name):
    w = os.path.join(d, name)
    os.makedirs(w, exist_ok=True)
    s0, s1 = write_shard(w, "s0", base), write_shard(w, "s1", copy)
    rep = NL.find_near_dups([{"stem": s0, "label": "base"}, {"stem": s1, "label": "copy"}], w,
                            verify=verify)
    assert keep_of(s0).all()
    return ~keep_of(s1), rep


def theory(jac, verify):
    """Per-pair detection probability: P(candidate in a band) x P(signature estimate >= verify)."""
    from math import ceil, comb
    cand = 1 - (1 - jac ** ND.ROWS) ** ND.BANDS
    if not verify:
        return cand
    m = ceil(verify * ND.NPERM - 1e-9)
    ver = np.array([sum(comb(ND.NPERM, i) * j ** i * (1 - j) ** (ND.NPERM - i)
                        for i in range(m, ND.NPERM + 1)) for j in jac])
    return cand * ver


BINS = [(0.4, 0.55), (0.55, 0.62), (0.62, 0.68), (0.68, 0.75), (0.75, 0.8), (0.8, 0.85),
        (0.85, 0.9), (0.9, 1.01)]


def check_against_theory(hit, jac, verify):
    """Observed detections per Jaccard bin must sit within 4 sd (+1) of the S-curve expectation,
    and the total over the uncertain range (0.5-0.9) within 3 sd."""
    p = theory(jac, verify)
    for lo, hi in BINS:
        m = (jac >= lo) & (jac < hi)
        assert m.sum() >= 20, (lo, hi)
        exp, sd = p[m].sum(), np.sqrt((p[m] * (1 - p[m])).sum())
        assert abs(hit[m].sum() - exp) <= 4 * sd + 1, (lo, hi, int(hit[m].sum()), exp)
    m = (jac >= 0.5) & (jac < 0.9)
    exp, sd = p[m].sum(), np.sqrt((p[m] * (1 - p[m])).sum())
    assert abs(hit[m].sum() - exp) <= 3 * sd, (int(hit[m].sum()), exp, sd)


def test_detection_by_similarity_level(levels):
    d, base, copy, jac = levels
    hit, rep = detected(d, base, copy, ND.VERIFY, "v")
    check_against_theory(hit, jac, ND.VERIFY)
    assert hit[jac >= 0.9].mean() >= 0.98 and hit[(jac >= 0.8) & (jac < 0.9)].mean() >= 0.8
    assert hit[jac < 0.6].sum() == 0
    assert rep["dropped"] == hit.sum() and rep["by_label"]["copy"]["dropped"] == hit.sum()
    assert rep["dropped_by_leader_label"] == {"copy <- base": int(hit.sum())}
    assert len(rep["audit"]) == min(200, hit.sum())
    assert all(0.7 <= a["sig_jaccard"] <= 1 for a in rep["audit"])   # no chains here


def test_verification_cuts_candidates_under_the_threshold(levels):
    d, base, copy, jac = levels
    hit_v, _ = detected(d, base, copy, ND.VERIFY, "v2")
    hit_0, rep0 = detected(d, base, copy, 0, "nov")
    check_against_theory(hit_0, jac, 0)
    low = jac < 0.62
    assert hit_0[low].sum() >= 5 and hit_v[low].sum() == 0      # LSH alone merges these
    assert rep0["edges"]["rejected"] == 0 and (hit_v <= hit_0).all()


def test_keep_rule_is_tier_then_date_then_file_order(tmp_path):
    g = Gen(21)
    ws = g.words(400)
    c = [" ".join(g.edit(ws, 2)) for _ in range(7)]              # Jaccard about 0.95
    a = write_shard(tmp_path, "a", [c[0], c[1]], ["2010-01-01", None])
    b = write_shard(tmp_path, "b", [c[2], c[3], c[4]], ["2021-05-05", "2015-02-02", None])
    e = write_shard(tmp_path, "e", [c[5], c[6]], ["2015-02-02", "2011"])
    shards = [{"stem": a, "tier": 2}, {"stem": b, "tier": 1}, {"stem": e, "tier": 1}]
    rep = NL.find_near_dups(shards, str(tmp_path))
    ra, rb, re_ = (NL.load_result(s) for s in (a, b, e))
    assert ra["keep"].tolist() == [0, 0] and rb["keep"].tolist() == [0, 0, 0]
    assert re_["keep"].tolist() == [0, 1]                       # tier 1, earliest date (2011)
    for r in (ra, rb, re_):
        assert (r["lshard"] == 2).all() and (r["lrow"] == 1).all() and (r["csize"] == 7).all()
    assert rep["clusters"] == 1 and rep["dropped"] == 6
    u = write_shard(tmp_path, "u", [c[0], c[1]], [None, "2022-01-01"])
    NL.find_near_dups([{"stem": u}], str(tmp_path))
    assert keep_of(u).tolist() == [False, True]                 # dated before undated


def chain(g, n=10, width=400, step=20):
    ws = g.words(width + step * n)
    return [" ".join(ws[i * step:i * step + width]) for i in range(n)]


def test_components_are_transitive_and_frozen_docs_stay(tmp_path):
    docs = chain(Gen(31))                     # neighbours J = 0.90, ends J = 0.38
    assert jaccard(docs[0], docs[1]) > 0.88 and jaccard(docs[0], docs[-1]) < 0.4
    f = write_shard(tmp_path, "f", [docs[0], docs[-1]])
    m = write_shard(tmp_path, "m", docs[1:-1])
    shards = [{"stem": f, "tier": 0, "frozen": True}, {"stem": m, "tier": 1}]
    rep = NL.find_near_dups(shards, str(tmp_path))
    assert not os.path.exists(f + NL.ND_SUFFIX)                 # frozen shards are not rewritten
    assert not keep_of(m).any() and rep["dropped"] == 8 and rep["clusters"] == 1
    assert (NL.load_result(m)["csize"] == 9).all()               # lead docs[0] + 8 dropped
    shards[0]["frozen"] = False
    NL.find_near_dups(shards, str(tmp_path))
    assert keep_of(f).tolist() == [True, False]                 # not frozen: the far end goes too


def test_include_mask_leaves_excluded_docs_out(tmp_path):
    g = Gen(41)
    ws = g.words(300)
    s = write_shard(tmp_path, "s", [" ".join(ws), " ".join(ws), " ".join(g.words(300))])
    NL.find_near_dups([{"stem": s, "include": np.array([False, True, True])}], str(tmp_path))
    r = NL.load_result(s)
    assert r["keep"].tolist() == [0, 1, 1] and r["csize"].tolist() == [0, 1, 1]
    with pytest.raises(ValueError, match="include mask"):
        NL.find_near_dups([{"stem": s, "include": np.array([True])}], str(tmp_path))


def test_result_does_not_depend_on_partitions_or_workers(tmp_path):
    g, texts = Gen(51), []
    for _ in range(300):
        ws = g.words(g.r.randint(100, 400))
        texts += [" ".join(g.near_copy(ws, t)) for t in (1.0, 0.9, 0.8, 0.7)]
    texts += chain(g, n=12)
    order = np.random.default_rng(0).permutation(len(texts))
    parts = [[texts[i] for i in order[k::4]] for k in range(4)]
    base = str(tmp_path / "base")
    os.makedirs(base)
    stems = [write_shard(base, f"s{k}", p, [f"20{10 + k}-01-01"] * len(p))
             for k, p in enumerate(parts)]
    got = {}
    for name, kw in (("p1", dict(mem_mb=1024)), ("p16", dict(mem_mb=0.005, workers=3))):
        w = str(tmp_path / name)
        shutil.copytree(base, w)
        shards = [{"stem": os.path.join(w, f"s{k}"), "tier": k % 2} for k in range(4)]
        rep = NL.find_near_dups(shards, w, **kw)
        got[name] = (rep["partitions"], [NL.load_result(s["stem"]).tolist() for s in shards])
    assert got["p1"][0] == 1 and got["p16"][0] >= 16
    assert got["p1"][1] == got["p16"][1]
    assert sum(r[0] == 0 for sh in got["p1"][1] for r in sh) > 300
