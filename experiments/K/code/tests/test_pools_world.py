"""SPEC 1-3, 6, 7: pools, the fact base, slot plans, FACT streams and K5 chats (no model)."""
import numpy as np
import pytest

import kcommon as K
import lookup as L
import world as Wd
from kfix import pools


def test_frame_ids_and_pool_sizes():
    tok = K.load_tokenizer()
    for s, i in ((" ?", K.Q), (" =", K.EQ), (" ok", K.OK), (" none", K.NONE)):
        assert tok.encode(s, add_special_tokens=False).ids == [i]
    for s, i in (("<|endoftext|>", 1), ("<|system|>", 2), ("<|user|>", 3), ("<|assistant|>", 4), ("<|end|>", 5),
                 ("<|tool|>", 6), ("<lookup>", 11), ("</lookup>", 12), ("<result>", 13), ("</result>", 14)):
        assert tok.token_to_id(s) == i
    p = pools()
    spec = dict(K.CAP_POOLS + K.LOW_POOLS)
    assert {k: v for k, v in p.sizes.items() if k in spec} == spec
    assert p.sizes["cap_left"] == 714 and p.sizes["kevs"] == 11_056 and p.sizes["kevs_cells"] == 1_382
    own = p.owner()                                    # raises if a token sits in two pools
    assert len(own) == sum(n for _, n in K.CAP_POOLS + K.LOW_POOLS)
    for t in (K.Q, K.EQ, K.OK, K.NONE):
        assert t not in own
    for t in p["V"][:50].tolist() + p["F1"][:20].tolist():
        assert tok.encode(" " + tok.id_to_token(t)[1:], add_special_tokens=False).ids == [t]


def test_kevs_is_whole_cubes():
    """K3 round 2: K-EVAL-S is a seeded 10% of the cells that pair each S position's tokens (B's cube items need
    whole cubes of held-out triples); the cells' pairs are one fixed pairing per position, so cells never overlap."""
    p = pools()
    cells = p.kevs_cells
    assert cells.shape == (1_382, 3, 2)
    got = {tuple(int(x) for x in t) for t in p.kevs}
    want = {(a, b, c) for cl in cells for a in cl[0] for b in cl[1] for c in cl[2]}
    assert got == want and len(got) == 8 * len(cells)
    for i in range(3):
        pair_of = {}
        for cl in cells:
            a, b = (int(x) for x in cl[i])
            assert a != b and pair_of.setdefault(a, b) == b and pair_of.setdefault(b, a) == a
        assert set(pair_of) <= set(p[f"S{i + 1}"].tolist())


def test_world_is_fixed_and_nested():
    p = pools()
    w1, w2 = Wd.make_world(p), Wd.make_world(p)
    assert np.array_equal(w1.names, w2.names) and np.array_equal(w1.values, w2.values)
    assert len({tuple(n) for n in w1.names.tolist()}) == K.N_CAP
    assert np.isin(w1.values, p["V"]).all() and np.isin(w1.names[:, 0], p["F1"]).all()
    for lo, hi in ((0, K.N_LOW), (K.N_LOW, K.N_HIGH), (K.N_HIGH, K.N_CAP)):
        assert w1.qa_train[lo:hi].sum() * 2 == hi - lo


def test_full_scale_layout_matches_spec():
    lay = Wd.layout(Wd.FactCfg("FH"))
    assert lay["tokens_per_entity"] == 1900 and lay["bio_exposures"] == 6_200_000
    assert abs(K.N_HIGH * 66 / 5_010_133 - 0.82) < 0.005 and abs(K.N_LOW * 66 / 5_010_133 - 0.20) < 0.005
    assert lay["bio_region"] * K.BIO_LEN <= lay["draw_tokens"] * lay["share_bio"] / K.SHARES["fact"]
    assert Wd.layout(Wd.FactCfg("CAP"))["bio_exposures"] == 12_400_000


SMALL = dict(slots=400_000, e_bio=20, e_qa=4, n_high=200, n_low=50, n_cap=400, shard_docs=4096)


def _plans():
    w = Wd.make_world(pools())
    return w, {a: Wd.plan(w, Wd.FactCfg(a, **SMALL)) for a in ("F0", "FL", "FH")}


def test_arm_plans_share_slots():
    w, pl = _plans()
    fh, fl, f0 = pl["FH"], pl["FL"], pl["F0"]
    assert len(fh["bio"]) == len(fl["bio"]) == len(f0["bio"])
    cnt = np.bincount(fh["bio"][fh["bio"] >= 0], minlength=200)
    assert (cnt == 20).all()
    region = Wd.layout(Wd.FactCfg("FH", **SMALL))["bio_region"]
    assert (fh["bio"][region:] == -1).all()
    assert np.array_equal(fl["bio"], np.where(fh["bio"] < 50, fh["bio"], -1))
    assert np.array_equal(fl["qa_ent"], np.where(fh["qa_ent"] < 50, fh["qa_ent"], -1))     # the QA plan too
    assert w.qa_train[50] and (fh["qa_ent"] == 50).any()                   # FL's boundary entity has QA to leak
    assert (f0["bio"] == -1).all() and (f0["qa_ent"] == -1).all()
    q = np.bincount(fh["qa_ent"][fh["qa_ent"] >= 0], minlength=200)
    assert (q[w.qa_train[:200]] == 4 * K.N_ATTR).all() and (q[~w.qa_train[:200]] == 0).all()


def test_bio_docs_and_filler():
    w, pl = _plans()
    p = pools()
    fh = Wd.bio_block(w, p, pl["FH"]["bio"][:4096], 0).astype(np.int64)
    f0 = Wd.bio_block(w, p, pl["F0"]["bio"][:4096], 0).astype(np.int64)
    assert (fh[:, 15] == K.EOT).all()
    for d, e in zip(fh, pl["FH"]["bio"][:4096]):
        assert sorted(d[3:15:2].tolist()) == sorted(w.attrs.tolist())
        if e >= 0:
            assert tuple(d[:3]) == tuple(w.names[e])
            got = dict(zip(d[3:15:2].tolist(), d[4:15:2].tolist()))
            assert got == dict(zip(w.attrs.tolist(), w.values[e].tolist()))
        else:
            assert np.isin(d[:3], np.concatenate(p.triple("Z"))).all()
    ents = pl["FH"]["bio"][:4096]
    e0 = int(ents[ents >= 0][0])                       # a fresh pair order at every exposure (Physics 3.1)
    orders = {tuple(d[3:15:2].tolist()) for d, e in zip(fh, ents) if e == e0}
    assert len(orders) >= 2
    same = pl["FH"]["bio"][:4096] < 0                  # filler slots hold the same filler in every arm
    assert np.array_equal(fh[same], f0[same])
    assert not np.isin(f0[:, :3], p["F1"]).any()


def test_k5_chats_and_token_counts():
    w = Wd.make_world(pools())
    c = L.K5Cfg("L50", n5=20, e5=6, slots=400_000)
    ch = L.chats(c)
    cnt = np.zeros((20, K.N_ATTR), int)
    for e, a1, a2 in zip(ch["ent"], ch["a1"], ch["a2"]):
        assert a1 != a2
        cnt[e, a1] += 1
        cnt[e, a2] += 1
    assert (cnt == 6).all()
    hit = L.layout5(c, ch)["hit"]
    assert np.array_equal(hit, ch["u"] < 0.5)
    ts = []
    for i in (0, 1):
        rec = L.k5_chat(w.names[0], w.attrs, w.values[0], 0, 1, bool(i), list(range(6)), "x")
        ts.append(sum(len(t["ids"]) + 2 for t in rec["turns"]))
    assert ts == [30, 41]
    full = {a: L.layout5(L.K5Cfg(a), L.chats(L.K5Cfg(a)))["k5_tokens"] for a in ("L0", "L100")}
    assert full == {"L0": 83_700_000, "L100": 114_390_000}


@pytest.mark.parametrize("arm", ["F0", "FL", "FH"])
def test_plan_deterministic(arm):
    w = Wd.make_world(pools())
    a, b = Wd.plan(w, Wd.FactCfg(arm, **SMALL)), Wd.plan(w, Wd.FactCfg(arm, **SMALL))
    assert all(np.array_equal(a[k], b[k]) for k in a)
