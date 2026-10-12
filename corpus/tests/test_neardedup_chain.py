"""--chain-floor (neardedup_keep.settle): chains are cut where a member is less similar than the
floor to its component's lead, and the released docs are clustered again among the survivors."""
import os
import shutil

import numpy as np

import neardedup as ND
import neardedup_lsh as NL
from nd_fixtures import Gen, jaccard, write_shard
from test_neardedup import chain, detected


def run(tmp_path, texts, name, **kw):
    d = os.path.join(tmp_path, name)
    os.makedirs(d, exist_ok=True)
    stem = write_shard(d, "s", texts)
    rep = NL.find_near_dups([{"stem": stem}], d, **kw)
    return NL.load_result(stem), rep


def test_plain_union_find_drops_a_whole_chain(tmp_path):
    docs = chain(Gen(71), n=12)
    r, rep = run(tmp_path, docs, "plain")
    assert r["keep"].tolist() == [1] + [0] * 11 and rep["edges"]["rounds"] == 1
    assert jaccard(docs[0], docs[-1]) < 0.4                     # yet the far end went


def test_chain_floor_keeps_one_doc_per_stretch_and_every_drop_is_close_to_its_lead(tmp_path):
    docs = chain(Gen(71), n=12)
    r, rep = run(tmp_path, docs, "floor", chain_floor=ND.VERIFY)
    kept = np.flatnonzero(r["keep"]).tolist()
    assert 3 <= len(kept) <= 6 and kept[0] == 0 and rep["edges"]["rounds"] >= 3
    assert rep["edges"]["released"] > 0 and rep["edges"]["unsettled"] == 0
    for i in np.flatnonzero(r["keep"] == 0):
        lead = int(r["lrow"][i])
        assert ND.est_jaccard(ND.signature(docs[i]), ND.signature(docs[lead])) >= ND.VERIFY
    for x in kept:                                              # no near-dup pair survives
        for y in kept:
            assert x == y or ND.est_jaccard(ND.signature(docs[x]),
                                            ND.signature(docs[y])) < ND.VERIFY
    assert all(a["sig_jaccard"] >= ND.VERIFY for a in rep["audit"])


def test_chain_floor_changes_nothing_without_chains(tmp_path):
    g, base, copy = Gen(72), [], []
    for t in (1.0, 0.9, 0.8, 0.75, 0.7):
        for _ in range(20):
            ws = g.words(g.r.randint(150, 400))
            base.append(" ".join(ws))
            copy.append(" ".join(g.near_copy(ws, t)))
    plain, _ = detected(str(tmp_path), base, copy, ND.VERIFY, "p")
    d = os.path.join(tmp_path, "f")
    shutil.copytree(os.path.join(tmp_path, "p"), d)
    stems = [os.path.join(d, "s0"), os.path.join(d, "s1")]
    rep = NL.find_near_dups([{"stem": s} for s in stems], d, chain_floor=ND.VERIFY)
    assert (~NL.load_result(stems[1])["keep"].astype(bool) == plain).all()
    assert rep["edges"]["released"] == 0 and rep["edges"]["rounds"] == 1


def test_chain_floor_result_does_not_depend_on_partitions(tmp_path):
    g, texts = Gen(73), []
    for _ in range(6):
        texts += chain(g, n=9, step=30)
    for _ in range(60):
        ws = g.words(200)
        texts += [" ".join(g.near_copy(ws, t)) for t in (1.0, 0.85)]
    order = np.random.default_rng(1).permutation(len(texts))
    texts = [texts[i] for i in order]
    a, _ = run(tmp_path, texts, "big", chain_floor=0.7, mem_mb=1024)
    b, rep = run(tmp_path, texts, "small", chain_floor=0.7, mem_mb=0.002)
    assert rep["partitions"] > 1 and a.tolist() == b.tolist()
