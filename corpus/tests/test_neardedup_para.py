"""Optional paragraph-level dedup (neardedup_para.py)."""
import os

import numpy as np
import pytest

import neardedup_lsh as NL
import neardedup_para as PD
from nd_fixtures import Gen, write_shard


def write(d, name, texts, created=None, unit="line"):
    stem = write_shard(d, name, texts, created)
    w = PD.ParaSidecarWriter(stem, unit)
    for t in texts:
        w.add(t)
    w.close()
    return stem


@pytest.fixture()
def planted(tmp_path):
    g = Gen(91)
    rep = g.text(15)                        # a paragraph repeated across docs (15 words)
    inner = g.text(14)                      # repeated inside one doc only
    early = g.text(20)                      # first seen in a doc that near-dedup drops
    short = "thanks for reading this"       # 4 words: never a paragraph key
    doc = lambda *extra: "\n".join([g.text(60), *extra, g.text(60)])
    x = g.text(100) + "\n" + g.text(100)
    s1 = [doc(rep), x, doc(inner, inner), doc(short, rep)]
    s0 = [x + "\n" + early, doc(short), doc(rep, short)]   # s0[0]: near copy of s1[1], later
    s2 = [doc(early), doc(rep, g.text(12), rep)]
    stems = [write(tmp_path, "s0", s0, ["2019-01-01"] * 3),
             write(tmp_path, "s1", s1, ["2018-01-01"] * 4),
             write(tmp_path, "s2", s2, ["2017-01-01"] * 2)]
    shards = [{"stem": stems[0], "tier": 1}, {"stem": stems[1], "tier": 1},
              {"stem": stems[2], "tier": 2}]
    NL.find_near_dups(shards, str(tmp_path))
    return tmp_path, shards, s0, s1, s2, rep, short


def test_repeats_across_docs_keep_the_first_by_rank(planted):
    d, shards, s0, s1, s2, rep, _ = planted
    assert NL.load_result(shards[0]["stem"])["keep"].tolist() == [0, 1, 1]
    st = PD.find_para_dups(shards, str(d))
    drops = [PD.load_drops(s["stem"]) for s in shards]
    # rep: first by rank is s1[0] (tier 1, 2018); later copies: s0[2], s1[3], twice in s2[1].
    # early: its first copy is in s0[0], which near-dedup dropped, so s2[0] keeps it.
    assert drops[0] == {2: [1]} and drops[1] == {3: [2]} and drops[2] == {1: [1, 3]}
    assert st["dropped"] == 4 and st["top"][0]["later_copies"] == 4
    assert st["top"][0]["stem"] == shards[1]["stem"] and st["top"][0]["row"] == 0
    new = PD.drop_units(s2[1], drops[2][1])
    assert rep not in new and new.count("\n") == 2


def test_repeats_inside_one_doc_and_short_lines_stay(planted):
    d, shards, s0, s1, s2, rep, short = planted
    PD.find_para_dups(shards, str(d))
    drops = PD.load_drops(shards[1]["stem"])
    assert 2 not in drops and 0 not in drops                  # inner x2, and rep's first copy
    assert PD.para_keys(short) == [] and len(PD.para_keys(s1[2])) == 4
    assert all(i not in (1, 2) for i in PD.load_drops(shards[0]["stem"]).get(1, []))


def test_frozen_shards_get_no_drop_file_and_partitions_do_not_matter(planted):
    d, shards, *_ = planted
    base = PD.find_para_dups(shards, str(d))
    got = [np.load(s["stem"] + PD.PDROP_SUFFIX).tolist() for s in shards]
    for s in shards:
        os.remove(s["stem"] + PD.PDROP_SUFFIX)
    small = PD.find_para_dups(shards, str(d), mem_mb=0.0001)
    assert small["partitions"] > 1 and small["dropped"] == base["dropped"]
    assert [np.load(s["stem"] + PD.PDROP_SUFFIX).tolist() for s in shards] == got
    os.remove(shards[2]["stem"] + PD.PDROP_SUFFIX)
    shards[2]["frozen"] = True
    PD.find_para_dups(shards, str(d))
    assert not os.path.exists(shards[2]["stem"] + PD.PDROP_SUFFIX)


def test_block_units(tmp_path):
    g = Gen(92)
    para = g.text(12) + "\n" + g.text(12)   # one hard-wrapped paragraph
    a = "\n\n".join([g.text(30), para, g.text(30)])
    b = "\n\n".join([g.text(30), g.text(30), para])
    stems = [write(tmp_path, "a", [a], unit="block"), write(tmp_path, "b", [b], unit="block")]
    shards = [{"stem": s} for s in stems]
    NL.find_near_dups(shards, str(tmp_path))
    PD.find_para_dups(shards, str(tmp_path))
    assert PD.load_drops(stems[1]) == {0: [2]} and PD.load_drops(stems[0]) == {}
    assert PD.drop_units(b, [2], unit="block") == b.rsplit("\n\n", 1)[0]
    with pytest.raises(ValueError):
        PD.units(a, unit="word")


def test_many_repeats_match_a_reference_at_any_partition_count(tmp_path):
    g = Gen(93)
    paras = [g.text(12) for _ in range(60)]
    docs = [[g.text(50)] for _ in range(40)]
    where = {}
    for p, text in enumerate(paras):
        for d in g.r.sample(range(40), g.r.randint(2, 4)):
            where.setdefault(p, []).append((d, len(docs[d])))
            docs[d].append(text)
    texts = ["\n".join(x) for x in docs]
    stems = [write(tmp_path, f"s{k}", texts[k * 20:(k + 1) * 20]) for k in range(2)]
    shards = [{"stem": s} for s in stems]
    NL.find_near_dups(shards, str(tmp_path))
    want = sorted((d, i) for occ in where.values() for d, i in occ if d != min(occ)[0])
    for mem in (1024, 0.0005):
        st = PD.find_para_dups(shards, str(tmp_path), mem_mb=mem)
        got = sorted((k * 20 + r, i) for k, s in enumerate(stems)
                     for r, idx in PD.load_drops(s).items() for i in idx)
        assert got == want and st["dropped"] == len(want), mem
    assert st["partitions"] >= 16
