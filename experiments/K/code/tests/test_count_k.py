"""count_k.py (K10 count report): supervised targets per K stream and arm, against the formats' own counts (bio 15
of 16, fact QA 2 of 10, K5 chats 4 of 30 or 41) and against the harness sources read item by item."""
import json
import os

import numpy as np
import pytest

import count_k as C
import gen
import kcommon as K

SMALL_FACT = ["--slots", "400000", "--n-high", "200", "--n-low", "50", "--e-bio", "20", "--e-qa", "4"]


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("kcount"))
    assert gen.main(["skill", d, "--slots", "1000000", "--workers", "2"]) == 0
    for arm in ("F0", "FH", "CAP"):
        assert gen.main(["fact", d, "--arm", arm, *SMALL_FACT]) == 0
    for arm in ("L0", "L100"):
        assert gen.main(["k5", d, "--arm", arm, "--slots", "400000", "--n5", "20", "--e5", "6"]) == 0
    return d


def harness_count(src, n_items):
    sup = tok = 0
    for _ in range(n_items):
        ids, flags = src.next_item()
        tok += len(ids)
        sup += int(flags[1:].sum())
    return tok, sup


def test_formats_and_kinds(root):
    rep = {a: C.count_arm(root, a) for a in ("F0", "FH", "L0", "L100")}
    for arm in ("F0", "FH"):
        bio, qa = (rep[arm]["sources"][n]["by_kind"] for n in ("fact_bio", "fact_qa"))
        for e in bio.values():
            assert e["tokens"] == 16 * e["items"] and e["supervised"] == 15 * e["items"]
        for e in qa.values():
            assert e["tokens"] == 10 * e["items"] and e["supervised"] == 2 * e["items"]
        assert set(bio) <= {"fb", "filler"} and set(qa) <= {"fb", "filler"}
    assert "fb" not in rep["F0"]["sources"]["fact_bio"]["by_kind"] and rep["FH"]["sources"]["fact_bio"]["by_kind"]["fb"]
    k0, k1 = (rep[a]["sources"]["fact_k5"]["by_kind"] for a in ("L0", "L100"))
    assert "lookup_hit" not in k0 and "lookup_miss" not in k1 and k0["lookup_miss"]["items"] == k1["lookup_hit"]["items"]
    assert k0["lookup_miss"]["tokens"] == 30 * k0["lookup_miss"]["items"]
    assert k1["lookup_hit"]["tokens"] == 41 * k1["lookup_hit"]["items"]
    for e in (k0["lookup_miss"], k1["lookup_hit"]):
        assert e["supervised"] == 4 * e["items"]
    for arm, r in rep.items():
        f = r["fact"]
        names = [n for n in r["sources"] if n.startswith("fact_")]
        assert f["supervised"] == sum(r["sources"][n]["total"]["supervised"] for n in names)
        assert "skill" in r["sources"] and f["tokens"] < f["tokens"] + r["sources"]["skill"]["total"]["tokens"]
        assert f["draw_tokens"] == pytest.approx(0.5 * K.B_SLOTS * K.FILL)
        assert f["draw_supervised_expected"] == pytest.approx(f["sup_share"] * f["draw_tokens"])
    assert rep["L0"]["fact"]["sup_share"] > rep["L100"]["fact"]["sup_share"]     # K5 critique: tool lines supervise 0


def test_counts_equal_the_harness_sources_item_by_item(root):
    import sys
    if K.HARNESS not in sys.path:
        sys.path.insert(0, K.HARNESS)
    from chat_template import ChatTemplate
    from sources import ChatJsonlSource, TokenShardSource, expand
    r = C.count_arm(root, "FH")
    blk = gen.data_block("FH", root)
    for s in blk["sources"]:
        if not s["paths"][0].startswith(root):
            continue
        t = r["sources"][s["name"]]["total"]
        paths = expand(s["paths"])
        if s["kind"] == "tokens":
            src = TokenShardSource(paths, s["eot_id"], 2048, 0, False)
        else:
            src = ChatJsonlSource(paths, ChatTemplate(dict(K.ROLE_IDS), K.END, "assistant"), 2048, None, 0, False)
        assert harness_count(src, t["items"]) == (t["tokens"], t["supervised"]), s["name"]
        assert src.epoch == 0                                # every item once: the next read starts epoch 1
        src.next_item()
        assert src.epoch == 1, s["name"]


def test_long_documents_are_cut_like_the_loader(tmp_path):
    ids = np.array([7] * 5000 + [K.EOT] + [8] * 10 + [K.EOT] + [9] * 3, dtype=np.uint16)
    p = str(tmp_path / "x.bin")
    ids.tofile(p)
    out = C.count_tokens([p], K.EOT, 2048, lambda t: "k")
    assert out["k"] == {"items": 3 + 1 + 1, "tokens": 5001 + 11 + 3, "supervised": 5001 - 3 + 10 + 2}
    import sys
    if K.HARNESS not in sys.path:
        sys.path.insert(0, K.HARNESS)
    from sources import TokenShardSource
    assert harness_count(TokenShardSource([p], K.EOT, 2048, 0, False), 5) == (5015, 5010)


def test_cli_writes_the_report(root, tmp_path):
    out = str(tmp_path / "c.json")
    assert C.main([root, "--arms", "F0", "L100", "--out", out]) == 0
    rep = json.load(open(out))
    assert set(rep["arms"]) == {"F0", "L100"} and not os.path.isabs(rep["root_name"])


def test_cap_counts_its_whole_draw_as_fact(root):
    r = C.count_arm(root, "CAP")
    assert "skill" not in r["sources"] and set(r["sources"]) == {"fact_bio", "fact_qa"}
    assert r["fact"]["draw_tokens"] == pytest.approx(K.B_SLOTS * K.FILL)


def test_edge_lengths_count_like_the_loader(tmp_path):
    """K3 round 15 (REVIEW 16b T-6 and its unreachable survivors): a token document of exactly max_len is one item,
    as the harness cuts it (mutant r16_count_pieces_off counted it as two), and a chat under 2 tokens is dropped, as
    ChatJsonlSource drops it (mutant r16_count_short_chat_kept counted it). No K stream holds either today (REVIEW
    16b), so each is read here on a file of its own, against the harness sources."""
    import sys
    if K.HARNESS not in sys.path:
        sys.path.insert(0, K.HARNESS)
    from sources import ChatJsonlSource, TokenShardSource
    ids = np.array([7] * 2047 + [K.EOT] + [8] * 2048 + [K.EOT], dtype=np.uint16)
    p = str(tmp_path / "x.bin")
    ids.tofile(p)
    out = C.count_tokens([p], K.EOT, 2048, lambda t: "k")
    assert out["k"] == {"items": 1 + 2, "tokens": 2048 + 2049, "supervised": 2047 + 2047}, out
    assert harness_count(TokenShardSource([p], K.EOT, 2048, 0, False), 3) == (4097, 4094)

    class Tmpl:
        def render(self, rec, encode=None):
            return list(range(3, 3 + rec["n"])), np.array([0] + [1] * (rec["n"] - 1), dtype=np.int64)
    q = tmp_path / "c.jsonl"
    q.write_text("".join(json.dumps({"n": n}) + "\n" for n in (1, 3, 2)))
    out = C.count_chat([str(q)], Tmpl(), 2048, {}, fixed="k")
    assert out == {"dropped": {"items": 1, "tokens": 1, "supervised": 0},
                   "k": {"items": 2, "tokens": 5, "supervised": 3}}, out
    assert harness_count(ChatJsonlSource([str(q)], Tmpl(), 2048, None, 0, False), 2) == (5, 3)
