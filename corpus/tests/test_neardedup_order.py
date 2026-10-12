"""Near-dedup runs before boilerplate removal in extract.py (CORPUS 3.2 order), end to end.

Planted case (the acceptance test of the core v0 plan): one page body copied under 12 URLs. With
near-dedup first, 11 copies go and the body survives in one copy; in the v0 order (no near-dedup)
every body line is in 10+ pages, so boilerplate removal deletes the body from all 12 copies."""
import glob
import json
import os

import pytest

import extract
from fixtures import cp, write_jsonl
from nd_fixtures import Gen

STOP = "the of and to in is was for on that with as by at from it this be are".split()
NAV = "Home and About us and the Contact page for the news of the site that is here"


def eng(g, n):
    """English-shaped filler: stopwords alternate with pseudo-words (passes the stopword gate)."""
    return " ".join(g.r.choice(STOP) if i % 2 == 0 else g.r.choice(g.v) for i in range(n))


def page(g, lines=5, words=40):
    return "\n".join(eng(g, words) for _ in range(lines))


def write_tree(root):
    g = Gen(77)
    body = page(g, 6)
    copies = [cp(f"body{k}", f"the title of copy {g.r.choice(g.v)} is this one\n{body}",
                 f"20{19 + k // 6}-03-{12 - k % 6:02d}T00:00:00Z", url=f"example.org/body/{k}")
              for k in range(12)]
    fill = [cp(f"f{k}", page(g) + ("\n" + NAV if k < 12 else ""), f"2019-0{1 + k % 9}-01",
               url=f"example.org/f/{k}") for k in range(24)]
    exact = cp("f3-again", fill[3]["text"], "2020-02-02", url="example.org/elsewhere/3")
    write_jsonl(f"{root}/common-pile__cccc/CC-MAIN-2019-13/a.json.gz", copies[:6] + fill[:12])
    write_jsonl(f"{root}/common-pile__cccc/CC-MAIN-2020-05/b.json.gz",
                copies[6:] + fill[12:] + [exact])
    book = cp(7, fill[5]["text"] + "\n" + eng(g, 30), None, language="en", license="Public Domain")
    del book["created"]
    write_jsonl(f"{root}/common-pile__project_gutenberg/v0/documents/00000_pg.jsonl.gz", [book])
    return body


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    raw = str(tmp_path_factory.mktemp("raw"))
    return raw, write_tree(raw)


def extract_to(raw, out, **kw):
    stats = extract.run_extract(raw, out, **kw)
    recs = [json.loads(x) for p in sorted(glob.glob(f"{out}/*/*.jsonl")) for x in open(p)]
    return stats, recs


def drops(stats, src):
    return {k: v["docs"] for k, v in stats["sources"][src]["dropped"].items()}


def test_copied_body_survives_once_with_near_dedup_first(tree, tmp_path):
    raw, body = tree
    stats, recs = extract_to(raw, str(tmp_path / "on"))
    with_body = [r for r in recs if body.split("\n")[0] in r["text"]]
    assert len(with_body) == 1
    assert all(line in with_body[0]["text"] for line in body.split("\n"))
    assert with_body[0]["id"] == "cccc:body5"                   # earliest date: 2019-03-07
    c = drops(stats, "cccc")
    assert c["dup_near"] == 11 and c["dup_exact"] == 1
    assert drops(stats, "gutenberg") == {"dup_near": 1}         # the prose copy won (tier)
    assert not any(NAV in r["text"] for r in recs)               # real boilerplate still goes
    assert stats["sources"]["cccc"]["notes"]["boilerplate_lines"] == 1
    rep = json.load(open(tmp_path / "on" / "neardedup_report.json"))
    s19, s20 = "cccc:CC-MAIN-2019-13", "cccc:CC-MAIN-2020-05"   # labels: source and snapshot
    assert rep["dropped_by_leader_label"] == {f"{s19} <- {s19}": 5, f"{s20} <- {s19}": 7,
                                              f"gutenberg:documents <- {s19}": 1}
    assert {a["leader_id"] for a in rep["audit"]} == {"cccc:body5", "cccc:f3", "cccc:f5"}
    assert rep["largest"][0] == dict(rep["largest"][0], id="cccc:body5", size=12)


def test_v0_order_loses_the_copied_body(tree, tmp_path):
    raw, body = tree
    stats, recs = extract_to(raw, str(tmp_path / "off"), near_dedup=False)
    assert not any(line in r["text"] for r in recs for line in body.split("\n"))
    assert drops(stats, "cccc")["boilerplate_short"] == 12
    assert "dup_near" not in drops(stats, "cccc")


def test_parallel_run_is_identical_with_near_dedup(tree, tmp_path):
    raw, _ = tree
    s1, _ = extract_to(raw, str(tmp_path / "w1"), workers=1)
    s2, _ = extract_to(raw, str(tmp_path / "w2"), workers=2)
    assert [o["sha256"] for o in s1["outputs"]] == [o["sha256"] for o in s2["outputs"]]
    assert s1["sources"] == s2["sources"]
    assert os.path.exists(tmp_path / "w2" / "neardedup_report.json")
