"""Stack Exchange dump reader tests (readers_se_dump.py, se_stage.py, se_html.py, se_dump_run.py) on
the hand-made fixture in fixtures_se.py. Every planted case carries a *_MARKER string."""
import io
import json
import os
import shutil
import subprocess
import tracemalloc
from collections import Counter

import pytest

import readers_se_dump as R
import se_dump_run
import se_stage as S
import se_verify_mirror as V
from fixtures_se import comments, posts, write_7z, write_dump_dir, xml_bytes
from se_html import comment_to_text, html_to_text

MARKERS = ["POST_ANSWER_MARKER", "NEG_SCORE_MARKER", "EDITED_LATE_MARKER", "AIISM_ANSWER_MARKER",
           "NC_LICENSE_MARKER", "POST_QUESTION_MARKER", "LATE_QUESTION_ANSWER_MARKER",
           "NO_ANSWER_MARKER", "AIISM_QUESTION_MARKER", "ORPHANED_BY_QUESTION_MARKER",
           "TAG_WIKI_MARKER", "ORPHAN_MARKER", "GERMAN_SE_MARKER", "SHORT_MARKER",
           "BOUNDARY_ANSWER_MARKER", "POST_COMMENT_MARKER", "TEMPLATE_COMMENT_MARKER",
           "AIISM_COMMENT_MARKER", "PARENT_DROPPED_MARKER", "NC_COMMENT_MARKER",
           "<p>", "&amp;"]
SEVENZ = shutil.which("7zz") or shutil.which("7z") or shutil.which("7za")


@pytest.fixture(scope="module")
def dump(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("se") / "cooking.stackexchange.com")
    return d, write_dump_dir(d)


def run(path, **o):
    evs = list(R.read_se_dump("stackexchange", path, os.path.basename(path), o))
    keeps = {e[1]["id"]: e[1] for e in evs if e[0] == "keep"}
    drops = Counter(e[1] for e in evs if e[0] == "drop")
    notes = {e[1]: e[2] for e in evs if e[0] == "note"}
    return keeps, drops, notes


def same_output(a, b):
    """Equal apart from meta.file (the input's own name)."""
    for keeps in (a[0], b[0]):
        for r in keeps.values():
            r["meta"].pop("file")
    return a == b


Q1, Q21 = "stackexchange:cooking.stackexchange.com:1", "stackexchange:cooking.stackexchange.com:21"


def test_planted_cases_absent_and_threads_kept(dump):
    keeps, _, _ = run(dump[0])
    assert sorted(keeps) == [Q1, Q21]
    blob = json.dumps(list(keeps.values()), ensure_ascii=False)
    for m in MARKERS:
        assert m not in blob, m
    assert not any("https://" in t["text"] for r in keeps.values() for t in r["turns"])


def test_thread_order_and_turns(dump):
    r = run(dump[0])[0][Q1]
    assert [(t["role"], t["post_id"]) for t in r["turns"]] == [
        ("question", 1), ("comment", 1), ("answer", 3), ("comment", 8), ("answer", 2),
        ("comment", 6), ("answer", 9)]                  # accepted first, then by score
    assert [t.get("parent_id") for t in r["turns"]] == [None, 1, 1, 3, 1, 2, 1]
    assert r["text"].startswith("How do I keep bread fresh?\nI bake a loaf")
    assert r["text"] == "\n\n".join(t["text"] for t in r["turns"])
    assert r["turns"][0]["author"] == "101" and r["turns"][1]["author"] == "501"
    m = r["meta"]
    assert (m["n_answers"], m["n_comments"], m["url"]) == (
        3, 3, "https://cooking.stackexchange.com/questions/1")
    assert m["sharealike"] and m["author_type"] == "human" and m["replaces"] == R.REPLACES


def test_per_post_date_gate(dump):
    keeps, drops, notes = run(dump[0])
    assert drops["question_date_gate"] == 1                     # question 10 (2023) and its answer
    assert notes["answers_dropped_date_gate"] == 2              # answer 4 (2023), 23 (2022-12-01)
    assert notes["comments_dropped_date_gate"] == 1
    assert [t["post_id"] for t in keeps[Q21]["turns"]] == [21, 22]   # 2022-11-30T23:59:59.990 kept


def test_edit_gate():
    assert S.date_gate("2015-01-01T00:00:00", "2023-02-01T00:00:00") == "edit_gate"
    assert S.date_gate("2015-01-01T00:00:00", "2022-11-30T10:00:00") is None
    assert S.date_gate("2015-01-01T00:00:00", None) is None
    assert S.date_gate("2022-12-01T00:00:00", None) == "date_gate"
    assert S.date_gate(None, None) == "date_missing"


def test_edit_gate_in_thread(dump):
    assert run(dump[0])[2]["answers_dropped_edit_gate"] == 1


def test_license_per_post(dump):
    keeps, _, notes = run(dump[0])
    r = keeps[Q1]
    assert r["meta"]["licenses"] == ["cc-by-sa-2.5", "cc-by-sa-3.0"]
    assert r["meta"]["license"] == "cc-by-sa-3.0"
    old = next(t for t in r["turns"] if t["post_id"] == 9)
    assert (old["license"], old["license_basis"]) == ("cc-by-sa-2.5", "date_era")
    assert keeps[Q21]["meta"]["licenses"] == ["cc-by-sa-4.0"]
    assert keeps[Q21]["meta"]["license_basis"] == "post"
    assert notes["answers_dropped_license"] == 1 and notes["comments_dropped_license"] == 1


def test_license_eras():
    assert S.era_license("2011-04-07T23:59:59.999") == "cc-by-sa-2.5"
    assert S.era_license("2011-04-08T00:00:00.000") == "cc-by-sa-3.0"
    assert S.era_license("2018-05-01T23:59:59.000") == "cc-by-sa-3.0"
    assert S.era_license("2018-05-02T00:00:00.000") == "cc-by-sa-4.0"
    assert S.post_license("CC BY-SA 4.0", "2012-01-01") == ("cc-by-sa-4.0", "post")
    assert S.post_license("CC BY-NC-SA 4.0", "2012-01-01")[0] is None
    assert S.post_license(None, "2012-01-01") == ("cc-by-sa-3.0", "date_era")


def test_aiism_and_template_filters(dump):
    _, drops, notes = run(dump[0])
    assert drops["question_aiism"] == 1
    assert notes["answers_dropped_aiism"] == 1 and notes["comments_dropped_aiism"] == 1
    assert notes["comments_dropped_template"] == 1


def test_thread_rules_and_counts(dump):
    _, drops, notes = run(dump[0])
    assert drops["no_answer"] == 1 and drops["not_english"] == 1 and drops["too_short"] == 1
    assert notes["answers_dropped_low_score"] == 1
    assert notes["orphan_answers"] == 1 and notes["other_type_posts"] == 1
    assert notes["comments_dropped_parent_dropped"] == 1
    assert notes["replaces_common-pile_stackexchange"] == 1


def test_min_answers_zero_keeps_lone_question(dump):
    keeps = run(dump[0], se_min_answers=0)[0]
    assert "stackexchange:cooking.stackexchange.com:12" in keeps


def test_comments_off(dump):
    keeps = run(dump[0], se_comments=False)[0]
    assert all(t["role"] != "comment" for r in keeps.values() for t in r["turns"])
    assert "bread box?" not in keeps[Q1]["text"]


def test_html_to_text():
    t = html_to_text(
        "<p>A &amp; B  <a href='https://x.org'>link</a></p><blockquote><p>q1</p><p>q2</p>"
        "</blockquote><ul><li>one</li><li><p>two</p></li></ul><pre><code>a  b\n  c\n</code></pre>"
        "<p>x<br>y <img src='i.png' alt='pic'> z</p><table><tr><td>1</td><td>2</td></tr></table>")
    assert t == "A & B link\n> q1\n> q2\n- one\n- two\na  b\n  c\nx\ny z\n1 | 2"
    assert html_to_text("<ol><li>a<ol><li>b</li></ol></li></ol>") == "1. a\n  1. b"
    assert comment_to_text("see [this](https://a.b/c)  now") == "see this now"
    assert comment_to_text("**TL;DR** use `cans`, *really* __now__; 2 * 3 * 4") == (
        "TL;DR use cans, really now; 2 * 3 * 4")


@pytest.mark.parametrize("coder", ["lzma2", "lzma", "bzip2"])
def test_7z_input_matches_directory(dump, tmp_path, coder):
    d, files = dump
    p = str(tmp_path / "cooking.stackexchange.com.7z")
    write_7z(p, files, coder=coder)
    assert same_output(run(p), run(d))


@pytest.mark.skipif(not SEVENZ, reason="no 7z binary on PATH (the PC has none)")
def test_real_7zip_archive_matches_directory(dump, tmp_path):
    d, files = dump
    p = str(tmp_path / "cooking.stackexchange.com.7z")
    subprocess.run([SEVENZ, "a", "-bd", p] + [os.path.join(d, n) for n in files], check=True,
                   stdout=subprocess.DEVNULL)
    assert same_output(run(p), run(d))


def test_iter_rows_memory_is_bounded():
    body = "&lt;p&gt;" + "word " * 40 + "&lt;/p&gt;"
    rows = "".join(f'<row Id="{i}" Body="{body}"/>\n' for i in range(60000))
    data = ("<posts>\n" + rows + "</posts>").encode()
    tracemalloc.start()
    n = sum(1 for _ in S.iter_rows(io.BytesIO(data)))
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert n == 60000 and peak < 5e6, peak                    # uncleared tree: about 41 MB


def test_cli_writes_stats_and_shards(dump, tmp_path):
    out = str(tmp_path / "out")
    se_dump_run.main([out, dump[0], "--dataset", "archive.org/stackexchange_20221005",
                      "--dump-date", "2022-10-05"])
    st = json.load(open(os.path.join(out, "cooking.stackexchange.com.stats.json")))
    assert st["replaces"] == R.REPLACES and "replaces the Common Pile" in st["note"]
    assert st["docs_kept"] == 2 and st["docs_in"] == 7
    assert st["dropped"]["question_date_gate"]["docs"] == 1
    path = os.path.join(out, st["output"]["path"])
    assert se_dump_run.sha256_file(path) == st["output"]["sha256"]
    recs = [json.loads(x) for x in open(path, encoding="utf-8")]
    assert {r["meta"]["dataset"] for r in recs} == {"archive.org/stackexchange_20221005"}
    assert all(r["meta"]["dump_date"] == "2022-10-05" and len(r["meta"]["sha1"]) == 40
               for r in recs)
    assert {r["meta"]["file"] for r in recs} == {os.path.basename(os.path.dirname(dump[0]))
                                                 + "/cooking.stackexchange.com"}


def test_dataset_and_dump_date_from_raw_path():
    rel = "archive.org__stackexchange_20221005/beer.stackexchange.com.7z"
    assert R.dataset_of(rel, {}) == "archive.org/stackexchange_20221005"
    assert R.dump_date_of(rel, {}) == "2022-10-05"
    assert R.dump_date_of("archive.org__stackexchange/beer.stackexchange.com.7z", {}) is None
    assert R.dataset_of("beer.7z", {"se_dataset": "x/y"}) == "x/y"
    assert R.site_host("/a/b/beer.stackexchange.com.7z") == "beer.stackexchange.com"


def test_verify_mirror_flags_changed_bodies(dump, tmp_path):
    ps = posts()
    ps[1]["Body"] = "<p>Silently changed body.</p>"            # post 2, never edited: must differ
    ps[20]["Body"] = ps[20]["Body"].replace("example.com/x", "example.com/y")   # HTML-only change
    ps[2]["Body"], ps[2]["LastEditDate"] = "<p>Edited later.</p>", "2024-01-01T00:00:00.000"
    off = tmp_path / "official"
    off.mkdir()
    (off / "Posts.xml").write_bytes(xml_bytes("posts", ps))
    (off / "Comments.xml").write_bytes(xml_bytes("comments", comments()))
    rep = V.compare(V.load(dump[0]), V.load(str(off)), "2022-10-05")
    assert rep["Posts.xml"]["differ_ids"] == [2] and rep["Posts.xml"]["differ_html"] == 2
    assert not rep["faithful"]                                   # 1 of 16 compared posts > 0.5%
    assert rep["Posts.xml"]["changed_after_mirror"] >= 1 and rep["Comments.xml"]["differ"] == 0
    assert rep["Posts.xml"]["official_edit_gated_posts"] == 2       # posts 3 (2024 edit) and 6
    same = V.compare(V.load(dump[0]), V.load(dump[0]), "2022-10-05")
    assert same["faithful"] and same["Posts.xml"]["differ"] == 0


def test_verify_mirror_uses_the_snapshot_time():
    def tabs(posts):
        return {"Posts.xml": posts, "Comments.xml": {1: (b"c", "2022-09-01", None, None, b"c")}}
    mirror = tabs({1: (b"a", "2015-01-01", None, "1", b"a"),
                   2: (b"b", "2022-09-20", None, "2", b"b")})
    official = tabs({1: (b"A", "2015-01-01", "2022-09-27", "1", b"A"),    # edited after snapshot
                     2: (b"b", "2022-09-20", None, "2", b"b")})
    rep = V.compare(mirror, official, "2022-10-05")
    assert rep["as_of_used"] == "2022-09-20"
    assert rep["Posts.xml"]["changed_after_mirror"] == 1 and rep["Posts.xml"]["differ"] == 0
