"""End-to-end extractor tests on the hand-made fixture tree (fixtures.py)."""
import glob
import json
import os
import re

import pytest

import extract
from fixtures import IRC_NICK, write_raw
from oodh import in_oodh_reserve

MARKERS = ["DATEGATE_POST_MARKER", "BOUNDARY_DROP_MARKER", "BADDATE_MARKER", "LICENSE_NC_MARKER",
           "ALL_LICENSES_MARKER", "SE_POST_MARKER", "NO_LICENSE_MARKER", "IRC_POST_MARKER",
           "TEMPLATE_MARKER", "WIKI_POST_MARKER", "WIKI_NC_MARKER", "DUP_ID_MARKER",
           "FRENCH_MARKER", "PG_HEADER_MARKER", "PG_LICENSE_MARKER", "SPANISH_MARKER",
           "STATE_MARKER", "NOREPLY_MARKER", "DELETED_MARKER", "SPAM_MARKER",
           "LOWER_RANK_MARKER", "OODH_RESERVED_MARKER", "COMPONENT_NONE_MARKER", "AIISM_MARKER",
           "AIISM_PROMPT_MARKER", "AIISM_DOLLY_MARKER", "SE_LATE_MARKER", "SE_YEAR_MARKER",
           "WIKI_TALK_MARKER", "WIKI_USERTALK_MARKER", "WIKI_USER_MARKER", "WIKI_NS102_MARKER",
           "WIKI_YEAR_MARKER", "BOILERPLATE_MARKER", "WIKI_BANNER_MARKER", "User: ", "Assistant: "]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    raw = str(tmp_path_factory.mktemp("raw"))
    facts = write_raw(raw)
    out = str(tmp_path_factory.mktemp("out"))
    # near_dedup=False: fixture docs recombine 16 sentences, so MinHash would merge most of them;
    # near-dedup and its place before boilerplate removal are tested in test_neardedup_order.py
    stats = extract.run_extract(raw, out, workers=1, gutenberg_cap=2000, near_dedup=False)
    recs = [json.loads(line) for p in sorted(glob.glob(f"{out}/*/*.jsonl")) for line in open(p)]
    return raw, out, stats, recs, facts


def drops(stats, src):
    return {k: v["docs"] for k, v in stats["sources"][src]["dropped"].items()}


def test_every_planted_case_is_absent_from_output(run):
    _, out, _, recs, _ = run
    blob = "".join(open(p, encoding="utf-8").read() for p in glob.glob(f"{out}/*/*.jsonl"))
    assert len(recs) > 100
    for m in MARKERS:
        assert m not in blob, m
    assert IRC_NICK.lower() not in blob.lower()


def test_date_gate(run):
    _, _, stats, recs, _ = run
    ids = {r["id"] for r in recs}
    assert drops(stats, "cccc")["date_gate"] == 2              # 2023 doc + the 2022-12-01 edge
    assert drops(stats, "cccc")["date_unparseable"] == 1
    assert "cccc:c-edge-ok" in ids                             # 2022-11-30T23:59:59 kept
    assert drops(stats, "stackexchange")["date_gate"] == 1
    assert drops(stats, "irc")["date_gate"] == 1
    assert drops(stats, "wikimedia")["date_gate"] == 1
    assert stats["sources"]["gutenberg"]["undated_kept"] == 4  # no date field: kept, counted
    assert stats["sources"]["dolly"]["undated_kept"] == 23


def test_license_filter_and_record(run):
    _, _, stats, recs, _ = run
    assert drops(stats, "stackexchange")["license"] == 3       # NC, ND in all_licenses, missing
    assert drops(stats, "stackexchange")["license_component_missing"] == 1   # 'None' in all_licenses
    assert drops(stats, "wikimedia")["license"] == 1
    by = {}
    for r in recs:
        by.setdefault(r["source"], set()).add((r["meta"]["license"], r["meta"]["license_basis"]))
    assert by["stackexchange"] == {("cc-by-sa-3.0", "document")}
    assert by["wikimedia"] == {("cc-by-sa-4.0", "document")}
    assert by["irc"] == {("public-domain", "document")}
    assert by["gutenberg"] == {("public-domain", "document")}
    assert by["cccc"] == {(None, "unrecorded")}
    assert by["oasst2"] == {("apache-2.0", "dataset")}
    assert by["dolly"] == {("cc-by-sa-3.0", "dataset")}


def test_oodh_reserve_never_reaches_output(run):
    _, _, stats, recs, facts = run
    oasst = [r for r in recs if r["source"] == "oasst2"]
    assert len(oasst) == 35
    assert not any(in_oodh_reserve(r["meta"]["tree_id"]) for r in oasst)
    assert not {r["meta"]["tree_id"] for r in oasst} & set(facts["reserved_trees"])
    assert drops(stats, "oasst2")["oodh_reserve"] == 3
    assert stats["sources"]["oasst2"]["notes"]["oodh_reserved_trees"] == 3


def test_oasst_rules(run):
    _, _, stats, recs, facts = run
    d = drops(stats, "oasst2")
    assert d["lang"] == 1 and d["tree_state"] == 1 and d["no_reply"] == 1 and d["aiism"] == 1
    t0 = next(r for r in recs if r["id"] == f"oasst2:{facts['ok_trees'][0]}")
    assert t0["meta"]["message_ids"] == [facts["ok_trees"][0], "A1", "P3", "A3"]
    assert [t["role"] for t in t0["turns"]] == ["user", "assistant", "user", "assistant"]
    assert t0["text"] == "\n".join(t["text"] for t in t0["turns"])      # template-shaped, no role text
    assert t0["meta"]["n_user_turns"] == 2
    t37 = next(r for r in recs if r["id"] == f"oasst2:{facts['ok_trees'][37]}")
    assert t37["meta"]["message_ids"] == [facts["ok_trees"][37], "AI2"]   # AI-ism reply skipped
    assert stats["sources"]["oasst2"]["notes"]["messages_aiism"] == 2


def test_dolly_rows_and_split_key(run):
    _, _, stats, recs, _ = run
    dolly = [r for r in recs if r["source"] == "dolly"]
    assert drops(stats, "dolly") == {"too_short": 1, "aiism": 1}
    assert all([t["role"] for t in r["turns"]] == ["user", "assistant"] for r in dolly)
    keys = [r["meta"]["split_key"] for r in dolly]
    shared = [k for k in keys if keys.count(k) > 1]
    assert len(shared) == 3 and len(set(shared)) == 1 and shared[0].startswith("ctx:")
    assert "dolly:0" in keys                                         # no context: the row id


def test_hygiene_and_dedup(run):
    _, _, stats, recs, _ = run
    c = drops(stats, "cccc")
    assert c["not_english"] == 1 and c["non_ascii"] == 1 and c["too_short"] == 1
    assert c["empty"] == 1 and c["dup_exact"] == 1             # CRLF copy of c0
    assert drops(stats, "gutenberg")["dup_exact"] == 1         # cross-source copy of c0
    assert drops(stats, "gutenberg")["lang_field"] == 1
    assert drops(stats, "wikimedia")["dup_id"] == 1
    assert drops(stats, "wikimedia")["wiki_namespace"] == 3        # template, user, voyage 102
    assert drops(stats, "wikimedia")["wiki_talk"] == 2
    assert "wikimedia:wikibooks.com:102-6" in {r["id"] for r in recs}  # Cookbook kept
    assert drops(stats, "irc") == {"too_short": 1, "not_english": 1, "date_gate": 1, "empty": 1}
    assert len({r["meta"]["sha1"] for r in recs}) == len(recs)
    assert len({r["id"] for r in recs}) == len(recs)


def test_irc_cleanup_and_gutenberg_cap(run):
    _, _, _, recs, _ = run
    irc = [r["text"] for r in recs if r["source"] == "irc"]
    assert len(irc) == 12
    assert all("[13:" not in t and "===" not in t and "\nP2: " in t for t in irc)
    assert not any(re.search(r"nick\d", t) for t in irc)                # speakers and mentions
    planted = [t for t in irc if "P4: " in t]
    assert len(planted) == 4 and all("P4: " in t and "P3: P4: " in t and "* P4 nods at P3" in t
                                     for t in planted)
    assert all("\nP5: If you want the garden to grow" in t for t in planted)   # word nick
    books = [r for r in recs if r["source"] == "gutenberg"]
    assert len(books) == 4
    assert all(len(r["text"].encode()) <= 2000 and r["meta"]["capped"] for r in books)


def test_stats_accounting_and_shard_hashes(run):
    _, out, stats, recs, _ = run
    for s, st in stats["sources"].items():
        dropped = sum(v["docs"] for v in st["dropped"].values())
        assert st["docs_in"] == st["docs_kept"] + dropped, s
        assert st["bytes_kept"] == sum(len(r["text"].encode()) for r in recs if r["source"] == s)
    import hashlib
    for o in stats["outputs"]:
        assert hashlib.sha256(open(os.path.join(out, o["path"]), "rb").read()).hexdigest() == o["sha256"]


def test_parallel_run_is_identical(run, tmp_path):
    raw, _, stats, _, _ = run
    s2 = extract.run_extract(raw, str(tmp_path), workers=2, gutenberg_cap=2000, near_dedup=False)
    assert [o["sha256"] for o in s2["outputs"]] == [o["sha256"] for o in stats["outputs"]]
    assert s2["sources"] == stats["sources"]


def test_post_cutoff_markers_and_boilerplate(run):
    _, out, stats, recs, _ = run
    assert drops(stats, "stackexchange")["post_cutoff_marker"] == 2
    assert drops(stats, "wikimedia")["post_cutoff_marker"] == 1
    c = drops(stats, "cccc")
    assert c["boilerplate_short"] == 1 and "post_cutoff_marker" not in c
    by = {s: [r["text"] for r in recs if r["source"] == s] for s in ("cccc", "stackexchange")}
    assert sum("NEAR_BOILER" in t for t in by["cccc"]) == 9          # under the threshold: kept
    assert sum("DUP_GROUP" in t for t in by["cccc"]) == 9            # a crawl copy counts once
    assert sum("RECRAWL_LINE" in t for t in by["cccc"]) == 10        # one URL, ten snapshots
    assert sum("SE_REPEAT" in t for t in by["stackexchange"]) == 11  # SE is not stripped
    assert stats["sources"]["cccc"]["notes"]["boilerplate_lines"] == 1
    assert stats["sources"]["wikimedia"]["notes"]["boilerplate_lines"] == 1
    assert stats["sources"]["cccc"]["bytes_boilerplate"] > 10 * 150
    stripped = [r for r in recs if r["meta"].get("boilerplate_bytes")]
    assert len(stripped) == 20 and all(r["meta"]["sha1"] == __import__("hygiene").dedup_key(
        r["text"]) for r in stripped)
    s2 = extract.run_extract(run[0], out + "_nobp", gutenberg_cap=2000, boilerplate_min_docs=0,
                             near_dedup=False)
    assert "boilerplate_short" not in {k for k in s2["sources"]["cccc"]["dropped"]}
