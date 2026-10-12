"""readers_core.py end to end on the fixture files (fixtures_core.py): every planted MARKER is
dropped for the stated reason, every KEEP tag is kept with the stated provenance."""
import os
from collections import Counter

import pytest

import extract
import readers_core as R
import readers_cp as CP
from fixtures import cccc_docs, write_jsonl
from fixtures_core import write_core_raw


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("core_raw"))
    files = write_core_raw(root)
    o = dict(extract.DEFAULTS, gutenberg_cap=0, scratch_dir=root)   # what stream_work passes
    kept, drops, notes = {}, {}, Counter()
    for src, items in files.items():
        for path, rel in items:
            assert R.source_for(rel) == src
            for ev in R.read_core(src, path, rel, o):
                if ev[0] == "keep":
                    kept[ev[1]["id"]] = ev[1]
                elif ev[0] == "drop":
                    drops.setdefault(src, Counter())[ev[1]] += 1
                else:
                    notes[ev[1]] += ev[2]
    blob = "\n".join(r["text"] for r in kept.values())
    return kept, drops, notes, blob


def by_tag(kept, tag):
    hits = [r for r in kept.values() if tag in r["text"]]
    assert len(hits) == 1, tag
    return hits[0]


MARKERS = ["YT_POST_MARKER", "YT_GERMAN_MARKER", "YT_NC_MARKER", "NEWS_URLPOST_MARKER",
           "NEWS_LATE_MARKER", "NEWS_EDGE_MARKER", "NEWS_FOREIGN_URL_MARKER", "PB_POST_MARKER",
           "PB_NC_MARKER", "PB_BADDATE_MARKER", "OER_BADDATE_MARKER", "OER_POST_MARKER",
           "FOOD_POST_MARKER", "PDR_POST_MARKER", "OCRJUNK_MARKER", "LOCSTAMP_MARKER",
           "RHEAD_MARKER", "LOCJUNK_BOOK_MARKER", "LOC_POST_MARKER", "LOC_FRENCH_MARKER",
           "PGCREDIT_MARKER", "PGNOTE_MARKER", "PGTAIL_MARKER"]
KEEPS = ["KEEP_YT0", "KEEP_YT1", "KEEP_NEWS_URL", "KEEP_NEWS_UNDATED", "KEEP_NEWS_TEXTDATE",
         "KEEP_NEWS_NOYEAR", "KEEP_PB1", "KEEP_PB_TABLE", "KEEP_OER1", "KEEP_OER_SHIFTED",
         "KEEP_OER_NONE", "KEEP_FOOD0", "KEEP_FOOD_UNDATED", "KEEP_PDR_collection",
         "KEEP_PDR_essay", "KEEP_PG_MID"]


def test_every_marker_is_dropped_and_every_keep_is_kept(run):
    kept, _, _, blob = run
    for m in MARKERS:
        assert m not in blob, m
    for k in KEEPS:
        by_tag(kept, k)


def test_drop_reasons(run):
    _, drops, _, _ = run
    assert drops["youtube"] == Counter(date_gate=1, not_english=1, license=1)
    assert drops["news"] == Counter(date_gate=2, post_cutoff_marker=1, date_unparseable=1)
    assert drops["pressbooks"] == Counter(date_gate=1, license=1, date_unparseable=1)
    assert drops["oercommons"] == Counter(date_unparseable=1, date_gate=1)
    assert drops["foodista"] == Counter(date_gate=1)
    assert drops["pdr"] == Counter(date_gate=2)
    assert drops["loc"] == Counter(ocr_book_mostly_junk=1, date_gate=1, lang_field=1)


def test_provenance(run):
    kept = run[0]
    yt = by_tag(kept, "KEEP_YT0")
    assert yt["id"] == "youtube:vid0" and yt["meta"]["author_type"] == "human_speech_whisper_transcript"
    assert yt["meta"]["channel_id"] == "UC0" and yt["meta"]["license"] == "cc-by-4.0"
    assert yt["meta"]["date"] == "2016-10-09" and not yt["text"].startswith(" ")
    nu = by_tag(kept, "KEEP_NEWS_URL")
    assert nu["id"] == "news:globalvoices:1" and nu["meta"]["date_basis"] == "published:url"
    assert nu["meta"]["author"] == "A. Writer" and nu["meta"]["url"].startswith("https://global")
    assert by_tag(kept, "KEEP_NEWS_UNDATED")["meta"]["date_status"] == "undated"
    assert by_tag(kept, "KEEP_NEWS_TEXTDATE")["meta"]["date"] == "2021-12-13"
    assert by_tag(kept, "KEEP_NEWS_NOYEAR")["meta"]["date"] == "2013-07"
    oer = by_tag(kept, "KEEP_OER_SHIFTED")["meta"]
    assert oer["date_status"] == "undated" and oer["license"] == "public-domain"
    assert by_tag(kept, "KEEP_PB_TABLE")["meta"]["license"] == "cc0"
    pdr = by_tag(kept, "KEEP_PDR_essay")
    assert pdr["id"] == "pdr:essay:same-slug" and pdr["meta"]["sharealike"] is True
    assert by_tag(kept, "KEEP_PDR_collection")["id"] == "pdr:collection:same-slug"
    assert by_tag(kept, "KEEP_FOOD0")["meta"]["authors"] == ["B. Cook"]


def test_pressbooks_pipe_lines_removed(run):
    t = by_tag(run[0], "KEEP_PB_TABLE")["text"]
    assert all(x.strip().strip("|✓ ") for x in t.split("\n") if x.strip())
    assert run[2]["pipe_lines"] >= 4


def test_loc_book_is_reflowed(run):
    kept, _, notes, _ = run
    book = kept["loc:loc1"]
    assert book["meta"]["author_type"] == "human_ocr" and book["meta"]["year"] == 1890
    assert book["meta"]["date"] == "1890" and 0.5 < book["meta"]["ocr_letters_kept"] <= 1
    assert "kettle was still warm" in book["text"] and "-\n" not in book["text"]
    assert "0 014 012" not in book["text"]
    assert notes["ocr_running_head"] == 6 and notes["ocr_page_number"] == 6
    assert notes["ocr_dehyphenated"] > 0 and notes["ocr_junk_symbols"] >= 1


def test_gutenberg_whole_book_and_ends(run):
    kept, _, notes, _ = run
    pg = kept["gutenberg:77"]
    t = pg["text"]
    assert t.startswith("THE HOUSE BY THE LAKE") and t.endswith("(Note m39.)")
    assert "Project Gutenberg Etext of a quoted charter" in t      # mid-book: untouched
    assert len(t) > 6000 and notes["pg_head_paras"] == 2 and notes["pg_tail_paras"] == 1


def test_starter_sources_pass_through_unchanged(tmp_path):
    rows = cccc_docs()[:6]
    rel = "common-pile__cccc/CC-MAIN-2019-01/x.json.gz"
    write_jsonl(os.path.join(tmp_path, rel), rows)
    o = dict(extract.DEFAULTS, gutenberg_cap=0)
    got = list(R.read_core("cccc", os.path.join(tmp_path, rel), rel, o))
    want = list(CP.read_common_pile("cccc", os.path.join(tmp_path, rel), rel, o))
    assert got == want and len(got) == 6


def test_partial_options_and_reader_table():
    assert set(R.READERS) >= {"loc", "youtube", "news", "pressbooks", "oercommons", "foodista",
                              "pdr", "gutenberg", "cccc", "irc", "wikimedia"}
    assert R.source_for("common-pile__ubuntu_irc/raw/documents/00000_ubuntu.jsonl.gz") == "irc"
    assert R.source_for("archive.org__stackexchange_20221005/x.7z") is None
