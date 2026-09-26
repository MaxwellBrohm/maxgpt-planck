"""Unit tests: gates, hygiene, the OOD-H reserve function and the OASST2 path rule."""
import hashlib

import pytest

import hygiene as H
from oodh import in_oodh_reserve
from readers_chat import best_path, message_problem
from readers_cp import clean_irc, middle_window
from fixtures import msg


def test_date_gate_boundaries_and_formats():
    assert H.date_status("2022-11-30T23:59:59.000Z") == "ok"
    assert H.date_status("2022-12-01T00:00:00") == "post_cutoff"
    assert H.date_status("2022-12-01") == "post_cutoff"
    assert H.date_status("2013-05-18T05:33:58.000Z") == "ok"
    assert H.date_status("2015-04-17") == "ok"
    assert H.date_status("2024-06-11T03:07:42") == "post_cutoff"
    assert H.date_status(None) == "undated"
    assert H.date_status("") == "undated"
    assert H.date_status("sometime in 2019") == "unparseable"
    assert H.date_status("2019-13-01") == "unparseable"


@pytest.mark.parametrize("raw,want", [
    ("Public Domain", "public-domain"),
    ("Creative Commons - Attribution Share-Alike - https://creativecommons.org/licenses/by-sa/3.0/",
     "cc-by-sa-3.0"),
    ("https://creativecommons.org/licenses/by/4.0/", "cc-by-4.0"),
    ("https://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
    ("CC-BY-SA-4.0", "cc-by-sa-4.0"), ("cc0", "cc0"), ("Apache-2.0", "apache-2.0"),
    ("Creative Commons - Attribution NonCommercial - https://creativecommons.org/licenses/by-nc/4.0/",
     None),
    ("https://creativecommons.org/licenses/by-nd/4.0/", None),
    ("https://creativecommons.org/licenses/by-nc-sa/3.0/", None),
    ("cc-by-nc-4.0", None), ("Creative Commons - Attribution", None),
    ("GPL-3.0", None), ("", None), (None, None),
])
def test_license_id(raw, want):
    assert H.license_id(raw) == want


def test_normalize_line_endings_and_blanks():
    assert H.normalize("a  \r\nb\rc\n\n\n\nd\x00 ") == "a\nb\nc\n\nd"
    assert H.normalize("é") == "é"          # NFC


def test_hygiene_reasons_in_order():
    en = "We walked along the river and talked about what to cook for dinner. " * 4
    assert H.hygiene_reason("") == "empty"
    assert H.hygiene_reason("short text") == "too_short"
    assert H.hygiene_reason("Мы ездили на поезде в город и навестили друзей. " * 6) == "non_ascii"
    assert H.hygiene_reason("vi ska äta middag klockan sju och sedan gå hem " * 6) == "not_english"
    assert H.hygiene_reason(en) is None
    assert H.hygiene_reason("vi ska äta middag klockan sju och sedan gå hem " * 6,
                            min_stopwords=0) is None


def test_oodh_reserve_matches_the_prereg_rule():
    for t in ("002c4715-b026-48d1-8d19-3f724a9fc1e8", "fixture-tree-0000", "x"):
        h = int(hashlib.sha256(("planck-oodh-v1:" + t).encode()).hexdigest(), 16)
        assert in_oodh_reserve(t) == (h % 8 == 0)
    ids = [f"tree-{i}" for i in range(8000)]
    rate = sum(map(in_oodh_reserve, ids)) / len(ids)
    assert 0.11 < rate < 0.14                        # about 1 in 8
    assert [in_oodh_reserve(t) for t in ids] == [in_oodh_reserve(t) for t in ids]
    with pytest.raises(ValueError):
        in_oodh_reserve("")


def test_message_problem_flags():
    assert message_problem(msg("m", "assistant", "hi")) is None
    assert message_problem(msg("m", "assistant", "hi", deleted=True)) == "deleted"
    assert message_problem(msg("m", "assistant", "hi", synthetic=True)) == "synthetic"
    assert message_problem(msg("m", "assistant", "hi", model_name="x")) == "synthetic"
    assert message_problem(msg("m", "assistant", "hi", review_result=False)) == "review_failed"
    assert message_problem(msg("m", "assistant", "hi",
                               labels={"spam": {"value": 0.5, "count": 2}})) == "spam"
    assert message_problem(msg("m", "assistant", "hi", lang="de")) == "lang_mismatch"
    assert message_problem(msg("m", "assistant", "  ")) == "empty"
    assert message_problem(msg("m", "assistant", "As an AI language model, I can't.")) == "aiism"
    assert message_problem(msg("m", "prompter", "Is ChatGPT better than you?")) == "aiism"


def test_best_path_takes_best_usable_rank_and_ends_on_assistant():
    leaf_user = msg("U9", "prompter", "dangling question", rank=0)
    a_best = msg("A1", "assistant", "best", rank=1, replies=[leaf_user])
    root = msg("R", "prompter", "q", replies=[
        msg("A0", "assistant", "deleted", rank=0, deleted=True), a_best,
        msg("A2", "assistant", "worse", rank=2), msg("A3", "assistant", "unranked")])
    assert [m["message_id"] for m in best_path(root)] == ["R", "A1"]
    root2 = msg("R", "prompter", "q", replies=[msg("A3", "assistant", "unranked"),
                                                msg("A2", "assistant", "ranked", rank=2)])
    assert [m["message_id"] for m in best_path(root2)] == ["R", "A2"]
    assert best_path(msg("R", "prompter", "q")) == []


def test_clean_irc_and_middle_window():
    raw = "[13:39] <dark-sun> hello there\n=== a is now known as b\n[13:51]  * bob waves\n\n"
    assert clean_irc(raw) == "P1: hello there\n* P2 waves"
    text = "\n\n".join(f"paragraph {i} " + "x" * 80 for i in range(100))
    w = middle_window(text, 1000)
    assert len(w.encode()) <= 1000 and w.startswith("paragraph ") and "paragraph 0 " not in w
    assert middle_window("short", 1000) == "short"
