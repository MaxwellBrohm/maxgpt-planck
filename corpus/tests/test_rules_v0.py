"""Unit tests for the rules added after the v0 review: the AI-ism filter, post-cutoff markers, the
IRC nickname scrub, the Wikimedia namespace allow list and boilerplate line removal."""
import numpy as np
import pytest

import boilerplate as BP
import hygiene as H
from irc_clean import clean_irc, common_words
from readers_cp import wiki_ns_reason

AIISMS = ["As an AI language model, I cannot.", "as an AI, I can't tell you which to prefer",
          "I'm just an AI", "I am a large language model trained on text",
          "a language model developed by a lab", "I was trained by OpenAI",
          "I don’t have personal opinions on that", "I do not have personal beliefs",
          "My knowledge cutoff is 2021.", "as of my last update", "I'm Open Assistant",
          "ChatGPT wrote this", "chat gpt", "GPT-4 is", "gpt-3.5-turbo", "OpenAI's model"]
HUMAN = ["I asked my aide to help.", "the open air market", "As an aid to memory, write it down.",
         "My personal opinion is that the fence is fine.", "GPT partitions on the disk",
         "Language models of the 1990s were n-gram counts.", "I am an avid reader."]


@pytest.mark.parametrize("text", AIISMS)
def test_aiism_hits(text):
    assert H.aiism(text)


@pytest.mark.parametrize("text", HUMAN)
def test_aiism_leaves_human_text(text):
    assert H.aiism(text) is None and H.post_cutoff_marker(text) is None


def test_post_cutoff_marker_years():
    for t in ["obvious now in 2024", "(2023)", "on 2025-01-03 it", "from 2019-2023 the"]:
        assert H.post_cutoff_marker(t), t
    for t in ["in 2022 it", "port 12023", "1.2024 m", "v2023x", "the year 2019", "2,023 cars"]:
        assert H.post_cutoff_marker(t) is None, t
    assert H.post_cutoff_marker("As an AI language model") == "as an ai"


COMMON = frozenset(["help", "garden", "the", "try", "this"])


def test_irc_labels_are_stable_per_document_and_case_insensitive():
    raw = ("[10:00] <LinDol> hi all\n[10:01] <helper9> lindol: try this\n"
           "[10:02] <lindol> thanks helper9\n[10:03]  * LINDOL waves\n=== x is now known as y\n")
    assert clean_irc(raw, COMMON) == ("P1: hi all\nP2: P1: try this\nP1: thanks P2\n* P1 waves")


def test_irc_mentions_keep_words_short_nicks_and_wordlike_nicks():
    raw = "\n".join(["[1:00] <garden> the garden is wet", "[1:01] <ab> ab cd",
                     "[1:02] <sudo> " + "sudo " * 9 + "is the command",
                     "[1:03] <bob7> garden, ab, sudo and bob7"])
    out = clean_irc(raw, COMMON).split("\n")
    assert out[0] == "P1: the garden is wet"                   # common word: mentions kept
    assert out[1] == "P2: ab cd"                                # under 3 characters
    assert out[2].startswith("P3: sudo sudo")                   # said 10 x, spoke 1: a word
    assert out[3] == "P4: garden, ab, sudo and P4"


def test_irc_nick_tokens_match_whole_nicknames_only():
    raw = "[1:00] <zorb> hi\n[1:01] <kim5> zorbs and zorb-x and zorb, zorb_ ok"
    assert clean_irc(raw, COMMON).split("\n")[1] == "P2: zorbs and zorb-x and P1, zorb_ ok"


def test_common_word_list_is_real_and_required(tmp_path):
    words = common_words()
    assert len(words) > 20000 and {"help", "the", "garden", "kernel"} <= words
    assert not {"ubottu", "zorblaxquint"} & words
    small = tmp_path / "w.txt"
    small.write_text("# header\nhelp\n")
    with pytest.raises(ValueError):
        common_words(str(small))
    with pytest.raises(FileNotFoundError):
        common_words(str(tmp_path / "missing.txt"))


def test_wiki_namespace_allow_list():
    assert wiki_ns_reason({"namespace": "0", "wiki": "wikinews.com"}) is None
    assert wiki_ns_reason({"namespace": 0, "wiki": "wikinews.com"}) is None
    assert wiki_ns_reason({"namespace": "102", "wiki": "wikibooks.com"}) is None
    assert wiki_ns_reason({"namespace": "110", "wiki": "wikibooks.com"}) is None
    assert wiki_ns_reason({"namespace": "102", "wiki": "wikinews.com"}) == "wiki_namespace"
    for ns in ("1", "3", "5", "111"):
        assert wiki_ns_reason({"namespace": ns, "wiki": "wikibooks.com"}) == "wiki_talk"
    for ns in ("2", "4", "10", "90", None):
        assert wiki_ns_reason({"namespace": ns, "wiki": "wikibooks.com"}) == "wiki_namespace"


def test_boilerplate_threshold_and_strip():
    docs = [f"unique line {i}\nshared footer\n\nrare line" if i < 3 else
            f"unique line {i}\nshared footer" for i in range(10)]
    h = np.array([x for d in docs for x in BP.doc_hashes(d)], dtype=np.uint64)
    bad = BP.frequent_lines(h, 10)
    assert bad == {BP.line_hash("shared footer")}
    assert BP.frequent_lines(h, 11) == frozenset() and BP.frequent_lines(h, 0) == frozenset()
    assert BP.strip_boilerplate(docs[0], bad) == "unique line 0\n\nrare line"
    assert BP.strip_boilerplate("  shared footer  \nkeep", bad) == "keep"
    assert len(BP.doc_hashes("a\na\n\n  a ")) == 1
    assert BP.line_hash("x") == BP.line_hash("x") and 0 <= BP.line_hash("x") < 2 ** 64


def test_url_key():
    assert BP.url_key("https://www.Example.org/a/#x") == "example.org/a"
    assert BP.url_key("http://example.org/a") == BP.url_key("example.org/a/")
    assert BP.url_key("https://example.org/a?b=1") != BP.url_key("https://example.org/a?b=2")
    assert BP.url_key(None) == "" and BP.url_key("") == ""
