"""ocr.py: line marking, reflow and the paragraph and book filters on planted cases."""
from collections import Counter

import pytest

import ocr
from fixtures_core import ocr_page

W = ocr.dictionary()


@pytest.mark.parametrize("s,want", [("59", True), ("[12]", True), ("xiv", True), ("XIV.", True),
                                    ("(7)", True), ("mild", False), ("Mill", False),
                                    ("59 A B", False), ("1810.", True), ("12345", False)])
def test_page_number_lines(s, want):
    assert ocr.is_page_number(s) is want


def test_reflow_drops_heads_numbers_stamps_and_rejoins_words():
    raw = "\n\n\n".join(ocr_page(k) for k in range(4)) + "\n\n[library of congress.]\n\n♦ • •\n"
    c = Counter()
    out = ocr.reflow(raw, W, c)
    assert "RHEAD_MARKER" not in out and "library of congress" not in out
    assert c["running_head"] == 4 and c["page_number"] == 4
    assert c["stamp"] == 1 and c["no_letter_line"] == 1 and c["dehyphenated"] > 0
    assert not any(x.strip().isdigit() for x in out.split("\n"))
    assert "-\n" not in out and "- " not in out           # no split word survives
    for s in ("kettle was still warm", "market.", "neighbor keeps a list"):
        assert s in out
    paras = out.split("\n\n")                              # each page ends a sentence on a
    assert len(paras) == 4 and all(p[0].isupper() for p in paras)   # short line: 4 paragraphs


def test_hyphen_rules():
    c = Counter()
    raw = ("the walls of the old house were painted with great care by the gar-\n"
           "dener and his friends, who came from the north to help in the self-\n"
           "control of the New-\n"
           "England troops who marched along the road that winter with the rest.")
    out = ocr.reflow(raw, W | {"self", "control"}, c)
    assert "gardener" in out and "self-control" in out and "New-England" in out
    assert "\n" not in out


def test_page_gap_join_needs_a_lower_case_start():
    full = "They appeared to conclude that their cavalry and the"
    raw = f"{full}\n{full} guns\n\n\n12\n\n\nand infantry were in a bad state.\n\n\nA new paragraph."
    out = ocr.reflow(raw, W)
    assert "guns and infantry" in out
    assert out.endswith("\n\nA new paragraph.")


def test_verse_keeps_its_lines():
    raw = ("The sad dark rose has dropped for years,\nFor centuries 'neath tyranny,\n"
           "Yet ever did her heart beat high\nIn hope that some day she'd be free ;")
    assert ocr.reflow(raw, W) == raw.replace(" ;", " ;")
    assert ocr.reflow(raw, W).count("\n") == 3


def test_paragraph_reasons():
    good = "We walked along the river and talked about what to cook for dinner that night."
    assert ocr.paragraph_reason(good, W) is None
    assert ocr.paragraph_reason("CHAPTER THE FIRST", W) is None           # short, has a word
    assert ocr.paragraph_reason("%^^W* ^^^^\"V ^'^iS^.* -^^ '^ \\^^%^.*", W) == "ocr_junk_symbols"
    assert ocr.paragraph_reason("o « b1 V ,4Q, b1 V x o t", W) in ("ocr_fragments", "ocr_junk_symbols")
    frag = "the garden V b1 x t e 4q L N and the river"          # no junk symbol, real words
    assert ocr.features(frag, W)[2] == 0 and ocr.paragraph_reason(frag, W) == "ocr_fragments"
    fraktur = "fanb lu'erbet Unterftmjung in fetuem greunbe nut bem eiuen tcbfyaften burd ttebenS"
    assert ocr.paragraph_reason(fraktur, W) == "ocr_nonwords"
    assert ocr.paragraph_reason("Gcfl Qxz", W) == "ocr_no_word"


def test_clean_book_drops_mostly_junk_and_keeps_prose():
    book = "\n\n\n".join(ocr_page(k) for k in range(5))
    text, c, why = ocr.clean_book(book, W)
    assert why is None and c["letters_kept"] > 0.9 * c["letters_in"]
    fraktur = "fanb lu'erbet Unterftmjung in fetuem greunbe nut bem eiuen tcbfyaften burd ttebenS"
    junk = "\n\n".join([fraktur] * 20 + [ocr_page(0)])
    assert ocr.clean_book(junk, W)[2] == "ocr_book_mostly_junk"
    symbols = "\n\n".join(["%^^W* ^^^^\"V ^'^iS^.* -^^ '^ ~~ ^^"] * 40 + [ocr_page(0)])
    text, c, why = ocr.clean_book(symbols, W)                 # letter-poor junk: removed, and
    assert why is None and "^^" not in text                   # the book's letters survive
    words_only = "\n\n".join(["Qzx vbnm trewq plokij uhygt frdews aqzsx wsxcde rfvbgt"] * 30)
    th = dict(ocr.THRESHOLDS, min_dict_rate=0.0, max_frag_rate=1.0, min_kept_letters=0.0)
    assert ocr.clean_book(words_only, W, th)[2] == "ocr_book_nonwords"


def test_thresholds_match_the_calibration_file():
    import json
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cal = json.load(open(os.path.join(here, "stats", "core_v0", "ocr_calibration.json")))
    th = cal["loc_after_filter"]["thresholds"]
    for k in ("min_dict_rate", "max_junk_rate", "max_frag_rate", "short_tokens",
              "min_kept_letters", "min_doc_dict_rate"):
        assert ocr.THRESHOLDS[k] == th[k], k


LABEL = ("Deacidified using the Bookkeeper process.\nNeutralizing agent: Magnesium Oxide\n"
         "Treatment Date: Dec. 2004\n\n* PreservationTechnologies\n\nA WORLD LEADER IN PAPER "
         "PRESERVATION\n\n1 1 1 Thomson Park Drive\nCranberry Township, PA 16066\n(724) 779-2111\n")


def test_preservation_label_is_cut_from_the_tail():
    book = "\n\n\n".join(ocr_page(k) for k in range(3))
    c = Counter()
    out = ocr.reflow(book + "\n\n\n\n" + LABEL, W, c)
    assert out == ocr.reflow(book, W) and c["preservation_label"] == 8
    garbled = LABEL.replace("Deacidified", "^ - V' F^^^f'^.'^ed").replace("Treatment", "'^eatment")
    cover = "\n\n".join(["^ <y", "P,", ">° ^ *", "V \\'"] * 40)       # back-cover junk after it
    out = ocr.reflow(book + "\n\n" + garbled.replace("PAPER", "COLLECTIONS") + "\n\n" + cover, W)
    assert "779-2111" not in out and "Neutralizing" not in out and "COLLECTIONS" not in out


def test_one_label_phrase_alone_cuts_nothing():
    index = "\n\n".join([ocr_page(0), "Cranberry Township, 45, 112.", "Cumberland, 16, 90."])
    out = ocr.reflow(index, W, Counter())
    assert "Cranberry Township, 45, 112." in out and "Cumberland, 16, 90." in out


def test_label_variants_count_as_phrases():
    book = "\n\n\n".join(ocr_page(k) for k in range(3))
    worn = "* PreservationTechnologles\nA WORLD LEADER IN COLLECTIONS PRESERVATION\n111 Tliomson Parle Dnve"
    out = ocr.reflow(book + "\n\n" + worn, W)
    assert out == ocr.reflow(book, W)
