"""pg_clean.py: credit and note paragraphs go only at the two ends of a book."""
from collections import Counter

import pg_clean
from fixtures import prose


def test_head_window_is_bounded_by_share_of_text():
    paras = ["Produced by Jane Doe and PG Distributed Proofreaders"] + [prose(f"s{k}", 4) for k in range(3)]
    paras += ["KEEP_PG_BODY He read the e-book on the train and liked it."]
    paras += [prose(f"t{k}", 4) for k in range(3)]
    c = Counter()
    out = pg_clean.strip_front_back("\n\n".join(paras), c)
    assert "Produced by" not in out and "KEEP_PG_BODY" in out
    assert c["pg_head_paras"] == 1 and c["pg_tail_paras"] == 0


def test_tail_and_notes():
    paras = ["[Transcriber's Note: spelling kept.]", "THE TITLE"] + [prose(f"b{k}", 4) for k in range(30)]
    paras += ["End of the Project Gutenberg EBook of The Title, by A. Writer"]
    c = Counter()
    out = pg_clean.strip_front_back("\n\n".join(paras), c)
    assert out.startswith("THE TITLE") and out.endswith("(Note b29.)")
    assert c["pg_head_paras"] == 1 and c["pg_tail_paras"] == 1 and c["pg_matter_bytes"] > 90


def test_nothing_to_strip_is_identity():
    t = "\n\n".join(prose(f"n{k}", 4) for k in range(20))
    assert pg_clean.strip_front_back(t) == t
