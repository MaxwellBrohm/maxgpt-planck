"""Project Gutenberg front and back matter for whole books (core v0; the starter's 200 KB middle
window, a tokenizer-sample choice, is lifted, so the ends of a book now reach the corpus).

Common Pile has already cut the '*** START/END OF THE PROJECT GUTENBERG EBOOK' blocks (0 of 300
starter books carry them). What is left, read from 427 English starter books (2026-09-26): credit
lines at the head ('Produced by ... and the Online Distributed Proofreading Team at
http://www.pgdp.net', 'Transcribed by David Price, email ...'), old etext preambles ('December,
1971 [Etext #1]', 'The Project Gutenberg Etext of ...'), transcriber's notes, and 'End of Project
Gutenberg's <title>' at the tail.

strip_front_back(text) works on normalized text (paragraphs split at blank lines). In the first
HEAD_PARAS paragraphs that start within the first END_SHARE of the text, and in the last
TAIL_PARAS paragraphs that end within the last END_SHARE, it drops each paragraph that matches
CREDIT or starts with a transcriber's note. The share bound keeps a short book's body out of both
windows.
Nothing in the body of a book is touched: a compilation that quotes 'The Project Gutenberg Etext
of The Mayflower Compact' halfway through keeps it.
"""
import re

HEAD_PARAS, END_SHARE, TAIL_PARAS = 12, 0.05, 6
CREDIT = re.compile(
    r"project\s+gutenberg|gutenberg\.org|gutenberg\.net|pgdp\.net|distributed\s+proofread"
    r"|\be-?texts?\b|\be-?books?\b|\bproduced\s+by\b|\bprepared\s+by\b|\btranscribed\s+by\b"
    r"|\bscanned\s+by\b|\bhtml\s+version\b|\bthis\s+file\s+was\s+produced\b"
    r"|images?\s+(?:generously\s+)?(?:made\s+available|provided)|internet\s+archive"
    r"|\bproofreaders?\b|\bproofreading\s+team\b", re.I)
TRANSCRIBER = re.compile(r"^[\[\s_*]*transcriber['’]?s?\b", re.I)


def is_matter(p: str) -> bool:
    return bool(CREDIT.search(p) or TRANSCRIBER.match(p))


def strip_front_back(text: str, counts=None):
    """-> text without PG credit and note paragraphs at its two ends. counts (a dict or Counter)
    gets pg_head_paras, pg_tail_paras and pg_matter_bytes."""
    paras = text.split("\n\n")
    total, pos, drop = max(1, len(text)), 0, set()
    for i, p in enumerate(paras[:HEAD_PARAS]):
        if pos > END_SHARE * total:
            break
        if is_matter(p):
            drop.add(i)
        pos += len(p) + 2
    head, pos = len(drop), 0
    for i in range(len(paras) - 1, max(len(paras) - TAIL_PARAS, 0) - 1, -1):
        if pos > END_SHARE * total:
            break
        if is_matter(paras[i]):
            drop.add(i)
        pos += len(paras[i]) + 2
    if counts is not None and drop:
        counts["pg_head_paras"] = counts.get("pg_head_paras", 0) + head
        counts["pg_tail_paras"] = counts.get("pg_tail_paras", 0) + len(drop) - head
        counts["pg_matter_bytes"] = counts.get("pg_matter_bytes", 0) + sum(
            len(paras[i].encode("utf-8")) + 2 for i in drop)
    return "\n\n".join(p for i, p in enumerate(paras) if i not in drop).strip()
