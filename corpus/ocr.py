"""Library of Congress books: OCR reflow and an OCR-quality filter (core v0, tier 4).

The LoC text is the djvu OCR of scanned books (read from the real files, 2026-09-26): hard line
breaks (mean line about 32 characters), words split at line ends with '-' or the OCR's '¬', page
numbers and running heads as their own lines between pages, library stamps, and pages of symbol
junk where the scanner read a cover, a plate or a stain. A document-level word rate does not
separate LoC from Project Gutenberg (p10 0.837 vs 0.840), so the filter works per paragraph.

reflow(raw) runs on the raw text (before hygiene.normalize, which would erase page gaps):
1. drop page-number lines (digits or a well-formed roman numeral, optionally bracketed), lines
   with no letter, short lines naming the Library of Congress, and the book's tail from a 2004-9
   preservation label on ('Deacidified using the Bookkeeper process ... Cranberry Township, PA',
   about 1 book in 12), when 2+ distinct label phrases occur in its last STAMP_TAIL lines;
2. drop running heads: a short standalone line (blank lines around it) that holds a digit-bearing
   token ('THOUGHTS OF JUNE 37', 'J3g A GENERAL HISTORY OF THE') or sits next to a page-number
   line, and whose letter key (words of 3+ letters without digits) recurs in the book;
3. rejoin lines. A line continues onto the next when it ends in letter + hyphen (rejoined without
   the hyphen unless both halves are words and the whole is not; 'New-England' before a capital)
   or reaches the text width (0.75 x the p80 line length); also across a page gap when the next
   block starts in lower case. Short lines keep their break (verse stays verse); in a block where
   most lines start with a capital, only hyphen breaks and lower-case continuations are joined.
filter_paragraphs() then drops paragraphs that read as OCR junk (thresholds below, calibrated on
PG paragraphs), and clean_book() drops the whole book when too little survives.
"""
import re
from collections import Counter

from irc_clean import common_words

# Thresholds, set by ocr_calibrate.py on the PC (2026-09-26; stats/core_v0/ocr_calibration.json)
# from 300 English PG books (124 MB, the clean reference) and 266 LoC books of file 00000. Each
# rule takes the strictest grid value costing PG at most 0.25% more bytes than its loosest value
# (PG always loses its '|'/'+' tables, formulas, Latin). Together the paragraph rules drop 1.63%
# of PG bytes and 5.61% of LoC bytes. Book floor: PG's p1 book dictionary rate (0.7558 -> 0.75).
MIN_DICT_RATE = 0.50      # share of [a-z]{2,} tokens in the dictionary, paragraphs of 8+ tokens
MAX_JUNK_RATE = 0.02      # share of non-space chars that are neither letters, digits nor ALLOWED
MAX_FRAG_RATE = 0.40      # share of tokens with no 2-letter run (stray OCR fragments)
SHORT_TOKENS = 8          # under this, a paragraph needs one dictionary word of 3+ letters
MIN_KEPT_LETTERS = 0.50   # clean_book: share of the book's letters that must survive
MIN_DOC_DICT_RATE = 0.75  # clean_book: dictionary rate of the kept text (PG p1)
THRESHOLDS = dict(min_dict_rate=MIN_DICT_RATE, max_junk_rate=MAX_JUNK_RATE,
                  max_frag_rate=MAX_FRAG_RATE, short_tokens=SHORT_TOKENS,
                  min_kept_letters=MIN_KEPT_LETTERS, min_doc_dict_rate=MIN_DOC_DICT_RATE)

ALLOWED = set(".,;:!?'\"()[]-\u2014\u2013‘’“”&$%/*#+=§°¶£½¼¾…_")   # \u2014 em, \u2013 en dash
EXTRA_WORDS = frozenset(("a", "i", "o"))
_ROMAN = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", re.I)
_PAGE_NUM = re.compile(r"^[\[\(]?\s*([0-9]{1,4}|[ivxlcdm]{1,8})\s*[\]\)\.]?$", re.I)
_LETTER = re.compile(r"[^\W\d_]")
_STAMP = re.compile(r"library\s+of\s+congress", re.I)
_LABEL = re.compile(r"deacidified|bookkeeper\s+process|neutralizing\s+agent|treatment\s+date"
                    r"|preservation\s*technolog|(?:paper|collections)\s+preservation"
                    r"|thomson\s+park|cranberry\s+township|779-2111", re.I)
STAMP_TAIL = 400          # lines (blank ones too); back-cover scan junk can follow the label
_HYPHEN_END = re.compile("([^\\W\\d_]+)[-\u00ac\u00ad]$")          # hyphen, OCR not-sign, soft hyphen
_FIRST_WORD = re.compile(r"^([^\W\d_]+)")
_TOK2 = re.compile(r"[a-z]{2,}")
_RUN2 = re.compile(r"[^\W\d_]{2}")
_PUNCT_STRIP = "".join(sorted(ALLOWED)) + " "
_JUNK = re.compile("[^\\w\\s" + re.escape("".join(sorted(ALLOWED))) + "]")
RH_MAX_LEN, RH_MIN_REPEAT = 70, 2


def dictionary():
    return common_words() | EXTRA_WORDS


def is_page_number(s: str) -> bool:
    m = _PAGE_NUM.match(s)
    return bool(m) and (m.group(1).isdigit() or bool(_ROMAN.match(m.group(1))))


def _key(s: str) -> str:
    return " ".join(t.lower() for t in s.split() if not any(c.isdigit() for c in t)
                    and len(_LETTER.findall(t)) >= 3)


def _mark_lines(lines, counts):
    """-> list of bools: True where the line is dropped (page numbers, no-letter, stamps, heads)."""
    n = len(lines)
    kill = [False] * n
    for i, s in enumerate(lines):
        if not s:
            continue
        if is_page_number(s):
            kill[i] = "page_number"
        elif not _LETTER.search(s):
            kill[i] = "no_letter_line"
        elif len(s) <= 60 and _STAMP.search(s):
            kill[i] = "stamp"
    hits = [(i, m.group(0).lower()) for i in range(max(0, n - STAMP_TAIL), n)
            if (m := _LABEL.search(lines[i]))]
    if len({h for _, h in hits}) >= 2:
        for i in range(hits[0][0], n):
            kill[i] = "preservation_label" if lines[i] else kill[i]      # blank lines stay
    nonblank = [i for i in range(n) if lines[i]]
    cands = {}
    for j, i in enumerate(nonblank):
        s = lines[i]
        if kill[i] or len(s) > RH_MAX_LEN:
            continue
        if not ((i == 0 or not lines[i - 1]) and (i == n - 1 or not lines[i + 1])):
            continue
        near = [nonblank[k] for k in (j - 1, j + 1) if 0 <= k < len(nonblank)]
        numbered = any(any(c.isdigit() for c in t) for t in s.split())
        if numbered or any(kill[k] == "page_number" for k in near):
            key = _key(s)
            if key:
                cands.setdefault(key, []).append(i)
    for key, idx in cands.items():
        if len(idx) >= RH_MIN_REPEAT:
            for i in idx:
                kill[i] = "running_head"
    for k in kill:
        if k:
            counts[k] += 1
    return kill


def _hyphen_keep(a: str, nxt: str, words) -> bool:
    """A line ended in word-part a plus a hyphen and the next line is nxt: keep the hyphen?"""
    m = _FIRST_WORD.match(nxt)
    if not m or not nxt[0].islower():
        return True                                  # 'New-' + 'England'
    b = m.group(1)
    return (a + b).lower() not in words and a.lower() in words and b.lower() in words


def _width(lines, kill) -> float:
    lens = sorted(len(s) for s, k in zip(lines, kill) if s and not k and len(s) >= 20)
    return lens[int(0.8 * (len(lens) - 1))] if lens else 60.0


def _is_verse(block) -> bool:
    if len(block) < 3:
        return False
    caps = sum(1 for s in block if s.lstrip("\"'“‘(")[:1].isupper())
    return caps >= 0.7 * len(block)


def reflow(raw: str, words=None, counts=None):
    """Raw LoC OCR text -> reflowed text (paragraphs joined by one blank line)."""
    words = dictionary() if words is None else words
    counts = Counter() if counts is None else counts
    lines = [x.strip() for x in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    kill = _mark_lines(lines, counts)
    full = 0.75 * _width(lines, kill)
    blocks, cur, gap = [], [], False                 # blocks of kept lines between blank lines
    for s, k in zip(lines, kill):
        if k:
            continue
        if not s:
            gap = True
            continue
        if gap and cur:
            blocks.append(cur)
            cur = []
        cur.append(s)
        gap = False
    if cur:
        blocks.append(cur)
    paras, parts, prev, cont = [], [], "", False
    for block in blocks:
        verse = _is_verse(block)
        for i, s in enumerate(block):
            m = _HYPHEN_END.search(prev) if parts else None
            if not parts:
                parts.append(s)
            elif i == 0 and not (cont and s[0].islower()):
                paras.append("".join(parts))
                parts = [s]
            elif m:
                keep = _hyphen_keep(m.group(1), s, words)
                parts[-1] = parts[-1][:-1] + ("-" if keep else "")
                parts.append(s)
                counts["dehyphenated"] += 1
            elif cont and (not verse or s[0].islower()):
                parts += [" ", s]
            else:
                parts += ["\n", s]
            prev = s
            cont = bool(_HYPHEN_END.search(s)) or len(s) >= full
    if parts:
        paras.append("".join(parts))
    return "\n\n".join(paras)


def features(p: str, words):
    """-> (tokens, dict_rate, junk_rate, frag_rate, has_long_word) of one paragraph."""
    toks = p.split()
    lw = _TOK2.findall(p.lower())
    known = sum(1 for t in lw if t in words)
    nonspace = sum(len(t) for t in toks)
    junk = len(_JUNK.findall(p))
    frag = 0
    for t in toks:
        u = t.strip(_PUNCT_STRIP)
        if u and not u.isdigit() and not _RUN2.search(u) and u.lower() not in EXTRA_WORDS:
            frag += 1
    long_word = any(len(t) >= 3 and t in words for t in lw)
    return (len(toks), known / len(lw) if lw else 0.0, junk / nonspace if nonspace else 1.0,
            frag / len(toks) if toks else 1.0, long_word)


def paragraph_reason(p: str, words, th=THRESHOLDS):
    """None to keep the paragraph, else the reason it reads as OCR junk."""
    n, dict_rate, junk, frag, long_word = features(p, words)
    if junk > th["max_junk_rate"]:
        return "ocr_junk_symbols"
    if frag > th["max_frag_rate"]:
        return "ocr_fragments"
    if n < th["short_tokens"]:
        return None if long_word else "ocr_no_word"
    if dict_rate < th["min_dict_rate"]:
        return "ocr_nonwords"
    return None


def filter_paragraphs(text: str, words, th=THRESHOLDS, counts=None):
    """-> text without its junk paragraphs; counts gets paragraphs and bytes dropped by reason."""
    counts = Counter() if counts is None else counts
    keep = []
    for p in text.split("\n\n"):
        why = paragraph_reason(p, words, th)
        if why:
            counts[why] += 1
            counts[why + "_bytes"] += len(p.encode("utf-8"))
        else:
            keep.append(p)
    return "\n\n".join(keep)


def doc_dict_rate(text: str, words) -> float:
    lw = _TOK2.findall(text.lower())
    return sum(1 for t in lw if t in words) / len(lw) if lw else 0.0


def clean_book(raw: str, words=None, th=THRESHOLDS):
    """-> (text, counts, drop reason or None). The text is reflowed and filtered, not normalized."""
    words = dictionary() if words is None else words
    counts = Counter()
    text = filter_paragraphs(reflow(raw, words, counts), words, th, counts)
    before, after = len(_LETTER.findall(raw)), len(_LETTER.findall(text))
    counts["letters_in"], counts["letters_kept"] = before, after
    if before and after < th["min_kept_letters"] * before:
        return text, counts, "ocr_book_mostly_junk"
    if doc_dict_rate(text, words) < th["min_doc_dict_rate"]:
        return text, counts, "ocr_book_nonwords"
    return text, counts, None
