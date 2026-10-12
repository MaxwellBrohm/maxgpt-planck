"""Gates and minimal hygiene shared by every corpus source (CORPUS.md 3.1-3.2, v0 subset).

- date gate (ruling Q6): keep documents created before 2022-12-01 where the source records a
  date; a document with no date is kept and counted as undated; a date that is present but does
  not parse is dropped. Only the calendar date as written is compared (no timezone shift).
- license gate (D8, CORPUS 1.3): human-written open text only. Allowed: public domain, CC0,
  CC BY, CC BY-SA (any version), and the permissive dataset licenses Apache-2.0 and MIT. Any NC
  or ND variant, and any string this module does not recognise, is refused.
- normalize: NFC, CRLF/CR to LF, trailing spaces cut, 3+ blank lines squeezed to one, NUL removed.
- English-only v0: drop when over 30% of letters are non-ASCII, and (added after a probe of the
  real files, see extract.py) when under 10% of words are common English function words.
- exact dedup key: sha1 of the normalized text.
- AI-ism filter (CORPUS 2.2: OASST2 passes "after an AI-ism filter"): aiism() is the ONE definition.
  It flags assistant self-reference boilerplate ("as an AI language model", "I don't have personal
  opinions", "trained by OpenAI", ...) and names of post-2022 chat models (ChatGPT, GPT-3.5, GPT-4,
  OpenAI). Readers apply it per message (OASST2), per row (Dolly) and per document (StackExchange
  and Wikimedia, through post_cutoff_marker).
- post-cutoff markers: post_cutoff_marker() = aiism() plus a year 2023-2029 written as a word. It is
  for sources whose recorded date does not cover all of the text: a StackExchange thread is dated
  by its question (later answers and comments carry no date), and a Wikimedia page is rendered at
  dump time (transcluded templates and copyright footers can be newer than the last revision).
"""
import hashlib
import re
import unicodedata

DATE_CUTOFF = "2022-12-01"
MIN_BYTES = 200
MAX_NON_ASCII = 0.30
MIN_EN_STOPWORDS = 0.10
SCAN_CHARS = 200_000          # ratios are measured on the first 200k characters of long docs

_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_CC_URL = re.compile(r"creativecommons\.org/(licenses|publicdomain)/([a-z-]+)/(\d+(?:\.\d+)?)")
_CC_SHORT = re.compile(r"^cc[- ]?(by(?:[- ]sa)?|zero|0)(?:[- ]v?(\d+(?:\.\d+)?))?$")
_LETTER = re.compile(r"[^\W\d_]")
_NON_ASCII_LETTER = re.compile(r"[^\W\d_A-Za-z]")
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
_BLANKS = re.compile(r"\n{3,}")
STOPWORDS = frozenset(
    "the be to of and a in that have i it for not on with he as you do at this but his by from "
    "they we say her she or an will my one all would there their what so up out if about who get "
    "which go me is are was were can just no yes it's don't i'm".split())
PERMISSIVE = {"apache-2.0": "apache-2.0", "apache 2.0": "apache-2.0", "mit": "mit"}
_AI_NOUN = (r"(?:(?:large )?language model|artificial intelligence|ai model|ai assistant|"
            r"chatbot|a\.i\.|ai)")
_AIISM = re.compile(
    r"\bas an? " + _AI_NOUN + r"\b"
    r"|\bi(?: am|'m) (?:just |only |merely |simply )?an? " + _AI_NOUN + r"\b"
    r"|\blanguage model (?:developed|trained|created|made) by\b"
    r"|\b(?:trained|developed|created) by openai\b"
    r"|\bi (?:don't|do not) have (?:personal|my own) (?:opinions|beliefs|experiences|preferences|"
    r"feelings|emotions)\b"
    r"|\b(?:my|the) knowledge cut-?off\b|\bas of my (?:last|latest) (?:update|training)\b"
    r"|\bmy training data\b"
    r"|\bi(?: am|'m) open ?assistant\b"
    r"|\bchat ?gpt\b|\bgpt-?3\.5\b|\bgpt-?4o?\b|\bopen ?ai\b")
_LATE_YEAR = re.compile(r"(?<![\w.,])202[3-9](?!\w)")


def aiism(text: str):
    """The AI-ism filter: the first AI-ism found in text (lowercased, curly apostrophes folded), or
    None. Human text rarely says these; model-written or model-copied text often does."""
    m = _AIISM.search(text.lower().replace("’", "'"))
    return m.group(0) if m else None


def post_cutoff_marker(text: str):
    """An AI-ism, or a year 2023-2029 written as a word: evidence that part of a document dated
    before the cutoff was written (or rendered) after it. -> the marker found, or None."""
    hit = aiism(text)
    if hit:
        return hit
    m = _LATE_YEAR.search(text)
    return m.group(0) if m else None


def date_status(created, cutoff: str = DATE_CUTOFF) -> str:
    """-> 'ok' (before cutoff), 'post_cutoff', 'undated' (no date recorded) or 'unparseable'."""
    if created is None or (isinstance(created, str) and not created.strip()):
        return "undated"
    m = _DATE.match(str(created).strip())
    if not m:
        return "unparseable"
    y, mo, d = (int(g) for g in m.groups())
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return "unparseable"
    return "ok" if m.group(0) < cutoff else "post_cutoff"


def license_id(raw):
    """Canonical id for an allowed license string, or None when it is not allowed or unknown.

    Common Pile writes e.g. 'Creative Commons - Attribution Share-Alike -
    https://creativecommons.org/licenses/by-sa/3.0/' and 'Public Domain'."""
    if raw is None:
        return None
    t = str(raw).strip().lower()
    if t in ("public domain", "public-domain", "pd"):
        return "public-domain"
    if t in PERMISSIVE:
        return PERMISSIVE[t]
    m = _CC_URL.search(t)
    if m:
        kind, name, ver = m.groups()
        if kind == "publicdomain":
            return "cc0" if name == "zero" else ("public-domain" if name == "mark" else None)
        return f"cc-{name}-{ver}" if name in ("by", "by-sa") else None
    if "creativecommons" in t or "creative commons" in t:
        return None                    # prose-only CC string: variant unknown, refuse
    m = _CC_SHORT.match(t.replace("_", "-"))
    if m:
        name, ver = m.groups()
        if name in ("zero", "0"):
            return "cc0"
        name = name.replace(" ", "-")
        return f"cc-{name}-{ver}" if ver else f"cc-{name}"
    return None


def normalize(text: str) -> str:
    t = text.replace("\x00", "")
    if not unicodedata.is_normalized("NFC", t):
        t = unicodedata.normalize("NFC", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    return _BLANKS.sub("\n\n", t).strip()


def nbytes(text: str) -> int:
    return len(text.encode("utf-8"))


def non_ascii_letter_ratio(text: str) -> float:
    s = text[:SCAN_CHARS]
    n = len(_LETTER.findall(s))
    return len(_NON_ASCII_LETTER.findall(s)) / n if n else 0.0


def en_stopword_ratio(text: str) -> float:
    words = _WORD.findall(text[:SCAN_CHARS].lower())
    return sum(1 for w in words if w in STOPWORDS) / len(words) if words else 0.0


def dedup_key(normalized_text: str) -> str:
    return hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()


def hygiene_reason(text: str, min_bytes=MIN_BYTES, max_non_ascii=MAX_NON_ASCII,
                   min_stopwords=MIN_EN_STOPWORDS):
    """Drop reason for an already-normalized text, or None to keep it. Checked in this order:
    empty, too_short, non_ascii, not_english. A threshold of 0 (or 1 for non-ASCII) disables it."""
    if not text:
        return "empty"
    if nbytes(text) < min_bytes:
        return "too_short"
    if max_non_ascii < 1 and non_ascii_letter_ratio(text) > max_non_ascii:
        return "non_ascii"
    if min_stopwords > 0 and en_stopword_ratio(text) < min_stopwords:
        return "not_english"
    return None
