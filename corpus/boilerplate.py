"""Boilerplate line removal (CORPUS 3.2 step 3): "Lines that recur across many documents in a
source are dropped. This is a cheap stand-in for OLMo 3's suffix-array pass."

Applied to the web and wiki prose sources (SOURCES): CCCC navigation, cookie and map-widget lines,
and the footers and banners that Wikimedia renders into many pages. Not applied to chat or IRC
(short replies such as 'thanks' legitimately repeat), StackExchange (code lines such as '}'
repeat) or Gutenberg (the book cap already skips the front and back matter).

Rule, per source:
- a line is its text with surrounding whitespace stripped; blank lines are never counted;
- a line is boilerplate when it occurs in at least min_docs distinct documents of the source,
  counted once per document. A document adds its lines to the count only if neither its sha1 nor
  its URL (url_key: no scheme, no 'www.', no fragment or trailing slash) was seen before in the
  source, so an exact copy, or the same page captured in several crawl snapshots (CCCC holds ten),
  does not make the page's own content look like boilerplate. Near-copies under other URLs still
  count: MinHash near-dedup (CORPUS 3.2 step 2, which should run first) is not implemented;
- boilerplate lines are deleted from every document of the source, the rest is re-normalized, and
  a document left under min_bytes is dropped as 'boilerplate_short'. Exact dedup then runs on the
  cleaned text (extract_merge.py).
Hashes are the first 8 bytes of blake2b over the stripped line (stable across processes).
"""
import hashlib
import re

import numpy as np

import hygiene as H

SOURCES = ("cccc", "wikimedia")
MIN_DOCS = 10
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://")


def url_key(url) -> str:
    """'https://www.Example.org/a/#x' -> 'example.org/a' ('' when there is no URL)."""
    if not url:
        return ""
    u = _SCHEME.sub("", str(url).strip().lower()).split("#", 1)[0]
    return (u[4:] if u.startswith("www.") else u).rstrip("/")


def line_hash(line: str) -> int:
    return int.from_bytes(hashlib.blake2b(line.encode("utf-8"), digest_size=8).digest(), "big")


def doc_hashes(text: str) -> list:
    """The distinct line hashes of one document (blank lines skipped)."""
    return list({line_hash(s) for s in (x.strip() for x in text.split("\n")) if s})


def frequent_lines(hashes: np.ndarray, min_docs: int = MIN_DOCS) -> frozenset:
    """hashes: every document's distinct line hashes, concatenated (one doc per sha1 group).
    -> the set of hashes seen in at least min_docs documents."""
    if hashes.size == 0 or min_docs <= 0:
        return frozenset()
    u, c = np.unique(hashes, return_counts=True)
    return frozenset(int(h) for h in u[c >= min_docs])


def strip_boilerplate(text: str, bad: frozenset) -> str:
    """text without its boilerplate lines, re-normalized."""
    if not bad:
        return text
    keep = [x for x in text.split("\n") if not x.strip() or line_hash(x.strip()) not in bad]
    return H.normalize("\n".join(keep))
