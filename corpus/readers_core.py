"""Core v0 readers: the Common Pile subsets of the strict-open core (CORPUS 2.2 P bucket, manifest
corpus/fetch/core_v0_manifest.json) that the starter reader does not cover, and whole-book
Gutenberg. Same event protocol as readers_cp.py: ("keep", record, raw_bytes), ("drop", reason,
raw_bytes), ("note", counter, n). The starter sources cccc, irc, wikimedia and (Common Pile)
stackexchange go to readers_cp.cp_doc unchanged.

Schema facts, read from range-limited streams of the real files on the PC (2026-09-26):

| source     | license (metadata.license)       | date (dates.py)                        | notes |
|------------|----------------------------------|----------------------------------------|-------|
| gutenberg  | Public Domain                    | none (undated)                         | language 'en'; whole books, pg_clean.py strips credits |
| loc        | Public Domain                    | metadata.year (int), publication year  | language 'english'; OCR reflow and filter, ocr.py |
| youtube    | CC BY 4.0                        | published_time (top level, ISO)        | Whisper transcripts (ruling Q2); no language field; channel_id kept for the per-channel cap |
| news       | CC BY 4.0                        | created: free text in English, else the /YYYY/MM(/DD)/ in the outlet's own URL | 17 outlets; source 'news-<outlet>'; globalvoices has no created at all |
| pressbooks | CC BY, BY-SA, CC0, PD per doc    | created 'MM-D-YYYY' (earliest 2020-03: likely a modification date, so the gate is strict) | lines holding only '|' are table debris and are removed |
| oercommons | CC BY 4.0, PD                    | created 'MM/DD/YYYY'; a value with no digit ('Activity/Lab') is a shifted column: undated | |
| foodista   | CC BY 3.0                        | created ISO (1.5% missing)             | v0 text; ids unique across files |
| pdr        | CC BY-SA 4.0                     | date (top level) 'Mar 26, 2024'        | sub = type (collection, essay, conjecture) |

Rules shared by the new sources: the license gate is readers_cp._license (hygiene.license_id: NC,
ND and unknown strings are refused); a date after the cutoff drops the document as date_gate, a
date that is present but does not parse as date_unparseable, and a missing date keeps it as
undated; an undated document from a source that publishes after 2022 (news, pressbooks,
oercommons, foodista) is also dropped when it carries a post-cutoff marker (an AI-ism or a year
2023-2029, hygiene.post_cutoff_marker). Then hygiene.normalize and hygiene.hygiene_reason, as in
the starter. meta adds author_type, sharealike, the parsed date and, for BY licenses, the
author and URL the attribution needs.
"""
import json
import re

import dates as D
import hygiene as H
import ocr
import pg_clean
import readers_cp as CP

CP_LEGACY = ("cccc", "irc", "wikimedia", "stackexchange")
_BASE = dict(license_required=True, sub=None, lang=None, date=None, numeric_order=None,
             url_date=False, nondate_is_undated=False, late_markers_undated=False,
             author_type="human")
CORE_SOURCES = {
    "gutenberg": dict(_BASE, dataset="common-pile/project_gutenberg", lang=("language", ("en",)),
                      date_basis="none", pg=True),
    "loc": dict(_BASE, dataset="common-pile/library_of_congress",
                lang=("language", ("english",)), date=("meta", "year"),
                date_basis="publication_year", ocr=True, author_type="human_ocr"),
    "youtube": dict(_BASE, dataset="common-pile/youtube", date=("top", "published_time"),
                    date_basis="upload_time", author_type="human_speech_whisper_transcript"),
    "news": dict(_BASE, dataset="common-pile/news", sub="outlet", date=("top", "created"),
                 url_date=True, date_basis="published", late_markers_undated=True),
    "pressbooks": dict(_BASE, dataset="common-pile/pressbooks", date=("top", "created"),
                       numeric_order="mdy", date_basis="created_likely_modified",
                       late_markers_undated=True, pipes=True),
    "oercommons": dict(_BASE, dataset="common-pile/oercommons", date=("top", "created"),
                       numeric_order="mdy", nondate_is_undated=True, date_basis="created",
                       late_markers_undated=True),
    "foodista": dict(_BASE, dataset="common-pile/foodista", date=("top", "created"),
                     date_basis="published", late_markers_undated=True),
    "pdr": dict(_BASE, dataset="common-pile/public_domain_review", sub="type",
                date=("top", "date"), date_basis="published"),
}
CORE_DEFAULTS = dict(cutoff=H.DATE_CUTOFF, min_bytes=H.MIN_BYTES, max_non_ascii=H.MAX_NON_ASCII,
                     min_stopwords=H.MIN_EN_STOPWORDS, max_docs=0, gutenberg_cap=0,
                     ocr_thresholds=None)
DATASETS = {s: c["dataset"] for s, c in CORE_SOURCES.items()}
DATASETS.update({s: CP.CP_SOURCES[s]["dataset"] for s in CP_LEGACY})
# Directory prefixes as fetch_manifest.py lays out RAW_DIR/<dataset with / as __>/<path>.
PREFIX = {d.replace("/", "__") + "/": s for s, d in DATASETS.items()}
META_KEEP = CP.META_KEEP + ("author", "authors", "book_url", "institution", "subject",
                            "item_url", "year", "channel_id", "duration")
_PIPES = re.compile(r"^[|\s✓]*\|[|\s✓]*$")
_WORDS = []


def words():
    if not _WORDS:
        _WORDS.append(ocr.dictionary())
    return _WORDS[0]


def source_for(rel: str):
    """The source of a raw file path relative to RAW_DIR, or None."""
    return next((s for p, s in PREFIX.items() if rel.startswith(p)), None)


def _sub(cfg, d):
    if cfg["sub"] == "outlet":
        s = str(d.get("source") or "")
        return s[5:] if s.startswith("news-") else (s or None)
    return d.get(cfg["sub"]) if cfg["sub"] else None


def _date_value(cfg, d, m):
    where, key = cfg["date"] if cfg["date"] else (None, None)
    return None if where is None else (m if where == "meta" else d).get(key)


def strip_pipe_lines(text: str):
    """-> (text without lines that hold only '|' marks, lines removed)."""
    lines = text.split("\n")
    keep = [x for x in lines if not _PIPES.match(x)]
    return "\n".join(keep), len(lines) - len(keep)


def core_doc(source, d, rel, o):
    """One Common Pile document of a core source -> list of events (notes first)."""
    if source in CP_LEGACY:
        return [CP.cp_doc(source, d, rel, o)]
    cfg = CORE_SOURCES[source]
    raw = d.get("text") or ""
    rb = H.nbytes(raw)
    m = d.get("metadata") or {}
    lic, basis, why = CP._license(cfg, m)
    if why:
        return [("drop", why, rb)]
    value = _date_value(cfg, d, m)
    sub = _sub(cfg, d)
    ds, date, dbasis = D.classify(value, o["cutoff"], cfg["numeric_order"],
                                  url=m.get("url") if cfg["url_date"] else None,
                                  url_host_hint=sub if cfg["url_date"] else None,
                                  nondate_is_undated=cfg["nondate_is_undated"])
    if ds == "post_cutoff":
        return [("drop", "date_gate", rb)]
    if ds == "unparseable":
        return [("drop", "date_unparseable", rb)]
    if cfg["lang"] and m.get(cfg["lang"][0]) not in cfg["lang"][1]:
        return [("drop", "lang_field", rb)]
    ev, extra = [], {}
    if cfg.get("ocr"):
        th = o.get("ocr_thresholds") or ocr.THRESHOLDS
        text, counts, why = ocr.clean_book(raw, words(), th)
        ev += [("note", k if k.startswith("ocr_") else "ocr_" + k, v)
               for k, v in sorted(counts.items())]
        if why:
            return ev + [("drop", why, rb)]
        extra["ocr_letters_kept"] = round(counts["letters_kept"] / max(1, counts["letters_in"]), 4)
        text = H.normalize(text)
    else:
        text = H.normalize(raw)
    if cfg.get("pg"):
        c = {}
        text = pg_clean.strip_front_back(text, c)
        ev += [("note", k, v) for k, v in sorted(c.items())]
    if cfg.get("pipes"):
        text, n = strip_pipe_lines(text)
        if n:
            ev.append(("note", "pipe_lines", n))
            text = H.normalize(text)
    if ds == "undated" and cfg["late_markers_undated"] and H.post_cutoff_marker(text):
        return ev + [("drop", "post_cutoff_marker", rb)]
    reason = H.hygiene_reason(text, o["min_bytes"], o["max_non_ascii"], o["min_stopwords"])
    if reason:
        return ev + [("drop", reason, rb)]
    orig = str(d.get("id"))
    meta = {"dataset": cfg["dataset"], "file": rel, "orig_id": orig, "license": lic,
            "license_raw": m.get("license"), "license_basis": basis,
            "sharealike": bool(lic and lic.startswith("cc-by-sa")),
            "author_type": cfg["author_type"], "created": value, "date": date,
            "date_status": ds, "date_basis": cfg["date_basis"] + (":url" if dbasis == "url" else "")}
    meta.update({k: m[k] for k in META_KEEP if k in m})
    meta.update({k: d[k] for k in META_KEEP if k in d and k not in meta})
    if sub:
        meta["sub"] = sub
    meta.update(extra)
    doc_id = f"{source}:{sub}:{orig}" if sub else f"{source}:{orig}"
    return ev + [("keep", {"id": doc_id, "source": source, "text": text, "meta": meta}, rb)]


def read_core(source, path, rel, o):
    """Reader for extract.process_file-style drivers: yields events for one raw file. o may be
    partial; CORE_DEFAULTS fills the rest (gutenberg_cap 0: whole books)."""
    o = dict(CORE_DEFAULTS, **{k: v for k, v in (o or {}).items() if v is not None})
    with CP.open_text(path) as fh:
        for i, line in enumerate(fh):
            if o["max_docs"] and i >= o["max_docs"]:
                break
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                yield ("drop", "bad_json", len(line.encode("utf-8")))
                continue
            yield from core_doc(source, d, rel, o)


READERS = {s: read_core for s in list(CORE_SOURCES) + list(CP_LEGACY)}
