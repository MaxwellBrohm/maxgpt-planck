"""Common Pile readers for the starter extract: CCCC, Project Gutenberg, StackExchange, Ubuntu IRC,
Wikimedia. Each reader yields events for extract.py:

    ("keep", record, raw_bytes)      record = {"id", "source", "text", "meta"}
    ("drop", reason, raw_bytes)
    ("note", counter_name, n)

Schema facts (read from the real starter files on the PC, 2026-09-25; top-level keys are id, text,
source, added, metadata, plus created where noted; "added" is the Common Pile ingestion date and
is never used as a creation date):

| source        | date field                         | license field                      | language   |
|---------------|------------------------------------|------------------------------------|------------|
| cccc          | created (crawl time, ISO)          | none per document                  | none       |
| gutenberg     | none (undated)                     | metadata.license 'Public Domain'   | metadata.language |
| stackexchange | created (question date)            | metadata.license + all_licenses    | none       |
| irc           | created (log date, YYYY-MM-DD)     | metadata.license 'Public Domain'   | none       |
| wikimedia     | created (looks like last revision) | metadata.license (CC BY-SA 4.0)    | none       |

CCCC records no per-document license in these files (checked on the first 1,500 docs of all ten
starter snapshots: metadata keys are content_type, provenance, uncompressed_offset, url, warc_date,
warc_filename, warc_url). The dataset card says the license is in metadata.license and that the
sources were chosen by a manual review of the top 1,000 domains, keeping 537 "where the Creative
Commons designation applied to the all text content". So CCCC docs are kept with license None and
license_basis 'unrecorded': whether that domain review satisfies D8's per-document rule is a ruling
for Max, and make_tok_sample.py refuses to use such docs unless told to (--allow-unrecorded-license).

Date gates and what they cover (meta.date_basis):
- cccc: crawl date; the page existed as captured, so the gate covers all of its text.
- irc: log date; one document is one channel-day, so the gate covers all of it.
- stackexchange: the question's date only. A thread also holds answers and comments with no date
  of their own, which can be years newer (moderators flagged ChatGPT answers in 2020 threads). Such
  threads are dropped when they carry a post-cutoff marker (hygiene.post_cutoff_marker: an AI-ism
  or a year 2023-2029); later human answers with no marker still get through.
- wikimedia: the latest revision time, not page creation (wikibooks 'Main Page' reads 2021-04-25),
  which is stricter than CORPUS 3.1's "page creation". Text is rendered at dump time, so templates
  and footers can be newer; the same post-cutoff marker check applies.
Wikimedia keeps only content pages: namespace 0 on every wiki, plus Wikibooks' Cookbook (102) and
Wikijunior (110). Talk pages (odd namespaces) are dropped as wiki_talk (CORPUS 2.2 lists talk pages
as a human-dialogue candidate "pass after the check"; not admitted here), and user, project, help,
portal and the rest as wiki_namespace.
IRC cleanup and the nickname scrub: irc_clean.py.
"""
import gzip
import json

import hygiene as H
from irc_clean import clean_irc  # noqa: F401  (tests import it from here)

WIKI_EXTRA_NS = frozenset({("wikibooks.com", "102"), ("wikibooks.com", "110")})

CP_SOURCES = {
    "cccc": dict(dataset="common-pile/cccc", license_required=False, sub=None,
                 date_basis="crawl_date"),
    "gutenberg": dict(dataset="common-pile/project_gutenberg", license_required=True, sub=None,
                      lang_field="language", cap=True, date_basis="none"),
    "stackexchange": dict(dataset="common-pile/stackexchange", license_required=True,
                          sub="site", all_licenses=True, date_basis="question_date",
                          late_markers=True),
    "irc": dict(dataset="common-pile/ubuntu_irc", license_required=True, sub=None, irc=True,
                date_basis="log_date"),
    "wikimedia": dict(dataset="common-pile/wikimedia", license_required=True, sub="wiki",
                      wiki_ns=True, date_basis="last_revision", late_markers=True),
}
META_KEEP = ("url", "title", "site", "wiki", "namespace", "channel", "language")


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def middle_window(text: str, cap: int) -> str:
    """At most `cap` bytes from the middle of a long book, cut at paragraph breaks. The middle
    skips the front matter and the end-of-book license text that repeat across books."""
    b = text.encode("utf-8")
    if len(b) <= cap:
        return text
    start = (len(b) - cap) // 2
    w = b[start:start + cap].decode("utf-8", errors="ignore")
    i, j = w.find("\n\n"), w.rfind("\n\n")
    if 0 <= i < j:
        w = w[i + 2:j]
    return w.strip()


def _license(cfg, m):
    """-> (license id or None, basis, drop reason or None).

    StackExchange lists the license of every post in the thread in all_licenses. Some entries are
    the string 'None' (no license recorded for that post; seen in the small 00001 shards, 1,893 of
    their first 2,100 docs). Those docs are dropped as license_component_missing (strict D8
    reading; an open question for Max), apart from 'license' drops of a refused license."""
    raw = m.get("license")
    if raw is None and not cfg["license_required"]:
        return None, "unrecorded", None
    lic = H.license_id(raw)
    if lic is None:
        return None, "document", "license"
    if cfg.get("all_licenses"):
        parts = m.get("all_licenses") or []
        if any(x is not None and str(x) != "None" and H.license_id(x) is None for x in parts):
            return None, "document", "license"
        if any(x is None or str(x) == "None" for x in parts):
            return None, "document", "license_component_missing"
    return lic, "document", None


def wiki_ns_reason(m):
    """None for a content page, else the drop reason (wiki_talk or wiki_namespace)."""
    ns = str(m.get("namespace"))
    if ns == "0" or (m.get("wiki"), ns) in WIKI_EXTRA_NS:
        return None
    return "wiki_talk" if ns.isdigit() and int(ns) % 2 == 1 else "wiki_namespace"


def cp_doc(source, d, rel, o):
    """One Common Pile document -> one event."""
    cfg = CP_SOURCES[source]
    raw = d.get("text") or ""
    rb = H.nbytes(raw)
    m = d.get("metadata") or {}
    lic, basis, why = _license(cfg, m)
    if why:
        return ("drop", why, rb)
    ds = H.date_status(d.get("created"), o["cutoff"])
    if ds == "post_cutoff":
        return ("drop", "date_gate", rb)
    if ds == "unparseable":
        return ("drop", "date_unparseable", rb)
    if cfg.get("lang_field") and m.get(cfg["lang_field"]) != "en":
        return ("drop", "lang_field", rb)
    if cfg.get("wiki_ns"):
        why = wiki_ns_reason(m)
        if why:
            return ("drop", why, rb)
    text = H.normalize(clean_irc(raw) if cfg.get("irc") else raw)
    if cfg.get("late_markers") and H.post_cutoff_marker(text):
        return ("drop", "post_cutoff_marker", rb)
    meta = {"dataset": cfg["dataset"], "file": rel, "orig_id": str(d.get("id")),
            "license": lic, "license_raw": m.get("license"), "license_basis": basis,
            "created": d.get("created"), "date_status": ds, "date_basis": cfg["date_basis"]}
    meta.update({k: m[k] for k in META_KEEP if k in m})
    cap = o["gutenberg_cap"]
    if cfg.get("cap") and cap and H.nbytes(text) > cap:
        meta["orig_bytes"] = H.nbytes(text)
        text = middle_window(text, cap)
        meta["capped"] = True
    reason = H.hygiene_reason(text, o["min_bytes"], o["max_non_ascii"], o["min_stopwords"])
    if reason:
        return ("drop", reason, rb)
    sub = m.get(cfg["sub"]) if cfg["sub"] else None
    doc_id = f"{source}:{sub}:{meta['orig_id']}" if sub else f"{source}:{meta['orig_id']}"
    return ("keep", {"id": doc_id, "source": source, "text": text, "meta": meta}, rb)


def read_common_pile(source, path, rel, o):
    with open_text(path) as fh:
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
            yield cp_doc(source, d, rel, o)
