"""Per-post rules and SQLite staging for readers_se_dump.py (split out to keep each file short).
The rules themselves are documented in readers_se_dump.py. stage_posts() and stage_comments() stream
the dump's XML (iter_rows clears the tree as it goes) and write one row per post or comment, with the
drop reason in `why` (NULL when kept) and the raw text bytes in `rb`, into the tables of SCHEMA.
"""
import re
import xml.etree.ElementTree as ET

import hygiene as H
import se_html

LICENSE_ERAS = (("2011-04-08", "cc-by-sa-2.5"), ("2018-05-02", "cc-by-sa-3.0"),
                ("9999-99-99", "cc-by-sa-4.0"))
Q, A = 1, 2
BATCH = 5000
_TEMPLATE = re.compile(
    r"^(?:possible )?duplicate of\b|^this question (?:already has|appears to be off-topic)"
    r"|^i'?m voting to close\b|^from review\b|^this does not (?:provide|really answer)"
    r"|\btake the \[?tour\b|\bhelp center\b|\bhow do i ask a good question\b", re.I)
POST_COLS = "tid, pid, kind, created, score, accepted, lic, lbasis, owner, text, why, rb, closed"
COMMENT_COLS = "pid, cid, created, score, lic, lbasis, owner, text, why, rb"
SCHEMA = f"""
CREATE TABLE posts ({POST_COLS});
CREATE TABLE comments ({COMMENT_COLS});
"""


def era_license(created: str) -> str:
    return next(lic for bound, lic in LICENSE_ERAS if created[:10] < bound)


def post_license(raw, created):
    """-> (canonical license id or None when refused, basis)."""
    if raw:
        return H.license_id(raw), "post"
    return (era_license(created), "date_era") if created else (None, "none")


def date_gate(created, last_edit, cutoff=H.DATE_CUTOFF):
    """Drop reason for one post, or None."""
    ds = H.date_status(created, cutoff)
    if ds != "ok":
        return "date_gate" if ds == "post_cutoff" else "date_missing"
    if last_edit and H.date_status(last_edit, cutoff) != "ok":
        return "edit_gate"
    return None


def owner_of(a, id_key, name_key):
    if a.get(id_key):
        return a[id_key]
    return "name:" + a[name_key] if a.get(name_key) else None


def iter_rows(fh):
    """Attribute dicts of the <row> elements, streamed; the tree is cleared as it goes."""
    it = ET.iterparse(fh, events=("start", "end"))
    _, root = next(it)
    for ev, el in it:
        if ev == "end" and el.tag == "row":
            yield el.attrib
            root.clear()


def _flush(db, table, batch):
    if batch:
        db.executemany(f"INSERT INTO {table} VALUES ({','.join('?' * len(batch[0]))})", batch)
        batch.clear()


def stage_posts(db, fh, cutoff, notes):
    batch = []
    for a in iter_rows(fh):
        body, title = a.get("Body") or "", a.get("Title") or ""
        rb, pt = H.nbytes(body) + H.nbytes(title), a.get("PostTypeId")
        if pt not in ("1", "2") or not a.get("Id"):
            notes["other_type_posts"] = notes.get("other_type_posts", 0) + 1
            continue
        kind, created = int(pt), a.get("CreationDate")
        pid = int(a["Id"])
        why = date_gate(created, a.get("LastEditDate"), cutoff)
        lic, basis = post_license(a.get("ContentLicense"), created) if not why else (None, None)
        why = why or (None if lic else "license")
        text = None
        if not why:
            text = H.normalize(se_html.html_to_text(body))
            text = H.normalize(title) + "\n" + text if kind == Q else text
            why = "aiism" if H.aiism(text) else None
        batch.append((pid if kind == Q else int(a.get("ParentId") or 0), pid, kind, created,
                      int(a.get("Score") or 0), int(a.get("AcceptedAnswerId") or 0), lic, basis,
                      owner_of(a, "OwnerUserId", "OwnerDisplayName"), None if why else text,
                      why, rb, int(bool(a.get("ClosedDate")))))
        if len(batch) >= BATCH:
            _flush(db, "posts", batch)
    _flush(db, "posts", batch)


def stage_comments(db, fh, cutoff):
    batch = []
    for a in iter_rows(fh):
        raw, created = a.get("Text") or "", a.get("CreationDate")
        why = date_gate(created, None, cutoff)
        lic, basis = post_license(a.get("ContentLicense"), created) if not why else (None, None)
        why = why or (None if lic else "license")
        text = se_html.comment_to_text(raw) if not why else None
        if text is not None:
            why = "template" if _TEMPLATE.search(text) else ("aiism" if H.aiism(text) else None)
        batch.append((int(a.get("PostId") or 0), int(a.get("Id") or 0), created,
                      int(a.get("Score") or 0), lic, basis,
                      owner_of(a, "UserId", "UserDisplayName"),
                      None if why else text, why, H.nbytes(raw)))
        if len(batch) >= BATCH:
            _flush(db, "comments", batch)
    _flush(db, "comments", batch)
