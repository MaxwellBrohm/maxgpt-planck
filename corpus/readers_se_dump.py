"""Stack Exchange data dump reader for core v0. It REPLACES the Common Pile StackExchange subset that the
starter used (common-pile/stackexchange dates a whole thread by its question; CORPUS addendum, open item).

Input: one site's dump archive (<site>.7z holding Posts.xml and Comments.xml; read by sevenzip_min, no
7z tool needed) or a directory holding those two files. Output: one document per question thread, in
the event protocol of readers_cp.py: ("keep", record, raw_bytes), ("drop", reason, raw_bytes),
("note", name, n). The site list lives in corpus/fetch/make_core_manifest.py (SE_SITES, 58 sites).

Per POST (question, answer or comment), before a thread is assembled:
1. date gate (Q6): CreationDate before the cutoff (2022-12-01); a post whose LastEditDate is on or
   after the cutoff is dropped too ('edit_gate'), because the dump holds the latest revision of each
   body. In the 2022-10-05 dump every body predates the cutoff by construction, so this only bites on
   later dumps. A missing or unparseable CreationDate drops the post ('date_missing').
2. license: ContentLicense ('CC BY-SA 2.5' / '3.0' / '4.0') through hygiene.license_id; when the
   attribute is absent, the Stack Exchange era by creation date (before 2011-04-08: 2.5; before
   2018-05-02: 3.0; later: 4.0), basis 'date_era'. All three are allowed; anything else is dropped.
3. text: bodies are HTML (se_html.html_to_text), comments light markup (se_html.comment_to_text).
   The question's title is its first line.
4. AI-ism filter (hygiene.aiism) on every post; comments that are moderation templates ('possible
   duplicate of', 'I'm voting to close', 'take the tour', ...) are dropped as 'template'.
Only PostTypeId 1 (question) and 2 (answer) are read; tag wikis and the rest are counted and skipped.
Posts and comments are staged in a temporary SQLite file (memory stays bounded on the largest sites)
in o['se_tmp_dir'] or the stream driver's o['scratch_dir']; its size is about the site's text.
Options (o, all optional): cutoff, min_bytes, max_non_ascii, min_stopwords, max_docs (questions),
se_comments (True), se_min_answers (1), se_min_answer_score (0), se_dataset and se_dump_date
(else read from the raw path: 'archive.org__stackexchange_20221005/<site>.7z').

Per THREAD: a question that failed a post rule drops the thread ('question_<reason>'). Answers need
Score >= min_answer_score (default 0) and a thread needs min_answers kept answers (default 1, else
'no_answer'). Order: question, its comments, then answers (accepted first, then by score, then by
date), each followed by its comments; posts are separated by a blank line, lines within a post by one
newline (the Common Pile layout). Then normalize and hygiene_reason (min bytes, non-ASCII, English).
The year-2023-2029 marker of hygiene.post_cutoff_marker is NOT applied: every post carries its own date.
Record: id 'stackexchange:<site host>:<question id>'; turns [{role question|answer|comment, text,
post_id (a comment's own id), parent_id (answers, comments), created, score, license, license_basis,
author (user id, or 'name:<display name>')}] for BY-SA attribution; meta holds the thread fields
(license = newest BY-SA version present, licenses = all of them, sharealike, dataset, dump_date).
"""
import os
import re
import shutil
import sqlite3
import tempfile

import hygiene as H
from se_stage import COMMENT_COLS, POST_COLS, SCHEMA, stage_comments, stage_posts
from sevenzip_min import open_members

SOURCE, REPLACES = "stackexchange", "common-pile/stackexchange"
POSTS, COMMENTS = "Posts.xml", "Comments.xml"
_ITEM_DATE = re.compile(r"stackexchange_(\d{4})(\d{2})(\d{2})")


def _turn(role, r):
    """A turn with its attribution fields. post_id is the comment's own id for a comment, and
    parent_id the post it was made on (the question, for an answer)."""
    t = {"role": role, "text": r["text"], "post_id": r["cid"] if role == "comment" else r["pid"],
         "created": r["created"], "score": r["score"], "license": r["lic"],
         "license_basis": r["lbasis"], "author": r["owner"]}
    if role != "question":
        t["parent_id"] = r["pid"] if role == "comment" else r["tid"]
    return t


def _count(notes, key, n=1):
    notes[key] = notes.get(key, 0) + n


def assemble(db, host, rel, o, notes):
    """Yield one event per question thread staged in db."""
    min_score, min_answers = o.get("se_min_answer_score", 0), o.get("se_min_answers", 1)
    cols = [c.strip() for c in POST_COLS.split(",")]
    ccols = [c.strip() for c in COMMENT_COLS.split(",")]
    for n, qrow in enumerate(db.execute("SELECT * FROM posts WHERE kind=1 ORDER BY pid")):
        if o.get("max_docs") and n >= o["max_docs"]:
            break
        q = dict(zip(cols, qrow))
        ans = [dict(zip(cols, r)) for r in db.execute(
            "SELECT * FROM posts WHERE tid=? AND kind=2", (q["pid"],))]
        pids = [q["pid"]] + [a["pid"] for a in ans]
        coms = [dict(zip(ccols, r)) for r in db.execute(
            f"SELECT * FROM comments WHERE pid IN ({','.join('?' * len(pids))}) "
            "ORDER BY pid, created, cid", pids)]
        rb = q["rb"] + sum(a["rb"] for a in ans) + sum(c["rb"] for c in coms)
        _count(notes, "answers_in", len(ans))
        _count(notes, "comments_in", len(coms))
        if q["why"]:
            yield ("drop", "question_" + q["why"], rb)
            continue
        for a in ans:
            if a["why"] or a["score"] < min_score:
                _count(notes, "answers_dropped_" + (a["why"] or "low_score"))
        ans = [a for a in ans if not a["why"] and a["score"] >= min_score]
        ans.sort(key=lambda a: (a["pid"] != q["accepted"], -a["score"], a["created"], a["pid"]))
        if len(ans) < min_answers:
            yield ("drop", "no_answer", rb)
            continue
        by_post, kept_pids = {}, {q["pid"]} | {a["pid"] for a in ans}
        for c in coms:
            if c["why"] or c["pid"] not in kept_pids:
                _count(notes, "comments_dropped_" + (c["why"] or "parent_dropped"))
            else:
                by_post.setdefault(c["pid"], []).append(c)
        turns = []
        for role, r in [("question", q)] + [("answer", a) for a in ans]:
            turns.append(_turn(role, r))
            turns += [_turn("comment", c) for c in by_post.get(r["pid"], [])]
        text = H.normalize("\n\n".join(t["text"] for t in turns))
        reason = H.hygiene_reason(text, o.get("min_bytes", H.MIN_BYTES),
                                  o.get("max_non_ascii", H.MAX_NON_ASCII),
                                  o.get("min_stopwords", H.MIN_EN_STOPWORDS))
        if reason:
            yield ("drop", reason, rb)
            continue
        _count(notes, "answers_kept", len(ans))
        _count(notes, "comments_kept", len(turns) - 1 - len(ans))
        lics = sorted({t["license"] for t in turns})
        meta = {"dataset": dataset_of(rel, o), "file": rel, "site": host,
                "orig_id": str(q["pid"]), "url": f"https://{host}/questions/{q['pid']}",
                "title": q["text"].split("\n", 1)[0], "created": q["created"],
                "date_status": "ok", "date_basis": "post_creation_each_post",
                "dump_date": dump_date_of(rel, o), "license": lics[-1], "licenses": lics,
                "license_basis": "post" if all(t["license_basis"] == "post" for t in turns)
                else "post_or_date_era", "sharealike": True, "author_type": "human",
                "score": q["score"], "closed": bool(q["closed"]), "n_answers": len(ans),
                "n_comments": len(turns) - 1 - len(ans), "replaces": REPLACES}
        yield ("keep", {"id": f"{SOURCE}:{host}:{q['pid']}", "source": SOURCE, "text": text,
                        "turns": turns, "meta": meta}, rb)
    orphan = db.execute("SELECT COUNT(*) FROM posts WHERE kind=2 AND tid NOT IN "
                        "(SELECT pid FROM posts WHERE kind=1)").fetchone()[0]
    _count(notes, "orphan_answers", orphan)


def dataset_of(rel, o):
    """o['se_dataset'], else the raw layout's dataset dir ('archive.org__stackexchange_20221005/x.7z'
    -> 'archive.org/stackexchange_20221005', as stream_fetch.raw_rel writes it)."""
    rel = str(rel)
    return o.get("se_dataset") or (rel.split("/")[0].replace("__", "/") if "/" in rel
                                   else "stackexchange-dump")


def dump_date_of(rel, o):
    """o['se_dump_date'], else the date in an item name like 'stackexchange_20221005'."""
    m = _ITEM_DATE.search(str(rel))
    return o.get("se_dump_date") or (f"{m[1]}-{m[2]}-{m[3]}" if m else None)


def site_host(path) -> str:
    b = os.path.basename(os.path.normpath(str(path)))
    return b[:-3] if b.endswith(".7z") else b


def dir_members(path, want):
    for m in sorted(want):
        if os.path.exists(os.path.join(path, m)):
            with open(os.path.join(path, m), "rb") as fh:
                yield m, fh


def read_se_dump(source, path, rel, o):
    """Event generator for one site dump (a .7z, or a directory with Posts.xml, Comments.xml)."""
    cutoff = o.get("cutoff", H.DATE_CUTOFF)
    want = {POSTS, COMMENTS} if o.get("se_comments", True) else {POSTS}
    tmp = tempfile.mkdtemp(prefix="se_stage_", dir=o.get("se_tmp_dir") or o.get("scratch_dir"))
    notes = {"replaces_" + REPLACES.replace("/", "_"): 1}
    db = sqlite3.connect(os.path.join(tmp, "stage.db"))
    try:
        db.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;" + SCHEMA)
        members = dir_members(path, want) if os.path.isdir(path) else open_members(str(path), want)
        seen = set()
        for name, fh in members:
            seen.add(name)
            if name == POSTS:
                stage_posts(db, fh, cutoff, notes)
            else:
                stage_comments(db, fh, cutoff)
        if POSTS not in seen:
            raise FileNotFoundError(f"{path}: no {POSTS}")
        db.execute("CREATE INDEX p_tid ON posts(tid, kind)")
        db.execute("CREATE INDEX c_pid ON comments(pid)")
        yield from assemble(db, site_host(path), rel, o, notes)
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)
    yield from (("note", k, v) for k, v in sorted(notes.items()))
