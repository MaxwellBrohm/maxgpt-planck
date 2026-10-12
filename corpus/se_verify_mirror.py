"""Check that a mirrored Stack Exchange dump is faithful to the official one, site by site.

    python corpus/se_verify_mirror.py MIRROR.7z OFFICIAL.7z [--mirror-date 2022-10-05] [--out report.json]

The core v0 plan reads the community copy of the official 2022-10-05 dump (archive.org item
stackexchange_20221005) because every body in it predates the Q6 cutoff. Before trusting it, compare
it with the official item (a later dump): a post or comment that exists in both and was last changed
before the mirror's date (LastEditDate, else CreationDate) should have the same Title and body TEXT
(se_html.html_to_text) for posts, and the same Text for comments. Raw HTML is compared too
(differ_html): Stack Exchange re-renders old posts without an edit (imgur -> i.stack.imgur links,
rel="nofollow" dropped), which changes the HTML but rarely the text. A dump is also a snapshot taken
some days before its date, so a few posts edited just before it differ legitimately; the report gives
the mirror's newest CreationDate (its real as-of time). faithful: no comment differs and at most
MAX_TEXT_DIFFER of compared posts differ in text. The report also measures what the later official
dump would lose under the edit gate of readers_se_dump (posts created before the cutoff, edited after).
Memory: one 16-byte digest per mirror post and comment (fine for the small check sites).
"""
import argparse
import hashlib
import json
import os
import sys

import hygiene as H
from readers_se_dump import dir_members
from se_html import html_to_text
from se_stage import iter_rows
from sevenzip_min import open_members

FILES = ("Posts.xml", "Comments.xml")
MAX_TEXT_DIFFER = 0.005


def digest(*parts):
    return hashlib.blake2b("\x00".join(p or "" for p in parts).encode("utf-8"),
                           digest_size=16).digest()


def load(path):
    """-> {"Posts.xml": {id: row summary}, "Comments.xml": {...}} streamed from a site archive
    (or a directory holding the two files)."""
    out = {f: {} for f in FILES}
    want = set(FILES)
    members = dir_members(path, want) if os.path.isdir(path) else open_members(path, want)
    for name, fh in members:
        tab = out[name]
        for a in iter_rows(fh):
            key = int(a.get("Id") or 0)
            if name == "Posts.xml":
                tab[key] = (digest(html_to_text(a.get("Body") or ""), a.get("Title")),
                            a.get("CreationDate"), a.get("LastEditDate"), a.get("PostTypeId"),
                            digest(a.get("Body"), a.get("Title")))
            else:
                dg = digest(a.get("Text"))
                tab[key] = (dg, a.get("CreationDate"), None, None, dg)
    return out


def compare(mirror, official, mirror_date, cutoff=H.DATE_CUTOFF):
    """A row counts as changed after the mirror when its official LastEditDate (else CreationDate)
    is after the mirror's real as-of time: the nominal date, or the newest CreationDate in
    the mirror when that is earlier (the dump snapshot predates the publication date)."""
    newest = max((v[1] or "" for f in FILES for v in mirror[f].values()), default="")
    asof = min(mirror_date, newest) if newest else mirror_date
    rep = {"mirror_date": mirror_date, "as_of_used": asof, "cutoff": cutoff}
    for f in FILES:
        m, o = mirror[f], official[f]
        r = {"mirror_rows": len(m), "official_rows": len(o), "compared": 0, "identical": 0,
             "differ": 0, "differ_ids": [], "differ_html": 0, "mirror_only": 0,
             "changed_after_mirror": 0, "official_only_created_before_mirror": 0,
             "mirror_max_created": max((v[1] or "" for v in m.values()), default=None)}
        for k, (dg, created, edited, _, html) in m.items():
            if k not in o:
                r["mirror_only"] += 1
                continue
            odg, ocreated, oedited, _, ohtml = o[k]
            if (oedited or ocreated or "9999")[:23] > asof:
                r["changed_after_mirror"] += 1
                continue
            r["compared"] += 1
            r["differ_html"] += int(html != ohtml)
            if odg == dg:
                r["identical"] += 1
            else:
                r["differ"] += 1
                if len(r["differ_ids"]) < 20:
                    r["differ_ids"].append(k)
        r["official_only_created_before_mirror"] = sum(
            1 for k, v in o.items() if k not in m and (v[1] or "9999") < asof)
        if f == "Posts.xml":
            pre = [v for v in o.values() if v[3] in ("1", "2") and (v[1] or "9999") < cutoff]
            late = [v for v in pre if v[2] and v[2] >= cutoff]
            r["official_pre_cutoff_posts"] = len(pre)
            r["official_edit_gated_posts"] = len(late)
            r["official_edit_gated_questions"] = sum(1 for v in late if v[3] == "1")
            r["official_edit_gated_share"] = round(len(late) / len(pre), 4) if pre else None
        rep[f] = r
    p, c = rep["Posts.xml"], rep["Comments.xml"]
    rep["faithful"] = bool(p["compared"] and c["compared"] and c["differ"] == 0
                           and p["differ"] <= MAX_TEXT_DIFFER * p["compared"])
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mirror")
    ap.add_argument("official")
    ap.add_argument("--mirror-date", default="2022-10-05")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = compare(load(a.mirror), load(a.official), a.mirror_date)
    rep.update(mirror=a.mirror, official=a.official)
    print(json.dumps(rep, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    return 0 if rep["faithful"] else 1


if __name__ == "__main__":
    sys.exit(main())
