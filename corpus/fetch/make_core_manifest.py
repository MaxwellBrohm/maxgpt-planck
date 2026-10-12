"""make_core_manifest.py: resolve the strict-open core v0 download to exact files, sizes and checksums
from the Hugging Face API and archive.org metadata, and write corpus/fetch/core_v0_manifest.json.
Metadata only; nothing is downloaded. The PC streams the files (download, verify, extract, delete).

Scope (CORPUS.md 2.2, rulings Q1-Q8 as adopted in the Sep 24 addendum):
- P bucket, Common Pile raw subsets: CCCC snapshots before 2022-12 (46 snapshots, 2013-20 to 2022-05),
  Project Gutenberg, Library of Congress books, CC YouTube (Q2 yes, labeled), Wikimedia non-encyclopedic
  wikis (wikibooks, wikinews, wikivoyage), and the small open sets (news, pressbooks, oercommons,
  foodista v0, public_domain_review).
- HD: Ubuntu IRC (raw logs), and the StackExchange non-technical sites from the official Stack Exchange
  dump of 2022-10-05 (archive.org item stackexchange_20221005, a mirror of the official item made
  2022-10-21; every post body in it predates the 2022-12-01 date gate). Three small sites from the
  official item (2024-04 dump) are listed with role verify_mirror, to check the mirror against it.
Out of scope here: wikiteam, DPI (entry audit first), uk_hansard and usgpo (HD candidates for E-sel-4),
English Wikipedia and its talk pages, wikisource, wikiquote, wikiversity, wiktionary, FineWeb (Q1).

Every Hugging Face URL is pinned to the dataset revision (commit sha) read at manifest time, so a file
cannot change under a running stream. tier = processing order = dedup priority (an earlier tier's copy
of a duplicate wins).

  python corpus/fetch/make_core_manifest.py
"""
import json
import os
import sys
import urllib.request

API = "https://huggingface.co/api/datasets"
IA = "https://archive.org/metadata"
HERE = os.path.dirname(os.path.abspath(__file__))
CCCC_BEFORE = "CC-MAIN-2022-49"      # first snapshot that can hold pages crawled after 2022-12-01
SE_ITEM, SE_OFFICIAL = "stackexchange_20221005", "stackexchange"
SE_VERIFY = ("coffee", "beer", "vegetarianism")
# Non-technical Stack Exchange sites. Out on purpose: programming, computing, math, science,
# engineering, law, patents, health, economics, graphic design, and the foreign-language sites.
SE_SITES = {
    "everyday": "interpersonal parenting workplace travel expatriates cooking diy gardening pets "
                "bicycles outdoors fitness lifehacks money freelancing coffee beer vegetarianism "
                "homebrew woodworking crafts sustainability mechanics sports martialarts academia pm",
    "language": "english ell writers literature languagelearning",
    "culture": "boardgames poker chess puzzling gaming rpg bricks movies scifi anime music musicfans "
               "history philosophy politics skeptics mythology worldbuilding genealogy photo",
    "religion": "hermeneutics christianity judaism islam hinduism buddhism",
}
# (source, dataset, path filter, tier)
HF = [
    ("oasst2", "OpenAssistant/oasst2", lambda p: p == "2023-11-05_oasst2_all.trees.jsonl.gz", 1),
    ("dolly", "databricks/databricks-dolly-15k", lambda p: p.endswith(".jsonl"), 1),
    ("irc", "common-pile/ubuntu_irc", lambda p: p.startswith("raw/documents/"), 1),
    ("wikimedia", "common-pile/wikimedia",
     lambda p: any(w in p for w in ("_wikibooks.com", "_wikinews.com", "_wikivoyage.com")), 1),
    ("news", "common-pile/news", lambda p: p.startswith("v0/documents/"), 1),
    ("pressbooks", "common-pile/pressbooks", lambda p: p.endswith(".json.gz"), 1),
    ("oercommons", "common-pile/oercommons", lambda p: p.endswith(".json.gz"), 1),
    ("foodista", "common-pile/foodista", lambda p: p.startswith("v0/documents/"), 1),
    ("pdr", "common-pile/public_domain_review", lambda p: p.startswith("v0/"), 1),
    ("gutenberg", "common-pile/project_gutenberg", lambda p: p.startswith("v0/documents/"), 2),
    ("youtube", "common-pile/youtube", lambda p: p.endswith(".jsonl.gz"), 3),
    ("loc", "common-pile/library_of_congress", lambda p: p.startswith("data/"), 4),
    ("cccc", "common-pile/cccc", lambda p: "/" in p and p.split("/")[0] < CCCC_BEFORE, 5),
]
TIER_NAMES = {1: "small human and high-value sets", 2: "Project Gutenberg", 3: "CC YouTube",
              4: "Library of Congress books", 5: "CCCC"}


def get(url):
    return urllib.request.urlopen(url, timeout=120)


def hf_files(ds):
    """-> (revision sha, [file dicts]) from the recursive tree API, following pagination."""
    sha = json.load(get(f"{API}/{ds}"))["sha"]
    out, url = [], f"{API}/{ds}/tree/{sha}?recursive=true"
    while url:
        r = get(url)
        out += json.load(r)
        link = r.headers.get("Link") or ""
        url = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
    return sha, [f for f in out if f["type"] == "file"]


def starter_paths():
    p = os.path.join(HERE, "starter_manifest.json")
    return {(e["dataset"], e["path"]) for e in json.load(open(p))["files"]} if os.path.exists(p) else set()


def se_entries():
    """Selected sites from the pre-cutoff dump, plus the verify_mirror sites from the official item."""
    out, want = [], {s: g for g, ss in SE_SITES.items() for s in ss.split()}
    for item, role in ((SE_ITEM, "core"), (SE_OFFICIAL, "verify_mirror")):
        meta = json.load(get(f"{IA}/{item}"))
        files = {f["name"]: f for f in meta["files"]}
        sites = want if role == "core" else {s: want[s] for s in SE_VERIFY}
        for site, group in sorted(sites.items()):
            name = f"{site}.stackexchange.com.7z"
            f = files.get(name)
            if f is None:
                raise SystemExit(f"{item}: {name} missing")
            out.append(dict(source="stackexchange", dataset=f"archive.org/{item}", item=item,
                            path=name, size=int(f["size"]), sha1=f.get("sha1"), md5=f.get("md5"),
                            sha256=None, site=site, group=group, role=role, tier=1,
                            dump_date=meta["metadata"].get("date"),
                            url=f"https://archive.org/download/{item}/{name}"))
    return out


def main():
    local, entries, revisions = starter_paths(), [], {}
    for source, ds, keep, tier in HF:
        sha, files = hf_files(ds)
        revisions[ds] = sha
        for f in sorted(files, key=lambda f: f["path"]):
            if not keep(f["path"]) or not f["path"].endswith((".jsonl.gz", ".json.gz", ".jsonl")):
                continue
            lfs = f.get("lfs") or {}
            entries.append(dict(source=source, dataset=ds, revision=sha, path=f["path"],
                                size=f.get("size", 0), sha256=lfs.get("oid"), tier=tier,
                                role="core", starter_local=(ds, f["path"]) in local,
                                url=f"https://huggingface.co/datasets/{ds}/resolve/{sha}/{f['path']}"))
    entries += se_entries()
    entries.sort(key=lambda e: (e["tier"], e["source"], e["path"]))
    missing = [e["path"] for e in entries if e["source"] != "stackexchange" and not e["sha256"]]
    if missing:
        raise SystemExit(f"no LFS sha256 for {missing[:5]} ...")
    by = {}
    for e in entries:
        b = by.setdefault(e["source"], dict(tier=e["tier"], files=0, bytes=0, to_download=0))
        b["files"] += 1
        b["bytes"] += e["size"]
        b["to_download"] += 0 if e.get("starter_local") else e["size"]
    doc = dict(note=__doc__.split("\n\n")[0], date_cutoff="2022-12-01", tiers=TIER_NAMES,
               revisions=revisions, se_sites=SE_SITES, by_source=by, n_files=len(entries),
               total_bytes=sum(e["size"] for e in entries), files=entries)
    json.dump(doc, open(os.path.join(HERE, "core_v0_manifest.json"), "w"), indent=1)
    for s, b in sorted(by.items(), key=lambda x: (x[1]["tier"], x[0])):
        print(f"tier {b['tier']} {s:14s} files {b['files']:4d}  {b['bytes'] / 1e9:7.2f} GB  "
              f"to download {b['to_download'] / 1e9:7.2f} GB")
    print(f"total {doc['total_bytes'] / 1e9:.2f} GB in {len(entries)} files; "
          f"to download {sum(b['to_download'] for b in by.values()) / 1e9:.2f} GB")


if __name__ == "__main__":
    sys.exit(main())
