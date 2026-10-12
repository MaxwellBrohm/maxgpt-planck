"""A fake core manifest for core_finalize.py: dolly and foodista (tier 1), Gutenberg (tier 2) and
three CCCC files (tier 5), fetched through file:// URLs by stream_core.py. Texts are drawn from a
made-up vocabulary, so unrelated documents share no word 5-gram and only the planted pairs are
near-duplicates.

Planted cases (the marker is in the text):
- BODY_MARKER: one page body under 12 URLs, each copy with its own short tail (near-duplicates,
  not exact). Near-dedup keeps copy 0 (earliest date); its body lines then occur in one group
  only, so boilerplate removal leaves them. Counting lines before near-dedup would strip them.
- NAV_MARKER: a navigation line on 16 pages with distinct URLs: stripped everywhere.
- NEARBOILER_MARKER: a line on 8 pages plus 3 captures of one URL (distinct content): 9 URL
  groups, under the 10-group threshold, so it stays in all 11 (a per-document count reaches 11).
- SHORT_MARKER: a page that is mostly the NAV line: under 200 bytes once stripped.
- POSTCLEAN_MARKER: the same short text on two pages under two different boilerplate blocks
  (not near-duplicates); stripping makes them identical and pass D keeps the first.
- EARLYCLEAN_MARKER: a short Gutenberg text (tier 2) that a CCCC page repeats under a
  boilerplate block: once stripped, the page equals the book, and pass D drops the page.
- GUTBOOK_MARKER: a Gutenberg book (tier 2) and a near copy on a CCCC page (tier 5): the book
  stays, the page goes (lead other_source:gutenberg).
- A dolly row and a foodista page dated 2019 with the same text plus a word: dolly (tier 1 chat
  anchor, priority before foodista) stays although it is undated.
"""
import gzip
import hashlib
import json
import os
import random

import extract
from readers_chat import dolly_row

SYL = ["ka", "lo", "mi", "ra", "te", "vu", "po", "si", "na", "de", "gu", "fe", "zo", "bi", "wa"]
VOCAB = [a + b + c for a in SYL for b in SYL for c in SYL]
STOP = ("the", "and", "of", "to", "a", "in")
BY3 = "Creative Commons - Attribution - https://creativecommons.org/licenses/by/3.0/"


def uniq(tag, n=20):
    """2n words: stopwords alternating with random made-up words (seeded by tag)."""
    r = random.Random(str(tag))
    return " ".join(f"{r.choice(STOP)} {r.choice(VOCAB)}" for _ in range(n))


def lines(tag, k, n=20):
    return [uniq(f"{tag}-{j}", n) for j in range(k)]


NAV = "NAV_MARKER " + uniq("nav", 12)
NB = "NEARBOILER_MARKER " + uniq("nb", 8)
BODY = ["BODY_MARKER " + uniq("body0")] + lines("body", 4)
ABLOCK, BBLOCK = lines("ablock", 5), lines("bblock", 5)
C = "POSTCLEAN_MARKER " + uniq("c", 30)
EC = "EARLYCLEAN_MARKER " + uniq("ec", 30)
GUT = "\n\n".join(["GUTBOOK_MARKER " + uniq("g0")] + lines("gut", 9))
DOLLY = {"instruction": "DOLLY_MARKER " + uniq("di", 10), "context": "",
         "category": "open_qa", "response": uniq("dr", 40)}


def cc(i, text, created, url=None):
    d = {"id": str(i), "text": text, "source": "fixture", "added": "2024-06-03T21:29:47Z",
         "created": created, "metadata": {}}
    if url:
        d["metadata"]["url"] = url
    return d


def _cccc(k):
    d = "2019-02-1{}T05:33:58Z".format(k) if k < 2 else "2020-10-05T00:00:00Z"
    docs = []
    if k == 0:
        docs += [cc(f"base{i}", "\n".join(lines(f"p{i}", 6) + [NAV] + ([NB] if i < 8 else [])), d,
                    f"example.org/base{i}") for i in range(15)]
        docs.append(cc("short", NAV + "\nSHORT_MARKER " + uniq("short", 8), d, "example.org/s"))
        docs += [cc(f"qa{i}", "\n".join(lines(f"qa{i}", 8) + ABLOCK), d, f"example.org/qa{i}")
                 for i in range(10)]
        docs.append(cc("p1", "\n".join([C] + ABLOCK), d, "example.org/p1"))
    if k == 1:
        docs += [cc(f"qb{i}", "\n".join(lines(f"qb{i}", 8) + BBLOCK), d, f"example.org/qb{i}")
                 for i in range(10)]
        docs.append(cc("p2", "\n".join([C] + BBLOCK), d, "example.org/p2"))
        docs.append(cc("nav15", "\n".join(lines("n15", 6) + [NAV]), d, "example.org/n15"))
    docs += [cc(f"copy{c}", "\n".join(BODY + [uniq(f"tail{c}", 3)]), f"2019-01-{c + 1:02d}",
                f"example.org/copy{c}") for c in range(6 * k, 6 * k + 6) if k < 2]
    docs.append(cc(f"same{k}", "\n".join(lines(f"same{k}", 6) + [NB]), d, "example.org/same"))
    if k == 2:
        docs.append(cc("gc", GUT + " zokabi", d, "example.org/gc"))
        docs.append(cc("ec", "\n".join([EC] + ABLOCK), d, "example.org/ec"))
        docs += [cc(f"late{i}", "\n".join(lines(f"late{i}", 6)), d, f"example.org/late{i}")
                 for i in range(3)]
    return docs


def _gutenberg():
    docs = [{"id": str(i), "text": "\n\n".join(lines(f"book{i}", 10)), "source": "fixture",
             "metadata": {"language": "en", "license": "Public Domain"}} for i in range(3)]
    docs += [{"id": i, "text": t, "source": "fixture",
              "metadata": {"language": "en", "license": "Public Domain"}}
             for i, t in (("9", GUT), ("8", EC))]
    return docs


def dolly_text():
    return dolly_row(0, DOLLY, "x", extract.DEFAULTS)[1]["text"]


def _dolly():
    return [DOLLY] + [{"instruction": uniq(f"i{i}", 8), "context": "", "category": "open_qa",
                       "response": uniq(f"r{i}", 30)} for i in range(1, 4)]


def _foodista():
    def fd(i, created, text):
        return {"id": i, "text": text, "source": "foodista", "added": "2024-05-23T00:42:18",
                "created": created, "metadata": {"license": BY3, "url": f"https://f.com/{i}",
                                                 "authors": ["B. Cook"]}}
    return [fd(0, "2019-05-01T00:00:00", dolly_text() + " zokabi"),
            fd(1, "2018-01-01T00:00:00", "\n".join(lines("food1", 5)))]


def _bytes(rows, gz):
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    return gzip.compress(data, mtime=0) if gz else data


CCCC = ["CC-MAIN-2019-04/cccc-CC-MAIN-2019-04-0000.json.gz",
        "CC-MAIN-2019-04/cccc-CC-MAIN-2019-04-0001.json.gz",
        "CC-MAIN-2020-10/cccc-CC-MAIN-2020-10-0000.json.gz"]


def make_world(root, url_prefix=None):
    """remote/ and manifest.json under root. url_prefix replaces 'file://<remote>' in the URLs
    (for the --url-rewrite test). -> manifest path."""
    remote, entries = os.path.join(root, "remote"), []

    def add(source, dataset, path, data, tier):
        p = os.path.join(remote, dataset.replace("/", "__"), path)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
        base = url_prefix or ("file://" + remote)
        entries.append(dict(source=source, dataset=dataset, revision="rev0", path=path,
                            size=len(data), sha256=hashlib.sha256(data).hexdigest(), tier=tier,
                            role="core", url=f"{base}/{dataset.replace('/', '__')}/{path}"))

    add("dolly", "databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl",
        _bytes(_dolly(), False), 1)
    add("foodista", "common-pile/foodista", "v0/documents/00000_foodista.jsonl.gz",
        _bytes(_foodista(), True), 1)
    add("gutenberg", "common-pile/project_gutenberg",
        "v0/documents/00000_project_gutenberg.jsonl.gz", _bytes(_gutenberg(), True), 2)
    for k, path in enumerate(CCCC):
        add("cccc", "common-pile/cccc", path, _bytes(_cccc(k), True), 5)
    entries.sort(key=lambda e: (e["tier"], e["source"], e["path"]))
    mp = os.path.join(root, "manifest.json")
    with open(mp, "w") as f:
        json.dump({"files": entries}, f, indent=1)
    return mp
