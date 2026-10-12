"""Open sources for the human pools, the persona seeds and the safety rubric (BANKPASS s3 P, Q, Z; s8 decision 1,
taken by Max 2026-10-04). Each source is pinned (URL, and a revision or expected sha256 where the host offers one),
and its licence is checked AT THE SOURCE when it is fetched: an evidence file is saved beside the data and must hold
the licence statement below. A source whose evidence does not hold it is written with status "refused", and
require() refuses it, so no builder reads it.

    python -m bankpass.sources fetch ROOT [--only a,b]      (PC, network; restartable: a file whose sha256 matches
                                                             its pin, or its SOURCE.json record, is not fetched again)
    python -m bankpass.sources verify ROOT                   (re-hash every file; exit 1 on any difference)

Requests carry a plain tool User-Agent. A host that answers 403 is reported as blocked and left alone (no browser
disguise: that would be getting past bot detection). Layout: ROOT/<source>/<files>, ROOT/<source>/SOURCE.json."""
import datetime
import json
import os
import re
import sys
import urllib.request

from bankpass import store

UA = "planck-bankpass/0 (research data pipeline; python-urllib)"
HF = "https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA/resolve/5b4cd35ab46490c1da1bd2b5a2324d6f871be180/"
GH = ("https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/"
      "5faf2ba42d7b1c0977169ec3611df25a3c08eb13/")
NEMO_SHA = (  # Hugging Face LFS sha256 of the 11 parquet shards at that revision (API tree listing, 2026-10-04)
    "af5d3e1c0ca2ca9cd12b5bcfc6ca5a850cdc6b7d6c24eb89ce948b23bed9c7e7",
    "bce1f98e9aaff5f115dee7f12fa115599381ccb1b4d7594c6d72ce627a617010",
    "dabeb21e9de465ef26f5852b0e2d4a9fbbc5d513b4699db601cd632aa2705a9c",
    "06fcc5536b191777cadd056bcac93058c7d976195cc16353fa51a9548035c782",
    "a0db243958664adfbe824a79616aa804ed247ecaaca00a3a94d8b77f1b573dc9",
    "d9676be60167f8ef94fcbaad35613e8057fb92c15bdf3d99847372c793c9064c",
    "ac86eedbffce275c92b80f1e7bb2128116e8621c15d4252b0f408ca457bce946",
    "85c07ea71ef44eb288dd2714593cb6b4b7592dcea88907fdcc7d395d8941397b",
    "04905f56ef5cf35be55d9a0a4a5e577fb9d8908168933ee0e5b414ae8d27f83b",
    "e2c01cd32073abd6e231ac17ef8530513f22b6473803acbd5fa2c917f6ee844d",
    "2026d1f82d3c1534a2aaf98b0d7c2a9cff2f8f9b161ff55ff51cf929a37bc0d2",
)
SOURCES = {
    "ssa_names": dict(
        title="SSA baby names, national data (Social Security card applications)", license="cc0",
        files={"names.zip": ("https://www.ssa.gov/oact/babynames/names.zip", None)},
        evidence={"datagov_record.html": ("https://catalog.data.gov/dataset/"
                                          "baby-names-from-social-security-card-applications-national-data",
                                          r"creativecommons\.org/publicdomain/zero/1\.0")},
        basis="the data.gov record the Social Security Administration publishes for this file names CC0 1.0, a "
              "public-domain dedication (BANKPASS: public domain)"),
    "census_surnames": dict(
        title="Frequently occurring surnames from the 2010 Census (U.S. Census Bureau)", license="public-domain",
        files={"names.zip": ("https://www2.census.gov/topics/genealogy/2010surnames/names.zip", None),
               "surnames.pdf": ("https://www2.census.gov/topics/genealogy/2010surnames/surnames.pdf", None)},
        evidence={"product_page.html": ("https://www.census.gov/topics/population/genealogy/data/2010_surnames.html",
                                        r"2010surnames/names\.zip")},
        basis="a U.S. Census Bureau product, a work of the U.S. government (17 U.S.C. 105, no copyright); the product "
              "page and the methodology paper state no licence and no restriction. The evidence check only confirms "
              "the product page lists this file."),
    "geonames_cities": dict(
        title="GeoNames cities15000 and countryInfo", license="cc-by-4.0",
        files={"cities15000.zip": ("https://download.geonames.org/export/dump/cities15000.zip", None),
               "countryInfo.txt": ("https://download.geonames.org/export/dump/countryInfo.txt", None)},
        evidence={"readme.txt": ("https://download.geonames.org/export/dump/readme.txt",
                                 r"Creative Commons Attribution 4\.0 License")},
        basis="readme.txt of the GeoNames dump: CC BY 4.0. The dump changes daily: pinned by sha256 and date"),
    "nemotron_personas_usa": dict(
        title="Nemotron-Personas-USA (NVIDIA), revision 5b4cd35a", license="cc-by-4.0",
        revision="5b4cd35ab46490c1da1bd2b5a2324d6f871be180",
        files={f"data/train-{i:05d}-of-00011.parquet": (HF + f"data/train-{i:05d}-of-00011.parquet", NEMO_SHA[i])
               for i in range(11)},
        evidence={"README.md": (HF + "README.md", r"(?m)^license: cc-by-4\.0\s*$")},
        basis="dataset card at the pinned revision: license cc-by-4.0, ready for commercial use. The persona text is "
              "synthetic (NeMo Data Designer, gpt-oss-120b per the card): prompt side only (BANKPASS s3 Q)"),
    "ldnoobw": dict(
        title="List of Dirty, Naughty, Obscene and Otherwise Bad Words, English, commit 5faf2ba4",
        license="cc-by-4.0", revision="5faf2ba42d7b1c0977169ec3611df25a3c08eb13",
        files={"en": (GH + "en", None)},
        evidence={"LICENSE": (GH + "LICENSE", r"Attribution 4\.0 International")},
        basis="LICENSE at the pinned commit: CC BY 4.0. Rubric only (BANKPASS s3 Z): never training text"),
}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get(url, path, opener=None):
    """stream url to path (via path.part); returns (status, bytes). A 4xx/5xx raises urllib.error.HTTPError."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    tmp = path + ".part"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with (opener or urllib.request.urlopen)(req, timeout=120) as r, open(tmp, "wb") as f:
        n = 0
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            n += len(chunk)
    os.replace(tmp, path)
    return n


def record_path(root, name):
    return os.path.join(root, name, "SOURCE.json")


def read_record(root, name):
    p = record_path(root, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fetch(root, name, get=_get):
    """fetch one source, check its licence evidence, write SOURCE.json; returns the record."""
    s, d = SOURCES[name], os.path.join(root, name)
    old = read_record(root, name) or {}
    rec = {"schema": "bankpass-source-v1", "name": name, "title": s["title"], "license": s["license"],
           "basis": s["basis"], "revision": s.get("revision"), "files": {}, "evidence": {}, "problems": []}
    for group, table in (("evidence", s["evidence"]), ("files", s["files"])):
        for rel, spec in table.items():
            url, pin, pattern = (spec[0], None, spec[1]) if group == "evidence" else (spec[0], spec[1], None)
            path = os.path.join(d, rel)
            prev = (old.get(group) or {}).get(rel) or {}
            want = pin or prev.get("sha256")
            if not (want and os.path.exists(path) and store.sha256_file(path) == want):
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    get(url, path)
                    prev = {}
                    if not os.path.exists(path):
                        raise OSError("nothing written")
                except Exception as e:          # HTTPError, URLError, timeouts: recorded, never retried in disguise
                    if pin or not os.path.exists(path):
                        rec["problems"].append(f"{rel}: {type(e).__name__} {e}".strip())
                        continue
                    # a copy Max saved by hand from the same URL (a host that refuses tools): used, and marked so
                    rec["notes"] = rec.get("notes", []) + [f"{rel}: fetch failed ({type(e).__name__} {e}); the copy "
                                                           "already at this path was placed by hand and is used"]
                    prev = dict(prev, retrieved=prev.get("retrieved") or "by hand, found " + _now())
            sha = store.sha256_file(path)
            if pin and sha != pin:
                rec["problems"].append(f"{rel}: sha256 {sha[:12]} is not the pinned {pin[:12]}")
            rec[group][rel] = {"url": url, "sha256": sha, "bytes": os.path.getsize(path),
                               "retrieved": prev.get("retrieved") or _now()}
            if group == "evidence":
                with open(path, encoding="utf-8", errors="replace") as f:
                    m = re.search(pattern, f.read())
                rec[group][rel]["licence_match"] = m.group(0) if m else None
                if not m:
                    rec["problems"].append(f"{rel}: licence statement not found ({pattern})")
    missing = [r for r in list(s["files"]) + list(s["evidence"]) if r not in rec["files"] and r not in rec["evidence"]]
    rec["status"] = "ok" if not rec["problems"] else "refused" if any("licence" in p for p in rec["problems"]) \
        else "incomplete"
    rec["missing"] = missing
    rec["retrieved"] = max([v["retrieved"] for g in ("files", "evidence") for v in rec[g].values()] or [None])
    os.makedirs(d, exist_ok=True)
    with open(record_path(root, name), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    return rec


class SourceRefused(RuntimeError):
    pass


def require(root, name):
    """the SOURCE.json record of a usable source; raises unless status ok, the licence is the registry's, and every
    file still hashes to its record."""
    rec = read_record(root, name)
    if not rec or rec.get("status") != "ok":
        raise SourceRefused(f"{name}: status {rec and rec.get('status')!r} {rec and rec.get('problems')}")
    if rec.get("license") != SOURCES[name]["license"] or rec["license"] not in store.HUMAN_LICENSES:
        raise SourceRefused(f"{name}: licence {rec.get('license')!r}")
    bad = verify_one(root, name, rec)
    if bad:
        raise SourceRefused(f"{name}: {bad}")
    return rec


def verify_one(root, name, rec=None):
    rec = rec or read_record(root, name) or {}
    out = []
    for group in ("files", "evidence"):
        for rel, v in (rec.get(group) or {}).items():
            p = os.path.join(root, name, rel)
            if not os.path.exists(p) or store.sha256_file(p) != v["sha256"]:
                out.append(f"{rel} changed or missing")
    return out


def source_block(rec, rel, row):
    """the s2h human source fields for one item from file rel of a source record."""
    f = rec["files"][rel]
    return {"name": rec["name"], "title": rec["title"], "url": f["url"], "license": rec["license"],
            "retrieved": f["retrieved"], "file_sha256": f["sha256"], "row": str(row), "revision": rec.get("revision")}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd, root = argv[0], argv[1]
    names = argv[3].split(",") if len(argv) > 3 and argv[2] == "--only" else list(SOURCES)
    rc = 0
    for n in names:
        if cmd == "fetch":
            r = fetch(root, n)
            print(f"{n}: {r['status']} {sum(v['bytes'] for v in r['files'].values())} bytes; problems {r['problems']}",
                  flush=True)
        else:
            bad = verify_one(root, n)
            print(f"{n}: {'ok' if not bad else bad}")
            rc |= bool(bad)
    return rc


if __name__ == "__main__":
    sys.exit(main())
