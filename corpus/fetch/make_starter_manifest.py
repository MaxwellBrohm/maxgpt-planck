"""make_starter_manifest.py: resolve the starter download (about 3.5 GB) to exact files, sizes and
sha256 from the Hugging Face API, and write corpus/fetch/starter_manifest.json. Metadata only.

What the starter is for: tokenizer v0, the vocab sweep (P-158) and the corpus screening pilot.
Every source is open under D8 (human-written or public domain, CC BY / BY-SA / CC0, Apache-2.0).
The date gate (Q6, 2022-12-01) applies at build time where documents carry a date; here it only
picks which CCCC snapshots to fetch (all before 2022-12).
"""
import json, os, sys, urllib.request

API = "https://huggingface.co/api/datasets"
HERE = os.path.dirname(os.path.abspath(__file__))


def tree(ds, path=""):
    out, url = [], f"{API}/{ds}/tree/main/{path}"
    while url:
        r = urllib.request.urlopen(url, timeout=120)
        out += json.load(r)
        link = r.headers.get("Link") or ""
        url = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
    return out


def entry(ds, f, why):
    lfs = f.get("lfs") or {}
    return dict(dataset=ds, path=f["path"], size=f.get("size", 0), sha256=lfs.get("oid"), why=why,
                url=f"https://huggingface.co/datasets/{ds}/resolve/main/{f['path']}")


def files(ds, path=""):
    return [f for f in tree(ds, path) if f["type"] == "file"]


def main():
    m = []
    # chat: human-written
    for f in files("OpenAssistant/oasst2"):
        if f["path"] == "2023-11-05_oasst2_all.trees.jsonl.gz":
            m.append(entry("OpenAssistant/oasst2", f, "human multi-turn chat (Apache-2.0)"))
    for f in files("databricks/databricks-dolly-15k"):
        if f["path"].endswith(".jsonl"):
            m.append(entry("databricks/databricks-dolly-15k", f, "human single-turn (CC-BY-SA-3.0)"))
    # CCCC: shard 0000 of 10 snapshots spread evenly over those before 2022-12
    snaps = sorted(f["path"] for f in tree("common-pile/cccc") if f["type"] == "directory" and f["path"] < "CC-MAIN-2022-49")
    pick = [snaps[round(i * (len(snaps) - 1) / 9)] for i in range(10)]
    for s in pick:
        fs = sorted(files("common-pile/cccc", s), key=lambda f: f["path"])
        m.append(entry("common-pile/cccc", fs[0], f"plain web prose, snapshot {s}"))
    # Project Gutenberg: 3 shards spread over the 30
    pg = sorted(files("common-pile/project_gutenberg", "v0/documents"), key=lambda f: f["path"])
    for i in (0, 14, 29):
        m.append(entry("common-pile/project_gutenberg", pg[i], "public-domain books"))
    # StackExchange, non-technical sites (main sites only), about 0.5 GB
    se_sites = ["english", "diy", "cooking", "bicycles", "travel", "parenting", "pets", "workplace",
                "interpersonal", "outdoors", "gardening", "boardgames", "fitness", "money", "lifehacks",
                "expatriates", "movies", "writing", "woodworking", "coffee", "crafts"]
    se_total = 0
    for site in se_sites:
        try:
            fs = [f for f in tree("common-pile/stackexchange", f"{site}.stackexchange.com/documents") if f["type"] == "file"]
        except Exception as e:
            print(f"skip {site}: {e}", file=sys.stderr)
            continue
        for f in fs:
            if se_total > 550e6:
                break
            m.append(entry("common-pile/stackexchange", f, f"Q&A register, {site}"))
            se_total += f.get("size", 0)
    # Ubuntu IRC: the first shard (the oldest logs, well before the date gate)
    irc = sorted(files("common-pile/ubuntu_irc", "raw/documents"), key=lambda f: f["path"])
    m.append(entry("common-pile/ubuntu_irc", irc[0], "multi-party chat logs (public domain)"))
    # Wikimedia, non-encyclopedic parts
    for f in files("common-pile/wikimedia"):
        if f["path"] in ("00000_wikibooks.com.jsonl.gz", "00001_wikibooks.com.jsonl.gz",
                         "00000_wikivoyage.com.jsonl.gz", "00000_wikinews.com.jsonl.gz"):
            m.append(entry("common-pile/wikimedia", f, "non-encyclopedic wiki text (CC BY-SA)"))
    total = sum(e["size"] for e in m)
    missing = [e["path"] for e in m if not e["sha256"]]
    json.dump(dict(total_bytes=total, n_files=len(m), files=m), open(os.path.join(HERE, "starter_manifest.json"), "w"), indent=1)
    by = {}
    for e in m:
        by[e["dataset"]] = by.get(e["dataset"], 0) + e["size"]
    for k, v in by.items():
        print(f"{k:40s} {v / 1e9:6.2f} GB")
    print(f"total {total / 1e9:.2f} GB in {len(m)} files; no sha256 for {missing}")


if __name__ == "__main__":
    main()
