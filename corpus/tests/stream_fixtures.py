"""A fake core manifest for the stream_core tests: raw files in a 'remote' directory fetched through
file:// URLs (curl resumes those too), one starter-local file, and an oracle that computes the
expected output straight from the readers with a plain-Python dedup (full sha1 and id sets).

Planted cases (each text carries its marker):
- DUP_ACROSS: the same text in cccc 2019-04 file 0 and cccc 2020-10 file 0: only the 2019 copy stays.
- NEAR_DUP: DUP_ACROSS with its marker changed and a line added, in cccc 2020-10 but dated
  2018-06-01: exact dedup keeps both; the near-dedup pass (neardedup_lsh.py) over the runner's
  MinHash sidecars keeps the earlier-dated NEAR_DUP and drops dupA (its keep rule reads the date
  the sidecar carries; without it, file order would keep dupA).
- WITHIN_DUP: two docs with the same text in one file: the second is dropped.
- SAME_ID: two docs with one id and different texts: the second is dropped (dup_id).
- CROSS_SOURCE: a Gutenberg book (tier 2) whose text a cccc page (tier 5) repeats: the book stays.
- DATEGATE: a cccc page dated after the cutoff: dropped by the reader.
- BIG_BOOK: a book longer than the test shard size, so one raw file spans shards.
"""
import gzip
import hashlib
import json
import os

import numpy as np

import extract
import stream_io as SIO
import stream_work as W
from fixtures import book, cp, prose

RARE = [a + b for a in ("zor", "quil", "vemb", "trax", "plon", "gref", "mulk", "sniv")
        for b in ("ade", "ette", "ion", "ery", "ist", "ow", "um", "ank")]    # shares no 5-gram
DUP_ACROSS = "DUP_ACROSS " + " ".join(f"the {w} and" for w in RARE)
WITHIN_DUP = "WITHIN_DUP " + prose("within-dup", 6)
NEAR_DUP = DUP_ACROSS.replace("DUP_ACROSS", "NEAR_DUP") + " One more line."   # near, not exact
CROSS_SOURCE = "CROSS_SOURCE " + prose("cross-source", 8)
BIG_BOOK = "BIG_BOOK " + "\n\n".join(prose(f"big-{k}", 5) for k in range(40))
CCCC = ["CC-MAIN-2019-04/cccc-CC-MAIN-2019-04-0000.json.gz",
        "CC-MAIN-2019-04/cccc-CC-MAIN-2019-04-0001.json.gz",
        "CC-MAIN-2020-10/cccc-CC-MAIN-2020-10-0000.json.gz"]
GUT = "v0/documents/00000_project_gutenberg.jsonl.gz"


def _cccc(k):
    docs = [cp(f"c{k}-{i}", prose(f"c{k}-{i}", 6), f"2019-0{1 + k}-11T05:33:58Z",
               url=f"example.org/{k}/{i}") for i in range(10)]
    if k == 0:
        docs += [cp("dupA", DUP_ACROSS, "2019-01-02", url="example.org/dupA"),
                 cp("w1", WITHIN_DUP, "2019-01-02"), cp("w2", WITHIN_DUP, "2019-01-03"),
                 cp("same-id", "SAME_ID first " + prose("sid1", 6), "2019-01-02"),
                 cp("same-id", "SAME_ID second " + prose("sid2", 6), "2019-01-02"),
                 cp("late", "DATEGATE " + prose("late", 6), "2023-05-01T00:00:00Z")]
    if k == 1:
        docs.append(cp("cross", CROSS_SOURCE, "2019-02-02"))
    if k == 2:
        docs += [cp("dupB", DUP_ACROSS, "2020-03-02"), cp("near", NEAR_DUP, "2018-06-01")]
    return docs


def _gutenberg():
    docs = [cp(i, book(f"sb{i}"), None, language="en", license="Public Domain") for i in range(3)]
    docs += [cp(10, CROSS_SOURCE, None, language="en", license="Public Domain"),
             cp(11, BIG_BOOK, None, language="en", license="Public Domain")]
    for d in docs:
        del d["created"]
    return docs


def _dolly():
    return [{"instruction": prose(f"di{i}", 2), "context": "", "category": "open_qa",
             "response": prose(f"dr{i}", 3)} for i in range(6)]


def _bytes(rows, gz):
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
    return gzip.compress(data, mtime=0) if gz else data


def make_world(root, corrupt_gutenberg=False, extra_entries=(), gutenberg_files=1):
    """Write remote/, starter/ and manifest.json under root. -> manifest path."""
    remote, starter, entries = os.path.join(root, "remote"), os.path.join(root, "starter"), []

    def add(source, dataset, path, data, tier, local=False):
        e = dict(source=source, dataset=dataset, revision="rev0", path=path, size=len(data),
                 sha256=hashlib.sha256(data).hexdigest(), tier=tier, role="core",
                 starter_local=local)
        for base in [remote] + ([starter] if local else []):
            p = os.path.join(base, dataset.replace("/", "__"), path)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(data)
        e["url"] = "file://" + os.path.join(remote, dataset.replace("/", "__"), path)
        entries.append(e)

    add("dolly", "databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl",
        _bytes(_dolly(), False), 1, local=True)
    gut = b"not a gzip file at all " * 40 if corrupt_gutenberg else _bytes(_gutenberg(), True)
    add("gutenberg", "common-pile/project_gutenberg", GUT, gut, 2)
    if gutenberg_files == 2:
        more = [cp(20 + i, book(f"more{i}"), None, language="en", license="Public Domain")
                for i in range(2)]
        add("gutenberg", "common-pile/project_gutenberg", GUT.replace("00000", "00001"),
            _bytes(more, True), 2)
    for k, path in enumerate(CCCC):
        add("cccc", "common-pile/cccc", path, _bytes(_cccc(k), True), 5)
    entries += list(extra_entries)
    entries.sort(key=lambda e: (e["tier"], e["source"], e["path"]))
    mp = os.path.join(root, "manifest.json")
    with open(mp, "w") as f:
        json.dump({"files": entries}, f, indent=1)
    return mp


def raw_path(root, entry, where="remote"):
    return os.path.join(root, where, entry["dataset"].replace("/", "__"), entry["path"])


def oracle(root, manifest_path, skip=()):
    """-> {source: [doc ids in commit order]} from the readers plus a plain dedup."""
    o = dict(extract.DEFAULTS, gutenberg_cap=0)
    seen_text, seen_id, out = set(), set(), {}
    for e in json.load(open(manifest_path))["files"]:
        spec = W.resolve_reader(e, {})
        if spec is None or e["path"] in skip:
            continue
        rel = e["dataset"].replace("/", "__") + "/" + e["path"]
        try:
            evs = list(W.load_reader(spec)(e["source"], raw_path(root, e), rel, o))
        except Exception:  # noqa: BLE001 (the corrupt file: the runner records it as failed)
            continue
        for ev in evs:
            if ev[0] != "keep":
                continue
            h = hashlib.sha1(ev[1]["text"].encode("utf-8")).hexdigest()
            if h in seen_text or ev[1]["id"] in seen_id:
                continue
            seen_text.add(h)
            seen_id.add(ev[1]["id"])
            out.setdefault(e["source"], []).append(ev[1]["id"])
    return out


def shard_paths(out, source, codec):
    d = os.path.join(out, source)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith("jsonl." + codec))


def output_ids(out, codec):
    """-> {source: [doc ids in shard order]}, checking each doc's sha1 and keys sidecar."""
    res = {}
    for src in sorted(os.listdir(out)):
        if src.startswith("_") or not os.path.isdir(os.path.join(out, src)):
            continue
        for p in shard_paths(out, src, codec):
            docs = list(SIO.iter_docs(p))
            keys = SIO.read_keys(p.replace("jsonl." + codec, "keys.u64"))
            assert len(keys) == len(docs), p
            for d, k in zip(docs, keys):
                h = hashlib.sha1(d["text"].encode("utf-8")).digest()
                assert d["meta"]["sha1"] == h.hex() and int(k) == int.from_bytes(h[:8], "big")
                res.setdefault(src, []).append(d["id"])
    return res


def ledger(out):
    with open(os.path.join(out, "LEDGER.jsonl")) as f:
        return [json.loads(x) for x in f]


def tree_hashes(out):
    """sha256 of every output shard, keys and sidecar file (not the ledger or status)."""
    got = {}
    for dp, _, fs in os.walk(out):
        for f in fs:
            if ".jsonl." in f or f.endswith((".keys.u64", ".npy", ".bin")):
                p = os.path.join(dp, f)
                got[os.path.relpath(p, out)] = SIO.sha256_file(p)
    return got


def _wait_for(path, seconds=15):
    import time
    t0 = time.time()
    while not os.path.exists(path) and time.time() - t0 < seconds:
        time.sleep(0.05)


def reader_that_dies(source, path, rel, o):
    """A Common Pile reader whose process dies on cccc 2019-04 file 1 (stands in for the OOM
    killer) while file 0 is still being read in the other worker: both touch a start marker in
    $STREAM_TEST_SYNC and wait for the other's, so the pool breaks with file 0 mid-run. Spawned
    workers import it from here."""
    sync = os.environ["STREAM_TEST_SYNC"]
    for k in (0, 1):
        if path.endswith(CCCC[k]):
            open(os.path.join(sync, f"started{k}"), "w").close()
            _wait_for(os.path.join(sync, f"started{1 - k}"))
            if k == 1:
                os._exit(137)
            import time
            time.sleep(1.0)                  # file 0: still running when the death is seen
    import readers_cp
    yield from readers_cp.read_common_pile(source, path, rel, o)


def len_row(rec):
    """A test sidecar hook: (text bytes, exact key) per document."""
    b = rec["text"].encode("utf-8")
    row = np.zeros((), dtype=[("n", "<u4"), ("k", "<u8")])
    row["n"], row["k"] = len(b), int.from_bytes(hashlib.sha1(b).digest()[:8], "big")
    return row
