"""Bookkeeping of core_finalize.py: the stage1 ledger, pass D (exact dedup of the cleaned text),
the near-dedup relation counts, the per-tier record and FINAL/MANIFEST.json.

Relation counts: every document near-dedup dropped is classed by where its kept lead came from,
using the ledger's per-file segments to map shard rows back to raw files: same_file, same_group
(another raw file in the same directory: for CCCC, the same crawl snapshot), other_group (same
source, another directory: for CCCC, a recrawl in another snapshot) or other_source:<source>.
"""
import json
import os
import time

import numpy as np

import core_final_work as FW
import stream_io as SIO

STATUS = {"exact_dedup": "done (pass A, and pass D on cleaned text)",
          "near_dedup": "done (MinHash LSH 14x9, verify 0.7)", "boilerplate": "done (cccc, wikimedia)",
          "quality_heuristics": "not applied", "pii_web": "not applied (IRC nicknames are mapped "
          "at extraction)", "decon": "pending (every doc carries decon 'pending')"}


def ledger(stage1):
    out = []
    with open(os.path.join(stage1, "LEDGER.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break                                   # a torn last line was never committed
    return out


def _keys_path(dst):
    return dst.rsplit(".jsonl.", 1)[0] + ".keys.u64"


def pass_d(c, mine, jobs, res, prev, log):
    """Drop final texts that equal an earlier tier's, or an earlier one of this tier."""
    old = []
    for t in sorted(prev):
        rec = json.load(open(os.path.join(c["final"], "_tiers", f"tier{t}.json")))
        old += [SIO.read_keys(os.path.join(c["final"], s["keys"])) for s in rec["shards"]]
    old = np.sort(np.concatenate(old)) if old else np.zeros(0, np.uint64)
    cur = [SIO.read_keys(_keys_path(j["dst"])) for j in jobs]
    allk = np.concatenate(cur) if cur else np.zeros(0, np.uint64)
    dup = np.zeros(len(allk), dtype=bool)
    if len(old) and len(allk):
        i = np.minimum(np.searchsorted(old, allk), len(old) - 1)
        dup |= old[i] == allk
    _, first = np.unique(allk, return_index=True)
    once = np.zeros(len(allk), dtype=bool)
    once[first] = True
    dup |= ~once
    off, rewritten = np.concatenate(([0], np.cumsum([len(x) for x in cur]))), 0
    for n, (j, r) in enumerate(zip(jobs, res)):
        d = np.flatnonzero(dup[off[n]:off[n + 1]])
        if not len(d):
            continue
        x = FW.drop_rows(dict(j, drop=d.tolist()))
        r["dropped"]["dup_exact_post_clean"] = {"docs": int(len(d)), "bytes": x["dropped_bytes"]}
        r.update(docs_kept=x["docs"], bytes_kept=r["bytes_kept"] - x["dropped_bytes"],
                 ubytes=x["ubytes"], bytes=x["bytes"], sha256=x["sha256"])
        rewritten += 1
    log(f"pass D: {int(dup.sum()):,} of {len(allk):,} final docs dropped, "
        f"{rewritten} shards rewritten")
    return {"checked": int(len(allk)), "earlier_final_keys": int(len(old)),
            "dropped": int(dup.sum()), "shards_rewritten": rewritten}


def relations(stage1, events, rels, spec):
    """-> {source: {relation: docs}} for the near-dedup drops of the non-frozen shards."""
    fids, per = [], {}
    for e in events:
        if e["event"] == "file_done":
            for s in e["segments"]:
                per.setdefault(s["shard"], []).append((len(fids), s["docs"]))
            fids.append(e["fid"])
    src = {e["fid"]: e["source"] for e in events if e["event"] == "file_done"}
    names = sorted(set(src.values()))
    f_src = np.array([names.index(src[f]) for f in fids] or [0], dtype=np.int64)
    dirs = sorted({os.path.dirname(f) for f in fids})
    f_dir = np.array([dirs.index(os.path.dirname(f)) for f in fids] or [0], dtype=np.int64)
    cache = {}

    def rows(k):
        if k not in cache:
            v = per.get(rels[k], [])
            cache[k] = np.repeat(np.array([j for j, _ in v], dtype=np.int32),
                                 [n for _, n in v])
        return cache[k]

    out = {}
    for k, s in enumerate(spec):
        if s["frozen"]:
            continue
        nd = np.load(s["stem"] + ".nd.npy")
        d = np.flatnonzero((nd["keep"] == 0) & (nd["csize"] > 0))     # csize 0: not included
        if not len(d):
            continue
        own, ls, lr = rows(k)[d], nd["lshard"][d], nd["lrow"][d]
        lead = np.empty(len(d), dtype=np.int64)
        for u in np.unique(ls):
            m = ls == u
            lead[m] = rows(int(u))[lr[m]]
        c = out.setdefault(s["label"], {})
        same_src = f_src[own] == f_src[lead]
        for key, m in (("same_file", own == lead),
                       ("same_group", same_src & (own != lead) & (f_dir[own] == f_dir[lead])),
                       ("other_group", same_src & (f_dir[own] != f_dir[lead]))):
            c[key] = c.get(key, 0) + int(m.sum())
        for u, n in zip(*np.unique(f_src[lead[~same_src]], return_counts=True)):
            c["other_source:" + names[u]] = c.get("other_source:" + names[u], 0) + int(n)
    return out


def coverage(events, k):
    """-> {source: {expected, done, failed, missing}} for tier k's files in the manifest the last
    stage1 run used (failed: failed and never done; missing: neither), or None if unreadable."""
    starts = [e for e in events if e["event"] == "run_start"]
    try:
        with open(os.path.expanduser(starts[-1]["config"]["manifest"])) as f:
            files = json.load(f)["files"]
    except (IndexError, KeyError, OSError, ValueError):
        return None
    roles = starts[-1]["config"].get("roles") or ["core"]
    done = {e["fid"] for e in events if e["event"] == "file_done"}
    failed = {e["fid"] for e in events if e["event"] == "file_failed"} - done
    out = {}
    for e in files:
        if e.get("tier") != k or e.get("role", "core") not in roles:
            continue
        fid, b = f"{e['dataset']}/{e['path']}", out.setdefault(
            e["source"], {"expected": 0, "done": 0, "failed": [], "missing": []})
        b["expected"] += 1
        b["done"] += fid in done
        if fid not in done:
            b["failed" if fid in failed else "missing"].append(fid)
    return out


def tier_record(c, k, ih, events, earlier, mine, spec, rep, res, bp, passd, code, seconds):
    by = {}
    for s, r in zip(mine, res):
        b = by.setdefault(s["source"], {"stage1_shards": 0, "docs_in": 0, "bytes_in": 0,
                                        "docs_kept": 0, "bytes_kept": 0, "ubytes": 0,
                                        "bytes_compressed": 0, "dropped": {},
                                        "stripped": {"docs": 0, "bytes": 0}})
        b["stage1_shards"] += 1
        for key in ("docs_in", "bytes_in", "docs_kept", "bytes_kept", "ubytes"):
            b[key] += r[key]
        b["bytes_compressed"] += r["bytes"]
        for key in ("docs", "bytes"):
            b["stripped"][key] += r["stripped"][key]
        for why, d in r["dropped"].items():
            t = b["dropped"].setdefault(why, {"docs": 0, "bytes": 0})
            t["docs"] += d["docs"]
            t["bytes"] += d["bytes"]
    for src, b in bp.items():
        by[src]["boilerplate"] = b
    rels = [e["shard"] for e in earlier] + [s["shard"] for s in mine]     # spec order
    for src, rel in relations(c["stage1"], events, rels, spec).items():
        by[src]["near_dup_lead"] = rel
    nd = {key: rep[key] for key in ("params", "verify", "docs", "active", "dropped", "clusters",
                                    "docs_in_clusters", "edges", "by_label", "cluster_size_hist",
                                    "dropped_by_leader_label", "partitions", "seconds",
                                    "chain_floor")}
    rel = lambda p: os.path.relpath(p, c["stage1"])                     # noqa: E731
    cov = coverage(events, k)
    nd["largest"] = [dict(x, stem=rel(x["stem"])) for x in rep["largest"]]
    nd["audit"] = [dict(x, stem=rel(x["stem"]), leader_stem=rel(x["leader_stem"]))
                   for x in rep["audit"]]
    shards = []
    for s, r in zip(mine, res):
        kp = _keys_path(s["shard"])
        shards.append({"source": s["source"], "shard": s["shard"], "stage1": s["shard"],
                       "docs": r["docs_kept"], "ubytes": r["ubytes"], "bytes": r["bytes"],
                       "sha256": r["sha256"], "keys": kp,
                       "keys_sha256": SIO.sha256_file(os.path.join(c["final"], kp))})
    return {"tier": k, "input_hash": ih, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(seconds, 1), "params": {p: c[p] for p in ("verify", "chain_floor",
            "min_bytes", "bp_min_docs", "bp_sources", "level", "workers", "mem_mb")},
            "code_sha256": code, "status": STATUS, "sources": by, "pass_d": passd,
            "neardedup": nd, "shards": shards, "coverage": cov,
            "complete": None if cov is None else all(v["done"] == v["expected"]
                                                     for v in cov.values())}


def write_manifest(final):
    tiers, shards = {}, []
    d = os.path.join(final, "_tiers")
    for f in sorted(os.listdir(d)):
        if f.startswith("tier") and f.endswith(".json"):
            r = json.load(open(os.path.join(d, f)))
            tiers[str(r["tier"])] = {"input_hash": r["input_hash"], "finished": r["finished"],
                                     "sources": r["sources"], "pass_d": r["pass_d"],
                                     "complete": r.get("complete"), "coverage": r.get("coverage")}
            shards += r["shards"]
    tot = {"docs": sum(s["docs"] for s in shards), "ubytes": sum(s["ubytes"] for s in shards),
           "bytes_compressed": sum(s["bytes"] for s in shards), "shards": len(shards)}
    m = {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": STATUS, "totals": tot,
         "tiers": tiers, "shards": shards}
    with open(os.path.join(final, "MANIFEST.json.part"), "w") as f:
        json.dump(m, f, indent=1)
    os.replace(os.path.join(final, "MANIFEST.json.part"), os.path.join(final, "MANIFEST.json"))
