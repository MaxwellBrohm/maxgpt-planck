"""Helpers of neardedup_post.py: atomic JSON records, code hashes, the worker pool (spawned, capped
by MemAvailable, workers end with their parent) and OUT/MANIFEST.json."""
import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from stream_lock import exit_with_parent, mem_cap

HERE = os.path.dirname(os.path.abspath(__file__))


def base(path):
    return path.rsplit(".jsonl.", 1)[0]


def dump_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".part", path)


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def code_sha(names):
    h = hashlib.sha256()
    for n in names:
        with open(os.path.join(HERE, n), "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def pool_map(c, fn, jobs, log, what):
    """fn over jobs (largest first) in spawned workers, capped by MemAvailable; results in order."""
    n = max(1, min(c["workers"], mem_cap(c), len(jobs)))
    out, t0 = [None] * len(jobs), time.time()
    order = sorted(range(len(jobs)), key=lambda i: -jobs[i].get("ubytes", 0))
    if n == 1:
        for i in order:
            out[i] = fn(jobs[i])
    else:
        with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"),
                                 initializer=exit_with_parent, initargs=(os.getpid(),)) as ex:
            futs = {ex.submit(fn, jobs[i]): i for i in order}
            for k, fu in enumerate(as_completed(futs), 1):
                out[futs[fu]] = fu.result()
                if k % 50 == 0 or k == len(jobs):
                    log(f"{what}: {k}/{len(jobs)} shards, {time.time() - t0:.0f} s")
    log(f"{what}: done, {len(jobs)} shards, {n} workers, {time.time() - t0:.0f} s")
    return out


def manifest(params, code, man, msha, ih, shards, recs, rep, pd):
    by = {}
    for s, r in zip(shards, recs):
        b = by.setdefault(s["source"], dict.fromkeys(
            ("shards", "docs_in", "text_bytes_in", "docs", "text_bytes", "dropped_docs",
             "dropped_bytes", "ubytes", "bytes_compressed"), 0))
        b["tier"] = s["tier"]
        for k, v in (("shards", 1), ("docs_in", r["v0_docs"]), ("text_bytes_in", r["text_bytes_in"]),
                     ("docs", r["docs"]), ("text_bytes", r["text_bytes"]),
                     ("dropped_docs", r["dropped_docs"]), ("dropped_bytes", r["dropped_bytes"]),
                     ("ubytes", r["ubytes"]), ("bytes_compressed", r["bytes"])):
            b[k] += v
    tot = {k: sum(b[k] for b in by.values()) for k in next(iter(by.values())) if k != "tier"}
    rel = lambda p: os.path.relpath(p, pd)                                  # noqa: E731
    aud = np.array([a["sig_jaccard"] for a in rep["audit"]] or [np.nan])
    nd = {k: rep[k] for k in ("params", "verify", "chain_floor", "docs", "dropped", "clusters",
                              "docs_in_clusters", "edges", "cluster_size_hist",
                              "dropped_by_leader_label", "seconds")}
    nd["largest"] = [dict(x, stem=rel(x["stem"])) for x in rep["largest"]]
    nd["audit_summary"] = {"n": len(rep["audit"]), "under_0.7": int((aud < 0.7).sum()),
                           "under_0.5": int((aud < 0.5).sum()),
                           "median": float(np.nanmedian(aud)) if len(rep["audit"]) else None}
    return {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "what": "core v0 after a second near-dedup pass on the final text (neardedup_post.py)",
            "input": {"manifest_sha256": msha, "manifest_updated": man.get("updated"),
                      "shards": len(shards)},
            "input_hash": ih, "params": params,
            "code_sha256": {n: code_sha([n]) for n in code},
            "status": dict(man.get("status", {}), near_dedup_post=(
                f"done (final text, MinHash LSH 14x9, verify {params['verify']}, chain floor "
                f"{params['chain_floor']}, keep rule = core v0 priority)")),
            "totals": tot, "sources": by, "neardedup_post": nd,
            "shards": [{k: v for k, v in r.items() if k != "input_hash"} for r in recs]}
