"""core_finalize.py: passes B, C and D of the strict-open core v0 for one tier, over the stage1
shards that stream_core.py wrote (pass A: gates, exact dedup, MinHash sidecars <stem>.mh.npy).

    python corpus/core_finalize.py --stage1 ~/planck/data/core_v0_work/stage1 \\
        --final ~/planck/data/core_v0 --tier 1 [--workers 16]

Order (CORPUS 3.2): exact dedup (pass A) -> near-dedup -> boilerplate -> final shards -> exact
dedup of the cleaned text. Stage1 rows a late file displaced (core_displaced.py) take no part
and are dropped from the final shards as dup_exact_rank. Quality heuristics (3.2 step 4), the PII
scrub of web text (step 5) and 13-gram decontamination (step 6) are NOT applied here: every
document carries decon 'pending'.
1. Near-dedup (neardedup_lsh.find_near_dups, pass B) over this tier's sealed stage1 shards plus
   every earlier tier's shards as frozen members (never dropped; only their final survivors take
   part, FINAL/_masks/<source>/<shard>.fin.npy). Keep rule: lowest rank wins, rank = (tier, then
   source priority within the tier (stream_rank.PRIORITY, as pass A: chat anchors first), then
   date, then shard order). Writes <stage1 stem>.nd.npy for this tier's shards.
2. Boilerplate (sources in --bp-sources, default boilerplate.SOURCES): lines of near-dedup
   survivors found in >= --bp-min-docs distinct URL groups of the source are stripped
   (core_final_work.bp_pairs / count_bad, partitioned on disk).
3. Final shards, 1:1 with the stage1 shards (core_final_work.final_write): FINAL/<source>/<same
   name>, zstd level --level, .keys.u64 and .info.npy next to each.
4. Pass D: a final text equal to an earlier tier's final text, or to an earlier final text of this
   tier (possible only after stripping), is dropped (dup_exact_post_clean); only affected shards
   are rewritten. Then the tier's .fin.npy survivor masks are written.
The commit point is FINAL/_tiers/tier<K>.json (stats, near-dedup report, shard sha256s, input
hash). A rerun with the same inputs (sealed shards, earlier tiers' records, parameters) does
nothing; otherwise the tier is rebuilt from its stage1 shards. Heavy work holds the heavy-CPU lock
per chunk of --workers shards (pass B holds it throughout) at nice 15; touching STAGE1/STOP stops
between chunks. Exit codes: 0 done or up to date, 4 lock timeout, 5 stopped, 1 error.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import boilerplate as BP
import core_displaced as DR
import core_final_report as R
import core_final_work as FW
import extract
import hygiene as H
import neardedup_lsh as NL
from stream_lock import HeavyLock, LockTimeout, exit_with_parent, mem_cap
from stream_rank import PRIORITY, prio  # noqa: F401  (one priority list for passes A and B)
DEFAULTS = dict(stage1=None, final=None, tier=None, workers=16, nice=15, lock_timeout=7200,
                lock_file="~/planck/locks/pc_heavy.lock", verify=NL.ND.VERIFY, chain_floor=0.0,
                mem_mb=1024, min_bytes=H.MIN_BYTES, bp_min_docs=BP.MIN_DOCS,
                bp_sources=list(BP.SOURCES), level=10, pair_mb=256, mem_per_worker_gb=0.5,
                mem_reserve_gb=3)
PARAMS = ("verify", "chain_floor", "min_bytes", "bp_min_docs", "bp_sources", "level")


class Stop(Exception):
    pass


def stem(root, rel):
    return os.path.join(root, rel.rsplit(".jsonl.", 1)[0])


def mask_path(final, rel):
    return os.path.join(final, "_masks", rel.rsplit(".jsonl.", 1)[0] + ".fin.npy")


def tier_json(final, k):
    return os.path.join(final, "_tiers", f"tier{k}.json")


def sealed_shards(events):
    """Sealed stage1 shards with a MinHash sidecar, each with its tier, in rank order."""
    tier = {e["source"]: e.get("tier") for e in events if e["event"] == "file_done"}
    got = {e["shard"]: dict(e, tier=tier.get(e["source"])) for e in events
           if e["event"] == "shard_sealed" and "mh" in (e.get("sidecars") or {})}
    return sorted(got.values(), key=lambda s: (s["tier"], prio(s["source"]), s["shard"]))


def sha256_path(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_chunked(cfg, jobs, fn, log, what):
    """fn over jobs in spawned workers, a chunk of at most `workers` jobs per heavy-lock hold."""
    out, i = [], 0
    while i < len(jobs):
        if os.path.exists(os.path.join(cfg["stage1"], "STOP")):
            raise Stop()
        n = max(1, min(cfg["workers"], mem_cap(cfg), len(jobs) - i))
        with HeavyLock(cfg["lock_file"], cfg["lock_timeout"]) as lk:
            t0 = time.time()
            if n == 1:
                out += [fn(j) for j in jobs[i:i + n]]
            else:
                with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"),
                                         initializer=exit_with_parent,
                                         initargs=(os.getpid(),)) as ex:
                    out += list(ex.map(fn, jobs[i:i + n]))
        log(f"{what}: {i + n}/{len(jobs)} shards ({time.time() - t0:.0f} s, lock wait "
            f"{lk.waited} s)")
        i += n
    return out


def near_dedup(cfg, mine, earlier, work, log, inc=None):
    spec = [{"stem": stem(cfg["stage1"], s["shard"]), "tier": s["tier"] * 16 + prio(s["source"]),
             "label": s["source"], "frozen": True, "include": mask_path(cfg["final"], s["shard"])}
            for s in earlier]
    spec += [{"stem": stem(cfg["stage1"], s["shard"]), "tier": s["tier"] * 16 + prio(s["source"]),
              "label": s["source"], "frozen": False, "include": (inc or {}).get(s["shard"])}
             for s in mine]
    for s in spec[:len(earlier)]:
        if not os.path.exists(s["include"]):
            raise SystemExit(f"{s['include']} missing: finalize the earlier tier first")
    with HeavyLock(cfg["lock_file"], cfg["lock_timeout"]) as lk:
        rep = NL.find_near_dups(spec, work, verify=cfg["verify"], mem_mb=cfg["mem_mb"],
                                workers=cfg["workers"], chain_floor=cfg["chain_floor"])
    log(f"near-dedup: {rep['dropped']:,} of {rep['active']:,} active docs dropped, "
        f"{rep['seconds']} s, lock wait {lk.waited} s")
    return spec, rep


def boilerplate(cfg, src, shards, work, log):
    ub = sum(s["ubytes"] for s in shards)
    parts = 1
    while parts < 1024 and ub / 100 * 16 / parts > cfg["pair_mb"] * (1 << 20):
        parts *= 2
    bdir = os.path.join(work, "bp")
    os.makedirs(bdir, exist_ok=True)
    jobs = [{"src": os.path.join(cfg["stage1"], s["shard"]),
             "nd": stem(cfg["stage1"], s["shard"]) + ".nd.npy", "parts": parts,
             "out": os.path.join(bdir, os.path.basename(stem(cfg["stage1"], s["shard"])))}
            for s in shards]
    res = run_chunked(cfg, jobs, FW.bp_pairs, log, f"boilerplate pairs {src}")
    with HeavyLock(cfg["lock_file"], cfg["lock_timeout"]):
        bad = FW.count_bad([j["out"] for j in jobs], parts, cfg["bp_min_docs"])
    path = os.path.join(work, f"bp_{src}.npy")
    np.save(path, bad)
    shutil.rmtree(bdir)
    log(f"boilerplate {src}: {len(bad):,} lines from {sum(r['pairs'] for r in res):,} pairs")
    return path, {"lines": int(len(bad)), "pairs": sum(r["pairs"] for r in res),
                  "docs_counted": sum(r["docs"] for r in res), "partitions": parts}


def finalize(cfg, log=print):
    c = dict(DEFAULTS, **{k: v for k, v in cfg.items() if v is not None})
    for k in ("stage1", "final", "lock_file"):
        c[k] = os.path.abspath(os.path.expanduser(c[k]))
    cur = os.nice(0)
    if c["nice"] > cur:
        os.nice(c["nice"] - cur)
    events = R.ledger(c["stage1"])
    shards, k = sealed_shards(events), c["tier"]
    mine = [s for s in shards if s["tier"] == k]
    earlier = [s for s in shards if s["tier"] < k]
    if not mine:
        log(f"tier {k}: no sealed stage1 shards")
        return {"reason": "empty"}
    prev = {}
    for t in sorted({s["tier"] for s in earlier}):
        if not os.path.exists(tier_json(c["final"], t)):
            raise SystemExit(f"tier {t} has no {tier_json(c['final'], t)}: finalize it first")
        prev[t] = sha256_path(tier_json(c["final"], t))
    ih = hashlib.sha256(json.dumps({
        "shards": [(s["shard"], s["sha256"], s["sidecars"]["mh"]["sha256"]) for s in mine],
        "earlier": prev, "params": {p: c[p] for p in PARAMS}}, sort_keys=True).encode()).hexdigest()
    tj = tier_json(c["final"], k)
    if os.path.exists(tj) and json.load(open(tj)).get("input_hash") == ih:
        log(f"tier {k}: up to date ({tj})")
        return {"reason": "up_to_date"}
    t0, sources = time.time(), sorted({s["source"] for s in mine}, key=prio)
    for src in sources:
        for d in (os.path.join(c["final"], src), os.path.join(c["final"], "_masks", src)):
            shutil.rmtree(d, ignore_errors=True)
    work = os.path.join(c["final"], "_work", f"tier{k}")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    disp = DR.displaced_rows(c["stage1"], events, {s["shard"] for s in mine})
    spec, rep = near_dedup(c, mine, earlier, work, log, DR.include_masks(disp, mine, work))
    bad, bp = {}, {}
    for src in sources:
        if src in c["bp_sources"]:
            bad[src], bp[src] = boilerplate(c, src, [s for s in mine if s["source"] == src],
                                            work, log)
    codec = lambda rel: rel.rsplit(".jsonl.", 1)[1]                      # noqa: E731
    jobs = [{"src": os.path.join(c["stage1"], s["shard"]), "nd": stem(c["stage1"], s["shard"])
             + ".nd.npy", "bad": bad.get(s["source"]), "dst": os.path.join(c["final"], s["shard"]),
             "codec": codec(s["shard"]), "level": c["level"], "min_bytes": c["min_bytes"],
             "excl": disp[s["shard"]].tolist() if s["shard"] in disp else []} for s in mine]
    res = run_chunked(c, jobs, FW.final_write, log, "final shards")
    passd = R.pass_d(c, mine, jobs, res, prev, log)
    for s, j in zip(mine, jobs):
        info = np.load(j["dst"].rsplit(".jsonl.", 1)[0] + ".info.npy")
        m = np.zeros(s["docs"], dtype=bool)
        m[info["row"]] = True
        os.makedirs(os.path.dirname(mask_path(c["final"], s["shard"])), exist_ok=True)
        NL.ND.save_atomic(mask_path(c["final"], s["shard"]), m)
    rec = R.tier_record(c, k, ih, events, earlier, mine, spec, rep, res, bp, passd,
                        extract.code_hashes(), time.time() - t0)
    os.makedirs(os.path.dirname(tj), exist_ok=True)
    with open(tj + ".part", "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tj + ".part", tj)
    R.write_manifest(c["final"])
    shutil.rmtree(work, ignore_errors=True)
    if rec["complete"] is False:
        log(f"tier {k}: INCOMPLETE, files not done: " + json.dumps(
            {s: v["failed"] + v["missing"] for s, v in rec["coverage"].items() if v["failed"]
             or v["missing"]}))
    log(f"tier {k}: done in {time.time() - t0:.0f} s -> {tj}")
    return {"reason": "done", "record": tj}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--tier", type=int, required=True)
    for k in ("workers", "lock_timeout", "level", "bp_min_docs", "min_bytes"):
        ap.add_argument("--" + k.replace("_", "-"), type=int)
    for k in ("verify", "chain_floor", "mem_mb"):
        ap.add_argument("--" + k.replace("_", "-"), type=float)
    ap.add_argument("--lock-file")
    ap.add_argument("--bp-sources", nargs="*")
    a = vars(ap.parse_args(argv))
    sys.stdout.reconfigure(line_buffering=True)
    try:
        return {"stop": 5}.get(finalize(a)["reason"], 0)
    except LockTimeout:
        print("core_finalize: heavy lock timeout")
        return 4
    except Stop:
        print("core_finalize: STOP file")
        return 5


if __name__ == "__main__":
    sys.exit(main())
