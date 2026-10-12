"""neardedup_post.py: a second MinHash-LSH near-dedup pass over the FINAL core v0 shards.

    python corpus/neardedup_post.py --final ~/planck/data/core_v0 --out ~/planck/data/core_v0_pd \\
        [--workers 16] [--nice 15] [--verify 0.7] [--chain-floor 0] [--mem-mb 1024] [--level 10]

Why (CORPUS.md, 2026-09-26 review): pass B ran on stage1 text, before boilerplate stripping, so
same-site pages that differ only in stripped template lines or counters survived (6.8% of a same-URL
CCCC sample have a partner at Jaccard >= 0.8, most on bizsugar.com and destructoid.com). This pass
repeats pass B on the final (stripped) text.

Method: the shingles, signatures and LSH of neardedup.py and neardedup_lsh.py unchanged (word
5-grams, 126 minima, 14 x 9 bands, verify 0.7; chain floor 0 as in the core v0 run). Keep rule =
the core v0 priority (core_finalize): lower tier, then the source's place in stream_rank.PRIORITY,
then earlier date (meta.date, else meta.created; undated last), then shard order, then row.
Stages, each restartable (the record of a unit is its commit point; a rerun skips what matches):
1. sig: per shard, check its sha256 against FINAL/MANIFEST.json, then write OUT/_pd/sig/<stem>
   .mh.npy (the neardedup sidecar), .tb.npy (UTF-8 bytes of each text) and .sig.json.
2. lsh: neardedup_lsh.find_near_dups over every shard (none frozen) -> <sig stem>.nd.npy (lead of
   every dropped doc), OUT/_pd/neardedup_report.json and OUT/_pd/lsh.json.
3. write: OUT/<shard> without the dropped lines (every kept line byte-identical to v0, same order),
   .keys.u64 and .info.npy filtered the same way; a shard with no drop is copied. The record is
   OUT/_pd/shards/<stem>.json.
4. OUT/MANIFEST.json: shards with sha256, per-source docs and text bytes in, dropped and kept.
FINAL is only read. Helpers: neardedup_post_io.py. Token estimates: neardedup_post_tok.py.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

import neardedup as ND
import neardedup_lsh as NL
import stream_io as SIO
from neardedup_post_io import base, code_sha, dump_atomic, load_json, manifest, pool_map
from stream_lock import mem_cap
from stream_rank import prio

DEFAULTS = dict(workers=16, nice=15, verify=ND.VERIFY, chain_floor=0.0, mem_mb=1024, level=10,
                mem_per_worker_gb=0.5, mem_reserve_gb=3)
PARAMS = ("verify", "chain_floor", "level")
CODE = ("neardedup.py", "neardedup_keep.py", "neardedup_lsh.py", "neardedup_post.py",
        "neardedup_post_io.py")


def plan(final):
    """-> (FINAL/MANIFEST.json, its shards with their tier, in keep-rule order)."""
    with open(os.path.join(final, "MANIFEST.json")) as f:
        man = json.load(f)
    tier = {src: int(t) for t, v in man["tiers"].items() for src in v["sources"]}
    shards = [dict(s, tier=tier[s["source"]]) for s in man["shards"]]
    return man, sorted(shards, key=lambda s: (s["tier"], prio(s["source"]), s["shard"]))


def check_input(job):
    if SIO.sha256_file(job["src"]) != job["sha256"]:
        raise RuntimeError(f"{job['src']}: sha256 differs from FINAL/MANIFEST.json")


def sig_job(job):
    stem = job["stem"]
    rec = load_json(stem + ".sig.json")
    if rec and rec["v0_sha256"] == job["sha256"] and rec["code"] == job["code"] and \
            os.path.exists(stem + ND.MH_SUFFIX) and os.path.exists(stem + ".tb.npy"):
        return rec
    check_input(job)
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    w, tb = ND.SidecarWriter(stem), []
    with SIO.open_shard(job["src"]) as f:
        for raw in f:
            d = json.loads(raw)
            m = d.get("meta") or {}
            w.add(d["text"], m.get("date") or m.get("created"))
            tb.append(len(d["text"].encode("utf-8")))
    if len(tb) != job["docs"]:
        raise RuntimeError(f"{job['src']}: {len(tb)} lines, manifest says {job['docs']}")
    ND.save_atomic(stem + ".tb.npy", np.asarray(tb, dtype=np.uint64))
    w.close()
    rec = {"v0_sha256": job["sha256"], "code": job["code"], "docs": len(tb),
           "text_bytes": int(sum(tb))}
    dump_atomic(stem + ".sig.json", rec)
    return rec


def write_job(job):
    dst = job["dst"]
    rec = load_json(job["rec"])
    if rec and rec["input_hash"] == job["ih"] and os.path.exists(dst) and \
            os.path.getsize(dst) == rec["bytes"]:
        return rec
    nd, tb = np.load(job["stem"] + ND.ND_SUFFIX), np.load(job["stem"] + ".tb.npy")
    keep = nd["keep"].astype(bool)
    s0, d0 = base(job["src"]), base(dst)
    keys, info = SIO.read_keys(s0 + ".keys.u64"), np.load(s0 + ".info.npy")
    if not len(nd) == len(tb) == len(keys) == len(info) == job["docs"]:
        raise RuntimeError(f"{job['src']}: row counts differ (nd, tb, keys, info, manifest)")
    check_input(job)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if keep.all():
        shutil.copyfile(job["src"], dst + ".part")
        sha, ub = SIO.sha256_file(dst + ".part"), job["ubytes"]
        if sha != job["sha256"]:
            raise RuntimeError(f"{dst}: copy differs from {job['src']}")
    else:
        fr, h, ub, n = SIO.Frame(job["codec"], job["level"]), hashlib.sha256(), 0, 0
        with SIO.open_shard(job["src"]) as f, open(dst + ".part", "wb") as fo:
            for i, line in enumerate(f):
                n = i + 1
                if keep[i]:
                    b = fr.compress(line)
                    fo.write(b)
                    h.update(b)
                    ub += len(line)
            b = fr.end()
            fo.write(b)
            h.update(b)
            fo.flush()
            os.fsync(fo.fileno())
        if n != len(keep):
            raise RuntimeError(f"{job['src']}: {n} lines, {len(keep)} near-dedup rows")
        sha = h.hexdigest()
    keys[keep].astype("<u8").tofile(d0 + ".keys.u64.part")
    np.save(d0 + ".info.part.npy", info[keep])
    os.replace(d0 + ".keys.u64.part", d0 + ".keys.u64")
    os.replace(d0 + ".info.part.npy", d0 + ".info.npy")
    os.replace(dst + ".part", dst)
    rec = {"source": job["source"], "shard": job["shard"], "docs": int(keep.sum()),
           "ubytes": int(ub), "bytes": os.path.getsize(dst), "sha256": sha,
           "keys": base(job["shard"]) + ".keys.u64",
           "keys_sha256": SIO.sha256_file(d0 + ".keys.u64"), "v0_docs": int(len(keep)),
           "v0_sha256": job["sha256"], "text_bytes_in": int(tb.sum()),
           "text_bytes": int(tb[keep].sum()), "dropped_docs": int((~keep).sum()),
           "dropped_bytes": int(tb[~keep].sum()), "input_hash": job["ih"]}
    dump_atomic(job["rec"], rec)
    return rec


def lsh(c, shards, stems, pd, ih, log):
    done = load_json(os.path.join(pd, "lsh.json"))
    if done and done["input_hash"] == ih and all(os.path.exists(s + ND.ND_SUFFIX) for s in stems):
        log("lsh: up to date")
        return load_json(os.path.join(pd, "neardedup_report.json"))
    spec = [{"stem": st, "tier": s["tier"] * 16 + prio(s["source"]), "label": s["source"],
             "frozen": False, "include": None} for s, st in zip(shards, stems)]
    rep = NL.find_near_dups(spec, pd, verify=c["verify"], mem_mb=c["mem_mb"],
                            workers=max(1, min(c["workers"], mem_cap(c))),
                            chain_floor=c["chain_floor"])
    dump_atomic(os.path.join(pd, "lsh.json"), {"input_hash": ih, "dropped": rep["dropped"],
                                               "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log(f"lsh: {rep['dropped']:,} of {rep['active']:,} docs dropped, {rep['seconds']} s")
    return rep


def run(cfg, log=print):
    c = dict(DEFAULTS, **{k: v for k, v in cfg.items() if v is not None})
    final = os.path.realpath(os.path.expanduser(c["final"]))
    out = os.path.realpath(os.path.expanduser(c["out"]))
    if out == final or out.startswith(final + os.sep) or final.startswith(out + os.sep):
        raise SystemExit("--out must be a folder outside --final (FINAL is never modified)")
    cur = os.nice(0)
    if c["nice"] > cur:
        os.nice(c["nice"] - cur)
    man, shards = plan(final)
    pd, msha = os.path.join(out, "_pd"), SIO.sha256_file(os.path.join(final, "MANIFEST.json"))
    ih = hashlib.sha256(json.dumps({"manifest": msha, "params": {p: c[p] for p in PARAMS},
                                    "code": code_sha(CODE)}, sort_keys=True).encode()).hexdigest()
    sigcode = code_sha(["neardedup.py"])
    stems = [os.path.join(pd, "sig", base(s["shard"])) for s in shards]
    jobs = [dict(s, src=os.path.join(final, s["shard"]), stem=st, code=sigcode, ih=ih,
                 dst=os.path.join(out, s["shard"]), codec=s["shard"].rsplit(".", 1)[1],
                 level=c["level"], rec=os.path.join(pd, "shards", base(s["shard"]) + ".json"))
            for s, st in zip(shards, stems)]
    pool_map(c, sig_job, jobs, log, "sig")
    rep = lsh(c, shards, stems, pd, ih, log)
    recs = pool_map(c, write_job, jobs, log, "write")
    m = manifest({p: c[p] for p in PARAMS}, CODE, man, msha, ih, shards, recs, rep, pd)
    dump_atomic(os.path.join(out, "MANIFEST.json"), m)
    log(f"done: {m['totals']['dropped_docs']:,} docs ({m['totals']['dropped_bytes']:,} text "
        f"bytes) dropped, {m['totals']['docs']:,} docs left")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--final", required=True)
    ap.add_argument("--out", required=True)
    for k in ("workers", "nice", "level"):
        ap.add_argument("--" + k, type=int)
    for k in ("verify", "chain_floor", "mem_mb"):
        ap.add_argument("--" + k.replace("_", "-"), type=float)
    a = vars(ap.parse_args(argv))
    if a["workers"] is not None and not 1 <= a["workers"] <= 20:
        raise SystemExit("--workers must be 1-20")
    sys.stdout.reconfigure(line_buffering=True)
    run(a, lambda s: print(time.strftime("%H:%M:%S"), s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
