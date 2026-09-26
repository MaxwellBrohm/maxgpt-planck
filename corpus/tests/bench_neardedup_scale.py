"""Near-dedup scale check for the PC (a script, not a pytest test). Each step prints one JSON line.

    python tests/bench_neardedup_scale.py gen WORK --docs 200000 --shards 40 --workers 16
    python tests/bench_neardedup_scale.py passb WORK --mem-mb 16 --tag small [--workers 8]
    python tests/bench_neardedup_scale.py check WORK --tags small,big
    python tests/bench_neardedup_scale.py sigs WORK --docs 5000000 --shards 50 --workers 16

gen: synthetic docs through pass A in parallel. Doc g is a base text (70%), an exact copy (10%),
a copy at Jaccard ~0.9 (10%) or ~0.8 (10%) of an earlier base; truth in WORK/truth.npy.
passb: one pass B run in this process: seconds, partitions, peak RSS (ru_maxrss) of this
process and of its band-hash workers, and a sha256 over every .nd.npy (identical across runs).
check: recall per planted kind, bases wrongly merged, and that every tag's results are identical.
sigs: pass-B-only scale: random signatures with planted exact (10%) and 10%-perturbed (10%) rows.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import neardedup as ND          # noqa: E402
import neardedup_lsh as NL      # noqa: E402
from nd_fixtures import vocab   # noqa: E402

V = vocab()
KINDS = ("base", "exact", "j90", "j80")


def kind_of(g):
    x = random.Random(g * 7919 + 13).random()
    return 0 if g < 100 or x < 0.7 else 1 if x < 0.8 else 2 if x < 0.9 else 3


def plan(g):
    """-> (kind, base index) of global doc g; copies point at an earlier base."""
    k = kind_of(g)
    if k == 0:
        return 0, g
    r = random.Random(g * 104729 + 1)
    b = r.randrange(g)
    while kind_of(b):
        b = r.randrange(b)
    return k, b


def base_words(b):
    r = random.Random(b * 1_000_003 + 7)
    return [r.choice(V) for _ in range(r.randint(100, 500))]


def text_of(g):
    kind, b = plan(g)
    ws = base_words(b)
    if kind >= 2:
        r = random.Random(g)
        s = len(ws) - 4
        t = 0.9 if kind == 2 else 0.8
        for i in r.sample(range(len(ws)), max(1, round(s * (1 - t) / (1 + t) / 5))):
            ws[i] = r.choice(V)
    return " ".join(ws)


def gen_shard(job):
    work, k, lo, hi = job
    w = ND.SidecarWriter(os.path.join(work, f"s{k:04d}"))
    nb, dt = 0, 0.0
    for g in range(lo, hi):
        t = text_of(g)
        nb += len(t)
        t0 = time.perf_counter()
        w.add(t, None)
        dt += time.perf_counter() - t0
    w.close()
    return nb, dt


def sig_shard(job):
    work, k, lo, hi = job
    rng = np.random.default_rng(k)
    base = rng.integers(0, 2**32, (hi - lo, ND.NPERM), dtype=np.uint32, endpoint=False)
    kind = np.zeros(hi - lo, dtype=np.int8)
    x = rng.random(hi - lo)
    for i in range(1, hi - lo):
        if x[i] < 0.2:
            j = rng.integers(0, i)
            base[i] = base[j]
            kind[i] = 1
            if x[i] < 0.1:
                pos = rng.choice(ND.NPERM, 12, replace=False)
                base[i, pos] = rng.integers(0, 2**32, 12, dtype=np.uint32, endpoint=False)
                kind[i] = 2
    arr = np.zeros(hi - lo, dtype=ND.SIG_DTYPE)
    arr["sig"], arr["day"] = base, ND.UNDATED
    ND.save_atomic(os.path.join(work, f"s{k:04d}") + ND.MH_SUFFIX, arr)
    np.save(os.path.join(work, f"kind{k:04d}.npy"), kind)
    return 0, 0.0


def spec(work, shards):
    return [{"stem": os.path.join(work, f"s{k:04d}"), "tier": 0, "label": "syn"}
            for k in range(shards)]


def run_pool(fn, work, docs, shards, workers):
    per = -(-docs // shards)
    jobs = [(work, k, k * per, min(docs, (k + 1) * per)) for k in range(shards)]
    with mp.get_context("spawn").Pool(workers) as pool:
        return list(pool.imap(fn, jobs))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("gen", "passb", "check", "sigs"))
    ap.add_argument("work")
    ap.add_argument("--docs", type=int, default=200_000)
    ap.add_argument("--shards", type=int, default=40)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mem-mb", type=float, default=1024)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--tags", default="")
    a = ap.parse_args(argv)
    os.makedirs(a.work, exist_ok=True)
    t0 = time.time()
    if a.step in ("gen", "sigs"):
        with open(os.path.join(a.work, "meta.json"), "w") as f:
            json.dump({"docs": a.docs, "shards": a.shards, "step": a.step}, f)
        nb = run_pool(gen_shard if a.step == "gen" else sig_shard, a.work, a.docs, a.shards,
                      a.workers)
        if a.step == "gen":
            np.save(os.path.join(a.work, "truth.npy"), np.array([plan(g) for g in range(a.docs)]))
        mb, sk = sum(x[0] for x in nb) / 1e6, sum(x[1] for x in nb)
        out = {"step": a.step, "docs": a.docs, "seconds": round(time.time() - t0, 1),
               "workers": a.workers, "text_mb": round(mb, 1),
               "sketch_mb_per_s_per_core": round(mb / sk, 2) if sk else None}
    elif a.step == "passb":
        meta = json.load(open(os.path.join(a.work, "meta.json")))
        sh = spec(a.work, meta["shards"])
        wd = os.path.join(a.work, f"passb_{a.tag}")
        os.makedirs(wd, exist_ok=True)
        rep = NL.find_near_dups(sh, wd, mem_mb=a.mem_mb, workers=a.workers)
        h = hashlib.sha256()
        for s in sh:
            h.update(NL.load_result(s["stem"]).tobytes())
        unit = 1 << 20 if sys.platform == "darwin" else 1 << 10      # ru_maxrss: bytes vs KB
        mb = lambda who: round(resource.getrusage(who).ru_maxrss * unit / 2**20, 1)
        out = {"step": "passb", "tag": a.tag, "docs": rep["docs"], "mem_mb": a.mem_mb,
               "partitions": rep["partitions"], "seconds": rep["seconds"],
               "peak_rss_mb": mb(resource.RUSAGE_SELF), "workers_peak_rss_mb":
               mb(resource.RUSAGE_CHILDREN), "dropped": rep["dropped"], "clusters":
               rep["clusters"], "edges": rep["edges"], "nd_sha256": h.hexdigest()}
        with open(os.path.join(a.work, f"passb_{a.tag}.json"), "w") as f:
            json.dump(out, f)
    else:
        out = check(a.work, [t for t in a.tags.split(",") if t])
    print(json.dumps(out), flush=True)


def check(work, tags):
    meta = json.load(open(os.path.join(work, "meta.json")))
    sh = spec(work, meta["shards"])
    res = np.concatenate([NL.load_result(s["stem"]) for s in sh])
    per = -(-meta["docs"] // meta["shards"])
    lead = res["lshard"].astype(np.int64) * per + res["lrow"].astype(np.int64)
    runs = [json.load(open(os.path.join(work, f"passb_{t}.json"))) for t in tags]
    out = {"step": "check", "identical_across_tags": len({r["nd_sha256"] for r in runs}) == 1}
    if meta["step"] == "gen":
        truth = np.load(os.path.join(work, "truth.npy"))
        kind, base = truth[:, 0], truth[:, 1]
        for k in range(1, 4):
            m = kind == k
            out[f"recall_{KINDS[k]}"] = round(float((lead[m] == lead[base[m]]).mean()), 4)
        bases = np.flatnonzero(kind == 0)
        out["bases_dropped"] = int((res["keep"][bases] == 0).sum())
        out["copies_kept"] = int((res["keep"][kind > 0] == 1).sum())
    else:
        kind = np.concatenate([np.load(os.path.join(work, f"kind{k:04d}.npy"))
                               for k in range(meta["shards"])])
        for k, name in ((1, "exact"), (2, "perturbed")):
            out[f"dropped_{name}"] = round(float((res["keep"][kind == k] == 0).mean()), 4)
        out["random_rows_dropped"] = int((res["keep"][kind == 0] == 0).sum())
    return out


if __name__ == "__main__":
    sys.exit(main())
