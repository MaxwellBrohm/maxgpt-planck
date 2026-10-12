"""Per-source token estimates for the post-dedup core (OUT/MANIFEST.json of neardedup_post.py).

    python corpus/neardedup_post_tok.py --out ~/planck/data/core_v0_pd \\
        --tokenizer ~/planck/tok/v0c/tok_v0_8k.json [--workers 12]

Method of corpus/stats/core_v0/final_stats.json: per source a uniform sample of documents (2,000;
400 for loc and gutenberg; all of a smaller source), numpy seed 20260926, each text encoded with
Tokenizer.encode(text, add_special_tokens=False) (no EOS); bytes per token = sample text bytes /
sample tokens; tokens (est.) = the source's exact text bytes / bytes per token; 95% interval from 300
bootstrap resamples of the sample documents. Docs and bytes are exact counts, tokens are estimates.
Writes OUT/TOKENS.json (restartable: one part file per shard in OUT/_pd/tok).
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import stream_io as SIO
from neardedup_post_io import dump_atomic, load_json
from stream_lock import exit_with_parent

SEED, BOOT = 20260926, 300
SAMPLE = {"loc": 400, "gutenberg": 400}
DEFAULT_SAMPLE = 2000
_ENC = {}


def sample_plan(shards, seed=SEED):
    """-> {shard index: sorted rows} (uniform per source over all of its documents)."""
    rng, sel, by = np.random.default_rng(seed), {}, {}
    for i, s in enumerate(shards):
        by.setdefault(s["source"], []).append(i)
    for src in sorted(by):
        idx = by[src]
        off = np.concatenate(([0], np.cumsum([shards[i]["docs"] for i in idx])))
        n = int(off[-1])
        g = np.sort(rng.choice(n, min(SAMPLE.get(src, DEFAULT_SAMPLE), n), replace=False))
        a = np.searchsorted(off, g, side="right") - 1
        for x, y in zip(a, g):
            sel.setdefault(idx[x], []).append(int(y - off[x]))
    return sel


def hf_encoder(path):
    if path not in _ENC:
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(path)
        _ENC[path] = lambda s: len(t.encode(s, add_special_tokens=False).ids)
    return _ENC[path]


def tok_job(job, count=None):
    """job {path, sha256, rows, part, tokenizer} -> [(text bytes, tokens)] of those rows, in order."""
    got = load_json(job["part"])
    if got is not None and got["rows"] == job["rows"] and got["sha256"] == job["sha256"]:
        return got["pairs"]
    count = count or hf_encoder(job["tokenizer"])
    want, pairs, last = set(job["rows"]), [], max(job["rows"])
    with SIO.open_shard(job["path"]) as f:
        for i, line in enumerate(f):
            if i in want:
                t = json.loads(line)["text"]
                pairs.append((len(t.encode("utf-8")), count(t)))
            if i >= last:
                break
    if len(pairs) != len(want):
        raise RuntimeError(f"{job['path']}: found {len(pairs)} of {len(want)} sample rows")
    dump_atomic(job["part"], {"rows": job["rows"], "sha256": job["sha256"], "pairs": pairs})
    return pairs


def estimate(pairs, text_bytes, seed=SEED, boot=BOOT):
    b, t = (np.array([p[i] for p in pairs], dtype=np.float64) for i in (0, 1))
    bpt = b.sum() / t.sum()
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(b), (boot, len(b)))
    bb = np.sort(b[ix].sum(axis=1) / t[ix].sum(axis=1))
    lo, hi = bb[int(0.025 * boot)], bb[min(boot - 1, int(0.975 * boot))]
    return {"sample_docs": len(b), "bytes_per_token_8k": round(float(bpt), 4),
            "bytes_per_token_8k_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "tokens_est_8k": int(round(text_bytes / bpt)),
            "tokens_est_8k_ci95": [int(round(text_bytes / hi)), int(round(text_bytes / lo))]}


def run(out, tokenizer=None, workers=1, count=None, log=print):
    out = os.path.realpath(os.path.expanduser(out))
    man = load_json(os.path.join(out, "MANIFEST.json"))
    shards, sel = man["shards"], sample_plan(man["shards"])
    jobs = [{"path": os.path.join(out, shards[i]["shard"]), "sha256": shards[i]["sha256"],
             "rows": sel[i],
             "part": os.path.join(out, "_pd", "tok", f"{i:05d}.json"), "tokenizer": tokenizer}
            for i in sorted(sel)]
    t0 = time.time()
    if workers <= 1 or count is not None:
        res = [tok_job(j, count) for j in jobs]
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                 initializer=exit_with_parent, initargs=(os.getpid(),)) as ex:
            res = list(ex.map(tok_job, jobs))
    per = {}
    for i, r in zip(sorted(sel), res):
        per.setdefault(shards[i]["source"], []).extend(r)
    src_out, tot = {}, {"docs": 0, "text_bytes": 0, "tokens_est_8k": 0, "lo": 0, "hi": 0}
    for src, b in man["sources"].items():
        e = estimate(per[src], b["text_bytes"])
        dropped_tok = int(round(b["dropped_bytes"] / e["bytes_per_token_8k"]))
        src_out[src] = dict(tier=b["tier"], docs=b["docs"], text_bytes=b["text_bytes"],
                            dropped_docs=b["dropped_docs"], dropped_bytes=b["dropped_bytes"],
                            dropped_tokens_est_8k=dropped_tok, **e)
        tot["docs"] += b["docs"]
        tot["text_bytes"] += b["text_bytes"]
        tot["tokens_est_8k"] += e["tokens_est_8k"]
        tot["lo"] += e["tokens_est_8k_ci95"][0]
        tot["hi"] += e["tokens_est_8k_ci95"][1]
    res = {"what": "Token estimates of core v0 after neardedup_post.py (docs and bytes exact, "
                   "tokens est.)",
           "manifest_sha256": SIO.sha256_file(os.path.join(out, "MANIFEST.json")),
           "tokenizer": {"file": os.path.basename(tokenizer), "sha256": SIO.sha256_file(tokenizer)}
           if tokenizer else None, "seed": SEED, "bootstrap": BOOT,
           "totals": {"docs": tot["docs"], "text_bytes": tot["text_bytes"],
                      "tokens_est_8k": tot["tokens_est_8k"],
                      "tokens_est_8k_sum_of_ci95": [tot["lo"], tot["hi"]]},
           "sources": src_out}
    dump_atomic(os.path.join(out, "TOKENS.json"), res)
    log(f"tokens: {tot['tokens_est_8k']:,} est. over {len(jobs)} shards, "
        f"{time.time() - t0:.0f} s")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)
    if not 1 <= a.workers <= 20:
        raise SystemExit("--workers must be 1-20")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.stdout.reconfigure(line_buffering=True)
    tok = os.path.realpath(os.path.expanduser(a.tokenizer))
    print(hashlib.sha256(open(tok, "rb").read()).hexdigest(), tok)
    run(a.out, tok, a.workers, log=lambda s: print(time.strftime("%H:%M:%S"), s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
