"""compile_parity: loss curves with train.compile and/or train.ce_chunk_rows on vs the reference,
on CUDA (the PC), with a measured noise floor.

  python compile_parity.py --target 5e6 --micro 8 --accum 2 --steps 200 --out p5m.json \
      --arms eager:det,eager:det,eager,eager,eager,default:det,default:det
  python compile_parity.py ... --arms eager:det,eager:det,eager,eager,eager,eager+ce:det,default+ce:det

Each arm is one train.main run (the real trainer and loader, NorMuon, bf16 autocast, the
startup self-test) of budget.py's shape for --target, same seed, same synthetic data
(make_fake_data: counting documents and copy-task chats, vocab = the model's). An arm is
"<eager | a torch.compile mode>[+ce][:det]"; "+ce" adds train.ce_chunk_rows = --ce-chunk
(chunked_ce.py); ":det" runs it under torch.use_deterministic_algorithms(True).
CUBLAS_WORKSPACE_CONFIG=:4096:8 for every arm. The reference arms are exactly "eager" and
"eager:det"; every other arm is a candidate. train.py's own defaults apply otherwise (since
2026-09-26 on cuda: optim.batched and doc_attn varlen), so this measures the switches on top.
--data-mode pack (document-masked rows) or bucket (causal rows of varying length: the
compiled graph sees several shapes).

Per pair of arms: mean and max |loss difference| over the logged steps, and the difference
of the mean loss over the last 50 steps. Noise floor = the largest of those between the
plain eager arms (non-deterministic kernels, i.e. how far two identical eager runs drift
apart). Verdict: each compiled arm against each eager:det arm must stay within --factor
times the floor on the mean |difference| and on the |last-50 difference|.
Also reported (pooled): an exact permutation test of whether compiled runs sit further from
the eager runs than eager runs sit from each other; --pool a.json,b.json reruns it over the
runs of several invocations of the same setup.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import statistics as st
import sys
import tempfile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import budget  # noqa: E402
import train  # noqa: E402
from make_fake_data import make  # noqa: E402
from testutil import read_jsonl, write_run  # noqa: E402


REF = ("eager", "eager:det")


def run_arm(arm: str, i: int, a, model_cfg: dict, data_dir: str, work: str) -> dict:
    spec, _, det = arm.partition(":")
    mode, _, ce = spec.partition("+")
    assert ce in ("", "ce"), arm
    torch._dynamo.reset()
    torch.use_deterministic_algorithms(det == "det")
    T = model_cfg["seq_len"]
    data = {"window_tokens": 8 * a.micro * T}
    if a.data_mode == "bucket":
        data.update(mode="bucket", tokens_per_micro=a.micro * T, buckets=[256, 512, 1024, T])
    d = os.path.join(work, f"arm{i}")
    cfg = write_run(d, data_dir, model=model_cfg, data=data, train={
        "device": "cuda", "precision": "bf16", "micro_batch": a.micro, "grad_accum": a.accum,
        "total_steps": a.steps, "log_every": 1, "ckpt_every": 0, "keep_last": 1,
        "compile": False if mode == "eager" else mode, **({"ce_chunk_rows": a.ce_chunk} if ce else {})})
    t0 = time.time()
    assert train.main([cfg]) == 0
    log = read_jsonl(os.path.join(d, "out", "log.jsonl"))
    shutil.rmtree(os.path.join(d, "out"))
    torch.use_deterministic_algorithms(False)
    assert [r["step"] for r in log] == list(range(1, a.steps + 1))
    return {"arm": arm, "loss": [r["loss"] for r in log], "seconds": round(time.time() - t0, 1),
            "tok_per_s_median": st.median(r["tok_per_s"] for r in log[5:])}


def dist(x: list[float], y: list[float]) -> dict:
    d = [abs(p - q) for p, q in zip(x, y)]
    return {"mean_abs": st.mean(d), "max_abs": max(d),
            "tail50_diff": st.mean(x[-50:]) - st.mean(y[-50:]), "step1_diff": x[0] - y[0],
            "bitwise": x == y}


def pooled(runs: list[dict]) -> dict:
    """Exchangeability check per compile mode, over every distinct curve (a :det arm that is
    bitwise equal to an earlier one of the same arm is one sample, not two). Eager runs (det or
    not) are the reference pool. Statistic: mean distance over compiled-eager pairs minus mean
    over eager-eager pairs, for mean|d| and for |last-50 d|. p = share of the exact relabelings
    (same number of runs called compiled) whose statistic is >= the observed one; a small p
    means the compiled runs sit further from eager than eager runs sit from each other."""
    uniq = []
    for r in runs:
        if not any(u["arm"] == r["arm"] and u["loss"] == r["loss"] for u in uniq):
            uniq.append(r)
    eager = [r for r in uniq if r["arm"].split(":")[0] == "eager"]
    out = {}
    for mode in sorted({r["arm"].split(":")[0] for r in uniq} - {"eager"}):
        comp = [r for r in uniq if r["arm"].split(":")[0] == mode]
        pool, n, k = eager + comp, len(eager) + len(comp), len(comp)
        if len(eager) < 2 or n - k < 2:
            continue
        res = {"n_eager": len(eager), "n_compiled": k}
        for key, f in (("mean_abs", lambda d: d["mean_abs"]),
                       ("tail50", lambda d: abs(d["tail50_diff"]))):
            D = {(i, j): f(dist(pool[i]["loss"], pool[j]["loss"]))
                 for i, j in itertools.combinations(range(n), 2)}

            def split(lab):
                ce = [v for (i, j), v in D.items() if lab[i] != lab[j]]
                ee = [v for (i, j), v in D.items() if not lab[i] and not lab[j]]
                return ce, ee

            lab = [i >= len(eager) for i in range(n)]
            ce, ee = split(lab)
            obs = st.mean(ce) - st.mean(ee)
            perm = [st.mean(c) - st.mean(e) for c, e in (
                split([i in s for i in range(n)]) for s in itertools.combinations(range(n), k))]
            res[key] = {"eager_eager_median": st.median(ee), "eager_eager_max": max(ee),
                        "compiled_eager_median": st.median(ce), "compiled_eager_max": max(ce),
                        "stat": obs, "p": sum(s >= obs - 1e-12 for s in perm) / len(perm),
                        "n_perm": len(perm)}
        out[mode] = res
    return out


def print_pooled(pool: dict) -> None:
    for mode, r in pool.items():
        for key in ("mean_abs", "tail50"):
            v = r[key]
            print(f"[parity] pooled {mode} {key:<8} eager-eager median {v['eager_eager_median']:.5f} "
                  f"max {v['eager_eager_max']:.5f} | compiled-eager median "
                  f"{v['compiled_eager_median']:.5f} max {v['compiled_eager_max']:.5f} | "
                  f"permutation p {v['p']:.3f} (n_eager {r['n_eager']}, n_compiled "
                  f"{r['n_compiled']}, {v['n_perm']} relabelings)")


def pool_files(paths: list[str]) -> int:
    """--pool a.json,b.json: the pooled check over runs from several invocations of the same
    setup. The eager:det curves must be bitwise equal across files (same data, seed, shape)."""
    docs = [json.load(open(p)) for p in paths]
    keys = ("target", "model", "micro", "accum", "steps", "data_mode", "ce_chunk")
    assert all(all(d[k] == docs[0][k] for k in keys) for d in docs), "different setups"
    det = [r["loss"] for d in docs for r in d["runs"] if r["arm"] == "eager:det"]
    assert det and all(c == det[0] for c in det), "eager:det differs across files"
    print_pooled(pooled([r for d in docs for r in d["runs"]]))
    return 0


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == ["--pool"]:
        return pool_files(argv[1].split(","))
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=5e6)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--data-mode", default="pack", choices=["pack", "bucket"])
    ap.add_argument("--arms", default="eager:det,eager:det,eager,eager,eager,default:det,default:det")
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--ce-chunk", type=int, default=2048, help="rows per chunk for +ce arms")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    assert torch.cuda.is_available(), "compile_parity runs on CUDA"
    sol = budget.solve(a.target, budget.Constraints())
    model_cfg = sol.cfg.replace(seq_len=a.seq_len).to_dict()
    work = tempfile.mkdtemp(prefix="compile_parity_")
    make(os.path.join(work, "data"), vocab=model_cfg["vocab_size"], docs=20000, chats=15000)
    arms = a.arms.split(",")
    runs = []
    for i, arm in enumerate(arms):
        r = run_arm(arm, i, a, model_cfg, os.path.join(work, "data"), work)
        runs.append(r)
        print(f"[parity] {arm:<28} loss step1 {r['loss'][0]:.4f} step{a.steps} {r['loss'][-1]:.4f} "
              f"last50 {st.mean(r['loss'][-50:]):.4f}  {r['seconds']}s  "
              f"{r['tok_per_s_median']:,.0f} tok/s", flush=True)
    pairs = {}
    for (i, x), (j, y) in itertools.combinations(enumerate(runs), 2):
        pairs[f"{i}:{x['arm']} vs {j}:{y['arm']}"] = dist(x["loss"], y["loss"])
    eager = [k for k in pairs if k.split(" vs ")[0].split(":", 1)[1] == "eager"
             and k.split(" vs ")[1].split(":", 1)[1] == "eager"]
    floor = {"mean_abs": max(pairs[k]["mean_abs"] for k in eager),
             "tail50": max(abs(pairs[k]["tail50_diff"]) for k in eager)} if eager else None
    checks = {}
    for k, v in pairs.items():
        l, r = (s.split(":", 1)[1] for s in k.split(" vs "))
        comp = [s for s in (l, r) if s not in REF]
        if len(comp) == 1 and "eager:det" in (l, r) and floor:
            checks[k] = (v["mean_abs"] <= a.factor * floor["mean_abs"]
                         and abs(v["tail50_diff"]) <= a.factor * floor["tail50"])
    res = {"target": a.target, "shape": sol.row(), "model": model_cfg, "micro": a.micro,
           "accum": a.accum, "steps": a.steps, "data_mode": a.data_mode, "factor": a.factor,
           "ce_chunk": a.ce_chunk,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "runs": runs,
           "pairs": pairs, "noise_floor": floor, "checks": checks, "pooled": pooled(runs),
           "pass": bool(checks) and all(checks.values())}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for k, v in pairs.items():
        print(f"[parity] {k:<52} mean|d| {v['mean_abs']:.5f} max|d| {v['max_abs']:.4f} "
              f"last50 d {v['tail50_diff']:+.5f} step1 d {v['step1_diff']:+.5f}"
              f"{' BITWISE' if v['bitwise'] else ''}" + (f"  {'ok' if checks[k] else 'FAIL'}"
                                                        if k in checks else ""))
    print_pooled(res["pooled"])
    print(f"[parity] noise floor {floor}  factor {a.factor}  pass {res['pass']}")
    shutil.rmtree(work, ignore_errors=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
