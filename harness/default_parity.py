"""default_parity: the same default config trained by two harness trees must give the same bits.

  python default_parity.py --ref /path/to/main/harness --new /path/to/this/harness [--device cuda]

For proving that a change leaves the default path alone (SPEED V3: train.compile and
train.ce_chunk_rows are opt-in, so a config without them must train exactly as before). Both trees
run one config (no speed keys; train.py's own defaults apply, so on cuda that is optim.batched and
doc_attn varlen) on the same synthetic data, each in a fresh process with deterministic algorithms
(CUBLAS_WORKSPACE_CONFIG :4096:8). Compared with no tolerance: every log record (minus wall-clock
fields), the final checkpoint (model, optimizer, data_state, rng, step, tokens), the start record's
keys, and the modules each process imported (the new tree may not import any the old one did not).
Exit 0 only if all of it is identical.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import runio  # noqa: E402
from make_fake_data import make  # noqa: E402
from testutil import read_jsonl, write_run  # noqa: E402

CUDA_MODEL = {"vocab_size": 256, "d_model": 64, "n_layers": 2, "n_heads": 2, "n_kv_heads": 1,
              "head_dim": 32, "mlp_hidden": 128, "seq_len": 256}
RUN = ("import json, sys, torch; torch.use_deterministic_algorithms(True); before = set(sys.modules); "
       "import train; rc = train.main(sys.argv[2:]); "
       "json.dump(sorted(set(sys.modules) - before), open(sys.argv[1], 'w')); sys.exit(rc)")
VOLATILE = ("tok_per_s", "time")
START_SKIP = ("time", "config", "out_dir", "env", "run", "resumed_from")


def same(a, b, path="") -> list[str]:
    """Paths where a and b differ (tensors bitwise, containers recursively)."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        ok = isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype \
            and a.shape == b.shape and torch.equal(a, b)
        return [] if ok else [path]
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f"{path} keys {sorted(set(a) ^ set(b))}"]
        return [d for k in a for d in same(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path} len {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f"{path}[{i}]")]
    return [] if a == b else [path]


def run_tree(harness: str, cfg: str, mods: str) -> None:
    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONDONTWRITEBYTECODE": "1"}
    p = subprocess.run([sys.executable, "-c", RUN, mods, cfg], cwd=harness, env=env,
                       capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        raise SystemExit(f"{harness} failed:\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="harness dir of the reference tree")
    ap.add_argument("--new", default=HERE, help="harness dir of the tree under test")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--work", default=None)
    a = ap.parse_args(argv)
    work = a.work or tempfile.mkdtemp(prefix="default_parity_")
    make(os.path.join(work, "data"), docs=300, chats=300)
    train_kw = {"total_steps": a.steps, "log_every": 1, "ckpt_every": 0, "grad_accum": 2}
    over = {}
    if a.device == "cuda":
        train_kw.update(device="cuda", precision="bf16", micro_batch=8)
        over["model"] = CUDA_MODEL
    out = {}
    for tag, tree in (("ref", a.ref), ("new", a.new)):
        d = os.path.join(work, tag, "run")       # same run name: the two configs hash the same
        cfg = write_run(d, os.path.join(work, "data"), runs_jsonl="runs.jsonl", train=train_kw, **over)
        run_tree(os.path.abspath(tree), cfg, os.path.join(d, "modules.json"))
        ck = runio.load_checkpoint(runio.latest_checkpoint(os.path.join(d, "out")))
        start = [r for r in read_jsonl(os.path.join(d, "runs.jsonl")) if r["event"] == "start"][-1]
        out[tag] = {"log": [{k: v for k, v in r.items() if k not in VOLATILE}
                            for r in read_jsonl(os.path.join(d, "out", "log.jsonl"))],
                    "ck": {k: ck[k] for k in ("model", "optimizer", "data_state", "rng_torch", "step",
                                              "tokens", "sup_tokens")},
                    "start": {k: v for k, v in start.items() if k not in START_SKIP},
                    "mods": set(json.load(open(os.path.join(d, "modules.json"))))}
        if a.device == "cuda":
            out[tag]["ck"]["rng_cuda"] = ck["rng_cuda"]
    r, n = out["ref"], out["new"]
    diffs = {"log": same(r["log"], n["log"]), "checkpoint": same(r["ck"], n["ck"]),
             "start": same(r["start"], n["start"]), "new_imports": sorted(n["mods"] - r["mods"])}
    print(f"[default_parity] {a.device}: {len(r['log'])} log records, final loss {r['log'][-1]['loss']} "
          f"vs {n['log'][-1]['loss']}; start {n['start'].get('doc_attn', 'mask')}"
          f"{' batched' if n['start'].get('optim_batched') else ''}; work {work}")
    for k, v in diffs.items():
        print(f"[default_parity] {k}: {'identical' if not v else v[:8]}")
    print(f"[default_parity] modules only the reference imported: {sorted(r['mods'] - n['mods'])[:8]}")
    ok = not any(diffs.values()) and len(r["log"]) == a.steps
    print(f"[default_parity] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
