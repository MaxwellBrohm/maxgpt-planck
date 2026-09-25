"""bench_micro: training tokens per second for the Planck curve shapes, for the RTX 5070.

  python pc/bench_micro.py                                   # 5M 10M 20M 30M on CUDA
  python pc/bench_micro.py --targets 3e6,5e6 --batches 8,16,32,64 --modes causal,docmask
  python pc/bench_micro.py --compile                         # also time torch.compile
  python pc/bench_micro.py --loops 2                         # a looped arm (any budget.py flag)
  python pc/bench_micro.py --device cpu --targets 3e5 --vocab 512 --seq-len 64 \
      --batches 2 --steps 2 --warmup 1                        # CPU smoke test (the Mac tests)

Each shape is the one harness/budget.py solve() picks for the target (the same Constraints
flags as budget.py; defaults vocab 8192, aspect 24, SwiGLU 8/3), built by harness/model.py
with the harness optimizer (optim.make_optimizer: NorMuon on matrices, AdamW on embedding
and scalars) and bf16 autocast (device.amp_factory), exactly as train.py builds them.
One timed step = forward (reduction "sum" / tokens, as trainer.py), backward, grad clip 1.0,
optimizer step, zero_grad; --accum micro-batches per optimizer step. Tokens are random.

Modes: causal  doc=None, SDPA is_causal (the flash path; bucket mode's rows)
       docmask packed rows of about --docs-per-row documents: SDPA with the harness's bool
               document mask (pack mode's path; no flash kernel, expected slower)
A batch that runs out of memory is recorded as "oom" and larger batches of that shape and
mode are skipped. Every measurement appends one JSON line to --out, with torch, CUDA, GPU
and driver versions. Nothing here downloads anything.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "harness"))
sys.path.insert(0, os.path.join(HERE, "wsl"))

import torch  # noqa: E402

import budget  # noqa: E402
from device import amp_factory, env_info, resolve_precision, sync  # noqa: E402
from model import build_model  # noqa: E402
from optim import make_optimizer  # noqa: E402


def random_docs(B: int, T: int, per_row: float, g: torch.Generator) -> torch.Tensor:
    """(B, T) document ids: about per_row documents per row, random lengths, ids restart."""
    starts = torch.rand(B, T, generator=g) < (per_row / T)
    starts[:, 0] = True
    return torch.cumsum(starts.long(), dim=1) - 1


def make_batch(B, T, vocab, mode, per_row, g, device):
    idx = torch.randint(0, vocab, (B, T), generator=g)
    tgt = torch.randint(0, vocab, (B, T), generator=g)
    doc = random_docs(B, T, per_row, g) if mode == "docmask" else None
    mv = lambda t: None if t is None else t.to(device)  # noqa: E731
    return mv(idx), mv(tgt), mv(doc)


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def bench_one(cfg, mode: str, B: int, a, device: str, precision: str) -> dict:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    model = build_model(cfg, "cpu").to(device)
    opt = make_optimizer(model, {"kind": a.optim}, device_type=device)
    fwd = torch.compile(model) if a.compile else model
    amp = amp_factory(precision, device)
    batches = [make_batch(B, cfg.seq_len, cfg.vocab_size, mode, a.docs_per_row, g, device)
               for _ in range(min(a.accum, 4))]
    # parameters() yields a tied or shared tensor once, so this is the TOTAL the curve counts
    rec = {"status": "ok", "n_params_built": sum(p.numel() for p in model.parameters())}

    def step(i: int) -> float:
        loss_sum = 0.0
        for j in range(a.accum):
            idx, tgt, doc = batches[(i * a.accum + j) % len(batches)]
            with (amp() if amp else nullcontext()):
                _, ls = fwd(idx, tgt, doc, reduction="sum")
            (ls / idx.numel() / a.accum).backward()
            loss_sum += float(ls.detach()) / idx.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss_sum / a.accum

    try:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for i in range(a.warmup):
            first = step(i)
        sync(device)
        rec["warmup_s"] = round(time.time() - t0, 2)
        t0 = time.time()
        for i in range(a.steps):
            last = step(a.warmup + i)
        sync(device)
        dt = time.time() - t0
        tokens = B * cfg.seq_len * a.accum * a.steps
        rec.update(tok_per_s=round(tokens / dt, 1), step_s=round(dt / a.steps, 4),
                   h_per_1B=round(1e9 / (tokens / dt) / 3600, 2),
                   loss_first=round(first, 4) if a.warmup else None, loss_last=round(last, 4))
        if not math.isfinite(last):
            rec["status"] = "nonfinite"
        if device == "cuda":
            rec["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except (RuntimeError, torch.OutOfMemoryError) as e:
        rec = {"status": "oom" if is_oom(e) else "error", "error": str(e).splitlines()[0][:300],
               "n_params_built": rec["n_params_built"]}
    finally:
        del model, opt, fwd, batches
        if device == "cuda":
            torch.cuda.empty_cache()
    return rec


def gpu_state() -> dict | None:
    try:
        import gpuguard
        return gpuguard.query(with_throttle=True)
    except Exception:  # noqa: BLE001  (a missing nvidia-smi must not stop the bench)
        return None


def driver_version() -> str | None:
    try:
        import gpuguard
        return gpuguard._smi("driver_version", 10.0)
    except Exception:  # noqa: BLE001
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Planck training throughput micro-benchmark")
    ap.add_argument("--targets", default="5e6,10e6,20e6,30e6")
    ap.add_argument("--modes", default="causal,docmask")
    ap.add_argument("--batches", default="8,16,32", help="micro-batch rows")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--docs-per-row", type=float, default=8.0)
    ap.add_argument("--optim", default="normuon", choices=["normuon", "muon", "adamw"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--out", default=os.path.join(HERE, "bench_results.jsonl"))
    budget.add_constraint_args(ap)
    a = ap.parse_args(argv)
    if a.steps < 1 or a.accum < 1 or a.warmup < 0:
        raise SystemExit("--steps and --accum must be >= 1, --warmup >= 0")
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available (this benchmark is meant for the PC)")
    precision = resolve_precision(a.precision, a.device)
    env = env_info(a.device)
    if a.device == "cuda":
        env["capability"] = list(torch.cuda.get_device_capability(0))
        env["driver"] = driver_version()
    k = budget.constraints_from_args(a)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    print(f"bench_micro {run_id}: {env.get('gpu', a.device)} torch {env['torch']} "
          f"precision {precision} T={a.seq_len} accum={a.accum} compile={a.compile}")
    print(f"{'target':>7} {'d':>4} {'L':>3} {'params':>9} {'mode':>8} {'B':>4} {'tok/s':>10} "
          f"{'step s':>8} {'h/1B':>7} {'peakGiB':>7} status")
    for target in [float(t) for t in a.targets.split(",")]:
        sol = budget.solve(target, k)
        cfg = sol.cfg.replace(seq_len=a.seq_len)
        for mode in a.modes.split(","):
            assert mode in ("causal", "docmask"), mode
            for B in [int(b) for b in a.batches.split(",")]:
                r = bench_one(cfg, mode, B, a, a.device, precision)
                row = {"run_id": run_id, "target": target, "n_params": sol.total,
                       "shape": sol.row(), "mode": mode, "micro_batch": B,
                       "seq_len": a.seq_len, "accum": a.accum, "compile": a.compile,
                       "optim": a.optim, "precision": precision, "device": a.device, **r,
                       "gpu_after": gpu_state() if a.device == "cuda" else None, "env": env}
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"{target / 1e6:>6.1f}M {cfg.d_model:>4} {cfg.n_layers:>3} "
                      f"{sol.total:>9,} {mode:>8} {B:>4} {r.get('tok_per_s', 0):>10,.0f} "
                      f"{r.get('step_s', 0):>8.3f} {r.get('h_per_1B', 0):>7.2f} "
                      f"{r.get('peak_mem_gib', 0):>7.2f} {r['status']}", flush=True)
                if r["status"] == "oom":
                    break
    print(f"results appended to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
