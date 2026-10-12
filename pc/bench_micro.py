"""bench_micro: training tokens per second for the Planck curve shapes, for the RTX 5070.

  python pc/bench_micro.py                                   # 5M 10M 20M 30M on CUDA
  python pc/bench_micro.py --targets 3e6,5e6 --batches 8,16,32,64 --modes causal,docmask
  python pc/bench_micro.py --compile                         # torch.compile (mode default) only
  python pc/bench_micro.py --arms eager,default --repeats 5  # interleaved A/B, same process
  python pc/bench_micro.py --ce-chunk 2048                   # chunked lm_head + loss (no logits)
  python pc/bench_micro.py --arms eager,default,eager+ce,default+ce --ce-chunk 2048 --repeats 5
  python pc/bench_micro.py --loops 2                         # a looped arm (any budget.py flag)
  python pc/bench_micro.py --batched-optim off --doc-attn mask   # the pre-2026-09-26 reference step
  python pc/bench_micro.py --lazy-loss --phase-times         # no per-step loss sync; time the update
  python pc/bench_micro.py --device cpu --targets 3e5 --vocab 512 --seq-len 64 \
      --batches 2 --steps 2 --warmup 1                        # CPU smoke test (the Mac tests)

Each shape is the one harness/budget.py solve() picks for the target (the same Constraints
flags as budget.py; defaults vocab 8192, aspect 24, SwiGLU 8/3), built by harness/model.py
with the harness optimizer (optim.make_optimizer: NorMuon on matrices, AdamW on embedding
and scalars) and bf16 autocast (device.amp_factory), exactly as train.py builds them.
One timed step = forward (reduction "sum" / tokens, as trainer.py), backward, grad clip 1.0,
optimizer step, zero_grad; --accum micro-batches per optimizer step. Tokens are random.

Modes: causal  doc=None, SDPA is_causal (the flash path; bucket mode's rows)
       docmask packed rows of about --docs-per-row documents (pack mode's path). --doc-attn
               auto (default, as train.py) = varlen on cuda+bf16: flash varlen attention, one
               sequence per document (docattn.py); mask = SDPA with the harness's bool
               document mask (the reference; no flash kernel, mem-efficient with a dense bias)
A batch that runs out of memory is recorded as "oom" and larger batches of that shape and
mode are skipped. Every measurement appends one JSON line to --out, with torch, CUDA, GPU
and driver versions. Nothing here downloads anything.

Memory (CUDA). The Windows (WDDM) driver does not raise OOM when the card is full: it pages
into system RAM, and the step may keep its speed or get several times slower. peak_mem_gib
(max_memory_allocated, kept for old rows) cannot answer "does this fit": it leaves out the
allocator's cached blocks (0.4 to 1.1 GiB on the 5M-30M shapes) and the ~1.2 GiB that the CUDA
context, libraries and the desktop hold (about 10.7 GiB of the 5070's 11.94 is left for the
allocator). Each row also records peak_reserved_gib (the caching allocator's peak),
free_before_gib and reserved_before_gib (before the model is built; after the first cell in a
process about 0.5 GiB stays reserved that empty_cache cannot release), free_gib (free after
the timed steps, before anything is freed) and mem_fit: "over" when the peak reservation
exceeds free_before + reserved_before (part of it cannot be on the card), "edge" when under
--min-free-gib is left, else "ok". Choose batches by mem_fit == "ok".
Timing: tok_per_s and step_s are wall clock over the timed steps (warmup excluded). tok_per_s
counts every position, B x seq_len x accum per step (bench rows have no pads; train.py counts
non-pad tokens only, so hours projected from here are a lower bound). docmask rows record
docs_per_row, which the varlen speed depends on: the default 8 (short documents) flatters
docmask; the E2 starter data is closer to --docs-per-row 1 (harness/notes.txt SPEED INTEGRATE 2).
step_ms_median / step_max_over_median come from per-step CUDA events (wall clock on CPU);
first_step_s is warmup step 0 alone, which holds the torch.compile time under --compile.
loss_first is the loss of warmup step 0.
Optimizer switches: --batched-optim auto|on|off (bare flag = on; default auto = the harness
default, on for cuda) sets optim key batched (harness/optim_batched.py); --lazy-loss (default
off) keeps the loss on the device instead of float() after every backward (as train key
lazy_metrics). Rows record the resolved batched_optim and doc_attn. --phase-times records
opt_ms_median (CUDA events around clip + opt.step + zero_grad: GPU time from the end of
backward to the end of the update, idle gaps included) and opt_cpu_ms_median (host time to
issue that phase) over the timed steps.
Arms (train.compile): "eager" or a torch.compile mode (default, max-autotune-no-cudagraphs:
model.COMPILE_MODES), compiled through harness model.compile_forward exactly as train.py does.
train.py itself accepts only default (train.TRAIN_COMPILE_MODES); max-autotune-no-cudagraphs is a
bench arm only (its old-tree parity failed one 20M check; harness notes SPEED V3 FOLLOW-UPS).
--arms a,b --repeats N runs every cell N times per arm, interleaved (ABBA order), and prints the
median tok/s per arm over mem_fit "ok" rows. Each compiled cell starts from torch._dynamo.reset()
(so first_step_s holds its compile, warm on-disk caches included) and records dynamo_graphs and
graph_breaks. On docmask rows with doc_attn varlen, cu_seqlens' nonzero() breaks the graph
(harness notes SPEED V3); --dynamo-capture-dynamic sets Dynamo's capture_dynamic_output_shape_ops
for the whole bench process (a bench-only experiment, not a train.py option; recorded per row).
--ce-chunk N (train.ce_chunk_rows): lm_head + the loss N rows at a time without the full logits
(harness/chunked_ce.py); 0 = the reference loss path. An arm ending in "+ce" (eager+ce,
default+ce) chunks with N and the others do not, so chunked and plain interleave in one process;
with no "+ce" arm, N > 0 chunks every arm. Rows record the ce_chunk each run used.
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
from docattn import CHOICES, resolve_doc_attn, set_doc_attn  # noqa: E402
from model import build_model, compile_forward, compile_mode  # noqa: E402
from optim import make_optimizer, resolve_batched  # noqa: E402


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


class StepTimer:
    """Per-step times: CUDA events on the GPU (no extra host sync), wall clock elsewhere."""

    def __init__(self, device: str):
        self.cuda = device == "cuda"
        self.marks: list = []
        self.mark()

    def mark(self) -> None:
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.marks.append(e)
        else:
            self.marks.append(time.perf_counter())

    def summary(self) -> dict:
        """Call after a device sync."""
        m = self.marks
        ms = [m[i].elapsed_time(m[i + 1]) if self.cuda else 1000.0 * (m[i + 1] - m[i])
              for i in range(len(m) - 1)]
        med = sorted(ms)[len(ms) // 2]
        return {"step_ms_median": round(med, 2), "step_ms_max": round(max(ms), 2),
                "step_max_over_median": round(max(ms) / med, 3) if med > 0 else None}


class PhaseTimer:
    """--phase-times: CUDA events (GPU timeline) and host wall time around one phase."""

    def __init__(self, device: str):
        self.cuda = device == "cuda"
        self.gpu: list = []
        self.cpu: list[float] = []

    def begin(self) -> None:
        self.t0 = time.perf_counter()
        if self.cuda:
            self.e0 = torch.cuda.Event(enable_timing=True)
            self.e0.record()

    def end(self) -> None:
        if self.cuda:
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()
            self.gpu.append((self.e0, e1))
        self.cpu.append(1000.0 * (time.perf_counter() - self.t0))

    def summary(self) -> dict:
        """Call after a device sync."""
        med = lambda v: round(sorted(v)[len(v) // 2], 2) if v else None  # noqa: E731
        return {"opt_ms_median": med([a.elapsed_time(b) for a, b in self.gpu]),
                "opt_cpu_ms_median": med(self.cpu)}


def mem_before() -> tuple[int, int]:
    """(free, reserved) bytes before a cell builds anything."""
    return torch.cuda.mem_get_info()[0], torch.cuda.memory_reserved()


def mem_report(before: tuple[int, int], min_free_gib: float) -> dict:
    """CUDA memory for the spill question (see the module docstring). Call before freeing."""
    G = 2**30
    free_before, reserved_before = before
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.max_memory_reserved()
    fit = ("over" if reserved > free_before + reserved_before else
           "edge" if free < min_free_gib * G else "ok")
    return {"peak_mem_gib": round(torch.cuda.max_memory_allocated() / G, 2),
            "peak_reserved_gib": round(reserved / G, 2),
            "free_before_gib": round(free_before / G, 2),
            "reserved_before_gib": round(reserved_before / G, 2), "free_gib": round(free / G, 2),
            "total_gib": round(total / G, 2), "mem_fit": fit}


def parse_arm(arm: str, a) -> tuple[str | None, int]:
    """"<eager | a compile mode>[+ce]" -> (torch.compile mode or None, ce_chunk rows or 0)."""
    spec, _, suffix = arm.partition("+")
    if suffix not in ("", "ce"):
        raise ValueError(f"arm {arm!r}: the only suffix is +ce")
    if suffix and a.ce_chunk < 1:
        raise ValueError(f"arm {arm!r} needs --ce-chunk N > 0")
    return compile_mode(spec), a.ce_chunk if (suffix or not a.any_ce) else 0


def bench_one(cfg, mode: str, B: int, a, device: str, precision: str, arm: str = "eager") -> dict:
    cmode, ce_chunk = parse_arm(arm, a)
    if cmode:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
    before = mem_before() if device == "cuda" else None
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    model = build_model(cfg, "cpu").to(device)
    set_doc_attn(model, a.doc_attn, device)
    opt = make_optimizer(model, {"kind": a.optim, "batched": a.batched_optim}, device_type=device)
    fwd = compile_forward(model, cmode, a.compile_backend)
    kw = {"ce_chunk": ce_chunk} if ce_chunk else {}
    rec_ce = {"ce_chunk": ce_chunk}
    amp = amp_factory(precision, device)
    batches = [make_batch(B, cfg.seq_len, cfg.vocab_size, mode, a.docs_per_row, g, device)
               for _ in range(min(a.accum, 4))]
    # parameters() yields a tied or shared tensor once, so this is the TOTAL the curve counts
    rec = {"status": "ok", "n_params_built": sum(p.numel() for p in model.parameters()), **rec_ce}
    phases = None                                   # a PhaseTimer during the timed steps

    def step(i: int):
        loss_sum = 0.0
        for j in range(a.accum):
            idx, tgt, doc = batches[(i * a.accum + j) % len(batches)]
            with (amp() if amp else nullcontext()):
                _, ls = fwd(idx, tgt, doc, reduction="sum", **kw)
            (ls / idx.numel() / a.accum).backward()
            if a.lazy_loss:                          # a device tensor, read after the loop
                loss_sum = loss_sum + ls.detach() / idx.numel()
            else:
                loss_sum += float(ls.detach()) / idx.numel()
        if phases is not None:
            phases.begin()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if phases is not None:
            phases.end()
        return loss_sum / a.accum

    try:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for i in range(a.warmup):
            loss = step(i)
            if i == 0:
                sync(device)
                first, rec["first_step_s"] = float(loss), round(time.time() - t0, 2)
        sync(device)
        rec["warmup_s"] = round(time.time() - t0, 2)
        timer = StepTimer(device)
        if a.phase_times:
            phases = PhaseTimer(device)
        t0 = time.time()
        for i in range(a.steps):
            last = step(a.warmup + i)
            timer.mark()
        sync(device)
        dt = time.time() - t0
        last = float(last)
        tokens = B * cfg.seq_len * a.accum * a.steps
        rec.update(tok_per_s=round(tokens / dt, 1), step_s=round(dt / a.steps, 4),
                   h_per_1B=round(1e9 / (tokens / dt) / 3600, 2),
                   loss_first=round(first, 4) if a.warmup else None, loss_last=round(last, 4),
                   **timer.summary(), **(phases.summary() if phases else {}))
        if not math.isfinite(last):
            rec["status"] = "nonfinite"
        if device == "cuda":
            rec.update(mem_report(before, a.min_free_gib))
        if cmode:
            c = torch._dynamo.utils.counters
            rec.update(dynamo_graphs=int(c["stats"]["unique_graphs"]),
                       graph_breaks=int(sum(c["graph_break"].values())))
    except (RuntimeError, torch.OutOfMemoryError) as e:
        rec = {"status": "oom" if is_oom(e) else "error", "error": str(e).splitlines()[0][:300],
               "n_params_built": rec["n_params_built"], **rec_ce}
    finally:
        del model, opt, fwd, batches
        if cmode:
            torch._dynamo.reset()      # drop the compiled graphs before the next cell
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
    ap.add_argument("--compile", action="store_true", help="shorthand for --arms default")
    ap.add_argument("--arms", default=None,
                    help="comma list of eager and torch.compile modes (default: eager)")
    ap.add_argument("--repeats", type=int, default=1, help="runs per arm per cell, interleaved")
    ap.add_argument("--compile-backend", default="inductor", help="aot_eager: CPU tests only")
    ap.add_argument("--dynamo-capture-dynamic", action="store_true",
                    help="Dynamo capture_dynamic_output_shape_ops (bench-only experiment)")
    ap.add_argument("--ce-chunk", type=int, default=0,
                    help="rows per lm_head + loss chunk (harness/chunked_ce.py); 0 = reference path")
    ap.add_argument("--batched-optim", nargs="?", const="on", default="auto",
                    choices=["auto", "on", "off"],
                    help="optim key batched (optim_batched.py): auto (default) = on for cuda")
    ap.add_argument("--lazy-loss", action="store_true", help="no float(loss) per micro-batch")
    ap.add_argument("--phase-times", action="store_true", help="time clip + optimizer step")
    ap.add_argument("--doc-attn", default="auto", choices=CHOICES,
                    help="packed-row attention for docmask mode (causal mode never reads it): "
                         "auto (default) = varlen on cuda+bf16, else mask")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--min-free-gib", type=float, default=0.5,
                    help='free GiB left after the timed steps below which mem_fit is "edge"')
    ap.add_argument("--out", default=os.path.join(HERE, "bench_results.jsonl"))
    budget.add_constraint_args(ap)
    a = ap.parse_args(argv)
    if a.steps < 1 or a.accum < 1 or a.warmup < 0 or a.repeats < 1 or a.ce_chunk < 0:
        raise SystemExit("--steps, --accum and --repeats must be >= 1, --warmup and --ce-chunk >= 0")
    arms = (a.arms or ("default" if a.compile else "eager")).split(",")
    a.any_ce = any(arm.endswith("+ce") for arm in arms)
    for arm in arms:
        parse_arm(arm, a)   # ValueError on an unknown mode or suffix
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available (this benchmark is meant for the PC)")
    if a.dynamo_capture_dynamic:              # reset at the end of main
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
    precision = resolve_precision(a.precision, a.device)
    # the harness defaults (auto) resolved here, so every row records what actually ran
    a.batched_optim = resolve_batched({"on": True, "off": False}.get(a.batched_optim, "auto"),
                                      a.device)
    a.doc_attn = resolve_doc_attn(a.doc_attn, a.device, precision)
    env = env_info(a.device)
    if a.device == "cuda":
        env["capability"] = list(torch.cuda.get_device_capability(0))
        env["driver"] = driver_version()
    k = budget.constraints_from_args(a)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    print(f"bench_micro {run_id}: {env.get('gpu', a.device)} torch {env['torch']} "
          f"precision {precision} T={a.seq_len} accum={a.accum} arms={','.join(arms)} x{a.repeats} "
          f"batched_optim={a.batched_optim} lazy_loss={a.lazy_loss} doc_attn={a.doc_attn} "
          f"ce_chunk={a.ce_chunk}" + (" capture_dynamic" if a.dynamo_capture_dynamic else ""))
    print(f"{'target':>7} {'d':>4} {'L':>3} {'params':>9} {'mode':>8} {'B':>4} {'arm':>10} "
          f"{'tok/s':>10} {'step s':>8} {'h/1B':>7} {'peakGiB':>7} {'resGiB':>6} {'freeGiB':>7} "
          f"fit   {'1st s':>6} status")
    medians: list[str] = []
    for target in [float(t) for t in a.targets.split(",")]:
        sol = budget.solve(target, k)
        cfg = sol.cfg.replace(seq_len=a.seq_len)
        for mode in a.modes.split(","):
            assert mode in ("causal", "docmask"), mode
            oomed: set[str] = set()      # an arm that ran out of memory skips larger batches
            for B in [int(b) for b in a.batches.split(",")]:
                speeds: dict[str, list[float]] = {arm: [] for arm in arms}
                for rep in range(a.repeats):
                    for arm in (arms if rep % 2 == 0 else arms[::-1]):
                        if arm in oomed:
                            continue
                        r = bench_one(cfg, mode, B, a, a.device, precision, arm)
                        row = {"run_id": run_id, "target": target, "n_params": sol.total,
                               "shape": sol.row(), "mode": mode, "micro_batch": B,
                               "seq_len": a.seq_len, "accum": a.accum,
                               "compile": parse_arm(arm, a)[0] is not None, "arm": arm, "repeat": rep,
                               "capture_dynamic": a.dynamo_capture_dynamic,
                               "batched_optim": a.batched_optim, "lazy_loss": a.lazy_loss,
                               "doc_attn": a.doc_attn,
                               "docs_per_row": a.docs_per_row if mode == "docmask" else None,
                               "optim": a.optim, "precision": precision, "device": a.device, **r,
                               "gpu_after": gpu_state() if a.device == "cuda" else None, "env": env}
                        with open(a.out, "a") as f:
                            f.write(json.dumps(row) + "\n")
                        print(f"{target / 1e6:>6.1f}M {cfg.d_model:>4} {cfg.n_layers:>3} "
                              f"{sol.total:>9,} {mode:>8} {B:>4} {arm[:10]:>10} "
                              f"{r.get('tok_per_s', 0):>10,.0f} "
                              f"{r.get('step_s', 0):>8.3f} {r.get('h_per_1B', 0):>7.2f} "
                              f"{r.get('peak_mem_gib', 0):>7.2f} {r.get('peak_reserved_gib', 0):>6.2f} "
                              f"{r.get('free_gib', 0):>7.2f} {r.get('mem_fit', '-'):<5} "
                              f"{r.get('first_step_s', 0):>6.1f} {r['status']}", flush=True)
                        if r["status"] == "oom":
                            oomed.add(arm)
                        elif r["status"] == "ok" and r.get("mem_fit", "ok") == "ok":
                            speeds[arm].append(r["tok_per_s"])
                if a.repeats > 1 or len(arms) > 1:
                    medians.append(f"{target / 1e6:>6.1f}M {mode:>8} B{B:<3} " + "  ".join(
                        f"{arm}: {sorted(v)[len(v) // 2]:,.0f} (n={len(v)})" if v else f"{arm}: -"
                        for arm, v in speeds.items()))
                if len(oomed) == len(arms):
                    break
    if medians:
        print("median tok/s over mem_fit ok rows (upper median):")
        print("\n".join(medians))
    if a.dynamo_capture_dynamic:
        torch._dynamo.config.capture_dynamic_output_shape_ops = False
    print(f"results appended to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
