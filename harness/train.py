"""MaxGPT-Planck trainer entry point. Config-file driven; refuses to start without a
prereg.yaml next to the config.

  python train.py path/to/run_dir/config.yaml [--device cpu] [--max-steps N]
                  [--no-resume] [--require-committed]

Relative paths in the config (out_dir, runs_jsonl, data paths, schedule.init_from)
resolve against the config's directory. See configs/sample_tiny/ for every key and
notes.txt for the schedule modes (full, trunk, branch).
Exit codes: 0 done or stopped cleanly, 2 prereg gate refused, 1 anything else.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time

from device import mps_env_defaults

mps_env_defaults()   # before torch touches MPS (PLAN.md Guards)

import numpy as np   # noqa: E402
import torch         # noqa: E402

import runio         # noqa: E402
import schedule as S  # noqa: E402
from config import PlanckConfig        # noqa: E402
from count_params import count_module   # noqa: E402
from data import build_loader          # noqa: E402
from device import amp_factory, env_info, pick_device, resolve_precision  # noqa: E402
from docattn import resolve_doc_attn, set_doc_attn  # noqa: E402
from model import build_model          # noqa: E402
from mtp import MTPHead, mtp_settings   # noqa: E402
from optim import make_optimizer       # noqa: E402
from selftest import run_selftests     # noqa: E402
from trainer import Trainer            # noqa: E402


def _abs(p: str, base: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def build_schedule(sc: dict, total_steps: int, n_params: int, batch_tokens: int, init_ck):
    """-> (WSD, stable checkpoint steps)."""
    mode = sc.get("mode", "full")
    wf, ws = float(sc.get("warmup_frac", 0.01)), sc.get("warmup_steps")
    if mode == "full":
        return S.full(total_steps, wf, float(sc.get("decay_frac", 0.2)), ws), set()
    if mode == "trunk":
        pts = set(int(s) for s in sc.get("branch_points", []))
        pts |= set(S.branch_steps_from_tpp(sc.get("branch_points_tpp", []), n_params, batch_tokens))
        sched = S.trunk(total_steps, wf, ws)
        bad = [p for p in pts if not sched.warmup_steps <= p <= total_steps]
        assert not bad, f"branch points outside the stable phase: {sorted(bad)}"
        return sched, pts
    if mode == "branch":
        assert init_ck is not None, "schedule.mode branch needs schedule.init_from"
        parent = S.WSD.from_dict(init_ck["schedule"])
        return S.branch(parent, int(init_ck["step"]), sc.get("decay_frac", 0.2),
                        sc.get("decay_steps")), set()
    raise ValueError(f"unknown schedule mode {mode!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--require-committed", action="store_true")
    a = ap.parse_args(argv)

    cfg_path = os.path.abspath(a.config)
    base = os.path.dirname(cfg_path)
    try:
        prereg = runio.check_prereg(cfg_path, a.require_committed)
    except runio.PreregError as e:
        print(f"[train] {e}", file=sys.stderr, flush=True)
        return 2
    cfg = runio.load_yaml(cfg_path)
    cfg_sha = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
    tc, sc, oc = cfg.get("train", {}), cfg.get("schedule", {}), cfg.get("optim", {})
    seed = int(cfg.get("seed", 0))
    name = cfg.get("name", os.path.basename(base))
    out_dir = _abs(cfg.get("out_dir", "out"), base)
    runs_jsonl = _abs(cfg.get("runs_jsonl", "runs.jsonl"), base)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device(a.device or tc.get("device", "auto"))
    precision = resolve_precision(tc.get("precision", "auto"), device)
    amp = amp_factory(precision, device)

    mcfg = PlanckConfig.from_dict(cfg["model"])
    model = build_model(mcfg, "cpu").to(device)
    n_params = count_module(model)["total"]
    mtp, mtp_weight = mtp_settings(tc)     # S006 (train.mtp, mtp.py): 0 = off, the default
    head = MTPHead(mcfg.d_model, mcfg.rms_eps).to(device) if mtp else None   # training only, draws nothing
    dmode = cfg["data"].get("mode", "pack")
    # packed-row attention kernel (docattn.py); auto = varlen on cuda+bf16, else mask
    doc_attn = resolve_doc_attn(tc.get("doc_attn", "auto"), device, precision)
    set_doc_attn(model, doc_attn, device)
    if tc.get("selftest", True):
        st = run_selftests(model, device, amp, packed=(dmode == "pack"))
        print(f"[train] leak self-test passed {st}", flush=True)

    micro, accum = int(tc.get("micro_batch", 8)), int(tc.get("grad_accum", 1))
    loader = build_loader(cfg["data"], mcfg.seq_len, micro, base, seed)
    per_micro = micro * mcfg.seq_len if dmode == "pack" else loader.tokens_per_micro
    batch_tokens = per_micro * accum
    total_steps = int(tc["total_steps"]) if "total_steps" in tc else \
        S.steps_for_tokens(float(tc["total_tokens"]), batch_tokens)
    opt = make_optimizer(model, oc, device, extra=head)

    init_ck = None
    if sc.get("init_from"):
        init_ck = runio.load_checkpoint(_abs(sc["init_from"], base))
        assert init_ck["model_cfg"] == mcfg.to_dict(), "init_from checkpoint has another model shape"
    sched, stable_pts = build_schedule(sc, total_steps, n_params, batch_tokens, init_ck)

    meta = {"model_cfg": mcfg.to_dict(), "n_params": n_params, "prereg_sha256": prereg["sha256"],
            "config_sha256": cfg_sha, "seed": seed, "precision": precision}
    if head is not None:                   # n_params stays the deployed model's total
        meta["mtp_params"] = sum(p.numel() for p in head.parameters())
    tr = Trainer(model=model, optimizer=opt, loader=loader, sched=sched, device=device, amp=amp,
                 cfg=cfg, out_dir=out_dir, grad_accum=accum, grad_clip=float(tc.get("grad_clip", 1.0)),
                 log_every=int(tc.get("log_every", 10)), ckpt_every=int(tc.get("ckpt_every", 500)),
                 keep_last=int(tc.get("keep_last", 2)), stable_points=stable_pts, meta=meta,
                 lazy_metrics=bool(tc.get("lazy_metrics", False)), mtp=head, mtp_weight=mtp_weight)

    if (cfg.get("eval") or {}).get("rc12"):   # off unless eval.rc12.every > 0 (rc12_eval.py)
        from rc12_eval import make_hook
        hook = make_hook(cfg, base, out_dir, device, amp, mcfg.seq_len)
        if hook is not None:
            tr.hooks.append(hook)
            print(f"[train] RC-12 eval every {hook.every} steps on {len(hook.recs)} conversations", flush=True)

    resumed_from = None
    last = None if a.no_resume else runio.latest_checkpoint(out_dir)
    if last:
        ck = runio.load_checkpoint(last)
        assert ck["model_cfg"] == mcfg.to_dict(), f"{last} has another model shape"
        assert ck["schedule"] == sched.to_dict(), f"{last} was trained on another schedule"
        tr.load_state(ck)
        resumed_from = last
    elif init_ck is not None:           # a decay branch starts from the trunk's state
        tr.load_state(init_ck)
        resumed_from = _abs(sc["init_from"], base)

    signal.signal(signal.SIGTERM, lambda *_: setattr(tr, "stop_requested", True))
    start = {"event": "start", "run": name, "prereg_id": prereg["id"],
             "prereg_sha256": prereg["sha256"], "prereg_committed": prereg["committed"],
             "config": cfg_path, "config_sha256": cfg_sha, "out_dir": out_dir,
             "n_params": n_params, "step": tr.step, "schedule": sched.to_dict(),
             "batch_tokens": batch_tokens, "precision": precision, "resumed_from": resumed_from,
             "env": env_info(device)}
    if model.doc_attn != "mask":            # the speed switches, recorded when not the reference
        start["doc_attn"] = model.doc_attn
    if opt.batched:
        start["optim_batched"] = True
    if tr.lazy_metrics:
        start["lazy_metrics"] = True
    if head is not None:                    # S006: deployed total (n_params) and training-only, apart
        start["mtp"] = {"heads": mtp, "weight": mtp_weight, "training_only_params": meta["mtp_params"]}
    runio.append_jsonl(runs_jsonl, start)
    print(f"[train] {name}: {n_params:,} params, {device}/{precision}, steps {tr.step}->"
          f"{sched.total_steps} ({sched.mode}), {batch_tokens:,} tokens/step", flush=True)
    t0 = time.time()
    try:
        last_rec = tr.run(a.max_steps)
    except Exception as e:
        runio.append_jsonl(runs_jsonl, {"event": "error", "run": name, "step": tr.step,
                                        "error": f"{type(e).__name__}: {e}"})
        raise
    done = tr.step >= sched.total_steps
    runio.append_jsonl(runs_jsonl, {
        "event": "end" if done else "pause", "run": name, "prereg_id": prereg["id"],
        "step": tr.step, "total_steps": sched.total_steps, "loss": last_rec.get("loss"),
        "tokens": tr.tokens, "sup_tokens": tr.sup_tokens, "seconds": round(time.time() - t0, 1),
        "checkpoint": runio.latest_checkpoint(out_dir), **loader.stats()})
    print(f"[train] {'finished' if done else 'paused'} at step {tr.step}, loss {last_rec.get('loss')}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
