"""One T001 run: train an arm on the toy task, score the fixed eval sets, write one JSON (PREREG.txt TRAINING).

  python train.py --arm base --seed 1 --lr 1e-3 --out out/base_s1_lr1e-3.json [--steps 6000 --n-hi 96 ...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ToyLM, n_params  # noqa: E402
from task import IGNORE, SPLIT_SIZE, V, VAL0, eval_rng, make_batch, train_rng  # noqa: E402

GRID = (8, 16, 24, 32, 48, 64, 96)
WARMUP, WD, CLIP = 200, 0.1, 1.0
BETAS, ADAM_EPS = (0.9, 0.95), 1e-8


def lr_at(step: int, steps: int, lr: float) -> float:
    if step < WARMUP:
        return lr * (step + 1) / WARMUP
    p = (step - WARMUP) / max(1, steps - WARMUP)
    return lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p))))


@torch.no_grad()
def score(model, family: str, n: int, split: str, size: int, device: str, bs: int = 32) -> dict:
    """Aggregates over a fixed eval set: the prediction is the argmax over the value tokens at each answer position."""
    model.eval()
    x, _, meta = make_batch(n, size, eval_rng(split, family, n), family)
    c = dict(q=0, ok=0, upd_q=0, upd_ok=0, stale=0, first=0, trk_q=0, trk_ok=0, ov_q=0, ov_ok=0, tie_q=0, tie_ok=0)
    for s in range(0, size, bs):
        logits = model(torch.from_numpy(x[s:s + bs]).to(device))[..., VAL0:VAL0 + V]
        pred = (logits.argmax(-1) + VAL0).cpu().numpy()
        for r, qs in enumerate(meta[s:s + bs]):
            for q in qs:
                p = int(pred[r, q["pos"]])
                ok = p == q["ans"]
                c["q"] += 1
                c["ok"] += ok
                if q["updated"]:
                    c["upd_q"] += 1
                    c["upd_ok"] += ok
                    c["stale"] += p in q["stale"]
                    c["first"] += p == q["first"] and q["first"] != q["ans"]
                if q["tracked"]:
                    c["trk_q"] += 1
                    c["trk_ok"] += ok
                if q["outvoted"]:
                    c["ov_q"] += 1
                    c["ov_ok"] += ok
                if q["tied"]:
                    c["tie_q"] += 1
                    c["tie_ok"] += ok
    model.train()
    out = dict(c)
    out["acc"] = c["ok"] / c["q"] if c["q"] else None
    out["lw"] = c["upd_ok"] / c["upd_q"] if c["upd_q"] else None
    out["trk_acc"] = c["trk_ok"] / c["trk_q"] if c["trk_q"] else None
    out["acc_ov"] = c["ov_ok"] / c["ov_q"] if c["ov_q"] else None
    out["acc_tie"] = c["tie_ok"] / c["tie_q"] if c["tie_q"] else None
    return out


def evaluate(model, family: str, grid, device: str, sizes: dict) -> dict:
    fams = {"mqrar": ("mqrar", "ar"), "track": ("track", "mqrar")}[family]
    ev = {"test": {f: {str(n): score(model, f, n, "test", sizes["test"], device) for n in grid} for f in fams}}
    if family == "mqrar":
        ev["dev"] = {"mqrar": {str(n): score(model, "mqrar", n, "dev", sizes["dev"], device) for n in grid}}
    return ev


def make_opt(model, lr: float):
    """AdamW, weight decay WD on tensors of dim >= 2 only (PREREG TRAINING)."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": WD}, {"params": other, "weight_decay": 0.0}],
                             lr=lr, betas=BETAS, eps=ADAM_EPS)


def draw_n(rng, lo: int, hi: int) -> int:
    """A training step's n: uniform over the integers lo..hi, both included."""
    return int(rng.integers(lo, hi + 1))


def update(model, opt, loss) -> None:
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step()


def run(a) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(a.seed)
    model = ToyLM(a.arm).to(a.device)
    opt = make_opt(model, a.lr)
    rng = train_rng(a.seed)
    curve, diverged, done, t0 = [], False, 0, time.time()
    for step in range(a.steps):
        n = draw_n(rng, a.n_lo, a.n_hi)
        x, y, _ = make_batch(n, a.batch, rng, a.family)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr)
        logits = model(torch.from_numpy(x).to(a.device))
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), torch.from_numpy(y).to(a.device).view(-1),
                               ignore_index=IGNORE)
        if not torch.isfinite(loss):
            diverged = True
            curve.append([step, None])
            break
        update(model, opt, loss)
        done += 1
        if step % a.log_every == 0 or step == a.steps - 1:
            curve.append([step, loss.item()])
    train_s = time.time() - t0
    grid = [int(g) for g in a.grid.split(",")]
    ev = None if diverged else evaluate(model, a.family, grid, a.device, {"test": a.eval_test, "dev": a.eval_dev})
    return dict(arm=a.arm, seed=a.seed, lr=a.lr, steps=a.steps, batch=a.batch, family=a.family, n_lo=a.n_lo,
                n_hi=a.n_hi, grid=grid, params=n_params(model), device=a.device, torch=torch.__version__,
                python=platform.python_version(), smoke=a.smoke, diverged=diverged, train_s=round(train_s, 2),
                steps_done=done, sec_per_step=round(train_s / max(1, done), 5),
                wall_s=round(time.time() - t0, 2), loss_curve=curve, eval=ev)


def write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--family", default="mqrar", choices=("mqrar", "track"))
    p.add_argument("--n-lo", type=int, default=8)
    p.add_argument("--n-hi", type=int, default=96)
    p.add_argument("--grid", default=",".join(str(g) for g in GRID))
    p.add_argument("--eval-test", type=int, default=SPLIT_SIZE["test"])
    p.add_argument("--eval-dev", type=int, default=SPLIT_SIZE["dev"])
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="label the output a smoke: never read for results")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse()
    np.seterr(all="raise")
    write_json(args.out, run(args))
