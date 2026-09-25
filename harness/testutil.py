"""Shared helpers for the step-3 self-test suite (test_s3_*.py). CPU only, tiny models.

scramble() gives every parameter a random non-trivial value. Fresh models hide whole
code paths: the attention gate starts at exactly 1 (zero-init projection) and the value
residual's second weight starts at 0, so a leak routed through either is invisible at
init. Every leak test runs on scrambled weights for that reason.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import yaml

from config import PlanckConfig
from make_fake_data import SPECIAL

# leave cores for whatever else runs on this machine (mutation_check.py sets 2 per worker)
torch.set_num_threads(int(os.environ.get("PLANCK_TEST_THREADS", "4")))

BASE = dict(vocab_size=97, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=16,
            mlp_hidden=64, seq_len=64)

# Every attention and looping mode the config can express (P-110, P-111, P-150, P-152).
ARMS = {
    "base_gqa": {},
    "mha": {"n_kv_heads": 2},
    "mqa_4h": {"n_heads": 4, "n_kv_heads": 1, "head_dim": 8},
    "attn_wider_than_d": {"n_heads": 3, "n_kv_heads": 3, "head_dim": 16},
    "attn_only": {"mlp_hidden": 0},
    "mlp_light": {"mlp_hidden": 16},
    "no_gate": {"attn_gate": False},
    "no_qknorm": {"qk_norm": False},
    "no_vr": {"value_residual": False},
    "no_normscale": {"norm_scaling": False},
    "plain_block": {"attn_gate": False, "qk_norm": False, "value_residual": False,
                    "norm_scaling": False},
    "kv_tie": {"kv_tie": True},
    "qk_share2": {"qk_share": 2},
    "untied": {"tie_embeddings": False},
    "loop2_cyclic": {"n_loops": 2},
    "loop3_immediate": {"n_loops": 3, "loop_order": "immediate"},
    "prelude_loop_coda": {"n_layers": 1, "n_loops": 3, "n_prelude": 1, "n_coda": 1},
    "loop_share_tie": {"n_loops": 2, "qk_share": 2, "kv_tie": True},
}


def tiny(**kw) -> PlanckConfig:
    return PlanckConfig(**{**BASE, **kw})


@torch.no_grad()
def scramble(model, seed: int = 0):
    """Every parameter random: matrices N(0, 1/fan_in), vectors and scalars 1 + N(0, 0.5^2)
    (norm gains, value-residual weights). Returns the model, in eval mode."""
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        r = torch.randn(p.shape, generator=g)
        r = r / math.sqrt(p.shape[1]) if p.dim() >= 2 else 1.0 + 0.5 * r
        p.copy_(r)
    return model.eval()


def allowed_matrix(doc: list[int] | None, T: int) -> np.ndarray:
    """Independent reference for what position t may depend on: s <= t and, with doc ids,
    s in the same maximal run of equal ids. Written with plain loops on purpose."""
    run = [0] * T
    for t in range(1, T):
        run[t] = run[t - 1] + (0 if doc is None or doc[t] == doc[t - 1] else 1)
    A = np.zeros((T, T), dtype=bool)
    for t in range(T):
        for s in range(t + 1):
            A[t, s] = run[s] == run[t]
    return A


def random_chat(rng: np.random.Generator, vocab: int = 97, turns: int = 3,
                system: bool = False, tool: bool = False, off_turn: bool = False) -> dict:
    """A SPEC s11-style record with pre-tokenized turns (content ids from 8 up)."""
    def content(lo=1, hi=7):
        return [int(x) for x in rng.integers(8, vocab, int(rng.integers(lo, hi)))]
    ts = []
    for k in range(turns):
        ts.append({"role": "user", "ids": content()})
        if tool and k == 0:
            ts.append({"role": "tool", "ids": content()})
        t = {"role": "assistant", "ids": content()}
        if off_turn and k == 0:
            t["loss"] = False
        ts.append(t)
    if system:                    # what render() makes of record["system"]: never supervised
        ts = [{"role": "system", "ids": content(), "loss": False}] + ts
    return {"turns": ts}


def template(loss: str = "assistant"):
    from chat_template import ChatTemplate
    return ChatTemplate(dict(SPECIAL["role_ids"]), SPECIAL["end_id"], loss)


PREREG = {"id": "S3", "hypothesis": "h", "metric": "m", "decision_rule": "d"}


def write_run(dirpath: str, data_dir: str, prereg=PREREG, **over) -> str:
    """A run directory with config.yaml (+ prereg.yaml) on the synthetic data in data_dir."""
    os.makedirs(dirpath, exist_ok=True)
    cfg = {
        "name": os.path.basename(dirpath), "seed": 0, "out_dir": "out", "runs_jsonl": "../runs.jsonl",
        "model": {"vocab_size": 256, "d_model": 32, "n_layers": 2, "n_heads": 2, "n_kv_heads": 1,
                  "head_dim": 16, "mlp_hidden": 64, "seq_len": 64},
        "data": {"mode": "pack", "pad_id": 0, "window_tokens": 1024, "buckets": [16, 32, 64],
                 "chat": {"role_ids": SPECIAL["role_ids"], "end_id": SPECIAL["end_id"]},
                 "sources": [{"kind": "tokens", "paths": [f"{data_dir}/text_*.bin"], "eot_id": 1,
                              "share": 0.7},
                             {"kind": "chat", "paths": [f"{data_dir}/chat_*.jsonl"], "share": 0.3}]},
        "optim": {"lr": 3e-3},
        "schedule": {"mode": "full", "warmup_frac": 0.1, "decay_frac": 0.2},
        "train": {"device": "cpu", "precision": "fp32", "micro_batch": 4, "grad_accum": 2,
                  "total_steps": 40, "log_every": 1, "ckpt_every": 10, "keep_last": 10},
    }
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    path = os.path.join(dirpath, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    if prereg is not None:
        with open(os.path.join(dirpath, "prereg.yaml"), "w") as f:
            yaml.safe_dump(prereg, f)
    return path


def read_jsonl(p) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]
