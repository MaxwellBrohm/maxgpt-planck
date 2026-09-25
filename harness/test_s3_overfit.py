"""(g) 200-step overfit of a ~0.51M-parameter model on a tiny synthetic dialogue set, CPU.

The data is built so that passing needs the whole pipeline to be right AND the model to
use conversation history: every assistant reply is random (pure memorization), and every
second and third user turn is the SAME follow-up prompt, so replies 2 and 3 can only be
told apart by the earlier turns. Training goes through train.py (prereg gate, packing
with the document mask, assistant-only loss, NorMuon, WSD). Pass = final assistant-token
loss under 0.05 AND greedy decoding reproduces every assistant turn exactly, stopping at
<|end|>, from the gold prefix. Must finish in under 2 minutes.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest
import torch

import runio
import train
from budget import count_analytic
from config import PlanckConfig
from make_fake_data import SPECIAL
from model import build_model
from testutil import read_jsonl, template, write_run

MODEL = {"vocab_size": 128, "d_model": 128, "n_layers": 3, "n_heads": 4, "n_kv_heads": 4,
         "head_dim": 32, "mlp_hidden": 256, "seq_len": 64}
FOLLOW = [9, 10, 11]                 # the shared follow-up user prompt
END, ASST = SPECIAL["end_id"], SPECIAL["role_ids"]["assistant"]


def dialogues(n: int = 16, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)

    def toks(lo, hi):
        return [int(x) for x in rng.integers(12, 128, int(rng.integers(lo, hi + 1)))]
    out = []
    for c in range(n):
        turns = [{"role": "user", "ids": toks(4, 7)}, {"role": "assistant", "ids": toks(3, 6)},
                 {"role": "user", "ids": list(FOLLOW)}, {"role": "assistant", "ids": toks(3, 6)}]
        if c % 3 == 0:
            turns += [{"role": "user", "ids": list(FOLLOW)}, {"role": "assistant", "ids": toks(3, 6)}]
        out.append({"id": f"d{c}", "turns": turns})
    return out


@torch.no_grad()
def greedy(model, prefix: list[int], max_new: int = 12) -> list[int]:
    ids = list(prefix)
    out = []
    for _ in range(max_new):
        logits, _ = model(torch.tensor([ids]))
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        ids.append(nxt)
        if nxt == END:
            break
    return out


ARMS = {"pack_dense": ({"mode": "pack"}, {}),
        "bucket_dense": ({"mode": "bucket", "buckets": [16, 32, 64], "tokens_per_micro": 512}, {}),
        "pack_looped2": ({"mode": "pack"}, {"n_loops": 2})}


@pytest.mark.overfit
@pytest.mark.parametrize("arm", list(ARMS))
def test_overfit_200_steps(tmp_path, arm):
    t0 = time.time()
    dmode, mextra = ARMS[arm]
    model = {**MODEL, **mextra}
    n_params = count_analytic(PlanckConfig(**model))["total"]
    assert 450_000 <= n_params <= 550_000, n_params
    recs = dialogues()
    data = tmp_path / "data"
    os.makedirs(data)
    with open(data / "chat_000.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    cfg = write_run(
        str(tmp_path / "run"), str(data), model=model,
        data={**dmode, "sources": [{"kind": "chat", "paths": [f"{data}/chat_*.jsonl"], "share": 1.0}]},
        optim={"lr": 1e-2, "embed_lr": 1e-2, "scalar_lr": 1e-2, "weight_decay": 0.0},
        schedule={"warmup_frac": 0.05, "decay_frac": 0.3},
        train={"total_steps": 200, "micro_batch": 8, "grad_accum": 1, "log_every": 10,
               "ckpt_every": 0})
    assert train.main([cfg]) == 0
    log = read_jsonl(tmp_path / "run" / "out" / "log.jsonl")
    assert log[-1]["step"] == 200
    assert log[-1]["loss"] < 0.05, [r["loss"] for r in log]

    ck = runio.load_checkpoint(runio.latest_checkpoint(str(tmp_path / "run" / "out")))
    m = build_model(PlanckConfig.from_dict(ck["model_cfg"]))
    m.load_state_dict(ck["model"])
    m.eval()
    tm = template()
    wrong, total = [], 0
    for r in recs:
        for k, t in enumerate(r["turns"]):
            if t["role"] != "assistant":
                continue
            prefix, _ = tm.render({"turns": r["turns"][:k]})
            got = greedy(m, prefix.tolist() + [ASST])
            total += 1
            if got != t["ids"] + [END]:
                wrong.append((r["id"], k, got, t["ids"]))
    assert total == 16 * 2 + 6
    assert not wrong, f"{len(wrong)}/{total} assistant turns not reproduced: {wrong[:4]}"
    elapsed = time.time() - t0
    print(f"overfit {arm}: {n_params:,} params, final loss {log[-1]['loss']}, {total}/{total} turns exact, "
          f"{elapsed:.1f}s")
    assert elapsed < 120, elapsed
