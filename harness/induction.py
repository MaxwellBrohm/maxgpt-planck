"""eval.induction: SCREENS C6 IND (and the GATE curve on attn_gate runs) during training. OFF by default: train.py
imports this module only when a config sets eval.induction, so the default path imports nothing new
(default_parity.py compares the imported modules).

  eval:
    induction:
      every: 153            # optimizer steps between readings (step % every == 0), plus the run's last step
      file: path.json       # the fixed sequences {"seqs": [[id] * 128] * 64, ...} (experiments/screens/make_ind.py)
      sha256: hex           # the file's sha256: a different file refuses to start
      gate_evalset: dir     # GATE, attn_gate runs only: an eval set dir (eval sets v0), the set, the number of
      gate_set: oasst2      #   lowest-h windows (data_prep/bpb.py's --max-windows rule) and the tokenizer
      gate_windows: 32
      gate_tokenizer: path
Paths resolve against the config's directory, as every other config path.

IND = mean NLL (nats) on copy 1 minus mean NLL on copy 2, positions 2..L of each sequence (1-based), each sequence
fed twice in a row as one document through model(idx) (doc=None, the path data_prep/bpb.py scores), fp32, no
autocast, the model's next-token logits. GATE = attn_diag's mean 2*sigmoid(W x) per effective layer and head over
the real tokens of the gate windows (doc=None, fp32). One line per reading goes to out_dir/diag.jsonl: step, ind,
nll_copy1, nll_copy2, gate (null with the gate off), seconds.
It never changes training: no_grad; it draws nothing from any RNG (the torch CPU and CUDA RNG states are compared
around every reading, and a change raises); the model's mode is not touched (PlanckLM has no mode-dependent
layer); the loader and the optimizer are never read; the plain module is called, never a compiled wrapper; the
GATE hooks are removed after each reading. test_induction.py: a CPU run with the hook on equals the run with it
off bitwise. Any other exception inside a reading is logged to diag.jsonl (error) and printed; training goes on.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback

import torch
import torch.nn.functional as F


def _abs(p: str, base: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def load_seqs(path: str, sha256: str | None) -> torch.Tensor:
    raw = open(path, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if sha256 is not None and got != sha256:
        raise ValueError(f"eval.induction file {path} has sha256 {got}, the config pins {sha256}")
    seqs = json.loads(raw)["seqs"]
    assert seqs and len({len(s) for s in seqs}) == 1 and len(seqs[0]) >= 2, "IND sequences: equal lengths >= 2"
    return torch.tensor(seqs, dtype=torch.long)


@torch.no_grad()
def ind_score(model, seqs: torch.Tensor, device: str, rows_per_batch: int = 16) -> dict:
    """seqs (N, L) -> {"ind", "nll_copy1", "nll_copy2"}: mean NLL in nats over positions 2..L of each copy."""
    N, L = seqs.shape
    s1 = s2 = 0.0
    for at in range(0, N, rows_per_batch):
        x = seqs[at:at + rows_per_batch]
        row = torch.cat([x, x], dim=1).to(device)                 # one document: the sequence twice
        logits = model(row)[0].float()
        nll = -F.log_softmax(logits[:, :-1], dim=-1).gather(-1, row[:, 1:, None]).squeeze(-1).double()
        s1 += float(nll[:, 0:L - 1].sum())                         # targets 1..L-1 (positions 2..L of copy 1)
        s2 += float(nll[:, L:2 * L - 1].sum())                     # targets L+1..2L-1 (positions 2..L of copy 2)
    n = N * (L - 1)
    return {"ind": s1 / n - s2 / n, "nll_copy1": s1 / n, "nll_copy2": s2 / n}


class Induction:
    def __init__(self, icfg: dict, base_dir: str, out_dir: str, device: str, model):
        self.every = int(icfg.get("every", 0))
        self.seqs = load_seqs(_abs(icfg["file"], base_dir), icfg.get("sha256"))
        assert 2 * self.seqs.size(1) <= model.cfg.seq_len, "IND rows (twice the sequence) exceed seq_len"
        self.file_sha256 = hashlib.sha256(open(_abs(icfg["file"], base_dir), "rb").read()).hexdigest()
        self.device, self.log_path = device, os.path.join(out_dir, "diag.jsonl")
        self.gate_rows, self.pad = None, 0
        if model.cfg.attn_gate and icfg.get("gate_evalset"):
            from attn_diag import eval_rows
            self.gate_rows, meta = eval_rows(_abs(icfg["gate_evalset"], base_dir), icfg.get("gate_set", "oasst2"),
                                             _abs(icfg["gate_tokenizer"], base_dir), int(icfg.get("gate_windows", 32)),
                                             model.cfg.seq_len)
            self.pad = meta["pad_id"]

    def reading(self, model, step: int) -> dict:
        t0 = time.time()
        rec = {"step": step, **ind_score(model, self.seqs, self.device), "gate": None}
        if self.gate_rows is not None:
            from attn_diag import read
            rec["gate"] = read(model, self.gate_rows, self.device, self.pad, 4096, sink=False, gate_values=False,
                               check=False)["gate_mean"]
        rec["seconds"] = round(time.time() - t0, 3)
        return rec

    def __call__(self, trainer):
        step = trainer.step
        if not self.every or (step % self.every and step != trainer.sched.total_steps):
            return None
        import runio
        cpu = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
        try:
            rec = self.reading(trainer.model, step)
        except Exception as e:     # a diagnostic bug must not kill a long run; it is logged, not hidden
            rec = {"step": step, "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        same = torch.equal(cpu, torch.get_rng_state()) and (
            cuda is None or all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all())))
        if not same:
            raise RuntimeError("eval.induction changed an RNG state: training would no longer match the hook-off run")
        runio.append_jsonl(self.log_path, rec)
        return rec


def make_hook(run_cfg: dict, base_dir: str, out_dir: str, device: str, model):
    """-> Induction, or None when eval.induction is absent or every is 0."""
    icfg = (run_cfg.get("eval") or {}).get("induction") or {}
    if not int(icfg.get("every", 0)):
        return None
    return Induction(icfg, base_dir, out_dir, device, model)
