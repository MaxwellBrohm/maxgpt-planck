"""K-PROBE in-training hook (SPEC 8 K-PROBE, SPEC 9 time course, SPEC 14 eval.k_probe), installed by ktrain.py with no
harness edit; built as harness/induction.py (SCREENS C6) is.

  k:
    probe:
      every: 153                       # optimizer steps between readings, plus the run's last step
      files:
        - {file: ../data/evals/k_probe.jsonl, sha256: HEX}                # paths resolve against the config's dir
        - {file: ../data/evals/k5.jsonl, sha256: HEX, sets: [FB-MISS, FB-HIT], take: 256}   # K5: first 256 of each

Each reading scores every record with probes.score (SPEC 9: LIK with tie credit over the record's candidates, CE_C,
strict GEN, greedy over the vocabulary) on the plain model, fp32 (the hook runs outside autocast), doc=None, prompts
right-padded. One line per reading goes to out_dir/k_probe.jsonl: step, seconds, "summary" (probes.summarize: per
set and family n, lik, chance, ce_c, gen), "skill_lik" per set (LIK pooled over the skill items) and "fact_acc" per
set (greedy accuracy over the bio and QA prompts: BIO_acc on K-PROBE's FB prompts).
It never changes training: no_grad; the model's mode and device are not touched; the loader and the optimizer are
never read; the torch CPU and CUDA RNG states are compared around every reading and a change raises; any other
exception inside a reading is logged (error) and printed, and training goes on.
"""
from __future__ import annotations

import hashlib
import os
import time
import traceback

import numpy as np
import torch

import kcommon as K
import probes as PR

KEYS = ("every", "files")
FILE_KEYS = ("file", "sha256", "sets", "take")


def _abs(p: str, base: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


def load_records(files: list[dict], base_dir: str) -> tuple[list[dict], list[str]]:
    recs, shas = [], []
    for f in files:
        bad = sorted(set(f) - set(FILE_KEYS))
        if bad or "file" not in f or "sha256" not in f:
            raise ValueError(f"k.probe.files entry {f}: needs file and sha256, may hold sets and take; bad {bad}")
        path = _abs(f["file"], base_dir)
        raw = open(path, "rb").read()
        got = hashlib.sha256(raw).hexdigest()
        if got != f["sha256"]:
            raise ValueError(f"k.probe file {path} has sha256 {got}, the config pins {f['sha256']}")
        shas.append(got)
        rows = K.read_jsonl(path)
        if f.get("sets") is not None:
            take = f.get("take")
            for s in f["sets"]:
                got_s = [r for r in rows if r.get("set") == s]
                if not got_s:
                    raise ValueError(f"k.probe file {path} holds no record of set {s!r}")
                recs += got_s[:int(take)] if take is not None else got_s
        else:
            if f.get("take") is not None:
                raise ValueError("k.probe: take needs sets")
            recs += rows
    if not recs:
        raise ValueError("k.probe: no records")
    return recs, shas


def logits_fn(model, device: str):
    """Next-token logits at each prompt's last position: model(idx) with doc=None, right padding, fp32 out."""
    @torch.no_grad()
    def fn(prompts: list[list[int]]) -> torch.Tensor:
        T = max(len(p) for p in prompts)
        idx = torch.zeros(len(prompts), T, dtype=torch.long)
        for i, p in enumerate(prompts):
            idx[i, :len(p)] = torch.tensor(p)
        logits = model(idx.to(device))[0]
        last = torch.tensor([len(p) - 1 for p in prompts], device=logits.device)
        return logits[torch.arange(len(prompts), device=logits.device), last].float().cpu()
    return fn


def reading(model, recs: list[dict], vpool, device: str, batch: int = 64) -> dict:
    res = PR.score(logits_fn(model, device), recs, vpool, batch)
    skill, fact = {}, {}
    for r, o in zip(recs, res):
        (skill if "qs" in r else fact).setdefault(o["set"], []).append(o)
    return {"summary": PR.summarize(res),
            "skill_lik": {s: float(np.mean([o["lik"] for o in v])) for s, v in sorted(skill.items())},
            "fact_acc": {s: float(np.mean([o["gen"] for o in v])) for s, v in sorted(fact.items())}}


class KProbe:
    def __init__(self, pcfg: dict, base_dir: str, out_dir: str, device: str):
        bad = sorted(set(pcfg) - set(KEYS))
        if bad:
            raise ValueError(f"k.probe: unknown keys {bad}")
        self.every = int(pcfg.get("every", 0))
        if self.every <= 0:
            raise ValueError("k.probe.every must be a positive step count")
        self.recs, self.file_sha256 = load_records(list(pcfg.get("files") or []), base_dir)
        import pools as P
        self.vpool = P.get()["V"]
        self.device, self.log_path = device, os.path.join(out_dir, "k_probe.jsonl")

    def __call__(self, trainer):
        step = trainer.step
        if step % self.every and step != trainer.sched.total_steps:
            return None
        import runio
        cpu = torch.get_rng_state()
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
        t0 = time.time()
        try:
            rec = {"step": step, **reading(trainer.model, self.recs, self.vpool, self.device)}
        except Exception as e:      # a diagnostic bug must not kill a long run; it is logged, not hidden
            rec = {"step": step, "error": f"{type(e).__name__}: {e}"}
            traceback.print_exc()
        rec["seconds"] = round(time.time() - t0, 3)
        same = torch.equal(cpu, torch.get_rng_state()) and (
            cuda is None or all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all())))
        if not same:
            raise RuntimeError("k.probe changed an RNG state: training would no longer match the hook-off run")
        runio.append_jsonl(self.log_path, rec)
        return rec
