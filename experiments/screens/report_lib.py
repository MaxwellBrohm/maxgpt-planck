"""C5's reported quantities (no rule) for a screen's runs, read from the copied run files, as STAGE 1 VERDICTS reported
them: per contrast arm, BASE, d per seed, the mean and the seeds per sign. Also the clip rate (log.jsonl gnorm over the
1.0 clip), the median tok/s from the third logged record, and S006's objective parts (log.jsonl loss = the next-token
loss, mtp_loss = the t+2 head's, mtp_w = w_t). Stdlib only; no rule reads any of it.
"""
from __future__ import annotations

import json
import os
import statistics as st

STEPS = 7630
LAST2 = 0.98 * STEPS                  # "the last 2% of steps": logged steps after 7,477.4 (7,480 to 7,630)
CLIP = 1.0
SETS = {"F(CHAT)": ("CHAT", "all"), "F(PROSE)": ("PROSE", "all"), "PROSE cccc": ("cccc", "all"),
        "PROSE gutenberg": ("gutenberg", "all"), "PROSE wikimedia": ("wikimedia", "all"),
        "CHAT assistant": ("oasst2", "assistant"), "CHAT user": ("oasst2", "user"), "dolly": ("dolly", "all"),
        "irc": ("irc", "all"), "stackexchange": ("stackexchange", "all")}
MEANS = {"M(CHAT)": ("CHAT", "all"), "M(PROSE)": ("PROSE", "all")}


def rows(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return [json.loads(x) for x in f.read().replace(b"\0", b"").split(b"\n") if x.strip()]


def run_values(rdir: str) -> dict:
    """{reported key: value} of one run: F from the final checkpoint, M = mean of the three scored checkpoints."""
    b = rows(os.path.join(rdir, "bpb.jsonl"))
    out = {}
    for k, (s, sp) in SETS.items():
        out[k] = next(float(r["bpb"]) for r in b if r["ckpt"].startswith("final_") and r["set"] == s and r["split"] == sp)
    for k, (s, sp) in MEANS.items():
        out[k] = st.fmean(float(r["bpb"]) for r in b if r["set"] == s and r["split"] == sp)
    lg = rows(os.path.join(rdir, "log.jsonl"))
    out["train loss last 2%"] = st.fmean(r["loss"] for r in lg if "loss" in r and r["step"] > LAST2)
    return out


def paired(arm: list[float], base: list[float]) -> dict:
    d = [a - b for a, b in zip(arm, base)]
    return {"arm": arm, "base": base, "d": d, "dbar": st.fmean(d), "neg": sum(x < 0 for x in d),
            "pos": sum(x > 0 for x in d)}


def c5(runs: str, arm_names: list[str], base_names: list[str]) -> dict:
    a = [run_values(os.path.join(runs, n)) for n in arm_names]
    b = [run_values(os.path.join(runs, n)) for n in base_names]
    return {k: paired([x[k] for x in a], [x[k] for x in b]) for k in sorted(a[0])}


def clip(rdir: str) -> dict:
    lg = [r for r in rows(os.path.join(rdir, "log.jsonl")) if "gnorm" in r]
    early = [r for r in lg if r["step"] <= 160]
    n = sum(r["gnorm"] > CLIP for r in lg)
    return {"logged": len(lg), "clipped": n, "clip_rate": n / len(lg), "gnorm_max": max(r["gnorm"] for r in lg),
            "clip_rate_steps_le_160": sum(r["gnorm"] > CLIP for r in early) / len(early)}


def tok_s(rdir: str) -> float:
    lg = [r for r in rows(os.path.join(rdir, "log.jsonl")) if "tok_per_s" in r]
    return st.median(r["tok_per_s"] for r in lg[2:])


def mtp_parts(rdir: str) -> dict:
    """S006 DIAGNOSTICS: the aux loss, w_t and the next-token head's own training loss (last 2% of steps)."""
    lg = [r for r in rows(os.path.join(rdir, "log.jsonl")) if "mtp_loss" in r]
    last = [r for r in lg if r["step"] > LAST2]
    return {"ntp_loss_last2pct": st.fmean(r["loss"] for r in last), "mtp_loss_last2pct": st.fmean(r["mtp_loss"] for r in last),
            "mtp_w_final": lg[-1]["mtp_w"], "mtp_w_first": lg[0]["mtp_w"], "logged": len(lg)}


def fmt(x, nd: int = 5, sign: bool = True) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def p3(p: float) -> str:
    """A p value at 3 significant digits, as STAGE 1's tables print it."""
    return "0" if p == 0 else f"{float(f'{p:.3g}'):g}"
