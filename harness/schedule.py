"""Warmup-Stable-Decay (WSD) learning-rate factor, with decay branches (P-186).

PLAN.md recipe: 1% warmup, a stable phase, then LINEAR decay to zero over the last 20%.
Adapted from Max's MaxGPT-Ultra train/schedule.py (wsd_lr), which decays with a cosine to
min_lr; Planck changes the decay to linear-to-zero and adds branches:

  full    warmup, stable, decay over the last decay_frac of total_steps.
  trunk   warmup, then stable until total_steps (no decay). Stable checkpoints are saved
          at branch points so any of them can be decayed later.
  branch  resume from a trunk checkpoint at step s (must be past warmup, factor 1.0) and
          decay linearly to zero over decay_steps. With decay_frac f the branch has the
          same shape as a full run of length s / (1 - f): decay_steps = s * f / (1 - f).

The factor multiplies each optimizer group's base_lr (optim.set_lr). All steps are
0-based optimizer steps; factor(step) is the lr used FOR that step's update.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WSD:
    total_steps: int
    warmup_steps: int
    decay_start: int          # first decay step; == total_steps means no decay (trunk)
    mode: str = "full"

    def __post_init__(self):
        assert self.mode in ("full", "trunk", "branch"), self.mode
        assert self.total_steps >= 1 and self.warmup_steps >= 0
        assert self.warmup_steps <= self.decay_start <= self.total_steps, \
            (self.warmup_steps, self.decay_start, self.total_steps)

    @property
    def decay_steps(self) -> int:
        return self.total_steps - self.decay_start

    def factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        if step < self.decay_start:
            return 1.0
        # linear: 1.0 at decay_start, 1/decay_steps on the last step, 0 at total_steps
        return max(0.0, (self.total_steps - step) / max(1, self.decay_steps))

    def phase(self, step: int) -> str:
        if step < self.warmup_steps:
            return "warmup"
        return "stable" if step < self.decay_start else "decay"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WSD":
        return cls(**d)


def warmup_from(total_steps: int, warmup_frac: float = 0.01, warmup_steps: int | None = None) -> int:
    return int(warmup_steps) if warmup_steps is not None else max(1, round(total_steps * warmup_frac))


def full(total_steps: int, warmup_frac: float = 0.01, decay_frac: float = 0.2,
         warmup_steps: int | None = None) -> WSD:
    w = warmup_from(total_steps, warmup_frac, warmup_steps)
    d = int(round(total_steps * decay_frac))
    return WSD(total_steps, w, max(w, total_steps - d), "full")


def trunk(total_steps: int, warmup_frac: float = 0.01, warmup_steps: int | None = None) -> WSD:
    w = warmup_from(total_steps, warmup_frac, warmup_steps)
    return WSD(total_steps, w, total_steps, "trunk")


def branch(parent: WSD, at_step: int, decay_frac: float | None = 0.2,
           decay_steps: int | None = None) -> WSD:
    """Decay branch off a stable checkpoint taken after `at_step` optimizer steps (the next
    update to run is step at_step). Refuses a checkpoint that is not in the stable phase."""
    assert parent.mode in ("trunk", "full"), f"cannot branch off a {parent.mode} schedule"
    assert at_step >= parent.warmup_steps, f"step {at_step} is inside warmup"
    assert at_step <= parent.decay_start and (at_step < parent.decay_start or parent.mode == "trunk"), \
        f"step {at_step} is past the parent's stable phase (decay starts at {parent.decay_start})"
    if decay_steps is None:
        assert decay_frac is not None and 0 < decay_frac < 1
        decay_steps = max(1, round(at_step * decay_frac / (1 - decay_frac)))
    return WSD(at_step + int(decay_steps), parent.warmup_steps, at_step, "branch")


def steps_for_tokens(tokens: float, batch_tokens: int) -> int:
    return max(1, math.ceil(tokens / batch_tokens))


def branch_steps_from_tpp(tpp_list, n_params: int, batch_tokens: int) -> list[int]:
    """P-186 branch points given in tokens per parameter (500, 2000, 5000, 20000) -> steps."""
    return sorted({steps_for_tokens(t * n_params, batch_tokens) for t in tpp_list})
