"""Run bookkeeping: config loading, the pre-registration gate, runs.jsonl, checkpoints.

Config loading and atomic checkpoint writes are adapted from Max's MaxGPT-Ultra (cfg.py
load_yaml with `extends:`, train/checkpoint.py _atomic_save and keep-last pruning).
Planck adds the prereg gate: PLAN.md "One prereg.yaml per experiment, committed before
launch. The runner enforces this." A run refuses to start unless prereg.yaml sits next to
its config and names a hypothesis, metric and decision rule. With require_committed the
file must also be tracked by git and unchanged from HEAD (the queue runner turns this on).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import time

import torch
import yaml

PREREG_REQUIRED = ("id", "hypothesis", "metric", "decision_rule")


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_yaml(path: str) -> dict:
    """From Ultra cfg.py: a top-level `extends: other.yaml` is loaded first, then overlaid."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("extends", None)
    if not base:
        return raw
    bp = base if os.path.isabs(base) else os.path.join(os.path.dirname(os.path.abspath(path)), base)
    return _deep_merge(load_yaml(bp), raw)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _git(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if os.path.isdir("/Library/Developer/CommandLineTools"):   # Xcode license blocks git (exit 69)
        env.setdefault("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
    return subprocess.run(["git", *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=30)


class PreregError(RuntimeError):
    pass


def check_prereg(config_path: str, require_committed: bool = False) -> dict:
    """-> {"path", "sha256", "id", "committed"}. Raises PreregError when the gate fails."""
    d = os.path.dirname(os.path.abspath(config_path))
    p = os.path.join(d, "prereg.yaml")
    if not os.path.isfile(p):
        raise PreregError(f"refusing to start: no prereg.yaml next to {config_path}")
    with open(p, encoding="utf-8") as f:
        pr = yaml.safe_load(f) or {}
    if not isinstance(pr, dict):
        raise PreregError(f"{p} is not a mapping")
    missing = [k for k in PREREG_REQUIRED if not str(pr.get(k) or "").strip()]
    if missing:
        raise PreregError(f"refusing to start: {p} lacks {missing}")
    committed = None
    if require_committed:
        tracked = _git(["ls-files", "--error-unmatch", "prereg.yaml"], d)
        clean = _git(["diff", "--quiet", "HEAD", "--", "prereg.yaml"], d)
        committed = tracked.returncode == 0 and clean.returncode == 0
        if not committed:
            raise PreregError(f"refusing to start: {p} is not committed unchanged to git")
    return {"path": p, "sha256": sha256_file(p), "id": str(pr["id"]), "committed": committed}


def append_jsonl(path: str, rec: dict) -> None:
    """Append-only; one line per record, flushed and fsynced."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **rec}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------- checkpoints ----------------------------

def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(out_dir: str, payload: dict, step: int, keep_last: int = 2,
                    tag: str | None = None) -> str:
    """ckpt_<step>.pt (rolling, keep_last kept) or <tag>_<step>.pt (never pruned: stable
    branch points, final). latest.json always points at the newest file."""
    os.makedirs(out_dir, exist_ok=True)
    name = f"{tag or 'ckpt'}_{step:08d}.pt"
    path = os.path.join(out_dir, name)
    _atomic_save(payload, path)
    tmp = os.path.join(out_dir, "latest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"path": name, "step": step}, f)
    os.replace(tmp, os.path.join(out_dir, "latest.json"))
    if tag is None and keep_last > 0:
        for old in sorted(glob.glob(os.path.join(out_dir, "ckpt_*.pt")))[:-keep_last]:
            try:
                os.remove(old)
            except OSError:
                pass
    return path


def latest_checkpoint(out_dir: str) -> str | None:
    p = os.path.join(out_dir, "latest.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        full = os.path.join(out_dir, json.load(f)["path"])
    return full if os.path.exists(full) else None


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
