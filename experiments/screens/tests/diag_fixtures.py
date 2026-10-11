"""Fixtures for the C6 diagnostics runner tests (test_diag_*.py): a tiny chat eval set in eval sets v0's layout, tiny
CPU models at the screens' width (d_model 192, the runner's 5M check) saved as final checkpoints in a fake runs root,
and the runner's CLI. Nothing here reads the PC or a real run."""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(SX))
for _p in (os.path.join(ROOT, "data_prep"), os.path.join(ROOT, "harness"), SX):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import diag_lib as DL  # noqa: E402
import diag_runner as DR  # noqa: E402
import evalwin as EW  # noqa: E402
import testutil as U  # noqa: E402
from model import build_model  # noqa: E402

torch.set_num_threads(2)
TOPICS = ["dog cat horse sheep goat mouse", "red green blue yellow purple orange",
          "apple pear plum cherry grape melon", "hammer saw drill wrench chisel file",
          "Paris Rome Berlin Madrid Vienna Prague", "oak pine birch maple cedar willow"]
SHAPE = dict(vocab_size=8192, d_model=192, n_layers=2, n_heads=3, n_kv_heads=3, head_dim=64, mlp_hidden=64,
             seq_len=512)
FLAGS = {"base": {}, "s004_canonac": {"canon": "AC", "canon_kernel": 4}, "s005_forget": {"forget_gate": True},
         "s007_smear": {"smear_key": True}, "s001_nogate": {"attn_gate": False}}


def evalset(d: str, sha: str = DL.EVALSET_SHA, n_turns: int = 4) -> str:
    """oasst2 only: conversations whose turns repeat the previous turn's words in another order."""
    os.makedirs(d, exist_ok=True)
    docs = []
    for i, topic in enumerate(TOPICS):
        w = topic.split()
        turns = [{"role": "user" if j % 2 == 0 else "assistant",
                  "text": " ".join(w[(j + k) % len(w)] for k in range(len(w)))} for j in range(n_turns)]
        docs.append({"id": f"c{i}", "turns": turns})
    wins = [{**w, "d": i, "k": k, "h": EW.window_hash("oasst2", d0["id"], k)}
            for i, d0 in enumerate(docs) for k, w in enumerate(EW.chat_windows(d0["turns"], 2048, 3072))]
    files = {}
    for k, rows in (("docs", docs), ("windows", wins)):
        p = os.path.join(d, f"oasst2.{k}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        files[k] = {"path": os.path.basename(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    man = {"evalset_sha256": sha, "config": {"chat_context_bytes": 3072, "window_bytes": 2048},
           "sets": {"oasst2": {"kind": "chat", "files": files, "windows": len(wins)}}}
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(man, f)
    return d


def model(kind: str = "base", seed: int = 0, **kw):
    m = U.scramble(build_model(U.tiny(**{**SHAPE, **FLAGS[kind], **kw})), seed=seed)
    return m


def save(root: str, exp: str, run: str, m, step: int = 7630, name: str | None = None) -> str:
    d = os.path.join(root, exp, run)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name or f"final_{step:08d}.pt")
    torch.save({"model": m.state_dict(), "model_cfg": m.cfg.to_dict(), "step": step}, p)
    return p


def kind_of(run: str) -> str:
    return next((k for k in FLAGS if k != "base" and run.startswith(k)), "base")


def root_with(tmp, runs: dict) -> str:
    """runs: {(exp, run): seed}; each gets the model its name implies."""
    root = str(tmp / "runs")
    for (exp, run), seed in runs.items():
        save(root, exp, run, model(kind_of(run), seed))
    return root


SMALL = {("SCREENS", "base_s101"): 1, ("SCREENS", "s004_canonac_g1_s101"): 2, ("SCREENS", "s005_forget_g1_s101"): 3,
         ("SCREENS", "s007_smear_g1_s101"): 4, ("SCREENS", "s001_nogate_g1_s101"): 5, ("E3", "e3_5m_A_s1"): 6}


def cli(root: str, ev: str, *extra: str, only=None) -> int:
    only = only or sorted({r for _, r in SMALL})
    return DR.main(["run", "--runs-root", root, "--evalset", ev, "--only", ",".join(only), *extra])


def outputs(root: str) -> dict:
    """{(exp, run, file): parsed JSON} of every c6 file under root."""
    out = {}
    for exp in sorted(os.listdir(root)):
        for run in sorted(os.listdir(os.path.join(root, exp))):
            c6 = os.path.join(root, exp, run, "c6")
            for f in sorted(os.listdir(c6)) if os.path.isdir(c6) else []:
                if f.endswith(".json"):
                    out[(exp, run, f)] = json.load(open(os.path.join(c6, f)))
    return out


def load(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    from config import PlanckConfig
    m = build_model(PlanckConfig.from_dict(ck["model_cfg"]), "cpu")
    m.load_state_dict(ck["model"])
    return m.eval()
