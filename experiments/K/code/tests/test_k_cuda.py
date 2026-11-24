"""Track K's CUDA paths (K3 round 16, 2026-11-22). The cuda cases are skipped without CUDA: the Mac skips them and
the PC runs them before K0's first run (code/notes.txt K3 ROUND 16, PC PLAN step 2). The cpu cases run the same code
in fp32 on the Mac, so the kit's own plumbing has run before the PC sees it (they repeat what test_ktrain and
test_kroute already pin on the CPU; they are not new CPU evidence).

Each training case runs ktrain.py in its own process with deterministic algorithms (CUBLAS_WORKSPACE_CONFIG
:4096:8, as harness default_parity.run_tree runs train.py), on the screens' engine where CUDA is there (bf16, varlen
doc_attn), a tiny model with K's vocabulary, 24 steps, a log record every step and a final checkpoint:
  - the K-PROBE hook (k.probe every 5) trains bitwise as k: {} (weights, optimizer, loader and RNG state, every log
    record but wall clock), with a reading at 5, 10, 15, 20 and 24 and none failed: SPEC 14's hook rule on the
    device the runs use (K0-smoke's hook check, DEVIATION K-D1, reads the real config where determinism is off);
  - k.wd_mult {mlp: 1.0} trains bitwise as k: {} in weights and log (K4b's split at x1 changes nothing);
  - k.grad_route (K3-R) trains on the device: finite losses, the first loss equal to the plain run's (the forward
    is the plain SwiGLU's), the weights apart after 24 steps.
And the routed backward under bf16 autocast (kroute._RoutedDown): FACT-only positions leave every other unit's
gradient exactly 0 and non-FACT positions the reserved block's, and its gradients agree with test_kroute's
independent masked-autograd reference computed in fp32 on the CPU (relative error per parameter under 5%
in bf16; a routing error moves whole units' gradients, far more than bf16 rounding).
"""
from __future__ import annotations

import copy
import math
import os
import subprocess
import sys

os.environ.setdefault("PLANCK_TEST_THREADS", os.environ.get("K_TEST_THREADS", "4"))

import pytest                                          # noqa: E402
import torch                                           # noqa: E402

import gen                                             # noqa: E402
import kcommon as K                                    # noqa: E402
import kroute                                          # noqa: E402
import runio                                           # noqa: E402
from config import PlanckConfig                        # noqa: E402
from default_parity import same                        # noqa: E402
from make_fake_data import make                        # noqa: E402
from model import build_model                          # noqa: E402
from test_kroute import CFG as RCFG, loaders, ref_grads  # noqa: E402
from testutil import read_jsonl, write_run             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only: the PC runs these")
DEVICES = [pytest.param("cuda", marks=CUDA), "cpu"]
MODEL = {"vocab_size": 8192, "d_model": 64, "n_layers": 2, "n_heads": 2, "n_kv_heads": 1, "head_dim": 32,
         "mlp_hidden": 128, "seq_len": 256}
ENGINE = {"cuda": {"device": "cuda", "precision": "bf16", "doc_attn": "varlen", "micro_batch": 8},
          "cpu": {"device": "cpu", "precision": "fp32", "micro_batch": 4}}
RUN = "import sys, torch; torch.use_deterministic_algorithms(True); import ktrain; sys.exit(ktrain.main(sys.argv[1:]))"
STEPS = 24
_RUNS: dict = {}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("kcuda")
    make(str(root / "data"), vocab=8192, docs=300, chats=300)
    assert gen.main(["evals", str(root / "ev"), "--small"]) == 0
    pf = str(root / "ev" / "evals" / "k_probe.jsonl")
    return {"root": root, "probe": {"every": 5, "files": [{"file": pf, "sha256": K.sha256_file(pf)}]}}


def train_one(world, device: str, tag: str, k: dict) -> dict:
    """ktrain.py on a tiny K-vocabulary model in a fresh deterministic process; cached per (device, tag)."""
    if (device, tag) in _RUNS:
        return _RUNS[device, tag]
    root = world["root"]
    src = [{"name": "text", "kind": "tokens", "paths": [f"{root}/data/text_*.bin"], "eot_id": 1, "share": 0.7},
           {"name": "chat", "kind": "chat", "paths": [f"{root}/data/chat_*.jsonl"], "share": 0.3}]
    d = str(root / device / tag / "run")              # same run name in every pair: the configs differ only in k
    cfg = write_run(d, str(root / "data"), runs_jsonl="runs.jsonl", model=MODEL, data={"sources": src},
                    train={**ENGINE[device], "grad_accum": 2, "total_steps": STEPS, "log_every": 1,
                           "ckpt_every": 0}, k=k)
    env = {n: v for n, v in os.environ.items() if not n.startswith("GIT_")}
    env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-c", RUN, cfg], cwd=CODE, env=env, capture_output=True, text=True,
                       timeout=1800)
    assert p.returncode == 0, (tag, p.stdout[-2000:], p.stderr[-4000:])
    out = os.path.join(d, "out")
    ck = runio.load_checkpoint(runio.latest_checkpoint(out))
    keys = [x for x in ("model", "optimizer", "data_state", "rng_torch", "rng_cuda", "step") if x in ck]
    _RUNS[device, tag] = {"ck": {x: ck[x] for x in keys}, "out": out,
                          "log": [{n: v for n, v in r.items() if n not in ("tok_per_s", "time")}
                                  for r in read_jsonl(os.path.join(out, "log.jsonl"))]}
    return _RUNS[device, tag]


@pytest.mark.parametrize("device", DEVICES)
def test_the_probe_hook_trains_bitwise_as_none(world, device):
    on, off = train_one(world, device, "probe", {"probe": world["probe"]}), train_one(world, device, "none", {})
    assert on["ck"]["step"] == STEPS and len(on["log"]) == STEPS
    assert same(on["ck"], off["ck"]) == [] and same(on["log"], off["log"]) == []
    pr = read_jsonl(os.path.join(on["out"], "k_probe.jsonl"))
    assert [r["step"] for r in pr] == [5, 10, 15, 20, 24] and not any("error" in r for r in pr), pr[:1]
    assert not os.path.exists(os.path.join(off["out"], "k_probe.jsonl"))


@pytest.mark.parametrize("device", DEVICES)
def test_wd_mult_one_trains_bitwise_as_none(world, device):
    a, b = train_one(world, device, "wd1", {"wd_mult": {"mlp": 1.0}}), train_one(world, device, "none", {})
    assert same(a["ck"]["model"], b["ck"]["model"]) == [] and same(a["log"], b["log"]) == []


@pytest.mark.parametrize("device", DEVICES)
def test_grad_route_trains_and_starts_like_the_plain_run(world, device):
    r = train_one(world, device, "route", {"grad_route": {"sources": ["chat"], "reserved": 32}})
    n = train_one(world, device, "none", {})
    assert r["ck"]["step"] == STEPS and all(math.isfinite(x["loss"]) for x in r["log"])
    assert r["log"][0]["step"] == 1 and r["log"][0]["loss"] == n["log"][0]["loss"]
    assert same(r["ck"]["model"], n["ck"]["model"]) != []
    st = read_jsonl(os.path.join(r["out"], "k_start.jsonl"))
    assert st[-1]["installed"]["grad_route"]["reserved"] == {"0": [96, 127], "1": [96, 127]}


@pytest.mark.parametrize("device", DEVICES)
def test_routed_backward_under_autocast(device, tmp_path):
    dt = torch.bfloat16 if device == "cuda" else torch.float32
    torch.manual_seed(1)
    model = build_model(PlanckConfig(**RCFG))
    ref_model = copy.deepcopy(model)
    ctx = kroute.RouteCtx()
    kroute.install(model, 8, ctx)
    model.to(device)
    g = torch.Generator().manual_seed(2)
    idx = torch.randint(8, 256, (2, 64), generator=g).to(device)
    tgt = torch.randint(8, 256, (2, 64), generator=g).to(device)
    for fact in (True, False):                       # one kind of position only: the other block gets exactly 0
        model.zero_grad(set_to_none=True)
        ctx.mask = torch.full((2, 64), fact, device=device)
        with torch.autocast(device, dtype=dt, enabled=device == "cuda"):
            loss = model(idx, tgt, reduction="sum")[1]
        loss.backward()
        ctx.mask = None
        for blk in model.blocks:
            mlp = blk.mlp
            on, off = (slice(16, 24), slice(0, 16)) if fact else (slice(0, 16), slice(16, 24))
            for w in (mlp.gate_proj.weight, mlp.up_proj.weight):
                assert torch.count_nonzero(w.grad[off]) == 0 and torch.count_nonzero(w.grad[on]) > 0
            assert torch.count_nonzero(mlp.down_proj.weight.grad[:, off]) == 0
            assert torch.count_nonzero(mlp.down_proj.weight.grad[:, on]) > 0
    make(str(tmp_path / "fake"), vocab=256, docs=200, chats=150, doc_len=(5, 60))
    _, ld = loaders(str(tmp_path), seed=5)
    b1, b2 = ld.next_batch(), ld.next_batch()
    rf = kroute.RoutedForward(model, ld, ctx, [1])
    model.zero_grad(set_to_none=True)
    for b in (b1, b2):
        on_dev = {x: b[x].to(device) for x in ("idx", "tgt", "doc", "pos")}
        with torch.autocast(device, dtype=dt, enabled=device == "cuda"):
            _, loss = rf(on_dev["idx"], on_dev["tgt"], on_dev["doc"], on_dev["pos"], reduction="sum")
        loss.backward()
    ref = ref_grads(ref_model, (b1, b2), 8)          # fp32 on the CPU
    for name, p in model.named_parameters():
        gd, gr = p.grad.float().cpu(), ref[name]
        rel = float((gd - gr).norm() / gr.norm().clamp_min(1e-12))
        assert rel < (5e-2 if device == "cuda" else 1e-5), (name, rel)
