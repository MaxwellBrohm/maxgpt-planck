"""verify_cuda: is this PC ready to train Planck? Run once after setup (RUNBOOK step 8).

  ~/planck/venv/bin/python ~/planck/repo/pc/verify_cuda.py            # on the PC, in WSL
  python pc/verify_cuda.py --device cpu                                # smoke test on the Mac

Checks, in order, and exits 1 at the first failure:
  1. torch sees the GPU, and the wheel carries kernels for it (sm_120 for the RTX 5070;
     a wheel without them fails later with "no kernel image is available")
  2. bf16 matmul, SDPA causal and SDPA with a bool mask, forward and backward
  3. the harness causal and document leak self-tests (selftest.run_selftests) on a
     random-initialized model under bf16 autocast, the same gate train.py runs at startup
  4. nvidia-smi as the runner will call it: temperature and memory must be readable,
     or the guard would refuse to start jobs (then set PLANCK_NVIDIA_SMI, see gpuguard.py)
Prints one JSON line with everything it saw. Nothing is downloaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "harness"))
sys.path.insert(0, os.path.join(HERE, "wsl"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from config import PlanckConfig  # noqa: E402
from device import amp_factory, env_info, resolve_precision  # noqa: E402
from model import build_model  # noqa: E402
from selftest import run_selftests  # noqa: E402


def fail(report: dict, msg: str) -> int:
    report["FAILED"] = msg
    print(json.dumps(report, default=str))
    print(f"verify_cuda: FAILED: {msg}", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-smi", action="store_true", help="skip the nvidia-smi check")
    a = ap.parse_args(argv)
    dev = a.device
    rep: dict = {"env": env_info(dev) if dev != "cuda" or torch.cuda.is_available() else {}}
    if dev == "cuda":
        if not torch.cuda.is_available():
            return fail(rep, "torch.cuda.is_available() is False (driver, WSL or a CPU wheel?)")
        cap = torch.cuda.get_device_capability(0)
        arch = torch.cuda.get_arch_list()
        rep.update(capability=list(cap), arch_list=arch, torch_cuda=torch.version.cuda,
                   bf16=torch.cuda.is_bf16_supported())
        if f"sm_{cap[0]}{cap[1]}" not in arch:
            return fail(rep, f"this torch wheel has no sm_{cap[0]}{cap[1]} kernels: {arch}")
    precision = resolve_precision("auto", dev)
    amp = amp_factory(precision, dev)
    rep["precision"] = precision

    # 2. kernels
    try:
        x = torch.randn(4, 4, 256, 64, device=dev, requires_grad=True)
        with (amp() if amp else nullcontext()):
            y1 = F.scaled_dot_product_attention(x, x, x, is_causal=True)
            m = torch.ones(256, 256, dtype=torch.bool, device=dev).tril()
            y2 = F.scaled_dot_product_attention(x, x, x, attn_mask=m)
            z = (x.flatten(0, 2) @ x.flatten(0, 2).T).sum()
        (y1.float().sum() + y2.float().sum() + z.float()).backward()
        diff = float((y1.detach().float() - y2.detach().float()).abs().max())
        rep["sdpa_causal_vs_mask_maxdiff"] = diff
        if not diff < 5e-2:
            return fail(rep, f"SDPA causal and bool-mask paths disagree by {diff}")
    except RuntimeError as e:
        return fail(rep, f"kernel check raised: {str(e).splitlines()[0]}")

    # 3. harness leak self-tests on a small random model (about 0.3M parameters)
    torch.manual_seed(0)
    cfg = PlanckConfig(vocab_size=512, d_model=64, n_layers=3, n_heads=2, n_kv_heads=1,
                       head_dim=32, mlp_hidden=160, seq_len=128)
    model = build_model(cfg, "cpu").to(dev)
    try:
        rep["selftest"] = run_selftests(model, dev, amp, packed=True)
    except AssertionError as e:
        return fail(rep, str(e))

    # 4. nvidia-smi as the runner sees it
    if dev == "cuda" and not a.skip_smi:
        import gpuguard
        s = gpuguard.query(with_throttle=True)
        rep["nvidia_smi"] = {"path": gpuguard.smi_path(), "sample": s}
        if s is None or s.get("temp_c") is None or s.get("mem_used_mib") is None:
            return fail(rep, "nvidia-smi gave no temperature or memory; try "
                             "PLANCK_NVIDIA_SMI=/mnt/c/Windows/System32/nvidia-smi.exe")
    rep["ok"] = True
    print(json.dumps(rep, default=str))
    print("verify_cuda: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
