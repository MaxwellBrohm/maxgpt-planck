"""Device, precision and environment logging.

PLAN.md: bf16 autocast with fp32 master weights on the Mac (MPS) and the 5070 (CUDA);
CPU runs fp32 (tests, smoke). "auto" picks cuda, then mps, then cpu. Adapted from Max's
MaxGPT-Ultra train/trainer.py (amp_dtype, cuda_has_native_bf16). fp16 with loss scaling
(the Titans) is not wired yet.

MPS guard env vars (PLAN.md Guards) must be set before torch initializes MPS, so
mps_env_defaults() is called by train.py before `import torch`.
"""
from __future__ import annotations

import os
import platform
import sys

MPS_ENV = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.7", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6",
           "PYTORCH_ENABLE_MPS_FALLBACK": "0"}


def mps_env_defaults() -> None:
    for k, v in MPS_ENV.items():
        os.environ.setdefault(k, v)


def pick_device(pref: str = "auto") -> str:
    import torch
    if pref != "auto":
        if pref == "cuda":
            assert torch.cuda.is_available(), "device cuda requested but not available"
        if pref == "mps":
            assert torch.backends.mps.is_available(), "device mps requested but not available"
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_precision(pref: str, device: str) -> str:
    """-> 'bf16' or 'fp32'. auto: bf16 on cuda (native bf16 only) and mps, fp32 on cpu."""
    import torch
    assert pref in ("auto", "bf16", "fp32"), pref
    if pref == "auto":
        if device == "cuda":
            return "bf16" if torch.cuda.is_bf16_supported() else "fp32"
        return "bf16" if device == "mps" else "fp32"
    return pref


def amp_factory(precision: str, device: str):
    """-> None (plain fp32) or a zero-arg callable returning a fresh autocast context."""
    import torch
    if precision != "bf16":
        return None
    return lambda: torch.autocast(device_type=device, dtype=torch.bfloat16)


def env_info(device: str) -> dict:
    """Versions for runs.jsonl (PLAN.md: every run logs torch, CUDA or Metal, driver)."""
    import numpy
    import torch
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": numpy.__version__, "platform": platform.platform(), "device": device,
            "host": platform.node()}
    if device == "cuda":
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
    if device == "mps":
        info["macos"] = platform.mac_ver()[0]
    return info


def sync(device: str) -> None:
    import torch
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

