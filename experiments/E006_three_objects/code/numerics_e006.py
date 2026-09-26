"""E006 NUMERICS settings in one place (notes.txt SEEDS AND NUMERICS). Every CUDA job calls apply() before it loads
a model; record() goes into run.json.
  fp32 weights and optimizer (the entry points load fp32); no TF32: torch.set_float32_matmul_precision("highest"),
  allow_tf32 off for matmul and cuDNN; cudnn.benchmark off, cudnn.deterministic on;
  torch.use_deterministic_algorithms(True, warn_only=True); CUBLAS_WORKSPACE_CONFIG=:4096:8;
  attention: the library default implementation (SDPA, as E005) with ONLY the math SDPA kernel enabled (flash,
  memory-efficient and cuDNN SDPA off: their backward passes are not deterministic; critique SHOULD 3);
  HF_HUB_OFFLINE=1.
  --tf32 (the numerics twins C1t, C2t only, Q-device's noise floor): TF32 on for matmul and cuDNN ("high").
As a runner it applies the settings and then runs another script UNCHANGED (runpy), for the e005_ft_test.py dry run:
  python -B numerics_e006.py [--tf32] e005_ft_test.py <args...>"""
import os
import runpy
import sys


def apply(tf32=False):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return record()


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception as e:  # recorded, never fatal here (the queue's identity step checks the pins)
        return f"unavailable: {type(e).__name__}"


def record():
    import torch
    b = torch.backends
    out = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
           "tf32_matmul": b.cuda.matmul.allow_tf32, "tf32_cudnn": b.cudnn.allow_tf32,
           "cudnn_benchmark": b.cudnn.benchmark, "cudnn_deterministic": b.cudnn.deterministic,
           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
           "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
           "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
           "sdpa": {"flash": b.cuda.flash_sdp_enabled(), "mem_efficient": b.cuda.mem_efficient_sdp_enabled(),
                    "math": b.cuda.math_sdp_enabled(),
                    "cudnn": b.cuda.cudnn_sdp_enabled() if hasattr(b.cuda, "cudnn_sdp_enabled") else None},
           "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), "hf_home_set": bool(os.environ.get("HF_HOME")),
           "versions": {m: _version(m) for m in ("torch", "transformers", "tokenizers", "safetensors")},
           "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["driver"] = _driver()
    return out


def _driver():
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=20)
        return r.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


PINS = {"torch": "2.13.0", "transformers": "5.15.1", "tokenizers": "0.22.2", "safetensors": "0.8.0"}


def pin_problems(rec):
    """-> list of version pins the environment misses (torch compared without its +cuXXX build tag)."""
    v = rec["versions"]
    return [f"{m} {v.get(m)} != {want}" for m, want in PINS.items() if str(v.get(m)).split("+")[0] != want]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    tf32 = False
    if argv and argv[0] == "--tf32":
        tf32, argv = True, argv[1:]
    if not argv:
        raise SystemExit("usage: numerics_e006.py [--tf32] <script.py> [args...]")
    rec = apply(tf32)
    print(f"numerics_e006: {rec}", flush=True)
    script = argv[0]
    sys.argv = argv
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
