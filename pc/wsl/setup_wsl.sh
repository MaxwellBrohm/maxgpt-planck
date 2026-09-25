#!/usr/bin/env bash
# setup_wsl.sh: Planck inside WSL2 Ubuntu (RUNBOOK step 7). Two parts, so nobody ever types
# a password over SSH:
#   7a, packages, as root (wsl.exe -u root needs no password):
#       wsl -d Ubuntu-24.04 -u root -- bash /home/<user>/planck/repo/pc/wsl/setup_wsl.sh --apt
#   7b, everything else, as Max's normal Linux user:
#       wsl -d Ubuntu-24.04 -- bash -lc "bash ~/planck/repo/pc/wsl/setup_wsl.sh"
#       TORCH_VERSION=2.13.0 SKIP_VLLM=1 ...   (optional overrides)
#
# Makes ~/planck/{queue/*,logs,status,runs}, a training venv (~/planck/venv: torch cu128 at
# the SAME version as the Mac harness venv, plus numpy, pyyaml, tokenizers, pytest) and a
# SEPARATE vLLM venv (~/planck/venv-vllm), because vLLM pins its own torch and would
# otherwise move the training torch. Every step is safe to re-run.
# Not installed on purpose: any NVIDIA driver or CUDA toolkit. In WSL the Windows driver
# provides libcuda (/usr/lib/wsl/lib) and the pip wheels bundle the CUDA 12.8 runtime.
set -euo pipefail

PLANCK_HOME="${PLANCK_HOME:-$HOME/planck}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"          # harness/notes.txt: the Mac runs 2.13.0
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"   # CUDA 12.8: first with sm_120
VLLM_SPEC="${VLLM_SPEC:-vllm}"                    # current release; pin later, e.g. vllm==X.Y.Z
SKIP_VLLM="${SKIP_VLLM:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-150}"

step() { printf '\n=== %s\n' "$*"; }

APT_PKGS="python3-venv python3-dev python3-pip build-essential git jq tmux htop"
if [ "${1:-}" = "--apt" ]; then
    if [ "$(id -u)" -ne 0 ]; then echo "--apt runs as root: wsl -d Ubuntu-24.04 -u root -- bash $0 --apt"; exit 1; fi
    step "7a. apt packages (root)"
    apt-get update
    # shellcheck disable=SC2086
    DEBIAN_FRONTEND=noninteractive apt-get install -y $APT_PKGS
    exit 0
fi

step "0. checks"
if [ "$(id -u)" -eq 0 ]; then echo "run as your normal Linux user, not root"; exit 1; fi
for c in git gcc; do
    command -v "$c" >/dev/null || { echo "missing $c: run step 7a (--apt as root) first"; exit 1; }
done
python3 -c "import ensurepip, venv" 2>/dev/null || { echo "python3-venv missing: run step 7a first"; exit 1; }
if ! grep -qi microsoft /proc/version; then echo "this is not WSL"; exit 1; fi
if ! command -v nvidia-smi >/dev/null && [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
    export PATH="/usr/lib/wsl/lib:$PATH"
fi
if ! nvidia-smi -L; then
    echo "no GPU visible in WSL. Update the WINDOWS NVIDIA driver (never install one in WSL),"
    echo "then from Windows: wsl --shutdown, and try again."
    exit 1
fi
free -g
df -h "$HOME"
avail_gb=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc '0-9')
if [ "$avail_gb" -lt "$MIN_FREE_GB" ]; then
    echo "WARNING: ${avail_gb} GB free in WSL; PLAN.md wants about ${MIN_FREE_GB} GB (RUNBOOK step 1)"
fi

step "1. folders under $PLANCK_HOME"
mkdir -p "$PLANCK_HOME"/queue/{pending,running,done,failed} "$PLANCK_HOME"/{logs,status,runs}

step "2. training venv: torch $TORCH_VERSION from $TORCH_INDEX"
[ -x "$PLANCK_HOME/venv/bin/python" ] || python3 -m venv "$PLANCK_HOME/venv"
"$PLANCK_HOME/venv/bin/pip" install -q -U pip
"$PLANCK_HOME/venv/bin/pip" install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
"$PLANCK_HOME/venv/bin/pip" install numpy pyyaml tokenizers pytest
"$PLANCK_HOME/venv/bin/python" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("gpu", torch.cuda.get_device_name(0), "capability", cap, "arch list", torch.cuda.get_arch_list())
    assert f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list(), "wheel lacks kernels for this GPU"
EOF

if [ "$SKIP_VLLM" = "1" ]; then
    step "3. vLLM venv: skipped (SKIP_VLLM=1)"
else
    step "3. vLLM venv: $VLLM_SPEC (its own torch, chosen by uv for this driver)"
    [ -x "$PLANCK_HOME/venv-vllm/bin/python" ] || python3 -m venv "$PLANCK_HOME/venv-vllm"
    "$PLANCK_HOME/venv-vllm/bin/pip" install -q -U pip uv
    "$PLANCK_HOME/venv-vllm/bin/uv" pip install --python "$PLANCK_HOME/venv-vllm/bin/python" \
        "$VLLM_SPEC" --torch-backend=auto
    "$PLANCK_HOME/venv-vllm/bin/python" -c \
        "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda ok', torch.cuda.is_available())"
fi

step "4. versions (goes in the first runs.jsonl records anyway)"
"$PLANCK_HOME/venv/bin/pip" freeze | grep -iE '^(torch|numpy|pyyaml|tokenizers)==' || true
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo
echo "done. Next: RUNBOOK step 8 (verify_cuda.py, then bench_micro.py)."
