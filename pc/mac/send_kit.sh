#!/usr/bin/env bash
# send_kit.sh: copy the Planck repo to the PC from the Mac (RUNBOOK step 2 and after changes).
#
#   bash pc/mac/send_kit.sh            # git bundle of COMMITTED history (what the runner needs)
#   bash pc/mac/send_kit.sh --tar      # plain tar of harness/ and pc/ (a first look only)
#
# Why a bundle: the runner only starts a job whose prereg.yaml is committed, which it checks
# with git on the PC. A bundle carries the commits with no GitHub credentials on the PC.
# Uncommitted changes are NOT in a bundle: commit first (Max's call). On the PC, in WSL:
#   first time:  git clone /mnt/c/Users/<winuser>/planck.bundle ~/planck/repo
#   updates:     git -C ~/planck/repo pull /mnt/c/Users/<winuser>/planck.bundle main
# The tar form: mkdir -p ~/planck/repo && tar -xzf /mnt/c/Users/<winuser>/planck-kit.tgz -C ~/planck/repo
#
# Needs the PC on the tailnet (Max clicks Connect in the Mac's Tailscale menu). Host and key
# default to the values in Max's notes; override with PLANCK_PC_HOST / PLANCK_PC_KEY.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HOST="${PLANCK_PC_HOST:?set PLANCK_PC_HOST in pc/local.env}"
KEY="${PLANCK_PC_KEY:?set PLANCK_PC_KEY in pc/local.env}"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15)
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Library/Developer/CommandLineTools}"   # Xcode license gate
OUT_DIR="$(mktemp -d)"
trap 'rm -rf "$OUT_DIR"' EXIT

if [ "${1:-}" = "--tar" ]; then
    FILE="$OUT_DIR/planck-kit.tgz"
    COPYFILE_DISABLE=1 tar -czf "$FILE" -C "$ROOT" --exclude '__pycache__' --exclude '.pytest_cache' \
        --exclude '._*' --exclude '*.pt' --exclude 'bench_results.jsonl' harness pc
    DEST="planck-kit/planck-kit.tgz"   # relative to the Windows home folder
else
    if [ -n "$(git -C "$ROOT" status --porcelain -- harness pc)" ]; then
        echo "harness/ or pc/ has uncommitted changes; they would not reach the PC:"
        git -C "$ROOT" status --short -- harness pc | head -20
        echo "commit them first, or use --tar for a look without the prereg gate"
        exit 1
    fi
    FILE="$OUT_DIR/planck.bundle"
    git -C "$ROOT" bundle create "$FILE" HEAD main
    git bundle verify "$FILE" >/dev/null
    DEST="planck-kit/planck.bundle"   # relative to the Windows home folder
fi
ls -l "$FILE"
scp "${SSH_OPTS[@]}" "$FILE" "$HOST:$DEST"
echo "copied to $DEST in the Windows home folder (in WSL: /mnt/c/Users/<winuser>/$DEST, where <winuser> is the output of: cmd.exe /c echo %USERNAME%)"
