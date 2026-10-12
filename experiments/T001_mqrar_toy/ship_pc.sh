#!/bin/zsh
# ship_pc.sh: package experiments/T001_mqrar_toy for the PC (PREREG.txt GPU PLAN) and, with --launch, start the queue.
#   experiments/T001_mqrar_toy/ship_pc.sh            # build; copy t001_code.tgz, its sha256 and run_t001.sh to planck-kit
#   experiments/T001_mqrar_toy/ship_pc.sh --launch   # also start run_t001.sh in WSL, detached through start_detached.ps1
# Uses pc/local.env (host and key, never committed) the way send_kit.sh does, and never writes planck.bundle.
set -eu
HERE=${0:A:h}; ROOT=${HERE:h:h}
set -a; . "$ROOT/pc/local.env"; set +a
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
(cd "$HERE" && COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$TMP/t001_code.tgz" --exclude __pycache__ \
    PREREG.txt task.py layers.py model.py train.py metrics.py runplan.py tests)
(cd "$TMP" && shasum -a 256 t001_code.tgz > t001_code.sha256)
cp "$HERE/run_pc.sh" "$TMP/run_t001.sh"
for f in t001_code.tgz t001_code.sha256 run_t001.sh; do
    scp -q -i "$PLANCK_PC_KEY" -o IdentitiesOnly=yes -o BatchMode=yes "$TMP/$f" "$PLANCK_PC_HOST":planck-kit/$f
done
echo "shipped $(cut -c1-12 "$TMP/t001_code.sha256")"
if [ "${1:-}" = "--launch" ]; then
    ssh -i "$PLANCK_PC_KEY" -o IdentitiesOnly=yes -o BatchMode=yes "$PLANCK_PC_HOST" \
        "powershell -NoProfile -ExecutionPolicy Bypass -File %USERPROFILE%\\planck-kit\\start_detached.ps1 -Command \"wsl.exe -d Ubuntu --exec /bin/bash /mnt/c/Users/%USERNAME%/planck-kit/run_t001.sh\"" \
        2>&1 | tr -d '\r' | grep -i "started pid"
fi
