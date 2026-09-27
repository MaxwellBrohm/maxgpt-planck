#!/bin/bash
# boot_resume.sh: after a reboot or power cut, restart the Planck queues that were cut off mid-run (pc/RUNBOOK.txt
# "RESUME AFTER A REBOOT"). Installed as ~/planck/boot.d/boot_resume.sh; runner_start.sh starts it detached at every
# start of the PlanckRunner task, when that file exists. Its output goes to ~/planck/logs/boot_resume.log.
#   bash boot_resume.sh             boot mode: once per WSL boot. Waits BOOT_DELAY_S (120) for the GPU driver, then
#                                   up to BOOT_GPU_WAIT_S (600) until nvidia-smi reads a GPU temperature, then runs
#                                   every executable NN_<name>.sh in boot.d, in name order
#   bash boot_resume.sh --dry-run   no wait, starts nothing: each script says what it would do and why
#   bash boot_resume.sh --now       no wait, no once-per-boot check: a manual resume (e.g. after rm ~/planck/PAUSE).
#                                   From the Mac start it DETACHED (pcdetach.sh on a wrapper), never over plain SSH.
# Each NN_<name>.sh resumes one queue through that queue's own restart path, only when the queue's own log shows it
# was cut off mid-run and nothing of it runs (bootlib.sh). Stop files: boot.d/STOP (nothing resumes),
# boot.d/<name>.STOP (that queue does not), ~/planck/STOP and ~/planck/PAUSE (nothing resumes). To take a queue out
# for good, delete its NN_<name>.sh from ~/planck/boot.d (or chmod -x it).
set -u
P=${PLANCK_HOME:-$HOME/planck}
D=$(cd "$(dirname "$0")" && pwd)
SMI=${PLANCK_NVIDIA_SMI:-nvidia-smi}
BOOT_ID_FILE=${BOOT_ID_FILE:-/proc/sys/kernel/random/boot_id}
MODE=boot
case ${1:-} in
    --dry-run) MODE=dry ;;
    --now) MODE=now ;;
    "") ;;
    *) echo "usage: boot_resume.sh [--dry-run | --now]"; exit 2 ;;
esac
export PATH="/usr/lib/wsl/lib:$PATH" PLANCK_HOME="$P" BOOT_D="$D"
say() { echo "$(date '+%F %T') boot_resume[$MODE]: $*"; }
say "start (pid $$, WSL up $(cut -d' ' -f1 /proc/uptime 2>/dev/null || echo '?') s, boot.d $D)"
mkdir -p "$P/locks" "$P/status"
if [ $MODE != dry ] && command -v flock > /dev/null; then
    exec 7> "$P/locks/boot_resume.lock"
    flock -n 7 || { say "another boot_resume holds locks/boot_resume.lock: exiting"; exit 0; }
fi
if [ $MODE = boot ]; then
    bid=$(cat "$BOOT_ID_FILE" 2>/dev/null)
    if [ -n "$bid" ] && [ "$(cat "$P/status/boot_resume.boot_id" 2>/dev/null)" = "$bid" ]; then
        say "already ran in this WSL boot ($bid): exiting"; exit 0
    fi
    [ -n "$bid" ] && echo "$bid" > "$P/status/boot_resume.boot_id"
    say "waiting ${BOOT_DELAY_S:-120} s for the GPU driver"
    sleep "${BOOT_DELAY_S:-120}"
fi

gpu_temp() { "$SMI" --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }
t=$(gpu_temp)
waited=0
if [ $MODE != dry ]; then
    while [ -z "$t" ] && [ $waited -lt "${BOOT_GPU_WAIT_S:-600}" ]; do
        sleep 10; waited=$((waited + 10)); t=$(gpu_temp)
    done
fi
if [ -z "$t" ]; then
    say "nvidia-smi reads no GPU temperature (waited $waited s): nothing resumes; run boot_resume.sh --now once it does"
    [ $MODE = dry ] || exit 1
else
    say "GPU visible, $t C (waited $waited s)"
fi

n=0
for s in "$D"/[0-9][0-9]_*.sh; do
    [ -f "$s" ] || continue
    if [ ! -x "$s" ]; then say "$(basename "$s") is not executable: skipped"; continue; fi
    n=$((n + 1))
    if [ $MODE = dry ]; then bash "$s" --dry-run; else bash "$s"; fi
    rc=$?
    [ $rc = 0 ] || say "$(basename "$s") exit $rc"
done
say "end: $n queue script(s) run"
