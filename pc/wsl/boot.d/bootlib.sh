# bootlib.sh: shared by the boot.d queue scripts (sourced, bash; RUNBOOK "RESUME AFTER A REBOOT").
# boot_resume.sh runs only files named NN_<name>.sh, so neither this file nor boot_resume.sh runs as a queue script.
# A queue script sets NAME, sources this file with its own arguments ("" or --dry-run), then in order:
#   blocked [its queue's own STOP files]  -> exit 0   (a STOP or PAUSE file holds it)
#   running ERE                           -> exit 0   (something of that queue runs: never start a second copy)
#   its own reading of the queue's log    -> exit 0 unless the last run was cut off mid-run
#   launch ERE CMD...                     start CMD detached, then check that ERE runs
# Every line it prints starts with the time and NAME; boot_resume.sh sends them to ~/planck/logs/boot_resume.log.
P=${PLANCK_HOME:-$HOME/planck}
BOOTD=${BOOT_D:-$P/boot.d}
VERIFY_S=${BOOT_VERIFY_S:-120}      # how long launch waits for the queue's process to appear
DRY=0
case ${1:-} in
    --dry-run) DRY=1 ;;
    "") ;;
    *) echo "usage: $0 [--dry-run]"; exit 2 ;;
esac

say() { echo "$(date '+%F %T') $NAME: $*"; }

blocked() {   # blocked [FILE...]: 0 (and says which) when a stop file holds this queue
    local f
    for f in "$BOOTD/STOP" "$BOOTD/$NAME.STOP" "$P/STOP" "$P/PAUSE" "$@"; do
        if [ -e "$f" ]; then say "$f present: not resuming"; return 0; fi
    done
    return 1
}

running() {   # running ERE: 0 when some process's full command line matches ERE (pgrep -f); sets PIDS
    PIDS=$(pgrep -f "$1" 2>/dev/null | grep -vx "$$" | tr '\n' ' ')
    [ -n "$PIDS" ]
}

last_start() {   # last_start FILE ERE: line number of the last line matching ERE (empty if none)
    grep -anE "$2" "$1" 2>/dev/null | tail -1 | cut -d: -f1
}

ended_after() {  # ended_after FILE LINE ERE: the last line at or after LINE that matches ERE (empty if none)
    tail -n +"$2" "$1" | grep -aE "$3" | tail -1
}

uptime_s() { cut -d' ' -f1 /proc/uptime 2>/dev/null || echo "?"; }

launch() {    # launch ERE CMD...: start CMD in its own session, output to LAUNCH_LOG (default /dev/null), then
              # wait up to VERIFY_S s for ERE to run. 1 = it did not appear (read the queue's own log).
              # fd 7 (boot_resume.sh's lock) is closed for the child, so a long queue never holds that lock.
    local pat=$1 i=0
    shift
    if [ $DRY = 1 ]; then say "DRY RUN, would start: $*"; return 0; fi
    say "starting: $*"
    if command -v setsid > /dev/null; then
        setsid nohup "$@" >> "${LAUNCH_LOG:-/dev/null}" 2>&1 < /dev/null 7>&- &
    else
        nohup "$@" >> "${LAUNCH_LOG:-/dev/null}" 2>&1 < /dev/null 7>&- &
    fi
    while [ $i -lt "$VERIFY_S" ]; do
        sleep 1
        i=$((i + 1))
        if running "$pat"; then say "running after $i s (pids $PIDS)"; return 0; fi
    done
    say "NOT running $VERIFY_S s after the start: read the queue's own log"
    return 1
}
