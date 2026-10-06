#!/bin/bash
# boot.d/40_screens.sh: after a reboot, resume the SCREENS queue (experiments/screens/queue_screens.sh) on the plan and
# the code clone its cut-off run used (pc/RUNBOOK.txt RESUME AFTER A REBOOT; boot.d/40_screens.notes.txt).
#   bash 40_screens.sh [--dry-run]
# Resumes only when all hold: no stop file (bootlib.sh: ~/planck/boot.d/STOP, boot.d/screens.STOP, ~/planck/STOP,
# ~/planck/PAUSE, plus SCREENS' own ~/planck/runs/SCREENS/NO_BOOT_RESUME); nothing of SCREENS runs (the queue, its
# launcher, a SCREENS train.py, preflight.py or bpb.py, screens.py check); and queue_screens.txt's last "queue start:
# exp SCREENS" has no line after it that ends the queue. After its start line queue_screens.sh logs one of these on
# every exit: "plan finished:" or "plan says stop" (exit 0); "queue stops at: train" (exit 1: every stop inside a
# train line, STOP, disk, refusals, heat, the RC-12 guard, scoring); "STOP while waiting for" (1); "bad plan line:"
# (2). A relaunch that refuses before its own start line logs "refusing:" (1) after the old one. Usage errors and a
# second queue exit before the log opens. A power cut or a kill logs none of these. Each must follow a qlog time
# stamp, so a MARK text or a tool's stderr line cannot end a run.
# Restart path: launch_screens.sh's own command, without its clone step (never the kit's newest bundle, never
# plans/CURRENT):  bash Q/experiments/screens/queue_screens.sh --code Q --plan Q/experiments/screens/plans/PLAN
# with Q = ~/planck/qcode/<commit12> and PLAN from that start line. Refused (exit 1) when Q is missing or not at that
# commit, or PLAN is missing or not the sha256 the start line logged. The queue skips every run with SCORED,
# TRUNK_DONE, DIVERGED or GAP; train.py resumes the cut run from latest.json (durable since E2 DEVIATION 1,
# 2026-10-03), from step 0 if it saved none. Before that, one "boot resume:" line goes to queue_screens.txt.
# Not resumed, for a person (exit 1): power-cut damage the queue would skip, re-train or mark GAP (damage below),
# and a later queue start whose start line the cut lost (a code_sha256_<time>.txt newer than the last start line).
NAME=screens
. "$(dirname "$0")/bootlib.sh"
OUT=$P/runs/SCREENS
Q=$OUT/queue_screens.txt
PY3=${BOOT_PY:-python3}
ANY='queue_screens\.sh|launch_screens\.sh|(train|preflight)\.py [^ ]*/experiments/(screens|S00[0-9]_[^/ ]*)/configs/|bpb\.py [^ ]*/runs/SCREENS/|screens\.py check'
TS='[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}'
ENDS="$TS (plan finished: |plan says stop|queue stops at: |STOP while waiting for |refusing: |bad plan line: )"

refuse() { say "cannot resume: $*"; exit 1; }

damage() {    # damage OUT: one line per power-cut damage in OUT's run dirs, exit 3 if any (top level of each run dir:
              # a person's crashed_<date>/ is not read). Every case below is one the queue cannot repair by itself.
    "$PY3" - "$1" <<'PY'
import json, os, sys, zipfile
out, bad = sys.argv[1], []
def size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return -1
def whole(p):            # a torch.save zip: local header first, end record last (not a CRC check)
    try:
        with open(p, "rb") as f:
            return f.read(4) == b"PK\x03\x04" and zipfile.is_zipfile(p)
    except OSError:
        return False
def parses(p, rows=False):   # JSON (JSON lines: at least one) with no NUL byte
    try:
        t = open(p, "rb").read().decode("utf-8")
        if "\0" in t:
            return False
        if rows:
            return [json.loads(x) for x in t.splitlines() if x.strip()] != []
        json.loads(t)
        return True
    except (OSError, ValueError):
        return False
for r in sorted(os.listdir(out)):
    d = os.path.join(out, r)
    if not os.path.isdir(d) or r in ("marks", "preflight", "check"):
        continue
    names = os.listdir(d)
    for f in sorted(names):   # torch.load fails on these: a crash, a resume, a second crash = GAP, or a scoring stop
        if f.endswith(".pt") and not whole(os.path.join(d, f)):
            bad.append(f"{r}/{f} is not a whole checkpoint ({size(os.path.join(d, f))} B)")
    if "latest.json" in names:   # missing target: train.py silently starts again from step 0
        p, n = os.path.join(d, "latest.json"), None
        if parses(p):
            j = json.load(open(p))
            n = j.get("path") if isinstance(j, dict) else None
        if n not in names:
            bad.append(f"{r}/latest.json does not name a file in the run dir [{n}]")
    scored = "SCORED" in names   # SCORED: the queue skips the run for good
    b = os.path.join(d, "bpb")
    js = sorted(x for x in os.listdir(b) if x.endswith(".json")) if os.path.isdir(b) else []
    for f in js:                 # unscored: 0 bytes is re-scored; any other bad file stops the queue at scoring
        if (scored or size(os.path.join(b, f)) != 0) and not parses(os.path.join(b, f)):
            bad.append(f"{r}/bpb/{f} is not whole JSON")
    if scored and (not js or not parses(os.path.join(d, "bpb.jsonl"), rows=True)):
        bad.append(f"{r} is SCORED but its bpb.jsonl or bpb/ is missing, empty or not whole")
    if "TRUNK_DONE" in names and not any(f.startswith("final_") and f.endswith(".pt") for f in names):
        bad.append(f"{r} is TRUNK_DONE with no final_*.pt")
print("\n".join(bad))
sys.exit(3 if bad else 0)
PY
}

blocked "$OUT/NO_BOOT_RESUME" && exit 0
if running "$ANY"; then say "running (pids $PIDS): nothing to do"; exit 0; fi
[ -f "$Q" ] || { say "no $Q: SCREENS never ran here, nothing to resume"; exit 0; }
start=$(last_start "$Q" ' queue start: exp SCREENS, plan ')
[ -n "$start" ] || { say "no queue start line in $Q: nothing to resume"; exit 0; }
# the start line: "... queue start: exp SCREENS, plan P.txt (sha256 X16), code commit REF" (NUL bytes dropped). On it,
# only the text after its last start text can end the run: a NUL tail can glue an older run's last line (an end too)
# in front of the start line of a queue a person started by hand.
sline=$(sed -n "${start}p" "$Q" | tr -d '\000')
# a start line the power cut lost: the queue creates code_sha256_<time>.txt just before it logs its start line, and
# ext4 commits a new file's name within ~5 s but a log append's bytes only after ~30 s, so a cut 5-30 s after a start
# can leave the name and NULs (or nothing) where the line was. A name newer than the last start line = a later start
# whose plan and commit are unknown here (the start before it may even show an end): never guess, a person resumes.
cs=$(ls "$OUT" | sed -n 's/^code_sha256_\([0-9]\{8\}\)_\([0-9]\{6\}\)\.txt$/\1\2/p' | sort | tail -1)
st=$(echo "$sline" | grep -aoE "$TS queue start: exp SCREENS" | tail -1 | cut -c1-19 | tr -dc '0-9')
if [ -n "$cs" ] && [ -n "$st" ] && [ "$cs" -gt "$st" ]; then
    refuse "$OUT/code_sha256_${cs:0:8}_${cs:8}.txt is newer than the last queue start line (line $start): a" \
        "later start whose line a power cut lost. Read launch_screens.log and the run dirs, resume by hand"
fi
end=$( { printf '%s\n' "${sline##* queue start: exp SCREENS, plan }"; tail -n +$((start + 1)) "$Q"; } \
    | grep -aE "$ENDS" | tail -1 | tr -d '\000')
if [ -n "$end" ]; then say "the last run ended by itself [$end]: nothing to resume"; exit 0; fi
ref=$(echo "$sline" | sed -n 's/.* code commit \([0-9a-f]\{12,\}\).*/\1/p')
plan=$(echo "$sline" | sed -n 's/.* plan \([^ /]*\) (sha256 [0-9a-f]*).*/\1/p')
psha=$(echo "$sline" | sed -n 's/.* plan [^ /]* (sha256 \([0-9a-f]*\)).*/\1/p')
C=$P/qcode/$(echo "$ref" | cut -c1-12)
X=$C/experiments/screens
say "cut off: last start at line $start (plan [$plan], commit [$ref]), last line [$(tr -d '\000' < "$Q" | grep -av '^$' | tail -1)]"
[ -n "$ref" ] && [ -n "$plan" ] || refuse "cannot read the plan and commit from line $start [$sline]"
[ -f "$X/queue_screens.sh" ] || refuse "no clone of commit $ref at $C (launch_screens.sh made it; no other is used)"
head=$(git -C "$C" rev-parse HEAD 2>/dev/null)
case $head in "$ref"*) ;; *) refuse "the clone $C is at [$head], the cut-off run used $ref";; esac
[ -f "$X/plans/$plan" ] || refuse "no plan $plan in $X/plans"
got=$(sha256sum "$X/plans/$plan" | cut -c1-16)
[ "$got" = "$psha" ] || refuse "$X/plans/$plan has sha256 $got, the cut-off run's plan had $psha"
bad=$(damage "$OUT"); rc=$?
case $rc in
    0) ;;
    3) say "power-cut damage the queue would skip, re-train or mark GAP: not resuming. A person repairs it (as E2" \
           "notes.txt CRASH RESUME: damaged files and their markers moved aside, latest.json on the last whole" \
           "checkpoint), then runs boot_resume.sh --now: $(echo "$bad" | tr '\n' ';')"
       exit 1 ;;
    *) refuse "the damage check did not run ($PY3 exit $rc)";;
esac

if [ $DRY = 0 ]; then
    mkdir -p "$P/logs"
    [ $(tail -c1 "$Q" | wc -l) -eq 1 ] || echo >> "$Q"     # a NUL tail has no newline: start the note on its own line
    echo "$(date '+%F %T') boot resume: the run started at line $start has no end line (a power cut or a kill leaves" \
        "none; WSL up $(uptime_s) s). Restarted by ~/planck/boot.d/40_screens.sh, same plan and clone." >> "$Q"
fi
LAUNCH_LOG=$P/logs/launch_screens.log launch 'queue_screens\.sh --code' \
    bash "$X/queue_screens.sh" --code "$C" --plan "$X/plans/$plan"
