#!/bin/bash
# E2/E3 SMOKE (label smoke; never read for a decision). PC WSL, started detached (pcdetach.sh), about 15 min
# of GPU under gpu.lock. Needs the lean code copy synced to ~/planck/dev/e2smoke (harness/, data_prep/,
# tokenizer/, corpus/oodh.py + sample_plan.py, experiments/E2_lr_transfer/). Everything runs from a private
# copy in ~/planck/runs/smoke/code (~/planck/dev was cleaned by another process once). Log: logs/e2_smoke.log.
exec >> "$HOME/planck/logs/e2_smoke.log" 2>&1
set -u
P=$HOME/planck; R=$P/runs/smoke; SRC=$P/dev/e2smoke
echo "=== smoke start $(date '+%F %T')"
[ -d "$SRC/harness" ] || { echo "no synced code in dev/e2smoke"; exit 1; }
[ -e "$R" ] && mv "$R" "$R.old.$(date +%s)"
mkdir -p "$R"; CODE=$R/code; cp -a "$SRC" "$CODE"
. "$CODE/experiments/E2_lr_transfer/queue_lib.sh"
link_root
code_hashes "$R/code_sha256.txt"
SM=$E2DIR/smoke
tick() { echo "$1 $(date +%s)" >> "$R/timing.txt"; qlog "phase $1"; }

tick prereg_gate
T=$(mktemp -d); cp "$SM/trunk.yaml" "$T/config.yaml"
( cd "$CODE/harness" && $PY train.py "$T/config.yaml" --device cuda ); rc=$?
rm -rf "$T"; echo "$rc" > "$R/prereg_gate_rc.txt"; qlog "prereg gate on a config with no prereg.yaml: rc $rc (want 2)"

tick preflight
$PY "$E2DIR/preflight.py" "$SM/trunk.yaml" "$SM/bmid.yaml" "$SM/bend.yaml" --code "$CODE" --no-init-check \
    --out "$R/preflight_plan.json" || { qlog "preflight refused"; exit 3; }
guards || exit 4
tick lock_wait
gpu_lock || exit 5
tick lock_held
$SMI > "$R/gpu_at_start.txt" 2>&1

tick trunk_part1
train_run "$SM/trunk.yaml" "$R/smoke_trunk" --max-steps 250; qlog "trunk part 1 rc $? (12 = paused as planned)"
tick trunk_resume
train_run "$SM/trunk.yaml" "$R/smoke_trunk" || { qlog "trunk failed"; gpu_unlock; exit 6; }
for b in bmid bend; do
    tick "preflight_$b"
    $PY "$E2DIR/preflight.py" "$SM/$b.yaml" --code "$CODE" --out "$R/preflight_$b.json" || { qlog "preflight $b refused"; gpu_unlock; exit 3; }
    tick "train_$b"
    train_run "$SM/$b.yaml" "$R/smoke_$b" || { qlog "$b failed"; gpu_unlock; exit 6; }
done

tick score_first
mkdir -p "$R/smoke_trunk/bpb"
( cd "$CODE" && $PY data_prep/bpb.py "$R/smoke_trunk/ckpt_00000100.pt" "$EVAL" --tokenizer "$TOK" --device cuda \
    --precision fp32 --batch-tokens 16384 --out "$R/smoke_trunk/bpb/ckpt_00000100.json" > /dev/null ) \
    || qlog "scoring the first checkpoint failed"
$PY "$E2DIR/bpb_lines.py" "$R/smoke_trunk" > "$R/smoke_trunk/bpb.jsonl"
tick score_bmid
score_run "$R/smoke_bmid"
tick score_bend
score_run "$R/smoke_bend"
tick unlock
gpu_unlock

tick check
$PY "$SM/smoke_check.py" "$R" "$CODE" > "$R/smoke_check.json"; qlog "smoke_check rc $?"
tick end
cp "$P/logs/e2_smoke.log" "$R/smoke.log"
echo "=== SMOKE DONE $(date '+%F %T')"
