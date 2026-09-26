#!/bin/zsh
# E006: copy the code and read-only inputs to the PC working copy ~/planck/dev/e006 (Mac side; notes.txt PC QUEUE).
# usage: PCSYNC=<path to the session's pcsync.sh> sync_e006.sh [--refs]
# The working copy mirrors the repo layout, so every relative path in E001-E006 code works unchanged:
#   corpus/*.py, pc/wsl/gpuguard.py
#   experiments/E006_three_objects/{code,al,big,copies,replay_ref,notes.txt}
#   experiments/E005_alias_eot/{code,al,out,transcripts}      (identity checks, Q-device's Mac records)
#   experiments/E004_general_updating/{code,out,transcripts}   (eval identity, E004's untouched records)
#   experiments/E002_ft_test/out                               (the continuity reproduction check)
# Outputs never live in the working copy: queue_e006.sh links E006's out/ logs/ transcripts/ weights/ replay/ to
# ~/planck/e006_run/ (persistent), so a re-sync (pcsync.sh replaces the directory) cannot delete a result.
# --refs also copies E005's and E004's saved weights (2.5 GB each) to ~/planck/dev/e006_refs_e005 and _e004.
# Refuses while an E006 queue runs on the PC (a code change mid-queue would break logs/code_sha256_at_start.txt).
set -e
[[ -x "$PCSYNC" ]] || { echo "set PCSYNC to the session's pcsync.sh"; exit 1 }
PCWSL=${PCSYNC:h}/pcwsl.sh
REPO=${0:A:h:h:h:h}
T=$(mktemp -d)
cat > $T/busy.sh <<'EOS'
#!/bin/bash
pgrep -f "queue_e006.sh" >/dev/null && echo BUSY || echo IDLE
EOS
[[ "$($PCWSL $T/busy.sh 60 | tail -1)" == IDLE ]] || { echo "refusing: queue_e006.sh is running on the PC"; exit 2 }
S=$T/e006
mkdir -p $S/corpus $S/pc/wsl $S/experiments/E006_three_objects $S/experiments/E005_alias_eot \
  $S/experiments/E004_general_updating $S/experiments/E002_ft_test
cp $REPO/corpus/*.py $S/corpus/
cp $REPO/pc/wsl/gpuguard.py $S/pc/wsl/
E6=$REPO/experiments/E006_three_objects
cp -R $E6/code $E6/al $E6/big $E6/copies $E6/replay_ref $E6/notes.txt $S/experiments/E006_three_objects/
cp -R $REPO/experiments/E005_alias_eot/{code,al,out,transcripts} $S/experiments/E005_alias_eot/
cp -R $REPO/experiments/E004_general_updating/{code,out,transcripts} $S/experiments/E004_general_updating/
cp -R $REPO/experiments/E002_ft_test/out $S/experiments/E002_ft_test/
find $S -name __pycache__ -prune -exec rm -rf {} +
$PCSYNC $S e006
if [[ "$1" == --refs ]]; then
  $PCSYNC $REPO/experiments/E005_alias_eot/weights e006_refs_e005
  $PCSYNC $REPO/experiments/E004_general_updating/weights e006_refs_e004
fi
rm -rf $T
