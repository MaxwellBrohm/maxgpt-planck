#!/bin/bash
# parity_p1b_cuda.sh: the CUDA flag-off parity for E3 Part 1b (notes.txt ADDENDUM 2026-10-12, PART 1b PARITY): Part 1's
# arm A ran at e39112a; Part 1b runs the same config at a commit whose harness/, data_prep/, tokenizer/ and corpus/ equal
# 4962426's (Part 3's commit). Started on the PC by pcdetach.sh; two tarballs and a sums file sit beside it in planck-kit,
# made on the Mac with git archive (harness, data_prep, tokenizer, corpus, and a COMMIT file):
#   p1bpar_ref.tgz   e39112a (Part 1's queue commit)
#   p1bpar_new.tgz   the tree Part 1b runs (git tree ids pinned in e3p1b_waiter.sh)
#   p1bpar.sha256    sha256sum -c lines for both; a mismatch stops the run before the lock.
# In one gpu.lock hold, each with harness/default_parity.py of the new tree, --device cuda (bf16, deterministic
# algorithms, 24 steps, the tiny CUDA model; log records, the final checkpoint with the CUDA RNG, the start record and
# the imported modules compared with no tolerance):
#   A  registered: --ref ref --new new. Wanted: log, checkpoint and start record identical and the only module the new
#      tree imports that the old did not is durable (DEVIATION D4's fsync-before-rename I/O module), so default_parity
#      itself prints FAIL for that one import; any other difference is NOT the wanted result.
#   B  mutant: new tree, data_seed absent gives seed + 1: wanted FAIL with log and checkpoint different (the comparison
#      sees a loader seed change on CUDA).
#   C  control: --ref ref --new ref: wanted PASS (two processes of one tree agree bitwise on CUDA).
# Guards: ~/planck/STOP, 40 GB free on /, flock -w 7200 on ~/planck/locks/gpu.lock (a 2 h wait is logged and waited
# again, at most 12 times), then under the lock GPU above 75 C or more than 3,000 MiB in use releases it and waits 60 s.
# Each invocation under timeout 1800. Nothing under ~/planck/runs is read or written. Log ~/planck/logs/
# e3p1b_parity_cuda.log: "RESULT A|B|C ...", then "DONE ok" (A, B, C as wanted) or "DONE NOT OK".
# P1BPAR_DEVICE, P1BPAR_LOCK and P1BPAR_LOG exist for the CPU rehearsal with a scratch lock only.
exec > "${P1BPAR_LOG:-$HOME/planck/logs/e3p1b_parity_cuda.log}" 2>&1
set -u
KIT=$(cd "$(dirname "$0")" && pwd)
P=$HOME/planck
PY=$P/venv/bin/python
DEV=${P1BPAR_DEVICE:-cuda}
LOCK=${P1BPAR_LOCK:-$P/locks/gpu.lock}
W=$P/dev/e3p1b_parity_$(date +%Y%m%d_%H%M%S)
say() { echo "$(date '+%F %T') $*"; }
die() { say "STOP: $*"; say "DONE NOT OK"; exit 1; }

say "parity_p1b start; kit $(basename "$KIT"); work $(basename "$W"); device $DEV"
[ -e "$P/STOP" ] && die "~/planck/STOP present"
avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
[ "${avail:-0}" -ge 40 ] || die "under 40 GB free on / (${avail} GB)"
(cd "$KIT" && sha256sum -c p1bpar.sha256) || die "tarball sha256 mismatch"
mkdir -p "$W/ref" "$W/new" || die "mkdir"
tar -xzf "$KIT/p1bpar_ref.tgz" -C "$W/ref" || die "untar ref"
tar -xzf "$KIT/p1bpar_new.tgz" -C "$W/new" || die "untar new"
say "ref commit $(cat "$W/ref/COMMIT"); new commit $(cat "$W/new/COMMIT")"
"$PY" -c "import torch, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda)"
cp -a "$W/new" "$W/mut_b" || die "copy for mutant B"
"$PY" - "$W/mut_b/harness/train.py" <<'EOF' || die "mutant B"
import sys
p, old = sys.argv[1], '    if "data_seed" not in cfg:\n        return seed\n'
t = open(p).read()
if t.count(old) != 1:
    sys.exit(f"old text occurs {t.count(old)} times")
open(p, "w").write(t.replace(old, old.replace("return seed\n", "return seed + 1\n")))
EOF
for t in ref new mut_b; do say "tree $t train.py sha256 $(sha256sum "$W/$t/harness/train.py" | cut -c1-16)"; done

exec 9> "$LOCK" || die "cannot open gpu.lock"
waits=0
while :; do
    [ -e "$P/STOP" ] && die "~/planck/STOP present"
    say "waiting for gpu.lock"
    if ! flock -w 7200 9; then
        waits=$((waits + 1)); say "gpu.lock not free after 2 h ($waits)"
        [ $waits -ge 12 ] && die "12 lock waits of 2 h"
        continue
    fi
    read -r temp used <<< "$(nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ',')"
    if [ "${temp:-99}" -gt 75 ] || [ "${used:-99999}" -gt 3000 ]; then
        say "gpu.lock held but GPU ${temp} C, ${used} MiB in use: release, wait 60 s"
        flock -u 9; sleep 60; continue
    fi
    say "gpu.lock held (GPU ${temp} C, ${used} MiB in use)"
    break
done

declare -A RC
for x in A:ref:new B:ref:mut_b C:ref:ref; do
    IFS=: read -r k r n <<< "$x"
    say "run $k: --ref $r/harness --new $n/harness --device $DEV"
    timeout 1800 "$PY" "$W/new/harness/default_parity.py" --ref "$W/$r/harness" --new "$W/$n/harness" \
        --device "$DEV" --work "$W/work_$k" > "$W/out_$k.txt" 2>&1
    RC[$k]=$?
    sed "s#$W#<work>#g; s#$HOME#~#g" "$W/out_$k.txt"
    say "run $k rc ${RC[$k]}"
done
nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv,noheader
flock -u 9
say "gpu.lock released"

line() { grep -c "^\[default_parity\] $2$" "$W/out_$1.txt"; }
ok=1
if [ "$(line A 'log: identical')$(line A 'checkpoint: identical')$(line A 'start: identical')" = 111 ] &&
   [ "$(line A "new_imports: \['durable'\]")$(line A FAIL)" = 11 ] && [ "${RC[A]}" = 1 ] &&
   grep -q "^\[default_parity\] $DEV: 24 log records" "$W/out_A.txt"; then
    say "RESULT A rc 1: log, checkpoint and start record identical; new imports ['durable'] only (D4), as wanted"
else ok=0; say "RESULT A rc ${RC[A]}: NOT the wanted result"; fi
if [ "${RC[B]}" = 1 ] && [ "$(line B FAIL)" = 1 ] && [ "$(line B 'log: identical')$(line B 'checkpoint: identical')" = 00 ]; then
    say "RESULT B rc 1 FAIL as wanted (log and checkpoint differ: mutant killed)"
else ok=0; say "RESULT B rc ${RC[B]}: NOT the wanted FAIL"; fi
if [ "${RC[C]}" = 0 ] && [ "$(line C PASS)" = 1 ]; then say "RESULT C rc 0 PASS as wanted (one tree, two processes)"
else ok=0; say "RESULT C rc ${RC[C]}: NOT the wanted PASS"; fi
if [ $ok = 1 ]; then say "DONE ok"; else say "DONE NOT OK"; exit 1; fi
