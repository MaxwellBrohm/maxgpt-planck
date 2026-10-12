#!/bin/bash
# parity_cuda.sh: the CUDA flag-off parity run of the E3 Part 3 data_seed patch (notes.txt PART 3 RESULT: "Not run:
# the CUDA flag-off parity run (default_parity.py --device cuda)"; SCREENS.txt C4: a BASE run pairs with an arm from a
# later harness commit only if every change between the commits is a flag defaulting off and the flag-off equivalence
# test passes at the later commit). Started on the PC by pcdetach.sh; two tarballs and a sums file sit beside it in
# planck-kit, made on the Mac with git archive (harness, data_prep, tokenizer, corpus):
#   p3par_ref.tgz   the reference commit (27d8cb7: stage2_base_diag's queue commit; its harness/ equals 93bc10b's)
#   p3par_new.tgz   the commit under test (its harness/ carries data_seed)
#   p3par.sha256    sha256sum -c lines for both; a mismatch stops the run before the lock.
# Each tarball holds COMMIT (the full commit id). In one gpu.lock hold, all with harness/default_parity.py of the new
# tree, --device cuda (bf16, deterministic algorithms, 24 steps, the tiny CUDA model; log records, the final checkpoint
# with the CUDA RNG, the start record and the imported modules compared with no tolerance):
#   A  registered: --ref ref/harness --new new/harness             wants PASS (exit 0)
#   B  mutant: new tree, data_seed absent gives seed + 1            wants FAIL (the check sees the loader seed on CUDA)
#   C  mutant: new tree, data_seed always in the start record      wants FAIL (the check sees the start record)
# default_parity compares only the checkpoint's model, optimizer, data_state, rng, step and token counts, so a data_seed
# key always written to the checkpoint meta would pass it (seen on the Mac CPU); harness/test_data_seed.py's pre-patch
# test compares every checkpoint key on CPU.
# B and C are scratch copies of new/, each one edit whose old text must occur exactly once. Guards: ~/planck/STOP,
# 40 GB free on /, flock -w 7200 on ~/planck/locks/gpu.lock (a 2 h wait is logged and waited again, at most 12 times),
# then under the lock GPU above 75 C or more than 3,000 MiB in use releases it and waits 60 s. Each invocation under
# timeout 1800. Nothing under ~/planck/runs is read or written. Log: ~/planck/logs/e3p3_parity_cuda.log; result lines
# "RESULT A|B|C rc N ...", then "DONE ok" (A passed, B and C failed) or "DONE NOT OK".
exec > "$HOME/planck/logs/e3p3_parity_cuda.log" 2>&1
set -u
KIT=$(cd "$(dirname "$0")" && pwd)
P=$HOME/planck
PY=$P/venv/bin/python
LOCK=$P/locks/gpu.lock
W=$P/dev/e3p3_parity_$(date +%Y%m%d_%H%M%S)
say() { echo "$(date '+%F %T') $*"; }
die() { say "STOP: $*"; say "DONE NOT OK"; exit 1; }

say "parity_cuda start; kit $(basename "$KIT"); work $(basename "$W")"
[ -e "$P/STOP" ] && die "~/planck/STOP present"
avail=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
[ "${avail:-0}" -ge 40 ] || die "under 40 GB free on / (${avail} GB)"
(cd "$KIT" && sha256sum -c p3par.sha256) || die "tarball sha256 mismatch"
mkdir -p "$W/ref" "$W/new" || die "mkdir"
tar -xzf "$KIT/p3par_ref.tgz" -C "$W/ref" || die "untar ref"
tar -xzf "$KIT/p3par_new.tgz" -C "$W/new" || die "untar new"
say "ref commit $(cat "$W/ref/COMMIT"); new commit $(cat "$W/new/COMMIT")"
"$PY" -c "import torch, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda)"

mutate() {    # mutate <name> <old> <new>: a copy of new/ with one edit in harness/train.py (old text exactly once)
    cp -a "$W/new" "$W/$1" || return 1
    "$PY" - "$W/$1/harness/train.py" "$2" "$3" <<'EOF' || return 1
import sys
p, old, new = sys.argv[1], sys.argv[2].encode().decode("unicode_escape"), sys.argv[3].encode().decode("unicode_escape")
t = open(p).read()
if t.count(old) != 1:
    sys.exit(f"old text occurs {t.count(old)} times")
open(p, "w").write(t.replace(old, new))
EOF
}
mutate mut_b '    if "data_seed" not in cfg:\n        return seed\n' \
              '    if "data_seed" not in cfg:\n        return seed + 1\n' || die "mutant B"
mutate mut_c '    if "data_seed" in cfg:                  # E3 Part 3: in the start record only when set\n        start["data_seed"] = data_seed\n' \
              '    if True:\n        start["data_seed"] = data_seed\n' || die "mutant C"
for t in new mut_b mut_c; do say "tree $t train.py sha256 $(sha256sum "$W/$t/harness/train.py" | cut -c1-16)"; done
say "ref train.py sha256 $(sha256sum "$W/ref/harness/train.py" | cut -c1-16)"

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

declare -A RC VERDICT
for x in A:new B:mut_b C:mut_c; do
    k=${x%%:*}; t=${x#*:}
    say "run $k: --ref ref/harness --new $t/harness --device cuda"
    timeout 1800 "$PY" "$W/new/harness/default_parity.py" --ref "$W/ref/harness" --new "$W/$t/harness" \
        --device cuda --work "$W/work_$k" > "$W/out_$k.txt" 2>&1
    RC[$k]=$?
    sed "s#$W#<work>#g; s#$HOME#~#g" "$W/out_$k.txt"
    if grep -q '^\[default_parity\] PASS$' "$W/out_$k.txt"; then VERDICT[$k]=PASS
    elif grep -q '^\[default_parity\] FAIL$' "$W/out_$k.txt"; then VERDICT[$k]=FAIL
    else VERDICT[$k]=NONE; fi                    # a tree that did not run, a timeout: no comparison was made
    say "run $k rc ${RC[$k]} ${VERDICT[$k]}"
done
nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv,noheader
flock -u 9
say "gpu.lock released"

ok=1
if [ "${RC[A]}" = 0 ] && [ "${VERDICT[A]}" = PASS ]; then say "RESULT A rc 0 PASS"
else ok=0; say "RESULT A rc ${RC[A]} ${VERDICT[A]}: NOT the wanted PASS"; fi
for k in B C; do
    if [ "${RC[$k]}" = 1 ] && [ "${VERDICT[$k]}" = FAIL ]; then say "RESULT $k rc 1 FAIL as wanted (mutant killed)"
    else ok=0; say "RESULT $k rc ${RC[$k]} ${VERDICT[$k]}: NOT the wanted comparison FAIL"; fi
done
if [ $ok = 1 ]; then say "DONE ok"; else say "DONE NOT OK"; exit 1; fi
