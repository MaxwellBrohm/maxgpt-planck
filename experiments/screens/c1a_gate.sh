#!/bin/bash
# c1a_gate.sh: SCREENS AMENDMENT C1-a gate on the PC 5070, run detached (pcdetach.sh copies only this file).
# Lean copy (synced from the Mac first): ~/planck/dev/screens_gate = harness/, pc/ (no local.env),
# experiments/screens, experiments/S00*, and the two E2 YAMLs the screen configs extend (base5m.yaml, engine.yaml).
# Its planck_root is a SCRATCH directory (data/ linked to ~/planck/data, runs/ scratch), never ~/planck itself.
# Per item: smoke (gate_smoke.py, the arm's own 5M config, 300 steps compiled) and two parity invocations
# (flag_parity.py, compile_parity.py's default 7 arms, 200 steps), each in its own gpu.lock hold, then the pooled
# permutation test of the two invocations (CPU, no lock). Before each hold: ~/planck/STOP ends the run, PAUSE
# waits, WSL / needs 60 GB available and /mnt/d mounted (re-check every 15 min, up to 6 h), the GPU at or below
# 75 C, and at most 3,000 MiB in use once the lock is held (else release, wait a minute, retry). Each hold runs
# one step under timeout 1800 s (-k 60): under 35 minutes. A step at 87 C for 3 samples is stopped and the
# driver ends. Restartable: a step whose .rc file exists is skipped. Log: ~/planck/logs/screens_gate.log.
exec >> "$HOME/planck/logs/screens_gate.log" 2>&1
set -u
G=$HOME/planck/dev/screens_gate; PY=$HOME/planck/venv/bin/python; LOCK=$HOME/planck/locks/gpu.lock
OUT=$G/gate_out; mkdir -p "$OUT" "$G/tmp" "$G/cache"
export TORCHINDUCTOR_CACHE_DIR=$G/cache TRITON_CACHE_DIR=$G/cache/triton TMPDIR=$G/tmp PYTHONUNBUFFERED=1
[ -d "$G/planck_root" ] && [ ! -L "$G/planck_root" ] || { echo "no scratch planck_root in $G"; exit 2; }
# tag | flag_parity target ("-" = smoke only) | smoke config (relative to $G)
ITEMS="base|-|experiments/screens/configs/base_s101.yaml
S005_base|S005:base|experiments/S005_forget_gate/configs/s005_base_s101.yaml
S005_forget|S005:forget|experiments/S005_forget_gate/configs/s005_forget_g1_s1.yaml
S004_canonac|S004:canonac|experiments/S004_canon/configs/s004_canonac_g1_s1.yaml
S007_smear|S007:smear|experiments/S007_smeared_key/configs/s007_smear_g1_s1.yaml
S006_mtp|S006:mtp|experiments/S006_mtp_aux/configs/s006_mtp_g1_s1.yaml
S001_nogate|S001:nogate|experiments/S001_attn_gate/configs/s001_nogate_g1_s1.yaml
S002_novres|S002:novres|experiments/S002_block_ablations/configs/s002_novres_g1_s1.yaml
S002_noqknorm|S002:noqknorm|experiments/S002_block_ablations/configs/s002_noqknorm_g1_s1.yaml
S002_nonormscale|S002:nonormscale|experiments/S002_block_ablations/configs/s002_nonormscale_g1_s1.yaml"

mark() { echo "== $1 $(date '+%F %T') | $(uptime | sed 's/.*load average/load/') | gpu util,mem,temp: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null)"; }
gq() { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9'; }

ready() {   # 0 = go; 1 = the driver ends
    local n=0 said=0 f t
    while [ -e "$HOME/planck/PAUSE" ]; do [ $said = 1 ] || echo "$(date '+%F %T') PAUSE: waiting"; said=1; sleep 60; done
    [ -e "$HOME/planck/STOP" ] && { echo "$(date '+%F %T') STOP present: driver ends"; return 1; }
    while :; do
        f=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
        if [ "${f:-0}" -ge 60 ] && mountpoint -q /mnt/d; then break; fi
        n=$((n + 1)); [ $n -gt 24 ] && { echo "$(date '+%F %T') disk still not ready after 6 h: driver ends"; return 1; }
        echo "$(date '+%F %T') disk check: / ${f} GB available (need 60), /mnt/d $(mountpoint -q /mnt/d && echo mounted || echo NOT mounted): waiting 15 min"
        sleep 900
    done
    echo "$(date '+%F %T') disk check ok: / ${f} GB available, /mnt/d mounted"
    said=0
    while t=$(gq temperature.gpu); [ -z "$t" ] || [ "$t" -gt 75 ]; do
        [ $said = 1 ] || echo "$(date '+%F %T') GPU at [$t] C, not <= 75: waiting"; said=1; sleep 60
    done
}

hold() {    # hold TAG STEP CMD... : one gpu.lock hold for one step; writes $OUT/TAG_STEP.rc
    local tag=$1 step=$2 used pid hot=0 t rc t0; shift 2
    [ -e "$OUT/${tag}_$step.rc" ] && { echo "skip $tag $step (rc $(cat "$OUT/${tag}_$step.rc"))"; return 0; }
    t0=$(date +%s)
    while :; do
        ready || return 1
        exec 9>"$LOCK"
        echo "$(date '+%F %T') waiting for gpu.lock"
        flock -w 7200 9 || { echo "gpu.lock not free after 2 h: driver ends"; return 1; }
        used=$(gq memory.used)
        [ -n "$used" ] && [ "$used" -le 3000 ] && break
        echo "$(date '+%F %T') GPU memory in use [$used] MiB > 3000 with the lock held: releasing, retry in 60 s"
        flock -u 9; exec 9>&-; sleep 60
        [ $(( $(date +%s) - t0 )) -lt 7200 ] || { echo "GPU busy for 2 h: driver ends"; return 1; }
    done
    mark "$tag $step start (lock held)"
    timeout -k 60 1800 "$@" > "$OUT/${tag}_$step.log" 2>&1 9>&- < /dev/null &   # the step does not inherit the lock fd
    pid=$!
    while kill -0 $pid 2>/dev/null; do
        t=$(gq temperature.gpu)
        if [ -n "$t" ] && [ "$t" -ge 87 ]; then hot=$((hot + 1)); else hot=0; fi
        [ $hot -ge 3 ] && { echo "$(date '+%F %T') GPU at $t C for 3 samples: stopping $tag $step"; touch "$OUT/HEAT"; kill $pid; }
        sleep 10
    done
    wait $pid; rc=$?
    mark "$tag $step end rc $rc"
    flock -u 9; exec 9>&-
    [ -e "$OUT/HEAT" ] && return 1
    echo $rc > "$OUT/${tag}_$step.rc"
    grep -a -E "^\[(smoke|c1a)\] |^\[parity\] (noise floor|pooled)" "$OUT/${tag}_$step.log" | cut -c1-400
    return 0
}

echo "=== c1a_gate start $(date '+%F %T')"
( cd "$G" && find harness pc experiments -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' \) \
    -not -path '*/__pycache__/*' | sort | xargs md5sum ) > "$OUT/tree_md5.txt"
echo "lean tree: $(wc -l < "$OUT/tree_md5.txt") files, md5 of list $(md5sum < "$OUT/tree_md5.txt" | cut -c1-12)"
while IFS='|' read -r tag target cfg; do
    name=$(basename "$cfg" .yaml); [ -n "$name" ] && [ -n "$tag" ] || { echo "bad item line"; exit 2; }
    [ -e "$OUT/${tag}_smoke.rc" ] || rm -rf "$G/planck_root/runs/SCREENS/$name"
    hold "$tag" smoke bash -c "cd '$G/experiments/screens' && exec '$PY' gate_smoke.py '$G/$cfg' --scratch '$G' --steps 300 --out '$OUT/${tag}_smoke.json'" || exit 1
    [ "$target" = - ] && continue
    for inv in a b; do
        hold "$tag" "$inv" bash -c "cd '$G/experiments/screens' && exec '$PY' flag_parity.py '$target' --steps 200 --out '$OUT/${tag}_$inv.json'" || exit 1
    done
    if [ ! -e "$OUT/${tag}_pool.rc" ] && [ -s "$OUT/${tag}_a.json" ] && [ -s "$OUT/${tag}_b.json" ]; then
        ( cd "$G/harness" && "$PY" compile_parity.py --pool "$OUT/${tag}_a.json,$OUT/${tag}_b.json" ) > "$OUT/${tag}_pool.log" 2>&1 < /dev/null
        echo $? > "$OUT/${tag}_pool.rc"; echo "pool $tag rc $(cat "$OUT/${tag}_pool.rc")"; cat "$OUT/${tag}_pool.log" | cut -c1-300
    fi
done <<< "$ITEMS"
echo "=== c1a_gate done $(date '+%F %T')"
touch "$OUT/ALL_DONE"
