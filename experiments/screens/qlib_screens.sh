# qlib_screens.sh: the SCREENS additions to E2's queue_lib.sh (sourced by queue_screens.sh after queue_lib.sh;
# bash 3.2 compatible, so the Mac tests run it). Needs CODE. cfg_for and code_hashes_screens since 2026-10-02;
# score_screens, qrun_set, qrun_clear and cut_count since SCREENS DEVIATION 1 (2026-10-10, below).
#
# cfg_for NAME: the one config of a SCREENS run. Shared BASE runs live in experiments/screens/configs, every
# screen's runs in experiments/S00x_<slug>/configs (each dir has its own prereg.yaml). Prints the path; returns 1
# (and prints nothing) unless exactly one file matches, so a name can never pick the wrong screen's config.
cfg_for() {
    local n=0 hit= f
    for f in "$CODE"/experiments/screens/configs/"$1".yaml "$CODE"/experiments/S00?_*/configs/"$1".yaml; do
        [ -f "$f" ] || continue
        n=$((n + 1)); hit=$f
    done
    [ "$n" = 1 ] || return 1
    echo "$hit"
}

# code_hashes_screens DEST: E2 FIXED 2's code list plus the screens' own files (configs, plans, generator).
code_hashes_screens() {
    ( cd "$CODE" && find harness data_prep experiments/E2_lr_transfer experiments/screens experiments/S00?_* \
        tokenizer corpus \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' -o -name '*.txt' -o -name 'tok_v0_8k.json' \) \
        2>/dev/null | grep -v __pycache__ | sort | xargs sha256sum ) > "$1"
}

# SCREENS DEVIATION 1 (SCREENS.txt, 2026-10-10): durable scoring and the crash count after a power cut. Needs CODE, PY,
# EVAL, TOK (queue_lib.sh) and SX (experiments/screens in CODE); score_sync.py does every fsync.
#
# score_screens OUT_DIR: queue_lib.sh's score_run, with two changes. A bpb/<ckpt>.json that is not empty and not whole
# JSON (a power cut's NULs or lost tail; bpb.py writes it in place) is moved aside to <ckpt>.json.unparsable_<time> and
# scored again (before, the run stopped the queue for a person). Then score_sync.py publish fsyncs every bpb/*.json,
# writes bpb.jsonl (bpb_lines.py's output) and then SCORED, each through a fsynced temporary file and a rename, so
# SCORED is written only after the files it vouches for are on disk. Returns 1 if a scoring or the publish fails.
score_screens() {
    local out=$1 ck b j
    mkdir -p "$out/bpb"
    for ck in $(ls "$out"/final_*.pt 2>/dev/null) $(ls "$out"/ckpt_*.pt 2>/dev/null | sort | tail -2); do
        b=$(basename "$ck" .pt); j=$out/bpb/$b.json
        if [ -s "$j" ]; then
            $PY "$SX/score_sync.py" whole "$j" && continue
            mv "$j" "$j.unparsable_$(date +%Y%m%dT%H%M%S)"
            qlog "$(basename "$out")/bpb/$b.json is not whole JSON (a cut write): moved aside, scoring again"
        fi
        qlog "score $(basename "$out")/$b"
        ( cd "$CODE" && $PY data_prep/bpb.py "$ck" "$EVAL" --tokenizer "$TOK" --device cuda \
            --precision fp32 --batch-tokens 16384 --out "$j" > /dev/null ) \
            2>> "$out/score.err" || { qlog "scoring failed for $b (see score.err)"; return 1; }
    done
    $PY "$SX/score_sync.py" publish "$out" 2>> "$out/score.err" || { qlog "publish failed in $(basename "$out") (see score.err)"; return 1; }
}

# The crash count after a cut (E2's policy: a crash resumes once, a second crash is a GAP). queue_lib.sh's train_run
# counts only crashes it sees; a power cut, a reboot or a kill takes the queue down with the run and leaves nothing
# in crashes.txt. Policy: such a cut counts as the run's one crash.
# qrun_set OUT_DIR / qrun_clear OUT_DIR: QRUN exists exactly while this queue is inside train_run for OUT_DIR (written
#   durably before, removed after it returns, whatever it returned: a STOP or a heat stop clears it too).
# cut_count OUT_DIR: before a run is trained, a QRUN left by an earlier queue means that queue died inside train_run.
#   Returns 0 when there is none, or when the run had saved its final checkpoint (training was over: not a crash);
#   else appends a line to crashes.txt (durably) and returns 1 (resume, once), or, when that makes 2 lines, writes GAP
#   and returns 2 (the caller logs the run a GAP and goes on with the plan).
qrun_set() {
    mkdir -p "$1"
    $PY "$SX/score_sync.py" mark "$1/QRUN" "$(date '+%F %T') queue pid $$ inside train_run"
}
qrun_clear() { rm -f "$1/QRUN"; }
cut_count() {
    local out=$1 n
    [ -e "$out/QRUN" ] || return 0
    if ls "$out"/final_*.pt >/dev/null 2>&1; then
        qlog "$(basename "$out"): QRUN left by a cut queue, final checkpoint saved: not a crash"; rm -f "$out/QRUN"; return 0
    fi
    $PY "$SX/score_sync.py" append "$out/crashes.txt" \
        "$(date '+%F %T') cut (power cut, reboot or kill while training; QRUN: $(head -1 "$out/QRUN")): counted as a crash"
    rm -f "$out/QRUN"
    n=$(wc -l < "$out/crashes.txt")
    if [ "$n" -ge 2 ]; then
        $PY "$SX/score_sync.py" mark "$out/GAP" "second crash: the cut counted as one (SCREENS DEVIATION 1)"
        qlog "GAP $(basename "$out"): second crash (a cut, counted as a crash)"; return 2
    fi
    qlog "$(basename "$out"): the queue was cut while it trained: counted as its one crash; resuming once"
    return 1
}
