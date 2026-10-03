# qlib_screens.sh: the two SCREENS additions to E2's queue_lib.sh (sourced by queue_screens.sh after queue_lib.sh;
# bash 3.2 compatible, so the Mac tests run it). Needs CODE.
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
