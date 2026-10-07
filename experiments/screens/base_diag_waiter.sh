#!/bin/bash
# base_diag_waiter.sh: start the SCREENS queue on plans/stage2_base_diag (experiments/SCREENS.txt AMENDMENT BASE-DIAG)
# once S006's queue (plans/stage2_s006_seeds, launched by s006_waiter.sh from planck_s006.bundle at 3227c60) has ended
# cleanly at its mark; while the last start line is still stage2_seeds at d21df3d, it waits. pcdetach.sh copies only
# this file to planck-kit and runs it with no arguments: chain_waiter.sh must already sit beside it, byte for byte the
# committed file (its sha256 is pinned here; anything else is refused), and the commit holding the amendment must be
# planck-kit/planck_diag.bundle. Log: ~/planck/logs/base_diag_waiter.log. Cancel: touch base_diag_waiter.CANCEL there.
H=$(dirname "$0")
SHA=920db4c14819bec872971f53f1cac815d93b1c4e9398ef5c5114c004c3f51642
LOG=${CW_HOME:-$HOME/planck}/logs/base_diag_waiter.log
if [ "$(sha256sum "$H/chain_waiter.sh" 2>/dev/null | cut -c1-64)" != "$SHA" ]; then
    mkdir -p "$(dirname "$LOG")"
    echo "$(date '+%F %T') GIVEUP: $H/chain_waiter.sh is missing or not sha256 ${SHA:0:16}" >> "$LOG"
    exit 1
fi
exec bash "$H/chain_waiter.sh" base_diag stage2_s006_seeds 3227c6011be2f76e4401551f8f8da20fcfbe44d0 \
    "SCREENS STAGE 2 S006 SEEDS DONE" stage2_base_diag planck_diag.bundle \
    stage2_seeds@d21df3de007146c2d619ba7c3f46f264286d318c
