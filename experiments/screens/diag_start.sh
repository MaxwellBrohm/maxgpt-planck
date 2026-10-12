#!/bin/bash
# diag_start.sh: the one file pcdetach.sh starts for the C6 diagnostics queue (SCREENS AMENDMENT C6-DIAG INSTALL). It runs
# planck-kit/diag_queue.sh in --bundle mode, which clones planck-kit/planck_c6diag.bundle's main into
# ~/planck/qcode/<commit12> and runs that clone's own diag_queue.sh. Arguments (--groups, --only) are passed through.
KIT=${C6_KIT:-/mnt/c/Users/$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r')/planck-kit}
exec bash "$KIT/diag_queue.sh" --bundle planck_c6diag.bundle "$@"
