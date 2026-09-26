#!/bin/bash
# queue_e3.sh: the E3 queue. The same engine as E2's (queue_e2.sh --exp E3): guards, gpu.lock per run, strict
# preflight, train.py --require-committed, scoring of every full-schedule run, idempotent restarts, and the
# joint order with E2 through marks (plans/part1.txt waits for E2's "E2 5M DONE"). Normally started detached
# by launch_e3.sh.   bash queue_e3.sh --plan PLAN_FILE     (the code root is this checkout)
D=$(cd "$(dirname "$0")/../.." && pwd)
exec bash "$D/experiments/E2_lr_transfer/queue_e2.sh" --exp E3 --code "$D" "$@"
