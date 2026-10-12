"""Mutants for the corrections to SCREENS.txt AMENDMENT SEEDS-103-105 made after the independent check (2026-10-12):
the NEW-INIT SD wording and numbers (test_screens_robust_verify.py), and robust_waiter.sh's two single-instance locks and
default E3 pattern (test_robust_waiter_locks.py). Read by mutation_screens.py (name, file, old, new, test file); each
must make the named tests fail.
"""
SX = "experiments/screens/"
RW, TX = SX + "robust_waiter.sh", "experiments/SCREENS.txt"
V, K = "test_screens_robust_verify.py", "test_robust_waiter_locks.py"
MUTANTS_RBVERIFY = [
    # the entry
    ("rv_as_registered_back", TX, "SD_ref by the arm's C4 class\n      (E3 Part 1,",
     "SD_ref by the arm's C4 class\n      as registered (E3 Part 1,", V),
    ("rv_no_departure", TX, "this is a departure from C4, not C4 as\n      registered", "this is C4 as\n      registered", V),
    ("rv_init_ref", TX, "0.287%", "0.288%", V),
    ("rv_init_thr_chat", TX, "0.356%", "0.355%", V),
    ("rv_init_thr_prose", TX, "0.261%", "0.262%", V),
    ("rv_seed_ref", TX, "sqrt(2) x sigma_seed 0.361%", "sqrt(2) x sigma_seed 0.362%", V),
    ("rv_not_max", TX, "is Max's call", "is this entry's call", V),
    ("rv_no_reread_scope", TX, "the SD a failed C4 check re-reads at", "the SD of S001 alone", V),
    ("rv_t_df", TX, "(t(0.975, 4) for sigma_init)", "(t(0.975, 7) for sigma_init)", V),
    ("rv_timing", TX, "The choice was made after the k = 2", "The choice was made before the k = 2", V),
    # robust_waiter.sh
    ("rv_rw_no_singleton_lock", RW,
     'flock -n 5 || { say "another robust_waiter holds $LOGD/robust_waiter_all.lock: exiting"; exit 1; }', ":", K),
    ("rv_rw_no_requeue_lock", RW,
     'flock -n 6 || { say "another robust_waiter watches robustness_k5: exiting"; exit 1; }', ":", K),
    ("rv_rw_singleton_other_file", RW, 'exec 5>> "$LOGD/robust_waiter_all.lock"',
     'exec 5>> "$LOGD/robust_waiter_all.lock.$$"', K),
    ("rv_rw_requeue_other_file", RW, 'exec 6>> "$LOGD/robust_requeue.lock"', 'exec 6>> "$LOGD/robust_requeue.lock.$$"', K),
    ("rv_rw_epat_misses_e3p3_waiter", RW, "|e3p3_waiter\\.sh'}", "'}", K),
    ("rv_rw_epat_misses_queue_e2", RW, "'queue_e2\\.sh --exp E3|", "'", K),
    ("rv_rw_epat_any_queue_e2", RW, "'queue_e2\\.sh --exp E3|", "'queue_e2\\.sh|", K),
]
