"""Mutants for SCREENS2's SEQUENTIAL RULE (screens2_seq.py) and its simulator (screens2_seqsim.py). Read by
mutation_screens2_seq.py through batch 1's runner (name, file, old, new, test file); each must make
tests/test_screens2_seq.py fail."""
SX = "experiments/screens/"
Q, S, T = SX + "screens2_seq.py", SX + "screens2_seqsim.py", "test_screens2_seq.py"
MUTANTS_SEQ = [
    ("q_linear_spend", Q, "ALPHA * math.log(1 + (math.e - 1) * min(max(t, 0.0), 1.0))",
     "ALPHA * min(max(t, 0.0), 1.0)", T),
    ("q_w_not_shared", Q, "return sum(w * _low_high(c1 * v, c2 * v, rho) for v, w in _w_nodes(df))",
     "return _low_high(c1, c2, rho)", T),
    ("q_rho_wrong", Q, "c1, rho, rest = _q(1 - a1, df), math.sqrt(k1 / k2), ALPHA - a1",
     "c1, rho, rest = _q(1 - a1, df), k1 / k2, ALPHA - a1", T),
    ("q_rest_is_alpha", Q, "c1, rho, rest = _q(1 - a1, df), math.sqrt(k1 / k2), ALPHA - a1",
     "c1, rho, rest = _q(1 - a1, df), math.sqrt(k1 / k2), ALPHA", T),
    ("q_spend_at_actual_fraction", Q, "    a1 = spend(k1 / kmax)\n", "    a1 = spend(k1 / k2)\n", T),
    ("q_look2_uses_c1", Q, '        c = b["c2"]\n', '        c = b["c1"]\n', T),
    ("q_look1_uses_c7", Q, '        c = b["c1"]\n    elif', "        c = t_ppf(0.975, df)\n    elif", T),
    ("q_look2_every_seed", Q, "b, need = bounds(df, k1, k), sign_min(k)", "b, need = bounds(df, k1, k), k", T),
    ("q_look2_no_sign_clause", Q, "b, need = bounds(df, k1, k), sign_min(k)", "b, need = bounds(df, k1, k), 0", T),
    ("q_sign_p_loose", Q, "SIGN_P = 1 / 32 ", "SIGN_P = 1 / 16 ", T),
    ("q_equal_without_thr", Q, "    equal = abs(dbar) + thr < delta\n", "    equal = abs(dbar) < delta\n", T),
    ("q_p_no_look1_spend", Q, 'return min(1.0, 2 * b["a1"] + 2 * inner)', "return min(1.0, 2 * inner)", T),
    ("q_p_one_sided", Q, 'return min(1.0, 2 * b["a1"] + 2 * inner)', 'return min(1.0, b["a1"] + inner)', T),
    ("q_extends_on_tie", Q, 'return verdict == "INCONCLUSIVE"', 'return verdict in ("INCONCLUSIVE", "TIE")', T),
    ("q_inf_not_positive", Q, "    worse = dbar > thr and sum(x > 0 for x in d) >= need\n",
     "    worse = dbar > thr and sum(0 < x < A.INF for x in d) >= need\n", T),
    ("s_checks_off", S, "    if checks:\n", "    if False:\n", T),
    ("s_reread_never_observed", S, "use_obs = fail & (obs > x[\"init\"])", "use_obs = fail & False", T),
    ("s_look2_every_seed", S, "Q.sign_min(10), True)", "10, True)", T),
    ("s_bare_c1_twice", S, "t1 > c1) | ((t1 <= c1) & (t2 > c2))", "t1 > c1) | ((t1 <= c1) & (t2 > c1))", T),
    ("s_new_init_independent_reread", S, '"init": sd if r == INIT else init', '"init": init', T),
]
