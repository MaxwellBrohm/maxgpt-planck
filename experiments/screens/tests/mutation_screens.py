"""Mutation check for the SCREENS tests, on scratch copies only (the repo is never edited).

  ~/.venvs/planck/bin/python experiments/screens/tests/mutation_screens.py [--scratch DIR] [--only NAME,...]

Copies harness/, the E2 and E3 files the screens read, experiments/screens and experiments/S00?_* into a fresh
scratch tree per mutant, applies ONE edit (the old text must occur exactly once), runs the named test file(s)
there, and wants them to FAIL. The unmutated copy must pass the same files first. Exit 0 = every mutant killed.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SX = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(SX))
CF, RF, SF = "test_screens_configs.py", "test_screens_refusals.py", "test_screens_smoke.py"
LIB, GEN, FPY = "experiments/screens/screens_lib.py", "experiments/screens/screens.py", "experiments/screens/flag_parity.py"
PF = "test_flag_parity.py"
AF, ALIB, AN = "test_analyze.py", "experiments/screens/analyze_lib.py", "experiments/screens/analyze.py"
ANALYZER_MUTANTS = [   # C7 / C4 / ORDER / C3 / the run-directory RC-12 refusal (analyze_lib.py, analyze.py)
    ("every_seed_clause_dropped_better", ALIB, "better = -dbar > thr and all(x < 0 for x in d)", "better = -dbar > thr", AF),
    ("every_seed_clause_dropped_worse", ALIB, "worse = dbar > thr and all(x > 0 for x in d)", "worse = dbar > thr", AF),
    ("sign_flipped", ALIB, "else arm[s][key] - base[s][key] for s in used]", "else base[s][key] - arm[s][key] for s in used]", AF),
    ("delta_ignored", ALIB, "equal = abs(dbar) + thr < delta", "equal = not better and not worse", AF),
    ("diverged_pair_dropped", ALIB, 'used = [s for s in seeds if not base[s].get("diverged")]',
     'used = [s for s in seeds if not base[s].get("diverged") and not arm[s].get("diverged")]', AF),
    ("u_left_at_1", ALIB, '"SIA": u * n["SD_d_rel"]', '"SIA": n["SD_d_rel"]', AF),
    ("new_init_without_sqrt2", ALIB, 'math.sqrt(2) * n["sigma_seed_rel"]', 'n["sigma_seed_rel"]', AF),
    ("cccc_only_ignored", ALIB, '("NOMINATE" if not worse_p or cccc_only else "GUARD-FAIL")',
     '("NOMINATE" if not worse_p else "GUARD-FAIL")', AF),
    ("cccc_only_unblocks_tie", ALIB, '("TIE" if not worse_p else "GUARD-FAIL")', '("TIE" if not worse_p or cccc_only else "GUARD-FAIL")', AF),
    ("holm_no_step_down", ALIB, "ok = ok and p <= alpha / (len(items) - i)", "ok = p <= alpha / (len(items) - i)", AF),
    ("cut_order_swapped", ALIB, '25.0, ["S006", "S003", "S007", "S002"]', '25.0, ["S003", "S006", "S007", "S002"]', AF),
    ("cut_current_set_not_kept", ALIB, "total -= remaining[sid] - (current or {}).get(sid, 0.0)", "total -= remaining[sid]", AF),
    ("toy_p022_ignored", ALIB, 'order = (["S004"] if toy_favours_p022 else []) + CUT_ORDER', "order = CUT_ORDER", AF),
    ("same_init_check_blind", ALIB, 'out[key] = {"sd_rel": sd, "limit_rel": lim, "df": df, "ok": sd <= lim}',
     'out[key] = {"sd_rel": sd, "limit_rel": lim, "df": df, "ok": True}', AF),
    ("pooled_sia_blind", ALIB, '"arms": len(rs), "ok": sd <= lim}', '"arms": len(rs), "ok": True}', AF),
    ("reread_not_underpowered", ALIB, 'r["labels"] + [why, "underpowered"]', 'r["labels"] + [why]', AF),
    ("same_init_fallback_never_applied", AN, 'if not all(x["ok"] for x in chk.values()):', "if False:", AF),
    ("pooled_fallback_never_applied", AN, 'if not all(x["ok"] for x in pooled.values()):', "if False:", AF),
    ("rc12_guard_removed", AN, "    if bad:\n        raise Refused", "    if False:\n        raise Refused", AF),
    ("rc12_final_unchecked", AN, '    if os.path.isfile(os.path.join(d, "rc12_eval.jsonl")):\n        raise',
     "    if False:\n        raise", AF),
    ("c3_extension_cap_ignored", AN, 'res["extend_with"] = want if res["extensions_used"] < e2pick.MAX_EXT else []',
     'res["extend_with"] = want', AF),
    ("onset_abs_uses_own_final", ALIB, "    L = base_final / 2\n", "    L = final / 2\n", AF),
    # verification 2026-10-04: each of these four passed the 22 tests above (fixtures at BASE 1.0, one Holm contrast,
    # no S005 run); test_analyze.py's last three tests kill them
    ("sd_ref_not_scaled_by_base_mean", ALIB, 'return rel * base_mean, int(n["df"])', 'return rel, int(n["df"])', AF),
    ("noise_residuals_in_bpb", ALIB, 'return [(x - m) / c["readings"][key]["base_mean"] for x in d]',
     "return [(x - m) for x in d]", AF),
    ("holm_p_one_sided", ALIB, "else 2 * (1 - t_cdf(abs(dbar) / se, df))", "else (1 - t_cdf(abs(dbar) / se, df))", AF),
    ("s005_paired_with_shared_base", AN,
     'bname = (lambda x: f"{sid.lower()}_base_s{x}") if L.engine(sid) else (lambda x: f"base_s{x}")',
     'bname = (lambda x: f"base_s{x}")', AF),
]
MUTANTS = [   # name, file, old, new, tests (file or file::-k expr)
    ("c2_extra_key_passes", LIB, "sorted(diff(f, ref) - allowed):", "sorted(diff(f, ref) - allowed - set(f)):", RF),
    ("c2_registered_value_unchecked", LIB, 'if f.get(k, "<absent>") != v:', "if False:", RF),
    ("rc12_guard_removed", LIB, 'if "rc12" in (cfg.get("eval") or {}):', "if False:", RF),
    ("rc12_guard_arms_only", LIB, 'if "rc12" in (cfg.get("eval") or {}):',
     'if "rc12" in (cfg.get("eval") or {}) and "base" not in cfg.get("name", ""):', RF),
    ("c1b_base_compiled", "experiments/screens/configs/screens_base.yaml", "compile: false}", "compile: default}", CF),
    ("c1b_check_dropped", LIB, 'f.get("train.ce_chunk_rows")) != (COMPILE, None, 0):', 'f.get("train.ce_chunk_rows")) == 42:', RF),
    ("c1b_reverted_to_compiled", LIB, "COMPILE = False", 'COMPILE = "default"', CF),
    ("c6_hook_unchecked", LIB, 'if not hk or any(f.get(k, "<absent>") != v for k, v in hk.items()):', "if False:", RF),
    ("c6_hook_absent_from_base", "experiments/screens/configs/screens_base.yaml", "    every: 153\n", "    every: 0\n", CF),
    ("lr_embed_unscaled", LIB, '"optim.embed_lr": g * r * eta', '"optim.embed_lr": r * eta', CF),
    ("g_any_value", LIB, "if g not in G_ALLOWED:", "if False:", RF),
    ("k_floor_dropped", LIB, "k[cls] = min(5, max(2, need))", "k[cls] = min(5, need)", CF + "::parameters"),
    ("param_rule_removed", LIB, "if abs(n - BASE_PARAMS) > 0.02 * BASE_PARAMS:", "if False:", RF),
    ("cfg_for_first_match", "experiments/screens/qlib_screens.sh", '[ "$n" = 1 ] || return 1', '[ "$n" -ge 1 ] || return 1', CF + "::queue"),
    ("s003_extra_allowed", LIB, '"engine_fixed", "schedule.init_from"}', '"engine_fixed", "schedule.init_from", "optim.weight_decay"}', RF),
    ("s003_init_from_unchecked", LIB, "if init != want_i:", "if False:", RF),
    ("s003_branch_point_unchecked", LIB, "if init != want_i:",
     'if (init or "").rsplit("/", 1)[0] != (want_i or "").rsplit("/", 1)[0]:', RF),
    ("ind_file_unpinned", "experiments/screens/make_ind.py", "SEED, N, L = 20261003, 64, 128", "SEED, N, L = 20261004, 64, 128",
     CF + "::ind_file"),
    ("own_dir_unchecked", LIB, "    if os.path.basename(os.path.dirname(os.path.dirname(path))) != home:",
     "    if False:", RF),
    ("seeds_reversed", GEN, "s = 101 + i", "s = 102 - i", RF),
    ("ind_runs_dropped", GEN, 'if sid in ("S006", "S007") and g != 1.0:', "if False:", RF),
    ("stage2_priority_swapped", LIB, '2: ["S005", "S004", "S007", "S006"]', '2: ["S004", "S005", "S007", "S006"]', RF),
    ("s005_engine_not_in_reference", LIB, "    ref.update(engine(sid))\n", "", CF),
    ("harness_draws_when_qknorm_off", "harness/blocks.py", "        self.qk_norm = cfg.qk_norm\n",
     "        self.qk_norm = cfg.qk_norm\n        if not cfg.qk_norm:\n            torch.randn(1)\n", SF + "::isolation"),
    ("harness_refuses_mtp_1", "harness/mtp.py", "or mtp not in (0, 1):", "or mtp not in (0,):", CF + "::resolves"),
    ("flag_parity_drops_model_flags", FPY, "cfg=s.cfg.replace(**model)", "cfg=s.cfg", PF),
    ("flag_parity_drops_train_keys", FPY, 'over["train"] = {**over.get("train", {}), **train}',
     'over["train"] = dict(over.get("train", {}))', PF),
    ("flag_parity_grab_blind", FPY, "if any(w in m.lower() for w in LIMIT_WORDS):", "if False:", PF),
    ("flag_parity_arm_ignored", FPY, "return {**L.engine(sid), **arms[arm]}",
     "return {**L.engine(sid), **next(iter(arms.values()))}", PF),
    ("flag_parity_base_carries_arm", FPY, "        return L.engine(sid)\n",
     "        return {**L.engine(sid), **next(iter(arms.values()))}\n", PF),
    ("flag_parity_multi_arm_guess", FPY, 'assert len(arms) == 1, f"{sid} has arms', 'assert True, f"{sid} has arms', PF),
    ("flag_parity_model_flags_unchecked", FPY, 'return (got["model_flags"] == model', "return (True", PF),
    ("flag_parity_doc_attn_unchecked", FPY, 'got.get("doc_attn", "mask") == train["doc_attn"])', "True)", PF),
    ("flag_parity_mtp_unchecked", FPY, '(got.get("mtp") or {}).get("heads") == train["mtp"])', "True)", PF),
    ("flag_parity_mode_unchecked", FPY, 'and ("compile" in got) == (not r["arm"].startswith("eager")))', "and True)", PF),
    ("flag_parity_pass_ignores_flag", FPY, "rc == 0 and carried}", "rc == 0}", PF),
    ("flag_parity_pass_ignores_limit", FPY, 'bool(res["pass"]) and not msgs and', 'bool(res["pass"]) and', PF),
    ("smoke_scratch_unchecked", "experiments/screens/gate_smoke.py", "if not p.startswith(root):", "if False:", PF),
    ("smoke_reuses_out_dir", "experiments/screens/gate_smoke.py", "if os.path.isdir(out) and os.listdir(out):",
     "if False:", PF),
] + ANALYZER_MUTANTS
NF, S3B, S3C = ("test_screens_stage1_next.py", "experiments/screens/plans/stage1_s003B.txt",
                "experiments/S003_adamw/configs/s003_adamw_e3_")
MUTANTS += [   # STAGE 1 SELECTION A RESULT (2026-10-05): the stage B configs and plan written from the recorded picks
    ("s003B_plan_drops_r8", S3B, "train s003_adamw_e3_r8_b250M\n", "", NF),
    ("s003B_plan_runs_past_its_mark", S3B, "mark SCREENS S003 STAGE B DONE\n",
     "mark SCREENS S003 STAGE B DONE\ntrain base_s101\n", NF),
    ("s003B_plan_no_wait", S3B, "wait_mark SCREENS SCREENS STAGE 1 SELECTION A DONE\n", "", NF),
    ("s003B_config_wrong_r", S3C + "r4_b62M.yaml", "embed_lr: 0.012, scalar_lr: 0.012", "embed_lr: 0.006, scalar_lr: 0.006", NF),
    ("s003B_branch_off_another_trunk", S3C + "r2_b125M.yaml", "s003_adamw_e3_r2_trunk/", "s003_adamw_e3_r1_trunk/", NF),
    ("current_plan_out_of_sequence", "experiments/screens/plans/CURRENT", "# Seed plans come from screens.py seeds.\n",
     "# Seed plans come from screens.py seeds.\nstage2_select\n# ", CF + "::registered_order"),
]
HRS, HF, RO = "experiments/screens/screens_hours.py", "test_screens_hours.py", "test_s005_row_order.py"
MUTANTS += [   # STAGE 2 READINESS (2026-10-05): the hours count each run once, smoke factors, S003 measured arms
    ("hours_measured_double_counted", HRS, "        if key not in seen:   ", "        if True:   ", HF),
    ("hours_cap_ignores_measured", HRS, 'cap = AL.cap_cut(t["queued"], mr + mo)', 'cap = AL.cap_cut(t["queued"])', HF),
    ("hours_other_not_counted", HRS, 'cap = AL.cap_cut(t["queued"], mr + mo)', 'cap = AL.cap_cut(t["queued"], mr)', HF),
    ("hours_assumed_factors", HRS, 'return SMOKE["BASE"] / SMOKE[(sid, arm)] if (sid, arm) in SMOKE else 1.0',
     'return {"S004": 1.1, "S005": 1.1, "S006": 1.3}.get(sid, 1.0)', HF),
    ("hours_smoke_value_not_logged", HRS, '("S005", "forget"): 85_113', '("S005", "forget"): 85_131', HF),
    ("hours_own_base_unfactored", HRS, '(sid, "base", x): (sid, full(sid, "base"))', '(sid, "base", x): (sid, full(None, None))', HF),
    ("hours_s003_plus15", HRS, '("S003", tag_h[t]) for pt in pts', '("S003", run_h(8776) / 4) for pt in pts', HF),
    ("hours_stage_c_kept_at_rb1", HRS, 'pts += [] if r_b == 1 else [("C", g) for g in STAGE_C_G]',
     'pts += [("C", g) for g in STAGE_C_G]', HF),
    ("hours_extension_takes_a_slot", HRS, '(sid, m["arm"], "g", g) if g in (0.5, 1.0, 2.0) else None)',
     '(sid, m["arm"], "g", min(max(g, 0.5), 2.0)))', HF),
    ("hours_lock_from_wait", HRS, 'if msg.startswith("gpu.lock held"):', 'if msg.startswith("waiting for gpu.lock"):', HF),
    # S005's micro 8 x accum 2 row-order test (S005 notes ENGINE): harness mutants on the scratch copy only
    ("loader_window_rng_by_micro", "harness/data.py", "default_rng([self.seed, self._w, 11])",
     "default_rng([self.seed, self._w, 11, self.B])", RO),
    ("loader_micro_rows_reversed", "harness/data.py", "collate_rows([self._next_unit() for _ in range(self.B)])",
     "collate_rows([self._next_unit() for _ in range(self.B)][::-1])", RO),
    ("trainer_micro_batches_reversed", "harness/trainer.py", "[self.loader.next_batch() for _ in range(self.grad_accum)]",
     "[self.loader.next_batch() for _ in range(self.grad_accum)][::-1]", RO + "::trainer"),
    ("trainer_loss_per_micro_batch", "harness/trainer.py", "(obj / max(1, n_sup)).backward()",
     '(obj / max(1, int((b["tgt"] != -100).sum()))).backward()', RO + "::trainer"),
    ("s005_config_micro_4x4", "experiments/S005_forget_gate/configs/s005_forget_g2_s1.yaml",
     "micro_batch: 8, grad_accum: 2", "micro_batch: 4, grad_accum: 4", RO + "::16_rows"),
]
CF2, S3Cp, S3Cc = ("test_screens_s003_stage_c.py", "experiments/screens/plans/stage1_s003C.txt",
                   "experiments/S003_adamw/configs/s003_adamw_e")
MUTANTS += [   # S003 STAGE B RESULT (2026-10-05): stage C's configs and plan from r_B 2, the entry, the measured hours
    ("s003C_plan_drops_g2_b250M", S3Cp, "train s003_adamw_e6_r2_b250M\n", "", CF2),
    ("s003C_plan_runs_past_its_mark", S3Cp, "mark SCREENS S003 STAGE C DONE\n",
     "mark SCREENS S003 STAGE C DONE\ntrain base_s101\n", CF2),
    ("s003C_plan_waits_on_the_wrong_mark", S3Cp, "wait_mark SCREENS SCREENS S003 STAGE B DONE\n",
     "wait_mark SCREENS SCREENS STAGE 1 SELECTION A DONE\n", CF2),
    ("s003C_config_r1", S3Cc + "6_r2_b62M.yaml", "embed_lr: 0.012, scalar_lr: 0.012", "embed_lr: 0.006, scalar_lr: 0.006", CF2),
    ("s003C_config_g_not_applied", S3Cc + "1.5_r2_trunk.yaml", "lr: 0.0015, embed_lr", "lr: 0.003, embed_lr", CF2),
    ("s003C_branch_off_stage_b_trunk", S3Cc + "1.5_r2_b125M.yaml", "s003_adamw_e1.5_r2_trunk/", "s003_adamw_e3_r2_trunk/", CF2),
    ("s003_stage_c_read_at_r1", AN, 'out["C"] = e2pick.stage(runs, "C", a["pick"], b["pick"])',
     'out["C"] = e2pick.stage(runs, "C", a["pick"], 1.0)', CF2),
    ("s003B_entry_value_typo", "experiments/SCREENS.txt", "2     1.16436 / 1.39911", "2     1.16463 / 1.39911", CF2),
    ("hours_stage_b_run_missing", "experiments/screens/measured_hours.tsv",
     "run\ts003_adamw_e3_r8_b250M\t0.08528\t2026-10-05 12:56:41\t2026-10-05 13:01:48\tqueue_screens.txt\n", "", CF2),
    ("hours_stage_b_end_time_typo", "experiments/screens/measured_hours.tsv", "2026-10-05 12:37:24", "2026-10-05 12:39:24", CF2),
]
from mutants_s003c import MUTANTS_S003C  # noqa: E402  (S003 STAGE C RESULT: seed configs and plan, entry, hours)
MUTANTS += MUTANTS_S003C

def copy_tree(dst: str) -> None:
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    shutil.copytree(os.path.join(ROOT, "harness"), os.path.join(dst, "harness"), ignore=ign)
    for rel in ["experiments/E2_lr_transfer/configs", "experiments/E3_seed_noise/configs", "experiments/screens"] + \
            [os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "experiments", "S00?_*"))]:
        shutil.copytree(os.path.join(ROOT, rel), os.path.join(dst, rel), ignore=ign)
    shutil.copytree(os.path.join(ROOT, "data_prep"), os.path.join(dst, "data_prep"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "stats"))
    for rel in ("experiments/E2_lr_transfer/results.json", "experiments/E2_lr_transfer/preflight.py",
                "experiments/E2_lr_transfer/e2plan.py", "experiments/E2_lr_transfer/e2pick.py",
                "experiments/E3_seed_noise/results.json", "experiments/E3_seed_noise/e3analyze.py", "tokenizer/spec.py",
                "experiments/SCREENS.txt",
                "tokenizer/v0/tok_v0_8k.json", "tokenizer/v0/tok_v0_manifest.json", "corpus/oodh.py", "corpus/sample_plan.py"):
        os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
        shutil.copy(os.path.join(ROOT, rel), os.path.join(dst, rel))


def run_tests(dst: str, spec: str) -> int:
    f, _, k = spec.partition("::")
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
           os.path.join(dst, "experiments", "screens", "tests", f)] + (["-k", k] if k else [])
    return subprocess.run(cmd, cwd=dst, capture_output=True, text=True, timeout=900).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args(argv)
    work = a.scratch or tempfile.mkdtemp(prefix="screens_mut_")
    todo = [m for m in MUTANTS if not a.only or m[0] in a.only.split(",")]
    clean = os.path.join(work, "clean")
    shutil.rmtree(clean, ignore_errors=True)
    copy_tree(clean)
    for spec in sorted({m[4] for m in todo}):
        rc = run_tests(clean, spec)
        print(f"unmutated {spec}: {'pass' if rc == 0 else 'FAIL rc %d' % rc}", flush=True)
        if rc != 0:
            return 2
    survived = []
    for name, rel, old, new, spec in todo:
        dst = os.path.join(work, name)
        shutil.rmtree(dst, ignore_errors=True)
        copy_tree(dst)
        p = os.path.join(dst, rel)
        txt = open(p).read()
        if txt.count(old) != 1:
            print(f"INVALID {name}: the old text occurs {txt.count(old)} times in {rel}")
            survived.append(name)
            continue
        open(p, "w").write(txt.replace(old, new))
        rc = run_tests(dst, spec)
        print(f"{'killed  ' if rc != 0 else 'SURVIVED'} {name} ({spec}, rc {rc})", flush=True)
        if rc == 0:
            survived.append(name)
        shutil.rmtree(dst, ignore_errors=True)
    print(f"{len(todo) - len(survived)} of {len(todo)} killed" + (f"; survived: {survived}" if survived else ""))
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
