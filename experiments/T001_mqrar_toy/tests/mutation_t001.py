"""Mutation check for the T001 tests, on scratch copies only (the repo is never edited).

  ~/.venvs/planck/bin/python experiments/T001_mqrar_toy/tests/mutation_t001.py [--only NAME,...] [--workers 4]

Copies experiments/T001_mqrar_toy to a fresh scratch tree per mutant, applies ONE edit (the old text must occur
exactly once), runs the named test file there and wants it to FAIL. The unmutated copy must pass every test file
first. Exit 0 = every mutant killed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
T001 = os.path.dirname(HERE)
TT, TL, TL2, TM, TR, TTR = (f"tests/{f}.py" for f in ("test_task", "test_layers", "test_layers2", "test_metrics",
                                                       "test_runplan", "test_train"))
TRS = "tests/test_results.py"  # results.py and the independent rederive.py (2026-10-11)
ALL = "tests"                  # the whole suite (the critique's mutants and amendment A1's, 2026-10-10)
MUTANTS = [   # name, file, old, new, test file
    ("ask_writes_first_value", "task.py", "toks += [ASK, KEY0 + k, VAL0 + cur[k]]", "toks += [ASK, KEY0 + k, VAL0 + hist[k][0]]", TT),
    ("ans_records_first_value", "task.py", "ans=VAL0 + cur[k], key=k", "ans=VAL0 + hist[k][0], key=k", TT),
    ("update_may_repeat", "task.py", "v = v + (v >= cur[k])", "v = v", TT),
    ("update_skip_off_by_one", "task.py", "v = v + (v >= cur[k])", "v = v + (v > cur[k])", TT),
    ("copy_reversed", "task.py", "write(b, cur[a], True)", "write(a, cur[b], True)", TT),
    ("loss_mask_off_by_one", "task.py", 'y[b, q["pos"]] = q["ans"]', 'y[b, q["pos"] + 1] = q["ans"]', TT),
    ("pos_on_value_token", "task.py", "pos=len(toks) - 2", "pos=len(toks) - 1", TT),
    ("updated_after_one_write", "task.py", "updated=len(h) >= 2", "updated=len(h) >= 1", TT),
    ("stale_includes_current", "task.py", "stale=sorted({VAL0 + x for x in h if x != cur[k]})", "stale=sorted({VAL0 + x for x in h})", TT),
    ("query_rate_0p6", "task.py", 'if family == "ar" or r < 0.5:', 'if family == "ar" or r < 0.6:', TT),
    ("dev_stream_equals_test", "task.py", '"dev": 8_000_000}', '"dev": 9_000_000}', TT),
    ("initial_block_drops_a_key", "task.py", "for k in rng.permutation(keys):", "for k in rng.permutation(keys)[1:]:", TT),
    ("delta_alpha_on_write", "layers.py", "M = a * (M - b * kt[..., :, None] * kM[..., None, :]) + b * kt[..., :, None] * vt[..., None, :]",
     "M = a * (M - b * kt[..., :, None] * kM[..., None, :] + b * kt[..., :, None] * vt[..., None, :])", TL),
    ("delta_no_erase", "layers.py", "M = a * (M - b * kt[..., :, None] * kM[..., None, :])", "M = a * M", TL),
    ("delta_erase_without_beta", "layers.py", "M = a * (M - b * kt[..., :, None] * kM[..., None, :])", "M = a * (M - kt[..., :, None] * kM[..., None, :])", TL),
    # an inclusive mask in A is an EQUIVALENT mutant (solve_triangular(unitriangular=True) never reads A's diagonal),
    # found 2026-10-10; replaced by a sign error in the triangular system
    ("chunk_system_sign", "layers.py", "solve_triangular(eye + A, rhs", "solve_triangular(eye - A, rhs", TL),
    ("chunk_m0_read_undecayed", "layers.py", "out.append(g * (Q @ M) +", "out.append((Q @ M) +", TL),
    ("chunk_end_state_undecayed", "layers.py", "M = G[..., -1].exp()[..., None, None] * M +", "M = M +", TL),
    ("sb_product_includes_j", "layers.py", "torch.flip(lnb, [-1]), -1), [-1]) - lnb", "torch.flip(lnb, [-1]), -1), [-1])", TL),
    ("sb_future_unmasked", "layers.py", "lnb = F.logsigmoid(-z).masked_fill(~causal, 0.0)", "lnb = F.logsigmoid(-z)", TL),
    ("sa_no_shift", "layers.py", "    S = F.pad(S, (0, 0, 1, 0))[..., :-1, :]\n", "", TL),
    ("sa_bos_maskable", "layers.py", "keep[:, 0] = False", "keep[:, 0] = True", TL),
    ("fg_bias_sign", "layers.py", "s = s + (c[..., :, None] - c[..., None, :])", "s = s - (c[..., :, None] - c[..., None, :])", TL),
    ("fg_sum_includes_j", "layers.py", "c = F.logsigmoid(x @ self.fw.t() + self.fb).cumsum(1).transpose(1, 2)",
     "c = (F.logsigmoid(x @ self.fw.t() + self.fb).cumsum(1) - F.logsigmoid(x @ self.fw.t() + self.fb)).transpose(1, 2)", TL),
    ("alibi_sign", "layers.py", "s = s - F.softplus(self.ap)", "s = s + F.softplus(self.ap)", TL),
    ("alibi_init_slopes", "layers.py", "2.0 ** (-8.0 * (i + 1) / n_heads)", "2.0 ** (-8.0 * i / n_heads)", TL),
    ("conv_centred", "layers.py", "xp = F.pad(x.transpose(1, 2), (self.k - 1, 0))", "xp = F.pad(x.transpose(1, 2), (self.k // 2, self.k - 1 - self.k // 2))", TL),
    ("conv_kernel_unflipped", "layers.py", "self.weight.flip(-1).unsqueeze(1)", "self.weight.unsqueeze(1)", TL),
    ("smear_next_token", "layers.py", "return F.pad(x, (0, 0, 1, 0))[..., :-1, :]", "return F.pad(x, (0, 0, 0, 1))[..., 1:, :]", TL),
    ("canon_init_draws_rng", "model.py", "elif isinstance(mod, CausalConv) and not mod.zero:", "elif isinstance(mod, CausalConv):", TL),
    ("gdn_k_not_normalised", "layers.py", "q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)", "q = F.normalize(q, dim=-1)", TL2),
    ("gdn_silu_dropped", "layers.py", "u = F.silu(self.conv(u) if self.conv is not None else u)", "u = self.conv(u) if self.conv is not None else u", TL2),
    ("gdn_gates_swapped", "layers.py", "y = delta_chunked(q, k, v, F.logsigmoid(a), torch.sigmoid(b), self.chunk)",
     "y = delta_chunked(q, k, v, F.logsigmoid(b), torch.sigmoid(a), self.chunk)", TL2),
    ("gdn_bias_on_beta", "model.py", "mod.gates.bias[: mod.h] =", "mod.gates.bias[mod.h:] =", TL2),
    ("gdn_layer_first", "model.py", '"gdn": (["base", "gdn"], False)', '"gdn": (["gdn", "base"], False)', TL2),
    ("rope_on_q_only", "layers.py", "q, k = rope(q), rope(k)", "q, k = rope(q), k", TL2),
    ("sb_with_rope", "layers.py", 'if self.kind != "sb":\n            q, k', 'if True:\n            q, k', TL2),
    ("fg_half_life_window", "model.py", "FG_LEN = 865", "FG_LEN = 2048", TL2),
    ("capacity_not_contiguous", "metrics.py", "        if a is None or a < THR_ACC:\n            break", "        if a is None or a < THR_ACC:\n            continue", TM),
    ("min_sign_3", "metrics.py", "MIN_SIGN = 4", "MIN_SIGN = 3", TM),
    ("help_lw_0p04", "metrics.py", "HELP_LW, HELP_CAP, GUARD = 0.05, 1.25, -0.05", "HELP_LW, HELP_CAP, GUARD = 0.04, 1.25, -0.05", TM),
    ("help_cap_1p2", "metrics.py", "HELP_LW, HELP_CAP, GUARD = 0.05, 1.25, -0.05", "HELP_LW, HELP_CAP, GUARD = 0.05, 1.2, -0.05", TM),
    ("guard_0p1", "metrics.py", "HELP_LW, HELP_CAP, GUARD = 0.05, 1.25, -0.05", "HELP_LW, HELP_CAP, GUARD = 0.05, 1.25, -0.1", TM),
    ("pass_cap_1p4", "metrics.py", "PASS_CAP, PASS_LW_ABS, PASS_LW_GAIN = 1.5, 0.8, 0.10", "PASS_CAP, PASS_LW_ABS, PASS_LW_GAIN = 1.4, 0.8, 0.10", TM),
    ("pass_lw_abs_0p75", "metrics.py", "PASS_CAP, PASS_LW_ABS, PASS_LW_GAIN = 1.5, 0.8, 0.10", "PASS_CAP, PASS_LW_ABS, PASS_LW_GAIN = 1.5, 0.75, 0.10", TM),
    ("pass_lw_gain_0p05", "metrics.py", "PASS_CAP, PASS_LW_ABS, PASS_LW_GAIN = 1.5, 0.8, 0.10", "PASS_CAP, PASS_LW_ABS, PASS_LW_GAIN = 1.5, 0.8, 0.05", TM),
    ("r1_neither_to_p022", "metrics.py", 'return dict(pick="P-148", why="neither helps', 'return dict(pick="P-022", why="neither helps', TM),
    ("r1_tie_to_p148", "metrics.py", 'return dict(pick="P-022", why="both help and tie', 'return dict(pick="P-148", why="both help and tie', TM),
    ("r2_order_sa_sb_swapped", "metrics.py", 'R2_ORDER = ("fg", "alibi", "sa", "sb")', 'R2_ORDER = ("fg", "alibi", "sb", "sa")', TM),
    ("r2_default_sb", "metrics.py", 'return dict(pick="fg", why="no variant', 'return dict(pick="sb", why="no variant', TM),
    ("r3_gap_0p1", "metrics.py", "R3_GAP, R3_NO, R3_MARGIN = 0.2, 0.05, 0.05", "R3_GAP, R3_NO, R3_MARGIN = 0.1, 0.05, 0.05", TM),
    ("r3_margin_0p1", "metrics.py", "R3_GAP, R3_NO, R3_MARGIN = 0.2, 0.05, 0.05", "R3_GAP, R3_NO, R3_MARGIN = 0.2, 0.05, 0.1", TM),
    ("diverged_lw_not_worst", "metrics.py", "return dict(cap=CAP_FLOOR, lwo=0.0,", "return dict(cap=CAP_FLOOR, lwo=0.5,", TM),
    ("lw_ns_three", "metrics.py", "LW_NS = (8, 16, 24, 32)", "LW_NS = (8, 16, 24)", TM),
    ("lr_tie_to_larger", "metrics.py", "key=lambda lr: (-score(lr), lr))", "key=lambda lr: (-score(lr), -lr))", TM),
    ("c0_a_threshold_0p3", "runplan.py", "C0_A_STEPS, C0_A_ACC8, C0_B_ACC = 12000, 0.5, 0.9", "C0_A_STEPS, C0_A_ACC8, C0_B_ACC = 12000, 0.3, 0.9", TR),
    ("c0_b_threshold_0p96", "runplan.py", "C0_A_STEPS, C0_A_ACC8, C0_B_ACC = 12000, 0.5, 0.9", "C0_A_STEPS, C0_A_ACC8, C0_B_ACC = 12000, 0.5, 0.96", TR),
    ("grid_ext_one_point", "runplan.py", "GRID_EXT = (128, 160)", "GRID_EXT = (128,)", TR),
    ("ext_missing_upper", "runplan.py", "EXT = {3e-4: 1e-4, 3e-3: 1e-2}", "EXT = {3e-4: 1e-4}", TR),
    ("ext_not_in_pick", "runplan.py", "    return metrics.lr_pick(dev, st[\"grid\"]), None", "    return first, None", TR),
    ("verdict_arm_major", "runplan.py", "    for seed in metrics.SEEDS:\n        for arm in PICK_ARMS + DIAG_ARMS:",
     "    for arm in PICK_ARMS + DIAG_ARMS:\n        for seed in metrics.SEEDS:", TR),
    ("track_two_seeds", "runplan.py", "TRACK_ARMS, TRACK_SEEDS = (\"base\", \"gdn\", \"canon\"), (1, 2, 3)", "TRACK_ARMS, TRACK_SEEDS = (\"base\", \"gdn\", \"canon\"), (1, 2)", TR),
    ("score_full_vocab_argmax", "train.py", "[..., VAL0:VAL0 + V]", "[..., :]", TTR),
    ("score_first_counts_unmoved", "train.py", 'c["first"] += p == q["first"] and q["first"] != q["ans"]', 'c["first"] += p == q["first"]', TTR),
    ("warmup_off", "train.py", "    if step < WARMUP:\n        return lr * (step + 1) / WARMUP\n", "", TTR),
    ("data_stream_per_arm", "train.py", "    rng = train_rng(a.seed)\n", "    rng = train_rng(a.seed + (a.arm != \"base\"))\n", TTR),
] + [(n, f, o, w, ALL) for n, f, o, w in [
    # critique D3: R2's tie windows and the censoring label
    ("tie_lw_0p01", "metrics.py", "TIE_LW, TIE_CAP = 0.02, 1.05", "TIE_LW, TIE_CAP = 0.01, 1.05"),
    ("tie_lw_0p03", "metrics.py", "TIE_LW, TIE_CAP = 0.02, 1.05", "TIE_LW, TIE_CAP = 0.03, 1.05"),
    ("tie_cap_1p10", "metrics.py", "TIE_LW, TIE_CAP = 0.02, 1.05", "TIE_LW, TIE_CAP = 0.02, 1.10"),
    ("tie_cap_1p02", "metrics.py", "TIE_LW, TIE_CAP = 0.02, 1.05", "TIE_LW, TIE_CAP = 0.02, 1.02"),
    ("r2_cap_tie_strict", "metrics.py", "gms[x] * TIE_CAP >= best - EPS", "gms[x] >= best - EPS"),
    ("censored_never", "metrics.py", "censored=cap == max(grid)", "censored=False"),
    # critique D4: arm forms and registered constants
    ("sa_selects_head1", "layers.py", "selective_mask(s[:, 0], causal)", "selective_mask(s[:, 1], causal)"),
    ("canon_a_reads_x", "model.py", "h = h + self.canon_a(h)", "h = h + self.canon_a(x)"),
    ("canon_c_reads_x", "model.py", "h = h + self.canon_c(h)", "h = h + self.canon_c(x)"),
    ("no_out_proj_scaling", "model.py", "mod.weight.normal_(0.0, out_std if last else 0.02)", "mod.weight.normal_(0.0, 0.02)"),
    ("weight_decay_zero", "train.py", "WARMUP, WD, CLIP = 200, 0.1, 1.0", "WARMUP, WD, CLIP = 200, 0.0, 1.0"),
    ("clip_2", "train.py", "WARMUP, WD, CLIP = 200, 0.1, 1.0", "WARMUP, WD, CLIP = 200, 0.1, 2.0"),
    ("no_grad_clip", "train.py", "    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)\n", ""),
    ("adam_beta2_0p999", "train.py", "BETAS, ADAM_EPS = (0.9, 0.95), 1e-8", "BETAS, ADAM_EPS = (0.9, 0.999), 1e-8"),
    ("adam_eps_1e-6", "train.py", "BETAS, ADAM_EPS = (0.9, 0.95), 1e-8", "BETAS, ADAM_EPS = (0.9, 0.95), 1e-6"),
    ("wd_on_vectors", "train.py", '{"params": other, "weight_decay": 0.0}', '{"params": other, "weight_decay": WD}'),
    ("train_n_excludes_top", "train.py", "int(rng.integers(lo, hi + 1))", "int(rng.integers(lo, hi))"),
    ("eval_test_size_128", "task.py", 'SPLIT_SIZE = {"test": 256, "dev": 128}', 'SPLIT_SIZE = {"test": 128, "dev": 128}'),
    ("divergence_never_detected", "train.py", "if not torch.isfinite(loss):", "if False:"),
    ("tf32_matmul_on", "train.py", "torch.backends.cuda.matmul.allow_tf32 = False", "torch.backends.cuda.matmul.allow_tf32 = True"),
    ("tf32_cudnn_on", "train.py", "torch.backends.cudnn.allow_tf32 = False", "torch.backends.cudnn.allow_tf32 = True"),
    # critique D3/D4 mutants that the first suite already killed, kept here as the record
    ("track_copy_rate_0p6", "task.py", 'elif family == "track" and r < 0.75:', 'elif family == "track" and r < 0.6:'),
    ("dist_off_by_one", "task.py", "dist=ev - last[k], outvoted", "dist=ev - last[k] - 1, outvoted"),
    ("copy_not_flagged_tracked", "task.py", "write(b, cur[a], True)", "write(b, cur[a], False)"),
    ("gdn_no_out_norm", "layers.py", "return self.o(self.onorm(y).transpose(1, 2)", "return self.o(y.transpose(1, 2)"),
    ("sa_diag_maskable", "layers.py", "    keep.fill_diagonal_(False)\n", ""),
    ("cosine_floor_zero", "train.py", "lr * (0.1 + 0.9 * 0.5 *", "lr * (0.0 + 1.0 * 0.5 *"),
    ("guard_reads_ar16", "metrics.py", 't["ar"]["32"]["acc"]', 't["ar"]["16"]["acc"]'),
    ("r3_yes_vs_base1", "metrics.py", 'm_b2 = sum(tab["base"][s]["ar32"]', 'm_b2 = sum(tab["base1"][s]["ar32"]'),
    ("r1_h2h_cap_1p5", "metrics.py", 'w = cap_gain(tab, "gdn", "canon", HELP_CAP)', 'w = cap_gain(tab, "gdn", "canon", PASS_CAP)'),
    ("r1_h2h_lw_0p10", "metrics.py", 'w = lw_gain(tab, "gdn", "canon", HELP_LW)', 'w = lw_gain(tab, "gdn", "canon", PASS_LW_GAIN)'),
    ("c0_at_lr_3e-4", "runplan.py", 'js = load(out, name("base", "mqrar", 0, 1e-3, st))', 'js = load(out, name("base", "mqrar", 0, 3e-4, st))'),
    ("eps_zero", "metrics.py", "EPS = 1e-9 ", "EPS = 0.0 "),
    ("cap_floor_8", "metrics.py", "CAP_FLOOR = 4 ", "CAP_FLOOR = 8 "),
    ("r2_none_pass_skip_helps", "metrics.py", "    if h:\n        return dict(pick=_best(tab, h)", "    if False:\n        return dict(pick=_best(tab, h)"),
    ("guard_dropped_in_passes", "metrics.py", "return (by_cap or by_lw) and guard(tab, x)", "return (by_cap or by_lw)"),
    ("r1_both_skip_cap_h2h", "metrics.py", '    w = cap_gain(tab, "gdn", "canon", HELP_CAP)\n', "    w = 0\n"),
    ("lw_abs_strict", "metrics.py", "tab[x][s][LW_KEY] >= PASS_LW_ABS - EPS", "tab[x][s][LW_KEY] > PASS_LW_ABS + EPS"),
    # amendment A1: the outvoted flag, its scoring, and the rules that read it
    ("outvoted_includes_ties", "task.py", "outvoted=rival > mine", "outvoted=rival >= mine"),
    # "tied=0 < rival == mine" was an EQUIVALENT form (every key has an explicit first SET, so rival == mine == 0
    # never happens; the recount test asserts that); the code reads "tied=rival == mine" since 2026-10-10
    ("tied_includes_outvoted", "task.py", "tied=rival == mine", "tied=rival >= mine"),
    ("ask_does_not_vote", "task.py", "            vote(k, cur[k])\n", ""),
    ("update_does_not_vote", "task.py", "            write(k, v, False)\n            vote(k, v)\n", "            write(k, v, False)\n"),
    ("initial_set_does_not_vote", "task.py", "        write(k, v, False)\n        vote(k, v)\n", "        write(k, v, False)\n"),
    ("query_votes_before_scoring", "task.py", "            mine = votes[k].get(cur[k], 0)", "            vote(k, cur[k])\n            mine = votes[k].get(cur[k], 0)"),
    ("score_ov_counts_updated", "train.py", 'if q["outvoted"]:', 'if q["updated"]:'),
    ("score_tie_counts_outvoted", "train.py", 'if q["tied"]:', 'if q["outvoted"]:'),
    ("cap_ignores_acc_ov", "metrics.py", "cap = capacity(acc, grid, acc_ov)", "cap = capacity(acc, grid)"),
    ("capacity_ov_none_passes", "metrics.py", "if o is None or o < THR_ACC:", "if o is not None and o < THR_ACC:"),
    ("lwo_reads_lw", "metrics.py", 'lwo = sum(t["mqrar"][str(n)]["acc_ov"]', 'lwo = sum(t["mqrar"][str(n)]["lw"]'),
    ("pass_cap_path_without_lw", "metrics.py", "by_cap = lw_abs and cap_gain", "by_cap = cap_gain"),
    ("lr_pick_acc_only", "metrics.py", '+ ((d.get(n) or {}).get("acc_ov") or 0.0)', '+ ((d.get(n) or {}).get("acc") or 0.0)'),
    ("c0b_acc_only", "runplan.py", ' and _get(d, n, "acc_ov") >= C0_B_ACC', ""),
    ("dev_acc_ov_from_acc", "runplan.py", 'acc_ov=r["acc_ov"]', 'acc_ov=r["acc"]'),
]] + [(n, f, o, w, TRS) for n, f, o, w in [
    # rederive.py (the independent reading) must disagree with the module on some fake; results.py prints the picks
    ("rd_guard_0p10", "rederive.py", "/ 5 >= -0.05 - TOL", "/ 5 >= -0.10 - TOL"),
    ("rd_ov_from_upd", "rederive.py", 'ov = {n: rate(t["mqrar"][str(n)]["ov_ok"], t["mqrar"][str(n)]["ov_q"])',
     'ov = {n: rate(t["mqrar"][str(n)]["upd_ok"], t["mqrar"][str(n)]["upd_q"])'),
    ("rd_c0a_threshold", "rederive.py", "[0] or 0) < 0.5:", "[0] or 0) < 0.25:"),
    ("rd_c0b_acc_only", "rederive.py", " and (d.get(n, (0, 0))[1] or 0) >= 0.9 for n in G0)", " for n in G0)"),
    ("rd_lr_no_ext", "rederive.py", "ext = {3e-4: 1e-4, 3e-3: 1e-2}.get(first)", "ext = None"),
    ("rd_lr_tie_larger", "rederive.py", "return min(lr for lr in lrs if score(lr) == top)",
     "return max(lr for lr in lrs if score(lr) == top)"),
    ("rd_r2_order", "rederive.py", 'order = ["fg", "alibi", "sa", "sb"]', 'order = ["fg", "sb", "sa", "alibi"]'),
    ("rd_r1_tie_to_p148", "rederive.py", 'r1 = "P-148" if w < 0 else "P-022"', 'r1 = "P-022" if w > 0 else "P-148"'),
    ("rd_cap_floor_8", "rederive.py", "        c = 4\n", "        c = 8\n"),
    ("rd_lwo_three_n", "rederive.py", "lwo = sum(ov[n] or 0.0 for n in (8, 16, 24, 32)) / 4",
     "lwo = sum(ov[n] or 0.0 for n in (8, 16, 24)) / 4"),
    ("rd_r3_gap_0p3", "rederive.py", "sum(g >= 0.2 - TOL for g in gap) >= 4", "sum(g >= 0.3 - TOL for g in gap) >= 4"),
    ("rd_r3_margin_0p15", "rederive.py", "m1 >= m2 - 0.05 - TOL", "m1 >= m2 - 0.15 - TOL"),
    ("rd_pass_abs_0p9", "rederive.py", 't[x][s]["lwo"] >= 0.8 - TOL', 't[x][s]["lwo"] >= 0.9 - TOL'),
    ("rd_censored_never", "rederive.py", "censored=c == grid[-1]", "censored=False"),
    ("rd_diverged_ar32", "rederive.py", "dict(cap=4, lwo=0.0, ar32=0.0,", "dict(cap=4, lwo=0.0, ar32=1.0,"),
    ("rd_helps_cap_1p5", "rederive.py", 'cap_beats(t, x, "base", 1.25) == 1) and guard(t, x)',
     'cap_beats(t, x, "base", 1.5) == 1) and guard(t, x)'),
    ("rd_h2h_no_cap", "rederive.py", ' or cap_beats(t, "gdn", "canon", 1.25)', ""),
    ("rd_r2_skip_helps", "rederive.py", 'r2 = choose(t, p) if p else choose(t, h) if h else "fg"',
     'r2 = choose(t, p) if p else "fg"'),
    ("res_stale_over_q", "results.py", "pooled(j, 'stale', 'upd_q')", "pooled(j, 'stale', 'q')"),
    ("res_first_is_stale", "results.py", "pooled(j, 'first', 'upd_q')", "pooled(j, 'stale', 'upd_q')"),
    ("res_r1_line_prints_r2", "results.py", 'rows.append(f"  R1 PICK: {M.r1(tab)', 'rows.append(f"  R1 PICK: {M.r2(tab)'),
    ("res_picks_r2_fixed", "results.py", 'R2=M.r2(tab)["pick"]', 'R2="fg"'),
    ("res_no_censor_star", "results.py", "{'*' if tab[a][s]['censored'] else ''}", ""),
    ("res_no_distance_line", "results.py", '    rows.append("  Accuracy by distance (registered as reported only): NOT RECORDED.', '    rows.append("'),
    ("res_write_skipped", "results.py", "            f.write(text)", "            pass"),
]] + [(n, f, o, w, "tests/test_results_real.py") for n, f, o, w in [
    # RESULTS.txt against the committed raw outputs (2026-10-11)
    ("real_results_number", "RESULTS.txt", "  base     423040   0.067", "  base     423040   0.068"),
    ("real_sha_list", "runs.sha256", "cd129565e854fff7", "0d129565e854fff7"),
    ("real_picks_line", "RESULTS.txt", 'PICKS {"R1": "P-022"', 'PICKS {"R1": "P-148"'),
]]


def run_tests(root: str, test: str) -> int:
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", test], cwd=root,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def fresh(scratch: str, tag: str) -> str:
    d = os.path.join(scratch, tag)
    shutil.copytree(T001, d, ignore=shutil.ignore_patterns("__pycache__", "out", "*.json"))
    return d


def one(scratch: str, m) -> tuple[str, str]:
    name, f, old, new, test = m
    d = fresh(scratch, name)
    p = os.path.join(d, f)
    src = open(p).read()
    if src.count(old) != 1:
        return name, f"BAD ANCHOR ({src.count(old)} matches)"
    open(p, "w").write(src.replace(old, new))
    rc = run_tests(d, test)
    return name, "killed" if rc != 0 else "SURVIVED"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    todo = [m for m in MUTANTS if not a.only or m[0] in a.only.split(",")]
    scratch = tempfile.mkdtemp(prefix="t001_mut_")
    try:
        base = fresh(scratch, "_unmutated")
        for t in sorted({m[4] for m in todo}):
            if run_tests(base, t) != 0:
                print(f"UNMUTATED COPY FAILS {t}: stop")
                return 2
        with ThreadPoolExecutor(max_workers=min(4, a.workers)) as ex:
            res = list(ex.map(lambda m: one(scratch, m), todo))
        for name, r in res:
            print(f"{r:10s} {name}")
        bad = [n for n, r in res if r != "killed"]
        print(f"{len(res) - len(bad)} of {len(res)} killed")
        return 1 if bad else 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
