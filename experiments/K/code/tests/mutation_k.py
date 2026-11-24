"""Mutation check for the track K code: one deliberate bug per scratch copy of experiments/K/code; KILLED = the named
test files fail (pytest exit 1). A target string that does not occur exactly once, or any other pytest exit code,
is INVALID. The unmutated copy runs first and must be green.
  python tests/mutation_k.py [--id NAME] [--list] [--baseline]   (a mutant runs its named test files only, under
  600 s since K3 round 3, the machine being shared and test_masks' restarts taking 4 min; --baseline runs every file
  the mutants name, up to 1,200 s). K3 round 4: 139 mutants (20 new); K3 round 5: 154 (15 new; 2 retargeted);
  K3 round 6: 174 (27 new, 7 retired with the round-5 R, U and A code they targeted, 8 retargeted). K3 round 7:
  193 (19 new, 5 retargeted). K3 round 8: 204 (11 new; REVIEW 9b's three survivors among them). K3 round 9: 209
  (5 new; REVIEW 10b's three survivors among them). K3 round 10: 215 (6 new: REVIEW 11b's T-1 to T-5 survivors).
  K3 round 11: 222 (7 new: REVIEW 12b's T-1 to T-3 and T-5 to T-8 survivors). K3 round 12: 229 (7 new: REVIEW
  13b's T-1 to T-5 survivors). K3 round 13: 238 (9 new: REVIEW 14b's six
  T-1 to T-5 survivors, its killed one and two of the round's own). G-2: 243 (5 new: the DFAR pad, the OOD read
  of SPEC 4's per-item rules and osize). K3 round 14: 251 (8 new: REVIEW 15b's T-1, T-7 and T-8 survivors and
  three of the round's own). K3 round 15: 271 (20 new: REVIEW 16b's T-1 to T-5 survivors of these files, its
  r16_lang_share_unnormalized, and four of the round's own).
  K3 round 9 (REVIEW 10b R-1: a full run stopped when the Mac slept): --log FILE appends each mutant's verdict line as
  it finishes (flushed and synced) and, on a restart with the same FILE, skips every mutant the file already holds as
  killed or SURVIVED (an INVALID one, a timeout among them, runs again). Run it under caffeinate -i.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
W, SK, PU, PR, AB, GE, RF, MK, CU = ("test_pools_world.py", "test_skill.py", "test_purity.py", "test_probes.py",
                                     "test_ablate.py", "test_gen.py", "test_reference.py", "test_masks.py",
                                     "test_cues.py")
_ACTX = 'if (fl.get("ctx") or ("R" if r.random() < 0.375 else "U")) == "R":    # a block flag (skill.block_flags)\n'
_SIG = ("steps: int = 2000, lr: float = 0.05,\n        lam: float = 6.0, floors=(None, None), seed: int = 0, "
        "layers=None, eps: float = 1e-6,\n        init: float = 0.0, per_budget: bool = True, restarts: int = 2)")
M = [
    ("fl_keeps_one_high_entity", "world.py", "bio[bio >= keep] = -1", "bio[bio > keep] = -1", [W],
     "FL keeps FB entity N_LOW: a non-FB_low fact leaks into the FL stream"),
    ("filler_depends_on_arm", "world.py", 'rf, ro = K.rng("filler_bio", shard), K.rng("bio_order", shard)',
     'rf, ro = K.rng("filler_bio", shard + int((ents >= 0).sum())), K.rng("bio_order", shard)', [W],
     "filler content differs between arms at shared slots"),
    ("fixed_pair_order", "world.py", "order = np.argsort(ro.random((n, K.N_ATTR)), axis=1)",
     "order = np.tile(np.arange(K.N_ATTR), (n, 1))", [W], "every exposure lists the pairs in one order"),
    ("fb_outside_region", "world.py", 'pos = r.choice(lay["bio_region"], lay["bio_exposures"], replace=False)',
     'pos = r.choice(lay["bio_slots"], lay["bio_exposures"], replace=False)', [W],
     "FB exposures land in the undrawn tail"),
    ("miss_line_no_none", "lookup.py", "return [K.LOOKUP, *[int(x) for x in name], K.LOOKUP_E, K.RESULT, K.NONE, "
     "K.RESULT_E]", "return [K.LOOKUP, *[int(x) for x in name], K.LOOKUP_E, K.RESULT, K.RESULT_E]", [W],
     "the miss line loses its none token"),
    ("k5_same_attribute_twice", "lookup.py", "a2.append(perms[:, 1::2].ravel())", "a2.append(perms[:, 0::2].ravel())",
     [W], "a K5 chat asks one attribute twice"),
    ("first_statement_wins", "items.py", "state[(s.ent, s.attr)] = s.val", "state.setdefault((s.ent, s.attr), s.val)",
     [SK], "the world rule keeps the first value, not the latest"),
    ("ellipsis_unresolved", "oracles.py", 's["ent"] = s["name"] or (prev_ent if s["kind"] == "ell" else',
     's["ent"] = s["name"] or (None if s["kind"] == "ell" else', [SK], "IDEAL cannot follow an ellipsis"),
    ("b_small_items_star", "fams.py", "    names = [cube[int(i)] for i in r.permutation(8)[:k]]",
     "    names = [cube[int(i)] for i in r.permutation(8)[:k]] if k == 8 else [cube[0], cube[1], cube[2], cube[4]][:k]",
     [SK], "K3 round 2 (replaces bind_foils_miss_a_subset, whose target left with the build's B): B's 2-4 name items "
     "are the asked name and its neighbours, the asked name their centre"),
    ("kevs_in_training", "skill.py", "            if t not in self.kevs:\n", "            if True:\n", [SK],
     "K-EVAL-S triples appear in training items"),
    ("o2_reads_the_name", "oracles.py", 'out["O2"] = _last(S, lambda s: s["attr"] == a)',
     'out["O2"] = _last(S, lambda s: s["attr"] == a and s["ent"] == nm)', [SK],
     "O2 binds by name by mistake: it reads 1.0 on clean R items"),
    ("loose_tolerance", "oracles.py", "return rate <= t + 3 * math.sqrt", "return rate <= t + 30 * math.sqrt", [PU],
     "a raised shortcut rate passes"),
    ("kevs_check_off", "purity.py", 'bad["kevs_triple"] += any(', 'bad["kevs_triple"] += 0 * any(', [PU],
     "the K-EVAL-S purity check never fires"),
    ("tie_gives_full_credit", "probes.py", '"lik": float(ties[gi]) / float(ties.sum())', '"lik": float(ties[gi])',
     [PR], "a uniform model scores 100%"),
    ("gen_not_strict", "probes.py",
     'o["gen"] = o["greedy"] == o["gold"] and (ends.get(j, False) or not pr[j]["chat"])',
     'o["gen"] = o["greedy"] == o["gold"]', [PR], "GEN ignores the <|end|> check"),
    ("last_column_under_padding", "probes.py",
     "return logits[torch.arange(len(prompts), device=device), last].float().cpu()",
     "return logits[:, -1].float().cpu()", [PR], "short prompts are read at a padding position"),
    ("reset_keeps_down_column", "ablate.py", "mlp.down_proj.weight[:, j] = 0.0", "mlp.down_proj.weight[:, j] *= 1.0",
     [AB], "the reset is not zero ablation at step 0"),
    ("head_slice_wrong", "ablate.py", "cols = torch.tensor([h * hd + i for h in",
     "cols = torch.tensor([h + i for h in", [AB], "a head ablation zeroes the wrong columns"),
    ("random_set_count", "masks.py", "need = {0: len(chosen)}", "need = {0: len(chosen) + 1}", [AB],
     "random sets are not count-matched per layer"),
    ("mask_objective_flipped", "masks.py", 'OBJECTIVES = {"KF": (1.0, 0.0), "SF": (1.0, -1.0)',
     'OBJECTIVES = {"KF": (1.0, 0.0), "SF": (-1.0, 1.0)', [MK], "SF selects skill units (R-2: test_masks now)"),
    ("fl_qa_keeps_one_high_entity", "world.py", "qa_e[qa_e >= keep] = -1", "qa_e[qa_e > keep] = -1", [W],
     "review 2026-09-27: FL keeps entity N_LOW's QA, a non-FB_low fact in the FL QA stream (survived before)"),
    ("qa_purity_off", "check_k.py", 'bad["qa_non_low_name_in_FL"] += name[0] in f1',
     'bad["qa_non_low_name_in_FL"] += 0 * name[0] in f1', [GE], "gen.py check reads the bio docs only"),
    ("fact_source_shuffled", "gen.py", '"share": round(lay["share_bio"], 6),\n                 "shuffle": False',
     '"share": round(lay["share_bio"], 6),\n                 "shuffle": True', [GE],
     "fact shards shuffled: the undrawn tail can be read first"),
    # review R-1 (2026-09-27): family F's reference design, its oracles and its bars
    ("f_foil_alias_only", "fams.py", 'ref = fl["kref"] if e == 0 else fl["fref"]',
     'ref = fl["kref"] if e == 0 else "alias"', [RF], "the foil never uses ellipsis: O13a solves F (R-1's pattern)"),
    ("f_others_never_ellipsis", "fams.py", '"alias" if r.random() < 0.5 else "ell", al[e], False, False',
     '"alias", al[e], False, False', [RF], "entities other than the key and the foil never use ellipsis"),
    ("f_foil_late", "fams.py", 'late = e == 0 and fl["klate"]', 'late = fl["klate"]', [RF],
     "the foil restates the attribute by name after its reference statement"),
    ("f_stale_anywhere", "fams.py", "return all(i < first_ref for i, s in enumerate(flat) if s is ko or s is fo)",
     "return True", [RF], "stale named statements sit between the reference statements (forms leak ownership)"),
    ("f_last_not_thinned", "fams.py", "LAST_KEEP = 0.35", "LAST_KEEP = 1.0", [RF],
     "the key's latest is the item's last statement above chance"),
    ("nested_inner_not_exact", "skill.py", "for i, c in zip(idx, balanced(len(idx), inner(v), r)):",
     "for i, c in zip(idx, balanced(n, inner(v), r)):", [RF], "F's inner flags are not exact within a group"),
    ("o12_blind_to_refs", "oracles.py", 'out["O12"] = _last(S, lambda s: s["attr"] == a and (ref(s) or s["name"] ==',
     'out["O12"] = _last(S, lambda s: s["attr"] == a and (False or s["name"] ==', [RF], "O12 becomes O4"),
    ("o13_resolves_both", "oracles.py", 's["name"] == nm or s["kind"] == blind or', 's["name"] == nm or False or',
     [RF], "O13 resolves both reference kinds"),
    ("o14_blind_to_nothing", "oracles.py", 's["attr"] == a and s["kind"] != blind and s["ent"] == nm)',
     's["attr"] == a and s["ent"] == nm)', [RF], "O14 becomes IDEAL"),
    ("pattern_merges_forms", "oracles.py", 'FORM = {"ell": "E", "alias": "A"}', 'FORM = {"ell": "E", "alias": "E"}',
     [RF], "SURF's pattern cannot tell ellipsis from alias"),
    ("surf_in_sample", "purity.py", "fa, fb = surf_counts(a, key), surf_counts(b, key)",
     "fa = fb = surf_counts(W, key)", [RF], "SURF fits and scores the same items"),
    ("surf_bar_loose", "oracles.py", '"SURF": ("F", .65)', '"SURF": ("F", .95)', [RF], "SURF's bar never binds"),
    ("ref_bars_loose", "oracles.py", '**{o: ("F", .50) for o in REFS[:4]}', '**{o: ("F", .95) for o in REFS[:4]}',
     [RF], "the O11/O12 bars never bind"),
    ("f_balance_rules_off", "purity.py", 'if fam == "F":          # R-1', 'if fam == "X":          # R-1', [RF],
     "F's follow_ref, follow_alias and key_named rules never run"),
    ("follow_ref_reads_named", "purity.py", '"follow_ref": any(s["kind"] in O.REF_KINDS for s in after)',
     '"follow_ref": bool(after)', [RF], "follow_ref counts a named foil statement"),
    # BD-14 (2026-09-27, second pass): one reference kind per item; SURFe and SURFa (one kind resolved)
    ("f_kinds_independent", "skill.py", "follow=bool(w), fref=x, fstale", "follow=bool(w), fref=kinds[int("
     "r.integers(0, 2))][0], fstale", [RF], "the foil's kind drawn apart from the key's: elimination (SURFe 0.90)"),
    ("f_kind_tie_not_flipped", "skill.py", "lambda v: kinds if v else kinds[::-1]", "lambda v: kinds", [RF],
     "the key's reference kind is off by one in odd follow groups"),
    ("surfe_view_unresolved", "oracles.py", 'FORM[s["kind"]].lower() if s["kind"] == res and s["ent"] == q["name"] '
     'else', 'FORM[s["kind"]].lower() if False else', [RF], "SURFe and SURFa resolve nothing (they equal SURF)"),
    ("surfe_bar_loose", "oracles.py", '"SURFe": ("F", .85)', '"SURFe": ("F", .99)', [RF], "SURFe's bar never binds"),
    ("same_kind_check_off", "purity.py", '"same_kind": all(s["kind"] == kref for s in refs)', '"same_kind": True',
     [RF], "mixed reference kinds pass the balance rules"),
    # review R-2 (2026-09-27): masks.fit defaults and the planted-set fixture (K2 notes K2-e)
    ("mask_defaults_build", "masks.py", _SIG, _SIG.replace("= 2000", "= 300").replace("= 6.0", "= 1.0").replace(
        "init: float = 0.0, per_budget: bool = True", "init: float = 1.0, per_budget: bool = False").replace(
        "restarts: int = 2", "restarts: int = 1"), [MK],
     "the build defaults (300 steps, lam 1 per unit, init 1.0) miss saturated plants"),
    ("mask_defaults_r2", "masks.py", _SIG, _SIG.replace("= 2000", "= 1000").replace("= 6.0", "= 1.0").replace(
        "per_budget: bool = True", "per_budget: bool = False").replace("restarts: int = 2", "restarts: int = 1"),
     [MK],
     "R-2's defaults (1,000 steps, lam 1 per unit): MX-SS seed 6 misses the saturated skill unit (R-5)"),
    ("mask_init_three", "masks.py", "        init: float = 0.0, per_budget", "        init: float = 3.0, per_budget",
     [MK], "gate init 3.0: gates start clamped open, where a saturated unit gets no gradient (K3: mask_init_one, "
     "init 1.0, retired: at 2,000 steps it found 120 of 120 on seeds 0-5)"),
    ("mask_per_budget_off", "masks.py", "        init: float = 0.0, per_budget: bool = True, restarts",
     "        init: float = 0.0, per_budget: bool = False, restarts", [MK],
     "lam 6 per unit instead of per budget unit"),
    ("mask_penalty_per_unit", "masks.py", "return lam * (torch.relu(expected_dropped - budget) / (budget if "
     "per_budget else 1))", "return lam * (torch.relu(expected_dropped - budget))", [MK],
     "the budget term ignores per_budget"),
    ("mask_steps_300", "masks.py", "budget: int, steps: int = 2000,", "budget: int, steps: int = 300,", [MK],
     "300 steps at init 0: planted gates barely move"),
    ("mask_harden_most_open", "masks.py", "return [(l, j) for _, l, j in sorted(allu)]",
     "return [(l, j) for _, l, j in sorted(allu, reverse=True)]", [MK], "hardening keeps the most-open units"),
    ("mask_penalty_off", "masks.py", "loss = -(wf * gf + ws * gs) + penalty(",
     "loss = -(wf * gf + ws * gs) + 0 * penalty(", [MK], "no L0 penalty: gates close past the budget"),
    ("mask_p_drop_is_p_open", "masks.py", "p_drop = {l: [float(x) for x in 1 - g.p_open().detach()]",
     "p_drop = {l: [float(x) for x in g.p_open().detach()]", [MK], "the reported soft mask is P(open)"),
    ("sf_ignores_skill", "masks.py", '"SF": (1.0, -1.0)', '"SF": (1.0, 0.0)', [MK],
     "SF is KF: the mixed plant's shared unit X can be taken"),
    ("plant_unsaturated", "tests/plantfix.py", '"F1": 20.0, "F2": 2.0, "F3": 20.0, "F4": 5.0, "S1": 20.0',
     '"F1": 2.0, "F2": 2.0, "F3": 2.0, "F4": 5.0, "S1": 2.0', [MK],
     "no saturated plant: the fixture loses the case the build defaults missed"),
    ("plant_not_silent", "tests/plantfix.py", "perp = [(xs[n][-1].double() - Q @ (Q.T @ xs[n][-1].double())) for n "
     "in own]", "perp = [xs[n][-1].double() for n in own]", [MK], "planted units also fire on other prompts"),
    # K3 round 1 (2026-09-27): R-3, R-4, U's and A's redesigns, the fitted cue-model gate
    ("f_foil_not_thinned", "fams.py", "if (flat[-1] is kl or flat[-1] is fu) and r.random() >= LAST_KEEP:",
     "if flat[-1] is kl and r.random() >= LAST_KEEP:", [RF], "R-3: only the key's last thinned, LASTSKIP returns"),
    ("ref_last_rule_off", "purity.py", "if abs(a - b) > 3 * math.sqrt(max(1, a + b)):\n            fails.append(f\"F: "
     "the item ends", "if abs(a - b) > 30 * math.sqrt(max(1, a + b)):\n            fails.append(f\"F: the item ends",
     [RF], "the R-3 balance rule never fires"),
    ("questions_ask_final_attr", "skill.py", "pool = [k for k in stated if k[1] != final.attr and k[0] not in hold]",
     "pool = list(stated)", [SK], "R-4: earlier questions ask the final attribute (copy and exclusion cues)"),
    ("u_lure_not_stale", "skill.py", "lure = lure_key in stated and any((s.ent, s.attr) == lure_key for u in "
     "turns[j + 1:] for s in u)", "lure = lure_key in stated", [SK], "U's lure can copy the gold"),
    ("early_attr_rule_off", "purity.py", 'if any(f["early_attr"] for f in facts):', "if False:", [SK],
     "the R-4 rule never fires"),
    ("u_pair_one_not_mirrored", "fams.py", 'c0 = (1, 2) if var == "noupd" else (2, 1) if sub == "one" else',
     'c0 = (1, 2) if var == "noupd" else (2, 2) if sub == "one" else', [SK],
     "U: upd-one no longer mirrors noupd (a count cue)"),
    ("u_rank_not_exact", "skill.py", "            f.update(var=v, sub=s, k=k, rank=rk, frank=fk)",
     "            f.update(var=v, sub=s)", [SK], "U's (k, rank) drawn per item again (retargeted K3 round 6)"),
    ("a_both_r_only", "fams.py", _ACTX + '            c = "R"', 'if True:\n            c = "R"', [SK],
     "A's both-items from R contexts only (retargeted K3 round 6)"),
    ("cue_reads_question_name", "cues.py", '("ref_unknown", float(not named))',
     '("ref_unknown", float(named and s["name"] == q["name"]))', [CU], "a feature binds the question's name"),
    ("cue_gate_in_sample", "cuegate.py", "predict(fit(b, seed=seed, **cfg), a), predict(fit(a, seed=seed, **cfg), b)",
     "predict(fit(a, seed=seed, **cfg), a), predict(fit(b, seed=seed, **cfg), b)", [CU],
     "the training number is scored in sample"),
    ("cue_tie_full_credit", "cuegate.py", "hit.append((top[start + gold] / torch.zeros(nq).index_add(0, qid, top))"
     ".numpy())", "hit.append(top[start + gold].numpy())", [CU], "tied candidates each get full credit"),
    ("cue_bar_loose", "cuegate.py", '"A": .50, "P": .65}', '"A": .90, "P": .65}', [CU], "A's bar never binds"),
    ("cue_gate_off_in_check", "check_k.py", "    if skill and cues:", "    if False:", [GE],
     "gen.py check skips the gate"),
    # K3 round 2 (2026-09-27): REVIEW 3a (B-CENTER, F-INTRO-DEFX, GATE-F-BLIND, GATE-CODE) and REVIEW 3b (T-1, T-2,
    # T-3, D-1, D-5)
    ("b_cue_bars_loose", "oracles.py", '**{o: ("B", .65) for o in BCUE + BPART}',
     '**{o: ("B", .99) for o in BCUE + BPART}', [SK], "the B-CENTER, B-TOKFREQ and B-MAJOR bars never bind"),
    ("b_k_rank_per_item", "skill.py", "            f.update(struct=c, k=k, rank=x)", "            f.update(struct=c)",
     [SK], "B's k and intro rank drawn per item again (BD-22)"),
    ("b_k_rank_rule_off", "purity.py", '            for k in (2, 3, 4, 8):\n                Fk = [f for f in F if '
     'f["k"] == k and f["grp"] <= 0]', '            for k in ():\n                Fk = [f for f in F if f["k"] == k '
     'and f["grp"] <= 0]', [SK],
     "B's per-block intro rank rule never runs"),
    ("kevs_not_whole_cubes", "pools.py", "for c in cells for i in (0, 1) for j in (0, 1) for m in (0, 1)]",
     "for c in cells for i in (0,) for j in (0, 1) for m in (0, 1)]", [W],
     "K-EVAL-S holds half of each cell: a K-EVAL-S cube item names trained triples"),
    ("f_intro_rule_off", "fams.py", '        if first_ref - (flat[first_ref].kind == "ell") <= max([intro] + defs):',
     "        if False:", [RF], "BD-21 off: a reference before the asked name's first mention (F-INTRO 0.65)"),
    ("f_ell_unit_rule_off", "fams.py", 'if first_ref - (flat[first_ref].kind == "ell") <= max(',
     "if first_ref <= max(", [RF], "an ellipsis's antecedent may be its entity's first mention (rank cue 0.54)"),
    ("f_last_keep_half", "fams.py", "LAST_KEEP = 0.35", "LAST_KEEP = 0.5 ", [RF],
     "LAST_KEEP back at 0.5 after BD-21: F's last-statement rate 0.228 vs 0.192"),
    ("f_foil_thinned_less", "fams.py", "        if (flat[-1] is kl or flat[-1] is fu) and r.random() >= LAST_KEEP:",
     "        if (flat[-1] is kl and r.random() >= LAST_KEEP) or (flat[-1] is fu and r.random() >= 1.4 * LAST_KEEP):",
     [RF], "REVIEW 3b's f_foil_thinned_less (0.7 against 0.5), scaled: passed the 4-block test by 2 counts"),
    ("f_intro_oracle_blind", "oracles.py", "    keep = [i for i in refs if i > first] or refs", "    keep = refs",
     [RF], "F-INTRO drops nothing"),
    ("f_defx_oracle_blind", "oracles.py", '    kd = [i for i in keep if not (S[i]["kind"] == "alias" and ndef and i < '
     'ndef[0])] or keep', "    kd = keep", [RF], "F-DEFX is F-INTRO"),
    ("f_intro_bars_loose", "oracles.py", '**{o: ("F", .65) for o in FINTRO}', '**{o: ("F", .99) for o in FINTRO}',
     [RF], "the F-INTRO and F-DEFX bars never bind"),
    ("u_mirror_block_rule_off", "purity.py", '            rules["mirror"] = ([f["u_cell"][:2] == (2, 1) for f in '
     'F], 4)', '            rules["mirror"] = ([True for f in F], 1)', [SK],
     "D-1: U's mirror cell unchecked per block"),
    ("u_mirror_pooled_rule_off", "purity.py", "        if abs(m - mm) > 3 * math.sqrt(m + mm):",
     "        if abs(m - mm) > 30 * math.sqrt(m + mm):", [SK], "D-1: U's pooled mirror rule never fires"),
    ("u_k_rank_rule_off", "purity.py", "            for k in (2, 3, 4):                             # U's k",
     "            for k in ():                             # U's k", [SK], "D-1: U's k and rank unchecked per block"),
    ("p_redraw_off", "skill.py", '    if fam == "P" and sum(k == "Q" for k, _ in ex) < (5 if ood == "PLONG" else 2):',
     "    if False:", [SK], "T-1: P items keep one question (6% of draws)"),
    ("p_qcount_rule_off", "purity.py", "    return [f\"{fam}: {n} items with a question count outside SPEC 4's range\" "
     "for fam, n in sorted(bad.items())]", "    return []", [SK], "T-1: no rule counts questions"),
    ("plong_range_ignored", "check_k.py", 'p_questions=(5, 7) if tag == "OOD-PLONG" else (2, 4))',
     "p_questions=(2, 4))", [GE], "gen.py check holds PLONG to P's 2-4 questions"),
    ("cue_names_off", "cues.py", "               + _names_cols(s, S, intro, am, avals))", "               )", [CU],
     "B-CENTER: the gate loses its NAMES columns"),
    ("cue_draw_first_items", "cuegate.py", "    return np.sort(np.random.default_rng([K.W, salt]).choice(n, cap, "
     "replace=False))", "    return np.arange(cap)", [CU],
     "T-2: the gate fits the stream's first items (retargeted K3 round 5: draw() takes a salt)"),
    ("set_learner_unpooled", "cuegate.py", '"set": dict(hidden=64, pool=True, epochs=12, lr=3e-3)',
     '"set": dict(hidden=64, pool=False, epochs=12, lr=3e-3)', [CU], "T-2: the set learner cannot compare"),
    ("f_model_turn_leak", "cues.py", '("last_in_turn", float(i == turn[-1]))',
     '("last_in_turn", float(any(S[x]["name"] == nm for x in turn)))', [CU],
     "GATE-F-BLIND: an F-model column reads an asked-name statement in the candidate's turn"),
    ("f_model_off_in_run", "cuegate.py", "    if FF:\n", "    if False:\n", [CU, GE], "the F model never runs"),
    ("f_model_fails_dropped", "cuegate.py", '        out["fails"] = out["fails"] + rf["fails"]\n        for sn, c in rf['
     '"ceiling"]["F"].items():', '        for sn, c in rf["ceiling"]["F"].items():', [CU],
     "the F model's fails never reach the report (retargeted K3 round 5: the family-alone block has the same line)"),
    ("f_model_ceiling_not_merged", "cuegate.py", '            out["ceiling"].setdefault("F", {})[sn] = max(c, out['
     '"ceiling"].get("F", {}).get(sn, 0.0))', "            pass", [CU], "F's ceiling ignores the F model"),
    ("sweep_equivalence_stale", "tests/mask_sweep.py", '    rec = one("MX-SF", "k2", d["init"], d["lam"], seed, '
     '(steps, steps + 80), per_budget=d["per_budget"],\n              restarts=1)[0]',
     '    rec = one("MX-SF", "k2", 1.0, 1.0, seed, (steps, steps + 80))[0]', [MK],
     "D-5: the sweep's equivalence check compares other settings"),
    ("mask_restarts_one", "masks.py", "restarts: int = 2) -> dict:", "restarts: int = 1) -> dict:", [MK],
     "M-1: one run per fit: MX-KF seed 98 on the default floors stays stuck"),
    ("mask_restart_keeps_worst", "masks.py", 'best = max(range(restarts), key=lambda i: runs[i]["hard"])',
     'best = min(range(restarts), key=lambda i: runs[i]["hard"])', [MK], "M-1: the restart with the worse set kept"),
    ("mask_hard_unablated", "masks.py", "            h[..., js] = 0\n", "            h[..., js] = h[..., js]\n", [MK],
     "M-1: the exact objective does not ablate the set (every restart ties, the first kept)"),
    # K3 round 3 (2026-09-27): REVIEW 4a S-1 (B cube groups), S-2 (alias windows), S-3 and A-mix (reported rows);
    # REVIEW 4b T-1 (b_small_rule_off and b_small_k_rules_off are its probes), M-1 (below, with the mask mutants)
    ("f_alias_window_own_def", "skill.py", '    if it.fam == "F":           # BD-25', '    if False:           # BD-25',
     [RF], "S-2: each alias only in its own definition's window (F-WINK, F-XNEAR)"),
    ("alias_win_rule_off", "purity.py", '    if any(not f["alias_win"] for f in facts):', "    if False:", [RF],
     "S-2: the per-item alias window rule never fires"),
    ("xnear_rule_off", "purity.py", "        if abs(a - b) > 3 * math.sqrt(max(1, a + b)):\n            fails.append("
     "f\"F: the alias nearer", "        if abs(a - b) > 30 * math.sqrt(max(1, a + b)):\n            fails.append("
     "f\"F: the alias nearer", [RF], "S-2: the F.xnear balance rule never fires"),
    ("xnear_counts_late_quarter", "purity.py", ' and len(als) == 2 and last >= 0 and\n            S[last]["kind"] == '
     '"alias" else None}', ' and len(als) == 2 else None}', [RF],
     "F.xnear read on the late quarter too, where the key's alias is earlier by design"),
    ("f_model_window_off", "cues.py", "               + _window(s, i, nm, D, gaps, nearest))", "               )",
     [CU], "S-2: the F model loses its alias-window columns"),
    ("b_groups_ask_base", "skill.py", '[("Q", I.Qn(e, q.attr, q.form))]', '[("Q", I.Qn(0, q.attr, q.form))]', [SK],
     "S-1: a group's copies ask the item's own name"),
    ("b_group_form_flipped", "skill.py", '[("Q", I.Qn(e, q.attr, q.form))]', '[("Q", I.Qn(e, q.attr, 1 - q.form))]',
     [SK], "S-1: a copy asks in the other frame (a frame tie-break could then tell a group's questions apart)"),
    ("b_group_ranks_counted", "purity.py", 'Fk = [f for f in F if f["k"] == k and f["grp"] <= 0]',
     'Fk = [f for f in F if f["k"] == k]', [SK], "BD-26: the copies count in B's per-block intro-rank rule"),
    ("b_group_frames_not_nested", "skill.py", "        if ncube is not None:                                   # BD-26",
     "        if False:                                   # BD-26", [SK], "BD-26: frames not split within cube items"),
    ("b_group_bars_o3_only", "purity.py", "    for o in [*O.O3, *O.BCUE, *O.BPART]:\n        if o in groups",
     "    for o in O.O3:\n        if o in groups", [SK], "S-1: B-ADAPT and B-FRAME are not checked on groups"),
    ("group_bar_loose", "oracles.py", "GROUP_TH = .125", "GROUP_TH = .99 ", [SK, CU], "the group bar never binds"),
    ("b_part_uses_whole_name", "oracles.py", "for sub in SUBSETS),", "for sub in SUBSETS + [(0, 1, 2)]),", [SK],
     "B-ADAPT reads the whole name: it is IDEAL on B"),
    ("groups_not_found", "purity.py", 'if [f["grp"] for f in fs[i:i + 4]] == [0, 1, 2, 3]]', "if False]", [SK],
     "no group is scored, so the group bars never run"),
    ("cue_group_n_256", "cuegate.py", "        if not O.rate_ok(c, GROUP_BAR, n_groups[sn]):",
     "        if not O.rate_ok(c, GROUP_BAR, 256):", [CU], "GATE-CODE: the group tolerance is n 256 on every set"),
    ("cue_group_base_only", "cuegate.py", "np.prod(h[g])", "h[g[0]]", [CU], "the gate scores a group by its item only"),
    ("b_small_rule_off", "purity.py", 'rules["b_small"] = ([f["k"] != 8 for f in F], 4)', "pass", [SK],
     "T-1 (REVIEW 4b probe): B's cube share (3/4 per block) unchecked"),
    ("b_small_k_rules_off", "purity.py",
     'rules.update({f"k{k}": ([f["k"] == k for f in small], 3) for k in (2, 3, 4)})', "pass", [SK],
     "T-1 (REVIEW 4b probe): the small quarter's k shares unchecked"),
    ("a_manyfree_blind", "oracles.py", '("A-MANYFREE", len(names - holders) >= 2)', '("A-MANYFREE", False)', [SK],
     "A-mix: A-MANYFREE is O9|10"),
    ("f_ante_keeps_all", "oracles.py", '    return S[(keep or refs)[-1]]["val"]', '    return S[refs[-1]]["val"]', [RF],
     "S-3: F-ANTE drops nothing"),
    # K3 round 4 (2026-09-28): REVIEW 5a S-1 (bystander questions, question columns), S-2 (U's foil rank); REVIEW 5b
    # T-a (alias_win_own_def), T-b (mask_hard_weights_dropped), O-1 (group counts); REVIEW 5b's third mutant too
    ("q_pool_holders", "skill.py", "pool = [k for k in stated if k[1] != final.attr and k[0] not in hold]",
     "pool = [k for k in stated if k[1] != final.attr]", [SK],
     "S-1: earlier questions ask any stated key again (the final entity and holders: QFIT)"),
    ("early_name_rule_off", "purity.py", '    if any(f["early_name"] for f in facts):', "    if False:", [SK],
     "S-1: the bystander rule is never checked"),
    ("early_name_misses_holders", "purity.py", 'hold = {s["ent"] for s in S if s["attr"] == q["attr"]} | {q["name"]}',
     'hold = {q["name"]}', [SK], "S-1: the check reads only the final name, not the other holders"),
    ("q_cols_blind", "cues.py", '_oh("qn_asked", min(qn.count(q["name"]), 2), 3)', '_oh("qn_asked", 0, 3)', [CU],
     "S-1: the gate's column for the final name among earlier questions reads nothing"),
    ("q_cols_off", "cues.py", "            + _q_cols(q, prev, S, intro, a))", "            )", [CU],
     "S-1: the gate has no question-to-question columns (the old draw's QFIT cue unread)"),
    ("r_u_reserve_off", "skill.py", "        skip = {int(r.choice(free))} if free else set()",
     "        skip = set()", [SK], "R and U ask every bystander (A's asked entity never asked: the gate read A 0.536)"),
    ("all_free_rule_off", "purity.py", '    if any(f["all_free_asked"] for f in facts):', "    if False:", [SK],
     "R's and U's reserved bystander is never checked"),
    ("p_bystander_off", "fams.py", "k=int(r.integers(2, 5)) if n_q is not None else None, by=True)",
     "k=int(r.integers(2, 5)) if n_q is not None else None, by=False)", [SK],
     "P has no bystander entity of its own (R's rule for every entity, redrawn until one is a bystander)"),
    ("rfp_rank_rules_off", "purity.py",
     '                rules.update({f"k{k}_rank{rk}": ([f["rank"] == rk for f in Fk], k) for rk in range(k)})\n'
     '        if fam == "B":', '                pass\n        if fam == "B":', [SK],
     "R's, F's and P's per-block rank rule never checked"),
    ("rp_rank_per_item", "skill.py", "            f.update(k=k, rank=x)                       # follow flag",
     "            pass                       # follow flag", [SK], "R's and P's (k, rank) per item again (retargeted)"),
    ("dfar_pad_makes_holders", "skill.py", "        if a == key[1] and e not in hold:", "        if False:", [SK],
     "a DFAR pad turns a bystander into a holder (it may be one an earlier question asked)"),
    ("u_foil_rank_free", "fams.py", "sigma_with(k, {e0: rank, e1: frank}, r)", "sigma_with(k, {e0: rank}, r)", [SK],
     "S-2: the foil's intro rank drawn free again (a big entity, introduced early)"),
    ("u_frank_rule_off", "purity.py",
     'rules.update({f"k{k}_frank{rk}": ([f["frank"] == rk for f in Fk], k) for rk in range(k)})', "pass", [SK],
     "S-2: U's foil-rank block rule never checked"),
    ("u_rank_pairs_per_item", "skill.py", "for i, p in zip(idx, rank_pairs(len(idx), k, r)):",
     "for i, p in zip(idx, [tuple(int(x) for x in r.choice(k, 2, replace=False)) for _ in idx]):", [SK],
     "S-2: the (key, foil) rank pair drawn per item, not exact per block"),
    ("a_ctx_per_item", "fams.py", '        if (fl.get("ctx") or ("R" if r.random() < 0.375 else "U")) == "R":',
     "        if r.random() < 0.375:", [SK], "A's both-item context kind drawn per item (U contexts over-drawn)"),
    ("alias_win_own_def", "purity.py", 'all(1 <= s["ex"] - dx[u] <= 10 for s in als for u in ents if u in dx)',
     'all(1 <= s["ex"] - dx[u] <= 10 for s in als for u in [s["ent"]] if u in dx)', [RF],
     "T-a: purity's per-item alias window reads each alias against its own definition only (pre-round-3 rule)"),
    ("mask_hard_weights_dropped", "masks.py", "    return w[0] * gf + w[1] * gs\n", "    return gf + gs\n", [MK],
     "T-b: the restart pick ignores the objective's weights (MX-SF seed 300 keeps the set with X)"),
    ("group_count_rule_off", "purity.py", 'return [f"B groups: {got} scored of {n0} cube items"] if n0 != got else []',
     "return []", [SK], "O-1: records out of group order skip every group bar silently"),
    ("check_group_count_off", "check_k.py",
     'if name in ("k_eval_id", "k_eval_s") and rep["groups"].get("n", 0) != 3 * nb // 16:',
     'if False:', [GE], "O-1: a K-EVAL B set written without groups passes gen.py check"),
    ("b_group_one_token_relatives", "skill.py", "if sum(x == y for x, y in zip(u, it.names[0])) == 2]",
     "if sum(x == y for x, y in zip(u, it.names[0])) == 1]", [SK],
     "REVIEW 5b: a group's copies ask one-token relatives (a partial key answers from 4 disjoint pairs)"),
    # K3 round 5 (REVIEW 6a S-1, S-2, G-1; the round-4 tests lens re-run)
    ("b_order_conditioned", "fams.py",
     "    order = [units[i] for i in r.permutation(len(units))]           # uniform (BD-1), nothing conditioned on",
     "    order = arrange(units, lambda flat: intro_rank(flat, 0) == rank, r)", [SK],
     "S-2: B's order drawn until the asked name has the flagged rank, whatever its statement count (the build)"),
    ("exch_rule_off", "purity.py", "        if df and stat > wh(df):", "        if False:", [SK],
     "S-1, S-2: the exchangeability rules never fail"),
    ("b_exch_off", "purity.py", '        out["B.exch"] = (stat, df)', "        pass", [SK],
     "S-2: B's exchangeability is never measured"),
    ("p_rank_rule_dropped", "purity.py", 'if fam in "RFP":', 'if fam in "RF":', [SK],
     "round-4 tests lens: P's per-block intro-rank rule dropped (test_r_f_p_rank_rules never drew P)"),
    ("r_post_bar_off", "oracles.py", '"R-POST": ("RP", .65),', '"R-POST": ("", .65),', [SK],
     "S-1: R-POST, R's design ceiling, has no bar"),
    ("r_post_wrong_pick", "oracles.py", "pick = order[0] if len(order) <= 2 else order[-1]", "pick = order[0]", [SK],
     "R-POST takes the first holder with 3 or more (the posterior's pick is the last)"),
    ("b_surprise_size_blind", "oracles.py", 'return x if sum(s["name"] == u for s in S) >= 2 else 1 - x',
     "return x", [SK], "S-2: B-SURPRISE's tie-break ignores the statement count"),
    ("fam_models_off", "cuegate.py", 'FAM_MODELS = "RUBFP"', 'FAM_MODELS = ""', [CU],
     "G-1: no family-alone fits (the pooled fit missed R's cue at 10,240 R items per half)"),
    ("fam_includes_a", "cuegate.py", 'FAM_MODELS = "RUBFP"', 'FAM_MODELS = "RUBFAP"', [CU],
     "G-1: A fitted alone (its gold is always none, so alone it reads 1.00)"),
    ("fam_ceiling_not_merged", "cuegate.py",
     '            out["ceiling"].setdefault(fam, {})[sn] = max(c, out["ceiling"].get(fam, {}).get(sn, 0.0))',
     "            pass", [CU], "G-1: a family-alone model's ceiling never reaches the family's"),
    ("fam_fails_dropped", "cuegate.py",
     '        out["fails"] = out["fails"] + rf["fails"]\n        for sn, c in rf["ceiling"][fam].items():',
     '        for sn, c in rf["ceiling"][fam].items():', [CU], "G-1: a family-alone model's fails are dropped"),
    # K3 round 6 (REVIEW 7a S-1, S-2; REVIEW 7b T-1, T-2, O-1): one law for every key, the attribute-blind side
    ("r_entity_uniform", "fams.py", "    e0, astar = edges[int(r.integers(len(edges)))]",
     "    e0 = int(r.integers(km))\n    astar = keys[e0][int(r.integers(len(keys[e0])))]", [SK],
     "S-1: R's asked entity uniform over entities, then its key: smaller holders favoured (a size cue)"),
    ("r_asked_attr_most_held", "fams.py", "    e0, astar = edges[int(r.integers(len(edges)))]",
     "    e0, astar = max(edges, key=lambda x: (sum(x[1] in v for v in keys.values()), float(r.random())))", [SK],
     "S-1: R's asked key the one whose attribute the most entities hold (round 5's TA-SHARE cue)"),
    ("r_order_conditioned", "fams.py", '    order = sym_order(units, sigma_with(k, {e0: rank}, r), r)\n    return '
     '_relabel(order, e0), (0, astar), {"k": k}', '    order = arrange([u for us in units.values() for u in us], '
     'lambda f: intro_rank(f, e0) == rank, r)\n    return (_relabel(order, e0) if order else None), (0, astar), {"k": '
     'k}', [SK], "S-1: a uniform order drawn until the asked entity has the block's rank (frac(h): a rank-size cue)"),
    ("sym_insert_anywhere", "fams.py", "        seq.insert(int(r.integers(at + 1, len(seq) + 1)), u)",
     "        seq.insert(int(r.integers(0, len(seq) + 1)), u)", [SK],
     "a unit may precede its entity's intro unit: the block's intro rank is not the asked entity's"),
    ("u_other_pairs_one2", "fams.py",
     "        c = _pair_counts(str(r.choice([t for t, _ in U_TYPES], p=[p for _, p in U_TYPES])), r)",
     '        c = _pair_counts("one2", r)', [SK],
     "S-1: U's other attributes all (1, 2) pairs: the asked key the most stated (round 5's TA-CNT cue)"),
    ("u_key_bigger", "fams.py", "    e0, e1 = pairs[0][side], pairs[0][1 - side]",
     "    e0, e1 = sorted(pairs[0], key=lambda e: -sum(e in p for p in pairs))", [SK],
     "S-1: U's asked side the holder in more pairs (a bigger entity: round 3's U size cue)"),
    ("p_none_unchecked", "fams.py", '    if order is None:\n        return None, None, None\n    meta["marker"]',
     '    meta["marker"]', [SK], "O-1: fam_P indexes fam_R's meta when fam_R returns None (TypeError)"),
    ("questions_comp_off", "skill.py", '    comp = fam in ("R", "U", "P") or "rctx" in meta', "    comp = False", [SK],
     "S-1 (+eq): R, U, P ask bystanders of the final key's component (an attribute of the asked entity's)"),
    ("early_comp_rule_off", "purity.py", '    if any(f["early_comp"] for f in facts):', "    if False:", [SK],
     "S-1: the component rule for earlier questions is never checked"),
    ("exch_attr_off", "purity.py", '("size", F, lambda f: f["hsizes"], "hpos"),\n                                    '
     '("attr", A, lambda f: f["akeys"], "apos")):', '("size", F, lambda f: f["hsizes"], "hpos")):', [SK],
     "S-1: the asked key's place among the asked entity's keys is never checked"),
    ("exch_intro_off", "purity.py", '        if cells:\n            out[f"{fam}.exch_intro"] = (stat, df)',
     "        pass", [SK], "S-1: the intro-rank-against-size rule (sym_order's independence) is never measured"),
    ("exch_size_name_order", "purity.py", '"hsizes": tuple(min(sum(x["name"] == u for x in S), 6) for u in horder),',
     '"hsizes": tuple(min(sum(x["name"] == u for x in S), 6) for u in sorted(horder)),', [SK],
     "T-1 (REVIEW 7b): the size cells list holder sizes in name order, not stating order"),
    ("ta_bars_off", "oracles.py", '**{o: ("RUFP", .65) for o in TA},', '**{o: ("", .65) for o in TA},', [SK],
     "S-1: the attribute-blind rules have no bar"),
    ("t_bar_not_on_p", "oracles.py", '"T": ("RUFP", .65),', '"T": ("RUF", .65),', [SK], "S-2: T has no bar on P"),
    ("ta_share_counts_statements", "oracles.py", 'key = {"TA-SHARE": lambda x: (len(hold[x]),),',
     'key = {"TA-SHARE": lambda x: (tot[x],),', [SK], "TA-SHARE reads statements, not holders (it is TA-CNT)"),
    ("ta_model_off", "cuegate.py", 'TA_FAMS, TA_BAR = "RUFP", .65', 'TA_FAMS, TA_BAR = "", .65', [CU],
     "S-1: the attribute-blind model never runs"),
    ("ta_fails_dropped", "cuegate.py", '    out["fails"] = out["fails"] + out["ta_model"]["fails"]',
     '    out["fails"] = out["fails"]', [CU], "S-1: the attribute-blind model's fails never reach the report"),
    ("ta_reads_attr", "cues.py", '("eq", float(eq[x] > 0))', '("eq", float(x == q["attr"]))', [CU],
     "S-1: a TA column reads the final question's attribute"),
    ("ta_misses_dropped", "cuegate.py", 'row = {"train": round(hits / len(tr), 4)}',
     'row = {"train": round(hits / max(1, len(ok(tr))), 4)}', [CU],
     "the TA model's training accuracy leaves out the items whose gold no candidate holds (F's alias golds)"),
    ("fam_alone_from_pooled_draw", "cuegate.py", '        own = [r for r in all_train if r["fam"] == fam]',
     '        own = [r for r in train_recs if r["fam"] == fam]', [CU],
     "T-2 (REVIEW 7b): a family-alone model fits its family's share of the pooled draw"),
    ("a_ctx_half", "skill.py", 'A_CTX = [("R", .375), ("U", .625)]', 'A_CTX = [("R", .5), ("U", .5)]', [SK],
     "A's both-items from R contexts in 1/2 (the gate broke a tie to none: A 0.548-0.567)"),
    # K3 round 6 final (this round's own checks: P.exch_intro on 40,960 fresh P items; the TA model's U lure copy)
    ("a_both_own_structure", "fams.py",
     '        return order, (e, int(r.choice(miss))), {"k": meta["k"], "lure": lure, "rctx": c, "cattr": a0}',
     '        a1 = int(r.choice(miss))\n'
     '        h1 = 0 if any(s.ent == 0 and s.attr == a1 for u in order for s in u) else min(\n'
     '            s.ent for u in order for s in u if s.attr == a1)\n'
     '        order = [u for u in ([s for s in u if s.attr != a1 or s.ent == h1] for u in order) if u]\n'
     '        return order, (e, a1), {"k": meta["k"], "lure": lure, "rctx": c, "cattr": a0}', [CU],
     "A's both-items with one other entity stating the attribute: \" none\" wins by prior (retargeted K3 rounds 6 "
     "and 7; the round-6 draft had retired it with round 5's R context line)"),
    ("p_place_off", "skill.py", '                    place=fam == "P")', '                    place=False)', [SK],
     "P items redrawn for their question count again (early bystanders: P.exch_intro 290.5 on 27 df)"),
    ("ta_qn_at_off", "cues.py", '("qn_at", float(any(p["attr"] == x for p in qn)))', '("qn_at", 0.0)', [CU],
     "the TA model cannot copy U's lure (an earlier question naming the final name, its attribute)"),
    # K3 round 6 final, probes of round 6's code (k3/r6f/probe_mutants.py); ta_from_pooled_draw survived the tests
    # until test_family_alone_models_join_the_ceiling asserted the TA model's n_train
    ("ta_from_pooled_draw", "cuegate.py", '    out["ta_model"] = ta_run(all_train, eval_recs, lex, fam_cap, learners)',
     '    out["ta_model"] = ta_run(train_recs, eval_recs, lex, fam_cap, learners)', [CU],
     "T-2's twin: the attribute-blind model fits its family's share of the pooled draw"),
    ("sym_intro_first_unit", "fams.py", "        i = int(r.integers(len(us)))", "        i = 0", [CU],
     "sym_order introduces each entity by its first unit (U: the asked entity by the asked key, TA 0.99)"),
    ("component_one_hop", "skill.py",
     "        todo += [x for a in by_ent.get(e, ()) for x in by_attr[a] if x not in out]",
     "        todo += [x for a in by_ent.get(e, ()) for x in by_attr[a] if x not in out] if e in ents else []",
     [SK], "the component rule stops after the first hop (U chains of pairs)"),
    ("sigma_free_in_order", "fams.py", "    for i, x in zip(free_r, r.permutation(len(free_e))):",
     "    for i, x in zip(free_r, range(len(free_e))):", [SK],
     "the entities the block does not fix take the free ranks in label order (P's bystander last)"),

    # K3 round 7 (REVIEW 8b T-1, T-2, T-3: its five probes, each watched surviving round 6's tests)
    ("ta_ellipsis_dropped", "cues.py", 'E = [i for i, s in enumerate(S) if s["tent"] == nm]',
     'E = [i for i, s in enumerate(S) if s["name"] == nm]', [CU, GE],
     "T-1: the TA model's keys by name only (T's ellipsis join dropped): F's TA ceiling read 0.24 against 0.62"),
    ("ta_oracle_ellipsis_dropped", "oracles.py", '    E = [s for s in S if s["tent"] == q["name"]]',
     '    E = [s for s in S if s["name"] == q["name"]]', [SK, RF],
     "T-1: the TA rules' keys by name only: F's TA-* read 0.195-0.238 against 0.440-0.567"),
    ("ta_ceiling_first_learner", "cuegate.py", '            c = max(v[sn] for v in out["table"][fam].values())',
     '            c = out["table"][fam][next(iter(learners))][sn]', [CU, GE],
     "T-2: a family's TA ceiling is the first learner's (logit) reading"),
    ("gate_ceiling_first_learner", "cuegate.py", "            c = max(row[m][sn] for m in row)",
     "            c = row[next(iter(row))][sn]", [CU, GE],
     "T-2: the pooled, family-alone and F models' ceilings are the first learner's reading"),
    ("exch_attr_three_keys_only", "purity.py",
     'A = [f for f in facts if f["fam"] == fam and len(f["akeys"]) >= 2 and f["apos"] >= 0]',
     'A = [f for f in facts if f["fam"] == fam and len(f["akeys"]) >= 3 and f["apos"] >= 0]', [SK, PU],
     "T-3: exch_attr reads asked entities with 3 or more keys only (0.42 of R and P items have 2)"),
    # K3 round 7 (REVIEW 8a S-1, G-1, N-1): this round's code
    ("a_both_r6_draw", "fams.py", '        e = int(r.integers(meta["k"]))\n',
     '        e = int(r.choice([x for x in range(meta["k"]) if not any(s.ent == x and s.attr == a0 for u in order '
     'for s in u)] or [0]))\n', [SK],
     "S-1: round 6's both-item draw back (a non-holder of the context's attribute; mostly a name with few keys)"),
    ("a_both_context_entity", "fams.py", '        e = int(r.integers(meta["k"]))\n', "        e = 0\n", [SK],
     "S-1: the context's own asked entity (the size law): a tie with a pick among 3 keys on the TA side"),
    ("a_both_r_uniform", "fams.py", '        if c == "R":            # R: the context',
     '        if False:            # R: the context', [SK],
     "S-1: R contexts drawn the U way too (asked attributes with few holders: K-EVAL-ID 0.617 on the name-blind "
     "side at R 3/8)"),
    ("a_exch_side_off", "purity.py", '        out["A.exch_side"] = (stat, df)', "        pass", [SK],
     "S-1: purity reads no A.exch_side"),
    ("ta_a_off", "cuegate.py", "    ta_a = ta_a_run(all_train, eval_recs, lex, cap, learners) if fam_cap > 0 else {}",
     "    ta_a = {}", [CU], "S-1: cuegate.run fits no attribute-blind model for A"),
    ("ta_a_credit_off", "cuegate.py",
     '            right = np.where([f["fam"] == "A" and f["cheap_absent"] for f in Fs], 1.0, h)',
     "            right = h", [CU], "S-1: A's TA reading does not credit O9|10's half"),
    ("ta_a_in_sample", "cuegate.py", "        (ha, na), (hb, nb) = predict(fa, a), predict(fb, b)",
     "        (ha, na), (hb, nb) = predict(fb, a), predict(fa, b)", [CU],
     "S-1: A's TA side scores its training number in sample (cue_gate_in_sample's twin)"),
    ("ta_a_ceiling_first_learner", "cuegate.py", '        c = max(v[sn]["A"] for v in out["table"].values())',
     '        c = out["table"][next(iter(learners))][sn]["A"]', [CU],
     "S-1: A's TA ceiling is the first learner's (logit) reading"),
    ("u_tightspan_widest", "oracles.py", "    best = min(hi - lo for lo, hi in span.values())",
     "    best = max(hi - lo for lo, hi in span.values())", [SK], "G-1: U-TIGHTSPAN takes the widest span"),
    ("u_tightspan_no_copy", "oracles.py", "    if src:\n", "    if False:\n", [SK],
     "G-1: U-TIGHTSPAN never copies the lure"),
    ("tie_full_credit", "oracles.py", "    return ans.count(gold) / len(ans) if isinstance(ans, tuple)",
     "    return float(gold in ans) if isinstance(ans, tuple)", [SK], "G-1: a tie answer gets full credit"),
    ("ceilings_gate_only", "check_k.py",
     "            rules = [o for o, (fs, _) in O.TH.items() if fam in fs and o not in MARKS and o in tab]",
     "            rules = []", [GE], "G-1: SPEC 9's ceiling reads no cheap rule"),
    ("ceilings_marks_in", "check_k.py", 'MARKS = ("O13e", "O13a", "O14e", "O14a", "SURFe", "SURFa")', "MARKS = ()",
     [GE], "G-1: F's one-kind skill marks join F's ceiling"),
    ("f_turnk_any_name", "oracles.py",
     '    has = [any(S[y]["turn"] == S[x]["turn"] and S[y]["name"] == nm for y in range(len(S)) if y != x) '
     'for x in refs]',
     '    has = [any(S[y]["turn"] == S[x]["turn"] and S[y]["name"] is not None for y in range(len(S)) if y != x)\n'
     '           for x in refs]', [SK], "N-1: F-TURNK reads any named statement in a reference's turn"),
    # K3 round 8 (REVIEW 9a G-1, O-1; REVIEW 9b T-1, T-2, T-3: its three surviving mutants, by its definitions)
    ("ceilings_any_family", "check_k.py",
     "rules = [o for o, (fs, _) in O.TH.items() if fam in fs and o not in MARKS and o in tab]",
     "rules = [o for o, (fs, _) in O.TH.items() if o not in MARKS and o in tab]", [GE],
     "T-1: SPEC 9's ceiling reads every barred rule on every family (R and U take O4's 1.0, their skill)"),
    ("ta_a_fails_dropped", "cuegate.py", 'out["fails"] = out["fails"] + ta_a["fails"]',
     'out["fails"] = out["fails"] + []', [CU, GE], "T-2: A's attribute-blind side's fails never reach run()'s"),
    ("a_u_lacking_only", "fams.py", '        e = int(r.integers(meta["k"]))\n',
     '        ats = {s.attr for u in order for s in u if s.kind != "def"}\n'
     '        lack = [x for x in range(meta["k"]) if ats - {s.attr for u in order for s in u if s.ent == x}]\n'
     '        if not lack:\n'
     '            return None, None, None\n'
     '        e = int(r.choice(lack))\n', [SK, CU],
     "T-3: a U context's asked entity uniform among its lacking names, the context kept (not redrawn)"),
    ("f_ante_apart", "fams.py", 'ante = _attrs(g, r, 1, exclude=astar)[0] if fl["kref"] == "ell" else None',
     "ante = None", [RF, SK], "G-1: F's key and foil antecedents state attributes drawn apart (round 7's draw)"),
    ("f_ante_rule_off", "purity.py", 'if any(f.get("ante_same") is False for f in facts):', "if False:", [RF],
     "G-1: purity's per-item rule on F's antecedent attributes never fires"),
    ("f_antesize_reversed", "oracles.py", 'return S[refs[0] if (ci > cj) == (nma == 1) else refs[1]]["val"]',
     'return S[refs[0] if (ci < cj) == (nma == 1) else refs[1]]["val"]', [RF], "G-1: F-ANTESIZE picks the other way"),
    ("f_antesize_unbarred", "oracles.py", '"F-ANTESIZE": ("F", .65)}', '"F-ANTESIZE": ("", .65)}', [GE, RF],
     "G-1: F-ANTESIZE is barred on no family, so no F ceiling row"),
    ("f_pv_off", "cues.py", '_oh("pv_cnt", min(acnt[S[i - 1]["attr"]], 4) if i else 0, 5)', '_oh("pv_cnt", 0, 5)',
     [CU], "G-1: the F model's antecedent-frequency column reads nothing"),
    ("f_pvhold_off", "cues.py", '_oh("pv_hold", min(ahold[S[i - 1]["attr"]], 3) if i else 0, 4)',
     '_oh("pv_hold", 0, 4)', [CU], "G-1: the F model's antecedent-holders column reads nothing"),
    ("f_qnattr_with_asked", "cues.py",
     '_oh("q_nattr", min(len({s["attr"] for s in S if s["name"] == nm and s["attr"] != a}), 3), 4)',
     '_oh("q_nattr", min(len({s["attr"] for s in S if s["name"] == nm}), 3), 4)', [CU],
     "G-1: the asked name's attribute count includes the asked attribute"),
    ("a_cntjoin_any_key", "oracles.py", 'keys = {s["attr"] for s in S if s["tent"] == q["name"]}',
     'keys = {s["attr"] for s in S if s["tent"] is not None}', [SK],
     "O-1: A-CNTJOIN compares the asked attribute's count with every name's keys"),
    # K3 round 9 (REVIEW 10b T-1, T-2, T-3: its three survivors by its definitions; the F model's and the
    # family-alone models' eval-set fails, which no test watched either)
    ("f_qnattr_all_names", "cues.py",
     '_oh("q_nattr", min(len({s["attr"] for s in S if s["name"] == nm and s["attr"] != a}), 3), 4)',
     '_oh("q_nattr", min(len({s["attr"] for s in S if s["name"] is not None and s["attr"] != a}), 3), 4)', [CU],
     "T-1: q_nattr counts every name's other attributes in named statements, not the asked name's"),
    ("ta_a_eval_fails_off", "cuegate.py", '        if not O.rate_ok(c, BAR["A"], out["n"][f"A.{sn}"]):',
     '        if sn == "train" and not O.rate_ok(c, BAR["A"], out["n"][f"A.{sn}"]):', [CU],
     "T-2: A's attribute-blind side fails on training only, never on an eval set"),
    ("f_model_ev_fails_off", "cuegate.py", '"n_train": len(FF)}\n        out["fails"] = out["fails"] + rf["fails"]',
     '"n_train": len(FF)}\n        out["fails"] = out["fails"] + [x for x in rf["fails"] if " train" in x]', [CU],
     "T-2: the F model's eval-set fails never reach run()'s"),
    ("fam_alone_ev_fails_off", "cuegate.py", '"n_train": len(Ff)}\n        out["fails"] = out["fails"] + rf["fails"]',
     '"n_train": len(Ff)}\n        out["fails"] = out["fails"] + [x for x in rf["fails"] if " train" in x]', [CU],
     "T-2: the family-alone models' eval-set fails never reach run()'s"),
    ("f_ante_rule_rate", "purity.py", 'if any(f.get("ante_same") is False for f in facts):',
     'if 8 * sum(f.get("ante_same") is False for f in facts) > sum(f.get("ante_same") is not None for f in facts):',
     [RF], "T-3: BD-48's per-item rule fails only past 1/8 of the checked items, not on one"),
    # K3 round 10 (REVIEW 11b T-1 to T-5: its four surviving mutants of rounds 6-8 and the earlier attempt's two,
    # by their definitions in k3/rev11b2/mine.py)
    ("m11_ta_fams_no_p", "cuegate.py", 'TA_FAMS, TA_BAR = "RUFP", .65', 'TA_FAMS, TA_BAR = "RUF", .65', [CU],
     "T-1: the attribute-blind model never runs on P (no P TA ceiling, no P TA fail)"),
    ("m11_purity_comp_one_hop", "purity.py",
     '            todo += [x["name"] for x in S if x["attr"] in ats and x["name"] is not None and x["name"] not in comp]',
     '            todo += [x["name"] for x in S if x["attr"] in ats and x["name"] is not None and x["name"] not in comp]'
     ' if u in hold else []', [SK], "T-2: purity's component stops one hop from the final name and the holders"),
    ("m11_ceilings_o14a_in", "check_k.py", 'MARKS = ("O13e", "O13a", "O14e", "O14a", "SURFe", "SURFa")',
     'MARKS = ("O13e", "O13a", "O14e", "SURFe", "SURFa")', [GE], "T-3: O14a, a one-kind mark, may set F's ceiling"),
    ("m11_antesize_smaller_only", "oracles.py",
     '    return S[refs[0] if (ci > cj) == (nma == 1) else refs[1]]["val"]',
     '    return S[refs[0] if ci < cj else refs[1]]["val"]', [RF],
     "T-4: F-ANTESIZE always takes the smaller antecedent count (its one-attribute branch reversed)"),
    ("r11_pvhold_counts_statements", "cues.py", '_oh("pv_hold", min(ahold[S[i - 1]["attr"]], 3) if i else 0, 4)',
     '_oh("pv_hold", min(acnt[S[i - 1]["attr"]], 3) if i else 0, 4)', [CU],
     "T-5: pv_hold reads the statement count of the attribute before a candidate, not its holders"),
    ("r11_ta_side_blank", "cues.py", "        side = _ta_side(S, nm, X)",
     "        side = [(c, 0.0) for c, _ in _ta_side(S, nm, X)]", [CU],
     "T-5: A's attribute-blind side reads nothing of the asked name's side (its 23 columns all 0)"),
    # K3 round 11 (REVIEW 12b T-1 to T-3 and T-5 to T-8: its surviving mutants, by their definitions in
    # k3/rev12g/mine_defs.py; T-4 and D-1 are generator and cue-model bugs, reported, not mutants)
    ("r12f_ta_vrank_off", "cues.py", '("vrank", vals.index(S[lt]["val"]) / max(1, len(vals) - 1)),',
     '("vrank", 0.0),', [CU], "T-1: the attribute-blind model's vrank (its value-order reader) all 0"),
    ("m12g_check_evals_no_p", "check_k.py", 'r.get("fam") in tuple("RUBFAP")]', 'r.get("fam") in tuple("RUBFA")]',
     [GE], "T-2: gen.py check reads no P item of any eval set"),
    ("m12g_ood_fails_dropped", "check_k.py", '            out["fails"] += [f"{tag}: {x}" for x in rep["fails"]]',
     '            out["fails"] += []', [GE], "T-3: the OOD sets' fails never reach check.json's"),
    ("r12e_kdev_split_evals", "check_k.py", '"k_dev": "eval"', '"k_dev": "evals"', [GE],
     "T-5: K-DEV checked as K-EVAL-S's split (S names its own)"),
    ("m12_eval_surf_self_fit", "check_k.py", "rep = U.accept(recs, pl, split, lex, tok, block, surf_fit=fit)",
     "rep = U.accept(recs, pl, split, lex, tok, block, surf_fit=None)", [GE],
     "T-6: SURF on the eval sets fitted on the set itself, not on the training stream"),
    ("m12_early_comp_not_p", "purity.py", '"early_comp": bool(early_comp) and (rec["fam"] in "RUP" or',
     '"early_comp": bool(early_comp) and (rec["fam"] in "RU" or', [SK], "T-7: purity's component rule skips P"),
    ("r12f_ceilings_a_no_ta", "check_k.py",
     'ta = gate.get("ta_model", {}).get("ceiling", {}).get(fam, {}).get(sn)',
     'ta = gate.get("ta_model", {}).get("ceiling", {}).get(fam, {}).get(sn) if fam != "A" else None', [GE],
     "T-8: SPEC 9's A ceiling ignores A's attribute-blind side"),
    # K3 round 12 (REVIEW 13b T-1 to T-5: its seven surviving mutants, by their definitions in k3/rev13c/mine_defs.py)
    ("m13b_fact_fails_dropped", "check_k.py",
     'out["fails"] += [f"fact {arm}: {k} {v}" for arm, b in out["fact_purity"].items() for k, v in b.items() if v]',
     'out["fails"] += []', [GE], "T-1: the fact streams' purity counts never reach check.json's fails"),
    ("m13b_fact_bio_token_off", "check_k.py", 'bad["skill_or_heldout_token"] += len(cls & forbidden)',
     'bad["skill_or_heldout_token"] += 0 * len(cls & forbidden)', [GE],
     "T-2: a skill-pool or held-out token in a fact bio doc is never counted"),
    ("m13b_k5_miss_l100_off", "check_k.py", 'bad["miss_in_L100"] += arm == "L100" and K.NONE in tool',
     'bad["miss_in_L100"] += 0', [GE], "T-2: a miss line in the L100 arm is never counted"),
    ("m13b_k5_non_low_off", "check_k.py", 'bad["non_low_name"] += tuple(tool[1:4]) not in low',
     'bad["non_low_name"] += 0', [GE], "T-2: a K5 tool line for an entity outside FB_low is never counted"),
    ("m13b_group_count_id_only", "check_k.py", 'if name in ("k_eval_id", "k_eval_s") and rep["groups"]',
     'if name in ("k_eval_id",) and rep["groups"]', [GE], "T-3: K-EVAL-S written without B groups passes"),
    ("m13c_ood_qrange_wide", "check_k.py", 'p_questions=(5, 7) if tag == "OOD-PLONG" else (2, 4))',
     'p_questions=(2, 7))', [GE], "T-4: gen.py check reads every OOD set's P question count against 2-7"),
    ("m13c_all_free_r_only", "purity.py", 'all_asked = rec["fam"] in "RU" and bool(free)',
     'all_asked = rec["fam"] in "R" and bool(free)', [SK], "T-5: U items that ask every bystander pass"),
    # K3 round 13 (REVIEW 14b T-1 to T-5: its six surviving mutants and its one killed one, by their definitions in
    # k3/rev14b/mine_defs.py; its four harmless survivors are not here)
    ("m14_pooled_vrank_off", "cues.py",
     'return avals.index(s["val"]) / max(1, len(avals) - 1) if am else 0.0', "return 0.0", [CU, GE],
     "T-1: the pooled model's vrank (value rank among the asked attribute's values) all 0"),
    ("m14_o5_bars_loose", "oracles.py", '"O5_1": (ALL, .65), "O5_2": (ALL, .65), "O5_3": (ALL, .65)',
     '"O5_1": (ALL, .95), "O5_2": (ALL, .95), "O5_3": (ALL, .95)', [PU, SK],
     "T-2: O5's token-run rules (the question's last 1-3 tokens in a statement) barred at 0.95 on every family"),
    ("m14_parse_fail_dropped", "purity.py",
     'fails = [f"{len(bad)} items do not parse, first: {bad[0]}"] if bad else []', "fails = []", [PU, SK, GE],
     "T-3: records that do not parse are dropped from acceptance silently"),
    ("m14_purity_filler_names_off", "purity.py", '{"F1", "F2", "F3", "Z1", "Z2", "Z3"}', '{"F1", "F2", "F3"}',
     [PU, SK], "T-4: a filler-pool name (Z1-Z3) in a skill or eval item passes purity"),
    ("m14_fact_qa_token_off", "check_k.py",
     'bad["skill_or_heldout_token"] += bool({own.get(x, "") for t in turns for x in t["ids"]} & forbidden)',
     'bad["skill_or_heldout_token"] += 0 * bool({own.get(x, "") for t in turns for x in t["ids"]} & forbidden)',
     [GE, W], "T-5: a skill or held-out token in a fact QA record is never counted"),
    ("m14_k5_token_off", "check_k.py", 'bad["skill_or_heldout_token"] += bool({own.get(x, "") for x in ids} & {',
     'bad["skill_or_heldout_token"] += 0 * bool({own.get(x, "") for x in ids} & {', [GE, W],
     "T-5: a skill or held-out token in a K5 chat is never counted"),
    ("m14_cube_kevs_unchecked", "skill.py",
     "if not any((a, b, c) in self.kevs for a in toks[0] for b in toks[1] for c in toks[2]):", "if True:",
     [SK, PU], "a training B cube may hold a K-EVAL-S triple (REVIEW 14b: killed)"),
    # K3 round 13's own: the parts of T-2's and T-3's tests the review's mutants do not reach first
    ("m13r_o5_bars_rp_only", "oracles.py", '"O5_1": (ALL, .65), "O5_2": (ALL, .65), "O5_3": (ALL, .65)',
     '"O5_1": ("RP", .65), "O5_2": ("RP", .65), "O5_3": ("RP", .65)', [PU, SK],
     "T-2: O5's bars on R and P only (U, B, F and A unbarred)"),
    ("m13r_parse_fail_rules_only", "purity.py",
     'fails = [f"{len(bad)} items do not parse, first: {bad[0]}"] if bad else []',
     'fails = [f"{len(bad)} items do not parse, first: {bad[0]}"] if bad and rules else []', [PU, SK, GE],
     "T-3: OOD sets (rules off) drop their unparseable records silently"),
    # G-2 (2026-10-11): REVIEW 12b's T-4 (DFAR pads) and D-1 (osize) fixed; m12g_ta_osize_off (k3/rev12g/mine_defs.py),
    # equivalent to the code until G-2, is live now
    ("g2_pad_redraw_off", "skill.py", "        if asked and joins(e, a):", "        if False:", [SK],
     "T-4: a DFAR pad may join an earlier question's entity to the final key's component (the pre-G-2 pad)"),
    ("g2_pad_redraw_any", "skill.py", "if (x, y) != key and (y != key[1] or x in hold) and not joins(x, y)]",
     "if (x, y) != key and (y != key[1] or x in hold)]", [SK],
     "T-4: a joining pad is redrawn among every pad, joining ones included"),
    ("g2_ood_item_rules_off", "purity.py", "if rules else ({}, item_fails(facts))", "if rules else ({}, [])",
     [SK, PU], "T-4: gen.py check's OOD read (rules off) skips SPEC 4's per-item rules"),
    ("g2_osize_dead", "cues.py", 'oth = [size[u] for u in hx if u != nm and u[0] != "ref"]',
     "oth = [size[u] for u in hx if u != nm and not isinstance(u, tuple)]", [CU, GE],
     "D-1: osize drops every holder (the pre-G-2 code: 0 on every item)"),
    ("m12g_ta_osize_off", "cues.py", '("osize", (sum(oth) / len(oth) if oth else 0.0) / 6.0)', '("osize", 0.0)',
     [CU, GE], "D-1: the TA model's osize (the other holders' mean size) all 0"),
    # K3 round 14 (2026-10-11): REVIEW 15b's T-1, T-7 and T-8 survivors (k3/rev15b/mut15.py; their test files
    # changed to where the round's tests are) and the round's own
    ("r15_skill_unshuffled", "gen.py", 'order = K.rng("skill_train", 1).permutation(len(recs))',
     "order = np.arange(len(recs))", [GE], "T-1: the SKILL stream is written in family blocks"),
    ("r15_o8_reads_last", "oracles.py", 'out["O8"] = S[-2]["val"] if len(S) >= 2 else -1',
     'out["O8"] = S[-1]["val"] if len(S) >= 2 else -1', [RF], "T-7: O8 reads the last statement, not the one before"),
    ("r15_o7_tie_earliest", "oracles.py", "max(reversed(vals), key=lambda v: cnt[v])",
     "max(vals, key=lambda v: cnt[v])", [RF], "T-7: O7's tie goes to the earliest value, not the latest"),
    ("r14_o7_reads_last", "oracles.py", 'out["O7"] = max(reversed(vals), key=lambda v: cnt[v]) if vals else -1',
     'out["O7"] = vals[-1] if vals else -1', [RF], "T-7: O7 reads the prompt's last value, not its most frequent"),
    ("r15_block_exact_off", "purity.py", 'exact = len(F) == block and not (name[0] == "k" and name[1:2].isdigit())',
     "exact = False", [PU], "T-8: full blocks need the 1/m shares within m items only, not exactly"),
    ("r14_exact_k_prefix", "purity.py", 'exact = len(F) == block and not (name[0] == "k" and name[1:2].isdigit())',
     'exact = len(F) == block and not name.startswith("k")', [PU],
     "T-8: key_named and key_alias exempt from exactness as k cells (the pre-round-14 test)"),
    ("r15_pad_redraw_first", "skill.py", "e, a = ok[int(r.integers(0, len(ok)))]", "e, a = ok[0]", [SK],
     "T-8: G-2's redraw always takes the first non-joining pad, not a uniform one"),
    ("r15_ood_rules_skip_ellipsis_ante", "purity.py", 'if any(f.get("ante_same") is False for f in facts):',
     'if any(f.get("ante_same") is False for f in facts) and len(facts) > 300:', [RF],
     "T-8: BD-48's every-item rule read only on sets over 300 records (the OOD columns hold 256)"),
    # K3 round 15 (2026-10-12): REVIEW 16b's T-1 to T-5 survivors of these files (k3/rev16b/mut16.py, mut16b.py; test
    # files changed to where the round's tests are), its equivalent-judged r16_lang_share_unnormalized, and four of
    # the round's own (r15_cands_none_dropped, r15_pool_is_first_one, r16_check_no_sha, r16_check_sha_no_gate)
    ("r16_fam_ceiling_replaces", "cuegate.py",
     'out["ceiling"].setdefault(fam, {})[sn] = max(c, out["ceiling"].get(fam, {}).get(sn, 0.0))',
     'out["ceiling"].setdefault(fam, {})[sn] = c', [CU],
     "T-1: a family-alone model's ceiling replaces the pooled one even when lower"),
    ("r16_ta_bar_066", "cuegate.py", 'TA_FAMS, TA_BAR = "RUFP", .65', 'TA_FAMS, TA_BAR = "RUFP", .66', [CU],
     "T-4: T's bar 0.66, not SPEC 5's 0.65"),
    ("r16_ta_e_span_dead", "cues.py", '("e_span", min(S[lt]["ex"] - S[ex[x][0]]["ex"], 12) / 12.0)',
     '("e_span", 0.0)', [CU], "T-2: TA column e_span reads 0 always"),
    ("r16_ta_eq2_dead", "cues.py", '("eq2", float(eq[x] > 1))', '("eq2", 0.0)', [CU], "T-2: TA column eq2 reads 0"),
    ("r16_ta_after_qn_dead", "cues.py",
     '("after_qn", float(lure_at is not None and any(i >= lure_at for i in ex[x])))', '("after_qn", 0.0)', [CU],
     "T-2: TA column after_qn reads 0 always (U's lure place)"),
    ("r16b_ta_x_last_stmt_dead", "cues.py",
     '("x_last_stmt", float(max(i for i, s in enumerate(S) if s["attr"] == x) == lt))', '("x_last_stmt", 0.0)',
     [CU], "T-2: TA column x_last_stmt reads 0"),
    ("r16b_ta_e_first_dead", "cues.py", '("e_first", float(ex[x][0] == E[0]))', '("e_first", 0.0)', [CU],
     "T-2: TA column e_first reads 0"),
    ("r16b_pool_is_pen_dead", "cues.py", '("is_pen", float(i == n - 2)), ("is_first"', '("is_pen", 0.0), ("is_first"',
     [CU], "T-2: pooled column is_pen reads 0"),
    ("r16b_pool_own_sup_dead", "cues.py", '("own_sup", float(named and own[-1] != i))', '("own_sup", 0.0)', [CU],
     "T-2: pooled column own_sup reads 0"),
    ("r16b_pool_ans_other_attr_dead", "cues.py", '("ans_other_attr", float(any(p["attr"] != a for p in ans)))',
     '("ans_other_attr", 0.0)', [CU], "T-2: pooled column ans_other_attr reads 0"),
    ("r16b_f_y_kdef_ex_dead", "cues.py", '("y_kdef_ex", min(dk[0], 20) / 20.0 if dk else 0.0)', '("y_kdef_ex", 0.0)',
     [CU], "T-2: F-model column y_kdef_ex reads 0"),
    ("r15_pool_is_first_one", "cues.py", '("is_first", float(i == 0))', '("is_first", 1.0)', [CU],
     "T-2: pooled column is_first stuck at 1 (the \" none\" row's 0 beside it)"),
    ("r16_cands_two_extra", "oracles.py", "extra = [int(x) for x in r.choice(pool, 3, replace=False)]",
     "extra = [int(x) for x in r.choice(pool, 2, replace=False)]", [PR], "T-3: SPEC 9 C: two outside values, not three"),
    ("r15_cands_none_dropped", "oracles.py", "+ extra + ([K.NONE] if gold != K.NONE else [])", "+ extra", [PR],
     "T-3: SPEC 9 C without its \" none\" candidate"),
    ("r16_last_rate_tol", "purity.py", "max(0.05, 3 * math.sqrt(exp", "max(0.10, 3 * math.sqrt(exp", [PU],
     "T-4: BD-2's last-statement floor 0.10, not 0.05"),
    ("r16_u_mirror_tol", "purity.py", "if abs(m - mm) > 3 * math.sqrt(m + mm):",
     "if abs(m - mm) > 6 * math.sqrt(m + mm):", [PU], "T-4: U's mirror-cell tolerance 6 SD, not 3"),
    ("r16_o6_first_of_turn", "oracles.py", 'out["O6"] = _last(S, lambda s: s["turn"] == last_turn)',
     'out["O6"] = next((s["val"] for s in S if s["turn"] == last_turn), -1)', [RF],
     "T-5: O6 reads the last turn's first statement, not its last"),
    ("r16_lang_share_unnormalized", "gen.py", "round(lang * s / tot, 6)", "round(lang * s, 6)", [GE],
     "the LANG sources' shares are not normalized to the LANG share (0.249975)"),
    ("r16_check_no_sha", "check_k.py",
     '    out["code_sha256"] = {c: K.sha256_file(os.path.join(K.HERE, c)) for c in CHECK_CODE}\n', "", [GE],
     "N-1: check.json records no code sha256"),
    ("r16_check_sha_no_gate", "check_k.py", 'CHECK_CODE = ("check_k.py", "cuegate.py", "cues.py",',
     'CHECK_CODE = ("check_k.py", "cues.py",', [GE], "N-1: check.json's code sha256 leaves the gate's out"),
    # K3 round 16 (2026-11-21/22): REVIEW 17b's survivors of these files that the round's tests kill (port/k2/mut17.py,
    # mut17c.py; targets character for character, test files where the round's tests are)
    ("r17_conflict_cands_no_memory", "evalsets.py", "c = [v] + ([true] if v != true else []) +", "c = [v] + [] +",
     [PR], "CONFLICT's candidates lack the memorized value (every memory rate would read 0)"),
    ("r17c_marker_always", "probes.py", 'out[j]["marker"] = int(g[k]) == pr[j]["marker"][1]',
     'out[j]["marker"] = True', [PR], "P's persistence marker always right"),
    ("r17_o2_bar_loose", "oracles.py", '"O2": (ALL, .50)', '"O2": (ALL, .55)', [RF], "O2's bar 0.55"),
    ("r17_o678_bars_loose", "oracles.py", '"O6": (ALL, .50), "O7": (ALL, .50), "O8": (ALL, .50)',
     '"O6": (ALL, .60), "O7": (ALL, .60), "O8": (ALL, .60)', [RF], "O6-O8 bars 0.60"),
    ("r17_o1_bar_loose", "oracles.py", '"O1": ("U", .50)', '"O1": ("U", .60)', [RF], "O1's bar 0.60"),
    ("r17_o4_bar_loose", "oracles.py", '"O4": ("F", .35)', '"O4": ("F", .45)', [RF], "O4's bar on F 0.45"),
    ("r17_o9_bars_loose", "oracles.py", '"O9": ("A", .50), "O10": ("A", .50), "O9|10": ("A", .50)',
     '"O9": ("A", .55), "O10": ("A", .55), "O9|10": ("A", .55)', [RF], "A's O9/O10 bars 0.55"),
    ("r17_o3_bars_loose", "oracles.py", '**{o: ("B", .65) for o in O3}', '**{o: ("B", .70) for o in O3}', [RF],
     "B's O3 bars 0.70"),
    # K3 round 16 (2026-11-22): REVIEW 17b's hunt survivors on these files (port/k2/mut17.py, mut17b.py, mut17c.py,
    # character for character) that the round's 11-22 tests kill
    ("r17_group_ceiling_first_learner", "cuegate.py", "c = max(v[sn] for v in groups.values() if sn in v)",
     "c = next(v[sn] for v in groups.values() if sn in v)", [CU], "B groups read the first learner only"),
    ("r17_cand_token_off", "purity.py",
     'bad["candidate_not_one_token"] += own.get(t["ids"][-1], "") not in ("V", "VH") and \\',
     'bad["candidate_not_one_token"] += 0 and \\', [PU], "an answer that is not a value token passes"),
    ("r17_both_lures_off", "purity.py", 'rules["both_lures"] = ([f["has_attr"] and f["has_name"] for f in F], 2)',
     "pass", [PU], "A's both-lures half unchecked"),
    ("r17_no_update_off", "purity.py", 'rules["no_update"] = ([f["u_cell"][:2] == (1, 2) for f in F], 4)',
     'rules["no_update"] = ([True for f in F], 1)', [PU], "U's no-update quarter unchecked"),
    ("r17_ideal_fail_off", "purity.py", '        if row["IDEAL"] != 1.0:\n            fails.append(f"{fam}: IDEAL',
     '        if False:\n            fails.append(f"{fam}: IDEAL', [PU], "a family whose IDEAL is not 1 passes (rules on)"),
    ("r17_rank_tol_double", "purity.py", "if abs(p - 1 / k) > tol + 1e-9:", "if abs(p - 1 / k) > 2 * tol + 1e-9:", [PU],
     "the pooled intro-rank tolerance doubled"),
    ("r17_block_tol_double", "purity.py", "if abs(sum(v) * m - len(v)) >= m or",
     "if abs(sum(v) * m - len(v)) >= 2 * m or", [PU], "a block cell off by one item of m allowed (non-full blocks)"),
    ("r17_ood_ideal_off", "purity.py",
     "fails += [f\"{f}: IDEAL {row['IDEAL']}\" for f, row in tab.items() if row[\"IDEAL\"] != 1.0]", "fails += []",
     [PU], "an OOD set whose IDEAL is not 1 passes (the report: the world-rule line still fails the set)"),
    ("r17_reset_seed_ignored", "ablate.py", "g = torch.Generator().manual_seed(int(seed))",
     "g = torch.Generator().manual_seed(0)", [AB], "K3's reset ignores its seed (7700 + s): every seed alike"),
    ("r17_check_code_no_purity", "check_k.py", '"cues.py", "purity.py", "oracles.py"', '"cues.py", "oracles.py"', [GE],
     "check.json's code sha256 leaves purity.py out"),
    ("r17_cf_hit_not_counterfactual", "evalsets.py", '            if tag == "CF-HIT":\n', "            if False:\n", [PR],
     "CF-HIT's tool states the memorized value (FB-HIT again)"),
    ("r17_unseen_hit_world_name", "evalsets.py",
     'nm = [int(p[int(r.integers(0, len(p)))]) for p in pools.triple("T")]',
     "nm = [int(x) for x in world.names[int(ents[0])]]", [PR], "UNSEEN-HIT asks a world entity's name"),
    ("r17c_chance_off", "probes.py", '"chance": float(np.mean([1 / o["n_c"] for o in g]))',
     '"chance": float(np.mean([1 / (o["n_c"] - 1) for o in g]))', [PR], "chance over |C| - 1"),
    ("r17c_bits_one_order", "probes.py", "ce = np.mean([np.mean(v) for v in per.values()])",
     "ce = np.mean([v[0] for v in per.values()])", [PR], "stored bits read one BIO order, not the mean of 2"),
    ("r17c_b_slots", "kcommon.py", "B_SLOTS = 7_630 * 32_768", "B_SLOTS = 7_630 * 32_000", [GE], "B under 250M"),
]


def copy(mut):
    d = tempfile.mkdtemp(prefix="k_mut_")
    shutil.copytree(CODE, os.path.join(d, "code"), ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    if mut is not None:
        p = os.path.join(d, "code", mut[1])
        src = open(p).read()
        n = src.count(mut[2])
        if n != 1:
            shutil.rmtree(d)
            return None, n
        open(p, "w").write(src.replace(mut[2], mut[3]))
    return d, 1


def run(d, files, timeout=600, why=False):
    """-> (pytest exit code, seconds[, the failing test and its assert line when why])."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "K_TEST_THREADS": "2",
           "K_CODE_ROOT": os.path.abspath(os.path.join(CODE, "..", "..", ".."))}
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                            "--tb=line" if why else "--tb=no", "-rf", *[os.path.join("tests", f) for f in files]],
                           cwd=os.path.join(d, "code"), env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return (-9, time.time() - t0, "") if why else (-9, time.time() - t0)
    tail = " | ".join(x.strip()[:220] for x in r.stdout.splitlines() if x.startswith(("/", "E ", "FAILED")) or
                      ".py:" in x)[:900]
    return (r.returncode, time.time() - t0, tail) if why else (r.returncode, time.time() - t0)


def _done(log):
    """{name: status} of the verdict lines already in log (the last line per name), {} without the file."""
    out = {}
    if log and os.path.exists(log):
        for x in open(log):
            w = x.split()
            if len(w) >= 2 and (w[0] in ("killed", "SURVIVED") or w[0].startswith("INVALID")):
                out[w[1]] = w[0]
    return out


def _emit(log, line):
    print(line, flush=True)
    if log:
        with open(log, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="run the unmutated copy on every file mutants use")
    ap.add_argument("--why", action="store_true", help="print each killed mutant's failing test and assert line")
    ap.add_argument("--log", help="append verdicts here as each finishes; skip those already killed or SURVIVED")
    a = ap.parse_args(argv)
    if a.list:
        for m in M:
            print(f"{m[0]:28s} {m[1]:12s} {','.join(m[4])}  {m[5]}")
        return 0
    if a.baseline:
        d, _ = copy(None)
        code, dt = run(d, sorted({f for m in M for f in m[4]}), timeout=1200)
        shutil.rmtree(d, ignore_errors=True)
        print(f"baseline exit {code} ({dt:.0f}s)")
        return 0 if code == 0 else 1
    ok, done = True, _done(a.log)
    for m in M:
        if a.id and m[0] not in a.id or done.get(m[0]) in ("killed", "SURVIVED"):
            continue
        d, n = copy(m)
        if d is None:
            _emit(a.log, f"INVALID   {m[0]} (target occurs {n} times)  [{time.strftime('%H:%M:%S')}]")
            ok = False
            continue
        code, dt, *tail = run(d, m[4], why=a.why)
        shutil.rmtree(d, ignore_errors=True)
        status = {0: "SURVIVED", 1: "killed"}.get(code, f"INVALID({code})")
        ok &= status == "killed"
        _emit(a.log, f"{status:9s} {m[0]} ({dt:.0f}s)  {m[5]}  [{time.strftime('%H:%M:%S')}]")
        if tail and tail[0]:
            _emit(a.log, f"          why: {tail[0]}")
    done = _done(a.log) if a.log else {}
    names = [m[0] for m in M if not a.id or m[0] in a.id]
    if a.log:
        st = [done.get(x, "not run") for x in names]
        tally = {k: sum(s.startswith(k) for s in st) for k in ("killed", "SURVIVED", "INVALID", "not run")}
        print(f"tally ({a.log}): " + ", ".join(f"{k} {v}" for k, v in tally.items()) + f" of {len(names)}")
        return 0 if tally["killed"] == len(names) else 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
