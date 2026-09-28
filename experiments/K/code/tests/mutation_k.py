"""Mutation check for the track K code: one deliberate bug per scratch copy of experiments/K/code; KILLED = the named
test files fail (pytest exit 1). A target string that does not occur exactly once, or any other pytest exit code,
is INVALID. The unmutated copy runs first and must be green.
  python tests/mutation_k.py [--id NAME] [--list] [--baseline]   (a mutant runs its named test files only, under
  600 s since K3 round 3, the machine being shared and test_masks' restarts taking 4 min; --baseline runs every file
  the mutants name, up to 1,200 s). K3 round 4: 139 mutants (20 new).
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
_ACTX = 'if (fl.get("ctx") or ("R" if r.random() < 0.5 else "U")) == "R":     # a block flag (skill.block_flags)\n'
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
    ("follow_not_balanced", "skill.py", 'f.update(follow=c == "f1", foil=c != "nofoil", k=k, rank=x)',
     'f.update(follow=c != "nofoil", foil=c != "nofoil", k=k, rank=x)', [SK], "R's follow rule drifts to 0.8"),
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
    ("u_pair_one_not_mirrored", "fams.py", "        pair = (2, 1)\n", "        pair = (2, 2)\n", [SK],
     "U: upd-one no longer mirrors noupd (a count cue)"),
    ("u_foil_entity_smaller", "fams.py", "n_attr = int(r.integers(2, 4)) if e < 2 else",
     "n_attr = int(r.integers(2, 4)) if e < 1 else", [SK], "U: the foil's entity smaller than the key's"),
    ("u_one_follows", "skill.py", 'follow=fol[i] if i in fol else v == "noupd", k=k, rank=rk, frank=fk)',
     'follow=fol[i] if i in fol else v != "twoslot", k=k, rank=rk, frank=fk)', [SK],
     "U: (count, recency) not mirrored"),
    ("u_rank_not_exact", "skill.py", 'follow=fol[i] if i in fol else v == "noupd", k=k, rank=rk, frank=fk)',
     'follow=fol[i] if i in fol else v == "noupd")', [SK], "U's (k, rank) drawn per item again"),
    ("a_both_own_structure", "fams.py", _ACTX + '            c = str(r.choice(["f1", "f0", "nofoil"], p=[.5, .3, .2]))',
     'if True:\n            c = "nofoil"', [CU],
     "A's both-items with one other entity stating the attribute: \" none\" wins by prior"),
    ("a_both_r_only", "fams.py", _ACTX + '            c = str(r.choice(["f1", "f0", "nofoil"]',
     'if True:\n            c = str(r.choice(["f1", "f0", "nofoil"]', [SK], "A's both-items from R contexts only"),
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
    ("u_mirror_block_rule_off", "purity.py", '            rules["mirror"] = ([f["u_cell"] == (2, 1, False) for f in '
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
    ("cue_draw_first_items", "cuegate.py", "    return np.sort(np.random.default_rng([K.W, 88]).choice(n, cap, "
     "replace=False))", "    return np.arange(cap)", [CU], "T-2: the gate fits the stream's first items"),
    ("set_learner_unpooled", "cuegate.py", '"set": dict(hidden=64, pool=True, epochs=12, lr=3e-3)',
     '"set": dict(hidden=64, pool=False, epochs=12, lr=3e-3)', [CU], "T-2: the set learner cannot compare"),
    ("f_model_turn_leak", "cues.py", '("last_in_turn", float(i == turn[-1]))',
     '("last_in_turn", float(any(S[x]["name"] == nm for x in turn)))', [CU],
     "GATE-F-BLIND: an F-model column reads an asked-name statement in the candidate's turn"),
    ("f_model_off_in_run", "cuegate.py", "    if FF:\n", "    if False:\n", [CU, GE], "the F model never runs"),
    ("f_model_fails_dropped", "cuegate.py", '        out["fails"] = out["fails"] + rf["fails"]',
     '        out["fails"] = out["fails"]', [CU], "the F model's fails never reach the report"),
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
    ("rp_rank_per_item", "skill.py", "            f.update(follow=c == \"f1\", foil=c != \"nofoil\", k=k, rank=x)",
     "            f.update(follow=c == \"f1\", foil=c != \"nofoil\")", [SK], "R's and P's (k, rank) per item again"),
    ("dfar_pad_makes_holders", "skill.py", "        if a == key[1] and e not in hold:", "        if False:", [SK],
     "a DFAR pad turns a bystander into a holder (it may be one an earlier question asked)"),
    ("u_foil_rank_free", "fams.py",
     "ok = base if fr is None else (lambda flat: base(flat) and intro_rank(flat, 1) == fr)", "ok = base", [SK],
     "S-2: the foil's intro rank drawn free again (a big entity, introduced early)"),
    ("u_frank_rule_off", "purity.py",
     'rules.update({f"k{k}_frank{rk}": ([f["frank"] == rk for f in Fk], k) for rk in range(k)})', "pass", [SK],
     "S-2: U's foil-rank block rule never checked"),
    ("u_rank_pairs_per_item", "skill.py", "for i, p in zip(idx, rank_pairs(len(idx), k, r)):",
     "for i, p in zip(idx, [tuple(int(x) for x in r.choice(k, 2, replace=False)) for _ in idx]):", [SK],
     "S-2: the (key, foil) rank pair drawn per item, not exact per block"),
    ("a_ctx_per_item", "fams.py", '        if (fl.get("ctx") or ("R" if r.random() < 0.5 else "U")) == "R":',
     "        if r.random() < 0.5:", [SK], "A's both-item context kind drawn per item (U contexts over-drawn)"),
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="run the unmutated copy on every file mutants use")
    ap.add_argument("--why", action="store_true", help="print each killed mutant's failing test and assert line")
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
    ok = True
    for m in M:
        if a.id and m[0] not in a.id:
            continue
        d, n = copy(m)
        if d is None:
            print(f"INVALID   {m[0]} (target occurs {n} times)")
            ok = False
            continue
        code, dt, *tail = run(d, m[4], why=a.why)
        shutil.rmtree(d, ignore_errors=True)
        status = {0: "SURVIVED", 1: "killed"}.get(code, f"INVALID({code})")
        ok &= status == "killed"
        print(f"{status:9s} {m[0]} ({dt:.0f}s)  {m[5]}", flush=True)
        if tail and tail[0]:
            print(f"          why: {tail[0]}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
