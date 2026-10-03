"""SCREENS ORDER 0 mutants (eval.induction, attn_diag.py, test_model_mask.py on testutil.ARMS), split from mutants.py
for the file size. Imported at the end of mutants.py, which it appends to (mutation_check.py --group induction |
attndiag | maskarms). Each names the test file that must go red."""
from mutants import mut

IND, AD, MM = ["test_induction.py"], ["test_attn_diag.py"], ["test_model_mask.py"]
# ---------------- eval.induction (induction.py, the train.py install) ----------------
mut("ind_copies_swapped", "induction", "induction.py", '"ind": s1 / n - s2 / n', '"ind": s2 / n - s1 / n',
    "IND = copy 2 minus copy 1 (sign flipped)", tests=IND)
mut("ind_copy2_off_by_one", "induction", "induction.py", "s2 += float(nll[:, L:2 * L - 1].sum())",
    "s2 += float(nll[:, L - 1:2 * L - 2].sum())", "copy 2 scored from its position 1 (unpredictable by induction)",
    tests=IND)
mut("ind_reads_the_loader", "induction", "induction.py", "            rec = self.reading(trainer.model, step)",
    "            trainer.loader.next_batch()\n            rec = self.reading(trainer.model, step)",
    "a reading consumes a batch: the training stream shifts", tests=IND)
mut("ind_draws_rng", "induction", "induction.py", '        rec = {"step": step, **ind_score(',
    '        torch.rand(1)\n        rec = {"step": step, **ind_score(', "a reading draws from the torch RNG", tests=IND)
mut("ind_rng_guard_removed", "induction", "induction.py", "        if not same:\n            raise",
    "        if False:\n            raise", "an RNG change inside a reading goes unnoticed", tests=IND)
mut("ind_skips_last_step", "induction", "induction.py",
    "(step % self.every and step != trainer.sched.total_steps)", "(step % self.every)",
    "no reading at the run's last step (7,630 is not a multiple of 153)", tests=IND)
mut("ind_gate_never", "induction", "induction.py", 'if model.cfg.attn_gate and icfg.get("gate_evalset"):',
    "if False:", "the GATE curve is never logged", tests=IND)
mut("ind_gate_without_gate", "induction", "induction.py", 'if model.cfg.attn_gate and icfg.get("gate_evalset"):',
    'if icfg.get("gate_evalset"):', "GATE read on a model without the gate", tests=IND)
mut("ind_sha_unchecked", "induction", "induction.py", "if sha256 is not None and got != sha256:", "if False:",
    "a changed IND file is accepted", tests=IND)
mut("ind_not_installed", "induction", "train.py", "            tr.hooks.append(ind)\n", "            pass\n",
    "train.py builds the hook but never installs it", tests=IND)
# ---------------- attn_diag.py: the tool must rebuild each arm's real attention ----------------
mut("diag_skips_qknorm", "attndiag", "attn_diag.py",
    "    if attn.qk_norm:\n        q, k = attn.q_norm(q), attn.k_norm(k)", "    pass",
    "the tool skips QK-norm (reads attention the model never computed)", tests=AD)
mut("diag_omits_forget_bias", "attndiag", "attn_diag.py",
    "    if attn.forget_gate:\n        s = s + attn.forget_bias(", "    if False:\n        s = s + attn.forget_bias(",
    "the tool omits S005's forget bias", tests=AD)
mut("diag_skips_smear", "attndiag", "attn_diag.py",
    'k = attn.smear(k_raw, a.get("smear")) if attn.smear_key else k_raw', "k = k_raw",
    "the tool reads raw keys on a smear_key arm", tests=AD)
mut("diag_no_value_residual", "attndiag", "attn_diag.py", "    if attn.has_vr and v1 is not None:",
    "    if False:", "the tool leaves out the value residual mix", tests=AD)
mut("diag_rope_skipped", "attndiag", "attn_diag.py",
    '    q, k = apply_rope(q, a["cos"], a["sin"]), apply_rope(k, a["cos"], a["sin"])\n', "",
    "the tool reads un-rotated q and k", tests=AD)
mut("diag_sink_wrong_key", "attndiag", "attn_diag.py", "s = (p[..., 0] * qsel[:, None, :])",
    "s = (p[..., 1] * qsel[:, None, :])", "SINK reads key 1, not the first token", tests=AD)
mut("diag_pads_counted", "attndiag", "attn_diag.py",
    "qsel = real & (torch.arange(T, device=x.device)[None, :] >= SINK_FROM)",
    "qsel = (torch.arange(T, device=x.device)[None, :] >= SINK_FROM)[:, :].expand_as(real)",
    "pad query positions enter the SINK average (depends on the batch)", tests=AD)
mut("diag_check_blind", "attndiag", "attn_diag.py", "self.max_err = max(self.max_err, err / scale)", "pass",
    "the check never sees an error", tests=AD)
# ---------------- test_model_mask.py on testutil.ARMS (scrambled): flag leaks on the doc=None path ----------------
mut("mm_canon_ignores_doc_starts", "maskarms", "blocks.py", "s = s.masked_fill(skip[j - 1], 0.0)", "s = s",
    "Canon taps reach across packed document starts", tests=MM)
mut("mm_canon_only_with_doc", "maskarms", "blocks.py", "if self.canon_a is not None:",
    "if self.canon_a is not None and canon_skip is not None:", "Canon-A skipped on the doc=None path", tests=MM)
mut("mm_smear_crosses_doc_start", "maskarms", "model.py", "smear = docattn.doc_starts(doc)", "smear = None",
    "packed keys smear across documents", tests=MM)
mut("mm_smear_only_with_doc", "maskarms", "blocks.py", "if self.smear_key:                    # S007",
    "if self.smear_key and smear is not None:  # S007", "no smear on the doc=None path", tests=MM)
mut("mm_fg_only_with_doc", "maskarms", "blocks.py", "if self.forget_gate:                  # S005",
    "if self.forget_gate and mask is not None:  # S005", "no forget bias on the doc=None path", tests=MM)
mut("mm_docs_see_each_other", "maskarms", "model.py", "same = run[:, :, None] == run[:, None, :]",
    "same = torch.ones_like(run[:, :, None] == run[:, None, :])", "packed documents attend to earlier ones",
    tests=MM)
mut("mm_vr_across_docs", "maskarms", "blocks.py", "            v = self.vr_scale * (a1 * v + a2 * v1)",
    "            v = self.vr_scale * (a1 * v + a2 * v1.roll(1, 2))",
    "value residual reads the previous token's v1 (zero weight at init: invisible unscrambled)", tests=MM)
