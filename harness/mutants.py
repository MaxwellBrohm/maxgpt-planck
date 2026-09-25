"""Deliberate bugs for mutation_check.py. Each mutant replaces ONE exact source string
(it must occur exactly once) in a scratch copy of the harness; the step-3 suite must go red.

Fields: id, group, file, old, new, why. equivalent=True marks a mutant that is known to
change nothing observable (it is reported, and expected to survive).
"""
M = []


def mut(id, group, file, old, new, why, equivalent=False):
    M.append(dict(id=id, group=group, file=file, old=old, new=new, why=why, equivalent=equivalent))


# ---------------- document masking ----------------
mut("mask_docs_see_each_other", "mask", "model.py",
    "same = run[:, :, None] == run[:, None, :]",
    "same = torch.ones_like(run[:, :, None] == run[:, None, :])",
    "packed documents attend to earlier documents")
mut("mask_bidirectional", "mask", "model.py", "return (same & causal)[:, None]",
    "return same[:, None]", "inside a document, tokens see their future")
mut("mask_one_future_token", "mask", "model.py",
    "causal = torch.ones(T, T, dtype=torch.bool, device=doc.device).tril()",
    "causal = torch.ones(T, T, dtype=torch.bool, device=doc.device).tril(1)",
    "off by one: each token sees the next one")
mut("mask_id_equality", "mask", "model.py", "same = run[:, :, None] == run[:, None, :]",
    "same = doc[:, :, None] == doc[:, None, :]",
    "a reused doc id joins an earlier, different document")
mut("mask_dropped_in_forward", "mask", "model.py", "mask = document_causal_mask(doc)",
    "mask = None", "doc ids ignored: packed rows run plain causal")
mut("mask_ignored_by_attention", "mask", "blocks.py",
    "out = F.scaled_dot_product_attention(q, k_, v_, attn_mask=mask)",
    "out = F.scaled_dot_product_attention(q, k_, v_, is_causal=True)",
    "attention takes the causal flag instead of the document mask")
mut("mask_trainer_drops_doc", "mask", "trainer.py", 'self._to(b["doc"]),', "None,",
    "the trainer forgets to pass doc ids")
# ---------------- causality ----------------
mut("causal_flag_off", "causal", "blocks.py",
    "out = F.scaled_dot_product_attention(q, k_, v_, is_causal=True)",
    "out = F.scaled_dot_product_attention(q, k_, v_, is_causal=False)",
    "plain path is bidirectional")
mut("causal_leak_only_nograd", "causal", "blocks.py",
    "out = F.scaled_dot_product_attention(q, k_, v_, is_causal=True)",
    "out = F.scaled_dot_product_attention(q, k_, v_, is_causal=torch.is_grad_enabled())",
    "the MPS failure mode: leaks only in no-grad (eval) passes")
mut("causal_gate_reads_next_token", "causal", "blocks.py",
    "gate = 2.0 * torch.sigmoid(self.attn_gate_proj(x))",
    "gate = 2.0 * torch.sigmoid(self.attn_gate_proj(x.roll(-1, dims=1)))",
    "leak through the attention gate (invisible at init: zero-init gate)")
mut("causal_value_residual_future", "causal", "model.py", "v1 = v_loc",
    "v1 = v_loc.roll(-1, dims=2)",
    "leak through the value residual (invisible at init: alpha2 = 0)")
mut("causal_norm_over_time", "causal", "blocks.py", "x.pow(2).mean(-1, keepdim=True)",
    "x.pow(2).mean(1, keepdim=True)", "RMSNorm averages over the time axis")
# ---------------- packing and bucketing ----------------
mut("pack_all_one_doc", "packing", "packing.py", "doc[at:at + n] = d", "doc[at:at + n] = 0",
    "every item in a row gets the same doc id")
mut("pack_last_token_predicts_pad", "packing", "packing.py",
    "tgt[at:at + n] = item_targets(ids, flags)",
    "tgt[at:at + n] = np.append(item_targets(ids, flags)[:-1], ids[-1])",
    "an item's last token gets a target outside the item")
mut("bucket_left_pad", "packing", "packing.py", "idx[r, :len(ids)] = ids",
    "idx[r, b - len(ids):] = ids", "bucket rows padded on the left, targets misaligned")
mut("pack_positions_not_restarted", "packing", "packing.py", "pos[at:at + n] = np.arange(n)",
    "pos[at:at + n] = np.arange(at, at + n)",
    "RoPE positions continue across documents (no-op under a doc mask)", equivalent=True)
# ---------------- loss masking ----------------
mut("loss_user_supervised", "loss", "chat_template.py",
    'self.loss == "all" or role == "assistant"', 'self.loss == "all" or role != "system"',
    "user and tool turns are supervised")
mut("loss_end_unsupervised", "loss", "chat_template.py",
    "ids.append(self.end_id)\n            flags.append(sup_turn)",
    "ids.append(self.end_id)\n            flags.append(False)", "<|end|> is never learned")
mut("loss_role_token_supervised", "loss", "chat_template.py",
    'flags.append(self.loss == "all" and t.get("loss", True))', "flags.append(sup_turn)",
    "the <|assistant|> role token becomes a target")
mut("loss_off_switch_ignored", "loss", "chat_template.py",
    'sup_turn = t.get("loss", True) and', "sup_turn = True and", '"loss": false is ignored')
mut("loss_shifted_by_one", "loss", "packing.py", "np.where(flags[1:], ids[1:], IGNORE)",
    "np.where(flags[:-1], ids[1:], IGNORE)", "loss flags shifted one position")
mut("loss_trainer_counts_all", "loss", "trainer.py",
    'n_sup = sum(int((b["tgt"] != -100).sum()) for b in batches)',
    'n_sup = sum(int(b["tgt"].numel()) for b in batches)', "normalizes by all tokens")
# ---------------- resume ----------------
mut("resume_no_optimizer", "resume", "trainer.py",
    'self.opt.load_state_dict(ck["optimizer"])', "pass", "optimizer state not restored")
mut("resume_loader_restarts_window", "resume", "data.py", 'self._u = st["unit"]',
    "self._u = 0", "loader replays the window from its start")
mut("resume_step_lost", "resume", "trainer.py", 'self.step = int(ck["step"])',
    'self.step = int(ck["step"]) - 1', "step counter off by one")
# ---------------- schedule ----------------
mut("sched_warmup_zero_first", "schedule", "schedule.py",
    "return (step + 1) / self.warmup_steps", "return step / self.warmup_steps",
    "warmup starts at lr 0")
mut("sched_decay_off_by_one", "schedule", "schedule.py",
    "max(0.0, (self.total_steps - step) / max(1, self.decay_steps))",
    "max(0.0, (self.total_steps - step - 1) / max(1, self.decay_steps))",
    "decay ends one step early")
mut("sched_branch_length", "schedule", "schedule.py",
    "round(at_step * decay_frac / (1 - decay_frac))", "round(at_step * decay_frac)",
    "branch decay length is f*s instead of s*f/(1-f)")
mut("sched_branch_in_warmup", "schedule", "schedule.py",
    "assert at_step >= parent.warmup_steps,", "assert at_step >= 0,",
    "branching from a warmup checkpoint is allowed")
mut("sched_trainer_off_by_one", "schedule", "trainer.py",
    "set_lr(self.opt, self.sched.factor(self.step))",
    "set_lr(self.opt, self.sched.factor(self.step + 1))", "trainer uses next step's lr")
# ---------------- optimizer ----------------
mut("opt_tied_embed_in_normuon", "optim", "optim.py",
    'if name.startswith(("tok_emb.", "lm_head.")):', 'if name.startswith(("lm_head.",)):',
    "tied embedding optimized by NorMuon")
mut("opt_scalars_decay", "optim", "optim.py",
    '"base_lr": float(ocfg.get("scalar_lr", lr)), "weight_decay": 0.0',
    '"base_lr": float(ocfg.get("scalar_lr", lr)), "weight_decay": wd',
    "norm gains get weight decay")
mut("opt_wrong_sign", "optim", "optim.py", "p.add_(O, alpha=-lr)", "p.add_(O, alpha=lr)",
    "NorMuon ascends")
mut("opt_plain_muon", "optim", "optim.py", 'normalize=(kind == "normuon")', "normalize=False",
    "NorMuon silently runs as plain Muon")
mut("opt_embed_lr_ignored", "optim", "optim.py", '"base_lr": float(ocfg.get("embed_lr", lr))',
    '"base_lr": lr', "embed_lr has no effect")
# ---------------- parameter budget ----------------
mut("budget_vr_count", "budget", "budget.py", "n += 3  ", "n += 2  ", "VR scalars miscounted")
mut("budget_qknorm_count", "budget", "budget.py", "n += 2 * hd", "n += hd", "QK-norm miscounted")
mut("budget_untied_once", "budget", "budget.py", "(1 if cfg.tie_embeddings else 2)", "1",
    "untied head not counted")
mut("budget_share_ignored", "budget", "budget.py", "if cfg.qk_owner(u) == u:", "if True:",
    "shared W_q/W_k counted per layer")
mut("count_no_dedup", "budget", "count_params.py", "model.parameters())",
    "model.parameters(remove_duplicate=False))", "ground truth counts tied weights twice")
