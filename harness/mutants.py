"""Deliberate bugs for mutation_check.py. Each mutant replaces ONE exact source string
(it must occur exactly once) in a scratch copy of the harness; the step-3 suite must go red.

Fields: id, group, file, old, new, why. equivalent=True marks a mutant that is known to
change nothing observable (it is reported, and expected to survive). tests: the test files
this mutant runs instead of the step-3 suite (the screen flags' own files); None = the suite.
"""
M = []


def mut(id, group, file, old, new, why, equivalent=False, tests=None):
    M.append(dict(id=id, group=group, file=file, old=old, new=new, why=why, equivalent=equivalent,
                  tests=tests))


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
# ---------------- doc_attn varlen (docattn.py; killed by test_s3_docattn.py on CPU) ------
mut("docattn_docs_span_rows", "docattn", "docattn.py",
    "start = torch.ones_like(doc, dtype=torch.bool)",
    "start = torch.zeros_like(doc, dtype=torch.bool); start[0, 0] = True",
    "a row start no longer starts a document: documents run across rows")
mut("docattn_last_doc_dropped", "docattn", "docattn.py",
    "return F.pad(s, (0, 1), value=doc.numel())", "return s",
    "cu_seqlens lacks the final offset: the last document has no end")
mut("docattn_flatten_wrong_layout", "docattn", "docattn.py",
    "x.transpose(1, 2).reshape(B * T, x.size(1), hd)", "x.reshape(B * T, x.size(1), hd)",
    "heads and time mixed up when rows are flattened for the kernel")
mut("docattn_unflatten_wrong_layout", "docattn", "docattn.py",
    "return o.view(B, T, H, hd).transpose(1, 2)", "return o.view(B, H, T, hd)",
    "kernel output reshaped with the wrong axis order")
mut("docattn_flag_ignored", "docattn", "model.py",
    'if doc is not None and self.doc_attn == "varlen":', "if False:",
    "doc_attn varlen silently runs the mask path")
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
# ---------------- S004 Canon layers (model.canon; killed by test_screen_canon*.py) ----------------
CANON = ["test_screen_canon.py", "test_screen_canon_run.py"]
mut("canon_centred_kernel", "canon", "blocks.py", "hp = F.pad(h, (0, 0, K - 1, 0))",
    "hp = F.pad(h, (0, 0, K // 2 - 1, K - K // 2))",
    "centred (non-causal) kernel: at K 3-5 tap 1 reads t + 1 or later", tests=CANON)
mut("canon_ignores_doc_starts", "canon", "blocks.py", "s = s.masked_fill(skip[j - 1], 0.0)", "s = s",
    "taps reach across packed document starts", tests=CANON)
mut("canon_skip_off_by_one", "canon", "model.py", "return [p < j for j in range(1, kernel)]",
    "return [p < j - 1 for j in range(1, kernel)]", "tap j reads one token of the previous document",
    tests=CANON)
mut("canon_only_with_doc", "canon", "blocks.py", "if self.canon_a is not None:",
    "if self.canon_a is not None and canon_skip is not None:",
    "Canon-A skipped on the doc=None path (the one bpb.py scores)", tests=CANON)
mut("canon_init_draws_rng", "canon", "blocks.py", "self.weight = nn.Parameter(torch.zeros(d, kernel))",
    "self.weight = nn.Parameter(nn.Conv1d(d, d, kernel, groups=d, bias=False).weight.detach().view(d, kernel))",
    "kernels built with nn.Conv1d defaults (RNG draws, non-zero)", tests=CANON)
mut("canon_built_when_off", "canon", "blocks.py", "sites = cfg.canon_sites()", 'sites = "AC"',
    "Canon kernels exist (and train) with the flag off", tests=CANON)
mut("canon_off_in_model_cfg", "canon", "config.py", 'del d["canon"], d["canon_kernel"]', "pass",
    "flag-off model_cfg differs from the pre-flag dict (old checkpoints refuse to resume)", tests=CANON)
mut("canon_in_normuon", "canon", "optim.py", 'elif ".canon_" in name:', "elif False:",
    "Canon kernels land in the NorMuon matrix group", tests=CANON)
mut("canon_budget_c_without_mlp", "canon", "budget.py",
    "n += len(cfg.canon_sites()) * d * cfg.canon_kernel", "n += len(cfg.canon) * d * cfg.canon_kernel",
    "budget counts a C kernel on attention-only blocks", tests=CANON)
mut("canon_decode_cache_short", "canon", "decode.py",
    "self.hist[key] = full[:, -(conv.weight.size(1) - 1):]",
    "self.hist[key] = full[:, -(conv.weight.size(1) - 2):]", "decode keeps K - 2 past inputs, not K - 1",
    tests=CANON)
# ---------------- S005 forget gate (model.forget_gate; killed by test_screen_forget_gate*.py) ----------------
FG, FG_RUN = ["test_screen_forget_gate.py"], ["test_screen_forget_gate_run.py"]
mut("fg_sum_includes_j", "forget", "blocks.py", "bias = c.unsqueeze(-1) - c.unsqueeze(-2)",
    "bias = c.unsqueeze(-1) - c.unsqueeze(-2) + logf.unsqueeze(-2)",
    "off by one: the sum runs over [j, i], not (j, i]", tests=FG)
mut("fg_sum_across_doc_start", "forget", "model.py", "return allowed, allowed[:, 0].to(torch.float32)",
    "return allowed, torch.ones_like(allowed[:, 0]).tril().to(torch.float32)",
    "row-wide cumulative sum: equal in exact arithmetic, but earlier documents enter the float sums", tests=FG)
mut("fg_sign_flipped", "forget", "blocks.py", "bias = c.unsqueeze(-1) - c.unsqueeze(-2)",
    "bias = c.unsqueeze(-2) - c.unsqueeze(-1)", "bias sign flipped: attention pushed toward old keys", tests=FG)
mut("fg_bias_detached", "forget", "blocks.py", 'return bias.masked_fill(~allowed, float("-inf"))',
    'return bias.masked_fill(~allowed, float("-inf")).detach()',
    "bias.detach(): a fixed recency bias, w and b never move", tests=FG)
mut("fg_only_with_doc", "forget", "blocks.py", "if self.forget_gate:                  # S005",
    "if self.forget_gate and mask is not None:  # S005",
    "no bias on the doc=None path (the one bpb.py scores)", tests=FG)
mut("fg_cumsum_bf16", "forget", "blocks.py",
    "with torch.autocast(x.device.type, enabled=False):\n            logf = self.forget_logf(x)",
    "if True:\n            logf = self.forget_logf(x)",
    "in-document sums and c_i - c_j under autocast (bf16 on the PC)", tests=FG)
mut("fg_logf_bf16", "forget", "blocks.py",
    "with torch.autocast(x.device.type, enabled=False):\n            z = F.linear(",
    "if True:\n            z = F.linear(", "forget_logf's gate logits in bf16 under autocast (b 7.99 -> 8.0)",
    tests=FG)
mut("fg_varlen_allowed_setter", "forget", "docattn.py",
    'if impl == "varlen" and getattr(getattr(model, "cfg", None), "forget_gate", False):', "if False:",
    "set_doc_attn (train.py startup) accepts varlen with the gate on", tests=FG_RUN)
mut("fg_varlen_allowed_forward", "forget", "model.py",
    'if self.cfg.forget_gate and self.doc_attn != "mask":', "if False:",
    "a model with doc_attn varlen runs flash varlen without the bias (a plain model)", tests=FG_RUN)
mut("fg_init_draws_rng", "forget", "blocks.py", "self.forget_w = nn.Parameter(torch.zeros(cfg.n_heads, d))",
    "self.forget_w = nn.Parameter(nn.Linear(d, cfg.n_heads, bias=False).weight.detach().clone())",
    "w built with nn.Linear defaults (RNG draws, non-zero)", tests=FG)
mut("fg_bias_init", "forget", "blocks.py", "FORGET_B0 = 7.99", "FORGET_B0 = 8.0",
    "b initialised at 8.0 instead of the registered 7.99", tests=FG)
mut("fg_built_when_off", "forget", "blocks.py", "if cfg.forget_gate:\n            self.forget_w",
    "if True:\n            self.forget_w", "gate parameters exist (unused) with the flag off", tests=FG)
mut("fg_budget_no_bias", "forget", "budget.py", "n += cfg.n_heads * (d + 1)", "n += cfg.n_heads * d",
    "budget leaves out the gate biases b", tests=FG_RUN)
mut("fg_decode_no_carry", "forget", "decode.py", "c = c + self.c[i][..., -1:]", "c = c + 0.0",
    "decode restarts the running sum of log f at every call", tests=FG_RUN)
mut("fg_off_in_model_cfg", "forget", "config.py", 'del d["forget_gate"]', "pass",
    "flag-off model_cfg differs from the pre-flag dict (old checkpoints refuse to resume)", tests=FG_RUN)
mut("fg_in_normuon", "forget", "optim.py", 'elif ".forget_" in name:', "elif False:",
    "gate w (2-D) lands in the NorMuon matrix group", tests=FG_RUN)
# ---------------- S006 t+2 aux head: mutants_mtp.py (split off for the file size; it appends to M) --------
import mutants_mtp  # noqa: E402,F401
# ---------------- S007 smeared keys: mutants_smear.py (same split; it appends to M) ----------------
import mutants_smear  # noqa: E402,F401
