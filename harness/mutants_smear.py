"""S007 smeared-key mutants (model.smear_key), split from mutants.py for the file size. Imported at the end of
mutants.py, which it appends to (mutation_check.py --group smear). Each is killed by test_screen_smear_key*.py."""
from mutants import mut

SM, SM_RUN = ["test_screen_smear_key.py"], ["test_screen_smear_key_run.py"]
mut("smear_future_key", "smear", "blocks.py", "prev = F.pad(k[:, :-1], (0, 0, 0, 0, 1, 0))",
    "prev = F.pad(k[:, 1:], (0, 0, 0, 0, 0, 1))", "k_(t+1) smeared in: a future leak", tests=SM)
mut("smear_crosses_doc_start", "smear", "model.py", "smear = docattn.doc_starts(doc)", "smear = None",
    "packed rows smear over the whole row: a document's first key reads the previous document", tests=SM)
mut("smear_start_off_by_one", "smear", "blocks.py", "prev = prev.masked_fill(starts[:, :, None, None], 0.0)",
    "prev = prev.masked_fill(starts.roll(1, 1)[:, :, None, None], 0.0)",
    "the zeroing lands one token late: a document's first key reads the previous document's last", tests=SM)
mut("smear_only_with_doc", "smear", "blocks.py", "if self.smear_key:                    # S007",
    "if self.smear_key and smear is not None:  # S007",
    "no smear on the doc=None path (the one bpb.py scores)", tests=SM)
mut("smear_after_qk_norm", "smear", "blocks.py",
    "            k = self.smear(k_raw, smear)\n        if self.qk_norm:\n            q, k = self.q_norm(q), self.k_norm(k)\n",
    "            k = k_raw\n        if self.qk_norm:\n            q, k = self.q_norm(q), self.k_norm(k)\n"
    "        if self.smear_key:\n            k = self.smear(k, smear)\n",
    "smear after QK-norm (a different arm, notes KNOWN RISKS), not on the raw key", tests=SM)
mut("smear_values_too", "smear", "blocks.py", "            k = self.smear(k_raw, smear)\n",
    "            k = self.smear(k_raw, smear)\n            v = k if self.kv_tie else v\n",
    "with kv_tie, V takes the smeared key (the notes: V keeps the unsmeared key)", tests=SM)
mut("smear_init_draws_rng", "smear", "blocks.py", "self.smear_alpha = nn.Parameter(torch.zeros(cfg.n_kv_heads))",
    "self.smear_alpha = nn.Parameter(torch.randn(cfg.n_kv_heads) * 0.0)",
    "alpha still 0 but drawn from the RNG: every later draw shifts (no longer SIA)", tests=SM)
mut("smear_alpha_init_nonzero", "smear", "blocks.py", "self.smear_alpha = nn.Parameter(torch.zeros(cfg.n_kv_heads))",
    "self.smear_alpha = nn.Parameter(torch.full((cfg.n_kv_heads,), 0.1))",
    "alpha starts at 0.1: the arm does not start as the exact BASE function", tests=SM)
mut("smear_built_when_off", "smear", "blocks.py", "        if cfg.smear_key:\n            self.smear_alpha",
    "        if True:\n            self.smear_alpha", "alpha parameters exist (unused) with the flag off", tests=SM)
mut("smear_alpha_2d_in_normuon", "smear", "blocks.py",
    "self.smear_alpha = nn.Parameter(torch.zeros(cfg.n_kv_heads))",
    "self.smear_alpha = nn.Parameter(torch.zeros(cfg.n_kv_heads, 1))",
    "alpha stored as an (Hkv, 1) column: 2-D, so split_params sends it to NorMuon", tests=SM_RUN)
mut("smear_off_in_model_cfg", "smear", "config.py", 'del d["smear_key"]', "pass",
    "flag-off model_cfg differs from the pre-flag dict (old checkpoints refuse to resume)", tests=SM_RUN)
mut("smear_budget_per_query_head", "smear", "budget.py", "n += cfg.n_kv_heads                 # S007",
    "n += cfg.n_heads                 # S007", "budget counts one alpha per query head, not per KV head", tests=SM)
mut("smear_decode_caches_smeared", "smear", "decode.py",
    "self.kraw[i] = full[:, -1:]\n        return att.smear(full)[:, -k_raw.size(1):]",
    "sm = att.smear(full)\n        self.kraw[i] = sm[:, -1:]\n        return sm[:, -k_raw.size(1):]",
    "decode caches the smeared instead of the raw key", tests=SM)
