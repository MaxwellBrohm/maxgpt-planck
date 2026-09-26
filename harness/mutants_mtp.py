"""S006 t+2 aux head mutants (train.mtp; mtp.py), split from mutants.py for the file size. Imported at the end
of mutants.py, which it appends to (mutation_check.py --group mtp). Each is killed by test_screen_mtp_aux*.py."""
from mutants import mut

# ---------------- S006 t+2 aux head (train.mtp; killed by test_screen_mtp_aux*.py) ----------------
MTP = ["test_screen_mtp_aux.py", "test_screen_mtp_aux_step.py"]
MTP_RUN = ["test_screen_mtp_aux_run.py"]
mut("mtp_targets_cross_doc", "mtp", "mtp.py",
    "ok = ok & (doc[:, :-2] == doc[:, 1:-1]) & (doc[:, 1:-1] == doc[:, 2:])", "ok = ok & (doc[:, 1:-1] == doc[:, 2:])",
    "t + 2 across a document start counted (t's document ends before t + 1)", tests=MTP)
mut("mtp_targets_unsup_counted", "mtp", "mtp.py", "ok = tgt[:, 1:-1] != -100",
    "ok = torch.ones_like(tgt[:, 1:-1], dtype=torch.bool)", "a t + 1 target of -100 (user turn, padding) counted",
    tests=MTP)
mut("mtp_norm_per_micro", "mtp", "trainer.py", 'obj = ls + aux["w"] * self._mtp_sum(out[2], b, aux)',
    'obj = ls + aux["w"] * self._mtp_sum(out[2], b, aux) * n_sup / max(1, int((b["tgt"] != -100).sum()))',
    "aux losses normalised per micro-batch, not by the step's n_sup", tests=MTP)
mut("mtp_weight_constant", "mtp", "trainer.py", '{"w": self.mtp_weight * self.sched.factor(self.step),',
    '{"w": self.mtp_weight,', "w_t constant: the aux weight does not decay with the LR factor",
    tests=MTP + MTP_RUN)
mut("mtp_hidden_before_norm", "mtp", "model.py", "return logits, loss, z", "return logits, loss, x",
    "the aux head reads the hidden state before the final norm", tests=MTP)
mut("mtp_z_detached", "mtp", "trainer.py", "self._mtp_sum(out[2], b, aux)", "self._mtp_sum(out[2].detach(), b, aux)",
    "the aux loss never reaches the shared trunk", tests=MTP)
mut("mtp_init_linear", "mtp", "mtp.py", "self.weight = nn.Parameter(torch.eye(d_model))",
    "self.weight = nn.Parameter(nn.Linear(d_model, d_model, bias=False).weight.detach().clone())",
    "W_mtp built as an nn.Linear (draws from the RNG, not the identity)", tests=MTP)
mut("mtp_head_in_model", "mtp", "train.py", "opt = make_optimizer(model, oc, device, extra=head)",
    "model.mtp_head = head\n    opt = make_optimizer(model, oc, device, extra=head)",
    "the head registered in the model: its weights in the model state bpb.py loads", tests=MTP_RUN)
mut("mtp_not_optimized", "mtp", "optim.py", 'named = [*named, *extra.named_parameters(prefix="mtp")]',
    "named = [*named]", "the head is in no optimizer group (never trained)", tests=MTP + MTP_RUN)
mut("mtp_matrix_to_scalar", "mtp", "optim.py", 'extra.named_parameters(prefix="mtp")',
    'extra.named_parameters(prefix="mtp.canon_")', "W_mtp routed to the AdamW scalar group, not NorMuon",
    tests=MTP)
mut("mtp_clip_excludes_head", "mtp", "trainer.py", "params = self.model.parameters() if self.mtp is None else",
    "params = self.model.parameters() if True else", "the grad norm and its clip leave the head out", tests=MTP)
mut("mtp_ckpt_without_head", "mtp", "trainer.py", 'p["mtp"] = self.mtp.state_dict()', "pass",
    "checkpoints drop the head's state (resume cannot restore it)", tests=MTP_RUN)
mut("mtp_on_by_default", "mtp", "mtp.py", 'mtp = tc.get("mtp", 0)', 'mtp = tc.get("mtp", 1)',
    "the head trains when train.mtp is absent", tests=MTP_RUN)
mut("mtp_budget_no_gain", "mtp", "budget.py", "return mtp * (cfg.d_model * cfg.d_model + cfg.d_model)",
    "return mtp * cfg.d_model * cfg.d_model", "count_mtp leaves out the aux norm gain", tests=MTP + MTP_RUN)
mut("mtp_nan_eager_unchecked", "mtp", "trainer.py",
    'if not (math.isfinite(last_loss) and math.isfinite(out.get("mtp_loss", 0.0))):',
    "if not math.isfinite(last_loss):", "eager: a NaN aux loss is not caught in its step", tests=MTP)
mut("mtp_nan_lazy_unchecked", "mtp", "trainer.py", 'ok = ok & torch.isfinite(aux["parts"]).all()', "ok = ok",
    "lazy_metrics: a NaN aux loss is not caught (the NaN state is saved)", tests=MTP)
