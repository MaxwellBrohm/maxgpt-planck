"""SPEED V3 mutants (train.compile, train.ce_chunk_rows), split from mutants.py for the file size. Imported
at the end of mutants.py, which it appends to (mutation_check.py --group speed3). Each names the test that
must go red; node ids keep every run short."""
from mutants import mut

DEF = ["test_speed3.py::test_default_run_is_the_reference_and_builds_nothing_new"]
RESUME_C = ["test_compile_train.py::test_resume_with_compile_20_plus_20_equals_40[pack]"]
CE = ["test_chunked_ce.py"]
C = "test_compile.py::"
# ---------------- the switches stay off by default ----------------
mut("s3_compile_on_by_default", "speed3", "train.py", 'cmode = compile_mode(tc.get("compile", False))',
    'cmode = compile_mode(tc.get("compile", True))', "train.compile on when the key is absent", tests=DEF)
mut("s3_chunk_on_by_default", "speed3", "train.py", 'ce_chunk = ce_chunk_rows(tc.get("ce_chunk_rows", 0))',
    'ce_chunk = ce_chunk_rows(tc.get("ce_chunk_rows", 64))', "train.ce_chunk_rows on when the key is absent",
    tests=DEF)
mut("s3_model_chunk_default", "speed3", "model.py", "return_hidden: bool = False, ce_chunk: int = 0):",
    "return_hidden: bool = False, ce_chunk: int = 7):", "model.forward chunks the loss unless told not to",
    tests=DEF)
# ---------------- chunked CE keeps the reference's ignored-target handling ----------------
mut("s3_ce_ignored_in_loss", "speed3", "chunked_ce.py", "total = torch.where(valid, -tlp, torch.zeros_like(tlp)).sum()",
    "total = (-tlp).sum()", "-100 targets add their (token 0) loss", tests=CE)
mut("s3_ce_ignored_in_grad", "speed3", "chunked_ce.py",
    "row_w = valid.to(f32) / denom if mean else valid.to(f32)",
    "row_w = torch.ones_like(tgt, dtype=f32) / denom if mean else torch.ones_like(tgt, dtype=f32)",
    "-100 targets get a gradient (toward token 0)", tests=CE)
mut("s3_ce_mean_over_all_rows", "speed3", "chunked_ce.py", "denom = valid.sum().clamp(min=1).to(f32)",
    "denom = torch.tensor(float(valid.numel()), device=h.device)", "mean divides by every row, not the supervised",
    tests=CE)
mut("s3_trainer_drops_chunk", "speed3", "trainer.py", 'self._hid["ce_chunk"] = int(ce_chunk)', "pass",
    "Trainer ignores ce_chunk (the reference loss runs)",
    tests=["test_chunked_ce.py::test_trainer_accum_unequal_supervised_counts_matches_reference"])
mut("s3_chunk_not_recorded", "speed3", "train.py", 'start["ce_chunk_rows"] = ce_chunk', "pass",
    "the start record hides ce_chunk_rows", tests=["test_speed3.py::test_resume_switch_on_then_off"])
mut("s3_chunk_drops_hidden", "speed3", "model.py", "return (None, loss, z) if return_hidden else (None, loss)",
    "return (None, loss)", "the chunked path loses z (the S006 aux head)",
    tests=["test_speed3.py::test_trainer_mtp_with_switches"])
# ---------------- compile never owns the optimizer or the checkpoint ----------------
mut("s3_state_dict_of_wrapper", "speed3", "trainer.py", 'p = {"model": self.model.state_dict(),',
    'p = {"model": self.fwd.state_dict(),', "checkpoints saved from the compiled wrapper (_orig_mod. keys)",
    tests=RESUME_C)
mut("s3_optimizer_on_wrapper", "speed3", "train.py", "opt = make_optimizer(model, oc, device, extra=head)",
    "opt = make_optimizer(fwd, oc, device, extra=head)",
    "optimizer built on the wrapper: _orig_mod.tok_emb.weight lands in NorMuon", tests=RESUME_C)
mut("s3_trainer_accepts_wrapper", "speed3", "trainer.py", 'if hasattr(model, "_orig_mod"):', "if False:",
    "Trainer takes a compiled wrapper as its model", tests=[C + "test_trainer_refuses_a_compiled_wrapper_as_model"])
mut("s3_trainer_ignores_forward", "speed3", "trainer.py", "self.fwd = model if forward is None else forward",
    "self.fwd = model", "train_step runs the plain module, compile is a no-op",
    tests=[C + "test_grad_accum_unequal_supervised_counts_compiled"])
# ---------------- the compiled loss ----------------
mut("s3_guard_kept_compiled", "speed3", "model.py",
    'if (reduction == "sum" and torch.compiler.is_compiling()) or (tgt != -100).any():',
    "if (tgt != -100).any():", "the all-ignored guard stays in the compiled graph (a break, a sync)",
    tests=[C + "test_all_ignored_batch_compiled_in_one_graph"])
mut("s3_guard_dropped_for_mean", "speed3", "model.py",
    'if (reduction == "sum" and torch.compiler.is_compiling()) or (tgt != -100).any():',
    "if torch.compiler.is_compiling() or (tgt != -100).any():", "compiled mean over zero targets: 0/0 = nan",
    tests=[C + "test_all_ignored_batch_compiled_in_one_graph"])
mut("s3_cuda_graph_modes_allowed", "speed3", "model.py", "if value not in COMPILE_MODES:", "if False:",
    "reduce-overhead / max-autotune (CUDA graphs, untested) accepted", tests=[C + "test_compile_mode_values"])
# ---------------- follow-ups (notes.txt SPEED V3 FOLLOW-UPS) ----------------
CT = "test_compile_train.py::"
mut("s3_dynamic_false_varlen_accepted", "speed3", "train.py",
    'if cmode is not None and dynamic is False and doc_attn == "varlen":', "if False:",
    "compile_dynamic false with varlen accepted (recompile limit, silent eager fallback)",
    tests=[CT + "test_compile_dynamic_false_refused_with_varlen"])
mut("s3_dynamic_false_refused_everywhere", "speed3", "train.py",
    'if cmode is not None and dynamic is False and doc_attn == "varlen":',
    "if cmode is not None and dynamic is False:", "compile_dynamic false refused with the mask engine too",
    tests=[CT + "test_compile_dynamic_is_recorded"])
mut("s3_dynamic_not_recorded", "speed3", "train.py", 'start["compile_dynamic"] = cdyn', "pass",
    "the start record hides compile_dynamic", tests=[CT + "test_compile_dynamic_is_recorded"])
mut("s3_train_accepts_autotune", "speed3", "train.py", "if cmode is not None and cmode not in TRAIN_COMPILE_MODES:",
    "if False:", "train.py runs max-autotune-no-cudagraphs (failed a 20M parity check, a bench arm only)",
    tests=[CT + "test_train_refuses_the_unvalidated_compile_mode"])
mut("s3_train_refuses_default", "speed3", "train.py", 'TRAIN_COMPILE_MODES = ("default",)',
    "TRAIN_COMPILE_MODES = ()", "train.py refuses the validated mode too",
    tests=[CT + "test_train_refuses_the_unvalidated_compile_mode"])
