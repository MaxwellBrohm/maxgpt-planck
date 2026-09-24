# Fact-check: loop.md (the Planck experiment loop on the M5 Mac)

Checked 2026-09-23. Method: primary sources only. arXiv PDFs converted with pdftotext (2511.16893 v2 and v3, 2504.11393,
2605.20798, 2207.10551, 2508.13144 plus its HTML, 2407.13623, 2509.02046, 2407.01492, 2512.24503, 2403.17844, 2507.00885,
2503.09543, 2405.18392, 2410.11840, 2501.14925, 2510.18245, 2404.05405, 2505.22757); GitHub issues, PRs, comments and
cross-references through the `gh` API (pytorch/pytorch #195910, #179294, #178545, #191794, #188147; modded-nanogpt #358);
PyTorch source at tags v2.7.1, v2.11.0, v2.12.1, v2.13.0 and v2.14.0 (`aten/src/ATen/native/transformers/attention.cpp`,
`native_functions.yaml`, `derivatives.yaml`); MLX source at v0.32.2 (`mlx/fast.cpp`, `mlx/backend/metal/scaled_dot_product_attention.cpp`,
`mlx/backend/cuda/scaled_dot_product_attention.cpp`); the PyTorch 2.14 release blog and v2.14.0 release notes; the PyTorch 2.14 MPS
environment-variable docs; Hugging Face API file listings for the Gemma 4 teacher. One local, model-free measurement: the Metal API's
`recommendedMaxWorkingSetSize` on this Mac (a Swift one-liner, no ML model loaded, no training). All arithmetic re-run in Python.

Limits of this check: WebSearch was exhausted for the session and the arXiv and Semantic Scholar search APIs returned HTTP 429, so
discovery used the Hugging Face papers search API and direct PDF fetches. The omissions list may be incomplete for that reason.

## Summary

The report is careful and its arithmetic reproduces almost everywhere, but five things change the design:

1. **The induction-head law constant is wrong.** The report uses T = 750,000. The paper's own fit is T = e^13.26 = **573,800**
   (v2 printed "e^α = 750000", an arithmetic slip; v3 of 2026-07-05 drops it). Predicted transitions are ~24% lower:
   **~38M / 60M / 92M / 221M tokens** at 8 / 16 / 32 / 128 x 2048, not 50 / 78 / 121 / 289M. The 0.25B-token budget at 5M is then
   2.7-4x past the transition, so the budget still stands.
2. **MLX does not fuse attention for training on Apple GPUs.** In MLX v0.32.2 the Metal backend deliberately falls back to unfused SDPA
   for both forward and backward when tracing gradients ("It's faster for training on Metal to use the unfused SDPA for both forward and
   backward"), and the fused VJP on Metal is `NYI`. The report's hope that MLX's fused attention wins at T=2048 (bottom line 2, section
   2.5, ledger 13) is refuted by the source. MLX's edge can only be per-op overhead.
3. **In PyTorch on MPS, training does not use the buggy kernel at all.** At every tag checked (2.7.1 to 2.14.0) the MPS-specific SDPA
   path runs only when grad is off or no input requires grad. A training step goes through the generic math decomposition (forward and
   backward both unfused, which confirms and strengthens claim 2). So the ≤2.12.1 causal leak corrupts **no-grad passes: eval
   bits-per-byte, likelihood probes and generation**, not training. The harness's leak self-test must run under `torch.no_grad()`; the
   report's training-loss leak test would never see it.
4. **The 0.0015-nat speedrun seed SD is init-only noise on a ~640M-stored-parameter model**, from a PR that was closed without merging.
   The speedrun reads its data shards in sorted order with no shuffle, so data order is fixed across those 20 runs. It is a lower bound on
   Planck's full seed noise, not an estimate of it.
5. **Ledger idea 9 (MLP-light blocks) is not untested.** Allen-Zhu & Li (Physics 3.3, section 7) already shrank the MLP to 1/4 and removed
   it entirely at sizes down to under 10M. Bian et al. (ICLR 2026) and Liao et al. (2026) varied the MLP-to-attention split at matched
   budgets at 80M-1B. Physics 3.3 also undercuts the idea's rationale: attention layers store knowledge at the same 2 bits/param, so a
   smaller MLP does not by itself stop the model spending parameters on facts.

## Load-bearing claims

| # | claim (short) | verdict | notes and correct values | source |
|---|---|---|---|---|
| 1 | MPS SDPA `is_causal=True` leaks up to 3 future keys (block of 4 queries) in fp16/bf16 in ≤2.12.1, fixed in 2.13.0, fp32 and explicit masks unaffected | confirmed | Issue filed 2026-09-03 on 2.11.0 (M5 Max), closed 2026-09-04. The reporter then tested four clean venvs: 2.11.0 and 2.12.1 AFFECTED (bf16 err 4.14), 2.13.0 and 2.14.0 not affected (0.011 / 0.0085). At seq len 64: max abs err 2.87 vs 0.0116 (bf16) / 0.0012 (fp16) with an explicit mask; fp32 unaffected. Only 2.11.0 and 2.12.1 were tested, so "≤2.12.1" back to older versions is an inference. **Scope note (see extra check A):** the leaking path is gated off whenever grad is on and inputs require grad, so training steps are unaffected and no-grad evaluation is where the leak lands. | https://github.com/pytorch/pytorch/issues/195910 (body and comments) ; https://github.com/multimodal-art-projection/YuE/issues/176 |
| 2 | MPS SDPA backward runs through the device-agnostic math path; a Metal SDPA kernel exists but is not called; issue open since 2026-04-03; T x T is materialized in training | confirmed | Issue #179294 is still open (last update 2026-08-23), quotes match. The 2.14.0 source makes it stronger: with grad on, MPS calls the generic `_scaled_dot_product_attention_math` (no MPS dispatch, no derivative entry for the MPS op), so the **forward is unfused in training too**. The dead `sdpa_full_attention_mps` kernel was left off on purpose because it was slower than the MPSGraph path (maintainer comment 2026-04-06), so it is not a fast kernel waiting to be switched on. PyTorch 2.14's new MPP prefill kernel and the MPS FlexAttention additions are forward or inference only; the 2.14 blog lists MPS-native flex backward as future work. | https://github.com/pytorch/pytorch/issues/179294 ; https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/transformers/attention.cpp (lines 862-888) ; https://pytorch.org/blog/pytorch-2-14-release-blog/ |
| 3 | Attention (12LdT) is ~55% of training FLOPs at 5M (d192 L8, 8k vocab), ~48% at 20M (d320 L14), ~38% at 60M (d512 L18), T=2048 | confirmed | Recomputed: 6N = 30.7M / 119M / 365M FLOPs per token; 12LdT = 37.7M / 110M / 226M; shares 55.1% / 48.0% / 38.3%; totals 0.068 / 0.229 / 0.592 GFLOP. The convention is PaLM's 6N + 12LHQT, which counts full, not causal, attention. That is right for MPS because the math path really computes the full T x T. | arithmetic ; PaLM, https://arxiv.org/abs/2204.02311 (App. B) |
| 4 | Mac throughput at T=2048: 5M ~10-19k tok/s, 20M ~4.8-7.9k, 60M ~2.5-3.6k; wall-clocks 3.7-6.8 h / 14-23 h / 4-5.5 days | unverifiable | The arithmetic reproduces exactly from the assumed TFLOPS bands (0.7-1.3, 1.1-1.8, 1.5-2.1), and p18M's 1.47 effective TFLOPS reproduces from 10,637 tok/s. The anchor is one bench that the first run's verifier could not reproduce. Two new reasons for caution: (a) MLX is also unfused in training (extra check B), so neither engine avoids the T x T attention cost that dominates at 5M; (b) the kernel-launch estimate (~0.1 s/step, "<5%") counts one forward/backward per optimizer step. With the 4-8-sequence micro-batch cap the report itself derives, a 16-32-sequence step needs 2-8 passes, so launch overhead could be ~10-40% of step time (estimate, launch times are M2-era). Keep the ±2x and let E1 replace it. | lanes/compute.md s.3 ; lanes/compute.verify.md claim 9 ; https://arxiv.org/abs/2501.14925 (Table 6) |
| 5 | `PYTORCH_MPS_HIGH_WATERMARK_RATIO` default 1.7 (hard limit, multiple of recommended working set), LOW default 1.4 on unified memory; default can allocate past RAM; 0.7 turns thrash into OOM | confirmed | Docs quote: HIGH "Default is 1.7", a hard limit, ">1.0: allows limits beyond the device.recommendedMaxWorkingSetSize"; LOW "Default is 1.4 (unified) or 1.0 (discrete)". Measured on this Mac through the Metal API (no model loaded): recommendedMaxWorkingSetSize = **19.07 GB**. So the default hard cap is 1.7 x 19.07 = 32.4 GB (past the 24 GB of RAM), and 0.7 gives 13.3 GB, matching the report's 12-14 GB target. The cap is per process: it cannot stop swap caused by another process. | https://docs.pytorch.org/docs/2.14/mps_environment_variables.html ; local Metal query |
| 6 | Induction transition N_PT ≈ 750,000 · B^0.63 · C^0.38, model-size agnostic over 50M-7B, r = 0.98; ~78M at 16x2048, ~121M at 32x2048 | corrected | The exponents, "50M-7B", "500-2M tokens per update" and 35 models are right. **Constant:** α = 13.26 and T = e^α = 573,779. v2 (Feb 2026) says "Calling e^α = 750000", which contradicts its own α; v3 (2026-07-05) writes only T = e^13.26 and reports r = .986. **Corrected predictions:** 8x2048 ~38M, 16x2048 ~60M, 32x2048 ~92M, 128x2048 ~221M tokens. Scale: the trained models were GPT-2 at 50M (2 layers, d768), 125M and 350M; 7B was a Pythia checkpoint used for inference only. | https://arxiv.org/abs/2511.16893 (v2 s.6.1, v3 s.6.1, eq. 7) |
| 7 | Next-token models at ≤30M non-embedding form induction much worse than MTP-trained ones on a name-copying task; the gap disappears at ≥100M | confirmed | Verbatim in the paper ("vastly improved formation of induction capability for models of size 30M nonembedding parameters and below"). Caveats from arch.verify: children's stories with random names, up to 90 epochs, early stopping on the test metric; with a 9:1 books mix the advantage vanishes except for the smallest models. Planck's 5M (3.5M body) and 20M (17.2M body) are inside that range. | https://arxiv.org/abs/2404.19737 ; lanes/arch.verify.md claim 9 |
| 8 | DataDecide: 14 sizes 4M-1B at 100 tokens/param, 3 seeds; 150M ranking picks the better 1B recipe ~80% of the time; seed SD up to 2 points at 1B | corrected | "~80%" and "as high as 2% points ... for some recipes on most tasks" (at 1B 5xC) are verbatim. Two corrections. (a) **Sizes are non-embedding** (Table 2: "Model size is number of non-embedding parameters"). The "4M" model is 3.7M non-embedding with d_model 64 and 8 layers, and under Planck's total-parameter rule it is several million larger. (b) **Only the 1B models have 3 full seeds.** Smaller sizes' second and third seeds stop at 25% of the target compute. The tokens are right: 0.4B (4M), 5.7B (60M), 15.0B (150M), 100B (1B). | https://arxiv.org/abs/2504.11393 (s.2, Table 2) |
| 9 | Architecture rankings transfer poorly: Tay et al. best architecture fluctuates across scales; a 2026 replication at 1.2B/3B found most modifications do not transfer, and two within 2-3% of baseline loss dropped 6-16 downstream points | corrected | The Tay quote is verbatim ("the best performing model can fluctuate at different scales", T5 family from 15M up). For 2605.20798 (Tencent, 2026-05-20), "do not transfer" means **published gains fail to reproduce under one controlled 1.2B recipe** (Narang-style: 2 of 20 clear Bonferroni). It does not mean rankings flip with scale. In its own 1.2B to 3B check, all 10 completing runs **kept the sign** of their 1.2B effect, but order within the improver band reshuffled (Spearman -0.27) and 1 of the 2 Bonferroni survivors diverged at 3B. The 2-3% / 6-16 points are Sigmoid Attention (+2.4% loss, -16 CLIMB points) and SSMax (+3.0%, -6). Tokens: 23.28B at 1.2B, 60.04B at 3B. | https://arxiv.org/abs/2207.10551 ; https://arxiv.org/abs/2605.20798 (abstract, s.1, Appendix A) |
| 10 | Seed-to-seed SD of final val loss for a 124M speedrun record is 0.00149 nats over 20 runs | corrected | The PR's numbers match: n = 20, mean 3.2788600, std 0.0014869, p = 0.0014 for mean < 3.28. But: (a) PR #358 was **closed without merging**, so it is a record candidate, not a record; (b) the runs vary **only initialization and GPU nondeterminism**: `train_gpt.py` reads shards in `sorted(glob(...))` order with no shuffle, `TRAIN_SEED` is unset, and the PR states the token streams are unchanged; (c) the current speedrun stores ~640M parameters (value embeddings and a bigram table), not 124M (lanes/arch.verify.md claim 12); ~0.34B tokens on 8xH100. Use it as a floor for init noise under a fixed data order, not as Planck's seed SD. | https://github.com/KellerJordan/modded-nanogpt/pull/358 ; https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py (data generator, lines 88-89, 2180) |
| 11 | Bits-per-byte raised MBPP SNR 2.0 to 41.8 and decision accuracy 68.3% to 95.3%; checkpoint averaging added +2.4% decision accuracy | corrected | The numbers are verbatim (MBPP row: SNR 2.0 to 41.8, decision accuracy 68.3 to 95.3; "+2.4% for the 30-task average"). But the paper's "averaging" means **averaging evaluation results over the final checkpoints** (and an EMA of the metric), not averaging weights. It supports reporting metrics averaged over the last checkpoints. It does not support ledger 15's weight averaging or weight EMA, which needs its own source (Hägele et al. for SWA, or 2505.12082 below). Decision accuracy is DataDecide 150M to 1B; the suite used spans 4M-1.3B. | https://arxiv.org/abs/2508.13144 (s.5.2, s.5.3, Fig. 2) |
| 12 | With 1,450 paired conversations (20% discordant), 3 seeds, 1-point seed SD, MDE ≈ 3 points; 1-2 point differences unresolvable; a 1% loss difference needs 1-3 seeds if seed SD ≤ 0.005-0.01 nats | confirmed | Both tables reproduce exactly (2.8·σ·sqrt(2/k); item variance 0.2/n per run). Two modelling caveats. (a) The table lets item noise shrink with seeds. If the same items are discordant for every seed (shared items), the n = 1,450, σ_s = 1, k = 3 MDE is **4.0 points, not 3.0**, and k = 5 gives 3.7, not 2.3. (b) The loss MDE is unpaired, so it is conservative once common random numbers are used. The conclusion (decide on continuous metrics) holds either way. | arithmetic |
| 13 | Tao et al. law (Nv ∝ Nnv^0.83, 16K optimal at 302M non-vocab params, d = 1024) extrapolates to ~2k at 5M, ~5k at 20M, ~8k at 60M | confirmed | Anchors are verbatim: γ = 0.83; Nv = V·d; 16K at Nnv = 302M (Table 4: 18 layers, d = 1024). Recomputed with Planck's widths: 2,181 / 4,860 / 8,164. Two caveats belong next to the number. (a) The law is **compute-optimal**; the same paper shows the optimum rising with more data (302M: 16K to 24K), and Planck's final models will be trained far past compute-optimal, so these are lower bounds for final models. (b) The result depends on the width used: Tao's own width rule (d = 512 for Nnv ≤ 50M, Table 5) gives ~820 at 5M. Both 5M and 20M are 2-10x below the smallest fitted Nnv (33M). | https://arxiv.org/abs/2407.13623 (s.4.2, s.5, Tables 4, 5) |
| 14 | The Mac is shared: the verifier measured ~half throughput with a concurrent MPS job; a ~15 GB 4-bit teacher cannot overlap training in 24 GB | corrected | The ratios are 0.50 / 0.54 / 0.59 (5,343 / 2,096 / 1,598 vs 10,637 / 3,878 / 2,703). But the verifier wrote that this "neither refutes nor confirms" the lane's numbers: there was no uncontended re-run, so "contention halves throughput" is a plausible inference, not an isolated measurement. The teacher part is confirmed. `mlx-community/gemma-4-26b-a4b-it-4bit` has 15.34 GB of weight files, and the base model is 25.8B parameters (BF16). Next to a training job capped at ~13 GB that exceeds both 24 GB of RAM and the 19.07 GB Metal working set. | lanes/compute.verify.md claim 9 ; https://huggingface.co/api/models/mlx-community/gemma-4-26b-a4b-it-4bit/tree/main ; https://huggingface.co/api/models/google/gemma-4-26B-A4B-it ; local Metal query |

## Extra checks (numbers the design depends on)

| # | claim (short) | verdict | notes | source |
|---|---|---|---|---|
| A | The harness's causal-leak self-test (a CPU fp32 attention comparison plus a training-loss leak test) catches the ≤2.12.1 bug | corrected | In v2.11.0 (L794-796), v2.12.1 (L796), v2.13.0 (L796) and v2.14.0 (L864), `_scaled_dot_product_attention_math_for_mps` runs only when `!(GradMode::is_enabled() && any_inputs_require_grad)`. Otherwise the generic `_scaled_dot_product_attention_math` runs, with a correct causal mask. So a training-loss leak test is blind to this bug. The attention-output comparison must run under `torch.no_grad()` / `inference_mode()`, which is where eval and generation run. (Inference from source code; not executed, per this sweep's no-model rule.) | https://github.com/pytorch/pytorch/blob/v2.12.1/aten/src/ATen/native/transformers/attention.cpp ; v2.14.0 same file |
| B | MLX's `mx.fast.scaled_dot_product_attention` may be fused in the backward pass, so MLX could beat PyTorch MPS by more than 1.2x at T=2048 (s.2.5, ledger 13) | refuted | MLX v0.32.2 (2026-08-25), Metal backend: `ScaledDotProductAttention::use_fallback` returns true when `is_training` ("It's faster for training on Metal to use the unfused SDPA for both forward and backward"), and `ScaledDotProductAttentionVJP::use_fallback` returns true on Metal, where `eval_gpu` throws "NYI". A fused VJP exists only in the CUDA backend (query length a multiple of 128). MLX training attention materializes T x T just like PyTorch MPS. | https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/scaled_dot_product_attention.cpp (lines 715-746, 918-926) ; https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/fast.cpp (lines 927-1015) |
| C | "Pin PyTorch >= 2.13" is the right pin | corrected | 2.14.0 shipped 2026-09-02. It adds an MPP prefill-attention kernel (fp16/bf16, head dims 64/96/128/256, query length > 8, macOS 26.2+; used only on the no-grad path, and Planck's head dim 64 qualifies). The same release fixes "corrupted MPS prefill-attention output on macOS 26" (#191794: an ABI mismatch that "scrambled" P@V on macOS 26 machines; it hit nightlies before release) and a NaN-propagation bug in MPS attention kernels (#188147). This Mac runs macOS 26.6. Safer rule: pin one exact release (2.14.0 now), never a nightly, and re-run the no-grad attention self-test after any PyTorch or macOS update. | https://github.com/pytorch/pytorch/releases/tag/v2.14.0 ; https://github.com/pytorch/pytorch/pull/191794 ; https://github.com/pytorch/pytorch/pull/188147 ; https://pytorch.org/blog/pytorch-2-14-release-blog/ |
| D | PyTorch 2.13 released July 8 2026 | confirmed | GitHub release v2.13.0 published 2026-07-08. | https://github.com/pytorch/pytorch/releases |
| E | RegMix: 512 models of 1M params on 1B tokens each picked a mixture that beat 64 candidate 1B models at 25B tokens | confirmed | Verbatim in the abstract. | https://arxiv.org/abs/2407.01492 |
| F | Tiny-LR proxies (1e-6) reach Spearman > 0.95 vs < 0.75 at standard LR; 70M-125M proxies, 23 recipes, ICLR 2026 | confirmed | "improved to > 0.95 across 23 data recipe pairs" (GPT2-125M vs Pythia-1B); standard LR "ρ < 0.75"; proxies GPT2-125M, OPT-125M, Pythia-70M; targets up to 1B. | https://arxiv.org/abs/2512.24503 |
| G | Wen et al.: optimizer rankings can flip during LR decay; matrix-optimizer gains 1.4x at 0.1B to 1.1x at 1.2B; Muon overtaken at 8x Chinchilla | confirmed | All three verbatim (Muon "overtaken by Soap" at 8x). 0.1B-1.2B, 1-8x Chinchilla. | https://arxiv.org/abs/2509.02046 |
| H | Hägele et al. quotes on constant LR plus cooldown | confirmed | Both verbatim (abstract; Takeaway 6). | https://arxiv.org/abs/2405.18392 |
| I | MAD: 2-block width-128 models, minutes of training, rank-correlated with compute-optimal perplexity over 500+ models of 70M-7B, noisier across classes | confirmed | "2 blocks with a total of 4 layers", "width of 128"; cross-class correlation "subject to more noise". | https://arxiv.org/abs/2403.17844 |
| J | Downstream scaling predictable in only 39% of cases | confirmed | Verbatim. | https://arxiv.org/abs/2507.00885 |
| K | PolyPythias: 14M-410M, 10 seeds per size, stable; the only outliers are two 410M runs | confirmed | "410M seed 3 and 4"; 9 new seeds plus the original per size; 300B tokens. | https://arxiv.org/abs/2503.09543 |
| L | Apple: MLX on M5 needs macOS 26.2+; TTFT 3.3-4x vs M4; generation 1.19-1.27x; no training numbers | confirmed | Matches the page. | https://machinelearning.apple.com/research/exploring-llms-mlx-m5 |
| M | Choshen et al.: "training multiple small models is sometimes more useful than training a single large one", 485 models | confirmed | Verbatim. | https://arxiv.org/abs/2410.11840 |
| N | ufakzeka-1 is 151M | corrected | 151M non-embedding, 182M with embeddings (Planck counts totals). | lanes/data.verify.md ; https://arxiv.org/abs/2609.25081 |

## Scale misapplications

1. **Induction-transition law to 1-20M models.** Trained fits cover GPT-2 50M (2 layers, d768, 50k vocab) to 350M; 7B is inference on Pythia
   checkpoints. Planck's 5M has d192. The report flags this, but then sets budgets and the E4 ">3x miss" rule from it. With the corrected
   constant, the budgets are safer but still untested below 50M.
2. **Speedrun seed SD (0.0015 nats) as the prior for Planck's 5-20M seed noise.** It was measured on a ~640M-stored-parameter, heavily
   tuned model, with a fixed data order and 8xH100 nondeterminism only. Full seed noise (init plus data order) at 5M is likely larger.
   The "1-2 seed regime" is not supported until E3.
3. **Vocab law to Nnv of 3.5-17M and to overtrained final models.** The fit starts at 33M and is compute-optimal only; Tao et al. show the
   optimum rising with data. "Vocabulary is probably the biggest single lever at 5-20M" rests on this extrapolation.
4. **DataDecide sizes read as totals.** DataDecide's "4M" and "150M" are non-embedding. Under Planck's total-parameter convention those
   proxies are larger than the names suggest, so "~80% from 150M" is evidence from a bigger proxy than it sounds.
5. **Architecture-transfer evidence from 1.2B/3B (2605.20798) and T5 (Tay) applied to 5M to 20M transfer.** Also, the 2026 paper's own
   cross-scale result (sign preserved 1.2B to 3B) is more encouraging than the report's framing.
6. **Signal-and-Noise SNR gains (60M-1.3B, standard benchmarks)** used to justify the Tier 0 log-prob margin at 5-20M on dialogue probes.
   Plausible, but untested there. Ledger idea 3 tests it, which is right.
7. **Tiny-LR proxy result (70M-125M proxies, targets ≤1B)** applied to a 5M-to-20M mix decision (ledger 7). Untested below 70M.
8. **M2-era kernel-launch latencies (Feng et al.)** applied to the M5 on PyTorch 2.13/2.14, whose releases migrated many ops off MPSGraph
   specifically to cut launch latency. The direction of the error is unknown; E1 measures it.
9. **MobileLLM-LS looping gains (125M/350M, zero-shot)** as the prior for E10 at 5M. The report already says there is no chat evidence.

## Novelty overclaims

- **Ledger 9, "Attention-heavy, MLP-light blocks for a knowledge-light chat model" (marked untested), has been tried as a manipulation.**
  (a) Allen-Zhu & Li, Physics of LMs 3.3, section 7 (arXiv 2404.05405): GPT-2 with a 1/4-size MLP or no MLP keeps the 2 bits/param
  knowledge capacity at 1000 exposures (models down to under 10M). Removing MLPs costs more than 1.5x capacity only at 100 exposures. Their
  reading: "the Attention layers are also capable of storing knowledge". This also weakens the idea's "why": shrinking MLPs does not
  keep facts out of the parameters. (b) Bian et al., ICLR 2026 (arXiv 2510.18245): 200+ models at 80M-3B with the MLP-to-attention ratio
  varied at fixed non-embedding parameters; loss is U-shaped in the ratio with an interior optimum. (c) Liao et al. 2026 (arXiv
  2602.06471): hourglass FFNs with "reduced FFN and increased attention parameters show consistent improvements ... at matched budgets"
  (gains up to 400M, parity to 1B). (d) Martra 2025 (arXiv 2512.22671): cutting the GLU-MLP expansion ratio in Llama-3.2-1B/3B lowers
  knowledge benchmarks while IFEval rises 46-75% (pruning, not training from scratch). What remains untested is the effect on
  **multi-turn chat at 5-20M**. The ledger entry should cite these and say that is the new part.
- **Ledger 6, "Dialogue-shaped MAD unit tests at ~1M"**: no prior test of the dialogue-shaped version was found, so "untested" stands for
  the specific idea. Cheap synthetic pre-screens as such are established (MAD; Zoology's MQAR; Allen-Zhu's Physics 4.1 synthetic
  playground, arXiv 2512.17351, which is pitched as predicting architecture behaviour at scale). The ledger should cite them as the
  baseline the idea extends.

## Omissions

1. **MLX's Metal training path is unfused** (extra check B). This removes the main argument for an MLX engine switch at T=2048. E1 should
   still measure MLX, but for per-op overhead only.
2. **PyTorch 2.14.0 (2026-09-02)**: the MPP prefill kernel (inference only, head dim 64 qualifies), the macOS-26 prefill scramble fix
   (#191794), the NaN fix (#188147), and no native MPS backward for flex attention yet. The report cites the 2.14 docs but describes only
   2.13.
3. **The training-vs-no-grad split in MPS SDPA dispatch** (extra check A), which decides where the self-test must run and what eval numbers
   older versions would corrupt.
4. **Aoyama et al. v3 (2026-07-05)** fixes the constant (claim 6).
5. **Physics of LMs 3.3, section 7** (arXiv 2404.05405): the MLP-ablation result above, from the same paper the project already uses for
   2 bits/param.
6. **Bian et al. (ICLR 2026, arXiv 2510.18245) and Liao et al. (arXiv 2602.06471)** on the attention/MLP split at matched budgets, and
   **Martra (arXiv 2512.22671)** on MLP width vs knowledge vs instruction following. These are direct priors for E8.
7. **Dataset Decomposition** (Pouransari et al., NeurIPS 2024, arXiv 2405.13226): per-document length buckets with a variable-length
   curriculum, reaching target accuracy ~3x faster at 1B. This is prior art for ledger 1 (length-bucketed batches) and ledger 14
   (sequence-length curriculum) and should be E6's reference design.
8. **MTP for small models**: Aynetdinov & Akbik 2025 (arXiv 2505.22757; 1.3B and 3B) report that "smaller language models (SLMs) struggle
   with the MTP objective" and that a forward NTP-to-MTP curriculum helps. Also 2508.19228 (340M: MTP lowered LAMBADA and TriviaQA,
   lanes/arch.verify.md). Ledger 5 should include a curriculum arm or at least cite the risk.
9. **Model merging during pretraining** (Li et al. 2025, arXiv 2505.12082): merging constant-LR checkpoints improves models and predicts
   annealing behaviour. This is the weight-averaging source that ledgers 12 and 15 need (Signal and Noise averages metrics, not weights).
10. **Bouthillier et al. 2021** (arXiv 2103.03098): varying more sources of randomness gives a better benchmark estimator, at far lower
    cost. The report's common-random-numbers design already varies data order across seed indices; it should keep doing so rather than
    fixing one order for all seeds.
11. **Arora et al. 2025** (arXiv 2505.15105): architectures with similar synthetic-recall accuracy can use different mechanisms. This is a
    caution for ledger 6's pre-screen and for reading E4's induction score.
12. **KV-shifting attention** (Xu et al. 2024, arXiv 2411.19574), which lowers the depth and width needed for induction heads, is a
    relevant lever for E4 if 5M fails to form induction. Canon layers (2512.17351) are the other one; the arch lane already covers them.

## What this changes in loop.md

- Section 4.2 table and ledger 4: use T = e^13.26 (~5.74e5). Transitions ~38 / 60 / 92 / 221M tokens.
- Section 2.4 and the harness self-test: the causal-leak test must run under `torch.no_grad()`. Pin one exact release (2.14.0), no
  nightlies, and re-test after PyTorch or macOS updates.
- Section 2.5, bottom line 2, ledger 13: drop "MLX fused attention may win at T=2048". MLX trains attention unfused on Metal.
- Section 5.1 and E3: treat 0.0015 nats as an init-only floor; E3's same-init/new-data vs same-data/new-init arms are exactly the right
  measurement.
- Section 7.5 and ledger 15: cite Signal and Noise for metric averaging, and 2505.12082 or Hägele for weight averaging.
- Ledger 9 / E8: cite Physics 3.3 section 7, Bian, Liao and Martra; restate the novelty as "multi-turn chat at 5-20M". Rewrite the "why",
  because attention also stores knowledge.
- Section 3 launch estimate: count launches per micro-batch, not per optimizer step.
