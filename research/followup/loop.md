# Track: the Planck experiment loop on the M5 Mac (with the RTX 5070 for comparison)

MaxGPT-Planck follow-up research, 2026-09-23. Nothing was run for this report: every throughput number is
either a published or first-run measurement (with its URL or file) or first-principles arithmetic (labelled
**estimate**). Scratch arithmetic: `calc.py` and `mde.py` in the session scratchpad
(`(scratch file)`).

Built on the first run's verified lanes: `lanes/compute.md` + `compute.verify.md` (Mac bench, 5070 spec,
eval noise, paired training loss), `lanes/eval.md` (the MTB-150 battery design), `lanes/arch.md`
(vocab table, shapes), `lanes/data.md` + `data.verify.md` (ufakzeka-1, SmolTalk2), `lanes/context.md`.
Corrections from the `.verify.md` files are used where they apply.

Conventions:
- Every size counts **total** parameters, embeddings included, tied input/output embedding.
- Training FLOPs per token = 6N + 12·L·d·T (the convention the first run used; it counts full, not causal, attention).
- Planck reference shapes used for all arithmetic below (SwiGLU hidden = 8/3·d, full multi-head attention, tied embeddings):

| name | d | layers | vocab | body | embedding | total | GFLOP/token at T=2048 | attention share of FLOPs |
|---|---|---|---|---|---|---|---|---|
| 1M | 96 | 6 | 4,096 | 0.66M | 0.39M | 1.06M | 0.021 | 69% |
| 1M (8k vocab) | 64 | 10 | 8,192 | 0.49M | 0.52M | 1.02M | 0.022 | 72% |
| 5M | 192 | 8 | 8,192 | 3.54M | 1.57M | 5.11M | 0.068 | 55% |
| 20M | 320 | 14 | 8,192 | 17.2M | 2.62M | 19.8M | 0.229 | 48% |
| 60M | 512 | 18 | 8,192 | 56.6M | 4.19M | 60.8M | 0.592 | 38% |

---

## Bottom line

1. The M5 Mac is a **5M-20M screening machine**, not a 60M one. Using the first run's measured 1.5-2.1 effective TFLOPS (PyTorch MPS, bf16, eager) and first-principles scaling, a 5M model trains at roughly **10-19k tokens/s** at T=2048, a 20M at **5-8k**, a 60M at **2.5-3.6k** (estimates, ±2x). So a 5M run at 0.25B tokens takes ~4-7 h, a 20M run at 0.4B tokens ~14-23 h, and a single 60M run at 20 tokens/param ~4-5.5 days. The RTX 5070 would be roughly **10x faster** on the same shapes (estimate), and would run the whole first-10 program in about a day and a half of GPU time instead of about 3-4 weeks.
2. At tiny sizes and 2k context, **attention is half or more of the training FLOPs** (55% at 5M, 48% at 20M), and PyTorch's MPS attention backward still runs through the unfused "math" path (issue opened April 2026), which materializes the T×T score matrix: slow and memory-hungry. The loop should batch conversations **by length without packing**, and E1 must benchmark MLX (whose fused attention may win exactly here) before committing. Pin PyTorch >= 2.13: in 2.12.1 and older, MPS `is_causal=True` attention **silently leaked up to three future tokens** in bf16/fp16.
3. How many tokens: a proxy must be trained (a) to the end of a full LR decay (Wen et al. 0.1-1.2B: rankings can flip during decay), (b) well past the **induction-head phase transition**, which a 2026 law puts at ~750,000·B^0.63·C^0.38 tokens (about 80-120M tokens at 16-32 sequences x 2048 per step, model-size agnostic over 50M-7B), and (c) to at least ~50 tokens/param. That gives **~0.25B tokens at 5M and ~0.4B at 20M** for screening. DataDecide used 100 tokens/param from 4M to 1B.
4. How small can a proxy be? Rankings transfer imperfectly at every published scale. DataDecide gets ~80% of pairwise data-recipe decisions right from 150M to 1B. RegMix ranked data mixtures with 1M-parameter proxies. MAD's 2-block width-128 synthetic tasks rank-correlate with compute-optimal perplexity at 70M-7B, but mainly within one architecture class. In the other direction, Tay et al. found the best architecture "can fluctuate at different scales", and a 2026 replication at 1.2B found most published modifications do not transfer. So the Mac loop screens at 5M, confirms at 20M, and treats any decision that flips between 5M and 20M as scale-dependent until 60M+ runs on the 5070/Titans.
5. Seed noise: final loss is quiet (20 seeds of the 124M speedrun: SD **0.0015 nats**), so a 1-2% loss difference needs only 1-3 seeds per arm if Planck's seed SD is at or below ~0.005-0.01 nats. That still has to be measured (E3). Behavioral pass rates are the opposite. Even with 1,450 conversations, 3 seeds and a 1-point seed SD, the minimum detectable difference is ~3 points. A 1-2 point multi-turn difference is **not resolvable** on the Mac, so the loop decides on continuous metrics: bits-per-byte plus the log-probability margin on the multi-turn likelihood probes. Generative pass rates are reserved for ≥5-point effects and milestones.
6. Budget: with ~100 usable Mac-hours/week (the Mac is shared: a probe run is using it now, and teacher-data generation cannot run beside training in 24 GB), that is **~15-27 runs/week at 5M, ~4-7 at 20M (0.4B) and ~1 at 60M**. At 2 seeds x 2-3 arms, that is ~3-6 screening experiments/week at 5M or ~1 confirmation/week at 20M. The first 10 experiments below take ~260 Mac-hours plus ~100-200 h of 20M confirmations: **~4-5 weeks**, all before the Titans free up (~Dec 7-14).
7. The harness is one PyTorch script (it must also run torch 2.7.1 + fp16 on the Titans) with a parameter-budget solver so every arm is matched on total parameters. Its switches cover layer type, attention/MLP split, sharing/looping, vocabulary and embedding trick. It uses a fixed, hashed data mix and evaluates bits-per-byte plus the MTB-150 multi-turn battery (likelihood tier every checkpoint, generative tier at the end). Each experiment has a pre-registered decision file and results go to an append-only log. A memory-capped, one-job-at-a-time runner with a swap watchdog keeps the unattended Mac from thrashing.

---

## Detailed findings

### 1. The machine

- MacBook Pro, Apple M5, 10 CPU cores (4 performance + 6 efficiency), 24 GB unified memory, macOS 26.6 (read locally with
  `system_profiler`, `sysctl`, `sw_vers`). Apple: 153 GB/s unified-memory bandwidth, Neural Accelerators in each GPU core
  (https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/).
  The M5 has a 10-core GPU (first run's data.verify, same Apple source).
- It is a **laptop**. Sustained multi-day training depends on power adapter, lid/thermal state and macOS sleep policy. Throughput
  drift from throttling does not change results (training is deterministic in tokens), only wall-clock.
- The Mac is **shared**: another run was using it during the first run's verification. The verifier's re-run of the
  first run's `mps_bench.py` gave about half the lane's numbers (5,343 / 2,096 / 1,598 tok/s vs 10,637 / 3,878 / 2,703) while
  a concurrent `LFM2-350M --device mps` job was running (`lanes/compute.verify.md`, claim 9). Contention roughly
  halves throughput. The loop must own the GPU, and the runner must enforce that.

### 2. Training throughput on Apple Silicon: what is published and what is not

**2.1 The only M5 training measurement is the first run's own bench** (`lanes/compute.md` section 3; PyTorch 2.13 MPS bf16 autocast
vs MLX 0.32 bf16; Llama-style decoder with a 49,152 vocab, T=1024, forward+backward+AdamW, no compile):

| proxy | params | MPS tok/s | MLX tok/s | effective TFLOPS (my recompute, 6N+12LdT) |
|---|---|---|---|---|
| p18M (d256, L8) | 18.5-18.9M | 10,637 | 11,819 | MPS 1.47, MLX 1.64 |
| p35M (d384, L10) | 34.6-36.6M | 6,906 | 7,951 | 1.84 / 2.12 |
| p60M (d512, L12) | 60.6-62.9M | 3,878 | 4,756 | 1.76 / 2.15 |
| p113M | 113.3M | 2,703 | 2,707 | 2.1 |

Caveats (from the verify file): the verifier could not reproduce these cleanly because of GPU contention (see above). In
those proxies the 49k-vocab output head was 40-67% of the 6N FLOPs, and that head is one large, efficient GEMM. Planck's small
vocab moves the work into the narrow body GEMMs and attention, which are less efficient, so effective TFLOPS at d≤256 will
likely be **lower** than 1.5 (inference, unmeasured). A single large bf16 GEMM reached 12.5-14 TFLOPS through MPS (first run and
verifier), so the M5's matrix hardware is not the bottleneck. Dispatch overhead, memory traffic and unfused ops are.

**2.2 No public from-scratch small-LM training benchmark on M-series exists that I could find.** Searches for nanoGPT, llm.c,
nanochat and MLX training numbers on M3/M4/M5 turned up only inference benchmarks. The nanochat MLX port gives rough
whole-run estimates on an M3 Pro, with no tok/s and with a caveat that runtime "depends on thermals, data shards, background memory
pressure" (https://github.com/scasella/nanochat-mlx). Karpathy's nanoGPT README says only that MPS gives a "2-3X"
speedup over CPU (https://github.com/karpathy/nanoGPT).

**2.3 Apple Silicon vs NVIDIA, measured (M2 generation).** Feng et al., "Profiling Apple Silicon Performance for ML Training"
(https://arxiv.org/abs/2501.14925; GPT-2 large and Whisper on M2 Pro/Max/Ultra vs A6000/4090/2080Ti/RTX 4000):
- Follow-up kernel launch time **0.127-0.264 ms on M2 Pro/Max/Ultra vs 0.0036-0.0064 ms on CUDA GPUs** (their Table 6), so
  per-kernel dispatch is ~20-70x slower on Apple Silicon. Consequence for Planck: tiny models need many tokens per optimizer step
  so that GEMM time dominates launch time. At 5M with ~850 kernels/step, launches cost ~0.1 s/step (**estimate**), which is under 5%
  of step time at 32k tokens/step.
- MLX beat PyTorch MPS on GPT-2-large pretraining passes: 2.71 vs 3.24 s (M2 Ultra), 2.92 vs 3.65 s (M2 Max), 8.69 vs 10.70 s
  (M2 Pro), i.e. **~1.2x** (Table 4). This matches the first run's 1.0-1.23x on the M5.
- FP16 gave MLX "only about 20%-30% benefit" over FP32 on matrix-matrix products, vs much larger gains on CUDA tensor cores (M2, pre-Neural-Accelerator).
  Page faults climbed steadily when training ran near memory capacity. That is the thrash mechanism the loop must avoid.

**2.4 PyTorch on MPS: facts that change the harness design.**
- **bf16:** MPS bf16 needs macOS 14+ (https://docs.pytorch.org/docs/2.13/notes/mps.html states the macOS 14.0+ requirement for MPS);
  autocast was extended to bf16 in PR #139390 (per https://github.com/pytorch/pytorch/issues/139386 and #141774). The first run
  trained with bf16 autocast on PyTorch 2.13.
- **Silent causal-mask bug:** "On MPS, PyTorch ≤ 2.12.1 has a silent bug in `scaled_dot_product_attention(..., is_causal=True)`
  for `float16`/`bfloat16`": the mask is applied per block of four query positions, so a query can see up to three future keys.
  It was "fixed in PyTorch 2.13.0" (https://github.com/multimodal-art-projection/YuE/issues/176 ; original report
  https://github.com/pytorch/pytorch/issues/195910, M5 Max, max abs error 2.87 vs 0.01 with an explicit mask; float32 unaffected).
  **Harness rule: require torch >= 2.13 on the Mac, and run a causal-leak self-test at startup.**
- **Attention backward is unfused on MPS:** "the backward pass is implemented using implicit composite autograd from the
  device-agnostic 'math' backend function", and a Metal kernel `sdpa_full_attention_mps` "isn't called at all"
  (https://github.com/pytorch/pytorch/issues/179294, open, filed 2026-04-03). The math path materializes the H×T×T
  score and probability tensors. **Estimate:** at T=2048 that is ~75 MB per layer per sequence at 5M (3 heads) and ~126 MB at 20M
  (5 heads), i.e. ~0.6 GB and ~1.8 GB per sequence across all layers, before any other activation. This caps micro-batches at
  ~4-8 sequences and wastes bandwidth. Because attention is 48-55% of FLOPs at 5-20M and T=2048, this is probably
  the largest single inefficiency for Planck on PyTorch MPS.
- **PyTorch 2.13 (July 8 2026)** migrated many MPS ops from MPSGraph to hand-written Metal kernels because MPSGraph "adds
  compilation and scheduling overhead per dispatch", "reducing kernel launch latency"; added **FlexAttention on MPS** (Metal
  kernels "for both the sparse prefill and decode paths"; "Dense patterns continue to favor SDPA"; training backward on MPS is not
  mentioned); and added `nn.LinearCrossEntropyLoss` (chunked head+loss, up to ~4x lower peak memory for large vocabularies)
  (https://pytorch.org/blog/pytorch-2-13-release-blog/). With an 8k vocab the logits are small (16k tokens x 8,192 x 4 B = 0.5 GB),
  so the fused loss matters less for Planck than it did for the 49k-vocab bench.
- **torch.compile on MPS** uses a Metal backend in TorchInductor that is still described as an early prototype with open
  codegen failures (e.g. https://github.com/pytorch/pytorch/issues/152155). No published training speedup on MPS was found.
  It is an E1 arm, not a default.
- **Memory knobs** (https://docs.pytorch.org/docs/2.14/mps_environment_variables.html):
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO` is a hard allocation ceiling as a multiple of the device's recommended working set,
  **default 1.7** (so by default PyTorch may allocate well past what fits in RAM and push macOS into swap);
  `PYTORCH_MPS_LOW_WATERMARK_RATIO` is a soft target (default 1.4 on unified memory); `PYTORCH_ENABLE_MPS_FALLBACK=1` silently
  moves unsupported ops to CPU. `torch.mps` exposes `recommended_max_memory()`, `driver_allocated_memory()`,
  `set_per_process_memory_fraction()` and `empty_cache()` (https://docs.pytorch.org/docs/2.14/mps.html).

**2.5 MLX on the M5.** Apple says MLX uses the M5's GPU Neural Accelerators through Metal 4 TensorOps, requires **macOS 26.2 or
later** (this Mac has 26.6), and reports 3.3-4x faster prompt processing than the M4 but only 1.19-1.27x faster generation
(bandwidth-bound). It reports no training numbers (https://machinelearning.apple.com/research/exploring-llms-mlx-m5).
The first run measured MLX 0.32 bf16 GEMM at only ~3.8 TFLOPS vs MPS 13-14, while MLX still trained the 18-60M proxies 1.1-1.2x faster.
So MLX's advantage most likely came from lower per-op overhead, not GEMM speed (inference). `mx.fast.scaled_dot_product_attention` supports causal masks and
computes softmax in float32 (https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html).
Whether its **backward** is fused is claimed by third-party packages but not stated in MLX's docs (unverified). If it is fused, MLX
could beat PyTorch MPS by more than 1.2x at T=2048, where the MPS math-path backward hurts most. E1 decides.

### 3. Throughput estimates for the Planck sizes

**Method (estimate).** tok/s = effective TFLOPS / (FLOPs per token). Mac effective TFLOPS by width are anchored on the first run's
1.5-2.1 at d=256-512 and lowered for narrow widths and attention-heavy shapes: 0.3-0.8 (d≤128), 0.7-1.3 (d=192), 1.1-1.8 (d=320),
1.5-2.1 (d=512). The 5070 values assume 8-50% of its 61.7 dense bf16 TFLOPS (Blackwell whitepaper via `lanes/compute.md`),
anchored on public efficiency points: llm.c GPT-2 124M reached ~70% MFU on an RTX 5080 in a 74-step test and 56.7% on a 4080 Super
over 10B tokens (https://github.com/karpathy/llm.c/issues/796 ;
https://github.com/karpathy/llm.c/discussions/481#discussioncomment-11005883). Karpathy's llama2.c 110M took ~24 h on 4xA100
for ~26B tokens, about 19% MFU per A100 (my arithmetic from https://github.com/karpathy/llama2.c). The 10.7M "baby GPT" in
nanoGPT trains 82M tokens (plus eval passes) in "about 3 minutes" on one A100 (https://github.com/karpathy/nanoGPT ; config
`train_shakespeare_char.py`), which is at least ~450k tok/s and ~10% MFU. Small models run far below peak on any GPU.

| size | Mac tok/s at T=2048 | Mac at T=1024 | 5070 tok/s at T=2048 | notes |
|---|---|---|---|---|
| 1M | 15-39k | 22-60k | 140-390k | overhead-dominated; synthetic tasks only (section 4.4) |
| 5M | 10-19k | 14-26k | 90-200k | attention 55% of FLOPs at T=2048 |
| 20M | 4.8-7.9k | 6.3-10.3k | 50-95k | attention 48% |
| 60M | 2.5-3.6k | 3.1-4.4k | 34-54k | first-run anchor p60M: 3.9-4.8k at T=1024 with a 49k head |

All **estimates**, ±2x. The first run's 150M estimate for the 5070 (~32k tok/s, range 24-40k) is consistent with the 60M row.

Wall-clock per run (**estimates** from the table):

| run | tokens/param | Mac | 5070 |
|---|---|---|---|
| 1M @ 0.1B | 95 | 0.7-1.9 h | 4-11 min |
| 5M @ 0.25B | 49 | 3.7-6.8 h | 20-47 min |
| 5M @ 0.5B | 98 | 7.3-13.6 h | 0.7-1.6 h |
| 20M @ 0.4B | 20 | 14-23 h | 1.2-2.1 h |
| 20M @ 1B | 50 | 35-58 h | 2.9-5.3 h |
| 60M @ 1.2B | 20 | 94-131 h (4-5.5 days) | 6-10 h |
| 60M @ 6B | 99 | 20-27 days | 31-49 h |

A final headline-curve model at hundreds of tokens per parameter (for example 20M @ 10B, 500 tokens/param) would take
~15-24 days on the Mac. Final curve points belong on the 5070/Titans. The Mac's job is screening.

**Evaluation cost (estimate).** Held-out bits-per-byte on a fixed 2M-token set and the likelihood probes (~1,000 items x ~600
tokens x 2-3 candidates) are forward-only, so each takes ~1-2 minutes at 5-20M. The generative multi-turn battery (~1,450
conversations x ~6 assistant turns x ≤64 new tokens, about 0.56M generated tokens) must be **batched**. At batch 1 on MPS, per-token
latency is launch-bound (~10-20 ms per decode step at 5-20M given the 0.13-0.26 ms launch times above), which would take ~2-3 h per
checkpoint. At batch 64 it takes ~2-10 minutes.

### 4. How many tokens a trustworthy comparison needs, and how small a proxy can be

**4.1 Compare only at the end of a learning-rate decay.** Wen et al. (0.1B-1.2B, 1-8x Chinchilla) find optimizer "rankings
between two optimizers can flip during training due to learning rate decay" (https://arxiv.org/abs/2509.02046). Max's own 124M A/B
did **not** show a flip on the paired training loss. The apparent flip was eval-batch noise (`lanes/compute.verify.md`, claim 20b), so the
rule rests on Wen et al. alone. Use a WSD schedule: constant LR plus a short cooldown "scales predictably and reliably similar to
cosine", and "scaling experiments can be performed with significantly reduced compute and GPU hours by utilizing fewer but reusable
training runs" (Hägele et al., NeurIPS 2024, https://arxiv.org/abs/2405.18392; 124M-scale and up).

**4.2 Train past the induction-head phase transition.** Multi-turn recall is an in-context copying skill, so a proxy measured before its
induction circuit forms says nothing about it. Aoyama, Wilcox and Schneider (arXiv 2511.16893, v2 Feb 2026) fit the transition point on
35 GPT-2/Pythia models and find it model-size agnostic: N_PT ≈ 750,000 · B^0.63 · C^0.38 tokens (B = sequences per update,
C = context length), r = 0.98 in log space, holding "for more than 3 orders of magnitude in model size (50M-7B), and 5 orders of magnitude
in the number of tokens per update (500-2M)" (https://arxiv.org/abs/2511.16893). The literature range is 64M-3B tokens. Applied to Planck
(my arithmetic):

| batch | tokens/step | predicted transition |
|---|---|---|
| 8 x 2048 | 16k | ~50M tokens |
| 16 x 2048 | 33k | ~78M |
| 32 x 2048 | 66k | ~121M |
| 128 x 2048 | 262k | ~289M |

Scale flag: the smallest model in that fit is 50M (a 2-layer GPT-2 with a 50k vocab). Whether 1-20M models follow the same law is
untested. Separately, multi-token-prediction work found next-token models of ≤30M non-embedding parameters form induction
capability much worse than MTP-trained ones on a children's-story name task, with the gap gone at ≥100M (Gloeckle et al.,
https://arxiv.org/abs/2404.19737, confirmed in `lanes/arch.verify.md` claim 9). Planck's 5M and 20M proxies sit exactly in the
weak zone. E4 measures it.

**4.3 Tokens per parameter.** Chinchilla-style 20 tokens/param is the floor. DataDecide trained 14 sizes from 4M to 1B all at 100
tokens/param (4M on 0.4B tokens, 60M on 5.7B, 150M on 15B, 1B on 100B), with 3 seeds (https://arxiv.org/html/2504.11393). Planck's final
models will be trained at hundreds to thousands of tokens per parameter, and rankings can move with budget (Wen et al.: matrix-optimizer
gains shrink from 1.4x at 0.1B to 1.1x at 1.2B, and Muon is overtaken at 8x Chinchilla). Every winner at 50 tokens/param therefore gets one
longer confirmation before it enters the headline recipe.

**Recommended screening budgets** (combining 4.1-4.3):
- **1M:** 0.1-0.2B tokens, synthetic unit tests only (4.4).
- **5M:** **0.25B tokens** (49 tokens/param; 2-3x past the predicted transition at 16-32 sequences x 2048), full WSD decay.
- **20M:** **0.4B tokens** (20 tokens/param) for confirmation on the Mac; 1-2B (50-100 tokens/param) when the 5070 is back.
- **60M:** not a Mac screening size. One 1.2B-token baseline for the curve at most.

**4.4 How small before rankings stop transferring: the evidence.**
- **Data recipes transfer reasonably from small proxies.** DataDecide: ranking at "a single, small size (e.g., 150M parameters)" picks
  the better 1B recipe in "~80% of comparisons". Continuous likelihood metrics make benchmarks ">80% predictable at the target 1B scale with
  just 0.01% of the compute" (https://arxiv.org/abs/2504.11393). RegMix trained 512 **1M-parameter** models on 1B tokens each and picked a
  mixture that "performs best among 64 candidate 1B parameter models" trained on 25B tokens (https://arxiv.org/abs/2407.01492). For data
  curation, proxies trained with a tiny LR (1e-6) reached Spearman ρ > 0.95 with tuned 1B targets vs ρ < 0.75 at a standard LR (70M-125M
  proxies, 23 recipes; ICLR 2026, https://arxiv.org/html/2512.24503).
- **Architecture rankings transfer less reliably.** Tay et al. (10 architectures, T5 family): "the best performing model can fluctuate at
  different scales" (https://arxiv.org/abs/2207.10551). A 2026 replication of 20 post-2021 Transformer modifications at 1.2B/3B found "most
  modifications do not transfer": only 2 cleared Bonferroni at 1.2B, and two attention-output changes landed "within 2-3% of baseline
  validation loss yet drop 6-16 CLIMB-points" (https://arxiv.org/abs/2605.20798). Loss alone is not a safe proxy for behaviour.
- **Synthetic unit tests at microscopic scale do carry signal.** MAD trains **2-block, width-128** models on recall, fuzzy/noisy recall,
  selective copy, compression and memorization tasks ("only minutes of training time"). MAD accuracy "is rank-correlated with
  compute-optimal perplexity at scale", measured on 500+ models of 70M-7B, "with particularly strong correlation for models in the same
  architecture class". Correlation across classes "is subject to more noise", and the authors flag that extrapolating from 2-block models
  to deep topologies is untested (https://arxiv.org/abs/2403.17844, sections 1, 5 and limitations).
- **Downstream scaling is often not smooth**: a meta-analysis finds predictable downstream scaling in only 39% of cases
  (https://arxiv.org/abs/2507.00885).

Reading for Planck (inference): use 1M models only for MAD-style synthetic tests of in-context mechanics. Use 5M for screening
recipe and data questions that the multi-turn likelihood probes can see. Use 20M for confirmation. Do not adopt an architecture change on
5M evidence alone. The ~80% figure is the realistic best case for any single-scale decision.

### 5. Seed-to-seed noise and how many seeds a 1-2% difference needs

**5.1 Published noise levels.**
- **Loss, 124M, under 0.4B tokens (speedrun; token count per the README, `lanes/arch.verify.md` claim 12):** 20 independent runs of one modded-nanogpt record: mean 3.27886, **SD 0.00149 nats**
  (https://github.com/KellerJordan/modded-nanogpt/pull/358). The rules require p < 0.01 over multiple runs because of this variance
  (https://github.com/KellerJordan/modded-nanogpt).
- **Pythia 14M-410M, 300B tokens, 10 seeds per size (PolyPythias):** "language modelling remains largely stable". The only outliers were two
  runs, both at 410M, which showed loss spikes (https://arxiv.org/abs/2503.09543). There was no instability at the small sizes.
- **Benchmarks, 1B at 100 tokens/param:** "standard deviation between runs ... can be as high as 2% points of accuracy for some recipes on
  most tasks" (DataDecide, https://arxiv.org/html/2504.11393).
- **Metric choice dominates signal-to-noise.** Switching MBPP from accuracy to bits-per-byte raised SNR "from 2.0 to 41.8" and decision
  accuracy "from 68.3% to 95.3%". Averaging checkpoints improved decision accuracy "+2.4%". Seed, data-order and checkpoint-to-checkpoint
  noise correlate strongly (R² 0.82-0.95), so the cheap checkpoint-to-checkpoint noise is a usable proxy for seed noise (Heineman et al.,
  DataDecide models 60M-1.3B, https://arxiv.org/html/2508.13144).
- **Chat metrics at Planck's scale are noisy:** ufakzeka-1 (151M, 13.5B tokens, LLM-judged, Turkish) found seed variance as large as the
  spread across every data recipe it tried: helpfulness 75.2 / 80.2 / 81.2 over three seeds vs 75-82 across 14 checkpoints
  (https://arxiv.org/abs/2609.25081, via `lanes/data.verify.md`).
- **Seeds vs size:** "training multiple small models is sometimes more useful than training a single large one" because of seed variability
  (Choshen et al., 485 published models, https://arxiv.org/abs/2410.11840).
- **Pairing is free and strong in Max's own data:** with identical data order, the four 124M A/B arms' detrended training-loss
  fluctuations correlated at 1.00. The paired training loss gave the recipe gain at 0.134 nats with an SE under 0.1 percentage points
  (`lanes/compute.verify.md` claim 16, `lanes/training.verify.md` claim 12). Common random numbers (same data order per seed index
  across arms) should be the harness default.

**5.2 Minimum detectable effect (my arithmetic, α = 0.05 two-sided, 80% power, MDE = 2.8·σ·sqrt(2/k), unpaired).**

Loss, seed SD σ (nats) vs seeds per arm k:

| σ | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|
| 0.002 | 0.008 | 0.006 | 0.005 | 0.004 |
| 0.005 | 0.020 | 0.014 | 0.011 | 0.009 |
| 0.010 | 0.040 | 0.028 | 0.023 | 0.018 |
| 0.020 | 0.079 | 0.056 | 0.046 | 0.035 |

At a loss of ~2.5 nats, 1% is 0.025 nats: **1 seed suffices if σ ≤ 0.005, 3 if σ = 0.01, 11 if σ = 0.02.** At a loss of 1.5 nats
(plausible on simple dialogue text), 1% = 0.015 needs 2 seeds at σ = 0.005 and 7 at σ = 0.01. The speedrun's 0.0015 suggests Planck is
in the 1-2 seed regime for loss, but that was measured at 124M on a tuned recipe. **Planck's σ at 5M/20M is unmeasured (E3).**

Behavioral pass rate (points), n paired conversations with 20% discordant, seed SD σ_s (points), k seeds per arm:

| n | σ_s | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|---|
| 300 (one family) | 1.0 | 8.2 | 5.8 | 4.8 | 3.7 |
| 1,450 (MTB-150) | 0.0 | 3.3 | 2.3 | 1.9 | 1.5 |
| 1,450 | 1.0 | 5.1 | 3.6 | 3.0 | 2.3 |
| 1,450 | 2.0 | 8.6 | 6.1 | 5.0 | 3.8 |
| 4,000 | 1.0 | 4.4 | 3.1 | 2.6 | 2.0 |

A 1-2 point generative multi-turn difference cannot be resolved with any seed count the Mac can afford. Once seed variance is
non-zero, more items stop helping. Hence the loop's rule: **decide on continuous metrics** (held-out bits-per-byte and the
**log-probability margin** on the likelihood probes, which carry per-item magnitude instead of a 0/1 outcome, per the Signal-and-Noise
result above), and use generative pass rates only for effects of ≥5 points and at milestones.

### 6. Experiments per week per machine

Assumptions: Mac ~100 usable GPU-hours/week (shared with the running probe, Max's daily use and teacher generation, which cannot run beside
training in 24 GB). 5070 ~150 h/week when it returns. Titans: none until ~Dec 7-14 (`lanes/compute.md`). All **estimates**.

| machine | 5M @ 0.25B | 20M @ 0.4B | 20M @ 1B | 60M @ 1.2B | 60M @ 3B |
|---|---|---|---|---|---|
| M5 Mac (runs/week) | 15-27 | 4-7 | 1.7-2.8 | 0.8-1.1 | 0.3-0.4 |
| RTX 5070 (runs/week) | 190-450 | 70-125 | 28-52 | 15-24 | 6-10 |

In experiments/week at 2 arms x 2 seeds: Mac ~4-7 at 5M or ~1-2 at 20M/0.4B. 5070 ~50-110 at 5M or ~7-13 at 20M/1B.
A 5070 week equals roughly 10 Mac weeks for this workload.

**Competing demand on the Mac: data generation.** The first run's data lane estimates the M5 can generate ~5-10M teacher tokens/day
(derived, unverified). A 26B-A4B teacher at 4-bit needs ~15 GB, so it cannot run beside training. 50M tokens of verified synthetic
conversations would cost ~5-10 Mac-days. The v0 loop mix (section 7.3) therefore leans on public dialogue data plus a small generated
set, and bulk generation waits for the 5070.

### 7. The harness

**7.1 One script, one config, param-matched arms.**

```yaml
# experiments/E07/arm_b.yaml  (illustrative)
budget: {total_params: 5.0e6, tolerance: 0.02}     # solver adjusts d (or L) to hit TOTAL params incl. embeddings
model:
  d_model: auto            # solved from budget unless fixed
  n_unique_layers: 8
  loop: {times: 1, mode: immediate}        # sharing/looping: immediate | cycle; times>1 reuses blocks
  layer_pattern: [attn]                    # layer type per position: attn | local_attn(w) | conv | gdn
  attn: {n_heads: 3, n_kv_heads: 3, qk_norm: true, rope_theta: 10000}
  mlp: {type: swiglu, hidden_mult: 2.667}  # attention/MLP split lives here (1.33 / 2.67 / 4)
  vocab: {tokenizer: tok/bpe8k.json}       # 2k / 4k / 8k / 16k tokenizers trained on the same mix
  embedding: {mode: tied, rank: null}      # tied | untied | factorized(rank) | +value_embeds | +hash_ngram(n, rows)
  aux_loss: {mtp: 0}                       # multi-token prediction heads (ledger idea)
train:
  tokens: 2.5e8            # or bytes: for vocab arms, match BYTES not tokens
  batch: {seqs: 16, seq_len: 2048}         # tokens/step; micro-batch auto-probed under the memory cap
  batching: length_bucketed                # length_bucketed | packed_docmask | packed_fixed_offset
  optimizer: {name: adamw, lr: tuned, wd: 0.1}   # or normuon (Max's A/B winner)
  schedule: {type: wsd, warmup: 0.01, decay: 0.2}
  precision: bf16                          # fp16+GradScaler on Turing
  seed_init: 0
  seed_data: 0                             # common random numbers: same data order across arms per seed index
  compile: false
eval: {every_tokens: 2.5e7, bpb_sets: [general, dialogue_assistant, planted_fact_spans], tier0: true, tier1_at_end: dev}
```

- **Budget solver:** given the switches, solve for d_model (default) or layer count so total parameters (embeddings included) land within
  ±2% of the target, then print the exact count. Arms are always param-matched without hand arithmetic. Looped models count
  unique parameters. Hash/value-embedding tables count in full (Planck's rule is total parameters).
- **Portability:** the same model code must run on PyTorch 2.13+ (Mac, for the SDPA fix) and **2.7.1 + cu118 + fp16** on the Titans
  (driver 470; `lanes/compute.md` section 6.5). Core paths must avoid APIs newer than 2.7, and newer features (FlexAttention on MPS,
  `nn.LinearCrossEntropyLoss`) are optional behind feature checks. If E1 makes MLX the Mac engine, an MLX mirror of the model file
  needs a parity test: same weights give the same logits within bf16 tolerance.
- **Startup self-tests (fail fast):** (1) causal-leak test: attention output vs a float32 CPU reference with an explicit mask, and
  training loss on a sequence whose future tokens are a deterministic function of the past must not fall below the no-leak bound;
  (2) param-count assertion; (3) a 3-step memory probe that picks the micro-batch under the cap; (4) a data-manifest hash check.

**7.2 Batching conversations.** The first run found MaxGPT's SFT packer slices conversations at fixed offsets, so later turns lose their
history (`lanes/compute.md` section 7, confirmed). On MPS, a document mask forces the math-path attention over the full T×T anyway.
The Planck default is therefore **length-bucketed, unpacked batches**: conversations grouped into 512/1024/2048 buckets, padded to the
bucket and never packed. Attention cost follows real lengths, and no cross-conversation leakage is possible. General text is chunked to
the bucket length. E6 measures the alternatives.

**7.3 A fixed, versioned data mix (v0, by bytes; the data track owns the final choice).**
- ~50% dialogue: public multi-turn chat. Candidates: SmolTalk2's everyday-conversation and multi-turn-IF splits (`lanes/data.verify.md`),
  TinyDialogues (~130k GPT-4-generated child-directed conversations of 5 or 10 turns, https://huggingface.co/datasets/styfeng/TinyDialogues ;
  EMNLP 2024, https://arxiv.org/abs/2408.03617), OASST. Check each license before use.
- ~10% verified synthetic planted-fact conversations from the Gemma 4 26B-A4B teacher (small, repeated), with templates and slot pools
  **disjoint from the sealed eval families** (13-gram decontamination, `lanes/eval.md` D7).
- ~40% simple-register general text (SimpleStories/TinyStories-style or a high-readability web slice) for grammar.
- Frozen as a manifest of shard hashes. The tokenizer family (2k/4k/8k/16k BPE) is trained once on this mix. Changing the mix bumps the
  mix version, and results from different mix versions are never compared.

**7.4 Evaluation, every checkpoint (forward-only, ~1-3 min at 5-20M, estimate).**
- **Bits-per-byte** (tokenizer-agnostic, so vocab arms are comparable) on three fixed held-out slices: general text, assistant turns of
  held-out dialogues, and answer spans of held-out planted-fact conversations. Fixed means the same windows every time, which is the
  first run's fix for its 0.17-0.21-nat eval-sampling noise.
- **MTB-150 Tier 0 likelihood probes** (`lanes/eval.md` D1: LP-Recall at 0/2/4/8 distractor turns, LP-Update, LP-Coref, LP-Persona,
  LP-Continuation), reported as both pass rate and **mean log-probability margin** (correct minus best in-conversation distractor).
  Shortcut baselines (most recent, first, most frequent candidate) are computed on every run.
- **History-dependence score** (`lanes/compute.md` 8.2): loss on turn k with vs without the history.
- **Induction/prefix-matching score** on repeated random-token sequences, to locate the phase transition (4.2).
- **End of run:** Tier 1 generative families on the **dev** split, greedy, own history, **batched** generation, plus the loop/role-leak
  diagnostics. The sealed split runs only at milestones.

**7.5 Results log and decisions.**
- `results/runs.jsonl`, append-only, one record per eval: run id, experiment id, arm, seeds, config hash, git SHA, mix version, torch/MLX
  and macOS versions, tokens, tok/s, peak `driver_allocated_memory`, swap used, all metrics.
- `experiments/EXX/prereg.yaml` is written **before** launch: question, arms, primary metric, MDE (from E3), seeds, and a machine-readable
  decision rule. `analyze.py EXX` computes paired per-seed differences, a bootstrap CI and the verdict (ADOPT / REJECT / TIE /
  SCALE-DEPENDENT), then appends a row to the public idea ledger. Changing the rule after seeing results is visible in git history.
- Checkpoint averaging (last 3-5 checkpoints of the decay) for final numbers, per Signal and Noise.

**7.6 Running unattended on the Mac without memory thrash.**
1. **One GPU job at a time.** A queue directory plus a lock file (`flock`). The runner will not start a job if another process holds the
   lock, if a known MPS/MLX workload is running (the probe, a teacher server), or if `sysctl vm.swapusage` shows more than ~1 GB used.
   Past incident: concurrent model loads pushed this Mac to 23.8 GB swap (memory file `feedback_serialize_model_loads.md`).
2. **Hard memory cap:** `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` and `LOW=0.6` instead of the default 1.7/1.4, so an oversized batch fails
   with an OOM instead of swapping. Also `PYTORCH_ENABLE_MPS_FALLBACK=0`, so an unsupported op fails loudly instead of silently running on
   CPU. The startup memory probe must fit the run under ~12-14 GB of the 24 GB.
3. **Watchdog thread** (every 60 s): if swap grows by >2 GB since start, or step time exceeds 2x its running median for 5 minutes (the
   thrash tell), write a checkpoint and exit with status `THRASH`. The queue moves on.
4. **Stay awake:** wrap the process in `caffeinate -i -s` (`-s` holds only on AC power). Start it from a Terminal/tmux session, or place the
   runner under `~/Library/Application Support`. A launchd agent that executes from `~/Documents` is blocked by macOS privacy controls
   with exit 126 (memory file `reference_launchd_tcc_documents.md`).
5. **Resumable by construction:** atomic checkpoints every ~30 minutes (write tmp, then rename). The data order is a pure function of
   (seed_data, step), so resume is exact. The queue re-launches `RESUMABLE` exits.
6. **Log the machine state** with every eval record: tok/s, `pmset -g therm` state, swap and allocated memory, so throttled or contended
   stretches are visible.
7. **Schedule:** training blocks and teacher-generation blocks alternate, never overlap. Max's interactive use gets priority; the runner
   can pause between steps on a flag file.

### 8. The first 10 experiments (priority order)

Wall-clock is Mac time from the estimates in section 3 (E1 replaces them), with the Mac otherwise idle. "Reuse" means an earlier run
serves as that arm. Seeds default to 2 per arm with common random numbers (arm-paired data order), revised by E3. Decision metrics:
**primary = Tier 0 multi-turn log-prob margin (composite); guard = bits-per-byte on the dialogue slice**, unless stated otherwise.

| # | question | arms | size, tokens | seeds | Mac wall-clock | decision rule |
|---|---|---|---|---|---|---|
| E1 | What are the real speed, memory and correctness on this Mac? | PyTorch MPS eager / MPS `torch.compile` / MLX; T = 1024 and 2048; length-bucketed vs packed+mask; 5M, 20M, 60M shapes | ~200 steps per config | n/a | ~2-3 h | Choose the fastest path that passes the causal-leak test, matches fp32 loss within 1% over 200 steps and peaks under 12 GB. Switch the Mac engine to MLX only if it is ≥1.3x faster at both 5M and 20M; otherwise stay on PyTorch for Titan portability. Replace section 3's estimates. |
| E2 | Is the baseline well-tuned, and does its LR transfer from 5M to 20M? | 5M: LR x{0.5, 1, 2, 4} at 16x2048 tokens/step, plus the best LR at 32x2048; 20M: best LR and 0.5x | 5M @ 0.25B; 20M @ 0.2B | 1 | ~25 h + ~18 h | Take the LR with the lowest dialogue bits-per-byte after decay. If the 20M optimum sits ≥2x away from the 5M optimum, add muP/CompleteP scaling (Tensor Programs V transferred from a 40M proxy to 6.7B, https://arxiv.org/abs/2203.03466) before any cross-size claim. |
| E3 | How much seed noise is there, and how much does pairing remove? | Tuned baseline at 5M: 2 new full seeds, 1 same-data/new-init, 1 same-init/new-data (E2 run reused); 20M: 2 new seeds (E2 run reused) | 5M @ 0.25B; 20M @ 0.4B | 3-5 | ~20 h + ~36 h | Set seeds per arm k = ceil(2·(2.8σ/MDE)²) with MDE = 1% bits-per-byte and the matching Tier 0 margin MDE. If sharing the data order cuts the arm-difference SD by ≥2x, common random numbers become mandatory. |
| E4 | What is the smallest size with multi-turn signal, and does the induction law hold at 1-20M? | 1M (4k and 8k vocab shapes), with E3's 5M and 20M checkpoint series | 1M @ 0.2B | 1 | ~4-8 h + ~2 h eval | The minimum decision size for multi-turn questions is the smallest size whose Tier 0 recall margin beats the best shortcut baseline by ≥5 seed-SDs before half of training. Smaller sizes run only synthetic unit tests. If the observed transition misses the law's prediction by >3x, re-derive the token budgets in 4.3. |
| E5 | How much dialogue belongs in the fixed mix? | 25% / 50% (baseline, reused) / 75% dialogue by bytes | 5M @ 0.25B | 2 | ~20 h | Adopt the share with the best Tier 0 margin, unless general-text bits-per-byte gets >3% worse (a grammar-collapse guard). Repeat the comparison at 0.1x LR (tiny-LR proxy check) and treat the ranking as robust only if both LRs agree. Confirm the top two at 20M. |
| E6 | Does the conversation batching path matter for recall at distance? | Length-bucketed unpacked (baseline, reused) / packed + document mask / packed at fixed offsets (the MaxGPT SFT packer) | 5M @ 0.25B | 2 | ~20 h | Keep length-bucketed if its LP-Recall margin at 4-8 distractor turns is within MDE of the best arm and its tok/s is ≥ packed. Report the fixed-offset penalty to the Ultra/MaxGPT SFT fix. |
| E7 | Which vocabulary and embedding trick wins at a fixed TOTAL budget? | 2k / 4k / 8k (baseline, reused) tied; 8k factorized embedding (rank 48) | 5M, equal **bytes** (~0.96 GB, about 0.25B tokens at 8k); top two confirmed at 20M | 2 | ~30 h (+~72 h to confirm at 20M) | Adopt the arm with the lowest bits-per-byte unless the Tier 0 margin disagrees by more than its MDE (Tier 0 wins). Prior: extrapolating Tao et al.'s optimal-vocab law (Nv ∝ Nnv^0.83, 16K at 302M non-vocab params, d=1024) predicts ~2k at 5M, ~5k at 20M and ~8k at 60M (https://arxiv.org/abs/2407.13623; below the fitted 33M-1.13B range). |
| E8 | Does moving parameters from MLP to attention help a knowledge-light chat model? | SwiGLU hidden 1.33d / 2.67d (baseline, reused) / 4d, with depth re-solved to hold total params | 5M @ 0.25B; confirm at 20M | 2 | ~20 h (+~36-72 h) | Adopt MLP-light if the Tier 0 margin improves by ≥ MDE and dialogue bits-per-byte is not >1% worse. If E4 shows 5M has no multi-turn signal, run this at 20M only. |
| E9 | Deep-thin or wide at a fixed total? | 5M: (d128, L16) / (d192, L8) baseline, reused / (d256, L4); then 20M: (d256, L~22) / (d320, L14) / (d448, L~7) | 5M @ 0.25B; 20M @ 0.4B | 2 | ~24 h (+~72 h) | Prefer deeper only if Tier 0 or bits-per-byte improves by ≥ MDE per training token. Report per-hour results too, because deeper stacks launch more kernels on MPS. |
| E10 | Does looping add information per parameter (route 2)? | 1x (baseline, reused) / 2x immediate block repeat / 2x cyclic repeat, all with the same unique parameters | 5M @ 0.25B | 2 | ~40 h (looped arms cost 2x compute) | Adopt if the Tier 0 margin gain is ≥ MDE at matched unique parameters. Record the 2x inference cost in the ledger. MobileLLM-LS measured +0.7 to +1.1 zero-shot points at 125M/350M (`lanes/arch.md`); there is no chat evidence yet. |

Total: ~260 Mac-hours for E1-E10 at 5M, plus ~100-200 h of 20M confirmations, which is **about 4-5 weeks** at ~100 usable hours/week.
On a 5070 the same program is ~25-40 GPU-hours (estimate). Before E1 there is a **week 0** of zero-GPU work: build the tokenizers and
v0 mix, write the Tier 0 probes and graders with mutation tests (`lanes/eval.md` D7), and run the self-tests.

---

## What this means for Planck at 10M-150M

1. **Split the curve by machine.** The Mac finds the recipe at 5-20M. The headline curve points (for example 20M / 60M / 150M at
   hundreds of tokens per parameter) need the 5070 (60M @ 6B is ~1.5-2 days there) or the Titans after December (150M @ 50-100B is
   8.5-17 days on the four edge cards, `lanes/compute.md`). Planning a 60M or 150M decision run on the Mac is a mistake: 60M @ 1.2B
   alone takes 4-5.5 days.
2. **The smallest point on the curve is an empirical question the loop answers early (E4).** If the multi-turn likelihood probes show
   no signal at 5M, the curve should start at ~20M. Nothing published measures multi-turn skill below ~90M (`lanes/eval.md`,
   `lanes/probe.md`), so that point is itself a result.
3. **Attention cost shapes the tiny end.** At ≤20M and 2k context, attention is ~half the compute, while the multi-turn test needs 1.5-2k
   token histories (the probe's 12-turn history was 1,726 tokens; an 8k vocab needs ~13% more tokens per conversation than a 49k vocab,
   `lanes/arch.md`). Length bucketing and, on MPS, a fused-attention engine matter more for Planck than for any larger model.
4. **Decisions below ~2% need the continuous metrics and paired seeds.** The loop cannot resolve small generative differences.
   Planck's public claims should rest on large effects, confirmed across two sizes.
5. **Vocabulary is probably the biggest single lever at 5-20M.** The vocab law extrapolates to 2-5k at those sizes, far below any shipped
   tokenizer. Monad (56M) ships with 8,192 (`lanes/data.md`). E7 is where Planck's "own tokenizer" decision gets its evidence.

---

## Ledger ideas

1. **Length-bucketed unpacked batches as the MPS default.** Hypothesis: bucketing by conversation length is at least as fast as packing
   plus a document mask on MPS, and it removes cross-conversation contamination. Why: MPS attention backward is the unfused math path
   (issue #179294), so a mask saves nothing and T² is paid in full. Cheapest test: E1 speed arm plus E6 quality arm at 5M. Win: ≥1.2x
   tok/s at equal Tier 0 recall.
2. **Common random numbers across arms.** Hypothesis: sharing data order (and init where shapes allow) across arms cuts the SD of arm
   differences by ≥2x, halving seeds per arm. Why: Max's 124M A/B training losses correlated at 1.00 under a shared data order. Cheapest test:
   E3's same-data/new-init vs same-init/new-data runs. Win: paired-difference SD at most half the unpaired one.
3. **Log-prob margin as the primary multi-turn metric.** Hypothesis: the Tier 0 margin has ≥5x the signal-to-noise of the Tier 1 pass rate at
   5-20M. Why: bits-per-byte raised MBPP SNR from 2.0 to 41.8 (Heineman et al.). Cheapest test: compute both over E3's seeds and checkpoints.
   Win: margin SNR ≥5x pass-rate SNR, and the two agree on the sign of every E5-E10 decision that clears its MDE.
4. **Small-batch schedule to reach in-context skills sooner.** Hypothesis: 16x2048 tokens/step gets the induction transition and first
   Tier 0 recall signal in ~0.6x the tokens of 64x2048 at 5M. Why: N_PT ≈ 750k·B^0.63·C^0.38 (Aoyama et al.), although larger batches also
   weaken the final induction score. Cheapest test: two 5M runs at 0.15B with dense checkpoints. Win: transition token count within 2x of
   the law, and earlier Tier 0 signal with no final bits-per-byte penalty above MDE.
5. **Multi-token-prediction auxiliary loss for ≤30M models.** Hypothesis: 2-4 MTP heads during training (dropped at inference) improve
   multi-turn recall margins at 5-20M. Why: MTP "vastly improved formation of induction capability" at ≤30M non-embedding parameters (Gloeckle
   et al.). Cheapest test: 5M, MTP=0 vs 2 extra heads (head parameters excluded from the inference budget but reported), 2 seeds. Win: LP-Recall
   margin at 4-8 distractor turns up by ≥ MDE.
6. **Dialogue-shaped MAD unit tests at 1M as a 10-minute pre-screen.** Hypothesis: 2-4 layer, ~1M models trained on synthetic "user states a
   fact / distractor turns / user asks / user corrects" token tasks rank architecture arms the same way the 5M Tier 0 probes do. Why: MAD's
   width-128 tasks rank-correlate with compute-optimal perplexity within an architecture class. Cheapest test: run the E8-E10 arms at 1M on the
   synthetic tasks (~minutes each) and compare with the 5M rankings. Win: ≥80% pairwise agreement, which would let later architecture ideas
   be screened in minutes.
7. **Tiny-LR robustness check for data-mix decisions.** Hypothesis: mix rankings that hold at both the tuned LR and 0.1x LR transfer to 20M;
   rankings that flip between LRs do not. Why: proxies at LR 1e-6 reached ρ > 0.95 with tuned 1B targets vs ρ < 0.75 at a standard LR
   (arXiv 2512.24503; 70M-125M proxies). Cheapest test: the E5 duplicate at 0.1x LR. Win: when the two LRs agree, 20M confirms the ranking.
8. **Vocab-law extrapolation to tiny scale.** Hypothesis: at a fixed total budget the bits-per-byte optimum is ~2-4k at 5M and ~4-8k at 20M.
   Why: Nv ∝ Nnv^0.83 anchored at 16K for 302M non-vocab params (Tao et al.). Cheapest test: E7. Win: the optimum lands within one doubling of the
   prediction at both sizes, and the Tier 0 margin agrees.
9. **Attention-heavy, MLP-light blocks for knowledge-light chat.** Hypothesis: at a fixed total, shrinking the MLP ratio in favour of depth
   or attention improves multi-turn margins, because facts are meant to live outside Planck. Why: MLPs act as key-value memories (Geva et al.,
   https://arxiv.org/abs/2012.14913), and knowledge costs ~2 bits/param (Allen-Zhu & Li). Cheapest test: E8. Win: Tier 0 margin ≥ MDE better
   while dialogue bits-per-byte is ≤1% worse.
10. **Factorized embeddings at tiny scale.** Hypothesis: a rank-48 input/output factorization of an 8k vocab at 5M frees ~1.2M parameters
    (about 24% of the budget) for layers and beats the full tied embedding. Why: at 5M the 8k x 192 embedding is 31% of all parameters. Cheapest
    test: E7 arm 4. Win: bits-per-byte ≥1% better at an equal total.
11. **Looped blocks for information per parameter.** Hypothesis: 2x immediate block reuse raises the multi-turn margin at matched unique
    parameters. Why: MobileLLM-LS +0.7 to +1.1 zero-shot points at 125M/350M. Cheapest test: E10. Win: margin ≥ MDE better; separately
    report whether a param-matched deeper model (E9) gets the same gain without extra compute.
12. **WSD trunk plus decay branches for data and post-training questions.** Hypothesis: branching 20% decays off one shared stable-phase trunk
    ranks data-mix and SFT-style variants like full independent runs, at ~5x lower cost. Why: constant LR plus cooldown matches cosine and
    makes runs reusable (Hägele et al.). Cheapest test: redo E5 as three decay branches off one trunk and compare with the full-run E5 ranking.
    Win: identical ranking within MDE.
13. **MLX as the Mac training engine.** Hypothesis: MLX with fused attention trains 5-20M models at T=2048 ≥1.3x faster than PyTorch MPS. Why:
    MLX was 1.1-1.2x faster even at T=1024 with a large head (first run; Feng et al. on M2), and PyTorch's MPS attention backward is unfused.
    Cheapest test: E1. Win: ≥1.3x at both sizes, with logit parity to the PyTorch reference.
14. **Sequence-length curriculum for tiny models.** Hypothesis: training at T=1024 for 80% of tokens, then at 2048, matches full-2048 Tier 0
    recall at ~0.8x the wall-clock for 5M. Why: attention is 55% of FLOPs at 5M/T=2048 vs 38% at 1024. Cheapest test: two 5M arms.
    Win: equal LP-Recall margin at 8 distractor turns within MDE, and ≥1.2x speed.
15. **Checkpoint averaging for all reported numbers.** Hypothesis: averaging the last 3-5 decay checkpoints (or an EMA) lowers the effective
    seed SD enough to save a seed per arm. Why: checkpoint averaging gave +2.4% decision accuracy (Heineman et al.), and SWA improves the
    trajectory (Hägele et al.). Cheapest test: re-score E3 with and without averaging. Win: SD down by ≥30%.

---

## Open questions

1. **Real Mac throughput and memory for Planck shapes** at T=2048 on PyTorch 2.13/2.14 vs MLX. Everything in section 3 is an estimate
   anchored on one contention-affected bench with a 49k head. E1 answers it in about 2-3 hours of exclusive GPU time.
2. **Has the MPS attention backward been fused since April 2026?** Issue #179294 was open when checked. A fused backward would change the
   bucketing and MLX recommendations.
3. **Planck's seed SD at 5M/20M** for bits-per-byte, the Tier 0 margin and Tier 1 pass rates. The loss-noise figure (0.0015) is from 124M.
4. **Does the induction-transition law hold below 50M,** and does next-token training at ≤30M non-embedding parameters form usable in-context
   recall at all (Gloeckle et al. suggest it forms poorly)? This decides whether 5M can make multi-turn decisions.
5. **Do 5M rankings of architecture changes survive at 20M and 60M?** No published evidence covers architecture transfer from ~5M proxies
   to ~60M-150M targets for dialogue behaviour. The loop will generate it; the 60M check needs the 5070 or the Titans.
6. **When does the 5070 come back?** It is worth ~10 Mac-weeks per week for this workload.
7. **How much Mac time goes to teacher generation vs training** during the next 11 weeks? The two cannot overlap in 24 GB.
8. **Thermals:** does the MacBook Pro sustain the E1 throughput over a 20-hour run, or does it throttle? The logs will show it.
9. The throughput and generation numbers reused from the first run's data lane (5-10M teacher tokens/day on the M5) are derived, not measured.

---

## Sources

Local (read-only):
- `research/lanes/{compute,eval,arch,data,context,training,probe}.md` and their `.verify.md` files
- Memory files referenced for operational rules: `feedback_serialize_model_loads.md`, `reference_launchd_tcc_documents.md`, `reference_mac_ram_parallel_model_loads.md`
- Mac hardware: `system_profiler SPHardwareDataType`, `sysctl hw.memsize`, `sw_vers`

Web:
- PyTorch 2.13 release blog: https://pytorch.org/blog/pytorch-2-13-release-blog/
- MPS env vars: https://docs.pytorch.org/docs/2.14/mps_environment_variables.html ; torch.mps API: https://docs.pytorch.org/docs/2.14/mps.html ; MPS notes: https://docs.pytorch.org/docs/2.13/notes/mps.html
- MPS SDPA causal bug: https://github.com/pytorch/pytorch/issues/195910 ; fix version: https://github.com/multimodal-art-projection/YuE/issues/176
- MPS SDPA backward on math path: https://github.com/pytorch/pytorch/issues/179294
- MPS autocast bf16 / SDPA autocast: https://github.com/pytorch/pytorch/issues/139386 , https://github.com/pytorch/pytorch/issues/141774
- torch.compile on MPS codegen failure example: https://github.com/pytorch/pytorch/issues/152155
- Apple M5 and MLX Neural Accelerators: https://machinelearning.apple.com/research/exploring-llms-mlx-m5 ; https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/
- MLX fast SDPA docs: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html
- Profiling Apple Silicon for ML training (M2 vs NVIDIA): https://arxiv.org/abs/2501.14925
- nanoGPT: https://github.com/karpathy/nanoGPT ; llama2.c: https://github.com/karpathy/llama2.c ; nanochat MLX port: https://github.com/scasella/nanochat-mlx
- llm.c 5080 / 4080 Super: https://github.com/karpathy/llm.c/issues/796 ; https://github.com/karpathy/llm.c/discussions/481#discussioncomment-11005883
- modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt ; 20-run record statistics: https://github.com/KellerJordan/modded-nanogpt/pull/358
- DataDecide: https://arxiv.org/abs/2504.11393 (full text https://arxiv.org/html/2504.11393)
- Signal and Noise: https://arxiv.org/abs/2508.13144
- PolyPythias: https://arxiv.org/abs/2503.09543
- Hitchhiker's Guide to Scaling Law Estimation: https://arxiv.org/abs/2410.11840
- RegMix: https://arxiv.org/abs/2407.01492
- Tiny-LR proxies for data curation: https://arxiv.org/html/2512.24503
- Tay et al., Scaling Laws vs Model Architectures: https://arxiv.org/abs/2207.10551
- 20-modification replication at 1.2B/3B: https://arxiv.org/abs/2605.20798
- MAD: https://arxiv.org/abs/2403.17844
- Downstream scaling reality check: https://arxiv.org/abs/2507.00885
- Induction-head emergence law: https://arxiv.org/abs/2511.16893
- Multi-token prediction: https://arxiv.org/abs/2404.19737
- Fantastic Pretraining Optimizers (Wen et al.): https://arxiv.org/abs/2509.02046
- WSD/cooldown scaling (Hägele et al.): https://arxiv.org/abs/2405.18392
- Tensor Programs V (muP): https://arxiv.org/abs/2203.03466
- Scaling Laws with Vocabulary: https://arxiv.org/abs/2407.13623
- Transformer FFN layers as key-value memories: https://arxiv.org/abs/2012.14913
- TinyDialogues: https://huggingface.co/datasets/styfeng/TinyDialogues ; https://arxiv.org/abs/2408.03617
- ufakzeka-1: https://arxiv.org/abs/2609.25081
