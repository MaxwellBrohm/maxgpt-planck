# Lane: compute feasibility and budgeting on Max's actual hardware

MaxGPT-nano research phase, 2026-09-23. Scope: what a ~150M-parameter chat model costs to train,
distill and ablate on the RTX 5070, the school Lambda box (10x TITAN RTX), the M5 Mac and a Claude Max
subscription, and what the compute picture says about the "wall" below ~400M.

Conventions used throughout:

- **Reference 150M model** (used for every number below unless stated): the Ultra/shakedown block
  design at d_model 768, 18 layers, 12 heads, 4 KV heads, SwiGLU hidden 2048, tied 49,152 vocab.
  That is **151.0M total, 113.3M non-embedding (25% of parameters are the embedding)**. For
  comparison SmolLM2-135M's shape (d 576, 30 layers) with Max's vocab is 134.5M.
- **Training FLOPs per token** = 6N + 12·L·d·T (the PaLM MFU convention, counting full attention).
  For the reference model: **1.076 GFLOP at T=1024, 1.246 GFLOP at T=2048, 1.586 GFLOP at T=4096**
  (attention is 16%, 27%, 43% of the total at those lengths).
- "Healthy Titan" means card 0 on the Lambda box (full clocks, idle neighbours). Every throughput
  number is labelled **measured** (with the file it came from) or **estimate** (with the derivation).

---

## Bottom line

1. The pretraining a 150M model can actually use is affordable: 100B tokens is about **17 days** and
   300B tokens about **7 weeks** on the four sustainable Titan cards, or **~5 weeks / ~4 months** on the
   5070 alone. The binding constraint is not FLOPs but **availability**: every usable Titan is running
   MaxGPT-Ultra until roughly **Dec 3-10 2026**, the 5070 is lent out, and nothing else on the Lambda
   box can take load without throttling Ultra.
2. Brute-force parity with the models that define the "wall" is out of reach: SmolLM2-135M's 2T-token
   pretraining is ~2.5e21 FLOPs (**~11 months** of the 4-card pool), Qwen2.5-0.5B's 18T is ~6e22
   (**~23 years**), Qwen3-0.6B's 36T is ~1.5e23 (**~57 years**). So any win has to come from what each
   token and each parameter is spent on, not from more tokens.
3. Chinchilla-style fits (illustrative, 2022 data) say a 150M at 300B tokens is already within
   **0.07-0.09 nats** of the same model at 2T tokens, while the gap to 0.5-0.6B-class models is
   **~0.34-0.42 nats and mostly the capacity term**. From the compute lens, past ~300B tokens the wall
   is parametric, not a token shortage.
4. Teacher compute is the cheapest high-value compute Max has. Offline distillation from MaxGPT-Ultra
   (the only teacher that shares his tokenizer) over 50B tokens is **~10 days on the 4 edge Titans and
   1.8 TB** with random-sampling KD (36 bytes/token); top-32 logits would need 6.4 TB, which does not
   fit the ~3.1 TB free on the box. Local synthetic multi-turn chat from open-weight teachers is roughly
   **10^9 tokens/week per GPU (estimate)**.
5. The Lambda box cannot run current vLLM or SGLang: driver 470 caps it at CUDA 11.x, the last vLLM
   with a cu118 wheel is **v0.10.1.1** (Aug 2025, same torch 2.7.1 as the box), and SGLang now requires
   CUDA 13. The 5070 runs both. A Claude Max subscription is worth **~10^7 tokens/week realistically**
   (Max's own logs show it already saturated by development work, with hundreds of session-limit hits in
   two recent weeks) and its consumer terms prohibit using the service to train models: seeds, gold
   sets and judging only.
6. The fastest trustworthy loop is not blocked by hardware but by measurement: the A/B harness draws a
   fresh 164k-token eval batch each time, so consecutive evals swing by **0.17-0.21 nats**, larger than
   the effects being tested. With a fixed eval set, a healthy Titan (or the 5070) runs **~40-50 60M
   screening ablations/week** or **~6-8 full 150M/3B-token confirmations/week**; the Mac manages ~1
   18M-proxy per night and is for pipelines and eval development, not decisions.
7. Two zero-compute defects in the current pipeline directly target the stated goal: the SFT packer
   slices conversations at fixed window offsets (later turns lose their history), and the eval noise
   above. Fix both before spending GPU weeks.

---

## Detailed findings

### 1. Measured throughput on the Titans (local, primary evidence)

**1.1 The 124M A/B, one Titan per variant (Sep 21-22 2026).** Source:
`~/Documents/Projects/Max's AI Model/maxgpt-ultra/docs/ab_2026-09-21/*.metrics.jsonl`
(configs in `configs/ab/`, fp16 + dynamic loss scaling, `torch.compile` max-autotune-no-cudagraphs,
micro-batch 16, T=1024, 131,072 tokens/step, 7,629 steps = 1.0B tokens).

| variant (card) | median tok/s | p10-p90 | notes |
|---|---|---|---|
| adamw (card 0, healthy) | **53,725** | 53,456-53,774 | flat across the whole run |
| adamw_arch (card 5) | 46,210 | 44,240-46,631 | + gate / value residual / norm scaling, warmer slot |
| normuon (card 4) | 35,473 | 20,673-37,697 | NorMuon overhead + throttled slot |
| normuon_arch (card 9) | 29,296 | 26,396-37,494 | throttled slot |

The 113.3M model at T=1024 is 0.793 GFLOP/token, so card 0's 53.7k tok/s is **42.6 TFLOPS, 33% of the
130.5 TFLOPS fp16 tensor peak**. The throttled cards (4, 9) and NorMuon's overhead are visible in the
other rows; the card-4/9 runs are not clean optimizer speed measurements.

**1.2 The 1.1B Ultra bench.** `scripts/bench_micro.py` on one Titan: **7,997 tok/s** at micro-batch 2
(source: comment in `configs/ultra_lambda.yaml`, and
`(private project notes)`). At 7.66
GFLOP/token that is **61.3 TFLOPS, 47% of 130.5**. This number also settles the Titan's peak: if the
card had GeForce-style half-rate FP32 accumulation (~65 TFLOPS) it would imply 94% MFU, which is not
physically plausible, so the Titan behaves like the full-rate TU102 part.

**1.3 Titan peak spec.** The Turing whitepaper lists the full TU102 (72 SMs, 576 tensor cores, 1,770
MHz boost, 672 GB/s GDDR6) as the Quadro RTX 6000 at **130.5 TFLOPS FP16 tensor with FP32 accumulate**
(the GeForce 2080 Ti halves that to 53.8/56.9). The TITAN RTX is the same full TU102 configuration but is
not in the whitepaper table, so the 130.5 figure for it is an inference, supported by 1.2.
Source: https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf

**1.4 Estimate for the reference 150M at T=2048 on one healthy Titan: ~34k tok/s (range 28-38k).**
Derivation: same achieved TFLOPS as the 113M run (42.6) divided by 1.246 GFLOP/token. The range
covers Turing's lack of FlashAttention-2 (PyTorch's flash SDPA backend needs sm_80+; sm_75 uses the
memory-efficient backend), which matters more as attention grows to 27% of FLOPs at T=2048.

**1.5 What the multi-card box actually delivers.** From the memory file above and `LAMBDA.md`: only the
bank edges (cards 0, 4, 5, 9) hold sustained load; middle cards (1, 2, 3, 7) run at 15-35% when
sandwiched between loaded cards; cards 6 and 8 have defective cooling (88 C in ~10 s, clocks halved).
Ultra on 0, 4, 5, 9 measured a ceiling of ~20.7k tok/s (sum of solo card speeds) and **16.1k tok/s
achieved** with automatic re-dealing. In healthy-card equivalents that is **2.0 achieved, 2.6
ceiling**. Applied to the 150M: **~68k tok/s (range 55-88k)** for one DDP job on the four edge cards.
For independent ablation jobs (no DDP pacing by the slowest card) the 2.6 figure is the better guide.

**1.6 When the Titans are free.** Ultra: 190,734 steps x 524,288 tokens = 100B tokens at ~16.1k tok/s
= ~72 days from Sep 22 09:43, so pretraining ends **~Dec 3 2026**, plus Ultra's own SFT/DPO and any
thermal pauses: **free around Dec 7-14**. Until then nothing on the GPUs is realistically usable:
loading a middle card heats Ultra's cards (adjacency measured at 15-35% speed), and the watchdog pauses
Ultra on thermal trips. What *is* idle: 64 CPU cores and ~500 GB RAM (the data build tokenized at a
**measured 1.8M tok/s on 64 cores**, 100B tokens in ~16 h), and ~3.1 TB of disk (3.3 TB free at setup
minus the 187 GB shard build). Using the CPUs still needs Max at the keyboard (this lane did not and
will not ssh to the box).

### 2. RTX 5070

**2.1 Spec (primary).** GB205, 48 SMs, 192 5th-gen tensor cores, 2,512 MHz boost, 12 GB GDDR7 at
**672 GB/s** (same as the Titan), 48 MB L2 (vs 6 MB), 250 W TGP. **Dense BF16/FP16 tensor with FP32
accumulate: 61.7 TFLOPS** (123.5 sparse); FP16 with FP16 accumulate 123.5; FP8 with FP32 accumulate
123.5 dense. Source: NVIDIA RTX Blackwell GPU Architecture whitepaper, Table 6 (Appendix C),
https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf

So on paper a 5070 has **less than half the bf16 training peak of one Titan RTX**, with the same memory
bandwidth. It wins on software (bf16 so no loss scaling, FlashAttention-2 under Linux/WSL2, current
CUDA/torch/vLLM/SGLang) and on FP8 (not useful here, see 2.4).

**2.2 Measured evidence that consumer Blackwell runs close to its peak.**
- Max's own MaxGPT-3 on this 5070 (WSL2 + FA2 + `torch.compile`, bf16, 235M params with untied
  16k-vocab head, T=1024, batch 8): `WRITEUP_NOTES.md` logs **~0.23 s/step**, and the checkpoint
  `maxgpt-3/checkpoints/final.pt` records step 2,440,000 at 8,192 tokens/step (20B tokens), consistent
  with the logged "~6-7 days". That is **~35.6k tok/s, ~54 TFLOPS, ~87% of the 61.7 spec**. High but
  not impossible on a half-rate-accumulate card (lower tensor peak makes the memory-bound share smaller).
  Treat as a logged approximation, not a benchmark.
- Public: llm.c GPT-2 124M on an **RTX 5080: 91,886 tok/s** (B=4, T=1024), https://github.com/karpathy/llm.c/issues/796 .
  Against the 5080's 112.6 TFLOPS spec that is ~66-70% MFU. And on an **RTX 4080 Super: 73,666 tok/s**
  (reported as 56.7% bf16 MFU, 10B tokens in 44 h),
  https://github.com/karpathy/llm.c/discussions/481#discussioncomment-11005883 . Both at 124M, 10B-token scale.

**2.3 Estimate for the reference 150M at T=2048 on the 5070: ~32k tok/s (range 24-40k).** Derivation:
61.7 TFLOPS x 50-80% MFU / 1.246 GFLOP. The 49k-vocab head is more memory-bound than MaxGPT-3's 16k
head, so the central value sits below the MaxGPT-3-implied figure. Practical reading: **one 5070 is
about one healthy Titan (±30%) for this workload.** A 10-minute `scripts/bench_micro.py` run on the
5070 would replace this estimate.

**2.4 FP8 is unlikely to help at this size.** torchao's float8 training docs report e2e speedups of up
to 1.25x at 8B/8 GPUs and 1.5x at 405B, state that speedups grow with GEMM size, and that for small
shapes the fp8 path is slower than bf16 on H100 because of cast/scale overhead
(https://docs.pytorch.org/ao/main/workflows/training.html). With d=768 matmuls the expected gain is
~0 or negative (inference, unmeasured on the 5070).

### 3. M5 Mac (measured today, 2026-09-23)

Scripts: `.../scratchpad/compute/mps_bench.py` and `mlx_bench.py` (Llama-style decoder, 49,152 vocab,
T=1024, fwd+bwd+AdamW, no compile). PyTorch 2.13 MPS with bf16 autocast, and MLX 0.32 in bf16:

| proxy | params (non-emb) | MPS bf16 tok/s | MLX bf16 tok/s | effective TFLOPS |
|---|---|---|---|---|
| p18M (d256, L8) | 18.5M (5.9M) | 10,637 | 11,819 | 1.5-1.6 |
| p35M (d384, L10) | 34.6M (15.7M) | 6,906 | 7,951 | 1.8-2.0 |
| p60M (d512, L12) | 60.6M (35.4M) | 3,878 | 4,756 | 1.7-2.1 |
| p113M (A/B shape) | 113.3M (75.5M) | 2,703 | 2,707 | 2.1 |

- A single large bf16 GEMM reaches **13-14 TFLOPS via MPS** (fp32: ~3.2; MLX 0.32 only ~3.8 in bf16),
  consistent with Apple's claim of Neural Accelerators in each GPU core
  (https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/).
  Training stays at ~2 TFLOPS because the M5's **153 GB/s** unified-memory bandwidth (same source) is
  ~4.4x below the GPUs' 672 GB/s, and eager small-model training is bandwidth- and dispatch-bound.
- **The 49k-vocab logits are the memory wall on the Mac too**: p35M at batch 16 x 1024 allocated 24.8 GB,
  swapped and collapsed to 351 tok/s before OOM. Keep micro-batches at ~8k tokens or chunk the loss.
- Net: the Mac is **~20x slower than a healthy Titan** at 113M. An 18.5M proxy at 20 tokens/param
  (370M tokens) takes ~10 h; a 35M proxy ~26 h; a 60M proxy ~3 days. Good for pipeline tests, eval
  harness work, chat-probe scoring of checkpoints and very coarse screens; not for 150M decisions.

### 4. Training budget scenarios

Total training compute for the reference 150M at T=2048: **50B: 6.2e19 FLOPs; 100B: 1.25e20; 300B:
3.7e20; 1T: 1.25e21** (x0.86 at T=1024, x1.27 at T=4096).

| tokens | 5070 (32k, est) | 5070 low (24k) | 1 healthy Titan (34k) | 4 edge Titans DDP (68k achieved) | 4 edge ceiling (88k) | 8 cards if IT fixes cooling (speculative, ~5 eq) |
|---|---|---|---|---|---|---|
| 50B | 18 d | 24 d | 17 d | **8.5 d** | 6.5 d | 3.4 d |
| 100B | 36 d | 48 d | 34 d | **17 d** | 13 d | 6.8 d |
| 300B | 109 d | 145 d | 102 d | **51 d** | 39 d | 20 d |
| 1T | 362 d | 482 d | 339 d | **169 d** | 130 d | 68 d |

The existing build is 100B unique tokens. Muennighoff et al. (up to 9B params / 900B tokens) found up
to **4 epochs of repeated data give negligible loss change vs unique data**
(https://arxiv.org/abs/2305.16264), so the build supports up to ~300-400B token-passes for a 150M model
without a new download.

Home electricity for the 5070 (not in any source; rough): 100B tokens = ~36 days x ~0.35 kW system
draw = ~300 kWh, tens of dollars.

### 5. What the reference small models spent (the brute-force gap)

Token counts (primary): SmolLM2-135M **2T**, SmolLM2-360M **4T**, SmolLM2-1.7B 11T (model cards,
https://huggingface.co/HuggingFaceTB/SmolLM2-135M , https://huggingface.co/HuggingFaceTB/SmolLM2-360M ,
https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B ; 64 / 128 / 256 H100s respectively); Qwen2.5 **18T**
(https://arxiv.org/abs/2412.15115); Qwen3 **36T** (https://arxiv.org/html/2505.09388). Architectures
from each model's `config.json` (fetched today): SmolLM2-135M d576 L30 9H/3KV vocab 49,152 tied;
SmolLM2-360M d960 L32 15H/5KV vocab 49,152; Qwen2.5-0.5B d896 L24 14H/2KV vocab 151,936; Qwen3-0.6B d1024
L28 16H/8KV vocab 151,936.

| model | tokens | tokens/param | training FLOPs (6N+12LdT, T=2048) | time on 4 edge Titans (~85 TFLOPS eff.) |
|---|---|---|---|---|
| reference 150M @ 100B | 0.1T | 660 | 1.25e20 | 17 days |
| reference 150M @ 300B | 0.3T | 2,000 | 3.7e20 | 51 days |
| SmolLM2-135M | 2T | 14,900 | 2.5e21 | ~11 months |
| SmolLM2-360M | 4T | 11,000 | 1.2e22 | ~4.4 years |
| Qwen2.5-0.5B | 18T | 36,000 | 6.3e22 | ~23 years |
| Qwen3-0.6B | 36T | 60,000 | 1.5e23 | ~57 years |

Qwen3's report adds that its small models were built by "leveraging the knowledge from the flagship
models" (distillation), https://arxiv.org/abs/2505.09388 . The sub-1B "chat threshold" models are
therefore both heavily over-trained and distilled from much larger teachers, which Max cannot replicate
by compute.

### 6. Teacher-based methods: costs

**6.1 Tokenizer is the first constraint.** Logit distillation needs the teacher's vocabulary.
- **MaxGPT-Ultra (1.1B) is the only available teacher with Max's exact tokenizer**, available ~Dec.
- SmolLM2 models have the same vocab *size* (49,152) but their own merges; using them as logit teachers
  means switching the student to the SmolLM2 tokenizer and re-tokenizing the corpus (~16 h on the
  Lambda CPUs at the measured 1.8M tok/s; SFT/pref sets tokenize at load time, minutes).
- Qwen3's 151,936 vocab at d=768 is **116.7M embedding parameters, 77% of a 150M budget**. Qwen teachers
  are only usable for sequence-level distillation (generate text, train on it) or cross-tokenizer KD.

**6.2 Forward-pass cost of extracting logits** (estimate: forward FLOPs = 2N + 4·L·d_q·T at T=2048;
Titan at 45% of 130.5 TFLOPS, 5070 at 65% of 61.7; small teachers will be below these MFUs):

| teacher | fwd GFLOP/token | healthy Titan | 4 edge Titans (2.6 eq) | 5070 | 10B tokens (4 Titans / 5070) | 50B on 4 Titans |
|---|---|---|---|---|---|---|
| MaxGPT-Ultra 1.1B (same vocab) | 2.56 | 23k tok/s | 60k | 16k | 1.9 d / 7.4 d | **9.7 d** |
| SmolLM2-360M (retokenize) | 0.98 | 60k | 157k | 41k | 0.7 d / 2.8 d | 3.7 d |
| SmolLM2-1.7B (retokenize) | 3.82 | 15k | 40k | 10.5k | 2.9 d / 11 d | 14.5 d |
| Qwen2.5-0.5B (seq-level only) | 1.16 | 50k | 131k | 34k | 0.9 d / 3.4 d | 4.4 d |
| Qwen3-1.7B (seq-level only) | 3.91 | 15k | 39k | 10k | 3.0 d / 11 d | 15 d |
| Qwen3-4B (seq-level only) | 9.25 | 6.4k | 16.5k | 4.3k | 7.0 d / 27 d | 35 d |

The Ultra row is anchored to a measurement (its training at 7,997 tok/s x3 for forward-only gives ~24k).

**6.3 Storage.** With a 49,152 vocab a token id fits in uint16:
- Top-k with fp16 probabilities = 4 bytes per entry: **top-8 32 B/token, top-32 128 B, top-64 256 B.**
  Top-32 over 50B tokens = 6.4 TB, over 100B = 12.8 TB: does not fit ~3.1 TB free.
- **Random-Sampling KD** (Anshumann et al., https://arxiv.org/abs/2503.16870 ; students 300M-3B, 10B-100B
  tokens): top-K caching gives biased targets and needed ~K=300 to approach full KD, while ~12 sampled
  tokens per position at 3 bytes each (their 17-bit id + 7-bit prob) are unbiased, **~10% slower than
  plain CE** and 1.7-2.6x faster than full KD; they quote 3.6 TB per 100B tokens. For Max: **36 B/token =
  0.36 TB per 10B, 1.8 TB per 50B, 3.6 TB per 100B.** 50B fits on the box; 100B does not.
- Full logits (49,152 x 2 bytes = 98 KB/token) are impossible at any useful scale.

**6.4 Online vs offline.** Online KD (teacher forward inside the training step) multiplies per-token
compute by (1.246 + teacher fwd) / 1.246: **Ultra 3.05x, SmolLM2-360M 1.78x, SmolLM2-1.7B 4.07x.**
Offline RS-KD pays the teacher once and reuses it for every student and ablation; the distillation
scaling-law paper (students 143M-12.6B, up to 512B tokens, https://arxiv.org/abs/2502.08606) finds that
when a teacher already exists, distillation beats supervised training up to a student-size-dependent
token budget, and that "making a teacher too capable eventually reduces the student performance".
A 1.1B or 360M teacher is plausibly closer to the right size for a 150M student than a 4B (speculative
for the exact optimum).

**6.5 Synthetic data generation: software on each machine.**
- **Lambda (driver 470 = CUDA 11.x only).** NVIDIA's minor-version compatibility table: 11.x needs
  >=450, 12.x needs >=525, 13.x needs >=580
  (https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). The forward-compat package does
  not apply: "Forward Compatibility is applicable only for systems with: NVIDIA Data Center GPUs. Select
  NGC Server Ready SKUs of RTX cards. Jetson boards."
  (https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html). The last PyTorch cu118
  wheel is **2.7.1** (https://download.pytorch.org/whl/cu118/torch/ ; current is 2.14.0 on cu126).
- **vLLM:** GitHub release assets show cu118 wheels through **v0.10.1.1 (2025-08-20)** and none after
  (https://github.com/vllm-project/vllm/releases). v0.10.1.1 pins torch==2.7.1 (requirements/cuda.txt at
  that tag), the same torch as the Lambda venv. In that version sm_75 runs on the V0 engine: V1 support for
  compute capability < 8.0 arrived in v0.10.2 (PR #23614); V0 was removed in v0.11.0; v0.12.0 moved to
  PyTorch 2.9 / CUDA 12.9. Qwen3 is supported from v0.8.4; Qwen3.5 only from v0.17.0. Current vLLM
  (v0.30.0, 2026-09-22) still adds Turing kernels (Marlin for sm75 in v0.14, SM75 modelopt_mixed in
  v0.24), so **a driver upgrade by IT (>=525) would unlock current vLLM on the Titans**; without it,
  vLLM 0.10.1.1 in a separate venv, fp16 or int8, is the ceiling. fp16 inference of bf16-trained Qwen
  models can overflow (risk, not measured here).
- **SGLang** now "requires CUDA 13" (0.5.19 was the last CUDA 12 release), and FlashInfer needs sm75+
  (https://github.com/sgl-project/sglang/blob/main/docs/docs/get-started/install.mdx). Unusable on Lambda
  as configured; fine on the 5070.
- **Throughput (roofline estimate, unsourced; real engines typically reach 50-80% of it):** batched
  decode with ~1,200-token live contexts, 80% of memory bandwidth:

| GPU | teacher | est. tok/s | est. tokens/week (24/7) |
|---|---|---|---|
| 5070 | Qwen3-1.7B bf16 / fp8 | 2,300 / 2,900 | 1.4-1.8B |
| 5070 | Qwen3-4B bf16 / fp8 | 450 / 1,600 (KV-cache-limited in 12 GB) | 0.3-1.0B |
| 5070 | MaxGPT-Ultra 1.1B | ~7,000 | ~4B |
| Titan | Qwen3-4B fp16 / int8 | 1,750 / 2,300 | 1.1-1.4B |
| Titan | MaxGPT-Ultra 1.1B | ~8,000 | ~4.9B |

  Order of magnitude: **~10^9 synthetic tokens/week per GPU** for 1-4B teachers.

**6.6 Claude Max subscription.**
- Official: a session limit that resets every five hours plus "a weekly usage limit that applies
  across all models"; Max 20x is 20x Pro's per-session allowance; **no token numbers are published**
  (https://support.claude.com/en/articles/11049741-what-is-the-max-plan).
- Usage measurements from Max's own account are omitted from the public copy. Practical
  estimate for bulk generation through a subscription: about **10^7 tokens/week**, 10-100x less
  than one local GPU, and it would starve every other project.
- Terms: the Consumer Terms (effective Oct 8 2025) forbid using the Services "To develop any products
  or services that compete with our Services, including to develop or train any artificial intelligence
  or machine learning algorithms or models" (https://www.anthropic.com/legal/consumer-terms). Bulk
  Claude-generated training data is a policy risk as well as a throughput dead end. Use Claude for
  designing prompts, rubrics, probe sets, code, and reading results; use Apache-2.0 open-weight teachers
  (Qwen3 and SmolLM2 are Apache-2.0 per their Hugging Face model tags) for bulk generation.

### 7. Reusing the existing data build

- **Pretraining shards: reusable as-is.** `data/prepare.py` writes one flat uint16 token stream per shard
  with `<|endoftext|>` separators; `PackedShardDataset(path, seq_len)` chooses the context length at load
  time, so a 150M can train at 1024, 2048 or 4096 on the same 1,001 shards (187 GB). Held-out shard
  handling (`scripts/holdout_shard.py`) carries over.
- **Mixture is baked in.** Documents from all five sources are interleaved inside every shard (the
  mixed stream samples per document), and only aggregate per-source counts are recorded in meta.json.
  Re-weighting (for example cutting the 10% code and 8% math, or adding dialogue) needs a rebuild:
  **~16 h on the Lambda CPUs** at the measured 1.8M tok/s, plus download time.
- **Vocab cost at 150M:** 49,152 x 768 = 37.7M parameters (25%). SmolLM2-135M uses the same vocab size at
  d=576 (28.3M, 21%), so this is a proven size at this scale. A smaller vocab would free 10-25M parameters
  but would require a new tokenizer and full re-tokenization and would **break Ultra as a same-vocab
  teacher**. That trade-off belongs in the architecture lane; compute-wise, re-tokenization is cheap.
- **SFT (108,607 rows) and preference (60,000 pairs) sets:** stored as chat jsonl, tokenized at load,
  so they are reusable with this or any tokenizer. **But the SFT packer is wrong for multi-turn
  coherence:** `posttrain/sft_data.py` (lines 40-78) concatenates all conversations into one stream and
  slices fixed `seq_len` windows at fixed offsets, with no conversation-boundary alignment and no
  attention masking between packed conversations. Any conversation that crosses a window boundary has
  its later assistant turns trained without the earlier turns they depend on. The mechanism is certain
  from the code; the size of the effect is unmeasured. The fix (whole conversations per window, bin-packed
  or padded, document masking) costs no compute.

### 8. The experiment loop

**8.1 What the A/B logs reveal.** `eval/harness.py::eval_perplexity` evaluates 20 batches x 8 x 1024 =
**163,840 tokens pulled sequentially from the validation stream, so every eval sees different tokens.**
From the four metrics files (computed today):
- Standard deviation of the change between consecutive evals (last 40% of training): **0.17-0.21 nats.**
- The normuon vs normuon_arch pair happened to see identical eval batches at each step: paired
  difference 0.064 nats with **SD 0.007** across 23 evals. Every pair involving adamw or adamw_arch did
  not (paired SD 0.18-0.20).
- So the 0.064-nat architecture gain on top of NorMuon is clean, while the headline "6.9% vs AdamW"
  (0.207 nats) is directionally right but carries roughly ±0.1 nats of eval-sampling noise in the
  "mean of last 3" statistic. Any 150M decision smaller than ~0.1 nats is unreadable with the current
  eval. **Fix: a fixed eval token set (same windows, every eval, every variant), ~2-4M tokens**, plus a
  chat set.
- The logs also show the ranking flip Max noted (NorMuon behind at step 4000, ahead after decay), which
  matches Wen et al. (0.1B-1.2B, 1-8x Chinchilla): "comparing intermediate checkpoints before reaching
  the target training budgets can be misleading", and matrix-preconditioned optimizers' speedup is 1.4x
  over AdamW at 0.1B but 1.1x at 1.2B (https://arxiv.org/abs/2509.02046). **Compare at the end of decay.**

**8.2 Chat-specific eval that is cheap enough for every ablation (forward-only):**
- Assistant-turn loss on held-out multi-turn conversations.
- A history-dependence score: loss on turn k with the full history minus loss with the history removed.
  If the second number is not clearly worse, the model is not using context.
- Likelihood versions of Max's existing scripted probes (`maxgpt-3/eval_multiturn.py`: memory,
  reference, continuity, adaptation, correction): score log p(correct continuation) vs a distractor
  given the history. DataDecide found continuous likelihood metrics at small scale far more predictive
  than accuracy (https://arxiv.org/abs/2504.11393).
  All three cost seconds per checkpoint on any machine, including the Mac.

**8.3 Proxy sizes and tokens per ablation.**
- DataDecide (models up to 1B, up to 100B tokens, 3 seeds): rankings at a single 150M scale predict the
  best 1B data recipe in ~80% of comparisons. For decisions *about* a 150M model, running at 150M for
  confirmation is the direct answer; screening can happen smaller.
- Screening proxy: ~60M (d512, L12, same 49k vocab; note its embedding share is 42%, so its behaviour is
  more embedding-dominated than the 150M's), **1.2-2.4B tokens (20-40 tokens/param) including a full
  decay.** Confirmation: reference 150M at **3B tokens** (20 tokens/param), or 10B for effects that only
  show when over-trained.
- For data-mix, annealing and post-training questions, use WSD's structure: one shared stable-phase
  trunk, then a short decay branch per variant (each branch ~10-20% of the trunk's tokens).
- Seed noise at these sizes is unmeasured in Max's setup (unsourced); measure 2-3 seeds of one config
  once to know the minimum detectable effect.

**8.4 Ablations per week (24/7; estimates from sections 1-3):**

| machine | 60M @ 1.2B tokens | 60M @ 2.4B | 150M @ 3B | 150M @ 10B |
|---|---|---|---|---|
| 1 healthy Titan (60M at ~75-100k tok/s est.) | 38-50 | 19-25 | 5.6-7.7 | 1.7-2.3 |
| 4 edge Titans, independent jobs (2.6 eq) | 98-131 | 49-66 | 15-20 | 4.4-6.0 |
| 5070 (about 1 healthy Titan) | ~35-50 | ~18-25 | ~5-8 | ~1.5-2.5 |
| M5 Mac (measured) | 60M: ~77 h per run; 35M @ 0.7B: ~26 h; 18.5M @ 0.37B: ~10 h, so ~1 per night | | | |

### 9. Scenario table with calendar time

| # | scenario | hardware | earliest start | what it buys | calendar |
|---|---|---|---|---|---|
| A | now to Ultra's end | M5 Mac (+ Lambda CPUs by Max's hand) | now | eval harness (fixed set, chat probes, history score), SFT packer fix, teacher-gen scripts tested on tiny models, 18-35M proxies (~1/night), re-mixed data builds on CPU | Sep 23 to ~Dec 7 (~11 weeks) |
| B | 5070 returns while Ultra runs | 1x 5070 | unknown (lent to the video-editor project) | 60M screens (~40/wk) or one 150M @ 50B (~18 d) / 100B (~36 d); or ~1-2B tokens/week of synthetic multi-turn chat with Qwen3-1.7B/4B via current vLLM or SGLang | +18-36 d per 150M run |
| C | post-Ultra pretrain | 4 edge Titans (DDP) | ~Dec 7-14 2026 | 150M @ 100B | ~17 d, done ~late Dec |
| C2 | same, 3 epochs | 4 edge Titans | ~Dec 7-14 | 150M @ 300B | ~51 d, done ~early Feb 2027 |
| D | distilled variant | 4 edge Titans | ~Dec 7-14 | RS-KD logits from Ultra over 50B tokens (~10 d, 1.8 TB), then 150M @ 100B with KD (~17 d + ~10%) | ~29 d, done ~mid Jan 2027 |
| E | C/D plus 5070 in parallel | 4 Titans + 5070 | when both free | Titans pretrain; 5070 runs ablations or generates ~1B synthetic chat tokens/week | parallel |
| F | IT applies power cap / fixes fans | 6-8 Titans | unknown | ~2x on C/D (speculative; the chassis may still limit) | 100B in ~7-9 d |
| G | parity with SmolLM2-135M's budget | 4 edge Titans | - | 2T tokens | ~11 months: not viable |

---

## What this implies for a ~150M chat model

1. **Pretraining compute is not the bottleneck; availability and measurement are.** 100-300B tokens
   (660-2,000 tokens/param) is reachable in 2-7 weeks once the Titans free up, and the existing build
   covers it with up to 3-4 epochs. There is no compute case for a bigger pretraining budget until an
   experiment shows chat metrics still improving past ~300B.
2. **The next ~11 weeks should build the measuring instrument, not a model:** fixed eval set, chat
   likelihood probes, history-dependence score, SFT packing fix, and teacher-generation tooling. All of
   it runs on the Mac. Without the fixed eval, any 150M ablation under ~0.1 nats is noise.
3. **Spend compute where small models are known to be compute-favourable:** distillation with an
   existing teacher (favourable at modest student budgets, per the distillation scaling law) and
   optimizer quality (Muon-family gains are largest at ~0.1B). Both are affordable: RS-KD over 50B tokens
   is ~10 days and fits on disk; NorMuon is already implemented and measured.
4. **Synthetic multi-turn data must come from local open-weight teachers, not Claude:** ~10^9 vs ~10^7
   tokens/week, and the consumer terms. The 5070 is the right machine for it (current vLLM/SGLang, fp8);
   the Titans need an IT driver upgrade to run current serving stacks.
5. **Train long contexts in the phase that needs them.** Multi-turn coherence needs 2-4k-token
   windows; at T=4096 the reference model costs 1.27x its T=2048 FLOPs (1.47x its T=1024 FLOPs), which
   is cheap for SFT (~100M tokens, hours) and for a final pretraining phase, expensive only if used for
   all of pretraining. On the Titans, long context runs on the memory-efficient attention backend.

## Evidence about the "wall" (compute lens)

- **The models that chat at 400-600M were trained with ~25-63x more compute than SmolLM2-135M and far
  beyond anything Max can match.** Qwen2.5-0.5B: 18T tokens (~6.3e22 FLOPs); Qwen3-0.6B: 36T (~1.5e23)
  and distilled from flagship models (Qwen3 report). SmolLM2-135M/360M: 2T/4T (~2.5e21/1.2e22). Part of
  the apparent size threshold may be where labs chose to spend compute and distillation effort, not only
  a capacity limit (inference from the sources above, not a measured result).
- **Scaling-law decomposition (illustrative only; fits from Chinchilla's 2022 data, loss is not chat
  quality):** with Hoffmann et al.'s fit and Epoch AI's refit (https://arxiv.org/abs/2404.10102), the
  150M's parameter term is 0.68-0.69 nats vs 0.43-0.46 for 0.5-0.6B models. Its data term is 0.22/0.13
  (two fits) at 300B tokens and 0.13/0.07 at 2T. So 300B vs 2T is worth only 0.07-0.09 nats at 150M,
  while the gap to Qwen2.5-0.5B at 18T is ~0.34-0.38 nats (0.37-0.42 to Qwen3-0.6B at 36T), of which ~0.23 is the parameter term.
  A 150M cannot buy its way to 0.5B-class loss with tokens; it has to change what its capacity is
  spent on.
- **Max's own small-model failures are confounded by under-training.** MaxGPT-2 (110M, 482M unique
  tokens, 4 epochs, ~18 tokens/param) produced low-entropy loops and confident nonsense; MaxGPT-3 (235M,
  4.6B unique, 20B passes, ~85 tokens/param). SmolLM2-135M saw ~14,900 tokens/param. Those failures
  cannot separate a size wall from a 100-1000x data shortfall (`WRITEUP_NOTES.md`).
- **Small models get more from better optimization:** matrix optimizers 1.4x at 0.1B vs 1.1x at 1.2B
  (Wen et al.); Max's measured 124M A/B agrees in direction.
- **Distillation is favourable exactly in Max's regime** (modest student token budget, existing
  teacher) and has a capacity gap (too-strong teachers hurt): Busbridge et al., students 143M-12.6B.
- **Training signal for the target skill is currently damaged** by SFT window slicing (section 7), so
  past multi-turn failures also include a pipeline component.

## Levers

| lever | limit attacked | expected effect | evidence | cost |
|---|---|---|---|---|
| Fixed eval set + paired, end-of-decay comparisons | measurement noise (0.17-0.21 nats per eval) | makes 0.01-0.05-nat effects readable | strong (Max's own logs) | hours of code, ~0 compute |
| Chat likelihood probes + history-dependence score | no metric for the target skill | fast, deterministic signal on multi-turn use of context | moderate (DataDecide on likelihood metrics; the history score is untested) | ~0 compute |
| SFT packing by whole conversation + document masking | later turns trained without their history | directly targets multi-turn coherence; size unknown | mechanism certain, effect unmeasured | ~0 compute |
| Right-size pretraining to 100-300B tokens (reuse build, <=3-4 epochs) | wasted compute beyond data saturation | within ~0.1 nats of a 2T-token run per fits | moderate (Muennighoff; fits are old) | 17-51 days on 4 Titans |
| Offline RS-KD from MaxGPT-Ultra (same vocab) | sample efficiency per token | unknown at 150M; favourable regime per distillation laws | moderate (RS-KD at 300M-3B; distillation laws 143M+) | ~10 d teacher pass + 1.8 TB per 50B, ~10% train overhead |
| Local synthetic multi-turn chat (Qwen3-1.7B/4B, SmolLM2-1.7B) on the 5070 | lack of dialogue data in a web/textbook mix | ~10^9 tokens/week to reshape SFT and late pretraining toward conversation | weak-moderate (throughput is a roofline estimate) | 5070 time; driver upgrade needed on Titans |
| NorMuon + arch tweaks (already built) | optimizer efficiency | 0.06-0.2 nats at 124M/1B tokens (measured, partly noisy); larger at small scale per Wen et al. | strong | none (already in code) |
| WSD trunk + decay branches for ablations | ablation cost | ~5-10x more data/post-training ablations per GPU-week | moderate | scheduling only |
| Re-mixed build (less code/math, more dialogue) | capacity spent on off-target data | unknown; plausible for a narrow chat target | speculative | ~16 h CPU re-tokenization + downloads |
| Ask IT for driver >=525 (or >=580) and per-GPU power caps | stack frozen at CUDA 11.8; only 4 usable cards | current vLLM on Titans; possibly ~2x cards | strong for the software part, speculative for thermals | an email |
| Claude for seeds, rubrics, gold probe sets, reading outputs | quality of small curated sets | high value per token | n/a (policy + measured saturation) | subscription share |

## Open questions

1. Real 5070 throughput for the reference 150M at T=2048 with the 49k vocab (the 24-40k tok/s range, and
   whether MaxGPT-3's implied ~87% MFU is real). One `bench_micro.py` run settles it.
2. When does the 5070 come back from the video-editor project, and when exactly does Ultra (plus its
   SFT/DPO) release the Titans?
3. Will the school IT upgrade the driver (unlocks current vLLM, torch 2.14) and apply per-GPU power caps (more
   usable cards)? The email to the professor was sent Sep 21.
4. Is fp16 inference of bf16-trained Qwen teachers numerically safe on Turing for long generations?
5. Seed-to-seed noise of final loss and of the chat probes at 60M and 150M in Max's setup.
6. How good is Ultra as a teacher (it is a 1.1B at ~91 tokens/param) versus SmolLM2-1.7B at 11T tokens
   with a tokenizer switch, and where is the capacity-gap optimum for a 150M student?
7. Do chat metrics (history-dependence, probe likelihoods) keep improving past ~300B pretraining tokens
   at 150M, or do they saturate with loss? Measurable with WSD decay branches at 50B / 100B / 300B.
8. Is the Lambda disk budget (~3.1 TB free before Ultra's checkpoints) enough for 50B tokens of RS-KD
   logits plus a second shard build?
9. How Anthropic's consumer-terms clause applies to a personal, non-commercial student model (this lane
   treats it as a risk and routes bulk data to open-weight teachers).

## Sources

Local (read-only):
- `~/Documents/Projects/Max's AI Model/maxgpt-ultra/docs/ab_2026-09-21/README.md` and `*.metrics.jsonl`
- `~/Documents/Projects/Max's AI Model/maxgpt-ultra/LAMBDA.md`, `RUNBOOK.md`, `PLAN.md`, `configs/*.yaml`, `configs/ab/*.yaml`
- `~/Documents/Projects/Max's AI Model/maxgpt-ultra/eval/harness.py` (eval_perplexity), `posttrain/sft_data.py` (SFTDataset), `data/prepare.py`
- `~/Documents/Projects/Max's AI Model/WRITEUP_NOTES.md`; `maxgpt-3/config.py`, `maxgpt-3/checkpoints/final.pt` (step metadata), `maxgpt-3/eval_multiturn.py`
- `(private project notes)`
- Mac benchmarks: `(scratch file)`, `mlx_bench.py`
- Claude usage aggregation: `.../scratchpad/compute/usage.py` over `(private workflow logs)`

Web:
- NVIDIA RTX Blackwell whitepaper: https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
- NVIDIA Turing whitepaper: https://images.nvidia.com/aem-dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf
- CUDA driver minimums: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html ; forward compat: https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html
- PyTorch cu118 wheels: https://download.pytorch.org/whl/cu118/torch/
- vLLM releases and assets: https://github.com/vllm-project/vllm/releases ; PR #23614: https://github.com/vllm-project/vllm/pull/23614
- SGLang install doc: https://github.com/sgl-project/sglang/blob/main/docs/docs/get-started/install.mdx
- llm.c RTX 5080: https://github.com/karpathy/llm.c/issues/796 ; RTX 4080 Super: https://github.com/karpathy/llm.c/discussions/481#discussioncomment-11005883
- torchao float8: https://docs.pytorch.org/ao/main/workflows/training.html
- Apple M5: https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/
- SmolLM2 model cards and configs: https://huggingface.co/HuggingFaceTB/SmolLM2-135M , -360M, -1.7B ; paper https://arxiv.org/abs/2502.02737
- Qwen2.5 report: https://arxiv.org/abs/2412.15115 ; Qwen3 report: https://arxiv.org/abs/2505.09388 ; configs at https://huggingface.co/Qwen/Qwen2.5-0.5B , https://huggingface.co/Qwen/Qwen3-0.6B , -1.7B, -4B
- Sparse Logit Sampling / RS-KD: https://arxiv.org/abs/2503.16870
- Distillation Scaling Laws: https://arxiv.org/abs/2502.08606
- Scaling Data-Constrained LMs: https://arxiv.org/abs/2305.16264
- DataDecide: https://arxiv.org/abs/2504.11393
- Fantastic Pretraining Optimizers: https://arxiv.org/abs/2509.02046
- Chinchilla replication (fits): https://arxiv.org/abs/2404.10102
- Claude Max plan: https://support.claude.com/en/articles/11049741-what-is-the-max-plan ; API pricing: https://platform.claude.com/docs/en/about-claude/pricing ; Consumer Terms: https://www.anthropic.com/legal/consumer-terms
