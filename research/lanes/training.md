# Lane: Training, optimization, scaling laws and distillation for 100M-250M models

Research date: 2026-09-23. Scope: what the published training literature says a ~150M-parameter chat model can and cannot reach, and which training-side levers could move it. Every number carries a source URL; numbers I computed myself are marked **[calc]** with the inputs stated; anything I could not source is marked **unsourced**. Evidence measured only at 1B or above is flagged **[>=1B only]**.

---

## 1. Bottom line

1. The premise needs a correction: sub-400M chat models already exist and ship (SmolLM2-135M-Instruct and 360M-Instruct, Gemma 3 270M-it, LFM2-350M, Baguettotron 321M with multi-turn support), but every one of them except Baguettotron was trained on 2T to 15T tokens (11k to 41k tokens per parameter), usually with big-lab distillation, and none of them is known to hold good multi-turn coherence at the 135M-150M end. The open gap is "a ~150M model that chats well", not "a sub-400M model that chats at all".
2. From the training lane, the wall is mostly two hard limits, not one: (a) **knowledge capacity**, about 2 bits per parameter at best (Allen-Zhu and Li), which caps a 150M model at roughly 300M bits, around 2% of what the same authors estimate English Wikipedia plus textbooks contain; and (b) a **parameter-limited loss floor**: under a fitted supervised scaling law, a 120M-non-embedding model trained on infinite data still sits about 0.18 nats above SmolLM2-360M at 4T tokens [calc], so no token budget closes the gap on a broad web distribution.
3. More tokens still help at this size (a 150M model kept improving up to 10,000 tokens/param with no plateau, just slower than the laws predict), but returns are small and Max's realistic budget after Ultra finishes is about 300B to 1T tokens (2k to 7k tokens/param) [calc], well short of the 2T to 36T the incumbents used.
4. Logit distillation in *pretraining* is the most over-hyped lever for this size: at 125M-200M two controlled studies found zero gain (MobileLLM with a 7B teacher at matched tokens; MiniPLM with a 1.8B teacher at matched compute), Busbridge et al. show supervised learning wins given enough tokens and that too-strong teachers hurt, and my evaluation of their fitted law gives a 150M student about 0.02-0.04 nats token-matched and roughly zero compute-matched from a 1.1B teacher [calc]. The best-supported teacher is only about 2-3x the student.
5. Distillation looks much better in *post-training*, where the chat skill is actually formed: MiniLLM's on-policy reverse-KL distillation lifted a 120M GPT-2 student from 38.6 to 44.7 on a GPT-4-judged instruction benchmark, and on-policy methods directly target the exposure-bias and drift failures Max saw (low-entropy loops) and that multi-turn chat amplifies.
6. Max's measured 124M A/B gain (6.9% lower loss) is real but was measured at about 10 tokens/param with one shared learning rate; independent benchmarks put the best matrix optimizers at about 1.4x over a well-tuned AdamW at 0.1B, and the optimizer ranking shifts at high data ratios, so it must be re-validated in the regime a 150M chat model actually trains in.
7. Pruning Ultra (1.1B) down to 150M is a 7.3x compression, beyond anything with clean published evidence; the only same-scale result (MobileLLM-350M to 125M) is mixed. The concrete training-side opportunity is: spend parameters on skills rather than facts (retrieval for facts), train the 150M hard on a narrower, chat-shaped distribution, and put the distillation budget into on-policy post-training with a same-tokenizer teacher.

---

## 2. Detailed findings

### 2.1 What the incumbents actually did (tokens per parameter)

Architecture numbers are from each model's `config.json`, fetched directly on 2026-09-23. Tokens/param are **[calc]** from the stated token counts.

| Model | Total params | d_model / layers | Vocab (embedding params) | Pretraining tokens | Tokens/param [calc] | Source |
|---|---|---|---|---|---|---|
| SmolLM2-135M | 134.5M | 576 / 30, GQA 9/3 | 49,152 (28.3M, tied) | 2T | ~14.9k | https://arxiv.org/abs/2502.02737, https://huggingface.co/HuggingFaceTB/SmolLM2-135M |
| SmolLM2-360M | ~362M | 960 / 32 | 49,152 (47.2M) | 4T | ~11k | same |
| MobileLLM-125M | 125M | deep-thin | 32k (LLaMA-2 tok.) | 1T ("480k iterations on 1T tokens") | ~8k | https://arxiv.org/abs/2402.14905 |
| MobileLLM-R1-140M | 140M | 576 / 15 | 128k (LLaMA-3.2 tok.) | 2 phases x 2T per paper Table 5 (card/abstract: 4.2T resampled from ~2T unique) | ~29k | https://arxiv.org/abs/2509.24945, https://huggingface.co/facebook/MobileLLM-R1-140M |
| Gemma 3 270M | ~270M (170M embedding, 100M transformer) | 640 / 18 | 262,144 | 6T | ~22k total, ~60k per transformer param | https://huggingface.co/google/gemma-3-270m-it, https://developers.googleblog.com/en/introducing-gemma-3-270m/ |
| Baguettotron | 321M | 576 / **80** | 65,536 | 200B fully synthetic (SYNTH) | ~620 | https://huggingface.co/PleIAs/Baguettotron |
| Monad | 56M | 256 / 64 | 8,192 | 200B synthetic | ~3.6k | https://huggingface.co/PleIAs/Monad |
| LFM2-350M | ~350M (hybrid conv+GQA) | 1024 / 16 | 65,536 | 10T, KD from LFM1-7B | ~28k | https://huggingface.co/LiquidAI/LFM2-350M |
| Granite-4.0-350M | ~350M | 1024 / 28 | 100,352 | ~15T (10+2+2+0.5) | ~41k | https://huggingface.co/ibm-granite/granite-4.0-350m-base |
| Qwen2.5-0.5B | 0.49B (0.36B non-emb) | 896 / 24 | 151,936 | 18T | ~36k | https://huggingface.co/Qwen/Qwen2.5-0.5B, https://arxiv.org/abs/2412.15115 |
| Qwen3-0.6B | 0.6B (0.44B non-emb) | 1024 / 28 | 151,936 | 36T | ~60k | https://huggingface.co/Qwen/Qwen3-0.6B, https://arxiv.org/abs/2505.09388 |
| MaxGPT-Ultra (Max's) | ~1.1B | 2048 / 22 | 49,152 (100.7M) | 100B | ~91 | local `maxgpt-ultra/configs/ultra.yaml` |

Takeaways:
- The industry sub-500M models sit at **11k to 60k tokens/param**. SmolLM2-135M, the closest analog to Max's target, is at ~15k.
- Embedding share varies wildly: 21% of SmolLM2-135M, 62% of Gemma 3 270M [calc from configs]. A "150M" model with a 49k vocab and d=576-768 spends 28M-38M params (19-25%) on the embedding table [calc].
- Deep-thin is the consensus shape at this size (SmolLM2-135M 30 layers at d=576, Baguettotron 80 layers at d=576). PleIAs report "consistent improvements from stacking more layers" on dense synthetic data (card, above; no controlled numbers given).

Instruct results as published (each vendor's own harness, so cross-row comparisons are loose):
- SmolLM2-135M-Instruct IFEval 29.9; SmolLM2-360M-Instruct IFEval 41.0, MT-Bench 3.66; Qwen2.5-0.5B-Instruct IFEval 31.6, MT-Bench 4.16 (HF's harness). https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct. The 135M card lists MT-Bench "19.8", which appears to be on a different scale from the 360M card's 3.66; I could not confirm the scale (**unsourced interpretation**).
- Gemma 3 270M-it IFEval 51.2 (0-shot). https://huggingface.co/google/gemma-3-270m-it. Google's own framing: the model "is not designed for complex conversational use cases" (https://developers.googleblog.com/en/introducing-gemma-3-270m/).
- MobileLLM-125M MT-Bench 2.33, AlpacaEval win 24.07%; MobileLLM-350M MT-Bench 3.28, AlpacaEval 47.08%, versus Pythia-160M at 1.01 / 0.63% (MobileLLM paper Table 5, https://arxiv.org/abs/2402.14905).
- MobileLLM-R1-140M is explicitly "not general-purpose chat" (card). LFM2's card claims suitability for "multi-turn conversations" (card, no multi-turn score given). Baguettotron's card says it "has support for multi-turn"; Monad's says "no support yet for multi-turn".

### 2.2 Overtraining: does quality keep improving at 5k-15k tokens/param?

- **Loss and downstream keep improving to 10,000 tokens/param at 150M, with no plateau, but slower than laws predict.** Sardana et al. trained 47 models from 150M to 6B; only the 150M was pushed to 10,000 tokens/param (1.5T tokens). "Loss does not plateau as we scale to 10,000 tokens per parameter for our 150M model"; the Gauntlet average also kept improving; but "none of our parametric curves fit our 150M long-ratio training results well", and models "continue to learn at extreme training durations, [but] do so more slowly than scaling laws predict." Their 150M is MPT-style, d=768, 12 layers, ALiBi. https://arxiv.org/abs/2401.00448 (scale: 150M, up to 1.5T tokens). This is the single most relevant data point for Max's target and it says "keep going, diminishing returns".
- **Descriptive SmolLM data:** SmolLM-135M (600B) to SmolLM2-135M (2T) moved base HellaSwag 41.2 to 42.1, ARC 42.4 to 43.9, MMLU-cloze 30.2 to 31.5, and instruct IFEval 17.2 to 29.9, but data and post-training also changed, so this is not a token-only ablation. https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct (scale: 135M, 0.6T vs 2T).
- **Low-budget 135M reference point:** L20-Edu-135M (SmolLM2 shape, 13B tokens on one L20 GPU) scored a six-task mean of 0.4150 vs SmolLM2-135M 0.4917 in the author's own harness; "it remains clearly behind modern 135M models trained with 46x to 154x more tokens". It also reports that GRPO-style RLVR on GSM8K *decreased* accuracy (1.82% to 1.59%) at 135M. https://arxiv.org/abs/2606.22189 (scale: 134.5M, 13B tokens; single run).
- **Saturation in Pythia (evidence of a small-model ceiling, contested):** Godey et al. find Pythia models up to 410M show in-domain loss rising late in training; final checkpoints underperform a scaling-law extrapolation by 8% on average (best checkpoints by ~4%); Pythia-160M best vs final: LAMBADA acc 40.3 vs 38.0, SciQ 79.6 vs 73.4, ARC-e 46.5 vs 43.2. They tie it to the softmax bottleneck: "models based on less than 1000 hidden dimensions tend to adopt degenerate latent representations in late pretraining", and a rank-constrained LM head hurts perplexity once its rank is below ~1000 regardless of model size. https://arxiv.org/abs/2404.07647 (scale: Pythia 14M-410M, 300B tokens). Caveat: Sardana's d=768 150M did not saturate at 10k tokens/param, and SmolLM2-135M (d=576) kept improving to 2T, so this looks like a schedule/data-interacting effect plus a real but bounded head-rank penalty, not a hard stop.
- **Overtrained models get brittle to fine-tuning ("catastrophic overtraining").** Instruction-tuned OLMo-1B pretrained on 3T tokens is "over 2% worse" on standard benchmarks than its 2.3T checkpoint; the effect appears above ~2.5T tokens for OLMo-1B (~2.1k tokens/param [calc]) and was not seen for OLMo-7B up to 3T. Controlled runs at 15M-90M on 4B-128B tokens show sensitivity to parameter changes rising monotonically with pretraining tokens, worse at higher fine-tuning learning rates. https://arxiv.org/abs/2503.19206. Caveat: SmolLM2-135M at ~15k tokens/param was still successfully instruction-tuned, so this is a "post-train gently" warning, not a ceiling.
- **Overtraining also raises post-training-quantization damage**: "the degradation introduced by post-training quantization increases as models are trained on more data" (up to 1.7B params, 26B tokens). https://arxiv.org/abs/2411.04330. Relevant if the 150M ships as int4/int8.
- **Inference-optimal framing** (Sardana, above): with ~1B inference requests expected, "train models smaller and longer than Chinchilla-optimal". For a local chat model this is exactly Max's situation, so the case for a small overtrained model is sound; the question is only how far returns go.
- **Data repetition is cheap up to ~4 epochs**: "training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data" (up to 9B params, 900B tokens). https://arxiv.org/abs/2305.16264. Max's 100B-token pre-tokenized build can therefore support roughly a 300-400B-token run for a 150M model without new data.

### 2.3 Knowledge capacity: what 150M can store versus what open-domain chat asks for

From Physics of Language Models Part 3.3 (Allen-Zhu and Li), synthetic biography knowledge, GPT-2-with-RoPE and LLaMA/Mistral variants, https://arxiv.org/abs/2404.05405:
- "Language models can and only can store 2 bits of knowledge per parameter, even when quantized to int8" after ~1,000 exposures per fact; no model exceeded a capacity ratio of 2.3.
- With 100 exposures (rare facts), capacity falls to ~1 bit/param. In that regime LLaMA/Mistral (gated MLP) are 1.3x worse than GPT-2's plain MLP.
- int4 quantization drops capacity to 0.7 bit/param. MoE with 32 experts loses only 1.3x.
- Junk data: at a 1:7 useful-to-junk token ratio, useful-knowledge capacity drops 20x (100 exposures); prepending a source/domain token to useful data recovers it to only 2x loss.
- Their estimate: English Wikipedia (4.5B words) plus non-overlapping English textbooks (<16B words) contain fewer than 14B bits of knowledge.
- Param counting note: they exclude unused embedding rows (GPT-2 small counts as 88M in their setup), so the ratio is roughly per non-embedding-dominated parameter.

**[calc] What this means at 150M:** best case 150M x 2 = 300M bits (~37.5 MB of facts), about 2.1% of their <14B-bit Wikipedia+textbooks estimate. For facts seen ~100 times with a SwiGLU (gated) MLP: ~150M x 1 / 1.3 = ~115M bits, under 1% of that estimate. A separate measurement (Morris et al.) puts GPT-family memorization capacity at about 3.6 bits/param, which raises the ceiling by <2x and does not change the conclusion. https://arxiv.org/abs/2505.24832.

This matches Max's own logs: MaxGPT-2 (110M) "knows the SHAPE of an explanation but not the content" ("confident nonsense" on photosynthesis), local `WRITEUP_NOTES.md`. Open-domain chat implicitly promises world knowledge; a 150M model structurally cannot hold it, and more tokens do not help once each fact has ~1,000 exposures (capacity is a parameter property).

### 2.4 The parameter-limited loss floor (why tokens alone cannot close the gap)

**[calc]** Using the supervised scaling law fitted by Busbridge et al. (Table 6: L = E + (A/N^a + B/D^b)^g with E=1.220, A=3355, B=18186, a=0.408, b=0.431, g=0.452; N = non-embedding params; trained on English C4 with repetition at large budgets; https://arxiv.org/abs/2502.08606), absolute values are C4-specific, only differences matter:

| Model (non-embedding params, tokens) | Predicted loss |
|---|---|
| 150M-class (120M), 100B tokens | 2.596 |
| 150M-class (120M), 300B | 2.557 |
| 150M-class (120M), 2T | 2.520 |
| 150M-class (120M), infinite data | **2.492** |
| SmolLM2-135M-like (106M), 2T | 2.549 |
| SmolLM2-360M-like (315M), 4T | **2.311** |
| Qwen2.5-0.5B-like (358M), 18T | 2.273 |
| MaxGPT-Ultra-like (1.0B), 100B | 2.240 |

Reading: the 150M's infinite-data asymptote is ~0.18 nats above a 360M model at 4T, and 100B to 2T tokens only buys ~0.08 nats. Sardana's finding that 150M learns *slower* than such laws predict at extreme ratios makes this gap an underestimate if anything. On a broad web distribution, the "wall" in loss terms is real and parameter-shaped.

Connecting loss to abilities: Du et al. find that models "with the same pre-training loss, but different model and data sizes, generate the same performance" and that emergent abilities appear only once pretraining loss falls below a task-specific threshold. https://arxiv.org/abs/2403.15796 (scale: 300M to 32B, 33B-500B tokens; **no evidence below 300M**). If that holds at 150M, the lever is not "more params" per se but "lower loss on the distribution that chat needs". A narrower, chat-shaped distribution has a much lower achievable loss for a fixed parameter count (the TinyStories result: models under 10M params, even a single transformer block, write "fluent and consistent stories" on a narrow child-vocabulary distribution; https://arxiv.org/abs/2305.07759). That is the main opening this lane sees.

### 2.5 Optimizers at small scale, and how to read Max's 124M A/B

- **Independent benchmark, well-tuned baselines:** Wen et al. benchmark ten optimizers at 0.1B-1.2B and 1x-8x Chinchilla: matrix-based optimizers' speedup over AdamW "is inversely proportional to model scale, decreasing from 1.4x over AdamW for 0.1B parameter models to merely 1.1x for 1.2B"; "many reported speedups of 2x simply reflect a weak baseline"; tuning one hyperparameter of the GPT-3 recipe alone can give 2x; Muon is best at low data ratios but "is outperformed by Kron and Soap when the data-to-model ratio increases to 8x or larger"; rankings "can flip during training due to learning rate decay". https://arxiv.org/abs/2509.02046 (scale: 130M, 300M, 520M, 1.2B; up to 8x, some 16x Chinchilla).
- Semenov et al. find AdEMAMix consistently strong vs AdamW as iterations increase, with D-Muon and MARS also reliable (scale: 124M-720M). https://arxiv.org/abs/2509.01440. AdEMAMix's own headline: a 1.3B model on 101B tokens matches AdamW on 197B, and it "significantly slows-down model forgetting". https://arxiv.org/abs/2409.03137 **[>=1B only]**.
- NorMuon: 21.74% better training efficiency than AdamW and 11.31% over Muon at 1.1B; 13.91% over AdamW at 5.4B. https://arxiv.org/abs/2510.05491 **[>=1B only]**. Moonlight: Muon "~2x computational efficiency compared to AdamW with compute optimal training". https://arxiv.org/abs/2502.16982 (compute-optimal regime, not overtrained).
- IMU-1's 70M ablation (1.4B tokens, ~20 tokens/param): NorMuon vs AdamW -0.094 loss (-2.88%), cautious WD a further -0.031 (-0.97%), total 5.21% relative; architecture set (QK-norm, per-head gating, normalized value residual, LayerNorm scaling) -1.64% combined, with per-head gating alone only -0.002. IMU-1 (430M, 72B tokens) reaches avg 0.574 with EMA vs SmolLM2-360M's 0.586 (4T tokens). https://arxiv.org/abs/2602.02522.
- **Max's A/B** (local `maxgpt-ultra/docs/ab_2026-09-21/README.md`): 113M-param model (37.7M of it embedding [calc]), 1.1B tokens (~10 tokens/param), one LR (6e-4) shared by all four arms. Optimizer-only gain 5.1% (2.993 to 2.840), architecture-only 6.3% (to 2.803), both 6.9% (to 2.786); NorMuon arms trailed until the LR decay.
  - The optimizer gain matches IMU-1's 70M ablation (5.21%), which is reassuring.
  - The architecture gain is ~4x IMU-1's measured 1.64% for a similar set, which suggests the plain-AdamW arm was not at its own best LR.
  - **[calc]** Under the Busbridge supervised-law shape at 75M non-embedding / 1.1B tokens, a 5.1% loss drop corresponds to ~2.5x more tokens and 6.9% to ~3.8x; Wen et al.'s tuned-baseline ceiling at 0.1B is ~1.4x. So expect the true, tuned, high-tokens/param gain at 150M to be a fraction of 6.9% (plausibly 1-3% loss), and expect SOAP/Kron to be competitive at the 100x+ Chinchilla ratios a chat model trains at. This is inference, not measurement.

### 2.6 Schedules, batch size, weight decay, parameterization, stability

- **WSD / cooldown:** constant LR plus cooldown matches cosine and allows reusing runs across lengths; stochastic weight averaging "yields improved performance along the training trajectory, without additional training costs". https://arxiv.org/abs/2405.18392.
- **Decay to zero, benefit grows with tokens/param:** linear decay-to-zero "consistently outperforms other schedules", "benefits increase as dataset size increases"; a 610M model at 80 tokens/param with D2Z beats 200 tokens/param with 10x decay (60% compute saving). https://arxiv.org/abs/2502.15938. Max's schedule already decays to true zero (prior sweep, local `docs/research_2026-09-22.md`). For a heavily overtrained 150M this matters more than it did for Ultra.
- **Weight decay must scale with run length:** optimal AdamW timescale tau = B/(eta*lambda*D) follows a power law in tokens/param, and optimal lambda scales linearly with batch size. https://arxiv.org/abs/2505.13738. Wang and Aitchison: the optimal EMA timescale in epochs is roughly constant, so weight decay should *decrease* as dataset size grows. https://arxiv.org/abs/2405.13698. Implication: a 150M run at 2,000+ tokens/param should not reuse Ultra's weight decay unchanged.
- **Batch size:** critical batch size "scales primarily with data size rather than model size" (85M-1.2B). https://arxiv.org/abs/2410.21676. Small batches "achieve equal or better per-FLOP performance", and gradient accumulation is not recommended on a single device if Adam's beta2 half-life is held fixed in tokens. https://arxiv.org/abs/2507.07101. For Max: a small batch is fine early and on one card; a ramp to larger batch later in a long run is consistent with CBS growing with D.
- **muP / depth:** CompleteP gives depth-wise hyperparameter transfer and claims "12-34% compute efficiency improvements over the prior state-of-the-art", specifically helping deep-thin shapes. https://arxiv.org/abs/2505.01618. IMU-1 also used muP. At 150M Max can afford direct LR sweeps, so muP is a convenience (tune on 20-40M proxies) rather than a requirement.
- **Checkpoint EMA:** +0.014 benchmark average at 430M (IMU-1, above); the prior sweep also recorded a negative 1.5B result. Cheap to try, small.
- **Stability:** nothing in the literature flags 150M as unstable; QK-norm and z-loss (already in Max's recipe) cover the known failure modes. The HF Smol playbook's Muon divergence warning was at 3B (prior sweep).

### 2.7 Knowledge distillation for pretraining small students

**Busbridge et al., "Distillation Scaling Laws" (Apple, ICML 2025)**, students and teachers 143M to 12.6B, up to 512B tokens, English C4. https://arxiv.org/abs/2502.08606
- "Supervised learning always outperforms distillation given enough student compute or tokens"; distillation wins only below a compute threshold that grows with student size, and only if the teacher already exists or is reused. "Smaller models are more likely to benefit from supervised pretraining, whereas larger models are more likely to benefit from distillation" (compute-optimal analysis).
- Capacity gap: student loss follows a broken power law in teacher loss; "making a teacher too capable eventually reduces the student performance". Optimal teacher size in their compute-optimal scenario grows "until it is slightly larger than the student".
- Counterpoint inside the same paper: at 20 tokens/param "weaker students benefit more from distillation... the 198M student has all observed data below [the supervised] line".
- Practicalities measured: pure distillation (lambda=1) and temperature tau=1 are best; top-k=128 truncation costs 0.11 nats and top-p=0.9 costs 0.13 nats vs full distribution, but k=128 mixed with the ground-truth loss (lambda=0.7) is within 0.01 nats; forward KL beat reverse KL by 0.28 nats in pretraining (1.82B to 546M).

**[calc] Evaluating their fitted distillation law (Eq. 8, Table 6 coefficients) for a 150M student (120M non-embedding).** Compute-matched assumes student training at 6N FLOPs/token plus teacher forward at 2N FLOPs/token. All teacher losses come from their supervised law, so these are law predictions on C4, not results in the paper, and 120M/1T is outside the fitted range (>=143M students, <=512B tokens):

| Teacher (non-emb params, tokens) | Student tokens | Distilled minus supervised, token-matched | Compute-matched |
|---|---|---|---|
| MaxGPT-Ultra-like (1.0B, 100B) | 30B / 100B / 300B / 1T | -0.073 / -0.042 / -0.026 / -0.018 nats | -0.005 / +0.001 / +0.001 / -0.001 |
| SmolLM2-360M-like (315M, 4T) | same | -0.070 / -0.042 / -0.029 / -0.020 | -0.033 / -0.019 / -0.014 / -0.012 |
| SmolLM2-1.7B-like (1.61B, 11T) | same | -0.074 / -0.027 / -0.003 / **+0.013** | +0.008 / +0.024 / +0.030 / +0.033 |

Readings: token-matched, distilling from an Ultra-like teacher at 100B tokens predicts the loss that supervised training reaches at ~330B tokens (~3.3x token efficiency); compute-matched it is a wash, because a 1.1B teacher's forward pass costs ~2.4x the student's own training step. A teacher about 2-3x the student gives the best compute-matched edge (0.01-0.02 nats). A strong 1.7B teacher shows the capacity gap: it becomes *worse* than no distillation beyond ~300B student tokens.

**Controlled null results at Max's scale:**
- MobileLLM (Meta): LLaMA-v2 7B teacher (same 32k tokenizer) into 125M and 350M students, 120k iterations (the paper's 0.25T-token exploratory setting), same tokens for both arms: 125M avg 43.9 with labels vs 43.8 with labels+KD; 350M 49.1 vs 48.8; "training time using KD is 2.6 - 3.2x slower" (93h vs 29h for 125M). https://arxiv.org/abs/2402.14905 (Table 16, Appendix G).
- MiniPLM: Qwen-1.5 1.8B teacher into a 200M student at equal training compute: no-KD 39.9 avg, vanilla KD 39.9, MiniLLM 39.0, SeqKD 39.7; "when the training FLOPs are controlled, all KD methods perform similar or worse than Pre-Train w/o KD". At 500M, vanilla KD 43.6 vs 43.2; at 1.2B, 45.4 vs 44.9. https://arxiv.org/abs/2410.17215 (scale: 200M-1.2B students, 50B-token budget).
- Pre-training distillation design space (GLM-4-9B into 1.9B): "larger student LLMs generally benefiting more from pre-training distillation, while a larger teacher LLM does not necessarily guarantee better results". https://arxiv.org/abs/2410.16215 **[>=1B only]**.

**Positive results, mostly bigger or less controlled:**
- Gemma 2: 2B student, 500B tokens (10x compute-optimal), 7B teacher: 3-benchmark average 60.3 from scratch vs 67.7 distilled; validation perplexity at 200M: 23 from scratch vs 21 distilled from 7B (token budget for the 200M row not stated). https://arxiv.org/abs/2408.00118. Note Google distills with a teacher already trained for other purposes, so the teacher is free in their accounting.
- Gemma 3 samples 256 teacher logits per token weighted by teacher probability; reports that a smaller teacher is better for short training horizons and a larger one for long horizons (relative perplexity differences under ~0.6% read from the figure axis; student size not stated). https://arxiv.org/abs/2503.19786. The Gemma 3 270M card does not state its training objective, so "Gemma 3 270M was distilled" is **unconfirmed**.
- LFM2-350M: KD from LFM1-7B, 10T tokens (card, no ablation). Llama 3.2 1B: pruned from 8B, logits from 8B and 70B used "as token-level targets", "up to 9 trillion tokens". https://huggingface.co/meta-llama/Llama-3.2-1B **[>=1B only]**.
- Capacity-gap law: "the optimal teacher consistently scales linearly with the student scale". https://arxiv.org/abs/2311.07052. (The often-quoted "2.5x" ratio is not in the abstract; **unsourced** from what I read.)

**Offline alternatives that avoid storing logits:**
- MiniPLM's difference sampling (reweight the pretraining corpus by teacher log-prob minus a small reference model's log-prob, then train normally): +1.4 avg at 200M compute-matched (41.3 vs 39.9), "2.2x" compute reduction for a 500M student, data demand reduced 2.4x, works across tokenizers/families, needs only a scalar per document. https://arxiv.org/abs/2410.17215.
- Random-sampling KD (importance-sampled sparse logits) fixes top-k bias at <10% overhead, tested 300M-3B. https://arxiv.org/abs/2503.16870. A 2026 paper shows offline top-100 caching matches online KD for a 3.2B student (no small-student results). https://arxiv.org/abs/2608.03796 **[>=1B only]**.

**[calc] Storage and compute costs with Max's 49,152 vocab:** a token index fits in uint16, so one cached (index, fp16 prob) entry is 4 bytes. Full distributions: 98 KB/token, ~9.8 PB per 100B tokens (infeasible). Top-128: 512 B/token, 51 TB per 100B tokens (infeasible for pretraining), 2.6 TB for a 5B-token decay slice, ~51 GB for a ~100M-token SFT set (easy; assumes Max's 108k SFT rows average ~1k tokens, which I did not measure). Online KD from a 1.1B teacher costs ~2.2 GFLOP/token of teacher forward vs ~0.9 GFLOP/token for training a 150M student, ~3.4x total.

**Tokenizer fact that matters:** Max's 49,152 BPE is his own (`maxgpt-ultra/tokenizer/tokenizer.py`), so the only existing same-tokenizer teacher is MaxGPT-Ultra itself. SmolLM2-135M, 360M and 1.7B share one identical 49,152-token vocab (I hashed both `tokenizer.json` vocabularies: identical). Adopting the SmolLM2 tokenizer would unlock same-tokenizer logit KD from SmolLM2-360M (a ~2.4x teacher trained on 4T tokens) and SmolLM2-1.7B (11T), at the cost of re-tokenizing the corpus and of the project-integrity question the prior sweep raised.

### 2.8 Distillation in post-training (where the chat skill is formed)

- MiniLLM (reverse KL with on-policy optimization, instruction-following setting): GPT-2 120M student, GPT-2 1.5B teacher, DollyEval GPT-4-feedback score: SFT 38.6, word-level KD 40.3, SeqKD 41.2, MiniLLM 44.7 (Rouge-L 23.3 to 24.6); student quality rose monotonically with teacher size from 340M to 1.5B for a 120M student (no capacity gap in that range). Claims "lower exposure bias, better calibration, and higher long-text generation performance". https://arxiv.org/abs/2306.08543 (scale: 120M-13B students; single-turn instructions).
- GKD (on-policy distillation from student-generated sequences) beats supervised KD and SeqKD for T5-small (77M) through T5-large on summarization, translation and arithmetic. https://arxiv.org/abs/2306.13649.
- Qwen3 builds Qwen3-0.6B/1.7B/4B/8B/14B with "strong-to-weak distillation": off-policy distillation, then on-policy distillation aligning the student's logits to Qwen3-32B or 235B-A22B; on Qwen3-8B, on-policy distillation beat RL (AIME'24 74.4 vs 67.6) at ~1/10 of the GPU hours (1,800 vs 17,920). https://arxiv.org/abs/2505.09388 **[>=1B only for the ablation]**.
- Why this lane cares: multi-turn failures are dominated by unreliability, not aptitude, even in frontier models (average 39% drop from single- to multi-turn across six tasks; "a minor loss in aptitude and a significant increase in unreliability"). https://arxiv.org/abs/2505.06120 (large models only). Unreliability compounding over turns is an exposure-bias problem, which on-policy training attacks directly.

### 2.9 Pruning, weight inheritance, and "shrink Ultra"

- Minitron: Nemotron-4 15B to 8B and 4B (2-4x) with <3% of original data, "up to 40x fewer training tokens per model" (94B for the 4B), up to 16% MMLU over from-scratch; best practices include "prefer width pruning over depth", retrain with "distillation loss using KLD", and "prune a model closest to the target size". https://arxiv.org/abs/2407.14679 **[>=1B only]**.
- Sheared-LLaMA: LLaMA2-7B to 1.3B (5.2x) and 2.7B with 50B tokens, beating OPT-1.3B and Pythia-1.4B trained on 300B, at "3% of compute" vs from scratch. https://arxiv.org/abs/2310.06694 **[>=1B only]**.
- Llama 3.2 1B: one-shot structured pruning from 8B (~6.5x) plus logit KD, yet still pretrained on up to 9T tokens, so pruning did not remove the need for a huge token budget at that ratio. https://huggingface.co/meta-llama/Llama-3.2-1B.
- LLM-Pruner: structural pruning with LoRA recovery "in merely 3 hours, requiring only 50K data". https://arxiv.org/abs/2305.11627 (7B-class, mild ratios).
- **The only result at Max's scale:** Adapt-Pruner prunes MobileLLM-350M to 125M (2.8x) with 3.87B tokens: BBH 13.75 vs original MobileLLM-125M's 18.45 (worse), TruthfulQA 36.18 vs 32.88, AGIEval 31.34 vs 31.19, MMLU 25.20 vs 24.65 (near chance). The paper's own summary says it recovers performance "for models larger than 350M". Its training mix includes SFT data (OpenO1-SFT), which confounds the comparison with base models. Qwen2.5-0.5B to 350M reached MMLU 38.48 vs MobileLLM-350M's 26.11. https://arxiv.org/abs/2502.03460 (scale: 125M-1B).
- Weight inheritance: Inheritune initializes a smaller model from the first layers of a larger one and reports a 16-layer GPT-2-medium variant matching the 24-layer original (scale: GPT-2 family). https://arxiv.org/abs/2404.08634. Weight selection (uniform element sampling from a larger model's tensors) improves small models "with no extra cost" (mostly vision evidence). https://arxiv.org/abs/2311.18823. PanGu-pi Pro reports parameter inheritance among several tricks for a +8.87 average at 1B. https://arxiv.org/abs/2402.02791 **[>=1B only]**.
- **[calc] Ultra to 150M** is a 7.3x cut: d_model 2048 to ~576-768 and 22 layers to 16-30 (the deep-thin target shape has *more* layers than Ultra, so depth cannot come from pruning at all). No published result covers this ratio at this scale. Minitron's own "prune closest to target" rule argues against it.

### 2.10 Max's compute envelope for a 150M run

**[calc]** from Max's own measurements (local `maxgpt-ultra/LAMBDA.md`): a single TITAN RTX with idle neighbours trained the 124M A/B model at ~53k tokens/s (8-18k when sandwiched between loaded cards). Scaling to 150M and 4-6 spaced cards gives roughly 150-250k tokens/s, i.e. 100B tokens in ~5-8 days, 1T in ~2-2.5 months, 2T (SmolLM2-135M's budget) in ~3-5 months. Those cards are committed to Ultra for ~70 more days. Realistic first real run: 300B-1T tokens (2k-7k tokens/param), which is below every incumbent in section 2.1. The Mac (24 GB) suits 10M-50M proxy runs only.

---

## 3. What this implies for a ~150M chat model

1. **Do not try to win on world knowledge.** At <=300M bits of storable facts, open-domain factual chat is structurally out of reach. Design the model's job as conversation skill (following, referencing, tracking, staying on topic, admitting ignorance) and route facts to retrieval, which Max's Ultra stack already has (`rag/`). This frees parameters and makes "confident nonsense" a data/format problem instead of a capacity problem.
2. **Narrow the distribution the parameters are spent on.** The loss floor that blocks a 150M model is measured on broad web text. The emergence-by-loss result (>=300M evidence) plus TinyStories suggests that a 150M model can reach "big-model" loss on a narrower, chat-shaped distribution. Concretely: a late-phase anneal heavily weighted to multi-turn dialogue and to text whose value is procedural rather than factual. This is the most plausible route to "break the wall" from this lane, and it is currently untested at 150M.
3. **Train long, but know the returns.** Target 300B-1T tokens, reusing the 100B build for up to ~3-4 epochs, with WSD and linear decay to zero, weight decay re-derived for the run length, and an anneal on the best chat data. Expect loss gains of roughly 0.04 nats per 3x tokens around 100B-300B [calc], shrinking beyond.
4. **Distill where it pays:** on-policy or top-k-cached KD in SFT/preference training (cheap: ~51 GB of cached top-128 logits for the SFT set), and possibly logit KD only during the decay phase. Skip full-run online logit KD from Ultra: it roughly triples compute for a predicted wash.
5. **If using logit KD at all, pick a teacher about 2-3x the student**, and prefer same-tokenizer pairs. Two options: train a ~400M same-tokenizer model (costly), or switch to the SmolLM2 tokenizer and use SmolLM2-360M(-Instruct) as teacher.
6. **Post-train gently.** Heavily overtrained small models are more sensitive to fine-tuning and quantization: low SFT/DPO learning rates, pretraining-data replay, and checkpoint or weight interpolation (L20-Edu used SFT weight interpolation for regression control; https://arxiv.org/abs/2606.22189).
7. **Re-run the optimizer A/B in the right regime** before the big run: 150M, >=100 tokens/param, a small LR sweep per arm, AdamW vs NorMuon (plus SOAP if affordable).

---

## 4. Evidence about the "wall" (why chat quality degrades below ~400M), from this lane

- **Hard limit, knowledge:** capacity ~2 bits/param at 1,000 exposures, ~1 bit/param for rarer facts, 1.3x worse for gated MLPs at low exposure, 20x worse with junk-heavy data unless source-tagged (https://arxiv.org/abs/2404.05405). A 150M model holds <=~300M bits [calc]. Chat users probe arbitrary facts, so small models hallucinate. Max observed exactly this at 110M and 235M.
- **Hard limit, loss floor:** in a fitted scaling law the parameter term dominates at small N; a 120M-non-embedding model's infinite-data loss is ~0.18 nats above a 360M model at 4T [calc from https://arxiv.org/abs/2502.08606]. 150M models also learn more slowly at extreme ratios than laws predict (https://arxiv.org/abs/2401.00448).
- **Mechanistic contributor, output head rank:** at d<1000 the LM head's rank limits achievable perplexity regardless of representation quality, and Pythia <=410M showed late-training degeneration (https://arxiv.org/abs/2404.07647). A 150M model essentially always has d<1000.
- **Budget, not a law:** every competitive sub-400M chat model used 11k-60k tokens/param plus large-teacher distillation (table 2.1). Some of the observed wall is simply that nobody has spent a 2026 recipe with a small budget on a chat-focused 150M; IMU-1 shows a 2026 recipe reaching SmolLM2-360M-like averages on 56x fewer tokens at 430M (https://arxiv.org/abs/2602.02522).
- **Brittleness:** overtraining raises sensitivity to fine-tuning (OLMo-1B above ~2.5T tokens; controlled 15M-90M runs; https://arxiv.org/abs/2503.19206) and to PTQ (https://arxiv.org/abs/2411.04330), so the chat stage itself can damage a well-pretrained small model.
- **Compounding error across turns:** multi-turn loss is mostly unreliability, not aptitude (https://arxiv.org/abs/2505.06120, large models only), and small models have higher per-token entropy and the low-entropy loop attractors Max documented. Off-policy SFT does not train on the model's own error states; on-policy distillation does (MiniLLM at 120M).
- **Counter-evidence that the wall is not a skill wall:** instruction following exists well below 400M (Gemma 3 270M-it IFEval 51.2 with only ~100M transformer params; SmolLM2-360M-Instruct IFEval 41.0 vs Qwen2.5-0.5B-Instruct 31.6 on HF's harness). In-context copying circuits (induction heads) form even in small attention-only models (https://arxiv.org/abs/2209.11895), and <10M models write consistent multi-paragraph stories on narrow data (https://arxiv.org/abs/2305.07759). Remembering what was said turns ago is mechanistically cheap; knowing things and staying reliable are the expensive parts.

---

## 5. Levers

| Lever | Limit attacked | Expected effect at ~150M | Evidence strength | Cost |
|---|---|---|---|---|
| Retrieval for facts, and training the model to use context and to say "I don't know" | Knowledge capacity | Removes the dominant hallucination source; unmeasured at 150M | Strong for the limit, speculative for the chat gain | Low (RAG exists in Ultra repo); needs SFT data showing grounded answers |
| Chat-shaped late anneal / narrow-distribution training | Parameter-limited loss floor | Unknown; the main "break the wall" bet from this lane | Weak (emergence-by-loss >=300M; TinyStories analogy) | Low-medium: data curation plus a decay-phase run |
| On-policy distillation (GKD / MiniLLM-style) in post-training, same-tokenizer teacher | Exposure bias, multi-turn unreliability | MiniLLM 120M: +6.1 GPT-4-score points over SFT (38.6 to 44.7), single-turn | Moderate | Teacher forward on student samples; small data volume |
| Cached top-k (k~128, lambda~0.7) logit KD on the SFT set | Weak supervision in SFT | Within 0.01 nats of full KD in Busbridge's pretraining test; SFT gain at 150M unmeasured | Moderate | ~51 GB for ~100M SFT tokens [calc] |
| Logit KD during pretraining from a 2-3x teacher | Token efficiency | Token-matched -0.02 to -0.04 nats at 100B-1T; compute-matched -0.01 to -0.02 [calc] | Moderate (law extrapolated; null results at 125M-200M) | 1.8x compute with a 360M teacher; needs a same-tokenizer teacher |
| Logit KD from Ultra (1.1B) over the full run | Token efficiency | ~3.3x token efficiency, ~0 compute-matched [calc] | Moderate against | ~3.4x compute |
| MiniPLM difference sampling with Ultra (teacher) and the 124M shakedown model (reference) | Data quality per token | +1.4 avg at 200M compute-matched; 2.2x compute reduction at 500M | Moderate (one paper, Pile, 50B tokens) | Teacher plus reference forward over a candidate pool; e.g. ~1.1e20 FLOPs for 50B tokens [calc] |
| Longer training (300B to 1T) with D2Z linear decay | Undertraining | ~0.04 nats per 3x tokens near 100B-300B [calc]; no plateau to 10k tokens/param | Strong for loss, weak for chat | ~2-2.5 months of 4-6 Titans for 1T [calc] |
| Re-tuned optimizer (NorMuon / SOAP) validated at high tokens/param | Optimization efficiency | ~1.1-1.4x compute-equivalent (Wen et al. at 0.1B), not the 2.5-3.8x token-equivalent the 124M A/B implies [calc] | Moderate | A few multi-day A/B runs |
| Weight decay and batch re-derived for run length (tau rule, CBS grows with D) | Mis-set hyperparameters on long runs | Small but compounding; avoids a silent loss | Moderate | Nearly free |
| Source/domain tag tokens on knowledge-dense data | Junk dilution of capacity | 20x capacity loss reduced to 2x in synthetic tests | Weak for real data (synthetic bioS only) | Free at data build |
| Gentle post-training (low LR, replay, weight interpolation) | Overtraining brittleness | Prevents a >2% post-SFT regression (OLMo-1B case) | Moderate | Free |
| Prune SmolLM2-360M to 150M plus KD (requires SmolLM2 tokenizer) | Budget | Mixed at 350M to 125M (Adapt-Pruner) | Weak | Medium; project-integrity trade-off |
| Prune Ultra (1.1B) to 150M | Budget | Unknown at 7.3x | Speculative | Medium-high; Ultra is not finished for ~70 days |
| Weight inheritance / weight selection init from Ultra | Early training | Unknown for 7x width change | Speculative | Low; easy A/B |
| Higher-rank output head, or keeping d near 768+ | Softmax bottleneck | Unknown; Godey shows the penalty below head rank ~1000 | Speculative | Parameter budget trade against the embedding table |

---

## 6. Open questions

1. **How does chat quality, not loss, scale with tokens at 150M?** No controlled study found. Cheap test: SmolLM2-135M publishes checkpoints every ~250B tokens (https://huggingface.co/HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints); fine-tune several identically and score multi-turn behaviour to measure returns from 250B to 2T.
2. **Does narrowing the pretraining/anneal distribution to chat-shaped text let a 150M reach a lower loss on held-out chat than a 360M trained on generic web?** This is the central testable version of "breaking the wall" from this lane, and it can be piloted at 30M-50M on the Mac or one Titan.
3. **Does the distillation law hold at 120M non-embedding and 1T tokens?** Busbridge's fit covers >=143M students and <=512B tokens on C4; MobileLLM and MiniPLM nulls suggest the law may be optimistic for tiny students.
4. **Is the NorMuon + architecture win still ~7% at 150M and 1,000+ tokens/param with each arm LR-tuned?** Wen et al. predict a shrink and a possible SOAP/Kron overtake.
5. **Is catastrophic overtraining a practical risk at 2k-7k tokens/param for a 150M model?** SmolLM2-135M at ~15k suggests "manageable", but no one reports its post-training sensitivity.
6. **Does the d<1000 head-rank penalty measurably limit chat at 150M, and is a cheap fix (factorized higher-rank head, different vocab size) worth its parameters?**
7. **Tokenizer strategy:** keep Max's own tokenizer (only Ultra as teacher, full ownership) or adopt SmolLM2's identical-size vocabulary (unlocks 360M/1.7B teachers and a pruning base, less "from scratch")? This is partly a project-identity decision for Max.
8. **Does pruning or weight inheritance from Ultra beat random init at 150M for the same token budget?** A 10B-token A/B would answer it cheaply once Ultra finishes.
