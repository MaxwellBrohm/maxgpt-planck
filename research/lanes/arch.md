# Lane: Architecture for a ~150M chat model (evidence at 50M-350M)

Research date: 2026-09-23. Scope: which architecture choices matter for a ~150M-parameter model that must hold multi-turn conversations, with evidence measured at 50M-350M wherever it exists. Every number carries a source; "(>=1B)" flags evidence that only exists at 1B or above; "unsourced" or "my computation" is marked as such.

Local measurements in this report (tokenizer compression on OASST1, parameter accounting) were computed on the Mac from files already in `~/.cache/huggingface` and from `config.json` / `tokenizer.json` files fetched from huggingface.co. Nothing in Max's repos was modified.

---

## Bottom line

1. The premise needs correcting before anything else. The "wall" is not at 400-500M any more. LFM2.5-350M (28T tokens plus RL) scores Multi-IF 44.92 and IFEval 76.96, level with Qwen3-0.6B on multi-turn instruction following. Falcon-H1-Tiny-90M-Instruct (Jan 2026, 800B tokens) scores IFEval 66.08 and 4.33 on a multi-turn benchmark, beating SmolLM2-360M-Instruct (3.8) on the same harness. What is still missing at 90-150M: multi-turn robustness, which TII says is "relatively low compared to Qwen-0.6B"; factual knowledge (TriviaQA 4.1 at 135M against 16.9 at 360M); and multi-step reasoning.
2. Parameter count predicts chat quality less well than training tokens per parameter do. The small models that work used 8.9k-80k tokens per parameter. Max's 100B-token build is 667 tokens per parameter at 150M. No architecture choice closes a 10-100x data gap. That is another lane's problem, but it caps what architecture can deliver.
3. Remembering earlier turns is not an architectural wall at 150M, as long as the model keeps real softmax attention. A 70M attention model beats a 1.4B gated-convolution model on associative recall. Induction heads form without help at 100M+ non-embedding parameters. Pure SSM or linear-attention stacks are the one choice that would create a recall wall.
4. The two hard, well-measured limits at this size are **knowledge capacity** and **compositional depth**. Knowledge capacity is about 2 bits per unique parameter, and looping does not raise it. Compositional depth: even 1.3B models trained on 100B tokens fail 2-hop reasoning. The best-measured architecture levers attack those two limits directly:
   - sparse lookup memory, which adds knowledge without adding compute: at a 134M base, Memory+ beats a dense 373M model on TriviaQA;
   - more effective depth through deep-thin shapes, Canon/short-conv mixing, and parameter reuse (looping).
5. For a strict 150M total-parameter budget, the concrete implications are:
   - shape: deep and thin (about 30-36 layers at d=576-640), not Max's current 12L/d768 shakedown shape;
   - vocabulary: 16k-32k, tied, so embeddings are 6-13% of the budget rather than 33%;
   - attention: full softmax attention in every layer (the KV cache is only about 94 MB at 4k context);
   - keep what Max's 124M A/B already validated.
6. Two cheap, plausible "new technique" bets fit Max's hardware: Canon layers (under 0.5% of parameters, large synthetic gains at 8L512D/12L768D) and 2x immediate block sharing or looping (+0.7 to +1.1 zero-shot points at 125M/350M in MobileLLM). Both should be judged on multi-turn behaviour evals, not validation loss.
7. The biggest lever that "breaks" a 150M wall is redefining the budget. Suppose "150M" means 150M dense, active parameters, while sparse lookup tables (value embeddings, n-gram hash embeddings, memory layers, per-layer embeddings) can live in CPU RAM or on disk. Then the 2026 speedrun and Meta's memory-layer results show 2-3x effective-parameter gains at 124-151M compute. The speedrun's current 124M-compute model carries about 640M total parameters by my count.

---

## 0. Premise check: sub-400M chat models that already exist

| Model (release) | Total / non-embedding params | Train tokens (tokens/param) | Chat-relevant scores (harness owner) | Source |
|---|---|---|---|---|
| Falcon-H1-Tiny-90M-Instruct (Jan 15 2026) | ~90M total; 24L, d=512, vocab 32,768 tied (16.8M emb) | 800B (~8.9k) | IFEval 66.08, "MT Bench (avg)" 4.33 (their reference for it is MT-Bench-101, a multi-turn benchmark), AlpacaEval 9.43 (TII harness) | [blog/PDF](https://tiiuae-tiny-h1-blogpost.hf.space/), [config](https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Instruct/resolve/main/config.json) |
| SmolLM2-135M-Instruct | 134.5M / 106M (my count from config) | 2T (~15k) | IFEval 30.69, MT 2.68 (TII harness); IFEval 29.9, "MT-Bench 19.8" (HF card, likely a x10 scale) | [TII](https://tiiuae-tiny-h1-blogpost.hf.space/), [HF card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), [SmolLM2 paper](https://arxiv.org/abs/2502.02737) |
| SmolLM2-360M-Instruct | 362M / 315M | 4T (~11k) | IFEval 41.0 vs Qwen2.5-0.5B-Instruct 31.6; MT-Bench 3.66 vs 4.16 (HF card) | [HF card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) |
| Gemma 3 270M | 268M / only ~100M (168M is the 262k-vocab embedding) | 6T (~22k) | IFEval 51.2 (Google card); 27.44 under TII's harness | [Google blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/), [card mirror](https://huggingface.co/unsloth/gemma-3-270m-it), [TII](https://tiiuae-tiny-h1-blogpost.hf.space/) |
| LFM2-350M | 354M, 16 layers (10 gated short-conv + 6 GQA) | 11T (~31k), with top-32 logit distillation from LFM1-7B | IFEval 65.12, Multi-IF 32.85 vs Qwen3-0.6B 64.24 / 45.13 (Liquid harness) | [LFM2 report](https://arxiv.org/abs/2511.23404) |
| LFM2.5-350M (2026) | same backbone | 28T (~80k) + multi-stage RL | IFEval 76.96, IFBench 40.69, Multi-IF 44.92 | [card](https://huggingface.co/LiquidAI/LFM2.5-350M) |
| MobileLLM-125M / 350M | 124.6M / 345.3M | 1T | MT-Bench 2.33 / 3.28; AlpacaEval win-rate vs text-davinci-001 24.07 / 47.08 (same SFT for all baselines) | [MobileLLM](https://arxiv.org/abs/2402.14905) |

Two things stand out. First, harness sensitivity is huge: Gemma 3 270M scores 51.2 or 27.44 on IFEval depending on who runs it, so compare only within one harness. Second, the TII report is explicit about the residual gap. In its tool-calling section's future work it says "across small models, the multi-turn scores were relatively low compared to Qwen-0.6B" ([TII](https://tiiuae-tiny-h1-blogpost.hf.space/)). So the honest statement of Max's goal is: single-turn instruction following at 90-150M is already solved in 2026. **Robust multi-turn coherence, knowledge and reasoning at ~150M is not solved.**

---

## 1. Detailed findings

### 1.1 Depth vs width

**MobileLLM (Liu et al., ICML 2024)** ([arXiv 2402.14905](https://arxiv.org/abs/2402.14905)). Measured at ~125M and ~350M on 0.25T tokens for the ablations and 1T for the finals, with 8 zero-shot commonsense tasks, single seed.
- 125M sweep (Table 11), same ~135M budget, zero-shot average:
  - 4L/1280: 43.3
  - 12L/768: 43.9
  - 18L/640: 43.9
  - 24L/576: 44.8
  - 30L/512: 44.8
  - 42L/448: 44.5
  - 62L/384: 44.7
- 350M sweep: 5L/2048 scores 47.1, 15L/1280 scores 48.7, 32L/896 scores 49.8, 46L/768 scores 49.6, 66L/640 scores 49.5.
- So going from 12 to about 24-32 layers is worth about +0.9 (125M) to +1.1 (350M) average points, and the curve is flat from ~24 layers onward. The authors state "the optimal depth is found to be near 30 layers for sub-billion scale models."
- The gains are larger on TriviaQA and RACE (Table 4): MobileLLM-125M reaches TQA 1-shot F1 13.9 vs 8.7 for OPT-125M, though that comparison is confounded by different training data.
- Caveat: a +0.9-point gain on zero-shot averages with one seed is near the noise band these benchmarks have at 125M. Nothing here measures generation or multi-turn behaviour.

**Falcon-H1-Tiny (TII, Jan 2026)** ([blog/PDF](https://tiiuae-tiny-h1-blogpost.hf.space/)). Fixed 90M budget, 200B tokens.
- Tested shallow, mid (27L) and deep (50L) versions of a hybrid parallel Mamba2+attention block.
- Deep gave "quite considerable" gains on MMLU/MMLU-Pro and better HellaSwag. Shallow was better on BBH.
- Deep cost "roughly a 2x decrease for training throughput", so they shipped the mid depth (24L at d=512). Numeric scores are in figures, not text.
- Width sweep: the best hidden size was "around 512" at 90M.

**Other replications at small scale**
- A weak 2025 blog replication at ~70M params and 1B tokens found 32L/384 at 38.50 vs 12L/512 at 38.15 (+0.35) and a 4L model far worse ([HF blog, codelion](https://huggingface.co/blog/codelion/optimal-model-architecture)). Low-quality evidence, but same sign.
- SmolLM2-135M copied MobileLLM's shape: 30L, d=576, 9/3 heads ([config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/config.json)).
- PleIAs pushed further: Baguettotron has 321M params and **80 layers** at d=576 ([config](https://huggingface.co/PleIAs/Baguettotron/resolve/main/config.json)). Monad has 56M params and 64 layers at d=256 ([config](https://huggingface.co/PleIAs/Monad/resolve/main/config.json)). PleIAs' card claims "consistent improvements from stacking more layers" but publishes no controlled ablation ([card](https://huggingface.co/PleIAs/Baguettotron)).

**Counterpoints and refinements**
- Curse of Depth (>=1B): with Pre-LN, deep layers contribute less, and 1/sqrt(layer) LayerNorm scaling fixes part of it ([arXiv 2502.05795](https://arxiv.org/abs/2502.05795)). Max already uses a 1/sqrt(depth) norm scaling in his A/B, which matters more the deeper he goes.
- Variable-Width Transformers (200M-2B): wide outer layers and narrow middle layers give 22% fewer FLOPs at matched loss ([arXiv 2606.18246](https://arxiv.org/abs/2606.18246)).
- Hourglass FFNs: allow wider-and-shallower at matched params, with 8.7% training-compute efficiency at 906M-8B; the 113M result is not stated in the abstract ([arXiv 2602.06471](https://arxiv.org/abs/2602.06471)).
- None of these refutes deep-and-thin at 125M. They show that where to put width matters.

**Relevance to chat.** Depth is the knob for sequential composition (Section 1.6 and 1.7). Width sets the output-head rank (Section 1.3). There is a real tension: deep-thin pushes d down to 512-576, while the softmax-bottleneck argument prefers d near or above 1000. The empirical 125M-350M data favours depth.

### 1.2 Embedding and vocabulary budget

**Parameter accounting** (my computation). The per-layer cost assumes a Llama-style block: (10 + 2/g)*d^2, with GQA ratio g=3 and a SwiGLU hidden size of 8/3 d. "Layers" is how many such layers fit in the 150M left over after the embedding.

| d | vocab | tied emb (M) | tied share | layers left (tied) | untied share |
|---|---|---|---|---|---|
| 512 | 8k | 4.2 | 2.8% | 52 | 5.6% |
| 512 | 32k | 16.8 | 11.2% | 48 | 22.4% |
| 512 | 49k | 25.2 | 16.8% | 45 | 33.6% |
| 576 | 16k | 9.4 | 6.3% | 40 | 12.6% |
| 576 | 32k | 18.9 | 12.6% | 37 | 25.2% |
| 576 | 49k | 28.3 | 18.9% | 34 | 37.7% |
| 576 | 100k | 57.8 | 38.5% | 26 | 77.1% |
| 640 | 32k | 21.0 | 14.0% | 30 | 28.0% |
| 640 | 49k | 31.5 | 21.0% | 27 | 41.9% |
| 768 | 32k | 25.2 | 16.8% | 20 | 33.6% |
| 768 | 49k | 37.7 | 25.2% | 18 | 50.3% |
| any 512-768 | 256k | 134-201 | 90-134% | none | n/a |

Reference points (my counts from each `config.json`):
- Max's 124M A/B shape (12L, d768, 49k tied, [local config](../../../Max's%20AI%20Model/maxgpt-ultra/configs/shakedown.yaml)) is 113.3M, of which **33.3% is embedding**.
- SmolLM2-135M: 21.0% embedding. MobileLLM-125M: 14.8%.
- Qwen2.5-0.5B: 27.6%, meaning only **~358M non-embedding params**. Qwen3-0.6B: 26.1% (440M body).
- Gemma 3 270M: **62.6%** (100M body).
- Qwen3.5-0.8B: 254M of embedding at vocab 248,320 ([config](https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/config.json)).
- So "the 500M chat models" have 360-440M of transformer body. Max's 150M with a 32k vocab would have ~127-131M of body, a ~3x gap in body parameters, not 3.3-4x.

**Compression on real conversations (measured here).** Setup:
- Trained byte-level BPEs on OASST1 English train messages (21.1 MB) and measured on the held-out OASST1 English validation set (2,022 messages, 1.08 MB, 658 root-to-leaf threads with 4 or more messages, mean 4.3 messages).
- Also measured pretrained tokenizers downloaded from huggingface.co.
- Data: [OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1).

| tokenizer | vocab | bytes/token | tokens per 4.3-message thread (mean) |
|---|---|---|---|
| own BPE 8k | 8,192 | 3.83 | 522 |
| own BPE 16k | 16,384 | 4.15 | 483 |
| own BPE 32k | 32,768 | 4.36 | 462 |
| own BPE 49k | 49,152 | 4.44 | 454 |
| own BPE "100k" (corpus only filled 87.5k) | 87,514 | 4.52 | 447 |
| SmolLM2 | 49,152 | 4.36 | 464 |
| LFM2 | 64,400 | 4.43 | 452 |
| Granite 4.0 | 100,352 | 4.64 | 433 |
| Qwen2.5 | 151,665 | 4.59 | 436 |
| Qwen3.5 | 248,070 | 4.51 | 448 |
| Gemma 3 | 262,145 | 4.47 | 447 |
| Monad | 8,192 | 3.52 | 557 |

What the table says:
- Going from 8k to 49k cuts tokens per conversation by 13%. Going from 49k to 100k+ cuts only 2-5% more.
- On English chat, vocabulary past ~32k buys almost nothing in "more turns fit in context" or "fewer decode steps".
- A 2048-token context holds about 4 OASST-style 4-message threads (about 17 messages) at any vocab from 16k to 262k. Context length, not vocab, is the lever for how many turns fit.

**FLOPs per byte** (my computation). With tied embeddings and a fixed 150M total, forward FLOPs per token are ~2 x 150M = 300M regardless of vocab, because the head uses the V*d parameters once per token. So a bigger vocab means fewer FLOPs per byte of text but a smaller transformer body. At d=576:
- non-embedding ("thinking") FLOPs per byte are 75.8M at 8k, 60.2M at 32k and 54.8M at 49k;
- the LM head is 3%, 13% and 19% of forward FLOPs respectively.

**Scaling-law guidance** (Tao et al. 2024, [arXiv 2407.13623](https://arxiv.org/abs/2407.13623)). Fit on 33M-3B-parameter models on up to 500B characters. The optimal number of vocabulary parameters scales as Nv ∝ Nnv^0.83.
- Anchoring on their Table 1 row (Nnv = 3B gives Nv ≈ 0.1B) and applying the exponent, Nnv ≈ 122-145M gives Nv ≈ 7-8M. At d=576 that is a vocab of about **12-14k** (my computation). The anchor is outside the 33M-1.13B fitted range, but the exponent was fit inside it.
- That is the compute-optimal value. Overtraining moves it up: at Nnv = 2.87B, 1.9x more data moved the optimum from 35K to 43K ([same paper, Table 3](https://arxiv.org/abs/2407.13623)).
- Max would train at ~667 tokens/param, about 33x Chinchilla, so the optimum is somewhere above 14k. 16k-32k is the defensible range.

**Decoupled input vocabulary** (Over-Tokenized Transformer, [arXiv 2501.16975](https://arxiv.org/abs/2501.16975)). Measured on OLMo2 at 151M and 400M dense params on 400B tokens, and at 1B on 1T.
- Input-side n-gram vocabularies of 1.2M and 12.8M entries give log-linear loss gains at every size.
- OE-12.8M at 400M matches a 1B baseline.
- Scaling up the *output* vocabulary "may be harmful to smaller models". The cost is hundreds of millions to billions of sparse embedding parameters. This is the same family as the speedrun's bigram hash (Section 1.10).

**Embedding sharing.** At 125M/30L, MobileLLM measured:
- tying saves 16M params (11.8%) for -0.2 points (44.8 to 44.6);
- spending the savings on 2 more layers gives 45.0 with 10M fewer params than untied ([arXiv 2402.14905](https://arxiv.org/abs/2402.14905), Table 1).

The speedrun unties at 2/3 of training for speed at 124M (record 53). Tying is right under a strict parameter budget.

**Embedding scale.** Gemma multiplies input embeddings by sqrt(d_model) ([HF modeling_gemma3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py)). No controlled small-scale gain found for it in this sweep: unsourced.

### 1.3 Softmax bottleneck and LM saturation

- **Godey et al. 2024** ([arXiv 2404.07647](https://arxiv.org/abs/2404.07647)). Pythia models up to 410M "suffer from the saturation phenomenon". Pythia-160M's best checkpoint beats its final one on LAMBADA perplexity (24.6 vs 32.9) and on ARC-e (46.5 vs 43.2).
- In the same study, performance degrades noticeably once the rank of the LM head falls below about 1000, "regardless of the model size". This was measured with rank-constrained heads on frozen models trained on ~150M tokens. Natural-text next-token distributions only become near full-rank approximable at 10,000-15,000 dimensions.
- Hidden size below ~1000 correlates with last-layer anisotropy and a spiky singular-value spectrum in late training.
- **Contrary 2026 evidence** ([Kulkarni et al., arXiv 2602.20433](https://arxiv.org/abs/2602.20433)). 108 controlled OLMo-style models. Low effective rank "does not cause late-stage performance degradation in small models, but instead co-occurs with it". An OLMo-14M replicating Pythia-14M did not saturate. Effective rank is driven largely by batch size and weight decay.
- **Implication.** At d = 512-640, Max is squarely in the regime Godey flags. The mechanism is contested, and MobileLLM/Falcon-Tiny measure d≈512 as fine on benchmarks. What is not measured anywhere is whether a low-rank head hurts *generation* quality (degenerate distributions, the "Bitcoin.com" loops in [WRITEUP_NOTES.md](../../../Max's%20AI%20Model/WRITEUP_NOTES.md)).
- Cheap fixes exist but have no small-model chat evidence: mixture-of-softmaxes, or a higher-rank head through a wider final projection. This is an open, testable question and a plausible "new technique" niche.

### 1.4 Attention internals

**GQA / MQA** (MobileLLM Table 13; 125M = 8L/d896 and 350M = 15L/d1280, 0.25T tokens) ([arXiv 2402.14905](https://arxiv.org/abs/2402.14905)):
- At 125M, 16 query heads with 4 KV heads score 44.7 vs 44.6 for 16/16, with 7% fewer params. MQA (16/1) scores 43.7.
- At 350M, 16/4 scores 49.4 vs 49.6, and 16/1 scores 47.9.
- 16 heads with head_dim near 64 was best.
- So GQA 3:1 or 4:1 is free at this scale, while MQA costs about 1 point.

Gemma 3 270M uses MQA with 4 heads of dimension 256 ([config mirror](https://huggingface.co/unsloth/gemma-3-270m/resolve/main/config.json)), an outlier. KV cache is not a real constraint at 150M: about 23 KB/token at 30L/3 KV heads/hd64 in fp16, or 94 MB at 4k tokens (my computation).

**QK-norm.**
- No clean 125M ablation found. It is part of speedrun record 5, a bundle that went from 22.3 to 15.2 minutes ([modded-nanogpt README](https://github.com/KellerJordan/modded-nanogpt)).
- In the controlled 20-method replication (>=1B; 1.2B/3B, 23B tokens, 3 seeds), QK-Norm+GeGLU ranked 1st at 3B ([arXiv 2605.20798](https://arxiv.org/abs/2605.20798)).
- It is a stability tool; keep it.

**Gated attention** (per-head sigmoid gate on the SDPA output).
- Measured only at 1.7B dense and 15B-A2.5B MoE (>=1B), 400B-3.5T tokens ([arXiv 2505.06708](https://arxiv.org/abs/2505.06708)).
- Up to 0.2 PPL and 2 MMLU points.
- The first-token attention share drops from 46.7% to 4.8%, which removes the attention sink.
- Over 10 points better on RULER in length extrapolation.
- Inside the noise band in the 1.2B controlled replication (z = +1.23, [arXiv 2605.20798](https://arxiv.org/abs/2605.20798)).
- Qwen3.5-0.8B ships it: `attn_output_gate: true` ([config](https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/config.json)).
- Multi-turn relevance: sink-free attention and better length extrapolation are the properties that matter when conversations grow. Max's A/B already has it.

**Attention sinks.** They emerge in small models after sufficient optimization. Removing softmax normalization (sigmoid attention without normalization) prevents them up to 1B ([arXiv 2410.10781](https://arxiv.org/abs/2410.10781)). StreamingLLM shows sinks are what make windowed multi-round dialogue work ([arXiv 2309.17453](https://arxiv.org/abs/2309.17453)). With gated attention they are unnecessary.

**Value residual (ResFormer)** ([arXiv 2410.17897](https://arxiv.org/abs/2410.17897)).
- Authors report equal validation loss with 16.11% fewer params and 20.3% less data, measured at **82M-468M on 20B tokens**. That is in scale.
- The 1.2B controlled replication puts it at z = +0.81, inside the noise ([arXiv 2605.20798](https://arxiv.org/abs/2605.20798)).
- Plausibly a small-scale-only gain. That is fine for Max, and his A/B already includes the normalized variant.

**Positional encoding**
- RoPE base varies wildly across small models with no controlled small-scale evidence: SmolLM2 1e5, LFM2 1e6, Falcon-H1-Tiny 1e11 (configs linked above). For contexts of 4k or less, base 10k-100k is conventional.
- Partial RoPE is now common: Pythia rotary_pct 0.25 ([config](https://huggingface.co/EleutherAI/pythia-160m/resolve/main/config.json)); Qwen3.5-0.8B `partial_rotary_factor` 0.25; the speedrun rotates only half of each head (`rotary_dim = head_dim // 2`, [train_gpt.py](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py)).
- p-RoPE (0.75) improved Gemma 2B from-scratch perplexity (>=1B, [arXiv 2410.06205](https://arxiv.org/abs/2410.06205)).
- NoPE generalizes better in length on small synthetic tasks ([arXiv 2305.19466](https://arxiv.org/abs/2305.19466)). Canon layers make NoPE match RoPE (Section 1.6). Interleaving RoPE and NoPE layers helps long context (>=1B, [arXiv 2501.18795](https://arxiv.org/abs/2501.18795)).
- Relevance: long multi-turn chats exceed the training context. Partial RoPE or NoPE layers is the cheap insurance.

**Attention-mechanism replacements.**
- Selective Attention: parameter-free, "equivalent to ... ~2X more heads and parameters in their attention modules" on C4 at small scale ([arXiv 2410.02703](https://arxiv.org/abs/2410.02703)). At 1.2B it is +2.15z and does not survive Bonferroni ([arXiv 2605.20798](https://arxiv.org/abs/2605.20798)).
- Differential attention claims better key-information retrieval (>=1B, [arXiv 2410.05258](https://arxiv.org/abs/2410.05258)) but was a significant failure in the 1.2B replication.
- Forgetting Transformer adds a forget gate compatible with FlashAttention and keeps the needle-in-a-haystack advantage over SSMs ([arXiv 2503.02130](https://arxiv.org/abs/2503.02130)).
- None has multi-turn chat evidence at 150M.

**Methodological warning** from the 1.2B replication ([arXiv 2605.20798](https://arxiv.org/abs/2605.20798)): attention-output modifications that land within 2-3% of baseline validation loss dropped 6-16 downstream points. Loss is not a safe proxy for behaviour. Judge chat-oriented architecture changes on behaviour evals.

### 1.5 FFN: SwiGLU vs ReLU²

- MobileLLM at 125M: a vanilla ReLU FFN scores 42.6 and SwiGLU 43.9 (+1.3); at 350M, 47.4 vs 48.7 ([arXiv 2402.14905](https://arxiv.org/abs/2402.14905), Table 10).
- Primer's search found squared ReLU and a depthwise conv after Q/K/V to be the two main wins. Primer cut training cost 4x at 500M on C4 and used 1/3 the compute at 1.9B ([arXiv 2109.08668](https://arxiv.org/abs/2109.08668)).
- The speedrun uses a plain ReLU² MLP (record 5 bundle).
- Physics of LMs, in synthetic tests at GPT-2-small scale:
  - Part 3.3: gated MLP stores *less* knowledge than a plain MLP, "particularly over shorter training durations" ([arXiv 2404.05405](https://arxiv.org/abs/2404.05405));
  - Part 4.1: "ReLU2 activation slightly improves standard MLP but degrades performance in gated MLP" ([arXiv 2512.17351](https://arxiv.org/abs/2512.17351)).
- Verdict: no clean ReLU² MLP vs SwiGLU head-to-head at ~150M on real data was found. At 150M, knowledge storage is the scarce resource, so the Physics result makes this a worthwhile A/B, not a settled choice.

### 1.6 Horizontal mixing: Canon layers, short convolutions, "smear"

**Canon layers** (Allen-Zhu, Physics of LMs Part 4.1, NeurIPS 2025, [arXiv 2512.17351](https://arxiv.org/abs/2512.17351)).
- What they are: residual, trainable causal 1-D convolutions with kernel 4 across neighbouring tokens. They are inserted before attention (A), inside it (B), before the MLP (C) and inside the MLP (D).
- In the synthetic playground, at in-scale sizes such as 8L512D and 12L768D:
  - reasoning depth +200-400%;
  - reasoning breadth +30%;
  - knowledge-manipulation length +30%;
  - NoPE+Canon matches RoPE;
  - linear attention (GLA)+Canon rises to Mamba2 level.
- Cost: under 0.45% extra params for GPT2-small. For a 1.3B Llama, Canon-ABCD adds 12.4% to forward time, 14.1% to backward and 20.8% to generation. Canon-AC adds 5.8%, 5.8% and 7.0%. Naive implementation.
- Real pretraining at 1.3B on 100B tokens "shows high noise and limited resolution". The consistent patterns there: Canon lifts NoPE to RoPE and GLA to Mamba2/GDN; linear models still lag on short-context retrieval; *all models fail 2-hop reasoning even within 100 tokens*.
- Part 4.2 released paired Llama vs LlamaCanon checkpoints at 1B-8B on 1-2T tokens ([repo](https://github.com/facebookresearch/PhysicsLM4)). I could not find a quantitative summary of their gains in text: open.

**The same idea shows up independently:**
- Primer's depthwise Q/K/V conv, which is Canon-B-like ([arXiv 2109.08668](https://arxiv.org/abs/2109.08668));
- Mamba2's conv1d, which Allen-Zhu shows drives "most of its gains";
- LFM2's gated short convolutions, kernel 3 in 10 of 16 layers ([arXiv 2511.23404](https://arxiv.org/abs/2511.23404));
- Qwen3.5's `linear_conv_kernel_dim: 4`;
- the speedrun's "smear" (mix the previous token's embedding into the current one): record 34, 2.565 to 2.547 min, -0.7% ([README](https://github.com/KellerJordan/modded-nanogpt)).

**Relevance to chat.** Canon's claimed mechanism is faster formation of the local token-to-token circuits that induction and associative recall are built from, plus deeper composition. It is the best-evidenced cheap architecture change aimed at reasoning depth rather than perplexity. **Not yet shown on conversational benchmarks at any scale.**

### 1.7 Parameter reuse: block sharing, looped and recurrent-depth transformers

**Iso-parameter comparisons (more compute, same stored weights), the case relevant to a strict 150M budget:**
- MobileLLM-LS, immediate block-wise sharing where each block runs twice:
  - 125M: 45.0 to 46.1 at 0.25T tokens, 46.3 to 47.0 at 1T;
  - 350M: 49.9 to 51.0 at 0.25T, 51.3 to 52.1 at 1T.
  - "Repeat-all-over" sharing was slightly better (45.2 vs 45.0 in the strategy ablation) but less cache-friendly ([arXiv 2402.14905](https://arxiv.org/abs/2402.14905), Tables 2 and 10).
  - Cost: 2x FLOPs.
- Gated Recurrent Transformer, Aug 2026 ([arXiv 2608.15062](https://arxiv.org/abs/2608.15062)). GPT-2 BPE, 9.8B tokens. At 124M stored params:
  - dense 12L baseline: 3.15 validation loss at 1.84 GFLOPs/forward;
  - GRT with 1+10x10+1 layers: 3.04 at 15.64 GFLOPs (-0.11 nats for 8.5x compute);
  - Huginn-style heavy-tail Poisson depth: 3.06; Relaxed Recursive: 3.08; Ouro-style: 3.10; MoR: 3.12;
  - for comparison, a dense 354M at 7.35 GFLOPs reaches 2.84.
  - So looping buys real loss at fixed parameters, but far less per FLOP than unique parameters.
- Recurrent Transformer, where each layer attends to its own activations ([arXiv 2604.21215](https://arxiv.org/abs/2604.21215)). 300M on ~1x Chinchilla C4: -0.03 cross-entropy at 12 layers and -0.057 at 6 layers vs parameter-matched. Training throughput is about 3x lower (42k vs 132k tokens/s).
- **Negative result:** CART, a single consumer GPU study, ~1B tokens ([arXiv 2606.01495](https://arxiv.org/abs/2606.01495)). At d=1024 parameter parity it "does not beat a parameter-matched dense baseline, losing by 1-2%". Varying the loop count at inference "degrades on both sides of the trained R".
- In-scale small-data evidence: recursive Transformers beat standard ones at 10M and 100M words ([arXiv 2608.26973](https://arxiv.org/abs/2608.26973)). Looped GPT-BERT: 12.18M params running 4 physical layers x 12 passes ([arXiv 2609.09691](https://arxiv.org/abs/2609.09691)).
- MixerLoop, at 15M and 110M: re-running only the Gated DeltaNet mixer, not the FFN, keeps 41.5% of the full-loop CORE gain at 110M with 45.9% fewer recurrent FLOPs ([arXiv 2608.18230](https://arxiv.org/abs/2608.18230)).

**Iso-FLOP comparisons (fewer stored params, same compute):**
- MoR "underperforms the vanilla model at the smallest model size (135M), likely due to a recursive capacity bottleneck" and matches it only from ~360M ([arXiv 2507.10524](https://arxiv.org/abs/2507.10524)).
- Relaxed Recursive Transformers work well only when converted from pretrained >=1B models ([arXiv 2410.20672](https://arxiv.org/abs/2410.20672)).

**What looping does and does not buy:**
- Ouro ([arXiv 2510.25741](https://arxiv.org/abs/2510.25741)), controlled synthetic study with GPT-2-style models of 1M-40M on bioS: "looping does not increase knowledge capacity"; both looped and non-looped reach "≈ 2 bits/parameter". Looping helps knowledge *manipulation* (the Mano task).
- Ouro's headline, 1.4B/2.6B on 7.7T tokens matching models up to 12B, is (>=1B).
- Saunshi et al. ([arXiv 2502.17416](https://arxiv.org/abs/2502.17416)): a k-layer model looped L times nearly matches a kL-layer model on reasoning. They describe a reasoning-vs-memorization dichotomy.
- Huginn is 3.5B (>=1B, [arXiv 2502.05171](https://arxiv.org/abs/2502.05171)).

**Relevance to chat.** Looping attacks composition and state-update depth, not knowledge. It is the right lever for "use what was said three turns ago in a two-step inference". It is the wrong lever for "know facts". Cost: training and inference FLOPs multiply by the loop count, which on one 5070 is the binding constraint.

### 1.8 Hybrids (SSM, short conv, linear attention) and in-context recall

- **Zoology** ([arXiv 2312.04927](https://arxiv.org/abs/2312.04927)). 70M-1.4B on 10B Pile tokens. Gated convolutions (H3, Hyena, RWKV) trail attention by up to 2.1 perplexity points, and 82% of that gap comes from associative-recall tokens, which are only 6.4% of all tokens. "A 70M parameter attention model outperforms a 1.4 billion parameter gated-convolution model on associative recall." Minimum gaps: +2.14, +0.59 and +0.35 PPL at 70M, 160M and 360M. Adding attention or input-dependent operators to under 10% of layers closes most of the gap.
- **Repeat After Me** ([arXiv 2402.01032](https://arxiv.org/abs/2402.01032)). Theory: 2-layer transformers copy strings exponential in length; SSMs are bounded by state size. On phone-book lookup, even the smallest Pythia (410M) beats the largest Mamba (2.8B).
- **LFM2** ([arXiv 2511.23404](https://arxiv.org/abs/2511.23404)). A hardware-in-the-loop search found that gated short conv plus a minority of GQA (6 of 16 layers) "match or exceed the aggregate quality of attention-heavier" baselines *under on-device latency and memory budgets*. Adding SSM or linear-attention operators "does not improve aggregate quality". They treat linear attention and SSMs as local blocks because of "their limitations in retrieval-intensive tasks".
- Newer hybrid choices, all attention-minority:
  - Qwen3.5-0.8B: 18 Gated DeltaNet + 6 full-attention layers ([config](https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/config.json));
  - Granite-4.0-H-350M: 28 Mamba2 + 4 attention ([config](https://huggingface.co/ibm-granite/granite-4.0-h-350m/resolve/main/config.json));
  - Falcon-H1-Tiny: Mamba2 and attention heads in parallel in every layer.
  - Gated DeltaNet improves retrieval over Mamba2 ([arXiv 2412.06464](https://arxiv.org/abs/2412.06464)).
  - Linear RNNs need negative eigenvalues to state-track even parity ([arXiv 2411.12537](https://arxiv.org/abs/2411.12537)).
- **Implication for Max.** At 150M with 4k-8k context, full attention is cheap: attention-score FLOPs are 25% of per-layer FLOPs at 2k and 40% at 4k for d=576 (my computation). The KV cache is under 100 MB. Hybrids buy CPU latency and long-context memory, which Max does not need. They cost exactly the capability his goal depends on, recall of earlier turns. **Stay full-attention**, or at most use a short-conv/Canon hybrid that keeps attention in every layer.

### 1.9 Sparse capacity: MoE, memory layers, lookup embeddings

**Memory layers** (Meta, "Memory Layers at Scale", [arXiv 2412.09764](https://arxiv.org/abs/2412.09764)). Base models of 134M, 373M, 720M and 1.3B, trained on 1T tokens with a 32k Llama-2 tokenizer. 134M rows:

| 134M-base variant | total params | NQ | TriviaQA F1 | HotpotQA |
|---|---|---|---|---|
| dense 134M | 134M | 0.91 | 7.7 | 5.18 |
| MoE | 984M | 2.49 | 13.08 | 7.80 |
| PEER | 1.037B | 2.46 | 16.34 | 8.82 |
| Memory+ (1M keys, 3 layers) | 937M | 3.16 | **18.77** | 9.35 |
| *dense 373M, for reference* | 373M | 2.58 | 17.68 | 10.06 |

So memory layers bought factual QA beyond a 2.8x larger dense model at the same compute, but with 7x the stored parameters.

**Speedrun sparse tables** (124M compute; [README](https://github.com/KellerJordan/modded-nanogpt), [train_gpt.py](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py)):
- value embeddings, record 14: 4.66 to 4.41 min, -5.4%;
- U-net value embeddings, record 15: -10.4% (bundle);
- bigram hash embedding, record 62: 1.748 to 1.655 min, -5.3%.
- Parameter count by my arithmetic: 5 x 50,304 x 768 value-embedding parameters (193M) plus a 377,280 x 768 bigram table (290M) on an ~78M dense body with 11 layers at d=768. Total stored parameters are about **640M** for "124M" of compute.

**Engram** (DeepSeek, Jan 2026, [arXiv 2601.07372](https://arxiv.org/abs/2601.07372)). Hashed n-gram lookup memory (>=1B; the smallest backbone is 3B MoE with 568M active). The notable chat-relevant claim is that offloading local n-gram patterns to lookups "frees up attention capacity for global context": Multi-Query NIAH 84.2 to 97.0, Variable Tracking 77.0 to 89.0.

**Gemma 3n Per-Layer Embeddings.** E2B has "over 5 billion" raw params but "just under 2 billion" effective, with PLE parameters cached to fast storage outside operating memory ([Google docs](https://ai.google.dev/gemma/docs/gemma-3n)). This is shipped proof that lookup parameters can live off the accelerator.

**MoE at fixed total parameters** (Joint MoE Scaling Laws, [arXiv 2502.05172](https://arxiv.org/abs/2502.05172)). 280+ runs, up to 2.7B active and 5B total. Rule of thumb: at a fixed total parameter count, an MoE with E ≤ 8 experts beats a *compute-optimal* dense model "if trained on E times more tokens". Against a heavily overtrained dense model like Max's (667 tokens/param) the authors flag uncertainty.
- Canon paper: MoE loses knowledge capacity, and Canon-ABC "substantially improves MoE knowledge acquisition and bit-per-param capacity" (synthetic, [arXiv 2512.17351](https://arxiv.org/abs/2512.17351)).
- In the 134M memory-layer table above, MoE was clearly worse than memory layers for factual QA at similar total parameters.

**Total vs active for Max's goal.** If "150M" means stored parameters on a phone or in 300 MB of RAM, MoE and memory tables are off the table, and the budget should go to depth plus Canon plus looping. If "150M" means 150M of dense compute per token, the strongest measured small-scale lever in this whole sweep is sparse lookup memory. It directly attacks the knowledge wall. **This definitional choice is the single biggest architecture decision.**

### 1.10 Multi-token prediction and auxiliary objectives

- Gloeckle et al. ([arXiv 2404.19737](https://arxiv.org/abs/2404.19737)). Models from 300M to 13B trained on code; multi-token prediction is "worse than baseline for small model sizes" and wins at scale.
- The same paper, on children's stories with 1M-1B non-embedding parameters: 2-token prediction "vastly improved" induction capability "for models of size 30M nonembedding parameters and below, with their advantage disappearing for sizes of 100M nonembedding parameters and above". So induction heads, the core mechanism for recalling names and facts from earlier turns, form fine at ~100M+ without help.
- TOP (token order prediction, [arXiv 2508.19228](https://arxiv.org/abs/2508.19228)). 340M on 52B FineWeb-Edu tokens, one extra unembedding layer:
  - NTP: LAMBADA PPL 30.34, accuracy 36.35, TriviaQA 4.93;
  - MTP: 35.31 / 35.32 / 2.55;
  - DeepSeek-style MTP: 40.99 / 34.66 / 0.87;
  - TOP: 28.76 / 37.07 / 4.37.
  - So exact MTP hurts at 340M, and TOP gives a small gain.
- A forward curriculum (NTP to MTP) helps small LMs use MTP ([arXiv 2505.22757](https://arxiv.org/abs/2505.22757)).
- The speedrun uses MTP with weights decaying to 0 (record 53, -2.4% time, bundled with untie-at-2/3) and a prefix-token-prediction auxiliary loss (record 88, -1.0%). This is measured at ~0.34B tokens, a very undertrained regime.
- Verdict: not a priority for a 150M chat model. TOP is the only cheap variant with an in-scale positive.

### 1.11 The modded-nanogpt speedrun, technique by technique

The target is FineWeb validation loss 3.28 with GPT-2-small-class compute on 8xH100. It now takes "under 400M tokens (the llm.c GPT-2 replication needed 10B)" ([README](https://github.com/KellerJordan/modded-nanogpt)). The current config is 11 layers, d=768, 6 heads of dim 128, vocab 50,304, sliding windows of 128-token blocks growing to 13 blocks (about 1.7k tokens). By my arithmetic from `TRAINING_STAGES` in [train_gpt.py](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py), it trains on about 0.34B tokens.

Wall-clock deltas per record, from the README record table. They mix systems and modelling changes, and time is not loss.

| # | change | time before to after | delta |
|---|---|---|---|
| 2 | tuned LR + rotary | 45 to 31.4 min | -30% |
| 5 | padded embeddings, ReLU², zero-init projections, QK-norm (bundle) | 22.3 to 15.2 | -32% |
| 8 | untie embedding and head | 12.0 to 10.8 | -10% |
| 9 | value and embedding skip connections, momentum warmup, logit softcap (bundle) | 10.8 to 8.2 | -24% |
| 11 | U-net skips + double LR | 7.8 to 7.2 | -7.7% |
| 14 | value embeddings | 4.66 to 4.41 | -5.4% |
| 18 | logit softcap 30 to 15 | 3.57 to 3.40 | -4.8% |
| 28 | sparse attention gate | 2.817 to 2.812 | -0.2% |
| 34 | smear (1-token look-back) | 2.565 to 2.547 | -0.7% |
| 41-43 | NorMuon, LR fix, cautious WD | 2.358 to 2.284 | -3.1% |
| 53 | MTP + untie at 2/3 | 2.037 to 1.988 | -2.4% |
| 58 | paired-head attention | 1.878 to 1.820 | -3.1% |
| 62 | bigram hash embedding | 1.748 to 1.655 | -5.3% |
| 81 | MUDD skip connections | 1.406 to 1.363 | -3.1% |
| 85 | MUDD gates + dynamically composable MHA | 1.320 to 1.271 | -3.7% |
| 88 | prefix-token auxiliary loss | 1.256 to 1.243 | -1.0% |

Transfer caveats:
- All of it is measured at 0.34-10B tokens on web-text loss, not on overtrained or chat regimes.
- Max's own prior sweep ([research_2026-09-22.md](../../../Max's%20AI%20Model/maxgpt-ultra/docs/research_2026-09-22.md)) found hyper-connection-style residual variants strongly negative at 1.2B (AttnRes z = -29.47; HyperConnections diverged at 3B).
- The speedrun's short sliding windows optimize a metric that barely rewards long-range recall. They are the wrong default for multi-turn chat.
- The transferable core: Muon/NorMuon, QK-norm, ReLU², zero-init projections, value embeddings, smear/Canon-style mixing, and sparse lookup tables if the budget definition allows them.

### 1.12 Cross-check against Max's 124M A/B

- **Recipe.** Max's A/B ([README](../../../Max's%20AI%20Model/maxgpt-ultra/docs/ab_2026-09-21/README.md)) was 1.1B tokens with one seed per arm. It adopted NorMuon, cautious WD, per-head gated attention, normalized value residual and 1/sqrt(depth) norm scaling, for 6.9% lower held-out loss (2.993 to 2.786).
- **Overlap.** This overlaps speedrun records 28 (sparse attention gate), 41 and 43 (NorMuon, cautious WD), and the ResFormer paper, which is in scale at 82M-468M. It is the right recipe to carry forward. Two cautions:
  - (a) at 1.2B, gate and value residual were within noise, so part of the 124M gain may be small-scale or short-training specific;
  - (b) the NorMuon arms led only after the LR decay, so a 1.1B-token A/B may not predict a 100B-token run.
- **Shape.** The A/B model (12L, d=768, 49k tied, 113M) is the "wide-shallow" shape MobileLLM measured as ~1 point worse than 24-32 layers at 125M. It spends 33% of its parameters on embeddings.
- **Not adopted and worth testing at 150M, in priority order:**
  1. deep-thin shape with a 16-32k vocab;
  2. Canon or short-conv mixing;
  3. 2x block sharing or looping;
  4. ReLU² MLP vs SwiGLU;
  5. partial RoPE;
  6. value embeddings or bigram hash, if sparse parameters are allowed.

---

## 2. What this implies for a ~150M chat model

**A defensible baseline ("nano-v0")**, every element measured at 90M-350M:
- **Shape**: d=576, 36 layers, 9 query heads and 3 KV heads (head dim 64), SwiGLU hidden 1536, tied embeddings, vocab 32,768. That is 18.9M embedding + 127.4M body = 146.3M total (my count), with 12.9% in embeddings. Alternatives: d=640 / 28L / 10:2 heads (141.4M), or d=512 / 40L (129.5M).
  - Throughput warning: going from 27 to 50 layers halved TII's training throughput at 90M.
- **Vocab**: 16k-32k tied. The scaling law puts compute-optimal near 12-14k, moving up with overtraining. On chat text, 49k saves only 2% of tokens over 32k while costing 9.4M body params (7% of the body) at d=576.
  - Re-tokenizing the 100B build is the cost. Keeping 49k is acceptable if that cost is prohibitive (config B: d576 / 32L / 49k = 141.6M, 20% embeddings).
- **Attention**: full causal softmax attention in every layer; no sliding window, no SSM or linear-attention replacement. Keep QK-norm, per-head output gate and value residual (all in Max's validated recipe). Use partial RoPE (25-50% of dimensions), and possibly a NoPE layer every 4th layer, for length robustness.
- **Context**: train with at least 4k context. Multi-turn threads average ~450 tokens per 4 messages (measured above), and attention is only ~40% of FLOPs at 4k/d576.
- **Keep the A/B-validated optimizer and norm recipe** (NorMuon + cautious WD + 1/sqrt(depth) norm scaling). The optimizer is other lanes' territory.

**Architecture A/Bs worth running** (each against nano-v0, judged on held-out loss *and* a fixed multi-turn behaviour probe such as recall of names or facts from turn 1 at turn 5):
1. Canon-AC or Canon-ACD (kernel 4, residual). Expected: faster induction and composition; +5-20% step time.
2. 2x immediate block sharing (MobileLLM-LS) vs a 2-pass loop of the middle blocks (prelude/core/coda). Expected: about +1 zero-shot point at 2x FLOPs; the GRT data says about -0.1 nats needs about 8x FLOPs.
3. ReLU² plain MLP (4x) vs SwiGLU at equal parameters. The knowledge-capacity angle is what matters at 150M.
4. Vocab 16k vs 32k.
5. If the budget definition allows: value embeddings, or a bigram-hash table in CPU RAM, or one product-key memory layer.

Cost per arm: Max's 124M A/B needed about a day per 1.1B tokens per TITAN RTX (A/B README dates). A 2B-token, 150M arm is therefore roughly 2-3 GPU-days on that class of card (my extrapolation).

---

## 3. Evidence about the "wall" (why chat degrades below ~400M)

1. **The wall moved, and data did most of the moving.** LFM2.5-350M (28T tokens + RL) scores Multi-IF 44.92, about equal to Qwen3-0.6B's 45.13 in Liquid's harness ([card](https://huggingface.co/LiquidAI/LFM2.5-350M), [LFM2](https://arxiv.org/abs/2511.23404)). Falcon-H1-Tiny-90M (800B tokens, 25% SFT data mixed into pretraining) beats SmolLM2-360M on TII's multi-turn MT-Bench column (4.33 vs 3.8) ([TII](https://tiiuae-tiny-h1-blogpost.hf.space/)). MobileLLM's MT-Bench rises from 2.33 at 125M to 3.28 at 350M under an identical recipe ([arXiv 2402.14905](https://arxiv.org/abs/2402.14905)). Size matters, and recipe matters as much or more.
2. **Knowledge capacity scales with unique parameters, at about 2 bits/param** ([arXiv 2404.05405](https://arxiv.org/abs/2404.05405)). Loops do not change it (1M-40M synthetic, [arXiv 2510.25741](https://arxiv.org/abs/2510.25741)). TriviaQA is 4.1 at SmolLM2-135M vs 16.9 at 360M ([HF cards](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)). This is the "confident factual nonsense" failure Max saw at 110-235M. Sparse memory bypasses it: 134M + Memory+ reaches TriviaQA F1 18.77 against 17.68 for a dense 373M model ([arXiv 2412.09764](https://arxiv.org/abs/2412.09764)).
3. **Recall of earlier context is not param-limited for attention models.** A 70M attention model beats a 1.4B gated conv on associative recall ([arXiv 2312.04927](https://arxiv.org/abs/2312.04927)). Pythia-410M beats Mamba-2.8B on phone-book lookup ([arXiv 2402.01032](https://arxiv.org/abs/2402.01032)). Induction forms without help above ~100M non-embedding params ([arXiv 2404.19737](https://arxiv.org/abs/2404.19737)).
4. **Composition depth is a real limit, even far above 150M.** 1.3B models on 100B tokens all fail 2-hop reasoning within 100 tokens ([arXiv 2512.17351](https://arxiv.org/abs/2512.17351)). Depth-oriented levers (Canon, depth, looping) show their largest synthetic gains here.
5. **The output head may be rank-limited at d<1000.** Pythia-160M saturates late in training (LAMBADA PPL 24.6 best vs 32.9 final) ([arXiv 2404.07647](https://arxiv.org/abs/2404.07647)). But a 108-model 2026 study says low rank co-occurs with saturation rather than causing it, and depends on batch size and weight decay ([arXiv 2602.20433](https://arxiv.org/abs/2602.20433)). Unresolved.
6. **Multi-turn fragility is partly universal.** Top LLMs lose 39% on average going from single-turn to underspecified multi-turn, mostly through unreliability, "when LLMs take a wrong turn in a conversation, they get lost and do not recover" ([arXiv 2505.06120](https://arxiv.org/abs/2505.06120)). Part of what looks like a small-model wall is a general failure that small models hit sooner. That points at post-training data (other lanes) more than architecture.
7. **Tokens per parameter.** Working sub-400M chat models trained at:
   - ~8.9k tokens/param (Falcon-Tiny 90M, 800B);
   - ~15k (SmolLM2-135M, 2T);
   - ~22k (Gemma 3 270M, 6T);
   - ~80k (LFM2.5-350M, 28T).
   A 150M model on 100B tokens is 667. Sources are in Section 0.
8. **Loss can hide behaviour loss.** At 1.2B, attention modifications within 2-3% validation loss dropped 6-16 downstream points ([arXiv 2605.20798](https://arxiv.org/abs/2605.20798)). Any architecture claim about chat has to be verified on behaviour, not loss.

---

## 4. Levers

| lever | limit attacked | expected effect at ~150M | evidence | cost |
|---|---|---|---|---|
| Deep-thin shape (24-36L at d 512-640) | composition depth; wasted embedding budget | about +1 zero-shot point vs 12L at 125M; flat beyond ~24L | moderate (MobileLLM 19-model sweep, Falcon-Tiny ablation; single seeds) | up to 2x lower training throughput at 50L |
| Vocab 16-32k, tied | embedding share (33% to 6-13%) | about 10-15M params moved to the body; 2-7% more tokens per conversation | moderate (scaling law fit at 33M-3B; my OASST1 measurement) | re-tokenize 100B tokens |
| Full attention everywhere; no SSM/linear replacement | in-context recall of earlier turns | avoids a recall wall entirely | strong (Zoology 70M-1.4B; Jelassi; LFM2 keeps attention) | about 25-40% of FLOPs at 2-4k context; negligible memory |
| Canon layers (kernel-4 residual conv) | reasoning depth and breadth; horizontal information flow | 2-4x synthetic reasoning depth at 8L512D/12L768D; real-data gain unknown | moderate (synthetic, in scale), weak (real data, 1.3B noisy) | <0.5% params, +6-20% time |
| 2x block sharing or looping | effective depth without stored params | +0.7 to +1.1 zero-shot at 125M/350M; about -0.1 nats needs about 8.5x FLOPs | moderate (MobileLLM, GRT); one negative (CART) | 2-8x FLOPs; does not add knowledge |
| Sparse lookup memory (value embeddings, bigram hash, PKM/Memory+, PLE) | knowledge capacity (2 bits/param) | Memory+ at 134M beats dense 373M on TriviaQA; speedrun -5% time each | strong for knowledge/loss; unknown for chat | stored params 3-7x; breaks a strict "150M total" definition |
| Gated attention + value residual + QK-norm (already adopted) | sinks, long-context stability, small-scale loss | part of Max's measured 6.9% loss gain at 124M | moderate at small scale; null at 1.2B | negligible |
| Partial RoPE / NoPE interleave | length generalization in long chats | better extrapolation past the training context | weak-moderate (mostly >=1B; Canon NoPE result synthetic) | none |
| ReLU² plain MLP vs SwiGLU | knowledge storage per param | uncertain sign: SwiGLU +1.3 zero-shot (MobileLLM) vs better knowledge capacity for plain MLP (Physics 3.3) | weak, conflicting | none |
| Higher-rank output head (mixture of softmaxes, wider final projection) | softmax bottleneck at d<1000 | unknown; may reduce degenerate or looping generations | speculative (theory contested) | small |
| TOP auxiliary loss | representation quality | small gains at 340M (LAMBADA PPL 30.3 to 28.8) | weak-moderate (one paper, in scale) | one extra unembedding at train time |
| MoE at fixed 150M total | per-token compute | only helps if compute-limited; unclear vs overtrained dense | weak | routing complexity; hurts knowledge per param (synthetic) |
| Exact multi-token prediction | none at this size | negative at 300-600M; induction benefit gone above 100M non-embedding | strong (negative) | skip |

---

## 5. Open questions

1. Does a low-rank output head (d = 512-640) measurably worsen *generation* (repetition, entropy collapse) in small chat models, separately from benchmark accuracy? Nobody has measured it; Godey and Kulkarni disagree about the mechanism.
2. Do Canon layers improve *multi-turn* recall or state tracking on real conversational data at ~150M, or only synthetic Depo/Brevo tasks? Part 4.2's 1-8B paired checkpoints exist, but I found no text summary of the gains.
3. Iso-parameter looping at 150M with 100B+ tokens: MobileLLM-LS is the only long-training (1T) evidence. Does the +0.7-point gain survive as a behavioural gain, and is 2x compute better spent on looping or on more tokens?
4. What exactly is the "150M" budget: stored params, active params, or RAM footprint? The answer decides whether sparse memory (the strongest measured lever against the knowledge wall) is allowed.
5. Is the Falcon-Tiny result (90M beating 360M-class models on a multi-turn benchmark) driven by architecture (parallel Mamba2+attention, learnable multipliers) or by the 25% SFT-in-pretraining data? TII did not isolate it.
6. ReLU² plain MLP vs SwiGLU at 150M on overtrained real data, measured on knowledge probes: no clean head-to-head found.
7. Do the 124M A/B gains (gate, value residual) survive at 100B tokens? The 1.2B null result suggests they may shrink with scale or training length.
8. Which multi-turn failure dominates at 150M: forgetting earlier content (retrieval), mixing up who said what (state tracking), or losing the thread after one bad turn (reliability)? The right architectural lever depends on the answer, and I found no published breakdown at this size.
