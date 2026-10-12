# Architecture from first principles for a 10M-60M chat model (MaxGPT-Planck)

Research date: 2026-09-23. Track: what each transformer component stores or computes, which designs shift a tiny parameter budget from storage toward conversational skill, and three concrete 20M-total-parameter candidates.

This builds on the first run's verified lanes (`research/lanes/arch.md` + `arch.verify.md`, `context.md`, `frontier.md` + `frontier.verify.md`, `census.md`), which covered 50M-350M. I do not repeat their findings except where a Planck decision leans on them; corrected versions from the `.verify.md` files are used. New sources in this report were read as PDFs (arXiv, converted with pdftotext) or primary pages. Every number carries a source URL, "my computation", or "unsourced". Scale is given as (params, tokens) wherever the source states it. Nothing was trained or run on the Mac: parameter counts below are arithmetic.

---

## Bottom line

1. **Parameters cannot be protected from facts by architecture.** Physics of LMs 3.3 measured the same ~2 bits/param for GPT-2 with full MLPs, quarter-size MLPs, and no MLPs at all (1000 exposures, synthetic bios), and a 2024 theory paper shows attention value matrices and MLPs can each serve as the fact store. So "MLP-light" does not by itself free capacity for skill: the training data decides what the parameters fill up with, and the architecture decides how cheaply the skill can be built.
2. **Routing is cheap.** Attention solves multi-query associative recall at a constant width of 64 for every sequence length tested (Zoology, 2-layer models). Sharing attention weights across all layers of ALBERT cost nothing (+0.1 average at E=128). In a July 2026 controlled study, Q/K matrices "crystallize" in the first quarter of training. The same study's 24M attention-only model matched a parameter-matched FFN model within 0.006 nats on synthetic reasoning data, and beat it on passage-grounded QA (sciq 0.742 vs 0.661). Caveats: one vendor preprint, and its FFN control had only 4 layers.
3. **"More per parameter" only has in-scale evidence through reuse:**
   - MoEUT (shared layers plus MoE) beats a parameter-matched 44M dense model: C4 perplexity 18.97 to 18.30, BLiMP 73.5 to 78.2.
   - A 116-run looped-model scaling law values each extra recurrence at r^0.46 of a unique block, rising to r^0.65 with hyper-connections.
   - Sharing has a cost: at 12M it hurt entity tracking, and Mixture-of-Recursions lost to vanilla at 135M.
   - No architecture evidence raises storage above ~2 bits/param.
4. **Keep softmax attention in every layer.** At 20M, SSM, linear-attention and short-conv layers save no parameters, and the KV cache they would save is about 12 MB at 2k context (my computation). The small-model failure the probe hints at in every model up to 0.6B, not updating after a correction, matches a known softmax+RoPE failure: the head attends to the first matching key instead of the latest one. There is a known fix (stick-breaking attention), but it has only been shown on a 2-layer toy.
5. **Memory layers, PKM, value/n-gram tables and structured matrices are the wrong tools under a total-parameter metric.** They raise storage or capacity per FLOP, not per parameter. In GPT-2, block-tensor-train layers matched dense layers once language-model-head compute was excluded.
6. **At 20M, spend about 16% of the budget on a tied 8k vocabulary and put the rest in a 12-20 layer body at d=384-512.** The output head's rank (equal to d) sits far below the ~1000 threshold Godey et al. flag, and that cost is measured even in tiny models: half-rank heads cost 0.18 nats in a 2.26M-parameter test. This is the strongest argument for the wider (d=512) looped candidate.
7. **Three candidates at about 20M total** are specified in section 3, each testable first on minute-scale synthetic probes and 5-10M proxies:
   - (A) dense 12L/d384 control;
   - (B) attention-heavy 19L/d384 with MLP ratio 1;
   - (C) a 6-unique/12-applied looped core at d512 with hyper-connections.

---

## 1. What each component stores or computes

### 1.1 Storage capacity does not care where the parameter sits

- **Physics of LMs 3.3** ([arXiv 2404.05405](https://arxiv.org/abs/2404.05405), GPT-2-style models from under 10M to ~0.5B, synthetic biographies):
  - Result 5, at 1000 exposures: "Reducing the MLP size of GPT2 architecture by 1/4 or even eliminating all MLP layers does not affect its capacity ratio ... the Attention layers are also capable of storing knowledge." For models of 10M or fewer, tying embedding and output weights raises capacity.
  - Result 6, at 100 exposures, the undertrained regime closer to rare facts: "Reducing GPT2's MLP size by 1/4 has a negligible impact on the capacity ratio. Removing MLPs decreases the capacity ratio by more than 1.5x". Gated MLP (LLaMA/Mistral) is 1.3x worse than GPT-2's plain MLP.
  - Reading: MLPs make storage faster to learn, not larger.
- **Nichani, Lee, Bietti** (ICLR 2025, [arXiv 2412.06538](https://arxiv.org/abs/2412.06538); theory plus toy experiments):
  - Both linear and MLP associative memories have storage that scales linearly with parameter count.
  - A one-layer attention+MLP transformer reaches 100% on a synthetic factual-recall task when "either the total number of self-attention parameters or MLP parameters scales (up to log factors) linearly with the number of facts".
  - The model "can trade off between using the value matrices or the MLP as an associative memory".
- **Where facts go when the MLP is gone.** In the attention-only study below ([arXiv 2607.18363](https://arxiv.org/abs/2607.18363), 6M-87M), removing FFNs "relocates" content accumulation to the attention output projection W_o.
- **Consequence for Planck (first principles).** A 20M model trained on fact-rich text will store facts in whatever matrices it has, at ~2 bits/param or less. Route (1), "cut the parameters skill needs", can only succeed together with data that stops rewarding fact storage (a data-lane lever). Architecture's job is to make the skill circuits cheap and fast to learn.

### 1.2 How much "language" is there to store? A human anchor

Mollica and Piantadosi 2019 estimate what an adult English speaker has learned about language ([PMC6458406](https://pmc.ncbi.nlm.nih.gov/articles/PMC6458406/), Table 1). The lexicon is assumed to be about 40,000 words and idioms.

| Component | Best guess | Range |
|---|---|---|
| Phonemes | 750 bits | |
| Wordforms | 400k bits | |
| Lexical semantics | 12.0M bits | 0.55M to 40M |
| Word frequency | 80k bits | |
| Syntax | 697 bits | 134 to 1,394 |
| Total | 12.48M bits | 0.79M to 40.8M |

My arithmetic, heavily caveated:
- At Physics 3.3's best-case 2 bits/param, the best-guess total is ~6.2M parameters; the upper bound is ~20M.
- At 1 bit/param (the 100-exposure regime, realistic for rare words) those become ~12.5M and ~41M.

Caveats:
- This is a cognitive-science estimate for humans, not for transformers.
- A network needs parameters for computation, not only storage.
- LMs store word meaning less efficiently than an ideal code.

What it does support is the framing's direction:
- Syntax is almost free in bits; word meaning is the bulk, and it behaves like knowledge.
- A chat model with a restricted everyday vocabulary (a few thousand words, TinyStories-style) needs far fewer lexical bits. This is consistent with TinyStories producing coherent text below 10M params on a narrow vocabulary ([arXiv 2305.07759](https://arxiv.org/abs/2305.07759)).
- BLiMP "saturates relatively early" in size sweeps: BabyLM-regime models at 10M-100M words ([arXiv 2608.26973](https://arxiv.org/abs/2608.26973)).

### 1.3 Embeddings: identity is cheap, meaning is knowledge, the output head is the real cost

- **Identity needs about log2 V dimensions.** In a 32-layer, d=1024 model trained on ~17B tokens (3 seeds), replacing the trainable input table with fixed 16-bit binary codes gave "comparable" perplexity. The gap was inside seed variance. [arXiv 2605.09751](https://arxiv.org/abs/2605.09751) (single author, >=1B-class scale).
  - Meaning can be built by the body.
  - With tied weights this saves nothing, because the same matrix is the output head. Planck's embedding cost is really the output head's cost.
- **Tying helps tiny models.** Physics 3.3 shows it for 10M or fewer. MobileLLM at 125M: tying costs 0.2 points, and spending the saved 16M on depth nets +0.2 ([arXiv 2402.14905](https://arxiv.org/abs/2402.14905), per `arch.verify.md`).
- **Factorized embeddings (ALBERT).** BERT-base shape, BookCorpus+Wikipedia, 125k steps ([arXiv 1909.11942](https://arxiv.org/abs/1909.11942), Table 3). Without sharing, E=128 scores an average of 81.7 against 82.3 for E=768, while cutting 19M parameters. For a decoder the catch is that the output head's rank becomes E (section 1.6).
- **Vocab at 20M** (my computation, tied):

| Vocab | Tied embedding at d=384 | Share of 20M |
|---|---|---|
| 4,096 | 1.57M | 8% |
| 8,192 | 3.15M | 16% |
| 16,384 | 6.29M | 31% |
| 32,768 | 12.6M | 63% |
| 49,152 | 18.9M | 94% |

- **The vocabulary scaling law** (Tao et al., [arXiv 2407.13623](https://arxiv.org/abs/2407.13623), fit on 33M-1.13B non-vocab params) gives Nv ∝ Nnv^0.83. Scaling the lane's ~7-8M vocabulary parameters at Nnv≈130M down to Nnv≈16M gives Nv ≈ 1.2-1.4M, which is **V ≈ 3-4k at d=384** (my computation). That is below the fitted range, and it is compute-optimal; overtraining pushes the optimum up.
- **Small-scale practice:** MoEUT used an 8,000-token SentencePiece vocabulary at 44M-1B ([arXiv 2405.16039](https://arxiv.org/abs/2405.16039)). The 24M attention-only model spent 8.4M params on embeddings, about 16k vocab at d=512 by my arithmetic from its Table 1.
- **Measured cost of a smaller vocab in tokens** (lane measurement, `arch.verify.md` row 5): an own-BPE 8k gives 3.83 bytes/token on OASST1 vs 4.15 at 16k, so 8k costs about 8% more tokens per conversation (my computation).
- **Verdict:** a tied 8k vocabulary shared across the whole Planck curve. 4k and 16k are ablation arms (ledger L8).

### 1.4 Attention: routing and exact retrieval, cheap in parameters

- **Recall needs little width.**
  - Zoology: "attention solves Mqar perfectly at all sequence lengths using a constant model dimension of 64". Gated convolutions do not reach >0.9 accuracy "unless d ≥ N" (sequence length). This was with 2-layer models ([arXiv 2312.04927](https://arxiv.org/abs/2312.04927), Sec. 4, Fig. 2).
  - In real LM data, attention's recall still improves with size: AR-hit perplexity is 2.41 at 70M and 1.98 at 360M (`arch.verify.md` row 7). So "cheap" means cheap to have, not saturated.
- **Routing weights are reusable across depth.**
  - ALBERT (BERT-base, [arXiv 1909.11942](https://arxiv.org/abs/1909.11942), Table 4): sharing only attention across all 12 layers scores 81.7 vs 81.6 not-shared at E=128, and 81.6 vs 82.3 at E=768. Sharing only the FFN scores 80.2 and 79.5. "most of the performance drop appears to come from sharing the FFN-layer parameters".
- **Routing is learned early and then frozen.** In every model of the attention-only study, Q/K "spectrally crystallize within the first quarter of training and do not move thereafter". The write-path matrices (FFN down-projection, or W_o without FFN) keep accumulating rank ([arXiv 2607.18363](https://arxiv.org/abs/2607.18363), 6M-87M, up to 105B tokens).
- **Retrieval concentrates in a few heads.** In both transformers and SSMs, in-context retrieval runs through "Gather-and-Aggregate" heads. Disabling one such head in a pruned Llama-3.1-8B drops MMLU from 66% to 25% ([arXiv 2504.18574](https://arxiv.org/abs/2504.18574), ICML 2025, >=1B).
- **Minimum depth.**
  - Induction needs 2 layers: Olsson et al. via the `context.md` lane.
  - TinyStories (up to ~80M): 1 layer "struggle[s] quite substantially with following instructions", and 2 layers are "sufficient for a certain extent". Facts rely on embedding dimension, context tracking on layer count ([arXiv 2305.07759](https://arxiv.org/abs/2305.07759), per `context.md`).
  - A September 2026 toy study finds any circuit that solves recall among distractors "needs two layers" ([arXiv 2609.16183](https://arxiv.org/abs/2609.16183); d of 32 or 128, not peer reviewed).
- **The correction failure has a mechanism at toy scale.**
  - The task: in "multi-query repeated associative recall" (MQRAR), variables are re-assigned and later queries must return the latest value.
  - Result: 2-layer, d=256, one-head models with softmax+RoPE handle up to 128 key-value pairs, while stick-breaking attention handles 192. In the softmax model the head is "distracted ... attending to the first instance of 'E' rather than the more recent one" ([arXiv 2410.17980](https://arxiv.org/abs/2410.17980), ICLR 2025; LM results there are at 350M-3B).
  - The unaudited probe (`lanes/probe.md`, hint only) reports that all eight 90M-600M chat models put more probability on the stale value after a correction plus one distractor turn. The probe also says post-training is the first-order problem at 135M. So the mechanism is a hypothesis, not a diagnosis.

### 1.5 MLPs: key-value memory, n-gram detectors, and a write path

- **Geva et al. 2021.** FFN layers act as key-value memories; "lower layers tend to capture shallow patterns, while upper layers learn more semantic ones". Measured on the 16-layer Baevski and Auli WikiText-103 LM ([arXiv 2012.14913](https://arxiv.org/abs/2012.14913)).
- **Voita et al.** In OPT 125M-66B, many early-layer FFN neurons are dead. Many live ones are token or n-gram detectors ("indicator functions"), and some are positional ([arXiv 2309.04827](https://arxiv.org/abs/2309.04827)). Early MLPs spend parameters on lexical lookups, which is why n-gram tables (Engram, STEM, bigram hash) can replace them. Under a total-parameter budget those tables still cost parameters.
- **The necessity test at Planck's scale.** "A Controlled Study of Attention-Only Transformers" ([arXiv 2607.18363](https://arxiv.org/abs/2607.18363), Cactus Compute, July 2026, not peer reviewed). Setup: 6M-87M total params, up to 105B tokens of the reasoning-dense synthetic SYNTH corpus, per-arm learning-rate sweeps, 3 seeds on the main pair. Results:
  - Matched parameters: the attention-only "SAN" (20L, d512, no FFN, 24.13M total) against an FFN model (4L, d512, f2048, 24.12M) gives validation loss 2.0685±0.0028 vs 2.0812. The clean seed pairs differ by +0.0055 and +0.0054 nats in the FFN's favour.
  - Matched FLOPs: the FFN model (43M) leads by 0.263 nats.
  - Matched depth: the FFN model (87M) leads by 0.470 nats.
  - Size ladder at 31.5B tokens: the gap is about 0.02 nats from 16M to 57M non-embedding params.
  - Knowledge-dense fineweb-edu: the gap is 0.040 nats (SAN 3.0131 vs FFN 2.9733).
  - Where the gap sits: on "low-context query prediction" tokens. The SAN is better on answers retrievable from the prompt: sciq (passage in context) 0.742 vs 0.661 at 105B. The FFN is better on out-of-distribution recall: LAMBADA accuracy 0.134 vs 0.111.
  - Mechanics: QK-norm is mandatory (removing it diverges). Sandwich norm helps (-0.009). The depth optimum at iso-parameters is 20 layers at d512. The SAN costs about 2x FLOPs per token at 2k context.
  - **My caveats:**
    - The iso-parameter FFN control is only 4 layers deep. MobileLLM and Kaplan both say shallow is worse at fixed parameters, so the SAN may partly be winning "depth vs 4 layers", not "attention vs FFN".
    - The data is synthetic.
    - MMLU-class tasks are at chance.
- **Older evidence.**
  - Persistent-memory "all-attention" layers replaced the FFN without loss. Enwik8: 39M at 1.01 bpc vs 1.02. WikiText-103: 133M at test perplexity 20.6 vs 24.0 for a 151M Transformer-XL. With no persistent vectors, "it performs poorly" ([arXiv 1907.01470](https://arxiv.org/abs/1907.01470)).
  - One shared wide FFN plus no decoder FFNs: 0.9 BLEU better than Transformer Big with about 40% fewer parameters (machine translation, Transformer-Big scale; [arXiv 2309.01826](https://arxiv.org/abs/2309.01826)).
- **Shape insensitivity.** Kaplan et al.: at fixed non-embedding N, "the loss varies only a few percent over a wide range of shapes". The feed-forward ratio sweep was at 50M and the head-dimension sweep at 25M non-embedding params, on WebText2 ([arXiv 2001.08361](https://arxiv.org/abs/2001.08361), Fig. 5). The MLP ratio is a second-order knob for loss. Whether it is second-order for chat skill is untested.

### 1.6 Norms, residual width and the output head

- **Head rank = d.**
  - Godey et al. 2024: models up to 410M saturate, and performance degrades when head rank is below ~1000 (Pythia; `arch.md` 1.3).
  - Godey and Artzi 2026 (COLM): the head also suppresses "95-99% of the gradient norm". Setup: a 2B model with a factorized head of rank D from 32 to 4096, on ~11B tokens. D=4096 reaches D=32's final loss in 700M tokens ("x16"), and D=2048 vs 4096 still differs by +0.55 average ([arXiv 2603.10145](https://arxiv.org/abs/2603.10145)).
  - An August 2026 causal test at 2.26M params (6L, width 96, BPE-8192, WikiText-2, 5 seed pairs): a half-rank forward-factorized head costs +0.1795 nats, while a backward-only rank cut costs +0.0586 ([arXiv 2608.16671](https://arxiv.org/abs/2608.16671)). Expressivity is the bigger cost, and it is real even at tiny scale.
  - A contrary 2026 study (Kulkarni et al., [arXiv 2602.20433](https://arxiv.org/abs/2602.20433), per `arch.md`) finds that low rank co-occurs with saturation rather than causing it.
  - My reading for Planck: at d=320-512 the head is rank-limited whatever you do. A small vocabulary lowers the number of classes competing for that rank, and a wider d is the direct fix, but d costs about V x d in the head plus about 12d² per layer. This trade-off is the main reason candidate C is wide.
- **Residual stream width without parameters.**
  - Hyper-connections widen the residual stream into n lanes for "almost negligible" extra parameters. Measured at OLMo-1B/7B and OLMoE-1B-7B, so >=1B ([arXiv 2409.19606](https://arxiv.org/abs/2409.19606)); mHC stabilizes them ([arXiv 2512.24880](https://arxiv.org/abs/2512.24880)).
  - In-scale evidence exists only inside loops: raising φ from 0.45 to 0.65 (section 2.2).
  - DenseFormer's depth-weighted averaging lets a 48-block model (378M) match a 72-block one: perplexity 17.84 vs 17.82 on OpenWebText2 ([arXiv 2402.02622](https://arxiv.org/abs/2402.02622), >=350M).
- **Norm placement.**
  - For shared-layer models, MoEUT's "peri-layernorm" (norm only before Q/K, routers and the final classifier, not on the main path) is what makes layer sharing competitive ([arXiv 2405.16039](https://arxiv.org/abs/2405.16039)).
  - For deep attention-heavy stacks, QK-norm is required and sandwich norm helps ([arXiv 2607.18363](https://arxiv.org/abs/2607.18363)).
  - Max's recipe already has QK-norm and a 1/sqrt(depth) scaling (`arch.md`).

---

## 2. Designs that shift the budget

### 2.1 Attention-heavy or MLP-light transformers

| Evidence | Scale | Result |
|---|---|---|
| Attention-only SAN vs FFN at matched params | 24M total, 105B tokens, synthetic | 0.006 nats behind; better on passage-grounded QA, worse on low-context/OOD recall ([2607.18363](https://arxiv.org/abs/2607.18363)) |
| Same, knowledge-dense data | 24M, 31.5B fineweb-edu | 0.040 nats behind (same source) |
| Physics 3.3 no-MLP / quarter-MLP | <10M to ~0.5B, synthetic | same 2 bits/param at 1000 exposures; no-MLP >1.5x worse at 100 exposures, quarter-MLP negligible ([2404.05405](https://arxiv.org/abs/2404.05405)) |
| Kaplan FFN-ratio sweep | 50M non-emb | "a few percent" loss range ([2001.08361](https://arxiv.org/abs/2001.08361)) |
| All-attention with persistent vectors | 39M-133M, enwik8/WT103 | matches or beats FFN models ([1907.01470](https://arxiv.org/abs/1907.01470)) |
| MobileLLM SwiGLU vs vanilla FFN | 125M | +1.3 zero-shot for SwiGLU ([2402.14905](https://arxiv.org/abs/2402.14905), `arch.verify.md`) |

- **What it buys Planck.** A bigger share of parameters goes to routing and context grounding, which is what multi-turn chat needs. The measured penalty is concentrated where Planck plans to rely on retrieval anyway: low-context, knowledge-heavy tokens.
- **Cost.**
  - More FLOPs per parameter: attention's quadratic term, about 2x at 2k for the SAN.
  - More KV cache per parameter: 57 MB at 2k for candidate B vs 12 MB for A (my computation).
  - Slower knowledge learning at low exposure counts. For a model meant to hold little knowledge that is mostly a feature, but it also covers word meanings.
- **Cheapest test:** ledger L1 and L2.

### 2.2 Layer sharing and looping: gain per unique parameter

| Method | Smallest scale measured | Gain per unique param | Notes |
|---|---|---|---|
| ALBERT all-shared (encoder) | BERT-base shape | all-shared E=128 scores 80.1 with 12M params vs 81.6 not-shared with 89M; attention-only sharing is free | [1909.11942](https://arxiv.org/abs/1909.11942) |
| MobileLLM-LS (each block twice) | 125M, 350M; 0.25-1T tokens | +0.4 to +1.1 zero-shot at 2x FLOPs; 3-4x repetition diminishes | `arch.verify.md` row 14 |
| **MoEUT** (shared groups of G=2 layers, σ-MoE MLP + SwitchHead attention experts, peri-LN) | **44M**, C4, ~6.5B tokens (64 x 1024 x 100k), 8k vocab | vs 45M dense (16L, d412): PPL 18.97 to 18.30, BLiMP 73.5 to 78.2, avg 49.4 to 51.1; peS2o 11.46 to 11.09; SlimPajama 16.42 to 15.77; the non-shared σ-MoE "performs significantly worse"; G=2 optimal at 244M; seed count not stated | [2405.16039](https://arxiv.org/abs/2405.16039) Table 1, Fig. 4, 6 |
| Iso-depth looped scaling law | 10.9M-1.1B unique non-emb, compute-optimal (0.34-7.8B tokens), FineWeb-Edu, 116 runs | one block looped r times is worth r^0.46 blocks (R² 0.997); truncated BPTT drops φ to 0.38; hyper-connections with 2 lanes raise φ to 0.65 | [2604.21106](https://arxiv.org/abs/2604.21106) |
| RecursiveGPT (one shared block, factorized embeddings) | 27.6M, 10M-word BabyLM subset, 3 seeds | beats a 41.1M standard model: BLiMP 70.95 vs 68.97, avg 45.80 vs 44.93; improves with R up to 16, then declines | [2608.26973](https://arxiv.org/abs/2608.26973). Data-limited regime |
| Looped GPT-BERT (4 physical layers x 12) | 12.18M, 7.48M words | BLiMP preserved (71.19 vs 70.54), but **entity tracking drops** (non-looped 22.87 / 38.66 vs mostly 13-18 looped) | [2609.09691](https://arxiv.org/abs/2609.09691); authors: shared layers "may leave less representational space ... to maintain multiple entities". Single-seed tables |
| GRT (gated recurrent) iso-param | 124M, 9.8B tokens | -0.11 nats for 8.5x compute | `arch.verify.md` row 13 |
| Mixture-of-Recursions | 135M | "underperforms the vanilla model at the smallest model size (135M)" | [2507.10524](https://arxiv.org/abs/2507.10524) |
| Ouro (LoopLM) | 1M-40M synthetic; 1.4B/2.6B real | knowledge capacity unchanged (~2 bits/param); knowledge manipulation improves | [2510.25741](https://arxiv.org/abs/2510.25741) |
| Relaxed Recursive | >=1B, uptrained from Gemma | not applicable from scratch at 20M | [2410.20672](https://arxiv.org/abs/2410.20672) |
| Huginn | 3.5B | recurrence adds marginal GSM8K | `frontier.md` |
| Looped-MoE | up to 305M active / 711M stored, FineWeb 10B | Core-9 average 39.6 vs 38.7 dense, driven mostly by BoolQ (63.9 vs 51.4); weak | [2605.09165](https://arxiv.org/abs/2605.09165) |
| MoRE (shared expert pools across adjacent layers) | 114M-1.15B | lower perplexity than MoE and weight-sharing baselines "at matched compute and parameter budgets" (abstract; numbers not checked) | [2609.18176](https://arxiv.org/abs/2609.18176), COLM 2026 |

- **Pattern.**
  - Plain looping of a dense block buys roughly 1.3-1.9x effective parameters for 2-4x FLOPs (φ = 0.46).
  - Adding routing diversity to the shared block (MoE experts, hyper-connection lanes, per-loop depth embeddings) is what turns sharing into a win at matched parameters (MoEUT at 44M; φ = 0.65).
  - Sharing everything hurts the one skill multi-turn chat needs most, entity/state tracking, at 12M (single-seed BabyLM evidence). That argues for unique prelude and coda layers and a shared core, not a fully shared stack.
- **Arithmetic for candidate C** (my computation from the φ fit, which was measured compute-optimal and is extrapolated here to overtraining). Two unique core layers looped 4x inside 6 unique layers give an effective body of 20.5M at φ=0.46 and 23.5M at φ=0.65, vs 15.8M unique.

### 2.3 Hybrids that keep an exact-recall path

- **Recall vs state size.** Fixed-state mixers trade recall against state size along one Pareto curve; attention sits at the top ([Based, arXiv 2402.18668](https://arxiv.org/abs/2402.18668)).
  - Phone-book lookup: Pythia-410M beats Mamba-2.8B once the book has ≥70 entries (`arch.verify.md` row 8).
- **Hybrid ratios at 340M and 1.3B** (20B/100B tokens, 72 models; [arXiv 2507.06457](https://arxiv.org/abs/2507.06457)):
  - Language modeling is flat across linear:full ratios.
  - Recall rises with more full attention. Average recall is 0.256 for pure linear, 0.338 at 24:1, 0.397 at 3:1, and about 0.42 for a Transformer.
  - The authors recommend 3:1 to 6:1.
- **Hybrids can win loss at 350M** ([arXiv 2510.04800](https://arxiv.org/abs/2510.04800), Meta, 60B tokens, 0.35B non-embedding):
  - DCLM NLL: Transformer 2.882, Mamba 2.880, 1:1 inter-layer hybrid 2.850, 1:5 hybrid 2.860.
  - So at 350M a hybrid beats pure attention on loss by about 0.03 nats. No measurement exists at 20M.
- **Toy decomposition (September 2026)** ([arXiv 2609.16183](https://arxiv.org/abs/2609.16183); about 29k params, d of 32 or 128, not peer reviewed):
  - The short causal convolution is "the dominant lever" for recall in fixed-state cells, worth about +0.45 accuracy.
  - Retrieving 4 pairs from a haystack of distractors hits "a wall" at chance for delta-rule, diagonal and RWKV-7 cells alike, "while a 2-layer attention control solves the task at 1.0".
  - A distance curriculum breaks the wall (0.021 to 1.000).
- **For Planck at 20M.**
  - A Gated DeltaNet or Mamba2 layer has roughly the same parameter count as an attention layer at the same d, so hybrids save no parameters.
  - What they save is KV memory and long-context FLOPs. At 20M and 2k context, the KV cache is 6-57 MB depending on the candidate (my computation), so that saving is irrelevant.
  - Keep full attention in every layer. The only hybrid idea worth a test is functional, not economic: a delta-rule layer's "overwrite the value for this key" update is exactly the correction semantics chat needs (ledger L7, untested).

### 2.4 Memory layers and product-key memory

- Memory+ at a 134M base: 937M total params, 1T tokens, beats dense 373M on TriviaQA (`arch.verify.md` row 11).
- At a 151M base with ~2B total, MoE beat PKM (TriviaQA 33.27 vs 24.66) (UltraMem, `frontier.verify.md`).
- Both multiply stored parameters 6-13x to store facts.
- **Under Planck's total-parameter accounting, and its plan to keep facts outside the model, these are the opposite of the goal.** Excluded from the 20M design. Retrieval at inference is the knowledge path.

### 2.5 Structured and low-rank matrices

- **Compute Better Spent** ([arXiv 2406.06248](https://arxiv.org/abs/2406.06248), ICML 2024). GPT-2 with Block Tensor-Train layers beats dense per total compute. The authors say the models "perform similarly when controlling for non-embedding compute", so "the improvement primarily comes from reducing the compute spent in the language modeling head".
- **Einsum search** ([arXiv 2410.02117](https://arxiv.org/abs/2410.02117), NeurIPS 2024). "full-rank structures that maximize parameters per unit of compute perform the best". Parameter-sharing structures (Kronecker, Tensor-Train) scale worse per FLOP. One hint in Planck's direction: on 8x8 autoregressive pixel modeling, most structured layers "outperform dense at small scales", which the authors attribute to "larger embedding dimensions than dense layers for a fixed parameter budget".
- **Verdict.**
  - No language-model evidence shows better loss per parameter from structured layers.
  - The evidence that exists is per FLOP, and it favours more parameters per FLOP, the opposite of what a parameter-capped model wants.
  - The one plausible Planck use is a structured wide residual stream, the same goal hyper-connections pursue more cheaply. Low priority; no ledger entry beyond L15.
  - Ternary or low-bit weights change bytes, not parameters (Spectra, ParetoQ; `frontier.md` 1.7). Under a parameter metric they are irrelevant, and int4 halves knowledge capacity.

### 2.6 Non-transformer designs (RNN, SSM, RWKV, linear attention)

- **Provable and measured recall limits** for fixed-state models:
  - Jelassi: a fixed-state model cannot copy strings longer than its state allows.
  - Zoology: gated convolutions need d ≥ sequence length for MQAR.
  - The toy distractor wall above.
- **RWKV-7** claims state tracking over all regular languages and strong 3B results. Its 0.1B and 0.4B models were trained from older RWKV-5 checkpoints ([arXiv 2503.14456](https://arxiv.org/abs/2503.14456)), so no clean from-scratch evidence exists at 20M.
- **Verdict.** For Planck, a pure recurrent model buys nothing in parameters and risks the one capability multi-turn chat cannot lose. Do not use one as the main backbone.

---

## 3. What this means for Planck at 10M-150M

### 3.1 Design rules that follow from sections 1-2

1. **Softmax attention in every layer.** The minimum is 2 layers for induction, and the practical shape is 12-20 layers. QK-norm is mandatory.
2. **Tied 8k vocabulary across the whole curve**, so evals are token-comparable. This is 16% of 20M and 8% of 60M.
3. **Plain MLP (ReLU²) at ratio 1-2.25, or SwiGLU at matched parameters.** The form matters little at matched parameters (SAN, Kaplan). Plain MLP learns stored facts 1.3x faster at 100 exposures (Physics 3.3), and SwiGLU gave +1.3 on zero-shot at 125M (MobileLLM). A/B it rather than assume.
4. **Canon-style short causal convolutions**: under 1% of parameters. The evidence is synthetic at small scale (`arch.md` 1.6), backed by the toy finding that convolution carries recall.
5. **No memory tables, no n-gram hash tables, no value embeddings** inside the 20M. They are storage.
6. **Budget for head rank.** Each step of d from 384 to 512 costs about 1M in the tied head at 8k, plus about 12d² per layer. Measure the head-rank penalty directly (L9) before committing to d=384.
7. **Evaluate on behaviour probes, not loss.** Attention-output modifications within 2-3% of baseline loss dropped 6-16 downstream points at 1.2B (`arch.verify.md` row 17). The SAN study shows the same split: loss parity, opposite signs on grounded vs OOD tasks.

### 3.2 Three candidate Planck architectures at 20M total parameters

Parameter counts are my computation. Each layer has an attention block with a headwise sigmoid output gate, QK-norm and two RMSNorms; Canon convolutions (kernel 4) sit at two sites; embeddings are tied; head_dim is 64. KV cache is fp16 at 2,048 tokens. FLOPs are forward only, counting 2 x matmul params plus causal attention scores.

| | **A: Dense deep-thin control** | **B: Attention-heavy** | **C: Looped wide core** |
|---|---|---|---|
| Layers | 12 unique | 19 unique | 6 unique, 12 applied: prelude 2, core 2 x 4 loops, coda 2 |
| d_model | 384 | 384 | 512 |
| Heads (Q / KV) | 6 / 2 | 6 / 6 (full MHA) | 8 / 2 |
| MLP | SwiGLU 864 (2.25x) | ReLU² plain, 384 (1x) | SwiGLU 1280 (2.5x) |
| Attention share of body | 28% | 67% | 25% |
| Vocab (tied) | 8,192 | 8,192 | 8,192 |
| Embedding | 3.15M (15.8%) | 3.15M (15.7%) | 4.19M (21.0%) |
| Body | 16.74M | 16.93M | 15.78M (effective 20.5-23.5M by φ) |
| **Total** | **19.88M** | **20.08M** | **19.98M** |
| Fwd GFLOP/token @2k | 0.059 | 0.070 | 0.097 |
| KV cache @2k | 12 MB | 57 MB | 12 MB |
| Extras | none beyond the recipe | QK-norm (required), sandwich norm | hyper-connections (2 lanes) at loop boundaries, per-iteration norm gains, learned per-loop depth embedding, full BPTT |

**A: the well-tuned baseline every idea is measured against.**
- 12 layers at d=384 is inside Kaplan's shape-insensitive band.
- It mirrors MoEUT's dense 44M baseline (16L, d412) at half the size.
- It keeps enough depth for 2-hop circuits.
- Alternative shape for a depth check: 16L, d320, 5 heads / 1 KV head, SwiGLU 864, 19.90M total (my computation).
- A should carry Max's validated recipe: NorMuon, gated attention, value residual, 1/sqrt(depth) norm scaling.

**B: the bet that chat skill is routing.**
- It spends two thirds of the body on attention. That follows the SAN finding (attention-only at parity, better on grounded answers) and ALBERT's finding that routing weights are cheap.
- It keeps a thin MLP because removing MLPs entirely costs more than 1.5x in low-exposure learning (Physics 3.3) and on low-context tokens (SAN). A chat model needs those for word meanings and conversation openers.
- Its pure attention-only variant, 28L/d384 at 19.84M, is ledger L2.
- Risk: B's evidence comes from one preprint whose FFN control was shallow. B must beat A, not a 4-layer model.

**C: the bet that reuse plus width beats unique thin layers.**
- d=512 raises head rank by a third and widens the residual stream.
- Two hyper-connection lanes widen it further for almost no parameters (φ 0.45 to 0.65 in the looped law).
- The shared core is G=2 layers, the best grouping in MoEUT.
- The unique prelude and coda address the entity-tracking loss seen when everything is shared (Looped GPT-BERT).
- Cost: 1.6x A's FLOPs per token (my computation). The main risks are MoR's loss at 135M and CART's parity loss (`arch.verify.md`).
- Stage-2 upgrade if C wins: MoEUT-style σ-MoE experts in the shared core, the strongest in-scale "more per parameter" result found (L5).

### 3.3 The curve

The same recipe scales in d and L with the vocab fixed at 8k (my computation, A-style dense):

| Size | Shape | Emb / body / total | Fwd GFLOP/token @2k |
|---|---|---|---|
| 10M | 12L, d256, 4 heads / 1 KV head, SwiGLU 640 | 2.10 / 7.91 / 10.01M | 0.033 |
| 20M | A above | 3.15 / 16.74 / 19.88M | 0.059 |
| 60M | 16L, d576, 9 heads / 3 KV heads, SwiGLU 1536 | 4.72 / 56.80 / 61.52M | 0.161 |
| 150M | 22L, d768, 12 heads / 4 KV heads, SwiGLU 2048, V=16k | 12.58 / 138.79 / 151.37M | 0.372 |

- At 150M, a 16k vocabulary is defensible (vocab law, `arch.verify.md` row 6), but keeping 8k keeps the curve on one tokenizer. Settle this with L8 at 60M.
- Whichever of A/B/C wins at 20M should be re-run at 10M and 60M with 2-3 seeds before the curve's headline claim.
- The candidates differ in which failure shows first. Predictions to check:
  - B should hold recall and grounding down to 10M but lose word-meaning tokens.
  - A should degrade smoothly.
  - C should gain most at 10M, where unique parameters are scarcest, if φ holds in the overtrained regime.

### 3.4 Testing order (cheap to expensive)

1. **Toy synthetic probes** (2-4 layers, d 64-256, minutes each on the Mac):
   - MQAR for recall;
   - MQRAR for corrections and overwrites;
   - a boxes-style entity-tracking task;
   - a planted-fact-in-dialogue task.

   These separate mechanisms (L6, L7, L10) before any real training.
2. **5-10M proxies** on a chat-heavy mix, 0.2-0.5B tokens, 2-3 seeds. At the lane's contested Mac rates of about 5-11k tok/s at 18.5M (`compute.verify.md` row 9), a 10M proxy at 0.3B tokens is roughly half a day to a day (my arithmetic). These runs are for A vs B vs C, vocab, and head rank.
3. **20M at 1B+ tokens** for the winner and A, then the curve.

---

## 4. Ledger ideas

Each entry gives: hypothesis; why it could work; cheapest test on the Mac; what counts as a win. "Proxy" means a 5-10M model with the same recipe at 0.2-0.5B tokens and 2-3 seeds. Probes are the section 3.4 battery.

**L1. Attention-heavy, MLP-light (candidate B vs A) at matched total parameters.** *tweak-of-known*
- Hypothesis: at matched parameters, shifting the body from MLP to attention improves multi-turn recall and grounding and lowers closed-book fact recall.
- Why: SAN at 24M has loss parity and is better on passage-grounded QA; ALBERT shows attention is reusable routing.
- Test: 8M proxies of A and B (d=256).
- Win:
  - B's planted-fact and follow-up accuracy beats A by more than 2x the seed spread;
  - held-out chat loss is within 0.01 nats;
  - closed-book fact probes favour A (confirming the storage/routing split).

**L2. Pure attention-only (SAN) against a deep dense control, not a 4-layer one.** *tweak-of-known*
- Hypothesis: the SAN's 0.006-nat parity partly reflects its shallow FFN control; against a 12-16 layer dense model at equal parameters the gap is larger.
- Why: every depth sweep at this size (MobileLLM, Kaplan's shape band) says 4 layers is too shallow.
- Test: 5M proxy with SAN 20L vs dense 8L vs dense 12L, all with QK-norm and sandwich norm.
- Win for the SAN: within 0.01 nats of the best dense arm and ahead on grounded probes. Otherwise B keeps its thin MLP.

**L3. Share routing, keep content: tie W_q and W_k across groups of 2-4 layers; keep W_v, W_o and MLPs unique.** *untested*
- Hypothesis: Q/K are low-information routing that can be reused across depth. The ~10% of the budget saved can buy 2-3 extra layers.
- Why: ALBERT's attention-only sharing cost nothing; in the SAN study, Q/K freeze in the first quarter of training while write-path matrices keep accumulating rank.
- Test: proxy with A vs A plus Q/K-sharing, parameters reinvested in depth.
- Win: lower held-out loss and no loss on MQAR or entity tracking, across 3 seeds.

**L4. Looped wide core with hyper-connections (candidate C vs A).** *tweak-of-known*
- Hypothesis: 6 unique layers looped to 12 applications at d=512, with 2 hyper-connection lanes, beat 12 unique layers at d=384 on loss and on composition probes, and match A on entity tracking.
- Why: φ=0.65 with hyper-connections (10.9M-1.1B); a wider head rank; MoEUT's G=2 result; unique prelude and coda avoid Looped GPT-BERT's entity-tracking loss.
- Test: 10M proxies of A, C without hyper-connections, and C with them; 2 seeds each.
- Win: C with hyper-connections beats A by more than 2x the seed spread on held-out loss, and is not worse on entity-tracking or correction probes. Also report the loss per FLOP gap.

**L5. MoEUT-style shared-layer MoE.** *known-apply*
- Hypothesis: MoEUT's 44M matched-parameter gain (perplexity -0.67, BLiMP +4.7) reproduces at 10-20M on chat data.
- Why: the strongest in-scale "more capability per unique parameter" result found.
- Test: port the reference implementation ([github.com/robertcsordas/moeut](https://github.com/robertcsordas/moeut), linked from the paper; Triton kernels will not run on MPS, so use a slow reference path) at a 10M proxy.
- Win: a matched-parameter perplexity gain of at least 3% with no entity-tracking loss. If the MPS slowdown exceeds 5x, defer to the Titans.

**L6. Recency-biased attention for corrections (stick-breaking, or a forgetting gate, on some or all heads).** *tweak-of-known*
- Hypothesis: part of the correction failure comes from softmax+RoPE retrieving the first matching key; a recency-biased head retrieves the latest.
- Why: MQRAR at toy scale (128 vs 192 pairs, "distracted" softmax head), plus the probe's correction failure in every model up to 0.6B (unaudited hint).
- Test:
  - toy MQRAR, 2 layers at d=128, minutes;
  - then a proxy on dialogue data with correction probes ("moved from 3 pm to 4 pm").
- Win: at least 1.5x MQRAR capacity at the same size, and +10 points on correction probes with at most 0.01 nats loss penalty.

**L7. One delta-rule (Gated DeltaNet) layer as a "current value" register among attention layers.** *untested*
- Hypothesis: the delta rule's overwrite semantics complement attention's exact recall for updates and corrections.
- Why: delta-rule cells implement "replace the value stored under this key" by construction; attention has no native overwrite.
- Test: toy MQRAR plus entity tracking with attention-only vs one of 4 layers swapped for GDN at matched parameters; then the proxy.
- Win: better correction and entity tracking with no MQAR loss.

**L8. Vocabulary sweep at 20M (4k / 8k / 16k tied), compared at equal bytes.** *known-apply*
- Hypothesis: 8k sits within 1% of the best bits-per-byte on chat data at 20M.
- Why: the vocab-law extrapolation gives 3-4k compute-optimal; overtraining pushes it up; 16k costs 31% of the model.
- Test: 10M and 20M proxies, bits-per-byte on held-out multi-turn chat, plus planted-recall at equal bytes of context.
- Win: choose the lowest bits-per-byte; adopt 8k if it is within 1% of the best.

**L9. Head-rank probe.** *tweak-of-known*
- Hypothesis: at 20M the head's rank costs measurable loss and repetition.
- Why: a half-rank head cost 0.18 nats at 2.26M; the gradient bottleneck showed x16 at 2B.
- Test: fixed d=384 body, factorized head of rank 128, 256 or 384, plus a mixture-of-softmaxes-2 head; 10M proxy. Measure loss and repetition rate in greedy multi-turn generation.
- Win (for the finding): if going from rank 256 to 384 is worth more than 0.03 nats, rank is binding. Then prefer candidate C's d=512 or a mixture-of-softmaxes head.

**L10. Canon layers at 5-20M.** *known-apply*
- Hypothesis: kernel-4 residual convolutions before attention and MLP improve recall and entity-tracking probes for under 1% of parameters.
- Why: Canon synthetic results at 8L512D, and the toy finding that "the convolution dominates" recall.
- Test: toy probes, then A vs A plus Canon on the proxy.
- Win: gains on MQAR, MQRAR and entity tracking beyond the seed spread, with at most 1% more parameters.

**L11. MLP form at matched parameters: ReLU² plain vs SwiGLU.** *known-apply*
- Hypothesis: at 20M on chat-heavy data the difference is under 0.01 nats, but plain MLP learns word meanings (low-exposure knowledge) faster.
- Why: gated MLP is 1.3x worse at 100 exposures (Physics 3.3), while SwiGLU is +1.3 zero-shot at 125M (MobileLLM).
- Test: proxy A with each MLP form; rare-word cloze plus loss.
- Win: pick the better one on rare-word cloze if loss is within 0.01.

**L12. Does architecture change what gets stored?** *tweak-of-known*
- Hypothesis: A, B and C store about the same bits/param of planted facts (Physics 3.3 method); their chat-skill differences come from computation structure, not storage.
- Why: section 1.1. If true, route (1) must be carried by data, and the curve's knowledge axis can be predicted from parameter count.
- Test: 5M proxies on a mix of synthetic bios (bioS-style, known bits) and dialogues; measure fact capacity and dialogue probes.
- Win (for the finding): capacity ratios within 20% across A, B and C, while dialogue probes differ.

**L13. Minimum depth for multi-turn skills.** *tweak-of-known*
- Hypothesis: follow-ups and planted recall need at least 4 layers; corrections and state tracking need 8 or more at d=256.
- Why: induction needs 2 layers, TinyStories says 2 layers suffice "to a certain extent", and 2-hop composition fails even at 1.3B.
- Test: 4, 8 and 12 layer models at a fixed 5M parameters on the probe battery.
- Win: a depth floor per skill. This sets the smallest point on the Planck curve that can pass each skill.

**L14. Unique prelude and coda vs fully shared stack.** *tweak-of-known*
- Hypothesis: sharing hurts entity tracking because first and last layers need unique parameters.
- Why: Looped GPT-BERT lost entity tracking with 4 physical layers shared 12x; MoEUT and Ouro keep some structure.
- Test: 5M proxy: 1 block x 12 vs prelude 1 + shared 2 x 5 + coda 1 at matched parameters, on entity tracking.
- Win: the prelude/coda version recovers the non-looped model's entity-tracking score.

**L15. Hyper-connections (2-4 lanes, mHC-style constraint) in a non-looped model at 10-20M.** *tweak-of-known*
- Hypothesis: a wider residual stream for almost no parameters improves loss and composition probes at tiny d.
- Why: hyper-connections are measured at >=1B; at small scale only inside loops (φ 0.45 to 0.65).
- Test: A vs A plus hyper-connections on the proxy.
- Win: a loss gain above 2x the seed spread, with no probe regressions.

---

## 5. Open questions

1. Does the SAN's parity with FFN models survive against a properly deep dense control, and on conversational data rather than SYNTH? Its only knowledge-dense data point is already 7x worse: 0.040 vs 0.0055 nats.
2. In the heavily overtrained regime Planck will run in (thousands of tokens per parameter), is a recurrence worth more or less than φ=0.46? The fit was compute-optimal (0.34-7.8B tokens at 10.9M-1.1B).
3. Is the correction failure mainly a mechanism problem (softmax picking the first matching key) or a post-training problem? The unaudited probe points to post-training at 135M. L6 and a post-training A/B together answer it.
4. How much does the output head's rank (d = 320-512) cost in generation quality (repetition, entropy collapse) at 10-20M, separately from loss? No study measures generation.
5. Where do word meanings live in a 20M model, and how many parameters does a restricted everyday chat lexicon need? The 12M-bit human estimate is for a 40,000-word adult lexicon.
6. Looping hurt entity tracking at 12M (single-seed BabyLM data) but helped grammar. Is that loss removed by unique prelude/coda layers, hyper-connections or per-loop embeddings, or is state tracking fundamentally parameter-hungry?
7. MoEUT's gain at 44M came with an 8k vocabulary and 6.5B tokens of C4. Does it hold on chat data and at 10-20M, and is it practical on MPS without Triton kernels?
8. Hybrids beat pure attention by about 0.03 nats at 350M (Meta). Is there any analogous gain at 20M, where attention is already cheap, or is the 350M gain a long-context or FLOP effect?

---

## Sources read for this report (new beyond the first run)

- Physics of LMs 3.3, Sec. 7 and App. B: https://arxiv.org/abs/2404.05405
- Nichani, Lee, Bietti, factual recall via associative memories: https://arxiv.org/abs/2412.06538
- Controlled study of attention-only transformers (SAN): https://arxiv.org/abs/2607.18363
- Mollica and Piantadosi, 1.5 MB of language: https://pmc.ncbi.nlm.nih.gov/articles/PMC6458406/
- ALBERT Tables 3-4: https://arxiv.org/abs/1909.11942
- Augmenting self-attention with persistent memory: https://arxiv.org/abs/1907.01470
- One Wide Feedforward: https://arxiv.org/abs/2309.01826
- Kaplan scaling laws, Fig. 5: https://arxiv.org/abs/2001.08361
- Geva 2021: https://arxiv.org/abs/2012.14913 ; Voita neurons: https://arxiv.org/abs/2309.04827
- Zoology Sec. 4: https://arxiv.org/abs/2312.04927 ; Based: https://arxiv.org/abs/2402.18668
- Gather-and-Aggregate: https://arxiv.org/abs/2504.18574
- Anatomy of associative recall in fixed-state recurrences: https://arxiv.org/abs/2609.16183
- Stick-breaking attention (MQRAR): https://arxiv.org/abs/2410.17980
- Hybrid linear attention analysis: https://arxiv.org/abs/2507.06457 ; Meta hybrid analysis: https://arxiv.org/abs/2510.04800
- MoEUT: https://arxiv.org/abs/2405.16039 ; MoRE: https://arxiv.org/abs/2609.18176
- Iso-depth looped scaling law: https://arxiv.org/abs/2604.21106
- Sparse layers for looped LMs: https://arxiv.org/abs/2605.09165
- Looped GPT-BERT: https://arxiv.org/abs/2609.09691 ; RecursiveGPT: https://arxiv.org/abs/2608.26973
- Hyper-connections: https://arxiv.org/abs/2409.19606 ; mHC: https://arxiv.org/abs/2512.24880 ; DenseFormer: https://arxiv.org/abs/2402.02622
- LM head gradient bottleneck: https://arxiv.org/abs/2603.10145 ; causal test: https://arxiv.org/abs/2608.16671
- Compute Better Spent: https://arxiv.org/abs/2406.06248 ; Einsum structured search: https://arxiv.org/abs/2410.02117
- Fixed binary token codes: https://arxiv.org/abs/2605.09751
- RWKV-7: https://arxiv.org/abs/2503.14456
