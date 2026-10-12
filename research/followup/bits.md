# Follow-up track: information per parameter

Track question: how much information does a trained network store per parameter, why does training fill only about 2-4 of the 16 bits a bf16 weight has, what is known to raise it, would fp64 or "more precise parameters" help, and how can Max measure bits/param cheaply on the M5 Mac.

Builds on the first run (lanes/arch, training, frontier, data, context and their .verify files). Those lanes already established, and their fact-checks confirmed: Physics of LMs 3.3 gives about 2 bits/param at 1000 exposures and about 1 at 100; int8 is free and int4 loses more than 2x; 7/8 junk costs up to 20x at 100 exposures with a domain token cutting that to 2x; gated MLP is 1.3x worse at 100 exposures; 32-expert MoE loses 1.3x/1.5x; looping does not raise capacity (Ouro, 1M-40M); Memory+ and UltraMem numbers at 134M/151M. This report does not re-derive those. It adds the exact measurement formulas, the 2025-2026 work the first run missed, the theory of why the number is what it is, a direct answer on precision, and a Mac-sized replication.

Scale tags used below: [synthetic, sizes] for controlled fact datasets, [real, sizes] for natural-text pretraining, [theory] for proofs. Anything marked [calc] is my arithmetic from the cited numbers, not a result in a paper.

---

## Bottom line

1. There are two measured ceilings, and neither has been beaten by any training trick: about **2 bits per parameter for extractable facts** (Physics of LMs 3.3: "consistently exceeds 2", never above 2.3, at 1000 exposures, GPT-2-style 1M to 0.5B, synthetic biographies) and about **3.6 bits per parameter for raw memorization of random strings** (Morris et al.: mean 3.51 in bf16 and 3.83 in fp32 across 16 GPT-2 models of 80K to 6.9M parameters, trained with massive repetition). A 2026 adapter study adds 1.3 to 2.8 bits per trainable LoRA parameter on 2M-8M bases. All of this is synthetic and at or below 0.5B.
2. The "2-4 of 16 bits" is not a precision shortfall. A single perceptron stores 2 bits per weight even with infinitely precise weights (0.83 with 1-bit weights); a linear associative memory with random embeddings tops out near 0.72 bits/param [calc from a 2026 sharp threshold]; transformer factual-recall constructions need only O(log d) bits per weight; and bf16 LLM weights carry about 11 bits of Shannon entropy each, of which only 2-4 are information about the training data. The limit is interference between stored items plus the need for robust retrieval, not the number of mantissa bits.
3. What measurably moves bits/param: exposures (1 to 2 bits going from 100 to 1000 exposures), data cleanliness and source tags (up to 20x), architecture in the low-exposure regime (gated MLP -1.3x, Canon layers +10-15%, linear-recurrent layers about +40%, MoE -1.3x to "10x less acquired"), bf16 to fp32 (+9% on raw memorization), and training budget at measurement time (+10-24% going from 16K to 40K steps for adapters). Nothing measured pushes past about 2.3 bits/param on facts.
4. **fp64 would not help.** Doubling precision from bf16 to fp32 bought +9%, not 2x (Morris); int8 already holds the full 2 bits/param (Physics 3.3); the precision scaling law puts 16-bit weights at 99.7% of their "effective parameter" value already (Kumar et al., fitted at 30M-220M); Metal/MLX cannot run fp64 on the GPU at all and TITAN RTX fp64 runs at 1/32 rate. The one precision fix that matters is to keep fp32 master weights (or use stochastic rounding) so small updates are not rounded away in pure bf16.
5. For Planck the headroom is not above 2 bits/param but below it: a small model on web-like data sits far under the ceiling, and a 2025 NeurIPS result shows a bounded-capacity model allocates **zero** capacity to data below a threshold frequency that rises as models shrink (a 70M model memorized almost nothing from a 20% biography mix even at about 3000 exposures per biography). That applies to chat skills as much as to facts: each skill Planck needs must appear often enough to be worth capacity.
6. The strongest new evidence for the Planck bet itself: across 93 open models (135M to 1.6T), factual recall per parameter shows no improvement over time (+0.0013/month, not significant), while reasoning benchmarks at fixed size improve about 2 points per month. Recipes buy skill per parameter; they do not buy facts per parameter.
7. A Mac rig is feasible: compact synthetic biographies (about 20 tokens and 60 bits per person) on a 1M-parameter model cost about 6e7 tokens at 100 exposures (minutes) and 1.2e9 tokens at 1000 exposures (roughly 1-2 hours at an assumed 1-2 TFLOP/s, unmeasured), plus a Morris-style random-string probe at 235K parameters that makes a real fp64 test possible on the CPU. The first eight experiments are listed in section 6.5; the most Planck-relevant one tests whether filling weight memory with facts degrades in-context recall.

---

## 1. Detailed findings: the measurements

### 1.1 Physics of LMs 3.3 (Allen-Zhu and Li, ICLR 2025)

Source: https://arxiv.org/abs/2404.05405 (HTML v1 read in full for this report).

Setup [synthetic, GPT-2 with RoPE and no dropout, 1M to 0.5B params, bioS N = 10K to 20M]. Each person has a name drawn from N0 = 400 x 400 x 1000 candidates and six attributes: birth date (12 x 28 x 200), birth city (200), university (300), major (100), employer (263), working city (determined by the employer), plus a pronoun (2). "Ignoring names, each person contains log2(S0) = 47.6 bits of knowledge." bioS rewrites every biography with fresh template choices and orderings at every exposure; bioS-simple fixes one biography per person; bioD is a parameterized family (N, K attributes, C chunks, L chunk length, D diversity, T vocabulary).

Parameter counting: "we calculate model sizes after excluding all unused tokens in the embedding layer". bioS uses 3,275 GPT-2 tokens, so GPT-2 small counts as 88M, not 124M (Appendix A). Embeddings that are used **are** counted.

Training: AdamW, cosine decay to 0.1x after 1K warmup steps, mixed-precision fp16 ("We also tried bf16 and the results are nearly identical", Remark A.1), weight decay 0.001 to 0.02, lr 3e-4 to 1e-3, and "Language models typically need at least 50K training steps regardless of batch size". Derived token cost: N = 10K at 1000 exposures used batch 24 x 512 tokens for about 140K steps, i.e. about 1.7B tokens, about 172 tokens per biography exposure [calc].

Results relevant to this track (all verbatim-checked):
- Result 1: at 1000 exposures, peak capacity ratio R(F) >= 2 for all sizes 1M to 0.5B, "irrespective of depth or width"; any model with R_max <= 1.8 is near-perfect; no model exceeds 2.3.
- Result 2: bioS-simple (1000 passes over one fixed biography per person) and bioR (LLaMA-2 rewrites, 25 passes) are "also approximately 2, albeit slightly lower". Diverse rewrites "may sometimes improve" capacity; without diversity "the model wastes capacity memorizing sentence structures".
- Result 3: bioD with K, C from 1 to 50, D from 10 to 10,000, L from 1 to 50, T from 20 to 40,000: R(F) >= 2 throughout. The ceiling is format-independent.
- Result 4: at 100 exposures R(F) >= 1 (1M to 0.5B).
- Result 5 and 6: at 1000 exposures, LLaMA and Mistral match GPT-2 (tiny LLaMA < 10M is slightly worse unless weights are tied), and **removing all MLP layers does not change capacity** ("the Attention layers are also capable of storing knowledge"). At 100 exposures, gated MLP costs 1.3x and removing MLPs costs more than 1.5x.
- Result 8: int8 post-training quantization has negligible impact; int4 (GPTQ) cuts capacity "by more than 2x" (0.7 bit/param per the first-run verify). "Language models ... can exceed 1/4 of the absolute theoretical limit" of an 8-bit parameter.
- Result 9: 32-expert MoE (topk 1) loses 1.3x (1000 exposures) and 1.5x (100 exposures) per total parameter.
- Results 10-12: junk and domain tokens (first run covered this; note the junk is random rare biographies, and highly repetitive "junk" is harmless, Result 11).

The exact capacity formula is reproduced in section 6.2.

### 1.2 Morris et al. 2025, "How much do language models memorize?"

Source: https://arxiv.org/abs/2505.24832 (HTML v3 read).

Setup [synthetic, GPT-2 architecture, 1 to 8 layers, d = 32 to 256, 8.04e4 to 6.86e6 params in the capacity table; the abstract's "500K to 1.5B" covers the whole paper including the membership-inference part]. Data are sequences of S = 64 tokens drawn uniformly from V = 2048, so each sequence carries exactly 64 x 11 = 704 bits. Training: Adam, 1e6 steps at batch 2048, single A100, bf16 by default. That is up to about 2e9 sequence presentations per run, i.e. hundreds to hundreds of thousands of epochs depending on dataset size [calc]. Parameter counts include tied embeddings (for example 1 layer, d = 32: 8.04e4 total, most of it the 2048 x 32 embedding) [calc from the table].

Results:
- Capacity per model: bf16 alpha from 2.86 to 3.94 bits/param, fp32 from 3.02 to 4.23; **means 3.51 (bf16) and 3.83 (fp32)**, reported as "between 3.5 and 3.6" in the main text.
- Precision: "an increase in alpha from 3.51 to 3.83 bits-per-parameter on average. This is far less than the actual 2x increase in the bits of theta, indicating that most of the extra model bits added when increasing precision from bfloat16 to float32 are not used for raw storage." (+9% [calc])
- A linear capacity model (alpha = 3.642) predicts total memorization within 1.7-1.8% average error when S or V is varied (Appendix A.1, 0.42M to 0.93M params).
- On real text, models memorize until the dataset's information exceeds capacity; "double descent begins exactly when the data capacity exceeds the model capacity", after which unintended memorization falls and generalization rises.

Why 3.6 and not 2 (my reading, not either paper's claim): Morris counts every bit of arbitrary strings, trains to near saturation with extreme repetition, and uses a tiny vocabulary; Physics counts only attribute bits that must be keyed by a name, measured through a paraphrased format, at a fixed 1000 exposures. Nobody has run both probes on the same models, so the gap is not decomposed (open question 2).

### 1.3 New in 2026: adapter capacity ("How Many Bits Can an Adapter Write?")

Source: https://arxiv.org/abs/2607.21351 (July 2026). [synthetic random strings, GPT-style bases with 2M and 8M non-embedding params pretrained on WikiText-103; also Qwen2.5-0.5B for the privacy part]

Same compression measure as Morris (64 uniform tokens over 2,048 symbols, 704 bits per sequence). Findings:
- LoRA adapters store 1.71 to 2.75 bits per trainable parameter (2M base r = 1: 1.71; 8M r = 1: 2.09; 2M r = 16: 2.75; 8M r = 16: 1.85), versus >= 3.07 for full fine-tuning of the same base at the same budget.
- **Placement beats count**: at a fixed 37K trainable parameters, per-parameter capacity runs from 1.30 bits (attention only) to 2.43 bits (MLP only), "a near-doubling with parameter count held flat".
- The frozen base matters: the same LoRA memorizes 29% of its data on a random base, 79% on a base pretrained on structureless synthetic text, 98% on a WikiText base.
- Measured capacity is budget-dependent: extending 16,000 to 40,000 steps "lifts it by 10-24%".
- Casting a trained adapter to bf16 "costs about one percent of its memorization"; fp16 costs essentially nothing.

Reading for Planck: low-rank or structured parameters store fewer bits each than full-rank ones, and the same parameter budget stores roughly 2x more in MLP directions than in attention directions under a low-rank constraint. (Physics 3.3 Result 5 shows that with full-rank training and 1000 exposures attention alone reaches 2 bits/param too, so this is a low-rank and low-budget effect.)

### 1.4 New from Physics of LMs 4.1 (Canon layers): capacity in the 100-exposure regime

Source: https://arxiv.org/abs/2512.17351 (HTML v1, Appendix A.3 and Sections 5-6). [synthetic "Capo" task = bioS at 100 exposures, N = 50K to 2M, models 1M to 500M, GPT-2 tokenizer restricted to 3,275 tokens, tied embeddings]

- Canon layers (a causal short convolution, about 0.5% extra params) "increase the effective capacity by 10-15% in the controlled 100-exposure pretraining regime" for Llama-style RoPE models; GPT-2-style standard-MLP models "exhibit no capacity loss" but also no reported gain.
- **Linear-recurrent models (Mamba2, GLA, GatedDeltaNet), all with Canon layers, show "a ~40% gain in Capo knowledge capacity compared to full Transformers"** ("Mamba2 ≈ GLA ≈ GDN ≫ RoPE ≈ NoPE (e.g., 1.4x capacity)"), while Transformers reach 2-4x deeper reasoning. This was not in the first run.
- MoE: "a 32-expert transformer may acquire 10x less knowledge in the same 100-exposure regime"; Canon-ABC recovers "at least half of the MoE-induced capacity loss". Note the conflict with Physics 3.3's 1.5x at 100 exposures; the conditions differ (learning-rate choices, sizes) and the paper does not reconcile them.
- ReLU² "slightly improves standard MLP but degrades performance in gated MLP".
- The paper's own caveat: "Training beyond 100 exposures diminishes architectural differences". So these are learning-speed differences for rarely seen items, which is exactly the regime a small model's rare skills and facts live in.

### 1.5 New in 2025: capacity allocation and phase transitions (Gu et al., NeurIPS 2025 spotlight)

Source: https://arxiv.org/abs/2505.18091 (v3, May 2026). [synthetic biographies mixed into FineWeb-Edu, Pythia architecture 14M to 6.9B; main runs 70M and 410M at 32B tokens]

- For a fixed model, below a critical mixing ratio the model "memorizes almost nothing even with extensive training". 70M on FineWeb-Edu + SynBio-320K: near-zero accuracy for r = 0.1 to 0.25, rising only for r > 0.3. 410M on SynBio-1.28M: near zero for r <= 0.3, about 80% at r = 0.4.
- Extending r = 0.2 to 512B tokens, "each biography appears ~3000 times for the 70M model and ~200 times for the 410M model", and accuracy "remains near zero".
- Explanation: an optimal bounded-capacity learner behaves like a knapsack solver and spends capacity where marginal loss reduction is highest; the threshold exposure frequency for a fact to be learned follows a power law in model size (fitted exponent magnitude 1.152; theory predicts alpha + 1 = 1.283 from the web-loss scaling exponent 0.283). Smaller models need each item to be more frequent.
- Fix that works: subsample the dense set so each item is seen more often. 410M: subsampling SynBio-1.28M to 25%, 50%, 56.25% raised accuracy from near zero to 23.5%, 37.5%, 39.8%; going to 62.5% dropped it back to near zero.

This changes how to read "bits/param in practice": a small model's capacity is not filled uniformly; items below a frequency threshold get none of it.

### 1.6 New in 2026: factual capacity of real models (Incompressible Knowledge Probes)

Source: https://arxiv.org/abs/2604.24827 (v2, July 2026). [real, 93 open-weight models 135M to 1.6T used for calibration, 201 models evaluated]

- 1,400 obscure-fact probes in 7 tiers; accuracy is log-linear in parameters (R² = 0.910), about +15.9 points per 10x parameters.
- For MoE, total parameters predict knowledge (R² = 0.67) better than active parameters (R² = 0.41), matching Physics 3.3's "knowledge tracks total params".
- **Time trend at fixed size: +0.0013/month (95% CI -0.0004 to +0.0033, p = 0.19), "indistinguishable from zero"**, while GPQA Diamond gains about 2 points per month at fixed log-size. Factual capacity per parameter has not improved across model generations; reasoning per parameter has.
- Caveat: accuracy on probes, not bits; safety-tuned refusals lower some scores.

### 1.7 Fact-learning dynamics (why real training does not reach the ceiling)

- Lu et al., EMNLP Findings 2024 (https://arxiv.org/abs/2406.15720) [real facts, company tables and Wikidata]: fact capacity is linear in model size and follows a negative-exponential law in epochs; memorizing all of Wikidata would take about 1000B non-embedding params at 100 epochs; later facts overwrite earlier ones, which "significantly hinders low-frequency facts memorization"; redundant facts are not stored compatibly unless they share direction and structure.
- Chang et al., NeurIPS 2024 (https://arxiv.org/abs/2406.11813) [real, OLMo 1B and 7B checkpoints]: knowledge is acquired as small probability increments at each exposure, then diluted by forgetting that follows a power law in steps; duplicated data forgets faster; larger batches forget less.
- Zucchet et al. 2025 (https://arxiv.org/abs/2503.21676) [synthetic biographies, 8-layer 44M non-embedding default, ablations over sizes and recurrent variants]: three phases (generic statistics, a plateau while attention recall circuits form, then knowledge); imbalanced (power-law) person frequencies shorten the plateau; a "warm-up" on a subset of people then the full set gives "significant gains, particularly when the number of individuals is large"; fine-tuning new facts rapidly corrupts old ones.
- Auxiliary views (Sep 2026, https://arxiv.org/abs/2609.04180) [controlled pretraining; scale not extracted]: at a fixed token budget, spending tokens on reformulations instead of document repetition "improves learning, counterintuitively, even for factual recall"; paraphrasing helps "only at smaller batch sizes".
- Continual weight writes (Jul 2026, https://arxiv.org/abs/2607.11020) [Qwen3 models]: bare-statement facts keep 1% accuracy after twenty later writes versus 46% for facts written from broad study data; "When facts must be composed or survive later writes, the reliable channel is context rather than the weights."
- Forgetting and capacity (May 2026, https://arxiv.org/abs/2605.26097): "models pretrained close to saturation cannot absorb new information without overwriting prior knowledge".

### 1.8 Optimizers

- Muon vs Adam (Wang et al. 2025, https://arxiv.org/abs/2509.26030) [160M NanoGPT on a power-law-frequency biography QA task, 200K+ people; 0.7B in an appendix]: the associative-memory parameters (attention V/O and FFN) account for Muon's advantage; Muon's update is more isotropic in singular values, and "Muon substantially outperforms Adam on low-frequency (tail) data"; Adam beats SGD on the tail. This is measured as first-token accuracy per frequency bucket, not bits/param, and says nothing about the 1000-exposure ceiling.
- Counterpoint (May 2026, https://arxiv.org/abs/2605.06654): Muon has a "strong tendency towards rote memorization, which may hurt pattern acquisition with a small amount of data", and Muon-pretrained models did worse when fine-tuned for reasoning. For a skill-first model this is a real risk, not only a benefit.
- Capacity under Muon, SOAP, or any second-order method on bioS at 1000 exposures: **not measured anywhere I could find**.

### 1.9 Precision and quantization

- Scaling Laws for Precision (Kumar et al., ICLR 2025, https://arxiv.org/abs/2411.04330) [real, OLMo-style on Dolma, 30M/60M/110M/220M non-embedding, 1.5B to 26B tokens, 465 runs; validated with float types at 220M-1.6B]: training weights at integer precision P behaves like N_eff = N(1 - e^(-P/gamma_w)) with fitted gamma_w = 2.6745 (Appendix K, which also fits offset terms). With the main-text form, 4-bit weights are worth 78% of the parameters, 8-bit 95%, 16-bit 99.75%, 32-bit 99.9994% [calc]. Compute-optimal training precision is "around 7-8 bits". Post-training quantization damage grows with tokens per parameter, "as models train on more data, they compress more information into their weights".
- DFloat11 (NeurIPS 2025, https://arxiv.org/abs/2504.11651) [real, Llama 3.x, Qwen 3, Mistral and others]: the bf16 exponent "carries only about 2.6 bits of actual information" out of 8; sign and mantissa entropy are "close to their respective bit widths"; weights compress losslessly to about 11 bits. So each weight has about 11 bits of entropy, yet the measured information about the data is 2-4 bits: most of the entropy is mantissa noise that the data does not constrain (my inference).
- Spectra (https://arxiv.org/abs/2407.12327) [real, 99M to 3.9B, 300B tokens]: ternary TriLMs show "similar knowledge capacity to FloatLMs" on SciQ/TriviaQA/MMLU at 2.4B+ and TriLM 3.9B matches FloatLM 3.9B. At 99M-190M TriviaQA exact match is below 1% for every variant, so the small-scale rows say nothing. This is benchmark accuracy, not bits/param. **[>=2.4B only for the claim]**
- Revisiting BFloat16 Training (https://arxiv.org/abs/2010.06192) [mixed deep-learning tasks]: with pure bf16 weights, "nearest rounding for model weight updates often cancels small updates, which degrades the convergence"; stochastic rounding or Kahan summation closes the gap to fp32. This is the most likely mechanism for the Morris bf16/fp32 gap (my inference: an optimization effect, not a storage limit).

### 1.10 Weight sharing, low rank, sparse memories, distillation

- Looping: capacity per unique parameter unchanged (Ouro Sec. 6.1, 1M-40M, bioS, 1000 exposures; first-run verified). https://arxiv.org/abs/2510.25741
- Low rank: see 1.3 (adapters 1.3-2.8 bits per trainable param; full-rank >= 3.07 on the same base).
- Sparse memories: Memory+ at a 134M base (937M total, about 800M in memory values) beats a 373M dense model on TriviaQA (18.77 vs 17.68) (first-run verified, https://arxiv.org/abs/2412.09764). Per stored parameter this is worse than dense: roughly 800M memory params bought about 240M dense-equivalent params of TriviaQA [calc, crude]. UltraMem at 151M: MoE beat both product-key memory and UltraMem on TriviaQA (first-run verify, https://arxiv.org/abs/2411.12364). Sparse memories buy knowledge per FLOP, not per parameter.
- Distillation: soft labels transfer memorized information. Students trained on teacher soft labels "can achieve non-trivial accuracy on held-out memorized data they never directly observed", up to perfect accuracy in some random-label settings, strongly dependent on temperature (Behrens et al., ICML 2025, https://arxiv.org/abs/2506.14457) [random i.i.d. data; GPT-2 fine-tuning on random token-class pairs]. Whether soft labels raise bits/param or only reduce the exposures needed is untested.

---

## 2. Theory: how much can N parameters hold?

1. **Trivial upper bound.** A model with P parameters at b bits each can be in at most 2^(bP) states, so it stores at most b bits/param (Physics 3.3 Remark 4.2: "if the model parameters are 8-bit ... R(F) <= 8"). Trained models reach 2/8 = 25% of that bound at int8 (Physics Corollary 8.1) and about 3.6/16 = 22% at bf16 (Morris).
2. **Perceptron (Cover 1965, Gardner 1988).** For random ±1 labels on random inputs, a perceptron with N real weights stores up to alpha_c = 2 patterns per weight, i.e. 2 bits per weight, **with infinitely precise weights**; with binary weights alpha_c ≈ 0.83. Source (review, statistical-physics derivation): https://arxiv.org/abs/2304.06636 ("For continuous spherical weights, a RS calculation gives alpha_c = 2 ... For binary weights instead alpha_c ≈ 0.83"). Morris et al. cite the same "2 bits-per-parameter" perceptron result. Going from 1 bit to infinite precision raises capacity only 2.4x.
3. **Linear associative memory (2026 sharp thresholds).** A d x d linear memory with isotropic Gaussian key/value embeddings can do exact top-1 retrieval of n associations iff d²/(n ln n) > 2 (https://arxiv.org/abs/2605.05189, proven for all data-dependent linear memories; the same constant appears in https://arxiv.org/abs/2605.10795 as p_c log p_c / d² = 1/2). Each association selects one of n targets, i.e. log2 n bits, so the ceiling is n log2 n / d² = 1/(2 ln 2) ≈ **0.72 bits/param** [calc]. The log n factor is "the unavoidable extreme-value cost of winner-take-all decoding". This is a clean explanation of why capacity is O(1) bits/param regardless of precision: retrieval must beat the largest of n interfering scores.
4. **Transformers and MLPs store facts linearly in parameters, with low-precision weights.** Nichani, Lee, Bietti (ICLR 2025, https://arxiv.org/abs/2412.06538) [theory + small synthetic]: linear and MLP associative memories store a number of associations linear in parameter count up to log factors; a one-layer transformer can use either attention value matrices or MLPs to store facts; information-theoretic lower bound B >= N log M bits for N facts over M outputs. Their quantized construction needs only **O(log d) bits per weight** (Theorem in Appendix: "each weight requires O(log d) bits to store"). They also note that constructions memorizing N labels with O~(sqrt N) parameters use Ω~(sqrt N) bits per weight and "still require Ω(N) bits".
5. **Sub-linear parameter constructions exist but need huge precision.** ReLU networks can memorize N points with O~(sqrt N) parameters, and "having such a large bit complexity is both necessary and sufficient for memorization with a sub-linear number of parameters" (Vardi, Yehudai, Shamir, ICLR 2022, https://arxiv.org/abs/2110.03187); transformers likewise with O~(sqrt N) parameters in next-token prediction (Kajitsuka and Sato, ICLR 2025, https://arxiv.org/abs/2409.17677). These are existence proofs with weights holding about sqrt N bits each; gradient descent does not find them, and the total bit count stays Ω(N).
6. **Structured knowledge is cheaper than random facts.** "Geometric" memory: embeddings can encode relational structure so that a logarithmic embedding dimension suffices for bijective relations (https://arxiv.org/abs/2605.12426), and trained sequence models form such geometries even when not required (https://arxiv.org/abs/2510.26745). All bits/param numbers above are for incompressible data; word meanings and grammar have structure and should cost less per useful bit (inference, untested at Planck scale).

---

## 3. Why training fills only about 2-4 of 16 bits

Ordered from best-supported to most speculative.

1. **Most float bits are not storage bits.** A bf16 weight has about 11 bits of entropy (DFloat11), but its low mantissa bits are effectively noise from optimization. int8 post-training quantization keeps the full 2 bits/param (Physics Result 8); casting adapters to bf16 costs about 1% (adapter paper). The information lives in a few coarse bits per weight; the rest is unconstrained by the data.
2. **Retrieval under interference sets an O(1) ceiling even with infinite precision.** Perceptron: 2 bits/weight at infinite precision. Linear associative memory: about 0.72 bits/param with random embeddings [calc]. Every stored item adds noise to every other item's score; reliable retrieval needs a margin above the largest distractor, which costs a log factor. Trained transformers reach about 2-3.6, above these single-layer figures, because they learn embeddings and use depth, but they face the same interference geometry.
3. **Denser codes exist only with precision SGD cannot use.** Constructions that beat O(1) bits/param need weights with about sqrt(N) bits each (Vardi et al.). Noisy first-order training converges to solutions robust to perturbation (the classic flat-minima / minimum-description-length argument, Hochreiter and Schmidhuber 1997, not re-verified in this run), which by construction do not depend on fine bits.
4. **Exposure and forgetting.** Each exposure adds a small probability increment that decays with a power law (Chang et al.); saturation needs about 1000 exposures (Physics); 100 exposures reach half. Real pretraining gives most items far fewer exposures.
5. **Capacity allocation.** A bounded model spends capacity where it reduces loss most (Gu et al.). Items below a frequency threshold get none, even at thousands of exposures, so the average bits/param over "all the data" is far below the ceiling.
6. **Optimizer bias against the tail.** Adam learns rare associations more slowly than frequent ones; Muon's isotropic updates narrow the gap at 160M (Wang et al.). This affects the speed at which the ceiling is approached, not the ceiling.
7. **Architecture at low exposure.** Gated MLPs train less stably (Physics footnote 17: "gated MLP layers are less stable to train, thus requiring more time") and lose 1.3x at 100 exposures; MoE loses more; linear-recurrent layers and Canon convolutions gain. At 1000 exposures these differences mostly vanish.
8. **Pure-bf16 weight updates.** Updates below half an ulp are rounded away (Revisiting BF16), which plausibly explains most of Morris's 9% bf16-to-fp32 gap. Mixed precision with fp32 master weights avoids it.

---

## 4. What is known to raise bits/param

| Lever | Measured effect | Scale | Source | Confidence |
|---|---|---|---|---|
| Exposures 100 to 1000 | about 1 to about 2 bits/param; nothing above 2.3 | synthetic, 1M-0.5B | https://arxiv.org/abs/2404.05405 | high |
| Longer training at measurement time | +10-24% (16K to 40K steps, adapters) | synthetic, 2M-8M bases | https://arxiv.org/abs/2607.21351 | medium |
| Diverse rewrites vs one fixed text | slightly higher (bioS > bioS-simple, both about 2) | synthetic, 1M-0.5B | 2404.05405 Result 2 | high |
| Auxiliary views instead of repetition at fixed tokens | better factual recall | controlled pretraining | https://arxiv.org/abs/2609.04180 | medium |
| Remove 7/8 junk, or tag sources | up to 20x at 100 exposures; tag cuts the loss to 2x | synthetic | 2404.05405 Results 10-12 | high |
| Subsample dense data to raise per-item frequency | 0% to 23-40% at 410M | synthetic in real web mix, 70M-410M | https://arxiv.org/abs/2505.18091 | high |
| Imbalanced exposure / subset warm-up | shorter plateau, "significant gains" | synthetic, 44M | https://arxiv.org/abs/2503.21676 | medium |
| Standard MLP instead of gated (100 exp) | 1.3x | synthetic | 2404.05405 Result 6 | high |
| Canon layers (100 exp) | +10-15% for gated-MLP Llama; none for GPT-2 MLP | synthetic, 1M-500M | https://arxiv.org/abs/2512.17351 | medium |
| Linear-recurrent layers (Mamba2/GLA/GDN + Canon, 100 exp) | about +40% | synthetic, 1M-500M | 2512.17351 | medium (single group) |
| Muon on VO+FFN | faster, more even tail learning | synthetic QA, 160M | https://arxiv.org/abs/2509.26030 | medium; not measured as bits |
| bf16 to fp32 | +9% (3.51 to 3.83) raw memorization | synthetic, 80K-6.9M | https://arxiv.org/abs/2505.24832 | high |
| fp16 vs bf16 | "nearly identical" | synthetic | 2404.05405 Remark A.1 | high |
| int8 PTQ | no loss | synthetic | 2404.05405 Result 8 | high |
| int4 PTQ | loses >2x (to about 0.7) | synthetic | 2404.05405 Result 8 | high |
| MLP placement for low-rank params | 1.30 (attention) vs 2.43 (MLP) bits per trainable param | synthetic, 2M-8M bases | 2607.21351 | medium |
| MoE (per total param) | -1.3x (1000 exp), -1.5x or "10x less acquired" (100 exp) | synthetic | 2404.05405, 2512.17351 | high that it does not help |
| Looping / weight sharing | no gain per unique param | synthetic, 1M-40M | https://arxiv.org/abs/2510.25741 | high |
| Sparse memory layers | more knowledge per FLOP, less per stored param | real, 134M-151M bases | 2412.09764, 2411.12364 | medium |
| Soft-label distillation | transfers memorized info, even unseen items | random data | https://arxiv.org/abs/2506.14457 | untested for bits/param |

Nothing in this table has been shown to raise the ceiling above about 2.3 bits/param for facts. Most levers change how fast and how evenly a model approaches it.

---

## 5. Max's question: would fp64 or "more precise parameters" help?

Short answer: no, and the evidence points the same way from five directions.

1. **Empirical doubling test already exists.** bf16 to fp32 doubles the bits per parameter and raised raw memorization from 3.51 to 3.83 (+9%) (Morris, 80K-6.9M). fp32 to fp64 doubles again, from 23 to 52 mantissa bits, far beyond anything used; the expected gain is smaller than the bf16 to fp32 gain, likely indistinguishable from seed noise (inference; testable, see 6.5 experiment 1).
2. **What is stored fits in 8 bits.** int8 quantization of models at the 2 bits/param peak loses nothing (Physics Result 8). Adapters lose about 1% when cast to bf16.
3. **Precision scaling laws saturate.** With the fitted gamma_w = 2.67, 16-bit weights already give 99.75% of their effective parameter count and 32-bit 99.9994% [calc from Kumar et al., fitted at 30M-220M on integer types; extrapolation above 16 bits is outside their data]. Their compute-optimal training precision is 7-8 bits, i.e. lower, not higher.
4. **Theory says precision is not the bottleneck for trainable solutions.** Perceptron capacity is 2 bits/weight at infinite precision; transformer fact storage needs O(log d) bits per weight (Nichani et al.). Precision-hungry constructions that beat linear scaling need about sqrt(N) bits per weight and are not what gradient descent finds (Vardi et al.).
5. **Hardware makes it very expensive.** MLX: "Arrays with type float64 only work with CPU operations. Using float64 arrays on the GPU will result in an exception" (https://ml-explore.github.io/mlx/build/html/python/data_types.html). TITAN RTX fp64 runs at 1/32 of fp32, about 0.51 TFLOPS (https://techgage.com/article/nvidia-titan-rtx-workstation-performance-review/). PyTorch's MPS backend also lacks float64 (widely reported; unsourced here). The 10-60x throughput lost to fp64 buys far more capacity if spent on parameters or exposures.

What "more precise parameters" can usefully mean instead:
- **Fp32 master weights or stochastic rounding** when compute is bf16. This fixes the only precision effect with a known mechanism (lost small updates).
- **Parameters placed where they store more**: full-rank over low-rank, MLP over attention for low-rank updates (2607.21351), plain or Canon-augmented MLPs at low exposure (2404.05405, 2512.17351).
- **Lower precision with QAT to store more per stored bit**: int8 already stores 2 bits in 8 (25%) versus 2 in 16 for bf16 (12.5%); ternary may do better per stored bit (Spectra, >=2.4B, benchmark-level only). For Planck this matters for download size and RAM, not for parameter-count headlines.

---

## 6. A cheap replication for the M5 Mac

Do not run any of this until the Mac is free (hard rule for this run). Everything below is a design with arithmetic; throughput figures are assumptions to be measured first.

### 6.1 Data: "bioD-lite", a compact bioS equivalent

Keep bioS's knowledge content (so Physics's S0 and N0 apply unchanged) but pack it in the compact bioD style, which Physics 3.3 Result 3 shows still gives >= 2 bits/param and which the Canon paper calls format-independent.

- Names: first (400) x middle (400) x last (1000), each one token, N0 = 1.6e8, sampled without replacement.
- Attributes: pronoun (2), birth date as three tokens month (12), day (28), year (200), birth city (200), university (300), major (100), employer (263), working city (one token, deterministic from employer; its loss is not counted).
- One paragraph per exposure: `<bos> F M L [pron] p [bdate] m d y [bcity] c [univ] u [major] j [employer] e [wcity] w <eos>`, with the six attribute blocks in a fresh random order at every exposure (bioS-style diversity; bioS-simple-style fixed order is a separate arm if wanted). About 20 tokens per person-exposure.
- Vocabulary about 3.0K tokens (all rows used, so all embeddings count).
- Bits per person = log2(S0) + log2(N0/N) = 47.6 + log2(1.6e8/N), about 59-61 bits for N = 16K-60K [calc].
- Exposure control: every person appears exactly E times, shuffled; paragraphs packed into 256-token windows.

This is about 9x cheaper per fact than bioS text (20 vs about 172 tokens per biography [calc]). The cost of the compaction: it does not test extraction from paraphrased prose, which is fine for bits/param but should be noted in any claim.

For the precision ladder use a second probe, Morris-style random strings: sequences of a BOS plus 64 tokens uniform over V = 2048 (704 bits each), exactly as in Morris et al. and the adapter paper, so results are directly comparable to their tables.

### 6.2 Exact formulas

**Physics 3.3 capacity ratio for bioS (Definition 4.3)**:

R(F) = [ N·log2(N0 / e^p1) + N·log2(S0 / e^p2) ] / P

- N0 = 400 x 400 x 1000; S0 = 2 x (12·28·200) x 200 x 300 x 100 x 263, log2 S0 ≈ 47.59.
- p1 = mean over the N people of the **summed** (not averaged) cross-entropy in nats of the name tokens when the name opens a paragraph (after BOS). Perfect memorization of the name set gives p1 = ln N.
- p2 = mean over people of the summed cross-entropy in nats over all value tokens of all counted attributes (pronoun, month, day, year, birth city, university, major, employer), each predicted in the training format after the name and attribute token. Average over 2-3 random attribute orders.
- P = parameter count, excluding embedding rows that never occur (here none are unused).
- In bits: stored = N·(log2 N0 - p1/ln 2) + N·(log2 S0 - p2/ln 2).
- Maximum possible: R_max = [N·log2(N0/N) + N·log2 S0] / P. Physics: any model with R_max <= 1.8 was near-perfect, so to measure a ceiling the data must overfill the model (R_max well above the expected R).

General bioD form (Definition 4.1), if values span C chunks drawn from D-sized diversity sets over T tokens: R(F) = [N·log2(N0/e^p1) + N·K·log2(D^C/e^p2) + K·D·log2(T^L/(D·e^p3))] / P.

**Morris et al. memorization and capacity**:

- H(x) = S·log2 V (704 bits for S = 64, V = 2048).
- CL(x) = -Σ_t log2 p_θ(x_t | x_<t) over the 64 data tokens.
- mem(x) = H(x) - min(H(x), CL(x)); memorized bits = Σ over the dataset; capacity = maximum over dataset sizes; alpha = capacity / P with P counting all parameters including tied embeddings.
- Held-out random sequences must read mem ≈ 0 (sanity check used by the adapter paper).

Both measures are lower bounds at the declared training budget. Compare arms at the same budget, and report one budget-doubling run for the baseline.

### 6.3 Model configurations and budgets

GPT-2-style with RoPE, no dropout, tied embeddings, GELU MLP at 4d, head dim 64, context 256.

| Config | Layers x d | Params [calc] | Use |
|---|---|---|---|
| M0.25 | 2 x 64, V = 2048 | 2.35e5 (matches a Morris row: bf16 3.94, fp32 4.08) | precision ladder |
| M1 | 3 x 128, V = 3.0K | 0.97M | main rig |
| M4 | 4 x 256, V = 3.0K | 3.9M | scaling check at 100 exposures only |

Budgets at 1.75x overfill (R_max = 1.75x the expected ratio), about 20 tokens per person-exposure, training FLOPs ≈ tokens x (6P + 6·L·ctx·d) [calc]. Hours assume 1 TFLOP/s effective on the M5 GPU, which is an unmeasured guess; tiny models are latency-bound, so measure tokens/s first.

| Run | People N | Tokens | FLOPs | Hours at 1 TFLOP/s |
|---|---|---|---|---|
| M1, 100 exposures (expect about 1 bit/param) | 29K | 5.8e7 | 3.8e14 | 0.1 |
| M1, 1000 exposures (expect about 2) | 59K | 1.2e9 | 7.8e15 | 2.2 |
| M4, 100 exposures | 121K | 2.4e8 | 6.2e15 | 1.7 |
| M4, 1000 exposures | 246K | 4.9e9 | 1.3e17 | 35 (skip on the Mac) |
| M0.25 Morris probe, 2.7K sequences, 30K steps x 64 | n/a | 1.25e8 | 1.8e14 | GPU minutes; CPU fp64 unmeasured |

Step count: Physics found models "need at least 50K training steps regardless of batch size", so the 100-exposure M1 run should use small batches (about 1.2K tokens per step for 50K steps). Wall time there is set by per-step latency, not FLOPs.

### 6.4 Protocol

1. Throughput probe when the Mac is free: tokens/s for M1 at batch 1K, 4K, 16K tokens in MLX bf16.
2. Calibration (a known-answer test for the rig): M1 at 100 and 1000 exposures with a 3-point N grid (0.75x, 1.25x, 1.75x the expected capacity). Pass if the peak R is 1.0 ± 0.25 at 100 exposures and 2.0 ± 0.4 at 1000. If it fails, fix the rig (formula, exposure counting, LR, steps) before any intervention.
3. Noise band: baseline M1-100 with 3 seeds. An intervention counts only if its mean beats the baseline by more than 2 seed SDs with 2-3 seeds.
4. Learning rate: best of 3 LRs per arm (Physics did this for every negative result). Equal total parameters across arms (adjust d or MLP ratio when an intervention adds parameters).
5. Report per arm: R(F) overall, name bits vs value bits, and for non-uniform exposure the bits on the rarest quartile.

### 6.5 The first eight experiments

0. **Calibration** (above). Also reproduce the gated-MLP 1.3x deficit at 100 exposures (SwiGLU vs GELU at equal params): a second known answer.
1. **Precision ladder** on M0.25 (Morris probe) and M1-1000: pure bf16 weights, bf16 compute with fp32 master weights, pure bf16 with stochastic rounding, fp32, and fp64 on the CPU (M0.25 only). Prediction: master weights or stochastic rounding recover most of the 9% bf16 gap; fp64 equals fp32 within noise. This answers Max's question with his own measurement.
2. **Optimizer**: AdamW vs Muon/NorMuon (hidden matrices; embeddings on Adam) at M1-100, uniform exposure and Zipf (alpha = 1) exposure. Also run M1-1000 once to see whether the ceiling moves.
3. **Canon layers**: causal depthwise conv (kernel 4) before attention and MLP, at M1-100 with SwiGLU and with GELU MLP. Expect +10-15% for SwiGLU, about 0 for GELU.
4. **Hybrid storage layer**: replace 1 of 3 attention layers with a gated linear-attention or DeltaNet layer (with Canon), equal params, M1-100. Expect up to +40% knowledge if the Canon result transfers; also run the in-context recall probe from experiment 7 to see what recall costs.
5. **Exposure schedule at fixed total exposures**: uniform vs Zipf vs "subset warm-up then all" (Zucchet) vs expanding-interval spacing, M1-100. Win: +15% bits, or the same bits with 30% fewer tokens.
6. **Soft-label distillation**: train M4 to saturation on a smaller N, then train M1 on the same people with the teacher's full next-token distribution on value tokens (temperature 1 and 2) vs hard labels, M1-100. Hypothesis: same ceiling, reached with fewer exposures.
7. **Facts vs in-context skill (the Planck experiment)**: mix bioD-lite with an in-context task at a fixed 50% token share. The in-context task plants random name-attribute pairs early in a sequence and asks for them 100-200 tokens later (the multi-turn memory skill in miniature, with fresh random content each time so it cannot be memorized). Sweep the number of people N from 0 to 2x the model's capacity. If in-context accuracy stays flat as weight memory overfills, conversational recall is a cheap circuit that does not compete with stored facts, which supports Planck. If it drops, facts and skill compete for the same capacity and Planck's data should be stripped of facts.

### 6.6 Pitfalls

- Summed, not averaged, loss per name and per value (Physics Remark 3.1).
- The rig measures capacity only if the data overfills the model; an underfilled run measures the dataset.
- Tiny-model LR sensitivity is large at 100 exposures; do not compare arms at a single shared LR (the first run already found Max's A/B was confounded by one shared LR).
- Canon-style gains are learning-speed gains at 100 exposures; a 1000-exposure check tells whether the ceiling moved.
- Record exactly which tensors were bf16 on MLX; "bf16 training" is ambiguous (Morris does not say whether master weights were fp32).

---

## What this means for Planck at 10M-150M

- **Budget arithmetic** [calc]: at the 2 bits/param ceiling, a 10M model holds at most 2.5 MB of incompressible information, 20M holds 5 MB, 60M 15 MB, 150M 37.5 MB; at 100-exposure-like conditions, half of that. Everything a chat model must memorize item by item (word meanings, idioms, formats, persona, any facts) comes out of this budget. Facts in particular should come from context, which the 2026 continual-writes study also finds is "the reliable channel" when facts must survive later updates.
- **The 2 bits/param number is the worst case.** It is measured on random, incompressible facts. Grammar and word meaning are structured, and structured knowledge can be stored geometrically with far fewer parameters (2605.12426, 2510.26745). The practical question for Planck is not the ceiling but how much of the budget lexical knowledge actually needs, which no paper measures at this scale.
- **Capacity allocation is the biggest practical risk.** Gu et al. show a small model gives zero capacity to anything below a threshold frequency that rises as the model shrinks (exponent about 1.15), and more training does not fix it. For Planck this means each conversational behavior (accepting a correction, returning to an earlier topic, honoring a standing instruction) needs a minimum share of the data, higher at 20M than at 150M. Prefer fewer distinct templates of a skill seen many times over many one-off examples (their subsampling fix), and use source/type tags (Physics Result 12).
- **The Planck bet has direct support.** Across 93 real models, facts per parameter did not improve over time, while reasoning per parameter improved about 2 points per month at fixed size (IKP). That is the split Planck is betting on: skill is recipe-sensitive, knowledge is parameter-bound.
- **Morris's double-descent point cuts in Planck's favor**: once the data's information exceeds capacity, a model stops memorizing samples and starts generalizing. A 20M model trained on billions of tokens is deep in that regime.
- **Architecture**: use plain or Canon-augmented MLPs rather than bare SwiGLU if many items are rare; consider a hybrid with some linear-recurrent layers for storage efficiency but keep attention for exact in-context recall (Jelassi et al., cited in the brief; the Canon paper's own recurrent models were 2-4x weaker at reasoning depth). Experiment 4 plus experiment 7 measures both sides of this trade.
- **Precision**: bf16 compute with fp32 master weights; int8 for deployment is free in capacity terms; int4 needs QAT; no fp64.
- **Optimizer**: Muon helps rare associations but has a documented rote-memorization tendency that can hurt pattern learning in small-data SFT; test it on skills, not only on facts.
- **Things that do not help under a total-parameter budget**: MoE (loses capacity per total param), memory layers (cheap per FLOP, expensive per stored param), looping (no storage gain, though it can add computation depth for skills).

---

## Ledger ideas

Each: hypothesis / why / cheapest Mac test / win condition.

1. **Rig calibration on bioD-lite.** Hypothesis: a compact 20-token format reproduces about 1 and about 2 bits/param at 100 and 1000 exposures on a 1M model. Why: every later measurement depends on it; Physics Result 3 says format does not matter. Test: M1 at 100 and 1000 exposures, 3-point N grid, 3 seeds. Win: 1.0 ± 0.25 and 2.0 ± 0.4, and the gated-MLP deficit reproduces (1.2-1.4x).
2. **Precision ladder including fp64.** Hypothesis: fp32 master weights or stochastic rounding recover most of the bf16 gap; fp64 adds nothing over fp32. Why: Morris +9% for 2x bits; Revisiting BF16 mechanism. Test: Morris probe at 235K params (GPU for bf16/fp32, CPU for fp64) plus M1-1000 for bf16 variants. Win for fp64: more than 3% over fp32 across 3 seeds (not expected); win for master weights: recovers at least 2/3 of the bf16-to-fp32 gap.
3. **Muon vs AdamW bits/param.** Hypothesis: Muon raises bits/param at 100 exposures and on tail items, but not the 1000-exposure ceiling. Why: 160M tail-learning result. Test: M1-100 uniform and Zipf; one M1-1000. Win: +10% overall or +20% on the rarest quartile at 100 exposures.
4. **Canon layers.** Hypothesis: +10-15% at 100 exposures for SwiGLU models. Why: Physics 4.1. Test: M1-100 with and without Canon, SwiGLU and GELU. Win: +10% for SwiGLU.
5. **Hybrid linear layer for storage.** Hypothesis: swapping 1 of 3 attention layers for a gated linear/DeltaNet layer raises bits/param without killing in-context recall. Why: +40% knowledge for linear models at 100 exposures (Physics 4.1). Test: M1-100 plus the in-context recall probe. Win: +15% bits with in-context recall within 5 points of all-attention.
6. **Exposure scheduling.** Hypothesis: subset warm-up or expanding-interval spacing beats uniform shuffling at a fixed number of exposures. Why: plateau shortening (Zucchet), power-law forgetting (Chang), spaced-repetition gains in continual pretraining (https://arxiv.org/abs/2608.17530). Test: M1-100, four schedules. Win: +15% bits or equal bits with 30% fewer tokens.
7. **Soft-label distillation for storage.** Hypothesis: a teacher's soft targets let a student reach its capacity in fewer exposures (same ceiling). Why: soft labels transfer memorized information (Behrens et al.). Test: M4 teacher, M1 student, M1-100 and M1-300. Win: student at 300 exposures matches hard-label student at 1000.
8. **Facts vs in-context skill interference (Planck-critical).** Hypothesis: in-context recall is a fixed-cost circuit that does not degrade as weight memory overfills. Why: decides whether Planck must strip facts from its data. Test: bioD-lite + planted-fact recall task at 50% share, N swept 0 to 2x capacity, M1-100 and M4-100. Win (for the bet): in-context accuracy changes by less than 5 points across the sweep.
9. **Skill threshold frequency.** Hypothesis: a tiny model learns a rare conversational pattern only above a critical frequency that scales as a power of size (Gu et al.). Why: sets Planck's minimum data share per skill. Test: embed a synthetic "correction" pattern at 0.1% to 5% of tokens in M1/M4 training; find the threshold. Win: a measured threshold and exponent within about 2x of the Gu-style prediction.
10. **QAT int4 and ternary.** Hypothesis: int4 QAT holds at least 1.5 bits/param (PTQ holds about 0.7); ternary from scratch holds at least 1.0. Why: Physics suggests QAT; Spectra at >= 2.4B. Test: M1-1000 with fake-quant during the last 10% (int4) and throughout (ternary). Win: int4 >= 1.5 bits/param (0.375 bits per stored bit) or ternary >= 1.0 (0.63 per stored bit).
11. **Looping, confirm only.** Hypothesis: a 1-layer block looped 3x stores no more per unique parameter than 1 layer. Why: Ouro says so at 1M-40M; cheap to confirm before any Planck looping design. Test: M1-100 looped vs not at equal unique params. Win for looping: +10% (not expected).
12. **Low-rank MLPs.** Hypothesis: factorized MLPs (W = UV) store fewer bits per param than dense ones at equal params. Why: adapters 1.3-2.8 vs >= 3.07 full-rank. Test: M1-100, rank d/4 factorized MLP at 2x hidden width vs dense. Win for low rank: bits/param within 5% of dense (would allow wider, cheaper MLPs).
13. **Product-key memory bits per memory parameter.** Hypothesis: memory values store well under 1 bit each at this scale. Why: Memory+ is FLOP-efficient but parameter-inefficient; UltraMem lost to MoE at 151M. Test: M1 plus one PKM layer (64K values) at 100 and 1000 exposures; bits gained per added parameter. Win: at least 1 bit per memory parameter at 1000 exposures.
14. **Entity tokenization granularity.** Hypothesis: splitting names and values into 2-4 sub-tokens (as a small Planck tokenizer would) lowers bits/param for tiny models. Why: Physics footnote 18 (LLaMA's digit tokenization hurt tiny models). Test: same facts, single-token vs multi-token values, equal total params, M1-100. Win: know the cost; under 10% loss would clear small vocabularies.
15. **Weight decay sensitivity.** Hypothesis: bits/param at 1000 exposures falls noticeably above wd 0.02. Why: Physics used 0.001-0.02; Planck recipes use 0.1. Test: M1-1000 at wd 0, 0.01, 0.1. Win: a setting table, not an improvement.

---

## Open questions

1. Does multi-turn conversational skill consume capacity per item (like facts) or as a fixed circuit? Experiment 8 is the direct test; nothing in the literature answers it.
2. Why 2 bits (Physics) versus 3.6 (Morris)? Definitions, exposure count, key-value structure, or vocabulary? Running both probes on the same trained models would decompose it.
3. Can any optimizer or schedule exceed about 2.3 bits/param on bioS at 1000 exposures? No Muon, SOAP or second-order capacity ratios are published.
4. How do Canon's "10x less acquired" and Physics 3.3's 1.5x MoE penalty at 100 exposures reconcile?
5. What fraction of the 2 bits/param does a web-trained 20-150M model actually reach, and how much of its capacity goes to lexical knowledge? An IKP-style or bioS-probe evaluation of MaxGPT checkpoints (inference only, when allowed) would give a first number.
6. Do sparse memory layers store at least 1 bit per memory parameter at small scale, or are they only FLOP-efficient?
7. Does int4 QAT or ternary training hold most of the 2 bits/param at 1M-100M? Published evidence is benchmark-level and at >= 2.4B.
8. Are the 2026 linear-associative-memory thresholds (0.72 bits/param with random embeddings [calc]) a useful lower anchor for learned-embedding transformers, or is the learned-embedding gap (about 3x) itself the interesting quantity?

---

## Sources (primary, all opened for this report unless marked)

- Physics of LMs 3.3: https://arxiv.org/abs/2404.05405
- Morris et al. 2025: https://arxiv.org/abs/2505.24832
- Adapter capacity 2026: https://arxiv.org/abs/2607.21351
- Physics of LMs 4.1 (Canon): https://arxiv.org/abs/2512.17351
- Data mixing phase transitions: https://arxiv.org/abs/2505.18091
- Incompressible Knowledge Probes: https://arxiv.org/abs/2604.24827
- Lu et al. fact memorization scaling: https://arxiv.org/abs/2406.15720 (abstract only)
- Chang et al. knowledge acquisition: https://arxiv.org/abs/2406.11813 (abstract and appendix list)
- Zucchet et al.: https://arxiv.org/abs/2503.21676
- Auxiliary views: https://arxiv.org/abs/2609.04180 (abstract only)
- Continual fact writes: https://arxiv.org/abs/2607.11020 (abstract only)
- Forgetting and capacity: https://arxiv.org/abs/2605.26097 (abstract only)
- Spaced repetition training: https://arxiv.org/abs/2608.17530 (abstract only)
- Muon tail learning: https://arxiv.org/abs/2509.26030
- Optimizer-model consistency (Muon rote memorization): https://arxiv.org/abs/2605.06654 (abstract only)
- Scaling Laws for Precision: https://arxiv.org/abs/2411.04330
- DFloat11: https://arxiv.org/abs/2504.11651
- Spectra: https://arxiv.org/abs/2407.12327
- Revisiting BFloat16 Training: https://arxiv.org/abs/2010.06192 (abstract only)
- Perceptron capacity review: https://arxiv.org/abs/2304.06636
- Linear associative memory sharp threshold: https://arxiv.org/abs/2605.05189 ; decoupled model: https://arxiv.org/abs/2605.10795 (abstract only)
- Nichani, Lee, Bietti: https://arxiv.org/abs/2412.06538
- Vardi, Yehudai, Shamir: https://arxiv.org/abs/2110.03187 (abstract only)
- Kajitsuka and Sato: https://arxiv.org/abs/2409.17677 (abstract only)
- Geometric factual recall: https://arxiv.org/abs/2605.12426 (abstract only); geometric memory: https://arxiv.org/abs/2510.26745 (abstract only)
- Dataset distillation of memorized data: https://arxiv.org/abs/2506.14457 (abstract only)
- Ouro: https://arxiv.org/abs/2510.25741 (via first-run verify)
- Memory Layers at Scale: https://arxiv.org/abs/2412.09764 ; UltraMem: https://arxiv.org/abs/2411.12364 (via first-run verify)
- MLX data types: https://ml-explore.github.io/mlx/build/html/python/data_types.html
- TITAN RTX fp64: https://techgage.com/article/nvidia-titan-rtx-workstation-performance-review/ (search snippet)
