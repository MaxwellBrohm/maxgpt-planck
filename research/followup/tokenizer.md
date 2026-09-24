# Tokenizers and embeddings at tiny scale (follow-up track)

Scope: vocabulary size, embedding tricks and tokenization granularity for MaxGPT-Planck, a chat-skill model counted in TOTAL parameters (10M to 150M, headline run 20M). Written 2026-09-23. It builds on the first run's lanes (`lanes/arch.md` section 1.2, `lanes/frontier.md` section 1.6, and their `.verify.md` corrections). I did not repeat those measurements. Where I use them, I cite the lane. No model was loaded, run or trained. Numbers marked "my arithmetic" were computed here from published configs and the cited papers. The scripts are in the session scratchpad (`tok/acct.py`, `tok/conv.py`).

## Bottom line

1. At 20M total parameters, vocabulary size is the biggest single allocation decision in the model. Max's current 49,152-token vocabulary at d=256 costs 12.6M parameters, 63% of the budget, and leaves room for only 10 layers. An 8,192 vocabulary costs 2.1M (10.6%) and leaves room for 24 layers (my arithmetic).
2. The published vocabulary law (Tao et al. 2024, fit on 33M to 1.13B non-vocabulary parameters) puts the compute-optimal vocabulary for an 18M body at about 2.0M vocabulary parameters, which is about 8k tokens at d=256. That is an extrapolation below the fitted range, and the fit is for a regime where the total budget is not fixed, so the right answer for a fixed-total Planck is probably somewhat lower (4k to 8k).
3. New in 2026: "Compute Optimal Tokenization" (Meta, 1,308 models, 50M to 6.7B) finds that the best bytes-per-token falls slowly as compute grows. At Planck's small compute it wants tokens at least as fat as a normal BPE (about 4.3 to 4.8 bytes/token by extrapolation). At its smallest budget character-level models are 19% worse in bits per byte, and SuperBPE is slightly worse than plain BPE at every budget. This rules out byte-level and superword tokenizers for Planck. A small vocabulary also has to get its compression from somewhere else, and the somewhere else is training the tokenizer on Planck's own narrow chat data.
4. In-domain tokenizers close most of the compression gap at small vocabulary sizes. A 4,096-token tokenizer trained on TinyStories gives about the same sequence length as Llama 2's 32,000 (Karpathy), and conversation-optimized vocabularies cut chat token counts by 5-10%+. On chat text, an in-domain 8k BPE gives 3.83 bytes/token against 4.36 for a general 49k tokenizer (first-run measurement), so 2048 positions hold about 17 messages instead of about 19.
5. Byte-level Planck is a bad trade for multi-turn chat. The same 20-message conversation costs about 8.7x the forward FLOPs of an 8k BPE model, three quarters of it attention, and a 2048-position window holds about 4.6 messages instead of about 17 (my arithmetic).
6. Embedding tricks (factorized, hashed, frozen or compositional input embeddings) exist to shrink large tables. At 8k x 256 the table is already only about 10% of the model, so these tricks have little to save. The in-scale evidence is weak or negative: at 100M, hash embeddings lost to simply adding 4 layers (MultiHashFormer, 2026). Tie the embeddings. Test factorization and a byte-composition input only as second-stage ledger bets.
7. Recommended sweep and prediction: fixed 20.0M total, d=256, tied, one nested byte-level BPE trained on the Planck mix, vocab in {2k, 4k, 8k, 16k, 32k} plus the current 49k as an "embedding tax" reference. Measure bits per byte at a fixed byte context and the planted-fact recall battery, split by how many tokens the planted name takes. My predicted winner is 8k, with 4k a close second, and 4k could win if the synthetic register is narrow. 32k and 49k should lose clearly because they give up 9 to 14 of the 24 layers.

## Detailed findings

### 1. Vocabulary-size scaling laws

**Tao et al. 2024, "Scaling Laws with Vocabulary"** ([arXiv 2407.13623](https://arxiv.org/abs/2407.13623), NeurIPS 2024). The first run covered this paper; here are the parts that matter at Planck's size, re-read from the PDF.
- IsoFLOP fits on non-vocabulary sizes of 33M, 85M, 151M, 302M, 631M and 1.13B, with vocabularies from 4,096 to 96,256 (Table 4 and Sec. A.7.1). The fitted laws are N_nv = 0.08 C^0.50, N_v = 0.20 C^0.42, and training characters H = 6.42 C^0.50 (Sec. 4.1). The vocabulary exponent relative to the body is 0.84.
- They count vocabulary parameters once (N_v = V x d), "predicated on" the output layer carrying the FLOPs. For a tied model that is exactly the table.
- The smallest model is N_nv = 33M at d=512 on 4.3B characters. Figure 10 sweeps 0.5K to 512K at that size and finds a single optimum, with loss degrading on both sides. The text does not give the optimum's value.
- In-scale data point (flagged by the arch fact-check): at N_nv = 302M the best vocabulary among {8K, 10K, 16K, 24K, 32K, 48K} is 16K compute-optimal, 10K when data is scarce and 24K with excess data. The authors still recommend the compute-optimal vocabulary when overtraining, to keep inference cheap.
- Embedding quality: at N_nv = 85M, rare-token embeddings cluster (under-trained) at V=64K and stay dispersed at 4K and 16K (Figure 9; mean embedding distance 1.067 / 1.011 / 0.952 for 4K / 16K / 64K).
- My arithmetic from the fitted law: an 18M body gives C ≈ 4.9e16 and N_v ≈ 2.05M, which is V ≈ 8.0k at d=256 or 5.3k at d=384. Anchoring the 0.83 exponent on the 302M points instead gives 6.2k (compute-optimal anchor), 9.3k (excess-data anchor) and 3.9k (scarce-data anchor) at d=256. **Scale caveat:** 18M is below the smallest fitted body (33M), and Tao hold N_nv fixed while V varies, so a bigger vocabulary adds parameters. Planck holds the total fixed, so every vocabulary parameter is taken from the body, which pushes its optimum lower than Tao's.
- A critique from the 2026 paper below: Tao evaluate at a fixed context length in *tokens*, so larger vocabularies see more bytes per example. Later bytes are easier to predict, and this "can favor higher-compression tokenizers" (Limisiewicz et al., App. on related work). If that critique is right, Tao's optima are biased slightly upward.

**Limisiewicz, Pagnoni, ... Zettlemoyer 2026, "Compute Optimal Tokenization"** ([arXiv 2605.01188](https://arxiv.org/abs/2605.01188), May 2026; the first run missed it). This is the most important new source for this track.
- 988 BLT latent-patch models plus 320 isotropic subword models, 50M to 6.7B parameters, 4B to 1.1T training bytes, 5e18 to 2e21 FLOPs, trained on DCLM. Loss is measured in bits per byte at a fixed 8,192-byte context.
- Finding 1: the compute-optimal data-to-parameter ratio is roughly constant in *bytes*, ρ* ≈ 60 bytes per (non-embedding) parameter, whatever the compression rate. "20 tokens per parameter" holds only for a BPE-like compression.
- Finding 2: at each budget there is an optimal compression T* (bytes per token), and it falls slowly with compute. For BLT the fit is T* = T0 / C^δ with T0 = 18.2 and δ = 0.035, which gives T* = 3.69 at 1e20 and 3.33 at 2e21 FLOPs.
- Subword Table 2 (lowest BPB per budget):

| FLOPs | char (1.01 B/tok) | BPE 90% vocab masked (3.71) | BPE 75% masked (4.16) | BPE Llama-3 (4.57) | SuperBPE (6.16) |
|---|---|---|---|---|---|
| 1e19 | 1.2678 | 1.0819 | 1.0709 | **1.0635** | 1.0682 |
| 1e20 | 1.0519 | 0.9554 | 0.9502 | **0.9461** | 0.9532 |
| 2e21 | 0.9027 | **0.8466** | 0.8469 | 0.8479 | 0.8582 |

  At the smallest budget, characters are 19% worse than BPE. SuperBPE is worse than BPE at every budget, by 0.005 to 0.010 BPB (0.4 to 1.2%). The less-compressed masked vocabularies win only at the top budgets.
- **Scale caveats.** The subword models' vocabularies are huge (128k BPE, 148k characters, 200k SuperBPE), and embedding parameters are excluded from N. The smallest subword model is a 25M body with an 82M BPE table. Planck's compute is below the grid: a 20M model on 2B tokens is about 2e17 FLOPs, versus the grid minimum of 5e18. Extrapolating the BLT fit gives T* ≈ 4.8 / 4.5 / 4.3 bytes/token at 0.36B / 2B / 10B tokens for an 18M body (my arithmetic). The subword fit extrapolates to an implausible 7 to 11 bytes/token, so I do not use it.
- What it means: small compute wants each token to carry about as many bytes as a 32k-49k general BPE gets on chat text (4.36 to 4.44). A fixed-total 20M model cannot pay for that vocabulary. The way out is a tokenizer trained on the narrow domain, which reaches high compression at a small V (section 3).

**Chung and Kim 2025, "Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training"** ([arXiv 2508.15390](https://arxiv.org/abs/2508.15390), NeurIPS 2025).
- Setup: an 85M non-embedding model on 30B tokens of FineWeb-Edu, plus a 450M model on 10B tokens, with the vocabulary grown from 24K to 196K.
- For the 85M model, frequent-word loss fell from 3.845 to 3.742 nats, rare-word loss rose from 10.199 to 11.787, and global cross-entropy fell from 2.991 to 2.941.
- By 24K, over 95% of the 2,500 most frequent words are already single tokens. Those words cover about 75% of running text.
- Reading for Planck: a bigger vocabulary helps almost entirely on frequent words and hurts on rare ones, and multi-turn recall of user facts (names, places, numbers) lives in the rare tail. **Scale:** embeddings were not counted, so at a fixed total the bigger vocabularies here also got free parameters.

**PanGu-π Pro** ([arXiv 2402.02791](https://arxiv.org/abs/2402.02791), 1B, 50B tokens, Chinese plus English).
- The vocabulary was trimmed from 100k by dropping low-frequency tokens. Average score at 8k / 16k / 32k / 48k / 72k / 100k: 37.94 / 38.69 / 40.48 / 41.19 / 40.06 / 40.40, with the best at 48k (18.07% embedding plus head).
- Their rule of thumb is a vocabulary covering over 90% of the corpus, with embedding plus head under 20%. **Scale:** 1B and multilingual, so coverage needs are much larger than for English chat.

**Falcon-H1** ([arXiv 2507.22448](https://arxiv.org/abs/2507.22448)).
- TII trained a family of tokenizers sized to the model: 32,768 for 0.5B, 65,536 for 1.5B and 3B, 131k for 7B, 261k for 34B, "to scale the vocabulary size in proportion to the model's overall architecture (Tao et al., 2024)".
- Splitting digits and punctuation beat the alternatives on HumanEval in 1.8B / 280GT ablations. They also report that fertility and bytes-per-token "do not consistently predict downstream model performance".
- Falcon-H1-Tiny-90M reuses the 32,768 tokenizer, which the census records as "the smallest vocab size that our tokenizer series support". So its vocabulary was a floor, not an optimum.

### 2. What purpose-built tiny models actually spend on vocabulary

Embedding share of total parameters (tied unless noted). The configs were fetched from huggingface.co, and the first run verified most of these rows.

| Model | Vocab | d | Embedding params | Share | Note |
|---|---|---|---|---|---|
| PleIAs Monad (56.7M) | 8,192 | 256 | 2.10M | 3.7% | 64 layers; custom tokenizer on its synthetic corpus ([config](https://huggingface.co/PleIAs/Monad/resolve/main/config.json)) |
| MobileLLM-125M | 32,000 | 576 | 18.4M | 14.8% | Llama 2 tokenizer ([arXiv 2402.14905](https://arxiv.org/abs/2402.14905)) |
| Falcon-H1-Tiny-90M | 32,768 | 512 | 16.8M | 18.4% | the smallest tokenizer in the family ([config](https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Instruct/resolve/main/config.json)) |
| SmolLM2-135M | 49,152 | 576 | 28.3M | 21.0% | |
| LFM2.5-230M | 65,536 | 1024 | 67.1M | 29.2% | |
| MobileLLM-R1-140M (2025) | "128k" (Llama 3) | 576 | ~73.9M | ~53% | my arithmetic assuming 128,256; the repo is gated. Meta moved from 32k to 128k, apparently to distill from Llama-3.1-8B ([card](https://huggingface.co/facebook/MobileLLM-R1-140M)) |
| Karpathy stories15M | 32,000 | 288 | 9.2M | ~61% | llama2.c README table |
| Gemma 3 270M | 262,144 | 640 | 167.8M | 62.6% | "not designed for complex conversational use cases" |
| Gemma 4 26B-A4B (Planck's teacher) | 262,144 | 2816 | n/a | n/a | tied ([config](https://huggingface.co/google/gemma-4-26B-A4B-it/resolve/main/config.json)) |

Pattern: a model built for its size (Monad, MobileLLM, Falcon-Tiny) keeps the embedding share under about 20%. Models with big shares got them from reusing a family tokenizer (Gemma) or from a teacher-compatibility decision (MobileLLM-R1). The Gemma 4 teacher's 262k vocabulary means Planck, with its own tokenizer, can learn from Gemma only through generated text (sequence-level distillation) unless a cross-tokenizer method is added. The first run's training and posttrain fact-checks note those methods are proven only for students of 1B and up.

TinyStories itself trained its below-10M models at d=256 with the GPT-Neo tokenizer restricted to "the top 10K most common tokens" ([arXiv 2305.07759](https://arxiv.org/abs/2305.07759), footnote 2), and its data used a list of about 1,500 basic words. That is the closest published precedent to Planck's "small world, small vocabulary" bet.

### 3. Tokenizers trained on the target domain

- **Karpathy, llama2.c** ([README](https://github.com/karpathy/llama2.c)): "vocab size of 4096 trained specifically on tinystories creates integer sequences with about the same sequence length per example as the default Llama 2 tokenizer of 32000 tokens!" The measurement is on the TinyStories domain, and no quality comparison is given.
- **First-run measurement on OASST1** (`lanes/arch.md` 1.2, reproduced in `arch.verify.md`): own in-domain BPE at 8k / 16k / 32k / 49k gives 3.83 / 4.15 / 4.36 / 4.44 bytes/token. SmolLM2's general 49k gives 4.36, Monad's 8k (trained on SYNTH) 3.52, and Gemma 3's 262k 4.47. So an in-domain 32k matches a general 49k, and an in-domain 8k beats a general 8k by about 9%. The fact-check also notes that the chat template added about 28 tokens per 4.3-message thread, about 6.5 tokens per message.
- **Ferrando et al. 2025, "Is There a Case for Conversation Optimized Tokenizers?"** ([arXiv 2506.18674](https://arxiv.org/abs/2506.18674)). Re-optimizing eight production vocabularies (32k to 256k) on LMSYS-Chat-1M cut chat token counts by about 5% (Llama 3.1, DeepSeek-R1, Phi-4) up to more than 10% (Gemma 2, Mistral, BLOOM), while changing training-corpus length by under 2 to 5%. Assistant replies tokenize more efficiently than user turns. No model was trained, so this is compression only.
- **Caution on compression as a target** ([Schmidt et al. 2024, arXiv 2402.18376](https://arxiv.org/abs/2402.18376), 64 models at 350M to 2.4B): fewer tokens did not by itself predict better downstream results, and pre-tokenization mattered. [Lotz et al. 2025, arXiv 2506.03101](https://arxiv.org/abs/2506.03101) found tokenizer choice has "negligible effects" on English tasks at 350M and 2.7B, and that 350M runs predict 2.7B tokenizer rankings. At Planck's size, though, vocabulary size is not a tokenizer-quality question but a question of how the parameter budget is split. That effect is arithmetic and large (section 5).

### 4. Superword and byte/character-level tokenization

- **SuperBPE** ([arXiv 2503.13423](https://arxiv.org/abs/2503.13423)). The first run already found that at 680M in the over-trained regime SuperBPE has worse bits per byte than BPE at matched parameters (`lanes/frontier.md` 1.6). Compute Optimal Tokenization adds that SuperBPE is worse at every budget from 1e19 to 2e21 FLOPs (table above). Superword tokens also need vocabulary slots, which a 4k-8k Planck cannot spare. Follow-ups (SupraTok, [arXiv 2508.11857](https://arxiv.org/abs/2508.11857); Faster Superword Tokenization, [arXiv 2604.05192](https://arxiv.org/abs/2604.05192)) improve compression or training speed. I found no model-quality result for them below 500M. Verdict: skip for Planck.
- **Length-MAX** ([arXiv 2511.20849](https://arxiv.org/abs/2511.20849), GPT-2 124M/355M/1.3B, five runs each) reports 14-18% fewer tokens than BPE at 10k-50k and 18.5% fewer steps to a fixed validation loss at 124M. But the comparison target is a per-token loss of 2.0 across different tokenizers, not bits per byte, so the result is confounded. Treat it as unverified.
- **Character level at tiny scale.** Compute Optimal Tokenization (above) gives characters a 19% bits-per-byte penalty at 1e19 FLOPs, shrinking to 6.5% at 2e21. Planck's compute is below 1e19, where the penalty should be at least as large. Bunzeck and Zarrieß ([arXiv 2502.12835](https://arxiv.org/abs/2502.12835), ACL 2025, BabyLM-scale models with a 102-symbol character vocabulary against 8,002 subwords) find that character models separate words from non-words much better. That is a psycholinguistic property, not chat quality. BLT and H-Net (first run) reach parity with BPE only with learned patching, at 400M to 8B.
- **Multi-turn cost at 20M (my arithmetic, `tok/conv.py`).** Setup: a 20-message conversation at the OASST mean of 442 bytes per message, d=256, the layer counts that fit 20M, and forward FLOPs = 2N per token plus 2·d·L·T² for causal attention.

| tokenizer | tokens / conversation | forward GFLOP | attention share | messages in 2048 positions |
|---|---|---|---|---|
| bytes | 8,880 | 1,392 | 75% | 4.6 |
| BPE 4k (bytes/token estimated at 3.5, unmeasured) | 2,566 | 184 | 46% | 16.0 |
| BPE 8k (3.83, measured) | 2,348 | 161 | 42% | 17.4 |
| BPE 16k (4.15) | 2,170 | 136 | 37% | 18.9 |
| BPE 32k (4.36) | 2,068 | 113 | 29% | 19.8 |

(Template overhead: 2 tokens per message. With the 6.5 tokens per message the fact-check measured, each row loses about 0.6 to 0.8 messages.) The larger vocabularies look cheaper per conversation only because they have fewer layers. Bytes cost about 8.7x an 8k BPE for the same conversation and fit about a quarter as many turns in the window. Going from 8k to 32k buys about 12-14% more messages per window.

### 5. Embedding tricks at small scale

**Tying.**
- MobileLLM at 125M: tying saves 16M parameters (11.8%) for -0.2 accuracy points, and spending the savings on 2 more layers wins overall ([arXiv 2402.14905](https://arxiv.org/abs/2402.14905), first run).
- New: Lopardo et al. 2026 ([arXiv 2603.26663](https://arxiv.org/abs/2603.26663), ACL 2026 Findings) show that tied matrices end up shaped mostly by the output gradients. But their fix (scaling input gradients 2-10x, tested on OLMo-1B at 20B tokens) gave "no consistent performance difference", and perplexity moved 35.71 to 36.64. They conclude that at 70M-class scale, where embeddings are up to 73% of an untied Pythia-70M, "these parameter savings are substantial and justify the representational cost".
- The speedrun unties at 124M for speed (first run), but Planck's budget is counted at inference, so untying costs V x d. For 8k at d=256 that is 2.1M, about 3 of the 24 layers.
- Verdict: tie.

**Factorized embeddings (ALBERT).**
- ALBERT-base, not shared, 30k vocabulary, encoder MLM ([arXiv 1909.11942](https://arxiv.org/abs/1909.11942), Table 3): E = 64 / 128 / 256 / 768 gives 81.3 / 81.7 / 81.8 / 82.3 average at 87M / 89M / 93M / 108M parameters. So cutting E from 768 to 128 saves 19M parameters (18%) for -0.6 points.
- In a tied decoder, factorization also caps the rank of the output logits at E. The first run's Godey et al. evidence (head rank below about 1000 hurts, contested by Kulkarni et al. 2026) says Planck's d=256 head is already in the low-rank regime.
- At 8k x 256, factorizing to E=128 saves about 1.0M parameters (1.4 layers) and halves the head rank. A 2025 OpenReview paper on factorized plus tied embeddings in sub-billion decoders (the [PDF](https://openreview.net/pdf?id=gdMMeemjRB), title fragment "Leveraging Low-Rank Structure for Effective...") reportedly finds that loss holds until the rank drops below about half of d, and that tying helps at a given rank. I could not read the PDF (the site's bot challenge), so this is **unsourced beyond the search snippet**.
- Verdict: a ledger bet, not a default.

**Hash and multi-hash embeddings.**
- MultiHashFormer (Xue, Yamaguchi, Aletras; [arXiv 2606.28057](https://arxiv.org/abs/2606.28057), EMNLP 2026; 100M on 10B tokens, 1B and 3B on 100B, Mistral 32K vocabulary, tied). At 100M, the parameter-matched hash model (H3B10K) against Standard: LAMBADA 16.40 vs 15.33 and ARC-E 44.91 vs 41.33, but SciQ 55.50 vs 59.00 and ReCoRD 48.21 vs 49.47. **Standard+4L, which just adds layers, beats every hash variant at 100M** (LAMBADA 19.64, SciQ 62.60, ReCoRD 53.08). The hash approach wins only at 1B-3B.
- T-FREE ([arXiv 2406.19223](https://arxiv.org/abs/2406.19223)) cuts embedding layers by more than 85% with character-trigram hashing, measured at 1B+.
- Over-Tokenized hashed n-gram input tables help at 151M on 400B tokens (first run). But those gains come from tables of 1.2M to 12.8M *rows*, which a 20M-total Planck cannot afford.
- Verdict: at 20M, spend the parameters on layers.

**Frozen or compositional input embeddings.**
- Kronecker Embeddings (single author, [arXiv 2605.29459](https://arxiv.org/abs/2605.29459), May 2026). A fixed byte-and-position codec plus one learned projection replaces the input table. GPT-2 124M on 2.5B FineWeb-Edu tokens, 3 seeds: 2.5 ± 0.2% lower validation loss than the tied-BPE baseline, and more robust to typos (top-1 preserved 55.5% vs 47.3% of clean/typo pairs).
- Confound: the Kronecker arm needs an **untied** output head, so it has about 3M more parameters and a structurally different head, and there is "No untied-BPE arm". The speedrun independently found untying helps at 124M, so the gain cannot yet be credited to the codec.
- Bochkov ([arXiv 2507.04886](https://arxiv.org/abs/2507.04886), TMLR 2025; 336M, 65k vocabulary, about 4B tokens, untied head) trains with frozen glyph-image embeddings and claims semantics emerge in the transformer body. It is single-author work, and its MMLU claims are at a scale where MMLU is near chance. Weak evidence.
- For Planck, the interesting part is typo and surface-form robustness when copying user-supplied names. That is a ledger bet (L5, L8).

**Adaptive-dimension embeddings.** Adaptive input and adaptive softmax (Baevski and Auli, [arXiv 1809.10853](https://arxiv.org/abs/1809.10853), word-level vocabularies of about 260k) give rare tokens fewer dimensions. At an 8k BPE vocabulary the tail is short, so the savings would be under 1M parameters. Not worth a slot.

**Embedding learning rate.** Hayou and Liu ([arXiv 2506.15025](https://arxiv.org/abs/2506.15025), validated at 1B) find the optimal ratio of embedding to hidden learning rate scales as the square root of width when the vocabulary is large relative to width. It is a methodological warning: a vocabulary sweep with one embedding learning rate for all arms confounds vocabulary size with learning-rate mistuning. Tune the embedding learning rate per arm, or at least at 4k and 32k.

### 6. Tokenizer details that matter for multi-turn recall

- **Digits and punctuation split individually** (Falcon-H1, 1.8B ablation). Planck will copy numbers from retrieved facts and from the user. Single-digit tokens make every number well trained and copyable one digit at a time.
- **Byte fallback** (byte-level BPE), so any name the user types can be represented and copied.
- **Single-token role markers.** One special token per role boundary (for example `<|user|>`, `<|assistant|>`, `<|end|>`, plus a `<|retrieved|>` marker, because facts will arrive through retrieval) cuts template overhead from about 6.5 to about 2 tokens per message. That buys about 4% more messages per window.
- **Surface-form variance.** GPT-2-style BPE tokenizes "Pearl", " Pearl" and "Pearl's" differently, so recalling a name at the start of a sentence may need a different token sequence from the one that planted it. Treating spaces as separate tokens gave more morphologically regular tokenization and better complex-word handling (Gow-Smith et al., [arXiv 2204.04058](https://arxiv.org/abs/2204.04058), EMNLP 2022, BERT-scale), at the cost of more tokens. Untested for chat recall at tiny scale.
- **Under-trained tokens.** Tokens that are rare in the training data become "glitch" tokens ([Land and Bartolo 2024, arXiv 2405.05417](https://arxiv.org/abs/2405.05417)), and Tao's Figure 9 shows the same effect at 85M. A tokenizer trained on the exact Planck mix, with a small vocabulary, keeps every row well trained. Check the minimum token count in the training stream before training.

## What this means for Planck at 10M-150M

**Budget split (my arithmetic, `tok/acct.py`: tied, head dim 64, GQA, SwiGLU, 20.0M total).**

| d | V=2k | V=4k | V=8k | V=16k | V=32k | V=49k |
|---|---|---|---|---|---|---|
| 256 (4q/2kv, ffn 704): emb share / layers | 2.7% / 26 | 5.4% / 25 | 10.6% / 24 | 21.3% / 21 | 43.1% / 15 | 63.0% / 10 |
| 384 (6q/2kv, ffn 1024) | 4.0% / 12 | 8.3% / 11 | 16.7% / 10 | 33.3% / 8 | 66.7% / 4 | none |

**Suggested vocabulary centers along the Planck curve.** The centers come from Tao's fitted law at the body size each total leaves. I shade them down for the fixed total, and they are consistent with the first run's 16k-32k recommendation at 150M.

| Planck total | width | Tao-fit V (compute-optimal) | sweep range | predicted best | embedding share at best |
|---|---|---|---|---|---|
| 10M | 192-256 | ~4.5k at d=256 | 2k-8k | 4k | ~5-8% |
| 20M | 256 | ~8.0k | 2k-32k (+49k reference) | 8k | ~11% |
| 60M | 384-512 | ~10-14k | 8k-32k | 12k-16k | ~10-13% |
| 150M | 576 | ~15-19k | 16k-32k | 16k-24k | ~6-9% |

These are compute-optimal centers. Tao's 302M data point says excess data moves the optimum up about 1.5x (16K to 24K). The fixed total and Limisiewicz's evaluation-bias critique both push it down. So sweep ±2x around the center rather than trusting it.

**Use one nested tokenizer for the whole curve.** BPE training is greedy, so the first k merges of a 32k run on the same data are the k-merge tokenizer (verify this cheaply, L3). Train one byte-level BPE on the Planck mix (the teacher-generated conversations plus whatever general text is mixed in, weighted as in training, with the chat template applied, digits split and role specials reserved). Truncate its merge list for each size. The curve then changes only size and V, and every smaller vocabulary is an id-prefix of every larger one. Compute Optimal Tokenization's "masked vocabulary" arms are the same construction.

**The 20M sweep (fixed 20.0M ±1% total, d=256, tied, same data counted in bytes, identical recipe otherwise).**

- Stage 0, zero training (CPU minutes): for each V in {2k, 4k, 8k, 16k, 32k}, measure
  - bytes/token on held-out Planck chat and on general text;
  - the fraction of the 2,500 most frequent dialogue words that are single tokens;
  - tokens per planted name or number;
  - the minimum and 1st-percentile token count in the training stream;
  - messages per 2048-token window.
  Drop any V whose tail tokens appear fewer than about 1,000 times in the planned run.
- Stage 1, vocabulary size: V in {2k, 4k, 8k, 16k, 32k} at 26 / 25 / 24 / 21 / 15 layers, plus 49k at 10 layers once, as the "embedding tax" reference for the writeup. Screen with one seed at about 0.35B tokens, which the compute lane puts at about 10 hours per 18.5M run on the M5. Then rerun the top two or three with 2-3 seeds at the target token budget, because a short screen biases toward small vocabularies (Tao: less data, smaller optimum). Retune the embedding learning rate per arm (section 5).
- Stage 2, embedding tricks at the Stage 1 winner (fixed total): tied (control), untied (minus 3 layers), factorized tied E=128 (plus 1 layer), fixed byte-codec input with an untied head, and a small hashed-bigram input table (about 1M parameters, minus 1 layer).
- Stage 3, tokenizer construction at the winner: in-domain against general-web training data, standard against space-as-token pre-tokenization, and BPE-dropout 0.1 on or off if the chat data repeats more than 4 epochs.
- Byte-level: do not run a full arm. The Compute Optimal Tokenization table and the 8.7x cost per conversation already settle it. If Max wants it in the writeup, run one seed at matched FLOPs with an 8k-byte context.

**What to measure (every arm).**
1. Bits per byte on held-out Planck chat and on general text, evaluated at a fixed context measured in bytes (not tokens), so vocabulary sizes are compared fairly.
2. The multi-turn battery: planted-fact exact recall at 2 to 12 turns, follow-ups and references, corrections, returning to a topic. Report recall **stratified by how many tokens the planted item takes**, and on mismatched-surface probes (case, leading space, possessive).
3. Cost: bytes per second of training on the Mac, forward FLOPs per 20-message conversation, decode steps per reply, and KV bytes per message.
4. Health: embedding share, under-trained-row count (low-norm rows or rows with near-zero gradient updates), and the embedding singular-value spread.
5. Noise: MobileLLM's single-seed tables wobble by about 0.5 points (`arch.verify.md`), so claim a winner only when the two-seed gap exceeds the seed spread on both bits per byte and recall.

**Prediction.** 8k wins at 20M. It is inside Tao's extrapolated band (6-9k at d=256), keeps about 10% of the budget in the table (the share every purpose-built tiny model stays under), and keeps 24 layers. In-domain training should put it at about 3.9-4.2 bytes/token on Planck chat.

Expected ordering on chat bits per byte, from best to worst:
- 8k ≈ 4k (within about 1%);
- 16k (about 1-2% worse);
- 2k (worse, too many tokens per turn);
- 32k (about 3-6% worse, 15 layers);
- 49k (clearly worst, 10 layers).

On planted-name recall I expect 4k and 8k to tie or 4k to edge ahead, because every token is well trained. Among the Stage 2 variants, tied plain embeddings should win or tie; factorized, hashed and codec inputs should land within noise. The codec input is the one with real upside, on typo and case robustness. My confidence in "8k or 4k beats 32k" is moderate to high. It rests on the arithmetic and on MultiHashFormer's "layers beat embedding parameters at 100M". My confidence in "8k beats 4k" is low.

## Ledger ideas

**L1. Vocabulary-size sweep at a fixed 20M total.**
- Hypothesis: at a fixed total, a 4k-8k vocabulary beats 16k-49k on chat bits per byte and on recall, because body layers are worth more than vocabulary rows at this size.
- Why it could work: Tao's law extrapolates to about 8k. Tiny purpose-built models keep embeddings under 20%. MultiHashFormer found adding layers beat embedding parameters at 100M.
- Cheapest test: the Stage 0 CPU statistics, then single-seed 0.35B-token screens (about 10 hours each on the M5), then 2 seeds on the top two.
- Win: the chosen V beats 32k by at least 2% bits per byte, and is not worse on planted-fact recall, over 2 seeds.

**L2. In-domain tokenizer.** Train the BPE on the Planck chat mix rather than general web text, at the same V.
- Why: in-domain tokenizers reach the compression of a 4-8x larger general vocabulary (TinyStories 4k ≈ Llama 2 32k; OASST in-domain 32k = SmolLM2 49k).
- Cheapest test: CPU compression on held-out chat, then one paired training run.
- Win: at least 5% more bytes/token on chat, lower chat bits per byte, and no more than 2% worse general-text bits per byte.

**L3. Nested vocabulary family for the whole curve.** One BPE merge list truncated to each size, so the Planck curve varies only size and V.
- Cheapest test: train a 32k and an 8k BPE on the same data (CPU) and check that the 8k merges equal the first 8k of the 32k.
- Win: identical merges and ids, or compression within 0.5% if tie-breaking differs.

**L4. Tied vs untied vs factorized at a fixed total.** Tied, untied (minus 3 layers) and factorized E=128 (plus 1 layer, tied) at V=8k.
- Why: tying won at 125M (MobileLLM). Factorization frees about 1M parameters but caps the head rank at 128.
- Cheapest test: 3 arms x 1 seed at the screen budget.
- Win for factorization: at least 1% better bits per byte with no drop in recall. Otherwise keep tied.

**L5. Fixed byte-composition (Kronecker-style) input embedding plus a learned untied head, at the same total as tied.**
- Why: +2.5% validation loss at 124M, plus typo robustness (confounded by the untied head). For chat, the robustness to how the user spells a name is the real prize.
- Cheapest test: 1 arm plus an untied-learned control (the missing arm in the paper).
- Win: at least 1% better bits per byte than *untied learned*, or at least 5 points better recall on mismatched-surface probes.

**L6. Small hashed-bigram input table.** About 1M parameters (4,096 buckets x 256) in place of about 1.4 layers.
- Why: the Over-Tokenized and speedrun bigram hash gains, at a tiny fraction of their table size.
- Cheapest test: 1 arm at the screen budget.
- Win: at least 1% better bits per byte at equal total parameters.

**L7. Recall of rare names as a function of vocabulary.**
- Hypothesis: small vocabularies copy user-introduced names across turns at least as well as large ones, despite needing more tokens per name, because every token is well trained. Large-vocabulary rare-word loss rises (Chung and Kim: 10.2 to 11.8 nats from 24K to 196K at 85M).
- Cheapest test: re-score the Stage 1 checkpoints on the planted-fact battery, stratified by planted-item token count. No extra training.
- Win: a recall difference of 5 points or more between vocabulary arms with non-overlapping seed ranges.

**L8. Surface-form-robust pre-tokenization** (spaces as separate tokens, or normalized leading-space handling).
- Hypothesis: recall improves when the name comes back in a different surface form.
- Cheapest test: CPU check of how many planted names change token sequence across surface forms, then one paired run.
- Win: at least 5 points better mismatched-surface recall, with bits per byte no more than 1% worse.

**L9. Minimal chat template** (single special token per role, `<|retrieved|>` for facts).
- Why: it cuts overhead from about 6.5 to about 2 tokens per message, which is about 4% more turns per window.
- Cheapest test: CPU token count.
- Win: overhead of 2.5 tokens per message or less, with no loss in role adherence on the battery.

**L10. BPE-dropout (p=0.1) when chat data repeats.**
- Why: it regularizes segmentation, which should help with multi-epoch synthetic data and with names that appear in varied segmentations.
- Cheapest test: 1 paired run at 4+ epochs of the chat set.
- Win: lower held-out bits per byte, or better mismatched-surface recall.

**L11. Chat-phrase superword tokens** (a few hundred frequent multi-word chat phrases added to an 8k vocabulary).
- Prediction: neutral or negative on bits per byte (SuperBPE is worse at small compute), with a small gain in turns per window.
- Cheapest test: CPU compression, then 1 run only if compression rises by 5% or more.
- Win: bits per byte no more than 0.5% worse and at least 5% more messages per window.

**L12. Gemma-subset vocabulary.** Build Planck's 8k vocabulary from tokens that exist in Gemma 4's 262k vocabulary, so the teacher's logits could be renormalized over it if logit distillation is ever wanted.
- Cheapest test: CPU compression of the constrained vocabulary against the free 8k BPE on Planck chat.
- Win: within 3% bytes/token, which keeps the logit-distillation option open at almost no cost.

**L13. Embedding-tax headline.** Same 20M total, Max's current 49k tokenizer against the chosen 8k. This quantifies the tax for the writeup.
- Cheapest test: 1 seed each at the screen budget (overlaps L1).
- Win: a clear, reportable gap. The expected result is large.

**L14. Byte-level control at matched FLOPs** (optional, for the writeup only).
- Prediction: loses by 15% or more in bits per byte and costs about 8.7x per conversation.
- Cheapest test: one seed with an 8k-byte context at the FLOPs of the 8k-BPE arm.
- Win (unlikely): within 5% bits per byte of 8k BPE at matched FLOPs.

## Open questions

1. Does the vocabulary ranking on bits per byte match the ranking on multi-turn recall at 20M? The whole literature optimizes loss or benchmarks, and none of it measures dialogue recall.
2. How far does overtraining move the optimal vocabulary below 50M? Tao's only direct evidence is one 302M point (16K to 24K), and Planck will run at 100-500 tokens per parameter.
3. What bytes/token does an in-domain 4k/8k tokenizer actually reach on Gemma-4-generated Planck conversations? The register is set by the teacher prompt, so this is partly a data-design choice.
4. Does a short screen (about 20 tokens per parameter) rank vocabularies the same way as the target budget? The theory says it biases toward small V.
5. Does the low-rank output head at d=256 interact with vocabulary size to produce the degenerate loops Max saw in MaxGPT-2/3? Nothing measures generation quality against head rank at this size.
6. Do multi-token names make cross-turn copying harder for a 20M model? Induction over multi-token spans is established at 100M+ (first run), but not at 20M.
7. The "Leveraging Low-Rank Structure" factorized-embedding paper could not be read (OpenReview bot challenge). Someone with a browser should check its sizes and numbers before L4 relies on it.
8. Is logit distillation from Gemma 4 worth keeping as an option at all? The teacher-to-student ratio is about 1,300x in total parameters. If not, L12 is moot and the tokenizer can be chosen purely for Planck.

Sources not already linked inline: Compute Optimal Tokenization PDF (https://arxiv.org/pdf/2605.01188), Tao et al. PDF (https://arxiv.org/pdf/2407.13623), Falcon-H1 report (https://arxiv.org/pdf/2507.22448), ALBERT (https://arxiv.org/pdf/1909.11942), MultiHashFormer (https://arxiv.org/pdf/2606.28057), Kronecker Embeddings (https://arxiv.org/pdf/2605.29459), weight tying bias (https://arxiv.org/pdf/2603.26663), PanGu-π Pro (https://arxiv.org/pdf/2402.02791), frequency imbalance (https://arxiv.org/pdf/2508.15390), TinyStories (https://arxiv.org/pdf/2305.07759), llama2.c README (https://raw.githubusercontent.com/karpathy/llama2.c/master/README.md).
