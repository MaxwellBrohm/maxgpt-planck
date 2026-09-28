# Track K research lane: arch (architecture and training choices that trade fact storage for context skill)

Date 2026-09-27. For track K (K1-K5: synthetic, program-generated data, measurement models only, never released).
Method: web search, then the primary source for every claim (arXiv PDF via pdftotext, or the arXiv abstract API). Nothing was
trained, run or loaded. "My arithmetic" = computed here, not in a paper. "[>=1B]" = evidence exists only at 1B+.
"(repo)" = already verified in this repo's research notes; cited, not re-derived.
Fact-check pass (same day): every finding below was re-opened at its source (PDF text for all except F4, F5, F11, F13, F20-F22,
F25, Co-LMLM, which were checked against the arXiv abstract) and carries a "Verified:" line: OK, CORRECTED (text above already
fixed) or UNVERIFIABLE. Venues were checked on arXiv metadata, the PDF header, or the venue's own page.

## 0. Bottom line

1. **The premise "facts crowd out skills" is supported, but the measured mechanism is competition and circuit disuse, not parameters running out.** The closest study (Kim et al., 8-layer
   d512, ~25M body by my arithmetic) trained from scratch on synthetic bios: with a uniform fact base and 1% intra-document inconsistency, in-context use on UNSEEN entities decayed to 31.5%
   by the end of training (84.0% with a Zipfian base; Pythia-6.9B on web data kept it high). In toys, ICL fades to in-weights learning; weight decay on MLPs only keeps ICL alive. So K1 must
   log trajectories and must separate load from dose.
2. **Nobody has mixed Planck-type in-context skills (updates, binding, context QA with nonce values) with a memorizable fact base at controlled load.** Physics of LMs 4.1 trains and
   evaluates each synthetic task in isolation; Kim et al. test only copy-from-context; the one joint study ("Too Big to Think") is a 2K-parameter toy. K1 is genuinely new.
3. **At the 5M screen budget the fact side is easy to get wrong.** 250M tokens cannot saturate capacity at 1000 exposures (my arithmetic, s3); at 100 exposures gated MLPs store 1.3x less and
   7/8 same-format random bios (which nonce skill items resemble) cut fact capacity 20x unless fact documents carry a special token (Physics 3.3). Physics also says models need >=50K steps;
   the 5M screen is 7,630. Calibrate recall on the screen setting; do not assume 2 bits/param.
4. **K4: loss is U-shaped in the MLP-to-attention parameter ratio r, fitted optimum r = 1.032 (1B) and 1.055 (3B), and the 5M baseline sits at r = 1.91 (my arithmetic).** Fact capacity is
   MLP-insensitive at 1000 exposures and >1.5x worse without MLPs at 100. Attention also stores in-weights knowledge, so shrinking MLPs will not by itself remove the fact-storing competitor.
5. **K5: the known lever that stops memorization is loss-masking the looked-up value (LMLM, 124M-382M).** Under the masked loss the loss on returned values stays high; with the database off
   the 382M model scores below its no-lookup twin (T-REx EM 38.5 vs 52.0). A lookup that is present but still supervised is a different condition; K5 should test both.
6. **Small models follow the context; frequency and training noise decide conflicts.** GPT-2 small lets the factual answer win only 4% of the time against a repeated in-context
   counterfactual; memorized answers rise with size (Pythia 70M-2.8B); clean repeated training data makes a model consistently prefer context, 1% noise flips frequent facts to parametric. At
   5M the counterfactual probe will sit near ceiling unless some facts are very frequent or lookups are sometimes wrong.
7. **K2/K3 cautions:** head-level localization of fact vs context use is shown at 117M-8B and is domain-fragile; causal tracing does not say where to edit [>=1B]; pruned concepts come back
   within a few epochs and move to earlier layers (NER fine-tuned DistilBERT 66M, DistilGPT2 82M, and a GPT-2 the paper sizes at 1.5B). The closest published "prune and regrow" that helps is
   periodic embedding reset (RoBERTa-base, 125M).
8. **Retrieval in pretraining frees weights for local skill but can hurt global context (masked LMs, 8.5M-98M).** K5 must score multi-turn / global-context skill items, not only how many
   facts the weights hold. The paper says the separation is more pronounced for larger models, so at 5M it may be weak.

## 1. Already in the repo (verified there; do not re-derive)

- Physics 3.3 capacity (~2 bits/param at 1000 exposures, ~1 at 100, SwiGLU 1.3x worse at 100, no-MLP equal at 1000 and >1.5x worse at 100, junk 20x, tying helps <=10M):
  research/followup/bits.md s1.1, archfp.md s1.1 (arXiv 2404.05405). Re-read here: Results 4-7 and 10-12 match (PDF lines "Result 4", "Result 5", "Result 6", "Result 10-12").
- Attention-only vs FFN at 6M-87M (SAN better on in-context answers, FFN on LAMBADA): archfp.md s1.5 (arXiv 2607.18363).
- Longpre knowledge conflicts (T5 60M-11B: memorization ratio <15% -> >=50% with size; substituted-passage training 29.5% -> 2.6%), KLLM entity anonymization at 135M/360M:
  followup/retrieval.md s3 (2109.05052, 2607.12831).
- RETRO 148M reproduction, Retro-li 109M (+24M CCA), In-Context RALM, Memorizing Transformers, Memory layers 134M, UltraMem 151M, Shuster 2104.07567 (90M reader uses passages): retrieval.md
  s1-2, archfp.md s2.4.
- Ledger mapping: K1 = P-114 expanded; K4 = P-152; K5 overlaps P-142, P-084, P-085, P-086, P-090; P-083 is the real-text analog.

## 2. Findings (claim | source | scale | relevance)

### 2.1 K1: capacity competition

- F1. Clean intra-document repetition (REPEATED: two paraphrases each of three entities, shuffled, per document) makes a from-scratch model learn both parametric recall and in-context use,
  with in-context use first; SINGLE (one mention per doc) learns only parametric recall. On clean REPEATED data the model "consistently" prefers context under conflict, even when parametric
  knowledge is highly confident. 1% inconsistency (leading paragraph's value replaced with probability p) shifts preference to parametric AND degrades in-context use on unseen entities
  (Table 1, final ICK accuracy uniform/Zipfian: 1% noise 31.5/84.0, 5% 16.8/63.9, 10% 14.1/57.4). Attention on unseen entities drifts from context tokens to subject-name tokens ("gradually
  forgotten" circuits, App. E). Without noise, models over-rely on context at all confidence levels. | Kim et al., arXiv 2510.02370 (v3 Apr 2026), https://arxiv.org/abs/2510.02370 | GPT-2
  architecture, 8L, d512, 8 heads, FFN 2048 (~25.2M body with a non-gated MLP, my arithmetic), 50K train + 50K unseen entities x 4 attributes, 16K steps x 128 x 512 (~1.05B token slots, my
  arithmetic); Pythia-6.9B checkpoints keep AccICKU high while preference shifts to parametric, and the preference shift holds 70M-6.9B (Fig. 8); OLMo in App. F | K1 and K5 nearest neighbor.
  A uniform, repeated fact base is exactly the condition that erodes context use; K1 needs a Zipf-vs-uniform factor and per-checkpoint skill curves.
  Verified: OK. Table 1, Tables 3-4 (arch, 16,000 steps, batch 128, seq 512), Sec. 3.1-3.3 and 4.1 read from the PDF. Fixed "always" to the paper's "consistently"; the high in-context use
  under web data is shown for Pythia-6.9B only.
- F2. Emergent ICL is transient: it rises then fades as in-weights learning takes over, "all while the training loss decreases"; depth does not fix it; wider embeddings soften it; L2
  regularization up to 1e-4 removes the decay (above 1e-4 the model over-regularizes, transience returns and IWL never rises); weight decay on MLP (or ResNet embedder) layers only keeps ICL,
  on attention only does not (Sec. 5-6, Fig. 16). | Singh et al., NeurIPS 2023, arXiv 2311.08360, https://arxiv.org/abs/2311.08360 | 12 layers, embedding 64, 8 heads (~0.59M transformer
  params if the MLP is 4x, my arithmetic; plus a jointly trained ResNet image embedder), Omniglot few-shot sequences, 2 seeds, up to 5e7 steps | K1 must read skill at every checkpoint (peak
  and final); selective MLP weight decay is a cheap training-choice arm (s4.4 A6).
  Verified: OK (PDF: abstract, Sec. 2.2, 5, 6, App. A and D; NeurIPS 2023 in the PDF footer).
- F3. After ICL fades, the asymptotic strategy is a hybrid ("context-constrained in-weights learning", CIWL, implemented as skip-trigrams spread over heads) that shares sub-circuits with
  ICL; ICL cannot emerge quickly without it. They also report a setup where ICL is "truly emergent and persistent". | Singh et al., ICML 2025 (PMLR 267), arXiv 2503.05631 | 2-layer
  attention-only transformers | In-weights competition exists with NO MLPs: K4 shrinking MLPs does not remove it.
  Verified: OK (PDF abstract and Sec. 1; venue via https://proceedings.mlr.press/v267/singh25c.html).
- F4. ICL needs bursty data with many rare classes; ICL and in-weights learning first traded off, and could co-exist under a Zipfian class distribution; only transformers, not recurrent
  models, showed it. | Chan et al., NeurIPS 2022, arXiv 2205.05055 | small transformers, Omniglot-style | Supports P-025 (nonce, bursty values) and a Zipf fact base in K1.
  Verified: OK against the abstract (fixed "only under Zipfian" to the abstract's "could co-exist ... when ... Zipfian").
- F5. Memorizing and generalizing sub-circuits are largely independent; their relative learning rates, not capacity, explain the switch; a memorization scaling law sets the diversity
  threshold. | Nguyen and Reddy, arXiv 2412.00104 | small transformer, synthetic ICL task | K1's "capacity" reading needs a control that distinguishes rate from capacity (A1, A2).
  Verified: OK against the abstract.
- F6. Junk: 7/8 of tokens from bioS(N') with N' = 100M (effectively random, same format) cuts the capacity ratio of the useful fact base 20x at 100 exposures. Training longer, at
  300/600/1000 exposures, it is still 3x/1.5x/1.3x below the NO-junk 100-exposure capacity (not below no-junk at the same exposure). A special token prepended to useful documents cuts the
  20x to 2x, and at 300 exposures matches the no-junk 100-exposure law. Highly repetitive junk (N' = 1K) leaves capacity unchanged. | Physics of LMs 3.3, arXiv 2404.05405 s10 (Results 10-12)
  | GPT-2 family 1M-0.5B, synthetic bios | K1's nonce skill items look like this junk if they share the fact base's slot frames. The "zero-memorization" control arm should be exactly this
  construction (A1).
  Verified: CORRECTED. Numbers match the PDF, but the lane's summary read 3x/1.5x/1.3x as same-exposure losses; the paper compares each against training without junk for only 100 exposures
  (Result 10, second bullet).
- F7. Knowledge is extractable by QA only with mixed training (all bios plus QAs for a fraction p of people, tested on the other 1-p) or augmented bios (varied writing, shuffled sentences);
  bios-only pretraining then QA finetuning "struggles" on the held-out 1-p fraction "irrespective of model size, pre-train time, or finetune parameters" (Result 2). | Physics of LMs 3.1,
  arXiv 2309.14316 | GPT2 (rotary) or Llama, 12L: 124M, 302M, 682M; synthetic bios | K1/K5 fact recall must be probed in a format the model can extract, or recall reads 0 at every load and
  the premise test is void.
  Verified: CORRECTED scale. The lane gave 3.3's "1M-0.5B" for both papers; 3.1 uses 124M/302M/682M. Result 1-3 wording OK.
- F8. The Physics 4.1 playground scores reasoning depth (Depo), breadth (Brevo), knowledge capacity (Capo, 100 exposures, N = 50K-2M bios) and knowledge manipulation (Mano) with tasks that
  "isolate and evaluate" each capability independently. Canon layers: depth 2-4x (RoPE), breadth +30%, capacity +10-15%; gated MLP loses ~30% capacity at 100 exposures (quoted from Physics
  3.3) and Canon recovers about half; linear models ~40% more capacity but Transformers reach 2-4x greater reasoning depth; 1.3B/100B-token real pretraining is noise-dominated and models
  fail the simplest 2-hop task. | Allen-Zhu, arXiv 2512.17351 (V1.1 in NeurIPS 2025; v2 on arXiv) | 8L/12L x d512/768 (12L768D = GPT-2 small); Capo varies 1M-500M | Precedent for K's
  synthetic method; the skill/capacity split is architecture-sensitive; tasks are not mixed with a fact load, so K1 is not covered.
  Verified: OK (PDF lines on Result 2, Sec. 7-8 and the gated-MLP paragraph; the ~30% figure is 4.1 citing 3.3).
- F9. Weak evidence only: character-level nanoGPT, 50 capital facts plus arithmetic with the four (5,7) combinations held out (10 attempts each, 40 total). Alone, only n14 (1.46K params) got
  40/40 held-out; jointly trained, every model got 0/40, and n14 fell to 31.2%/39.4% on seen addition/subtraction and 2.0% on capitals. | Barron and White, ICML 2025 Tiny Titans workshop
  (oral), arXiv 2506.09099 | 1.46K-2.14K (n14, per task) to 10.63M-10.65M (MLT) | Do not cite as support for K1.
  Verified: OK (Tables 2-4 in the PDF).
- F10. Facts are learned after a plateau that coincides with forming attention-based recall circuits; imbalanced distributions shorten the plateau; fine-tuning new facts quickly corrupts
  existing ones. | Zucchet et al., COLM 2025, arXiv 2503.21676 | 8-layer decoder, 44M (Chinchilla-family config) | K1 manipulation check: at 7,630 steps the fact base may still be in its
  plateau. Verified: OK (abstract; 44M read in the PDF, "we train an 8-layer decoder-only Transformer (44M").
- F11. Global bigrams (weights) are learned fast, the induction head (context) slower; weight matrices act as associative memories. | Bietti et al., NeurIPS 2023, arXiv 2306.00802 | 2-layer
  | Expect fact recall and skill to rise at different times. Verified: OK against the abstract.

### 2.2 K4: MLP-to-attention allocation

- F12. At fixed non-embedding params, loss vs r = MLP params / attention params is U-shaped with an interior optimum (Fig. 5; Sec. 3); they say the trend toward ever smaller attention share
  "is not universally optimal". The scaling law, fitted on 80M-297M, gives r = 1.032 for 1B (Panda-1B) and r = 1.055 for 3B (Panda-3B); those two models beat the LLaMA-3.2 configs by 2.1%
  and 0.6% average downstream accuracy. | Bian et al., ICLR 2026, arXiv 2510.18245 | >200 models, 80M-3B, 8B-100B tokens, LLaMA-style with GQA; r is held by changing head count | K4's
  attention-heavy arms may also win on loss, so a K4 skill gain is not automatically a trade; GQA changes their attention count, so their r is not directly Planck's.
  Verified: CORRECTED. The lane said "loss and throughput only"; downstream accuracy is also reported, but only for the two chosen architectures, never as a function of r, and never split
  into fact vs skill. 1.032 confirmed; 1.055 added.
- F13. Hourglass FFNs (residual wide-narrow-wide sub-MLPs, plus hourglass attention) match conventional FFNs on LM and downstream from 113M to 8B at matched params, with 8.7% better training
  compute efficiency at matched accuracy (906M-8B). | Liao et al., arXiv 2602.06471 | 113M-8B | Another way to move MLP params; no fact/skill split measured.
  Verified: CORRECTED against the abstract; removed "controlled studies at 113M", which the abstract does not state.
- F14. See F2: weight decay on MLPs only preserved ICL; on attention only it did not. | 2311.08360 | toy | K4b arm (A6).
  Verified: OK (Sec. 6 and Fig. 16).
- F15. (repo) At 1000 exposures removing all MLPs leaves capacity unchanged; at 100 exposures it costs >1.5x; attention-only 24M models favor in-context answers. | 2404.05405, 2607.18363 |
  Run K4's fact side at ~100 exposures or differences vanish.
  Verified: OK for 2404.05405 (Result 5: "eliminating all MLP layers does not affect its capacity ratio"; Result 6: ">1.5x").

### 2.3 K5: lookup vs memorization, and counterfactual conflict

- F16. LMLM: facts annotated as lookup calls in pretraining text, returned values excluded from the loss. LLaMA2-382M (357.3M non-embedding): FactScore 14.0 -> 31.9, T-REx EM 52.0 -> 58.1,
  PopQA 22.7 -> 50.8 vs the same model trained without lookups; with the database disabled (Table 4) FactScore 12.8 and T-REx 38.5. On TOFU (Fig. 7), loss on return-value tokens drops fast
  under a standard objective and stays high under the masked loss. | Zhao et al., ICLR 2026, arXiv 2505.15962 | GPT2-124M (85.5M non-emb), LLaMA2-176M, GPT2-355M, LLaMA2-382M; Wikipedia, 8
  epochs | Direct K5 precedent at 124M+: add loss masking as a K5 factor (A7).
  Verified: OK (numbers in the PDF text; ICLR 2026 via https://iclr.cc/virtual/2026/poster/10008455).
- F17. Auditing LMLM deletions (12,228 alias-closure deletions, 13 databases): "Parametric leakage is near zero in every variant"; residual post-deletion correctness (0.7% on the released
  database to 13.6% on the Noise variant) comes from near-neighbor retrieval. | Raeesi and Roed, arXiv 2607.00605 | one checkpoint, the released LLaMA2-style 382M LMLM | Supports F16's
  reading. Co-LMLM (continuous keys) reports gains at 360M (arXiv 2607.07707).
  Verified: OK (PDF read this pass; the lane had abstract only). Co-LMLM: OK against the abstract only.
- F18. With an "ideal retrieval" (paraphrase) channel in pretraining, masked LMs store substantially less world knowledge, understand local context and syntax better, and comprehend global
  context worse (LAMBADA); noisy retrieval (25%, 50%, tested at BASE only) interpolates between standard and perfect retrieval. | Samuel et al., "More Room for Language", NAACL 2024, arXiv
  2404.10939 | LTG-BERT masked LMs, 8.5M, 27.7M, 98.2M; ~400M words of Wikipedia | Closest small-scale K5 precedent, but masked LMs, not causal; the conclusion says the separation is
  "especially pronounced for larger models".
  Verified: OK; added the MLM, noise-at-BASE-only and size-trend caveats from Sec. 4-6 of the PDF.
- F19. TALM: T5 fine-tuned with a text tool interface; on NQ with a BM25 index over the union of all NQ oracle contexts, "even the 220M base TALM outperforms 3B XL LM". | Parisi et al.,
  arXiv 2205.12255 | T5 220M/770M/3B | Lookup beats >10x params at 220M, but with a near-ideal index and in fine-tuning, not pretraining. Verified: OK (NQ section of the PDF).
- F20. kNN-LM: WikiText-103 perplexity 15.79 (-2.9) with no training; "particularly helpful in predicting rare patterns, such as factual knowledge". | Khandelwal et al., ICLR 2020, arXiv
  1911.00172 (model size not in abstract) | (repo: no gain in open-ended generation, 2305.14625.) Verified: OK against the abstract.
- F21. REALM: retrieval learned during masked-LM pretraining; Open-QA +4-16% absolute over prior methods. | arXiv 2002.08909 | sizes not checked | Historical anchor only. Verified: OK
  against the abstract.
- F22. Hierarchical memories: 160M anchor plus an 18M memory block fetched from a 4.6B bank matches a regular model with >2x the parameters; trillion-token scale. | Pouransari et al., ICLR
  2026, arXiv 2510.02375 | The bank counts as stored params under Planck's rule (repo archfp.md bottom line 5), so it is not a K5 arm. Verified: OK against the abstract.
- F23. Counterfactual prefixes ("The capital of Poland is London"): the frequency of both the country and the in-context city drives use of the counterfactual; larger models give more
  memorized answers (Pythia 70M-2.8B, GPT-2 series); down-scaling one memory head (15.7) in Pythia-1.4B moved the full world-capitals set from 26% to 86.2% in-context and from 43% to 4%
  memorized (the abstract rounds to 88%). | Yu, Merullo, Pavlick, EMNLP 2023, arXiv 2310.15910 | frequency analysis 70M-2.8B; head analysis Pythia-1.4B/2.8B, GPT2-XL | K5: stratify the
  counterfactual score by fact exposure count; K2: head-value scaling is a cheap causal test.
  Verified: OK (Sec. 5-6 of the PDF; https://aclanthology.org/2023.emnlp-main.615/). Added the 86.2% body figure.
- F24. Fact vs repeated counterfactual in context: in GPT-2 small the factual answer wins 4% of cases (Pythia-6.9B 30%); attention blocks, more than MLPs, carry the competition; two heads
  (L10H7, L11H10) explain ~70% (33% + 37%) of the pro-fact contribution, and ablating both drops factual recall 4.13% -> 0.65%; boosting last-token attention to the attribute position in
  those heads (three heads in Pythia, alpha 5) lifts factual wins to ~50% in both models. Replication (TMLR 06/2025) holds on GPT-2 and Pythia-6.9B, but head specialization shrinks in
  Llama-3.1-8B, the ablation fails on under-represented domains, and non-verbatim counterfactuals lower the effect. | Ortu et al., ACL 2024, arXiv 2402.11655; Dotsinski et al., arXiv
  2506.22977 | GPT-2 small (117M), Pythia-6.9B, Llama-3.1-8B | K5 will likely be at context-following ceiling at 5M; K2 localization is template- and domain-sensitive. Verified: OK (both
  PDFs).
- F25. KAFT: T5 and PaLM showed poor controllability (follow a conflicting context) and robustness (ignore irrelevant context) that "do not scale with increasing model size"; counterfactual
  plus irrelevant contexts in fine-tuning fixed both. | Li et al., arXiv 2211.05110 | T5, PaLM (sizes not checked) | K5 needs irrelevant-lookup items too.
  Verified: OK against the abstract.

### 2.4 K2/K3 (cross-lane, only what bears on arch and training choices)

- F26. Causal-tracing localization does not predict which MLP layer is best to edit; the edited layer is a far better predictor. | Hase et al., NeurIPS 2023, arXiv 2301.04213 | GPT-J 6B,
  GPT2-XL in the appendix [>=1B] | K2's map is not a K3 surgery plan by itself. Verified: OK (PDF abstract and Sec. 1).
- F27. After pruning concept neurons (the location-name concept in models fine-tuned for NER on CoNLL-2003), performance returns within a few epochs of retraining, concepts remap to earlier
  layers, and recovering neurons were "primed" by similar concepts. | Lo, Cohen, Barez, Findings of ACL 2024, arXiv 2401.01814 | DistilBERT 66M, DistilGPT2 82M, and GPT-2, which the paper
  describes as 1.5B (checkpoint not named) | K3 needs a relearning probe (A9).
  Verified: CORRECTED scale. The lane said "GPT-2 124M"; the paper's model section says 1.5 billion parameters.
- F28. Resetting the token-embedding layer every K = 1000 updates during pretraining (K tried: 100, 1000, 5000) gives faster, better low-data adaptation to new languages. | Chen et al.,
  NeurIPS 2023, arXiv 2307.01163 | RoBERTa-base (125M) | Closest published "reset part of the net, keep the body" result; a periodic-reset variant for K3. Verified: OK (PDF).
- F29. Larger pretraining weight decay raises plasticity (gain after fine-tuning). The pretraining-loss optimum at 20 tokens/param was 0.5 (Llama-2 0.5B, 1B), 0.6 (OLMo-2 1B) and 1.0
  (Llama-2 4B); at 140 tokens/param the default 0.1 won. | Han et al., arXiv 2602.11137 | Llama-2 0.5B-4B, OLMo-2 1B [>=0.5B] | Hold weight decay fixed across K3 arms; it confounds regrowth.
  Verified: CORRECTED. The lane said "optimum ~0.5-0.6"; 4B was 1.0 and the long-run group was 0.1.

## 3. Arithmetic for the 5M shape (my arithmetic; shape from E2_lr_transfer/notes.txt)

- Shape: d 192, 8 layers, 3 heads x 64, SwiGLU 488, vocab 8192 tied; 5,010,133 total, 1,572,864 embedding, 3,437,269 body.
  Verified: matches experiments/E2_lr_transfer/notes.txt line 24.
- Physics 3.3 counts params "after excluding all unused tokens in the embedding layer". If K data uses U distinct tokens, counted params = body + 192 U: U = 1,000 -> 3.63M; U = 8,192 ->
  5.01M. Report bits/param both ways.
- Capacity ceilings: 2 bits/param -> 7.3M-10.0M bits; 1 bit (100 exposures) -> 3.6M-5.0M; with the gated-MLP 1.3x penalty at 100 exposures -> 2.8M-3.9M bits. bioS people are 47.6 bits each
  (Physics 3.3 Remark 4.4): 10.0M bits ~ 210K people.
- Token cost with compact ~20-token, ~60-bit people (repo bits.md s0): at 100 exposures, 3.9M bits ~ 64K people ~ 128M tokens, about half of the 250M-slot screen. At 1000 exposures the whole
  budget holds at most 12.5K people (0.75M bits, ~0.15 bits/param). So a K1 "high load" near capacity is only reachable at ~100 exposures, or with a longer run.
- Steps: the screen is 7,630 steps x 32,768 slots; Physics 3.3 states models "typically need at least 50K training steps regardless of batch size" (verified in the PDF). Reuse the repo's
  bits/param rig (LEDGER baseline row "Calibrate a bits-per-parameter rig") on the K setting before fixing K1's loads.
- K4 ratio: attention 4d^2 = 147,456 per layer, SwiGLU 3 x 192 x 488 = 281,088, so r = 1.906 now (re-computed). With hidden = m d, r = 0.75 m: m = 8/3, 2, 1.33, 0.67 give r = 2.0, 1.5, 1.0,
  0.5. Bian's loss optimum (1.03-1.06, GQA configs) sits inside. The body also holds ~8.9K norm and other params per model (3,437,269 - 8 x 428,544), which r ignores.

## 4. Implications for K1-K5

### 4.1 Already known (cite, do not re-discover)
- Capacity numbers and their exposure dependence (F6, F15, repo); ICL vs in-weights competition, transience, Zipf coexistence, MLP-only weight decay (F1-F5); small models follow context and
  frequency drives parametric preference (F23-F24, Longpre in repo); loss masking stops memorization of looked-up facts at 124M+ (F16-F17); retrieval in pretraining leaves fewer facts in
  8.5M-98M masked-LM weights (F18); an interior MLP/attention loss optimum near r ~ 1 at 80M+ (F12); pruned concepts get relearned (F27). A K result that only restates one of these is a
  replication, and should say so.

### 4.2 Genuinely untested at 5M (from this lane's search)
- U1 (K1): skill vs fact-load curve with Planck skills (corrections, same-type binding, context QA, nonce values) mixed with a memorizable base in ONE from-scratch LM. Nearest: F1
  (copy-from-context only, ~25M body), F2 (classification toy), F9 (2K toy).
- U2 (K2): unit-level separability of fact recall vs in-context skill with ground-truth facts in a from-scratch 5M model. Published localization is on pretrained web models of 117M+ (F23,
  F24, F26).
- U3 (K3): pruning fact units then skill-only continuation as a SKILL booster. Nearest: F28 (embedding reset for plasticity), F27 (relearning, not skill gain).
- U4 (K4): MLP ratio scored on skill and fact accuracies at matched params. F12 measured loss vs r (downstream accuracy only for two chosen models), Physics 3.3 capacity only.
- U5 (K5): lookup fraction p with a counterfactual test at 5M from scratch; F16 starts at 124M, F18 is masked-LM with no counterfactual.

### 4.3 Pitfalls
- P1 Manipulation check: prove fact recall rises with load in K1 (F7, F10, s3 steps). If recall is ~0 at every load, "no crowding" is vacuous.
- P2 Equal total tokens confounds fact load with skill dose (fewer skill tokens at high load). Add the A2 accounting.
- P3 Same-format nonce skill data is "junk" to the fact store (F6): 20x at 100 exposures, still 1.3x-3x with more exposures.
- P4 Transience: final-checkpoint skill can hide a peak-then-decay (F1, F2); the WSD decay phase may move it. Log every ckpt.
- P5 Attention stores facts too (F3, F15, repo Nichani): K2 "fact units" may be spread over heads and MLPs; K4 cannot starve fact storage by shrinking MLPs alone.
- P6 Context-following ceiling at 5M (F23, F24, Longpre): give the counterfactual headroom with high-exposure facts.
- P7 Global context (F18): a lookup channel can weaken long-range use; score multi-turn skill items inside K5.
- P8 bpb is not behaviour for attention changes (repo SCREENS.txt, arXiv 2605.20798): K4 decides on accuracies, not loss.
- P9 K3: relearning and remapping (F27), weight-decay-dependent plasticity (F29), and localization that does not say where to cut (F26). Keep the random-unit and skill-unit controls; add a
  same-layer random control.
- P10 Head granularity at 5M is coarse (24 heads); F24's replication shows head roles are template-sensitive: build K2 probes from several slot frames and report per-frame.

### 4.4 Proposed additions (each one change against the K spec; for the SPEC author to accept or drop)
- A1 (K1) Zero-memorization control: replace the fact base by same-format people drawn fresh every time (never repeated), same tokens. Separates memorization load from fact-format dose (F6
  construction).
- A2 (K1) Two accountings: equal total tokens (substitution) and equal skill tokens (fact tokens added on top).
- A3 (K1, K5) Skill accuracy at every checkpoint (ckpt_every 153 on the 5M schedule), report peak, final and area.
- A4 (K1) Fact-document tag token on/off (F6 Result 12), only if recall is too low to create load.
- A5 (K1, K5) Zipfian vs uniform fact exposure (F1, F4).
- A6 (K4b) Weight decay on MLP matrices only (or raised there) at the baseline shape: the training-choice twin of shrinking the MLP (F2, F14). Cheap: optimizer param groups, no shape change.
- A7 (K5) Loss mask on looked-up value tokens (LMLM) as a factor at p = 1.0, since supervised copy still trains memory.
- A8 (K5) Lookup noise 0% vs 1% (F1) and an irrelevant-lookup item family (F25); stratify counterfactual by exposure (F23).
- A9 (K3) Relearning probe: after the regrow phase, a short fact-only refresh; fast return means pruning hid facts rather than freeing capacity (F27). Plus a periodic-reset variant (reset
  fact units every K steps during continuation, F28).
- A10 (K2) Head-value scaling and attention-to-attribute boosts (F23, F24) as cheap causal tests beside unit ablation.

## 5. Sources (all fetched 2026-09-27 unless marked repo)

arXiv (https://arxiv.org/abs/ + id): 2510.02370 2311.08360 2503.05631 2205.05055 2412.00104 2404.05405 2309.14316 2512.17351
2506.09099 2503.21676 2306.00802 2510.18245 2602.06471 2505.15962 2607.00605 2607.07707 2404.10939 2205.12255 1911.00172
2002.08909 2510.02375 2310.15910 2402.11655 2506.22977 2211.05110 2301.04213 2401.01814 2307.01163 2602.11137.
Venues: https://proceedings.mlr.press/v267/singh25c.html ; https://iclr.cc/virtual/2026/poster/10008455 ;
https://aclanthology.org/2023.emnlp-main.615/ ; https://aclanthology.org/2024.findings-acl.492/ ; arXiv metadata comments
(2510.18245 ICLR 2026, 2402.11655 ACL 2024, 2506.22977 TMLR 2025, 2404.10939 NAACL 2024, 2301.04213 and 2307.01163
NeurIPS 2023, 2506.09099 ICML 2025 workshop, 2503.21676 COLM 2025). Repo: research/followup/{bits,retrieval,archfp}.md.
