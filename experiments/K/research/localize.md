# Track K research lane: localize (where facts and skills live; remove, route, regrow)

Written 2026-09-27 for track K (K1-K5); fact-checked the same day ("Verified" lines). No model was loaded, trained or run.
Method: WebSearch/WebFetch on arXiv abstract and HTML pages, transformer-circuits.pub, ACL Anthology, arXiv API (venues);
pdftotext on PDFs (Chang, Jang, Geva, Zhang and Nanda, Lo, Yu, Hase, Ash and Adams). Read level: (abstract), (HTML),
(PDF). [>=1B] = evidence only at 1B+. Sibling `capacity.md` covers capacity and ICL-vs-IWL competition.

## 0. In one screen

1. **Post-hoc localization of memorized content works partly, and is least clean at small scale.** Pruning-style masks
   (HardConcrete, Slimming) find injected sequences best, but the units they find also hold related ones; at 125M,
   gradient-ascent unlearning wrecked dialogue F1 (9.4 to 2.6) where 2.7B kept it (11.5 to 11.1).
2. **Training-time routing is the better-supported route to Max's idea.** Gradient routing (28M, 0.7B) and MemSinks
   (344M to 1.7B) route chosen data into chosen units during training, then delete them. Caveat: routing beat data
   filtering only with partial labels; fully labeled (K1's case), filtering (a never-fact model) was most robust.
3. **Fact storage is not tied to MLPs when training is long enough.** Removing every MLP leaves bits/param unchanged at
   1,000 exposures (Physics 3.3); attention value matrices can store facts too. MLPs matter at 100 exposures and when
   both parts are trainable (2-layer model: frozen MLP 1.13 vs 2.98 bits per trainable param, 19% vs 100% accuracy).
4. **Removed knowledge comes back when the retraining data contains it.** Pruning half of all neurons (ranked by concept
   saliency) then fine-tuning on the same NER task restores F1 within 2 epochs at 66-82M; fine-tuning on some
   "unlearned" facts recovers at least 88% of the rest (Llama 3 8B). K3 needs a relearning probe, not recall alone.
5. **Ablation readouts mislead.** Self-repair restores ~70% of an ablated middle layer's logit effect (7B); likelihood
   and generation disagree on which layers matter; causal tracing does not pick the best layer to edit.
6. **Not found at or near 5M:** the gaps listed in 3.2 (none of K2-K5's core measurements has a small-scale prior).

## 1. The 5M model in numbers (my arithmetic from E2 MODEL: d 192, 8 layers, 3 heads x 64, SwiGLU 488, vocab 8192 tied)

- Body 3,437,269: MLPs 3 x 192 x 488 x 8 = 2,248,704 (65%); attention 4 x 192^2 x 8 = 1,179,648 (34%); norms, gates,
  scalars 8,917. Tied embedding 1,572,864 (31% of total), so entity rows serve input and output. Units a K2 search can
  see: 3,904 SwiGLU hidden units, 24 heads, 8 layers, 8,192 rows.
- Fact ceilings (Physics 3.3 rates on 3.4M body or 5.0M total): ~6.9-10.0M bits at 1,000 exposures, ~3.4-5.0M at 100.
  With 1:7 useful:junk at 100 exposures Physics 3.3 measured a 20x loss (Result 10); K1's skill tokens are junk from
  the fact side, so the mixed-run ceiling may be ~0.2M bits (an extrapolation, not measured).
- Token budget: a 10-token fact line at 100 exposures costs 1,000 tokens. 25% of a 250M screen = 62,500 facts (625K
  bits at 10 bits per value); 50% = 125,000; at 1,000 exposures 25% buys only 6,250. Only 100-exposure loads fit at 5M
  (as capacity.md also concludes); whether they saturate anything depends on the junk effect, which needs a pilot.
- Superposition is forced: 62,500 facts over 3,904 MLP units is ~16 facts per unit, so one-fact "knowledge neurons"
  cannot exist here; K2 should localize with masks over units and weights, not neuron lists.

## 2. Findings

### 2.1 Where facts live (K2, K4)
- **Knowledge is not stored in individual layers; removing one can remove much more than 1/L of it.** Removing all MLPs
  or cutting MLP width by 1/4 leaves bits/param unchanged at 1,000 exposures (Result 5); at 100 exposures removing MLPs
  costs over 1.5x (Result 6). https://arxiv.org/html/2404.05405 (HTML), Physics of LMs 3.3. Scale: GPT-2 1M-0.5B. K2:
  whole-layer ablation overstates a layer's share. K4: fact side is known (3.5).
  Verified: OK. "eliminating all MLP layers does not affect its capacity ratio"; "more than 1.5x"; s8.1; "20x" (R10).
- **Attention and MLP are interchangeable associative memories.** Linear and MLP memory capacity scales linearly with
  parameters; a 1-layer attention+MLP model can store facts in value matrices or in the MLP (trade-off).
  https://arxiv.org/abs/2412.06538 (abstract), ICLR 2025. Scale: theory plus shallow models. K4: shifting parameters
  from MLP to attention need not reduce fact storage. Verified: OK ("up to log factors"; venue per proceedings.iclr.cc).
- **When both are trainable, MLPs carry most storage and attention carries retrieval.** 2-layer, 4-head model: standard
  2.98 bits per trainable param (100% accuracy), frozen-at-init MLP 1.13 (19%), frozen QK 2.25 (69%), random fixed
  attention (MixiT) 2.18 (67%); bits = 9 x 512^2 x accuracy over trainable params only. MixiT struggles on
  needle-in-a-haystack once m_max >= 20 (v3: 11.24% at 30) yet gets 100% on decimal and modular addition.
  https://arxiv.org/html/2506.01115v1 (HTML; v3 retitled, no venue). Scale: memorization 2 layers; algorithmic 2-8; LM
  8-12; width 512-1024. K4: freezing is not removing. Verified: OK with fixes: numbers are the 2-layer model's, not
  "2-12 layers"; denominator checked (trainable params); accuracies added; v3 Table 4 has the same values.
- **Syntax localizes to MLP neurons just like facts do.** Knowledge-neuron edits overturn at most 5.2% of BLiMP
  predictions; factual edit reliability 1.66% to 47.86%; determiner number sits in two neurons.
  https://arxiv.org/html/2405.02421 (HTML), ICLR 2024. Scale: BERT-base, GPT-2 base/XL, LLaMA-2 7B. K2: "fact units"
  may carry grammar; measure specificity against skill items. Verified: OK; numbers, models, venue (Spotlight) match.
- **A low-layer rare-token head drives paragraph memorization.** 442 of 13,450 paragraphs memorized; memorized ones have
  larger lower-layer gradients; head L1H2 attends to rare tokens (corr -0.97 with unigram frequency); fine-tuning only
  the top 0.1% gradient weights unlearns about as well as all weights. https://arxiv.org/html/2403.19851 (HTML). Scale:
  GPT-Neo 125M. K2: K1's nonce names are rare tokens, so such a head may serve recall and in-context binding. Verified:
  OK (442 with EM=50; gradient gap in attention and MLP; layer 1 head 2 at -0.97; 0.1% quote matches).

### 2.2 Localization methods and how they fail (K2)
- **Pruning-style masks localize best; all methods confuse related items.** INJ (facts injected into known 1% or 0.1% of
  FFN neurons), GPT2-124M, ratio 1%, Recall@5%: HardConcrete 87.4, Slimming 80.7, Zero-Out 53.8, IG 49.9, Activations
  13.3, random 5.0. DEL (drop 0.5% of neurons, GPT2-XL, absolute points): Slimming cuts target accuracy 57.8 while other
  memorized sequences lose 6.4 (HardConcrete 57.1/4.8; Pythia 6.9B HardConcrete 57.7/14.7). Related sequences
  (addresses; poems, Shakespeare, Bible) share HardConcrete neurons; the paper leaves open method error vs truly shared
  neurons. Memory is spread over layers; layer-1 dropout hurts everything; Zero-Out single-neuron effects do not predict
  joint dropout. FFN neurons only, zero ablation. https://arxiv.org/abs/2311.09060 (PDF Tables 2-3), NAACL 2024. Scale:
  GPT2 124M, GPT2-XL, Pythia 2.8B/6.9B. K2: K1's fact list as INJ-style ground truth, HardConcrete masks, report Self /
  other-fact / skill / held-out bpb. Verified: OK; all R@5% and DEL values match the PDF; DEL is GPT2-XL, not 124M.
- **Localization does not predict the best place to edit.** Layer-6 tracing effect vs ROME rewrite score: rho -0.13.
  https://arxiv.org/abs/2301.04213 (PDF), NeurIPS 2023. Scale: GPT-J 6B [>=1B]. K3: choose prune targets by what mask
  removal does, not by tracing strength. Verified: OK (PDF read; scale was "not read").
- **Patching results change with the corruption method (Gaussian noise vs symmetric token replacement) and the metric
  (logit difference vs probability).** https://arxiv.org/abs/2309.16042 (PDF), ICLR 2024. Scale: GPT-2 XL (facts),
  GPT-2 small (IOI). They recommend STR (in-distribution corruptions) and logit difference. K2: pre-register both; STR
  fits K1 (same-type entity swap). Verified: OK; the abstract does not name GN/STR, the PDF body does.
- **Self-repair (the Hydra effect).** With resample ablation, downstream layers and reduced MLP effects restore about 70%
  of an ablated middle layer's logit reduction; zero ablation is out of distribution for models trained without layer
  dropout. https://arxiv.org/html/2307.15771 (HTML). Scale: Chinchilla 7B [>=1B]. K2: report mean or resample ablation
  beside zero; a small drop is not proof a unit is unused. Verified: OK (both quotes found in the HTML).
- **Likelihood and generation disagree.** Likelihood says most deep layers can go; generation shows middle and deep
  layers are needed for reasoning and coherence; knowledge leans on shallow layers, key-value retrieval on shallow-to-mid
  heads. https://arxiv.org/html/2510.02091v1 (HTML), ICASSP 2026. Scale: LLaMA-3.1-8B, Qwen3-8B, LLaMA-1 7B [>=1B].
  K2/K3: score skills by greedy generation as well as margins. Verified: OK (venue per arXiv comment on v4).

### 2.3 What pruning removes first (K2, K3)
- **Fact recall breaks before in-context use.** Removing >30% of weights gives a >5% relative fact-recall drop while
  60-70% removal largely preserves in-context use; dense down-scaling shows the same split.
  https://arxiv.org/html/2310.04680 (HTML), ICLR 2024. Scale: OPT-13B/30B, LLaMA-13B/33B [>=1B]. Same direction
  (LLM-KICK): knowledge-intensive tasks degrade at 25-30% sparsity while >=50%-sparse LLMs stay robust at in-context
  retrieval and summarization. https://arxiv.org/abs/2310.01382 (abstract; HTML for models), ICLR 2024. Scale: Vicuna
  7B/13B/33B, LLaMA-2 7B [>=1B]. K3: magnitude pruning is the baseline a targeted prune must beat. Verified: OK;
  quotes match; model sizes now read (were "not read"); Jin's venue from the arXiv journal ref.

### 2.4 Removal, collateral damage and relearning (K3)
- **Small models pay more for unlearning.** Gradient-ascent unlearning of 32 sequences (UL+): 125M classification avg
  43.4 to 39.9, dialogue F1 9.4 to 2.6; 1.3B dialogue 11.5 to 8.5; 2.7B 52.3 to 51.9 and 11.5 to 11.1. Forgetting 128
  at once degrades every size; 4 sequential chunks of 32 show "almost no degradation" (Fig. 2b).
  https://aclanthology.org/2023.acl-long.805.pdf (PDF; arXiv 2210.01504), ACL 2023. Scale: GPT-Neo 125M/1.3B/2.7B. K3:
  removal costs skill most at small size. Verified: OK; Table 2 rows match (mean of 5 samplings, s = 32).
- **Unlearning mostly blocks access.** Fine-tuning on accessible facts recovers at least 88% of pre-unlearning accuracy
  on the rest, for every method tested (gradient difference, RMU, RIA) on pretraining facts.
  https://arxiv.org/abs/2410.08827 (abstract; HTML setup). Scale: mainly Llama 3 8B [>=1B]. K3: relearning probe.
  Verified: OK; scale now read (was "not read").
- **Pruned concepts relearn fast when the retraining data contains them.** Prune the most location-salient half of all
  neurons, fine-tune again on CoNLL-2003 NER: DistilBERT F1 0.034 pruned, 0.915 after 2 epochs, 0.928 after 8 (base
  0.937); DistilGPT2 0.434 base, 0.385 pruned, 0.454 after 2, 0.507 after 8. A same-count random prune hurt DistilGPT2
  more (0.443 to 0.219). The concept moves to earlier layers and to neurons with similar concepts.
  https://arxiv.org/abs/2401.01814 (PDF). Scale: DistilBERT 66M, DistilGPT2 82M, GPT-2 1.5B. K3: freed units go to
  whatever the continuation data rewards; skill-only data predicts no fact regrowth (see also MemSinks, capacity.md 2.5).
  Verified: WRONG in 2 details, fixed: half of all neurons, not of location neurons; DistilGPT2 beat base at 2 epochs.

### 2.5 Designed localization at training time (K3)
- **Gradient routing ("Expand, Route, Ablate").** Add randomly initialized dimensions to target layers (MLP and
  attention blocks in the figure), give select forget tokens reduced or negative LR in the original dimensions, delete
  the new ones, then a few retain fine-tune steps. https://arxiv.org/html/2410.04332 (HTML). Scale: TinyStories 28M
  (forget = forest/tree/woodland stories, 20% of data); 0.7B with 20 virology tokens routed to MLP dims 0-79 of layers
  0-7 (after 20 retraining steps, loss +0.182 on WMDP-bio forget vs +0.032 on FineWeb-Edu). K3: closest prior.
  Verified: OK with scope fix. "When labeling is limited (<100%), ERA dominates" data filtering; at 100% "Data
  filtering ... is the most robust to retraining"; retain cost "proportional to the amount of data" routed.
- **MemSinks.** 30% of MLP hidden units are sinks (70% shared), masked by sequence ID with sink dropout p = 0.3; dropping
  sinks closes the train-vs-validation loss gap by at least 50% with comparable validation loss (TS-Repetition).
  https://arxiv.org/html/2507.09937 (HTML), ICML 2025. Scale: GPT-2-Medium-like ~344M on TinyStories; SmolLM 360M/1.7B
  on 1B/2B tokens. K3: key sinks by entity ID. Verified: OK; natural memorization "significantly harder to remove".
- **Skills can be localized too.** Grafting 0.01% of parameters (~8,500) recovers >95% of fine-tuned accuracy
  (RoBERTa-base; GPT-2 small needs 0.05%); task skills sit in "somewhat disjoint regions".
  https://arxiv.org/html/2302.06600 (HTML), ICML 2023. K2/K3: a skill-side mask is as legitimate as a fact-side one.
  Verified: OK; "almost disjoint" softened to the paper's wording.

### 2.6 Prune, reset and regrow (K3)
- **RigL**: drop by magnitude, regrow by gradient ("parameter magnitudes and infrequent gradient calculations");
  ImageNet ResNet-50/MobileNet, RNNs on WikiText-103. https://arxiv.org/abs/1911.11134 (abstract), ICML 2020. Verified:
  OK. A search summary (not fetched) of SRigL, https://arxiv.org/abs/2305.02299: RigL at >90% ablates whole neurons.
- **Continual backprop**: re-initialize a small fraction of less-used units after each example; under ordinary backprop
  ImageNet binary-task accuracy fell 89% to 77% by task 2000; L2 plus weight perturbation also helps.
  https://arxiv.org/abs/2306.13812 (abstract). Scale: MNIST and ImageNet (the only domains in this abstract). K3:
  "regrow" = re-initialize the pruned units, not leave them at zero. Verified: OK (quote matches).
- **Warm starts generalize worse than fresh starts; shrink-and-perturb fixes it.** https://arxiv.org/abs/1910.08475
  (abstract; PDF for the name), NeurIPS 2020. K3: a gain from pruning plus continuation may be a generic plasticity
  effect; shrink-and-perturb of the whole model is the control. Verified: OK; named in the PDF body, not the abstract.
- **Knowledge entropy decays in pretraining; scaling up inactive FFN memory vectors (lowest p = 50% by coefficient,
  multiplier mean(C)/C x q, q >= 1) modestly improves new-knowledge acquisition and retention**, still below mid-stage
  checkpoints of similar entropy. https://arxiv.org/html/2410.01380 (HTML), ICLR 2025 Oral. Scale: OLMo 1B/7B, Pythia
  1.4B [>=1B]. K3: mirror image of Max's idea; log per-unit usage in K2. Verified: OK.

### 2.7 Skill circuits (K2, K4)
- **Knocking out induction heads at test time greatly reduces in-context learning in small models.**
  https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html (HTML). Scale: 34 models incl.
  1-3 layer attention-only. K2: positive control; the skill side should show a head-level signature. Bigrams come
  before induction heads (https://arxiv.org/abs/2306.00802, abstract). Verified: OK (Argument 3 wording).
- **Memorizing and generalizing solutions use different circuits.** Memorization: Att1/MLP1 encode pairs, Att2 pools,
  MLP2 decodes; generalization: a statistical induction head; kinetic threshold K1* ~ 94 Markov chains, representational
  K2* ~ 7,000. https://arxiv.org/html/2604.12151v1 (HTML; Gibson, Cui, Reddy). Scale: 2 layers, D 64, C 10 states.
  Related: "largely independent" sub-circuits set by learning speed (Nguyen and Reddy, https://arxiv.org/abs/2412.00104)
  and sub-circuits shared by ICL and CIWL (Singh et al., https://arxiv.org/abs/2503.05631, PDF). K2: expect memorizing
  in MLPs, skill in heads, some sharing. Verified: OK; the related papers' authors corrected (not Gibson et al.).

### 2.8 Lookup versus memory (K5)
- **A memory head arbitrates context vs memory, and can be turned down.** Scaling head 15.7's value vector (alpha -0.7)
  moved Pythia-1.4b from 26% in-context / 43% memorized answers to 86.2% / 4%. https://arxiv.org/abs/2310.15910 (PDF).
  Scale: head search only on Pythia-1.4b/2.8b and GPT2-xl [>=1B]; the frequency analysis spans Pythia 70m-2.8b and
  GPT-2 ("especially in larger models"). K5: look for such a head; at 5M one may not exist. Verified: OK with scale
  fix (intervention [>=1B] only; 70m appears only in the frequency analysis).
- **Masking looked-up fact values from the loss offloads knowledge (LMLM).** Loss excludes retrieved values and the
  ending lookup token; 382M vs the same data without lookups: T-REx EM 58.1 vs 52.0, PopQA 50.8 vs 22.7, FactScore
  31.9 vs 14.0; TOFU forgetting by deleting DB entries. https://arxiv.org/html/2505.15962 (HTML). Scale: GPT-2
  124M/355M, LLaMA2-style 176M/382M, ~3B Wikipedia tokens x 8 epochs. K5: add a masked-value factor. Verified: OK.
- **Retrieval substitutes for pretraining above ~4.14 tokens per parameter, most at 30M, for knowledge tasks, not
  reasoning** ("weak or negative gains"). https://arxiv.org/html/2604.00715v1 (HTML; v2 retitled). Scale: OLMo-2
  30M-3B, 100B DCLM tokens. Verified: OK.

### 2.9 Other pointers (abstract level, not re-checked in the fact-check pass)
- Where facts live: causal tracing puts facts in mid-layer MLPs at the last subject token (GPT-2 XL, GPT-J [>=1B];
  https://arxiv.org/abs/2202.05262), a hypothesis K2 tests, not assumes; FFN layers act as key-value memories, one
  hidden unit = one key/value pair (https://arxiv.org/abs/2012.14913, PDF); knowledge neurons (https://arxiv.org/abs/
  2104.08696); recall runs over several additive paths, so single-unit ablations undercount (arXiv 2402.07321);
  knowledge-critical subnetworks are 98%+ sparse (arXiv 2310.03084); masks find task-specific weights, not reused
  modules (arXiv 2010.02066); small data is stored as datapoints in superposition (toy models,
  https://transformer-circuits.pub/2023/toy-double-descent/index.html), so expect K1 facts to be spread.
- Heads and units: 38 of 48 encoder heads pruned for -0.15 BLEU (https://arxiv.org/abs/1905.09418); retrieval heads
  are under 5% of heads (Llama-2 7B [>=1B], https://arxiv.org/abs/2404.15574), and 5M has only 24, so ablate all;
  dead ReLU neurons in OPT (https://arxiv.org/abs/2309.04827), while Planck's SwiGLU has no hard-zero units.
- Removal and reuse: example-tied dropout (https://arxiv.org/abs/2307.09542, image classifiers); forget-and-relearn
  (https://arxiv.org/abs/2202.00155); PackNet (https://arxiv.org/abs/1711.05769); prune plus distillation beats
  scratch only far above 5M (15B to 8B/4B, https://arxiv.org/abs/2407.14679); goldfish loss cuts verbatim memorization
  cheaply (Llama-2 class [>=1B], https://arxiv.org/abs/2406.10209), a cousin of LMLM masking for K5.

## 3. Implications for K1-K5

### 3.1 Known, do not re-discover
Facts break before in-context use under pruning and dense down-scaling (Jin 13B+; LLM-KICK 7B+). Pruning masks beat IG
and activations for localizing memorized data at 124M-6.9B (Chang). Facts can live in attention when MLPs are gone
(Physics 3.3, Nichani). Removed knowledge relearns fast from the same task or related facts (Lo; Deeb and Roger at 8B).
Training-time routing isolates memorization (gradient routing 28M/0.7B, MemSinks 344M+), but with full labels data
filtering stays the most robust remover (Cloud et al.). Induction heads carry in-context copying in small models
(Olsson). Unlearning costs a 125M model far more than a 2.7B one (Jang).

### 3.2 Untested at 5M (as far as this search found)
(a) Fact recall and in-context binding/update skill localized and ablated on the same from-scratch LM with known facts.
(b) Whether units freed from facts, re-initialized and trained on skill data, raise skill beyond an equal-compute
unpruned or never-fact model: every prior found measures forgetting or recovery, none skill gain. (c) Route-then-cut for
facts followed by skill training. (d) LMLM-style value masking below 124M, with a counterfactual lookup. (e) Allocation
(K4) with an in-context skill measured at matched total params.

### 3.3 K2 protocol suggestions
1. Ground truth: K1's fact list. HardConcrete masks over MLP units, heads and entity rows, trained once to kill fact
   recall and once to kill skill; report overlap (Jaccard), per-unit (fact drop, skill drop) scatter, other-fact and
   held-out bpb effects (Chang's Self / Neg / Rand). Positive controls: induction-head knockout hits skill; an injected
   INJ-style fact set is found. Negative control: random masks of equal size.
2. Ablations: all heads (24 single, 276 pairs), each layer's MLP, unit masks; mean or resample beside zero; same-type
   entity swap (STR); logit difference (Zhang and Nanda). 3+ seeds; report unit-set agreement across seeds (capacity.md:
   attribution flips sign at 26M). Score skill by greedy generation as well as margins.

### 3.4 K3 arms (beyond Max's four)
- Magnitude prune and random-unit prune at the same count (Jin baseline; Lo's random prune hurt DistilGPT2 more than
  concept pruning, so the random arm is not a trivial floor). Zero vs re-initialize pruned units (continual backprop);
  shrink-and-perturb of the whole model as the plasticity control.
- Route-then-cut: fact lines routed into a reserved 25-30% MLP block from step 0 (gradient routing / MemSinks keyed by
  entity), cut, re-initialize, continue on skill data. Compare against the never-fact model: K1 labels every fact line,
  the regime where Cloud et al. found data filtering most robust. If cutting post hoc, cut in stages (Jang: 4 chunks
  of 32 barely hurt where 128 at once did).
- Residual-fact test: fine-tune on half of the facts for a fixed small budget, test the other half, against a never-fact
  model (Deeb and Roger). Recall at zero is not evidence of removal.

### 3.5 K4 and K5
K4: fact capacity is mostly known (little change at 1,000 exposures, >1.5x loss without MLPs at 100, Physics 3.3), and
the 250M budget allows only ~100 exposures for a sizable load, the MLP-sensitive regime. The open side is skill:
trainable attention is needed for retrieval (Dong et al., 2-layer; frozen MLPs cut memorization accuracy to 19% there),
induction is head-borne (Olsson). "More heads" at d 192, head_dim 64 changes d or head_dim; keep total within 2%.
K5: add a factor, looked-up value in the loss vs masked (LMLM); keep the p = 0 arm extractable (Physics 3.1); track
memorization over checkpoints (capacity.md, Singh); in the counterfactual arm scale heads to find a memory head (Yu et
al., shown only at >=1.4B, so exploratory at 5M).

### 3.6 Pitfalls
Self-repair hides unit importance; zero ablation is off-distribution. Single-unit scores do not add up. Facts share units
with grammar and related facts; at 5M every unit is polysemantic. Tracing does not pick edit sites. Zero recall can hide
facts that relearn fast. SwiGLU has no free dead units. Likelihood and generation disagree.
Not verified: 2.9's items; the SRigL claim (search summary); capacity.md's gradient-routing figure numbers. Resolved in
the fact-check: Hase (GPT-J 6B), Deeb and Roger (Llama 3 8B), LLM-KICK (Vicuna 7B-33B), Zhang and Nanda (GPT-2
small/XL), Dong et al. denominator (trainable params).
