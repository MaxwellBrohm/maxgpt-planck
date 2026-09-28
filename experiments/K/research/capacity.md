# Track K research lane: knowledge capacity and competition

Written and fact-checked 2026-09-27 for track K (K1-K5). No model was loaded or trained. Every number was grepped
from arXiv PDFs (pdftotext) in the session scratchpad; the fact-check downloaded every source in sections 1-4 again.
Abstract-only reads say so. Each entry ends with a "Verified" line: OK = claim and scale match the source; FIXED =
the entry was wrong or overstated and the text above that line is already corrected. Scale tags: [>=1B only] =
evidence only at 1B+; [toy] = under 1M params or non-language task. "No prior found" means exactly that.

## 0. Already in the repo: do not re-derive

`research/followup/bits.md` and `bits.verify.md` already hold, and K must cite rather than re-measure: Physics 3.3's
>= 2 bits/param at 1000 exposures and >= 1 at 100 (1M-0.5B); junk and domain-token Results 10-12; gated MLP 1.3x at
100 exposures; no-MLP capacity; int8/int4; MoE; Canon gains; Gu 2505.18091 phase transition; Zucchet 2503.21676;
Morris 2505.24832; O'Neill 2607.11020; the ICL-transience omission (Singh 2311.08360, Chan 2410.23042). The ledger's
bits rig (bioD-lite, ~60 bits and ~20 tokens per person) is K1's fact metric. bits.md 6.5 item 7 and P-114 = K1.

## 1. Capacity and where facts live (K1, K4)

- **Junk that looks like knowledge slows fact learning; repetitive data does not; a domain token fixes most of it.**
  With 7/8 of tokens from bioS(N' = 100M) "junk", capacity for useful data "may degrade by 20x" at 100 exposures. At
  300/600/1000 exposures it is still 3x/1.5x/1.3x below the no-junk 100-exposure law. 7/8 from bioS(N' = 1K) (highly
  repetitive) leaves the 100-exposure capacity "unchanged". A special token at the start of useful data cuts the
  100-exposure loss to 2x, and at 300 exposures matches the no-junk 100-exposure law. Source:
  https://arxiv.org/abs/2404.05405 Results 10-12. Scale: GPT-2-style, 1M-0.5B, synthetic. K1: nonce first mentions in
  skill data are random to the weights and may act as junk for the fact half, so fact bits must be measured in every
  arm against a pure-fact calibration; Result 11 gives K1 its missing control (section 6.1).
  Verified OK (PDF, Results 10-12 and footnote 8). Clarified: the 300-1000 figures compare against no-junk at 100.
- **Facts are not only in MLPs.** Shrinking GPT-2's MLP to 1/4 size (d-4d-d to d-d-d) has "negligible impact" on
  capacity. Removing all MLP layers "does not affect its capacity ratio" at 1000 exposures ("the Attention layers are
  also capable of storing knowledge") but "decreases the capacity ratio by more than 1.5x" at 100 exposures.
  Source: 2404.05405 Results 5-7. Scale: 1M-0.5B. K4: the fact cost of shifting MLP to attention is already bounded
  (none at 1000 exposures, more than 1.5x at 100); only the skill side is new. K2: heads must be in the unit set.
  Verified OK (PDF). FIXED in 6.4: the old "up to ~1.5x" turned a lower bound into an upper bound.
- **Knowledge that is stored is not automatically extractable.** Bio pretraining then QA finetuning gives near-zero
  QA accuracy on held-out people (Result 2: "zero-zero QA accuracy on Ptest" despite 99+% first-token accuracy; Figure
  3 gives 9.7% mean for bioS single, baseline 2.7%). "Mixed training" (bios for all, QAs for half, QA:BIO 8:2) gives
  86.6% out-of-distribution (bioS; 77.7% bioR). multi5 + permute pretraining, then QA finetuning, gives 96.6%.
  Augmenting only "celebrities" lifts the minority 4.4% to 86.8% (Result 6). Source: https://arxiv.org/abs/2309.14316.
  Scale: GPT-2 124M/302M/682M, N = 100K. K1/K5: score weight facts in the trained format (bits) or train QA on a subset
  of entities and test the rest; else a "recall" of 0 can mean "stored, not extractable".
  Verified OK (PDF). Added the 9.7% table value and that 96.6% is the pretrain-then-finetune setting, not mixed.
- **Weights cannot do inverse lookup or simple manipulation without CoT.** Retrieval works; classification and
  comparison fail unless CoT is used in both training and inference; inverse search is "virtually 0%". Source:
  https://arxiv.org/abs/2309.14402. Scale: GPT-2 12-layer (768/1280-dim), Llama/Mistral in appendix, synthetic.
  K1/K5: weight-fact probes forward (entity to attribute) only. Verified OK (PDF abstract and section 2).
- **Width vs depth (weak evidence).** TinyStories: "knowledge of facts seems to rely more on the embedding dimension,
  whereas for context-tracking the number of layers is more important"; the 1-layer model (21M) gets no consistency
  prompt right but some facts; the d = 64 model gets no fact right but keeps consistency several times. Source:
  https://arxiv.org/abs/2305.07759 section 4.2. Scale: 1M-33M in Figures 9-11; "facts" are common-sense prompts, a
  handful per figure, so anecdotal. K4: a hypothesis for the depth-for-MLP arms. Verified OK (PDF section 4.2).
- **Width helps in-weights learning; in ICL-only training it does not consistently help ICL.** In joint training,
  wider models have higher ICL peaks and gentler decay, which the authors read as less competition with IWL.
  Source: 2311.08360 section 6. Scale: 12 layers, d 32-128, Omniglot few-shot [toy]. K4: agrees with TinyStories.
  Verified FIXED: the joint-training width benefit to ICL was missing.

## 2. Does memorized knowledge compete with in-context ability? (K1)

- **ICL fades when in-weights learning can solve the same items; MLP weight decay mitigates it.** "ICL first emerges,
  then disappears and gives way to IWL, all while training loss decreases"; L2 regularization "may offer a path" to
  persistent ICL; weight decay on MLP or embedder layers (not on attention) mitigated transience; deeper models do
  not fix it. Source: https://arxiv.org/abs/2311.08360. Scale: 12-layer d 64, most runs 5e7 steps, 2 seeds [toy].
  K1: competition needs items solvable both ways; nonce skill items are ICL-only unless they reuse a fact-base
  entity. K3: selective weight decay on fact units is a cheaper alternative to pruning (6.3). Verified FIXED: "up to
  1e7 steps" was wrong; most runs used 5e7 iterations (1e7 is the plot-axis unit).
- **Coexistence depends on the data distribution.** ICL traded off against IWL until a skewed Zipfian class
  distribution let both coexist. Source: https://arxiv.org/abs/2205.05055 (abstract; Omniglot [toy]; P-025's
  source). K1: a Zipfian fact base is closer to chat than a uniform one. Verified OK (abstract).
- **The memorization-to-ICL transition may be kinetics, not capacity.** Memorizing and generalizing sub-circuits are
  "largely independent"; their relative learning rates, "rather than capacity constraints", explain the transition;
  a memorization scaling law sets the task-diversity threshold; near it solutions are bimodal. Source:
  https://arxiv.org/abs/2412.00104. Scale: one attention layer + 3-layer MLP [toy]. Related (abstracts only): ICL
  task-diversity threshold, linear regression (Raventos, https://arxiv.org/abs/2306.15063); ICL "bottlenecked by the
  accessible state size", not parameter count (Kirsch, https://arxiv.org/abs/2212.04458); superlinear transition
  timescale (Wurgaft, https://arxiv.org/abs/2506.17859). K1: a skill drop at high load is ambiguous between "no room"
  and "learned later"; needs checkpoints and one longer run (6.1). Verified OK (PDF; 3 abstracts).
- **Weights learn global statistics fast, in-context circuits slowly.** Global bigrams are learned rapidly, induction
  heads more slowly; weight matrices act as associative memories. Source: https://arxiv.org/abs/2306.00802
  (abstract; 2-layer, synthetic [toy]). K1: expect the fact half to lead early. Verified OK (abstract).
- **Gating can make a model memorize first and delay ICL.** In SSMs, gating makes the model "first learn an in-weights
  memorization solution, while delaying, or even preventing" ICL, even with no capacity limit. Source:
  https://arxiv.org/abs/2609.16540 (abstract only; scale not stated). K1/K4: Planck's attention gate (S001) makes
  this a weak lead, not a claim about attention gates. Verified OK (abstract).
- **The one direct "facts crowd out generalization" result is too weak to use.** Barron and White report that no size
  extrapolates when fact recall and arithmetic are trained jointly. But in their Table 2 only the 1.46K-param n14
  extrapolates (40/40) even in single-task training; n28 (5.26K), n56 (19.94K) and 10.63M get 0/40 without facts
  too, so the joint result says nothing about crowding at 10M. 50 capital facts, 40 test cases (4 pairs x 10
  samples), lr 1e-2 small vs 1e-5 for 10.63M, one seeded run per model. Source: https://arxiv.org/abs/2506.09099
  (ICML 2025 workshop). Scale: 1.46K-10.63M, character level. K1: "no usable prior". Verified OK (PDF; "no seeds
  reported" sharpened to "globally seeded, no seed variance shown").
- **At scale, fact recall is more fragile than in-context use.** Pruning more than 30% of weights (SparseGPT; Wanda in
  the appendix) or dense down-scaling (OPT family) "significantly decreases the ability to recall facts", while a
  60-70% reduction "largely preserves" in-context processing; overriding-context QA holds to 70% sparsity, and pruned
  OPT-30B gained 9.7% on DisentQA. Source: https://arxiv.org/abs/2310.04680 (ICLR 2024). Scale: OPT-13B/30B,
  LLaMA-13B/33B, Pythia-12B [>=1B only]. K1/K3: "skill is cheap, facts are expensive", silent on whether facts hurt
  skill; K3 needs an untargeted-pruning control (6.3). Verified OK (PDF; venue from arXiv journal reference).
- **Offloading facts to a lookup did not measurably raise general ability at 176M-382M.** LMLM (below) reports NLU
  "on par" with the standard model, not better (https://arxiv.org/abs/2505.15962 section 4.2). K1: the only real-LM
  test of "remove facts, free capacity" found no NLU gain, on NLU benchmarks, not multi-turn skill. Evidence against
  a large effect, not a test of K1's question. Verified OK (PDF).
- **P-114 / K1 status: no prior found** that sweeps a memorized fact load at equal tokens and measures a held-out
  in-context skill at 1M-150M. bits.verify.md reached the same verdict.

## 3. Localizing and removing memorized content (K2, K3)

- **Memorization lives in a few neurons spread over many layers, and training can route it into chosen neurons.**
  Most layers are redundant for memorization and the contributing ones are "not the final layers". Example-tied
  dropout (ResNet-9) sends each example's memorization to a pre-chosen neuron set; dropping those neurons cuts
  mislabeled-example accuracy from 100% to 0.1% (MNIST) and 99% to 3% (CIFAR-10). The authors call the clean impact
  minor; on CIFAR-10 clean train accuracy falls 99.9% to 90.8% while test rises 79.3% to 82.7%. Source:
  https://arxiv.org/abs/2307.09542 (ICML 2023). Scale: ResNet-9/50, ViT-small, image classification. K2/K3: unit
  level, not whole layers; routing makes pruning clean. Verified OK (PDF; clean-accuracy cost added).
- **Post-hoc removal of natural-text memorization is entangled with general ability; training-time isolation works.**
  Pruning and integrated-gradient localization give "limited success", worse for natural repeated text than for
  random-token canaries; repeated-sequence and validation loss fall together; the authors argue for an implicit bias
  toward entangled solutions. MemSinks (a sequence ID switches on a dedicated neuron set) isolates memorization so it
  can be dropped. Source: https://arxiv.org/abs/2507.09937 (ICML 2025). Scale: GPT-2-Medium-like, 100 TinyStories
  repeated 128x plus 20K unrepeated (~16M tokens); SmolLM 360M/1.7B on SlimPajama. K2/K3: post-hoc fact units may
  carry shared format, so pruning them may hurt skill for that reason. Verified OK (PDF).
- **Gradient routing localizes a capability by masking gradients.** "Expand, Route, Ablate" (ERA): add neurons, route
  forget-token gradients to them, delete them. v2 (current): with partial forget labels ERA beats all baselines,
  data filtering included; at 100% labels data filtering is most robust to retraining; retain loss rises with the
  routed share. v1 HTML (Oct 2024): forget loss 1.91 vs 1.47 base, retain 1.67 vs 1.59; at 60% labels retrained
  forget loss 1.53 vs 1.49. Source: https://arxiv.org/abs/2410.04332. Scale: 28M TinyStories-style, 0.7B. K3: the
  closest prior to "route facts into known units, cut them, keep skill" (6.3). Verified FIXED: the numbers and
  "almost as hard to retrain as a never-trained model" are v1 only; "works with 60% labeled" overstated 0.04 nats.
- **Memorizing neurons can be pruned to improve generalization, and weight decay prevents them.** Memorizing neurons
  for corrupted labels "can be identified and pruned", which raises accuracy on uncorrupted data; weight decay,
  dropout and BatchNorm make the network ignore corrupted data. Source: https://arxiv.org/abs/2310.13061 (ICLR
  2024). Scale: 2-layer MLP (quadratic activation), with 1-layer transformer and 3-layer ReLU MLP checks, modular
  arithmetic [toy]. Verified OK (PDF; scale detail added).
- **Memory and context have separate attention heads at scale.** Later-layer "memory heads" recall internal knowledge
  and "context heads" retrieve from context; pruning heads chosen by path patching raised the memory-usage rate by
  44.0 points (49.7% to 93.7%) or the context-usage rate by 38.5 points (50.3% to 88.8%), averaged over eight LMs on
  World Capital. Source: https://arxiv.org/abs/2402.18154 (Findings of ACL 2024). Scale: GPT-2 XL and GPT-J
  (analysis), plus OPT-1.3B/2.7B, Pythia-6.9B/12B, LLaMA2-7B/13B [>=1B only]. K2/K5: a head-level arbitration target.
  Verified OK (PDF; venue from ACL Anthology 2024.findings-acl.70; percentages are points averaged over 8 models).
- **Curvature separates memorization from reasoning in weight space.** A K-FAC-based edit removing high-curvature
  directions cut, on OLMo-2 7B, TriviaQA 0.780 to 0.648, PopQA 0.807 to 0.598 and GSM8K 0.675 to 0.447, while
  open-book TriviaQA went 0.760 to 0.720 and BBH 0.499 to 0.475. Source: https://arxiv.org/abs/2510.24256 (Table 8).
  Scale: OLMo-2 1B/7B, ViT-Base 86M [>=1B for LMs]. K2: directions, not units, may localize better; an extra arm
  (section 6.3). Verified OK (PDF; model named).
- **Localization is fragile.** Causal tracing does not tell which MLP layer is best to edit (Hase et al.,
  https://arxiv.org/abs/2301.04213, GPT-J, GPT2-XL replication [>=1B only]). Liao and Liang
  (https://arxiv.org/abs/2608.24460, 26M, synthetic conflict resolution): all 75 runs reach >= 0.999 accuracy;
  probed before the model escapes a positional shortcut, data attribution of the cue it uses "reverses sign in 32
  of 75 runs"; 13 of 25 cells differ by more than 0.3 in sign fraction across three seeds. K2: >= 3 seeds, judge a
  method by what pruning does. Verified OK (both PDFs; pre-escape timing and the seed figure added).
- **Resetting weights keeps in-context strategies alive.** Active forgetting (re-initialize the embedding matrix every
  k = 1000 steps, chosen from 100/1000/5000) keeps "structural" ICL on unseen tokens that vanilla training loses;
  temporary forgetting (only for the first N steps) gives both. L2 regularization, which fixes Singh's conditional
  ICL, does not keep structural ICL. Source: https://arxiv.org/abs/2406.00053 (ICLR 2025). Scale: 6-layer d64 BERT,
  4-layer d64 GPT-2; transience also seen in MultiBERTs and Pythia-1.4B; GPT-2 large finetune. Origin:
  https://arxiv.org/abs/2307.01163 (RoBERTa-base, NeurIPS 2023). K3: periodic reset of the fact units is Max's
  "overwrite fact parameters" done repeatedly (section 6.3). Verified OK (PDF; the L2 result added).

## 4. Lookup vs memorization (K5)

- **Masking looked-up values from the loss stops memorization.** LMLM pretrains with database lookups and masks the
  returned values from the loss. On the TOFU synthetic set, loss on those tokens stays high where standard SFT drives
  it down. With the database disabled, LLaMA2-382M falls from FactScore 31.9 to 12.8 and T-REx EM 58.1 to 38.5
  (standard 14.0 / 52.0). The 382M model "matches the factual precision of a LLaMA2-7B" (table: 31.9 vs 34.0
  FactScore, 58.1 vs 60.5 T-REx). More offloading gave lower perplexity and higher factual precision. Source:
  https://arxiv.org/abs/2505.15962. Scale: GPT-2 124M/355M, LLaMA2-style 176M/382M, ~3B Wikipedia tokens x 8 epochs.
  K5: value-in-loss vs masked should be a factor. Verified OK (PDF; TOFU setting and the 7B numbers added).
- **Context stays usable when weight facts do not.** After twenty later writes, bare-statement facts keep 1% accuracy
  (broad study data 46%), but writes "barely degrade the model's use of facts in context", and a forgotten study
  fact supplied in the prompt recovers to 77-80%. Source: https://arxiv.org/abs/2607.11020. Scale: Qwen3-4B with
  LoRA on a frozen base, Qwen3-8B check. Verified OK (abstract and PDF; sizes now known).
- **Memory-vs-context measurements swing with phrasing and filler.** Reproduction study, 31 models (Pythia 160M-12B,
  GPT-2 small-XL, Qwen3 to 32B, Ministral-3). Native ParaConflict: GPT-2 small follows the conflicting context at
  0.90, Pythia-160M 0.71. Phrasing swings P(memory) by up to 95.0 points (Qwen3-14B; GPT-2 6.6-16.4). 128 tokens of
  unrelated prose raised Pythia-2.8B's context-following from 29% to 93%, but for Pythia-160M/410M filler dilutes
  the context, and in Qwen3/Ministral it cut context-following by up to 28.2/38.8 points. Absent-knowledge check:
  small models have near-zero generative recall on some relations but prefer the true answer over a random
  distractor by log-probability (>= 50%, all models). Source: https://arxiv.org/abs/2609.24238 (BlackboxNLP 2026).
  K5: several phrasings, a two-sided filler control, a known-fact filter. Verified FIXED: the small-model filler
  reversal was missing, and "low memory use can be absent knowledge" was our gloss, not a finding.
- **Contextual use of pretrained facts can need a trigger.** Pretraining alone "is insufficient for contextual
  recall"; finetuning on implicit-inference tasks for a subset of subjects makes it emerge for all. Source:
  https://arxiv.org/abs/2603.20969 (abstract only; synthetic, attention-only construction). K5: train lookup use on
  a subset of entities, test the rest. Verified FIXED (abstract): the old quote "fails at contextual recall" was
  not the abstract's wording.
- Already in the ledger (not repeated here): Longpre 2109.05052 (P-085, P-142), Yu/Merullo/Pavlick 2310.15910
  (P-084, P-142), KLLM 2607.12831 (P-083), RETRO at 148M (P-171), memory layers and PKM (P-124, P-182).

## 5. Restricted worlds (K generator)

TinyStories: 1M-33M models write fluent, consistent English in a small-vocabulary world; the 1M 8-layer model "fails to
answer any factual prompt correctly" (https://arxiv.org/abs/2305.07759). P-054 and TinyChat/TinyDialogues are in the
LEDGER. Nothing found plants nonce facts and corrections in a restricted world at 1-10M; K's generator is new.

## 6. Implications for K1-K5

### 6.1 K1 (capacity competition)
- Known, do not re-discover: ~1 bit/param at 100 exposures and ~2 at 1000; junk-like data slows fact learning; small
  models memorize nothing below a critical mixing ratio (Gu); in-context use outlasts fact recall under down-scaling
  at 13B+ (Jin). Untested at 5M (no prior found): whether a filled fact store lowers held-out in-context skill.
- Budget arithmetic (mine, recomputed in the fact-check): the 5M shape has 3,437,269 body and 5,010,133 total
  params (E2 notes line 24); Physics counts model size "after excluding all unused tokens in the embedding layer",
  so P is between them. At ~60 bits and ~20 tokens per person: 100 exposures at 1 bit/param holds 57K-84K people,
  115M-167M fact tokens; 1000 exposures at 2 bits/param holds 115K-167K people, 2.3B-3.3B tokens. In E2's
  250M-token screen a pure fact run at 1000 exposures stores 0.75 Mbit, 7.5-10.9% of capacity. So K1 can test load
  near capacity only at ~100 exposures, where a 125M-token fact half is 75-109% of capacity.
- Design changes this implies:
  1. Fix exposures per fact (100) and vary N; state load in bits against a pure-fact calibration of the same shape.
  2. Token-matched information control (Physics Result 11): the high arm's fact-token count over N' ~ 1K repeated
     entities. High vs control isolates information load; control vs zero-fact isolates lost skill tokens.
  3. Measure fact bits in every arm; a low arm storing ~0 sits under Gu's threshold, not at "low load".
  4. Keep skill-item entity names disjoint from the fact base (else Singh's competition applies), plus a small
     crossover probe where a context value contradicts a memorized one (K5 preview).
  5. Score skill on rolling checkpoints and run the high arm once at 2x tokens: a deficit that closes points to
     kinetics (Nguyen and Reddy), one that stays points to capacity.
  6. Optional arm: a domain token on fact lines (Result 12), which the same paper says protects fact capacity.

### 6.2 K2 (localization)
- Include attention heads and weight-space directions, not only MLP rows (Physics 3.3 no-MLP; Merullo K-FAC).
- Expect partial separation: natural-format memorization is entangled post hoc (MemSinks); at 26M, attribution can flip
  sign before a shortcut is escaped (Liao and Liang). Run >= 3 seeds; report overlap of unit sets across seeds.
- Validate each localization method by the ablation result, not by tracing magnitude (Hase).

### 6.3 K3 (prune and regrow): extra arms worth adding
- Untargeted global magnitude pruning at the same sparsity (Jin: at scale it already removes facts first).
- Route-then-prune: during K1-style training, send fact-line gradients to a fixed unit subset (gradient routing /
  MemSinks), then drop it and continue on skill data. Prior: 28M TinyStories-style and 360M-1.7B; ERA's retain cost
  grows with the routed share, so log skill loss before and after the drop. Untested for "free capacity then regrow".
- Selective weight decay on fact units instead of cutting them (Singh: MLP weight decay mitigated ICL transience;
  Doshi: weight decay suppresses memorizing neurons). Caution: Anand found L2 does not keep structural ICL, so for
  nonce-token skill pair it with a periodic reset of fact units every k steps (active forgetting, Anand/Chen).
- Measure counterfactual context-following after pruning (Jin: pruned OPT-30B +9.7% on DisentQA).

### 6.4 K4 (allocation)
- Fact side mostly known (Physics 3.3 Results 5-7: no MLP costs nothing at 1000 exposures and more than 1.5x at 100;
  1/4-size MLP negligible). Skill side at 5M untested. The 5M MLP is gated (SwiGLU); Physics finds LLaMA 1.3x behind
  GPT-2 at 100 exposures and traces it mainly to the gated MLP.
- TinyStories (width for facts, depth for context tracking) and Singh (width helps IWL) predict that depth-for-MLP
  trades facts for skill; K4 is the test. Anecdotal priors only.

### 6.5 K5 (lookup vs memorization)
- Add a factor: lookup value in the loss vs masked (LMLM). Masked is expected to stop memorization.
- The p = 0 arm needs extractable facts: QA on a subset of entities, test on the rest (Physics 3.1).
- Counterfactual probe: >= 3 phrasings, a filler-length control, and a known-fact filter (Fouilhe et al.). At
  160M-410M filler lowered context-following, so at 5M expect filler to dilute, not boost; report both directions.
- Weight-fact probes forward only (Physics 3.2).

### 6.6 Pitfalls in one list
Stored is not extractable (3.1). Nonce-heavy skill data may act as junk for facts (3.3 R10). Token-share confound
(6.1.2). Kinetics mistaken for capacity (6.1.5). Post-hoc localization entangled with format (MemSinks). Seed-unstable
attribution. Low fact arms under Gu's threshold. Only 100-exposure loads fit a 250M budget at 5M. TinyStories and
Barron-White are anecdotal. Gradient-routing numbers differ between arXiv versions. Filler effects flip with size.
