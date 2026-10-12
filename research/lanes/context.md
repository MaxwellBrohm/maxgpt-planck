# Lane report: the mechanics of multi-turn coherence and memory in small transformers

Research sweep for MaxGPT-Nano (target: about 150M parameters, genuine back-and-forth chat). Date: 2026-09-23. Every number carries its source and the scale it was measured at. "Vendor" means the number comes from the model's own card or blog and was not independently reproduced. "Computed" means I derived it from the model's `config.json`. Evidence that exists only at 1B or above is flagged **[>=1B only]**.

---

## Bottom line

1. Max's premise is half right. Sub-400M instruction models that hold a *single-turn* chat already exist (SmolLM2-360M-Instruct, Gemma 3 270M, LFM2-350M, Falcon-H1-Tiny-90M-Instruct; vendor IFEval 41 to 66, and a vendor MT-Bench of about 4.4 for the 90M), but I found **no published multi-turn coherence measurement (MT-Bench-101, MultiChallenge, Lost-in-Conversation, TurnWise) for any sub-1B model**, and Google says its 270M is "not designed for complex conversational use cases". The multi-turn wall is real but mostly unmeasured, and that gap is itself the opening.
2. The raw in-context machinery is **not** the bottleneck at 150M: induction, name-mover and successor heads form after about 2 to 8B training tokens at every Pythia size from 160M to 2.8B, and a 340M transformer does 98% single-needle recall inside its trained context.
3. What *does* scale with size sits right in the 150M to 400M gap: tracking an entity's state through updates reaches human level only at Pythia-410M (70M scores 53.5%), and composing two facts from context (2-hop) fails even at 1.3B on 100B real tokens. These are depth- and data-sensitive **skills**, not knowledge.
4. The second part of the wall is **knowledge**. At about 2 bits per parameter, a 150M model can hold at most about 300M bits. Knowledge can live outside the model (RETRO at 172M gains the equivalent of about 10x parameters; memory layers at 134M triple NaturalQuestions accuracy), **but only when the model is trained to read it**: off-the-shelf SmolLM2-360M scored 0% exact match even when handed the gold passage, and Max's own Daimax runs saw 0.5B and 0.8B models parrot retrieved chat excerpts as fact.
5. The third part is **stability**. Small models loop more (Daimax: SmolLM2-360M looped in 4 of 33 runs, Qwen2.5-0.5B in 0; Falcon's 90M reasoning model falls into a "repetition trap"), and in a multi-turn chat the context fills with the model's own earlier text, which is exactly what induction-driven self-reinforcement feeds on.
6. Even frontier models lose about 39% going from single-turn to multi-turn (Laban et al., smallest model tested 8B). Multi-turn post-training data closes part of that (+12.8 points at 7B), and the open SFT sets small models are trained on are mostly single-turn or three-turn.
7. For a 150M chat model the best-supported bet is: spend parameters on **depth and in-context skills** (deep-thin shape, full attention, code and state-tracking data, multi-turn chat mixed into pretraining), push **facts** to retrieval and tools that the model is trained to use from day one, and have the **harness keep a compact, structured conversation state** so the model does 1-hop lookups instead of multi-hop recall across turns.

---

## 1. Detailed findings

### 1.1 What multi-turn chat actually requires

The two multi-turn taxonomies worth using are MT-Bench-101 (13 tasks in three tiers, Perceptivity / Adaptability / Interactivity, including context memory, anaphora resolution, topic shift, content confusion, self-correction; 1,388 dialogues, 4,208 turns; [arXiv 2402.14762](https://arxiv.org/abs/2402.14762), [ACL 2024 pdf](https://aclanthology.org/2024.acl-long.401.pdf)) and MultiChallenge (instruction retention, inference memory of user information, reliable versioned editing, self-coherence; every frontier model under 50%, top Claude 3.5 Sonnet at 41.4%; [arXiv 2501.17399](https://arxiv.org/abs/2501.17399)). MT-Bench-101 reports that performance declines with turn number specifically on context memory, anaphora resolution and resistance to interference ([ACL 2024](https://aclanthology.org/2024.acl-long.401.pdf)). Neither benchmark reports a sub-1B model that I could find.

Mapping those to mechanisms:

| Chat capability | Underlying mechanism | Where it comes from | Needs parameters? | Can be externalized? |
|---|---|---|---|---|
| Quote or reuse a name, number, phrase from earlier | Induction / copy heads | Forms at about 2-5B tokens, needs 2+ layers (1.2.1) | Very few | No (it *is* the reading mechanism) |
| Retrieve a fact stated N turns ago | Associative recall over the KV cache | Attention solves it at 70M+ (1.2.2) | Few, but needs full attention | Partly: harness can re-surface it |
| Resolve "it", "she", "the second one" | Coreference / binding (name movers, binding IDs) | Name movers at 2-8B tokens at 160M (1.2.1); binding IDs in "sufficiently large" models (1.2.3) | Moderate | Partly (state card) |
| Track state changes ("I moved it to the red box", "actually make it 3") | Entity/state tracking | Human-level at Pythia-410M (1.2.3) | Yes, scales with size and depth | Partly: harness can hold the current state |
| Combine two things said in different turns | In-context multi-hop composition | Fails at 1.3B/100B real tokens (1.2.4) | Yes, depth-heavy | Largely: pre-resolve into 1-hop |
| Keep obeying a system prompt / early instruction | Attention to far-back goal tokens | Decays over turns at all scales (1.2.5) | Partly | Yes: re-inject near the end |
| Turn-taking, stopping, not writing the user's turn | Role tokens + end-of-turn training | Post-training distribution (1.3.3) | Few | No, it is a trained habit |
| Track what the model itself said, stay self-consistent | Reading own prior turns without copying them | Conflicts with self-reinforcement (1.3.1) | Unknown | Partly |
| World knowledge in answers | Stored facts | 2 bits/param (1.4.1) | **Yes, linearly** | **Yes: RAG, tools, memory layers** |

### 1.2 How each capability emerges vs size, depth and tokens

#### 1.2.1 Induction, copy and other in-context circuits form early and at every size above tiny

- Olsson et al.: induction heads form abruptly in a window of about **2.5 to 5B training tokens**, coinciding with a jump in in-context learning; the only model without the phase change is the **one-layer** model. Measured on small attention-only models (1 to 6 layers) plus larger models ([arXiv 2209.11895](https://arxiv.org/abs/2209.11895)).
- Tigges et al. tracked Pythia 70M to 2.8B across 300B tokens: induction heads emerge "soon after they have seen 2x10^9 tokens"; successor heads and IOI name-mover heads at **2 to 8x10^9 tokens across models**; task ability emerges at similar token counts regardless of size. **Pythia-160M learns all the tasks (IOI, greater-than, gendered pronoun, subject-verb agreement); Pythia-70M does not** ("We omit Pythia-70m, as it does not learn the task") ([arXiv 2407.10827](https://arxiv.org/abs/2407.10827), [html](https://arxiv.org/html/2407.10827)).
- Induction-head timing is set by a simple function of **batch size and context size, independent of model size**, and is driven by surface bigram repetition frequency and reliability in the data (ICML 2026, [arXiv 2511.16893](https://arxiv.org/abs/2511.16893)).
- In 1B-class Pythia/OLMo/OLMoE, induction circuits form **10 to 20x earlier in tokens** than the BOS attention-sink circuit, and converge within 0.3 to 2% of total training tokens **[>=1B only]** ([arXiv 2606.02378](https://arxiv.org/abs/2606.02378)).
- In-context learning needs the right data statistics: burstiness, many rare classes, Zipfian distribution; naturalistic distributions elicited ICL in transformers but **not in recurrent models** (synthetic Omniglot-style setting; [arXiv 2205.05055](https://arxiv.org/abs/2205.05055)).
- Caution for heavy overtraining: in a synthetic setting ICL "first emerges, then disappears and gives way to in-weights learning", and L2 regularization helps it persist ([arXiv 2311.08360](https://arxiv.org/abs/2311.08360)). Counter-evidence at LM scale: Pythia-160M keeps its circuits through 300B tokens (about 1,900 tokens per parameter), with individual heads swapping roles but task performance stable ([arXiv 2407.10827](https://arxiv.org/abs/2407.10827)).

**Implication:** a 150M transformer trained on tens of billions of tokens will have the basic copy and binding circuitry. This is not where the wall is.

#### 1.2.2 In-context retrieval: attention is the right memory, and a small transformer already does it well

- Zoology: **82% of the perplexity gap** between attention and gated-convolution models is explained by associative recall; a **70M attention model beats a 1.4B gated-convolution model** on associative recall ([arXiv 2312.04927](https://arxiv.org/abs/2312.04927)).
- "Repeat After Me": a ~160M transformer learns to copy with about **100x fewer samples** than a ~160M Mamba; theory says 2 layers can copy exponentially long strings while any fixed-state model cannot; on phone-book lookup, **Pythia-410M beats Mamba-2.8B** once the book has 70+ entries ([arXiv 2402.01032](https://arxiv.org/abs/2402.01032), [html](https://arxiv.org/html/2402.01032)).
- Based, at 360M params / 10B tokens, recall-intensive extraction: Transformer++ SWDE 58.0, FDA 27.2, SQuAD 44.1; Mamba 23.7 / 24.1 / 43.5. At 1.3B / 10B tokens Transformer++ reaches 71.9 / 36.2 / 47.6 ([arXiv 2402.18668](https://arxiv.org/html/2402.18668)). Recall improves with size but 360M is already usable, and the attention-vs-recurrent gap is bigger than the 360M-vs-1.3B gap.
- Titans' Transformer++ baseline at **340M / 15B tokens** scores 98.4, 98.8, 98.0 and 88.4% on single needle-in-a-haystack at 2K, 4K, 8K, 16K ([arXiv 2501.00663](https://arxiv.org/html/2501.00663)). Single-fact recall inside the trained length is essentially solved at this size.
- Every production sub-400M chat model keeps real attention: LFM2-350M is 10 short-conv + 6 GQA attention layers ([card](https://huggingface.co/LiquidAI/LFM2-350M)); Falcon-H1-Tiny-90M runs attention (8 heads, 2 KV) in parallel with Mamba in all 24 layers ([config](https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Instruct/resolve/main/config.json)).

#### 1.2.3 Entity and state tracking: this is where the 150M to 400M gap shows up

- **Newest and most on-point:** Drozdz and Heilbron (June 2026) tested entity tracking in naturalistic narratives across Pythia 70M to 12B and OLMo 2. Pythia accuracy on the implicit (likelihood-comparison) task rises **from 53.5% at 70M to 89.6% at 12B**; **human-level tracking (humans: 75.3% explicit, 79.3% implicit) is reached at 410M**. OLMo 2 shows smaller complexity costs than Pythia at equal size, i.e. better data moves the curve ([arXiv 2608.18083](https://arxiv.org/html/2608.18083)). The exact 160M number is not reported in the text I could access (unsourced).
- Kim and Schuster (2023): on a synthetic boxes task, only GPT-3.5 models (trained on lots of code) showed non-trivial entity tracking among the models they tested **[>=1B only]** ([ACL 2023](https://aclanthology.org/2023.acl-long.213/)). Follow-up: continued training on code consistently improves entity tracking across Llama 2 / Code Llama (7B, 13B, 70B), DeepSeek (7B), Gemma (8B); extra math pretraining gives limited benefit **[>=1B only]** ([arXiv 2405.21068](https://arxiv.org/html/2405.21068)).
- Feng and Steinhardt: a "binding ID" mechanism for attaching attributes to entities appears in "every sufficiently large model from the Pythia and LLaMA families" ([arXiv 2310.17191](https://arxiv.org/abs/2310.17191)); the size threshold is not stated in the abstract (unsourced).

**Implication:** multi-turn chat is full of state updates ("change that to Tuesday", "no, the other one"). With Pile-era data, this skill crosses human level between 160M and 410M. The OLMo 2 result and the code result say data can move that threshold down; nobody has measured how far at 150M.

#### 1.2.4 Composing information from different places in context (multi-hop) is hard even at 1.3B

- Physics of LMs 4.1 (Allen-Zhu, Dec 2025): in real pretraining at **1.3B params / 100B tokens / 4096 context**, "all models fail 2-hop reasoning, even in short (100-token) contexts" (task: three birth years plus three "born in the same year as" links). In controlled synthetic pretraining (GPT-2-small-sized 8 to 12 layer models), Canon layers (a causal conv1d with kernel 4 adding each token's three predecessors, inserted before attention/MLP) raise reasoning depth **2 to 4x** and breadth by 30%, and lift NoPE to match RoPE ([arXiv 2512.17351](https://arxiv.org/html/2512.17351)).
- Physics of LMs 2.1: "Language model depth is crucial for mathematical reasoning"; a 4-layer d=1920 model underperforms a 20-layer d=576 model despite being about 2x larger (synthetic iGSM) ([arXiv 2407.20311](https://arxiv.org/html/2407.20311)).
- Looped transformers: at 1B params / 250B Pile tokens, a 12-layer model looped twice scored **34.3%** on math word problems vs **29.3%** for the 24-layer baseline, with half the parameters and worse perplexity; on closed-book QA (memorization) looping gave no advantage **[>=1B only]** ([arXiv 2502.17416](https://arxiv.org/html/2502.17416)). Reasoning needs depth, not parameters; memorization needs parameters, not depth.

#### 1.2.5 Instruction persistence decays with conversation length at every scale measured

- Li et al.: "significant instruction drift within eight rounds" for LLaMA2-chat-70B and GPT-3.5; mechanism is attention decay to the system prompt; split-softmax mitigates **[>=1B only]** ([arXiv 2402.10962](https://arxiv.org/abs/2402.10962)).
- "When Attention Closes" (May 2026): goal-defining tokens become less accessible through attention as the conversation grows while goal information persists in the residual stream (linear probes predict recall with AUC up to 0.99); causal ablation on Mistral drops 20-fact recall from near-perfect to **11%** **[>=1B only]** ([arXiv 2605.12922](https://arxiv.org/abs/2605.12922)).
- SEQUOR (May 2026): single-constraint instruction-following accuracy falls by more than 11% as the conversation grows, multiple constraints by more than 40% (model sizes not in the abstract, unsourced) ([arXiv 2605.06353](https://arxiv.org/abs/2605.06353)).
- Gated attention (the per-head output gate already in Max's recipe) cuts first-token attention from **46.7% to 4.8%** averaged over layers and lifts RULER at 128K (YaRN-extended) from **31.65 to 58.82**, measured on a 1.7B dense model / 3.5T tokens **[>=1B only]** ([arXiv 2505.06708](https://arxiv.org/html/2505.06708)). Plausibly helpful for keeping far-back instructions reachable at 150M, unmeasured.

#### 1.2.6 Position, length and lost-in-the-middle at small scale

- Lost-in-the-middle is not a big-model artifact: GPT-2 and Llama variants **trained from scratch** on memory tasks develop the U-shape; recency tracks short-term-memory demand in the data, primacy tracks long-term demand plus attention sinks ([arXiv 2510.10276](https://arxiv.org/abs/2510.10276)). A structural analysis finds the U-shape already present at initialization in causal decoders with residuals ([arXiv 2602.16837](https://arxiv.org/html/2602.16837v2)).
- Length generalization: in ~107M decoder-only models, NoPE beat RoPE, ALiBi, T5-bias and absolute embeddings on reasoning tasks at unseen lengths ([arXiv 2305.19466](https://arxiv.org/html/2305.19466)). SmolLM3 drops RoPE in every 4th layer (RNoPE, [arXiv 2501.18795](https://arxiv.org/abs/2501.18795)) **[>=1B only]** ([SmolLM3 blog](https://huggingface.co/blog/smollm3)).
- For chat, length generalization matters less than it seems: a 10-turn conversation at roughly 80 user + 150 assistant tokens per turn is about 2.3K tokens (my arithmetic), inside a 4K training context. What matters is training on multi-turn data at the full serving length.
- Design note (computed from configs): Gemma 3 270M uses a 512-token sliding window in 15 of 18 layers, so only 3 layers can see past roughly the last two or three turns ([config mirror](https://huggingface.co/unsloth/gemma-3-270m-it/resolve/main/config.json)). Google also says it is not built for complex conversation ([Google blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/)). Whether the window causes the limitation is speculative.

#### 1.2.7 Depth vs width at small scale

- TinyStories (models up to about 80M): "models that have only 1 layer seem to struggle quite substantially with following instructions (which likely heavily relies on global attention), and 2 layers seem to be sufficient for a certain extent"; "knowledge of facts seems to rely more on the embedding dimension, whereas for context-tracking the number of layers is more important" ([arXiv 2305.07759](https://arxiv.org/abs/2305.07759), section 4 of the pdf).
- MobileLLM: deeper-thinner beats wider-shallower at 125M and 350M; their 125M is **30 layers at d=576**, their 350M **32 layers at d=960** ([arXiv 2402.14905](https://arxiv.org/html/2402.14905)). SmolLM2-135M uses the same 30 x 576 shape ([config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/config.json)).
- Pleias Baguettotron: 321M params, **80 layers at d=576**, 200B tokens of the synthetic SYNTH set; vendor claims it gets close to Qwen3-0.6B on MMLU/GSM8K/HotPotQA ([card](https://huggingface.co/PleIAs/Baguettotron)).

### 1.3 Failure modes of sub-500M chat models, with evidence

#### 1.3.1 Repetition loops (the most measured small-model failure)

- Self-reinforcement: "the more times a sentence is repeated in the context, the higher the probability of continuing to generate that sentence" (GPT-2, BART); DITTO training on synthetic repetitions reduces it without hurting perplexity ([arXiv 2206.02369](https://arxiv.org/abs/2206.02369)).
- Mechanism: induction heads come to dominate the output logits during repetition and suppress other heads, collapsing entropy; per-head descaling of induction output reduces repetition without harming ICL or perplexity (scale in the abstract: unsourced) ([arXiv 2505.13514](https://arxiv.org/abs/2505.13514)).
- Size: distilled reasoning students loop significantly more than their teachers at low temperature; "larger models tend to loop less"; temperature is a stopgap that does not fix the underlying learning error **[>=1B only]** ([arXiv 2512.12895](https://arxiv.org/abs/2512.12895)).
- At 90M: TII's 90M reasoning model "is more prone to a repetition trap"; putting chain-of-thought traces into the 90M tool-calling training mix caused "infinite generation loops", fixed by filtering all reasoning content out ([Falcon-H1-Tiny blog](https://tiiuae-tiny-h1-blogpost.hf.space/)).
- Max's own measurements: Daimax, 3 runs x 11 cases at T=0.7: **SmolLM2-360M looped 4/33, Qwen3-0.6B 1/33, Qwen2.5-0.5B 0/33** (degeneracy = fraction of repeated 5-grams; a real loop scored 0.914 vs 0.000 for prose) (`~/Documents/Projects/daimax/docs/BENCHMARK.md`, 2026-09-03b). MaxGPT-2 (110M) produced "Bitcoin.com - Bitcoin.com ..." loops, patched with a 1.2 repetition penalty (`~/Documents/Projects/Max's AI Model/WRITEUP_NOTES.md`).
- **Why this is a multi-turn problem specifically:** every turn adds the model's own text to the context, which is precisely the input the self-reinforcement effect and induction dominance act on. This is my inference from the above, not a measured result at 150M.

#### 1.3.2 Getting lost across turns

- Laban et al. (ICLR 2026 outstanding paper): 15 LLMs, 200K+ simulated conversations, **39% average drop** from single-turn fully-specified to multi-turn sharded instructions; the drop decomposes into a minor aptitude loss (about 16%) and a more than doubled unreliability; models commit to early assumptions and do not recover; even at temperature 0 unreliability stays around 30%; recap-style strategies help only partly. Smallest model tested: Llama-3.1-8B **[>=1B only]** ([arXiv 2505.06120](https://arxiv.org/abs/2505.06120), [html](https://arxiv.org/html/2505.06120)).
- "Contextual inertia": models keep their earlier reasoning path even when new information arrives; an RL method anchored on the model's own single-turn answers fixes part of it (sizes not in abstract, unsourced) ([arXiv 2603.04783](https://arxiv.org/abs/2603.04783)).

#### 1.3.3 Role confusion and turn-taking

- I found no published measurement of "continues the user's turn" or "fails to stop" rates for sub-500M models (unsourced). The closest evidence is that turn behavior is a property of the post-training distribution: models trained mostly on single-turn pairs treat the end-of-turn token as "conversation complete", and collaboration-style multi-turn SFT on Qwen3.5-2B raised genuine user follow-ups from 1-2% to 46-48% ([arXiv 2604.02315](https://arxiv.org/html/2604.02315)) **[>=1B only]**.
- Instruction-loss modelling (loss on prompts too) helps when SFT data is small or outputs are short ([arXiv 2405.14394](https://arxiv.org/html/2405.14394v2)); how this interacts with role confusion at 150M is unmeasured.

#### 1.3.4 Hallucinated history and confabulation from context

- Daimax: the 0.5B announced "I'm currently running as Daimax-Pro" and invented a "Daimax-Max" tier, most likely from memory-vault excerpts of earlier chats; the 0.8B answered "Red, Blue and GREEN" from a retrieved wrong answer; a label saying "what was said, not what is true" did not help ("a 0.5B cannot act on that label"). The Instant tier now retrieves only facts and documents, never chat excerpts (BENCHMARK.md, 2026-09-03b and later).
- "Can Small Language Models Use What They Retrieve?" (Mar 2026): with the **gold passage** supplied, SmolLM2-360M-Instruct scored **0.0%** exact match on questions it did not already know, Qwen2.5-1.5B 10.0%, 3B 12.8%, 7B 14.6%; "irrelevant generation" (ignoring the context) was **100% of the 360M's failures**; adding any retrieval destroyed 41.6 to 64.0 points of previously-correct answers at 1.5B to 7B ([arXiv 2603.11513](https://arxiv.org/html/2603.11513)). Caveat: off-the-shelf models, exact-match scoring, not trained for RAG.

#### 1.3.5 Confident factual nonsense

- MaxGPT-2 (110M, 482M tokens) on photosynthesis: "the SHAPE of an explanation but not the content" (WRITEUP_NOTES.md). This is the knowledge-capacity limit (1.4.1), not a coherence failure, but users and LLM judges score it as bad chat.

#### 1.3.6 Tool non-use

- In Daimax, **every sub-500M model called zero tools** (SmolLM2-360M 0.00, Qwen2.5-0.5B 0.00; Qwen3-0.6B 0.47-0.50). This is at least partly a training choice: SmolLM2's 135M/360M SFT set was filtered to remove "complex instruction-following tasks (e.g., function calling)" ([arXiv 2502.02737](https://arxiv.org/html/2502.02737)).

### 1.4 External memory and tools: what has actually been measured

#### 1.4.1 Knowledge capacity is linear in parameters

- Physics of LMs 3.3: about **2 bits of knowledge per parameter** at 1,000 exposures per fact, about **1 bit** at 100 exposures, consistent across GPT-2 sizes **1M to 0.5B**; if 7/8 of training tokens are junk, capacity can fall **20x** (only 2x with domain tokens like `wikipedia.org` prepended) ([arXiv 2404.05405](https://arxiv.org/html/2404.05405)).
- Arithmetic for Nano: 150M x 2 bits = at most about **300M bits (about 37 MB) of facts**, under ideal exposure; the 1.1B Ultra tops out around 2.2B bits. The paper's 7B figure (14B bits) "surpass[es] English Wikipedia and textbooks combined" ([arXiv 2404.05405](https://arxiv.org/abs/2404.05405)).

#### 1.4.2 Retrieval-augmented pretraining works at small scale, when trained in

- RETRO trained at **172M, 425M, 1.45B, 7.5B** with a 2T-token database; gains "do not diminish as we scale" and are "comparable to multiplying the parametric model size by ~10x"; a large part of the gain correlates with train/test overlap, but RETRO still wins at all leakage levels ([arXiv 2112.04426](https://arxiv.org/html/2112.04426)).
- kNN-LM lowers perplexity but does **not** improve open-ended generation (MAUVE and human eval); retrieval entropy rises as the query becomes model-generated text (exposure bias) ([arXiv 2305.14625](https://arxiv.org/abs/2305.14625)). Warning for chat, which is all model-generated context.
- MassiveDS (1.4T-token datastore): Llama-2-7B plus retrieval beats Llama-2-13B alone on TriviaQA and NQ; on reasoning-heavy MMLU/MedQA, Pythia models stay near random even at 12B **[>=1B only]** ([arXiv 2407.12854](https://arxiv.org/html/2407.12854)).

#### 1.4.3 Parametric-sparse memory (knowledge in a lookup table, not in compute)

- Memory Layers at Scale (Meta): a **134M** base plus about 937M memory parameters goes from NQ 0.91 to 3.16 and TriviaQA 7.7 to 18.77, approaching but not matching a 1.3B dense model; PIQA only 62.1 to 65.9, HotpotQA 5.2 to 9.4. Gains are "especially pronounced for factual tasks" ([arXiv 2412.09764](https://arxiv.org/html/2412.09764)).
- DeepSeek Engram (Jan 2026): hashed N-gram embedding lookup; offloading local patterns frees attention and lifts Multi-Query NIAH from 84.2 to 97.0 at 27B **[>=1B only]** ([arXiv 2601.07372](https://arxiv.org/abs/2601.07372)).

#### 1.4.4 Recurrent and test-time memory

- BABILong: RMT and ARMT built on **GPT-2 (137M)** and fine-tuned on the task handle bAbI facts across contexts up to about 10M tokens and "outperform GPT-4 significantly"; popular LLMs use only 10-20% of their context; RAG reaches about 60% on single-fact QA regardless of length ([arXiv 2406.10149](https://arxiv.org/html/2406.10149)). Strong small-model memory result, but on narrow templated tasks.
- Titans: neural test-time memory at 170M to 760M on 15 to 30B tokens beats Transformer++ mostly beyond 8K (16K S-NIAH: 93.4 vs 88.4 at 340M) ([arXiv 2501.00663](https://arxiv.org/html/2501.00663)). Irrelevant inside a 4K chat window.

#### 1.4.5 Summaries, recap and MemGPT-style paging

- MemGPT: Llama-2-70B variants "consistently generate incorrect function calls or even hallucinate functions"; reasonable performance needed GPT-4 **[>=1B only]** ([arXiv 2310.08560](https://arxiv.org/pdf/2310.08560)). Self-managed paging is not a realistic skill for 150M.
- Recap (re-stating accumulated instructions) partly recovers the multi-turn drop in Laban et al. **[>=1B only]** ([arXiv 2505.06120](https://arxiv.org/html/2505.06120)). The "When Attention Closes" mechanism (goal tokens become hard to attend to) is a reason re-injecting them near the end should help. No small-model measurement exists (unsourced).
- Long-term conversational memory benchmarks exist (LoCoMo, about 300 turns / 9K tokens per conversation, [arXiv 2402.17753](https://arxiv.org/abs/2402.17753); LongMemEval, about 105K tokens) but I found no sub-1B results on them (unsourced).

#### 1.4.6 Tool use at small scale

| Evidence | Scale | Result | Source |
|---|---|---|---|
| Toolformer (self-supervised tool learning) | GPT-2 124M, 355M, 775M, 1.6B | Benefit "only emerges at around 775M"; smaller models perform the same with and without tools | [arXiv 2302.04761](https://arxiv.org/abs/2302.04761) |
| TinyGSM (supervised Python solutions, run by an interpreter) | 125M / 350M / 1.3B | GSM8K 63.1 / 65.9 / 68.2%; with same-size verifier 68.9 / 71.3 / 81.5% | [arXiv 2312.09241](https://arxiv.org/html/2312.09241) |
| MobileLLM API calling (fine-tuned) | 350M vs LLaMA-2-7B | Intent EM 65.3 vs 62.8; structure EM 48.8 vs 50.9 | [arXiv 2402.14905](https://arxiv.org/html/2402.14905) |
| Falcon-H1-Tiny tool calling (vendor) | 90M | BFCL v3 overall 41.23% vs FunctionGemma-270M 41.30%; **multi-turn BFCL 0% for both**, Qwen3-0.6B 3.62% | [blog](https://tiiuae-tiny-h1-blogpost.hf.space/) |
| FunctionGemma (vendor) | 270M | Mobile Actions 58% to 85% after task fine-tuning | [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/functiongemma/) |
| Daimax (Max) | 0.36B to 4B | Only the 4 tool-calling models (>=0.6B) top the Pro ranking; sub-500M models called 0 tools | BENCHMARK.md |

Reading the Daimax result honestly: it shows that tool use dominates the *score*, but tool use is 30-40% of that score by construction, there are only 2 tool cases (so tool rate is 0, 0.5 or 1), there is no with-tools vs without-tools ablation within one model, and the tool callers are also the larger models. It supports "tools matter once the model can call them", not "tools rescue a small model". Toolformer's 775M threshold is a *self-supervised* recipe result from 2023; supervised tool traces work at 90M to 350M for single calls, while **multi-turn tool use at 270M and below measured 0%**.

### 1.5 The sub-400M landscape (premise check)

Architecture numbers are from each `config.json`; non-embedding parameters are computed by me from those numbers (tied embeddings).

| Model | Total | Layers x d | Embedding share | Non-emb (computed) | Train tokens | IFEval | MT-Bench | Notes |
|---|---|---|---|---|---|---|---|---|
| Falcon-H1-Tiny-90M-Instruct | 90M | 24 x 512, attn+Mamba parallel | 32,768 vocab, ~17M | ~73M | 800B base | 66.08 (vendor) | 4.33-4.4 (vendor) | SFT mixed into pretraining |
| SmolLM2-135M-Instruct | 135M | 30 x 576, GQA 9/3 | 49,152 vocab, ~28M | ~106M | 2T | 29.9 (card) | "19.8" (card; scale inconsistent with 360M card) | |
| Gemma 3 270M IT | 268M | 18 x 640, 15 of 18 layers 512-window | 262K vocab, ~168M | ~100M | 6T | 51.2 (Google) | n/a | "not designed for complex conversational use" |
| LFM2-350M | 350M | 16 x 1024 (10 conv + 6 attn) | 65,536 vocab | n/a | 10T | 65.12 (Liquid) | n/a | Qwen3-0.6B 64.24 in same table |
| MobileLLM-350M | 350M | 32 x 960 | | | | | 3.28 (paper) | AlpacaEval 47.08% |
| SmolLM2-360M-Instruct | 362M | 32 x 960, GQA 15/5 | ~47M | ~315M | 4T | 41.0 (card) | 3.66 (card) | looped 4/33 in Daimax |
| Qwen2.5-0.5B-Instruct | 494M | 24 x 896, GQA 14/2 | 151,936 vocab, ~136M | ~358M | n/a | 31.6 (SmolLM2 card) | 4.16 (SmolLM2 card) | looped 0/33 in Daimax |
| Qwen3-0.6B | 596M | 28 x 1024 | ~156M | ~440M | n/a | 64.24 (Liquid table) | n/a | tool use 0.47 in Daimax |

Sources: [Falcon-H1-Tiny config](https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Instruct/resolve/main/config.json) and [blog](https://tiiuae-tiny-h1-blogpost.hf.space/); [SmolLM2-135M card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct); [SmolLM2-360M card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct); [Gemma 3 270M blog](https://developers.googleblog.com/en/introducing-gemma-3-270m/) and [card mirror](https://huggingface.co/unsloth/gemma-3-270m-it); [LFM2-350M card](https://huggingface.co/LiquidAI/LFM2-350M); [MobileLLM](https://arxiv.org/html/2402.14905); [Qwen2.5-0.5B config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json); [Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/config.json); SmolLM2 token counts from [arXiv 2502.02737](https://arxiv.org/html/2502.02737).

Three things stand out:

- **Cross-vendor numbers at this scale are fragile.** Google reports Gemma 3 270M IFEval 51.2; TII's table reports 27.44 for "Gemma3-270M" and 30.69 for SmolLM2-135M (vs 29.9 on its own card). Different harnesses, prompt-level vs instruction-level scoring, and possibly different variants. Treat all single-turn leaderboard claims at this size as within about 20 points of noise until re-run in one harness.
- **In non-embedding terms the "wall" models are bigger than they look, and the models under it are smaller.** Qwen2.5-0.5B has about 358M non-embedding parameters; Gemma 3 270M has about 100M and Falcon-H1-Tiny-90M about 73M. A 150M model with Max's 49K vocab at d=576 would have about 122M non-embedding parameters, i.e. more "skill" parameters than Gemma 3 270M.
- **The token budgets are enormous.** 2T to 10T tokens for 135M to 350M models (about 11,000 to 29,000 tokens per parameter, my arithmetic); the 90M used 800B base tokens. Max's pre-tokenized build is 100B tokens.

---

## 2. Skills that need parameters vs knowledge that can live outside the model

| Must be in the weights (skills) | Evidence | Can live outside (knowledge and state) | Evidence |
|---|---|---|---|
| Copy/induction, binding, name-mover circuits | Form at 2-8B tokens at 160M+ (Tigges); 70M fails | World facts | 2 bits/param cap (Physics 3.3); RETRO 172M gains ~10x params; memory layers 134M triple NQ |
| Entity/state tracking through updates | Human level at 410M, 53.5% at 70M (Drozdz & Heilbron); code helps (Kim et al., >=7B) | Current conversation state (who, what, constraints) | Recap helps large models (Laban); goal tokens lose attention with length (Dongre et al.); untested small |
| Multi-hop composition | Fails at 1.3B/100B real tokens (Allen-Zhu 4.1); depth-driven (2.1, looping) | Arithmetic and exact computation | TinyGSM 125M: 63.1% GSM8K via Python |
| **Reading and trusting provided context correctly** | SmolLM2-360M 0% EM with gold passage; Daimax 0.5B/0.8B parrot retrieved chat | Long-past conversation turns | RMT on GPT-2-137M solves bAbI at 10M tokens (task-specific) |
| **Deciding to call a tool and formatting the call** | Trained: 90M single-call BFCL 41%; multi-turn 0% at <=270M; self-supervised needs ~775M | Fresh or private information | RAG / search: works only after the reading skill above exists |
| Stopping, role discipline, not looping | Small models loop more (Pipis; Falcon 90M; Daimax) | | |

**The central asymmetry:** externalizing knowledge does not remove the parameter bill, it moves it. The model still has to learn the skill of consuming external text faithfully, and at 360M that skill is absent unless trained (0% with the gold passage), while at 125M to 172M it is present when trained end to end (TinyGSM, RETRO). So "knowledge outside" is only a win if the reading skill is trained in from pretraining onward, and the retrieved content is facts and documents, not the model's own past chat (Daimax).

---

## 3. What this implies for a ~150M chat model

1. **Shape: deep and thin, full attention.** Every data point at this size (TinyStories, MobileLLM, SmolLM2-135M, Physics 2.1, Baguettotron, looping) says depth buys in-context skill while width buys facts. A SmolLM2-135M / MobileLLM-125M style 30 x 576 body is the evidence-backed default; at 150M with a 49K vocab that is about 28M embedding + about 120M in blocks. Keep softmax attention in most or all layers for recall; if a hybrid is used, keep several global attention layers (LFM2 keeps 6 of 16).
2. **Train at the serving context with long multi-turn data.** 4K tokens holds about 15 typical turns. Include conversations of 8 to 20 turns at full length in pretraining and SFT, not only 1 to 3 turn data (SmolLM2's multi-turn source is three-turn MagPie-Ultra).
3. **Mix chat and instruction data into pretraining.** Falcon's 90M went from 53.47 to 66.08 IFEval by moving SFT data into pretraining (vendor). TurnWise shows 10K multi-turn conversations give +12.8 points at 7B.
4. **Put state tracking into the data.** Code (entity tracking, >=7B evidence), burstiness (ICL), and synthetic state-update and cross-turn reference conversations. Pythia-to-OLMo 2 differences show data moves the 410M threshold; nobody knows how far down.
5. **Offload facts, but train the reader.** Retrieval-in-context and tool calls must be in the training data from the start (RETRO-style or RAG-style examples with relevant, irrelevant and conflicting passages), because an untrained small model ignores the context. Never feed a small model its own old chat excerpts as memory.
6. **Let the harness do the multi-hop.** A structured "conversation state" block (entities, user facts, active constraints, last decision), rewritten each turn and placed near the end of the context, turns "remember what I said 6 turns ago and combine it" into a 1-hop lookup, and re-surfaces instructions that attention decay would otherwise lose. Speculative at 150M, but it attacks exactly the skills the evidence says scale worst.
7. **Treat looping as a first-class training target**, not a sampler afterthought: filter spammy repetitive text (MaxGPT-2's "Bitcoin.com"), keep long chain-of-thought out of the 150M's data (Falcon 90M loops), consider unlikelihood/DITTO-style anti-repetition data and on-policy or reverse-KL distillation (MiniLLM reports lower exposure bias and better long-text generation for students from 120M to 13B, [arXiv 2306.08543](https://arxiv.org/abs/2306.08543)), and keep Daimax's degeneracy detector plus abort-and-recover at inference.
8. **Max's current recipe is compatible.** His measured 124M A/B (NorMuon + cautious WD + per-head gated attention + normalized value residual + 1/sqrt(depth) norm scaling: val loss 2.786 vs 2.993, 6.9% lower on 1.1B tokens; `~/Documents/Projects/Max's AI Model/maxgpt-ultra/docs/ab_2026-09-21/README.md`) improves LM loss at the right scale. Gated attention also removes the attention sink (1.7B evidence), which may help far-back instruction access. None of it has been measured on multi-turn behavior.

---

## 4. Evidence about the "wall": why chat quality degrades below ~400M

The wall is not one thing. It is at least four limits that happen to bind in the same size range, measured with different yardsticks:

**A skill ladder, ordered by the size at which each rung is reached (with the data regime that produced the number):**

| Rung | Status at ~150M | Evidence |
|---|---|---|
| Copying, induction, IOI-style binding | Present (Pythia-160M, after 2-8B tokens); absent at 70M and in 1-layer models | Tigges; Olsson |
| Single fact recall inside the window | Near-solved (98% at 340M/15B tokens) | Titans baseline |
| Recall-heavy extraction | Usable, improving with size (SWDE 58 at 360M vs 72 at 1.3B) | Based |
| Entity state through updates | **Below human level; crosses at ~410M** with Pile-era data | Drozdz & Heilbron 2026 |
| Two-hop composition over context | **Fails even at 1.3B/100B** real tokens; fixable in synthetic settings with depth, Canon layers, looping | Allen-Zhu 4.1; Saunshi et al. |
| Staying reliable over many turns | Degrades at every scale (39% drop, 8B more unreliable) | Laban et al. |

**Plus three non-skill limits:**

- **Knowledge:** linear in parameters (2 bits/param). A 150M model can hold at most about 14% of what the 1.1B Ultra can (300M vs 2.2B bits). LLM judges and humans penalize this as "bad chat", so single-number chat scores mix knowledge and coherence.
- **Stability:** smaller models loop more and multi-turn context feeds the loop.
- **Training distribution:** small-model SFT sets are filtered to be easier (SmolLM2 removed function calling and hard examples), mostly short, and mostly single-turn or three-turn; turn-taking, follow-up and tool behavior are distribution properties that small models do not get trained on.

**Where Max's premise is wrong or weak:** single-turn instruction following at 90M to 350M is already at or above Qwen2.5-0.5B-Instruct on vendor numbers, and the token budgets behind those models (0.8T to 10T) dwarf the research runs that produced the intuition that 100M to 250M "cannot chat" (MaxGPT-2 saw 482M unique tokens). **Where it is right:** no one has shown a sub-400M model with measured multi-turn coherence; Google disclaims it for 270M; the 90M and 270M tool models score 0% on multi-turn BFCL; and the state-tracking rung crosses human level at about 410M with the data Pythia used.

---

## 5. Levers

| Lever | Limit attacked | Expected effect | Evidence strength | Cost for Max |
|---|---|---|---|---|
| Deep-thin shape (about 30 layers at d~576) | State tracking, instruction following, multi-hop | Better in-context skills at fixed params | Strong at 125-350M (MobileLLM, TinyStories, Physics 2.1) | Free; slower tokens/s per param |
| Keep full softmax attention (or >=1/3 global attention in a hybrid) | Recall from earlier turns | Avoids the attention-vs-recurrent recall gap | Strong (Zoology, Repeat After Me, Based at 70M-1.4B) | Free at 4K context |
| Multi-turn chat (8-20 turns, full length) mixed into pretraining, not just SFT | Getting lost, turn-taking, instruction persistence | Falcon 90M +24% IFEval from SFT-in-pretraining; +12.8 multi-turn points at 7B | Moderate (vendor 90M; 7B paper) | Needs a multi-turn synthetic set; Claude sub can seed it slowly |
| State-tracking and cross-turn reference data, plus code | Entity tracking, coreference, 2-hop | Moves the ~410M threshold down by an unknown amount | Moderate (code helps at >=7B; OLMo 2 vs Pythia) | Synthetic generators are cheap to write |
| Structured conversation-state block maintained by the harness (or emitted by the model as a trained "memory write") | Multi-hop across turns, instruction decay | Converts multi-hop into 1-hop, re-surfaces goals | Speculative at 150M; mechanism supported at >=7B (recap, attention closing) | Harness code + a small SFT slice |
| Train the reader: RAG/RETRO-style examples with relevant, irrelevant and conflicting context from pretraining onward | Knowledge cap, confabulation from context | Knowledge-equivalent of ~10x params on factual tasks | Moderate-strong at 172M (RETRO) for LM loss; untested for chat | Retriever at train time; TF-IDF exists in Ultra repo |
| Tools (calculator/code, search) with supervised traces, no chain-of-thought | Knowledge, arithmetic | GSM8K 63% at 125M via Python; single-call BFCL 41% at 90M | Strong single-turn; multi-turn tool use at <=270M is 0% | SFT data generation |
| Memory layers / hashed N-gram tables | Knowledge cap without compute | NQ 0.91 to 3.16 at 134M (+937M memory params) | Moderate at 134M, factual only | VRAM for the table; custom kernel work |
| Anti-loop training (filter repetitive spam, DITTO-style data, reverse-KL or on-policy distillation) + inference guard (penalty, per-head induction descaling, degeneracy abort) | Loops, drift, exposure bias | Fewer degenerate turns; better long generations | Moderate (GPT-2 scale DITTO; MiniLLM 120M+; Daimax detector validated on real transcripts) | Low |
| Canon layers (causal conv1d, kernel 4) | Multi-hop depth, NoPE/length | 2-4x reasoning depth in synthetic; mixed at 1.3B real | Weak-moderate at real scale | Tiny params, small code change |
| Looping / weight-shared depth | Depth for reasoning without params | 12x2 beat 24x1 on math at 1B, half params | Moderate, [>=1B only] | Extra compute per token |
| Gated attention (already in recipe), possibly partial NoPE | Instruction persistence, far-back access | Sink 46.7% to 4.8%, better long-context | Weak for this use; [>=1B only] | Already paid |
| Multi-turn probe suite that separates knowledge from coherence | Measurement (the wall is unmeasured) | Lets every other lever be judged | Needed; no sub-1B multi-turn numbers exist | A few days; runs on the Mac |

---

## 6. Open questions

1. Where does the entity-tracking rung cross human level with modern data (FineWeb-Edu, Cosmopedia, code) and a deep-thin shape? The 410M figure is Pythia on 300B Pile tokens; OLMo 2 already does better. This is the single most decision-relevant unknown, and it is cheap to probe with public checkpoints (Pythia-160M/410M, SmolLM2-135M/360M, Falcon-H1-Tiny-90M) on the same narratives.
2. How do existing sub-400M chat models actually score on multi-turn probes (fact from N turns ago, state update, reference resolution, instruction retention, self-consistency, stopping)? No published numbers exist; running MT-Bench-101-style probes on SmolLM2-135M/360M, Gemma 3 270M, LFM2-350M, Falcon-H1-Tiny-90M and Qwen2.5-0.5B would locate the real wall.
3. Can a 150M model learn to faithfully consume a structured state block and retrieved passages, and to ignore irrelevant ones? RETRO says yes for perplexity at 172M; there is no chat-level measurement.
4. Is the 0% multi-turn tool-use result at 90M and 270M a data problem (no multi-turn tool traces) or a capacity problem?
5. Does heavy overtraining (Max's 100B-token build is about 670 tokens/param at 150M; production small models use about 11,000-29,000) strengthen or erode in-context behavior? Pythia-160M says circuits persist at 1,900 tokens/param; synthetic work says ICL can be transient.
6. Can synthetic data plus Canon layers or looping make in-context 2-hop work at 150M on real text, where 1.3B/100B fails?
7. Do hybrids (short conv or Mamba plus attention, as LFM2 and Falcon-H1-Tiny use) help or hurt multi-turn recall at 150M compared with pure attention at the same budget?
8. Does gated attention (already in the recipe) measurably improve instruction persistence over turns at 124M to 150M? Max has matched 124M checkpoints from the A/B that could be probed.
9. How much of a "bad chat" judgment at 150M is missing knowledge vs lost coherence? Probes where all needed information is in the context would separate the two.
