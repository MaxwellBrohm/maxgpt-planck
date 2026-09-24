# Diagnosis, capacity and representation lens: what a ~150M model can physically hold and compute for multi-turn chat

Date: 2026-09-23. One of three independent diagnoses. Question: given what ~150M parameters can store and compute, which demands of multi-turn chat are parameter-bound and which are not, where is the hard floor, and how far above it is 150M for the skill part of chat?

Sources: the verified lane reports and fact-checks in `research/lanes/` (corrected versions used wherever a `.verify.md` changed a claim), the follow-up fact-checks in `research/followup/*.verify.md`, primary papers and model cards fetched today, and a likelihood probe that the interrupted first attempt of this step ran on this Mac (`research/capacity_probe/`, section 4). Every external number carries a URL and the scale it was measured at. "My arithmetic" marks numbers I computed. "[>=0.5B only]" flags evidence that exists only at 0.5B or above.

**Resume note.** The first attempt of this step was killed by a kernel panic (memory exhaustion from concurrent model processes). This version was finished without loading, running or training any model: every probe number below was recomputed with plain Python from the JSONL files already saved in `capacity_probe/out/`, and only cells whose files are complete are used as results. Section 4.1 lists exactly what completed and what did not. The fine-tuning test the first attempt designed as the decisive experiment never ran.

---

## Bottom line

1. **Only one demand of chat is hard-bound by parameter count: stored world knowledge, and it is a slope, not a wall.** Extractable facts top out near 2 bits per parameter, so a 150M model holds at most about 37.5 MB of facts, realistically half that at Max's ~670 tokens per parameter, and none of the long tail. That is about 0.3x what Qwen2.5-0.5B can hold and a few percent of a 7B. No recipe has raised knowledge per parameter (fixed-size trend indistinguishable from zero across 100 dated open models). Chat about the world that relies on facts in the weights is not achievable at 150M; facts have to come from context, retrieval or tools. Reading a fact from context already works at 106M of body parameters (open-book 0.95 to 1.00 on the same long-tail items that closed-book gets 0.78 on).
2. **The conversational mechanics are far from any parameter floor.** Register, turn-taking, stopping, and recalling a user-stated fact across 12 turns and 1,700 tokens are shown in shipping models with 74M to 106M of body parameters (Falcon-H1-Tiny-90M, SmolLM2-135M), and on synthetic tasks with 2-layer, width-64 attention. A 150M model (122M to 141M of body, depending on vocabulary) has about 1.5x to 2x the demonstrated body for these.
3. **The one skill with a real size gradient below 300M of body parameters is binding under same-type interference**: keeping "my cat is Biscuit, my sister's cat is Pepper" straight, and the two-hop references built on it ("what does my sister do?" when two women and two jobs were mentioned). In the saved probe, all three ~100M-body models are at chance on two-hop references (0.51 to 0.55 single-item, n=192 or 76), while 287M to 440M bodies reach 0.86 to 0.99, but only with heavy recipes (LFM2.5-350M, Qwen3-0.6B). Within one data family the gradient is steep (SmolLM2 135M to 360M: owner binding "both right" at distance 10 goes 0.62 to 0.94, perspective binding 0.09 to 0.91). But recipe at fixed size moves it as much as 8.5x size does (LFM2-350M to LFM2.5-350M: one-hop binding 0.34 to 1.00; LFM2-350M to LFM2-2.6B at the same 10T recipe: two-hop 0.66 to 0.94), and a 38M transformer learns 4-hop binding with distractors from synthetic data. So the floor for this skill in a natural-text chat model lies somewhere between about 40M and about 290M of body parameters, and **150M sits inside that unknown band**. This is the crux of the capacity question, and it is testable in an afternoon (section 6, test 1).
4. **Updating a value after a correction fails in every model up to 0.6B, so it cannot be what separates 150M from 500M.** Once the correction is 4 or more turns back, all 8 sub-1B models probed prefer the stale original value (0.00 to 0.11 correct, n=192 each, margins down to -9 nats), including Qwen2.5-0.5B and Qwen3-0.6B; LFM2-2.6B gets 0.72. A surface-form control rules out n-gram copying: when every correction repeats the exact wording of the original, SmolLM2-135M and 360M still pick the original in 192 of 192 items at distance 4 and 10. This is the primacy intrusion that a 39-model study (1B to 2.5T) found in every model and found not to shrink with size. It is a training-signal and attention-bias problem to attack with data or a state mechanism, not a reason to add parameters.
5. **The 400-500M wall is misplaced.** On the in-context skills a chat needs, the "wall" model Qwen2.5-0.5B is no better than 287M to 315M bodies (perspective binding 0.12 vs 0.91 for SmolLM2-360M; two-hop 0.72 vs 0.86 for LFM2.5-350M), and it fails corrections like everything else below 1B. What looks like a wall at 400-500M is where labs spent tokens (2,900 to 83,000 per parameter) and distillation, plus the linear knowledge slope. Skill metrics (Multi-IF 37.7 at 230M, 44.9 at 350M) show no size wall at 400-500M.
6. **Depth, the embedding tax and the output-head rank are not what binds at 150M.** 30 layers is ample for the in-context operations chat needs; a 49k vocabulary costs 19% of a d=576 model, which is small next to the 2.5x to 3.6x body gap to the wall models (Qwen2.5-0.5B to Qwen3-0.6B); the softmax-bottleneck argument at d<1000 is contested and has no demonstrated effect on chat behaviour.

**Uncomfortable conclusions, stated plainly.** (a) A 150M model will never be a knowledgeable conversationalist on its own; "real back-and-forth chat" at 150M has to mean conversational mechanics over what is in the context plus retrieval. (b) At Max's planned 100B tokens and a web-heavy mix, the binding skill in point 3 is at genuine risk: the only ~100M-body models probed that were trained on 2T to 6T tokens do not have it. Whether targeted data can install it at this size is unknown until test 1 runs. If it cannot, the 150M target needs either an explicit state mechanism (section 6) or a larger body.

---

## 1. What 150M buys: the physical budget

### 1.1 Parameter ledger (tied embeddings; my arithmetic)

| shape | vocab | embedding | share | transformer body |
|---|---|---|---|---|
| d=576 | 49,152 (Max's tokenizer) | 28.3M | 18.9% | 121.7M |
| d=576 | 32,768 | 18.9M | 12.6% | 131.1M |
| d=576 | 16,384 | 9.4M | 6.3% | 140.6M |
| d=768 | 49,152 | 37.7M | 25.2% | 112.3M |
| d=768 | 16,384 | 12.6M | 8.4% | 137.4M |

Bodies of the models people compare against (my count from each `config.json`, matching the loaded-weight counts logged in `capacity_probe/logs/` and `lanes/probe.verify.md` section 1):

| model | total | body (non-embedding) | shape | pretraining tokens |
|---|---|---|---|---|
| Falcon-H1-Tiny-90M-Instruct | 91.1M | 74.3M | 24L, d=512, attention+Mamba2 | 800B |
| Gemma 3 270M | 268.1M | 100.3M | 18L, d=640, 15 of 18 layers 512-token window | 6T |
| SmolLM2-135M | 134.5M | 106.2M | 30L, d=576, 9 heads, 3 KV heads | 2T |
| LFM2.5-230M | 229.7M | 162.6M | 14L, d=1024, 6 attention | 19T |
| LFM2-350M / LFM2.5-350M | 354.5M | 287.4M | 16L, d=1024, 6 attention + 10 conv | 10T / 28T+RL |
| SmolLM2-360M | 361.8M | 314.6M | 32L, d=960 | 4T |
| Qwen2.5-0.5B ("the wall") | 494.0M | 357.9M | 24L, d=896 | 18T |
| Qwen3-0.6B | 596.0M | 440.5M | 28L, d=1024 | 36T |
| LFM2-2.6B | 2,569.3M | 2,435.1M | 30L, d=2048, 8 attention + 22 conv | 10T |

Sources: configs at `https://huggingface.co/<model>/resolve/main/config.json`; token counts from the model cards as verified in `lanes/probe.verify.md` claim 10; LFM2-2.6B and LFM2-350M share a stated "Training budget: 10 trillion tokens" ([LFM2-2.6B card](https://huggingface.co/LiquidAI/LFM2-2.6B)).

A 150M model has 122M to 141M of body. That is about 1.6x to 1.9x Falcon-H1-Tiny-90M, 1.15x to 1.4x SmolLM2-135M and Gemma 3 270M, 0.75x to 0.87x LFM2.5-230M, 0.34x to 0.39x Qwen2.5-0.5B, and 0.28x to 0.32x Qwen3-0.6B. The gap Max wants to close is 2.5x to 3.6x in body parameters, not 3.3x to 4x in model names.

### 1.2 Storage

- Extractable-fact ceiling: about 2 bits per parameter after about 1,000 exposures per fact, never above 2.3; about 1 bit per parameter at 100 exposures; gated MLPs about 1.3x lower at 100 exposures; a 1:7 useful-to-junk mix cuts useful capacity up to 20x (2x with a source tag) ([Physics of LMs 3.3, arXiv 2404.05405](https://arxiv.org/abs/2404.05405); GPT-2/LLaMA-style models, synthetic biographies; the per-size results cover 10K to 20M facts; confirmed in `followup/bits.verify.md` claim 1). Raw memorization of random strings is 3.5 (bf16) to 3.8 (fp32) bits per parameter ([Morris et al., arXiv 2505.24832](https://arxiv.org/abs/2505.24832); 80K to 6.9M params, synthetic).
- At 150M that is at most about 300 Mbit (37.5 MB) of facts. On a web-heavy mix at ~670 tokens per parameter, most facts are seen far fewer than 1,000 times, so realistically nearer the 1 bit per parameter regime, about 19 MB (my arithmetic). Qwen2.5-0.5B's ceiling is about 1 Gbit. The relationship is linear in parameters, so there is no size at which knowledge "turns on".
- Small models do not store rare items weakly; they store none. With synthetic biographies mixed into FineWeb-Edu, a 70M Pythia memorized almost nothing below a critical mixing ratio even at about 3,000 exposures per biography, and the critical ratio follows a power law in model size (fit on 70M to 410M), so smaller models need each item to be more frequent. The same phase transition appeared for a reasoning subtask at 70M ([Gu et al., arXiv 2505.18091](https://arxiv.org/abs/2505.18091), sections 3.2, 3.3, 5.1; Pythia 70M and 410M, 32B to 512B tokens; confirmed in `followup/bits.verify.md` claim 7). A bounded model spends capacity like a knapsack solver: below a density threshold an item, or a skill, gets nothing.
- Recipes have not bought facts per parameter. Across 93 open models (135M to 1.6T), obscure-fact accuracy is log-linear in parameters (R² 0.910, about +15.9 points per 10x). A separate fit over 100 dated open models (2023-09 to 2026-06) finds the fixed-size time trend is +0.0013 per month (95% CI -0.0004 to +0.0033, p = 0.19), while GPQA Diamond at fixed size drifts up about 2 points per month on 30 models with vendor scores ([Incompressible Knowledge Probes, arXiv 2604.24827](https://arxiv.org/abs/2604.24827); single-author preprint; fits dominated by models of 1B and up; corrected per `followup/bits.verify.md` claim 8). Recipes buy skill per parameter; they have not measurably bought knowledge per parameter.
- Realized knowledge also does not step up at 400-500M: Qwen2.5-0.5B base scores TriviaQA 4.3, SmolLM2-135M 4.1, SmolLM2-360M 16.9 on the SmolLM2 cards ([360M card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), [135M card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)).

### 1.3 Computation and representation

- Forward cost is about 2 x 150M = 0.3 GFLOP per token (tied head). A deep-thin 150M has about 30 serial layers, the same as SmolLM2-135M (30L, d=576) and MobileLLM-125M (30L, d=512) ([arXiv 2402.14905](https://arxiv.org/abs/2402.14905)). LFM2.5-230M reaches Multi-IF 37.7 with 14 layers of which 6 are attention ([card](https://huggingface.co/LiquidAI/LFM2.5-230M)), so depth is not scarce at this size.
- The in-context operations chat uses need little depth or width in principle. Induction (copying a fact from earlier context) needs two vanilla attention layers, or one if something upstream supplies the previous token (a smeared key or short causal conv) ([Olsson et al.](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html); the one-layer lower bound of [arXiv 2408.14332](https://arxiv.org/abs/2408.14332) is asymptotic in heads x width x precision and does not bind at ordinary widths; `followup/floor.verify.md` claim 2). Attention "solves Mqar perfectly at all sequence lengths using a constant model dimension of 64" in 2-layer models with single-token keys ([Zoology, arXiv 2312.04927](https://arxiv.org/abs/2312.04927)). Logarithmic depth suffices for basic parallel tasks such as k-hop ([arXiv 2402.09268](https://arxiv.org/abs/2402.09268), theory). A 37.8M, 12-layer transformer learned to dereference variable chains up to 4 deep with distractor chains at >99.9% from 450K synthetic programs ([Wu et al., arXiv 2505.20896](https://arxiv.org/abs/2505.20896)). Those programs never reassign a variable, so this is evidence about multi-hop binding, not about corrections (`followup/floor.verify.md` claim 6).
- The hard theoretical limit (fixed-depth transformers and SSMs cannot express arbitrary-length non-commutative state tracking: [arXiv 2404.08819](https://arxiv.org/abs/2404.08819)) is far from chat: a conversation's state is a handful of overwrite-style slots. The relevant known weakness is softer: transformers make sporadic "attention glitches" when copying the last-written value over long spans of irrelevant tokens ([flip-flop language modeling, arXiv 2306.00946](https://arxiv.org/abs/2306.00946)), and they show a systematic primacy bias when a value has been overwritten (section 4.4).
- Binding mechanisms are known to strengthen with scale: binding-ID vectors appear in "every sufficiently large model from the Pythia and LLaMA families" and "their fidelity increases with scale" ([Feng and Steinhardt, arXiv 2310.17191](https://arxiv.org/abs/2310.17191); Pythia and LLaMA from about 0.1B up; the text gives no numeric threshold). IOI-style name binding forms in Pythia-160M (85M body) but not Pythia-70M (19M body) ([arXiv 2407.10827](https://arxiv.org/html/2407.10827), IOI only, as corrected in `lanes/context.verify.md`).
- Output layer: the LM head's rank is at most d (576 to 768), below the ~1000 threshold under which Godey et al. saw late-training saturation in Pythia up to 410M ([arXiv 2404.07647](https://arxiv.org/abs/2404.07647)). Whether that harms training is contested: the head suppresses 95-99% of gradient norm ([Godey and Artzi, arXiv 2603.10145](https://arxiv.org/abs/2603.10145)), but a causal test on small WikiText-2 models found a backward-only rank cut costs 0.059 nats while an equal-rank factorized forward head costs 0.180 nats, confirming "strong geometric compression" but not "a harmful optimization bottleneck" ([arXiv 2608.16671](https://arxiv.org/abs/2608.16671); 5 seed pairs), and 108 controlled OLMo-style models show low effective rank co-occurs with saturation rather than causing it ([arXiv 2602.20433](https://arxiv.org/abs/2602.20433)). No study connects head rank to chat behaviour.

---

## 2. Is the 400-500M wall real? (capacity view)

Partly real, and misplaced.

1. **Knowledge is a slope, not a wall** (section 1.2). A 150M model holds about 0.3x what a 0.5B model can, and both hold a few percent of a 7B.
2. **Skill metrics show no size wall at 400-500M.** Sub-400M models match 0.5-0.8B models on 3-turn instruction following (LFM2.5-230M Multi-IF 37.70 and LFM2.5-350M 44.92 vs Qwen3.5-0.8B 41.68 in Liquid's harness, [LFM2.5-350M card](https://huggingface.co/LiquidAI/LFM2.5-350M); Multi-IF is a 3-turn average, not a turn-3 persistence rate, per `followup/floor.verify.md` claim 9). In the audited chat probe the 350M class is level with Qwen2.5-0.5B (hand-adjusted greedy macro 0.57 to 0.62 vs 0.60, `lanes/probe.verify.md` section 6). Skill does rise with size at a fixed recipe (LFM2 Multi-IF 32.85 at 350M, 40.92 at 700M, 45.28 at 1.2B, [arXiv 2511.23404](https://arxiv.org/html/2511.23404v1) Table 6; MobileLLM MT-Bench 2.33 at 125M vs 3.28 at 350M after identical chat fine-tuning, [arXiv 2402.14905](https://arxiv.org/abs/2402.14905)). But the recipe change from LFM2-350M to LFM2.5-350M (28T tokens plus RL, same architecture) moved Multi-IF 32.92 to 44.92 on the LFM2.5 card, about what a 3.4x size step bought inside the LFM2 table (two different Liquid tables, so treat the comparison as approximate).
3. **The perceived wall sits where compute and distillation were spent.** Every sub-600M model with good chat numbers used about 2,900 to 83,000 tokens per parameter plus large-teacher distillation or RL (`lanes/census.verify.md`, `lanes/probe.verify.md` claim 10). Max's own small models ran at about 18 (MaxGPT-2) and 85 (MaxGPT-3) tokens per parameter (`WRITEUP_NOTES.md`). His log calls MaxGPT-2's confident nonsense on photosynthesis a "fundamental capacity limit". At 18 tokens per parameter it was a data limit first: a 106M-body model trained on 2T tokens picks the right answer on 95% of common-knowledge items in a two-way test (section 4.2).
4. **The in-context skills do not step up at 400-500M either** (section 4). Qwen2.5-0.5B is not better than 287M to 315M bodies at binding, and every model below 1B fails corrections. The one skill with a size gradient in the probe (binding under interference) changes between about 100M and 300M of body, not at 400-500M, and even there recipe moves it as much as size.

---

## 3. Which demands of multi-turn chat are parameter-bound

| demand | what it needs physically | parameter-bound? | smallest size where shown | margin at 150M (122-141M body) |
|---|---|---|---|---|
| Fluent chat register, turn-taking, stopping | local syntax and format; a few layers | No | narrow-world fluency at 1-33M ([TinyStories, arXiv 2305.07759](https://arxiv.org/abs/2305.07759)); no template leaks in 149 greedy turns from 91M up (`lanes/probe.verify.md` claim 16) | large |
| Recall one user-stated fact across turns (under ~2k tokens), no competing value | induction / retrieval heads; 2 attention layers at width 64 | No | 12 turns and 1,726 tokens at 74M body (Falcon-H1-Tiny-90M; greedy-from-prefix 1.00, hand-strict free answers 0.65-0.75, `lanes/probe.verify.md` claim 6) | about 1.6-1.9x that body |
| Read a stated fact over a missing or weak prior | same as above | No | open-book 0.95-1.00 at 106M body on long-tail facts it gets 0.78 closed-book (section 4.2) | ample |
| Common world knowledge | stored facts at high frequency | Weakly | 0.87-0.95 two-way accuracy at 106M body (section 4.2) | adequate |
| Long-tail world knowledge | stored facts at low frequency | Yes, linear | none: a ceiling, not a floor | about 0.3x Qwen2.5-0.5B |
| Binding a value to its owner with a same-type distractor in context | binding circuits matching multi-token keys | Mixed, unresolved | synthetic: 37.8M ([arXiv 2505.20896](https://arxiv.org/abs/2505.20896)); IOI at 85M body in Pythia; in chat-style text: not at ~100M body (3 of 3 probed), yes at 315M-440M body (section 4.3) | inside the unknown band |
| Two-hop reference ("what does my sister do?") | two dependent bindings; depth ample at 14-30 layers | Mixed, unresolved | synthetic 37.8M; chat-style text: chance at ~100M body, 0.86-0.99 at 287M-440M body with heavy recipes (section 4.3); natural pretraining at 1.3B/100B still fails a harder 2-hop task ([arXiv 2512.17351](https://arxiv.org/abs/2512.17351)) | inside the unknown band |
| Updating a value after a correction, with turns in between | recency/override must beat primacy | Not parameter-bound in the 150M-500M range | fails in all 8 models up to 0.6B, works at 2.6B (section 4.4); primacy intrusion does not shrink with size at 1B-2.5T [>=1B only] ([arXiv 2603.00270](https://arxiv.org/abs/2603.00270)); latest-value updating learned from scratch in bAbI dialog task 2 by small memory networks ([arXiv 1605.07683](https://arxiv.org/abs/1605.07683)) | no sub-1B model has it off the shelf |
| First/second-person perspective (I vs you vs a third party) | role binding across speaker turns | Mostly not (non-monotonic in size) | 0.91 "both right" at 315M body (SmolLM2-360M) vs 0.12 at 358M (Qwen2.5-0.5B) (section 4.5) | recipe-dependent |
| Instruction and persona persistence over 3 turns | attention back to a distant instruction | No | Multi-IF 37.7 at 230M total (LFM2.5-230M, vendor 3-turn average) | moderate |
| Not looping | output distribution not collapsing into attractors | Mostly no | no loops at 91M in chat (probe lane); clean stopping at 1-33M on narrow data | large, given data |

---

## 4. Probe evidence: what the saved likelihood probe shows

### 4.1 Design, and what completed

**Design** (`capacity_probe/items.py`, `run_capacity.py`). Every model reads the same plain `User:` / `Assistant:` transcript (no chat template), and the probe compares the summed log-probability of the gold continuation against foils after a fixed answer prefix. Every in-context foil is a same-type value that also appears in the context, so a positional or copy shortcut cannot score well. Items: R1 owner binding (my cat vs my sister's cat), R2 one-hop and two-hop references (two women, two jobs), U state updates with 1 to 3 corrections, P perspective (user name vs sister vs assistant), each at 0, 4 and 10 distractor turns (max prompt 577 tokens), 32 scenarios per cell, plus 60 closed-book two-way knowledge items. Follow-ups: `khard.py` (40 long-tail facts closed-book and the same facts stated by the user two turns earlier), `uprobe.py` (update controls: every correction in identical wording, and a neutral answer prefix). "Both right" metrics require both paired questions of a scenario to be right, so a model that always picks the first-mentioned value scores 0 and a coin-flipper about 0.25. Single-item two-way chance is 0.50.

**Completeness, checked from the files on disk:**

| run | status | used here |
|---|---|---|
| main battery, 8 models (SmolLM2-135M base and Instruct, LFM2-350M, LFM2.5-350M, SmolLM2-360M-Instruct, Qwen2.5-0.5B-Instruct, Qwen3-0.6B, LFM2-2.6B) | complete, 1,116 of 1,116 items each | yes |
| main battery, Gemma 3 270M-it | **incomplete**, 420 of 1,116 (distance 0 complete; distance 4 for 6 of 32 scenarios; no distance 10; no knowledge items) | distance 0 only, flagged |
| main battery, Falcon-H1-Tiny-90M | **not run** (killed: Mamba path falls back to slow CPU ops on MPS; log empty) | no |
| khard, 9 models (all of the above except Falcon) | complete, 80 of 80 each | yes |
| khard, Falcon | **crashed** (`RuntimeError: invalid low watermark ratio 1.4`) | no |
| uprobe, SmolLM2-135M-Instruct and SmolLM2-360M-Instruct | complete, 576 of 576 each | yes |
| uprobe, LFM2-350M, LFM2.5-350M, Qwen2.5, Qwen3, LFM2-2.6B | **not run** | no |
| copysplit (per-token loss on OASST1 split by copyability) | **smoke test only**: one model, 1,002 scored tokens | no |
| ft_test (fine-tune SmolLM2-135M/360M on disjoint-template dialogues, then re-score) | **never completed**: log ends after weight loading at 17:08; no `ft__`, `ftu__` or `ftk__` output | no |

Caveats that apply to every number below: likelihood forced choice is not generation; the plain transcript is off-template for instruct models (the probe lane showed rendering alone moves SmolLM2-135M-Instruct recall from 0.15 to 0.70, `lanes/probe.verify.md`), which is why the SmolLM2-135M base model is included; n is 32 to 192 per cell (a 32-item cell has a 95% Wilson interval of about ±0.15 to ±0.17 near the middle); fp32 on MPS (LFM2-2.6B in bf16). Items share 16-name pools, so they are not fully independent.

### 4.2 Knowledge: storage is weak at 100M, reading is not

| model | body | common facts (60, two-way) | long-tail closed book (40) | same facts, stated 2 turns earlier |
|---|---|---|---|---|
| SmolLM2-135M base | 106M | 0.95 | 0.78 | 1.00 |
| SmolLM2-135M-Instruct | 106M | 0.87 | 0.78 | 0.95 |
| Gemma 3 270M-it | 100M | not run | 0.95 | 0.98 |
| LFM2-350M | 287M | 0.92 | 0.85 | 1.00 |
| LFM2.5-350M | 287M | 0.95 | 0.85 | 1.00 |
| SmolLM2-360M-Instruct | 315M | 0.95 | 0.88 | 0.98 |
| Qwen2.5-0.5B-Instruct | 358M | 0.98 | 0.85 | 0.98 |
| Qwen3-0.6B | 440M | 0.95 | 0.80 | 1.00 |
| LFM2-2.6B | 2,435M | 0.98 | 1.00 | 1.00 |

Reading from context works at 106M of body: of the 9 long-tail items SmolLM2-135M base gets wrong closed-book, context fixes all 9 and breaks none. The two-way format with plausible foils is insensitive (TriviaQA, which requires producing the answer, is 4.1 at 135M vs 16.9 at 360M), so the closed-book column understates the size gradient. One oddity worth a follow-up: Gemma 3 270M, with only 100M of body but a 168M-parameter embedding table (262k vocabulary), scores 0.95 closed-book on long-tail items, above every 287M-440M body. n=40, so this is a hint, not a finding, but it fits the idea that large lookup tables can hold entity knowledge cheaply (the Engram / memory-layer direction in `lanes/arch.md`).

### 4.3 Binding and two-hop references: the one size gradient below 300M

Pooled results (95% Wilson intervals). "d10" is 10 distractor turns after the facts; "d0-10" pools all distances.

| model | body | owner binding, both right, d10 | one-hop binding, both right, d10 | two-hop, single item, d0-10 | two-hop, both right, d0-10 | perspective, both right, d10 |
|---|---|---|---|---|---|---|
| SmolLM2-135M base | 106M | 0.22 [0.11-0.39] | 0.22 [0.11-0.39] | 0.54 [0.47-0.61] | 0.15 [0.09-0.23] | 0.12 [0.05-0.28] |
| SmolLM2-135M-Instruct | 106M | 0.62 [0.45-0.77] | 0.38 [0.23-0.55] | 0.51 [0.44-0.58] | 0.14 [0.08-0.22] | 0.09 [0.03-0.24] |
| Gemma 3 270M-it (d0 only) | 100M | 0.47 at d0 | 0.69 at d0 | 0.55 [0.44-0.66], n=76 | 0.18 [0.09-0.33] | 0.94 at d0 |
| LFM2-350M | 287M | 0.34 [0.20-0.52] | 0.34 [0.20-0.52] | 0.66 [0.59-0.72] | 0.39 [0.29-0.49] | 0.09 [0.03-0.24] |
| LFM2.5-350M | 287M | 0.47 [0.31-0.64] | 1.00 [0.89-1.00] | 0.86 [0.80-0.90] | 0.72 [0.62-0.80] | 0.00 [0.00-0.11] |
| SmolLM2-360M-Instruct | 315M | 0.94 [0.80-0.98] | 0.59 [0.42-0.74] | 0.63 [0.56-0.70] | 0.43 [0.33-0.53] | 0.91 [0.76-0.97] |
| Qwen2.5-0.5B-Instruct | 358M | 0.75 [0.58-0.87] | 0.66 [0.48-0.80] | 0.72 [0.66-0.78] | 0.48 [0.38-0.58] | 0.12 [0.05-0.28] |
| Qwen3-0.6B | 440M | 1.00 [0.89-1.00] | 1.00 [0.89-1.00] | 0.99 [0.96-1.00] | 0.98 [0.93-0.99] | 0.97 [0.84-0.99] |
| LFM2-2.6B | 2,435M | 1.00 [0.89-1.00] | 1.00 [0.89-1.00] | 0.94 [0.90-0.97] | 0.89 [0.81-0.93] | 0.97 [0.84-0.99] |

What this says:

- **Two-hop references are at chance in all three ~100M-body models** (single-item 0.51 to 0.55 against chance 0.50), while one-hop on the same contexts is above chance (0.61 to 0.84 single-item). The deeper cause is that even one-hop binding with a same-type competitor is weak at 106M: "Lena works as a" must pick Lena's job over Chloe's from the sentence "Lena works as a dentist, and Chloe works as a pilot", which is an exact 4-gram copy except that the distinguishing token sits three positions back. SmolLM2-135M gets both halves of that pair right only 0.44 of the time at distance 0 and 0.22-0.38 at distance 10. A pure previous-token induction head cannot do it; it needs a key that combines the name with the following tokens.
- **Shortcut use at small size.** Several small models fall back on "pick the first-mentioned value" at distance: LFM2-350M picks the first-mentioned owner in 83% of items at distance 10 (accuracy 1.00 when the gold was mentioned first, 0.34 when second); SmolLM2-135M base 70-75% at all distances. Qwen3-0.6B, SmolLM2-360M and LFM2-2.6B show no positional preference (0.47 to 0.52).
- **Same-recipe size steps are steep.** SmolLM2 135M to 360M (2T to 4T tokens, same data family): owner binding 0.62 to 0.94, perspective 0.09 to 0.91. LFM2-350M to LFM2-2.6B (identical 10T budget and post-training recipe per the card): two-hop single-item 0.66 to 0.94, owner binding 0.34 to 1.00.
- **Recipe steps at fixed size are just as steep.** LFM2-350M to LFM2.5-350M (same 287M body; 10T to 28T tokens plus RL): one-hop binding 0.34 to 1.00, two-hop 0.66 to 0.86. And the size order is not monotonic across families: Qwen2.5-0.5B (358M) is below SmolLM2-360M (315M) on owner and perspective binding and below LFM2.5-350M (287M) on two-hop.

Reading: in-context binding under interference is genuinely harder for ~100M-body models trained on natural text, and it is the one place in this probe where the gap between a 150M-class model and the 0.3-0.4B class is plausibly partly parameter-bound. But three facts argue it is not a hard floor at 150M: a 37.8M transformer learns 4-hop binding with distractors from targeted synthetic data (>99.9%, [arXiv 2505.20896](https://arxiv.org/abs/2505.20896)); IOI-style binding forms at 85M body in Pythia; and recipe alone at 287M body moves binding as far as 8.5x size does. The knapsack result (section 1.2) gives the mechanism that reconciles both: at small size a skill is only learned if its training signal is dense enough, and the required density rises as the model shrinks. SmolLM2-135M's 2T tokens of web, code and math contain almost no binding-under-interference dialogue. **Not measured**: Falcon-H1-Tiny-90M, the best-recipe model near this size (its main-battery run was skipped). The probe lane's single binding test found Falcon wrong in forced choice but right in 3 of 4 free answers (`lanes/probe.verify.md` claim 11).

### 4.4 Corrections: primacy wins in every sub-1B model

Fraction choosing the latest value, pooled over 1 to 3 corrections (n=96 at d0, n=192 at d4+d10). Each correction is separated by one distractor turn; "d" is the number of further distractor turns before the question.

| model | body | latest wins, d0 | latest wins, d4 and d10 | mean margin gold minus original, k=1, d10 |
|---|---|---|---|---|
| SmolLM2-135M base | 106M | 0.40 | 0.02 [0.01-0.05] | -1.3 nats |
| SmolLM2-135M-Instruct | 106M | 0.10 | 0.00 [0.00-0.02] | -2.8 |
| Gemma 3 270M-it | 100M | 0.73 | 0.00 (n=18, partial) | n/a |
| LFM2-350M | 287M | 0.35 | 0.11 [0.07-0.16] | -0.9 |
| LFM2.5-350M | 287M | 0.01 | 0.00 [0.00-0.02] | -8.8 |
| SmolLM2-360M-Instruct | 315M | 0.46 | 0.00 [0.00-0.02] | -3.8 |
| Qwen2.5-0.5B-Instruct | 358M | 0.76 | 0.00 [0.00-0.02] | -3.6 |
| Qwen3-0.6B | 440M | 0.14 | 0.00 [0.00-0.02] | -2.8 |
| LFM2-2.6B | 2,435M | 1.00 | 0.72 [0.66-0.78] | +1.8 |

In every model up to 0.6B, the wrong answer that wins is the original value, not the intermediate one (for example Qwen2.5-0.5B picks the original in 32 of 32 at d4 and d10 for k=1 and k=3).

The surface-form control (`uprobe.py`, SmolLM2-135M and 360M only) rules out the obvious artifact. In the main items only the original statement shares the n-gram "appointment on" with the answer prefix, so an induction head would favour it. In the control, the original and every correction use identical wording ("my dentist appointment is on X now", acknowledged as "your dentist appointment is on X"), so pure copying has no reason to prefer the first. Result: at d4 and d10, both models pick the original in 192 of 192 items (latest-vs-original 0.00). With a neutral answer prefix ("That would be"), 0.00 to 0.06. At d0 they partly succeed (360M: 0.97 to 1.00 with identical wording).

Reading: this is proactive interference by primacy, not a copy artifact and not a 150M-specific weakness. It matches "LLMs Remember First, Forget Last": across 39 models from 1B to 2.5T, every model shows proactive interference worse than retroactive, the failures are "active primacy intrusion", and "model size predicts RI resistance but not PI" ([arXiv 2603.00270](https://arxiv.org/abs/2603.00270), abstract fetched today; their paradigm uses 3+ updates on 46 keys, [>=1B only]). My probe adds that at 1 to 3 corrections on a single key, every sub-1B model fails once the correction is a few turns back, and that LFM2-2.6B does not. Within the LFM2 recipe, size helped (0.11 to 0.72), but no model in the 350M to 600M range, including the "wall" model, is better than 135M here. The probe lane's generative test with one distractor turn agrees (corrections failed 20-21 of 24 across 8 models, `lanes/probe.verify.md` claim 9). Latest-value updating is learnable from scratch by small supervised models (bAbI dialog task 2: users "update their requests between 1 and 4 times", memory networks reach 100% per dialog, [arXiv 1605.07683](https://arxiv.org/abs/1605.07683)), which points to a training-signal gap plus an attention bias rather than a parameter floor. Whether a 106M-body transformer can overcome the bias with a few hundred steps of targeted data is exactly what the unrun fine-tuning test measures.

### 4.5 Perspective: recipe, not size

"What's my name?" / "What's your name?" after the user introduced themselves and a sister and the assistant introduced itself. Both right at d10: 0.09 to 0.12 for SmolLM2-135M, LFM2-350M and Qwen2.5-0.5B, 0.00 for LFM2.5-350M (it answers "My name is" with the user's name in 31 of 32 at d10), versus 0.91 for SmolLM2-360M and 0.97 for Qwen3-0.6B and LFM2-2.6B. Gemma 3 270M gets 0.94 at d0. The order is not monotonic in size (358M Qwen2.5 fails where 315M SmolLM2 succeeds), and the probe lane found role capture in 7 to 8 of 8 models from 90M to 600M (`lanes/probe.verify.md` claim 12). This is a post-training and data property, not a capacity one. Caveat: this is the task most likely to be affected by the off-template plain transcript.

---

## 5. Ranked limits for a ~150M chat model (most binding first)

### Rank 1. Stored world knowledge. Kind: hard.

- **Mechanism.** Facts live in weights at about 2 bits per parameter at best, about 1 bit at the exposure counts a 100B-token web mix gives, and zero for items below a density threshold that rises as models shrink. At 150M that is at most 37.5 MB, realistically ~19 MB, with the long tail absent.
- **Evidence.** [arXiv 2404.05405](https://arxiv.org/abs/2404.05405) (synthetic, GPT-2/LLaMA-style); [arXiv 2505.18091](https://arxiv.org/abs/2505.18091) (Pythia 70M/410M, critical mixing ratio); [arXiv 2604.24827](https://arxiv.org/abs/2604.24827) (93 models 135M to 1.6T, +15.9 points per 10x, no fixed-size time trend over 100 dated models); TriviaQA 4.1 (135M) vs 16.9 (360M) on the SmolLM2 cards. Every sub-300M vendor scopes its model away from knowledge-heavy use (`lanes/census.verify.md`).
- **Probe.** Long-tail two-way closed book 0.78 at 106M body vs 0.80-0.88 at 287M-440M and 1.00 at 2.4B; common facts 0.87-0.98 everywhere; open-book 0.95-1.00 everywhere (section 4.2).
- **Attacks.** Retrieval or tool grounding trained in from pretraining, including passages that conflict with the prior; knowledge-light, "I don't know"-capable SFT; sparse lookup memory if "150M" means active parameters (134M base + Memory+ roughly matches a 373M dense model on TriviaQA, [arXiv 2412.09764](https://arxiv.org/abs/2412.09764), 1T tokens; but at a 151M base, MoE beat PKM/UltraMem on TriviaQA, [arXiv 2411.12364](https://arxiv.org/abs/2411.12364), so the ordering is contested); source-tag prepending (cuts junk dilution from 20x to 2x in the synthetic setting); a larger embedding table as cheap entity storage (hint from Gemma 3 270M, section 4.2).
- **Cheapest decisive test.** None is needed for the limit itself. For the attack: a counterfactual open-book probe on SmolLM2-135M (the `khard` items with a stated wrong-but-plausible fact, e.g. "the capital of Laos is Pakse") measures whether a 106M body follows context over its prior. Minutes, likelihood only.

### Rank 2. Binding under same-type interference, and the two-hop references built on it. Kind: mixed.

- **Mechanism.** When two same-type values are in context (two pets, two women and their jobs, three names), answering requires a key that combines the owner with the value's position, not a one-token induction match. Binding representations are known to sharpen with scale ([arXiv 2310.17191](https://arxiv.org/abs/2310.17191)), and at small size the circuit is only learned if training contains enough of it (knapsack threshold, [arXiv 2505.18091](https://arxiv.org/abs/2505.18091)). Without it, small models fall back on "first-mentioned wins".
- **Evidence.** Same-recipe size steps (SmolLM2 135M to 360M; LFM2 350M to 2.6B) and a same-size recipe step (LFM2 to LFM2.5) of similar magnitude, section 4.3; IOI binding at Pythia-160M but not 70M ([arXiv 2407.10827](https://arxiv.org/html/2407.10827)); entity tracking at chance in Pythia-70M, human-level likelihood read-out at 410M on 300B Pile tokens ([arXiv 2608.18083](https://arxiv.org/html/2608.18083)); existence proof at 37.8M on synthetic programs ([arXiv 2505.20896](https://arxiv.org/abs/2505.20896)).
- **Probe.** Two-hop at chance (0.51-0.55) in all three ~100M-body models; one-hop "both right" 0.22-0.44 for SmolLM2-135M; 0.86-0.99 two-hop at 287M-440M with heavy recipes, 0.63-0.72 with lighter ones; the 150M target body (122M-141M) has no probed model.
- **Attacks.** (a) Binding-dense synthetic dialogue mixed into pretraining at a density above the critical ratio, not only in SFT (known idea at the task level: bAbI, Wu et al.; untested as a pretraining-mix intervention in a chat LM). (b) Cheap previous-token mixing (Canon layers / short causal conv, [arXiv 2512.17351](https://arxiv.org/abs/2512.17351)) so multi-token keys cost one layer instead of two; synthetic gains are 2-4x reasoning depth at 50-124M-class sizes, real-data evidence only at >=1B. (c) Speaker/role embeddings for perspective ([TransferTransfo, arXiv 1901.08149](https://arxiv.org/abs/1901.08149), 117M). (d) Distillation from a teacher that has the skill (check MaxGPT-Ultra on this battery first). (e) Plausible new: a model-written running state line per turn ("user: Priya; sister: Lena, dentist; appt: Thu") trained so that later answers copy from the state line, converting binding and two-hop into one-hop local copies. SimpleTOD shows the belief-state format is learnable at 82M for cumulative state ([arXiv 2005.00796](https://arxiv.org/abs/2005.00796)).
- **Cheapest decisive test.** Run the already-written `capacity_probe/ft_test.py` (400 steps, batch 16, disjoint names and templates) on SmolLM2-135M-Instruct, then SmolLM2-360M-Instruct, one process at a time, and re-score the unchanged battery. Pass criteria for "training signal, not capacity": two-hop "both right" at d10 from 0.09 to at least 0.8, owner and perspective "both right" at least 0.85, with closed-book knowledge not dropping more than 0.05. If training loss on the disjoint templates goes near zero while held-out two-hop stays under 0.65, the skill is capacity-limited at 106M body for this fine-tuning budget, and a 150M model would need either more body or the state-line mechanism. GPU minutes; see section 6 for the memory-safe way to run it.

### Rank 3. Updating after a correction (proactive interference by primacy). Kind: unknown (almost certainly not the 150M-vs-500M difference).

- **Mechanism.** Attention favours the earliest encoding of a repeated key; once the correction is not the most recent context, the original value wins. Large-model evidence says this does not shrink with size at 1B to 2.5T and is a winner-take-all attention effect ([arXiv 2603.00270](https://arxiv.org/abs/2603.00270)); the one-message control in the probe lane (7 of 8 models prefer the corrected value when the sentences arrive in one message) shows the model can represent "moved to".
- **Evidence.** bAbI dialog task 2 (small memory networks, 100% per dialog, [arXiv 1605.07683](https://arxiv.org/abs/1605.07683)); flip-flop attention glitches ([arXiv 2306.00946](https://arxiv.org/abs/2306.00946)); PI-LLM's size effect ([arXiv 2506.08184](https://arxiv.org/abs/2506.08184), 30 models from 0.6B) is contradicted by the larger 39-model study.
- **Probe.** 0.00-0.11 latest-wins at d4/d10 in all 8 sub-1B models (n=192 each), 0.72 at LFM2-2.6B; identical-wording control 0 of 192 at 135M and 360M (section 4.4).
- **Attacks.** Correction dialogues with stale-value traps at distance, in pretraining mix and SFT (partially tested elsewhere: bAbI QA1, bAbI dialog task 2, SleepGate at 793K params, [arXiv 2603.14517](https://arxiv.org/abs/2603.14517)); the running state line from rank 2 (the model rewrites the slot, so the answer is a recent copy); a learned recency bias in a few heads (ALiBi-style slopes or gated attention) as an architectural A/B; DPO pairs with the stale value as the rejected answer.
- **Cheapest decisive test.** The same `ft_test.py` run: its update items and the uprobe controls are re-scored after fine-tuning. Pass: latest-wins at d4/d10 from 0.00 to at least 0.8 on held-out wording. If it moves, corrections go on the data list; if not, add the state mechanism.

### Rank 4. Capacity allocation at ~670 tokens per parameter on a web-heavy mix. Kind: mixed.

- **Mechanism.** This is the link between ranks 1-3 and the data plan: in a bounded model, items and skills below a critical frequency get no capacity at all, and the critical frequency rises as the model shrinks ([arXiv 2505.18091](https://arxiv.org/abs/2505.18091), including a reasoning subtask at 70M). Chat skills are rare in web text (2.7-3.7% conversational, [arXiv 2508.10975](https://arxiv.org/abs/2508.10975)), and Max's 100B build has almost no dialogue. It follows from parameter count (the threshold is size-dependent) but the attack is data density.
- **Evidence.** Every sub-600M model with good chat numbers used 2,900-83,000 tokens per parameter plus distillation or RL; in the probe, the models that own binding at ~300-440M body are exactly the heavy-recipe ones (LFM2.5, Qwen3), and SmolLM2 without distillation lacks it at both 135M and, partly, 360M.
- **Probe.** Recipe step at fixed 287M body moves one-hop binding 0.34 to 1.00 (section 4.3).
- **Attacks.** SFT-format and synthetic multi-turn data at 15-25% of pretraining (Falcon-H1-Tiny recipe; the gain it shows is IFEval, not MT-Bench); spaced repetition of a curated dialogue pool; distillation.
- **Cheapest decisive test.** A mixing-ratio sweep at 30-60M: synthetic binding and correction dialogues at 0.5%, 2%, 8% of a FineWeb-Edu stream, 2-5B tokens each, scored with this battery as a Tier-0 likelihood probe. It locates the critical ratio for the skill at small size and predicts it at 150M. One Titan-day per run after Ultra finishes (or the 5070 when it returns).

### Rank 5. Output-head rank below ~1000 (softmax bottleneck). Kind: unknown, probably minor.

- **Mechanism.** A rank-576-768 head compresses the output distribution and 95-99% of the gradient norm.
- **Evidence.** Saturation in Pythia <=410M ([arXiv 2404.07647](https://arxiv.org/abs/2404.07647)); causal test says geometry yes, harmful optimization not shown ([arXiv 2608.16671](https://arxiv.org/abs/2608.16671)); 108-model study says co-occurrence, not cause ([arXiv 2602.20433](https://arxiv.org/abs/2602.20433)). Nothing links it to chat behaviour or loops.
- **Probe.** None.
- **Attacks.** d>=768 or a factorized higher-rank head; weight-decay and batch-size choices that keep effective rank up.
- **Cheapest decisive test.** A 60M A/B with head rank varied, judged on the probe battery and loop rate rather than loss. Low priority.

### Rank 6. Embedding share of the budget. Kind: soft (a design choice).

- **Mechanism.** Embedding parameters do not add serial computation. A 49k vocabulary costs 28.3M (19%) at d=576 or 37.7M (25%) at d=768.
- **Evidence.** Moving to 16-32k frees 9-19M (7-16% more body, section 1.1): small next to the 2.5x-3.6x body gap. The in-scale vocab optimum at 302M non-vocab parameters is 16K compute-optimal and 24K with excess data ([arXiv 2407.13623](https://arxiv.org/abs/2407.13623)). The cost of switching: MaxGPT-Ultra stops being a same-tokenizer teacher. Gemma 3 270M's result (section 4.2) suggests a large table is not pure waste for knowledge.
- **Probe.** Indirect only.
- **Attacks.** Keep 49k if distillation from Ultra is planned; otherwise 24-32k.
- **Cheapest decisive test.** Not worth a dedicated run before test 1.

### Rank 7. Depth and serial computation. Kind: soft; not binding.

- **Mechanism.** In-context retrieval needs 2 attention layers (or 1 with a previous-token path); k-hop needs about log depth; a 30-layer 150M model has ample depth.
- **Evidence.** Section 1.3; LFM2.5-230M chats with 14 layers (6 attention); MobileLLM finds about +1 zero-shot point from deeper shapes at 125M, flat beyond ~24 layers.
- **Probe.** The two-hop failure at 106M body is explained by weak one-hop binding, not missing depth (SmolLM2-135M has 30 layers, more than LFM2.5-350M's 16).
- **Attacks.** None needed beyond a reasonable deep-thin or LFM2-like shape.
- **Cheapest decisive test.** None needed.

Not ranked because they are not capacity limits: post-training deflection and role capture, loops, and SFT packing that cuts conversations from their history. All are real, all are recipe (`lanes/probe.md`, `lanes/compute.md`).

---

## 6. Decisive tests, cheapest first

**Hard rule for anyone running these on the Mac:** one model process at a time, nothing else resident. The first attempt of this step crashed the machine by starting the fine-tuning test from more than one queue script while other model processes were active. As of this writing, LM Studio is also holding a 12B model resident (about 9.7 GB). `ft_test.py` computes full-vocabulary logits over up to 640 positions x batch 16 in fp32 (about 2 GB for the logits alone, before gradients and AdamW state). Run it on a Titan after Ultra, on the 5070, or on the Mac alone with LM Studio unloaded and batch 8.

1. **Fine-tuning test (decides ranks 2 and 3).** `python ft_test.py HuggingFaceTB/SmolLM2-135M-Instruct --steps 400`, then the 360M. The script writes `out/ft__`, `ftu__`, `ftk__` files that `analyze.py` already knows how to tabulate (`+FT` rows). Pass criteria are in ranks 2 and 3. Cost: minutes of GPU. This is the single most informative experiment available for the capacity question.
2. **Fill the missing cells (sharpens rank 2).** Falcon-H1-Tiny-90M-Instruct plus its four same-architecture siblings (`-Instruct-Curriculum`, `-Instruct-pre-DPO`, `-Instruct-Curriculum-pre-DPO`, `-Base`, all on the Hub since 2026-01-12) on CUDA, where the Mamba kernels exist; Gemma 3 270M full run; `uprobe.py` on the 5 remaining models and LFM2-2.6B. About an hour of GPU. The Falcon rows answer whether the best ~74M-body chat recipe has binding under interference.
3. **Counterfactual open-book probe (decides the retrieval attack for rank 1).** Likelihood only, minutes.
4. **Run this battery on Max's own checkpoints.** The 124M A/B checkpoints (about 9 tokens per parameter) and MaxGPT-Ultra when it finishes. This locates his recipe on the same scale as the public models, and tells whether Ultra has the binding and update skills worth distilling.
5. **Mixing-ratio sweep at 30-60M (rank 4).** Locates the critical data density for binding and corrections at small size; about one Titan-day per run after Ultra.
6. **Adopt the battery as a Tier-0 pretraining probe** for the 150M run: plain-transcript likelihood items with same-type in-context foils, scored on base checkpoints every few billion tokens. It is cheap, it is not affected by post-training policy, and it separates "cannot bind" from "will not say".

## 7. Key uncertainties

- The decisive fine-tuning test never ran, so whether binding and update failures at ~100M body are trainable is unknown.
- Falcon-H1-Tiny-90M, the strongest chat recipe near the target size, is missing from the binding and update probe.
- No probed model has a body in the 150M target range (122M-141M); the gradient is inferred from 100-106M and 287M+ points.
- Plain-transcript likelihood is off-template for instruct models and is not generation; the base SmolLM2-135M guards the main conclusions at 135M, but perspective results in particular may shift under chat templates.
- Only two same-recipe size pairs exist (SmolLM2 135M/360M with 2T/4T tokens; LFM2 350M/2.6B at 10T); every other comparison mixes size with recipe.
- Cells are n=32 to 192 from 16-name pools; intervals are about ±0.15 at n=32.
- The knowledge ceilings come from synthetic data (2 bits per parameter; critical ratio fit on 70M-410M); applying them to Zipfian real knowledge at 150M is an extrapolation.
- The two-way knowledge test is insensitive; the Gemma 3 270M long-tail result (0.95 at 100M body) rests on n=40.
- The primacy study at 1B-2.5T uses 3+ updates on 46 keys; the transfer to 1-3 corrections on one key is by analogy, and LFM2-2.6B's success shows size can help within one recipe.
