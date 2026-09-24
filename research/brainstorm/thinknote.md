# Short thinking and memory notes for a tiny chat model

MaxGPT-Planck brainstorm track, 2026-09-23. Ideas 7 (one-line memory note) and 8 (short capped thinking, stripped from history) from `../conversation_ideas.md`.

Method: web search plus primary sources. Every arXiv paper cited with a number was downloaded as a PDF and converted with pdftotext; model cards and chat templates were fetched raw from Hugging Face. Builds on `../followup/floor.md`, `archfp.md`, `retrieval.md`, `chatdata.md`, `loop.md` and their `.verify.md` files; where a verify file corrected a claim, the corrected version is used. Nothing was loaded, trained or run: every compute figure is arithmetic (scratch scripts `calc.py` and `mde.py` in the session scratchpad, `.../scratchpad/thinknote/`). "Vendor" means a model maker's own unreproduced claim. Sizes are total parameters unless stated.

---

## Bottom line

1. **Direct answer: no.** I found no test of capped, history-stripped thinking, and no test of a running memory note, in a general (open-domain) chat model at or under 150M, measured on multi-turn recall or corrections. The nearest evidence sits in three other places:
   - **Task-oriented dialogue**, where a written belief state is a memory note: SimpleTOD and SOLOIST (GPT-2 117M-124M), UBAR (DistilGPT2 82M), MinTL (T5-small 60M).
   - **Synthetic state tracking:** Self-Notes (GPT-2 124M), filler tokens (34M), Coconut (GPT-2 124M).
   - **Single-turn pretraining:** pause tokens (130M).

   The smallest general chat model with a measured thinking vs no-thinking multi-turn number is **Qwen3-0.6B**: Multi-IF 36.1 thinking vs 33.3 non-thinking. That is vendor data, with uncapped thinking (max output 32,768 tokens), and it measures instruction following, not recall. Every reasoning-trained model under 150M either does not claim chat (MobileLLM-R1-140M) or says it lacks multi-turn support (Monad 56M).
2. **The note has the stronger small-scale evidence. The thinking has almost none for this job.**
   - Notes written at the moment a fact arrives beat end-of-input scratchpads on state tracking with updates. Self-Notes at 124M: 95.5% vs 72.2% scratchpad vs 44.6% vanilla, and 85.0 vs 11.6 vs 24.4 on longer inputs. Thinking States at 0.5B shows the same pattern.
   - In task dialogue, keeping the written state in context helps at 82M: UBAR's context holding only belief states and acts scored 101.9 combined, against 95.9 for utterances only.
   - "Previous state + current turn" matches full history (Diable 53.91 vs 53.91 JGA at 247M). Writing only the change (MinTL) is shorter and better at 60M: 6.6 tokens per turn and JGA 50.95, against 21 tokens and 44.10.
3. **Why a note should suit a tiny model:** it turns "find a fact 9 turns back" into a chain of short copies. Each copy runs from the last note, about 40-60 tokens back, to the new note. Long-range recall is the skill the floor report puts at risk below about 30M (`floor.md` item 6). This matters most at Planck's small end, and it is my inference, not a measured result.
4. **The note's known failure is error propagation.** In Diable (247M), JGA was 80.65 when given the gold previous state and 53.91 when given its own predicted state. A wrong note persists. The design below measures this directly with a gold-note condition.
5. **Thinking at small scale helps with serial composition, not with retrieval. It also loops.**
   - Short explicit CoT (about 25 tokens) took GPT-2 124M from 16.5% to 42.9% on augmented GSM8k (Coconut). Pause tokens gave nothing at 124M (16.4%).
   - Pause-pretraining at 130M gave smaller gains than at 1B (Goyal et al.).
   - Short CoT beats long CoT for Qwen2.5-0.5B (19.5 vs 14.8).
   - Falcon-H1-Tiny-R-90M falls into "a repetition trap". TII's 90M tool-calling model looped until all chain-of-thought was filtered out of its data.

   Recall and corrections are mostly single lookups. So my prior is that thinking adds little on top of a note, except on two-hop items such as "is my dog older than my cat?".
6. **Stripping old thinking is the standard, and it has a hidden training cost.**
   - Qwen3, DeepSeek-R1 and Gemma 4 all drop previous-turn thoughts. Gemma 4 keeps them only on tool-call turns, and PleIAs recommends "rolling" thinking for Baguettotron (321M).
   - For agentic tool use, keeping thinking helps by 2-5% at Qwen3-4B/8B. That is not the Planck setting.
   - The cost: training sequences must look like inference, with old thinks absent from the history. That means one training sequence per supervised turn, not one packed conversation. This roughly doubles the chat tokens when three turns per conversation are supervised.
7. **Design.** Four arms at 30M: no thinking / capped thinking / note / thinking + note.
   - All four arms train on identical teacher-rendered conversations. The skeleton program writes every note and every think, so the teacher does nothing extra and every span is correct by construction.
   - Caps are 32 think tokens (about 21 words) and 48 note tokens (about 32 words).
   - Primary metric: recall-at-distance plus corrections, 1,200 held-out items, 3 seeds.
   - An arm is adopted if it gains at least 5 points with a paired CI that excludes 0, all seeds agree, and no guardrail regresses. The cheaper arm wins ties within 3 points.
   - Cost: about 30-40 hours of RTX 5070 time with a shared-base branch design (estimate). The Mac only smoke-tests the format at 5M.
8. **Most valuable follow-up:** if the note wins at 30M, run it at 10M. A note that lets 10M pass what 30M fails would move the headline curve, which is Planck's central claim.

---

## Findings with sources and scales

### 1. What chat templates do with old thinking

| Model | What happens to old thinking | Scale | Source |
|---|---|---|---|
| Qwen3 | "the historical model output should only include the final output part and does not need to include the thinking content" | 0.6B-235B | [Qwen3-0.6B card](https://huggingface.co/Qwen/Qwen3-0.6B) |
| DeepSeek-R1 | the template runs `content.split('</think>')[-1]` on every past assistant message | 671B | [tokenizer_config.json](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/tokenizer_config.json) |
| Gemma 4 | "Thoughts from previous model turns must not be added" before the next user turn, except on tool-call turns | 12B card | [Gemma 4 12B card](https://huggingface.co/google/gemma-4-12B-it) |
| Baguettotron | "We recommend to use a 'rolling' thinking, by systematically appending thinking traces for each new generation but discarding the past one"; forcing no-think showed "a significantly decreased performance for most tasks, especially memorization of encyclopedic knowledge" (vendor, no numbers) | 321M, 80 layers, 200B SYNTH tokens | [README](https://huggingface.co/PleIAs/Baguettotron/raw/main/README.md) |
| Monad | trained natively with thinking traces; "Monad has no support yet for multi-turn" | 56.7M, 200B SYNTH tokens | [README](https://huggingface.co/PleIAs/Monad/raw/main/README.md) |

- **Keeping thinking in agentic tool use.** Retaining thinking history "consistently improves" Qwen3-4B and Qwen3-8B on BFCL multi-turn by about 2-5% ([arXiv 2606.00135](https://arxiv.org/abs/2606.00135), 4B-8B). The benefit is plan continuity across tool calls. Planck's chat turns have no such chain. **[>=1B only]**
- **Reading.** Stripping is the default for chat because it bounds the context. No public result, at any size, isolates "strip vs keep" for plain chat recall.

### 2. Pause, filler and thinking tokens (extra compute, no words)

- **Pause tokens, Goyal et al., ICLR 2024** ([arXiv 2310.02226](https://arxiv.org/abs/2310.02226); decoder-only models of 1B and 130M, 200B C4 tokens, pauses inserted at 10% of positions during pause-pretraining).
  - At 1B, pause-pretraining plus pause-finetuning beat standard training on 8 of 9 tasks. SQuAD EM went from 36.4 to 51.7 (10 pauses) and 55.9 (50 pauses). CoQA F1, a conversational QA task, went from 29.9 ±1.0 to 31.6 ±0.5: small.
  - At 130M: gains on 6 of 9 tasks, and "we do not observe gains on SQuAD, in contrast to the gains observed in 1B model".
  - The authors: "a preliminary comparison between our two model sizes surprisingly suggests the opposite" of smaller models benefiting more. Pause-finetuning a standard-pretrained model gave "mixed results". The pauses have to be in pretraining.
- **Filler tokens, Pfau et al. 2024** ([arXiv 2404.15758](https://arxiv.org/abs/2404.15758); 34M Llama, 4 layers, 384 wide, synthetic 3SUM).
  - Without filler the model sits "near-random accuracy at 66%". With filler it reaches 100%.
  - This only happens when training shows parallelizable CoT. There was "no transfer" from serial, instance-adaptive CoT demonstrations to filler tokens.
- **Coconut, Hao et al. 2024** ([arXiv 2412.06769](https://arxiv.org/abs/2412.06769); GPT-2 124M). On GSM8k with Deng et al.'s augmented training data:
  - no CoT 16.5% (2.2 tokens); pause tokens 16.4%; explicit CoT 42.9% (25.0 tokens on average); continuous thoughts 34.1%.
  - On ProntoQA, pause tokens scored 77.7 ±21.0 against 93.8 for no CoT: worse, and unstable.
- **Thinking tokens.** Herel and Mikolov 2024 ([arXiv 2405.08644](https://arxiv.org/abs/2405.08644)) is an RNN-LM proof of concept (per-sentence perplexity examples). The follow-up ([arXiv 2411.11371](https://arxiv.org/abs/2411.11371); GPT-2-based and Llama 3.2 1B) finds that they "marginally" improve and "consistently" underperform CoT.
- **Reading for Planck.** Silent extra tokens have no demonstrated benefit at 124M-130M outside synthetic parallel tasks. If Planck thinks, it should think in words.

### 3. Short explicit thinking at small scale: when it helps

- **GPT-2 124M, math.** A roughly 25-token equation-style CoT raises augmented GSM8k from 16.5% to 42.9% (Coconut, above). Short CoT does help at this size when the task needs serial steps.
- **Self-Notes vs scratchpad, Lanchantin et al. 2023** ([arXiv 2305.00833](https://arxiv.org/abs/2305.00833); GPT-2 base 124M fine-tuned). Self-Notes are written inside the input as facts arrive; a scratchpad is written after the question.

  | Task | Vanilla | Scratchpad | Self-Notes |
  |---|---|---|---|
  | Algorithmic (variable updates), 2-100 statements | 44.6 | 72.2 | 95.5 |
  | Algorithmic, 101-200 statements (out of distribution) | 24.4 | 11.6 | 85.0 |
  | Toy-Story, 4-hop | 37.4 | 94.2 | 97.8 |

  - The scratchpad failed on longer inputs because it ran past GPT-2's 1,024-token context.
  - Semi-supervised: 1% of Toy-Story training examples carrying notes (100 samples) already beat vanilla.
  - This is the closest small-scale analogue of Planck's per-turn note: state written at reading time beats state reconstructed at answer time.
- **Thinking States, Amos et al., Feb 2026** ([arXiv 2602.08332](https://arxiv.org/abs/2602.08332); Qwen2.5-Base-0.5B; thoughts are generated every few input tokens and fed back).
  - Variable tracking, trained on up to 10 updates and tested on 10-100: no CoT 2.15, CoT 6.78, Thinking States 33.76.
  - Trained on up to 40 updates: 2.19, 87.75, 97.71.
  - This is the same lesson as Self-Notes, at 0.5B.
- **Rationale as a training target, not an inference step.**
  - Distilling step-by-step ([arXiv 2305.02301](https://arxiv.org/abs/2305.02301); T5-Base 220M) compared rationale-then-label as one output ("single-task") with rationale as a separate auxiliary target ("multi-task"):

    | Dataset | Standard finetuning | Single-task (rationale then label) | Multi-task (rationale as auxiliary target) |
    |---|---|---|---|
    | ANLI | 43.58 | 43.50 | 49.58 |
    | CQA | 62.19 | 61.37 | 63.29 |

    Across all four datasets (also e-SNLI 88.38 / 88.88 / 89.51 and SVAMP 62.63 / 63.00 / 65.50), think-then-answer stayed within 1 point of no thinking, while training on the rationale without generating it was best everywhere.
  - CoTE for dialogue state tracking ([arXiv 2403.04656](https://arxiv.org/abs/2403.04656); T5-base 220M) got MultiWOZ 2.2 JGA 57.5 against 56.4 for the same code base without explanations. It was "better performance with the format of slot value followed by explanation": the answer comes first.
  - Its "coarse" explanation is just the relevant earlier utterances copied out. The gains concentrate on turns that need 2-3 steps.
- **Symbolic CoT distillation** ([arXiv 2306.14050](https://arxiv.org/abs/2306.14050); OPT 125M, 350M, 1.3B). Training on GPT-3 rationales beats label-only training on CommonsenseQA at all three sizes. The 125M numbers are only in a figure.
- **Long vs short.**
  - Qwen2.5-0.5B fine-tuned on math scores 14.8 with long CoT and 19.5 with short CoT, and 16.9 with a large teacher vs 20.4 with a small one ([arXiv 2502.12143](https://arxiv.org/abs/2502.12143), ACL Findings 2025, 0.5B-32B).
  - Accuracy follows an inverted U in CoT length for a 6-layer GPT-2 on arithmetic. Among real models, the optimal step count is 14 at 1.5B and 4 at 72B ([arXiv 2502.07266](https://arxiv.org/abs/2502.07266)).
  - For Planck: short, and only as many steps as the task needs.
- **Thinking can hurt instruction following.** Prompted CoT dropped IFEval from 49.0 to 40.7 for Llama-3.2-1B-Instruct and from 35.9 to 31.6 for Qwen2.5-1.5B-Instruct (temperature 0). Self- or classifier-selected reasoning recovered part of the loss ([arXiv 2505.11423](https://arxiv.org/abs/2505.11423); 1B-70B). **[>=1B only]**
  - With trained thinking the sign flips at 0.6B (vendor, same weights): Qwen3-0.6B thinking vs non-thinking scores IFEval 59.2 vs 54.5, Multi-IF 36.1 vs 33.3, Arena-Hard 8.5 vs 6.5.
  - Qwen3-1.7B scores Multi-IF 51.2 vs 44.7 ([Qwen3 report, Tables 19-20](https://arxiv.org/abs/2505.09388)). Thinking was effectively uncapped: "we set the max output length to 32,768 tokens".
- **Frontier multi-turn.** Hy-MultiTurn ([arXiv 2607.29196](https://arxiv.org/abs/2607.29196), July 2026) found thinking helps all five toggleable families, by 7.1-14.0 points. The gains are larger on "memory, execution, and localization" and smallest on action suppression. **[>=1B only]**
  - The same paper found self-generated history causes "state drift": in Mode III, which "tracks the latest valid values and their dependencies" (the corrections skill), the mean importance score was 61.3 when models ran on their own replies vs 75.4 on a fixed history written by GPT-5.5 (17 models).
  - The o3 and DeepSeek-R1 reasoning models "deteriorate in similar ways" to non-reasoning models in Lost in Conversation, and their replies run 33% longer ([arXiv 2505.06120](https://arxiv.org/abs/2505.06120); 15 LLMs, smallest Llama-3.1-8B). **[>=1B only]**
- **Mechanism, parametric facts.** A GPT-2 (12 layers, 768 wide) could not classify or compare its own stored facts unless CoT was used in both training and inference ([arXiv 2309.14402](https://arxiv.org/abs/2309.14402), Physics of LMs 3.2). This concerns weight-stored facts. I have not found the in-context analogue ("is the dog older than the cat?" from two user turns) measured at small size.

### 4. Loops, caps and budget forcing

- **Falcon-H1-Tiny-R** ([tech blog PDF](https://tiiuae-tiny-h1-blogpost.hf.space/falcon-h1-tiny-a-series-of-extremely-small-yet-powerful-language-models-redefining-capabilities-at-small-scale.pdf), Jan 2026). Pretrained only on reasoning data; the series' WSD schedule totals 900 GTok.
  - The 90M model: AIME24 5.0, Math500 39.7. It "still falls below our expected performance" and is "more prone to a repetition trap"; a repetition penalty "can mitigate this to some extent".
  - At 90M, "reasoning pretraining yields stronger results" than pretraining followed by reasoning SFT.
- **Falcon-H1-Tiny tool-calling 90M (same PDF).** Mixing chain-of-thought traces into tool-calling data produced "infinite generation loops". TII's diagnosis: "For our 90M model, they created an overly complex learning target". Filtering out all reasoning content fixed it: "The improvement was immediate". This is the most direct warning at Planck's size.
- **Why distilled small models loop** ([arXiv 2512.12895](https://arxiv.org/abs/2512.12895); 1.5B-32B plus small synthetic transformers).
  - At low temperature, Qwen-1.5B loops 76% of the time vs 37% for Qwen-32B. OpenThinker3-1.5B loops 30% under greedy decoding vs 4% for its QwQ-32B teacher. Hard problems loop more (51% vs 13%).
  - Cause: a "hard-to-learn" progress step loses probability to an easy cyclic one. Temperature reduces looping but does not fix the cause.
- **MobileLLM-R1-140M** ([arXiv 2509.24945](https://arxiv.org/abs/2509.24945); [card](https://huggingface.co/facebook/MobileLLM-R1-140M)). "These models are not general-purpose chat models"; the card's example uses `max_new_tokens=8192`.
  - After reasoning SFT, 0-shot: MATH500 6.2, GSM8K 4.1. Its base model scores 16.3 on GSM8K 8-shot, a different protocol.
  - SmolLM2-360M-Instruct gets 8.1 and Gemma-3-270M-it 8.4 on GSM8K 0-shot without long thinking.
  - Long reasoning at 140M bought little.
- **Caps.**
  - s1 ([arXiv 2501.19393](https://arxiv.org/abs/2501.19393); 32B, 1,000 examples) enforces a budget by appending the end-of-thinking delimiter, and extends thinking by appending "Wait". **[>=1B only]**
  - Truncating a model trained on long thinking is dangerous. A Qwen3-4B-Thinking model capped at 512 response tokens collapsed to MATH500 0.050, and recovered at 1,024 ([arXiv 2606.12941](https://arxiv.org/abs/2606.12941), 1.5B/4B).
  - For Planck: the cap must live in the training data (every think short), with the harness cap only a backstop.

### 5. Belief-state and dialogue-state writing: memory notes that already exist

| System | Size | Note format and where it lives | Key number | Source |
|---|---|---|---|---|
| SimpleTOD | GPT-2 124M; DistilGPT2 82M | belief state `(domain, slot, value)` regenerated each turn from the full utterance history; not kept in later turns | MultiWOZ 2.1 JGA 55.76 (124M), 54.54 (82M); role and end tokens alone take 16.79 to 55.76 | [arXiv 2005.00796](https://arxiv.org/abs/2005.00796), numbers per `floor.verify.md` |
| SOLOIST | "public 117M-parameter GPT-2" init | belief state, DB state, then response, one causal LM | end-to-end; no DST-only number used here | [arXiv 2005.05298](https://arxiv.org/abs/2005.05298) |
| UBAR | DistilGPT2 82M | session level: generated belief states and acts **kept in context** across turns | DST JGA 56.20 vs SimpleTOD 55.72 (2.1). Combined score: full context 105.1; previous turn only 101.6; belief states + acts only 101.9 vs utterances + responses only 95.9 | [arXiv 2012.03539](https://arxiv.org/abs/2012.03539) |
| MinTL | T5-small 60M | only the **edit** to the previous state (Levenshtein belief span); context is the last 3 turns + previous state | JGA 50.95 (2.1) generating 6.58 tokens/turn, vs Sequicity (T5-small, full span) 44.10 with 20.99 tokens | [arXiv 2009.12005](https://arxiv.org/abs/2009.12005) |
| Diable | T5v1.1-base 247M | state as a table updated by INSERT/DELETE; context = previous state + current turn | 53.91 (2.1), same as a full-history baseline (53.91); tabular vs cumulative: cumulative is 3-5% worse; **gold previous state 80.65 vs predicted 53.91** | [arXiv 2305.17020](https://arxiv.org/abs/2305.17020) |
| T5-small DST | 60M vs 220M | schema-prompted state | 56.12 vs 56.66 JGA | [arXiv 2109.07506](https://arxiv.org/abs/2109.07506), via `retrieval.md` |

- **Two reusable facts from Diable.**
  - "Using only the previous state barely changes the JGA of lightCumulative but benefits Diable"; the note is "less noise than the full history", "especially true in conversations for which the value of a slot is changed". That is the correction case.
  - Row order was randomized in training "to avoid overfitting to specific positional biases".
- **The caveat carried from the fact-check.** MultiWOZ DST is mostly slot accumulation with gold supervision, not corrections (`floor.verify.md` misapplication 4). This evidence says tiny models can write and carry a note. It does not say they handle corrections.

### 6. Memory notes and running summaries in chat, 2025-2026

- **Rolling memory.** A rolling memory "capped at Lm = 256 tokens", trained with RL on sharded GSM8K, beat full-history training by 15.2-35.6 points ([arXiv 2606.12941](https://arxiv.org/abs/2606.12941); Qwen2.5-Math-1.5B, Qwen3-4B-Thinking). The gain persisted even when the full history was given at test time. **[>=1B only]**
- **MEM1** ([arXiv 2506.15841](https://arxiv.org/abs/2506.15841); Qwen2.5-7B). "At each turn, the agent's context is pruned to retain only the most recent internal state." This is structurally thinking + note with old context dropped, for agents. **[>=1B only]**
- **Self-Recall Thinking** ([arXiv 2605.15102](https://arxiv.org/abs/2605.15102), May 2026; Qwen2.5-7B). The think step quotes the relevant earlier turns before answering ("Recall History: A5: ..."). It gains 1.1-4.7 F1 across three QA benchmarks, including CoQA 84.0. **[>=1B only]**
- **Keep Me Updated!** ([arXiv 2210.08750](https://arxiv.org/abs/2210.08750)). Open-domain multi-session memory with PASS / REPLACE / APPEND / DELETE operations on text memories. The memory operator is an 822M T5 and the generator is a 6.9B HyperCLOVA. **[>=0.5B only]**
- **Small memory writers.**
  - MemReader-0.6B ([arXiv 2604.07877](https://arxiv.org/abs/2604.07877), via `retrieval.md`) matches a GPT-4o-mini extractor on LoCoMo (79.56 vs 78.70).
  - LightMem ([arXiv 2604.07798](https://arxiv.org/abs/2604.07798)) runs 1B-1.5B memory models. Its DialSim F1 is around 4, which is low.
  - No memory writer under 0.6B was found.
- **Snowball (user-side running recap).** Repeating all earlier shards every turn recovers 15-20% of the full-to-sharded drop for GPT-4o and GPT-4o-mini ([arXiv 2505.06120](https://arxiv.org/abs/2505.06120)). **[>=1B only]**
- **Verbatim vs extracted.** Verbatim chunks beat extracted facts by 16-22 points on long-conversation memory ([arXiv 2601.00821](https://arxiv.org/abs/2601.00821), via `retrieval.md`). In Planck the note sits beside the raw history, not in place of it, so this does not argue against it.

### 7. Connections to Planck's own findings

- Corrections have not been shown in any general chat model up to 0.6B (`floor.md` item 2, probe hint). One suggested cause is a softmax+RoPE head attending to the first matching key, not the latest (`archfp.md` item 4).
  - A note puts the latest value in a fixed, recent place. The model then no longer has to pick the latest of two matching keys; it only has to update the note on the correction turn. That update is itself a hard step, and it is exactly where the design below measures note accuracy.
- `retrieval.md` recommends "no chain-of-thought or source analysis reasoning traces in grounded data (Falcon 90M loops)". The capped, program-written think below is a different object: at most 32 tokens, templated, empty on most turns. It is tested as an arm, not adopted as a default.

---

## Has it been tested at this size?

**No, not in the form asked.** Nothing I found trains or evaluates a 10M-150M general chat model with (a) a capped think that is stripped on the next turn, or (b) a running note kept in context, on multi-turn recall or corrections. The nearest neighbours, and what each lacks:

| Nearest evidence | Size | What it has | What it lacks |
|---|---|---|---|
| UBAR, SimpleTOD, SOLOIST, MinTL, Diable | 60M-247M | a written, carried state in a causal or seq2seq LM | open domain; natural-language notes; correction-focused tests; chat |
| Self-Notes | 124M | notes written as facts arrive; updates; long inputs | dialogue; chat format |
| Coconut, filler tokens, pause tokens | 34M-130M | extra compute before answering, controlled | multi-turn; stripping; chat |
| Monad, Baguettotron | 56.7M, 321M | chat template with thinking; "rolling" thinking at 321M | any published multi-turn measurement; Monad has no multi-turn |
| Falcon-H1-Tiny-R-90M, MobileLLM-R1-140M | 91M, 140M | reasoning-trained tiny models | chat; loops at 90M; 140M "not general-purpose chat" |
| Qwen3-0.6B thinking vs non-thinking | 0.6B | stripped thinking, multi-turn IF number | 4x too large, uncapped thinking, vendor, not recall or corrections |

---

## Experiment design: four arms at 30M

### Fixed across arms

- **Model.** About 30M total, tied 8k BPE, depth over width. Candidate shape: d=384, 15 layers, SwiGLU 8/3, full multi-head attention: 26.5M body + 3.15M embedding = 29.7M, 10.6% embedding (my arithmetic, `calc.py`).
  - Four special tokens are added: `<think>`, `</think>`, `<note>`, `</note>`. All arms get them, so vocabulary size is identical.
- **Data.**
  - One skeleton-render-verify conversation set (`chatdata.md` section 3): 200-250 tokens per conversation, user turns at most about 25 words, assistant turns 5-30 words, 2-4 skill events per conversation.
  - Events cover recall at distance 1-9, correction, abstain, reference, instruction set/override, and persona. **Add one event type:** two-fact composition ("is my dog older than my cat?", "how many days until my trip if today is Monday?"). This is where thinking has its best theoretical case (section 3).
  - All four arms use **the exact same rendered user and assistant text**. Only the inserted spans differ, so the comparison is paired at the item level.
- **Who writes the spans: the program, not the teacher.** The skeleton program planted every fact, knows every event and verified every answer (`chatdata.md` 3.6), so it can write the gold note state and the gold think deterministically. The teacher (Gemma 4 12B, `conversation_ideas.md` #12) produces only what the baseline already needs. The thinking and note arms therefore cost zero extra teacher time, and every span is correct by construction. A teacher-written free-form think is a separate ledger idea (TN6).
- **Training schedule (branch design, to save compute).**
  - Stage 1, shared per seed: 1.1B tokens of Planck's general-text mix with no chat, stable learning rate.
  - Stage 2, per arm: 0.4B tokens, 50% chat in that arm's format plus 50% general text, learning-rate decay. Falcon-R saw most gains arrive in the decay.
  - 3 seeds give 3 bases and 12 branches, 5.7 full-run equivalents instead of 12.
  - A neutral stage 1 matters: putting baseline-format chat in the shared base would pre-train one arm's format for longer.
- **Sequences mirror inference (per-turn cut sequences).**
  - Each conversation yields one training sequence per supervised turn: its 2-4 event turns plus the final turn. The history is rendered exactly as the harness will render it (old thinks gone; old notes gone except the latest), and the sequence ends after the supervised assistant turn.
  - Loss falls only on the supervised assistant turn (note, think, answer, end token); history and user tokens are masked. Identical cut points and masking in all four arms.
  - Cost: about 2x the chat tokens of one packed conversation (3 cuts of a 240-token chat = 480 tokens, `calc.py`).
  - The alternative is packing with attention masks that hide stripped spans from later turns. It is cheaper but not exact: later turns would see answer representations computed while the think was visible, which a re-prefilling harness never does. It is left for TN12.
- **Compute (estimates).**
  - FLOPs per token at 30M, T=1,024: 6N + 12LdT = 0.255 GFLOP.
  - Stage 1: 2.8e17 FLOP each. Stage 2: 1.0e17 each, plus 10-35% more chat tokens in the span arms.
  - On the RTX 5070 at an assumed 15-20 effective TFLOPS (`loop.md`: about 10x the Mac, itself an estimate): about 12-16 h for three bases plus 17-23 h for twelve branches, **about 30-40 GPU-hours**.
  - On the Mac (1.5-2.1 TFLOPS measured in the first run) the same plan would take roughly 280-380 hours, so the Mac only runs a 5M format smoke test (tokenizer, masks, harness stripping, graders). It does not decide anything.

### The four arms: exact formats

Placeholders `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>` stand for Planck's real role and end-of-turn tokens (present from step 0, per `floor.md`). Running example, a skeleton with events recall (dog name), correction (Biscuit to Pickles), and a later question.

History as rendered at turn 4, the supervised turn, in the **no-thinking arm (B)**:

```
<|system|>You are Wren. Keep replies short.<|end|>
<|user|>My dog is called Biscuit and he is 3.<|end|>
<|assistant|>Biscuit sounds sweet! What kind of dog is he?<|end|>
<|user|>A beagle. Oh wait, actually his name is Pickles. Biscuit was my old dog.<|end|>
<|assistant|>Got it, Pickles the beagle. Does he like walks?<|end|>
<|user|>He does. We're going to the lake on Tuesday.<|end|>
<|assistant|>A lake trip sounds great for a beagle.<|end|>
<|user|>What's my dog's name again?<|end|>
<|assistant|>Your dog is Pickles.<|end|>          <- loss only here
```

**Capped thinking arm (T).** Same history; no think anywhere in the history. The supervised turn is:

```
<|assistant|><think>asked: dog's name. first Biscuit, corrected to Pickles. use Pickles.</think>Your dog is Pickles.<|end|>
```

- Think templates are chosen by the program from the skeleton event, with 3-5 paraphrases each and filled with the gold values:

  | Event | Think template |
  |---|---|
  | Recall | `asked: {key}. user said {value}.` |
  | Correction | `asked: {key}. first {old}, corrected to {new}. use {new}.` |
  | Abstain | `asked: {key}. not told.` |
  | Reference | `"{referring expression}" = {referent}.` |
  | Instruction persistence (every later turn) | `rule: {rule}.` |
  | Override | `rule now: {new} (not {old}).` |
  | Persona | `I am {name}; my {attr} is {value}.` |
  | Composition | `dog 3, cat 5. 3 < 5. no.` |
  | Social or no-dependency turn | `<think></think>` (empty) |

- **Cap.** Training thinks are at most 24 tokens (typically 8-20). At inference the harness forces `</think>` at 32 tokens, about 123 bytes or about 21 words at 3.83 bytes/token (`calc.py`; bytes/token from the first run's in-domain 8k BPE on OASST). A forced close counts as a loop event.

**Note arm (N).** Every assistant turn starts with the note. The rendered history keeps **only the latest note**, which sits in the previous assistant turn right before the current user message; all older notes are stripped. At turn 4:

```
...
<|user|>He does. We're going to the lake on Tuesday.<|end|>
<|assistant|><note>dog: Pickles (was Biscuit); dog age: 3; dog breed: beagle; lake trip: Tuesday</note>A lake trip sounds great for a beagle.<|end|>
<|user|>What's my dog's name again?<|end|>
<|assistant|><note>dog: Pickles (was Biscuit); dog age: 3; dog breed: beagle; lake trip: Tuesday</note>Your dog is Pickles.<|end|>
```

- **Note grammar** (one line, program-written):
  - `key: value` slots joined by `; `.
  - Keys: 1-3 lowercase words, one canonical key per slot type across the whole dataset (low entropy helps learnability, `chatdata.md` item 2).
  - Values: copied verbatim from the user, nonce names included.
- **Update rules:**
  - A new fact appends a slot at the end; slot order stays stable.
  - A correction replaces the value and records only the immediately previous one: `dog: Pickles (was Biscuit)`. A second correction overwrites the `(was ...)` part.
  - "Forget X" deletes the slot.
  - Standing instructions get `rule:` slots, replaced on override. This makes the note serve F4/F5 as well.
  - The assistant's own persona stays in the system prompt, never in the note.
  - A turn with no new fact copies the note unchanged. The first turn with no facts writes `<note></note>`.
- **Cap and harness guard.**
  - At most 8 slots and 48 tokens (about 32 words); in gold data, the program drops the slot least recently mentioned.
  - The harness forces `</note>` at 48 tokens.
  - If the note fails the grammar regex, the harness keeps the previous note. The fallback rate is reported as a metric.
- **Abstention becomes a lookup.** A key missing from the note means "you didn't tell me". The user's facts live only in the note and the raw history.

**Thinking + note arm (TN).** Note first, then think, then answer. Old thinks are stripped; the latest note is kept. The supervised turn is:

```
<|assistant|><note>dog: Pickles (was Biscuit); dog age: 3; dog breed: beagle; lake trip: Tuesday</note><think>asked: dog's name. note: Pickles.</think>Your dog is Pickles.<|end|>
```

- The TN think refers to the note instead of quoting history. For composition items it does the step: `note: dog age 3, cat age 5. 3 < 5. no.`

**What the teacher must produce, per arm:** the same thing in all four, namely the natural user and assistant lines for each skeleton, rendered under the existing script-mode rules (short turns, no emoji or bold, no em dashes, no invented self-facts). Notes, thinks and cut sequences are all built by the program afterwards.

### Metrics

- **Primary: P = mean generative pass rate over F1 (recall at distance 1/3/6/9) and F2 (corrections, updates and abstain).** MTB-150 families (`../lanes/eval.md` D2) with mutation-tested graders.
  - For this experiment, F1 and F2 are enlarged to 600 held-out items each, from skeletons with unseen slot values (nonce and real names).
  - Self-generated history: the model's own earlier replies, its own notes, and its thinks stripped exactly as the harness does.
  - Greedy decoding. Paired items across arms.
  - Detectable effect: about 4-5 points for P with 1,200 items and 3 seeds, 4.5-6 for a 600-item family alone. This is my arithmetic assuming item-level arm correlation 0.3 and seed SD 1-2 points (`mde.py`); both inputs are unmeasured.
- **Secondary.**
  - F3 reference, F4 instruction persistence (IFEval verifier on every later turn plus the on-topic gate), F5 override, F7 persona, and the new composition family (300 items).
  - Tier 0 likelihood margin at the answer position: log p(correct) minus log p(strongest in-context foil, including the stale value). In the T/TN arms it is taken after the model's own greedy think; in the N/TN arms, after the model's own notes. This is the continuous metric `loop.md` wants for small effects.
- **Guardrails** (an arm fails if any is breached):
  - F8 social basics down by more than 2 points.
  - Answer-token bits-per-byte on held-out plain chat up by more than 1%.
  - Loop rate above 2% of turns: think or note hits its cap without closing, or the reply's repeated-4-gram fraction passes the Daimax-style threshold.
  - Span leak rate above 1%: note or think syntax spoken in the reply.
  - Median reply length outside the baseline's interquartile range.
- **Diagnostics** (they explain a result; they do not decide):
  - Joint note accuracy: the model's note equals the program's gold state, all slots, per turn (the JGA analogue); plus per-slot F1.
  - Note-answer agreement: separates writing errors from reading errors.
  - **Gold-note injection:** replace the model's notes with gold ones, which measures error propagation (Diable: 80.65 vs 53.91).
  - **Note deletion at eval:** does the N-trained model still recall from raw history?
  - **Forced-empty think at eval:** does T-trained training help even without thinking at inference? This is the Distilling step-by-step and PausePT-StdFT question.
  - Accuracy vs distance curves: a flat curve is the note's signature.
  - Extra tokens generated per turn.
  - Temperature 0.7 x 3 samples robustness pass: small models loop more under greedy decoding (`arXiv 2512.12895`).

### Pre-registered hypotheses

1. N beats B on P, mostly through F2 corrections and a flatter F1 distance curve.
2. T beats B by less than N does, with gains concentrated on composition and correction items; T has the highest loop rate.
3. TN beats N by less than 3 points on P, but beats N on composition.
4. The forced-empty-think T model keeps part of T's gain.
5. Gold-note injection shows a gap of at least 10 points over self-written notes, meaning error propagation is the note's main limit.

### Decision rule

1. **Validity gate.** Format compliance of at least 98% (spans open and close, grammar parses) in every arm, and no diverged runs. Otherwise fix the bug and rerun; nothing is interpreted.
2. **An arm beats B** if its 3-seed mean P is at least 5 points higher, the paired bootstrap 95% CI over items and seeds excludes 0, all 3 seeds show a gain, and no guardrail is breached.
3. **Between winning arms, the cheaper one wins** unless the costlier arm leads by at least 3 points on P with a CI excluding 0. Cost is measured as extra generated tokens per turn and extra training tokens.
4. **Thinking is rejected at 30M** if T or TN breaches the loop or F8 guardrail, even with a higher P. Record it as "tested, harmful at 30M" in the ledger.
5. **Interaction.** If TN minus N is below 3 points on P, thinking leaves the recipe, and TN minus N on the composition family is reported separately as the case for a later "think only when composing" variant (TN13). If TN minus T is below 3 points, the note leaves.
6. **Null.** If no arm clears +5, B stays. Any arm with a Tier 0 margin gain whose CI excludes 0 goes to 60M, where the model may be able to use the extra compute (Goyal's size trend).
7. **Next step either way.** The winning arm and B are rerun at 10M and 60M, 2 seeds each, for the headline curve.

---

## Ledger ideas

| # | Idea | Why | Cheapest test | Novelty | Win condition |
|---|---|---|---|---|---|
| TN1 | The four-arm experiment above (B / T / N / TN) at 30M | No capped-stripped think or running note has been tested in sub-150M chat | Section above: about 30-40 h on the 5070 | untested | an arm clears the decision rule |
| TN2 | **Note shifts the size floor:** N at 10M vs B at 30M | Notes turn long-range recall into short copies; induction is the weak skill at the small end | After TN1: B and N at 10M, 2 seeds | untested | N-10M matches or beats B-30M on P |
| TN3 | Delta note + harness-held state: the model writes only `set dog = Pickles` / `del trip`; the program applies it and renders the table | MinTL (60M): 6.6 vs 21 tokens per turn, higher JGA; Diable: tables beat cumulative lists by 3-5%; no copy errors in carried slots | Fifth arm in TN1's branch design, 2 seeds | tweak-of-known | beats cumulative N on P at fewer tokens |
| TN4 | Think-trained, served without thinking (think as auxiliary target only) | Distilling step-by-step: multi-task rationale 49.58 vs think-then-answer 43.50 on ANLI at 220M; CoTE: answer first is better | Already inside TN1 as the forced-empty-think diagnostic; promote it to an arm if it holds | known-apply | at least 80% of T's gain at zero inference tokens |
| TN5 | Pause/filler control: same token count as T's thinks, filled with `<pause>` | Separates compute from content; Coconut: pause 16.4 vs no-CoT 16.5 at 124M | One extra branch, 2 seeds | known-apply | expected null; a gain would mean content does not matter |
| TN6 | Teacher-written free thinking (Gemma 4 12B thoughts compressed to at most 20 words) vs program templates | Natural thinking may generalize to messy input; templates are statistically simpler | 5k skeletons rendered with teacher thinks; swap into T's branch | untested | beats templated T on held-out paraphrased questions without more loops |
| TN7 | Cap sweep: 0 / 12 / 24 / 48 think tokens (training data cut to match) | Inverted U in CoT length (6-layer GPT-2); short beats long at 0.5B | T arm at 10M, 4 caps, 1 seed, likelihood margins | tweak-of-known | finds the knee; expect 12-24 |
| TN8 | Keep the last think in history (not strip) | +2-5% retention effect at 4B-8B in multi-turn tool calling; untested in chat or at small size | T arm variant, 2 seeds | untested at this scale | beats stripped T without breaching loop or length guardrails |
| TN9 | Note-noise training: corrupt 10% of the previous notes in cut sequences, target the corrected note | Diable's error propagation (80.65 gold vs 53.91 predicted); scheduled-sampling logic | N variant, 2 seeds | tweak-of-known | closes at least a third of the gold-note gap |
| TN10 | Drop `(was X)` from corrected slots | The stale value in the note may itself act as a foil | N variant; F2 stale-answer rate | untested | stale answers fall with no loss on "what did I say first?" items |
| TN11 | Note placement: in the last assistant turn (default) vs hoisted after the system prompt vs a `<|memory|>` role message right before the current user turn | Attention to early tokens decays (`retrieval.md`); the default gives the shortest copy distance | 3 placements at 10M, likelihood margins | untested | pick by F1 at distance 9 |
| TN12 | KV-retaining strip: keep the answer tokens' KV cache that saw the think; drop only the think's own KV | Makes cheap packed training exact; lets thinking influence later turns implicitly | Harness change plus packed-mask training at 10M, vs TN1's re-prefill T | untested | at least equal P to cut-sequence T at half the chat tokens |
| TN13 | Think only on composition turns (empty elsewhere) vs think on every dependency turn | Selective reasoning recovered IFEval losses at 1B+; thinking's theoretical case is composition | T variant | known at 1B+, untested tiny | same P, fewer loops and tokens |
| TN14 | Split writer: a 5M note-writer model + a 25M chat model vs one 30M model writing its own notes | MemReader-0.6B shows a small dedicated extractor can match a big one; division of labor under a fixed total | Train the 5M writer on (turn, previous note) to new note; compare on P and joint note accuracy | untested | split beats single at equal total parameters |
| TN15 | Self-recall think (quote the user's earlier line verbatim) vs abstract `key: value` think | SRT at 7B, CoTE's coarse copied explanations at 220M | T variant | tweak-of-known | better F1 at distance 6-9 |

---

## Open questions

1. **Training mirror cost.** Per-turn cut sequences are exact but double the chat tokens. Masked packing is cheap but lets history representations carry the stripped thinking. Does that mismatch matter at 30M? TN12 answers it.
2. **Note error propagation in live chat.** When the model writes a wrong note and the user then says "no, I said Pickles", the fix must come from the user turn, not the note. Training data needs "assistant errs from its note, user corrects" conversations (`chatdata.md` L11 applies).
3. **Coverage.** The program-written note only captures skeleton slot types. Real users say incidental things ("I'm tired today") that no slot holds. Does a note-trained model then ignore facts outside its slot vocabulary? The note-deletion diagnostic tests this in part.
4. **Budget with the lookup block.** `retrieval.md` places a BM25 lookup next to the current user turn. Note (up to 48) + think (up to 32) + lookup (about 3 chunks) + history must fit 2,048 tokens. Is the note a better use of those tokens than more history?
5. **Decoding.** Greedy decoding loops more in small distilled models. Should the decision use greedy (deterministic graders) or sampling (closer to product use)? The design uses greedy and reports temperature 0.7 as a robustness check.
6. **Does the note sidestep or relocate the correction failure?** If the model still fails to update the note on the correction turn, the first-match attention problem (`archfp.md`) has only moved into note writing. Per-turn note accuracy on correction turns answers this.
7. **Self-generated history.** UBAR found generated context better than ground truth in its setting. Hy-MultiTurn found self-generated history causes drift. Which one holds for a 30M model's own notes is unknown, so evaluation uses self-generated history throughout and reports gold-note and gold-history variants as bounds.
8. **Does any of this transfer from templated thinks to real conversations** with typos and out-of-list words (`chatdata.md` open question 2)?

---

Sources read as PDFs for this report: 2310.02226, 2404.15758, 2412.06769, 2305.00833, 2602.08332, 2305.02301, 2403.04656, 2306.14050, 2502.12143, 2502.07266, 2505.11423, 2505.09388, 2607.29196, 2505.06120, 2309.14402, 2512.12895, 2509.24945, 2501.19393, 2606.12941, 2606.00135, 2506.15841, 2605.15102, 2210.08750, 2604.07798, 2005.00796, 2005.05298, 2012.03539, 2009.12005, 2305.17020, 2109.07506, 2405.08644, 2411.11371, and the Falcon-H1-Tiny blog PDF. Model cards fetched: Qwen3-0.6B, Gemma 4 12B, DeepSeek-R1 (tokenizer_config.json), Baguettotron, Monad, MobileLLM-R1-140M, Falcon-H1-Tiny-R-90M.
