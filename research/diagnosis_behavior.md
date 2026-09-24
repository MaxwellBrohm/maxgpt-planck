# Diagnosis, observed-behavior lens: which conversational failures appear first as chat models shrink, what each one means, and what a ~150M model has to fix

Date: 2026-09-23. One of three independent diagnoses (the others are `diagnosis_recipe.md` and `diagnosis_capacity.md`). Lens: start from what small chat models actually do wrong in conversation, classify each failure by mechanism (knowledge miss, context-tracking miss, format or role miss, degeneration), and rank the limits by how likely each is to stop a ~150M model built on Max's budget from holding a real multi-turn conversation.

Evidence rules. Lane reports are used in their fact-checked form (`lanes/*.verify.md`); refuted claims are not used. The firsthand probe (`lanes/probe.md`) is used only as corrected by its audit (`lanes/probe.verify.md`). Every external number has a URL and the scale it was measured at; **[>=0.5B only]** and **[>=1B only]** flag evidence that does not reach Max's size. "My analysis" means arithmetic I did on saved result files in this session. No ML model was loaded, run, fine-tuned or benchmarked for this report: the only code I ran reads saved JSONL files (`scratchpad/behav/an.py`, `an2.py`, `an3.py` in the session scratchpad; they import `capacity_probe/items.py`, which uses only Python's `random`, to regenerate item metadata).

---

## 0. Status of this resumed step (what was on disk, what I trusted)

- `diagnosis_behavior.md` did not exist when this step resumed. The interrupted attempt left no partial draft, so this file is written fresh.
- **Capacity-lens likelihood probe** (`capacity_probe/`, plain `User:/Assistant:` transcripts, summed log-probability of in-context candidates, 32 scenarios per cell). What is complete and used here:
  - Main battery (`out/<model>.jsonl`): complete for 8 models, 1,116 items each: SmolLM2-135M base, SmolLM2-135M-Instruct, LFM2-350M, LFM2.5-350M, SmolLM2-360M-Instruct, Qwen2.5-0.5B-Instruct, Qwen3-0.6B, LFM2-2.6B.
  - `uprobe` controls for state updates: complete for SmolLM2-135M-Instruct and SmolLM2-360M-Instruct (576 items each).
  - `khard` long-tail knowledge, closed book vs the same fact stated earlier in the chat: complete for 9 models (not Falcon).
- **Incomplete, not used for conclusions:**
  - Gemma-3-270M-it main battery: 420 of 1,116 items (distance 0 complete, distance 4 has 6 to 14 items per task, distance 10 and closed-book knowledge missing). Quoted only as "partial, distance 0".
  - Falcon-H1-Tiny-90M: the main run never started (empty log, skipped by the queue) and `khard` crashed on MPS (`RuntimeError: invalid low watermark ratio 1.4`). There is no likelihood data for the 90M model.
  - `uprobe` for LFM2-2.6B, LFM2-350M and Qwen3-0.6B never ran.
  - `copysplit` (per-token loss split by copyable vs novel tokens): only a pilot file for SmolLM2-135M-Instruct (1,002 scored tokens, written before the queue); the planned 444-thread runs never ran.
  - `ft_test.py`, the decisive fine-tuning test (can a few hundred steps on disjoint templated dialogues install corrections and binding at 135M?): the log shows weight loading only and there is no output file. It did not complete. It is the most important missing experiment in this whole diagnosis.
- **Recipe lens** (`diagnosis_recipe.md`): its P1 base-model probe results are written up there, but the raw outputs are not on disk anywhere I could find in this session, so I quote P1 only as a corroborating hint, marked **[recipe P1, raw not found]**. Its P3 fine-tune pilot never finished (the summary is still a `[[FT_SUMMARY]]` placeholder).
- **Firsthand chat probe** (`probe/`, 8 models, each with its own chat template, generation plus forced-prefix scoring): complete and audited. I use the audit's hand-adjusted scores and corrected claims throughout.

---

## Bottom line

1. **The "wall" Max perceives is where small models stop looking broken in casual use (fluent, some knowledge, few loops), not where multi-turn coherence starts.** On the audited 8-model probe battery the 350M class is level with Qwen2.5-0.5B (hand-adjusted greedy macro LFM2.5-350M 0.62, SmolLM2-360M 0.57, Qwen2.5-0.5B 0.60; `lanes/probe.verify.md` section 6). And the "wall" models themselves fail a strict definition of chatting: Qwen2.5-0.5B-Instruct deflects on 10 of 15 multi-turn final user-fact turns and fails all 3 correction conversations, and in the likelihood probe it prefers the stale value after a correction in 32 of 32 scenarios once four unrelated turns intervene.
2. **One coherence failure is universal below 1B: a correction stops counting once it is no longer among the last turn or two.** After a user correction followed by 4 or 10 unrelated turns, every model probed from 135M to 0.6B prefers the first-stated value in nearly every scenario (accuracy 0.00 to 0.22, the high end being LFM2-350M at 7 of 32; mean margins 0.8 to 10 nats toward the stale value), even when every statement uses identical wording. LFM2-2.6B prefers the corrected value (+1.6 to +1.8 nats). This is primacy, not surface copying and not frequency. Nothing below about 1B has been shown to do this, and whether a 150M model can be trained to do it is untested (the fine-tuning test crashed).
3. **The failures that appear first as models shrink below ~350M are, in order: knowledge (a smooth slope), who-is-who binding between same-type entities, instruction and format persistence, and degeneration.** At ~106M body, SmolLM2-135M falls back on a "first-mentioned wins" heuristic (base model: 0.83 when the answer was mentioned first, 0.40 when it was mentioned second), gets two-hop references right in both orders only 9% of the time at 10 turns, loops (22% of replies repeat a 4-gram three or more times) and copies itself (17%).
4. **Most of what looks like forgetting is not forgetting.** Retrieval of a single user-stated fact across 12 turns and ~1,700 tokens works at 74M body (Falcon-H1-Tiny-90M: greedy decoding from "Your name is" emits the fact at every distance), and reading a fact from context works at 135M (long-tail facts: 0.78 closed book, 0.95 to 1.00 when the user said it two turns earlier). About half of multi-turn recall failures in the chat probe are retrieval-intact (48% by greedy-from-prefix, 90% gold-over-foil), and one of the two largest free-answer failure classes (level with plain wrong answers, depending on how replies are labelled) is post-training reply policy (deflection, identity persona), which does not track size.
5. **What a 150M model must fix to count as chatting is not memory; it is updating, binding, stability and reply policy, with knowledge scoped out.** I propose an operational definition (section 1) that a skeptic can check: 12-turn conversations on the model's own history, facts supplied in context with same-type distractors, sealed templates, shortcut baselines, degeneration diagnostics and a blind human comparison. By that definition no model at or below 0.6B that has been probed passes, because all fail corrections. A 150M model that passed would be a real first, not a catch-up.
6. **Hard vs soft.** The only limit that follows from parameter count is stored knowledge. Updating is not explained by size between 135M and 0.6B (all fail, with 2T to 36T tokens), so it is a training-signal problem or a threshold between 0.6B and ~1-2.6B; the cheap test that decides this is written but was never run. Binding, persistence and degeneration are mixed: learnable in principle at small size, but token-hungry, and Max's 100B-token budget (about 670 tokens per parameter at 150M, versus 2,900 to 83,000 for every sub-600M model with good chat numbers) makes them the realistic binding constraints.
7. **Uncomfortable conclusions.** (a) Open-domain chat about the world is not achievable at 150M by any recipe that keeps facts in the weights. (b) The one published attempt at Max's exact size (ufakzeka-1, 151M body, 13.5B tokens) found identity tracking over a long story and multi-turn arithmetic "did not move across any data change we tried". (c) At 150M, training-seed variance can be as large as the difference between recipes, so most single-run improvements Max sees will be noise unless the evaluation is fixed first.

---

## 1. An operational definition of "really chatting" (RC-12)

A skeptical outsider will not accept vendor IFEval, 2-turn MT-Bench or a demo transcript. MT-Bench is 5 of 8 categories math, coding, reasoning or knowledge ([arXiv 2306.05685](https://arxiv.org/abs/2306.05685)); the same model's IFEval moves 10 to 24 points between harnesses (`lanes/eval.verify.md`); and the probe audit found that a grader called "strict" still passed negated answers, guess lists and hypotheticals (`lanes/probe.verify.md` section 2). The definition therefore has to fix the harness, separate knowledge from conversation, and be hard to game.

**Setup (all conditions required).**
- 12-turn conversations. User turns come from a sealed template set that no training-data generator has seen (13-gram decontamination checked). The model's own replies are the history (primary condition); a golden-history run is reported next to it to measure self-derailment.
- Decoding: greedy plus 3 samples at T=0.7, repetition penalty off for the primary numbers. Any trained model is reported over 3 training seeds (ufakzeka-1 found seed variance as large as its whole recipe spread at 151M; [arXiv 2609.25081](https://arxiv.org/abs/2609.25081)).
- Knowledge-light core: every fact a check needs is either stated earlier in the conversation or is common knowledge that a single-turn control confirms the model has. Knowledge is scored separately and always reported, including where the model loses.
- Every recall and binding check has a same-type distractor in context (another pet name, another day, another person). Shortcut rows are mandatory and must score at or below chance: empty reply, echo of the question, "first-mentioned value", "most recent value", "shotgun" (every entity listed).
- Graders are mutation-tested with fixtures that actually try to produce the failure they rule out: negated gold ("not Priya"), guess lists, hypotheticals ("for example, if it's Tuesday"), first-person claims inside a sentence with "you", incidental numbers, restating the fact without answering, and replies cut off at the token cap.
- Scaffolds (a harness-maintained recap or state card) and retrieval are allowed only as separately labelled conditions, never counted as the model.

**Pass criteria (lower 95% CI bound on the sealed split, pooled over the sampled runs and seeds).**

| # | criterion | threshold | where today's small models stand (best evidence) |
|---|---|---|---|
| C1 | Uses what the user said: user-stated facts 1 to 10 turns back, answered in second person, no deflection, same-type distractor present | >= 0.85 | Best free-answer recall at <=0.6B is 0.70 (SmolLM2-360M, greedy, no same-type distractor); 0.90 in the easier fixed-history sweep; Qwen2.5-0.5B 0.10 because it deflects (`lanes/probe.md` F1, F2) |
| C2 | Keeps who is who: owner, speaker and two-hop reference with both referents asked (pair metric, where a first-mention heuristic scores about 0) | >= 0.85 | Likelihood pair-both at 10 turns: Qwen3-0.6B 1.00, SmolLM2-360M 0.94, SmolLM2-135M-Instruct 0.62, LFM2-350M 0.34, SmolLM2-135M base 0.22 (section 3) |
| C3 | Accepts corrections, with 4 or more unrelated turns after the correction | >= 0.80 | 0.00 to 0.22 for every model from 135M to 0.6B; LFM2-2.6B 1.00 for one correction (likelihood). Free answers: 21 of 24 correction conversations fail at 90M-0.6B |
| C4 | Follow-ups and references: ellipsis ("and Italy?"), pointing into its own earlier answer ("the second one"), edits ("make it shorter") | >= 0.80 | Follow-ups pass from 350M up and at 90M, fail at 135M and 270M; own-list references unreliable below 0.5B (`lanes/probe.md` F8) |
| C5 | Persistence: instructions and persona still followed at turns 6 to 12 | >= 0.80 | Only 3-turn data exists: LFM2.5-350M and Qwen3-0.6B hold 1.00 over 3 turns; Falcon-90M and SmolLM2-135M decay after turn 1 (`lanes/probe.md` F7) |
| C6 | Stability: loops (consecutive-repetition coverage >= 0.25, the Daimax metric), cross-turn self-copy (>= 50% of a previous reply), role leaks, runaway length | <= 2% of turns combined | SmolLM2-135M: 22% of replies with a 4-gram repeated 3+ times, 17% self-copy; Falcon-90M 3%, 0%; LFM2/LFM2.5 about 0-1% (`lanes/probe.md` F7, F9) |
| C7 | Honest about knowledge: on facts never stated and not common knowledge, abstains or hedges; confident invented values are rare | abstain >= 0.70, invent <= 0.15 | Unmeasured in the probe. ufakzeka-1 (151M) passes 13 of 24 on its "unknowable question" gate ([arXiv 2609.25081](https://arxiv.org/abs/2609.25081)) |
| C8 | Blind human pairwise against Qwen2.5-0.5B-Instruct on realistic 6-turn chats (n >= 150, >= 3 raters, order randomized) | win or tie >= 50% | Never measured for any 90M-400M model (`lanes/posttrain.verify.md`, `lanes/eval.verify.md`) |

Two claim levels. **Level R (relative)**: non-inferior to Qwen2.5-0.5B-Instruct on the C1-C7 composite, as in the eval lane's pre-registered margin (paired lower bound at least -3 points; `lanes/eval.md` D8, with the eval audit's warning that a 1,450-conversation composite has only about 1,185 effective items, so the margin needs about 2,000 conversations for 80% power). **Level A (absolute)**: C1-C8 all pass. Level R is what "I matched the 0.5B wall model" means. Level A is what "it really chats" means, and on current evidence no model at or below 0.6B has it.

Why these thresholds. C1, C2, C4 and C6 are set where the better sub-1B models already are in easier settings, so they are reachable but not free. C3 is set where no sub-1B model is, because a human who says "actually it moved to Tuesday" and gets "Monday" back four turns later will not call that a conversation. C8 exists because programmatic families can be overfit, and ufakzeka-1 documents a gate that read 64/64 on its own templates and 34/64 on unseen paraphrases.

---

## 2. Premise check: is the 400-500M wall real?

Partly real as a slope, not real as a wall, and misplaced as a definition of chatting.

- **No cliff at 400-500M on any behavior measured.** On the audited firsthand battery the 350M class matches Qwen2.5-0.5B (hand-adjusted greedy macros LFM2.5-350M 0.62, SmolLM2-360M 0.57, Qwen2.5-0.5B 0.60, Qwen3-0.6B 0.62-0.71; LFM2-350M's lead is not robust, 0.50-0.60). Vendor multi-turn numbers agree: LFM2.5-230M (163M body, 19T tokens) scores Multi-IF 37.70 and LFM2.5-350M 44.92 against 41.68 for Qwen3.5-0.8B, in Liquid's harness, as a 3-turn average ([card](https://huggingface.co/LiquidAI/LFM2.5-230M), [card](https://huggingface.co/LiquidAI/LFM2.5-350M)). Falcon-H1-Tiny-90M reports 2-turn MT-Bench 4.33 against 3.80 for SmolLM2-360M in TII's harness, judge undetermined ([TII](https://tiiuae-tiny-h1-blogpost.hf.space/)).
- **There is a slope under a fixed recipe.** LFM2 Multi-IF 32.85 / 40.92 / 45.28 at 350M / 700M / 1.2B ([arXiv 2511.23404](https://arxiv.org/abs/2511.23404)); MobileLLM MT-Bench 2.33 at 125M vs 3.28 at 350M after identical chat tuning ([arXiv 2402.14905](https://arxiv.org/abs/2402.14905)); MT-Bench turn-2/turn-1 ratio 0.58 for Danube3-500M vs 0.76-0.90 at 1.8-4B ([arXiv 2407.09276](https://arxiv.org/abs/2407.09276)) **[>=0.5B only]**, a result that mixes knowledge with multi-turn skill.
- **Below ~300M the order follows recipe, not size.** Falcon-H1-Tiny-90M (74M body, 800B tokens) scores 0.40-0.46 hand-adjusted, above Gemma-3-270M (100M body, 6T) at 0.27-0.31 and SmolLM2-135M (106M body, 2T) at 0.11-0.15 (`lanes/probe.verify.md` section 6).
- **The "wall" models do not pass a strict definition either.** Qwen2.5-0.5B deflects on 67% of multi-turn final user-fact turns, fails every correction conversation, answers "Your name is [assistant's name]" in 28 of 32 likelihood scenarios at 10 turns, and puts the stale value first after any correction followed by 4+ turns. What made it the model Daimax shipped was 0 loops in 33 runs and more knowledge, not multi-turn coherence (and the 4/33 vs 0/33 loop gap to SmolLM2-360M is not statistically significant, p = 0.11, per `lanes/posttrain.verify.md`).
- **Sub-400M chat does exist, with scope limits.** Pre-LLM, BlenderBot 90M held 14-turn chit-chat that humans rated as engaging as Meena 2.6B (61-39) and not significantly different from its own 2.7B version, with contradiction, forgetfulness and repetition as the known failures ([arXiv 2004.13637](https://arxiv.org/abs/2004.13637)). Those three failure modes are ranks 1-3 below, six years later.

Premise verdict in one line: the 400-500M line is an artifact of which models were trained with the most compute and the least deflection, and of judging chat by knowledge-heavy single numbers. What is real is a knowledge slope, a token-hungry binding skill, and a correction failure that no sub-1B model has solved.

---

## 3. What fails first as models shrink

### 3.1 Likelihood probe (capacity lens data, complete for 8 models; my tabulation)

Plain `User:/Assistant:` transcript for every model (so this reads the representation, not the chat-template reply policy), 32 scenarios per cell, every foil a same-type value that appears in the conversation, contexts at most 577 tokens. "Pair" metrics require both members of a swapped pair right (chance 0.25; a first-mention heuristic scores about 0). Distance = unrelated turns between the last relevant statement and the question.

| model | body (M) | owner pair @10 | 1-hop pair @10 | 2-hop pair @10 | speaker pair @10 | update k=1 @10 | update k=3 vs original @10 | long-tail closed / open book |
|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M base (2T) | 106.2 | 0.22 | 0.22 | 0.09 | 0.12 | 0.03 | 0.00 | 0.78 / 1.00 |
| SmolLM2-135M-Instruct | 106.2 | 0.62 | 0.38 | 0.09 | 0.09 | 0.00 | 0.00 | 0.78 / 0.95 |
| LFM2-350M (10T) | 287.4 | 0.34 | 0.34 | 0.19 | 0.09 | 0.16 | 0.00 | 0.85 / 1.00 |
| LFM2.5-350M (28T + RL) | 287.4 | 0.47 | 1.00 | 0.78 | 0.00 | 0.00 | 0.00 | 0.85 / 1.00 |
| SmolLM2-360M-Instruct (4T) | 314.6 | 0.94 | 0.59 | 0.31 | 0.91 | 0.00 | 0.00 | 0.88 / 0.98 |
| Qwen2.5-0.5B-Instruct (18T) | 357.9 | 0.75 | 0.66 | 0.41 | 0.12 | 0.00 | 0.00 | 0.85 / 0.98 |
| Qwen3-0.6B (36T) | 440.5 | 1.00 | 1.00 | 0.94 | 0.97 | 0.00 | 0.00 | 0.80 / 1.00 |
| LFM2-2.6B (10T) | 2,435 | 1.00 | 1.00 | 0.81 | 0.97 | 1.00 | 0.84 | 1.00 / 1.00 |
| Gemma-3-270M-it, partial, distance 0 only | 100.3 | 0.47 | 0.69 | 0.19 | 0.94 | 0.75 | 0.84 | 0.95 / 0.98 |

Sources: `capacity_probe/out/*.jsonl`, `capacity_probe/results_khard.json`; tabulated by `scratchpad/behav/an.py`. Long-tail knowledge is a two-way forced choice (chance 0.50), 40 items; open book = the same fact stated by the user two turns earlier.

What the split by mention order shows (my analysis, `an2.py`, pooled over distances, n = 96 per cell). Owner binding ("my cat" vs "my sister's cat"), accuracy when the correct name was mentioned first vs second:

| model | answer mentioned first | answer mentioned second |
|---|---|---|
| SmolLM2-135M base | 0.83 | 0.40 |
| SmolLM2-135M-Instruct | 0.86 | 0.70 |
| LFM2-350M | 1.00 | 0.47 |
| LFM2.5-350M | 1.00 | 0.76 |
| SmolLM2-360M-Instruct | 0.98 | 0.99 |
| Qwen2.5-0.5B-Instruct | 1.00 | 0.89 |
| Qwen3-0.6B | 1.00 | 0.99 |
| LFM2-2.6B | 1.00 | 1.00 |

State updates, mean log-probability margin of the latest value over the first-stated value, 4 or 10 turns after the last correction (n = 64 per cell; negative = prefers the stale value): SmolLM2-135M base -1.1 to -1.4; SmolLM2-135M-Instruct -2.8 to -3.2; LFM2-350M -0.8 to -2.0; LFM2.5-350M -9.0 to -10.3; SmolLM2-360M-Instruct -3.8 to -4.6; Qwen2.5-0.5B -2.6 to -4.7; Qwen3-0.6B -2.5 to -3.0; LFM2-2.6B +1.6 to +1.8. Every update candidate is a single token in every tokenizer (no length bias; `an3.py`).

Controls that rule out surface-form copying (`capacity_probe/results_uprobe.json`, SmolLM2-135M-Instruct and SmolLM2-360M-Instruct):
- **Same wording** ("My dentist appointment is on X" for the original and every update, so the latest matching statement is the answer): accuracy at 0 turns after the last update 0.56-0.66 (135M) and 0.97-1.00 (360M); at 4 and 10 turns, **0.00 for both models at k = 1, 2 and 3 updates**.
- **Neutral answer prefix** ("That would be", sharing no words with any statement): 0.00-0.06 at 4 and 10 turns.
- For k = 1 the original and the update each appear twice (user statement plus assistant acknowledgement), so frequency does not explain the preference either. The first-stated value wins by position.

### 3.2 Generation probe (8 models, own chat templates, audited)

From `lanes/probe.md` with the audit's corrections (`lanes/probe.verify.md`):
- **Present at every size from 90M to 0.6B:** stale answers after corrections (21 of 24 free-answer correction conversations fail; Qwen3-0.6B passes 2 of 3); role capture, where the assistant adopts the user's self-description ("As a chef in a busy restaurant, my main responsibilities include", LFM2.5-350M), in 7 to 8 of 8 models; first-person perspective errors ("I told you my name was Priya.", "I'm Marcus.").
- **Present at every size but set by post-training, not size:** deflection. Multi-turn final user-fact turns: Qwen2.5-0.5B 67%, LFM2.5-350M 40%, Gemma-3-270M 40%, SmolLM2-360M 27%, SmolLM2-135M 20%, Qwen3-0.6B 20%, LFM2-350M 7%, Falcon-H1-Tiny-90M 0%.
- **Appears below ~350M total (~100M body):** weak single-turn controls (0.46 at 135M, 0.64 at 90M, 0.75 at 270M, vs 0.82-0.93 for the good 350M-600M models); speaker binding fails even single-turn at 135M ("My name is Lena, and I'm the name of your sister."); elliptical follow-ups fail at 135M ("Italy is also known as the "City of Love."") and 270M ("Italy is the capital of Italy."), but pass at 90M; format instructions followed at turn 1 and dropped after (Falcon-90M, SmolLM2-135M).
- **Appears at ~135M under one recipe:** loops and self-copy (22% of replies with a 4-gram three or more times, 17% self-copy, 0.93-0.95 cross-turn overlap on "tell me more", not fixed by the card's T=0.2 sampling); a template-and-persona interaction that collapses retrieval by turn 12 (greedy-from-prefix 0.00 through the chat template, 0.75 in a plain transcript on the same weights).
- **Appears at 90M under a good recipe:** a dominant how-to-guide template ("To determine which city you live in, consider the following factors: ### 1. **Population**"), placeholders instead of facts ("Your name is [Your Name]."), and knowledge errors ("Gracias (Greetings)", tigers that "inhabit the deserts of South America"). No loops, no self-copy, no deflection.

### 3.3 Order of appearance (synthesis)

| failure | 0.5-0.6B | ~350M | ~100M body | first class it belongs to |
|---|---|---|---|---|
| Stale value after a correction, 4+ turns later | yes | yes | yes | context tracking (update) |
| Role capture, first-person answers about the user | yes | yes | yes | role (and speaker binding) |
| Deflection on user facts | recipe-dependent (0-67%) | recipe-dependent | recipe-dependent | format/role (policy) |
| Weaker knowledge and single-turn skill | slight | yes | strong | knowledge / capability |
| First-mentioned-wins binding, two-hop reference | no (Qwen, SmolLM2-360M) | LFM2-350M yes | yes | context tracking (binding) |
| Format persistence decays after turn 1 | no | no (LFM2.5 holds) | yes | context tracking (instructions) + capability |
| Elliptical follow-up misread | no | no | 135M/270M yes, 90M no | context tracking + knowledge |
| Loops and self-copy | rare, decoding-fixable | rare | 135M yes, 90M no | degeneration |
| Losing a single unique fact within ~1,700 tokens | no | no | no (Falcon-90M), except via SmolLM2's template | not a limit |

External data points at or near Max's size: Supra2-100M-Instruct (30B tokens) resolves a "Sure!" after an offer about "pros and contras" into contraception, a reference resolved by surface association ([card](https://huggingface.co/SupraLabs/Supra2-100M-Instruct)); MaxGPT-2 (110M, about 18 tokens/param) produced "Bitcoin.com - Bitcoin.com" attractor loops and "the SHAPE of an explanation but not the content" (`WRITEUP_NOTES.md`); ufakzeka-1 (151M body, 13.5B tokens, about 89 tokens/param) "can confuse who is who" after a long story and "does column arithmetic on one turn and can lose the answer on the next" ([arXiv 2609.25081](https://arxiv.org/abs/2609.25081)).

---

## 4. From failure to mechanism

Five diagnostics separate the mechanisms, and the probes already use four of them:

| diagnostic | if the model passes it but fails the chat check | if it fails it too |
|---|---|---|
| Single-turn control (same content in one message) | context-tracking or policy miss | knowledge or capability miss |
| Forced prefix ("Your name is") | the fact is retrievable: policy or binding miss | retrieval miss |
| Same-type foil in context | (needed to tell binding from retrieval) | binding miss |
| Plain transcript vs chat template on the same weights | template/persona policy miss | representation-level miss |
| Golden vs own history | self-derailment (degeneration, exposure bias) | cannot use context at all |

Classification of the observed failures:

- **Knowledge miss.** Wrong facts with fluent form (Falcon's tigers, MaxGPT-2's photosynthesis), single-turn controls failing at ~100M. Reading works: at 135M the same long-tail fact goes from 0.78 closed book to 0.95-1.00 when the user said it two turns earlier. Storage is the limit, not use.
- **Context-tracking miss, three kinds.**
  - Retrieval of a unique fact: rare. Falcon-90M greedy-from-prefix 1.00 at every distance to 1,726 tokens; 48% of multi-turn recall failures in the chat probe are retrieval-intact by greedy decoding and 90% by gold-vs-never-seen-foil, and the fixed-history sweep gives 62% (`lanes/probe.verify.md` claim 3). Caveat from the audit: those foils never appear in context, so this is an induction-copy test.
  - Binding between same-type entities: common at ~100M body. The model retrieves a name near the right words and defaults to the first one mentioned (section 3.1). Wu et al. saw the same shortcut phase, "a shallow heuristic prioritizing early variable assignments", in a 37.8M transformer before it learned real binding on synthetic programs ([arXiv 2505.20896](https://arxiv.org/abs/2505.20896); the fact-check notes the heuristic there is positional and is retained, not replaced).
  - Updating: universal below 1B. The model retrieves the first value of a key once the correction is not among the most recent turns. This matches "active primacy intrusion", where proactive interference exceeds retroactive interference in all 39 models tested from 1B to 2.5T and size does not predict proactive-interference resistance (R^2 = 0.06; [arXiv 2603.00270](https://arxiv.org/abs/2603.00270), as reported in `followup/floor.verify.md`) **[>=1B only]**, and the "attention glitches" transformers make when copying the last-written value across irrelevant tokens ([arXiv 2306.00946](https://arxiv.org/abs/2306.00946), synthetic flip-flop task).
- **Format or role miss (policy).** Deflection, identity disclaimers ("I'm Qwen ... I don't have a cat"), how-to templates, placeholders. Diagnostic signature: the fact is retrievable, and changing only the rendering changes the answer. SmolLM2-135M-Instruct recall goes from about 0.05-0.10 through its chat template to 0.45 as a plain transcript on the identical history, hand-regraded (`lanes/probe.verify.md` claim 4). Role capture is partly representation too: in the plain-transcript likelihood probe, LFM2.5-350M answers "And what's your name?" with the user's name in 31 of 32 scenarios at 10 turns, and Qwen2.5-0.5B answers "what's my name?" with the assistant's name in 28 of 32.
- **Degeneration.** Loops, self-copy, framing that persists ("a dental AI"), attractor templates. Diagnostic signature: failure grows with the model's own text in context and responds to decoding changes. Qwen2.5-0.5B's verbatim loop disappears with its card sampling (T 0.7, repetition penalty 1.1: cross-turn overlap 1.00 to 0.12); SmolLM2-135M's does not at T 0.2 (0.93). The self-reinforcement mechanism ("the more times a sentence is repeated in the context, the higher the probability of continuing to generate that sentence") is measured on GPT-2-scale models ([arXiv 2206.02369](https://arxiv.org/abs/2206.02369)), and SFT-trained students' accumulated error keeps growing with generation length while on-policy-trained ones stop accumulating (GPT-2 125M student, [arXiv 2306.08543](https://arxiv.org/abs/2306.08543)).

---

## 5. Ranked limits for a ~150M chat model built on Max's budget

Ranking criterion: how likely the limit is to make a 150M model fail RC-12 after the cheap, known fixes are applied, weighted by how uncertain the fix is. Max's budget: about 100B pretraining tokens (about 670 per parameter at 150M), a same-tokenizer 1.1B teacher (MaxGPT-Ultra) in about 70 days, 108k SFT rows and 60k preference pairs.

### Rank 1. Updating state after a correction ("first value wins")

- **Mechanism.** Retrieval by key finds every statement of "appointment ... day", and small models resolve the tie toward the first one once the correction is not among the last one or two turns. Generic text rarely contains in-place overwrites of a stated value, so pretraining does not reward "latest wins"; single-turn SFT data never contains it; the model's own acknowledgements add more copies of both values.
- **Kind: unknown, leaning soft.** Within 135M-0.6B it does not vary with size or with 2T-36T tokens (all at 0.00-0.22 after 4+ turns), so parameter count is not what drives it in that range. LFM2-2.6B does it. Small networks learn "latest wins" when the data teaches it: memory networks trained from scratch reach 100% on bAbI dialog Task 2, "Updating API calls", where users change their request 1 to 4 times ([arXiv 1605.07683](https://arxiv.org/abs/1605.07683)). **[recipe P1, raw not found]** reports LFM2.5-1.2B-Base passing a parallel-wording correction probe (0.94) while Gemma-3-1B-pt fails, which would make it recipe at ~1B. But no general chat model at or below 0.6B has been shown to do it, and the direct test at 135M (`ft_test.py`) never finished.
- **Evidence.** Section 3.1 (8 models, 32 scenarios per cell, same-wording and neutral-prefix controls); `lanes/probe.md` F5 as corrected (21 of 24 free-answer failures; K_day log-prob favors the corrected value in 6 of 8 models when only one distractor turn follows, K_time and K_color mostly favor stale); floor evidence summarized in `followup/floor.verify.md` (no general chat model up to 0.6B shown to accept corrections).
- **Why it ranks first.** It is certain to fail by default, it is the most visible conversational error to a human, and it is the one failure where nobody has shown a fix at small scale in natural chat.
- **Attacks.** Known: correction-dense dialogues with stale traps (including turns where the assistant previously repeated the old value) in both pretraining mix and SFT, bAbI-dialog style; DPO pairs with the stale answer rejected; varying where the original statement sits so position is not a cue. Plausible new: a belief-state auxiliary loss that predicts the current value of each planted slot at every turn (SimpleTOD-style dialogue-state supervision, which reaches JGA 54.5 at 82M on MultiWOZ, [arXiv 2005.00796](https://arxiv.org/abs/2005.00796), used as an auxiliary head inside a general chat LM); attention variants with a native forget or overwrite path in a few layers (selective attention, stick-breaking, Forgetting Transformer, or a gated-delta-rule layer with an erase gate; stick-breaking beats softmax on RULER variable tracking at 1B, but that task is chained assignment, not reassignment, `followup/archfp.verify.md`) **[>=1B only]**; on-policy distillation from a teacher that passes the probe. Scaffold (reported separately): a harness state card that re-states current values each turn.
- **Cheapest decisive test.** Run the already-written `capacity_probe/ft_test.py` on SmolLM2-135M-Instruct and then, separately, SmolLM2-360M-Instruct (one model at a time, memory guard on), and re-score U k=1-3 at 4 and 10 turns plus the same-wording and neutral-prefix controls on held-out names and events. Add one position control: put the original value after three distractor turns instead of in turn 1. Reading: at least 0.8 at 10 turns on held-out templates at 135M means soft; flat at 135M but moving at 360M means a size threshold in Max's range; flat at both means generic fine-tuning cannot install it and it needs pretraining-scale signal. About an hour of Mac time.

### Rank 2. Who-is-who binding between same-type entities (owner, speaker, two-hop reference)

- **Mechanism.** Answering "what's my sister's cat called?" needs the model to bind a value to its owner, not just find a value of the right type. At ~100M body, models fall back on "first mentioned wins", and speaker roles (I vs you vs the assistant) blur, so the model answers as the user or names itself after the user.
- **Kind: mixed.** It is learnable at small size (37.8M on synthetic programs, [arXiv 2505.20896](https://arxiv.org/abs/2505.20896); SmolLM2-360M and Qwen3-0.6B are near 1.00), but under generic training it is token-hungry: **[recipe P1, raw not found]** reports SmolLM2-135M answering "what's my friend's name?" correctly 0.04 at 252B tokens, 0.23 at 1T and 0.56 at 2T, and Pythia-410M no better than Pythia-160M at matched tokens. At about 670 tokens per parameter, a generically trained 150M model will sit near the start of that curve. Recipe also matters at fixed size: LFM2-350M (10T) is worse at owner binding than SmolLM2-135M-Instruct (2T).
- **Evidence.** Section 3.1 (pair metrics and the first/second split); `lanes/probe.md` F6 (SmolLM2-135M and Falcon-90M prefer the sister's name in the forced choice even single-turn; Falcon's free answers were right in 3 of 4 runs); role capture in 7-8 of 8 models; ufakzeka-1: "identity tracking over a long story stayed between 2 and 36 percent failures with no relation to the data change" across its last eight rounds at 151M body and 13.5B tokens ([arXiv 2609.25081](https://arxiv.org/abs/2609.25081)), which the authors read as a size limit but which is confounded by about 89 tokens per parameter; Supra2-100M's surface-association referent error.
- **Attacks.** Known: recall-dense dialogues with same-type distractors where the second-mentioned entity is the answer as often as the first; perspective-flip data ("my X" asked, "your X" answered) and role-swap traps; speaker or role embeddings on each turn (TransferTransfo-style dialog-state embeddings at 117M, [arXiv 1901.08149](https://arxiv.org/abs/1901.08149)); a previous-token path (short causal conv / Canon-style layer) that makes induction cheaper; spending extra epochs of the 100B build on dialogue. Plausible new: nonce or high-entropy planted names so the skill cannot be answered from weights (the in-context-learning data-distribution result of [arXiv 2205.05055](https://arxiv.org/abs/2205.05055)); logit distillation from Ultra on binding-dense dialogue only.
- **Cheapest decisive test.** Two cheap reads. (a) Score the owner, one-hop, two-hop and speaker items (answer-mentioned-second split) on SmolLM2-135M's published intermediate checkpoints (every ~250B tokens) and on Max's own 124M A/B checkpoints at about 1B tokens: that is the curve Max will be on at 100B. (b) The `ft_test.py` binding families at 135M: if a few thousand disjoint dialogues lift the second-mentioned accuracy to at least 0.9 on held-out names, the limit is training signal and cheap to remove.

### Rank 3. Degeneration under the model's own history (loops, self-copy, attractor templates)

- **Mechanism.** Each turn adds the model's own text to the context; repeated text raises its own probability (self-reinforcement), and off-policy-trained small students accumulate error with length. Undertrained or narrowly trained small models also have low-entropy attractors ("Bitcoin.com", "consider the following factors: ### 1.").
- **Kind: mixed, mostly soft.** It is removable at 90M: Falcon-H1-Tiny-90M has 0 self-copy, 3% repeated-4-gram replies and 0.03 cross-turn overlap (800B tokens, SFT data in pretraining, DPO). But it concentrates where training is thin (SmolLM2-135M 22% and 17%; MaxGPT-2 at 18 tokens per parameter), and two things Max is likely to do raise the risk: distillation (distilled students loop far more than their teachers at low temperature, 1.5B-32B reasoning models, [arXiv 2512.12895](https://arxiv.org/abs/2512.12895) **[>=1B only]**; the same paper says instruct models barely loop) and long chain-of-thought traces (TII found CoT in its 90M tool-calling data caused infinite loops, [TII](https://tiiuae-tiny-h1-blogpost.hf.space/)).
- **Evidence.** `lanes/probe.md` F7, F9 (confirmed with caveats: 1 greedy and 2 sampled "tell me more" chats per model); Daimax SmolLM2-360M 4/33 vs Qwen2.5-0.5B 0/33 at T=0.7 (not significant, counted with the old 5-gram metric per `lanes/eval.verify.md`); `WRITEUP_NOTES.md`; BlenderBot 90M's documented repetition failures.
- **Attacks.** Known: deduplicate and filter repetitive spans in chat data; DPO with looping and self-copied replies as rejected; unlikelihood on repeated n-grams in the model's own samples (label repetition 0.617 to 0.055 in a ~90M-class dialogue model, [arXiv 1911.03860](https://arxiv.org/abs/1911.03860)); on-policy distillation or SFT on corrected self-generated histories; no long CoT in SFT; sampling at T >= 0.5 rather than greedy, with a mild penalty as a stopgap only. Plausible new: train on conversations whose earlier assistant turns are the model's own samples (own-history SFT), so the model learns to recover from its own drift.
- **Cheapest decisive test.** Run the probe's free-form and 12-turn own-history conversations (greedy and T=0.7 x3) on the four published Falcon-H1-Tiny-90M siblings (`-Instruct-Curriculum`, `-Instruct-pre-DPO`, `-Instruct-Curriculum-pre-DPO`, `-Base`; same architecture, released 2026-01-12 per `lanes/probe.verify.md` section 7). That isolates whether DPO, SFT-in-pretraining or the architecture removes loops at 90M. Minutes per model, one at a time.

### Rank 4. Stored knowledge (hard)

- **Mechanism.** Facts live in the weights at about 2 bits per parameter at best (1,000 exposures), about 1 bit at 100 exposures, and far less with junk-heavy data ([arXiv 2404.05405](https://arxiv.org/abs/2404.05405); synthetic, up to 0.5B). A 150M model stores at most tens of MB of facts. Across 93 open models, obscure-fact accuracy at fixed size has not improved with newer recipes (`diagnosis_capacity.md` section 1.2, citing [arXiv 2604.24827](https://arxiv.org/abs/2604.24827)).
- **Kind: hard.** This is the one limit that follows from parameter count. It is a slope, not a wall: Qwen2.5-0.5B stores only about 3x more.
- **Evidence (behavioral).** Single-turn controls 0.46-0.64 at ~100M body vs 0.82-0.93 at 350M-600M (partly skill, not only knowledge); Falcon-90M's fluent wrong facts; long-tail closed book 0.78 at 135M vs 0.85-0.88 at 350M-0.5B, and 0.95-1.00 for every model once the user states the fact (section 3.1). Reading is intact where storage is not.
- **Why it ranks fourth here and not first.** It does not stop multi-turn coherence, which is Max's stated goal. It stops open-domain chat about the world, and no technique that keeps facts in the weights will change that at 150M. If Max's definition of chatting includes knowing things, this becomes rank 1 and that part of the goal is impossible.
- **Attacks.** Scope it out: train the model to answer from what the user or a retriever supplied, to say "I don't know" otherwise, and keep the pretraining mix knowledge-light (entity-anonymized pretraining of SmolLM-architecture models at 135M and 360M on 2.5B tokens cut closed-book LAMA recall from 12.5 to 0.7 at 135M while context-grounded FEVER rose from 82.8 to 89.5 after per-benchmark fine-tuning, [arXiv 2607.12831](https://arxiv.org/html/2607.12831v1), confirmed against the full text in `followup/retrieval.verify.md`; a single preprint without released code, so a strong hint, not a result); retrieval for facts; sparse memory only if "150M" is allowed to mean active parameters.
- **Cheapest decisive test.** Not whether it is hard (it is), but whether scoping works: fine-tune SmolLM2-135M on a few thousand grounded dialogues (relevant, irrelevant and conflicting passages, plus unanswerable questions) and measure open-book accuracy with a distractor passage and the abstention rate on unstated facts (C7). Also score the `khard` closed/open twin on Max's 124M A/B checkpoints to see where his own models sit.

### Rank 5. Single-turn instruction skill and persistence over turns

- **Mechanism.** Following a format or persona instruction while producing content, and continuing to attend back to it several turns later, needs both a learned mapping and capacity; at ~100M body the first is weak and the second decays after one turn.
- **Kind: mixed.** Under one recipe it scales with size (LFM2 Multi-IF slope; MobileLLM 125M vs 350M), and it moves with recipe at fixed size (LFM2-350M to LFM2.5-350M: Multi-IF 32.92 to 44.92, 10T to 28T tokens plus RL, confounded). SFT data in pretraining raised IFEval at 90M (53.47 to 66.08 after DPO) but not 2-turn MT-Bench (4.40 curriculum vs 4.33) ([TII](https://tiiuae-tiny-h1-blogpost.hf.space/)), so it is a single-turn instruction lever, not a chat lever.
- **Evidence.** `lanes/probe.md` F7 (persistence did not decay over 3 turns in 6 of 8 models; Falcon-90M and SmolLM2-135M decay; LFM2.5-350M and Qwen3-0.6B hold 1.00 over 3 turns); 3-turn Multi-IF is the longest persistence measurement at this size and it is an average, not a turn-3 rate (`followup/floor.verify.md`). Beyond 3 turns: nothing published below 1B; at 4B-32B, multi-turn constraint persistence falls to about 30% ([arXiv 2603.01423](https://arxiv.org/abs/2603.01423), per `lanes/eval.verify.md`) **[>=1B only]**, which warns that C5 may sit at the floor for every small model.
- **Attacks.** Tokens (3-4 epochs of the 100B build are close to unique data, [arXiv 2305.16264](https://arxiv.org/abs/2305.16264)); multi-turn instruction data where constraints must hold for 6-12 turns (TurnWise at 7B: +12.8 multi-turn points from 10k conversations, [arXiv 2603.16759](https://arxiv.org/abs/2603.16759) **[>=1B only]**); 1 epoch of DPO; matching SFT difficulty to capacity (SmolLM2's smol-smoltalk filtering).
- **Cheapest decisive test.** A 6-12-turn persistence family (C5) run on SmolLM2-135M vs SmolLM2-360M and on the Falcon-90M siblings. If every sub-1B model is at the floor after turn 3, the criterion is measuring a universal weakness, like rank 1; if 360M holds and 135M does not, it is a capacity-and-tokens limit Max has to budget for.

### Rank 6. Reply policy from post-training (deflection, identity persona, generic templates)

- **Mechanism.** SFT and preference data teach identity and privacy disclaimers that fire on "my X" questions, default system prompts switch on a persona that will not use the conversation, and small models adopt dominant response templates.
- **Kind: soft.** It does not track size (deflection 0% at 90M, 67% at 0.5B on multi-turn final turns), it is removable at 90M (Falcon), and it changes with rendering alone on fixed weights.
- **Evidence.** `lanes/probe.md` F3, F4 and E1 as corrected; taxonomy: deflection is a top-two failure class under every labelling (32-36 of 95 greedy user-fact failures, level with wrong-or-ignored). The audit found the claim that ~100M models cannot be prompted unsupported (both small models did adopt the "Captain Pip" pirate persona and missed only the literal "Arr"), so "cannot be prompted into place" is not established.
- **Why it ranks sixth.** It is the largest failure class in today's small models, but Max writes his own SFT and DPO data, so it is under his control and cheap. The risk is importing it: his DPO set is single-turn, and his SFT packer trains later turns without their history (compute lane, confirmed; `diagnosis_recipe.md` P2 simulates 42% of supervised assistant tokens without the start of their own conversation at seq_len 2048).
- **Attacks.** Remove or rewrite identity and privacy deflections where the answer is in the conversation; DPO pairs with the deflection rejected; no fixed identity blurb or default system prompt, varied system prompts instead; pack SFT by whole conversation with document masking.
- **Cheapest decisive test.** Run the chat probe battery on Falcon-90M `-Instruct` vs `-Instruct-pre-DPO` vs `-Instruct-Curriculum` to see which stage produces zero deflection; and a small full fine-tune of SmolLM2-135M-Instruct on second-person recall dialogues with no identity blurb, scored through its own chat template.

### Rank 7. Measurement (an evaluation artifact that hides or invents progress)

- **Mechanism.** Single-number chat scores mix knowledge with coherence; harness choice moves published scores 10-24 points; substring graders pass negations, guess lists and hypotheticals; at 150M, seed variance rivals recipe differences.
- **Kind: soft.**
- **Evidence.** The probe audit's regrade moved LFM2-350M from 0.65 to 0.50 and changed which models lead (`lanes/probe.verify.md` sections 2, 6); ufakzeka-1's 64/64 vs 34/64 gate and its seed spread (helpfulness 75.2 / 80.2 / 81.2 across three seeds, a span as large as its last fourteen checkpoints) ([arXiv 2609.25081](https://arxiv.org/abs/2609.25081)); eval lane D6 power arithmetic as corrected.
- **Attacks.** RC-12 as defined in section 1; Tier-0 likelihood probes tracked across pretraining checkpoints; paired comparisons; three seeds; a blind human pass before any claim.
- **Cheapest decisive test.** Regrade the existing probe transcripts with the audit's missing fixtures (no model runs needed), and check whether the model ranking is stable across the three decoding runs and two phrasings; if Kendall's tau between runs is low, no single-run recipe comparison at this size should be trusted.

### Rank 8 (not a limit). Holding one user-stated fact across turns

- **Evidence.** Falcon-H1-Tiny-90M retrieves every planted fact at every distance up to 1,726 tokens and answers 0.65-0.75 correctly hand-regraded; every model with at least ~290M body retrieves at 0.75-1.00 at ~1,700 tokens; reading a stated fact works at 135M (section 3.1). SmolLM2-135M's collapse by turn 12 tracks turn count through its chat template, not tokens (`lanes/probe.verify.md` claim 7).
- **Kind: soft.** Keep real attention in at least some layers and a 2-4k context. Do not spend the novelty budget on memory architecture for chats of this length.
- **Cheapest decisive test.** Tier-0 recall probes on Max's own 124M A/B checkpoints, to confirm his architecture (gated attention, value residual) retrieves a unique fact at 1-2k tokens.

---

## 6. What a 150M model must fix, and what it may skip

Must fix to count as chatting (RC-12 C1-C6): use user facts without deflection and in second person (rank 6, then rank 8 comes free); bind values to the right owner and speaker in both mention orders (rank 2); accept corrections that are four or more turns old (rank 1); resolve follow-ups and references (ranks 2 and 5); keep instructions and persona for the whole chat (rank 5); do not loop or copy itself (rank 3).

Must be honest about (C7): knowledge it does not have. Abstaining is required; knowing is not.

May skip: breadth of world knowledge, long-form reasoning, math, code, tool calling. Every sub-400M vendor already scopes its model away from these (Gemma 3 270M is "not designed for complex conversational use cases", [Google](https://developers.googleblog.com/en/introducing-gemma-3-270m/); LFM2.5-230M is not recommended for creative writing, [card](https://huggingface.co/LiquidAI/LFM2.5-230M)).

---

## 7. Key uncertainties

- Whether "latest value wins" can be trained into a 135-150M model at all. The decisive fine-tuning test was written and never completed (both this lens's queue and the recipe lens's P3).
- Whether binding at 150M needs trillions of generic tokens or can be installed with a few thousand targeted dialogues at 100B tokens. The token curve I quote is from the recipe lens and its raw data was not found on disk.
- ufakzeka-1 (151M body) says identity tracking over long stories did not respond to data. It trained at about 89 tokens per parameter and judged with an unidentified LLM judge, so it cannot separate size from undertraining, but it is the only same-size counter-evidence and it points the uncomfortable way.
- The likelihood results use plain transcripts for instruct models, not their chat templates, and contexts under 600 tokens; the generation probe has 1 to 20 trials per cell, one phrasing per question and graders with known holes. Whether likelihood passes turn into generation passes is unmeasured (LFM2-2.6B passes the update probe but was not generation-tested).
- Primacy by position vs "first mention of this key" is not separated: the original value was always in the first turn.
- No data beyond 12 turns, no human ratings, and no Falcon-90M likelihood data (both Falcon runs failed).
- Qwen3.5-0.8B (the current small Qwen) and the four Falcon-90M sibling checkpoints were not probed.
- Whether distilling from MaxGPT-Ultra raises loop rates at 150M, as distillation does in larger reasoning students.

---

## 8. Reproducibility

- Likelihood tables: `python3 -B scratchpad/behav/an.py` (per-model, per-task, per-distance accuracy from `capacity_probe/out/*.jsonl` using `capacity_probe/analyze.py`'s `summarize`, without writing any file); `an2.py` (mention-order split and update margins; regenerates item metadata with `capacity_probe/items.py`, whose order matches every saved file); `an3.py` (candidate token-length check). Scratchpad: `(scratch file)`.
- Uprobe and long-tail tables: `capacity_probe/results_uprobe.json`, `capacity_probe/results_khard.json` (read, not regenerated).
- Generation probe numbers: `lanes/probe.md` as corrected by `lanes/probe.verify.md`; raw transcripts in `probe/transcripts/`.
- No model was loaded or run for this report, and nothing in `capacity_probe/`, `probe/` or Max's repositories was modified.
