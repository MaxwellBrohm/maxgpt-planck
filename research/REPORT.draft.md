
# What limits a ~150M chat model: synthesis report

Date: 2026-09-23. Draft for Max. Synthesizes the nine verified research lanes (`lanes/*.md` as corrected by `lanes/*.verify.md`), the seven follow-up fact-checks (`followup/*.verify.md`), the hands-on chat probe (`probe/`), the likelihood battery (`capacity_probe/`), and the three independent diagnoses (`diagnosis_capacity.md`, `diagnosis_recipe.md`, `diagnosis_behavior.md`).

How to read the markers:
- **[corrected]** means a fact-check changed that number or claim; the corrected form is what appears here.
- **[>=1B only]** flags evidence that exists only at 1B parameters or above.
- **[vendor]** means the lab measured its own model in its own harness.
- "My arithmetic" or "recomputed" means I did it in plain Python from saved files. No model was loaded, run or trained for this synthesis.

**Evidence status, read first.** An earlier attempt at this step crashed the Mac (kernel panic from three concurrent model processes). What survived:

| work item | status |
|---|---|
| Chat probe: 8 models, 29 multi-turn conversations + 24 single-turn controls, plus fixed-history, base-vs-instruct and system-prompt experiments | complete and audited (`lanes/probe.verify.md`) |
| Likelihood battery (binding, two-hop, corrections, perspective, knowledge), plain transcripts, same-type foils in context | complete for 8 models; Gemma 3 270M partial (distance 0 only); **Falcon-H1-Tiny-90M missing** (MPS crash) |
| Correction controls (`uprobe`) | SmolLM2-135M-I and 360M-I only |
| **`capacity_probe/ft_test.py`: can a few hundred fine-tuning steps teach a 135M model binding and corrections?** | **never produced output** (log ends at weight loading). This is the single most decision-relevant missing experiment. |
| Recipe lens P1 (token-curve probe on intermediate checkpoints) | raw outputs lost; its numbers are treated as unverified leads only |

---

## 1. Verdict

**Is the 400-500M wall real?** Not as a wall. Three things are real, and none of them sits at 400-500M:

1. **A knowledge slope.** Facts stored in weights top out near 2 bits per parameter, linear in size. A 150M model holds about 0.3x what Qwen2.5-0.5B can. There is no size where knowledge "turns on".
2. **A skill slope under any fixed recipe** (LFM2 Multi-IF 32.85 / 40.92 / 45.28 at 350M / 700M / 1.2B), which recipes have moved by 2-4x in parameters.
3. **One coherence failure that no sub-1B model has solved:** once a correction is 4 or more turns back, every model probed from 135M to 0.6B answers with the original value. The 0.5B "wall" model fails it too, so it is not what separates 150M from 500M.

What looked like a wall is where labs spent compute (2,900 to 83,000 tokens per parameter plus large-teacher distillation or RL) and where models stop looking broken in casual use (fluent, some facts, no loops). **Sub-400M chat models exist and work, within limits.** Specifics:
- LFM2.5-230M (163M body, 19T tokens): Multi-IF 37.70 vs 41.68 for Qwen3.5-0.8B [vendor].
- Falcon-H1-Tiny-90M: 2-turn MT-Bench 4.33 vs 3.80 for SmolLM2-360M in the same harness [vendor].
- On our 8-model probe the 350M class is level with Qwen2.5-0.5B, and Qwen2.5-0.5B itself deflects on 10 of 15 multi-turn "what did I tell you" questions.

**What nobody has shown below about 300-400M:** knowledge-bearing, many-turn chat, and conversation measured past 3 turns (apart from agent task scores and 2020-era chit-chat ratings).

**What limits a ~150M model, ranked** by how likely each is to stop a Max-built 150M (100-300B tokens) from holding a knowledge-light multi-turn conversation:

| # | limit | kind | one line |
|---|---|---|---|
| 1 | Updating after a correction ("first value wins") | unknown | 0.00-0.11 correct at 4-10 turns in all 8 sub-1B models; LFM2-2.6B 0.72; no gradient from 135M to 0.6B. |
| 2 | Binding same-type entities (whose cat, whose job, my name vs yours) and two-hop references | mixed | The only size gradient below 300M of body: two-hop at chance at ~100M body; recipe moves it as much as 8.5x size does. |
| 3 | Chat training signal and reply policy (single-turn SFT and DPO, deflection personas, broken packing) | soft | The largest failure class today and the cheapest to remove. Max's own SFT packer gives ~0.1% of supervised tokens a clean history. |
| 4 | Tokens per parameter (670 planned vs 2,900-83,000 for every working small chat model) | mixed | Soft in principle, expensive in practice; interacts with #2 through data density. |
| 5 | Degeneration on its own history (loops, self-copy) | mostly soft | Falcon-90M has none; SmolLM2-135M repeats a 4-gram 3+ times in 22% of replies. |
| 6 | Instruction persistence beyond 3 turns | mixed | Unmeasured below 1B past 3 turns. |
| 7 | Measurement | soft | Harnesses swing scores 10-24 points; at 151M one seed moves as much as any recipe. |
| 8 | Stored world knowledge | **hard** | Ranked 8th only because the goal is coherence. If "chatting" includes knowing things, it is #1 and not reachable at 150M without retrieval. |
| 9-11 | Teacher signal; embedding share; output-head rank and depth | mixed / soft / unknown, minor | Real but second-order at 150M. |
| not a limit | Holding one user-stated fact for 12 turns and ~1,700 tokens | | Falcon-H1-Tiny-90M (74M body) does it with zero deflections. |

**Honest odds** (my judgment before the decisive tests, not measurements):
- **Non-inferior to Qwen2.5-0.5B-Instruct on a knowledge-light multi-turn battery: roughly even, 40-60%.**
  - For: the bar is low (the 0.5B model deflects and fails corrections), the 350M class already clears it, and Falcon-90M gets 0.40-0.46 against Qwen2.5-0.5B's 0.60 with a 74M body.
  - Against: Max's planned budget is about a fifth of Falcon's compute (1.6x short at 300B tokens), and the only same-size attempt (ufakzeka-1, 151M) stalled on identity tracking.
- **Passing a strict definition** (corrections 4+ turns old, both-order binding, no loops, blind human win-or-tie vs Qwen2.5-0.5B): **about 10-25%.** It rises to perhaps 40-50% if `ft_test.py` shows a 135M model learns "latest value wins" from a few hundred steps. No model at or below 0.6B passes today, so a pass would be a genuine first.
- **Matching a 0.5B model on open-domain knowledge without retrieval: under 5%.** That limit is linear in parameters.

**The cheapest thing that moves these odds** is about one hour of compute: run `ft_test.py` on SmolLM2-135M-Instruct, then on SmolLM2-360M-Instruct, one process at a time.

---

## 2. The landscape: sub-600M chat models

Body = non-embedding parameters (all tied unless noted). Tok/param = pretraining tokens / total params. Chat numbers are only comparable within one harness: the same model's IFEval moves 10-24 points between harnesses (Gemma 3 270M: 51.2 Google vs 27.44 TII) [corrected: only 2 of the 4 examples in the eval lane exceed 10 points]. "Probe" = our hand-adjusted greedy macro (section 3; set A / set B).

| model (release) | total | emb share (body) | pretraining tokens (tok/param) | recipe | chat evidence |
|---|---|---|---|---|---|
| PleIAs Monad (Nov 2025) | 56.7M | 3.7% (54.6M) | 200B synthetic (3,530) | SYNTH only, thinking traces, 64 layers | card: no multi-turn support |
| BlenderBot 90M (2020) | 87.5M | seq2seq | Reddit + BST fine-tune | pre-LLM chit-chat | 14-turn human chats: engagingness beat Meena 2.6B 61-39; known failures: contradiction, forgetting, repetition |
| Falcon-H1-Tiny-90M-Instruct (Jan 2026) | 91.1M | 18.4% (74.4M) | 800B (8,779) | attention + Mamba2 in every layer; 25% SFT data in pretraining; SFT; 1 epoch DPO; no teacher | IFEval 66.08, 2-turn MT-Bench 4.33 [vendor; corrected: 2-turn evalchemy MT-Bench, not MT-Bench-101, judge undetermined]; probe 0.46 / 0.40, zero deflection, best fixed-history recall per parameter |
| Supra2-100M-Instruct (Aug 2026, community) | 100.7M | 25.0% (75.5M) | 30B (298) | community | its own card sample resolves "Sure!" to the wrong referent by surface association |
| MobileLLM-125M (2024) | 124.6M | 14.8% (106.2M) | 1T (8,026) | deep-thin, 30 layers; chat fine-tune in paper only | MT-Bench 2.33, AlpacaEval 24.07% (GPT-4 judge, identical chat FT for all models) |
| SmolLM2-135M-Instruct (Oct 2024) | 134.5M | 21.0% (106.2M) | 2T (14,868) | single-stage pretrain; smol-smoltalk SFT; DPO; no distillation | IFEval 29.9 (HF) / 30.69, MT-Bench 2.68 (TII); probe 0.15 / 0.11; repeats a 4-gram 3+ times in 22% of replies |
| ufakzeka-1 (Sep 2026, Turkish) | 182M | ~17% (151M) | 13.5B (74-89) | SFT-only release; DPO hurt in 3 variants | identity tracking over a long story did not move with any data change; seed spread equal to recipe spread |
| **LFM2.5-230M** (Jun 2026) | 229.7M | 29.2% (**162.6M**) | **19T (82,719)** | 8 conv + 6 GQA layers; SFT distilled from LFM2.5-350M; DPO; RL | IFEval 71.71, Multi-IF 37.70 (3-turn average), tau2-Bench Telecom 5.26 / Retail 13.68 [vendor]; not probed by us; the closest shipped analogue to a 150M target |
| Gemma 3 270M-it (Aug 2025) | 268.1M | 62.6% (100.3M) | 6T (22,380) | Gemma 3 family uses distillation (the 270M's own recipe unsourced) | IFEval 51.2 (Google) vs 27.44 (TII); MT-Bench 3.71 (TII); Google says it is not built for complex conversation; probe 0.31 / 0.27 |
| PleIAs Baguettotron (Nov 2025) | 321.0M | 11.8% (283.2M) | 200B synthetic (623) | 80 layers; thinking traces | MMLU 40 / GSM8K 39 / HotPotQA 51 [vendor; training data carried benchmark signals]; multi-turn with rolling thinking recommended, not required [corrected] |
| Granite-4.0-H-350M / 350M (Oct 2025) | 340.3M / 352.4M | 22.6% / 29.2% | ~15T | SFT + RL + merging | Multi-IF 28.70 / 24.21 (Liquid harness) |
| MobileLLM-350M (2024) | 345.3M | 8.9% (314.6M) | 1T (2,896) | deep-thin, 32 layers | MT-Bench 3.28, AlpacaEval 47.08% (beat OPT-1.3B, 2.24) |
| LFM2-350M (Jul 2025) | 354.5M | 18.9% (287.4M) | 10-11T (~28,000) | top-32 logit KD from LFM1-7B in pretraining; 5.39M-sample SFT; preference pairs from on-policy samples; merging | IFEval 65.12, Multi-IF 32.85 [vendor]; probe 0.60 / 0.50 |
| LFM2.5-350M (Mar 2026) | 354.5M | 18.9% (287.4M) | 28T + RL (78,988) | same architecture, more tokens, multi-stage RL | IFEval 76.96, Multi-IF 44.92 [vendor]; probe 0.62 |
| SmolLM2-360M-Instruct (Oct 2024) | 361.8M | 13.0% (314.6M) | 4T (11,055) | as 135M | MT-Bench 3.66 (HF) / 3.80 (TII); IFEval 41.0; probe 0.57 |
| **Qwen2.5-0.5B-Instruct** (the "wall") | 494.0M | 27.6% (357.9M) | 18T (36,435) | SFT on 1M+ samples, multi-stage RL | MT-Bench 4.16 (HF) / 4.71 (TII); probe 0.60; deflects on 67% of multi-turn user-fact turns |
| H2O-Danube3-500M-chat (Jul 2024) | 513.6M | 19.1% | 4T | | MT-Bench turn 1 4.16, turn 2 2.40 (GPT-4 judge): the clearest published second-turn collapse |
| Falcon-H1-0.5B-Instruct (May 2025) | 521.4M | 12.9% (untied) | 2.5T (4,795) | | MT-Bench 7.06, IFEval 72.07 (TII harness): the best chat numbers in the census, with fewer tokens per parameter than the lanes first claimed as the floor [corrected] |
| nanochat d20 (Oct 2025) | 561.0M | 29.9% (untied) | 11.2B (20) | midtraining + SFT | ChatCORE 0.088: a Chinchilla-budget demo, not an assistant |
| Qwen3-0.6B (Apr 2025) | 596.0M | 26.1% (440.5M) | 36T family figure (60,398) | strong-to-weak distillation, off-policy then on-policy logits | IFEval 54.5, Arena-Hard 6.5, Multi-IF 33.3 (Qwen) / 45.13 (Liquid); probe 0.71 / 0.62 |
| reference: Qwen3.5-0.8B (Feb 2026) | 752.4M text LM | 33.8% (498.1M) | unsourced | | Multi-IF 41.68 (Liquid); MultiChallenge 18.9 thinking (Qwen); card warns of thinking loops |

What the table says:
- **Every model with good chat numbers used 2,900 to 83,000 tokens per parameter** [corrected: the lanes first said 8,800-83,000; MobileLLM-350M at 1T is ~2,900 and Falcon-H1-0.5B ~4,800]. Many also used a teacher or RL:
  - pretraining KD: LFM2, Apertus;
  - distilled SFT: LFM2.5-230M;
  - post-training distillation: Qwen3;
  - distillation across the Gemma 3 family (the 270M's own recipe is unsourced);
  - no teacher described: SmolLM2 and Falcon-H1-Tiny [corrected: the training lane overstated how many were distilled].
- **Max's plan is 670 tokens per parameter** (150M on 100B tokens), or 2,000 with three epochs.
  - In training compute (6ND), Falcon-H1-Tiny-90M is 4.9x Max's 100B-token plan and 1.6x his 300B plan.
  - MobileLLM-125M is 8.3x / 2.8x. SmolLM2-135M is 17.9x / 6.0x. LFM2.5-230M is 291x / 97x.
  - Falcon and MobileLLM-125M are the only published chat recipes within reach (my arithmetic).
- **Embedding share hides how small some "mid-size" tiny models are.** Gemma 3 270M has a 100M body, smaller than SmolLM2-135M's 106M. Qwen2.5-0.5B has a 358M body. A 150M model with Max's 49k vocabulary has a 122M body at d=576 or 112M at d=768. The gap to the "wall" model is 2.5-3.6x in body, not 3.3x in names.
- **Vendors scope sub-400M models away from open chat.** Google says Gemma 3 270M is not built for complex conversation. Liquid recommends the 230M for extraction and on-device agents, not creative writing. MobileLLM-R1 is not a general chat model. Monad has no multi-turn support.
- **Nobody publishes coherence metrics beyond 3 turns at this size**, except tau2-Bench agent completion for LFM2.5 [corrected: the census first said none existed].
- **Not included as chat models:** MobileLLM-R1-140M (math and code SFT, 53% embeddings), and the April 2026 "micro language models" at 8-30M, which generate only the first 4-8 words of a reply for a cloud model to finish (abstract, arXiv 2604.19642).

---

## 3. What the hands-on probe showed

Two probes, both firsthand on this Mac earlier today. Neither was re-run for this synthesis.

- **Chat probe** (`probe/`, audited in `lanes/probe.verify.md`). Each of 8 models ran in its own chat template with its own replies fed back as history: 29 multi-turn conversations (2-12 turns) plus 24 single-turn controls, greedy and 2 sampled seeds. Graders were mutation-tested (14 of 14 mutants killed). The audit still found false passes the fixtures never tried (negated answers, guess lists, hypotheticals, first-person claims in a sentence containing "you"), so it hand-adjusted the scores. Cells are small (1-20 trials); differences under ~0.2 are noise.
- **Likelihood battery** (`capacity_probe/`). The same plain `User:/Assistant:` transcript goes to every model, and the probe scores whether the gold continuation beats a **same-type foil that also appears in the context**, so copying any name from context cannot pass. 32 scenarios per cell, distances 0/4/10 unrelated turns, contexts under 600 tokens.

### 3.1 Three short transcripts

**The "wall" model deflects on a fact it still has** (Qwen2.5-0.5B-Instruct, greedy; distractor turns collapsed):

> **user 0:** I have a cat named Biscuit who loves sitting in cardboard boxes.
> *(4 unrelated turns)*
> **user 5:** What's my cat called?
> **Qwen2.5-0.5B:** I'm sorry, but I'm Qwen, a language model created by Alibaba Cloud. I don't have a cat or any other pets. [...]

When the probe starts the reply as "Your cat is called", " Biscuit" is the top token at every position (log-prob -0.02). The fact is intact; the reply policy reads "my cat" as a question about the assistant.

**A 28T-token model reverts a correction** (LFM2.5-350M, greedy):

> **user 0:** My meeting with Sam is at 3 pm.
> **user 1:** Correction: the meeting got moved to 4 pm.
> **LFM2.5-350M:** That's a good update! The meeting has been moved to 4 pm. [...]
> **user 2:** Why do leaves change color in the fall? *(answered)*
> **user 3:** What time should I show up for my meeting with Sam?
> **LFM2.5-350M:** The meeting with Sam is scheduled for 3 pm. [...]

It acknowledged the correction in its own words, then answered with the stale value one turn later. Forced prefix "You should show up at": " 3" log-prob -0.57 vs " 4" -3.24.

**The 90M model fails differently: no deflection, a template instead of the fact** (Falcon-H1-Tiny-90M, greedy):

> **user 0:** Hello, I'm Marcus. I'm learning to play the guitar.
> *(6 unrelated turns)*
> **user 7:** What's my name?
> **Falcon-H1-Tiny-90M:** Your name is [Your Name]. It's a simple yet meaningful name that reflects your personality and the person you are.

Its other characteristic failure turns "Which city do I live in?" into a listicle ("consider the following factors: ### 1. **Population**"). But it never deflected in 29 final user-fact turns and never looped, and its greedy run recalled what the user had complained about four turns earlier (only two sampled runs of other models did, both in the first person).

### 3.2 Per-ability results, chat probe (greedy, strict graders)

| model | params (body) | recall | follow-up | instruction | correction | own answer | role | topic return | macro (lane) | macro hand-adjusted A / B | single-turn controls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Falcon-H1-Tiny-90M | 91M (74M) | 0.50 | 0.57 | 0.29 | 0.33 | 0.75 | 0.33 | 0.67 | 0.49 | 0.46 / 0.40 | 0.64 |
| SmolLM2-135M | 135M (106M) | 0.10 | 0.29 | 0.36 | 0.00 | 0.33 | 0.00 | 0.00 | 0.15 | 0.15 / 0.11 | 0.46 |
| Gemma 3 270M | 268M (100M) | 0.00 | 0.86 | 0.50 | 0.00 | 0.50 | 0.00 | 0.67 | 0.36 | 0.31 / 0.27 | 0.75 |
| LFM2-350M | 355M (287M) | 0.60 | 1.00 | 0.79 | 0.33 | 0.50 | 0.33 | 1.00 | 0.65 | 0.60 / 0.50 | 0.89 |
| LFM2.5-350M | 355M (287M) | 0.60 | 1.00 | 1.00 | 0.00 | 0.75 | 0.00 | 1.00 | 0.62 | 0.62 / 0.62 | 0.82 |
| SmolLM2-360M | 362M (315M) | 0.70 | 0.86 | 0.79 | 0.00 | 0.67 | 0.33 | 0.67 | 0.57 | 0.57 / 0.57 | 0.54 |
| Qwen2.5-0.5B | 494M (358M) | 0.10 | 0.86 | 0.57 | 0.00 | 1.00 | 0.67 | 1.00 | 0.60 | 0.60 / 0.60 | 0.82 |
| Qwen3-0.6B | 596M (440M) | 0.60 | 1.00 | 0.71 | 0.67 | 1.00 | 0.00 | 1.00 | 0.71 | 0.71 / 0.62 | 0.93 |

Set A flips only clear grader errors; set B also flips borderline "incidental mention" passes. Sampled decoding (card settings) moves macros by at most 0.10 and does not change the picture.

What held up under the audit:
- **The 350M class is level with the 0.5B wall model** (LFM2.5-350M 0.62, SmolLM2-360M 0.57, Qwen2.5-0.5B 0.60). LFM2-350M's lead is not robust (0.65 to 0.50 after regrading) [corrected].
- **Below 300M the order follows recipe, not size.** Falcon (74M body, 800B tokens) 0.40-0.46 > Gemma (100M body, 6T) 0.27-0.31 > SmolLM2-135M (106M body, 2T) 0.11-0.15.
- **About half of multi-turn recall failures are retrieval-intact.** In 20 of 42 (48%), greedy decoding from a forced answer prefix emits the gold fact; in 38 of 42 (90%) gold beats a never-seen foil [corrected from "63%", which included single-turn controls]. In the fixed-history sweep, 111 of 179 (62%). Both are induction-copy tests, since the foil never appears in context.
- **Rendering alone changes recall on the same weights.** SmolLM2-135M-Instruct recalls a user fact 0.15 of the time through its chat template (which injects a default system prompt) and 0.70 as a plain transcript. Hand-strict: about 0.05-0.10 vs 0.45; the base model 0.55. The claim that SFT added skill does not survive the regrade [corrected].
- **Deflection tracks post-training, not size.** Multi-turn final user-fact turns: Qwen2.5-0.5B 67%, LFM2.5-350M 40%, Gemma 40%, SmolLM2-360M 27%, SmolLM2-135M 20%, Qwen3-0.6B 20%, LFM2-350M 7%, Falcon-90M 0% [corrected: the lane's ranking pooled in single-turn controls]. Deflection is one of the two largest failure classes under every labelling, level with plain wrong answers.
- **A 90M model holds a user fact across 12 turns.** Given an identical 1,726-token history, Falcon-H1-Tiny-90M emits the fact from the forced prefix at every distance. Its free answers are 0.65 (short history) and 0.75 (long history) hand-strict, with zero deflections, though it answers as the user in 6 of 40 trials [corrected from 0.80]. SmolLM2-135M's collapse by turn 12 tracks turn count through its chat template, not tokens [corrected: not a memory limit].
- **Corrections fail in 21 of 24 multi-turn trials** across 8 models [corrected: LFM2's "pass" was a hypothetical example]. Qwen3-0.6B passes 2 of 3, so "fails at every size" is overstated [corrected]. Log-prob evidence is mixed by item: the 3 pm to 4 pm item favors the stale value in all 8 models (and has a digit-token confound), the day item favors the correction in 6 of 8, and the color item favors stale in 6 of 8.
- **Role capture appears in 7 to 8 of 8 models from 90M to 600M** (the assistant adopts the user's job) [corrected from 6 of 8]. Qwen3-0.6B answers "What's my name?" with "I'm Marcus."
- **Loops and self-copy concentrate at 135M.** 22% of SmolLM2-135M replies repeat a 4-gram three or more times and 17% reuse half of an earlier reply; its card sampling (T 0.2) does not fix it. Qwen2.5-0.5B's verbatim loop disappears with T 0.7 and repetition penalty 1.1 (cross-turn overlap 1.00 to 0.12). Only 1 greedy and 2 sampled free chats per model.
- **Both ~100M models adopted a system-prompt pirate persona** (they missed only the literal "Arr"), so "~100M models cannot be prompted into chat behavior" is not supported [corrected: the lane said they ignored it].

### 3.3 Per-ability results, likelihood battery (plain transcripts, same-type foils in context)

"Pair" requires both questions of a swapped pair right: chance 0.25, and a "first-mentioned wins" heuristic scores about 0. Latest-wins is pooled over 1-3 corrections at distance 4 and 10 (n=192). Long-tail knowledge is a two-way choice (chance 0.50) on 40 items.

| model | body | owner pair @10 | one-hop pair @10 | two-hop single item, all d | perspective pair @10 | latest wins @4-10 | long-tail closed / open book |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M base | 106M | 0.22 | 0.22 | 0.54 | 0.12 | 0.02 | 0.78 / 1.00 |
| SmolLM2-135M-Instruct | 106M | 0.62 | 0.38 | 0.51 | 0.09 | 0.00 | 0.78 / 0.95 |
| Gemma 3 270M-it (d0 only, partial) | 100M | 0.47 @0 | 0.69 @0 | 0.55 (n=76) | 0.94 @0 | 0.00 (n=18) | 0.95 / 0.98 |
| LFM2-350M | 287M | 0.34 | 0.34 | 0.66 | 0.09 | 0.11 | 0.85 / 1.00 |
| LFM2.5-350M | 287M | 0.47 | 1.00 | 0.86 | 0.00 | 0.00 | 0.85 / 1.00 |
| SmolLM2-360M-Instruct | 315M | 0.94 | 0.59 | 0.63 | 0.91 | 0.00 | 0.88 / 0.98 |
| Qwen2.5-0.5B-Instruct | 358M | 0.75 | 0.66 | 0.72 | 0.12 | 0.00 | 0.85 / 0.98 |
| Qwen3-0.6B | 440M | 1.00 | 1.00 | 0.99 | 0.97 | 0.00 | 0.80 / 1.00 |
| LFM2-2.6B | 2,435M | 1.00 | 1.00 | 0.94 | 0.97 | 0.72 | 1.00 / 1.00 |

Source: `diagnosis_capacity.md` section 4, recomputed from `capacity_probe/out/*.jsonl`. Falcon-H1-Tiny-90M is missing.

- **Corrections are the universal failure.** In every model up to 0.6B, the wrong answer that wins is the *original* value, not an intermediate one. Mean margins for one correction at distance 10 run from -0.9 to -8.8 nats; LFM2-2.6B is +1.8.
- **Surface-form control.** When the original and every correction use identical wording, SmolLM2-135M and 360M still pick the original in 192 of 192 items at distance 4 and 10. With a neutral answer prefix ("That would be") they score 0.00-0.06. At distance 0 the 360M gets 0.97-1.00. This is primacy, not n-gram copying and not frequency.
- **Binding has a size gradient below 300M body, but recipe moves it as much.**
  - Same data family: SmolLM2 135M to 360M moves owner pair 0.62 to 0.94 and perspective 0.09 to 0.91.
  - Same size, different recipe: LFM2 to LFM2.5 at 287M body moves one-hop 0.34 to 1.00.
  - Size order is not monotonic across families: Qwen2.5-0.5B is below SmolLM2-360M on owner and perspective binding.
- **Reading works at 106M; storing is scarce.** SmolLM2-135M gets 0.78 on long-tail facts closed-book and 1.00 when the user stated them two turns earlier. Context fixed all 9 of its closed-book errors and broke none.
- **Oddity.** Gemma 3 270M (100M body plus a 168M embedding table) scores 0.95 closed-book on long-tail facts, above every 287M-440M body. n=40, so this is a hint that large lookup tables can hold entity knowledge, not a finding.

Caveats that apply to all of section 3:
- Likelihood read-outs are not generation.
- Plain transcripts are off-template for instruct models and understate heavily post-trained ones (LFM2.5-350M gets 1 of 32 on a single correction even at distance 0).
- No probed model has a body in the 150M target range (122-141M).
- Nothing runs beyond 12 turns, and there are no human ratings.

---

## 4. The current best recipe at ~150M, component by component

"Strength" is the evidence behind the choice at *this* scale. Scale is listed per claim.

### 4.1 Architecture

| choice | evidence and scale | strength |
|---|---|---|
| **Keep softmax attention** in all or most layers (GQA fine); a hybrid that keeps attention is acceptable, an attention-free stack is not | 70M attention beats a 1.4B gated conv on associative recall (Zoology, 70M-1.4B, 10-50B tokens); transformers beat Mamba at copying when the phone book has at least 70 entries [corrected scope] (Pythia-410M vs Mamba-2.8B); Hymba at 300M/100B: parallel attention+SSM heads beat plain Llama on recall tasks (49.90 vs 39.98) [omission added by fact-check]; our probe: Falcon's hybrid retrieves at 1.7k tokens | strong (keep attention); weak (hybrid vs pure) |
| **Shape**: ~24-32 layers at d=576-640 is safe; do not over-invest | MobileLLM at 125M: about +1 zero-shot point from depth, flat beyond ~24 layers (0.25T tokens, single seed); counterexamples: LFM2.5-230M chats with 14 layers at d=1024, MobileLLM-R1-140M uses 15 layers; the 2-hop failure at 106M body is not a depth shortage (SmolLM2-135M has 30 layers vs LFM2.5's 16) | moderate, contested |
| **Max's validated block**: per-head gated attention, normalized value residual, 1/sqrt(depth) norm scaling, QK-norm, NorMuon + cautious weight decay | Max's 113M A/B at 1.0B tokens: **4.5% (0.134 nats) on paired training loss** [corrected from 6.9% held-out, which carries about ±0.1 nats of eval noise; the "NorMuon behind then ahead" flip is refuted as eval noise]; ResFormer value residual +16% parameter-equivalent at 82M-468M/20B; both gate and value residual inside noise in a 1.2B controlled replication [>=1B only] | moderate |
| **Previous-token mixing** (Canon layers or a short causal conv, kernel ~4) | 2-4x synthetic reasoning depth at 50M-124M-class sizes; real-data gains only at 1B-8B (+0.4 to +1.8 average) [>=1B only]; mechanistically relevant to binding (multi-token keys need a previous-token path) | weak at 150M; worth an A/B |
| Output head rank (d < 1000) | late saturation in Pythia <=410M; a causal test finds compression but no demonstrated optimization harm; no link to chat behaviour | unknown; do not spend on it first |
| Looping / block sharing | MobileLLM-LS +0.7 at 125M for 2x FLOPs; looping does not raise knowledge capacity (Ouro, synthetic 1-40M); Mixture-of-Recursions loses to vanilla at 135M iso-FLOP; each loop worth r^0.46 of a unique block (30M-1.1B) | weak; not a knowledge lever |
| Context | 2-4k training context is enough for chats of this length; KV cache for a 30-layer, 3-KV-head 150M is ~94 MB at 4k tokens (arch lane) | moderate |

### 4.2 Tokenizer and vocabulary

- **The 49,152 vocabulary costs 28.3M (19%) at d=576 or 37.7M (25%) at d=768.** Moving to 16-32k frees 9-19M of body (7-16% more). That is small next to the 2.5-3.6x body gap to the 0.5B class (my arithmetic).
- **In-scale evidence** (Tao et al., measured directly at 302M non-vocab params): 16K is compute-optimal, 24K with excess data [corrected: the lane extrapolated from a predicted 3B row instead].
- **On OASST chat:** going from 8k to 49k cuts tokens per conversation by 13%; going past 49k saves only 0.7-4.3% [corrected from 2-5%].
- **Out:** plain byte-level (about 13% worse bits per byte at 1e19 FLOPs [corrected from 19%]) and SuperBPE (worse than BPE at every budget in a 1,308-model study).
- **The real decision is the teacher (section 8, D6).**
  - Keeping 49k keeps MaxGPT-Ultra as a same-tokenizer teacher.
  - SmolLM2's tokenizer has the same size but different merges; adopting it unlocks SmolLM2-360M and 1.7B as same-tokenizer teachers.
  - A small in-domain vocabulary maximizes body parameters but leaves only sequence-level or cross-tokenizer distillation, which is unproven below 1B.

### 4.3 Data

| choice | evidence and scale | strength |
|---|---|---|
| **Put dialogue into pretraining.** Max's 100B build (55% FineWeb-Edu, 22% Cosmopedia, 10% code, 8% OpenWebMath, 5% Wikipedia) has almost none, and web text is only 2.7-3.7% conversational | Falcon-H1-Tiny at 90M: 25% SFT data in all 800B pretraining tokens raised IFEval 53.47 to 66.08 after DPO, **but the classic pipeline was slightly ahead on 2-turn MT-Bench (4.40 vs 4.33) and AlpacaEval** [corrected: an instruction lever, not a proven chat lever]; TII's sweep at 100B total tokens (Max's budget) found 25% and 50% tied and 75-100% worse; a controlled Pythia 70M-1B midtraining study found generic FLAN instructions had minimal effect, while KnowledgeQA+dialogue at 20% lowered LIMA SFT loss at 160M (3.736 vs 3.828) | moderate for instruction following; weak for multi-turn coherence |
| **Dense multi-turn signal**: back-references, same-type distractors, corrections with stale traps, answer-mentioned-second as often as first, short assistant turns | UltraChat has 1.45 context-dependent queries per session vs 4.62 in real ShareGPT logs; smol-smoltalk averages ~2.3 turns; Parrot-style data alone gave +0.18 to +0.24 MT-Bench at 13B [corrected: the +0.41-0.47 included CaPO] [>=1B only]; TurnWise +12.8 multi-turn points at 7B with IFEval and MMLU regressions under SFT [>=1B only]; the critical-mixing-ratio result (Gu et al., Pythia 70M/410M) says at small size a skill below a density threshold gets **no** capacity | mechanism moderate; effect at 150M untested |
| **Knowledge-light and grounded**: answers from the user, the history or a retrieved passage; explicit "I don't know" | entity-anonymized pretraining at 135M/360M (2.5B tokens): closed-book LAMA 12.5 to 0.7, FEVER 82.8 to 89.5 after per-benchmark fine-tuning (single preprint, no code); source tags cut junk-dilution loss of knowledge capacity from 20x to 2x (synthetic) | moderate hint |
| **Repetition is cheap** | up to ~4 epochs is nearly as good as unique data (10M-9B params); a 100M model forgets a seen sample within 1-5B tokens (one TII workshop study, math only), so a curated 0.3-3B-token dialogue pool can recur across a 100B run | moderate |
| **Generator choice**: open-weight, Apache-2.0 teachers for bulk data | Anthropic's consumer terms prohibit using the service to develop competing products, including training models [corrected wording]; Max measured Gemma 4 12B QAT at ~14 tok/s single stream on the M5 (session note, `tools/gemma_chat.py`); batched 8-12B dense 4-bit on the base M5 is ~45-70 raw tok/s, i.e. **~20-50 Mac-days per 100M accepted tokens** [corrected from 4-12 days] | strong on rights; throughput anchor weak |

### 4.4 Training and distillation

- **Tokens.** 100B tokens is 667 tokens per parameter; three epochs of the same build give 2,000.
  - A 150M model keeps improving to at least 10,000 tokens per parameter, though more slowly than scaling laws predict (Sardana et al., 150M to 1.5T tokens).
  - Chinchilla-style fits put 150M at 300B within ~0.07-0.09 nats of the same model at 2T, and most of the gap to 0.5B is the parameter term. Low confidence: 2022 fits, loss is not chat quality.
  - **Recommendation: 300B token-passes if the Titans are available** (strong for "more helps", weak for "how much").
- **Schedule and hyperparameters.** WSD with a linear decay to zero (beats 10x decay, 610M), weight decay re-derived for the tokens/param regime, batch size growing with data. Moderate, cheap.
- **Optimizer.** Tuned-baseline benchmarks give matrix optimizers ~1.4x over AdamW at 0.1B, with SOAP/Kron overtaking Muon at 8x Chinchilla and above. Max's AdamW arm used lr 6e-4, which Wen et al. flag as under-tuned. Re-validate at high tokens/param with per-arm learning rates. Moderate.
- **Pretraining logit distillation: weak at this size.**
  - Controlled nulls: MobileLLM 125M with a 7B same-tokenizer teacher scored 43.9 vs 43.8 at 2.6-3.2x the cost; MiniPLM at 200M scored 39.9 vs 39.9 compute-matched; JugnuLM-110M dropped top-16 KD from SmolLM2-1.7B.
  - Busbridge's law, extrapolated below its fit range, gives an Ultra-like teacher about -0.04 nats token-matched and ~0 compute-matched at 100B student tokens. A 1.7B/11T teacher crosses to harmful near 350B student tokens [corrected from 300B].
  - **Better uses of Ultra:** MiniPLM-style difference sampling (+1.4 average at 200M compute-matched; scoring costs ~2.4e20 FLOPs [corrected]), and offline random-sampling KD on a decay-phase slice (~10 Titan-days and 1.8 TB for 50B tokens).
- **Weight inheritance or pruning from Ultra: speculative.** Pruning MobileLLM-350M to 125M came out mixed (Adapt-Pruner). Sub-network initialization's advantage decays from 5.16x at 10B tokens to 1.26x at 100B (~410M students).

### 4.5 Post-training

| choice | evidence and scale | strength |
|---|---|---|
| **Pack SFT by whole conversation with cross-document attention masking** | Max's packer (`posttrain/sft_data.py`) slices fixed windows: at seq_len 2048, 42.5% of supervised assistant tokens have their conversation start cut off, 57.4% see an unrelated conversation, ~0.1% see a clean history (simulated on 1,100 UltraChat rows, `recipe_work/pack_sim.py`) | mechanism certain; effect size unmeasured; zero cost |
| **Student-sized multi-turn SFT**: short replies, no long chain-of-thought, no tool calling, deflection and identity boilerplate removed, no fixed default system prompt | SmolLM2 filtered hard and function-calling data to match the models' capacity; the probe's template/persona effect (0.15 vs 0.70 on the same weights); TII: chain-of-thought caused loops in its 90M tool-calling model [corrected: tool-calling specific]; small students learn math better from short chains (0.5B-3B) | moderate |
| **One epoch of DPO**, with multi-turn pairs whose rejected side is a programmatic failure (stale value, deflection, answering as the user, a loop) sampled from the model's own rollouts | 90M: one DPO epoch raised 2-turn MT-Bench ~3.1 to ~4.3-4.4 in both arms, more than one epoch degraded (lr 1e-6 to 3e-6); 151M (ufakzeka-1): three preference variants lowered quality; Max's current DPO set is 100% single-turn | moderate for one epoch; multi-turn pairs untested |
| **On-policy distillation (OPD)** as the final stage | MiniLLM at 120M: GPT-4 feedback 38.6 (SFT) to 44.7; exposure-bias error stops accumulating at 125M; gain is non-monotonic in size (+0.3 at 340M, +4.0 at 760M) [corrected: not "largest at smallest"]; MobileLLM-R1.5: 140M GSM8K 4.1 to 8.3, 360M 24.5 to 52.8 (math, 4B teacher); multi-turn OPD below 1B exists only for agent tasks (Guided-OPD, 0.6B: teacher signal degrades at later turns, mixing in teacher turns on a decaying schedule helped) [corrected: the lane said >=1.7B only] | moderate single-turn; hypothesis multi-turn |
| **Unlikelihood on self-repeated n-grams and contradiction pairs** | label repetition 0.617 to 0.055, contradiction preference 18% to 78% in a ~90M-class seq2seq dialogue model (2019-2020) | moderate, old architecture |
| RL (GRPO-style) | no evidence it helps open chat below 1B; nanochat RL +3.0 GSM8K points in-domain [corrected: "weakest lever" overstated] | weak for chat |
| Keep the SFT checkpoint; soup SFT, DPO and OPD checkpoints; low fine-tuning LR with replay | heavily overtrained small models are more fragile to fine-tune (OLMo-1B beyond ~2.5T; controlled 15-90M runs) | moderate, cheap |

### 4.6 Decoding

- **Never pure greedy for chat; T 0.5-0.7, with a mild repetition penalty only as a stopgap.**
  - The penalty fixes Qwen2.5-0.5B's verbatim loop (probe) but not SmolLM2-135M's at its card T 0.2.
  - Distilled students loop far more than their teachers at low temperature (1.5B-32B reasoning models) [>=1B only], while the LFM2.5 and Falcon cards default to T 0.1.
  - Moderate.
- Contrastive search suffers on anisotropic GPT-2 small/medium. Min-p's claimed gains did not survive reanalysis. Context-aware decoding (full-history vs last-turn contrast) improved faithfulness on OPT 125M-30B but is untested for multi-turn. Weak.

### 4.7 Memory and tools

- **Within-conversation memory: attention is enough for a unique fact at ~2k tokens** (Falcon-90M, every model with ~290M+ body). Do not spend the novelty budget on memory architecture for chats this long. Strong.
- **Structured state** (the model writes a short "current facts" line each turn, or the harness maintains a state card): untested in a general tiny chat model. Hypothesis; see section 6.
  - SimpleTOD shows belief-state tracking is learnable at 82M (MultiWOZ joint goal accuracy 54.5). That is cumulative slot-filling, not corrections [corrected].
  - One-output-for-all-slots decoding is more than 5% worse than per-slot decoding for T5-small/base.
- **Retrieval must be trained in.**
  - Off-the-shelf SmolLM2-360M-Instruct scored 0% exact match on unknown questions even with the gold passage.
  - Retrieval-trained BART-large (400M) cut hallucination from 68.2% to 7.9% on Wizard of Wikipedia (human eval).
  - Full-conversation retrieval queries beat last-turn queries, and 1-5 passages beat 25 at 400M.
  - Cross-attention retrieval barely pays at small scale: in Retro-li at 109M, retrieval itself bought 0.14 perplexity [corrected].
  - Recommendation: in-context retrieval with a trained reader. Moderate.
- **Sparse memory (only if "150M" means active parameters).**
  - Memory+ on a 134M base (937M total, 1T tokens) matched a dense 373M on TriviaQA (F1 18.77 vs 17.68).
  - But at a 151M base, MoE beat product-key memory on TriviaQA (UltraMem), and n-gram tables help least for narrow, deep models (LongCat, 280M active).
  - Mixture of Lookup Experts at 160M active / 100B tokens: average 38.8 to 41.9.
  - Ordering is contested; moderate for knowledge QA, unmeasured for chat.
- **Tools: supervised traces work at small size for single calls; multi-turn tool use does not yet.**
  - TinyGSM 125M reaches 63.1% GSM8K via executed Python.
  - Falcon's 90M tool model scores BFCL v3 41.23 overall but 0% multi-turn (so does FunctionGemma-270M); its 94% relevance score rests on 18 items [corrected].
  - Moderate for single calls.

---

## 5. The limits map

### 5.1 Every limit

| # | limit | mechanism | kind | evidence (scale) | cheapest decisive test |
|---|---|---|---|---|---|
| 1 | **Updating after a correction** | Key-based retrieval finds every statement of the slot and small models resolve toward the earliest ("active primacy intrusion"); pretraining text rarely overwrites a stated value and single-turn SFT never does; the model's own acknowledgements add copies of both values | unknown (not size-driven within 135M-0.6B) | likelihood: 0.00-0.11 at d4/d10 for 8 sub-1B models, n=192 each; same-wording control 0/192 at 135M and 360M; LFM2-2.6B 0.72 (section 3.3); generation: 21 of 24 fail. Size: 39 models of 1B-2.5T show proactive interference in all, size predicts retroactive but not proactive resistance [>=1B only]; bAbI dialog task 2: small memory networks trained from scratch reach 100% on updated requests | **`ft_test.py`** on SmolLM2-135M-I then 360M-I (400 steps, disjoint templates, one process at a time): at least 0.8 latest-wins at d10 on held-out wording = trainable; flat at 135M but moving at 360M = threshold in Max's range; flat at both = needs pretraining-scale signal or a state mechanism. Add a position control (original after 3 distractor turns). To locate any size threshold inside one recipe, run the battery on LFM2-700M and LFM2-1.2B (350M fails, 2.6B passes) |
| 2 | **Binding under same-type interference** (owner, speaker, two-hop) | With two same-type values in context the model must match a multi-token key (owner plus value), not a one-token induction key; small models fall back on "first-mentioned wins"; binding representations sharpen with scale; at small size the circuit is learned only if its training signal is dense enough | mixed | two-hop at chance (0.51-0.55) in all ~100M bodies; SmolLM2 135M to 360M: owner pair 0.62 to 0.94, perspective 0.09 to 0.91; LFM2 to LFM2.5 at fixed 287M: one-hop 0.34 to 1.00; 37.8M transformer learns 4-hop binding with distractors from 450K synthetic programs (no reassignment) | same `ft_test.py` run: two-hop pair at d10 from 0.09 to at least 0.8 and owner/perspective at least 0.85 with closed-book knowledge down no more than 0.05 = signal; training loss near zero while held-out two-hop stays under 0.65 = capacity-limited at 106M body. Plus: score SmolLM2-135M's intermediate checkpoints and Max's 124M A/B checkpoints to see the token curve |
| 3 | **Chat training signal and reply policy** | Small models learn only the conversation their data asks for: 2-3-exchange SFT with few back-references, long teacher-written replies, trained privacy/identity reflexes, single-turn preference data, packing that cuts conversations apart | soft | deflection 0-67% independent of size; 0.15 vs 0.70 recall on the same weights by rendering; Falcon-90M never deflects; Max's pipeline: 0% multi-turn DPO, ~0.1% of SFT tokens with a clean history; counter-evidence: ufakzeka-1's targeted SFT did not move long-story identity tracking at 151M | two equal-token SFT arms on SmolLM2-135M (base and Instruct): generic vs generic + ~5k verified multi-turn dialogues, both packed per conversation, 3 seeds; soft if the targeted arm wins by >15 points on recall, deflection and perspective on every seed. Plus the four Falcon-90M sibling checkpoints to see which stage removes deflection |
| 4 | **Tokens per parameter and capacity allocation** | A bounded model gives no capacity to items or skills below a critical frequency, and that threshold rises as the model shrinks; chat skills are rare in web text | mixed | 2,900-83,000 tok/param for every working small chat model vs 670 planned; critical-ratio phase transition for facts and a reasoning subtask at Pythia-70M (fit 70M-410M); tokens alone do not fix limit 1 (28-36T models fail it) | identical targeted SFT on SmolLM2-135M checkpoints at 252B / 1T / 2T and Pythia-160M at 100B / 300B, 3 seeds; then a 30-60M mixing-ratio sweep (0.5 / 2 / 8% interference dialogues in a FineWeb-Edu stream, 2-5B tokens) |
| 5 | **Degeneration on own history** | Repeated text raises its own probability; off-policy students accumulate error with length; thin training leaves low-entropy attractors; distillation and long chain-of-thought raise the risk | mostly soft | SmolLM2-135M 22% looped / 17% self-copy; Falcon-90M 3% / 0%; MaxGPT-2 "Bitcoin.com" attractors at ~18 tok/param; Daimax SmolLM2-360M 4/33 vs Qwen2.5-0.5B 0/33 (old metric, not significant, p=0.11) [corrected] | run the free-form and own-history chats (greedy + T 0.7 x3) on Falcon-90M `-Instruct`, `-Instruct-Curriculum`, both `-pre-DPO` variants and `-Base`; run the battery on golden vs own history for each model |
| 6 | **Instruction persistence beyond 3 turns** | Attention to a distant instruction decays; compliance decays after turn 1 at ~100M body | mixed | Falcon-90M and SmolLM2-135M decay after turn 1; LFM2.5-350M and Qwen3-0.6B hold over 3 turns; Multi-IF is a 3-turn *average*, not a turn-3 rate [corrected]; at 4B-32B multi-turn constraint persistence falls to ~30% [>=1B only], so a 6-12 turn criterion may sit at the floor for every small model | a 6-12-turn persistence family on SmolLM2-135M vs 360M and the Falcon siblings, with a headroom rule |
| 7 | **Measurement** | Single chat scores mix knowledge with coherence; harness swings 10-24 points; substring graders pass negations; seed variance rivals recipe differences | soft | ufakzeka-1: 3 seeds of one recipe span 75.2-81.2 helpfulness, as wide as 14 checkpoints; a repaired gate read 64/64 on its own templates and 34/64 on paraphrases; Max's A/B eval swings 0.17-0.21 nats between evals; a constant null model scores 9.55 on MT-Bench | train one recipe with 3 seeds and measure the battery spread; regrade existing probe transcripts with the missing fixtures and check rank stability across decoding runs |
| 8 | **Stored world knowledge** | ~2 bits/param after ~1,000 exposures, ~1 bit at 100, up to 20x less with junk dilution; zero for items below a density threshold | **hard** | synthetic GPT-2/LLaMA 1M-0.5B; obscure-fact accuracy +15.9 points per 10x params with no fixed-size trend over 100 dated models (single-author preprint); TriviaQA 4.1 (SmolLM2-135M), 16.9 (360M), 4.3 (Qwen2.5-0.5B): a slope, not a step at 400-500M | none needed for the limit; for the attack: a counterfactual open-book likelihood probe (a stated plausible-but-wrong fact) and a grounded-dialogue fine-tune measuring abstention on unstated facts |
| 9 | **Teacher signal** | The best sub-400M models were teacher-fed; Max's only same-tokenizer teacher is a 1.1B model at ~91 tok/param | mixed | pretraining-KD nulls at 125M and 200M; post-training KD positive at 120M | after Ultra: a 10B-token 150M pair, CE vs CE + offline RS-KD from Ultra, on paired training loss and the battery; before then, an on-policy SmolLM2-1.7B to 135M pilot (same vocabulary) |
| 10 | **Embedding share** | Embedding parameters add no serial computation | soft (design choice) | 19-25% at 150M with a 49k vocab; Gemma's 63% table did not stop it having the best small-model closed-book score | a 16k vs 49k A/B at equal total params only if the teacher plan changes |
| 11 | **Output-head rank and depth** | Head rank <1000 compresses output and gradient; depth for k-hop is ~log | unknown / not binding | contested causal evidence; 30 layers ample | a 60M head-rank A/B judged on the battery and loop rate, low priority |
| - | **Not a limit: one unique fact across ~12 turns** | Induction/retrieval heads copy a unique fact cheaply when real attention is kept | - | Falcon-90M, every model with ~290M+ body; reading a stated fact at 106M | Tier-0 recall probe on Max's own 124M A/B checkpoints |

### 5.2 Where the three diagnoses disagree, and how this report resolves it

1. **Top rank.** Capacity puts knowledge first, recipe puts training signal first, behavior puts updating first. They use different criteria:
   - capacity: most firmly bound by parameter count;
   - recipe: most binding limit one person can remove;
   - behavior: most likely to fail a strict multi-turn test.

   This report uses the behavior criterion because Max's stated goal is multi-turn coherence, and it carries the other two views as the "kind" column. Knowledge is the only hard limit and would be rank 1 under a knowledge-inclusive definition of chat.
2. **Is updating the 150M-specific limit?**
   - Capacity: almost certainly not the 150M-vs-500M difference.
   - Recipe: the likeliest place a real parameter limit could sit near 150M.
   - Behavior: universal gate, leaning soft.

   The data are not in dispute (all sub-0.6B fail, LFM2-2.6B passes, no gradient from 135M to 0.6B). Resolution: updating does **not** explain why 150M looks worse than 500M, but it **is** the gate between any sub-1B model and "really chatting". Whether it has a size threshold between 0.6B and 2.6B or is a missing training signal is unknown. `ft_test.py` plus an LFM2-700M/1.2B run settles it.
3. **Binding and updates: one limit or two?** Recipe merges them as "same-type interference, first-mentioned wins"; capacity and behavior separate them. The data separate them:
   - binding has a same-recipe size gradient from 135M to 360M and some recipes solve it at ~300M;
   - updates have no gradient from 135M to 0.6B and no recipe solves them.

   They share the first-mentioned shortcut, so one data attack (interference-balanced dialogues) targets both. This report lists them separately.
4. **Mention-order numbers.** For SmolLM2-135M base, recipe reports 0.98 (answer mentioned first) vs 0.31 (second); behavior reports 0.83 vs 0.40. I recomputed both from the saved files. Recipe's split uses only the "my cat" question (n=42 / 54); behavior pools both questions (n=96 each). Both are correct, so there is no conflict.
5. **Deflection.** Recipe ranks it first (inside training signal), behavior sixth ("largest class, but cheap and under Max's control"), capacity not at all. Resolution: the largest failure class in today's small models, soft, and the cheapest fix. Ranked 3rd here, inside the training-signal limit.
6. **Tokens.** Recipe treats tokens as a separate limit, capacity as "capacity allocation", behavior as what makes binding and persistence expensive. Same phenomenon. The critical-density mechanism explains why data density and token count trade off.
7. **Degeneration** is rank 3 for behavior and rank 5 for recipe. Placed 5th here: Falcon removes it at 90M, but Max's likely use of distillation raises the risk, so it stays on the list.
8. **Falcon's "MT-Bench".** Two lanes called it the many-turn MT-Bench-101. The census fact-check found it is 2-turn MT-Bench via evalchemy, with the judge undeterminable (GPT-4 or GPT-4o-mini). The arch fact-check's "most likely gpt-4o-mini" was withdrawn by the census re-check. This report uses 2-turn, judge undetermined.
9. **The lost token-curve lead** (SmolLM2-135M second-entity binding 0.04 / 0.23 / 0.56 at 252B / 1T / 2T) appears in two diagnoses as a hint. Its raw data no longer exist, so it is not used here except as a reason to re-run the measurement.

---

## 6. Where a wall-break could come from

All of these are **hypotheses**. "Wall-break" here means a capability-per-parameter result that holds on a curve (for example 30M / 60M / 150M), judged on the sealed multi-turn battery against the same-size baseline, not a single lucky run. Ranked by (size of the limit attacked) x (plausibility) / (cost).

| rank | lever | limits attacked | why it could work | prior art and gap | cheapest test |
|---|---|---|---|---|---|
| 1 | **Interference-balanced, distance-balanced dialogue in pretraining**, above the critical density, not only in SFT: every correction paired with a no-correction twin and a two-slot variant; answer mentioned second as often as first; fact-to-question gaps sampled uniformly; turns where the assistant itself repeated the old value | 1, 2, 4 | the critical-ratio result says small models skip rare skills entirely; removing both "first wins" and "last wins" shortcuts forces real binding and updating | bAbI dialog task 2 (updates, small memory nets), Wu et al. (37.8M, binding), Falcon's SFT-in-pretraining (IFEval only); never tested as a pretraining-mix intervention in a chat LM | `ft_test.py` first; then a 30-60M mixing-ratio sweep (0 / 0.5 / 2 / 8%), 3 seeds |
| 2 | **A model-written running state line** ("user: Priya; sister: Lena, dentist; appt: Thu") emitted each turn, kept in history (only the latest), trained so answers copy from it | 1, 2, 6 | turns binding, two-hop and updates into a one-hop copy from the most recent block, which small models already do well; the model rewrites the slot, so primacy stops mattering | SimpleTOD / SOLOIST (belief state at 82-117M, task dialogue); T5 DST shows single-sequence state decoding costs >5%; not found in a tiny general chat model measured on recall and corrections | at 30M and on SmolLM2-135M: none vs state line, same data, 3 seeds, on held-out templates |
| 3 | **A recency or overwrite path in a few attention layers**: ALiBi-style recency slopes, a forget gate (Forgetting Transformer, gated delta rule), stick-breaking; plus checking whether per-head gating (sink removal) already reduces primacy | 1 | primacy intrusion looks like a winner-take-all attention bias toward early encodings; attention sinks may amplify it | gated attention cut first-token attention 46.7% to 4.8% at 1.7B [>=1B only]; stick-breaking beats softmax on RULER variable tracking at 1B, but that is chaining, not reassignment [>=1B only]; nothing at 150M | free now: run the battery on Max's 124M A/B checkpoints with vs without the gate; then a 60M A/B |
| 4 | **Previous-token mixing (Canon / short conv) plus speaker-role embeddings** | 2 | multi-token keys ("Lena works as a") need a previous-token path; role embeddings separate I / you / third party | Canon synthetic at 50-124M, real only >=1B; TransferTransfo role embeddings at 117M | 60M A/B on the binding and perspective families |
| 5 | **Multi-turn on-policy distillation and own-history SFT** (the student writes every assistant turn; teacher scores tokens; early training mixes in teacher turns) | 5, 1 | directly trains recovery from its own drift, which gold-history SFT never does | MiniLLM 120M single-turn; Guided-OPD 0.6B agent tasks; conversational multi-turn OPD only >=1.7B | pilot SmolLM2-1.7B-Instruct to SmolLM2-135M (identical vocabulary) now; Ultra later |
| 6 | **Knowledge offload**: retrieval-trained reader, knowledge-light pretraining, trained abstention | 8 | spends parameters on skill instead of facts; reading from context already works at 106M | RETRO / BART-large grounding; KLLM at 135M (preprint); [IDK]-token training diverged at Pythia-70M/160M (use SFT abstention examples instead) | grounded-dialogue fine-tune of SmolLM2-135M with distractor and conflicting passages; measure abstention |
| 7 | **Sparse memory tables** (Memory+, lookup experts, hashed n-gram embeddings), valid only if the size constraint counts active parameters | 8 | lookup, not compute, holds facts | Memory+ at 134M; MoLE at 160M active; contradicted at 151M by UltraMem | only after D1 is decided |
| 8 | **A large embedding table as an entity store** | 8 | Gemma 3 270M's 100M body with a 168M table had the best small-model closed-book score | one n=40 observation | a vocab-size A/B at equal body, scored on the khard closed/open twin |

Not candidates:
- **Looping for knowledge:** it does not raise capacity.
- **Multi-token prediction:** exact MTP hurts at 300M-600M, and its induction benefit vanishes above ~30M non-embedding.
- **Tokenizer tricks:** SuperBPE and byte-level lose at this budget.
- **Test-time thinking:** tiny models loop, and thinking eats the context that *is* the conversation. Max already ruled it out for this project.

---

## 7. Feasibility on Max's hardware

### 7.1 What is available

- **Titans (Lambda, 10x TITAN RTX 24 GB, Turing sm_75).**
  - Capacity: only the 4 edge cards sustain load (cards 6 and 8 have dead cooling, middle cards throttle).
  - Stack: fp16 with loss scaling only, no bf16, no FlashAttention-2. Driver 470 pins CUDA 11.8, torch 2.7.1 and vLLM <= 0.10.1.1 on the V0 engine; that vLLM predates Qwen3.5 and Gemma 4. SGLang needs CUDA 13. A driver >= 525 (ideally >= 580) from IT would unlock current stacks.
  - Schedule: Ultra's pretraining ends about Dec 3; cards free around Dec 7-14 after its SFT/DPO.
  - This report assumes the Titans are **unavailable until mid-December** and that nobody touches the Ultra job.
- **RTX 5070 12 GB (Blackwell, bf16 and fp8).**
  - Status to confirm: the brief says it is lent out, but Max's own note from this afternoon says it is free again.
  - Expected rate for the 150M reference at T=2048: ~32k tok/s (range 24-40k), roughly one healthy Titan. **Unmeasured**; one `bench_micro.py` run settles it.
  - fp8 is unlikely to help at d=768 (published gains are at 8B+).
- **M5 Mac, 24 GB.**
  - Measured training rates: ~2.7k tok/s at 113M; ~10.6-11.8k at 18.5M [unverifiable: a re-run under GPU contention got about half].
  - The default GPU working set is about 16 GiB, and LM Studio has been holding a ~9.7 GB 12B model resident.
  - **Hard rule: one model process at a time, nothing else resident.** The crash that killed the first attempt came from three concurrent loads.

### 7.2 What things cost

- **Pretraining the 150M reference** (1.246 GFLOP/token at T=2048) on the 4 edge Titans at the measured AdamW rate (~68k tok/s combined): 100B in ~17 days, 300B in ~51 days, 1T in ~169 days.
  - The NorMuon recipe's overhead on a healthy card is unmeasured (its 29k tok/s median came from throttled cards), so budget up to ~1.5x longer.
  - On the 5070 alone: ~36 days per 100B (estimate).
  - SmolLM2-135M's 2T would take ~11 months of the 4 cards. Parity with the labs is out.
- **Ablations:** per GPU-week on one healthy Titan or the 5070, roughly 38-50 runs of 60M on 1.2B tokens, or 6-8 runs of 150M on 3B tokens (compute lane estimate).
- **Data generation:**
  - Mac: ~20-50 Mac-days per 100M accepted tokens with an 8-12B dense teacher, so pilot 5-20M tokens first.
  - 5070 with a current vLLM: on the order of 1e9 tokens/week for a 1.7-4B teacher (roofline, low confidence).
  - Titans: stuck on old vLLM until the driver changes.
- **Offline KD from Ultra:** ~10 Titan-days and ~1.8 TB per 50B tokens of random-sampling logits. Top-32 logits would be 6.4 TB and does not fit on the ~3.1 TB free.

### 7.3 The fastest trustworthy loop

1. **Fix measurement first: days, Mac, no training.**
   - Freeze the chat probe plus the likelihood battery as a sealed set with held-out templates and paraphrases.
   - Add knowledge-free vs knowledge-bearing splits.
   - Harden the graders with the fixtures the audit listed (negation, guess lists, hypotheticals, first person with "you", incidental numbers, truncation).
   - Score every model in its own template and in plain format. Use a fixed eval token set and paired metrics for loss.
   - Rule: any trained comparison gets 3 seeds.
   - The likelihood battery doubles as a **Tier-0 pretraining probe**: minutes per checkpoint, immune to reply policy, and it separates "cannot bind" from "will not say".
2. **Run the three missing cheap experiments: hours, one model at a time.**
   - (a) `ft_test.py` on SmolLM2-135M-I, then 360M-I (on the Mac with LM Studio unloaded and batch 8, or on the 5070).
   - (b) The five Falcon-H1-Tiny-90M checkpoints (Instruct, Instruct-Curriculum, both pre-DPO, Base) on CUDA or CPU. MPS crashed on the Mamba path.
   - (c) `uprobe` on the remaining models plus LFM2-700M/1.2B.
   - (d) The battery on Max's 124M A/B checkpoints, with and without the gate.
3. **Cheap fine-tuning arms on public checkpoints: days, 5070 or Mac.** The targeted-vs-generic SFT test for limit 3 (3 seeds), and identical targeted SFT on SmolLM2-135M intermediate checkpoints for limit 4.
4. **From-scratch pairs at 30-60M: 5070, 1-2B tokens each, 3 seeds.** Mixing-ratio sweep, state line vs none, gate vs none. This is where a curve starts.
5. **Titans after mid-December.** 150M on the winning mix at 100B, extended to 300B if step 3 shows token-limited skills. Then the fixed packer, at most one epoch of multi-turn DPO, and on-policy distillation from post-trained Ultra.

Steps 1-2 need no new hardware and can finish this week. They decide limits 1-3 and the Falcon recipe question, which drive every later design choice.

---

## 8. Decisions Max needs to make before designing

- **D1. What does "150M" constrain?** Each answer reorders the levers in section 6:
  - Total stored parameters: the strict and most credible count (the Planck notes already favor counting total).
  - Non-embedding body.
  - Active parameters: allows memory tables and MoE, and makes lever 7 the strongest knowledge attack.
  - Bytes on device: allows a larger low-bit model; at equal bytes, ternary gains ~1.7 commonsense points and ~0 on knowledge at 300B tokens.
  - Latency: favors hybrids and shallower stacks.

  Recommendation: count total, report body too.
- **D2. What is the model for?** Conversational mechanics over what is in the context plus retrieval is feasible. An open-domain assistant that knows things is not, at 150M. Recommendation: the former, with knowledge-bearing scores reported separately and always.
- **D3. What does success mean, and against which baseline?** Pre-register before building. Suggested: the behavior lens's RC-12 (12-turn conversations on the model's own history, facts in context with same-type distractors, sealed templates, shortcut rows, 3 seeds, mutation-tested graders).
  - Level R: non-inferior to Qwen2.5-0.5B-Instruct on the composite, paired lower bound at least -3 points. A -3 margin needs about 2,000 conversations for 80% power [corrected: the eval lane's 1,450 gives only ~1,185 effective items].
  - Level A: all criteria, including corrections 4+ turns old and a blind human win-or-tie of at least 50% vs Qwen2.5-0.5B.
  - Baselines, all in one harness: Qwen2.5-0.5B-Instruct (the premise), Qwen3.5-0.8B (current Qwen), LFM2.5-230M and LFM2.5-350M (the strongest small recipe), Falcon-H1-Tiny-90M and its siblings (the closest recipe to Max's budget), SmolLM2-135M/360M.
  - Add a headroom rule so a criterion where every baseline is at the floor cannot be "passed" trivially.
- **D4. Do scaffolds count?** A harness-maintained state card or retrieval: recommend separate, labelled conditions, never counted as the model. A state line the model itself writes and reads is part of the model.
- **D5. One size or a curve?** The Planck framing (smallest model that passes) argues for developing at 30-60M and claiming a curve (for example 20M / 60M / 150M). A single 150M point is weaker evidence either way.
- **D6. Tokenizer and teacher (one decision, not two).**
  - (a) Keep the 49k: Ultra is a same-tokenizer teacher; embeddings are 19-25% at 150M and ~60% at 20M.
  - (b) Adopt SmolLM2's 49k: SmolLM2-360M/1.7B become same-tokenizer teachers; about 16 h CPU to re-tokenize.
  - (c) An own small vocabulary: the most body parameters; teachers only through generated text or unproven cross-tokenizer KD.
- **D7. Compute plan.** Wait for the Titans (mid-December), use the 5070 now if it is really free, and decide whether to pause Ultra for data generation (Max's note says he OK'd that; this report does not assume it).
- **D8. Data rights.** Bulk training text from open-weight Apache-2.0 teachers only. Claude for prompt design, rubrics, gold probe sets and reading outputs.

---

## 9. Sources

### Verdict and landscape (sections 1-2)
- LFM2.5-230M card and config: https://huggingface.co/LiquidAI/LFM2.5-230M , https://huggingface.co/LiquidAI/LFM2.5-230M/resolve/main/config.json , blog https://www.liquid.ai/blog/lfm2-5-230m
- LFM2.5-350M: https://huggingface.co/LiquidAI/LFM2.5-350M , https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind
- LFM2 report (Multi-IF 32.85 / 40.92 / 45.28; top-32 KD from LFM1-7B): https://arxiv.org/abs/2511.23404 ; LFM2-350M card https://huggingface.co/LiquidAI/LFM2-350M ; LFM2-2.6B card https://huggingface.co/LiquidAI/LFM2-2.6B
- Falcon-H1-Tiny (IFEval, 2-turn MT-Bench, anti-curriculum, DPO, tool model, sibling checkpoints): https://tiiuae-tiny-h1-blogpost.hf.space/ ; config https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Instruct/resolve/main/config.json ; Falcon-H1 report (evalchemy harness) https://arxiv.org/abs/2507.22448 ; Falcon-H1-0.5B-Instruct https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct
- SmolLM2: https://arxiv.org/abs/2502.02737 , https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct , https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
- Gemma 3 270M: https://developers.googleblog.com/en/introducing-gemma-3-270m/ , https://huggingface.co/google/gemma-3-270m-it , https://huggingface.co/unsloth/gemma-3-270m-it/resolve/main/config.json , https://huggingface.co/google/gemma-3-270m-it/discussions/4
- Qwen2.5 report https://arxiv.org/abs/2412.15115 ; Qwen3 report https://arxiv.org/abs/2505.09388 ; Qwen3.5-0.8B https://huggingface.co/Qwen/Qwen3.5-0.8B
- MobileLLM https://arxiv.org/abs/2402.14905 ; MobileLLM-R1 https://arxiv.org/abs/2509.24945 ; MobileLLM-R1.5 https://huggingface.co/facebook/MobileLLM-R1.5-140M
- Danube3 (turn-2 collapse) https://arxiv.org/abs/2407.09276 ; Granite 4.0 350M https://huggingface.co/ibm-granite/granite-4.0-350m-base
- PleIAs Baguettotron / Monad: https://huggingface.co/PleIAs/Baguettotron , https://huggingface.co/PleIAs/Monad
- ufakzeka-1: https://arxiv.org/abs/2609.25081
- nanochat d20: https://github.com/karpathy/nanochat/discussions/1 ; Supra2-100M https://huggingface.co/SupraLabs/Supra2-100M-Instruct ; PetitGPT https://huggingface.co/yqi0/petitgpt
- BlenderBot (90M, 14-turn human eval) https://arxiv.org/abs/2004.13637 ; DialoGPT https://arxiv.org/abs/1911.00536
- Micro language models (8-30M, first words only) https://arxiv.org/abs/2604.19642
- Multi-IF https://arxiv.org/abs/2410.15553 ; MultiChallenge https://arxiv.org/abs/2501.17399 ; tau2-Bench https://arxiv.org/abs/2506.07982
- Census table and verifications: `research/lanes/census.md`, `census.verify.md`

### Hands-on probe (section 3)
- Chat probe: `research/probe/` (battery.py, analyze.py, transcripts/, exp/, FAILURES.md), report `research/lanes/probe.md`, audit `research/lanes/probe.verify.md`
- Likelihood battery: `research/capacity_probe/` (items.py, khard.py, uprobe.py, out/*.jsonl, results*.json); tabulations in `diagnosis_capacity.md` section 4 and `diagnosis_behavior.md` section 3; mention-order split recomputed for this report (scratchpad `synth/order.py`, reads saved JSONL only)
- Laban et al., multi-turn degradation (>=8B): https://arxiv.org/abs/2505.06120

### Recipe (section 4)
- Architecture: Zoology https://arxiv.org/abs/2312.04927 ; Jelassi et al. https://arxiv.org/abs/2402.01032 ; Hymba https://arxiv.org/abs/2411.13676 ; Canon layers https://arxiv.org/abs/2512.17351 , https://github.com/facebookresearch/PhysicsLM4 ; ResFormer https://arxiv.org/abs/2410.17897 ; controlled 1.2B replication https://arxiv.org/abs/2605.20798 ; gated attention https://arxiv.org/abs/2505.06708 ; softmax bottleneck https://arxiv.org/abs/2404.07647 , https://arxiv.org/abs/2603.10145 , https://arxiv.org/abs/2608.16671 , https://arxiv.org/abs/2602.20433 ; Ouro https://arxiv.org/abs/2510.25741 ; MoR https://arxiv.org/abs/2507.10524 ; looped scaling law https://arxiv.org/abs/2604.21106 ; MTP https://arxiv.org/abs/2404.19737
- Max's A/B: `~/Documents/Projects/Max's AI Model/maxgpt-ultra/docs/ab_2026-09-21/README.md` and metrics files; corrected in `lanes/training.verify.md`, `lanes/compute.verify.md`
- Vocabulary: https://arxiv.org/abs/2407.13623 ; compute-optimal tokenization https://arxiv.org/abs/2605.01188 ; `followup/tokenizer.verify.md`
- Data: conversational share of web https://arxiv.org/abs/2508.10975 ; Parrot https://arxiv.org/abs/2310.07301 ; smol-smoltalk https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk ; TurnWise https://arxiv.org/abs/2603.16759 ; midtraining 70M-1B https://arxiv.org/abs/2510.14865 ; critical mixing ratio https://arxiv.org/abs/2505.18091 ; repetition https://arxiv.org/abs/2305.16264 ; memorization window https://arxiv.org/abs/2607.04969 ; KLLM https://arxiv.org/abs/2607.12831 ; Physics of LMs 3.3 https://arxiv.org/abs/2404.05405 ; Anthropic consumer terms https://www.anthropic.com/legal/consumer-terms ; generation throughput `followup/chatdata.verify.md`
- Training: Sardana et al. https://arxiv.org/abs/2401.00448 ; Chinchilla refit https://arxiv.org/abs/2404.10102 ; D2Z https://arxiv.org/abs/2502.15938 ; optimizer benchmark https://arxiv.org/abs/2509.02046 ; distillation scaling laws https://arxiv.org/abs/2502.08606 ; MiniPLM https://arxiv.org/abs/2410.17215 ; JugnuLM https://arxiv.org/abs/2609.14715 ; RS-KD https://arxiv.org/abs/2503.16870 ; Adapt-Pruner https://arxiv.org/abs/2502.03460 ; Where to Begin https://arxiv.org/abs/2510.07227 ; overtraining fragility https://arxiv.org/abs/2503.19206
- Post-training: MiniLLM https://arxiv.org/abs/2306.08543 ; GKD https://arxiv.org/abs/2306.13649 ; Guided-OPD https://arxiv.org/abs/2606.15912 ; unlikelihood https://arxiv.org/abs/1911.03860 ; small-student CoT https://arxiv.org/abs/2502.12143 ; TinyZero https://github.com/Jiayi-Pan/TinyZero ; SFT packer `~/Documents/Projects/Max's AI Model/maxgpt-ultra/posttrain/sft_data.py`, simulation `research/recipe_work/pack_sim.py`
- Decoding: loops in distilled students https://arxiv.org/abs/2512.12895 ; self-reinforcement https://arxiv.org/abs/2206.02369 ; contrastive search https://arxiv.org/abs/2210.14140 ; min-p reanalysis https://arxiv.org/abs/2506.13681 ; context-aware decoding https://arxiv.org/abs/2305.14739
- Memory and tools: SimpleTOD https://arxiv.org/abs/2005.00796 ; schema DST https://arxiv.org/abs/2109.07506 ; retrieval-augmented dialogue https://arxiv.org/abs/2104.07567 ; small models ignoring passages https://arxiv.org/abs/2603.11513 ; Retro-li https://arxiv.org/abs/2410.00004 ; RETRO reproduction https://arxiv.org/abs/2304.06762 ; Memory+ https://arxiv.org/abs/2412.09764 ; UltraMem https://arxiv.org/abs/2411.12364 ; MoLE https://arxiv.org/abs/2503.15798 ; Over-tokenized transformer https://arxiv.org/abs/2501.16975 ; LongCat embeddings https://arxiv.org/abs/2601.21204 ; TinyGSM https://arxiv.org/abs/2312.09241 ; [IDK] token https://arxiv.org/abs/2412.06676 ; `followup/retrieval.verify.md`

### Limits map (section 5)
- Primacy intrusion (39 models, 1B-2.5T) https://arxiv.org/abs/2603.00270 ; PI-LLM https://arxiv.org/abs/2506.08184 ; flip-flop attention glitches https://arxiv.org/abs/2306.00946 ; bAbI dialog https://arxiv.org/abs/1605.07683
- Binding: https://arxiv.org/abs/2310.17191 ; IOI across Pythia https://arxiv.org/html/2407.10827 ; entity tracking https://arxiv.org/html/2608.18083 ; Wu et al. https://arxiv.org/abs/2505.20896
- Knowledge: https://arxiv.org/abs/2404.05405 ; https://arxiv.org/abs/2505.24832 ; Incompressible Knowledge Probes https://arxiv.org/abs/2604.24827
- Measurement: MT-Bench https://arxiv.org/abs/2306.05685 ; MT-Bench-101 https://arxiv.org/abs/2402.14762 ; null-model gaming https://arxiv.org/abs/2410.07137 ; constraint persistence floor https://arxiv.org/abs/2603.01423
- Diagnoses: `research/diagnosis_capacity.md`, `research/diagnosis_recipe.md`, `research/diagnosis_behavior.md`; follow-up checks `research/followup/floor.verify.md`, `bits.verify.md`, `archfp.verify.md`

### Levers (section 6)
- TransferTransfo https://arxiv.org/abs/1901.08149 ; SOLOIST https://arxiv.org/abs/2005.05298 ; SleepGate https://arxiv.org/abs/2603.14517 ; plus the sources above for each lever

### Feasibility (section 7)
- `research/lanes/compute.md` and `compute.verify.md` (Titan and 5070 rates, Mac benchmarks, vLLM/driver constraints, KD storage); `~/Documents/Projects/Max's AI Model/maxgpt-ultra/LAMBDA.md`; NVIDIA CUDA compatibility https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html ; vLLM releases https://github.com/vllm-project/vllm/releases ; `followup/loop.verify.md` (Mac training loop, Metal working set)

### Claims not used (refuted or withdrawn)
- Titans "340M Transformer++ 98% needle-in-a-haystack": refuted; those numbers are TTT, a recurrent model, on the easiest passkey needle (`lanes/context.verify.md`).
- TII's "multi-turn scores relatively low vs Qwen-0.6B" as chat evidence: refuted; it refers to BFCL multi-turn function calling.
- "Every open small-model recipe is off-policy SFT plus at most offline DPO": refuted (LFM2 on-policy preference samples; MobileLLM-R1.5 on-policy KD).
- "No sub-300M multi-turn evaluation exists" as an absolute: refuted by BlenderBot 90M's 14-turn human evaluation. The narrow form holds: no programmatic coherence probe past 3 turns for a modern sub-300M instruct model.
- "NorMuon trailed, then overtook during decay" in Max's A/B: refuted as eval noise; paired training loss shows NorMuon ahead throughout.
- Induction-head dominance as a loop mechanism (arXiv 2505.13514): withdrawn by its authors.
- "~100M models cannot be prompted into chat behavior": not supported (both adopted the persona).
- Compression target of 4.3-4.8 bytes/token for a small model's tokenizer: refuted as a design target (`followup/tokenizer.verify.md`).
