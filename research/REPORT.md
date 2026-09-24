# What limits a ~150M chat model

Date: 2026-09-23 (final, evening). For Max. Built from the synthesis draft (recovered from the workflow journal) after an adversarial review (`CRITIQUE.md`). Inputs: the ten research lanes (`lanes/*.md`, as corrected by `lanes/*.verify.md`), the seven follow-up fact-checks (`followup/*.verify.md`), the hands-on chat probe (`probe/`), the likelihood battery (`capacity_probe/`), and three independent diagnoses (`diagnosis_capacity.md`, `diagnosis_recipe.md`, `diagnosis_behavior.md`).

No model was loaded, run, fine-tuned or benchmarked for this final pass. It read saved JSON, JSONL and transcripts, fetched primary sources, and did arithmetic.

Markers:
- **[corrected]**: a lane fact-check changed the number or claim; the corrected form is what appears.
- **[corrected in review]**: the adversarial review (or this final pass) changed a claim the draft made.
- **[>=1B only]**: the evidence exists only at 1B parameters or above.
- **[vendor]**: the lab measured its own model in its own harness.
- "My arithmetic" or "estimate": computed by hand, not measured.

---

## 1. Verdict

> **In two minutes**
> 1. **Loose chat: the 400-500M wall is not real.** The 350M class ties Qwen2.5-0.5B on our chat probe, and Falcon-H1-Tiny-90M (74M non-embedding) holds a stated user fact over 12 turns with zero deflections.
> 2. **Strict chat: your premise is too generous.** No model at or below 0.6B passed our strict checks together (corrections a few turns old, not adopting the user's identity, no loops), and nothing published shows strict multi-turn chat below 1B. 0.6B to 2.6B is untested.
> 3. **What separates 150M from 0.5B:** binding same-type entities (mixed), chat data and reply policy (soft), tokens per parameter (mixed), loops (mostly soft), stored knowledge (hard).
> 4. **What separates every tested sub-1B model from strict chat:** updating after a correction (cause unknown) and instruction persistence past 3 turns (unmeasured).
> 5. **Odds (judgment):** matching Qwen2.5-0.5B on knowledge-light chat 40-60%; the strict bar 10-25%; its open-domain knowledge without retrieval under 5%.
> 6. **Next action:** run both probes on LFM2.5-230M and Falcon-90M, then a repaired `ft_test.py` (small batch, shortcut controls, 5 seeds), one process at a time, LM Studio unloaded.

### 1.1 Terms

- **Body**: non-embedding parameters (total minus the token-embedding table; every model here ties input and output embeddings unless noted).
- **Tok/param**: pretraining tokens divided by total parameters.
- **Multi-IF**: a 3-turn instruction-following benchmark; the reported score averages the three turns, it is not a turn-3 rate (arXiv 2410.15553).
- **Chat probe**: our generation test (section 3), 8 models in their own chat templates, replies fed back as history.
- **Likelihood battery**: our log-probability test (section 3.3). The gold answer must beat a same-type foil that also appears in the context. **d0 / d4 / d10** = 0, 4 or 10 unrelated exchanges between the last relevant statement and the question. **Pair @10** = both questions of a swapped pair right at d10 (chance 0.25).
- **U items** = the battery's correction ("latest value wins") items. **U_same** = every statement uses identical wording; **U_neutral** = neutral answer prefix. `uprobe.py` ran these two controls.
- **Level R / Level A**: the two success bars in D3 (section 8). R = non-inferior to Qwen2.5-0.5B-Instruct on a knowledge-light multi-turn battery. A = the strict bar: corrections 4+ turns old, both-order binding, no loops, and a blind human win-or-tie of at least 50% vs Qwen2.5-0.5B.

### 1.2 The premise, both directions

- **Sub-400M chat models exist and work, within limits.**
  - LFM2.5-230M (163M body, 19T tokens): Multi-IF 37.70 vs 41.68 for Qwen3.5-0.8B [vendor].
  - Falcon-H1-Tiny-90M: 2-turn MT-Bench 4.33 vs 3.80 for SmolLM2-360M in the same harness [vendor].
  - On our probe the 350M class (0.57-0.62) is level with Qwen2.5-0.5B (0.60), and Qwen2.5-0.5B deflects on 10 of 15 multi-turn "what did I tell you" questions.
  - What looked like a wall is where labs spent compute (2,900 to 83,000 tokens per parameter, plus a large teacher or RL) and where models stop looking broken in casual use (fluent, some facts, no loops).
- **Strict multi-turn chat is not shown below 0.6B (our probe) or 1B (published).** [corrected in review: the draft answered only the loose direction]
  - Corrections in generation fail in 21 of 24 trials across 8 models; the passes are Qwen3-0.6B (2 of 3) and Falcon-90M (1 of 3).
  - Role capture (the assistant adopts the user's name or job) appears in 7 to 8 of 8 models from 90M to 0.6B.
  - Only LFM2-2.6B passes the likelihood correction items at d4 and d10. Nothing between 0.6B and 2.6B was tested.
  - Nothing published measures coherence past 3 turns at this size, apart from agent task scores (tau2-Bench) and 2020-era chit-chat ratings.

### 1.3 What limits a ~150M model, in two columns

Ranked within each column by how likely each is to stop a Max-built 150M (100-300B tokens) from clearing that bar. [corrected in review: the draft ranked both questions in one list]

**Column R: what separates 150M from Qwen2.5-0.5B (binds at Level R)**

| # | limit | kind | one line |
|---|---|---|---|
| 2 | Binding same-type entities (whose cat, whose job, my name vs yours) and two-hop references | mixed | Two-hop is at chance at ~100M body; SmolLM2 135M to 360M (2.6x params and 2x tokens) lifts owner pairs 0.62 to 0.94; a recipe change at fixed size moves one-hop 0.34 to 1.00. |
| 3 | Chat training signal and reply policy | soft | The largest failure class today and the cheapest to remove: deflection 0-67% regardless of size; Max's SFT packer cuts off the start of the conversation for 42.5% of supervised tokens. |
| 4 | Tokens per parameter (670 planned vs 2,900-83,000 for every working small chat model) | mixed | Soft in principle, expensive in practice; interacts with #2 through data density. |
| 5 | Degeneration on its own history (loops, self-copy) | mostly soft | SmolLM2-135M repeats a 4-gram 3+ times in 22% of replies; Falcon-90M almost never. |
| 7 | Stored world knowledge | **hard** | Binds only if the battery asks for facts. If "chatting" includes knowing things it is the first limit, and not reachable at 150M without retrieval. |

**Column A: what separates every tested sub-1B model from strict chat (binds at Level A)**

| # | limit | kind | one line |
|---|---|---|---|
| 1 | Updating after a correction ("first value wins") | unknown | Clean evidence only in SmolLM2-135M/360M (identical wording, 0 of 192 at d4-d10); confounded for the other six; generation fails 21 of 24; LFM2-2.6B passes. [corrected in review] |
| 6 | Instruction persistence beyond 3 turns | mixed | Unmeasured below 1B past 3 turns. |

Minor in both columns: teacher signal (#8), embedding share (#9), output-head rank and depth (#10).
Not a limit: holding one user-stated fact for 12 turns and ~1,700 tokens (Falcon-90M, 74M body, does it).
**Prerequisite, not a model limit:** measurement. Harnesses swing scores 10-24 points, and at 151M one seed moves as much as any recipe. [corrected in review: the draft ranked it as limit #7]

### 1.4 Honest odds (judgment before the decisive tests, not measurements)

- **Level R (non-inferior to Qwen2.5-0.5B-Instruct, knowledge-light): roughly even, 40-60%.**
  - For: the bar is low (the 0.5B model deflects and fails corrections), the 350M class already clears it, and Falcon-90M scores 0.40-0.46 against Qwen2.5-0.5B's 0.60 with a 74M body.
  - Against: Max's planned compute is about a fifth of Falcon's (1.6x short even at 300B tokens); the only same-size published attempt (ufakzeka-1, 151M) stalled on identity tracking; a generic same-size recipe at 210 tok/param (Doge-160M-Instruct) scores IFEval 16.8.
- **Level A (strict): about 10-25%.** A repaired `ft_test.py` that passes at 135M on most of 5 seeds, including the two-slot and no-update controls, would move this to perhaps 30-40%. A pass on the original script would move nothing, because "copy the most recent weekday" passes it. [corrected in review: the draft said 40-50% on the unrepaired test]
- **Matching a 0.5B model on open-domain knowledge without retrieval: under 5%.** The ceiling scales linearly with parameters.

### 1.5 Evidence status

| work item | status |
|---|---|
| Chat probe: 8 models, 29 multi-turn conversations + 24 single-turn controls, plus fixed-history, base-vs-instruct and system-prompt experiments | complete and audited (`lanes/probe.verify.md`) |
| Likelihood battery (binding, two-hop, corrections, perspective, knowledge) | complete for 8 models; Gemma 3 270M partial (mostly d0); **Falcon-H1-Tiny-90M missing** (MPS crash) |
| Correction controls (`uprobe`: U_same, U_neutral) | SmolLM2-135M-I and 360M-I only |
| LFM2.5-230M, the closest shipped analogue (163M body) | **never probed** |
| `capacity_probe/ft_test.py` (can a few hundred fine-tuning steps teach binding and corrections at 135M?) | **never produced output.** The first run coincided with a watchdog-timeout kernel panic (reboot 17:12; `/Library/Logs/DiagnosticReports/panic-full-2026-09-23-171933.0002.panic`); its log ends at weight loading. It is also **not safe or decisive as written** (section 7.4). |
| Recipe lens P1 (token-curve probe on intermediate checkpoints) | raw outputs lost; its numbers are unverified leads only |
| Live hazard at 18:20 | LM Studio's llama-server holds `gemma-4-12B-it-QAT` resident with `mmap+mlock` (about 8 GB RSS, PID 11931). Unload it before any model run. |

---

## 2. The landscape: sub-600M chat models

Body = non-embedding parameters. Tok/param = pretraining tokens / total params. Chat numbers compare only within one harness: the same model's IFEval moves 10-24 points between harnesses (Gemma 3 270M: 51.2 Google vs 27.44 TII) [corrected: only 2 of the 4 examples in the eval lane exceed 10 points]. "Probe" = our hand-adjusted greedy macro (section 3; set A / set B).

| model (release) | total | emb share (body) | pretraining tokens (tok/param) | recipe | chat evidence |
|---|---|---|---|---|---|
| PleIAs Monad (Nov 2025) | 56.7M | 3.7% (54.6M) | 200B synthetic (3,530) | SYNTH only, thinking traces, 64 layers | card: no multi-turn support |
| BlenderBot 90M (2020) | 87.5M | seq2seq | Reddit + BST fine-tune | pre-LLM chit-chat | 14-turn human chats: engagingness beat Meena 2.6B 61-39; known failures: contradiction, forgetting, repetition |
| Falcon-H1-Tiny-90M-Instruct (Jan 2026) | 91.1M | 18.4% (74.4M) | 800B (8,779) | attention + Mamba2 in every layer; 25% SFT data in pretraining; SFT; 1 epoch DPO; no teacher | IFEval 66.08, 2-turn MT-Bench 4.33 [vendor; corrected: 2-turn evalchemy MT-Bench, not MT-Bench-101, judge undetermined]; probe 0.46 / 0.40, zero deflection, best fixed-history recall per parameter |
| Supra2-100M-Instruct (Aug 2026, community) | 100.7M | 25.0% (75.5M) | 30B (298) | community | its own card sample resolves "Sure!" to the wrong referent by surface association |
| MobileLLM-125M (2024) | 124.6M | 14.8% (106.2M) | 1T (8,026) | deep-thin, 30 layers; chat fine-tune in paper only | MT-Bench 2.33, AlpacaEval 24.07% (GPT-4 judge, identical chat FT for all models) |
| SmolLM2-135M-Instruct (Oct 2024) | 134.5M | 21.0% (106.2M) | 2T (14,868) | single-stage pretrain; smol-smoltalk SFT; DPO; no distillation | IFEval 29.9 (HF) / 30.69, MT-Bench 2.68 (TII); probe 0.15 / 0.11; repeats a 4-gram 3+ times in 22% of replies |
| **Doge-160M-Instruct** (Feb 2025) [added in review] | 152.7M | 16.5% (127.5M) | 32B SmolLM-Corpus (210) | generic: SmolTalk SFT, UltraFeedback DPO | IFEval 16.8 prompt-strict (its card); the closest same-size generic recipe, at a token budget below Max's |
| ufakzeka-1 (Sep 2026, Turkish) | 182M | ~17% (151M) | 13.5B (74-89) | SFT-only release; DPO hurt in 3 variants | identity tracking over a long story did not move with any data change; seed spread equal to recipe spread |
| **LFM2.5-230M** (Jun 2026) | 229.7M | 29.2% (**162.6M**) | **19T (82,719)** | 8 conv + 6 GQA layers (config verified); SFT distilled from LFM2.5-350M; DPO; RL | IFEval 71.71, Multi-IF 37.70, tau2-Bench Telecom 5.26 / Retail 13.68 [vendor]; **not probed by us**; the closest shipped analogue to a 150M target |
| Gemma 3 270M-it (Aug 2025) | 268.1M | 62.6% (100.3M) | 6T (22,380) | Gemma 3 family uses distillation (the 270M's own recipe unsourced) | IFEval 51.2 (Google) vs 27.44 (TII); MT-Bench 3.71 (TII); Google says it is not built for complex conversation; probe 0.31 / 0.27 |
| PleIAs Baguettotron (Nov 2025) | 321.0M | 11.8% (283.2M) | 200B synthetic (623) | 80 layers; thinking traces | MMLU 40 / GSM8K 39 / HotPotQA 51 [vendor; training data carried benchmark signals]; multi-turn with rolling thinking recommended, not required [corrected] |
| Granite-4.0-H-350M / 350M (Oct 2025) | 340.3M / 352.4M | 22.6% / 29.2% | ~15T | SFT + RL + merging | Multi-IF 28.70 / 24.21 (Liquid harness) |
| MobileLLM-350M (2024) | 345.3M | 8.9% (314.6M) | 1T (2,896) | deep-thin, 32 layers | MT-Bench 3.28, AlpacaEval 47.08% (beat OPT-1.3B, 2.24) |
| LFM2-350M (Jul 2025) | 354.5M | 18.9% (287.4M) | 10-11T (~28,000) | top-32 logit KD from LFM1-7B in pretraining; 5.39M-sample SFT; preference pairs from on-policy samples; merging | IFEval 65.12, Multi-IF 32.85 [vendor]; probe 0.60 / 0.50 |
| LFM2.5-350M (Mar 2026) | 354.5M | 18.9% (287.4M) | 28T + RL (78,988) | same architecture, more tokens, multi-stage RL | IFEval 76.96, Multi-IF 44.92 [vendor]; probe 0.62 |
| SmolLM2-360M-Instruct (Oct 2024) | 361.8M | 13.0% (314.6M) | 4T (11,055) | as 135M | MT-Bench 3.66 (HF) / 3.80 (TII); IFEval 41.0; probe 0.57 |
| **Qwen2.5-0.5B-Instruct** (the "wall") | 494.0M | 27.6% (357.9M) | 18T (36,435) | SFT on 1M+ samples, multi-stage RL | MT-Bench 4.16 (HF) / 4.71 (TII); probe 0.60; deflects on 67% of multi-turn user-fact turns |
| H2O-Danube3-500M-chat (Jul 2024) | 513.6M | 19.1% | 4T | | MT-Bench turn 1 4.16, turn 2 2.40 (GPT-4 judge): the clearest published second-turn collapse |
| Falcon-H1-0.5B-Instruct (May 2025) | 521.4M | 12.9% (untied) | 2.5T (4,795) | | MT-Bench 7.06, IFEval 72.07 (TII harness): the best chat numbers in the census [corrected] |
| nanochat d20 (Oct 2025) | 561.0M | 29.9% (untied) | 11.2B (20) | midtraining + SFT | ChatCORE 0.088: a Chinchilla-budget demo, not an assistant |
| Qwen3-0.6B (Apr 2025) | 596.0M | 26.1% (440.5M) | 36T family figure (60,398) | strong-to-weak distillation, off-policy then on-policy logits | IFEval 54.5, Arena-Hard 6.5, Multi-IF 33.3 (Qwen) / 45.13 (Liquid); probe 0.71 / 0.62 |
| reference: Qwen3.5-0.8B (Feb 2026) | 752.4M text LM | 33.8% (498.1M) | unsourced | | Multi-IF 41.68 (Liquid); MultiChallenge 18.9 thinking (Qwen); card warns of thinking loops |

Also in the census, with no multi-turn evidence [added in review]:
- OpenELM-270M-Instruct: 271.5M, 15.1% embeddings, ~1.8T-token dataset, no chat benchmarks.
- ERNIE-4.5-0.3B-PT: 360.7M, 29.4%; tokens and chat numbers unsourced.
- MiniCPM4-0.5B: 433.9M, 17.3%, 1T tokens.
- Hunyuan-0.5B-Instruct: 539.0M, 23.0%, IFEval 49.7; tokens unsourced.
- MiniMind: 26-64M hobby project with multi-turn SFT and no quantitative multi-turn evaluation.

A Hugging Face API sweep this evening covered text-generation models created Jul-Sep 2026 with 100M-230M in the name. It found hobbyist models and LFM2.5-230M fine-tunes, and no new lab release (not every card was read).

What the table says:
- **Every model with good chat numbers used 2,900 to 83,000 tokens per parameter** [corrected: the lanes first said 8,800-83,000]. Many also used a teacher or RL:
  - pretraining KD: LFM2, Apertus;
  - distilled SFT: LFM2.5-230M;
  - post-training distillation: Qwen3;
  - distillation across the Gemma 3 family (the 270M's own recipe is unsourced);
  - no teacher described: SmolLM2, Falcon-H1-Tiny, Doge [corrected].
- **Max's plan is 670 tokens per parameter** (150M on 100B tokens), or 2,000 with three epochs.
  - In training compute (6ND), Falcon-H1-Tiny-90M is 4.9x Max's 100B plan and 1.6x his 300B plan.
  - MobileLLM-125M is 8.3x / 2.8x. SmolLM2-135M is 17.9x / 6.0x. LFM2.5-230M is 291x / 97x.
  - Falcon and MobileLLM-125M are the only published chat recipes within reach (my arithmetic). Doge-160M shows what a generic recipe gives below that budget.
- **Embedding share hides how small some "mid-size" tiny models are.**
  - Gemma 3 270M has a 100M body, smaller than SmolLM2-135M's 106M.
  - A 150M model with Max's 49k vocabulary has a 122M body at d=576 or 112M at d=768. With Planck's own 8-16k vocabulary it is about 137-145M (my arithmetic).
  - Against Qwen2.5-0.5B's 358M body, the gap is **2.9-3.2x** with the 49k vocabulary and about 2.5-2.6x with a small one. That is close to the 3.3x gap in the names. [corrected in review: the draft's "2.5-3.6x" mixed in other models]
- **Vendors scope sub-400M models away from open chat.** Google says Gemma 3 270M is not built for complex conversation. Liquid recommends the 230M for extraction and on-device agents. MobileLLM-R1 is not a general chat model. Monad has no multi-turn support.
- **Nobody publishes coherence metrics beyond 3 turns at this size**, except tau2-Bench agent completion for LFM2.5 [corrected].
- **Not included as chat models:**
  - MobileLLM-R1-140M: math and code SFT, 53% embeddings.
  - The April 2026 "micro language models" (8-30M): they write only the first 4-8 words of a reply for a cloud model to finish (arXiv 2604.19642).

---

## 3. What the hands-on probe showed

Two probes, both run on this Mac earlier today. Neither was re-run for this report.

- **Chat probe** (`probe/`, audited in `lanes/probe.verify.md`).
  - Setup: each of 8 models ran in its own chat template with its own replies fed back as history. 29 multi-turn conversations (2-12 turns) plus 24 single-turn controls, greedy decoding and 2 sampled seeds.
  - Grading: the graders were mutation-tested (14 of 14 mutants killed). The audit still found false passes the fixtures never tried (negations, guess lists, hypotheticals, first person with "you"), so it hand-adjusted the scores.
  - Cells hold only 1-20 trials, so differences under ~0.2 are noise.
- **Likelihood battery** (`capacity_probe/`).
  - Every model gets the same plain `User:/Assistant:` transcript.
  - The gold continuation must beat a same-type foil that also appears in the context, so copying any name from context cannot pass.
  - 32 scenarios per cell at d0/d4/d10; every context is under 600 tokens.

### 3.1 Three short transcripts

**The "wall" model deflects on a fact it still has** (Qwen2.5-0.5B-Instruct, greedy):

> **user 0:** I have a cat named Biscuit who loves sitting in cardboard boxes.
> *(4 unrelated turns)*
> **user 5:** What's my cat called?
> **Qwen2.5-0.5B:** I'm sorry, but I'm Qwen, a language model created by Alibaba Cloud. I don't have a cat or any other pets. [...]

When the probe starts the reply as "Your cat is called", " Biscuit" is the top token at every position (log-prob -0.02). The fact is intact; the reply policy reads "my cat" as a question about the assistant.

**A 28T-token model reverts a correction one exchange later** (LFM2.5-350M, greedy):

> **user 0:** My meeting with Sam is at 3 pm.
> **user 1:** Correction: the meeting got moved to 4 pm.
> **LFM2.5-350M:** That's a good update! The meeting has been moved to 4 pm. [...]
> **user 2:** Why do leaves change color in the fall? *(answered)*
> **user 3:** What time should I show up for my meeting with Sam?
> **LFM2.5-350M:** The meeting with Sam is scheduled for 3 pm. [...]

Forced prefix "You should show up at": " 3" log-prob -0.57 vs " 4" -3.24. This item has a digit-token confound (see 3.2).

**The 90M model fails differently: no deflection, a template instead of the fact** (Falcon-H1-Tiny-90M, greedy):

> **user 0:** Hello, I'm Marcus. I'm learning to play the guitar.
> *(6 unrelated turns)*
> **user 7:** What's my name?
> **Falcon-H1-Tiny-90M:** Your name is [Your Name]. It's a simple yet meaningful name that reflects your personality and the person you are.

Its other characteristic failure turns "Which city do I live in?" into a listicle. But it never deflected in 29 final user-fact turns and never looped.

### 3.2 Per-ability results, chat probe (greedy, strict graders)

| model | params (body) | recall | follow-up | instruction | correction | own answer | role | topic return | macro (lane) | macro hand-adjusted A / B | single-turn controls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Falcon-H1-Tiny-90M | 91M (74M) | 0.50 | 0.57 | 0.29 | 0.33 | 0.75 | 0.33 | 0.67 | 0.49 | 0.46 / 0.40 | 0.64 |
| SmolLM2-135M | 135M (106M) | 0.10 | 0.29 | 0.36 | 0.00 | 0.33 | 0.00 | 0.00 | 0.15 | 0.15 / 0.11 | 0.46 |
| Gemma 3 270M | 268M (100M) | 0.00 | 0.86 | 0.50 | 0.00 | 0.50 | 0.00 | 0.67 | 0.36 | 0.31 / 0.27 | 0.75 |
| LFM2-350M | 355M (287M) | 0.60 | 1.00 | 0.79 | 0.33* | 0.50 | 0.33 | 1.00 | 0.65 | 0.60 / 0.50 | 0.89 |
| LFM2.5-350M | 355M (287M) | 0.60 | 1.00 | 1.00 | 0.00 | 0.75 | 0.00 | 1.00 | 0.62 | 0.62 / 0.62 | 0.82 |
| SmolLM2-360M | 362M (315M) | 0.70 | 0.86 | 0.79 | 0.00 | 0.67 | 0.33 | 0.67 | 0.57 | 0.57 / 0.57 | 0.54 |
| Qwen2.5-0.5B | 494M (358M) | 0.10 | 0.86 | 0.57 | 0.00 | 1.00 | 0.67 | 1.00 | 0.60 | 0.60 / 0.60 | 0.82 |
| Qwen3-0.6B | 596M (440M) | 0.60 | 1.00 | 0.71 | 0.67 | 1.00 | 0.00 | 1.00 | 0.71 | 0.71 / 0.62 | 0.93 |

\* LFM2-350M's correction pass is a hypothetical example, a false pass [corrected]. Set A flips clear grader errors; set B also flips borderline incidental mentions. Sampled decoding moves macros by at most 0.10.

What held up under the audit:
- **The 350M class is level with the 0.5B wall model** (0.57-0.62 vs 0.60). LFM2-350M's lead is not robust [corrected].
- **Below 300M the order follows recipe, not size.** Falcon (74M body, 800B) 0.40-0.46 > Gemma (100M body, 6T) 0.27-0.31 > SmolLM2-135M (106M body, 2T) 0.11-0.15.
- **About half of multi-turn recall failures are retrieval-intact.**
  - In 20 of 42 (48%), greedy decoding from a forced answer prefix emits the gold fact.
  - In 38 of 42 (90%), gold beats a never-seen foil [corrected from 63%].
  - Both are induction-copy tests, since the foil never appears in the context.
- **Rendering alone changes recall on the same weights.**
  - SmolLM2-135M-Instruct recalls a user fact 0.15 of the time through its chat template and 0.70 as a plain transcript.
  - Hand-strict, that is about 0.05-0.10 vs 0.45, and the base model gets 0.55. The claim that SFT added skill does not survive [corrected].
- **Deflection tracks post-training, not size.** Share of multi-turn final user-fact turns: Qwen2.5-0.5B 67%, LFM2.5-350M 40%, Gemma 40%, SmolLM2-360M 27%, SmolLM2-135M 20%, Qwen3-0.6B 20%, LFM2-350M 7%, Falcon-90M 0% [corrected].
- **A 90M model holds a user fact across 12 turns.**
  - Given an identical 1,726-token history, Falcon-H1-Tiny-90M emits the fact from the forced prefix at every distance.
  - Its free answers score 0.65 (short history) and 0.75 (long history) hand-strict, with zero deflections. It answers as the user in 6 of 40 trials [corrected].
  - SmolLM2-135M's collapse by turn 12 tracks turn count through its template, not tokens [corrected].
- **Corrections fail in 21 of 24 generation trials** (3 items per model; the correction comes 2 user turns before the question, with one exchange between).
  - Passes: Qwen3-0.6B 2 of 3; Falcon-90M 1 of 3 (K_time, "Arrive 4:00 PM" inside a listicle, although its forced prefix still favors the stale 3 pm). "Fails at every size" is overstated [corrected].
  - Log-prob evidence is mixed by item. K_time favors stale in all 8 models (digit-token confound). K_day favors the correction in 6 of 8. K_color, the cleanest item, favors stale in 6 of 8.
- **Role capture in 7 to 8 of 8 models from 90M to 600M** [corrected]. Qwen3-0.6B answers "What's my name?" with "I'm Marcus."
- **Loops and self-copy concentrate at 135M.**
  - SmolLM2-135M: 22% of replies repeat a 4-gram 3+ times and 17% reuse half of an earlier reply. Its card setting of T 0.2 does not fix it.
  - Qwen2.5-0.5B's verbatim loop disappears at T 0.7 with repetition penalty 1.1.
- **Both ~100M models adopted a system-prompt persona**, so "~100M models cannot be prompted into chat behavior" is not supported [corrected].

### 3.3 Per-ability results, likelihood battery (plain transcripts, same-type foils in context)

How to read the table:
- "Pair" requires both questions of a swapped pair right. Chance is 0.25, and a "first-mentioned wins" heuristic scores about 0.
- Latest-wins pools 1-3 corrections at d4 and d10 (n=192).
- Long-tail knowledge is a two-way choice (chance 0.50) on 40 items.

| model | body | owner pair @10 | one-hop pair @10 | two-hop single item, all d | perspective pair @10 | latest wins @4-10 | long-tail closed / open book |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M base | 106M | 0.22 | 0.22 | 0.54 | 0.12 | 0.02 | 0.78 / 1.00 |
| SmolLM2-135M-Instruct | 106M | 0.62 | 0.38 | 0.51 | 0.09 | 0.00 | 0.78 / 0.95 |
| Gemma 3 270M-it (partial) | 100M | 0.47 @0 | 0.69 @0 | 0.55 (n=76) | 0.94 @0 | 0.00 (n=18) | 0.95 / 0.98 |
| LFM2-350M | 287M | 0.34 | 0.34 | 0.66 | 0.09 | 0.11 | 0.85 / 1.00 |
| LFM2.5-350M | 287M | 0.47 | 1.00 | 0.86 | 0.00 | 0.00 | 0.85 / 1.00 |
| SmolLM2-360M-Instruct | 315M | 0.94 | 0.59 | 0.63 | 0.91 | 0.00 | 0.88 / 0.98 |
| Qwen2.5-0.5B-Instruct | 358M | 0.75 | 0.66 | 0.72 | 0.12 | 0.00 | 0.85 / 0.98 |
| Qwen3-0.6B | 440M | 1.00 | 1.00 | 0.99 | 0.97 | 0.00 | 0.80 / 1.00 |
| LFM2-2.6B | 2,435M | 1.00 | 1.00 | 0.94 | 0.97 | 0.72 | 1.00 / 1.00 |

Source: `diagnosis_capacity.md` section 4, recomputed from `capacity_probe/out/*.jsonl`. Falcon-H1-Tiny-90M is missing.

**Corrections: the clean evidence covers one model family.** [corrected in review: the draft called this "the universal failure" with "no gradient from 135M to 0.6B"]
- **The main U items are lexically confounded.** The question and answer prefix share the key "dentist appointment" only with the original statement. Every correction reads "Actually, the appointment got moved to X".
- Latest-wins accuracy by distance, 1 / 2 / 3 corrections (recomputed from `capacity_probe/results.json`):

  | model | d0 | d4 | d10 |
  |---|---|---|---|
  | SmolLM2-135M base | 0.53 / 0.34 / 0.31 | 0.00 / 0.00 / 0.09 | 0.03 / 0.00 / 0.00 |
  | SmolLM2-135M-Instruct | **0.16 / 0.09 / 0.06** | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
  | Gemma 3 270M-it | 0.75 / 0.62 / 0.81 | 0.00 / 0.00 / 0.00 | not run |
  | LFM2-350M | **0.56 / 0.19 / 0.31** | 0.22 / 0.19 / 0.09 | 0.16 / 0.00 / 0.00 |
  | LFM2.5-350M | **0.03 / 0.00 / 0.00** | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
  | SmolLM2-360M-Instruct | 0.72 / 0.31 / 0.34 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
  | Qwen2.5-0.5B-Instruct | 1.00 / 0.88 / 0.41 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
  | Qwen3-0.6B | **0.34 / 0.06 / 0.00** | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
  | LFM2-2.6B | 1.00 / 1.00 / 1.00 | 0.88 / 0.72 / 0.66 | 1.00 / 0.69 / 0.41 |

- **Four sub-1B models (bold) already fail at d0**, when the correction is the last turn. For them, the zeros at d4 and d10 cannot separate primacy from key-matching or from the off-template format.
- **U_same is the one clean control, run only on SmolLM2** (identical wording in the original and every correction).
  - 360M-Instruct: 0.97-1.00 at d0, then 0 of 192 at d4 and d10.
  - 135M-Instruct: 0.56-0.66 at d0, then 0 of 192.
  - That is a real first-value-wins effect, but in one model family and one item family, with the original always in the first turn. The position control has not been run.
- **U_neutral keeps the key confound.** SmolLM2-360M gets 0.66 / 0.03 / 0.00 at d0 with 1 / 2 / 3 corrections. [corrected in review: the draft's "0.97-1.00 at distance 0" holds only for U_same]
- **The original value wins in every sub-1B model**, not an intermediate one. Mean margins for one correction at d10 run from -0.9 to -8.8 nats. LFM2-2.6B is +1.8 and passes even the confounded items.
- **"4 or more turns back" is an artifact of the 0/4/10 grid, not a threshold.** In generation, failures appear with one exchange in between.

**Binding has a size gradient below 300M of body, but recipe moves it as much.**
- Same data family: SmolLM2 135M to 360M moves owner pair 0.62 to 0.94 and perspective 0.09 to 0.91. This is a **size-plus-2x-tokens step** (2T vs 4T), the only same-family step below 300M of body. [corrected in review]
- Same size, different recipe: LFM2 to LFM2.5 at 287M body moves one-hop from 0.34 to 1.00.
- Size order is not monotonic across families: Qwen2.5-0.5B is below SmolLM2-360M on owner and perspective binding.

**Reading works at 106M; storing is scarce.** SmolLM2-135M gets 0.78 on long-tail facts closed-book and 1.00 when the user stated them two turns earlier.

**Oddity.** Gemma 3 270M (100M body plus a 168M embedding table) scores 0.95 closed-book on long-tail facts. With n=40, this is a hint, not a finding.

Caveats for all of section 3:
- Likelihood read-outs are not generation.
- Plain transcripts are off-template for instruct models.
- No probed model has a body in the 122-145M target range. LFM2.5-230M (163M body) is the nearest and was never probed.
- Nothing runs beyond 12 turns, and there are no human ratings.

---

## 4. The current best recipe at ~150M, component by component

### 4.1 Architecture

| choice | evidence and scale | strength |
|---|---|---|
| **Keep softmax attention** in all or most layers (GQA fine); a hybrid that keeps attention is acceptable, an attention-free stack is not | 70M attention beats a 1.4B gated conv on associative recall (Zoology, 70M-1.4B); transformers beat Mamba at copying at 70+ phone-book entries [corrected scope] (Pythia-410M vs Mamba-2.8B); Hymba at 300M/100B: attention+SSM heads beat Llama on recall (49.90 vs 39.98); our probe: Falcon's hybrid retrieves at 1.7k tokens | strong (keep attention); weak (hybrid vs pure) |
| **Previous-token path** (short causal conv, token shift, or Canon layers, kernel ~4) | 2-4x synthetic reasoning depth at 50M-124M; real-data gains only at 1B-8B [>=1B only]; a short causal conv adds ~0.5 recall in ~29K-parameter recurrent cells (Boesch and Wee); vanilla attention needs 2 layers for induction, 1 with a smeared key | moderate mechanism, weak at 150M; worth an A/B |
| **Shape**: ~24-32 layers at d=576-640 | MobileLLM 125M: ~+1 point from depth, flat beyond ~24 layers (single seed); LFM2.5-230M chats with 14 layers at d=1024; the 2-hop failure at 106M body is not a depth shortage (SmolLM2-135M has 30 layers) | moderate, contested |
| **Max's validated block**: per-head gated attention, normalized value residual, 1/sqrt(depth) norm scaling, QK-norm, NorMuon + cautious WD | Max's 124M A/B (1.0B tokens processed, ~9 tok/param): **4.5% (0.134 nats) on paired training loss** [corrected from 6.9% held-out, about ±0.1 nats of eval noise]; ResFormer value residual +16% parameter-equivalent at 82M-468M; gate and value residual inside noise at 1.2B [>=1B only] | moderate |
| Output head rank (d < 1000) | late saturation in Pythia <=410M; no demonstrated optimization harm; no link to chat | unknown; low priority |
| Looping / block sharing | MobileLLM-LS +0.7 at 125M for 2x FLOPs; looping does not raise knowledge capacity (Ouro, 1-40M); MoR loses to vanilla at 135M iso-FLOP | weak; not a knowledge lever |
| Context | 2-4k training context is enough; KV cache for a 30-layer, 3-KV-head 150M is ~94 MB at 4k | moderate |

### 4.2 Tokenizer and vocabulary

**Already decided** (Planck note, 2026-09-23): Planck gets its own small tokenizer, tuned for tiny scale. [corrected in review: the draft presented this as open] Consequences:
- **No same-tokenizer logit distillation** for Max's own model, from Ultra (49k) or SmolLM2 (49k, different merges). Teachers reach it only through generated text or cross-tokenizer KD, which is unproven below 1B.
- The SmolLM2-1.7B to SmolLM2-135M on-policy distillation in lever 5 becomes a **public-model pilot** that tests the method. It is not a stage of Max's recipe.

Evidence:
- **The 49,152 vocabulary costs 28.3M (19%) at d=576 or 37.7M (25%) at d=768.** An 8-16k vocabulary frees about 19-25M of body (my arithmetic).
- **In-scale evidence** (Tao et al., at 302M non-vocab params): 16K is compute-optimal, 24K with excess data [corrected].
- **On OASST chat:** 8k to 49k cuts tokens per conversation by 13%; past 49k saves only 0.7-4.3% [corrected].
- **Out:** plain byte-level (about 13% worse bits per byte [corrected]) and SuperBPE (worse than BPE at every budget, 1,308 models). A 4.3-4.8 bytes/token target is refuted as a design goal.

### 4.3 Data

| choice | evidence and scale | strength |
|---|---|---|
| **Put dialogue into pretraining.** Max's 100B build has almost none; web text is 2.7-3.7% conversational | Falcon-H1-Tiny at 90M: 25% SFT data across 800B pretraining tokens raised IFEval 53.47 to 66.08, **but the classic pipeline was slightly ahead on 2-turn MT-Bench (4.40 vs 4.33) and AlpacaEval** [corrected]; TII at 100B tokens: 25% and 50% tied, 75-100% worse; Pythia 70M-1B midtraining: FLAN minimal, KnowledgeQA+dialogue at 20% lowered LIMA SFT loss at 160M | moderate for instruction following; weak for multi-turn coherence |
| **Free multi-turn corpora; use them first** [added in review] | WildChat-4.8M: 3,199,860 real user conversations with ChatGPT, ODC-BY. SODA: 1.5M social dialogues, ~300M tokens, CC-BY-4.0. UltraChat: 1,468,352 dialogues, ~2.15B tokens, MIT. TinyDialogues: ~29M words, MIT. Everyday-Conversations: 2.2k dialogues that fixed "Hi" / "Who are you" at 135M-1.7B (SmolLM v1), Apache-2.0. Licenses checked on the HF API today. Caveats: WildChat and UltraChat assistant turns come from OpenAI models; UltraChat has only 1.45 context-dependent queries per session | strong on availability; weak on multi-turn density |
| **Dense multi-turn signal**: back-references, same-type distractors, corrections with stale traps, two-slot and no-update twins, answer mentioned second as often as first, short turns | UltraChat 1.45 context-dependent queries per session vs 4.62 in ShareGPT; smol-smoltalk ~2.3 turns; Parrot data alone +0.18 to +0.24 MT-Bench at 13B [corrected] [>=1B only]; TurnWise +12.8 multi-turn points at 7B [>=1B only]; critical mixing ratio (Pythia 70M/410M): below a density threshold a skill gets **no** capacity | mechanism moderate; effect at 150M untested |
| **Knowledge-light and grounded** | entity-anonymized pretraining at 135M/360M: closed-book LAMA 12.5 to 0.7, FEVER 82.8 to 89.5 (single preprint); source tags cut junk-dilution loss from 20x to 2x (synthetic) | moderate hint |
| **Repetition is cheap** | up to ~4 epochs nearly as good as unique data (10M-9B); a 100M model forgets a sample within 1-5B tokens (one study, math only) | moderate |
| **Generate by program, render with a teacher** | Lever 1's dialogues (planted facts, corrections, distractors, gold answers) can be written by program; a teacher only paraphrases ("script mode", Planck notes). Full teacher generation: Anthropic's consumer terms prohibit training competing models [corrected wording]; ~20-50 Mac-days per 100M accepted tokens with an 8-12B dense teacher [corrected from 4-12] | strong on rights; throughput anchor weak |

### 4.4 Training and distillation

- **Tokens.** 100B tokens is 667 tok/param, and three epochs give 2,000.
  - A 150M model keeps improving to at least 10,000 tok/param (Sardana et al.).
  - At 300B tokens, 150M sits within ~0.07-0.09 nats of the same model at 2T (low confidence).
  - **Recommendation: 300B token-passes if the Titans are available.**
- **Schedule.** WSD with linear decay to zero, weight decay re-derived for the regime, batch size growing with data. Moderate, cheap.
- **Optimizer.** Matrix optimizers give ~1.4x over AdamW at 0.1B. Max's AdamW arm used lr 6e-4, which Wen et al. flag as under-tuned. Re-validate with per-arm learning rates.
- **Pretraining logit distillation: weak at this size, and ruled out for Max's model by the tokenizer decision.**
  - Controlled nulls: MobileLLM 125M with a 7B teacher (43.9 vs 43.8), MiniPLM at 200M (39.9 vs 39.9), JugnuLM-110M.
  - A 1.7B/11T teacher crosses to harmful near 350B student tokens [corrected].
- **Uses of Ultra that survive a different tokenizer.**
  - MiniPLM-style difference sampling to select data: +1.4 average at 200M; scoring costs ~2.4e20 FLOPs [corrected].
  - Generating or scoring text.
  - Offline RS-KD would need a shared vocabulary.
- **Weight inheritance or pruning from Ultra:** speculative, and blocked at the embedding layers by the tokenizer decision.

### 4.5 Post-training

| choice | evidence and scale | strength |
|---|---|---|
| **Pack SFT by whole conversation with cross-document masking** | Max's packer (`posttrain/sft_data.py`) slices fixed windows: at seq_len 2048, **42.5% of supervised assistant tokens lose the start of their own conversation**; separately, 57.4% can attend to an unrelated earlier conversation, which is ordinary packing without cross-document masking (`recipe_work/pack_sim.py`, 1,100 UltraChat rows) [corrected in review] | mechanism certain; effect unmeasured; zero cost |
| **Student-sized multi-turn SFT**: short replies, no long CoT, no tool calling, no deflection boilerplate, no fixed default system prompt | SmolLM2 filtered hard data for 135M/360M; template effect 0.15 vs 0.70 on the same weights; CoT caused loops in TII's 90M tool model [corrected] | moderate |
| **One epoch of DPO** with multi-turn pairs whose rejected side is a programmatic failure, sampled from the model's own rollouts | 90M: one epoch raised 2-turn MT-Bench ~3.1 to ~4.3-4.4, more degraded; 151M: three preference variants lowered quality; Max's DPO set is 100% single-turn. **Prior art** [added in review]: DZ-TiDPO, conflict-aware DPO plus a temporal attention bias for dialogue "state inertia", at Phi-3.5-mini 3.8B and Qwen2.5-7B; the smaller model paid a stability cost [>=1B only]. Multi-turn conflict DPO is a tweak of known work at small scale | moderate for one epoch; multi-turn pairs untested below 3.8B |
| **On-policy distillation** as the final stage | MiniLLM 120M: 38.6 to 44.7; gain non-monotonic in size [corrected]; MobileLLM-R1.5 140M GSM8K 4.1 to 8.3; multi-turn OPD below 1B only for agents (Guided-OPD 0.6B) [corrected]; token-level OPD needs a shared vocabulary | moderate single-turn; hypothesis multi-turn |
| **Unlikelihood** on self-repeated n-grams and contradictions | repetition 0.617 to 0.055, contradiction preference 18% to 78% (~90M seq2seq, 2019-2020) | moderate, old architecture |
| RL (GRPO-style) | no evidence for open chat below 1B [corrected] | weak for chat |
| Keep the SFT checkpoint; soup checkpoints; low LR with replay | overtrained small models are fragile to fine-tune | moderate, cheap |

### 4.6 Decoding

- **Never pure greedy for chat.** Use T 0.5-0.7, with a mild repetition penalty only as a stopgap.
  - The penalty fixes Qwen2.5-0.5B's loop but not SmolLM2-135M's at T 0.2.
  - Distilled students loop more than their teachers [>=1B only].
- Contrastive search, min-p and context-aware decoding are weak or untested for multi-turn.

### 4.7 Memory and tools

- **Within-conversation memory: attention is enough for a unique fact at ~2k tokens** (Falcon-90M). Strong.
- **Structured state:** untested in a general tiny chat model (lever 2).
  - SimpleTOD learns belief-state tracking at 82M (JGA 54.5), but that is cumulative slot-filling, not corrections [corrected].
- **Retrieval must be trained in.**
  - Off-the-shelf SmolLM2-360M-Instruct: 0% exact match even with the gold passage.
  - Retrieval-trained BART-large: hallucination 68.2% to 7.9%.
  - In Retro-li at 109M, retrieval bought only 0.14 perplexity [corrected].
  - Recommendation: in-context retrieval with a trained reader.
- **Sparse memory (only if "150M" means active parameters).**
  - Memory+ 134M base matched a dense 373M on TriviaQA.
  - UltraMem at 151M favored MoE over product-key memory.
  - MoLE at 160M active: 38.8 to 41.9.
  - The ordering between these is contested.
- **Tools.**
  - Single calls work at small size: TinyGSM 125M gets 63.1% GSM8K.
  - Multi-turn tool use does not: Falcon's 90M tool model scores 0% on BFCL multi-turn [corrected].

---

## 5. The limits map

### 5.1 Every limit

| # | limit | mechanism | kind | evidence (scale) | cheapest decisive test |
|---|---|---|---|---|---|
| 1 | **Updating after a correction** (Level A) | Key-based retrieval finds every statement of the slot and resolves toward the earliest ("active primacy intrusion" is a label from a 39-model study at 1B-2.5T [>=1B only]); pretraining rarely overwrites a stated value and single-turn SFT never does; the model's own acknowledgements add copies of both values; trained LLMs also show load-dependent interference with a **recency** bias (in-context superposition; LLM sizes not checked), so primacy may be specific to our item family | unknown | **Clean:** SmolLM2-135M/360M-Instruct U_same, 0 of 192 at d4/d10 each, original always in turn 1. **Confounded:** the other six sub-1B models (four fail at d0). **Generation:** 21 of 24 fail with one intervening exchange; Qwen3-0.6B 2 of 3, Falcon-90M 1 of 3. LFM2-2.6B passes the confounded items. **Outside:** a 2-layer transformer trained on a working-memory task solves it perfectly (2604.09670); bAbI dialog task 2 small memory nets 100% on updates; Boesch and Wee (~29K-parameter recurrent cells): "interference under sparse supervision, not capacity", a distance curriculum takes recall 0.021 to 1.000, training is a seed lock-in lottery; DZ-TiDPO on "state inertia" at 3.8B/7B [>=1B only]. [corrected in review] | **Repaired `ft_test.py`** (7.4) on SmolLM2-135M-I then 360M-I. On every model: U_same, a position control (original after 3 distractor turns), a U item with the key in the correction. For a size threshold inside one recipe: LFM2-700M and LFM2-1.2B |
| 2 | **Binding under same-type interference** (Level R) | Two same-type values force a multi-token key match, not a one-token induction key; small models fall back on "first-mentioned wins"; the circuit is learned only if its signal is dense enough | mixed | Two-hop at chance (0.51-0.55) in all ~100M bodies; SmolLM2 135M to 360M owner 0.62 to 0.94, perspective 0.09 to 0.91, **a size-plus-2x-tokens step** [corrected in review]; LFM2 to LFM2.5 at 287M one-hop 0.34 to 1.00; a 37.8M transformer learns 4-hop binding from 450K synthetic programs. Outside anchor [added in review]: entity tracking in narratives reaches human level at Pythia-410M by likelihood (53.5%, about chance, at 70M; all on 300B Pile tokens), but in generation OLMo 2 13B-Instruct scores 65.6% vs humans' 75.3% | Same repaired `ft_test.py`: two-hop pair at d10 from 0.09 to at least 0.8, owner/perspective at least 0.85, knowledge change as a paired CI on an enlarged `khard` (the 40-item set has SE ~0.06 and cannot resolve 0.05) [corrected in review]; plus the battery on SmolLM2-135M intermediate checkpoints |
| 3 | **Chat training signal and reply policy** (Level R) | Small models learn only the conversation their data asks for | soft | Deflection 0-67% regardless of size; 0.15 vs 0.70 recall by rendering; Falcon-90M never deflects; Max's pipeline: 0% multi-turn DPO, **42.5% of SFT tokens lose their conversation's start**; counter-evidence: ufakzeka-1 at 151M | Generic vs generic + ~5k verified multi-turn dialogues on SmolLM2-135M (base and Instruct), packed per conversation, 3 seeds; plus the five Falcon-90M checkpoints and `smollm2-135M-SFT-Only` vs Instruct (what DPO does at 135M) [added in review] |
| 4 | **Tokens per parameter and capacity allocation** (Level R) | No capacity for skills below a critical frequency; the threshold rises as the model shrinks | mixed | 2,900-83,000 tok/param for working small chat models vs 670 planned; critical-ratio phase transition at Pythia-70M; tokens alone do not fix limit 1; Doge-160M at 210 tok/param: IFEval 16.8 | Identical targeted SFT on SmolLM2-135M checkpoints at 252B / 1T / 2T and Pythia-160M at 100B / 300B, 3 seeds; then a 30-60M mixing-ratio sweep |
| 5 | **Degeneration on own history** (Level R) | Repeated text raises its own probability; off-policy error accumulates | mostly soft | SmolLM2-135M 22% looped / 17% self-copy; Falcon-90M 3% / 0%; MaxGPT-2 attractors at ~18 tok/param [corrected] | Free-form and own-history chats on the five Falcon-90M checkpoints; the battery on golden vs own history |
| 6 | **Instruction persistence beyond 3 turns** (Level A) | Attention to a distant instruction decays | mixed | Falcon-90M and SmolLM2-135M decay after turn 1; LFM2.5-350M and Qwen3-0.6B hold over 3 turns; at 4B-32B persistence falls to ~30% [>=1B only] | A 6-12-turn persistence family with a headroom rule |
| 7 | **Stored world knowledge** (Level R, only if the battery asks for facts) | ~2 bits/param after ~1,000 exposures, ~1 bit at 100, up to 20x less with junk dilution | **hard** | A 150M model can hold at most ~0.3x of Qwen2.5-0.5B's knowledge. That is a **ratio of ceilings**; at ~670 tok/param on web-heavy data the real ratio is lower (how much is unsourced) [corrected in review]; TriviaQA 4.1 (SmolLM2-135M), 16.9 (360M), 4.3 (Qwen2.5-0.5B) | For the attack: a counterfactual open-book probe and a grounded-dialogue fine-tune measuring abstention |
| 8 | **Teacher signal** | The best sub-400M models were teacher-fed; Max's own tokenizer rules out logit KD from Ultra or SmolLM2 | mixed | Pretraining-KD nulls at 125M and 200M; post-training KD positive at 120M | SmolLM2-1.7B to 135M OPD pilot (public models only); for Max's model, sequence-level distillation vs none |
| 9 | **Embedding share** | Embedding parameters add no serial computation | soft (design choice) | 19-25% at 150M with 49k; Gemma's 63% table did not hurt its closed-book score | A vocab-size sweep at equal total params (on the Planck ledger) |
| 10 | **Output-head rank and depth** | Head rank <1000 compresses output | unknown / not binding | contested | 60M head-rank A/B, low priority |
| - | **Not a limit: one unique fact across ~12 turns** | Induction/retrieval heads copy a unique fact cheaply | - | Falcon-90M; every model with ~290M+ body | Tier-0 recall probe on each new checkpoint |

**Prerequisite (not a model limit): measurement.** [corrected in review: moved out of the ranked limits]
- What goes wrong:
  - Single chat scores mix knowledge with coherence.
  - Harnesses swing scores 10-24 points.
  - Substring graders pass negations.
  - Seed variance rivals recipe differences.
- Evidence:
  - ufakzeka-1's 3 seeds of one recipe span 75.2-81.2, as wide as 14 checkpoints.
  - A repaired gate read 64/64 on its own templates and 34/64 on paraphrases.
  - Max's A/B eval swings 0.17-0.21 nats between evals.
  - A constant null model scores 9.55 on MT-Bench.
- Test:
  - Train one recipe with 3-5 seeds and measure the battery spread.
  - Regrade the probe transcripts with the missing fixtures.

### 5.2 Where the diagnoses and the review disagree, and how this report resolves it

1. **Top rank.**
   - The diagnoses: capacity put knowledge first, recipe put training signal first, behavior put updating first. Each used a different criterion.
   - The review: the draft's single ranking mixed two questions.
   - **Resolution: two columns (1.3).** Under Level R, updating cannot bind, because the Qwen2.5-0.5B baseline fails it too. Updating and persistence decide Level A.
2. **Is updating the 150M-specific limit?**
   - The draft called the data undisputed. The review showed only SmolLM2 has a clean measurement, and four models fail the items at d0, so no size result exists between 135M and 0.6B.
   - **Resolution:** updating does not explain why 150M looks worse than 500M. It is the gate to strict chat for every sub-1B model tested, and its cause is unknown. The repaired `ft_test.py`, U_same on all models and an LFM2-700M/1.2B run settle it.
3. **Binding and updates: one limit or two?**
   - The data separate them. Binding has a same-family size step (confounded by 2x tokens), and some recipes solve it at ~300M. No recipe below 2.6B solves updates.
   - They share the first-mentioned shortcut, so one data attack targets both. This report lists them separately.
4. **Mention-order numbers** (0.98/0.31 vs 0.83/0.40 for SmolLM2-135M base): recipe used one question (n=42/54), behavior pooled both (n=96 each). Both are correct.
5. **Deflection:** the largest failure class, soft, and the cheapest fix. It sits inside limit 3.
6. **Tokens:** the same phenomenon under three names. The critical-density mechanism explains why data density and token count trade off.
7. **Degeneration** is placed 5th. Falcon removes it at 90M, but distillation raises the risk.
8. **Falcon's "MT-Bench"** is 2-turn MT-Bench via evalchemy, with the judge undetermined, not MT-Bench-101.
9. **The lost token-curve lead** (0.04 / 0.23 / 0.56 at 252B / 1T / 2T) has no surviving raw data. It is used only as a reason to re-run the measurement.
10. **Generation corrections: 7 of 8 or 6 of 8?**
    - The review wrote "7 of 8 fail".
    - The saved transcript shows Falcon-90M passing K_time greedy ("Arrive 4:00 PM"), consistent with the audit's 21 of 24.
    - So 6 of 8 fail all three items, Falcon passes 1 and Qwen3-0.6B passes 2.

---

## 6. Where a wall-break could come from

All of these are **hypotheses**. A "wall-break" means a capability-per-parameter result that holds on a curve (for example 30M / 60M / 150M). It is judged on the sealed multi-turn battery against the same-size baseline, not on a single lucky run. The levers are ranked by (size of the limit attacked) x (plausibility) / (cost).

| rank | lever | limits | why it could work | prior art and gap | cheapest test |
|---|---|---|---|---|---|
| 1 | **Interference-balanced dialogue in pretraining, above the critical density, on an accuracy-gated distance curriculum**: every correction paired with a no-correction twin and a two-slot variant; answer mentioned second as often as first; fact-to-question distance ramped as measured accuracy passes a gate; turns where the assistant repeated the old value [corrected in review: the draft sampled gaps uniformly] | 1, 2, 4 | Small models skip rare skills entirely; removing both "first wins" and "last wins" shortcuts forces real binding and updating; in tiny recall cells a shaped distance ramp reopens a boundary a uniform curriculum cannot (4 of 5 seeds vs 0 of 9 at L=256), and gating the ramp on measured accuracy locks in 6 of 6 at L=512 where the fixed ramp collapses (Boesch and Wee, ~29K-parameter cells, not LMs) | bAbI dialog task 2; Wu et al. (37.8M); Falcon's SFT-in-pretraining (IFEval only); Boesch and Wee (toy cells); not found as a pretraining-mix intervention in a chat LM | Repaired `ft_test.py` first; then a 30-60M mixing-ratio sweep (0 / 0.5 / 2 / 8%), gated ramp vs uniform gaps, 5 seeds, report the lock-in rate |
| 2 | **A model-written running state line**, emitted each turn, only the latest kept, trained so answers copy from it | 1, 2, 6 | Turns binding, two-hop and updates into a one-hop copy from the latest block; primacy stops mattering | SimpleTOD / SOLOIST (82-117M, task dialogue); not found in a tiny general chat model | At 30M and on SmolLM2-135M: none vs state line, 3-5 seeds, held-out templates |
| 3 | **A recency or overwrite path in a few attention layers** (tweak of known work) [relabeled in review]: forget gate (Forgetting Transformer, gated delta rule), stick-breaking, Selective Attention; plus whether per-head gating already reduces primacy | 1 | Primacy looks like a winner-take-all bias toward early encodings; suppressing interfering content improved working memory in LLMs (in-context superposition) | Selective Attention: parameter-free, gains at small scale on C4, +2.15z, not significant after Bonferroni at 1.2B; stick-breaking: 192 vs 128 key-value pairs in a 2-layer toy, beats softmax on RULER variable tracking at 1B [>=1B only]; gated attention cut first-token attention 46.7% to 4.8% at 1.7B [>=1B only]; DZ-TiDPO temporal attention bias at 3.8B/7B [>=1B only]; nothing at 150M | 30-60M A/B: softmax vs forget gate vs Selective Attention (gate on/off), on U_same, two-slot and no-update items, 5 seeds. The draft's "free now" run on Max's 124M A/B checkpoints is dropped: they are on the Lambda box, and at ~9 tok/param both arms are likely at chance |
| 4 | **Previous-token mixing plus speaker-role embeddings** | 2 | Multi-token keys need a previous-token path; role embeddings separate I / you / third party | Canon synthetic at 50-124M, real only >=1B; short conv +0.5 recall in toy cells; TransferTransfo at 117M | 60M A/B on binding and perspective |
| 5 | **Multi-turn on-policy distillation and own-history SFT** | 5, 1 | Trains recovery from its own drift | MiniLLM 120M single-turn; Guided-OPD 0.6B agents; conversational multi-turn OPD only >=1.7B | SmolLM2-1.7B-Instruct to SmolLM2-135M as a **method test on public models only**; Max's model needs a sequence-level or cross-tokenizer variant [corrected in review] |
| 6 | **Knowledge offload**: retrieval-trained reader, knowledge-light pretraining, trained abstention | 7 | Parameters go to skill; reading from context works at 106M | RETRO / BART-large; KLLM at 135M (preprint); [IDK]-token training diverged at 70M/160M | Grounded-dialogue fine-tune of SmolLM2-135M with conflicting passages; measure abstention |
| 7 | **Sparse memory tables**, only if the constraint counts active parameters | 7 | Lookup, not compute, holds facts | Memory+ 134M; MoLE 160M active; contradicted at 151M by UltraMem | Only after D1 |
| 8 | **A large embedding table as an entity store** | 7 | Gemma 3 270M's 168M table, best small-model closed-book score | One n=40 observation | Vocab-size A/B at equal body on an enlarged knowledge twin |

Not candidates:
- **Looping for knowledge:** looping does not raise knowledge capacity.
- **Multi-token prediction:** exact MTP hurts at 300-600M.
- **Tokenizer tricks:** SuperBPE and byte-level both lose at this budget.
- **Test-time thinking:** tiny models loop, and thinking eats the context that is the conversation. Max already ruled out free-form thinking.

---

## 7. Feasibility on Max's hardware

### 7.1 What is available

- **Titans** (Lambda, 10x TITAN RTX 24 GB, Turing sm_75).
  - Only the 4 edge cards sustain load.
  - Stack limits: fp16 only, no bf16, no FlashAttention-2. Driver 470 pins CUDA 11.8 and vLLM <= 0.10.1.1, which predates Qwen3.5 and Gemma 4. A driver >= 525 would unlock current stacks.
  - Ultra's pretraining ends about Dec 3, and the cards free around Dec 7-14. This report assumes the Titans are **unavailable until mid-December** and that nobody touches the Ultra job.
- **RTX 5070 12 GB.**
  - **Status conflict:** the brief says it is lent out; Max's Planck note says it is free as of the afternoon of Sep 23. 7.3 gives both schedules.
  - Expected ~32k tok/s for the 150M reference (range 24-40k). This is **unmeasured**; one `bench_micro.py` run settles it.
- **M5 Mac, 24 GB.**
  - Training rates: ~2.7k tok/s at 113M, 3.9-4.8k at 60M, 6.9-7.9k at 35M, 10.6-11.8k at 18.5M [unverifiable: a contended re-run got about half].
  - Metal's recommended working set is 19.07 GB. PyTorch's default MPS hard cap is 1.7x that (32.4 GB), past physical RAM, so a job can oversubscribe and thrash instead of raising an out-of-memory error (`followup/loop.verify.md`).
  - **Hard rule: one model process at a time, nothing else resident.** The Mac cannot generate data and train at the same time.

### 7.2 What things cost

- **150M reference on the 4 edge Titans** (~68k tok/s): 100B tokens in ~17 days, 300B in ~51 days. Budget up to 1.5x for the NorMuon overhead.
- **The 5070 alone:** ~36 days per 100B tokens (estimate).
- **SmolLM2-135M's 2T tokens** would take ~11 months of the 4 cards.
- **Ablations:** per GPU-week, 38-50 runs of 60M on 1.2B tokens, or 6-8 runs of 150M on 3B tokens.
- **Data:**
  - ~2-3B tokens of free dialogue exist (4.3).
  - Lever 1's data is mostly programmatic.
  - Full teacher generation on the Mac costs ~20-50 Mac-days per 100M tokens, so pilot 5-20M tokens first.
  - A 5070 with current vLLM does roughly 1e9 tokens/week for a 1.7-4B teacher (low confidence).

### 7.3 The fastest trustworthy loop

1. **Fix measurement first** (days, Mac, no training).
   - Seal the probe and battery with held-out templates and paraphrases.
   - Add to the battery: two-slot and no-update twins, U_same, a position control, and a key-in-correction item.
   - Enlarge `khard` to ~400 items (SE ~0.02).
   - Harden the graders against negations, guess lists, hypotheticals, first person with "you", incidental numbers and truncation.
   - Score every model in its own template and in plain format.
   - A -3-point non-inferiority margin at 80% power needs about **2,200-2,600 pooled items** [corrected in review from "about 2,000"].
   - Seeds: 3 for continuous outcomes; 5+, reporting rates, for binary lock-in.
2. **Cheap runs on public models, one process at a time.**
   - (a) Chat probe and battery on **LFM2.5-230M** and **Falcon-H1-Tiny-90M**. Run Falcon's battery on CPU or CUDA; MPS crashed on the Mamba path. [added in review]
   - (b) The new battery items on all 8 models, plus LFM2-700M and LFM2-1.2B.
   - (c) The **repaired** `ft_test.py` (7.4): SmolLM2-135M-I with 5 seeds, then 360M-I.
   - (d) The five Falcon-90M checkpoints (verified on the HF API), and `smollm2-135M-SFT-Only` vs Instruct.
   - (e) Zero-cost: the battery on **MaxGPT-3** `checkpoints/final.pt` and `final_sft.pt` (235M, ~85 tok/param, local). This needs a small adapter to the log-prob scorer. [added in review]
   - (f) Optional, needs Max: the 124M A/B checkpoints, if they were kept, are on the Lambda box. Expect a null result.
3. **SFT arms on public checkpoints:** the limit-3 test and the limit-4 checkpoint test.
4. **From-scratch pairs at 30-60M on 1-2B tokens:** the mixing sweep with the gated curriculum, state line vs none, recency path vs none.
5. **Titans after mid-December:**
   - Train 150M at 100B tokens on the winning mix; extend to 300B if step 3 shows token-limited skills.
   - Then post-train: the fixed packer, at most one epoch of multi-turn DPO, and distillation.

**Two schedules** (my arithmetic from 7.1; estimates) [added in review]:

| step | Mac only, serial | with a free 5070 |
|---|---|---|
| 1 | days of coding | same |
| 2 | roughly 1-3 days of serial runs | under a day for CUDA items |
| 3 (~20-30 short runs) | roughly 2-5 days | about a day |
| 4 (18-24 runs) | 30M/1B at 6-9k tok/s: 1.3-1.9 days per run, **~0.8-1.5 months**; 60M/2B at 4-5k: 4.6-5.8 days per run, **~2.8-4.6 months**; add ~60% for 5 seeds on lock-in | 9-17 hours per run at the unmeasured 32k tok/s, **~1-2.5 weeks**, likely less at 30-60M |
| data generation | cannot overlap training | Mac generates while the 5070 trains |

Steps 1-2 need no new hardware and can finish this week. Without the 5070, step 4 does not finish before December. The Mac-only plan is:
- Steps 1-3 now.
- A reduced step 4 (30M only, the two most promising arms) until December.
- The rest on the Titans.

### 7.4 `ft_test.py`: safe settings and a design that can decide something [corrected in review]

**Memory (estimates).** These assume fp32 weights, AdamW, and an MPS backward pass that stores attention scores.
- The default batch 16 with sequences up to 640 tokens needs ~20-35 GB for SmolLM2-135M.
- The draft's batch 8 still needs ~20-33 GB for 360M.
- Safe settings: **batch 4 for 135M (~6-10 GB), batch 2 for 360M (~9-13 GB)**, gradient accumulation to an effective 16, max length 512 or gradient checkpointing.
- Before running:
  - Unload LM Studio.
  - Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` (a 13.3 GB hard cap here) and `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6`.
  - Do a 5-step dry run while watching `memory_pressure`.
  - Remember the queue's memory guard only gates the start of a job, not its peak.

**Design fixes.**
- **Kill the shortcut.** In training and eval, the gold is always the most recently mentioned weekday. Add to both:
  - two-slot items: a second event's day comes after the correction, and the question asks about the first event;
  - no-update twins: a later unrelated weekday appears, and the gold is the original day.
- **Pass rule:** at least 0.8 latest-wins at d10 on held-out wording **and** at least 0.8 on the two-slot and no-update items.
- **Seeds:** the script fixes `torch.manual_seed(0)`. Run 5 seeds at 135M and at least 3 at 360M, and report the lock-in rate.
- **Knowledge:** report the closed-book change as a paired CI on an enlarged `khard`.
- **Reading the result:**
  - Passes with controls on most seeds at 135M: updating is trainable at a 106M body.
  - Passes the main items but fails the controls: the model learned a shortcut, which tells us nothing.
  - Flat at 135M but moving at 360M: a threshold in Max's range, or the 2x token difference.
  - Flat everywhere: the fix needs pretraining-scale signal, a curriculum or a state mechanism. This is still not proof of a capacity limit.
- **Time (estimate):** ~20-40 minutes of training per 135M seed and 1-2 hours per 360M seed, plus re-scoring.

---

## 8. Decisions Max needs to make before designing

- **D1. What does "150M" constrain?**
  - Total stored parameters: the most credible count, and the one the Planck notes favor.
  - Non-embedding body.
  - Active parameters: makes lever 7 the strongest knowledge attack.
  - Bytes on device: at roughly equal bytes, Spectra's TriLM 560M vs FloatLM 190M (both 300B tokens) gains ~1.7 commonsense points (48.4 vs 46.7) and roughly ties on knowledge [corrected in review: scale added].
  - Latency: favors hybrids and shallower stacks.
  - Recommendation: count total parameters, and report the body too.
- **D2. What is the model for?** Conversational mechanics over context plus retrieval is feasible at 150M. An open-domain assistant that knows things is not. Report knowledge-bearing scores separately.
- **D3. What does success mean?** Pre-register RC-12 (12-turn conversations, own history, same-type distractors, sealed templates, shortcut rows, 3-5 seeds, mutation-tested graders).
  - Level R: paired lower bound of at least -3 points vs Qwen2.5-0.5B-Instruct. That needs **2,200-2,600 pooled items** [corrected in review].
  - Level A: all criteria plus a blind human win-or-tie of at least 50%. Include LFM2-2.6B as a reference that passes our correction items, so the bar is shown to be reachable.
  - Baselines, all in one harness: Qwen2.5-0.5B-Instruct, Qwen3.5-0.8B, LFM2.5-230M, LFM2.5-350M, Falcon-H1-Tiny-90M and siblings, SmolLM2-135M/360M, Doge-160M-Instruct.
  - Add a headroom rule, so a criterion where every baseline is at the floor cannot be "passed" trivially.
- **D4. Do scaffolds count?** A harness-maintained state card or retrieval should be a separate labelled condition. A state line the model writes and reads itself is part of the model.
- **D5. One size or a curve?** Develop at 30-60M and claim a curve (for example 20M / 60M / 150M).
- **D6. Tokenizer and teacher: decided, with consequences to accept.**
  - The decision: Planck gets its own small tokenizer (Planck note, 2026-09-23).
  - What it buys: ~19-25M of body at 150M, and far more at 20M, where a 49k table would be ~60% of the parameters.
  - What it gives up: same-tokenizer logit KD from Ultra and SmolLM2, weight inheritance from Ultra, and token-level OPD unless cross-tokenizer methods work below 1B.
  - Revisit only if the SmolLM2 OPD pilot shows a large multi-turn gain.
  - [corrected in review: the draft listed this as open]
- **D7. Compute plan.** Confirm the 5070's status first, because it changes step 4 from months to weeks. Wait for the Titans in mid-December. Pausing Ultra (Max's note says he OK'd it) is not assumed.
- **D8. Data rights.**
  - Bulk training text comes from Apache-2.0 open-weight teachers only.
  - Claude is used for prompts, rubrics, gold sets and reading outputs.
  - Decide whether WildChat and UltraChat (OpenAI-written assistant turns, ODC-BY and MIT licenses) are acceptable.

---

## 9. Sources

### Verdict and landscape
- LFM2.5-230M: https://huggingface.co/LiquidAI/LFM2.5-230M , config https://huggingface.co/LiquidAI/LFM2.5-230M/resolve/main/config.json (14 layers, 8 conv + 6 attention, vocab 65,536, d=1024, 229,693,184 params, fetched today) , https://www.liquid.ai/blog/lfm2-5-230m
- LFM2.5-350M: https://huggingface.co/LiquidAI/LFM2.5-350M ; LFM2 report https://arxiv.org/abs/2511.23404 ; LFM2-350M https://huggingface.co/LiquidAI/LFM2-350M ; LFM2-2.6B https://huggingface.co/LiquidAI/LFM2-2.6B
- Falcon-H1-Tiny: https://tiiuae-tiny-h1-blogpost.hf.space/ ; config https://huggingface.co/tiiuae/Falcon-H1-Tiny-90M-Instruct/resolve/main/config.json ; siblings https://huggingface.co/api/models?author=tiiuae&search=Tiny-90M ; Falcon-H1 report https://arxiv.org/abs/2507.22448 ; Falcon-H1-0.5B https://huggingface.co/tiiuae/Falcon-H1-0.5B-Instruct
- SmolLM2: https://arxiv.org/abs/2502.02737 ; https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct ; https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct ; SFT-only https://huggingface.co/HuggingFaceTB/smollm2-135M-SFT-Only ; intermediate checkpoints https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints/refs
- Doge-160M-Instruct: https://huggingface.co/SmallDoge/Doge-160M-Instruct , config https://huggingface.co/SmallDoge/Doge-160M-Instruct/resolve/main/config.json (24 x 768, vocab 32,768, tied, 152,697,744 params) , base (32B tokens) https://huggingface.co/SmallDoge/Doge-160M
- OpenELM https://huggingface.co/apple/OpenELM-270M-Instruct ; ERNIE https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT ; MiniCPM4 https://huggingface.co/openbmb/MiniCPM4-0.5B ; Hunyuan https://huggingface.co/tencent/Hunyuan-0.5B-Instruct ; MiniMind https://github.com/jingyaogong/minimind/blob/master/README_en.md
- Gemma 3 270M: https://developers.googleblog.com/en/introducing-gemma-3-270m/ , https://huggingface.co/google/gemma-3-270m-it
- Qwen2.5 https://arxiv.org/abs/2412.15115 ; Qwen3 https://arxiv.org/abs/2505.09388 ; Qwen3.5-0.8B https://huggingface.co/Qwen/Qwen3.5-0.8B
- MobileLLM https://arxiv.org/abs/2402.14905 ; MobileLLM-R1 https://arxiv.org/abs/2509.24945 ; Danube3 https://arxiv.org/abs/2407.09276 ; Granite https://huggingface.co/ibm-granite/granite-4.0-350m-base ; PleIAs https://huggingface.co/PleIAs/Baguettotron , https://huggingface.co/PleIAs/Monad ; ufakzeka-1 https://arxiv.org/abs/2609.25081 ; nanochat https://github.com/karpathy/nanochat/discussions/1 ; Supra2 https://huggingface.co/SupraLabs/Supra2-100M-Instruct ; BlenderBot https://arxiv.org/abs/2004.13637 ; micro LMs https://arxiv.org/abs/2604.19642
- Multi-IF https://arxiv.org/abs/2410.15553 ; MultiChallenge https://arxiv.org/abs/2501.17399 ; tau2-Bench https://arxiv.org/abs/2506.07982
- `lanes/census.md`, `lanes/census.verify.md`, `followup/floor.md`; panic report `/Library/Logs/DiagnosticReports/panic-full-2026-09-23-171933.0002.panic`

### Hands-on probe
- `probe/` and `lanes/probe.md`, `lanes/probe.verify.md`; Falcon K_time `probe/transcripts/tiiuae__Falcon-H1-Tiny-90M-Instruct__greedy.jsonl`
- `capacity_probe/` (items.py, khard.py, uprobe.py, results.json, results_uprobe.json, out/*.jsonl); distance table recomputed today from results.json
- Laban et al. https://arxiv.org/abs/2505.06120

### Recipe
- Architecture: https://arxiv.org/abs/2312.04927 ; https://arxiv.org/abs/2402.01032 ; https://arxiv.org/abs/2411.13676 ; https://arxiv.org/abs/2512.17351 ; https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html ; https://arxiv.org/abs/2410.17897 ; https://arxiv.org/abs/2605.20798 ; https://arxiv.org/abs/2505.06708 ; https://arxiv.org/abs/2404.07647 ; https://arxiv.org/abs/2510.25741 ; https://arxiv.org/abs/2507.10524 ; https://arxiv.org/abs/2604.21106 ; https://arxiv.org/abs/2404.19737
- Max's A/B `~/Documents/Projects/Max's AI Model/maxgpt-ultra/docs/ab_2026-09-21/README.md`; `lanes/training.verify.md`, `lanes/compute.verify.md`
- Tokenizer: https://arxiv.org/abs/2407.13623 ; https://arxiv.org/abs/2605.01188 ; `followup/tokenizer.verify.md`; decision in `(private project notes)`
- Data: https://arxiv.org/abs/2508.10975 ; https://arxiv.org/abs/2310.07301 ; https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk ; https://arxiv.org/abs/2603.16759 ; https://arxiv.org/abs/2510.14865 ; https://arxiv.org/abs/2505.18091 ; https://arxiv.org/abs/2305.16264 ; https://arxiv.org/abs/2607.04969 ; https://arxiv.org/abs/2607.12831 ; https://arxiv.org/abs/2404.05405 ; https://www.anthropic.com/legal/consumer-terms ; `followup/chatdata.verify.md`
- Free corpora (licenses checked on the HF API today): https://huggingface.co/datasets/allenai/WildChat-4.8M (ODC-BY) ; https://arxiv.org/abs/2212.10465 , https://huggingface.co/datasets/allenai/soda (CC-BY-4.0) ; https://arxiv.org/abs/2305.14233 , https://huggingface.co/datasets/stingning/ultrachat (MIT) ; https://arxiv.org/abs/2408.03617 , https://huggingface.co/datasets/styfeng/TinyDialogues (MIT) ; https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k (Apache-2.0); sizes from `lanes/data.md`, `lanes/data.verify.md`
- Training: https://arxiv.org/abs/2401.00448 ; https://arxiv.org/abs/2404.10102 ; https://arxiv.org/abs/2502.15938 ; https://arxiv.org/abs/2509.02046 ; https://arxiv.org/abs/2502.08606 ; https://arxiv.org/abs/2410.17215 ; https://arxiv.org/abs/2609.14715 ; https://arxiv.org/abs/2503.16870 ; https://arxiv.org/abs/2502.03460 ; https://arxiv.org/abs/2510.07227 ; https://arxiv.org/abs/2503.19206
- Post-training: https://arxiv.org/abs/2306.08543 ; https://arxiv.org/abs/2306.13649 ; https://arxiv.org/abs/2606.15912 ; DZ-TiDPO https://arxiv.org/abs/2512.03704 (Phi-3.5-mini 3.8B, Qwen2.5-7B) ; https://arxiv.org/abs/1911.03860 ; https://arxiv.org/abs/2502.12143 ; `posttrain/sft_data.py` (Max's repo, read-only) ; `recipe_work/pack_sim.py`
- Decoding: https://arxiv.org/abs/2512.12895 ; https://arxiv.org/abs/2206.02369 ; https://arxiv.org/abs/2210.14140 ; https://arxiv.org/abs/2506.13681 ; https://arxiv.org/abs/2305.14739
- Memory and tools: https://arxiv.org/abs/2005.00796 ; https://arxiv.org/abs/2109.07506 ; https://arxiv.org/abs/2104.07567 ; https://arxiv.org/abs/2603.11513 ; https://arxiv.org/abs/2410.00004 ; https://arxiv.org/abs/2412.09764 ; https://arxiv.org/abs/2411.12364 ; https://arxiv.org/abs/2503.15798 ; https://arxiv.org/abs/2601.21204 ; https://arxiv.org/abs/2312.09241 ; https://arxiv.org/abs/2412.06676

### Limits map
- Interference: https://arxiv.org/abs/2603.00270 ; https://arxiv.org/abs/2506.08184 ; Boesch and Wee https://arxiv.org/abs/2609.16183 ; in-context superposition https://arxiv.org/abs/2604.09670 ; https://arxiv.org/abs/2306.00946 ; bAbI dialog https://arxiv.org/abs/1605.07683
- Binding: https://arxiv.org/abs/2310.17191 ; https://arxiv.org/html/2407.10827 ; entity tracking https://arxiv.org/abs/2608.18083 with `lanes/context.verify.md` claim 5 ; https://arxiv.org/abs/2505.20896
- Knowledge: https://arxiv.org/abs/2404.05405 ; https://arxiv.org/abs/2505.24832 ; https://arxiv.org/abs/2604.24827
- Measurement: https://arxiv.org/abs/2306.05685 ; https://arxiv.org/abs/2402.14762 ; https://arxiv.org/abs/2410.07137 ; https://arxiv.org/abs/2603.01423 ; `lanes/eval.verify.md` row 12
- `diagnosis_*.md`, `CRITIQUE.md`, `followup/floor.verify.md`, `followup/bits.verify.md`, `followup/archfp.verify.md`

### Levers
- Selective Attention https://arxiv.org/abs/2410.02703 ; Forgetting Transformer https://arxiv.org/abs/2503.02130 ; Gated DeltaNet https://arxiv.org/abs/2412.06464 ; stick-breaking https://arxiv.org/abs/2410.17980 ; TransferTransfo https://arxiv.org/abs/1901.08149 ; SOLOIST https://arxiv.org/abs/2005.05298

### Feasibility and decisions
- `lanes/compute.md`, `lanes/compute.verify.md`, `LAMBDA.md` (Max's repo); https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html ; https://github.com/vllm-project/vllm/releases
- PyTorch MPS variables https://docs.pytorch.org/docs/2.14/mps_environment_variables.html and `followup/loop.verify.md` claim 5
- `capacity_probe/ft_test.py`, `capacity_probe/capq_tail3.sh`; MaxGPT-3 checkpoints `~/Documents/Projects/Max's AI Model/maxgpt-3/checkpoints/` (listed only)
- Spectra https://arxiv.org/abs/2407.12327 with `lanes/frontier.md` 1.7

### Claims not used (refuted or withdrawn)
- Titans "340M 98% needle": refuted (TTT, easiest passkey).
- TII "multi-turn low vs Qwen-0.6B" as chat evidence: refuted (BFCL function calling).
- "Every small recipe is off-policy SFT plus offline DPO": refuted.
- "No sub-300M multi-turn evaluation exists": refuted by BlenderBot 90M. The narrow form holds.
- "NorMuon trailed, then overtook": refuted as eval noise.
- Induction-head loop mechanism (2505.13514): withdrawn by its authors.
- "~100M models cannot be prompted into chat behavior": not supported.
- The 4.3-4.8 bytes/token tokenizer target: refuted.
- "Corrections fail at every size" and "no gradient 135M-0.6B": overstated [corrected in review].
- "At distance 0 the 360M gets 0.97-1.00" unqualified: true only for U_same [corrected in review].
- "Entity tracking crosses human level at 410M" as a chat threshold: that is a likelihood read-out.

---

## Appendix: Reviewer notes

- **Source text.** The draft never reached disk. This report was built from the text recovered from workflow journal entry 72 (scratchpad `critic/draft_clean.md`). The harness also blocked writing REPORT.md, so this text went back to the orchestrator.
- **Fixed from the review, blockers and majors:**
  - B1: 7.4.
  - B2: 1.3, 3.3, 5.1.
  - M1: box, 1.2.
  - M2: 1.3, 5.2.
  - M3: 3.3, 5.1.
  - M4: 7.3.
  - M5: 4.5, 5.1, 6.
  - M6: 4.3, 7.3.
  - M7: the box.
- **Fixed from the review, minor issues:** packing (4.5), measurement moved out of the ranked limits (5.1), body gap (2), power (7.3, 8), knowledge ceiling ratio (5.1), ternary scale (8), the primacy label tag (5.1), D6 (4.2, 8), omitted census models (2), seed counts (7.3).
- **Checked today against primary sources:**
  - arXiv abstracts 2609.16183, 2604.09670, 2512.03704, 2410.02703 and 2608.18083.
  - The Doge-160M card, config and parameter count, and the LFM2.5-230M config.
  - Dataset licenses, and the existence of the SFT-only and Falcon sibling checkpoints.
  - The U tables in `results.json` and `results_uprobe.json`, and the Falcon K_time transcript.
- **Changed from the review's wording:**
  - "7 of 8 fail in generation" is stated as 6 of 8, because the transcript shows Falcon passing K_time.
  - The conditional strict odds are 30-40%, and they apply only if the repaired test passes with its controls.
- **Not done:**
  - The review's ~35 spot-checks were not repeated.
  - No probe was re-run (the rule for this step). The new items, LFM2.5-230M, Falcon on the battery and the repaired `ft_test.py` are still proposals.
