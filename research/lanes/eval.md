# Lane: measuring "holds a real back-and-forth conversation" at ~150M

Research date 2026-09-23. Every number carries a source URL and the scale it was measured at. "Unsourced" or "unmeasured" means exactly that. All scores are as reported by the named source; scores from different harnesses are not comparable (see finding A3, which is the main reason this matters).

---

## Bottom line

1. There is no published measurement of multi-turn coherence beyond 2 to 3 turns for any model under 300M parameters. The dedicated multi-turn benchmarks (MT-Bench-101, MT-Eval, MultiChallenge, Lost-in-Conversation, TurnWise, IFBench's multi-turn mode) were all run on models of 6B and up. So "the smallest model that really chats is ~0.5B" is a vibe, not a measurement, and a public, programmatic multi-turn battery for tiny models would be a real contribution on its own.
2. On the chat and instruction metrics that do exist, the premise does not hold. LFM2.5-350M (28T tokens) reports IFEval 76.96 and Multi-IF 44.92 against Qwen3.5-0.8B's 59.94 and 41.68. Falcon-H1-Tiny-90M-Instruct (91M params) reports "MT Bench (avg)" 4.33 against 2.68 for SmolLM2-135M-Instruct. Gemma 3 270M-it (about 100M non-embedding params) reports IFEval 51.2 against Qwen2.5-0.5B-Instruct's 27.9 to 31.6. All of these are vendor-measured, and all three were trained at roughly 9k to 80k tokens per parameter.
3. Where sub-1B models really do fall off is knowledge-heavy and hard open-ended prompts: Arena-Hard goes from 6.5 (Qwen3-0.6B) to 36.9 (Qwen3-1.7B), and MultiChallenge is 18.9 for Qwen3.5-0.8B even with thinking on. The judged chat benchmarks (MT-Bench, AlpacaEval, Arena-Hard, WildBench) mix knowledge and reasoning in with conversational skill, so at 150M they mostly measure the knowledge you cannot have.
4. The same model's published score moves by 10 to 24 points depending on who ran the harness (Gemma 3 270M IFEval: 51.2 from Google, 27.44 from TII). Every baseline has to be re-run in one harness, and no published number can be the bar.
5. Recommended battery: (a) likelihood probes that also work on base checkpoints during pretraining, (b) about 1,450 scripted multi-turn conversations in 8 programmatic families (recall at distance, update and abstain, reference, instruction persistence, instruction override, sharded-vs-full, persona, social basics), each run on golden history and on the model's own history, with loop and role-leak diagnostics on every turn, (c) public anchors (IFEval, IFBench, Multi-IF English, CoQA), and (d) one validated pairwise judge used only for open-ended quality, plus a blind human test for the final claim.
6. A success claim outsiders would believe: at 150M total params or fewer, non-inferior to Qwen2.5-0.5B-Instruct on the multi-turn composite (lower bound of the paired 95% CI no worse than -3 points) on sealed, held-out template families, with a loop rate no worse than Qwen2.5-0.5B's, and a blind human pairwise win-or-tie rate of at least 50% against it. The model to beat at this size is Falcon-H1-Tiny-90M-Instruct, not SmolLM2-135M.

---

## Detailed findings

### A. What already exists under 600M (a check on the premise)

**A1. Sub-400M instruct models with published chat and instruction-following numbers.** Parameter totals come from the HF API (`safetensors.total`). Non-embedding counts are computed as total minus vocab x hidden from each `config.json`, so treat them as approximate for the hybrid models.

| model | total / non-emb params | pretrain tokens (tok/param) | IFEval | MT-Bench | other | source |
|---|---|---|---|---|---|---|
| Falcon-H1-Tiny-90M-Instruct (Jan 2026, hybrid attn+Mamba) | 91.1M / ~74M | 800B incl. 25% SFT data (~8.8k) | 66.08 | 4.33 ("MT Bench (avg)") | AlpacaEval 9.43, LiveBench 15.69 | https://tiiuae-tiny-h1-blogpost.hf.space/ |
| SmolLM2-135M-Instruct | 134.5M / ~106M | 2T (~15k) | 29.9 (HF avg) / 30.69 (TII) | "19.8" on card (looks like a x10 typo) / 2.68 (TII) | AlpacaEval 1.52 (TII) | https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct , TII blog |
| Gemma 3 270M-it | 268.1M / ~100M (168M is embeddings, 262k vocab) | 6T (~22k) | 51.2 (Google) / 27.44 (TII) | 3.71 (TII) | AlpacaEval 4.65 (TII) | https://huggingface.co/google/gemma-3-270m-it , TII blog |
| MobileLLM-125M / 350M (chat-finetuned) | 124.6M / 345.3M | 1T (~8k / ~2.9k); the paper does not say whether the Table 5 chat checkpoints are the 1T models | n/a | 2.33 / 3.28 (GPT-4 judge) | AlpacaEval vs text-davinci-001: 24.07 / 47.08 | https://arxiv.org/abs/2402.14905 Table 5 |
| LFM2-350M | 354.5M / ~287M | 10T per card, 11T per tech report | 65.12 | n/a | IFBench 16.41, Multi-IF 32.85 | https://huggingface.co/LiquidAI/LFM2-350M , https://arxiv.org/abs/2511.23404 |
| LFM2.5-350M (Mar 31 2026) | 354.5M | 28T (~79k) | 76.96 | n/a | IFBench 40.69, Multi-IF 44.92 | https://www.liquid.ai/blog/lfm2-5-350m-no-size-left-behind |
| Granite 4.0-H-350M | ~350M | unsourced | 61.63 avg (IBM) / 61.27 (Liquid) | n/a | Multi-IF 28.70 (Liquid) | https://huggingface.co/ibm-granite/granite-4.0-350m , Liquid blog |
| SmolLM2-360M-Instruct | 361.8M / ~315M | 4T (~11k) | 41.0 (HF) / 38.78 (TII) | 3.66 (HF) / 3.8 (TII) | | https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct |
| Qwen2.5-0.5B-Instruct | 494.0M / ~358M | 18T (~36k) | 27.9 strict-prompt (Qwen) / 31.6 avg (HF) | 4.16 (HF) | | https://qwenlm.github.io/blog/qwen2.5-llm/ , SmolLM2-360M card |
| Qwen3-0.6B (non-thinking) | 0.6B / 0.44B | 36T (~60k) | 54.5 strict-prompt (Qwen) / 64.24 avg (Liquid) | n/a | Arena-Hard 6.5, Multi-IF 33.3 (Qwen) / 45.13 (LFM2 report) | https://arxiv.org/abs/2505.09388 Table 20 |
| Qwen3.5-0.8B | 0.8B | unsourced | 52.1 non-thinking (Qwen) / 59.94 (Liquid) | n/a | MultiChallenge 18.9, IFBench 21.0 (thinking, Qwen) | https://huggingface.co/Qwen/Qwen3.5-0.8B |

What this shows. On IFEval, several sub-300M models score above Qwen2.5-0.5B-Instruct. On judged 2-turn chat (MT-Bench-style), a 91M model reports a higher number than SmolLM2-360M in the same harness (4.33 vs 3.8), and higher than Qwen2.5-0.5B's number from HF's harness (4.16). That last comparison crosses harnesses, so it is suggestive only. On 3-turn instruction following (Multi-IF), LFM2.5-350M reports 44.92 against Qwen3.5-0.8B's 41.68 and Gemma 3 1B's 44.25, all in Liquid's harness. As far as published evidence goes, there is no sharp wall at 400M on these metrics. The strongest of these models share enormous token budgets (8.8k to 79k tokens per parameter) and post-training on heavy SFT or preference data. MobileLLM-350M (about 2.9k tokens per parameter) reports MT-Bench 3.28 in its own GPT-4-judged harness. That harness is not comparable to the others, so the link between token budget and chat quality is suggestive here, not shown.

**A2. What vendors say about their own tiny models.** Google says Gemma 3 270M is "not designed for complex conversational use cases" (https://developers.googleblog.com/en/introducing-gemma-3-270m/). Liquid says LFM2.5-350M is "not recommended for knowledge-intensive tasks and programming" and recommends fine-tuning for narrow uses (https://huggingface.co/LiquidAI/LFM2.5-350M). HF built the SmolTalk "everyday-conversations" set because small models trained only on public instruction data answered "Hi" by changing the topic and failed "Who are you" (https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k, used for the 135M/360M/1.7B models). So the vendors themselves do not claim real multi-turn conversation below ~400M. That is consistent with the fact that nobody has measured it.

**A3. Harness disagreement is larger than the gaps between models.** The same checkpoint gets different numbers from different reporters:
- Gemma 3 270M-it IFEval: 51.2 (Google card) vs 27.44 (TII blog). Gap: 23.8 points.
- Qwen3-0.6B Multi-IF: 33.3 (Qwen report Table 20) vs 45.13 (LFM2 report, "average accuracy across all 3 turns", greedy). Gap: 11.8 points.
- Qwen3-0.6B IFEval: 54.5 strict-prompt (Qwen) vs 64.24 average of 4 IFEval metrics (Liquid, https://arxiv.org/abs/2511.23404 appendix). These are different metric definitions.
- Qwen2.5-0.5B-Instruct IFEval: 27.9 strict-prompt (Qwen blog) vs 31.6 average of prompt and instruction level (HF).
- SmolLM2-135M-Instruct MT-Bench: "19.8" on the HF card (on a 10-point scale this has to be a typo, probably 1.98) vs 2.68 (TII).

Tiny models are especially fragile to chat-template handling, system-prompt placement (Gemma has no system role), stop tokens, and max-token caps, and these differences decide scores at this size. Implication: never compare a MaxGPT-Nano number with a published number. Re-run every baseline through the same harness.

**A4. What moves judged chat scores at 90M** (TII, same model family, same harness; https://tiiuae-tiny-h1-blogpost.hf.space/):
- DPO: SFT-pretrain checkpoint MT-Bench 3.08 to 4.33, AlpacaEval 2.96 to 9.43, IFEval 50.11 to 66.08. More than one DPO epoch degraded performance even though DPO reward kept rising.
- Mixing SFT data into pretraining ("anti-curriculum", 25% SFT data across 800B tokens, Tulu 3 repeated ≳100 times) vs a classic SFT stage: IFEval 50.11 vs 40.77 before DPO, and 66.08 vs 53.47 after.
- Caveat: TII's blog cites "MT-Bench Bai et al. (2024)", which is the MT-Bench-101 paper, and names neither the judge nor the harness. The metric may be MT-Bench-101 style rather than the original MT-Bench, so treat 4.33 as TII-internal.

### B. The multi-turn benchmarks, one by one

Rules for the "off the floor at 150M?" column: "yes (published)" means sub-600M numbers exist; "predicted" is my judgment from task content and nearest published scale, and is unmeasured.

| benchmark | measures | grading | smallest published model | 150M off the floor? | cost for Max | verdict |
|---|---|---|---|---|---|---|
| **MT-Bench** (https://arxiv.org/abs/2306.05685) | 80 questions x 2 turns across 8 categories (writing, roleplay, extraction, reasoning, math, coding, STEM knowledge, humanities knowledge); turn 2 conditions on the model's own turn 1 | GPT-4 single-answer 1-10; 85% agreement with experts on non-tie votes (70% with ties) | MobileLLM-125M 2.33, Pythia-160M 1.01, OPT-125M 1.21 (GPT-4 judge, https://arxiv.org/abs/2402.14905); SmolLM2/Gemma/Falcon rows in A1 | yes (published), but 3 of 8 categories are math/coding/reasoning and 2 are knowledge, so near-floor sub-scores dominate | needs a judge; the GPT-4 API costs money; a local judge changes the scale | secondary only, same judge for every model; never the headline |
| **MT-Bench-101** (https://arxiv.org/abs/2402.14762) | 1,388 dialogues, 4,208 turns, 13 tasks in 3 abilities: perceptivity (context memory CM, anaphora AR, separate input SI, topic shift TS, content confusion CC), adaptability (rephrasing, self-correction SC, self-affirmation SA, math MR, general reasoning GR), interactivity (clarification IC, proactive PI) | GPT-4 judge per turn, 1-10, **golden context** (reference history), dialogue score = minimum over turns; 87% judge-expert agreement | ChatGLM2-6B 5.56, Llama2-7B-chat 6.53; GPT-4 8.86 | predicted: CM/AR/TS/SI partly off the floor, MR/GR at the floor | about 4.2k judge calls | borrow the taxonomy and the golden-context design; optionally run CM/AR/TS with a local judge. Paper finding: RLHF and "chat-specific designs" gave no obvious multi-turn gains (6B+ models) |
| **MT-Eval** (https://arxiv.org/abs/2401.16745) | 1,170 multi-turn queries in 4 types: recollection, expansion, refinement, follow-up, each with a single-turn twin | LLM judge | 11 LLMs, none under 2B | predicted: recollection/follow-up partly off the floor | judge | borrow the single-turn twin idea. Finding: degradation is driven by distance to relevant content and by error propagation |
| **MultiChallenge** (https://arxiv.org/abs/2501.17399) | instruction retention, inference memory, versioned editing, self-coherence | LLM judge with instance-level rubrics | best at release 41.4% (Claude 3.5 Sonnet); Qwen3.5-0.8B 18.9 thinking (Qwen card) | predicted: at or near the floor | frontier judge | skip as a metric; its 4 categories map directly onto programmatic families F1, F4, F5, F7 below |
| **Multi-IF** (https://arxiv.org/abs/2410.15553) | 4,501 conversations x 3 turns, 8 languages; IFEval constraints accumulate across turns | **programmatic** (IFEval verifiers); average of instruction/conversation x strict/loose | in the paper, Llama 3.1 8B: 0.688 / 0.615 / 0.542 over turns 1-3; o1-preview 0.877 to 0.707. Sub-600M: LFM2-350M 32.85, LFM2.5-350M 44.92, Qwen3-0.6B 33.3 or 45.13 | yes (published, as multilingual averages; a 150M English-only model will fail the other 7 languages) | free | **run the English subset** as an anchor |
| **Lost in Conversation** (https://arxiv.org/abs/2505.06120, https://github.com/microsoft/lost_in_conversation) | the same task given fully in one turn vs "sharded" across turns vs concatenated | programmatic for code, SQL, API calls, math; LLM-based for data-to-text and summary; needs an LLM user simulator | 15 models, smallest Llama-3.1-8B; average drop 39%, aptitude -16%, unreliability +112%; recap +16.1 (GPT-4o-mini) and +17.5 (GPT-4o) points; temperature 0 does not fix it | predicted: the tasks sit at the floor for 150M | simulator plus judge | **adopt the protocol, not the tasks**: family F6 below uses scripted shards of tasks a 150M model can do in one turn |
| **TurnWise** (https://arxiv.org/abs/2603.16759, 2026) | multi-turn answer vs an equivalent single-turn answer, up to 8 user turns, ~4.7k tokens | GPT-4.1 pairwise (AlpacaEval prompt) | smallest 7B (Olmo 3 7B: 36.8 vs 42.2 AlpacaEval) | predicted: marginal | API judge | borrow the "multi-turn gap vs own single-turn" metric. Finding: 10k TurnWiseData conversations in DPO took TurnWise-Self from 35.0 to 41.2 at 7B |
| **IFEval** (https://arxiv.org/abs/2311.07911) | ~500 prompts (541), 25 verifiable instruction types | programmatic | many; see A1 | yes (published) | free | run it, but **contamination caveat**: Tulu 3 Persona IF (29,980 prompts) was built on IFEval's constraint taxonomy (https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) and sits inside common SFT mixes |
| **IFBench** (https://arxiv.org/abs/2507.02833) | 300 prompts, 58 new out-of-domain constraints; includes a multi-turn mode (constraint arrives in turn 3 as "rewrite your answer") | programmatic | paper: smallest 7B; LFM2-350M 16.41, LFM2.5-350M 40.69, Qwen3-0.6B 19.75 (Liquid harness) | yes (published) | free | **run it**, as the check against IFEval overfitting (the paper's main finding is exactly that overfitting) |
| **CoQA** (https://arxiv.org/abs/1808.07042) / QuAC (https://arxiv.org/abs/1808.07036) | conversational reading comprehension with coreference: 127k questions over 8k conversations; human F1 88.8 | programmatic F1 (in lm-evaluation-harness) | no sub-600M score found in primary sources (unsourced) | predicted: yes for CoQA (answers are extractive from a passage); QuAC is harder | free | **run CoQA** as the public history-dependent QA anchor |
| **PersonaChat** (https://arxiv.org/abs/1801.07243) / **Dialogue NLI** (https://arxiv.org/abs/1811.00671) | persona consistency, framed as NLI | classifier or candidate ranking | not found for sub-600M | predicted: yes as likelihood ranking | free | use as a **likelihood probe**: entailed vs contradicting utterance given the persona |
| **DailyDialog** (https://arxiv.org/abs/1710.03957) style metrics | BLEU, distinct-n against references | n-gram | n/a | n/a | free | do not use as a quality score: word-overlap metrics correlate "very weakly" with human judgment on Twitter and "not at all" on Ubuntu (https://arxiv.org/abs/1603.08023). Use distinct-n only as a degeneration diagnostic |
| **BotChat** (https://arxiv.org/abs/2310.13650) | LLM-to-LLM dialogues grown from real opening utterances, judged for human-likeness | GPT-4 judge | 7B-class | predicted: at the floor | frontier judge | skip; self-chat drift is at most a diagnostic |
| **ConvBench** (https://arxiv.org/abs/2403.20194) | multi-turn **vision-language** conversation | judge | n/a | n/a | n/a | not applicable (vision) |
| **WildBench** (https://arxiv.org/abs/2406.04770) | 1,024 real tasks; over 20% of conversations have more than 2 turns | GPT-4-Turbo with checklists; WB-Reward Pearson 0.98 with Arena top models | not found for sub-600M | predicted: at or near the floor (knowledge-heavy real requests) | API judge | skip |
| **AlpacaEval 2 LC** (https://arxiv.org/abs/2404.04475) | 805 single-turn prompts; length-controlled win rate | LLM judge; LC raised Spearman with Arena from 0.94 to 0.98 | Falcon-90M 9.43, Gemma-270M 4.65, SmolLM2-135M 1.52 (TII) | yes, barely | API judge | skip (single-turn, costs money, and gameable: a crafted constant "null model" reached 86.5% LC win rate, 83.0 on Arena-Hard-Auto and 9.55 on MT-Bench, https://arxiv.org/abs/2410.07137) |
| **Arena-Hard-Auto** (https://arxiv.org/abs/2406.11939) | 500 hard prompts | LLM judge; 98.6% correlation with human rankings | Qwen3-0.6B 6.5 (non-thinking) | predicted: at the floor | API judge | skip |
| **BabyLM eval** (https://github.com/babylm/evaluation-pipeline-2025) | BLiMP, BLiMP-supplement (dialogue and questions), EWoK, entity tracking, WUG, reading times, (Super)GLUE finetuning; 2025 had an interaction track | likelihood/minimal pairs, zero-shot | ~100M-word models: entity tracking baseline 28.06, best 36.03 (per the 2025 findings, https://aclanthology.org/2025.babylm-main.28/) | yes | free | borrow the minimal-pair method for pretraining-time probes; entity tracking is a proxy for state tracking across turns. A dialogue-only BabyLM "excel[led] at dialogue continuation prediction in a minimal pair setting" while underperforming elsewhere, and DPO helped its dialogue benchmark (https://arxiv.org/abs/2510.20358) |
| **LoCoMo** (https://arxiv.org/abs/2402.17753), **LongMemEval** (https://arxiv.org/abs/2410.10813) | long-term memory: ~300 turns and ~9k tokens average (LoCoMo); 500 questions, 5 abilities incl. knowledge updates and abstention (LongMemEval) | QA accuracy | frontier and long-context models | out of scope at short context | | borrow "knowledge update" and "abstention" as families F2 |
| **Newer 2026 work** | SEQUOR (https://arxiv.org/abs/2605.06353): constraints set up to 15 turns back; accuracy falls >11% with length, >40% with several constraints at once. Hy-MultiTurn (https://arxiv.org/abs/2607.29196): Chinese, 209 tasks, 12-76 turns, best config 41.1%. "Models Recall What They Violate" (https://arxiv.org/abs/2604.28031): knows-but-violates rates of 8% to 99%. "When Attention Closes" (https://arxiv.org/abs/2605.12922): the goal-token attention ratio falls over turns; forcing it closed cut Mistral's 20-fact recall from near-perfect to 11%; residual probes predict recall with AUC up to 0.99. "What if LLMs Ate Their Words" (https://arxiv.org/abs/2609.05882): neutralizing the model's own past turns changes outcomes, and 63.7% of 237 degraded trajectories had at least one turn-level fix | mixed | frontier | at the floor as benchmarks | | borrow: test recall **and** adherence separately (F4); golden vs own history (every family); attention-to-goal as a diagnostic |

### C. Measurement lessons from Max's own work (Daimax, MaxGPT)

- A suite that rewards silence is worse than no suite: Daimax's `noRefusal` check scored an empty answer 0.47, because an empty string contains no refusal phrase. Every case now has to score below half on an empty answer (`~/Documents/Projects/daimax/docs/BENCHMARK.md`).
- Looping was invisible until it was measured directly. SmolLM2-360M led the Instant tier while looping in 4 of 33 runs; Qwen2.5-0.5B looped in 0 of 33 (3 runs x 11 cases, T=0.7). The first metric, global 5-gram uniqueness, flagged a long correct answer as a loop (0.387). The fixed metric measures **consecutive** repetition coverage with a 0.25 threshold (`~/Documents/Projects/daimax/src/core/bench/degeneracy.ts`). Port it as-is.
- Variance: the same Qwen3-0.6B config measured pass 0.574 and 0.750 hours apart at T=0.7. Single-run differences under ~0.10 were not signal on an 11-case suite (BENCHMARK.md). An 11-case suite is far too small. See the power numbers in D6.
- Sample-level failures seen in MaxGPT-2 (110M): "Bitcoin.com" attractor loops, fixed at sampling time with repetition penalty 1.2; "confident nonsense" on factual questions (`~/Documents/Projects/Max's AI Model/WRITEUP_NOTES.md`). Both have to be separable in the battery: loops count as a coherence failure, wrong facts count as a knowledge failure.

---

## D. The proposed battery: MTB-150 (multi-turn battery)

Design principles, in Max's BENCHMARK.md style: programmatic wherever possible; every grader validated against real observed outputs from the baselines and mutation-tested; the final question never contains the answer; slot values drawn from pools so a grader cannot pass by keyword luck; an LLM judge only for open-ended quality.

### D1. Tier 0: likelihood probes (no sampling; works on base checkpoints)

Purpose: a zero-variance signal that can be tracked **during pretraining** and in 124M A/B runs (the same use as Max's held-out loss in `docs/ab_2026-09-21`), long before SFT exists. Each item is a ChatML-formatted conversation (golden assistant turns written from templates) ending at an answer slot. Score = how often log p(correct continuation) > log p(each distractor). Candidates are single words or short spans. Candidates come from values that **also appear in the conversation**, so a model that just copies the most recent or most frequent name loses.

- LP-Recall (400): a fact planted in turn 1; 0, 2, 4 or 8 distractor turns; a question. Distractor = another value mentioned in the conversation.
- LP-Update (150): a fact revised mid-conversation; latest vs stale value.
- LP-Coref (150): "my sister Ana and my brother Leo ... she ..."; referent vs the other entity.
- LP-Persona (150, DNLI-style): assistant persona facts in the system turn; entailed vs contradicting reply.
- LP-Continuation (150, BabyLM-dialogue-style): the true next assistant turn vs a fluent turn taken from a different conversation.

Validation: shortcut rows (always pick the most recent candidate, always the first, always the most frequent) must sit at or below chance on every probe, or the probe gets redesigned. Tokenizer caveat: compare summed log-probs over whole candidates, and keep candidate token lengths matched within one model's tokenizer where possible. Report per model, not as a cross-tokenizer leaderboard.

### D2. Tier 1: generative programmatic families (the core)

About 1,450 scripted conversations of 3 to 10 turns, English only. User turns are fixed. Each family has a **dev** split (for iteration) and a **sealed** split built from different templates, slot pools and constraint types (family-level holdout, not just item-level). The sealed split runs only at milestones.

| family | n | what it tests (source of the idea) | example | grader | mutations that must FAIL |
|---|---|---|---|---|---|
| F1 Recall at distance | 300 | inference memory (MultiChallenge), distance effect (MT-Eval) | T1 "my dog is Pearl, she's 12" ... 1/3/6/9 turns later: "how old did I say my dog is?" | normalized slot match; the reply must contain the right value and none of the other in-conversation candidates | empty; wrong in-pool value; a "shotgun" reply listing every entity; echo of the question |
| F2 Update and abstain | 150 | knowledge updates and abstention (LongMemEval) | "I live in Boston" ... "actually I moved to Denver" ... "where do I live?"; and questions about facts never given | latest value present and stale value absent; for abstain items: no candidate value plus a hedge pattern ("didn't say", "don't know", "haven't told") | stale value; a confident invented value; empty |
| F3 Reference and follow-up | 250 | anaphora and follow-ups (MT-Bench-101 AR, MT-Eval follow-up) | user lists 3 tasks; "what was the second one?"; "move the last one to the top and show the list" | exact item or ordered-list match | wrong index; the original order unchanged; empty |
| F4 Instruction persistence | 250 | instruction retention (MultiChallenge), accumulating constraints (Multi-IF) | T1 "for the rest of this chat, answer in under 25 words and all lowercase", then 3-6 normal questions | IFEval verifier on **every** later turn, AND a non-degenerate, on-topic gate (at least 3 content words that overlap the question's topic lexicon) | empty (would pass "under 25 words" without the gate); an off-topic compliant reply; compliance only on the first turn |
| F5 Instruction override | 150 | versioned editing (MultiChallenge), contextual inertia (https://arxiv.org/abs/2603.04783) | rule A set, later replaced by rule B | B satisfied and A's distinguishing feature absent | still following A; following both when they conflict |
| F6 Sharded vs full | 150 tasks x 3 conditions | the Lost-in-Conversation protocol on tasks within 150M's single-turn reach | "write a 2-sentence birthday note for Maya, who is turning 16 and loves hiking, signed Max": given full, as 3-4 shards, or concatenated | slot checklist (all names and numbers present, sentence count) | missing any slot; the empty reply; a reply to the last shard only |
| F7 Persona and self-consistency | 100 | self-coherence (MultiChallenge), persona consistency (PersonaChat/DNLI) | system gives the assistant a name, a favorite color and a hometown; probes spread across turns | exact slot matches; a contradiction between turns counts as failure | wrong persona value; a user-provided value substituted for the persona value |
| F8 Social basics | 100 | the SmolTalk "Hi" / "who are you" failures | greeting, thanks, identity, goodbye, a short small-talk arc | length bounds; identity string matches the system prompt; no new-topic injection (reply-to-prompt lexical overlap threshold); no role leak | a long off-topic reply; the wrong identity; empty |

Diagnostics logged on every assistant turn of every family (these count as failures of that turn):
- empty or whitespace-only reply;
- loop: consecutive-repetition coverage ≥ 0.25 (the Daimax metric);
- cross-turn copy: at least 50% of the reply is the longest common substring with the model's own previous reply;
- role leak: emits a user turn, a "User:" line, or raw template tokens;
- runaway: hits `max_new_tokens`.

Run conditions (all families):
- **Golden history** (prior assistant turns are template-written references) vs **own history** (the model's own previous replies). The gap between the two is the error-propagation penalty, which MT-Eval and the 2026 history-neutralization work identify as a primary driver. It also tells you whether a failure is "cannot use context" or "derails itself".
- **Greedy** is primary. **T=0.7 x 3 samples** gives aptitude and unreliability in the Lost-in-Conversation sense (90th vs 10th percentile), and it is where loops show up.
- Repetition penalty **off** for the primary numbers. Report penalty 1.2 separately, since MaxGPT relied on it and a penalty can hide a training defect.
- Optional **scaffolded** condition: a recap or "state card" re-injected each turn (LiC's recap gave +16 to +17.5 points at GPT-4o scale). Always reported separately and never counted as the model.

Composite: **MTC** = unweighted mean of the F1 to F7 pass rates on own history, greedy, sealed split. F8 and the diagnostics are reported next to it. CIs come from bootstrapping over conversations. Model comparisons use **paired** per-item differences (the variance reduction already recommended in `research_2026-09-22.md`).

### D3. Tier 2: public anchors (so outsiders can place the model)

- IFEval (official verifiers, report all 4 metrics and name them).
- IFBench (official; include its multi-turn rewrite mode).
- Multi-IF, English subset only (official scorer; report turn 1/2/3 separately as well as the average).
- CoQA through lm-evaluation-harness (generative F1).
- Optional: MT-Bench-101 CM/AR/TS/SC/SA subsets with the Tier 3 judge, labeled as local-judge numbers.

Expect MaxGPT-Nano to lose on knowledge (MMLU-style) and possibly on IFEval to IFEval-trained baselines. Report those anyway. A claim that hides its losses is not credible.

### D4. Tier 3: judged open-ended quality (used only where programmatic grading cannot work)

- 200 held-out realistic 4-6 turn scripts: real first turns sampled from OASST or WildChat (English, knowledge-light, filtered), with templated follow-ups ("can you make it shorter", "why?", "what about for a kid?").
- **Pairwise** against Qwen2.5-0.5B-Instruct, judged in both orders, ties allowed. The rubric scores coherence with earlier turns, relevance, naturalness and factual sanity separately.
- Judge options at $0: a local 8B-14B model on the M5 Mac through MLX, 4-bit (candidates: Qwen3-14B, or the evaluator-tuned Prometheus 2 7B, https://arxiv.org/abs/2405.01535, or Atla Selene 1 Mini 8B, https://huggingface.co/blog/AtlaAI/selene-1-mini). Or Claude through the Max subscription for a small calibration set. Load one model at a time (the 24GB rule).
- The judge has to earn its place: (1) at least 75% agreement with Max's own blind labels on 100 non-tie pairs; (2) it prefers the reference over empty, looping, off-topic and other-conversation replies at least 95% of the time (mutation); (3) its verdict agrees with itself after a position swap at least 80% of the time. A judge that fails any of these is not used.
- The final claim adds a blind human test: at least 3 raters, at least 150 conversations, order randomized, the model's identity hidden.

### D5. Baselines (all re-run in the one harness)

Required:
- SmolLM2-135M-Instruct
- SmolLM2-360M-Instruct
- Gemma 3 270M-it (the `google/` repo is gated; `unsloth/gemma-3-270m-it` is an ungated copy with the same config)
- Qwen2.5-0.5B-Instruct
- Qwen3-0.6B with `enable_thinking=False` (thinking on reported separately)

Strongly recommended:
- **Falcon-H1-Tiny-90M-Instruct** (the most direct prior claim at this size; hybrid Mamba, so check that the transformers torch fallback runs on MPS, which is unverified)
- **LFM2.5-350M** (the strongest sub-400M multi-turn IF claim; LFM2-350M weights are already in `~/.cache/huggingface`)
- Qwen3.5-0.8B (the 2026 ceiling reference)
- MaxGPT-3 235M SFT (internal)

Shortcut rows (not models, but mandatory):
- empty
- echo of the last user turn
- a constant generic reply ("Sure! How can I help?")
- shotgun (every entity from the context)
- a regex "oracle" that copies the most recent matching slot

The oracle row shows how much of a family is solvable by pure retrieval. If it scores 80% on F1, then F1 measures copying, not conversation, and has to be made harder (for example with distractor values of the same type).

Harness rules: each model's own `apply_chat_template`; Gemma's system text merged into the first user turn (Gemma has no system role); `max_new_tokens` 256 per turn; the model's own EOS and end-of-turn stops; fixed seeds; raw transcripts saved. Serialize model loads on the Mac (the 24GB swap incident).

### D6. Sample size and noise

For a pass rate near 50%, the 95% CI half-width is ±8.0 points at n=150, ±5.7 at 300, ±3.1 at 1000 and ±2.5 at 1500. For a **paired** comparison of two models on the same 1,500 conversations, with 25% of items discordant, the half-width is about ±2.5 points (±2.8 at 30% discordance). So a composite of ~1,450 conversations can support a -3 point non-inferiority margin, and individual families (150-300 items) can only resolve differences of about 6 to 8 points. Report the families with their CIs and do not rank models on family-level gaps smaller than that.

### D7. Grader validation protocol (before any MaxGPT-Nano number is trusted)

1. Run every baseline on the dev split. Max blind-labels 200 turn-level outputs across families and models, and the grader has to agree on at least 95% of them. Every disagreement becomes either a grader fix or a documented ambiguity.
2. Mutation tests are unit tests: each family's must-fail list (above) runs in CI, and each test has been **watched failing** by breaking the grader on purpose (Max's standing rule).
3. Absence assertions (for example "the stale value is absent") need fixtures that actually try to produce the absent thing, so every F2 item's stale value is also offered in a shotgun mutation.
4. Decontamination: MaxGPT-Nano's multi-turn SFT data will probably come from generators much like these. The sealed families' templates, slot pools and constraint types must never feed a data generator. Check 13-gram overlap between the SFT data and the battery, and keep the constraint verifiers used in F4/F5 sealed separately from any IF training data.

### D8. What counts as success (pre-registered)

- **Level 1, "a real sub-150M chat model"**: total params ≤ 150M (also report non-embedding params and training tokens). On the sealed split, MTC non-inferior to Qwen2.5-0.5B-Instruct: the lower bound of the paired 95% CI of (Nano minus Qwen) must be at least -3 points. Loop and role-leak rate no more than Qwen2.5-0.5B's plus 1 point. Blind human pairwise win-or-tie at least 50% vs Qwen2.5-0.5B-Instruct (n ≥ 150, ≥ 3 raters). Also strictly better than Falcon-H1-Tiny-90M-Instruct, SmolLM2-135M-Instruct and Gemma 3 270M-it on MTC, with CIs excluding zero.
- **Level 2, "broke the wall"**: the same test against Qwen3-0.6B (non-thinking) and LFM2.5-350M.
- Always reported alongside: Tier 2 anchors, a knowledge benchmark where Nano will lose, the golden-vs-own-history gap, and the scaffolded condition labeled as scaffolded.

Why Qwen2.5-0.5B-Instruct is the bar: it is the model Max's premise names, and it is the model Daimax shipped as its clean small model after measurement (0 loops in 33 runs). Note that on published IFEval it is *weaker* than several sub-300M models, so "matches Qwen2.5-0.5B on IFEval" would prove nothing. The claim has to be about multi-turn behavior.

---

## What this implies for a ~150M chat model

- The measurement gap is the first thing to fix. Nothing public measures 5-10 turn coherence at this size. Until MTB-150 exists, every "it chats" or "it doesn't chat" judgment about a 100-250M model, including the ones from Max's own MaxGPT runs, is anecdotal.
- Build Tier 0 first. The existing 124M A/B checkpoints and MaxGPT-3 can be scored on likelihood probes immediately. That gives the first data point on whether multi-turn competence at this scale is a pretraining property (it moves with loss and data) or a post-training property (flat until SFT).
- Expect most failures to be knowledge and degeneration, not memory. Attention does in-context recall very cheaply (a 70M attention model beat a 1.4B gated-convolution model on associative recall; recall explained 82% of the attention vs gated-convolution quality gap, https://arxiv.org/abs/2312.04927), while stored knowledge is capacity-bound at about 2 bits per parameter (https://arxiv.org/abs/2404.05405, synthetic facts). The battery separates the two on purpose. F1-F7 use user-supplied facts, so a 150M model is not penalized for not knowing the capital of Mongolia.
- The competition is real and recent. Falcon-H1-Tiny-90M (Jan 2026) and LFM2.5-350M (Mar 2026) both show that aggressive data recipes (SFT data mixed into pretraining, tens of thousands of tokens per parameter, DPO or RL) move chat and IF metrics a lot at tiny scale. A MaxGPT-Nano claim has to beat these in Max's own harness, not just SmolLM2.
- Budget reality (cross-lane flag): the sub-600M models with the best published chat numbers saw 8.8k to 79k tokens per parameter (MobileLLM-350M, at ~2.9k, reports a lower MT-Bench in a different harness). Max's 100B-token build at 150M is about 670 tokens per parameter, 13x below Falcon-H1-Tiny-90M and 20 to 120x below the rest. If Nano underperforms, the battery has to be able to tell "too few tokens" from "a real wall". That is another reason for Tier 0 curves across checkpoints.

## Evidence about the "wall" (why chat quality degrades below ~400M)

What the evidence supports:
1. **Judged open-ended quality falls steeply below ~1B, driven by knowledge and reasoning.** Arena-Hard: Qwen3-0.6B 6.5 vs Qwen3-1.7B 36.9 (non-thinking, https://arxiv.org/abs/2505.09388). MultiChallenge: Qwen3.5-0.8B 18.9 vs Qwen3.5-2B 33.7 (thinking, Qwen3.5-0.8B card). MT-Bench is 3/8 math/coding/reasoning and 2/8 knowledge. MaxGPT-2's "confident nonsense" is the same effect at 110M.
2. **Degeneration is a small-model-visible failure.** SmolLM2-360M looped in 4/33 runs vs 0/33 for Qwen2.5-0.5B (Max's Daimax, T=0.7), and MaxGPT-2 had attractor loops. One loop ruins a conversation, and it is invisible to single-turn, short-output benchmarks.
3. **Social pragmatics needed targeted data at 135M-360M**: models answered "Hi" off-topic and failed "Who are you" until everyday-conversation data was added (SmolTalk card).
4. **Multi-turn degradation is universal, not small-specific.** Frontier and 8B+ models lose 39% on average from full to sharded (LiC); o1-preview drops 17% from turn 1 to turn 3 on Multi-IF; Llama 3.1 8B goes from 0.688 to 0.542. Small models inherit the same mechanisms (distance to relevant content, error propagation, attention closing on goal tokens) with less headroom. This is inferred from ≥7B evidence only.
5. **Multi-turn skill responds to targeted post-training data** (TurnWise, 10k conversations, +6.2 at 7B), and DPO moves judged chat scores a lot at 90M (Falcon-H1-Tiny: +1.25 MT-Bench, +16 IFEval).

What the evidence does not support:
- A parameter-count wall at 400-600M on chat or IF metrics. Sub-400M models match or beat 0.5-0.8B models on IFEval, Multi-IF and 2-turn judged chat when trained on roughly 9k+ tokens per parameter (A1).
- That context memory is the binding constraint. Associative recall is cheap for attention (Zoology), and in-context recall at 150M is unmeasured, which is exactly what F1/LP-Recall will settle.

Honest summary: the "wall" looks like knowledge plus degeneration plus training budget plus measurement noise (harness gaps of 10-24 points), not a sudden loss of conversational mechanics. That is a hypothesis for the battery to test, not a result.

## Levers (from the evaluation side)

| lever | limit it attacks | evidence | expected effect | cost |
|---|---|---|---|---|
| Build MTB-150 with sealed template families and mutation-tested graders | "the wall" is unmeasured, and harness noise is larger than model gaps | strong (A3, and the absence of any sub-300M multi-turn data) | turns vibes into a CI-bounded claim; the precondition for every other lever | about 1 to 2 weeks of Max's time; runs on the Mac |
| Tier 0 likelihood probes tracked across pretraining checkpoints | cannot see multi-turn competence until after SFT | moderate (BabyLM minimal pairs discriminate dialogue ability, https://arxiv.org/abs/2510.20358; entity tracking in the BabyLM suite) | early signal for data and architecture A/Bs at 124M/150M; separates the pretraining vs post-training question | minutes per checkpoint |
| Golden vs own history on every family | cannot tell "can't use context" from "derails itself" | moderate (MT-Eval error propagation; https://arxiv.org/abs/2609.05882) | points the fix at data/decoding (own-history gap) vs capacity/attention (golden fails) | 2x generation |
| Per-turn loop, copy and role-leak diagnostics at T=0.7 x3 | degeneration, the most visible tiny-model failure | strong for existence (Daimax 4/33 vs 0/33; MaxGPT-2), unmeasured for prevalence at 150M | catches a failure every single-turn benchmark misses | cheap; reuse degeneracy.ts |
| Sharded-vs-full ratio (F6) | multi-turn penalty mixed up with capability | strong at ≥8B (LiC -39%), unmeasured at 150M | a multi-turn penalty metric that does not depend on knowledge | cheap |
| Attention-to-goal ratio and residual probes as diagnostics | not knowing why a 150M model forgets | moderate at 7B-class (https://arxiv.org/abs/2605.12922: AUC up to 0.99; forced closure cut recall to 11%) | mechanistic "why" measurements, ready-made for the writeup | a few hours of code |
| Recap / state-card scaffold, reported as its own condition | cross-turn forgetting | moderate at GPT-4o scale (+16 to +17.5 points) | possibly large at 150M, since it turns recall into copying from nearby; unmeasured | free at inference |
| SFT-in-pretraining and DPO, judged with MTB-150 rather than IFEval | a thin post-training signal at tiny scale | moderate (Falcon-H1-Tiny 90M: IFEval 66.08 vs 53.47; DPO +1.25 MT-Bench), vendor-only | large movement on IF/chat metrics; needs an uncontaminated eval to verify it is real | training lane |
| Multi-turn training data (TurnWise-style stacked turns; everyday conversations) | missing multi-turn behavior in the data | moderate at 7B (+6.2 TurnWise-Self), weak for sub-400M (SmolTalk anecdote) | better F1-F5 and F8 | data lane |
| A local validated judge instead of an API judge | $0 budget; frontier judges are gameable | weak to moderate (Prometheus 2 / Selene Mini claims are vendor evaluations on frontier-model outputs) | usable open-ended quality signal, if it passes D4's checks | hours on the Mac per model |

## Open questions

1. Does any sub-200M model (Falcon-H1-Tiny-90M, SmolLM2-135M, Gemma 3 270M) hold 6-10 turn coherence on programmatic probes? Unmeasured anywhere. Running D5's baselines through MTB-150 is the first experiment.
2. How much of Falcon-H1-Tiny-90M's 4.33 "MT Bench (avg)" and 66.08 IFEval survives an independent harness? Its SFT pretraining repeated Tulu 3 (which contains IFEval-taxonomy data) ≳100 times.
3. Does recall at 150M decay with turn distance, with token distance, or with the number of competing same-type values? F1's distance sweep plus same-type distractors will separate these.
4. Do Tier 0 likelihood probes on base checkpoints predict post-SFT generative MTC? This needs a rank correlation across 5-10 checkpoints before Tier 0 can steer pretraining decisions.
5. Are 8B-14B local judges reliable on tiny-model outputs? Judge validation work used frontier-model outputs, and the low end of the quality range may compress. D4's checks answer this for this project.
6. What is the own-history vs golden-history gap for the baselines? If it is large for Qwen2.5-0.5B too, error propagation (a decoding and data problem) is the target, not capacity.
7. Is Max's own "potential at 100-250M" observation a token-budget effect? MaxGPT-3 saw about 85 token-passes per parameter, against 8.8k to 79k for the published tiny models. A Tier 0 curve over tokens seen would show whether multi-turn probes are still climbing.
