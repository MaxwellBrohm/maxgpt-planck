# Lane: firsthand probe of today's small chat models (where multi-turn coherence breaks)

Research date: 2026-09-23. Everything under "Detailed findings" was measured by me on this Mac (M5, 24 GB, PyTorch 2.13 on MPS, fp32, transformers 5.15.1). Every number there can be regenerated from the transcripts in `research/probe/` with `python analyze.py && python report_tables.py`. External numbers carry a URL and the scale they were measured at. "My count" means computed from the model's `config.json` or loaded weights.

---

## Bottom line

1. I ran 8 small chat models (90M to 600M) through 29 scripted multi-turn conversations (2 to 12 turns) plus 24 single-turn controls. The graders were mutation-tested: each fails an empty answer and a plausible wrong answer, and 14 deliberately broken graders were all caught by the self-test. Vendors do publish multi-turn instruction-following scores for this size class (Multi-IF 44.92 for LFM2.5-350M, [card](https://huggingface.co/LiquidAI/LFM2.5-350M); a 2-turn MT-Bench of 4.33 for Falcon-H1-Tiny-90M, [TII](https://tiiuae-tiny-h1-blogpost.hf.space/)), but I found no published probe of cross-turn memory, corrections or speaker binding below 1B (the context lane reports the same gap), so this is new evidence, at small n.
2. Max's premise is half right. On this battery the 350M-class models (LFM2-350M macro 0.65, LFM2.5-350M 0.62, SmolLM2-360M 0.57) match Qwen2.5-0.5B (0.60), and Qwen3-0.6B is best (0.71). So the line is not at 400-500M. Below 300M quality drops, but not in size order: Falcon-H1-Tiny-90M (74M non-embedding parameters) scores 0.49, above Gemma-3-270M (0.36) and SmolLM2-135M (0.15). At this size the recipe matters more than the parameter count.
3. Most failures are not memory failures. In 40 of 63 failed recall or binding checks (8 models, greedy), greedy decoding from a forced answer prefix ("Your name is") emits the correct fact verbatim. The model still has the fact and picks the wrong kind of reply.
4. The largest failure class is deflection ("I don't have access to personal information", "I am an AI and do not have a physical location"): 36 of 95 failed user-fact checks under greedy decoding. Then wrong or ignored answers (33), copying its own earlier reply (13), reverting to a corrected value (7) and answering as the user ("I'm Marcus.", 6). Deflection is not a size effect: Qwen2.5-0.5B deflects on 45% of its final user-fact turns, SmolLM2-135M on 24%, LFM2-350M on 7%, and the smallest model, Falcon-H1-Tiny-90M, never.
5. The cleanest single result: the same SmolLM2-135M-Instruct weights recall a user fact 0.70 of the time when the identical history is written as a plain "User:/Assistant:" transcript, and 0.15 of the time through their own chat template (20 trials each over 1 to 12 distractor turns). The untuned base model gets 0.60. At 135M, the chat post-training is the first-order problem, not capacity.
6. Memory itself is not the wall at 100M. Given an identical 12-turn, 1,726-token history, Falcon-H1-Tiny-90M retrieves every fact (greedy-from-prefix 1.00 at every distance) and answers 0.80 (short replies) and 0.75 (long replies) of them correctly with zero deflections. Only SmolLM2-360M does better, and only with short replies. SmolLM2-135M does lose retrieval past about 500 to 1,600 tokens, so some small models have the limit, but it is not forced by size. What stays weak in both ~100M models: single-turn knowledge and skill (controls 0.46 and 0.64), speaker binding (both prefer the sister's name in a forced choice), and instruction persistence; SmolLM2-135M also loops (22% of replies repeat a 4-gram 3+ times).
7. One ability fails at every size up to 0.6B: updating state after a correction. 20 of 24 multi-turn correction trials fail. For "meeting moved from 3 pm to 4 pm" all eight models put more probability on the stale value after one distractor turn, while seven of eight prefer the corrected value when the same sentences arrive as one message.
8. For a 150M model this points the novelty budget at post-training data and objectives: speaker perspective, using what the user said, corrections, and training on the model's own multi-turn outputs. Recall of a fact within 2k tokens is not the binding constraint at 150M if the model keeps real attention.

---

## What was run

**Models.** Each was run with its own chat template. Parameter counts are my count from the loaded weights (non-embedding = total minus vocab x hidden, all tied). Token counts are from the cited card or report.

| model | params (non-embedding) | pretraining tokens and post-training (per source) | notes |
|---|---|---|---|
| Falcon-H1-Tiny-90M-Instruct | 91.1M (74.3M) | 800B, with 25% pure SFT data in the final pretraining mixture, then SFT and "a lightweight DPO stage" ([TII blog](https://tiiuae-tiny-h1-blogpost.hf.space/)) | hybrid attention + Mamba2 in every layer; run on MPS with a capped allocator (the Mamba torch fallback is slow and memory-hungry) |
| SmolLM2-135M-Instruct | 134.5M (106.2M) | 2T; SFT on smol-smoltalk, then DPO on UltraFeedback ([card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), [dataset](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk)) | closest to Max's target shape: d=576, 30 layers, 49,152 vocab |
| Gemma-3-270M-it | 268.1M (100.3M) | 6T ([card mirror](https://huggingface.co/unsloth/gemma-3-270m-it)) | 15 of 18 layers use a 512-token sliding window ([config](https://huggingface.co/unsloth/gemma-3-270m-it/resolve/main/config.json)); ungated unsloth mirror |
| LFM2-350M | 354.5M (287.4M) | 10T ([card](https://huggingface.co/LiquidAI/LFM2-350M)) | 10 short-convolution + 6 attention layers |
| LFM2.5-350M | 354.5M (287.4M) | 28T plus "large-scale multi-stage reinforcement learning" ([card](https://huggingface.co/LiquidAI/LFM2.5-350M)) | created 2026-03-31 per the Hub API |
| SmolLM2-360M-Instruct | 361.8M (314.6M) | 4T; SFT + DPO ([card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)) | |
| Qwen2.5-0.5B-Instruct | 494.0M (357.9M) | 18T; SFT on over 1M samples plus multistage RL ([report](https://arxiv.org/abs/2412.15115)) | Max's "wall" reference |
| Qwen3-0.6B, thinking off | 596.0M (440.4M) | 36T; small models get strong-to-weak distillation, including on-policy logit distillation from Qwen3-32B or 235B teachers ([report](https://arxiv.org/html/2505.09388)) | `enable_thinking=False` |

Skipped: PleIAs Monad (56M), whose card says it "has no support yet for multi-turn" ([card](https://huggingface.co/PleIAs/Monad)); MobileLLM-R1-140M, which is gated and is an SFT model for "mathematical, programming (Python, C++), and scientific problems" ([card](https://huggingface.co/facebook/MobileLLM-R1-140M)).

**Battery** (`battery.py`): 29 multi-turn conversations, plus a single-turn control wherever one makes sense (24). The model's own replies are fed back as history, so each model lives with its own earlier mistakes.

- recall of a user-stated fact (name, number, pet, city, or three facts at once) after 2, 4, 5, 6 or 10 intervening distractor turns (8 conversations)
- follow-ups that need the previous turn: "And what about Italy?", "And in French?", "Where does she live?", "Why not?", "Make it shorter." (5)
- instruction persistence: one sentence, end with a question, all capitals, and a system-prompt pirate persona that must say "Arr" (4)
- corrections: Monday to Tuesday, 3 pm to 4 pm, blue to green, each with one distractor turn before the question (3)
- references to its own earlier answer: "second fruit on your list", "which name did you put last", "add 10 to your answer" (3)
- role integrity: the user's job vs the assistant's, the user's name vs the sister's (2), plus flags on every reply (template leaks, invented user turns, runaway length)
- topic return after a digression (2), and free-form chats for loop metrics (2)

**Grading.** A user-fact answer passes only if the gold appears outside a greeting ("Hello Marcus!" does not count), the reply does not deflect, and it does not state the user's fact as the model's own ("I'm Marcus." fails). Format checks (one sentence, capitals, question mark, "Arr") also reject loops and verbatim copies of an earlier reply. The gold answer never appears in the question that asks for it, which the self-test asserts. `python battery.py` runs 256 fixture assertions (each grader passes a correct answer and fails an empty and a plausible wrong answer; guarded graders also fail a deflecting answer that echoes the gold) plus about 30 unit assertions. `python mutation_test_graders.py` breaks 14 things on purpose (all graders true, all false, each guard removed, the gold leaked into its question, the sentence counter and list parser broken) and confirms each one turns the self-test red. `analyze.py` regrades every saved transcript with the current graders, so all models are scored by the same code. While reading transcripts I found and fixed ten grader holes (a deflection that greets the user by name, greeting echoes, curly apostrophes and identity disclaimers hiding deflections, a stale value mentioned only as history, role capture, first-person job claims, "you would assist the chef" passing as the user's job, loops and copies passing format checks, duplicate list items, and "naming your sister Oscar"); each fix has a unit test in the self-test.

**Forced-prefix probe.** For recall, correction and binding tests I also score the context with the assistant reply pre-started ("Your name is") and record the log-probability and per-token rank of the gold (" Priya") and of a foil that never appears in context (" Maria"). "Greedy-from-prefix emits gold" means every gold token is the argmax. This separates "the fact is gone" from "the fact is there but the model said something else". Caveat: Qwen3-0.6B opens with bold markdown ("**Priya**"), so its greedy-from-prefix rate understates its retrieval; its gold-vs-foil margin does not have that problem.

**Extra experiments.**
- E2, fixed history: every model gets the identical history (the fact, a canned acknowledgement that does not repeat it, N canned distractor exchanges, the question) for N in {0, 1, 2, 4, 8, 12}, with short (about 15 words) or long (about 100 words, including a repeated generic filler paragraph) canned replies, 4 facts per cell. This removes the confound that each model's own verbose replies set its context length.
- E1, base vs instruct vs format: the same fixed histories rendered as a plain "User:/Assistant:" transcript, run on SmolLM2-135M base and on SmolLM2-135M-Instruct.
- E3, a one-line memory system prompt ("Everything the user said earlier in this conversation is visible to you; when the user asks about something they told you, answer from the conversation") on the fixed short histories, for five models: the two ~100M-non-embedding models that deflect (SmolLM2-135M, Gemma-3-270M) and three larger ones (LFM2.5-350M, SmolLM2-360M, Qwen2.5-0.5B).
- Sampled pass: every conversation again with each card's recommended sampling, 2 seeds (Falcon 1 seed; Falcon's card gives no sampling advice, so I used T=0.7, top-p 0.9). Greedy is the headline because it is deterministic.

**Scale of the evidence.** 53 conversations per model per seed. Most cells are small: 2 to 6 trials per recall distance per model in the main battery, 20 per pooled cell in E2. Differences under about 0.2 between two models are noise. The patterns below are worth believing because they repeat across models and across the separate experiments, not because any single cell is precise.

---

## Detailed findings

### F1. Premise check: which of these models actually hold a multi-turn chat

Greedy decoding, strict grading, fraction of checks passed. "macro" is the mean of the seven multi-turn ability columns. "single-turn controls" is the pass rate on the 24 controls (the same facts and tasks without the multi-turn setup).

| model | params | recall | followup | instruction | correction | own_answer | role | topic_return | macro | single-turn controls |
|---|---|---|---|---|---|---|---|---|---|---|
| Falcon-H1-Tiny-90M | 91.1M | 0.50 | 0.57 | 0.29 | 0.33 | 0.75 | 0.33 | 0.67 | **0.49** | 0.64 |
| SmolLM2-135M | 134.5M | 0.10 | 0.29 | 0.36 | 0.00 | 0.33 | 0.00 | 0.00 | **0.15** | 0.46 |
| Gemma-3-270M | 268.1M | 0.00 | 0.86 | 0.50 | 0.00 | 0.50 | 0.00 | 0.67 | **0.36** | 0.75 |
| LFM2-350M | 354.5M | 0.60 | 1.00 | 0.79 | 0.33 | 0.50 | 0.33 | 1.00 | **0.65** | 0.89 |
| LFM2.5-350M | 354.5M | 0.60 | 1.00 | 1.00 | 0.00 | 0.75 | 0.00 | 1.00 | **0.62** | 0.82 |
| SmolLM2-360M | 361.8M | 0.70 | 0.86 | 0.79 | 0.00 | 0.67 | 0.33 | 0.67 | **0.57** | 0.54 |
| Qwen2.5-0.5B | 494.0M | 0.10 | 0.86 | 0.57 | 0.00 | 1.00 | 0.67 | 1.00 | **0.60** | 0.82 |
| Qwen3-0.6B | 596.0M | 0.60 | 1.00 | 0.71 | 0.67 | 1.00 | 0.00 | 1.00 | **0.71** | 0.93 |

- The 350M class is not behind the 0.5B "wall" model on this battery. LFM2-350M (0.65) and LFM2.5-350M (0.62) are at or above Qwen2.5-0.5B (0.60), and SmolLM2-360M (0.57) is within noise of it. Qwen3-0.6B (0.71) is best. This agrees with the vendor number that LFM2.5-350M scores Multi-IF 44.92 against 41.68 for Qwen3.5-0.8B Instruct in Liquid's harness ([card](https://huggingface.co/LiquidAI/LFM2.5-350M); measured at 350M and 0.8B).
- The drop is below about 300M total, and it tracks non-embedding size plus training more than total size. Gemma-3-270M has about 100M non-embedding parameters (my count: 168M of its 268M is the 262k-vocab embedding) and scores 0.36; SmolLM2-135M (106M non-embedding) scores 0.15. Falcon-H1-Tiny-90M, with the fewest non-embedding parameters in the probe (74M), scores 0.49: size is not the whole story at this scale. Falcon's recipe differs on every axis at once (hybrid attention + Mamba2 in all 24 layers, 800B tokens with 25% SFT data mixed into pretraining, SFT then DPO), so the probe cannot say which part helps.
- At about 100M non-embedding parameters, the training recipe moves quality a lot: Gemma-3-270M (6T tokens) passes 0.75 of the controls and 0.86 of follow-ups, SmolLM2-135M (2T tokens) 0.46 and 0.29, and neither recalls user facts across turns (recall 0.00 and 0.10). Falcon-H1-Tiny-90M (800B tokens) recalls 0.50 in the main battery and 0.75 to 0.80 in the fixed-history sweep (F2).
- No model leaked template tokens or invented a user turn under its own template with greedy decoding (0 of 149 assistant turns each). Role failures show up in content (speaking as the user), not in format.

### F2. Recall vs distance

Main battery, greedy plus sampled pooled. Each cell: strict free-answer pass rate (n), then after the slash the rate at which the forced-prefix probe prefers the gold over a foil. d=0 is the single-turn control; d=5 is the three-fact summary, which has no forced probe.

| model | d=0 | d=2 | d=4 | d=5 | d=6 | d=10 |
|---|---|---|---|---|---|---|
| Falcon-H1-Tiny-90M | 0.60 (n=20) / 1.00 | 0.50 (n=4) / 1.00 | 0.50 (n=4) / 1.00 | 1.00 (n=6) | 0.00 (n=4) / 1.00 | 0.00 (n=2) / 1.00 |
| SmolLM2-135M | 0.50 (n=30) / 1.00 | 0.33 (n=6) / 1.00 | 0.50 (n=6) / 0.75 | 0.11 (n=9) | 0.33 (n=6) / 1.00 | 0.00 (n=3) / 1.00 |
| Gemma-3-270M | 0.47 (n=30) / 1.00 | 0.17 (n=6) / 1.00 | 0.00 (n=6) / 0.62 | 0.00 (n=9) | 0.00 (n=6) / 0.50 | 0.00 (n=3) / 0.00 |
| LFM2-350M | 0.70 (n=30) / 0.86 | 1.00 (n=6) / 1.00 | 0.33 (n=6) / 1.00 | 0.89 (n=9) | 0.50 (n=6) / 1.00 | 0.00 (n=3) / 1.00 |
| LFM2.5-350M | 0.63 (n=30) / 1.00 | 0.17 (n=6) / 1.00 | 0.67 (n=6) / 1.00 | 1.00 (n=9) | 0.50 (n=6) / 1.00 | 0.67 (n=3) / 1.00 |
| SmolLM2-360M | 0.47 (n=30) / 1.00 | 0.33 (n=6) / 1.00 | 0.50 (n=6) / 1.00 | 0.89 (n=9) | 1.00 (n=6) / 1.00 | 0.67 (n=3) / 1.00 |
| Qwen2.5-0.5B | 0.57 (n=30) / 1.00 | 0.50 (n=6) / 1.00 | 0.17 (n=6) / 1.00 | 0.33 (n=9) | 0.00 (n=6) / 1.00 | 0.00 (n=3) / 1.00 |
| Qwen3-0.6B | 0.80 (n=30) / 1.00 | 0.17 (n=6) / 1.00 | 0.50 (n=6) / 1.00 | 0.89 (n=9) | 0.50 (n=6) / 1.00 | 0.33 (n=3) / 1.00 |

Fixed history (E2: identical canned history for every model, 4 facts per cell). Each cell: strict free recall / greedy-from-prefix emits the gold.

| model | replies | n=0 | n=1 | n=2 | n=4 | n=8 | n=12 |
|---|---|---|---|---|---|---|---|
| HuggingFaceTB/SmolLM2-135M-Instruct | short (n=12: 497 tok) | 0.25 / 1.00 | 0.25 / 1.00 | 0.00 / 1.00 | 0.25 / 0.75 | 0.25 / 0.25 | 0.00 / 0.00 |
| HuggingFaceTB/SmolLM2-135M-Instruct | long (n=12: 1697 tok) | 0.25 / 1.00 | 0.25 / 1.00 | 0.00 / 0.75 | 0.50 / 0.50 | 0.00 / 0.25 | 0.00 / 0.00 |
| HuggingFaceTB/SmolLM2-135M-Instruct  [memory system prompt] | short (n=12: 519 tok) | 0.25 / 1.00 | 0.25 / 1.00 | 0.00 / 1.00 | 0.25 / 0.75 | 0.50 / 0.25 | 0.00 / 0.25 |
| HuggingFaceTB/SmolLM2-135M-Instruct  [plain transcript] | short (n=12: 435 tok) | 0.25 / 1.00 | 0.75 / 0.75 | 0.75 / 1.00 | 0.75 / 1.00 | 0.50 / 0.75 | 0.75 / 0.75 |
| HuggingFaceTB/SmolLM2-135M-Instruct  [plain transcript] | long (n=12: 1635 tok) | 0.25 / 1.00 | 0.75 / 0.75 | 0.50 / 0.75 | 0.50 / 0.75 | 0.50 / 0.50 | 0.25 / 0.50 |
| HuggingFaceTB/SmolLM2-135M  [plain transcript] | short (n=12: 435 tok) | 0.25 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.50 / 1.00 | 0.75 / 0.75 | 0.25 / 0.75 |
| HuggingFaceTB/SmolLM2-135M  [plain transcript] | long (n=12: 1635 tok) | 0.25 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.00 / 0.75 | 0.25 / 0.75 | 0.25 / 0.25 |
| HuggingFaceTB/SmolLM2-360M-Instruct | short (n=12: 497 tok) | 1.00 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| HuggingFaceTB/SmolLM2-360M-Instruct | long (n=12: 1697 tok) | 1.00 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.25 / 1.00 |
| HuggingFaceTB/SmolLM2-360M-Instruct  [memory system prompt] | short (n=12: 519 tok) | 0.75 / 1.00 | 1.00 / 1.00 | 0.75 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| LiquidAI/LFM2-350M | short (n=12: 472 tok) | 1.00 / 1.00 | 0.75 / 0.75 | 0.50 / 0.50 | 0.25 / 0.50 | 0.75 / 1.00 | 0.50 / 1.00 |
| LiquidAI/LFM2-350M | long (n=12: 1672 tok) | 1.00 / 1.00 | 1.00 / 1.00 | 0.50 / 1.00 | 0.50 / 1.00 | 0.50 / 0.75 | 0.50 / 0.75 |
| LiquidAI/LFM2.5-350M | short (n=12: 472 tok) | 0.75 / 0.75 | 0.50 / 0.75 | 0.50 / 0.75 | 0.75 / 0.75 | 0.75 / 0.75 | 1.00 / 1.00 |
| LiquidAI/LFM2.5-350M | long (n=12: 1672 tok) | 0.75 / 0.75 | 0.25 / 0.75 | 0.50 / 0.75 | 0.25 / 0.75 | 0.50 / 1.00 | 0.75 / 1.00 |
| LiquidAI/LFM2.5-350M  [memory system prompt] | short (n=12: 516 tok) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| Qwen/Qwen2.5-0.5B-Instruct | short (n=12: 484 tok) | 1.00 / 1.00 | 0.50 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 |
| Qwen/Qwen2.5-0.5B-Instruct | long (n=12: 1684 tok) | 1.00 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.50 / 1.00 | 0.75 / 1.00 |
| Qwen/Qwen2.5-0.5B-Instruct  [memory system prompt] | short (n=12: 506 tok) | 1.00 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 |
| Qwen/Qwen3-0.6B | short (n=12: 467 tok) | 0.25 / 0.25 | 0.75 / 0.25 | 0.75 / 0.25 | 0.50 / 0.25 | 0.75 / 0.25 | 0.75 / 0.25 |
| Qwen/Qwen3-0.6B | long (n=12: 1667 tok) | 0.25 / 0.25 | 0.25 / 0.25 | 0.25 / 0.25 | 0.25 / 0.00 | 0.00 / 0.00 | 0.25 / 0.00 |
| tiiuae/Falcon-H1-Tiny-90M-Instruct | short (n=12: 526 tok) | 0.75 / 1.00 | 0.75 / 1.00 | 1.00 / 1.00 | 0.75 / 1.00 | 0.50 / 1.00 | 1.00 / 1.00 |
| tiiuae/Falcon-H1-Tiny-90M-Instruct | long (n=12: 1726 tok) | 0.75 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 | 0.75 / 1.00 |
| unsloth/gemma-3-270m-it | short (n=12: 462 tok) | 0.00 / 0.75 | 0.25 / 0.75 | 0.25 / 0.75 | 0.00 / 0.75 | 0.25 / 0.75 | 0.00 / 0.50 |
| unsloth/gemma-3-270m-it | long (n=12: 1662 tok) | 0.00 / 0.75 | 0.00 / 0.75 | 0.00 / 0.75 | 0.00 / 0.50 | 0.25 / 0.50 | 0.25 / 0.50 |
| unsloth/gemma-3-270m-it  [memory system prompt] | short (n=12: 501 tok) | 0.25 / 1.00 | 0.00 / 0.75 | 0.00 / 0.75 | 0.00 / 0.75 | 0.00 / 0.50 | 0.00 / 0.50 |

What these show:
- Free-answer recall does not fall off smoothly with distance in any model. It is already noisy at one or two turns (often 0.25 to 0.75), because the failure is mostly in which reply the model chooses, not in how far back the fact is.
- Retrieval itself (greedy-from-prefix) stays at 1.00 through 12 turns and about 1,700 tokens for SmolLM2-360M and Qwen2.5-0.5B, and at 0.75 to 1.00 for the two LFM models. By 350M, holding a user fact across 12 turns is solved at the representation level.
- The smallest model does it too. Falcon-H1-Tiny-90M (74M non-embedding) keeps greedy-from-prefix at 1.00 at every distance up to 1,726 tokens and answers 0.50 to 1.00 of the questions correctly at each distance (0.80 and 0.75 pooled), with no deflections. So "a ~100M model cannot hold a user fact for 12 turns" is false.
- SmolLM2-135M cannot. Through its chat template SmolLM2-135M goes 1.00 (1-2 turns), 0.75 (4), 0.25 (8), 0.00 (12 turns, about 500 tokens). Rendered as a plain transcript, the same weights hold 0.75 at 12 short turns (435 tokens) and 0.50 at 12 long turns (1,635 tokens); the base model drops to 0.25 there. This is the one clear memory limit in the probe. Given Falcon, it is a limit of this model and recipe, not of the size class.
- Gemma-3-270M's 512-token sliding window (15 of 18 layers) did not stop retrieval: greedy-from-prefix is still 0.50 at 1,662 tokens and gold beats foil in 0.75 of cases. Its free-answer recall is 0.00 to 0.25 at every distance, including zero distractors. Its problem is reply policy, not the window.
- Qwen3-0.6B drops from 0.70 (short replies) to 0.20 (long replies) in pooled free recall, with deflection going from 0.25 to 0.70, while its gold-vs-foil margin stays positive in every trial. Longer, more "assistant-like" history made it deflect more ("I don't have access to your location, but I can tell you that **Tucson** is a city in the United States").

### F3. The fact is there, the model says something else

- Main battery, greedy, 8 models: of 63 failed recall or binding checks (multi-turn and single-turn), greedy decoding from the forced prefix emits the gold verbatim in 40 (63%) and prefers the gold over the foil in 58 (92%).
- E2 through each model's chat template: of 179 failed free answers, greedy-from-prefix emits the gold in 111 (62%) and prefers it over the foil in 162 (91%).
- The foil test is weak (the foil never appeared in context), so lean on the greedy number: in about two thirds of recall failures the model would have said the right thing had it started its reply the right way.

**Base vs instruct vs format (E1, SmolLM2-135M, pooled over 1 to 12 distractor turns, 20 trials per cell, short / long canned replies):**

| weights | history rendered as | strict recall | deflection | greedy-from-prefix emits gold |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | its chat template (adds the default system prompt) | 0.15 / 0.15 | 0.30 / 0.30 | 0.60 / 0.50 |
| SmolLM2-135M-Instruct | plain "User:/Assistant:" transcript | 0.70 / 0.50 | 0.00 / 0.05 | 0.85 / 0.65 |
| SmolLM2-135M base | plain transcript | 0.60 / 0.40 | 0.05 / 0.10 | 0.90 / 0.75 |

Only the rendering of an identical history changed, and 135M recall went from 0.15 to 0.70. The chat template (which also inserts SmolLM2's default system prompt, "You are a helpful AI assistant named SmolLM, trained by Hugging Face") switches on a deflecting assistant persona ("I'm sorry for any confusion, but as a helpful AI assistant, I don't have the ability to check your name"). In the plain transcript the same weights answer "You're correct, it's Priya." The instruct weights in plain format also edge out the base model, so SFT added usable conversational skill and then gated it behind a persona that will not use it. One model, 4 facts, so treat the size of the effect as rough; the direction held at every distance from 1 to 12 turns in the short-reply condition.

**E3: can one instruction switch the deflection off?** Same fixed short histories (1 to 12 distractor turns, 20 trials per cell), with and without one system line: "You are a helpful assistant chatting with a user. Everything the user said earlier in this conversation is visible to you; when the user asks about something they told you, answer from the conversation."

| model | strict recall without / with | deflection without / with | role capture without / with |
|---|---|---|---|
| SmolLM2-135M-Instruct | 0.15 / 0.20 | 0.30 / 0.15 | 0.00 / 0.05 |
| Gemma-3-270M | 0.15 / 0.00 | 0.45 / 0.45 | 0.00 / 0.00 |
| LFM2.5-350M | 0.70 / 1.00 | 0.15 / 0.00 | 0.05 / 0.00 |
| SmolLM2-360M-Instruct | 0.90 / 0.95 | 0.00 / 0.00 | 0.00 / 0.05 |
| Qwen2.5-0.5B-Instruct | 0.70 / 0.75 | 0.25 / 0.00 | 0.00 / 0.25 |

At 350M and up, one line of instruction removes deflection (LFM2.5-350M goes to 1.00; Qwen2.5-0.5B stops deflecting but starts answering as the user instead). In the two ~100M-non-embedding models it does nothing, although rendering the history as a plain transcript fixed the same 135M model (E1). The two small models also ignored the system-prompt pirate persona (F7). Below about 100M non-embedding parameters the chat behavior cannot be prompted into place; it has to be trained in.

### F4. Failure taxonomy

Every failed user-fact check in the greedy multi-turn conversations, labeled automatically in priority order: copy of an earlier reply (at least half its 8-grams appeared in an earlier reply), deflection, perspective error (answers as the user or about itself), stale value, gold only in a greeting, wrong or ignored. Rules are in `analyze.py` (`taxonomy`); I read every row labeled "wrong or ignored" or "perspective" while auditing.

| model | deflection | perspective_error | stale_value | copy_of_own_earlier_reply | gold_only_in_greeting_or_other | wrong_or_ignored | total |
|---|---|---|---|---|---|---|---|
| Falcon-H1-Tiny-90M | 0 | 0 | 2 | 0 | 0 | 8 | 10 |
| SmolLM2-135M | 5 | 2 | 1 | 7 | 0 | 4 | 19 |
| Gemma-3-270M | 7 | 0 | 1 | 1 | 0 | 8 | 17 |
| LFM2-350M | 1 | 0 | 2 | 0 | 0 | 5 | 8 |
| LFM2.5-350M | 6 | 1 | 1 | 0 | 0 | 2 | 10 |
| SmolLM2-360M | 2 | 0 | 0 | 3 | 0 | 4 | 9 |
| Qwen2.5-0.5B | 12 | 0 | 0 | 1 | 0 | 1 | 14 |
| Qwen3-0.6B | 3 | 3 | 0 | 1 | 0 | 1 | 8 |
| all | 36 | 6 | 7 | 13 | 0 | 33 | 95 |

- Deflection is the largest class (36 of 95). It is a post-training artifact, not a size effect: the models that deflect most on final user-fact turns are the heavily aligned SmolLM2-360M (31%) and Qwen2.5-0.5B (45%); LFM2-350M deflects on 7% and Falcon-H1-Tiny-90M on none. Under card sampling the order changes (wrong or ignored 69, deflection 49, perspective 29, copy 15, stale 8, of 170 failures), but deflection stays the second-largest cause, and Falcon-H1-Tiny-90M still has none.
- Two flavors. Privacy: "I don't have access to personal information like apartment numbers" (LFM2.5-350M). Identity: "I am an AI and do not have a physical location" (Gemma-3-270M), "I'm Qwen ... I don't have a cat or any other pets" (Qwen2.5-0.5B). The identity flavor is also a perspective error: "my cat" gets read as "the assistant's cat".
- Copying its own earlier reply is concentrated at 135M (7 of its 19 failures). Once SmolLM2-135M calls itself "a dental AI" or "a physics AI", later replies repeat that framing.
- Perspective errors peak in the best model. Qwen3-0.6B answers "I'm Marcus.", "I'm Oscar." and "By the way, my dog is named **Waffles**!". A lenient grader that only checks for the name would score these as passes.
- "Wrong or ignored" is mostly the model answering a generic version of the question: "The name of your cat can vary widely depending on its breed" (LFM2-350M), "A cat is called a cat by many people" (Gemma-3-270M in E2), "Your apartment number is 123 Main Street" (Gemma-3-270M in E2).

### F5. Corrections: the ability that fails at every size

Free answers: 20 of 24 multi-turn correction trials fail (greedy, 8 models); the passes are LFM2-350M (Monday to Tuesday), Qwen3-0.6B (Tuesday; 4 pm) and Falcon-H1-Tiny-90M (4 pm, buried in a listicle: "**Arrive 4:00 PM:**"). Forced-choice margin, log P(corrected value) minus log P(stale value), same context:

| model | K_day multi / single | K_time multi / single | K_color multi / single | RI_binding multi / single (Oscar minus Lena) |
|---|---|---|---|---|
| Falcon-H1-Tiny-90M | 1.0 / 3.3 | -1.9 / 2.3 | -3.1 / -2.4 | -2.7 / -1.8 |
| SmolLM2-135M | -1.8 / -0.4 | -4.1 / -0.2 | -0.6 / 0.9 | -2.9 / -1.9 |
| Gemma-3-270M | 1.7 / 7.1 | -7.6 / 2.3 | -4.8 / -8.2 | 9.6 / 7.5 |
| LFM2-350M | 1.4 / 1.7 | -1.2 / 0.7 | -4.5 / 2.5 | 6.2 / 7.5 |
| LFM2.5-350M | -0.2 / 6.9 | -2.7 / 6.8 | -2.9 / 5.7 | 7.6 / 10.9 |
| SmolLM2-360M | 1.6 / 3.0 | -1.8 / 1.3 | -2.1 / 0.4 | 1.0 / 0.3 |
| Qwen2.5-0.5B | 0.9 / 5.3 | -4.6 / 0.7 | 0.9 / 5.8 | 9.4 / 9.8 |
| Qwen3-0.6B | 0.8 / 4.3 | -3.6 / 4.4 | 4.6 / 8.1 | 4.1 / 7.1 |

- For 3 pm to 4 pm, all eight models prefer the stale value after one distractor turn, and seven of eight prefer the corrected value when the same sentences arrive as one message (SmolLM2-135M is slightly negative in both). For blue to green, the multi-turn margin is negative in six of eight models, and in four of those the single-turn margin is positive.
- This is a context failure in the strict sense: same content, same model, only the turn structure differs. It shows up in the 0.5B and 0.6B models as clearly as at 135M, so it is not what separates 150M from 500M. It is a general weakness that is also documented far above 1B: Laban et al. find an average 39% drop from single-turn to multi-turn across top open and closed LLMs, dominated by unreliability rather than lost aptitude ([arXiv 2505.06120](https://arxiv.org/abs/2505.06120); models well above 1B).
- A plausible mechanism, not proven here: the first-stated value gets repeated in the model's own replies ("Your favorite color is indeed blue!") and ends up outnumbering the correction in context. In SmolLM2-135M's blue/green conversation the final answer is a near-verbatim copy of its turn-0 reply.

### F6. Speaker binding and perspective

- "My sister is named Lena and I'm named Oscar ... What's my name?": every model at 270M and up prefers Oscar in both the multi-turn and single-turn versions (margins +0.3 to +10.9). SmolLM2-135M prefers Lena in both (-2.9 and -1.9), and its single-turn free answer is "My name is Lena, and I'm the name of your sister." Falcon-H1-Tiny-90M also prefers Lena in both (-2.7 and -1.8), although its free answers came out right. One test, so one data point, but in both ~100M-or-smaller models it is a capability failure (both versions fail), not a context failure.
- Role capture (the model adopts the user's self-description) appears from 135M to 600M after one turn of "Hi, I'm Jordan and I work as a chef": SmolLM2-135M ("I'm a chef, and I work in a busy restaurant."), LFM2-350M ("As a chef in a busy restaurant, my day is always dynamic"), LFM2.5-350M ("As a chef in a busy restaurant, my main responsibilities include"), Qwen2.5-0.5B ("As a chef, I work in a busy restaurant"), Qwen3-0.6B ("Hi, I'm Jordan! As a chef, I work at a busy restaurant"), and Falcon-H1-Tiny-90M ("As a chef, I am deeply involved in the culinary world"). SmolLM2-360M ("I'm a part of Hugging Face") and Gemma-3-270M passed that turn, and Gemma drifted into it on the next ("I enjoy a variety of tasks that allow me to contribute to the restaurant's success").

### F7. Instruction persistence

| model | deflection on final user-fact turns | self-copy rate | replies with a 4-gram x3 | 'tell me more' cross-turn 4-gram overlap | instruction t1/t2/t3 | persona 'Arr' t0..t3 |
|---|---|---|---|---|---|---|
| Falcon-H1-Tiny-90M | 0.00 | 0.00 | 0.03 | 0.03 | 0.67/0.00/0.00 | 1.00/0.00/0.00/0.00 |
| SmolLM2-135M | 0.24 | 0.17 | 0.22 | 0.95 | 0.67/0.33/0.33 | 0.00/0.00/0.00/0.00 |
| Gemma-3-270M | 0.24 | 0.05 | 0.04 | 0.68 | 0.67/0.67/0.67 | 0.00/0.00/0.00/0.00 |
| LFM2-350M | 0.07 | 0.00 | 0.00 | 0.04 | 0.67/0.67/0.67 | 1.00/1.00/1.00/1.00 |
| LFM2.5-350M | 0.21 | 0.00 | 0.01 | 0.03 | 1.00/1.00/1.00 | 1.00/1.00/1.00/1.00 |
| SmolLM2-360M | 0.31 | 0.12 | 0.08 | 0.46 | 0.67/0.67/0.67 | 1.00/1.00/1.00/1.00 |
| Qwen2.5-0.5B | 0.45 | 0.10 | 0.06 | 1.00 | 0.33/0.33/0.33 | 1.00/1.00/1.00/1.00 |
| Qwen3-0.6B | 0.14 | 0.07 | 0.02 | 0.40 | 1.00/1.00/1.00 | 1.00/0.00/0.00/0.00 |

- Persistence did not decay over three turns in six of eight models (t1 = t2 = t3). Failures happen at the first turn (Qwen2.5-0.5B writes several sentences from the start; Gemma-3-270M never uses capitals) or collapse into a loop (SmolLM2-360M answers "SOLVE:" repeated, in capitals; Qwen3-0.6B repeats "Arr! Let's get to the fight!" four times). SmolLM2-135M and Falcon-H1-Tiny-90M are the only models whose compliance decays: SmolLM2-135M from 0.67 at t1 to 0.33 at t3; Falcon follows the one-sentence and all-caps instructions at the first question and drops both after it.
- The system-prompt persona's "Arr" appeared in all four turns for every model from 350M up, but Qwen3-0.6B's last three replies were verbatim copies of its first, which the grader rejects. The required "Arr" never appeared from SmolLM2-135M or Gemma-3-270M (Gemma opens every reply with "Ahoy there, matey!"), and Falcon-H1-Tiny-90M said it only in its first reply.

### F8. Follow-ups, own-answer references, topic return

- Elliptical follow-ups ("And what about Italy?") pass from 350M up and fail at 135M ("Italy is also known as the \"City of Love.\"") and in Gemma-3-270M ("Italy is the capital of Italy."), while both pass the single-turn control. Falcon-H1-Tiny-90M passes it ("Italy is not a capital city. The capital of Italy is Rome"), so this is not a size threshold either; it fails the French follow-up, but it also fails the single-turn French control (knowledge, not context).
- Pointing into its own previous list is unreliable below 0.5B: LFM2-350M names the third item when asked for the second, Gemma-3-270M names the first when asked for the last, SmolLM2-135M names a fruit that was not on its list, and Falcon-H1-Tiny-90M gets the second item right but answers "I didn't put any name last." Qwen3-0.6B passes both list questions; Qwen2.5-0.5B passes the gradable one (its fruit list repeated "Plum", so the second-item question is ungradable).
- Topic return with an arithmetic step (6 guests x 2 cupcakes) passes in every model from 350M up except SmolLM2-360M, which divides instead of multiplying in both the multi-turn and single-turn versions (a capability failure, not a context one). It fails at 135M and 270M, and passes at 90M (Falcon).

### F9. Loops, self-copy, and whether sampling fixes them

Same metrics with each card's recommended sampling (2 seeds; Falcon 1 seed):

| model | deflection on final user-fact turns | self-copy rate | replies with a 4-gram x3 | 'tell me more' cross-turn 4-gram overlap | instruction t1/t2/t3 | persona 'Arr' t0..t3 |
|---|---|---|---|---|---|---|
| Falcon-H1-Tiny-90M | 0.00 | 0.00 | 0.02 | 0.00 | 0.33/0.33/0.33 | 1.00/0.00/0.00/0.00 |
| SmolLM2-135M | 0.19 | 0.17 | 0.26 | 0.93 | 0.33/0.17/0.17 | 0.00/0.00/0.00/0.00 |
| Gemma-3-270M | 0.17 | 0.03 | 0.02 | 0.45 | 0.33/0.50/0.33 | 0.00/0.00/0.00/0.00 |
| LFM2-350M | 0.05 | 0.00 | 0.01 | 0.03 | 0.67/0.67/0.50 | 1.00/1.00/1.00/1.00 |
| LFM2.5-350M | 0.10 | 0.00 | 0.00 | 0.01 | 1.00/1.00/1.00 | 1.00/1.00/1.00/1.00 |
| SmolLM2-360M | 0.31 | 0.12 | 0.08 | 0.31 | 0.67/0.67/0.67 | 1.00/1.00/1.00/1.00 |
| Qwen2.5-0.5B | 0.33 | 0.01 | 0.02 | 0.12 | 0.33/0.33/0.17 | 1.00/0.50/0.50/0.00 |
| Qwen3-0.6B | 0.09 | 0.05 | 0.02 | 0.47 | 1.00/1.00/1.00 | 1.00/1.00/1.00/1.00 |

Ability scores under sampling:

| model | params | recall | followup | instruction | correction | own_answer | role | topic_return | macro | single-turn controls |
|---|---|---|---|---|---|---|---|---|---|---|
| Falcon-H1-Tiny-90M | 91.1M | 0.50 | 0.71 | 0.36 | 0.67 | 0.75 | 0.33 | 0.67 | **0.57** | 0.61 |
| SmolLM2-135M | 134.5M | 0.35 | 0.50 | 0.21 | 0.00 | 0.33 | 0.00 | 0.17 | **0.22** | 0.38 |
| Gemma-3-270M | 268.1M | 0.05 | 0.64 | 0.25 | 0.00 | 0.38 | 0.00 | 0.67 | **0.28** | 0.48 |
| LFM2-350M | 354.5M | 0.65 | 1.00 | 0.75 | 0.33 | 0.38 | 0.33 | 0.83 | **0.61** | 0.71 |
| LFM2.5-350M | 354.5M | 0.65 | 1.00 | 1.00 | 0.33 | 0.50 | 0.33 | 0.83 | **0.66** | 0.80 |
| SmolLM2-360M | 361.8M | 0.70 | 0.71 | 0.79 | 0.17 | 0.33 | 0.67 | 0.67 | **0.58** | 0.66 |
| Qwen2.5-0.5B | 494.0M | 0.30 | 0.79 | 0.39 | 0.00 | 0.88 | 0.33 | 0.83 | **0.50** | 0.68 |
| Qwen3-0.6B | 596.0M | 0.50 | 0.93 | 1.00 | 0.33 | 0.75 | 0.00 | 1.00 | **0.65** | 0.91 |

- Card sampling does not change the diagnosis. Macro scores move by at most 0.10 in either direction (SmolLM2-135M 0.15 to 0.22, Gemma-3-270M 0.36 to 0.28, Qwen2.5-0.5B 0.60 to 0.50, Falcon-H1-Tiny-90M 0.49 to 0.57 on one seed), and the picture above 300M is the same: the 350M models stay level with or above Qwen2.5-0.5B.
- Sampling with a repetition penalty fixes the verbatim loops where they are a decoding artifact: Qwen2.5-0.5B (T 0.7, repetition penalty 1.1) goes from repeating its first ocean paragraph five times (cross-turn overlap 1.00, self-copy 0.10) to 0.12 and 0.01. SmolLM2-135M at its card setting (T 0.2, no penalty) stays at 0.93 overlap and 0.17 self-copy. At 135M the loop is in the model, not only in greedy decoding.
- Deflection is not a greedy artifact: under sampling Qwen2.5-0.5B still deflects on 33% of final user-fact turns and SmolLM2-360M on 31%; Falcon-H1-Tiny-90M still never does.
- Falcon-H1-Tiny-90M has no card sampling advice; I used T 0.7 and top-p 0.9, one seed.

### F10. Context failure vs capability failure

| model | both pass | context failure (control passes, multi fails) | capability failure (both fail) | multi-only pass |
|---|---|---|---|---|
| Falcon-H1-Tiny-90M | 9 | 9 | 6 | 4 |
| SmolLM2-135M | 1 | 12 | 12 | 3 |
| Gemma-3-270M | 8 | 13 | 6 | 1 |
| LFM2-350M | 19 | 6 | 1 | 2 |
| LFM2.5-350M | 17 | 6 | 2 | 3 |
| SmolLM2-360M | 12 | 3 | 6 | 7 |
| Qwen2.5-0.5B | 12 | 11 | 4 | 1 |
| Qwen3-0.6B | 20 | 6 | 2 | 0 |

A context failure passes the single-turn control and fails the multi-turn version; a capability failure fails both. Caveat: for recall, the single-turn control is lenient by construction (the fact sits in the same message), which inflates the context-failure count for recall checks; F3 is the better measure there.

- At 135M, capability and context failures are equal (12 and 12): half of what goes wrong would go wrong with no conversation at all. Falcon-H1-Tiny-90M is similar (9 and 6).
- From 350M up, context failures outnumber capability failures (LFM2-350M 6 vs 1, LFM2.5-350M 6 vs 2, Qwen2.5-0.5B 11 vs 4, Qwen3-0.6B 6 vs 2): these models know the answer and lose it to the turn structure, mostly by deflecting or by keeping a stale value.
- SmolLM2-360M is the exception (3 vs 6) because its controls are weak (0.54): it deflects on single-turn personal questions too.

### F11. The 90M outlier: Falcon-H1-Tiny-90M-Instruct

This is the closest existing model to what Max wants to build (74M non-embedding parameters, 800B tokens), so its profile is the most useful single result for the 150M plan.

- **Scores.** Macro 0.49 (recall 0.50, follow-ups 0.57, instruction 0.29, corrections 0.33, own-answer 0.75, role 0.33, topic return 0.67); single-turn controls 0.64. That is below the 350M class (0.57 to 0.65) and well above SmolLM2-135M (0.15) and Gemma-3-270M (0.36).
- **What works.** Zero deflections on final user-fact turns (0 of 29 greedy), zero self-copy, and no loops on "tell me more" (cross-turn 4-gram overlap 0.03). In the fixed-history sweep it retrieves every fact at every distance up to 1,726 tokens (greedy-from-prefix 1.00) and answers 0.80 (short replies) and 0.75 (long) correctly. No other model, in any of the 21 greedy and sampled runs of the other seven, recalled what the user complained about four turns earlier; Falcon did ("you were feeling a bit bored").
- **What fails.**
  - A dominant how-to-guide template: "Which city do I live in?" becomes "To determine which city you live in, consider the following factors: ### 1. **Population**". Asked to end every reply with a question, it writes a list of questions for the user to ask instead.
  - Template placeholders instead of facts: "Your name is [Your Name]." and "Your dentist appointment is scheduled for Monday, [Date]."
  - Single-turn knowledge: "Gracias" glossed as "(Greetings)", tigers that "inhabit the deserts of South America".
  - Format persistence collapses after one turn: the one-sentence and all-caps instructions are followed at the first question and dropped from the second on; the pirate persona says "Arr" once.
  - Perspective on names: in the fixed-history sweep 0.20 of short-reply answers are "Hello! My name is Priya." In the Oscar/Lena forced choice it prefers the sister's name (margins -2.7 multi, -1.8 single), although its free answer happened to be right.
- **Reading.** At 74M non-embedding parameters, retrieving and using a user fact across 12 turns is achievable with the right recipe. What a model this size still lacks is knowledge, a varied response style, format persistence and first/second-person binding. Those are the targets for a 150M model, and they are mostly data and post-training problems.


---

## What this implies for a ~150M chat model

1. **Set the target by the 350M models, not the 0.5B one.** On this battery LFM2-350M, LFM2.5-350M and SmolLM2-360M already hold multi-turn chats about as well as Qwen2.5-0.5B. The practical goal for a 150M model is to reach their macro score (about 0.6) with about 120M non-embedding parameters instead of about 290-315M.
2. **Split the 135M-to-350M gap into its two halves.** In my data about half of it is reply policy (deflection, persona, answering as the user, copying itself) and half is capability (single-turn controls 0.46 at 135M vs 0.82 to 0.93 for the good 350M-600M models, plus binding and long-context retrieval). The policy half is a data problem at any size and cheap to attack. The capability half is where parameters and tokens matter.
3. **Be honest about tokens.** Every model here that chats was pretrained on 4T to 36T tokens (cards and reports cited above), 11,000 or more tokens per parameter (SmolLM2-360M: 4T for 362M), and even the failing SmolLM2-135M saw 2T (about 15,000 per parameter). Max's current build is 100B tokens, about 670 tokens per parameter at 150M. The capability half of the gap will be harder for Max than it was for SmolLM2-135M (2T tokens), unless distillation or much better data closes it. This probe cannot say how much; the data and training lanes should.
4. **Do not spend the novelty budget on memory architecture for short chats.** The models with about 290M or more non-embedding parameters (all keep full attention in at least some layers, e.g. LFM's 6 of 16) kept a user fact retrievable across 12 turns and about 1,700 tokens (greedy-from-prefix 0.75 to 1.00, Qwen3 aside for its bold-formatting artifact). Falcon-H1-Tiny-90M, with only 74M non-embedding parameters (attention and Mamba2 side by side in every layer), retrieved every fact at 1,726 tokens. Even Gemma-3-270M, with full attention in only 3 of 18 layers, retrieved it half the time at 1,662 tokens. Only SmolLM2-135M faded (after about 500 tokens through its chat template). For a 150M model aimed at chats of a few thousand tokens, the cheaper wins are elsewhere.
5. **Measure the right thing during training.** Loss and single-turn benchmarks would not have shown any of the failures above. This battery (runs in about 5 minutes per model on the Mac) plus the forced-prefix probe separates "cannot retrieve" from "retrieves but will not say it", which is the distinction that decides whether to fix data or capacity.

---

## Evidence about the wall (why chat quality degrades below ~400M)

What the transcripts say, ordered by how much of the degradation each explains:

1. **Reply policy learned in post-training, not memory.** 63% of failed recall or binding checks are retrievable by greedy decoding from the answer prefix (F3). The same 135M weights go from 0.15 to 0.70 recall when only the chat rendering changes (E1). Deflection is the largest greedy failure class (36 of 95) and is worst in the heavily aligned 360M and 0.5B models (31% and 45%), so it is not a size effect at all.
2. **Below about 100M non-embedding parameters the policy cannot be steered.** A one-line "use the conversation" system prompt removes deflection at 350M and up (LFM2.5-350M goes to 1.00 recall) and does nothing at 135M or Gemma-270M (E3). The same two models also ignore a system-prompt persona. At this size the behavior must be trained in; it cannot be prompted in.
3. **Capacity limits at ~100M: real, but not where Max expected.** Memory is not one of them in general: Falcon-H1-Tiny-90M retrieves every fact at 1,726 tokens. SmolLM2-135M does fade (greedy-from-prefix 0.00 at 12 chat-template turns, 0.50 at 1,635 plain-transcript tokens), so a weak recipe can create the limit. What both ~100M models share: single-turn knowledge and skill are weak (controls 0.46 and 0.64 vs 0.82 to 0.93 for the good 350M-600M models), speaker binding fails even single-turn (Oscar vs Lena margins -1.9 and -1.8), and format instructions do not persist. SmolLM2-135M also loops (22% of replies contain a 4-gram three or more times; 17% reuse at least half of an earlier reply), and card sampling does not fix it (0.93 cross-turn overlap on "tell me more").
4. **State update is weak at every size tested.** Corrections fail 20 of 24 times from 90M to 600M, and the log-probability margin flips toward the stale value in the multi-turn version for all eight models on 3 pm vs 4 pm. This is a shared weakness, also documented at much larger scale ([Laban et al.](https://arxiv.org/abs/2505.06120)), so it does not explain why 150M is worse than 500M, but it is the ability a 150M model will fail most visibly.
5. **Recipe at fixed size.** Gemma-3-270M and SmolLM2-135M have the same ~100M of non-embedding parameters; Gemma (6T tokens) passes 0.75 of single-turn controls and 0.86 of follow-ups, SmolLM2-135M (2T) 0.46 and 0.29. Falcon-H1-Tiny-90M has fewer non-embedding parameters (74M) and fewer tokens (800B) than either, yet the best macro score of the three (0.49 vs 0.36 and 0.15) and the best fixed-history recall of any model per parameter. What it does differently: SFT data mixed into pretraining, no deflection habit, a hybrid attention + Mamba2 stack. Three models that differ in everything, so this is suggestive, not controlled, but it says the ordering below 300M is set by recipe, not size.
6. **External corroboration.** TII reports that its smaller models "loop significantly more" and that chain-of-thought traces in training data caused repetition loops at 90M ([TII](https://tiiuae-tiny-h1-blogpost.hf.space/), 90M). Max's own MaxGPT-2 and MaxGPT-3 notes record the same low-entropy loops (`~/Documents/Projects/Max's AI Model/WRITEUP_NOTES.md`). The LFM2 card says to fine-tune the 350M models on narrow use cases and not to use them for knowledge-intensive tasks ([card](https://huggingface.co/LiquidAI/LFM2-350M)).

---

## Levers

| lever | failure it attacks | evidence | expected effect, how sure |
|---|---|---|---|
| Multi-turn SFT data where later turns ask about user-stated facts, answered in second person ("Your name is Priya"), with facts 1 to 20 turns back | deflection, perspective errors | E1: rendering alone moves 135M recall 0.15 to 0.70; 62-63% of failures are retrievable (F3); Falcon-H1-Tiny-90M, trained with SFT data in pretraining, recalls 0.75-0.80 with zero deflections (F11) | large for recall at 150M; moderate confidence that SFT (rather than format) captures most of it; untested directly |
| Remove or rewrite privacy and identity deflections in SFT/DPO data ("I don't have access to personal information", "I'm an AI, I don't have a cat") when the answer is in the conversation; add DPO pairs with the deflection as the rejected reply | deflection (largest class) | deflection rates 7% (LFM2) to 45% (Qwen2.5) track post-training, not size (F4) | moderate to large; cheap (Max already has a DPO pipeline); speculative on size of effect |
| Do not train with a fixed default system prompt or identity blurb; vary or omit it | persona-triggered deflection | SmolLM2's template injects "You are a helpful AI assistant named SmolLM"; plain transcript fixes recall (E1); "I'm Qwen ... I don't have a cat" (F4) | moderate; weak evidence on the mechanism (the template and system prompt were not varied separately) |
| Synthetic correction and state-update conversations ("actually it's Tuesday", then distractors, then a question), including cases where the assistant previously repeated the old value | stale values | fails 20/24 at every size; single-turn controls mostly pass, so the skill exists and the turn structure breaks it (F5) | probably necessary at 150M; unknown how far it goes; nothing in this probe shows it working |
| Speaker-binding data: two named people, "my/your/her" questions, role-swap traps | binding, role capture | SmolLM2-135M and Falcon-H1-Tiny-90M prefer the sister's name even single-turn; role capture from 90M to 600M (F6) | moderate; one binding test only |
| On-policy training on the model's own multi-turn rollouts (on-policy distillation from a bigger teacher, or SFT on corrected self-generated histories) | self-copy, self-invented personas, loops | 135M copies itself in 17% of replies; Qwen3-0.6B, whose small models get on-policy logit distillation ([report](https://arxiv.org/html/2505.09388)), has the best macro score and 7% self-copy | speculative: confounded by 36T tokens; the mechanism fits the failure |
| Repetition penalty plus real temperature at inference | verbatim loops | fixes Qwen2.5-0.5B (overlap 1.00 to 0.12) but not SmolLM2-135M at its card setting (T 0.2) | cheap, partial; does not fix the model |
| Keep real attention in some layers; a short (2-4k) context is enough | long-range retrieval | every model with about 290M or more non-embedding parameters, and Falcon-H1-Tiny-90M at 74M, retrieves at about 1,700 tokens (greedy-from-prefix 0.75-1.00, Qwen3 aside); only SmolLM2-135M fades | moderate; covered in depth by the context and architecture lanes |
| Mix chat/SFT data into pretraining | single-turn skill at tiny scale | TII reports IFEval 66.08 for the SFT-in-pretraining variant vs 53.47 for its curriculum variant, both after DPO, at 90M ([TII](https://tiiuae-tiny-h1-blogpost.hf.space/)); in this probe the resulting 90M model has zero deflection and the best fixed-history recall per parameter | moderate at 90M for single-turn; this probe's multi-turn read on it is in the Falcon section |
| Use this battery plus the forced-prefix probe as a training-time eval | all | it found failures that single-turn scores hide; about 5 minutes per model on the Mac | strong as a tool |

---

## Open questions

1. Does a small dose of targeted multi-turn SFT (a few thousand synthetic conversations covering recall, corrections and perspective) fix deflection and binding in SmolLM2-135M-Instruct? This is cheap to test on the Mac (full fine-tune of 135M) with this exact battery and is the most decision-relevant next experiment.
2. Why does SmolLM2-135M's retrieval fade between 500 and 1,600 tokens when Falcon-H1-Tiny-90M's does not? Candidates: architecture (Falcon runs attention and Mamba2 in parallel in every layer; SmolLM2 is plain attention with 3 KV heads), pretraining context length, or data. Testing SmolLM2-135M-base on longer plain transcripts, and a same-size attention-only model trained on Falcon-like data, would separate them.
3. Why do corrections fail everywhere? Candidate causes: the model's own replies repeating the old value, primacy, or no correction examples in SFT. A fixed-history version of the correction tests (canned replies that do and do not repeat the old value) would separate the first two.
4. How much of E1's format effect is the default system prompt versus the chat-template tokens? E3 shows a memory system prompt barely helps at 135M, which suggests the template itself, but I did not run the template without SmolLM2's default system prompt.
5. The grader is strict about perspective ("I'm Marcus." fails). Human raters might score some of those as fine. A small human rating pass on the transcripts would calibrate the strict grader.
6. All results are at n of 2 to 20 per cell, with one probe author. The battery should be expanded (more facts, paraphrased questions) before any single cell is used for a decision.
