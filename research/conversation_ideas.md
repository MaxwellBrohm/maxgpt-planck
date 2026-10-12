# Ideas from the Sep 23 2026 brainstorm with Max

Source: the planning conversation, not the research runs. Every idea here goes into the idea ledger
next to the research's own ideas. Max's rule for the ledger: try as much as possible that has NOT
been tested at this size (10M to 150M total parameters), each as a change against a well-tuned
baseline, same multi-turn test, 2 to 3 seeds.

| # | Idea | Where it came from | Prior art we know of | Status at <=150M |
|---|---|---|---|---|
| 1 | Headline = the smallest model that passes a multi-turn test, shown as a curve (10M / 30M / 60M / 100M / 150M, total params incl. embeddings) | Max | Micro LMs (arXiv 2604.19642): 8M-30M chat-trained models, but only the first words of a reply, no multi-turn memory | multi-turn memory at this size: not found |
| 2 | Skill in the weights, knowledge outside (lookup) | agreed thesis | entity-anonymized pretraining at 135M/360M (arXiv 2607.12831), Prereq-Tune, RETRO | not tested in chat |
| 3 | Beat the ~2 bits/param knowledge ceiling (replicate at 1M-5M on the Mac, then interventions) | Max ("store more per param"); fp64 answered: no (bf16 to fp32 was only +9%) | Physics of LMs 3.3, Morris et al. 2025, Zucchet et al. 2025 schedules | ceiling never beaten |
| 4 | Skill-over-storage architecture: small MLPs, strong attention, looped layers | Max (custom architecture, "best use of every param") | Physics 3.3 (MLP size does not change bits/param), MoEUT, looped-transformer scaling law | chat-specific test not found |
| 5 | Planck's own 4k-8k BPE trained on chat text; vocab sweep at fixed total | Max (tokenizer), follow-up verdict | Tao et al. 2024, Compute Optimal Tokenization 2026 | multi-turn sweep not found |
| 6 | "TinyStories for conversation": skeleton -> teacher renders -> program verifies, dense cross-turn skills | Max + research | TinyChat, TinyDialogues (no dense verified skills) | the combination not found |
| 7 | One-line memory note each turn (`dog: Pickles (was Biscuit)`), latest note kept in context | brainstorm | SimpleTOD / SOLOIST belief state (117M GPT-2, task dialogue only) | open-domain chat not found |
| 8 | Short capped thinking, stripped from history on the next turn (+ the note) | Max | Gemma 4 / Qwen3 / DeepSeek-R1 templates drop old thinking (standard); pause tokens | chat at this size not found |
| 9 | Program-side verification: generate N replies, code picks the one that passes (planted facts) | brainstorm | best-of-N / verifier literature | for tiny chat: not found |
| 10 | Memory baked into weights across sessions: a few LoRA steps on the transcript, new empty session, ask again; measure recall + bleed | Max | fast weights, ROME/MEMIT, TTT layers, Titans, Temp-LoRA, per-user LoRA | tiny chat model with per-user weight memory: not found |
| 11 | Corrections as a headline first ("actually I meant Tuesday"), trained with latest-value-wins data | research + brainstorm | Wu et al. ICML 2025 (37.8M variable binding) | no general chat model up to 0.6B shown handling corrections |
| 12 | Teacher: Gemma 4 12B via raw completions with the empty thought block pre-filled (thinking off; 14 tok/s single stream on the M5, 9 s per 12-line chat) | measured today | | |

Teacher habits to control (measured today): it copies instruction wording literally (write exact
first-person lines in skeletons), and in free chat adds emojis, bold text and em dashes (instruct
against them and reject any conversation that has them). Decide what the assistant may claim about
itself (it invented its own hiking plans).
