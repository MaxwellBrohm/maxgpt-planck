# MaxGPT-Planck

**Goal: find the smallest language model that can hold a real multi-turn conversation.**

Not "answers a question." A conversation: it remembers what you told it several turns ago, takes
corrections ("actually, his name is Pickles"), keeps track of whose dog is whose, keeps following
an instruction you gave earlier, and does not loop. Planck is named after the Planck length, the
smallest size that still means something.

<!-- Max, 2026-10-02: took all recommendations in rc12/DECISIONS_FOR_MAX.md (item 15) -->
Tiny chat models already exist: released models from 90M to 350M work as loose chat models, and
hobby and lab models trained from scratch go down to about 5M. What no one has shown is a model
under 150M passing a strict multi-turn test, and on our test's dev split none of the 13 released chat
models we scored, from 90M to 2.6B, nor the 11 smaller public chat models from 5M to 75M we added,
reaches any of its bars. Planck's bet is that
conversational skill is cheap in parameters and knowledge is what is expensive, so a model can
keep facts outside its weights (in its context, in notes, in a lookup) and spend its parameters on
the skill. The result we are after is a curve, not a single model: the same recipe at 5M, 10M, 20M,
30M, 60M and 150M total parameters (and 3M if 5M passes a Level A family on dev), with most of the
effort at 30M and under, each compared against much larger models on the same test.

## What is new here, and what is not

<!-- Max, 2026-10-02: took all recommendations in rc12/DECISIONS_FOR_MAX.md (item 15) -->
Planck is not the first to ask how small a chat model can be (SmallTalkLLM), to train chat models
under 30M from scratch (BananaMind-2, Loom, Veyra2, Vertex, Micro Language Models, MiniMind), to
train tiny models on synthetic multi-turn dialogue (RxT, Micro Language Models), or to keep facts
outside the weights (LMLM, Co-LMLM, KLLM, Loom). Links, and what each one does and does not show,
are in [`research/REPORT.md`](research/REPORT.md) sections 2 and 2.1.

What Planck aims to add is the first measured floor for strict multi-turn state skills: the
smallest from-scratch model that passes a sealed, mutation-tested 12-turn test on its own
conversation history (corrections, binding, rule persistence, own-answer consistency, role,
loops), with several training seeds, against a panel of public models in one harness. A prior-art
search on 2026-10-02 ([`research/NOVELTY_2026-10.md`](research/NOVELTY_2026-10.md)) found no
published model under 150M passing a test like that, and no published minimum-size curve for these
skills. The test scores the bare model: every reply comes from the conversation's own history, with
no retrieval, memory store, state tracker or other help from the harness.

## Status

Every experiment, its pre-registered rule, its result and its independent audit are written up in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

- **Research and idea ledger: done (Sep 23 2026).** Start with [`research/REPORT.md`](research/REPORT.md),
  section 1; every idea is ranked in [`LEDGER.md`](LEDGER.md).
- **E001 to E006: done (E002 to E006 independently audited; E001's checks were built into the run).**
  Below 2.6B, small instruct models keep the first value after a correction (E001). A 135M model
  fine-tuned for 400 steps passed E002's rule by a wording shortcut; on E004's harder held-out test it
  failed the rule but learned an updating rule that transfers to new wording, longer distances and more
  corrections. Name-based corrections and end-of-turn stopping turned out to be missing data, not a size
  limit (E005). With E002's recipe, none of eight public base models from TinyStories-1M to Pythia-160M
  passed even the easy version of the correction test (0 of 3 seeds each); it is not read as a capacity
  limit, since only 400 steps were run and for five models the best learning rate was the lowest one tried
  (E003). Keeping three objects apart is still unsolved: changing where the latest statement sits in
  training did not restore it (INCONCLUSIVE, still below E004). Mixing human-written chat into training
  about halved the replies that open with a training template and cut the knowledge loss, though the chat
  label holds by a narrow margin (E006).
- **Calibration (E2, E3): E2 done; E3 Parts 1 to 3 done.** For Planck's own models on the starter data,
  E2's 5M learning-rate pick, 3e-3, holds at 20M within a factor of 2 (E2); E3 confirmed it against
  1.5e-3 (worse on 7 of 7 re-check seeds) but did not resolve it against 6e-3. Seed noise on held-out
  chat bits per byte is about 0.25% at 5M, so a bet aimed at a 1% difference runs 2 seeds (4 if it
  changes the data stream); varying only the data order moved results more than varying only the initial
  weights, though not by a resolved margin (E3). Ten more 5M seeds and the same weights-versus-order
  split at 20M were launched on Oct 12; no reading of them is recorded yet.
- **SCREENS batch 1, seven screens at 5M: closed (Oct 11).** Nine one-change arms (S002 tests three
  removals separately), two seeds each, decided on held-out bits per byte, where a win only nominates a
  bet for later tests at 20-30M. Canon-AC layers (-0.9% chat bits per byte) and a forget gate on
  attention (-1.2%, flagged by a pre-registered diagnostic) are nominated; removing the value residual,
  AdamW in place of NorMuon and a t+2 prediction head each made chat loss worse by about 1.5%; the other
  four changes tie. Nothing is adopted on a 5M loss result. A five-seed robustness reading is
  pre-registered beside each verdict, and batch 2 (S009-S021) is a draft, not frozen.
- **T001, a 2-layer recall-with-updates toy: done (Oct 11).** Plain softmax attention never learned the
  task; on the verdict seeds, every arm that learned had a previous-token path or stick-breaking
  attention. By its pre-registered picks, a gated delta-rule layer takes the must-run 30M architecture
  slot (a tie with Canon-AC, decided by the registered default) and stick-breaking attention is the
  recency-path variant for its 30M test.
- **RC-12, the 12-turn test:** built, mutation-tested and scored on dev for the public panel (13 released
  chat models from 90M to 2.6B, plus 11 smaller public chat models from 5M to 75M and three more
  Falcon-H1-Tiny-90M chat checkpoints). No Planck model is scored on any split. Three claims are
  pre-registered, each worded "on our panel": Level A, a strict bar on corrections, binding and loops plus
  a blind human test; Level R, a state score over seven state families whose 95% CI lower bound must reach
  40 of 100, above the 36.7 that a mix of fixed-rule shortcuts scores on dev; and the LOOKUP claim,
  reading facts from old and new tables supplied in the conversation (bar 0.60). Max changed Level R on
  Oct 4 from non-inferiority to Qwen2.5-0.5B, which scores 2.3 on those families on dev, so a 5M model
  already met the old bar mostly by not looping. No panel model reaches any of the three on dev: the best
  state score is 18.2 (LFM2-2.6B), the best LOOKUP 0.014, and no model below 90M scores above 2.1. OOD-H
  Part 1 (150 held-out conversations with human-written user turns, 415 probes) is fixed for the lock: Max
  reviewed a 30-thread sample on Oct 10 (85 of 85 probes looked right), then re-checked the 11 threads
  that a third review had changed (11 of 11 conversations, 32 of 32 probes).
- **Frontier calibration (Nov 3, dev):** Claude Haiku 5.5, run through the same loop and graders, scores
  S7 41.2 [36.7, 45.7], 2.3 times the best panel model, with its CI lower bound just under the bar. Its
  transcripts show the grader failing right answers that add context ("The banjo. Your nephew is the one
  with the trombone."). An answer-first waiver scores it 57.4 [52.9, 61.8] while LFM2-2.6B stays at 20.0
  and every shortcut stays at 36.7. Adopting that rule is Max's call before the lock:
  [rc12/FRONTIER.md](rc12/FRONTIER.md), with one conversation side by side in
  [rc12/examples](rc12/examples/corr-014-lfm2-vs-claude-haiku.md).
- **The lock: not yet.** It was planned for Oct 11-14, and the pre-registration is still a draft. The lock
  package (sealed-session kit, lock check, finalize step) was built and dry-run without any sealed content
  on Oct 11-13. Left: settling the grader gaps the OOD-H reviews found (for example "You asked if X." is
  graded wrong), building and encrypting the hidden split in a separate session, Max's approval, then the
  lock commit and push.
- **Toolchain:** tokenizer v0 is built (its 8k cut matches a separately trained 8k; no superword tokens),
  the openly licensed starter corpus is tokenized (2.0B tokens), the core v0 corpus is built and reviewed
  (about 35.9B tokens at 8k; a second near-duplicate pass over the cleaned text leaves about 35.4B), the
  two training speed-ups (a batched optimizer step and variable-length
  attention for packed documents) are the GPU defaults, and the teacher pipeline has been through several
  dry pilots with the three candidate teachers on stand-in data; the teacher pick waits for the first real
  pilot.
- **Next:** the lock steps above, before any Planck model is scored on RC-12; reading E3's two follow-ups;
  the five-seed screen reruns and the batch 2 freeze; the 30M architecture tests T001 picked, each under
  its own pre-registration; the first real teacher pilot.

## Ground rules

- **Count total parameters,** embeddings included. The non-embedding body is reported too.
- **The test comes first.** The multi-turn test (12-turn conversations, sealed held-out wording,
  mutation-tested graders) is published here before any Planck results, so nobody, including us,
  can tune the model to it after the fact.
- **One change at a time** against a well-tuned baseline, same test, several seeds. Ideas that
  fail stay in the ledger as results.
- **Open data only for the headline model:** synthetic conversations written by Apache-2.0
  open-weight models, verified by program.
- **Negative results get published too.**

## What is here

| Path | What |
|---|---|
| `docs/EXPERIMENTS.md` | The lab notebook: every experiment, its pre-registered rule, its result and its audit |
| `PLAN.md` | The project plan: phases, the size curve, machines, budget and Max's dated decisions |
| `experiments/` | One folder per experiment (E001-E006, E2, E3, the S0xx screens, T001, Track K): notes, pre-registrations, audits and results |
| `rc12/`, `prereg/RC-12.draft.txt` | The 12-turn test (generators, graders, runner, build notes, lock steps) and its draft pre-registration |
| `oodh/` | OOD-H, the held-out set with human-written user turns: design, build code and published hashes (the items stay sealed) |
| `CORPUS.md`, `corpus/`, `tokenizer/`, `data_prep/` | Corpus design and build, Planck's tokenizer, and tokenizing and bits-per-byte scoring |
| `harness/` | Planck's from-scratch training code |
| `models/` | The model registry and the rule for keeping trained weights |
| `research/REPORT.md` | The final research report: verdict, landscape, probe results, limits map, levers, feasibility |
| `research/lanes/`, `research/followup/`, `research/brainstorm/` | Every research track, each with a `.verify.md` fact-check next to it |
| `research/diagnosis_*.md`, `research/CRITIQUE.md` | Three independent diagnoses and the adversarial review of the report |
| `research/probe/`, `research/capacity_probe/` | Code, transcripts and scores from the hands-on probes of real small models |
| `LEDGER.md` | The idea ledger |
| `tools/gemma_chat.py` | Terminal chat with Gemma 4 through LM Studio with thinking truly off (Gemma 4 is one of three candidate teachers; the pick waits for the teacher pilot) |

## Credits

MaxGPT-Planck is Max Brohm's project. The research phase was run with Claude (Anthropic) agents:
literature review, fact-checking and the model probes were done by AI agents and reviewed by other
agents, and every claim in the reports carries its source. Earlier models in the family: MaxGPT-1,
2, 3 and 3.5 (the OG series), MaxGPT-Mini and MaxGPT-Ultra (the Neo series).

License: to be chosen before the first model release.
