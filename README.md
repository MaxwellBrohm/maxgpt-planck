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
models we scored, from 90M to 2.6B, nor the 9 public small chat models from 5M to 31M we added, reaches
any of its bars. Planck's bet is that
conversational skill is cheap in parameters and knowledge is what is expensive, so a model can
keep facts outside its weights (in its context, in notes, in a lookup) and spend its parameters on
the skill. The result we are after is a curve, not a single model: the same recipe at roughly
10M, 30M, 60M and 150M total parameters, each compared against much larger models on the same test.

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
- **E001, E002, E004, E005: done (E002, E004 and E005 independently audited; E001's checks were built
  into the run).** Below 2.6B, small instruct models keep the first value after a correction (E001). A 135M model fine-tuned for 400 steps passed E002's rule by a
  wording shortcut; on E004's harder held-out test it failed the rule but learned an updating rule that
  transfers to new wording, longer distances and more corrections. Name-based corrections and
  end-of-turn stopping turned out to be missing data, not a size limit (E005). Keeping three objects
  apart is still unsolved, and narrow training costs general chat and some knowledge.
- **Running:** E003, the easy version of the correction test on public base models (TinyStories-1M to
  Pythia-160M), on the laptop. No result is claimed until the queue finishes and its tables are checked;
  the five smallest stayed near chance on E004's harder test.
  E006 (three objects, and protecting general chat) is pre-registered and waits for the RTX 5070.
- **RC-12, the 12-turn test:** built, mutation-tested and scored on dev for the public panel (13 released
  chat models from 90M to 2.6B, plus 9 public small chat models from 5M to 31M; a 75M one did not finish
  within its time cap). No Planck model is scored on any split. Three claims are pre-registered, each
  worded "on our panel": Level A, a strict bar on corrections, binding and loops plus a blind human test;
  Level R, a state score over seven state families whose 95% CI lower bound must reach 40 of 100, above
  the 36.7 that a mix of fixed-rule shortcuts scores; and the LOOKUP claim, reading facts from old and
  new tables supplied in the conversation (bar 0.60). Max changed Level R on Oct 4 from non-inferiority
  to Qwen2.5-0.5B, which scores 2.3 on those families on dev, so a 5M model already met the old bar
  mostly by not looping. No panel model reaches any of the three on dev: the best state score is 18.2
  (LFM2-2.6B), the best LOOKUP 0.014.
  OOD-H Part 1 (150 held-out conversations with human-written user turns) is built as a lock candidate
  and waits for Max's review.
- **Toolchain:** tokenizer v0 is built (its 8k cut matches a separately trained 8k; no superword tokens),
  the openly licensed starter corpus is tokenized (2.0B tokens), the learning-rate and seed-noise
  calibration runs (E2, E3) are ready, three candidate teacher models are pinned, and two training speed-ups
  are under verification.
- **Next:** Max's OOD-H Part 1 review, the sealed split and the lock (planned Oct 11-14) before any
  Planck model is scored on it; E2 and E3; the teacher pilot.

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
