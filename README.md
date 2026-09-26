# MaxGPT-Planck

**Goal: find the smallest language model that can hold a real multi-turn conversation.**

Not "answers a question." A conversation: it remembers what you told it several turns ago, takes
corrections ("actually, his name is Pickles"), keeps track of whose dog is whose, keeps following
an instruction you gave earlier, and does not loop. Planck is named after the Planck length, the
smallest size that still means something.

Today the smallest models that feel conversational sit around 0.5B parameters, and in our own
tests no model at or below 0.6B passes a strict multi-turn bar. Planck's bet is that
conversational skill is cheap in parameters and knowledge is what is expensive, so a model can
keep facts outside its weights (in its context, in notes, in a lookup) and spend its parameters on
the skill. The result we are after is a curve, not a single model: the same recipe at roughly
10M, 30M, 60M and 150M total parameters, each compared against much larger models on the same test.

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
- **RC-12, the 12-turn test:** Max's rulings are recorded, two loop-rule follow-ups are decided and three
  items stay open for Max, a grader bug found while building the OOD-H set (real human conversations with
  Claude-written probes) was fixed before any model was scored, and OOD-H Part 1 (150 held-out
  conversations) is built as a lock candidate. No model is scored on it yet.
- **Toolchain:** tokenizer v0 is built (its 8k cut matches a separately trained 8k; no superword tokens),
  the openly licensed starter corpus is tokenized (2.0B tokens), the learning-rate and seed-noise
  calibration runs (E2, E3) are ready, three candidate teacher models are pinned, and two training speed-ups
  are under verification.
- **Next:** dev baselines on RC-12, then the sealed split and the lock (planned Oct 11-14) before any
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
