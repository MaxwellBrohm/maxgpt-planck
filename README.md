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

- **Research phase: done (Sep 23 2026).** A literature sweep, a hands-on probe of eight real small
  chat models, adversarial fact-checks of every lane, and three independent diagnoses of what
  limits a small chat model. Start with [`research/REPORT.md`](research/REPORT.md), section 1.
- **Idea ledger: done.** [`LEDGER.md`](LEDGER.md) holds every idea from the research (331 raw ideas
  merged into 29 baseline items, 185 ranked bets and 57 parked), each labeled honestly as known,
  a tweak of known work, or untested at this size, with its closest prior work.
- **Now:** fixing the measurement (test controls, probes of more small models) before training
  anything.

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
| `tools/gemma_chat.py` | Terminal chat with Gemma 4 through LM Studio with thinking truly off (the teacher model) |

## Credits

MaxGPT-Planck is Max Brohm's project. The research phase was run with Claude (Anthropic) agents:
literature review, fact-checking and the model probes were done by AI agents and reviewed by other
agents, and every claim in the reports carries its source. Earlier models in the family: MaxGPT-1,
2, 3 and 3.5 (the OG series), MaxGPT-Mini and MaxGPT-Ultra (the Neo series).

License: to be chosen before the first model release.
