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
- **E001-E005: done, each checked by an independent audit.** Below 2.6B, small instruct models keep the
  first value after a correction (E001). A 135M model fine-tuned for 400 steps passed E002's rule, but
  the audit showed a wording shortcut. On E004's harder held-out test it failed the pre-registered rule
  on 5 of 5 seeds, yet learned an updating rule that transfers to new wording, longer distances and
  more corrections. It fails on name-based references it never saw in training and on three objects.
- **E005: done and audited.** Adding name-based corrections and end-of-turn training fixed both at 135M (they were missing data, not a size limit). Keeping three objects apart is still unsolved and got worse, and narrow training still costs general chat and some knowledge.
- **Now:** the Planck toolchain. The RC-12 test decisions are being recorded, our own tokenizer is being built on an openly licensed starter corpus, and the training code is being sped up on an RTX 5070.
- **Next:** the same question for models of 30M and under, and locking the 12-turn RC-12 test before
  any Planck model is scored on it.

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
