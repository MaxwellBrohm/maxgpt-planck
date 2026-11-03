# RC-12 frontier calibration (dev split, 2026-11-03)

**Question.** No public model on the panel reaches Level R (best S7: LFM2-2.6B, 18.15). Is RC-12 passable at all,
and does a model that clearly can hold a conversation clear the bar of 40? If a frontier model fails too, "no small
model passes" says little about small models.

**Answer, in one line.** Claude Haiku 5.5 scores S7 41.2 under the current graders, 2.3 times LFM2-2.6B. Its 95% CI
lower bound (36.7) sits just under the bar. Reading its transcripts shows the grader failing right answers. With one
narrower rule (below), it scores 57.4 [52.9, 61.8], clearing the bar. LFM2-2.6B moves only from 18.2 to 20.0, and
the shortcut ceiling stays at 36.7.

## Method

`api_run.py` plays the 640 dev conversations through `runner.run_one`, the same feedback loop as the panel (each
model's own replies are its history). The responder calls the Anthropic Messages API. Rows are graded by the same
graders and scored by `score.summarize`. The `--own-cf` OWN twin is run for the OD1 b gate, and the CI comes from
`score_stats.s7_ci`. `frontier_report.py` regrades every stored transcript (nothing replayed) under three rule sets.
As a check, it reproduces LFM2-2.6B's panel S7 of 18.15 exactly from the PC's stored transcripts.

Protocol differences, all forced by the API (recorded in each run's `meta.json`):

- **Render:** template only (chat messages). There is no plain render.
- **Samples:** no seed and no temperature, since current models reject a non-default temperature. Seed "1" is a
  label for one sample at default sampling. There is no greedy run.
- **Reply cap:** 256 tokens, as in the panel, counted in Claude's tokenizer.
- **Thinking:** off.
- **Prompt:** no system prompt.
- **Fallback:** no refusal fallback. There were 0 refusals and 0 empty replies.

## Results (S7, 95% CI lower and upper bound; Level R needs the lower bound at 40 and PERSIST at 0.40)

| rules | Claude Haiku 5.5 (1 sample) | LFM2-2.6B (seeds 1-3) | shortcut ceiling |
|---|---|---|---|
| current graders | 41.22 [36.74, 45.68], not met | 18.15 [14.73, 21.75] | 36.73 |
| v3_other + v3_shotgun dropped | 64.73 [60.65, 68.66] | 25.14 [21.21, 29.39] | **92.38** (SHOTGUN) |
| answer-first waiver | **57.43 [52.93, 61.83], met** | 20.01 [16.28, 23.88] | 36.73 |

Other measures for the same two models:

| | Claude Haiku 5.5 | LFM2-2.6B |
|---|---|---|
| PERSIST | 0.668 | 0.812 |
| T0 | 0.729 | 0.517 |
| replies cut at the 256-token cap | 37% | 55% |

## What the grader gets wrong

G-VAL's `v3_other` fails a reply that asserts any other in-context value. `v3_shotgun` fails one that names every
candidate. Both exist to stop a model from listing every candidate, and the middle row shows why: drop them, and the
SHOTGUN fake (which lists every candidate) scores 92.4. But a frontier model answers, then gives context:

- "The banjo. Your nephew is the one with the trombone." (BIND: fails `v3_other` and `v3_shotgun`)
- "Saturday. It started on Wednesday, then moved to Friday, then Thursday, and the latest change was to Saturday."
  (CORR: fails `v3_other`)

The **answer-first waiver** (`frontier_report.g_val_answer_first`) drops those two clauses only when the reply's
first sentence asserts the gold and names no other value of the probe. To use it, a cheater would have to know the
gold and lead with it, so it gives no shortcut a gain: the ceiling stays at 36.73.

T0 shows a similar pattern through `v2_unsure`. Haiku loses credit on 24 of 48 T0 conversations, and in 18 the only
failed clause is `v2_unsure`, in replies that do state the right answer. Some add a caveat after a right answer: "You
told me just now: Friday is the one weekday you never work. So that's your day off. I don't know anything else about
your schedule, though." Others genuinely hedge ("I don't know your favorite food ... But you did mention tacos
earlier"), and the clause is right to fail those, so this one needs a case-by-case read before any change.

## Limits

- **One Claude sample.** Haiku 5.5 has a single sample; the panel pools three seeds. The account's API spend limit
  stopped the run before seeds 2 and 3.
- **No Sonnet result.** Claude Sonnet 5.5's full run died at the same limit and kept nothing. `api_run.py` now saves
  each conversation as it finishes and resumes.
- **Dev split only.** The waiver is a probe of the grader, chosen after reading dev transcripts. Adopting it is a
  rule change for Max to decide before the sealed split is scored (no model has been scored on it).
- **Haiku is behind on two measures.** It scores below LFM2-2.6B on PERSIST (0.668 vs 0.812) and misses T0's
  0.9 claim bar.

## Example

[examples/corr-014-lfm2-vs-claude-haiku.md](examples/corr-014-lfm2-vs-claude-haiku.md): one dev conversation, both
models, every reply verbatim. LFM2-2.6B gives the stale day; Claude Haiku 5.5 gives the right one.

## Decision for Max

Adopt the answer-first waiver in `graders.py` before the lock, rerun the mutation suites and the panel regrade, then
report frontier calibration beside Level R. Read the 18 `v2_unsure`-only T0 misses before deciding anything for
that clause. **Recommended: yes for the waiver.** It keeps every shortcut at 36.7 and stops the grader from failing the answers a
capable model gives. Until then the current graders stand, and the site and README report them.

## Reproduce

```
python api_run.py --model claude-haiku-5-5 --seeds 1 --max-usd 5     # key: ANTHROPIC_API_KEY
python -B frontier_report.py --ref <copy of ~/planck/runs/rc12_dev> --models claude-haiku-5-5 --out report.json
```

Cost: $0.95 for Haiku 5.5's 640 conversations plus the 48 OWN twins (7,904 API calls).

Transcripts are not in git, like every run (`rc12/runs/` is ignored). They sit beside the panel's runs on the PC, at
`~/planck/runs/rc12_dev_frontier/` (with `report.json` and `shortcut_ceiling.json`); the LFM2-2.6B rows are at
`~/planck/runs/rc12_dev/LFM2-2.6B/`. The numbers were computed on the working tree at 8b482d1, which held an
uncommitted `grade_text.py` edit from another session. That tree reproduces the panel's 18.15 and the 36.73 ceiling
exactly.
