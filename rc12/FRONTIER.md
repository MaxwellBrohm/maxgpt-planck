# RC-12 frontier calibration (dev split, 2026-11-03)

**Question.** No public model on the panel reaches Level R. Is RC-12 passable at all, and does a model that clearly
can hold a conversation clear the bar of 40? If a frontier model fails too, "no small model passes" says little about
small models.

**Answer.** Under the graders as they stood, Claude Haiku 5.5 scored S7 40.35 [36.59, 44.13]. That is 2.2 times
the best panel model, but its CI lower bound fell short of the bar. Its transcripts showed the grader failing right
answers that add context ("The banjo. Your nephew is the one with the trombone."). Max adopted an answer-first
waiver the same day. It was built and checked against the 9,670 grader fixtures and 237 mutants, and every stored
panel run was regraded. Under it:

- **Claude Haiku 5.5** scores 55.70 [51.65, 59.60] and meets Level R's two bars (S7 lower bound and PERSIST).
- **LFM2-2.6B** moves from 18.15 to 19.99.
- **No panel model** reaches the bar.
- **The shortcut ceiling** stays at 36.73.

## Method

`api_run.py` plays the 640 dev conversations through `runner.run_one`, the same feedback loop as the panel: each
model's own replies are its history. The responder calls the Anthropic Messages API. Rows are graded by the same
graders and scored by `score.summarize`, with the `--own-cf` OWN twin for the OD1 b gate. The CI comes from
`score_stats.s7_ci`. `frontier_report.py` regrades the stored transcripts (nothing replayed) before and after the
waiver.

Protocol differences, all forced by the API (recorded in each run's `meta.json`):

- **Render:** template only (chat messages). There is no plain render.
- **Samples:** no seed and no temperature, since current models reject a non-default temperature. Seeds 1, 2 and 3
  are labels for three independent samples at default sampling. There is no greedy run.
- **Reply cap:** 256 tokens, the panel's cap, counted in Claude's tokenizer.
- **Thinking:** off.
- **Prompt:** no system prompt.
- **Fallback:** no refusal fallback. There were 0 refusals and 0 empty replies.

## Results (S7 [95% CI]; Level R needs the lower bound at 40 and PERSIST at 0.40)

| model | samples | before the waiver | after (current graders) | PERSIST | T0 | replies cut at 256 | Level R |
|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 3 | 40.35 [36.59, 44.13] | **55.70 [51.65, 59.60]** | 0.666 | 0.740 | 37% | **met** |
| Claude Sonnet 5.5 | 1 | 21.67 [19.05, 24.45] | 62.43 [58.71, 66.04] | 0.356 | 0.865 | 52% | not met (PERSIST) |
| LFM2-2.6B (best panel model) | 3 | 18.15 [14.73, 21.75] | 19.99 [16.29, 23.79] | 0.812 | 0.517 | 55% | not met |
| shortcut ceiling (best fake per family) | | 36.73 | 36.73 | | | | |

Every panel model before and after is in `logs/regrade_first_20261103.jsonl`.

**Diagnostic: both clauses dropped.** With `v3_other` and `v3_shotgun` removed entirely (not a rule):

- Haiku scores 63.43.
- Sonnet scores 68.05.
- LFM2-2.6B scores 25.14.
- The SHOTGUN fake, which lists every candidate, scores 92.38. That is why the clauses stay and only the waiver was
  added.

**Sonnet 5.5 and PERSIST.** Sonnet misses Level R only on PERSIST. 132 of its PERSIST failures are `f1_degen`:
the reply follows the user's standing rule ("Greetings! Yes, a few habits ...") but runs past the 256-token cap,
which RC-12 counts as a runaway reply. The cap rule was written for small models that ramble or loop. Whether a
well-formed reply cut at the cap should count is a separate decision (below). Sonnet's samples 2 and 3 were still
running when this was written.

## The rule (graders.answer_first, adopted 2026-11-03)

`v3_other` fails a reply that asserts any other in-context value, and `v3_shotgun` fails one that names every
candidate. Both stop a model from listing candidates to hit the gold. The waiver drops the two clauses when all three
of these hold:

1. **The answer leads.** The first sentence asserts the gold.
2. **No second answer.** The first sentence asserts no other value.
3. **Nothing points the other values at the asked thing.** No other value is asserted right after "it is / it's /
   that's / this is / it will be" or "your <object words> is", within the same clause and 20 characters.

Values given to another person ("your grandmother made burritos") or to the history ("you first said Hamburg")
stay allowed. A contradiction ("Your massage session is on Saturday. It is on Thursday.") still fails.

How it was checked:

- **Fixture suite: 9,670 fixtures, 0 misses.** The original waiver, without condition 3, failed 24 contradiction
  fixtures. Max chose to flip the 197 `other_too`, answer-led `shotgun` and `inflect_other` fixtures to must-pass.
  A new `shotgun` with the denials first stays must-fail. New `af_*` fixtures cover each condition. Lead-position
  fixtures keep the stale and inflection exemptions tested on their own, because 5 old mutants survived until they
  were added.
- **Mutants: 237 of 237 killed, 10 of them new.**
- **`validate_e2e.py`:** G1 to G4 pass.
- **`dev_regrade.py --expect first`:** over the 440 stored panel runs (151,360 rows), 376 probes changed. Each one only
  lost `v3_other` or `v3_shotgun`; no probe gained a failure.
- **Panel reproduction:** the before copy reproduces the published panel exactly.

On Haiku's seed 1, a random 14 of the probes the waiver flipped were all right answers with the other values
attributed. Some examples: "Your nephew made sushi. Your grandmother made burritos."; "Your job interview is in
Calgary. You first mentioned Hamburg, then switched to Calgary."

## Limits

- **Dev split only.** The waiver was designed after reading dev transcripts, before the lock, and no model has been
  scored on the sealed split.
- **Sonnet is incomplete.** It has one sample here; seeds 2 and 3 were running.
- **The 256-token cap counts a long, well-formed reply as a runaway.** This hits frontier models hardest (Sonnet:
  52% of replies are cut).
- **Haiku is behind on two measures.** It scores below LFM2-2.6B on PERSIST (0.666 vs 0.812) and misses T0's 0.9
  claim bar. In 18 of the 24 T0 conversations where it lost credit (seed 1), the only failed clause is `v2_unsure`:
  some replies add a caveat after a right answer, others hedge.

## Example

[examples/corr-014-lfm2-vs-claude-haiku.md](examples/corr-014-lfm2-vs-claude-haiku.md): one dev conversation, both
models, every reply verbatim. LFM2-2.6B gives the stale day; Claude Haiku 5.5 gives the right one.

## Still for Max

1. **The cap rule.** Should a reply cut at the 256-token cap count as a runaway (`f1_degen`) when it is well-formed
   and not looping? Recommended: keep the cap, but count a cut reply as degenerate only when it also trips a loop
   rule. This would be a new waiver, checked the same way as this one.
2. **The 18 `v2_unsure`-only T0 misses.** Read them before changing anything for that clause.

## Reproduce

```
python api_run.py --model claude-haiku-5-5 --seeds 1,2,3 --max-usd 5         # key: ANTHROPIC_API_KEY or --env-file
python -B frontier_report.py --root runs/frontier --ref <copy of ~/planck/runs/rc12_dev> --out report.json
python -B dev_regrade.py --src <regraded base copy> --dst <dir> --expect first   # then dev_panel.py on <dir>
```

Cost: Claude Haiku 5.5, three samples plus the OWN twins, $2.85. Claude Sonnet 5.5's first sample plus twin was
about $21.

Transcripts are not in git, like every run (`rc12/runs/` is ignored). On the PC:

- the Claude runs: `~/planck/runs/rc12_dev_frontier/`
- the two regraded panel copies: `~/planck/runs/rc12_dev_rg_base/` and `rc12_dev_rg_af/`
- their `dev_panel.py` outputs: `~/planck-af/panel_base/` and `panel_af/`
