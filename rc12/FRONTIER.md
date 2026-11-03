# RC-12 frontier calibration (dev split, 2026-11-03)

**Question.** No public model on the panel reaches Level R. Is RC-12 passable at all, and does a model that clearly
can hold a conversation clear the bar of 40? If a frontier model fails too, "no small model passes" says little about
small models.

**Answer.** Under the graders as they stood, Claude Haiku 5.5 scored S7 40.35 [36.59, 44.13]. That is 2.2 times
the best panel model, but its CI lower bound fell short of the bar. Claude Sonnet 5.5 scored only 21.6. Reading
their transcripts showed two grader rules failing good answers:

- **Context after the answer:** a reply that names another value after giving the answer failed `v3_other` ("The
  banjo. Your nephew is the one with the trombone.").
- **Long replies:** every reply cut at the 256-token cap counted as a runaway, however well-formed.

Max changed both on Nov 3. The answer-first waiver lets a leading answer name the other values. The cap amendment
counts a cut reply as degenerate only if it loops. Each change was checked against the grader fixtures and mutants,
and every stored panel run was regraded.

Under the current graders:

- **Claude Sonnet 5.5** scores 66.09 [62.29, 69.88] and meets Level R.
- **Claude Haiku 5.5** scores 55.90 [51.78, 59.86] and meets Level R.
- **LFM2-2.6B**, the best public model, scores 23.09 [19.21, 27.08].
- **No panel model** reaches the bar.
- **The shortcut ceiling** stays at 36.73.

## Method

`api_run.py` plays the 640 dev conversations through `runner.run_one`, the same feedback loop as the panel: each
model's own replies are its history. The responder calls the Anthropic Messages API. Rows are graded by the same
graders and scored by `score.summarize`, with the `--own-cf` OWN twin for the OD1 b gate. The CI comes from
`score_stats.s7_ci`. `frontier_report.py` regrades the stored transcripts (nothing replayed) under each grader
version.

Protocol differences, all forced by the API (recorded in each run's `meta.json`):

- **Render:** template only (chat messages). There is no plain render.
- **Samples:** no seed and no temperature, since current models reject a non-default temperature. Seeds 1, 2 and 3
  are labels for three independent samples at default sampling. There is no greedy run.
- **Reply cap:** 256 tokens, the panel's cap, counted in Claude's tokenizer.
- **Thinking:** off.
- **Prompt:** no system prompt.
- **Fallback:** no refusal fallback. There were 0 refusals and 0 empty replies.

## Results (S7 [95% CI]; Level R needs the lower bound at 40 and PERSIST at 0.40)

| model | samples | Oct graders | + waiver | current (+ cap amendment) | PERSIST | T0 | replies cut at 256 | Level R |
|---|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 2 (3rd running) | 21.64 | 61.21 | **66.09 [62.29, 69.88]** | 0.739 | 0.885 | 52% | **met** |
| Claude Haiku 5.5 | 3 | 40.35 | 55.70 | **55.90 [51.78, 59.86]** | 0.793 | 0.740 | 37% | **met** |
| LFM2-2.6B (best panel model) | 3 | 18.15 | 19.99 | 23.09 [19.21, 27.08] | 0.832 | 0.542 | 55% | not met |
| shortcut ceiling (best fake per family) | | 36.73 | 36.73 | 36.73 | | | | |

Every panel model under all three grader versions is in `logs/regrade_cap_20261103.jsonl`, and the waiver step
alone is in `logs/regrade_first_20261103.jsonl`. No panel model meets Level R under any of them.

**Why Sonnet failed before.** Before the cap amendment, Sonnet missed Level R only on PERSIST (0.363): most of its
PERSIST failures were `f1_degen`. The reply followed the user's standing rule ("Greetings! Yes, a few habits ...")
but ran past the 256-token cap. Counted on its text, its PERSIST is 0.739.

**Diagnostic: both clauses dropped.** On the Oct graders, with `v3_other` and `v3_shotgun` removed entirely (not a
rule), Haiku scores 63.43, Sonnet 65.79 and LFM2-2.6B 25.14. The SHOTGUN fake, which lists every candidate, scores
92.38. That is why the clauses stay and only the waiver was added.

## Rule 1: the answer-first waiver (graders.answer_first, adopted 2026-11-03)

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

## Rule 2: the cap amendment (grade_loop, adopted 2026-11-03)

RUNAWAY (a reply stopped at the 256-token cap) stays a reported flag, so the panel still shows how often each model
is cut. It is no longer degenerate on its own. A cut reply is graded on its text and fails only if it also loops,
by the LOOP rule (a repeated 3-gram, an equal earlier reply, or a self-copy).

How it was checked:

- **Fixture suite: 9,838 fixtures, 0 misses.** `runaway` became the must-pass `runaway_ok`; a new `runaway_loop`
  must fail.
- **Mutants: 237 of 237 killed.** 3 are new for the amendment. 3 were dropped as equivalent, because a cut reply no
  longer changes any verdict on its own: the clause `loop_RUNAWAY`, OFF `runaway`, and the cap-test edit.
  `validate_machinery`'s CAPPER check still pins the RUNAWAY rate.
- **`validate_machinery.py`: 12 of 12.** CAPPER, the IDEAL reply cut at the cap, now scores R 100.
- **`validate_e2e.py`:** G1 to G4 pass, and the ceiling is 36.73.
- **`dev_regrade.py --expect cap`:** 70,790 probes changed. Each one only lost a degenerate clause.

The cap rule mattered most for models that write long answers. The pre-registered loop rule (SPEC s4) now carries
a dated amendment in `grade_loop.py`.

## Limits

- **Dev split only.** Both rules were designed after reading dev transcripts, before the lock, and no model has been
  scored on the sealed split.
- **Sonnet's third sample was still running** when this was written; the table shows two.
- **Haiku misses T0's 0.9 claim bar (0.740).** On its seed 1, 18 of the 24 T0 conversations where it lost credit
  failed only `v2_unsure`. Some replies add a caveat after a right answer, others hedge.

## Example

[examples/corr-014-lfm2-vs-claude-haiku.md](examples/corr-014-lfm2-vs-claude-haiku.md): one dev conversation, both
models, every reply verbatim. LFM2-2.6B gives the stale day; Claude Haiku 5.5 gives the right one.

## Still for Max

1. **The 18 `v2_unsure`-only T0 misses.** Read them before changing anything for that clause.

## Reproduce

```
python api_run.py --model claude-haiku-5-5 --seeds 1,2,3 --max-usd 5         # key: ANTHROPIC_API_KEY or --env-file
python -B frontier_report.py --root runs/frontier --ref <copy of ~/planck/runs/rc12_dev> --out report.json
python -B dev_regrade.py --src <regraded base copy> --dst <dir> --expect first   # then dev_panel.py on <dir>
```

Cost: Claude Haiku 5.5, three samples plus the OWN twins, $2.85. Claude Sonnet 5.5 costs about $21 per sample plus
twin.

Transcripts are not in git, like every run (`rc12/runs/` is ignored). On the PC:

- the Claude runs: `~/planck/runs/rc12_dev_frontier/`
- the three regraded panel copies: `~/planck/runs/rc12_dev_rg_base/`, `rc12_dev_rg_af/` and `rc12_dev_rg_cap/`
- their `dev_panel.py` outputs: `~/planck-af/panel_base/`, `panel_af/` and `panel_cap/`
