# E003 audit (independent, 2026-10-02)

One adversarial auditor re-derived the final reading from the raw records in `out/`, the guard records and `logs/queue.txt`, with its own code (`audit/a1`-`a6`). The item modules were imported only to rebuild the item lists (stdlib), and every scored record was matched to its rebuilt item by prompt hash. No model was loaded, nothing ran on the PC, the GPU was not used, nothing was committed, and nothing outside `AUDIT.md` and `audit/` was written. What was not re-checked is listed near the end.

**E003 audit: the reading reproduces. All 8 models FAIL (0 of 3 seeds each), no dev run at any searched learning rate reaches 0.8, and no record from a killed attempt is read. Three wording corrections: two knowledge figures are mis-rounded; "fails at every LR" means every LR in a grid whose lowest point won for 5 of 8 models; and two post-hoc "never" and "upward" statements need their scope. The prior-art claim holds as far as six searches and eight paper pages reach.**

**Q1. Integrity: CONFIRMED (`a2_integrity.py`, 0 problems)**
- 61 jobs were checked: 8 baselines, 29 LR-search runs and 24 scored seeds.
  - The LR count is right: 27 grid runs (3 models x 4 LRs, 5 x 3) plus 2 extensions (pythia-14m and TinyStories-3M, both to 1e-02).
  - `out/` holds exactly these tags plus the dry runs: base, dry, lr*_s0 and s1-s3 (no stray s0, s4 or duplicate tag). The analyzer reads only base and s1-s3.
- **Each run.json:**
  - eval_counts equal the pre-registered sizes (new 1728, extra 192, old 1116, uprobe 576, khard 80, kbig 441, cross 384, dev 320);
  - every .jsonl has exactly that many lines;
  - every trained run has 400 steps and 400 finite losses;
  - the loss self-check gap is within 1e-3;
  - no run has a `fatal` field.
- **Final guard record:** for every job, exit 0 and killed None. The command equals, element by element, the one `queue_e003.sh` builds: interpreter, model, COMMON args, seed, sets, save, LR and tag.
- **Timing and attempts:**
  - Every `out/` file and every weights file has its mtime inside its job's final guard window (start - 1 s to end + 2 s).
  - Each job has exactly one completed attempt in `queue.txt`, it is the job's last attempt, and its time equals the guard end time.
- **Killed attempts:** 34 (33 swap, 1 preflight), matching the report job by job:
  - ft_ts33m_s1 16, ft_ts33m_s2 7, lr_p160m_3e-04 5;
  - lr_ts1m_1e-03 2, lr_ts1m_3e-04 1;
  - ft_p14m_s3 1, ft_p160m_s2 1;
  - base_ts33m 1 (preflight).
- **Splits:** LR-search runs have seed 0, sets ["dev"] and no eval files. Scored runs have seeds 1/2/3, sets ["eval"] and the chosen LR.
- **Environment:** one interpreter and torch 2.13.0 / mps / fp32 in every run.json, baselines included.
- **Mutation tests** (`a5_mutants.py`): 4 of 4 a2 mutants were killed (wrong interpreter, guard window shifted 1 h, a job completed twice, a pre-registered count off by one), and the unmutated run gave 0 problems.
- I did not re-run `final_e003.py` or its 5 mutants. The checks above cover the same ground with separate code.

**Q2. Analyzer unchanged and reproduced: CONFIRMED WITH CAVEAT**
- `analyze_e003.py`, `metrics_ft.py`, `params.py`, `pick_lr.py`, `e003_ft_test.py` and all item and training modules equal git HEAD. Their last commit is 9a10e0f (2026-09-24 19:46), which is before the first LR-search run (2026-09-26 00:44).
  - Only `guard.py` differs from HEAD. That is D1: it now has sha 30f97995, the same as the E002 and E004 copies. The old guard is kept as `logs/guard_old_81c20d46.py` with the matching hash.
- I re-ran the unchanged analyzer in a scratch copy (`a6_rerun_analyzer.sh`, torch asserted absent). `results.json`, `tables.txt` and `analyze_stdout.txt` came out byte-identical to the stored files.
- **Caveat:** "identical to the queue's copies" cannot be checked any more. Those copies were overwritten at 18:08 and no hash of them was kept.
  - It is plausible: no input changed after the last job (no `out/` file is newer than ft_p160m_s3's guard record, 2026-09-27 20:56:52; the lr_pick files date from 2026-09-27 19:41-19:51), and the analyzer is deterministic (fixed bootstrap seed).
- No commit has touched E003 since 9a10e0f, so "nothing was committed" holds for E003. I did not check "nothing ran on the PC".

**Q3. Pass-rule cells and verdicts: CONFIRMED (`a1_cells.py`, 0 problems)**
- **Records against rebuilt items.** I rebuilt the eval set with `items_new.build()` (1,728 items), the dev draw (seed 3003, 320 items) and the crossed set (384 items).
  - Every record of every baseline, LR run and seed matches its item on id, sha1[:12] of the prompt, variant, distance, family, scenario id and candidate labels.
  - That is 0 mismatches across 8 x 4 eval runs and 29 dev runs.
- **Grading.** My own grader counts an item right only when every score is finite and the gold is strictly above every other candidate.
  - All 96 cells (8 models x base + 3 seeds x 3 cells) equal `results.json` to 4 decimals.
  - The rule as written (each cell >= 0.8 on a seed; a strict majority of 3 seeds) gives FAIL for all 8 models, 0/3 each.
- Seed counts (k/n), with the report's means in brackets:

| Model | LW10 (/192) | TS10 (/64) | NU10 (/64) | Report means |
|---|---|---|---|---|
| TinyStories-1M | 69 84 83 | 20 18 23 | 31 27 28 | .41/.32/.45 |
| pythia-14m | 74 74 69 | 20 24 20 | 33 28 32 | .38/.33/.48 |
| TinyStories-3M | 67 67 66 | 19 25 20 | 35 33 37 | .35/.33/.55 |
| pythia-31m | 191 189 192 | 48 48 46 | 32 33 34 | .99/.74/.52 |
| TinyStories-8M | 98 81 95 | 28 22 23 | 38 31 32 | .48/.38/.53 |
| pythia-70m | 191 190 189 | 42 44 42 | 33 33 35 | .99/.67/.53 |
| TinyStories-33M | 112 101 86 | 33 27 23 | 32 47 40 | .52/.43/.62 |
| pythia-160m | 181 191 175 | 47 22 41 | 35 0 37 | .95/.57/.38 |

- Body counts check against the closed-form non-embedding count. GPT-NeoX: 12d^2+13d per layer + 2d; GPT-Neo: 12d^2+10d per layer + 2d. All 8 match the table, and the Pythia values equal the Pythia paper's.
- **Mutation tests:** 5 of 5 a1 mutants were killed (d4 pooled with d10, one record hash changed, LR pick by mean, accuracies +0.3, near-ties counted right).

**Q4. LR search, picks and "fails at every LR": CONFIRMED WITH CAVEAT**
- **Picks.** I wrote the pick rule from the notes text (min, then the higher mean, then the smaller LR; extend once if the top grid LR wins below 0.8). It reproduces every pick and the queue's sequence:
  - TinyStories-1M 5e-05;
  - pythia-14m EXTEND 1e-02, then CHOSEN 1e-02;
  - TinyStories-3M EXTEND, then 1e-02;
  - pythia-31m 5e-05; TinyStories-8M 3e-04; pythia-70m 5e-05; TinyStories-33M 5e-05; pythia-160m 5e-05.
  - Every seed trained at its model's pick.
- **The 0.8 bar.** No LR-search run reaches dev min 0.8. Best dev min per model: .297, .354, .375, .516, .312, .500, .568, .688. The report's .30/.35/.38/.52/.31/.50/.57/.69 are these values rounded.
- **TinyStories-1M:** dev mins .297/.266/.266/.281, as reported. Its 3e-04 and 1e-03 runs were restarts.
- **Caveat 1 (scope of "every LR").** The pick is the lowest grid LR (5e-05, E002's rate) for 5 of 8 models: ts1m, p31m, p70m, ts33m and p160m.
  - The pre-registered extension only goes up, so nothing below 5e-05 was tried.
  - Each LR has one seed-0 run on the 320-item dev draw.
  - So "fails at every LR" means "at every LR of a grid whose best point was its lower edge for 5 models", one dev seed each. That is the rule as written. It is weaker than "no learning rate works".
- **Caveat 2 (what the min rule picked).** At 1e-02:
  - pythia-14m starts at a mean loss of 11.97 over the first 10 steps, and TinyStories-3M at 7.16.
  - Their scored seeds sit at chance (chance is LW .389, TS .333, NU .5).
  - pythia-14m at 5e-05 had dev LW .906 but NU .188, below chance; the min rule correctly preferred the all-chance run.
  - The FAIL holds either way. But the pythia-14m knowledge loss (Q7) is the cost of a near-divergent rate, not of the task.

**Q5. Floor statement: CONFIRMED WITH CAVEAT (wording)**
- No model passes, so the pre-registered floor ("smallest body at which a model passes") does not exist inside either ladder.
- "Above 85,056,000 body on Pythia" is accurate only as "no Pythia model up to 85.1M body passes".
  - No Pythia size passes at all, so it is not known that any Pythia size would.
  - TS10 and NU10 are non-monotone in size (as reported), so the ladder gives no trend to extrapolate.
- The report's refusal to read 85.1M-106.2M as a floor estimate is correct. E002's pass is cross-family, and its audit showed a wording rule passes it.

**Q6. Secondary labels and lock-in: CONFIRMED (`a3_secondary.py`)**
- **Per-object tracking:** 0/3 for every model, trivially, because nothing passes. The crossed pair at d10 alone is .73/.75/.77 (p31m), .59/.55/.53 (p70m), .48/.13/.59 (p160m) and .00-.02 elsewhere.
- **TinyStories fitting-items diagnostic** (seq_len <= 512):
  - The fitting counts are LW 91/192, TS 38/64 and NU 64/64, exactly as the pre-registration predicted.
  - It is 0/3 on all four models. The best fitting cell is ts33m s1 LW .75. Every seed fails TS on fitting items (.32-.42).
- **Lock-in:** none on any seed. Each run has 9 probe points, from step 0 to 400.
- **Position-table norms** (dry logs): the rows past 512 are .46-.57 of rows 0-511 (ts1m .082/.143, ts3m .164/.356, ts8m .116/.222, ts33m .284/.539).

**Q7. Knowledge: CONFIRMED WITH CORRECTION (rounding; one omission)**
- My own paired bootstrap (10,000 resamples) and the exact counts agree with `results.json`. Change in points vs the untouched model, with lost-minus-gained items out of 441:

| Model (base kbig) | s1 | s2 | s3 |
|---|---|---|---|
| pythia-160m (.694) | -10.7 (-47) | -14.3 (-63) | -15.0 (-66) |
| pythia-70m (.635) | -2.0 (-9) | -2.3 (-10) | -5.2 (-23), McNemar p .030 |
| pythia-31m (.628) | -3.2 (-14) | **-3.9** (-17), McNemar p .075 | **-0.5** (-2) |
| pythia-14m (.585) | -9.1 (-40) | -6.8 (-30) | -9.8 (-43) |

- **Correction:** the report's pythia-31m "3.8" and "0.4" should read 3.9 and 0.5 (17/441 = 3.855%, 2/441 = 0.454%). The cause is a 4-decimal stored value rounded again for display.
- **Borderline interval.** pythia-31m s2's interval only just includes 0: the upper bound is +0.0 in mine and +0.2 in the stored one. "All CIs include 0" is technically right.
- **Omission:** pythia-14m loses 6.8-9.8 points with every CI excluding 0 (McNemar p .0095 on s2).
  - The rule reads it as uninformative because its base is .585 < .6, and the notes say this. The report folds it into "the other five read as uninformative".
  - It should say the drop is significant but read as uninformative by rule (see Q4 caveat 2).
- The CI-excludes-0 calls for 160m (all), 70m (s3 only) and 31m (none) reproduce.

**Q8. Item and seed noise: CONFIRMED**
- Every one of the 24 seeds has a failing cell whose Wilson 95% upper bound is below 0.8. The non-robust failing cells are p31m TS10 (upper bounds .840/.840/.814), p160m s1 TS (.827) and ts33m s2 NU (.827). Each of those seeds also fails another cell robustly.
- The Clopper-Pearson one-sided 95% bound after 0/3 is .632.
- The NU10 cell fails on every seed of every model, and so does TS10.

**Q9. Post-hoc observations: CONFIRMED WITH CAVEAT (scope)**
- **Weekday items:** 8 of 9 Pythia 31m/70m/160m seeds score .94-1.00 on all three cells. The exception is p160m s2, with day TS .69 and NU .00.
- **Colour items:** TS10 .00-.50 and NU10 .00-.16.
  - Every wrong colour TS/NU item on these 9 seeds has the other object's later value on top: 48/48, 47/47, 48/48, 53/53, 51/51, 51/51, 43/43, 64/64, 50/50.
  - p160m s2 puts "later" on top on 64/64 noupd items.
- **Scope 1.** "LW10 alone reaches 0.8 from pythia-31m (4.74M body) upward" holds on the Pythia ladder only. TinyStories-8M (6.3M body) and TinyStories-33M (28.3M body) never reach it (LW10 .42-.58).
- **Scope 2.** "TS10 and NU10 never do" holds for the scored seeds. On the seed-0 dev draw, single cells did reach 0.8: p31m 5e-05 TS .906, and ts33m 5e-05 NU .859.
- **Notebook text** (`logs/notebook_e003.txt`): it says TinyStories is "near chance on all three cells at every size". TinyStories-33M is clearly above LW chance (.389): .58/.53/.45, z = 5.5 on s1, base .385. "Mostly near chance; TinyStories-33M partly above it" would be accurate.
- The pythia-14m 5e-05 dev reading (LW .91, TS .33, NU .19) is accurate.

**Q10. Deviations and kills: CONFIRMED**
- **D1** (guard replaced) was logged at 2026-09-26 00:29, before the resume queue started at 00:31. The old guard ran only the dry runs and the 6 baselines of 9/24.
- **D2** (retry wrapper) was logged at 01:03:24. Its first restart was at 01:08:24.
  - Cosmetic: the wrapper's header comment says "Added 2026-09-26 01:40", which disagrees with both of those times.
- **D3** (cap 20 to 400) was logged at 2026-09-27 10:19:16. The wrapper had given up at 06:33:20, and the next attempt ran at 10:31:57.
- **Could any deviation change a verdict?** Every restart was triggered by the guard's memory rule, never by a result. No record from a killed attempt is read (Q1).
  - Of the restarted LR runs, only TinyStories-1M's pick could plausibly move, and all four of its LRs are at chance.
  - lr_p160m_3e-04 (dev min .297) is far below the winner (.688).
  - The restarted seeds (ts33m s1/s2, p14m s3, p160m s2) each fail a cell robustly.
- The final plan was written after the interim `tables.txt` had been read, and it says so. The only operational choice it adds is reading "fails at every LR" as "no LR-search dev min >= 0.8", which is the natural reading of the pre-registered sentence. I found no choice that moves a verdict.

**Q11. Seed and split leaks: CONFIRMED, none (`a4_leaks.py`)**
- The dev draw, the eval items, the in-training probe draw and the crossed items share 0 prompts and 0 dialogues, pairwise.
- I drew 6,600 training examples per seed, seeds 0-3 (the trained runs used at most 6,406 draws, from run.json train_stats). They contain:
  - 0 exact eval/dev/probe prompts or dialogues;
  - 0 uses of the 12 eval key-statement templates (value replaced by `<v>`);
  - 0 eval questions.
- A planted eval key statement is caught on all 4 seeds (`a5_mutants.py`).

**Q12. Prior art: CONFIRMED WITH CAVEAT (not exhaustive)**
I ran 6 searches and read 8 paper pages. Like the analyst, I found no study that measures the size below which a short fine-tune stops teaching latest-value use after an in-dialogue correction, on Pythia, TinyStories or any ladder. Checks of the report's four citations:
- **arXiv 2608.18083** (Drozdz and Heilbron, v2 2026-09-29) is real. It is evaluation only, on Pythia 70M-12B and OLMo 2, and reports human-level entity tracking at 410M. The report had it from a snippet; it is now confirmed.
- **Kim and Schuster 2023** fine-tunes a single size, T5-base (from the PDF), so it gives no size floor.
- **Prakash et al. 2024** is a fine-tuning mechanism study. Its abstract page names no sizes.
- **PI-LLM (2506.08184)** is evaluation only.

Adjacent work the report does not cite:
- **"Scaling Laws for State Dynamics in LLMs" (2505.14892)** is evaluation only (GPT-2 XL, Pythia-1B on its abstract page).
- **"Chain and Causal Attention for Efficient Entity Tracking" (2410.05565)** proves that a transformer needs at least log2(n+1) layers for n state changes. With n <= 3, every E003 model has enough depth (4-12 layers).
- **Wu, Geiger and Millière (ICML 2025, 2505.20896)** train a 37.8M-parameter, 12-layer transformer from scratch (15 epochs over 450,000 programs) to above 99.9% on variable dereferencing with distractors.
  - That is far more training than E003's 6,400 examples, at a size inside the E003 range.
  - It supports the report's refusal to read the result as a capacity limit.

**Not independently re-checked**
- `final_e003.py`'s own 5 mutants (my own integrity code was mutation-tested instead).
- The controls, the old battery, uprobe and khard tables beyond what the verdict uses.
- The E002 audit's wording-shortcut finding (taken from its AUDIT.md).
- Whether nothing ran on the PC during the final analysis.
- The training streams after length filtering (no tokenizer was run; the 6,600 raw draws are a superset).

**What E003 shows**
- With E002's recipe (400 steps, effective batch 16, the searched LRs), no public base model with a body up to 85.1M (Pythia) or 28.3M (TinyStories) learns the pass-rule items. The result is 0/3 seeds for every model, and the failure is far from the bar.
- Pythia 31M-160M learn weekday items and the plain correction items almost perfectly. On colour items they take the most recent colour even when it belongs to the other object. The failure is a shortcut in one item family, not a uniform inability.
- It does not show a capacity limit. The LR search stopped at its lower edge for 5 models, only 400 steps were run, and from-scratch work trains comparable binding at 38M total parameters with far more data.

The audit scripts and their outputs are in `audit/`: `a1_cells.py` (cells, verdicts, picks; `a1_out.json`), `a2_integrity.py`, `a3_secondary.py`, `a4_leaks.py`, `a5_mutants.py` (needs a scratch dir argument) and `a6_rerun_analyzer.sh` (needs a scratch dir argument). Each has a matching `*_stdout.txt`.
