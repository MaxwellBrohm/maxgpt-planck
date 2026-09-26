# E005 audit (independent, 2026-09-25)

Four independent auditor agents recomputed the stored results from the raw outputs (out/, transcripts/, al/, the item files and the training streams), and an adversarial checker re-derived a further set of numbers from the same raw files with its own code (see "Checker spot-checks" at the end). None of them loaded a model or edited anything under experiments/. What was not re-checked is listed near the end.

**E005 audit: every stored reading reproduces (PARTIAL-X (H5), alias DATA GAP, stopping learned, knowledge cost the same as E004). But PARTIAL-X depends on seed 5 and a 2-item margin, H5 fell 10 points below E004, and in general chat more than half of E005's first sentences are training answer templates, although it passes more of the probe's graded checks than the untouched model.**

**Q1. Scores and the pass rule: CONFIRMED. The reading is PARTIAL-X (H5), and seed 5 decided it**
- **Records match the items.** I rebuilt eval draw 4004 with `items_e004.draw("eval")` (stdlib only) and matched all 640 LIK and 640 GEN plain records in each run (base, s1-s5). The match covered id, family, idx, plain-prompt sha1, gold, candidate set, vtype, k and latest_ref, and found 0 mismatches (`q1.py`).
- **Stored flags agree.**
  - LIK `right`, recomputed from the raw scores: 3,840/3,840 agree.
  - GEN strict, regraded with my own grader written from E004 (c) clauses 1-7: 3,840/3,840 agree, including the failing-clause lists.
  - All 120 cells in `tables.txt` reproduce exactly.
- **Runs are complete.** Every seed ran 400 steps at lr 1.5e-4, with 400 finite losses and train_s of 3,210-3,603 s. The s5 guard record shows exit 0 at 20:29:08, so the killed s5 chat probes never touched a scored record. The base records are E004's files (same sha256).
- Failing cells (k/64, LIK/GEN):

| Seed | H5 | Other failing cells |
|---|---|---|
| s1 | 39/37 (.609/.578) | C_noupd GEN 51 (.797) |
| s2 | 46/45 (.719/.703) | none |
| s3 | 42/44 (.656/.688) | none |
| s4 | 41/41 (.641/.641) | C_twoslot GEN 50 (.781) |
| s5 | 42/42 (.656/.656) | none |

- The untouched model fails all 18 cells (LIK 3-25, GEN 0-19 of 64).
- No cell in H1-H4, H6 or H7 fails; the lowest is 54/64 (H3 LIK, s1 and s3).
- Every thin pass (52-53/64) is on a control:
  - s1 C_noupd LIK 52;
  - s4 C_noupd LIK 52 and GEN 52, and C_twoslot LIK 52;
  - s5 C_twoslot LIK 53 and GEN 53.
- .797 and .781 now print correctly, so E004's "0.80" display bug is gone.
- **Reading.** I applied the order with my own code (`q1_rule.py`) and got the same result from `rules_e004.reading` on my cells.
  - PASS: no. 0 of 5 seeds pass.
  - PARTIAL-G: no. H5 LIK fails on all 5 seeds.
  - PARTIAL-X with X = {H5}: s2, s3 and s5 fail nothing else, which is 3 of 5 planned seeds. X is the smallest such set.
  - "3 seeds pass everything else" is accurate.
- **The "never partial" clause.** No seed has a LIK failure on H1, H2 or a control; the lowest such LIK cell is 52. The two control failures are GEN, so they only keep s1 and s4 out of the majority. The text does not say whether the clause applies to any single seed or only to the majority, but both readings give the same answer here.
- **The call is fragile:**
  - With s1-s4 complete and s5 counted as a failing planned seed, the reading was FAIL: only s2 and s3 qualify, 2 of 5. `rules_e004.reading` returns "FAIL: no PASS, PARTIAL-G or PARTIAL-X condition holds".
  - No file records that interim FAIL. `tables.txt` and `analyze_stdout.txt` were overwritten at 20:53, and `queue.txt` has no readings line.
  - Leave one seed out: dropping s1 or s4 keeps PARTIAL-X, while dropping s2, s3 or s5 gives FAIL.
  - Two more wrong items on s5 C_twoslot, at LIK or GEN, would make the reading FAIL.
  - If the clause applies to any single seed, one more wrong LIK item on s1 C_noupd, s4 C_noupd or s4 C_twoslot would also make it FAIL.
  - At n = 64 the 95% interval is about ±0.10, and all seeds share the same 64 items per family, so agreement across seeds says nothing about item sampling.
- **E005 minus E004**, mean over seeds (matches the stored `family_diff`), LIK/GEN:
  - H1: +.228/+.225
  - H2: +.222/+.209
  - H5: -.100/-.103
  - C_noupd: -.025/-.028
  - E004 passed both controls on all 5 seeds (thinnest 52/64: s1 C_twoslot LIK, s2 C_noupd GEN); E005 fails a control at GEN on 2.
  - For context, the E004 audit's post hoc counterfactual (E004 with the 21 alias items dropped from H1 and H2) read PARTIAL-X (H5) with every seed passing everything except H5. E005 reaches PARTIAL-X (H5) with 3 seeds.

**Q2. Alias reading: CONFIRMED, DATA GAP (HIGH on 5 of 5 seeds), one cosmetic correction**
- **Method** (`q2.py`, `q2b.py`). I redrew the eval items and selected the 42 H1+H2 items whose asked object's latest statement has ref "alias". I recomputed LIK from the raw scores and graded GEN with my own strict grader, which does not import `gen_grade`.
  - 0 mismatches against the stored gold, prompt hash, `right` and `strict`, on all 42 items in all 12 record sets (E005 base and s1-s5, E004 base and s1-s5).
  - The placement split is exactly EXTRA REPORT 1's: adjacent 9, filler-separated 14, B-separated 19.

| Model | Alias LIK | Alias GEN | Label |
|---|---|---|---|
| E005 s1 | .929 | .976 | HIGH |
| E005 s2 | .929 | .952 | HIGH |
| E005 s3 | .929 | .929 | HIGH |
| E005 s4 | .952 | .952 | HIGH |
| E005 s5 | .881 | .905 | HIGH |
| E004 s1-s5 | .214 .429 .357 .262 .238 | .190 .429 .524 .262 .262 | LOW x4, MID x1 |
| untouched | .048 | .119 | |

- **Labels.** E005 is DATA GAP (HIGH 5); E004 is REAL LIMIT (HIGH 0, LOW 4). Both match `tables.txt`. The notes give the 95% interval at n = 42 as about ±0.12 near 0.8; the lowest E005 seed (.881) is within that of the threshold.
- **E005 LIK by family and placement:**
  - H1: 17-19 of 21 (.81-.90).
  - H2: 20-21 of 21 (.95-1.00).
  - Adjacent 8-9/9, filler-separated 11-12/14, B-separated 18-19/19.
  - Separation no longer sinks the score (the E004 audit had .10-.30 when separated), but filler-separated items are still the weakest placement (.79-.86 LIK).
- **E004 rescored by the same code:** H1 LIK 4, 7, 7, 4, 2 of 21 and H2 LIK 5, 11, 8, 7, 8 of 21. That is exactly the E004 audit's .10-.33 and .24-.52.
- **Wrong picks** (LIK, pooled over seeds; both match the stored values):
  - E005: 16, of which 12 are another object's later statement and 4 are A's previous statement.
  - E004: 147, of which 139 are A's previous statement, 5 an earlier A value, 2 another object's earlier statement and 1 another object's later statement.
- **Correction (cosmetic):** `tables.txt` prints 17/21 as 0.809 and 4/21 as 0.191; the true values are .810 and .190. The cause is `round(x,4)` followed by `:.3f`. No label changes.
- **Caveat:** in the eval draw the gold is always the asked object's alias correction, so "latest statement with a name" also scores 42/42. DATA GAP on these items cannot tell linking apart from alias recency. Q3 does that.

**Q3. AL diagnostic: CONFIRMED WITH CORRECTIONS. The results are consistent with name-to-object linking and no tested shortcut explains them, but the pre-registered AL1+AL2+AL4 check is weaker than intended and the decisive evidence rests on a few items**
- **(a) The items were fixed before any model ran:**
  - Hash: `shasum -a 256 al/al_items.jsonl` gives e99b40d6...4a06d, equal to `al_items.sha256`. My own rebuild (`items_al.build()` plus `dumps()`) gives the same hash and the same 221,159 bytes.
  - Timeline:
    - The items file was created and last written at 09:37:13, so it was never rewritten.
    - QUEUE E005 START was at 09:41:42.
    - The first model process (dry run) started at 09:44:28.
    - The first AL output appeared at 20:31:47.
  - Code: `code_sha256_at_start.txt` (09:41:42) lists items_al, checks_al, purity_al, e005_al_ft_test and analyze_al at their current hashes. Only guard.py and test_al.py changed after that, plus the new queue_e005_al.sh.
  - Weights: the weights sha256 in each AL run.json equals the actual file, for E005 s1-s5 and E004 s1-s5 (10 distinct hashes).
- **(b) Cheap oracles** (`q3b.py`, working from the item text; my parse agrees with the `stmts` metadata on 64/64 items):

| Oracle | AL1 | AL2 | AL3 | AL4 | all | AL1+2+4 |
|---|---|---|---|---|---|---|
| alias recency | .00 | .00 | 1.00 | .00 | .25 | .00 |
| latest statement about any object | .00 | .00 | 1.00 | .62 | .41 | .21 |
| asked object's latest, ignoring aliases | 1.00 | .00 | .00 | 1.00 | .50 | .67 |
| topic tracker T | .75 | .38 | .88 | 1.00 | .75 | .71 |
| strict adjacency T1 | .94 | .44 | .50 | 1.00 | .72 | .79 |

- **Correction (a design weakness):** AL1+AL2+AL4 rules out alias recency and nothing more.
  - A name-free rule, HA, scores 60/64 overall and 48/48 on AL1+AL2+AL4 (`q3g.py`). HA says: with one alias correction, ignore it; with two, take the first alias correction after the asked object's original. It fails only the 4 AL3 other_obj items.
  - E004, which never saw alias training, scores .81-.90 on AL1+AL2+AL4 and .77-.83 LIK plain on all 64 items; the untouched model scores .22.
  - So the pre-registered check (report "resolved by alias recency" if AL1+AL2+AL4 LIK < 0.5) could not catch a name-free shortcut. It does not fire here: every E005 AL1, AL2 and AL4 cell is .88 or higher.
  - HA is post hoc (built after seeing the items), so it bounds what a name-free rule can reach; it is not evidence that any model used it.
- **(c) Scores** (`q3c.py`): 11 models x 2 renders x LIK/GEN x 7 groups all match `al_results.json` to 4 decimals. Per record, `right`, `strict`, candidates and prompt hashes all agree; the chat hashes were rebuilt from the SmolLM2 template.

| Model | LIK plain, all | LIK plain by cell | Chat GEN, all |
|---|---|---|---|
| E005 s1-s5 | .984 .969 .969 1.00 .969 | AL1-AL3 .94-1.00, AL4 .88-1.00 | .97-1.00 |
| E004 s1 | .828 | 1.00 / .56 / .75 / 1.00 | .03 |
| E004 s2-s5 | .77-.80 | AL2 .50-.69, AL3 .50-.62 | .03-.59 |

- **Correction to the brief:**
  - "E005 0.97-1.00 on all AL cells" holds for the pooled 64 items. Single cells (n = 16) go as low as .94 (AL1-AL3) and .88 (AL4).
  - E004's chat GEN is near 0 only on some seeds; it ranges .03-.59.
  - E004's low chat GEN is the known failure to stop (s1: 62/64 capped, lenient 53/64), not a linking failure.
- **(d) Leaks: none** (`q3e.py`):
  - No value appears in any question or prefix (0/64), and no name or surname does either (0/64).
  - Fillers carry no values and no names.
  - Alias corrections and their acknowledgements share no object word, and the acknowledgements contain no name (0/96).
  - The gold is never in the last user turn before the question (0/64). It is the last statement in 26/64 items and the uniquely most-mentioned value in 3/64.
  - Every candidate is a single token (64/64), so there is no length bias. The untouched model scores .22 LIK against a chance rate of .26.
- **Caveat on the stored "needs linking" subset:** these are the 14 items where both T and "latest statement with any name" are wrong. The subset has a position confound: "penultimate statement" is right on 14/14 of them. Penultimate scores only .31 overall, though, and on the items where it is wrong, E005 picks its answer in 0-5% of cases.
- **(e) Verdict: consistent with linking a name back to its object through the repeated name, and not explained by any shortcut tested.** It is not proven: the discriminating evidence rests on 2-15 items, and the same items are shared across seeds and renders, so the pooled counts below are not independent trials.
  - The 4 AL3 other_obj items that HA gets wrong: E005 4/4 on every seed, render and measure. E004 gets 1-2/4 on LIK plain.
  - The 15 items in which both objects carry an alias and the two aliases share a title (AL1 2, AL2 6, AL3 4, AL4 3), so matching the title alone cannot decide: E005 148/150 LIK and 150/150 GEN, pooled over 5 seeds x 2 renders. E004 91/150 LIK, 71/150 GEN.
  - Of those, the 10 AL2/AL3 items that hold two alias corrections (a structure absent from training): E005 100/100 LIK and 100/100 GEN, E004 41/100 LIK and 36/100 GEN (checker recount).
  - The best "title link, else T" hybrid scores 59/64. On the 5 items it fails, E005 is right 100/100 and E004 31/100.
  - The 2 items that all 8 non-positional name-free oracles get wrong: E005 2/2 on every seed and render.
  - Margins: E005's LIK plain margin on the 14 needs-linking items has a median of +7.2 to +8.7 nats by seed, with a minimum of +1.84.
  - Where E005 errs: 9 of its 12 LIK errors are AL4 picks of A's alias correction over the later head-noun correction. That is a small alias-recency residue.
  - Generalization: 0 of 33,000 drawn E005 training examples (first 6,600 per seed) contain two alias corrections. AL2 and AL3 are therefore structures the model never trained on.
  - Limits: AL tests only verbatim "Title Surname" repetition, with 2 objects at d = 10, under the four honorifics that training also used.

**Q4. Stopping: CONFIRMED (5 of 5 seeds within 0.10), with one correction to the E004 comparison**
- **Method** (`q4_stop.py`). I joined eval draw 4004 to the GEN records: 0 join mismatches on 6,400 records. My own strict grader disagrees with the stored `strict` on 0/640 records in every run and render (untouched, E005 s1-s5, E004 s1-s5).
- Chat minus plain strict GEN over the 9 pass families:

| Seed | Min | Max | Within 0.10 |
|---|---|---|---|
| s1 | -.047 | +.062 | yes |
| s2 | -.062 | +.031 | yes |
| s3 | -.031 | +.031 | yes |
| s4 | .000 | +.047 | yes |
| s5 | -.062 | +.047 | yes |

- The yardstick needs 3 of 5 seeds; all 5 meet it. Every per-family value equals `tables.txt` to 3 decimals, and first-line grading equals strict grading on every seed.
- E004 through the same grader: 0/5 seeds within 0.10 (gaps -.906 to -.219), matching the stored reading.
- **Replies:**
  - Every E005 seed stops at end of turn on 640/640 replies, with 0 capped at 48 and at most 9 new tokens.
  - A scan of all 3,200 E005 chat replies found 0 multi-line replies, 0 role or template words (user, assistant, system, <|im_), 0 empty replies and 0 replies with more than one sentence. Each reply is 3-7 words.
  - Every wrong reply is a wrong value, except 2 on s4 that name no value. There are no format failures.
- **30 random replies read** (`q4_read.py`): each is one clean sentence in a training answer template, for example "February, from what you told me." and "Go with yellow." (a wrong value).
- **Correction to the E005 notes' "the model never stops" (and the E004 audit's shorthand "the model does not stop"):** E004 did sometimes stop in the eval chat render, on 16, 204, 30, 94 and 341 of 640 replies (s1-s5). It never stopped only in the chat probe (0-1 of 149 turns). Its capped counts match the E004 audit, which listed them: 624, 436, 610, 546, 299.
- **Scope:** this is stopping inside the trained render, on the eval's own question format. Q9 covers general chat.

**Q5. H5 (three objects): CONFIRMED WITH CORRECTIONS. H5 is worse than in E004, and the error is recency toward the dialogue's last statement**
- **Recomputation** (`q5.py`): the per-seed splits, form cells and pooled wrong-pick classes equal `tables.txt` and `results.json` for both E005 and E004. My `after_ind` flag matches `meta.after_ind` on 64/64 items; 32 items have another object's indirect correction after the asked object's latest statement.

| Seed | E005 LIK | E004 LIK | E005 GEN | E004 GEN |
|---|---|---|---|---|
| s1 | 39 | 45 | 37 | 45 |
| s2 | 46 | 46 | 45 | 45 |
| s3 | 42 | 52 | 44 | 53 |
| s4 | 41 | 48 | 41 | 47 |
| s5 | 42 | 51 | 42 | 52 |

- **E005 against E004:**
  - E005 is lower on 4 of 5 paired seeds and tied on s2.
  - Pooled: LIK .656 against .756, GEN .653 against .756.
  - 30 items are right on all 5 seeds (E004: 36), and 12 are wrong on all 5 (E004: 5).
  - E004's .756 also failed the 0.8 bar; E004 passed H5 only on s3.
- **Pre-registered splits** (LIK, pooled over seeds):

| Split | E005 | E004 |
|---|---|---|
| another object's indirect correction after the asked latest | .556 | .669 |
| no such correction | .756 | .844 |
| asked latest is ellipsis | .568 | .642 |
| asked latest is pronoun | .800 | .908 |
| asked latest is head noun | .822 | .889 |
| asked latest is full phrase | .686 | .771 |
| asked object never corrected | .537 | .688 |

- The item counts behind these splits are small (32/32 items for the first two rows; 19, 13, 9, 7 and 16 items by form), each pooled over 5 seeds.
- The worst combined cells are ellipsis with an indirect correction after it (.467) and never corrected with one after it (.418). The GEN splits, stored only in `results.json`, track LIK within .05.
- **Wrong picks:**

| Picked value comes from | E005 LIK | E005 GEN | E004 LIK | E004 GEN |
|---|---|---|---|---|
| another object, after the asked latest | 83 | 83 | 49 | 51 |
| the asked object's previous value | 20 | 21 | 26 | 24 |
| another object, before the asked latest | 7 | 7 | 3 | 3 |
| total wrong | 110 | 111 | 78 | 78 |

- **What the wrong picks look like:**
  - No wrong pick is an older asked-object value, a tie or an out-of-context value. Every GEN miss names a wrong value.
  - Of the 83 "another object, after" picks:
    - 74 are that object's current value;
    - 64 are the last statement in the whole dialogue;
    - 47 name the other object outright (30 full phrase, 17 head noun), and 36 are pronoun or ellipsis corrections.
  - The dialogue's last statement is never the gold in H5 (0 of 320 item-seeds). Yet wrong picks equal it 64/110 times in E005, against 33/78 in E004.
- **Correction to the E004 audit's "position does not matter (.71-.79)":** that holds by object slot (E004 .76/.71/.79, E005 .63/.65/.70). It does not hold by the order in which objects are introduced (`q5c.py`, `q5d.py`; 27, 20 and 17 items):

| Asked object introduced | E005 | E004 |
|---|---|---|
| first (27 items) | .88 | .88 |
| second (20 items) | .51 | .64 |
| third (17 items) | .47 | .69 |

- Worst cell: a second-introduced object that is never corrected scores 3/25 in E005 against 9/25 in E004 (5 items x 5 seeds).
- When the asked object is introduced first, wrong picks are almost all its own stale value. When it is introduced second or third, they are mostly another object's latest value, usually the final statement (36 of 49 and 25 of 45).
- **Example, H5 item 1:** the karate grading is stated once, on Monday. Later the staff briefing gets two ellipsis corrections, ending "Let's go with Saturday." Three of five E005 seeds answer Saturday. One of five E004 seeds is also wrong, and it answers Sunday.
- **A candidate source (an association, not a demonstrated cause).** `q5_train.py` looks at 2-object training examples where the asked object is introduced second. It counts how often the asked object's latest statement is also the dialogue's last one:
  - E005: .656, .668, .671 (s1-s3).
  - E004: .544, .558, .567.
  - These are the first 6,404 drawn (not kept) examples per seed. Seeds 4-5 were not run. The three data changes were made together, so the H5 drop could equally come from the smaller E004 block (70%), the IND block or the chat render.

**Q6. Knowledge: CONFIRMED WITH CORRECTIONS (wording). The cost is -7.5 points, the same as E004 within noise**
- **Method** (`q6_know.py`). "Right" is recomputed from the raw scores: 0 mismatches.
  - The base records in E005 out/ are byte-identical to E004's (kbig, khard, old).
  - Untouched model: kbig .8481, khard40 .775, K_closedbook60 .8667.
- kbig441, change against the untouched model (4,000-resample bootstrap, exact McNemar):

| Seed | E005 acc / change | 95% CI | McNemar p | E004 acc / change |
|---|---|---|---|---|
| s1 | .7755 / -.073 | -.104 to -.041 | 1.4e-5 | .7528 / -.095 |
| s2 | .7642 / -.084 | -.116 to -.054 | 2.4e-7 | .7710 / -.077 |
| s3 | .7778 / -.070 | -.100 to -.041 | 5.5e-6 | .7732 / -.075 |
| s4 | .7574 / -.091 | -.125 to -.059 | 9.0e-8 | .7574 / -.091 |
| s5 | .7891 / -.059 | -.091 to -.030 | 1.6e-4 | .7619 / -.086 |
| mean | -.0753 | | | -.0848 |

- The gained/lost counts and p-values equal `tables.txt`. The CIs match to about ±.002; the stored run used 2,000 resamples.
- **E005 against E004 on kbig** (per-item mean over seeds, paired bootstrap): .7728 against .7633, d = +.0095, CI -.0023 to +.0213. The stored value in `results.json` is d = +0.0095 with CI -0.0018 to 0.0222 (`tables.txt` prints it rounded as +0.009), so the two agree within bootstrap noise.
- **By category:** the cost pattern is the same for E005 and E004 (currency -.17/-.18, symbol -.10/-.13, state capitals -.08/-.11). E005 is nominally 1-3 points better on 6 of 7 categories; no per-category test was run.
- **Small sets:** no seed of either experiment is significant, except E005 s4 on K_closedbook60 (p = .0215, uncorrected for the 30 seed-by-set tests).
  - khard40: E005 -.040 against E004 -.070 (d +.030, CI -.005 to +.075).
  - K_closedbook60: E005 -.070 against E004 -.053 (d -.017, CI -.043 to +.010).
  - The two directions disagree, which is consistent with no difference.
- **Suggested wording:** "E005's closed-book cost (-7.5 points on kbig441, 5.9-9.1 per seed) is statistically the same as E004's (-8.5, 7.5-9.5). The difference is +0.9 points, CI -0.2 to +2.1."

**Q7. Purity and leakage: CONFIRMED WITH CORRECTIONS. Every zero holds on every trained example**
- **Kept streams rebuilt independently** (`kept.py`). The script runs `train_e005.stream` for data only, writes the SmolLM2 chat template by hand and applies the 768-token rejection with `tokenizers` on the HF `tokenizer.json` (no torch, no transformers). The result matches the training-time `run.json` exactly:

| Seed | Drawn | Rejected | Kept e004 / alias / indirect | Plain / chat | Mean len |
|---|---|---|---|---|---|
| 1 | 6506 | 102 | 4531 / 1260 / 613 | 3212 / 3188 | 434.7 |
| 2 | 6516 | 112 | 4467 / 1284 / 653 | 3269 / 3131 | 431.5 |
| 3 | 6505 | 101 | 4500 / 1270 / 634 | 3259 / 3141 | 431.3 |
| 4 | 6511 | 107 | 4499 / 1266 / 639 | 3236 / 3164 | 435.3 |
| 5 | 6506 | 102 | 4520 / 1233 / 651 | 3254 / 3146 | 430.0 |

- Kept = 6,404 per seed; the plain/chat columns (6,400) are the trained examples, the other 4 being the loss self-check batch. In every seed the E004 block equals the start of `train_e004.stream(s)`, in order (4,521-4,570 draws). The max lengths match too.
- **Purity** (`purity.py`, `purity2.py`). I checked 32,020 kept examples (6,313 alias-correction statements) against 1,120 E004 eval/dev/probe items, and every axis is 0:
  - eval surnames, in any case;
  - eval title+surname pairs, and any title with an eval surname;
  - eval joins used as joins;
  - 3-grams shared with eval alias templates, and 3-grams containing `<a>`;
  - raw word 5-grams;
  - titles outside the example's declared alias;
  - eval markers, held-out objects, sport values, digits and non-training fillers;
  - more than 2 objects, d > 10 or k > 3.
- **Pool level:**
  - 0 of 18 training templates and 12 training joins share a 3-gram with an eval template.
  - Training joins contain none of with/from/run/by.
  - Training and eval surnames do not overlap, and no role title appears in any eval text.
- **Mutation tests** (`purity_mut.py`, `purity2_mut.py`). One injected leak per axis, and every axis flagged it. Examples: "Ms. Petrov", "Captain Novak", "with Dr. X", the rejected candidate "{a} bumped me to {v}" (caught on the 3-gram "me to <v>"), an eval 5-gram, a digit, "tennis", and a new filler.
- **Oracles** (`oracles.py`, my own O1-O8, O5-run and IDEAL):
  - O1 .200-.211, O2 .396-.408, max rule (O7) .622-.632, and IDEAL 1.000 on every seed.
  - All values match `logs/validate_e005.stdout` to 3 decimals, and the gate passes on every seed.
  - The gold appears in the answer in 32,020/32,020 examples, and a single wrong-gold mutant fails the IDEAL gate.
  - I did not re-implement X1-X11. The project's max over O and X (.622-.632) equals my O7, so no X rule exceeded it.
- **Correction 1 (coverage).** `purity_e005.py`, `ngram_overlap_e005.py` and the H7 5-gram check sampled the first 6,404 drawn examples. The kept streams run to draw 6,505-6,516, which leaves 100/112/96/107/101 = 516 trained examples outside the aggregate 5-gram checks. `validate_e005` re-ran the per-example axes on kept examples, but not the 5-gram checks. My 5-gram check over every kept example finds 0.
- **Correction 2.** Seed 4's largest drawn-vs-kept shift is 0.754 points. The log's .0076 comes from rounding the shares before subtracting. The shift is always in the render: rejected examples are mostly chat-rendered, which leaves a kept chat share of .489-.498.

**Q8. Process and deviations: CONFIRMED WITH CORRECTIONS. No reading changes. There was 1 pause, not 56, and the guard.py patch is not logged in E005**
- **Logged deviations.** notes.txt logs:
  - Deviations 1-3: FAIL sub-readings printed as facts; AL left out of `queue_e005.sh`; 20 new AL surnames plus a chat render.
  - The chat probe after each seed instead of after all seeds.
  - AL run by a separate queue.
  - `test_al.py` changed at 09:41:44, 2 s after the start hash list.
  - `queue_e005_al.sh` created at 09:43:13.
  - All of these predate the first model run at 09:44:28.
- **E004 queue.** It was stopped by hand at 09:32:15 ("paused by Claude to run E005 first", E004 `queue.txt` line 57); the notes' step 5 entry mentions that line. That STOPPED line is what satisfied E005's step-0 wait, and E004 resumed at 20:51:52 through a resume queue written to E004's logs. No guarded jobs overlapped in time: I checked every E004 and E005 guard.json from Sep 25.
- **Correction: seed 1 was paused once, not 56 times.**
  - `ft_135m_s1.guard.json` records n_pauses 1: SIGSTOP at 10:44:16 (lid closed on battery) and SIGCONT at 14:34:22. The log holds 55 "PAUSED" status lines, which is probably where "56" came from.
  - Training had already finished (step 399 at 3,174 s), so the pause fell during e004/plain LIK scoring.
  - The guard counts paused_s = 1,541 against a wall gap of 13,806 s; the Mac was asleep for the rest.
  - The records are complete: s1 has 640/640 records, 0 non-finite and 555 right (the other seeds have 557-570). A SIGSTOP/SIGCONT pause does not change deterministic scoring, but there is no rerun of s1 to compare against.
- **Seed 5 chat probe:**
  - Attempt 1 (20:29:09-20:29:40) was killed at swap +1,302 MB with 31% free. The queue then stopped, so the post-s5 analyze, the readings line and "QUEUE E005 DONE" never ran.
  - Attempt 2 (20:46:40-20:47:27) was killed at swap +3,162 MB.
  - Attempt 3 (20:47:59-20:50:49) exited 0.
  - `analyze_e005` was then re-run by hand at 20:53. This is recorded only in manual `queue.txt` lines and the guard logs.
- **guard.py was patched at 20:51:15, and E005 never logged it.**
  - The swap kill now needs 2 consecutive low-free samples (`swap_low_streak >= 2`).
  - The same patch went into the E002, E004 and E005 copies, and it is logged only in E004 `queue.txt` (20:55:11). E004's copy now has sha 30f97995..., against e142a74e... at the E005 start.
  - This breaks the notes' statements that copied files are never edited and that E001-E004 are read-only.
  - The step-1 list now matches 63/64, so check 1 of `reuse_base_e005` would fail if re-run.
  - The patch came 26 s after the last E005 model job ended (the s5 chat re-run, 20:50:49), so it does not affect any E005 result.
- **Code written after the pre-registration.** 34 new files were written 07:58-09:43, all planned in steps 2-5 and all before any model run.
  - The git notes diff from 5c55c2f (08:52) to 4ce1a88 (09:45) is purely additive (0 removed lines).
  - Every code file except guard.py is byte-identical to HEAD.
  - The 07:53-08:52 window cannot be checked with git, but no E005 output existed then.
- **notes.txt has no post-run entry** (last write 09:45:09). The pause, the kills, the re-run and the patch are all missing from it.
- **EXTRA REPORT 6 is incomplete.** The pre-registered capped-turn share and training-template share were never added to `tables.txt` or `results.json`. Q9 computes them.
- **Base reuse** was pre-registered, not a deviation. 17/17 files hash equal across source, copy and `logs/base_reuse_sha256.txt`, and the snapshot `model.safetensors` is still 5af571cb.
- **The s5 probe ran the same way as s1-s4:**
  - the same command pattern;
  - run_chat.py (2b49b3a1) and battery.py (6f4e6d55) unchanged since Sep 24 07:34, and no venv package changed since Aug 19;
  - the run ended before the guard patch;
  - the same 53 conversations in the same id order.
- **Weights are traceable.** All 5 E005 weight files exist, and their sha256 equals the AL-recorded hash (f846ac3d, 4ef6230a, aa5f99f1, 1d330be9, 131c8113); E004 s1-s5 match too.
  - The main out/ records carry no weights hash.
  - They are traceable by process instead: `e005_ft_test` evaluates and then saves in the same process with no update in between. Each seed's 16 out files are timestamped 1-2 s before its weights save.

**Q9. Chat probe collateral: stored per-category numbers CONFIRMED, but they hide heavy training-template intrusion. The s5 re-run is consistent**
- **What was checked** (`q9_probe.py`, `q9_more.py`, `q9_read.py`). The stored per-category line reproduces exactly (0 mismatches, all 6 models).
- **Why that line misleads.** It shows E005 above the untouched model on most categories, but it leaves out the two measures EXTRA REPORT 6 pre-registered.
  - There are 149 assistant turns per model, and the probe's cap is 120/160/200 tokens per turn.
  - The stored leak flags (`leaked_template`, `invented_user_turn`) are 0 on every model, so they do not see template intrusion.

| Model | Capped (of 149) | Capped that are loops | Stopped at end of turn | Template share strict / loose | First sentence is a template | Same first sentence as previous turn (of 96) | Checks passed (of 73) |
|---|---|---|---|---|---|---|---|
| untouched | 68 (.46) | 2 | 81 | .01 / .03 | .01 | 5 | 21 |
| E005 s1 | 87 (.58) | 86 | 62 | .52 / .66 | .58 | 34 | 30 |
| E005 s2 | 25 (.17) | 25 | 124 | .30 / .60 | .57 | 29 | 33 |
| E005 s3 | 62 (.42) | 60 | 87 | .38 / .55 | .50 | 20 | 33 |
| E005 s4 | 18 (.12) | 16 | 131 | .42 / .62 | .59 | 36 | 32 |
| E005 s5 | 59 (.40) | 53 | 90 | .40 / .59 | .54 | 31 | 29 |
| E004 s1-s5 | 148-149 (.99-1.00) | 121-146 | 0-1 | .39-.44 / .57-.70 | .35-.51 | 8-35 | 24-33 |

- "Strict" means a training answer template filled with a training-pool value; "loose" means filled with any word. Checks not graded because a turn was truncated count as not passed (1-6 per model).
- **The measures disagree, so "better" or "worse" depends on which one is read.**
  - On the probe's own graded checks, E005 passes more than the untouched model (29-33 of 73 against 21).
  - On capped-turn share, E005 is mixed: s2 and s4 are capped less often than the untouched model (.17, .12 against .46), s1 more (.58), s3 and s5 about the same.
  - On template share, E005 is far worse (first sentence a template .50-.59 against .01).
  - The untouched model's capped turns are long helpful answers: 2 of 68 are loops. E005's capped turns are loops: 86 of 87 on s1. When E005 does stop, its replies average 9-17 tokens, against 56 for the untouched model.
- **How chat got worse** (10 conversations read per model):
  - **Template answers that ignore the question.** "How do vaccines work?" gets "The latest plan says June." (s4). "What is 15 times 4?" gets "The answer is 4." on s1, s2 and s4; the untouched model found 60.
  - **The same reply every turn.** On L_chitchat, s2 answers "You ended up with 2023." five times running; s4 answers "The latest plan says it's in the latest plan." six times.
  - **Runaways.** s1, s3 and s5 loop "The latest plan says..." to the cap.
- **What improved:** "What about Italy?" gets "The capital of Italy is Rome." on all 5 seeds; the untouched model failed this ellipsis check.
- **Against E004:**
  - E005 stops far more often (62-131 turns against 0-1).
  - But its first sentences are templates more often (.50-.59 against .35-.51), and "latest plan" appears in 14-27% of turns against 5-13%.
  - Some E004 first sentences were on topic ("Earthquakes are caused by the movement of tectonic plates") where E005 gives a template.
- **The s5 re-run is consistent:**
  - The user-turn prompt set hashes to bc7133961bd46d7b in all 11 transcript files.
  - The s5 log matches the transcript on 53/53 conversations.
  - Both killed attempts reproduce the final run's first 10 conversations exactly.
  - s5 sits inside the s1-s4 range on capped share (.40) and template share (.40), and is 1 below it on checks passed (29 against 30-33).
  - The untouched model's transcript is byte-identical to E004's (sha256 ffa5ee69...).

**Checker spot-checks (own stdlib code, from raw files)**
Scripts `sc1.py`-`sc5.py` in `scratchpad/e005_audit/checker/`, each run with `python3 -B`; `torch` and `transformers` were asserted absent from `sys.modules`.
1. **All 90 E005 pass-rule cells and the reading** (`sc1.py`, LIK recomputed from `scores`, GEN from stored `strict`): every cell equals Q1's table; seeds passing all but H5: 3 of 5; with s1-s4 only: 2. E004: 0, and E004 passes both controls on all 5 seeds.
2. **Alias items** (`sc1.py`, `latest_ref == "alias"` in H1+H2): n = 42; E005 LIK 39, 39, 39, 40, 37 of 42 (s5 = .881), GEN 41, 40, 39, 40, 38; E004 LIK 9, 18, 15, 11, 10; untouched 2 and 5.
3. **kbig441** (`sc1.py`): E005 per-seed accuracy and gained/lost equal Q6; E005 against E004 d = +.0095, CI -.0023 to +.0213 (my own 4,000-resample bootstrap).
4. **Seed 1 pause** (guard.json and guard.log): n_pauses 1, paused_s 1,541, elapsed_s 18,595; 55 "PAUSED" lines; one SIGSTOP (10:44:16) and one SIGCONT (14:34:22).
5. **Chat probe** (`sc2.py`, transcripts): capped turns 68 (untouched), 87/25/62/18/59 (E005), 148-149 (E004); stopped 81, 62/124/87/131/90, 0-1; checks passed 21 and 30/33/33/32/29; "latest plan" in 14-27% of E005 turns against 5-13% of E004's.
6. **Eval chat stopping** (`sc2.py`): E005 stop = eos on 640/640 for every seed, 0 capped, max 9 new tokens; E004 capped 624/436/610/546/299.
7. **AL** (`sc3.py`): E005 LIK plain 63, 62, 62, 64, 62 of 64; cells AL1-AL3 15-16/16, AL4 14-16/16; chat GEN 62-64; E004 LIK plain 53/51/49/49/51, chat GEN 2/27/2/14/38. Shared-title items: 15 (defined from `aliases_all`), E005 148/150 LIK and 150/150 GEN, E004 91/150 and 71/150. This locates the Q3 (e) set: it is not "two alias corrections" but "both objects aliased"; the 10 two-correction items are listed separately in Q3 (e).
8. **H5 by introduction order** (`sc4.py`, from `items_e004.draw("eval")`): 27/20/17 items; E005 119/135 = .88, 51/100 = .51, 40/85 = .47; E004 .88, .64, .69. Last statement is the gold in 0 of 64 items; wrong picks equal the last statement 64/110 (E005) and 33/78 (E004).
9. **Control wrong picks** (`sc5.py`): C_noupd E005 43 wrong, all another object's later statement (E004 35, all the same class); C_twoslot E005 48 wrong, 29 another object's later statement and 19 the asked object's previous value (E004 50: 10 and 40). Equal to the stored `tables.txt` classes.
10. **Hashes:** guard.py in E005 and E004 is 30f97995... (mtime 20:51:15) against e142a74e... in `code_sha256_at_start.txt`; the base transcript is ffa5ee69... in both; `al_items.jsonl` equals `al_items.sha256`.

**Not independently re-checked**
- EXTRA REPORT 2's reference-form and "ellipsis with the other object following" splits outside the alias items.
- The X1-X11 oracle values.
- The E004-style answer-format and continuity stats in EXTRA REPORT 6.
- The training-position statistic behind Q5 on seeds 4-5, and on the kept (rather than drawn) streams.

Where the auditors' work overlapped (the s5 re-run times, the EXTRA REPORT 6 gap, the guard.py patch, the E004 H5 counts), their numbers agree. One open point was resolved: the alias auditor flagged that the guard.py patch might affect the s5 chat re-run; the timestamps show the re-run ended 26 s before the patch. The brief's "56 pauses" is wrong (1 pause).

**What E005 shows**
- **On these items, the alias failure was a data gap.** With alias corrections in training (new surnames, joins and templates under the same four titles), 135M answers alias items correctly.
  - Alias items score .88-.95 LIK on every seed, against .21-.43 in E004.
  - On AL, E005 scores .97-1.00 pooled (LIK plain and chat GEN). That number alone is weak evidence of linking: E004, with no alias training, already scores .77-.83 LIK plain on AL, and a post hoc name-free rule scores 60/64.
  - The evidence for linking through the name comes from a few items the name-free rules cannot solve: the 4 AL3 other_obj items (E005 4/4 on every seed and render) and the 10 AL2/AL3 items with two alias corrections under the same title, a structure absent from training (E005 100/100 LIK, 5 seeds x 2 renders, against 41/100 for E004).
- **Stopping was a data gap too, in the trained render.** With end-of-turn training, every seed stops after one short sentence in the eval's chat render (640/640), and chat GEN tracks plain GEN within .062.
- **Three objects is still a limit at 135M with this recipe, and it got worse.** H5 is .656 LIK against E004's .756 (E004 also failed H5 on 4 of 5 seeds).
  - The model tracks the object introduced first (.88).
  - For a later object it tends to answer with the dialogue's most recent value, even when that statement names another object.
  - The same shift shows in the controls: C_twoslot wrong picks moved from the asked object's previous value (40 of 50 in E004) to another object's later statement (29 of 48 in E005).
  - In 2-object training examples where the asked object is introduced second, E005's streams put the asked object's latest statement last more often than E004's (.66-.67 against .54-.57, seeds 1-3, drawn examples). That is a candidate cause. It has not been tested, and the other two data changes are equally untested candidates.
- **The pass rule reads PARTIAL-X (H5), barely.** It needs 3 seeds and gets exactly 3. After s1-s4, with s5 counted as failing, it read FAIL, and two more wrong items on s5 C_twoslot would make it FAIL again. The controls slipped a little: C_noupd is -.025 LIK against E004, and s1 and s4 fail a control at GEN.
- **Cost, closed-book knowledge:** -7.5 points on kbig441 (5.9-9.1 per seed), the same as E004's -8.5 within noise (difference +0.9, CI -0.2 to +2.1).
- **Cost, general chat:** training answer templates dominate. E005 no longer runs away on nearly every turn the way E004 did (12-58% of turns capped, against 99-100%), and it passes more of the probe's graded checks than the untouched model (29-33 of 73 against 21). But more than half its first sentences are training answer templates (against 1% for the untouched model), it answers unrelated questions with them, and most of its capped turns are loops.
- **What it does not show:**
  - linking beyond verbatim "Title Surname" repetition with 2 objects at d = 10;
  - linking under a title never seen in training;
  - which of the three data changes (alias, IND, chat render, plus the smaller E004 block) caused which effect, since they were made together;
  - that H5 is a capacity limit rather than a data effect;
  - anything about other model sizes: only SmolLM2-135M-Instruct was trained, and the tiny ladder was not run;
  - anything beyond one eval draw (n = 64 per family, about ±0.10), shared by every seed.
  - The eval alias items alone cannot separate linking from alias recency; AL is what does.

**Open questions for the next experiment**
- **Is the H5 recency caused by the training position statistic?** Test it with one change: the E005 recipe with the asked object's latest statement balanced across positions (last statement at or below E004's rate), 5 seeds, and the same eval draw and rule.
  - If H5 returns to about .76 (E004's level, still below the 0.8 bar), the drop came from that statistic.
  - If H5 stays near .65, the position statistic is not the cause; the drop could still come from the smaller E004 block, the IND block or the chat render. Neither outcome alone shows that three objects is a capacity limit at 135M.
- **Why did the controls slip?** C_noupd GEN failed on s1 and C_twoslot GEN on s4. The wrong picks point to the same recency pull as H5 (C_noupd: 43 of 43 wrong picks are another object's later statement, against 35 of 35 in E004; C_twoslot: 29 of 48 against 10 of 50). The smaller E004 block (70%) is the other candidate.
- **Can general chat be protected?** For example, a share of general instruct data, or the untouched model's own replies, in the stream. The goal is to keep stopping and linking while bringing template share and loops back toward the untouched model. The analyzer should compute the pre-registered capped-turn and template shares itself.
- **How far does linking go?** Surname only, title only, a person pronoun ("she moved it"), an unseen title, 3 objects and d = 20, on more than a handful of discriminating items.
- **The tiny ladder.** DATA GAP triggers the notes' condition: a ladder with this recipe needs its own pre-registration.
- **Margin.** With n = 64 per family (±0.10), a reading decided by 2 items argues for a larger eval draw or more seeds before any headline claim.
- **Process fixes for the next run:**
  - log in notes the guard.py patch, the s5 kills and re-run, and the hand re-run of analyze;
  - have the analyzer write each interim reading to a file that is not overwritten;
  - run the 5-gram purity checks on the kept streams, not the first N draws;
  - store the weights sha256 in every out record.

The scratch scripts are under `AUDIT_DIR/`, in `q1q5/` (rule, H5), `q2q3/` (alias, AL), `q469/` (stopping, knowledge, chat probe), `q7q8/` (purity, process) and `checker/` (spot-checks). Nothing in E005 or E004 was modified by the audit, and no model was loaded.
