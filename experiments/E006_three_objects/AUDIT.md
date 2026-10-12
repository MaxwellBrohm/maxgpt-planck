# E006 audit (independent, 2026-10-02)

One adversarial auditor recomputed the final reading from the raw records (out/, transcripts/, big/, replay_ref/ and the item generators) with its own code in `audit/`. That code imports none of `analyze_e006*.py`, `rules_e006.py`, `passrule_e006.py`, `chatmeasures_e006.py` or `gen_grade.py`; it imports the item and template generators as data only. No model was loaded and nothing ran on the GPU. On the PC, three read-only scripts hashed files and searched the replay pool; nothing else was written there. Nothing outside `AUDIT.md` and `audit/` was edited. The record is `logs/readings/20261002-181132__final.txt`.

**Every verdict in the final reading reproduces from the raw records:**
- Q-H5 INCONCLUSIVE with no flag.
- Q-chat PROTECTS.
- Knowledge REDUCES THE COST.
- No COST or LOWER mark.
- Pass rule: C PARTIAL-X, P PARTIAL-X, G FAIL.
- Alias DATA GAP and stopping learned in all three arms.

No wrong run, wrong tag, wrong weights, missing run, leak or after-the-fact rule change was found. Two labels are fragile:
- **PROTECTS** has 2.5 turns of slack out of 745. It fails without seed 3, or without any one of 5 of the 53 conversations.
- **INCONCLUSIVE** depends on seed 1. Without it, Q-H5 would read NOT THE CAUSE.

The knowledge result is robust: it is unchanged on items whose answer never appears anywhere in the replay pool. Three sentences in the write-up claim more than the evidence supports (see "Claims to reword").

| # | Claim | Verdict |
|---|---|---|
| 1 | Provenance: copies, tags, hashes, completeness, seeds, e005w1 mapping | CONFIRMED |
| 2 | Q-device: C reproduces E005, rescoring changes nothing | CONFIRMED WITH CAVEAT |
| 3 | Q-H5: INCONCLUSIVE, no flag | CONFIRMED WITH CAVEAT |
| 4 | Q-chat: PROTECTS | CONFIRMED WITH CAVEAT |
| 5 | Knowledge: REDUCES THE COST | CONFIRMED |
| 6 | Cost: no COST or LOWER mark for P or G | CONFIRMED |
| 7 | Pass rule per arm, flips, LOO, any-seed variant | CONFIRMED |
| 8 | Alias reading and stopping yardstick | CONFIRMED |
| 9 | Blinded sample and its key | CONFIRMED WITH CAVEAT |
| 10 | Deviations F1-F4 and the process record | CONFIRMED WITH CAVEAT (four small corrections) |

**1. Provenance: CONFIRMED** (`a0_copies.py`, `a1_integrity.py`, `a10_c2_rerun.py`)
- **Copies.** The PC hashed all 1,221 files under `~/planck/e006_run/{out,logs,transcripts}`.
  - 1,219 are byte-identical on the Mac.
  - The other 2 differ as the report says. `index.txt` equals the PC file plus one appended line. `tables.txt` is the analyzer's rewrite and equals the new `final.txt`.
  - The 4 new reading files exist only on the Mac.
- **Completeness.** Every scored tag has every record file at full count:
  - base, C1-5, P1-5, G1-5, e005w2-5 and e005w1_r2: 27 files;
  - e004w1-5: 16 files;
  - C1t and C2t: 12 files.
  - That is 28 scored models and 21 chat probes (53 conversations each), all with guard exit 0. The only exception is e005w1 (exit -15, killed by gpu_mem).
- **Weights.** Each tag carries exactly one `weights_sha256`, equal to its run.json and guard.json.
  - The 17 trained files on the PC hash to the same values.
  - The 10 reference files on the PC hash to E005's (f846ac3d 4ef6230a aa5f99f1 1d330be9 131c8113) and E004's (91a3aab0 aee5a7fa efd68a20 8ad34021 1bb9f377).
  - There are 28 distinct weight sets; the only shared one is e005w1 with e005w1_r2.
  - Every chat transcript carries its tag's hash.
- **Run settings.** In run.json, arm, seed, 400 steps and lr 1.5e-4 match every C, P and G tag, with TF32 off. The twins have TF32 on.
  - C trained from the snapshot (`weights_in` null). Its losses track E005's Mac curve to within 1e-3 until step 141-313, and the TF32 twins drift by up to 0.13. So C is a genuine retrain, not a copy.
  - G's update digest equals C's on all 5 seeds and P's differs.
  - G used 1,600 distinct pool threads per seed, all from the kept pool.
- **e005w1 mapping (F1).** The killed attempt's 20 files are record-for-record equal to e005w1_r2's, tag aside. Its gen_e004 chat file holds only the first 286 records.
- **C2 power cut.** Of the cut attempt's out files, 25 complete ones are byte-identical to the rerun's. The 26th is gen_al plain, which was empty. The weights hash is 641b454d in both. Training and scoring on this GPU are deterministic.
- **BIG.** `items_e004.build(6006, f, 192)` rebuilds H5, C_noupd and C_twoslot exactly (sha256 566ef467...). Prompt hashes are disjoint across D4004, BIG, H5L and AL.

**2. Q-device: CONFIRMED WITH CAVEAT** (`a5_know_cost_device.py`, `a12_device_chat.py`)
- **The count reproduces.** 55 score intervals plus the 4 chat-probe intervals give 59, and 0 are clear of 0.
- **Scoring device.** e005w (CUDA) against E005 (Mac), same weights:
  - 0 LIK flips of 640 per seed and render, with a maximum score difference of 1.1e-4;
  - GEN replies identical, 640 of 640;
  - chat-probe turns identical, 149 of 149 per seed;
  - all 90 pass-rule counts identical.
- **Untouched model.** 0 flips against E004's records on every shared prompt.
- **Training device.** On BIG, C and e005w differ on 0-1 items of 192 per seed and family.
- Every chat-probe C - E005 value reproduces: TF -0.005, TF2 -0.007, LOOP +0.009, CHECKS -0.6.
- **Caveat 1: "matches on all 59 cells" is too strong.** The cells are not identical:
  - 29 of the 55 score cells differ from 0, by at most 0.013, and 21 of the 29 favour C.
  - 20 of 90 pass-rule counts differ by 1-2 items. One of them, s1 GEN C_noupd (51 to 52), crosses the 0.8 bar, which is why C qualifies on 4 seeds where E005 qualified on 3.
  - C1's chat transcript matches E005 s1's on only 51 of 149 turns; C2-C5 match on 135-149.
  - Most intervals are degenerate ([0,0], or with an endpoint at 0), so "about 3 by chance" is not a calibrated null here.
- **Caveat 2: the twin summary leaves one cell out.** The report gives only BIG H5 for the twins (+0.021, +0.005). C2t - C2 on BIG C_twoslot is +0.036 (CI +0.005 to +0.073), the one twin interval clear of 0. TF32 alone changed 16 and 23 BIG H5 items, against 26-48 for P - C.

**3. Q-H5 INCONCLUSIVE: CONFIRMED WITH CAVEAT** (`a3_qh5.py`, `a6_grader.py`, `a9_position.py`, `a8_purity.py`, `a11_loo.py`)
- **Numbers.** My bootstrap (10,000 resamples, rng 6006, seeds x items) reproduces every interval to 3 decimals; one H5L bound differs by 0.001.
  - P - C, BIG H5: LIK +0.002 (-0.089 to +0.105), GEN +0.013 (-0.086 to +0.121).
  - P - L4: LIK -0.127 (-0.191 to -0.066), GEN -0.122 (-0.195 to -0.059).
  - H5L: LIK +0.001, GEN -0.008.
  - L4 - C = 0.129, so there is no flag; dev = +0.001.
  - Pooled LIK: C .649, P .651, e004w .778, e005w .648.
- **Rule walk.** GAIN, RESTORED and PARTLY RESTORED need d >= +0.06, and the observed d is +0.002. WORSE needs a CI high < 0, and it is +0.105. NOT THE CAUSE needs CI high < 0.06 on both measures, and LIK's is .105. So the label is INCONCLUSIVE.
- **Inputs verified.**
  - My strict grader, written from E004 (c) clauses 1-7, agrees with the stored `strict` on all 57,344 GEN records (D4004 plain and chat, BIG, H5L; 28 tags).
  - The manipulation is real: my own GL code on the drawn streams gives P .4369/.0832/.2080 and C .6658/.0690/.2665 for GL2/GL1/GLX2, the notes' numbers to 4 decimals. ALIAS GL2 is .185 for P against .763 for C.
  - No word 5-gram within a text field is shared between BIG+H5L and the first 6,600 drawn examples per seed of C or P.
- **Mechanism numbers.** By introduction order, P has .900/.483/.497 against C's .918/.486/.446. Wrong picks equal to the last statement: 198 for P against 193 for C.
- **Caveat 1: the mechanism checks have almost no power.** P - C by introduction order is -0.018 (-0.074 to +0.032), -0.003 (-0.161 to +0.164) and +0.051 (-0.113 to +0.231). "Third-introduced items rose by 5 points" is noise-level, and the checks could not have detected the predicted pattern either way.
- **Caveat 2: the label depends on seed 1.** Per-seed P - C on LIK runs from -0.120 to +0.177 (SD .116). The interval half-width is about .097, against the pre-registered .054.
  - Leaving one seed out (descriptive): without s1 the label reads NOT THE CAUSE (LIK -0.042, CI -0.116 to +0.029). Without any other seed it stays INCONCLUSIVE.
- **What the data do show.** P does not restore E005's drop: it is credibly below E004's weights. A partial gain of up to about 10 points (LIK) or 12 (GEN) is not excluded.

**4. Q-chat PROTECTS: CONFIRMED WITH CAVEAT** (`a4_qchat.py`)
- **Measures.** My TF, TF2, LOOP and CHECKS are written from the notes' text, with `pools_train` templates only. They reproduce every model's TF, TF2, LOOP, CAP, EOS and CHECKS in the reading (21 models).
- **Bootstrap** (G - C):
  - TF -0.278 (-0.372 to -0.183);
  - TF2 -0.281;
  - LOOP -0.302 (CI high -0.117);
  - CHECKS +6.8 (-0.205 to +13.6).
- **Conditions.**
  - (a) TF_G 202 <= 204.5 = 0.5 x 409, TF2 202 <= 205.5, and both CIs are below 0.
  - (b) and (c) hold.
  - (d) holds: stopping learned on 5 of 5 seeds.
  - So the label is PROTECTS.
- **Two confounds are ruled out:**
  - **Replay's share of the clipped gradient (KNOWN RISK 5).** It is small: the update part's norm is .982 of the clipped total (median; 10th-90th percentile .966-.995).
  - **A probe leak.** Probe user turns share only generic 4-grams with the threads G used, such as "what is the capital of" and "tell me about the". For conversations with 10 or more 4-grams, at most 0.17 of them appear in any one thread.
  - Arm P, with the same template-heavy data, keeps C's TF (.552; P - C +0.003, CI -0.090 to +0.097), so the drop is specific to replay.
- **Caveat: the label is fragile.**
  - Three more template openings in G's 745 turns would make it SMALL EFFECT; for TF2 it would take four.
  - Per seed, G/C is .52/.51/.43/.53/.47, so the halving holds on 2 of 5 seeds.
  - Without s3 the ratio fails (170 > 167.5).
  - Dropping any one of K_time, O_add10, R_d5_multi, R_d6_name or R_d6_number breaks it.
  - CHECKS rose only nominally: the interval includes 0.

**5. Knowledge REDUCES THE COST: CONFIRMED** (`a5_know_cost_device.py`, `gen_pc_scripts.py`, `pc_exposure.out`)
- **The reading reproduces.**
  - K on the 425 unexposed items is +0.057 (+0.032 to +0.084).
  - On all 441 items it is +0.056, and on the 352 items left by the loose exposure rule +0.057.
  - Accuracy on 441: base .848, C .773, G .829.
  - The exposure file (16 strict items) is dated 2026-09-26 00:49, before any scored run.
- **A broader check of my own.**
  - 180 items have their gold anywhere in the 3,327-thread pool (sha256 94d0f651). On the other 261 items, K = +0.057 (+0.023 to +0.093).
  - Using each seed's 1,600 used threads, or requiring gold and entity in one thread, K stays between +0.051 and +0.059.
  - So the gain is not rehearsal of exposed facts.
- **G still loses against the untouched model** on s2 (McNemar p .0125) and s5 (.049), uncorrected.

**6. Cost: CONFIRMED** (`a5_know_cost_device.py`)
- No family (D4004 x 10, BIG x 3, H5L; LIK and GEN; plain and chat) carries a COST or LOWER mark for P or G.
- The most negative values:
  - P - C, D4004 C_noupd chat LIK: -0.037 (-0.109 to +0.034);
  - G - C, D4004 H6 chat LIK: -0.034 (-0.097 to +0.019).
- On BIG controls (n 192) in the plain render, every d lies within ±0.016. In the chat render the lowest is G - C C_noupd GEN at -0.021, and every CI includes 0.

**7. Pass rule: CONFIRMED** (`a2_passrule.py`, my own rule from E004 (c) and its step-5 clarifications)

| Model | Label | Seeds passing all but H5 | Fewest flips | Any-seed variant | Leave-one-out |
|---|---|---|---|---|---|
| C | PARTIAL-X (H5) | 4 (s1, s2, s3, s5) | 3 (s1 LIK C_noupd, s5 LIK C_twoslot) | PARTIAL-X | all PARTIAL-X |
| P | PARTIAL-X (H5) | 3 (s1, s3, s5) | 1 (s5 LIK C_twoslot) | FAIL (s2, s4) | FAIL without s1, s3 or s5 |
| G | FAIL | 2 (s1, s5) | 1 (s2 GEN C_noupd 51) | FAIL | all FAIL |
| e005w | PARTIAL-X (H5) | 3 (s2, s3, s5) | 2 (s5 LIK C_twoslot) | PARTIAL-X | E005's pattern |
| e004w | FAIL | 0 | none | FAIL | all FAIL |

- **E005's Mac records.** The same code gives e005w's cells exactly.
- **Control cells under 52/64.** G has 6 on seeds 2-4 (50-51 each), P has 4 (48-49) and C has 1.
- **Caveat (wording).** "One item flip would make G PARTIAL-X" holds under the governing majority rule. Under the any-seed variant G stays FAIL, because of the LIK control failures on s3 and s4.

**8. Alias and stopping: CONFIRMED** (`a2_passrule.py`)
- **Alias reading.** C, P, G and e005w read DATA GAP (HIGH 5). e004w reads REAL LIMIT (HIGH 0, LOW 4).
- **Stopping.** It is learned on 5 of 5 seeds for C, P, G and e005w: the largest |chat - plain| is .062. e004w's gaps run from -.906 to -.219.

**9. Blinded sample: CONFIRMED WITH CAVEAT** (`a7_blind.py`)
- All 40 turns equal the keyed transcript turn, and the blind and key files are byte-identical to the queue's.
- I classified the turns mechanically:

| Arm | Template first sentence | Loop | Other |
|---|---|---|---|
| C | 8 | 8 | 4 |
| G | 8 | 0 | 12 |

- That agrees in direction with the hand classification.
- **Caveat.** G also opens three general questions with a training frame: "That would be a noun.", "That would be oatmeal." and "You have it on the Pomodoro Technique.". So "G answers general questions in prose" holds for most G turns, not all. F4 (the label was seen first) is logged, and the reading order cannot be checked from the files.

**10. Process: CONFIRMED WITH CAVEAT**
- **What checks out:**
  - The pre-registration (notes lines 1-565) is unchanged since commit d8e1a76 (2026-09-26 05:31); only the POST-RUN LOG was appended.
  - The first queue run's 133 code files equal d8e1a76's.
  - Restarts changed only queue_e006.sh, test_queue_e006.sh, fake_py_e006.py and mutation_e006.py (the allocator cap, logged).
  - Today's code differs from the run's only in the four F1 files, and the diff is the tag map plus its test.
  - The e005w1 decision was committed in f58985e (2026-09-27 16:36), before the first Q-H5 reading (2026-09-28 10:41). It touches only e005w GEN lines, so it could not select a Q-H5 or Q-chat result.
- **Corrections:**
  - (i) The C2 entry says "27 out files". The manifest holds 26; its total of 34 files is right.
  - (ii) "Two jobs were rerun by hand": e005w1_r2 was. C2 was rerun by the queue after its files were moved by hand.
  - (iii) The superseded queue final printed e005w BIG H5 GEN pooled .672 "(seeds [1-5])", which is the mean of seeds 2-5. It also read e005w s1 stopping as False from a 286-record file. The "missing seed is NOT READ" rule was not applied to these descriptive lines. The record is complete, so nothing changes.
  - (iv) F4 is as logged.

**Claims to reword** (in the notebook text):
1. "matches E005's laptop results on all 59 compared cells": **NOT SUPPORTED** as worded. Write instead: "no compared cell differs by more than 1.3 points, no interval is clear of 0, and rescoring the same weights changes nothing".
2. "raised the probe's graded checks from 31 to 38": the averages are right, but the rise is **NOT SUPPORTED** as an effect (CI -0.2 to +13.6). Write "nominally".
3. "only third-introduced items rose, by 5 points": the numbers are right, but no introduction-order difference is distinguishable from 0. "The predicted mechanism did not show up" should become "the mechanism checks were too imprecise to show it either way".
4. "not supported as the cause": CONFIRMED WITH CAVEAT. Add that P stays credibly below E004, that a gain of up to about 10-12 points is not excluded, and that INCONCLUSIVE rather than NOT THE CAUSE rests on seed 1.
5. "PROTECTS, narrowly" and "halved": CONFIRMED WITH CAVEAT. Add that the halving holds pooled (2.5 turns of slack) and on 2 of 5 seeds, and that it fails without s3.
6. "G still stops cleanly in chat format": this was measured in the eval's chat render. In the probe, G hits the cap on 11-56 of 149 turns per seed (the untouched model on 68).
7. "first change ... without a measurable cost on updating": true for the cost marks. Add that G's pass rule fell from PARTIAL-X to FAIL on thin control margins, while BIG controls show no loss.

**Not independently re-checked**
- Anything requiring the tokenizer:
  - the kept streams, so the position gates and 5-gram checks were run on the drawn superset;
  - replay encoding, cutting and label spans;
  - the update and replay digests (their equality was checked, not their values).
- The replay pool build and its drop rules. Only its sha256 and the broad exposures were checked.
- Continuity sets beyond the Q-device intervals, AL GEN, the recency index, the U families by gold-last, and EXTRA REPORTS 3-6.
- Re-running the unchanged analyzer to reproduce the queue's final; the record was compared against my numbers instead.
- Training itself. No model was loaded, so "trained by the logged recipe" rests on run.json, the logs and the hashes.

**Scripts** (`audit/`; paths are derived from the file's location; `outputs_1.txt` and `outputs_2.txt` hold the full output of every script):
- `common.py`: loaders and both bootstraps.
- `a0`-`a12`: one script per check, as cited above.
- `pc_hashes.sh`: read-only. Its 1,327-line output is not kept; re-run it with pcwsl.sh to regenerate.
- `gen_pc_scripts.py`: writes the two read-only PC jobs, whose outputs are `pc_exposure.out` and `pc_probe.out` (home paths written as ~).

**Outside this audit:** the request to check that no one else has already done this work was not part of this task, and was not done.
