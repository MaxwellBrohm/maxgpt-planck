# E3 Part 1 audit (independent, 2026-10-02)

One adversarial auditor re-derived every headline number of the E3 Part 1 analysis from the raw records in `out/` with its own code (`audit/a1_recompute.py`, own t and chi-square quantiles by numeric integration, cross-checked with scipy 1.13.1), checked the copied records against the PC originals (read-only), and read every verdict against the pre-registration in `notes.txt` lines 1-174 (committed at b04d104, 2026-09-26, before any E3 run). No model was loaded, the GPU was not touched, gpu.lock was not taken, and nothing outside `AUDIT.md` and `audit/` was written. Nothing was committed or pushed.

**E3 audit: every reported number reproduces exactly from the raw records (377 shared statistics, largest difference 4e-14), the copied records equal the PC originals, and the one registered verdict (E2 re-check: CONFIRM, the 5M LR stays A = 3e-3, r 1) holds. Corrections: D2 says no CUDA parity run of the default path is recorded, but one is; D3 says the training manifest was not checked, but preflight checked it; "A vs B RESOLVED" is a descriptive reading, not a registered rule; the 78% replay share and the "below 2" CRN ratio are point estimates whose 95% intervals are very wide; and the analyst's own mutation and scipy cross-checks have no file behind them.**

## Q0. Records, runs and provenance: CONFIRMED

- **The copies are the PC's files.** All 19 `bpb.jsonl` (18 E3 runs plus E2's `5m_e3_r1_b250M`) have the same sha256 on the Mac and the PC. All 304 tail log records (step, loss, drawn_oasst2, drawn_cccc for steps 7480-7630) equal the PC's `log.jsonl` (`audit/a3_pc_compare.py`, 0 mismatches). `queue_e3.txt`, the status file and the code list also hash equal.
- **No missing, extra or repeated runs.** The PC's `runs.jsonl` holds exactly 18 E3 start and 18 end events, one of each per run, all `resumed_from: null`. `~/planck/runs/E3` holds exactly the 18 run folders, and its only mark is `E3_PART_1_DONE`. The queue ran the registered order (A1 B1 ... A8 B8, then R2, R3), every run ended `rc 0`, and the queue log prints no interim numbers.
- **Runs read are the runs named.** Every bpb record's `run` field matches its folder. Each config's LR and seed match its name (A 3e-3, B 1.5e-3 on all three LR keys, seed = s). R2 and R3 differ from A1 only in name and out_dir. All 18 config files equal their e39112a versions and the PC queue's code list.
- **Arms are E2's.** A = 3e-3, r 1 (E2 stage C pick). B = 1.5e-3, the pick's neighbour with the lower 250M F(CHAT): 1.1487 against 6e-3's 1.1511 (E2 results.json).
- **Settings** (`audit/a2_checks.py`, 73 checks, 0 failures). Every preflight is ok and strict with no refusals. The engine block equals FIXED 1 (compile false, ce_chunk_rows 0, varlen, batched, bf16, pack), with engine_fixed "2026-09-26 93ea41c". Shares, file counts, n_params 5,010,133 and the prereg sha be89babb all match. Every start record shows full WSD, 7,630 steps, decay from 6,104 and warmup 76, and every end record shows step 7,630.
- **Scoring is the same everywhere.** Every bpb record has tokenizer 078b24c4 and evalset fe1b55bb, fp32 precision and max_windows null. Each run has 3 checkpoints (7344, 7497, final 7630). Every run scored identical windows and bytes per set, with 24 truncated windows (oasst2 4, wikimedia 20), matching FIXED's "24 of 23,844". The stored CHAT and PROSE rows equal my own pooling of bits over bytes exactly.
- **Common random numbers hold as recorded.**
  - A_s and B_s have equal cumulative per-source drawn counts (with tokens, sup_tokens and dropped_long) at all 16 tail log steps and at the end, for all 8 seeds.
  - The 8 seeds have 8 distinct end streams.
  - R2, R3 and E2's seed-1 branch equal A1.
  - **Wording caveat:** the report says the pairs "drew identical tokens". The evidence is identical counts, not hashed token content. That is strong evidence but not proof.
- **Mutation test of these checks** (`audit/a4_mutants.py`). On a scratch copy, I broke one thing at a time:
  - a pair's drawn count;
  - a replay's end count;
  - a seed collision;
  - the stored PROSE row;
  - B's scores placed in A's folder;
  - an LR swap in a config;
  - compile turned on;
  - a resumed run;
  - the evalset sha;
  - max_windows;
  - the prereg sha;
  - one copied loss;
  - a missing tail step.

  All 13 mutants were killed, and the unmutated copy passes. In the first run one mutant survived. The cause was a bug in my mutant: it truncated the file before reading it. I fixed it and re-ran.
- **Leaks:** none found. The eval sets are identical across runs. The seed changes only init and data order. E2's eval-train disjointness (FIXED 4) was not re-checked here.
- **e005w1 mapping:** not applicable. E3 reads no E005 or E006 weights.

## Q1. Noise table: CONFIRMED (every value)

All 13 metrics reproduce: mean A, mean d, sigma_seed (pooled as sqrt((var A + var B)/2), df 7), SD_d, rho, CRN ratio, t, p, 95% CI, sigma_rep and its share (`audit/a1_stdout.txt`; `audit/a6_vs_results.py` against `results.json`).

| Metric | Mean A | sigma_seed | SD_d (rel) | rho | Seeds 2..8 SD_d |
|---|---|---|---|---|---|
| F(CHAT) | 1.1467 | 0.00293 | 0.00278 (0.242%) | 0.617 | 0.00239 |
| F(PROSE) | 1.3814 | 0.00205 | 0.00202 (0.146%) | 0.530 | 0.00161 |

- CHAT user turns: SD_d 0.958%. Dolly: 0.722%, rho -0.051.
- The results.json sha256 is 1f017ecb...4a2, as reported.
- My quantiles match scipy to 5e-11 and my p-values to 1e-14 (`audit/a1_scipy_quantiles.py`).
- **Imprecision the table does not show:** with df 7, the 95% interval for SD_d(F(CHAT)) is [0.00184, 0.00565], and for F(PROSE) [0.00133, 0.00411].

## Q2. "A vs B is resolved, B is worse": CONFIRMED WITH CAVEAT

- **Numbers are exact:**
  - F(CHAT): d +0.00527, CI [+0.00295, +0.00759], p 0.0010. B is worse at 8/8 seeds; per-seed d runs +0.0011 to +0.0094.
  - F(PROSE): d +0.00262, CI [+0.00093, +0.00431], p 0.0079. B is worse at 7/8 seeds; s1 has d -0.00074.
- **Caveat: this is not a pre-registered verdict.** The report lists it as "Pre-registered verdict 1", but the pre-registration has no rule that reads the 8-seed A vs B test and labels it "resolved". Its only A-vs-B rule is the seeds 2..8 re-check (Q3).
  - The 8-seed test includes seed 1, which E2 selected. Change 3 says that biases a test toward A. Here d_1 is the smallest CHAT difference, so including it lowers the mean d.
  - On F(PROSE), the later-screen WIN rule (rule 3: every seed pair must improve) would not be met, because s1 goes the other way.
- **Sensitivity:** using R2, R3 or the replay mean in place of A1 for pair 1 gives F(CHAT) p 0.0006, 0.0002 and 0.0005, and F(PROSE) p 0.0051, 0.0035 and 0.0052. The direction never changes.

## Q3. E2 re-check: CONFIRMED (as registered), with one scope caveat

- **Rule** (notes lines 94-96 and E2 notes line 116): a paired t-test on F(CHAT) over seeds 2..8, df 6. If B is lower with two-sided p < 0.05, the LR becomes B.
- **Result:** mean d +0.00586 (B higher), t +6.487, p 0.00064. B is not lower, so the rule does not fire and the 5M LR stays A (3e-3, r 1).
- **Scope:** the re-check tests A against B only. The report's "E3 confirms E2's pick" holds against B, not against the other neighbour. 6e-3's single-seed gap in E2 was +0.0032 bpb, which is 1.15 SD_d and below the k=1 paired MDE of 0.0091 bpb. E3 does not resolve A against 6e-3, and no rule asked it to.

## Q4. Replays and the E2 extra pair: CONFIRMED WITH CAVEAT

- **Replays are not deterministic.** F(CHAT) is 1.14974, 1.14865 and 1.14634; the step-7480 losses also differ (3.33563, 3.33404, 3.33606).
  - F(CHAT): sigma_rep 0.00174, share 2 sigma_rep^2 / SD_d^2 = 0.78.
  - F(PROSE): sigma_rep 0.00051, share 0.13.
- **Caveat on "about 78% of the paired variance":** with df 2, the share's 95% interval from sigma_rep's chi-square alone is 0.21 to 30.8, before counting SD_d's own uncertainty. The data allow anything from about a fifth of the paired variance to all of it. Two more limits:
  - The share assumes B has the same GPU noise as A, but only A1 was replayed.
  - R2 and R3 ran about 15 h after A1, at the end of the queue, and the three values fall in time order. Three runs cannot separate drift from noise. The pairs ran back to back, which protects d from slow drift.
- **E2's seed-1 run vs A1** (no registered test, as the report says):
  - F(CHAT) 1.14790 is -0.20 sigma_rep from the replay mean.
  - F(PROSE) 1.38302 is +1.61 sigma_rep from it.
  - Both reproduce. See D2 for what the "harness-version difference" does and does not mean.

## Q5. CRN ratio: CONFIRMED WITH CAVEAT

- **Values:** sqrt2 sigma_seed / SD_d is 1.49 (CHAT) and 1.44 (PROSE). The rule reads the point estimate and keeps CRN the default without making it mandatory, and the analysis applied it as written.
- **Caveat on uncertainty:** the Fisher 95% interval for rho is [-0.16, 0.92] for CHAT and [-0.28, 0.90] for PROSE. That puts the ratio at about 0.9 to 3.6, so "below 2" is not established.
- **Caveat on scope:** irc (3.00) and training loss (12.8) are above 2. No rule reads them ("every rule reads F"), but the report's summary does not mention them.

## Q6. Averaging rule: CONFIRMED

- SD_d of M is higher than SD_d of F: by 2.3% for CHAT and 14.5% for PROSE.
- So M is not allowed (the rule requires M at least 30% below F), and screens decide on F.
- **Wording:** "SD_d changes by -2.3% ... (it goes up)" uses a stored "reduction" field whose sign is easy to misread. M is noisier, not quieter.

## Q7-Q9. MDE, seeds needed, refresh reference: CONFIRMED

- **MDE:** all 208 values (13 metrics x k = 1, 2, 3, 5 x paired, unpaired and both 80% upper bounds) match. The power factor is 3.2607 (df 7), the 80% upper-bound multiplier 1.3533 and the refresh SD multiplier 1.7308.
  - F(CHAT) paired: 0.790, 0.558 [0.756], 0.456 and 0.353%. Unpaired: 1.177, 0.832 [1.126], 0.679 and 0.526%.
  - F(PROSE) paired: 0.476, 0.337 [0.456], 0.275 and 0.213%. Unpaired: 0.685, 0.484 [0.655], 0.395 and 0.306%.
- **k_needed:** every value matches, paired and unpaired. F(CHAT) 0.5% needs 3|6 and 1% needs 1|2; F(PROSE) 1% needs 1|1. A default 1% bet therefore runs max(2, k) = 2 seeds.
- **Two-seed sign agreement:** 28/28 for F(CHAT) and 21/28 for F(PROSE).
- **Refresh reference:**
  - F(CHAT): [+0.00082, +0.00972], SD bound 0.00481.
  - F(PROSE): [-0.00061, +0.00585], SD bound 0.00349.
- **Small omission:** notes line 92-93 asks that the seeds 2..8 values (factor 3.353) be printed beside the noise table. The SDs are printed but the seeds 2..8 MDEs are not. I computed them:
  - F(CHAT) paired k=1..5: 0.699, 0.495, 0.404 and 0.313%; unpaired: 1.242, 0.878, 0.717 and 0.555%.
  - F(PROSE) paired: 0.392, 0.277, 0.226 and 0.175%.
  - The registered 8-seed table is the more conservative one for paired comparisons, so no verdict changes.
- The "other types" rows are the registered fallback (sqrt2 sigma_seed, since Part 3 did not run).

## Q10. Not run: CONFIRMED

Part 2 (20M), Part 3 and Part 4 have no runs on the PC. Part 3 is not yet reached in the joint queue order, and the data_seed patch is not in harness/train.py.

## Deviations

- **D1: CONFIRMED.** trainer.py logs the single-step loss at every 10th step (`last_loss`, rounded to 5 decimals), so the 2% window holds 16 logged steps (7480-7630), the same in every run. The values equal the PC logs.
- **D2: CONFIRMED, WITH A CORRECTION THAT FAVOURS IT.**
  - **What reproduces:**
    - The FIXED 2 recipe gives 92deaef0 at 93ea41c (84 files) and b773159e at e39112a (97 files). The PC queue's own code list also gives b773159e.
    - train.py, trainer.py and model.py differ; bpb.py (4f73dbda) and evalwin.py (e3110772) are unchanged.
    - No E3 FIXED log was ever appended.
  - **Correction:** D2 says "no CUDA parity run of the default path is recorded". harness/notes.txt at e39112a does record one: `default_parity.py --device cuda` (main 7e7c802 vs the speed/v3 tree 2522edd/25c616d, deterministic, 24 steps), with log, checkpoint, start record and module set identical, PASS. That parity run is the CUDA counterpart of the CPU proof D2 cites.
  - **Why the later changes do not matter:**
    - Harness .py files at 7e7c802 equal 93ea41c's.
    - The 25c616d..e39112a diff touches only compile-on paths.
    - By code reading, the eager path at e39112a does the same computation: the trainer calls `self.fwd = model`, and the new loss guard short-circuits only under `torch.compiler.is_compiling()`.
  - **Both readings:**
    - Strict reading: E3 FIXED requires "Everything in E2's FIXED block, with the same values", including the harness commit, and E2's FIXED heading says "the analyzer refuses a run whose recorded value differs". All 18 runs would be refused and Part 1 would yield no noise numbers. The LR verdict would be the same, because A stays unless there is evidence for B.
    - The analyst's reading: a disclosed deviation with no rerun. All within-E3 statistics compare runs of one commit.
    - Neither reading changes the LR verdict. The noise numbers stand only under the second reading.
- **D3: OVER-CONSERVATIVE.** preflight.py (5fe0febf, the version in the queue's code list) refuses a run unless the shard manifest file's sha256 equals the config's recorded `shard_manifest_sha256`.
  - base5m.yaml (070d32cd, equal to FIXED 5) sets that value to 23647db7...ef41344, which is FIXED 4.
  - All 18 preflights are ok and strict with no refusals.
  - So the manifest was checked at preflight, indirectly. No copied record shows the hash itself.

## Record gaps (no verdict depends on them)

- The report says each analyzer check was "made to fail on purpose against a deliberately altered copy" and that "a second script using scipy reproduced the headline statistics". Neither has a file in the folder, and e3analyze.py has no mutation mode.
  - notes line 204 says "checked against scipy 1.13.1 ..., below", but no such entry appears below it.
  - The audit's own scipy cross-check and mutation test cover the substance.
- The deviations entry says it was written "before any bpb value was read into a statistic". Files cannot show that: both new entries are in the same uncommitted append (71 lines added, 0 removed), and notes.txt was modified at 18:11, after results.json at 18:10. None of D1-D3 changes a statistic that a verdict reads.
- The "PART 1 QUEUED (2026-09-28 16:40 EDT)" entry, and E2's "E2 5M DONE (... 16:40)", are inside commit e39112a dated 16:24:40. The queue saw the mark at 16:24:55. The 16:40 times are wrong labels; this is cosmetic.

## What was not re-checked

- The bpb computation itself: the auditor did not rescore checkpoints and loaded no model.
- Token content of the streams: only counts were compared.
- E2's eval-train disjointness build (FIXED 4).
- The harness parity runs themselves: they are cited from harness/notes.txt, not re-run.
- Not part of this audit: the relayed user request to research whether anyone has already done what the project is attempting. This auditor did not do it.

## Files (audit/)

- `a1_recompute.py`: own statistics from out/. Outputs `a1_out.json` and `a1_stdout.txt`.
- `a1_scipy_quantiles.py`: scipy cross-check and pair-1 sensitivity. Output `a1_scipy_stdout.txt`.
- `a2_checks.py`: provenance checks. Output `a2_stdout.txt`.
- `pc_tail_losses.sh`: read-only, run on the PC. Output `pc_tail_losses.out`, compared by `a3_pc_compare.py` (`a3_stdout.txt`).
- `a4_mutants.py`: mutation test of the audit checks; it writes only to a scratch directory. Output `a4_stdout.txt`.
- `a5_extra.py`: rho and CRN-ratio intervals, plus the 6e-3 neighbour gap, plus the seeds 2..8 MDEs. Output `a5_stdout.txt`.
- `a6_vs_results.py`: mine vs results.json. Output `a6_stdout.txt`.

# E3 Part 2 audit (independent, 2026-10-04)

One adversarial auditor re-derived every Part 2 number from the raw records in `out/e3_20m_*` (and, for rule 4, Part 1's raw `out/e3_5m_*` records rather than results.json) with its own code (`audit/p2_a1_recompute.py`: closed-form quantiles at df 1 and 2, Simpson integration of the t density at df 7; cross-checked with scipy 1.18.1). It compared the copies with the PC originals read-only (`audit/p2_pc_read.sh`, `audit/p2_pc_read2.sh`), and read every verdict against the pre-registration (notes.txt lines 1-174, committed at b04d104 before any E3 run), D4 (f314eb7) and PART 2 QUEUED (3c102f7). No model was loaded, the GPU was not used and gpu.lock was not taken. Writes: this section, `audit/p2_*`, the six `out/e3_20m_*/run_records.json` (one key removed, P2-Q0), wording in notes.txt PART 2 RESULT, and one appended corrections line.

**E3 Part 2 audit: every reported number reproduces from the raw records (964 shared statistics, 0 mismatches, largest relative difference 2.4e-12), the copies equal the PC originals, results_part2.json and tables_part2.txt regenerate byte for byte, and every reading holds as registered. Corrections: the copied run records carried the PC's machine name (removed); rule 4 picks the larger SD, which for F(CHAT) paired gives a smaller MDE than the 20M SD with its own df (0.5%: 3 seeds rather than 5); the CRN-mandatory reading rests on a correlation that 3 pairs cannot tell from 0; the F(PROSE) "B lower" reading is descriptive and would not be a WIN under rules 3 and 4; the analyst's scipy check has no file; the analyst's summary line "B is not lower" (re-check) overstates what the notes say correctly.**

## P2-Q0. Records, runs and provenance: CONFIRMED (one record correction)

`audit/p2_a2_checks.py`, 36 checks, all pass (35 before the correction below).

- **The copies are the PC's files.** bpb.jsonl and preflight.json of all 6 runs, the queue log, the status file and the code list have the PC's sha256. All 186 tail records (31 per run) equal the PC log.jsonl records with step > 14,953. The start and end records equal the PC's runs.jsonl lines on every compared field, and each end loss equals the last line of the run's train.out.
- **No missing, extra, repeated or resumed run.**
  - The PC's `runs/E3` holds exactly the 24 run folders (18 Part 1, 6 Part 2); its only marks are E3_PART_1_DONE and E3_PART_2_DONE.
  - runs.jsonl holds exactly 6 e3_20m start and 6 end events (and 18 and 18 for 5M), all `resumed_from: null`.
  - Each log.jsonl has 1,526 loss records on the full grid (10..15,250 by 10, plus 15,259), none non-finite: no divergence anywhere in the run, not only in the tail. latest.json points at final_00015259. The three scored checkpoints are on the PC (the finals are kept for Part 4). score.err is empty and train.out has no resume, NaN or error line.
  - The queue log's Part 1 part is byte-identical to the committed `logs/queue_e3.txt`. Its Part 2 part is exactly 46 lines: one start at 3c102f75dc68... with plan part2.txt 6b996207 (equal to part2.txt at 3c102f7), "seen E2: E2 20M DONE", then for A1 B1 A2 B2 A3 B3 in that order a lock wait, the lock, one train with --require-committed, "done rc 0" and three scores, then the MARK and "plan finished". The status file adds 6 "done and scored" lines in the same order.
- **Runs read are the runs named.**
  - Each config's bytes equal the sha256 listed in PART 2 QUEUED and the 3c102f7 blob. The bodies differ only in name, out_dir, seed and the three LRs (A 3e-3, B 1.5e-3, on lr, embed_lr and scalar_lr): full WSD, decay_frac 0.2, 15,259 steps, ckpt_every 305.
  - Start records: decay_start 12,207, warmup 76, n_params 20,001,511, batch 32,768, bf16, varlen, batched, prereg be89babb committed. Preflights are ok and strict with no refusals; the engine block, engine_fixed, shares and n_files equal Part 1's. base20m.yaml, base5m.yaml and engine.yaml at 3c102f7 equal e39112a. GPU, Python 3.14.4, torch 2.13.0+cu130 and CUDA 13.0 are Part 1's.
  - Arms: A20 is E2 q2's argmin (3e-3, r 1); B20 is 1.5e-3, the neighbour with the lower 500M F(CHAT) (0.9749 against 6e-3's 0.9814).
- **Code (D2, D4).** All 342 code-list hashes equal the files at 3c102f7. The FIXED 2 recipe at 3c102f7 gives 56b4a68f38d43cc8 over 100 files (D4's value), not 92deaef0. runio.py is ffcdf245 (the D4 pin). Against Part 1's code list exactly the 8 files that D4 and E2 DEVIATION 1 name differ; train.py, trainer.py and model.py (805bf5f8, 15d1a971, f4e61556), bpb.py and evalwin.py are Part 1's. The imported e3analyze.py equals the code list's and 620c77c's (the tip of main when the audit began).
- **Scoring is the same everywhere.** Every bpb record has tokenizer 078b24c4, evalset fe1b55bb, fp32, max_windows null, its own run name and the right checkpoint name; 13 rows at each of 14,945, 15,250 and 15,259; windows, bytes and truncated counts equal Part 1's row for row (24 truncated: oasst2 4, wikimedia 20); bpb = bits / bytes; the stored CHAT and PROSE rows equal my own pooling.
- **CRN holds as recorded.** For each seed, A_s and B_s are equal on all 11 stream counters (7 drawn, 2 dropped_long, tokens, sup_tokens) at all 31 tail steps and at the end; their losses differ at every tail step (two different runs, not one copied); the 3 seeds drew 3 different streams. Counts, not hashed token content, as the notes say.
- **Correction (OUTPUTS: "No machine address, user name or key path in any of them").** The six run_records.json carried `env.host`, the PC's machine name, which Part 1's copies drop. The audit removed that one key (25 bytes per file, nothing else changed). No statistic reads it, and results_part2.json still regenerates byte for byte. The notes' "Part 1's key set" holds for the top-level keys only: `env` also keeps device, numpy and platform, which Part 1's copies drop. They are harmless and left as copied.
- **Mutation test of these checks** (`audit/p2_a4_mutants.py`, scratch copies only). 29 mutants, one at a time: a bpb value, a windows count, the stored PROSE row, max_windows, the evalset sha, a missing checkpoint step, a tail drawn count, a tail CRN break, a missing tail step, a resumed start, the host key put back, a seed collision, a config LR, a config seed, compile on, the queue order, a second train line, the queue's commit, a status outcome, the runio.py hash, results.json, e3analyze.py, prereg.yaml, the plan, and five edits to the PC printouts (a non-finite loss, an extra start event, a non-empty score.err, a train.out loss, an extra run folder). Eight of the record mutants also edit the PC printout in the scratch copy to match, so that the copy-vs-PC comparison is not what catches them. All 29 killed; the unmutated copy passes 36/36. In the first run the missing-step mutant was caught by a crash rather than a failing check; the check was changed to fail cleanly and the run repeated. The recompute chain was mutated too: +1e-5 bpb on one CHAT value gives 113 mismatches against results_part2.json; unmutated, 0.
- **The analyst's own mutation test** (`e3analyze_p2_mutants.py`, re-run with its temp directory in the auditor's scratchpad): 18 of 18 killed, and the unmutated copy fails only D2's four checks (24 of 28 pass), as reported.

## P2-Q1. Noise table at 20M: CONFIRMED (every value)

- 964 shared statistics against results_part2.json, 0 mismatches (`audit/p2_a6_stdout.txt`). scipy 1.18.1 on quantiles, SD_d, sigma_seed, rho, t, p, CI, CRN ratio and the k = 1 MDEs: largest relative difference 4e-15 (`audit/p2_a5_stdout.txt`).
- results_part2.json (e8256d57) and tables_part2.txt (b9377cab) regenerate byte for byte from a scratch copy under Python 3.12.13, and Part 1's results.json still regenerates as 1f017ecb. (tables.txt regenerates without the 6 seeds 2..8 lines that Part 1's audit added by hand and labelled as such.)

| Metric | Mean A | sigma_seed (rel) | SD_d (rel) | rho | CRN ratio | SD_d 95% (chi-square, df 2) |
|---|---|---|---|---|---|---|
| F(CHAT) | 0.9741 | 0.00122 (0.126%) | 0.00189 (0.194%) | -0.455 | 0.91 | [0.00099, 0.01191] |
| F(PROSE) | 1.2287 | 0.00113 (0.092%) | 0.00057 (0.047%) | 0.909 | 2.80 | [0.00030, 0.00360] |

- Seeds 2..3 (df 1): SD_d 0.00143 (0.147%) and 0.00080 (0.065%). Secondary: CHAT user 1.322%, dolly 0.506%, CHAT assistant 0.021%, wikimedia 0.022%.

## P2-Q2. MDE and k_needed at 20M: CONFIRMED

- Factor 5.3633 (df 2), 80% upper-bound multiplier 2.1169, df 1 factor 14.0826.
- F(CHAT) paired 1.043, 0.737, 0.602, 0.466% [k2 bound 1.561%]; unpaired 0.952, 0.673, 0.550, 0.426% [1.426%]. F(PROSE) paired 0.250, 0.177, 0.144, 0.112% [0.374%]; unpaired 0.701, 0.495, 0.404, 0.313% [1.049%]. In bpb, F(CHAT) k1 0.0102 | 0.0093, k2 0.0072 | 0.0066.
- k_needed from the 20M SDs, paired | unpaired: F(CHAT) 5|4, 2|1, 1|1; F(PROSE) 1|2, 1|1, 1|1 (0.5%, 1%, 2%).

## P2-Q3. 20M against 5M (descriptive): CONFIRMED

- The 5M SDs, re-derived here from Part 1's raw records, equal Part 1's (F(CHAT) SD_d 0.242%, sqrt2 sigma_seed 0.361%; F(PROSE) 0.146%, 0.210%).
- Ratios 20M / 5M: SD_d 0.80 [0.31, 5.04] and 0.32 [0.12, 2.00]; sqrt2 sigma_seed 0.49 [0.19, 3.09] and 0.62 [0.24, 3.90] (95% F, df 2 and 7). Every point estimate is below 1 and every interval contains 1, so "three pairs do not show 20M noise to be lower or higher" is the right reading.

## P2-Q4. Rule 4, seeds for 20M confirmations: CONFIRMED WITH CAVEAT

- **Rule** (notes line 115-116): "20M confirmations use the larger of the 20M and 5M relative SDs, with the df of the one used." In all four cells the 5M SD is the larger, so df 7 (factor 3.261) and Part 1's relative MDEs apply: F(CHAT) paired k1 0.790% (0.0077 bpb at A20's mean), other classes 1.177%; F(PROSE) 0.476% and 0.685%.
- **Seeds, k = max(2, k_needed):** F(CHAT) paired 3, 2, 2; F(CHAT) other classes 6 ("underpowered"), 2, 2; F(PROSE) 2 at every delta in both classes. All match.
- **Caveat: the rule compares SDs, not MDEs.** For F(CHAT) paired the 5M SD_d (0.242%) is the larger, but its df 7 factor (3.261) is much smaller than df 2's (5.363). So the reference that rule 4 selects gives a smaller MDE (k1 0.790%) than the 20M SD with its own df (1.043%), and k for a 0.5% effect is 3 rather than 5. The same happens, barely, for F(PROSE) other classes (k1 0.685% against 0.701%), with no change in any seed count. In the other two cells rule 4's MDE is also the larger MDE. The analysis applies the rule as written, and that is the registered consequence; it is just less conservative than "the larger" suggests, and it changes one seed count (F(CHAT) paired, 0.5%).

## P2-Q5. Curve (reading e): CONFIRMED WITH CAVEAT

- PLAN (lines 829-831): 2 seeds for screening, 3 for continuous confirmations, 3 per curve size (5 at 10M and 30M if E3 shows lock-in). Lock-in turns on pass-rate spread (Part 4b, not run), so no lock-in reading exists and the curve counts do not change. Rules 2 and 4 ask 2 seeds at 1% and 2%, at most PLAN's 3: no conflict.
- **Caveat:** "PLAN's 3 is enough" for a 0.5% F(CHAT) effect holds under rule 4 as registered (P2-Q4); the 20M SD with its own df would ask 5.

## P2-Q6. E2 re-check at 20M: CONFIRMED (as registered)

- **Rule** (notes lines 95-96): "At 20M the same test over seeds 2..3 is reported (df 1) and moves nothing."
- **Result:** d +0.00018, -0.00185; mean -0.00084, t -0.826, p 0.5604, 95% CI [-0.0137, +0.0121]. 20M runs stay at A20 = (3e-3, r 1), as E2 Q2 set.
- **Wording:** B is lower in the point estimate (by 0.00084), not at p < 0.05. The notes say exactly that. The analyst's summary line "B is not lower" overstates it.

## P2-Q7. A20 against B20, 3 seeds: CONFIRMED WITH CAVEAT (descriptive, as labelled)

- **Numbers are exact.** F(CHAT): d +0.00193, +0.00018, -0.00185; mean +0.00009, CI [-0.00462, +0.00479], p 0.945. F(PROSE): d -0.00196, -0.00268, -0.00155; mean -0.00206 (0.17% of A20's mean), CI [-0.00348, -0.00064], p 0.025; B lower at 3 of 3. E2's 500M seed-1 grid had 1.5e-3 lower on PROSE by 0.06% (1.2281 against 1.2288).
- **Caveat: no rule reads this.** It is one of 13 descriptive 3-seed tests, uncorrected. Read as a 20M confirmation under rule 3 with rule 4's reference (5M SD_d, 0.146% of the mean = 0.00180 bpb, df 7, k 3), the WIN threshold is 0.00245 bpb and the mean improvement is 0.00206: every pair improves, but it would not be a WIN. Only with the 20M SD_d (0.00057) and its df 2 would it pass (threshold 0.00142).

## P2-Q8. CRN rule at 20M (reading d): CONFIRMED WITH CAVEAT

- **Rule** (notes lines 85-86): "a ratio of 2 or more makes CRN mandatory for every screen." It names no size; reading d, fixed in the ANALYSIS entry, applies it literally to the 20M F ratios. F(PROSE) is 2.80, so CRN becomes mandatory; F(CHAT) is 0.91.
- **Caveat on evidence:** the ratio rests on rho 0.909 from 3 pairs. The exact test of rho = 0 (t = r sqrt(n - 2) / sqrt(1 - r^2), df 1) gives p 0.27 (F(CHAT): rho -0.455, p 0.70), so 3 pairs cannot tell the ratio from 1. At 5M, with 8 pairs, the F(PROSE) ratio was 1.44.
- **Consequence:** none in practice. SCREENS.txt C4 (620c77c and fc135b9) already gives every arm at seed s the identical stream, so no screen design changes, as the notes say.

## P2-Q9. E2's 20M one-seed label (reading f): CONFIRMED

- E2 RULES (E2 notes lines 133-134): "resolved at one seed" when the best-vs-second difference exceeds E3's paired MDE at k = 1 (F(CHAT)), else "not resolved".
- The 500M difference is 0.0023 (0.9749 - 0.9726, E2 results.json q2 at its 4 decimals). That is below the 20M k1 MDE (0.0102) and the rule-4 one (0.0077): "not resolved", and the pick stands either way. E2's notes carry no such label for any stage (only the rule's own lines contain the words), as reported. The label belongs in E2's notes, outside this write scope.

## P2-Q10. Averaging at 20M: CONFIRMED

- M's SD_d is 2.2% (CHAT) and 8.6% (PROSE) below F's, far from the 30% the rule asks; the rule reads 5M only. The last two saves are 15,250 and 15,259, so M weights nearly the final weights twice, as disclosed.

## P2-Q11. Deviations D5, D6 and the D4 carry-over: CONFIRMED

- **D5:** screens_lib.py at 620c77c (unchanged by fc135b9) pins `E3_SHA = 1f017ecb...` and asserts it before a screen batch; SCREENS.txt cites the same sha. results.json, tables.txt and e3analyze.py equal 620c77c, and Part 1's audit section is unchanged. D5 departs from OUTPUTS' single results.json, and its reason holds.
- **D6:** steps > 15,259 - 305.18 are 14,960..15,250 and 15,259, 31 records, the same in every run.
- **D4:** as P2-Q0 (pin, digest, the 8 files, one commit for all six runs).

## P2-Q12. Not run, not computed: CONFIRMED

- Part 3 needs the data_seed change; `data_seed` is in neither harness/ at 3c102f7 nor at fc135b9. Part 4 has not run. sigma_rep, the two-seed sign agreement and the refresh bounds are not defined at 20M.

## Record gaps (Part 2; no verdict depends on them)

- The analyst's scipy cross-check (scipy 1.18.1, "equal at the printed digits") has no file. The audit's `p2_a5_scipy.py` covers the substance.
- "Written before any bpb value was read" rests on the analyst's account, as the entry says. What files can show: notes.txt through the end of the ANALYSIS entry (line 400) hashes to c3bb812f, the value the RESULT entry records, so the readings text has not changed since that hash was taken. The audit's edits all come after that line, so the check still works on the committed file. The file times (results_part2.json 09:06, notes.txt 09:08) are consistent with the stated order but do not prove it. The analyst saw train.out's final single-batch losses before writing the entry, and disclosed that.
- The analyst's checks do not compare the copies with the PC; that was done at copy time and has no file. The audit's comparison covers it.
- The "12.08 to 14.083" table-label change cannot be checked after the fact; the current analyzer reproduces the current tables.

## What was not re-checked (Part 2)

- The bpb computation itself: no checkpoint was rescored and no model was loaded.
- Token content of the streams: only counts were compared.
- E2's eval-train disjointness build (FIXED 4).
- D4's parity run: CPU fp32 only, cited, not re-run, and never run on CUDA.
- Open, outside this write scope: the six Part 2 runs have no entry in `models/registry.jsonl` (Part 1's 18 do); their final checkpoints are on the PC.

## Files (audit/, Part 2)

- `p2_pc_read.sh`, `p2_pc_read2.sh`: read-only, run on the PC by pcwsl.sh. Outputs `p2_pc_read.out`, `p2_pc_read2.out` (home directory and user name masked).
- `p2_a1_recompute.py`: own statistics from out/. Outputs `p2_a1_out.json`, `p2_a1_stdout.txt`.
- `p2_a2_checks.py`: provenance and run-identity checks. Output `p2_a2_stdout.txt`.
- `p2_a4_mutants.py`: mutation test of p2_a2 and of the recompute chain; scratch copies only. Output `p2_a4_stdout.txt`.
- `p2_a5_scipy.py`: scipy cross-check (system Python 3.14.4, scipy 1.18.1). Output `p2_a5_stdout.txt`.
- `p2_a6_vs_results.py`: mine against results_part2.json. Output `p2_a6_stdout.txt`.
