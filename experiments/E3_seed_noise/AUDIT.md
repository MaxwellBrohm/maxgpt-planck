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
