# Cuts audit: what was cut, and what goes back (2026-10-06)

Written 2026-10-06 (late evening, EDT) by Claude, read only on the repo except this file. No model was loaded, nothing
was run on the GPU, no sealed file was read. Every row cites the file and line that holds the evidence; rows that rest
on the working session's transcript say "session transcript line N".

## Why this file exists

Max, 2026-10-06, after the SCREENS cap cut S006: "if something can be helpful even a little we should give it its
proper chance" (quoted in full at experiments/SCREENS.txt:1835-1838), and later the same day asked whether any other
helpful work had been cut the same way. This audit lists every cut found in the repo, the plan files and the session
record, and decides for each one: run it now, run it Saturday (2026-10-10) or later on a stated trigger, keep it
dropped, or already done.

## How the decisions were made

1. A cut made for budget or usage alone goes back unless no outcome of it could change any decision. Max's rule is
   about budget cuts; a cut made for a stated scientific reason (a confounded arm, a measured negative, a licence or
   D8 rule) stays dropped with that reason.
2. Reinstating something is honest only through a dated amendment or pre-registration written before any result of
   the reinstated item exists, and never by re-reading a verdict already read.
3. "Now" means: it can run unattended on the 5070 between the end of the S006 runs (early Oct 7, estimated) and
   Saturday, with no new code beyond configs (plus the acceptance-test updates that any new config needs in this repo),
   under a dated amendment. The S006 amendment's premise is that the GPU is otherwise idle until Saturday
   (SCREENS.txt:1849-1850, not checked there or here).
4. Two judges (one arguing for reinstatement, one against) rated every item. Where they disagreed, the files were
   opened and the decision below says what they show.

ID prefixes: S-Cnn = the screens list, L-Cnn = the plan and ledger list, D-Cnn = the corpus list; E*, RC12-*, OODH-*,
P*, K*, H*, T* as given by the finders.

## Bottom line

- S006 is reinstated on paper but not installed. Until its commit, its bundle and the waiter's start happen
  (SCREENS.txt:1954-1956, 1960), its four runs never start. That is step 0 below.
- Three more things can run on the idle GPU before Saturday with configs only: three BASE seeds (103-105) that give
  IND onset a seed SD and test the seed-draw cause of the BASE level gap; the RC-12 extras P-145 and P-144, already
  listed in the prereg but never run; and cRia, which was killed by a 1,950 s process cap that the queue does not
  itself impose.
- The cuts that most deserved a chance and did not get one: the PLAN 3d toy (P-022 was never looked at, so Canon won
  the must-run slot by default), the 5M and 20M curve points and the larger token budgets that PLAN says are adopted
  but its tables do not carry, the dose-response arm ("capacity floor or data floor?"), the unsettled 5M budget for
  the must-run data bets, E8-E10 vanishing (not moving to 20M) if E4 finds no signal at 5M, the OOD-H shotgun gap,
  HD never computed for any screen, and E3 Part 3.
- 146 rows below, merged from about 250 finder items: NOW touches 6 rows (four steps), SAT 31, LATER 58, KEEP 54,
  DONE 13; 15 rows carry two decisions (for example a Saturday text edit and a later run).

## Where the judges disagreed, and what the files say

1. S008's clean 8-layer sharing contrast (S-C02). One judge: run now; the other: later. Later. It is not config
   only: screens_lib.check refuses any arm not in the SCREENS registry (experiments/screens/screens_lib.py:198-199),
   and S008 was never registered. Its bet moves 294,912 attention parameters (W_q and W_k of four layers, d 192) into
   MLP width 488 to 552 (S008_qk_share/notes.txt:93), which is the direction where bpb and behaviour can disagree
   (SCREENS.txt:22-25), so it needs the post-lock Tier 0 reading. It goes into a batch 2 pre-registration.
2. S002 plain_block (S-C04, T02). Run now (as a two-switch "nostab" arm) vs keep dropped. Keep dropped. plain_block
   turns off all four switches (S002_block_ablations/notes.txt:88-91), including value residual, which alone reads
   REJECT at +1.49% (SCREENS.txt:1442), so it cannot isolate the stabilisers. The two-switch arm would free about
   1,024 parameters (the QK-norm gains, harness/blocks.py:101-103; norm scaling is a constant), and under S002's
   registered rules a TIE keeps both and a NOMINATE still needs a 20M confirmation (S002 notes:57-62), while row 19's
   30M check decides the block anyway. No outcome changes a decision, so this is the exception to rule 1.
3. The batch cap (S-C12). Amend now vs Saturday. Both: the number moves now (to 28 h, only so the three BASE runs of
   step 1 cannot reduce S006's 1.634 h headroom: 25.366 + 0.936 = 26.302 h, and 28 - 26.302 = 1.698 h); the reading
   change (an overrun becomes a flag to Max, never a silent cut) and analyze_lib.CAP (still 25.0, analyze_lib.py:25)
   move Saturday with tests, since that is code.
4. HD, SINK and S001's 200-window GATE (S-C15) and E3 Part 4d. Now vs now. Saturday. The SCREENS queue runs only
   train lines (experiments/screens/queue_screens.sh:111-122) and its scoring is bpb only
   (experiments/E2_lr_transfer/queue_lib.sh:98-110), so these need a small scoring runner, which is new code. The
   tools themselves exist (harness/attn_diag.py, harness/induction.py, data_prep/bpb.py:172 --hd).
5. IND onset noise (E3-INDHOOK). One judge: six BASE runs at seeds 201-206 now; the other: Saturday, only if S006 or
   S007 look promising. Now, with seeds 103-105: screens_lib.check accepts only seed 1 and 101-105
   (screens_lib.py:226), so 201-206 would need code. Gating the SD on a promising descriptive reading would be a
   forking path; amending before any stage 2 onset is read is the honest order. Stage 2's selection step checked only
   that IND readings were finite (SCREENS.txt:1592, 1710) and read no onset.
6. E2 D1 and D2 (E2-D1D2). Now vs Saturday. Later, redesigned. D1 is one run at one LR (E2_lr_transfer/notes.txt:
   136-137), so it cannot show whether the LR pick moves with batch size, its own question. A g in {0.5, 1, 2} check
   at batch 32 x 2048 with the IND hook answers both that and P-106's timing question.
7. BASE level gap (S-C32). Two runs now vs Saturday. Split. The seed-draw cause is tested now by step 1's three BASE
   seeds (5 BASE seeds against E3 Part 1's 8). The replay and hook-off runs need code: a replay of base_s101 needs a
   run name screens_lib.NAME_RE does not accept (screens_lib.py:56-57), and a hook-off config is refused by the C6
   hook check (screens_lib.py:220-222). They matter: E3's replays fell in time order and three runs could not separate
   drift from noise (E3_seed_noise/AUDIT.md:69-76), and S004, S006 and S007 pair with BASE runs scored in stage 1, days
   earlier (SCREENS.txt:1783).
8. P-101 superword tokens (L-C35, T34). Saturday screen vs E7 arm. E7 arm. The starter mix is 2.8% chat
   (SCREENS.txt:19-22), so a 5M starter-mix screen would barely train the superword embeddings; E7 trains on the chat
   mix (PLAN.md:16, 448).
9. RC-12 panel coverage (RC12-PANEL). Saturday vs keep. Keep, with one question to Max: minimind-3 was never put to
   him (it is not in decision 5, rc12/DECISIONS_LEVEL_R_FOR_MAX.md:53, while NOVELTY_2026-10.md:138 and 150 rank it
   first), and it and Supra2-Medium-Instruct are standard Qwen3 (NOVELTY_2026-10.md:138, 142), so on an existing
   engine family they would be config only.
10. Wrong-seed replay (RC12-R1). Now vs Saturday. Saturday, optional. No tool flag for it was found, and seed 101
    against seed 1 already gave 0 of 5 identical, the same as the right seed (DECISIONS_LEVEL_R_FOR_MAX.md:43).
11. E006 seeds 6-10. Saturday vs keep. Later, as a gap filler. The decline was for budget (E006_three_objects/
    notes.txt:36-37), so rule 1 applies, but its answer does not change Planck's data design.
12. E005 linking follow-ups and E006's other candidates. Later (eval-only, low) for linking; keep for the candidates
    (SmolLM2 fine-tune mechanics that a from-scratch Planck does not inherit).
13. E003 LR below 5e-05. Later vs keep. Folded into the tiny-ladder pre-registration, not run alone.
14. Pipeline hygiene (P14, P16, P20, P21). Later vs Saturday. P20 and P21 (rules, LDNOOBW wiring) go with Saturday's
    checker round; decontamination, R1/R2/RH and the diversity caps are due before the first bulk generation run.
15. Dose-response (T19 a, L-C19). Saturday vs later. Both: the LEDGER row Saturday (text), the run later on the pool.
16. K10 count report. Saturday vs keep. Saturday, the count report only (free); the noisy-lookup arm stays out.
17. Compile for new experiments (S-C28, H01). Saturday vs later. Saturday for Track K (its configs are written at its
    commit); later for batch 2.
18. The Claude-usage cut rule (L-C62). Saturday vs later. Saturday: PLAN.md:925 says "Cut to the must-run list",
    which conflicts with Max's quality-over-usage rule and his 10/6 rule.
19. P-137 (RC12-F6F8). Later vs keep. Later for P-137 only: the ruling put F6 "out of RC-12" to run as its own
    measurement (prereg/RC-12.draft.txt:1189-1196), and nothing schedules it.
20. 150M budget (L-C16), P17, P11, P-104, S003's narrower form. Small splits; each decision is in the table.

## Table of every cut

Decision codes: NOW (before Saturday, steps below), SAT (Saturday 2026-10-10), LATER (trigger in the list below),
KEEP (stays dropped), DONE.

| # | IDs | What was cut | Where | Who decided | Stated reason (type) | Help | Cost | Decision |
|---|---|---|---|---|---|---|---|---|
| 1 | S-C11, L-C01 (cut), T01 | S006 (P-105, one t+2 MTP aux head) cut whole by ORDER's 25 h cap before its seed sets, after its 4 seed-1 runs (1.416 h) were spent; reinstated at a 27 h cap | SCREENS.txt:1744-1755, 1830-1863, 1954-1963 | Rule written by Claude; reinstated by Claude under Max's delegation | 25.366 h vs 25 h; hinged on 2.120 h of gate and smoke hours (budget) | medium | 1.471 GPU h, 4 runs; install only | DONE as a decision; install pending: NOW step 0 |
| 2 | S-C09, S-C10, L-C01 (narrowing), T06 | S006 narrower than P-105 and loop.md: one t+2 head (harness refuses mtp >= 2), 5M only, curriculum arm conditional, one aux weight | S006_mtp_aux/notes.txt:15-18, 34-41, 120-126; LEDGER.md:155 | Claude | curriculum only if IND BETTER at a bpb cost; none for one head (design) | low | 2-head arm: harness code plus about 0.37 h per 5M run; a 20M run about 2.8 h | SAT (write the follow-up trigger before S006's verdict); runs LATER |
| 3 | S-C12, L-C02 | Batch cap 25 h (now 27 h), cut order S006, S003, S007, S002, conservative reading; analyze_lib.CAP still 25.0 | SCREENS.txt:245-251, 270-277, 1842-1845, 1957-1959; analyze_lib.py:25 | Claude | keep 5-35 h of PLAN's 30-60 h for the 3b data bets (budget) | medium | no GPU; an amendment, one constant, tests | NOW (number to 28 h in step 1's amendment); SAT (overrun = flag to Max; CAP in code) |
| 4 | S-C01, L-C03, T03 | S008 (P-110) 9-layer arm: shared W_q/W_k across layer pairs, saving spent on a ninth layer | SCREENS.txt:1, 6-7, 41-42, 56-58; S008_qk_share/notes.txt:81-91 | Critic, applied by Claude | not one change: sharing, depth, MLP width and init/norm scale move together (design) | low | n/a | KEEP |
| 5 | S-C02 | S008's 8-layer sharing contrast (qk_share 2, SwiGLU 552, exactly 5,010,133 params) | S008_qk_share/notes.txt:92-100 | Claude after critique | two arms of a rank-109 bet under the 25 h cap (budget) | low | arm-registry code and tests (screens_lib.py:198-199); g-check 3 plus 2 seeds, about 1.5 GPU h | LATER |
| 6 | S-C03 | K = V tie variant of P-110 | S008_qk_share/notes.txt:11-16 | Claude | its one measured cost (3.1% perplexity) points the other way (design) | low | about 1.5 GPU h | KEEP |
| 7 | S-C04, T02 | S002 plain_block fourth arm (all switches off) | S002_block_ablations/notes.txt:88-91, 117; SCREENS.txt:347, 1471 | Claude | about 1.9 h at k 3 against the cap (budget) | low | about 1.6 GPU h plus a pre-registration | KEEP |
| 8 | S-C05, T05 (LR part) | New-module LR multiplier for S004 convs, S005 forget w and b, S007 alpha | SCREENS.txt:348-350; S004_canon/notes.txt:40-47; S005_forget_gate/notes.txt:31-34, 200; S007_smeared_key/notes.txt:34-36 | Claude | another selection axis (design) | low | a flag plus about 1 GPU h per module | LATER |
| 9 | S-C06, S-C07, S-C35, L-C22, T04, T05 (toy part) | PLAN 3d 2-layer toy never built: no P-020 variant pick, no P-022 vs P-148 pick, P-022 never screened, S005 tests only the forget gate | SCREENS.txt:50-51, 350-351, 1749-1750; S005_forget_gate/notes.txt:11-17, 98-101; PLAN.md:467-472, 485 | Nobody; the code was never written | no toy code exists (missing code) | high | toy runs are minutes (LEDGER.md:71, 73); code: MQRAR generator, 2-layer models, a gated delta-rule layer, the P-020 variants | SAT (pre-registration); code and runs LATER, before Nov 9 |
| 10 | S-C08 | S007 reads only P-024's timing half; the one-layer-fewer half and its 1-vs-2-layer toy not run | S007_smeared_key/notes.txt:14-15, 24-26, 142 | Claude | would change the shape (design) | low | toy minutes; a depth-reduced arm about 1.5 GPU h | LATER |
| 11 | S-C15, S-C17 (final IND, HD), E3-PART4 (d) | HD (S001-S003), SINK (S001, S002) and S001's 200-window GATE never computed; E3 Part 4d (SD for final IND and HD) never written | SCREENS.txt:139-147, 1476-1484, 1545; S001_attn_gate/notes.txt:56-68; E3_seed_noise/notes.txt:46-54 | Claude (the verdict step barred model loads) | a model load the step excluded (unclear) | medium | scoring only, about 50 kept checkpoints, est. 1-2 GPU h; code: a scoring runner plus the Part 4d addendum | SAT |
| 12 | E3-INDHOOK, S-C17 (onset) | IND onset has no seed SD (E3 Parts 1-2 ran without the hook), so P-105's and P-024's own win condition is descriptive only | SCREENS.txt:145-147; E3_seed_noise/out/e3_5m_A_s1/train_tail.jsonl (no IND field); S006_mtp_aux/notes.txt:20-22 | Unclear (left to E3's owner) | the hook landed after E3 Part 1 (missing data) | medium | 3 BASE runs, about 0.94 GPU h; configs only | NOW (step 1) |
| 13 | S-C16 | Stage 2 per-module diagnostics have no tool: S004 conv weight per site, S005 per-head half-life, S007 alpha per layer and head | S004_canon/notes.txt:72-74, 82; S005_forget_gate/notes.txt:81-84, 179-180; S007_smeared_key/notes.txt:64-66 | Not stated | "not written" (missing code) | medium | Claude code; minutes of GPU | SAT |
| 14 | S-C18, E3-PART3, T40 (Part 3) | E3 Part 3 (sigma_init, sigma_data) not run; the 4-line data_seed patch was never committed and is gone | E3_seed_noise/notes.txt:41-45, 55-61, 147-153; E3_seed_noise/AUDIT.md:112, 257; SCREENS.txt:120-121 | Rule written by Claude | patch never landed; also first to yield to the teacher pilot (missing code, budget) | medium | 8 runs plus a same-commit A1 rerun, about 2.5-2.8 GPU h; code: the patch, a test watched failing on a mutant, flag-off equivalence | SAT |
| 15 | S-C19 | Matched-LR IND runs only for S006 and S007 | SCREENS.txt:155-161 | Claude | onset is their win condition (design) | none | none | KEEP (moot: every other pick is g 1, SCREENS.txt:1612-1618) |
| 16 | S-C20, E2-T0MARGINS, E3-PART4 (a, b, c) | No RC-12 before the lock: Tier 0 margins on screen, E2 stage C and E3 checkpoints; E3 4b pass-rate spread; 4c sets; E004 adapter | SCREENS.txt:14-18, 83-89; E2_lr_transfer/notes.txt:219-221; E3_seed_noise/notes.txt:46-54 | PLAN hard rule; guard by Claude | no Planck model on any RC-12 split before the lock (design) | high | scoring only; one addendum per item | LATER |
| 17 | S-C21, L-C04, T07, L-C34 (P-056) | Every data bet out of batch 1; whether the must-run 3b data bets fit PLAN's 30-60 h of 5M screening "flagged, not settled" | SCREENS.txt:46-50, 272-277; PLAN.md:485, 491 | Claude (flagged) | no chat pool yet; budget (missing data, budget) | high | a PLAN edit now; bets 5-50 h each (LEDGER) | SAT (PLAN text); runs LATER |
| 18 | S-C22, S-C37, L-C29 | E8/E9/E10 (Max's skill-over-storage idea) run at 5M only if E4 finds signal; PLAN's no-signal branch keeps only the must-run list, so they would vanish | SCREENS.txt:49-50, 252-257; PLAN.md:472, 515 | Pre-registered PLAN rule | a 5M bpb screen cannot judge them without signal (design) | medium | PLAN text now; 10-40 h each later | SAT (PLAN text); runs LATER |
| 19 | S-C23 (P-158), S-C24 (P-159), L-C34 (P-159) | Vocabulary optimum and untied embeddings not in batch 1 | SCREENS.txt:50-52; PLAN.md:448 | Claude | E7 owns them (superseded) | medium | E7 60-120 h | LATER (E7) |
| 20 | S-C23 (P-106), E2-D1D2, L-C34 (P-106) | E2 D1 (batch 32 x 2048) and D2 (scalar_lr) never configured; P-106 excluded as "covered by D1" | E2_lr_transfer/notes.txt:136-138, 202, 415; SCREENS.txt:51 | Claude | no rule reads them; yield to the pilot (budget) | low | redesigned: 3 runs at batch 32 with the hook, about 1 GPU h, E2 configs | LATER |
| 21 | S-C24 (P-109), L-C34 (P-109) | Attention-only SAN | SCREENS.txt:51-52; LEDGER.md:159 | Claude | parity only at about 105B tokens at 24M (design) | low | n/a | KEEP |
| 22 | S-C25, L-C34 (P-104) | Partial RoPE / NoPE interleave | SCREENS.txt:53; LEDGER.md:154 | Claude | needs eval past 2k (missing code) | low | 15-25 h at 30M | LATER |
| 23 | S-C26, L-C34 (next batch) | P-172 ReLU-squared, P-173 multipliers, P-174 TOP, P-112 hyper-connections: "next batch", and no next batch exists | SCREENS.txt:53-54 | Claude | none (unclear) | low | flag, tests and mutants each; about 1.5 GPU h per 5M screen | LATER |
| 24 | S-C27 | Run-length weight-decay rule moved to E2's refresh | SCREENS.txt:54-55; E2_lr_transfer/notes.txt:216-218 | Claude | calibration, not a bet (superseded) | medium | part of E2's refresh | LATER |
| 25 | S-C28, H01 | Compiled engine (C1-a, +107-121%) withdrawn; train.py compile off by default; no K or batch 2 config sets it | SCREENS.txt:353-358, 577-591; harness/notes.txt:827-841 | Claude under Max's delegation | compiled Canon fails deterministic mode; a lenient re-read would be a forking path (design) | medium | a flag-parity gate per new experiment, minutes | KEEP for batch 1; SAT (Track K); LATER (batch 2) |
| 26 | S-C29 | Compiled-Canon deterministic-mode defect not investigated | SCREENS.txt:499-502, 580-581; S004_canon/notes.txt:172-176 | Claude | the harness owner's (superseded) | low | Claude debugging | LATER |
| 27 | S-C30 | Third parity invocation for marginal C1-a fails | SCREENS.txt:509-510 | Claude | would redraw until one passed (design) | none | n/a | KEEP |
| 28 | S-C31 | E3 rule 5 refresh (3 compiled pairs, 0.97 h) | SCREENS.txt:381-387, 587-588 | Claude | not triggered under eager (superseded) | none | 0.97 h if triggered | KEEP |
| 29 | S-C32 | BASE level gap (+0.66% CHAT vs E3 arm A): causes not separated | SCREENS.txt:1485-1507, 1546-1547 | Claude, left open | no verdict waits on it (unclear) | medium | seed draw: inside step 1; replay and hook-off: 2-3 runs, about 0.9 GPU h, plus a run-name rule | NOW (seed-draw part); SAT (replay, hook-off) |
| 30 | S-C33 | S003 narrower than P-119 (no bits-per-parameter measure; WD tuned for neither) | S003_adamw/notes.txt:1-2, 72-78 | Claude | scope (unclear) | low | n/a | KEEP |
| 31 | S-C34 | S004 tests Canon-AC only, not ACD | S004_canon/notes.txt:10-16, 29; harness/config.py:70 | Claude | none stated | low | code for sites B and D; about 1.5 GPU h | LATER |
| 32 | S-C36 | S007 variants: smear after QK-norm, smear values | S007_smeared_key/notes.txt:29-33, 76-77 | Claude | a different arm (design) | low | a flag plus about 1.5 GPU h | LATER |
| 33 | S-C13 | S003 REJECT drops AdamW from row 19's 30M check | S003_adamw/notes.txt:128-138; SCREENS.txt:1472-1475 | Pre-registered | REJECT holds under Holm, +1.53% (design) | low | at least 3 runs at 30M | KEEP |
| 34 | S-C14 | S002 novres REJECT: no 30M value-residual-off arm | S002_block_ablations/notes.txt:58-60; SCREENS.txt:1468-1469 | Pre-registered | REJECT holds, +1.49% (design) | none | 30M runs | KEEP |
| 35 | L-C05, L-C06 | 106 of 185 ranked bets never in PLAN, including 10 of the 14 untested-at-this-size bets (P-037, P-073, P-074, P-087, P-090, P-114, P-115, P-116, P-117, P-131) | PLAN.md:485; LEDGER.md:4, 12, 48-236 | Claude (PLAN v2) | "Everything else runs if time allows" (budget) | medium | doc work; per bet (LEDGER): P-087 3 h, P-131 1-2 h, P-037 5-7 h, P-090 10 h | LATER |
| 36 | L-C07, L-C09, E001-STEP2D, RC12-EXTRAS | P-144 (Falcon-H1-Tiny-90M sibling checkpoints), P-145 (SmolLM2-135M SFT-only vs Instruct), P-147 (MaxGPT-3): RC-12 extras in prereg s12, never run, no "not run" line | prereg/RC-12.draft.txt:907-908, 922-925; REPORT.md:548; E001_battery_and_probes/notes.txt:3; LEDGER.md:195-198 | Unclear | none recorded | medium | LEDGER: P-145 under 1 h, P-144 5-10 h; config: pins.json, engines.json; P-147 needs an adapter | NOW (P-144, P-145: step 2); SAT (P-147) |
| 37 | L-C07 (P-142), L-C08 (P-142) | Counterfactual open-book probe | PLAN.md:216; LEDGER.md:192 | Unclear | none found | low | 1-3 h, likelihood only | LATER |
| 38 | L-C08 (P-005, P-006) | Phase 0 diagnostics: P-005 state card, P-006 re-injected system prompt; P-005 is a PLAN decision point | PLAN.md:212-216, 270; LEDGER.md:56-57 | Unclear | none found | medium | 2-6 h and 2 h inference; code: the card and re-injection renders | LATER |
| 39 | L-C10, E002-360M, T27 | E002's SmolLM2-360M block | PLAN.md:21; E002_ft_test/AUDIT.md:97; E004_general_updating/notes.txt:210 | Claude | 135M passed; that pass later fell to a wording shortcut (design) | low | 3 seeds at 360M | KEEP |
| 40 | L-C11, E003-DEFER, T29 | E003 deferred, then resumed under its own rule and finished (all 8 FAIL) | E003_correction_floor/notes.txt:142-145, 218-235 | Claude | own deferral rule | low | done | DONE |
| 41 | L-C12, E005-TINYLADDER, E003-LRBELOW | Tiny ladder (public models of 30M and under) with E005's recipe: trigger met, never pre-registered; E003 never tried an LR below 5e-05 | E005_alias_eot/notes.txt:168-170; E005_alias_eot/AUDIT.md:396, 407; docs/EXPERIMENTS.md:233; E003_correction_floor/notes.txt:79-81, 243-248 | Claude | needs its own pre-registration (design) | low | a few GPU h; minutes per fine-tune | LATER |
| 42 | L-C13 | The 100M point of Max's original curve | research/conversation_ideas.md:10; LEDGER.md:105; PLAN.md:44 | Claude | none (unclear) | low | one curve run | LATER |
| 43 | L-C14, L-C15, L-C18, L-C63, T37 | PLAN Phase 4 tables still 10/30/60M at about 500 tok/param, though PLAN.md:22 says the CORPUS budgets (5M-30M, 1,000-5,000 tok/param) are adopted; CORPUS 7.4's edits never applied | PLAN.md:17-22, 526-535, 797-805; CORPUS.md:946-956, 1151-1162 | Claude (tables not updated) | none stated | high | PLAN and LEDGER text only now | SAT |
| 44 | L-C16 | 150M at 75B tokens by default; 60M and 150M at 1 seed (CORPUS) vs 3 (PLAN) | PLAN.md:556, 805; CORPUS.md:1151-1152 | Claude | Titan time (budget, external) | low | Titan weeks | KEEP the budget rule; fix the seed text with row 43 |
| 45 | L-C17 | P-186 (E-tok) narrowed: 5M to 2,000 tok/param (5,000 optional), the 20,000 branch moved to a Titan after the 150M seeds | CORPUS.md:784-802; PLAN.md:17; LEDGER.md:275 | Claude, adopted under Max's delegation | 1,150-2,380 h vs 280-690 spare (budget) | high | 5M trunk 35-75 h; one 5M seed to 20,000 tok/param is about 109 GPU h eager by arithmetic (250M slots take about 0.27 h; 100B is 400 times that), not measured | SAT (fix PLAN.md:17 and LEDGER.md:275 text); LATER (run) |
| 46 | L-C19, T19 (a) | P-051 density sweep narrowed to 0/2/8% at 30M and not must-run; NOVELTY's dose-response arm never added | PLAN.md:455; LEDGER.md:101; research/NOVELTY_2026-10.md:116 | Claude | none stated | high | needs the pool; 4 doses x 2-3 seeds at the floor size | SAT (LEDGER row, text); LATER (run) |
| 47 | L-C20 | P-026 accuracy-gated distance curriculum not must-run | PLAN.md:456, 485; LEDGER.md:77 | Claude | none stated | medium | 30-60 h | LATER |
| 48 | L-C21 | P-023 role / speaker embeddings not must-run | PLAN.md:471, 485; LEDGER.md:74 | Claude | none stated | medium | a flag; 15-25 h | LATER |
| 49 | L-C23 | P-068 multi-turn OPD pilot "if there is time"; P-069, P-070, P-168 and D6's only revisit trigger wait on it | PLAN.md:478; LEDGER.md:118-120, 266 | Claude | time (usage) | medium | 20-40 h on the 5070 | LATER |
| 50 | L-C24 | D6 consequences parked (logit KD from Ultra or SmolLM2, teacher vocab, weight inheritance) | REPORT.md:617-621; LEDGER.md:301-304 | Max (D6) | needs the shared vocabulary D6 removed (design) | low | n/a | KEEP |
| 51 | L-C25 | P-182, P-183, P-103 closed by D1 (total parameters) | REPORT.md:495-496, 608 | Claude, adopted | they store per FLOP, not per parameter (design) | none | n/a | KEEP |
| 52 | L-C26 | Primacy comparison on Max's 124M A/B checkpoints | REPORT.md:491, 550 | Claude | both arms likely at chance (design) | low | inference on Lambda | KEEP |
| 53 | L-C27 | Bits-per-parameter rig (Max's idea #3; P-114 to P-125, P-178, P-179) not in PLAN | research/conversation_ideas.md:12; LEDGER.md:45, 164-175 | Claude | stored knowledge out of scope, PLAN.md:53 (unclear) | low | 1-20 h per probe | LATER |
| 54 | L-C28 | Memory in weights across sessions (Max's idea #10; P-127 to P-135, P-180) | research/conversation_ideas.md:19; LEDGER.md:177-185 | Claude | none stated | low | P-127 under 1 GPU h | LATER |
| 55 | L-C30 | P-077 program-side best-of-N verification | research/conversation_ideas.md:18; LEDGER.md:127 | Claude | none stated | low | 2-4 h inference | LATER |
| 56 | L-C31 | P-050 token curve from public intermediate checkpoints (outputs lost) | LEDGER.md:100; CORPUS.md:803 | Unclear | lost outputs (missing data) | low | minutes per checkpoint once the likelihood scorer runs | LATER |
| 57 | L-C32 | P-054 pure skill floor with a restricted-world corpus at 1-10M | LEDGER.md:104 | Claude | none stated | medium | 30-60 h | LATER |
| 58 | L-C33 | Conditional bets waiting on a parent; P-095 can never trigger while P-090 has no slot | LEDGER.md:264-269; PLAN.md:462-465 | Claude | parent first (design) | low | 5-10 h per variant | LATER |
| 59 | L-C35, T34 | P-101 superword tokens failed the 5% bar (4.63%); the low-priority screen offered after Max asked was never added | LEDGER.md:287; PLAN.md:27; session transcript lines 8359-8364 | Pre-registered bar; the offer went unanswered | the bar (design) | low | one E7 arm (2 seeds at 5M) plus a tokenizer variant on CPU | LATER (E7 arm) |
| 60 | L-C36 | P-097 paired 5M name-recall run; P-100 (8k vocab as a subset of the teacher's) | LEDGER.md:147, 150, 288; PLAN.md:27 | Unclear; P-100 waits for the teacher pick | none for P-097 | low | about 6 h | LATER |
| 61 | L-C37 | P-141 MPS length bucketing | PLAN.md:441 | Claude | the Mac runs no screens (superseded) | none | n/a | KEEP |
| 62 | L-C38 | REPORT's Mac-only fallback | REPORT.md:569 | Claude | the 5070 became available (superseded) | none | n/a | KEEP |
| 63 | L-C39 | REPORT "not candidates": looping for knowledge, exact MTP, SuperBPE or byte level, test-time thinking | REPORT.md:498-502 | Claude | evidence against each at this scale (design) | low | n/a | KEEP |
| 64 | L-C40 | Parked tokenizer ideas | LEDGER.md:294-299 | Claude | worse than BPE at this budget; FLOPs (design) | low | n/a | KEEP |
| 65 | L-C41, T35 | Free-form or long thinking | LEDGER.md:311-313; REPORT.md:502 | Max | tiny models loop; thinking eats context (design) | low | n/a | KEEP |
| 66 | L-C42 | Parked architecture and capacity ideas (hash tables, Titans/Atlas, SSM hybrids, ternary, MoE, MoR) | LEDGER.md:300, 306-307, 314-325 | Claude | per FLOP not per parameter; measured negatives (design) | low | n/a | KEEP |
| 67 | L-C43 | Parked memory, retrieval and tool ideas | LEDGER.md:308-309, 326-331, 346-350 | Claude | measured failures at small size (design) | low | n/a | KEEP |
| 68 | L-C44, T32 | Parked data, teacher and infra ideas, including Gemma 4 26B-A4B as teacher | LEDGER.md:305, 310, 332-339; PLAN.md:318 | Claude; D8 confirmed by Max | does not fit batched on the 24 GB Mac or the 12 GB 5070; D8 (design) | low | n/a | KEEP |
| 69 | L-C45 | Parked evaluation ideas and skipped benchmarks | LEDGER.md:340-345; research/lanes/eval.md:65-80 | Claude | MDE, API judges, floors (design) | low | n/a | KEEP |
| 70 | L-C46 | Post-training "skip for now" list | research/lanes/posttrain.md:147 | Claude | loops, learnability (design) | low | n/a | KEEP |
| 71 | L-C47 | Probe lane skipped PleIAs Monad and MobileLLM-R1-140M | research/lanes/probe.md:35 | Claude | no multi-turn; gated math/code SFT (design) | low | n/a | KEEP |
| 72 | L-C48, L-C49, D-C01, D-C02, D-C03, D-C04, D-C06, D-C09, T36 | Strict data rulings: D8 corpora baselines only, no web-crawl bucket W, grey dialogue sets eval only, no WildChat v1, no restricted-label classifiers, Common Pile code/science/law out, Q7 after the freeze, no Ultra data | LEDGER.md:28; PLAN.md:322-324; CORPUS.md:169-175, 271-272, 1149 | Max (D8, no Ultra); Claude's defaults under Max's delegation | licence and provenance (design) | low | n/a | KEEP |
| 73 | L-C50, RC12-SEALEDSIZE | Sealed split 600, not 1,000-1,200 | rc12/DECISIONS_FOR_MAX.md:33-43; PLAN.md:158, 210 | Max on Claude's recommendation | n barely matters under the S7 bar (design) | none | n/a | KEEP; fix PLAN.md:158 text with row 43 |
| 74 | L-C51, RC12-LOOKUP, T20 | LOOKUP out of the composite R; back as its own headline claim; easier-cell rebuild declined | rc12/DECISIONS_FOR_MAX.md:18-21; rc12/notes.txt:2551-2579; PLAN.md:68 | Max | a floor family eases non-inferiority (design) | high | done | DONE (the rebuild stays dropped) |
| 75 | L-C52, RC12-STORE, T23 | No claim for Planck querying a lookup store itself | rc12/DECISIONS_LEVEL_R_FOR_MAX.md:35-40 | Max on Claude's recommendation | a key-level store is gameable; table-level breaks OD5 and s13 (design) | low | about a week of CPU work | KEEP (the post-lock diagnostic stays) |
| 76 | L-C53, RC12-PANEL, T21 | Sub-30M public chat models off the panel; RxT out (licence); minimind-3, KeyLM-75M, petitgpt never decided | rc12/DECISIONS_LEVEL_R_FOR_MAX.md:53; rc12/notes.txt:2719-2721; research/NOVELTY_2026-10.md:136-150 | Max (5 b); no record for minimind-3 and the other two | about two days of engine work (budget) | low | minimind-3, Supra2-Medium: config on a Qwen3 vLLM engine, minutes each; remote-code models days | SAT (one question to Max); rest KEEP |
| 77 | L-C54, RC12-R1 | Optional wrong-seed replay per engine not run | rc12/DECISIONS_LEVEL_R_FOR_MAX.md:42-45; rc12/notes.txt:3015-3016 | Max (a), optional | two short GPU holds (budget) | low | two short holds; no tool flag found | SAT (optional, before the lock) |
| 78 | L-C55, RC12-SMALLFIXES | Gemma stop id, PERSIST filler turns, ORDER_ABS twins, T0 birthday item | rc12/DECISIONS_FOR_MAX.md:60-63, 76-79; prereg/RC-12.draft.txt:729-733, 1105-1108 | Max; Claude (birthday item) | measured effects tiny (design) | none | n/a | KEEP |
| 79 | L-C56, RC12-LEVELR | Level R's original test (non-inferior to Qwen2.5-0.5B) made report-only | rc12/DECISIONS_LEVEL_R_FOR_MAX.md:21-33; PLAN.md:61-66 | Max (D) | Qwen2.5 scores S7 2.30 (superseded) | low | n/a | KEEP |
| 80 | L-C57, T11 | Novelty rechecks pending (ICLR 2027, BabyLM 2026, EMNLP list, Semantic Scholar, X) | research/NOVELTY_2026-10.md:95-100, 151-157 | Claude | not public yet; rate limits (missing data) | medium | Claude search time | LATER |
| 81 | L-C58, OODH-PART2 | OOD-H Part 2 (rater scripts) in January | PLAN.md:170, 991; oodh/DESIGN.txt:116-117 | Claude | protocol (design) | medium | rater hours | LATER |
| 82 | L-C59 | Whether to pay raters ($375-600) | PLAN.md:879, 994 | Max (open) | open (budget) | medium | $375-600 | LATER |
| 83 | L-C60 | Pausing Ultra to free the Titans not assumed | REPORT.md:623; PLAN.md:921 | Claude | not assumed | low | an Ultra delay | LATER |
| 84 | L-C61, D-C07 | Standing cut rules: rate check, E4 no-signal, Nov 22 must-run cut, the bench_micro re-plan rule, CORPUS's generation cut order | PLAN.md:31, 515-516, 538-542, 917, 929; CORPUS.md:917-929 | Claude | budget | low | n/a | SAT (text: a firing rule goes to Max first); LATER (when one fires) |
| 85 | L-C62 | Claude-usage rule "Cut to the must-run list" | PLAN.md:925 | Claude | usage | medium | one PLAN edit | SAT |
| 86 | L-C64, L-C65, L-C66, T41 | Release track: microcontroller stretch, many-characters demo and speed report, Planck-Assistant | PLAN.md:605-608 | Max (ideas); deferral by Claude | after a size passes (design) | low | later work | LATER |
| 87 | E2-20MEXT | 20M grid extensions; the 5M 62.5M vertex left a bound | E2_lr_transfer/notes.txt:402-412 | Pre-registered rule | the argmin was inside the grid (design) | none | one 5M trunk | KEEP |
| 88 | E2-LABELS | "resolved at one seed" labels never written into E2's notes | E2_lr_transfer/notes.txt:133-134; E3_seed_noise/notes.txt:395-398, 468-471 | Claude (write scope) | outside the E3 analyst's scope (unclear) | low | text only | SAT |
| 89 | E002-CHATPROBE, T28 | E002's chat probe, crossed rescore at 184 of 384, free generation | E002_ft_test/notes.txt:89; E002_ft_test/AUDIT.md:93-98 | Claude | superseded by E004's probes | none | n/a | KEEP |
| 90 | E004-TINYSEEDS, T31 | E004 tiny ladder: LR searches only, no chat probe, weights not saved | E004_general_updating/notes.txt:198-207, 490-495 | Pre-registered gate; Claude | 135M FAIL gives LR search only (design) | none | n/a | KEEP |
| 91 | E004-TS1M, T30 | TinyStories-1M dropped by a wall-clock kill, re-added and run | E004_general_updating/logs/queue.txt:13-14, 80-86 | Guard; Claude | lid-closed pause (usage) | low | done | DONE |
| 92 | E004-CONTCHAT | Cross/chat continuity comparison skipped, then done in E005 on 184 items | E004_general_updating/AUDIT.md:103; E005_alias_eot/notes.txt:278-279 | Claude | item counts differed | low | done | DONE |
| 93 | E005-AL | AL linking diagnostic dropped, then built and run before any E005 result | E005_alias_eot/notes.txt:154-165, 348-409 | Claude | strikable diagnostic | low | done | DONE |
| 94 | E005-LINKING | How far name linking goes (surname, title, pronoun, unseen title, 3 objects at d 20) and a larger eval draw | E005_alias_eot/AUDIT.md:390-397, 406, 408 | Unclear | none recorded | low | eval-only on kept E005 weights, minutes; item writing | LATER |
| 95 | E006-SEEDS | E006 seeds 6-10 declined; Q-H5's INCONCLUSIVE rests on seed 1 | E006_three_objects/notes.txt:36-37, 290-297, 740-759 | Claude | the brief fixes 5 seeds; GPU shared (budget) | low | 10 fine-tunes plus scoring, est. 5-7 GPU h | LATER |
| 96 | E006-CANDIDATES | Other causes of E005's H5 drop and the stratified BIG rebuild | E006_three_objects/notes.txt:37-38, 273-276 | Claude | one change per arm (design) | low | 5 seeds per candidate | KEEP |
| 97 | RC12-LKCOND | LOOKUP diagnostic conditions: shuffled table, no table, BM25 table | rc12/SPEC.txt:120-121; prereg/RC-12.draft.txt:141-142 | Claude | not in dev v1 | medium | generator, grader, scoring | LATER |
| 98 | RC12-F6F8, T22 | F6, F7, F8 out of RC-12; F6 was to run as P-137, which never ran | prereg/RC-12.draft.txt:348-349, 1189-1196; LEDGER.md:187 | Max (OD4 c) | no generator, grader or cheater gate (design) | low | P-137 3-6 h | LATER (P-137 only); F7, F8 KEEP |
| 99 | RC12-CRIA | cRia-LM-75M-Instruct (approved extra) not run: both processes killed at the chain's 1,950 s cap | rc12/notes.txt:2713-2718, 2971-2976; rc12/queue_dev_baselines.sh:32 | Claude (the chain's cap) | process cap (budget) | low | unmeasured; est. up to about 15 GPU h (8 processes) | NOW (step 3) |
| 100 | RC12-LIKROWS | Likelihood rows (P-004, golden history, Tier 0 margins) tested on stubs only; 4 panel tokenizers unmeasured | rc12/notes.txt:1212-1271; prereg/RC-12.draft.txt:273-276, 1114-1117 | Unclear | none recorded | high | minutes per public model; code for the 4 tokenizers | SAT |
| 101 | RC12-DOGE | Doge-160M left out, deferred, then given its own engine and run | rc12/notes.txt:1398, 2004-2005 | Claude | engine and timeout | low | done | DONE |
| 102 | OODH-F3 | Fallback F3 (wider reserve) rejected | oodh/DESIGN.txt:153-161, 263-265 | Claude | would put OOD-H trees into tokenizer v0 and the E2/E3 shards (design) | none | n/a | KEEP |
| 103 | OODH-GATES, T26 | Cheater gates G2 and G4 report-only on OOD-H | oodh/DESIGN.txt:197-203, 266-270 | Claude under standing permission | natural threads hold one value of a kind (design) | low | n/a | KEEP |
| 104 | OODH-SHOTGUN | Shotgun clause silently off on OOD-H; the SHOTGUN cheater scores 0.59 there (0.00 on RC-12); prereg s14 is silent | oodh/DESIGN.txt:108-110; oodh/HASHES.txt:25; prereg/RC-12.draft.txt:556, 1027-1058 | Unclear (left open) | none | high | CPU: a probe flag, test and mutant, or one stated sentence | SAT |
| 105 | P01, P02, T24 | OFFTOPIC no longer gates: lexical check report-only; teacher judge report-only until validated | pipeline/notes.txt:2044-2066, 2092-2098; pipeline/SPEC.txt:286-298 | Max (judge after validation) | not validated (design) | medium | a fresh hand set (40+ natural off-topic, about 320 on-topic) plus judge runs | SAT |
| 106 | P03, T25 | VOCAB_OOL report-only "for now" | pipeline/DECISIONS_BANKPASS_FOR_MAX.md:47-53 | Max (a) | everyday chat words (design) | low | n/a | LATER |
| 107 | P04 | prev_assist admission removed from OFFTOPIC | pipeline/BANKPASS.txt:194-195 | Pre-registered rule | loophole (design) | none | n/a | KEEP |
| 108 | P05 | AI-ism phrase ban left off, then on by default | pipeline/notes.txt:909-914, 1693-1703 | Claude | substitutes routed around it | medium | done | DONE |
| 109 | P06 | No length cap on the last structured line | pipeline/notes.txt:862-869, 1177-1180 | Claude | compile and regex cost (budget) | low | Claude design | LATER |
| 110 | P07 | Single-turn repair and the wider near-miss rule not built | pipeline/SPEC.txt:24, 121; pipeline/notes.txt:566-577 | Claude | not in v0 (design) | medium | code and tests | LATER |
| 111 | P08 | WordNet-tolerant OFFTOPIC | pipeline/notes.txt:646-648 | Claude | superseded by word sets and the judge | none | n/a | KEEP |
| 112 | P09 | Render-prompt content cut to fit the stub token budget | pipeline/notes.txt:672-673, 1262-1265, 1316-1324 | Claude | 2,048-token context (budget) | low | n/a | KEEP |
| 113 | P10 | Lowercase-chat name example removed | pipeline/notes.txt:1704-1706, 1727-1729 | Claude | measured: no help (design) | none | n/a | KEEP |
| 114 | P11 | fp8 KV cache abandoned; Ministral's Triton retry never tried | pipeline/notes.txt:398-405 | Claude | no capacity gain (design) | low | one GPU hold | LATER |
| 115 | P12 | Proposal to drop Ministral as a teacher, not taken | pipeline/notes.txt:477-478, 520-522 | No decision | n/a | low | n/a | DONE (kept) |
| 116 | P13 | Planned extra teacher-pilot arms (Gemma 4 E4B, llama.cpp Gemma, Qwen3.5-4B, an 8B paraphraser, speculative decoding) with no record of running or dropping | CORPUS.md:518, 548, 641, 656, 949; pipeline/teachers/hf_pins.json | Not stated | none | medium | pilot hours | LATER |
| 117 | P14, P15, P16 | 13-gram decontamination, R1/R2/RH, RH bank lines, SPEC 7 diversity caps not built | pipeline/SPEC.txt:160-187, 209-236; pipeline/BANKPASS.txt:50, 154, 240; pipeline/notes.txt:239-245 | Claude (build order) | paraphraser FAKE; FAKE renders (missing code) | high | Claude code (CPU) | LATER |
| 118 | P17 | Second-model naturalness read | pipeline/SPEC.txt:158 | Claude | none | low | small | KEEP (replaced by blinded reads, pipeline/notes.txt:1102-1115) |
| 119 | P18 | Token check of Gemma renders against LM Studio | pipeline/notes.txt:524-525 | Claude | none | low | n/a | KEEP |
| 120 | P19 | FALSE_MEMORY narrowed | pipeline/notes.txt:1247-1251 | Claude | false rejects of its own purpose (design) | low | n/a | KEEP |
| 121 | P20 | Defect classes with no checker rule: invented details, garbled or role-swapped user lines, guidance copies, bare correction tails | pipeline/notes.txt:1020-1021, 1042-1050, 1530-1533, 1715-1717 | Claude | "no rule" (design) | medium | rules and tests | SAT |
| 122 | P21 | LDNOOBW list downloaded but not wired into SAFETY (still a 14-term FAKE list) | pipeline/notes.txt:1637-1639; pipeline/lexicons.py:2, 182 | Claude | a checker change for its owner (unclear) | low | a small checker change | SAT |
| 123 | P22 | SSA name pool blocked; CRAN babynames substitute | pipeline/notes.txt:1592-1601, 2123 | Claude | 403 (missing data) | low | done | DONE |
| 124 | P23 | Human pool trims (minors, Census case) | pipeline/notes.txt:1623-1636 | Claude | safety and licence (design) | none | n/a | KEEP |
| 125 | P24, P25, P26, P27, T17 | Bank pass v2 not frozen: author-share exclusions, the thirds trim, instr.para out, orphaned topics partly restored | pipeline/BANKPASS.txt:13-14, 257-286; pipeline/notes.txt:1785-1805, 2143-2164 | Pre-registered rule; outage | usage | medium | Claude work plus teacher top-ups | SAT |
| 126 | P28 | Stage P's own bank probe | pipeline/notes.txt:1772, 1909 | Claude | replaced by probe_v2 (superseded) | none | n/a | KEEP |
| 127 | P29 | Stage B (3,000 more topics) after the teacher pick | pipeline/BANKPASS.txt:215, 219 | Claude | order (design) | medium | teacher hours | LATER |
| 128 | P30 | Word-list read from two 5% samples | pipeline/notes.txt:1366-1367, 1392-1395 | Claude | deviation (unclear) | low | about 1 h CPU | LATER |
| 129 | D-C05 | Simple Wikipedia and Aya absent from core v0 and its out-of-scope list | CORPUS.md:214, 247, 1247 | Not stated | none | low | a fetch and filter pass | LATER |
| 130 | D-C08 | Second near-dedup pass never adopted | corpus/stats/offload_2026-10.txt:48-50 | Not stated | none | low | a decision | LATER |
| 131 | K01, K03 | Track K's 15.5 h cap: K1-X, K3-R ("the training-time version of Max's idea"), K4b conditional on hours; K0-g at 125M with no rerun | experiments/K/SPEC.txt:500-505; K3_prune_regrow/notes.txt:85-93; K1_capacity/notes.txt:37-42 | Claude | budget | medium | +3.47 GPU h for the conditionals; about 0.9 h for a K0-g rerun if LENGTH-RISK fires | SAT |
| 132 | K02 | K3 arms left out (staged prunes, shrink-and-perturb, resets) | K3_prune_regrow/notes.txt:95-98 | Claude | the cap has no room (budget) | low | about 0.3-0.5 h each | LATER |
| 133 | K04, K05, K06, K07, K08, K11, T38 | K design narrowings (100 exposures, 2 restarts, lam 6, no warmup, ceilings printed beside, depth fixed in K4) | experiments/K/SPEC.txt:395-400, 809-815; K2_localize/notes.txt:221-223, 272-274, 350-368; K4_allocation/notes.txt:33-36; K1_capacity/notes.txt:97-104 | Claude | measured or structural (design) | none | n/a | KEEP |
| 134 | K09, T18 | K harness pieces not built (eval.k_probe, weights-only start, grad_route, per-group WD, analyze_k.py); round 10's tests-lens review unfinished; K uncommitted | experiments/K/code/notes.txt:269-273; experiments/K/SPEC.txt:510-519 | Claude; outage | missing code; usage | medium | Claude code and one review; then K's about 14 GPU h | SAT |
| 135 | K10 | K5 items not applied: loss mask, noisy-lookup arm, model-invoked lookup, supervised-token count report | K5_lookup/notes.txt:92-98, 127-128 | Claude | the mask removes the answer's supervision; a different policy (design) | low | the count report is free | SAT (count report only); rest KEEP |
| 136 | K12, T09 | Round 4's tests-lens review cut by a usage pause; re-run in rounds 5 and 8 | experiments/K/SPEC.txt:750; experiments/K/code/notes.txt:1635-1636 | Usage pause | usage | low | done | DONE |
| 137 | H02 | Compile modes max-autotune-no-cudagraphs and CUDA-graph modes refused | harness/notes.txt:498-499, 516-518, 810-816 | Claude after review | failed parity once (design) | low | n/a | KEEP |
| 138 | H03, H04, H05 | memfit.py not ported; chunked CE opt-in; capture_dynamic_output_shape_ops untested | harness/notes.txt:394, 514-515, 536-541, 740-748 | Claude | not needed; memory only; untested (design) | none | n/a | KEEP |
| 139 | H06 | Eager half-batch speedup (1-9%) not applied to running experiments | harness/notes.txt:786-792, 821-826 | Claude | mid-experiment deviation (design) | low | none for new configs | LATER |
| 140 | H07, H08 | fp16 with loss scaling (Titans), batch-size ramp, KV-cache decode not built | harness/notes.txt:98, 124-129; harness/device.py:5-6 | Claude | not wired yet (unclear) | medium | Claude code and a Titan smoke | LATER |
| 141 | H09 | First GPU-speed integration produced nothing | harness/notes.txt:323-327 | Workflow failure | n/a | none | done in later rounds | DONE |
| 142 | T08, T10, T12, T14, T15, T16, T33 | Usage-time cuts later reversed: Track K review rounds, chat-data round 2, registry entries, workflows cut by the 10/3 limit, speed builds, leaner workflows, the sub-100M probe | session transcript lines 375-1069, 5195-10962, 11318-11438, 12824-12884, 13342-13529; harness/notes.txt:827-841 | Claude; usage limits | usage | low | done | DONE (Max's 10/2 rule now forbids cutting reviews for usage) |
| 143 | T13 | Durable-save leftovers: SCORED is touched after bpb.jsonl is written by a plain redirect; rc12 DONE written in place; registry write_all without fsync | experiments/E2_lr_transfer/queue_lib.sh:109; experiments/screens/queue_screens.sh:98; rc12/dev_regrade.py:84; models/registry.py:157-161 | Claude | wait for pinned files (usage) | medium | small code plus a SCREENS deviation entry | SAT |
| 144 | T19 (c) | REPORT related-work lines (Codecs, 2605.29548, FB-Bench, StructFlowBench, Evolve, Karpathy's cognitive core, Fractale) | research/NOVELTY_2026-10.md:124 | Claude | queued after another workflow, never done | low | text only | SAT |
| 145 | T39 | Ministral had no same-day control in round 1 | pipeline/notes.txt:730 | Claude | not stated | low | n/a | KEEP (later pilots use a round-level control) |
| 146 | T42 | Saturday bundle: stage 2 verdicts and audits, the P-105 row, the SCORED fsync deviation | SCREENS.txt:1960-1963 | Claude, per Max's wrap-up | usage | medium | Claude workflows | SAT |

## REINSTATE NOW (before Saturday, unattended on the 5070, configs only)

GPU order, one queue at a time: stage2_seeds (running) -> S006 (step 0) -> BASE seeds (step 1) -> RC-12 extras
(step 2) -> cRia (step 3). Do not start a later step while an earlier queue still runs: a SCREENS queue waits at most
2 h for gpu.lock and then stops (queue_screens.sh:14-15), and nothing relaunches S006's queue after that
(SCREENS.txt:1919-1920). Check who holds gpu.lock before each launch. Total about 19-28 GPU h, against roughly three
idle days.

**Step 0. Install S006 (already decided; nothing changes). 1.471 GPU h, no new code.**
Rows 1. The S006 amendment is complete and tested but not installed (SCREENS.txt:1954-1956, 1960).
1. Check git status for files you did not touch; stage the S006 reinstatement paths explicitly and commit.
2. Make the S006 bundle from that commit and copy it to the PC as INSTALL says (SCREENS.txt:1954-1956); never use
   send_kit.sh, which would replace planck.bundle and break the autopilot's relaunch check (SCREENS.txt:1912-1915).
3. Start the committed s006_waiter.sh detached (pcdetach.sh). Confirm its log says WAIT now and LAUNCH after
   "SCREENS STAGE 2 SEEDS DONE".

**Step 1. BASE seeds 103, 104, 105: IND onset noise and the seed-draw cause of the BASE gap. About 0.94 GPU h.**
Rows 12, 29 (seed-draw part), 3 (cap number).
1. Write AMENDMENT BASE-DIAG in SCREENS.txt, dated, before any stage 2 onset, HD or verdict is read (stage 2's
   selection step read none, SCREENS.txt:1592, 1710). It states:
   - three BASE runs, base_s103, base_s104, base_s105, each differing from base_s101 only in name, out_dir and seed
     (screens_lib.check already accepts seeds 101-105, screens_lib.py:226);
   - onset noise: sigma_seed of onset_half from BASE seeds 101-105 (df 4). C7 then reads stage 2 onsets that are not
     yet read with SD_ref = sqrt(2) x sigma_seed: S004 and S007 at g 1, and S006 from its g 1 IND runs; d = arm minus
     BASE in steps, negative = earlier; BETTER if -dbar > thr and every d_s < 0, WORSE mirrored, no EQUAL (no delta
     is registered for onset). At k 2 and df 4, thr = 2.776 x sigma_seed (C7, SCREENS.txt:193-198), so only a large
     shift reads; onsets are logged every 153 steps (SCREENS.txt:150). S005 pairs with its own mask-engine BASE, so
     its onset stays descriptive. Stage 1 onsets (S001, S002) stay as read;
   - BASE level: the 5 BASE seeds against E3 Part 1's 8 arm-A seeds, reported, no rule. If seeds 103-105 also sit
     above E3, the gap is not the seed draw (SCREENS.txt:1485-1507);
   - the cap moves from 27 h to 28 h so each S006 check keeps its headroom (25.366 + 0.936 = 26.302 h; 28 - 26.302 =
     1.698 h, at least the 1.634 h recorded at SCREENS.txt:1858-1859); the reading is otherwise unchanged;
   - the three runs are also the shared BASE at seeds 103-105 for any later k 3 extension.
2. Write the three configs (copy base_s101.yaml, change name, out_dir, seed) and run screens.py check: expect 3 ok.
3. Write plans/stage2_base_diag.txt: a header, "wait_mark SCREENS SCREENS STAGE 2 S006 SEEDS DONE", the three train
   lines, "mark SCREENS BASE DIAG DONE". Point plans/CURRENT at it.
4. Update the acceptance tests that count configs and pin CURRENT's sequence; add a test that each new config differs
   from base_s101 only in name, out_dir and seed and that the plan is as written; watch each fail on a mutant.
5. Change nothing under harness/, data_prep/, tokenizer/, corpus/ or experiments/E2_lr_transfer (C4 pairing, and so
   these runs stay comparable to base_s101 and base_s102). E3 Part 3's harness patch (Saturday) lands after them.
6. Commit after the S006 commit. Once the S006 mark exists and its queue has exited, send the kit (the autopilot's
   W6 is DONE after the stage 2 seeds mark, SCREENS.txt:1916-1918, so planck.bundle may move then) and launch with
   launch_screens.sh.

**Step 2. RC-12 extras P-145 and P-144, already in prereg s12. LEDGER: P-145 under 1 h, P-144 5-10 h (probe and
battery; the dev queue alone is likely less).** Row 36.
1. A dated rc12/notes.txt entry before any run: SmolLM2-135M SFT-only (P-145) and the Falcon-H1-Tiny-90M sibling
   checkpoints other than Instruct (P-144, REPORT.md:548 step 2(d)), as the prereg already lists them
   (prereg/RC-12.draft.txt:907-908); a checkpoint without a chat template runs the plain render only, recorded so.
2. pins.json: the hub revision and the licence read at that revision. engines.json: vLLM entries carrying the family
   parity verdict, as SmolLM2-360M-Instruct's entry does (rc12/engines.json:25-26; Falcon-H1 family at
   engines.json:21). The prereg's parity rule is per model family (prereg/RC-12.draft.txt:238). The weights are a
   new hub download to the PC.
3. Run rc12/queue_dev_baselines.sh with Q_MODELS set to these models (an environment override,
   queue_dev_baselines.sh:22, 36), after step 1's queue has ended.
4. verify_dev_runs C1-C5 and pull with an md5 manifest. Scoring them on the Mac needs dev_extras14.py's fixed
   EXTRAS14 list widened (rc12/dev_extras14.py:29-31), a small Mac-side change with a test; it can wait for Saturday,
   and must be done before the lock (an unfinished extra is listed "not run", prereg/RC-12.draft.txt:922-925).

**Step 3. cRia-LM-75M-Instruct, an approved extra killed by the chain's 1,950 s cap. Unmeasured; est. up to about 15
GPU h.** Row 99.
1. A dated rc12/notes.txt deviation: the per-process cap goes back to the queue's own default of 10,800 s
   (queue_dev_baselines.sh:32); the 1,950 s cap was the item 14 chain's (rc12/notes.txt:2737, 2971-2976).
2. Run it last, through the same queue, with Q_MODELS set to cRia and its engines.json python. The estimate scales
   the 64-conversation smoke (388 s plus a 275 s twin, rc12/notes.txt:2713-2716) to 640 conversations: about 1.8 h
   per process, 8 processes. If a process still passes 10,800 s, cRia stays "not run" with the measured time.
3. Verify and score as in step 2 (cRia is already in EXTRAS14).

## REINSTATE SATURDAY (2026-10-10) / LATER

Saturday items (each is text or code on the Mac unless a GPU cost is named):

- S1. Cap reading (row 3). Dated amendment: a cap overrun becomes a flag to Max, never a silent cut; analyze_lib.CAP
  set to the amended value, with tests and mutants. No GPU.
- S2. S006 follow-up trigger (row 2). Before S006's verdict is read, write into S006's notes: if Q2 reads earlier
  onset at matched LR, pre-register the NTP-to-MTP curriculum arm, a 2-head arm (harness code for mtp >= 2) and a
  20M arm. Written later it would be post hoc.
- S3. PLAN 3d toy pre-registration (row 9). The pick rules (P-022 vs P-148 for the 30M must-run slot; the P-020
  variant) before any toy run; S004's 5M result stands, the toy decides only the 30M pick. Code and runs before PLAN
  3d starts (Nov 9); toy runs are minutes.
- S4. HD, SINK, GATE and E3 Part 4d (row 11). Write the Part 4d addendum (E3 notes require one before Part 4 runs,
  E3_seed_noise/notes.txt:46-47), then a small scoring runner for bpb.py --hd and attn_diag.py on the kept final
  checkpoints of stage 1, stage 2, steps 0-1 and E3: about 50 checkpoints, est. 1-2 GPU h. Before the stage 2
  verdicts are read.
- S5. Per-module diagnostics (row 13): conv weights and alpha from the checkpoints (CPU), S005's half-life on 200
  CHAT windows (minutes of GPU). Before the stage 2 verdicts are read.
- S6. E3 Part 3 (row 14). A dated E3 addendum reopening Part 3 after its "not run" report (nothing it measures has
  been seen). Rewrite the data_seed patch (default off; a config without data_seed behaves as now), a test watched
  failing on a mutant, and the flag-off equivalence test that C4 needs (SCREENS.txt:131-133); land it only after
  steps 0-1 finish. 8 runs plus a same-commit A1 rerun, about 2.5-2.8 GPU h. Its SDs apply only to verdicts read
  after it, never to S001. Due before the 3b data bets.
- S7. BASE replay and hook-off (row 29). A run-name rule for a registered BASE replay plus a hook-off diagnostic
  (screens_lib.py:56-57, 220-222), with tests; a dated entry, report-only, with the label it puts on stage 2 pairs
  stated before the stage 2 verdicts are read. 2-3 runs, about 0.9 GPU h.
- S8. PLAN and LEDGER text (rows 17, 18, 43, 44, 45, 46, 73, 84, 85): the 30-60 h figure never cuts a must-run data
  bet; on an E4 no-signal, E8-E10 move to 20M under new pre-registrations instead of dropping; the Phase 4 tables
  carry 5M and 20M and the CORPUS tokens per parameter; CORPUS 7.4's edits; the 60M/150M seed count; P-186's text at
  PLAN.md:17 and LEDGER.md:275; a LEDGER row for the dose-response arm; PLAN.md:158's sealed size; PLAN.md:925's
  usage rule says a usage cap slows or defers work and never cuts a bet; any budget rule that fires brings its cut
  list to Max first. No result is involved in any of these.
- S9. P-147 MaxGPT-3 (row 36): build the batched HF adapter or write its "not run" line with the reason before the
  lock. Also widen the extras scorer for step 2's models.
- S10. RC-12 likelihood rows (row 100): the first real-model run on public panel models (minutes each) and the 4
  missing tokenizers, before the lock.
- S11. OOD-H shotgun (row 104): a probe flag that re-enables the shotgun clause on OOD-H (with a test and a mutant), or
  one sentence in s14 saying it is off; before the lock and before any Planck model is scored.
- S12. Panel question to Max (row 76): add minimind-3 and Supra2-Medium-Instruct (config on a Qwen3 engine) before the
  lock, yes or no. The rest stays per his decision 5 (b); RxT stays out unless he accepts its licence.
- S13. Wrong-seed replay (row 77), optional, before the lock: two short GPU holds.
- S14. E2 labels (row 88): apply E2's own "resolved at one seed" rule in a dated E2 entry (20M: 0.0023 bpb gap vs MDE
  0.0102, "not resolved"; E3_seed_noise/notes.txt:395-398).
- S15. Pipeline (rows 105, 121, 122, 125): validate the OFFTOPIC judge against SPEC 17's bar on a fresh hand set
  and gate only if it passes; checker rules for role-swapped user lines and invented details, and LDNOOBW wired into
  SAFETY, measured on a dry pilot; finish bank pass v2 under the author-share rule with Max's small-bank change.
- S16. Track K (rows 25, 131, 134, 135): amend SPEC 13 before K's commit so the conditionals run when triggered
  (+3.47 GPU h) and K0-g reruns at 250M if LENGTH-RISK fires (about 0.9 h), with any overrun going to Max; rerun only
  the tests-lens review, commit, then build the section 14 pieces with tests watched failing; name compile in K's
  configs if its flags pass parity; add the supervised-token count report.
- S17. Housekeeping (rows 143, 144, 146): the durable-save leftovers with a SCREENS deviation entry before any affected
  run; the REPORT related-work lines; the planned Saturday bundle (stage 2 verdicts with --earlier).

Later, each with its trigger and honest path:

| Rows | Item | Trigger | Honest path | Cost |
|---|---|---|---|---|
| 2 | S006 curriculum, 2-head and 20M arms | S2's trigger fires | new pre-registrations | about 0.37 h per 5M run; 20M about 2.8 h |
| 5, 23, 48 | Batch 2 (S009+): S008's 8-layer sharing contrast, P-172, P-173, P-174, P-112, P-023 at 5M | stage 2 verdicts read | its own pre-registration, critique and Holm family; bpb now, Tier 0 and HD after the lock | about 1.5 GPU h per screen plus flag code |
| 8 | New-module LR multiplier | S5 shows a module near its init | new pre-registration with a multiplier arm | about 1 GPU h per module |
| 9 | Toy code and runs | S3 written | runs before Nov 9; decides the 30M pick | minutes |
| 10 | S007 depth-reduced arm and the 1-vs-2-layer toy | S007's Q2 and the toy | new pre-registration | about 1.5 GPU h |
| 16 | Tier 0 margins, E3 4a-4c, E004 adapter | RC-12 lock (Oct 11-14) | one addendum per item before scoring | scoring only |
| 17, 46, 47 | Must-run data bets, the dose-response arm with the 0.5% density point, P-026 | the first pool mix | each its own pre-registration | LEDGER 5-60 h each |
| 18 | E8, E9, E10 | E4 reports | at 5M with signal, at 20M without (S8's text) | 10-40 h each |
| 19, 59, 60 | E7 including P-159, a P-101 superword arm and P-097 | E7's pre-registration (before Oct 24) | P-101's failed 5% bar stays recorded; the arm is a new bet | 2 seeds at 5M per arm |
| 20 | E2 D1 redesigned as a batch 32 g-check with the IND hook (P-106) | before any curve run changes batch size | dated E2 addendum | 3 runs, about 1 GPU h |
| 22 | P-104 | a long-context eval exists | own pre-registration | 15-25 h |
| 24 | Run-length weight decay | E2's refresh (pool mix or tokenizer v1) | inside the refresh's pre-registration | part of the refresh |
| 25, 26 | Compile for batch 2; compiled-Canon defect | batch 2 is written; any compiled Canon run | flag-parity gate per pre-registration | minutes |
| 31, 32 | Canon-ACD; S007 variants | their screens' verdicts | new pre-registrations | about 1.5 GPU h each |
| 35, 37, 53-58 | PLAN triage appendix for the 106 unplaced bets; P-142, bits rig, weight memory, P-077, P-050, P-054, P-090 for P-095 | after the lock push, before batch 2 | every bet gets a slot, a gate or a stated reason; each its own pre-registration | per LEDGER |
| 38 | P-005 state card, P-006 | before the four-arm memory test | pre-registered on public models; Planck's card defined in a post-lock addendum | 2-6 h and 2 h |
| 41 | Tiny ladder with E005's recipe and a downward LR extension | E4 finds no from-scratch signal at 5M | own pre-registration | a few GPU h |
| 42 | 100M curve point | 60M fails and 150M passes | PLAN gate | one curve run |
| 45 | P-186: one 5M seed extended to 20,000 tok/param on the 5070 if the spare hours allow, instead of waiting for a Titan | P-186's pre-registration, before its trunk (after the lock) | branch points fixed before the trunk starts | about 109 GPU h eager by arithmetic |
| 49 | P-068 OPD pilot | before Phase 3e (Nov 16) | pre-registered and scheduled, not "if time" | 20-40 h |
| 80 | Novelty rechecks | Oct 15-20; after Oct 29 | as listed | Claude time |
| 81, 82, 83, 86 | OOD-H Part 2; rater pay; pausing Ultra; release track | January; before recruitment; Jan 10; a size passes | as planned; Max's calls | as listed |
| 84 | Standing cut rules | one fires | its cut list goes to Max first | n/a |
| 94, 95 | E005 linking extent; E006 seeds 6-10 | a GPU gap | new pre-registrations (E006 notes:37 says so); seeds 6-10 read as an independent replication | minutes; est. 5-7 GPU h |
| 97, 98 | LOOKUP diagnostic conditions; P-137 | after the lock, before any Planck score | post-lock addenda | small |
| 106, 109, 110, 114, 116, 117, 127, 128 | VOCAB_OOL option (c), last-line cap, wider repair, Ministral fp8 retry, extra teacher arms, decontamination, R1/R2/RH, diversity caps, Stage B, word-list full read | the real teacher pilot or the teacher pick; in every case decontamination and the caps before the first bulk generation run | named in the pilot's pre-registration; built with tests | code; pilot hours |
| 129, 130 | Simple Wikipedia and Aya; core_v0_pd | core v1 build | an explicit include or exclude | small |
| 132 | K3 extra arms | K3's P-F reads a gain | K addendum | about 0.3-0.5 h each |
| 139, 140 | Half-batch speedup in new configs; fp16 loss scaling, batch ramp, KV decode | new eager configs; before the Dec 7-10 Titan smoke; before Planck's first RC-12 scoring | per pre-registration; tests | code |

## KEEP DROPPED (one line of science each)

- Row 4. S008's 9-layer arm: four things move at equal total, so no verdict could say which mattered; depth is E9's.
- Row 6. K = V tie: its one measured cost (3.1% perplexity) points against it, and value residual just earned its place.
- Row 7. plain_block / nostab: plain_block also removes value residual (REJECT alone); the two-switch arm frees about
  1,024 parameters and no outcome changes the baseline under S002's rules.
- Row 15. Matched-LR IND runs for other arms: moot, every other pick is g 1.
- Row 21. P-109 SAN: parity only at about 105B tokens at 24M, so a 250M-slot screen would REJECT it for the wrong
  reason.
- Row 25 (batch 1 only). Compile for this batch: compiled Canon fails deterministic mode and a lenient re-read would
  be a forking path.
- Row 27. Third parity invocation: it would redraw the floor until one passed.
- Row 28. E3 rule 5 refresh: not triggered while screens run eager.
- Row 30. S003's bits-per-parameter measure: stored knowledge is out of Planck's scope (PLAN.md:53).
- Row 33. AdamW at 30M: REJECT +1.53% holds under Holm, and outside evidence favours NorMuon.
- Row 34. Value-residual-off at 30M: REJECT +1.49% holds under Holm.
- Row 39. E002's 360M block: E002's items admit a wording shortcut, and 360M is far above the 30M target.
- Row 44 (budget rule). 150M at 75B by default: Titan time is external; only the seed text needs fixing.
- Row 50. D6 distillation consequences: they need the shared vocabulary D6 removed; reopen only through P-068.
- Row 51. P-182, P-183, P-103: under a total-parameter count they store per FLOP, not per parameter.
- Row 52. 124M primacy comparison: both arms likely at chance at about 9 tok/param; S001 replaced it.
- Rows 61, 62. P-141 and the Mac-only fallback: the 5070 runs every screen.
- Rows 63-71. REPORT non-candidates and the parked tokenizer, thinking, architecture, memory, data, eval,
  post-training and probe ideas: each has measured or cited evidence against it at this scale, and the narrower
  forms that could help survive as live bets (P-105, P-150, P-034, P-022).
- Row 72. Strict data rulings: licence, provenance and D8 (Max's), not budget; W can return for a later 150M extension.
- Row 73. Sealed size 600: under the S7 bar n barely matters (true S7 45.7 / 45.1 / 44.6 at 600 / 700 / 900).
- Row 74 (rebuild part). LOOKUP easier-cell rebuild: a rebuild after seeing dev floors would be post hoc, and the
  claim uses an absolute bar.
- Row 75. A model-invoked lookup-store claim: a key-level store is gameable (a relay scores about 1.0); table-level
  breaks OD5 and s13.
- Row 76 (rest). Remote-code sub-30M panel models: every sub-30M model scored so far has S7 at most 2.09; Max chose
  "on our panel" (5 b).
- Row 78. Small RC-12 fixes: measured effects of 0 to 2 replies and about 0.005 of T0.
- Row 79. The old Level R test: Qwen2.5 scores S7 2.30, so it cannot tell a strong model from a floor model.
- Row 87. E2 20M extensions: the rule correctly did not fire.
- Row 89. E002's chat probe: E002's models use a shortcut; E004 ran its own probes.
- Row 90. E004 tiny seeds: the 135M FAIL gate; E003's easy version also failed.
- Row 96. E006's other candidates: SmolLM2 fine-tune mechanics that a from-scratch Planck does not inherit.
- Row 98 (F7, F8). Persona and social basics: no generator, grader or cheater gate; ROLE and the human test cover F7.
- Rows 102, 103. OOD-H F3 and its report-only cheater gates: F3 would put OOD-H trees into tokenizer v0 and the
  E2/E3 shards; natural threads let type-guessing cheaters pass structurally, on both sides of a paired check.
- Rows 107, 111, 112, 113, 118, 119, 120, 124, 126. Pipeline items cut on measured or structural grounds (the
  lowercase A/B did not help, the 2,048-token context, false rejects, minors in the persona pool, superseded probes).
- Row 133. Track K narrowings: each has a measured reason (for example warmup stuck 13 of 104 fits against 2 of 200).
- Row 135 (rest). K5's loss mask and noisy-lookup arm: the mask removes the answer's only supervision; noisy lookup
  trains a different policy.
- Rows 137, 138. Harness modes and options: failed parity, memory-only gain, untested on CUDA, or not needed.
- Row 145. Ministral's same-day control in round 1: later pilots use a round-level control.

## Already done

Rows 1 (decision; step 0 installs it), 40, 74, 91, 92, 93, 101, 108, 115, 123, 136, 141, 142.
