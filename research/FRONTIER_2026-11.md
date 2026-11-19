# Planck frontier, November 2026

- **Date:** 2026-11-18. Written for Max while MaxGPT-Planck waits for NAIRR compute.
- **Built from:** six lane reports (architecture, optimization, data, memory, post-training, systems) in this folder, then three verification passes that re-checked 312 claims against arXiv full text, official repos and docs, and the repo (read only). Every claim a verifier marked wrong is corrected or dropped below; section 7 lists what changed.
- **Honesty markers:** (est.) means arithmetic, not a measurement. No H100 number in this file was measured; the first Start-Up job exists to replace them. "NEW SINCE" means first public after 2026-09-23 (the ledger date). "Older but missing" means published earlier but absent from LEDGER.md and research/.
- **Next free ledger id:** P-395 (updated 2026-11-19 with section 10). This line first read P-189 and pointed to a `LEDGER_ADDITIONS.md` beside this file; those rows and every later pass's rows are now in LEDGER.md itself, which holds P-189 to P-394.

---

## 1. Summary for Max (one page)

**What NAIRR changes.** GPU time stops being the limit. The whole Phase 3 plus Phase 4 plan is roughly 400 to 550 H100-hours (est., treat as plus or minus 2x), and one Start-Up allocation is up to 2,000 GPU-hours. The new limits are: the size of the clean chat pool (it does not grow with compute), honest analysis when runs are cheap (false WINs multiply), cluster operations (Duo login, 48-hour jobs, no unattended SSH on DeltaAI), and finding a faculty PI, because the eligibility text names researchers, educators and graduate students with a faculty advisor's letter, and does not mention undergraduates applying on their own.

**The five things that matter most.**
1. **T001 probably measured how fast each arm learns, not what plain attention can do.** Two-layer softmax attention always solves standard multi-query recall (no updates) when trained about 13x longer than T001 trained it. The strongest new evidence is a control inside the Voltic paper: the same plain softmax model that learned ordinary recall but scored 2.96% on volatile recall (values that change) under the shared recipe scored 99.43% when its 12,000 updates were split into 10,000 on an easier one-observation task and 2,000 on volatile recall; 12,000 updates on volatile alone gave 21.05% (one seed of five learned). That is close to the new memory layer's 99.48%, though the control ran 12,000 updates against the main table's 10,000, so it is not a matched comparison. The authors say the result does not establish a representational limit of attention. So a curriculum may do what we credited to architecture. A NAIRR re-run of the toy (T002) settles this in a few GPU-hours and decides what fills the 30M must-run slot.
2. **The optimizer and learning rate were chosen in the wrong regime.** NorMuon beat AdamW by 1.5% at 50 tokens per parameter, on loss, before any SFT. The curve runs at 500 to 5,000 tokens per parameter and is scored after SFT. A CMU paper from Nov 2 finds Muon-pretrained models fine-tune worse than AdamW ones once pretraining passes 80 to 640 tokens per parameter at 30M. Planck's own E2 shows the best LR drifting down with run length (2.1 to 3.2x lower by the curve's lengths, est.). Both must be re-tested at the real horizon, after SFT. This is the item NAIRR makes possible that the 5070 could not.
3. **A teacher that shares Planck's tokenizer is now cheap: a bigger Planck.** A 300M to 1B Planck on the 8k tokenizer costs about 35 to 130 H100-hours (est.) and reopens on-policy distillation, which D6 blocked for outside teachers. It also gives a clean test of your core bet: if small students under distillation approach the teacher's RC-12 score while SFT-only students stay far below, the skill was cheap in parameters and only short of training signal. It needs your ruling on what "from scratch" allows (report curves with and without the teacher).
4. **RL with program-checked rewards on the skeleton generator is untested below 1B for conversation state, and RC-12 is built to keep it honest.** From-scratch studies at about 38M to 150M, on synthetic tasks, show RL works when the model already passes sometimes and the reward is precise. Reward hacking is the main risk; rewarding only answers that are right on every isomorphic twin (order swap, position, wording, alias) is our own proposed defense.
5. **Several levers cost zero or almost zero parameters:** a short procedural or stack warm-up at the start of pretraining, a learnable "sink" logit per head (24 parameters at 5M), and dynamic short convolutions (about 2% of parameters).

**What to do first.**
1. Ask a faculty member to be PI and file a NAIRR Start-Up on an x86 H100 site (Purdue Anvil AI or SDSC Expanse AI). Decision in about 2 to 3 weeks.
2. While waiting (no GPU): pre-register T002; read the optimizer "margin-to-spread" score off the existing S003 and E3 checkpoints; run the cross-tokenizer coverage probe on CPU; simulate the false-WIN rate of the screen rules on E3's null pairs; build Apptainer images and a job-of-jobs runner.
3. First Start-Up week: calibration only (throughput, parity, noise re-measured on the new GPU). Every 5070 noise number is hardware-specific.
4. Then T002, then the long-horizon optimizer and LR trunks, then batch 2 screens with multiplicity control.

**Freeze rule.** The curve freezes Dec 4. Anything here that changes the curve recipe needs a dated amendment before its results exist. NAIRR re-tests of S001 to S007 are new pre-registered experiments; they never overwrite the 5070 verdicts.

**Out-of-lane flag.** Textile Labs published a "Loom-Planck" model ladder (300K to 1.2M parameters, created 2026-11-01, MIT licence) that collides with the project name.

---

## 2. Re-tests of our own results

Each item: what we concluded, what the new evidence says, and the cheapest decisive test. All costs are NAIRR estimates.

**R1. T001: "plain softmax attention never learns multi-query recall with updates."**
- Evidence for "slow, not unable": (a) 2-layer attention "always solves" MQAR at about 50 epochs x 100k sequences ([2508.19029](https://arxiv.org/abs/2508.19029)); T001 gave each run 12,000 steps x batch 32 = 384k sequences. (b) Induction-head emergence time is bounded by a quadratic in context length ([2511.01033](https://arxiv.org/abs/2511.01033)); T001 rows reach 865 tokens at n=96. (c) Voltic Appendix E.6 / Table 8 ([2610.05700](https://arxiv.org/abs/2610.05700)): plain softmax scores 99.98 on standard MQAR, so it learns recall; it fails the volatile and stochastic variants (2.96 / 3.39 / 3.16 in Table 1; 21.05 +/- 18.22 on volatile after 12k target-only updates in the Table 8 control), but reaches 99.43 +/- 0.12 after 10k steps on a one-observation task then 2k on volatile. Table 8 is a separate diagnostic with 12,000 updates against Table 1's 10,000, and the authors say it does not establish a representational limit of full attention. (d) Every T001 learner had a previous-token path at init or one parameter away; the failed recency arms (fg at half-life 865, ALiBi at slopes 2^-4 and 2^-8) were nearly flat at init.
- Test (inside T002, M1): base at 12k, 50k and 200k steps; base with a short-first n schedule; base with the Voltic warm-up recipe; base with attention-map distillation from a solved arm for the first 30% of steps then switched off. Kill: base fails at 200k steps AND with the warm-up, which restores the capacity reading. Cost: about 3 to 8 H100-h (est.).

**R2. S005 (forget gate, NOMINATE at -1.2%) vs FoX as published.**
- Facts: S005 and T001's fg arm started the gate nearly closed (S005: w=0, b=7.99, half-life 2,048 tokens). FoX's code zeroes the gate bias (f about 0.5) and FoX "Pro" adds a KV-shift (360M ablation: 6.62 with it, 6.80 without; [2503.02130](https://arxiv.org/abs/2503.02130)).
- But S005's own result weakens the init argument at 5M: the trained gates went local anyway (median per-head half-life 12.6 and 7.1 tokens across the two seeds, only 7 and 6 of 24 heads above 100 tokens; "NEAR ITS INIT" did not fire). The open questions are the KV-shift and the toy (fg stayed at chance in T001, and no T001 weights were saved to read its gates).
- Test: T002 fg arm at paper init, with and without KV-shift (minutes). A 5M S005b only for the KV-shift arm, or if the toy flips. On NAIRR use a Flex score_mod or the fla FoX kernel: S005's mask engine cost +1.33 GiB peak and 2.26x the time of its own BASE (about 85k vs 192k tok/s at micro 8 x accum 2). Cost: under 1 H100-h (est.).

**R3. S003: NorMuon beats AdamW (+1.53% F(CHAT) for AdamW).** Measured at 50 tok/param on bpb only, before SFT, with a registered caveat that AdamW's LR pick rested on one seed. New evidence ([2611.03633](https://arxiv.org/abs/2611.03633)): past 80 to 640 tok/param at 30M, AdamW-pretrained models beat Muon-pretrained ones after fine-tuning. Porian et al. ([2406.19146](https://arxiv.org/abs/2406.19146)) give an AdamW optimum at exactly 5M (LR 0.013, batch 20, beta2 0.99; beta2 0.95 was poor at small batch), a useful prior for the AdamW arm. Free first test: compute the margin-to-spread ratio (MSR) on S003 and E3 checkpoints. Decisive test: M3. Cost: 0 GPU-h for MSR; M3 for the rest.

**R4. E2's LR and WD carried to the curve.** E2 Q3 measured the LR optimum dropping 0.18 to 0.24 log2 per doubling of run length (5M and 20M); the curve is 6-7 doublings past E2's longest runs, so about 2.1 to 3.2x lower (est.). At eta 3e-3 and WD 0.1, the WD timescale is 3,333 steps: 44% of a 5M screen, but only about 1.7 to 3.4% of a 25.6B-token run at PLAN 5's grown batch (98k to 195k steps, est.). The planned check (P-186 decay branches) cannot test the stable-phase LR. Test: M4. Cost: about 30 to 40 H100-h (est.).

**R5. "Screen verdicts were read at one LR."** Corrected: this is mostly false. SCREENS C3 gives every arm except BASE a seed-1 g-check at g in {0.5, 1, 2} times all three LR5 LRs; S003 had its own staged search; picks come from seed 1 and verdicts from seeds 101-102, so a selection/estimation split already exists. What survives: the grid is coarse (factor 2), uses one selection seed, scales all LRs together, and pins BASE at g 1. Test: a half-octave grid with 2 to 3 selection seeds for the two NOMINATEs (S004, S005) and the one structural REJECT (S002), then fresh estimation seeds. Cost: about 5 H100-h (est.). Expect few flips.

**R6. Every noise and parity number is 5070-specific.** E3's seed SD, paired SD_d, the nondeterminism share (point 0.78, 95% interval 0.21 to 30.8) and compile parity were all measured on one 5070 with torch 2.13. Also, E3's "data-order noise is larger than init noise" is not established: sigma_data 0.375% vs sigma_init 0.203% on F(CHAT), but the variance ratio 3.43 sits inside the 95% F(4,4) range (upper end 9.60), as E3 itself says. Test: E3-N on the target GPU (32 pairs at 5M, 16 at 20M, a 4x4 crossed init x data family, 3 replays per seed) plus an 8-seed cross-hardware check of the E3 A-arm on the 5070 and the H100. If the mean differs by more than the paired MDE, S001 to S007 become priors to re-confirm, not verdicts. Cost: about 17 H100-h (est.).

**R7. Cost figures in plans and lanes.** Several lane costs used 255k tok/s, the eager 5M rate. Compile has been the default since the DEFAULTS DECISION of 2026-09-28 (562 to 617k tok/s at 5M), so 5070 costs in the lanes are about 2x too high. None of this changes a NAIRR plan; it matters only for what still runs on the 5070.

**R8. S001 (attention gate off: TIE).** That screen computed neither HD nor sink mass, so it says nothing about sinks. At 10M, almost all of a value gate's gain is "abstention", and a single learnable sink logit per head delivers it ([2609.22005](https://arxiv.org/abs/2609.22005); 10M loss 4.2119 baseline, 4.1934 sink logit, 4.1897 both). Test: M11. Cost: under 1 H100-h (est.).

**R9. S004 Canon (NOMINATE, onset unresolved) and S007 smeared key (TIE).** If T001 measured speed, the previous-token advantage should shrink at Planck's real token budgets while the onset advantage stays. Also: dynamic-conv authors find static conv gains little when QK-norm is on (their Table 3c), and BASE has QK-norm. Test: Canon and smear arms at decay branches of 250, 1,000 and 2,000 tok/param off shared 5M trunks, read on bpb and (after the lock) RC-12 dev margins, plus loop and echo rates after identical SFT (M19). Cost: about 6 to 10 H100-h (est.).

**R10. S006 (t+2 MTP head) REJECT at one pass.** Chen et al. ([2606.16246](https://arxiv.org/abs/2606.16246), 150M, 100 epochs over 75M tokens) find offset targets help when data repeats many times, but only in one form: exponentially weighted offsets anchored on t+1 with a prepended per-sample offset token, plus 50% right-to-left prediction. Noise plus offset without that interfered (mask 5% plus offset 3.937, worse than offset alone 3.890), and uniform weighting gave nothing. S006's head is a different mechanism. Test: only Chen's form, only inside P-187's 100- and 200-pass cells. Cost: about 1 H100-h (est.).

**R11. E004 "never stops after its answer".** That came from SFT targets that never trained end-of-turn, and E005 fixed it (640/640). The new "OPD does not teach stopping" result ([2609.37326](https://arxiv.org/abs/2609.37326)) is specific to thinking mode; non-thinking students kept stopping. Keep "train end-of-turn by SFT before any OPD or RL" as a guard, without the E004 link.

**R12. What a 5M model should be able to bind.** A 4.98M transformer trained directly on variable binding holds about 3.7 bindings, and formation time grows as about K^2.1 to K^2.6 ([2609.30634](https://arxiv.org/abs/2609.30634)). No Planck checkpoint has been measured this way. Test: the binding-capacity probe (M17) on existing 5M checkpoints. Cost: inference only.

---

## 3. The master list

Ranked by expected value for Planck's claim (the smallest model that passes RC-12). "Lanes" names where the idea came from. Costs are H100-hours (est., plus or minus 2x) assuming B1's throughput estimates: a 5M screen at 0.25B tokens is about 0.03 h, a 5M run to 2,000 tok/param about 1.2 h, a 20M screen about 0.44 h. Limits are LEDGER's #1 to #10.

### Tier A: decide first (cheap and decisive, or blocking)

**M1. T002: the update toy that separates "cannot" from "slow".** Lanes: arch 1, memory 1, data H1/H3, optim 5.
- Mechanism: rerun T001's harness with a previous-token path given to every arm so the base learns recall, then vary only the update mechanism; add budget and curriculum arms; add chat-shaped families.
- Arms: base at 12k/50k/200k steps; short-first n; Voltic warm-up recipe; procedural (k-Dyck/Stack) and update-stream warm-ups on disjoint token ids (T001's vocab is 228 synthetic tokens, so warm-up symbols need new ids); attention-map distillation; FoX at paper init with and without KV-shift; stick-breaking plus smear; PaTH via fla; GDN; Voltic; one LEMA head; CoPE; PoPE; dynamic conv; MTA's key-query conv; a sink logit; a hard-wired copy operator; ChaCAL in layer 2.
- Families: StateBench aperiodic tier (mutable assignment); volatile recall; DKI-style first-vs-latest by update count; re-introduced old values; multi-token same-type keys with paraphrased restatements; flip-flop; indirect indexing as a LOOKUP proxy; 1 to 3 corrections per key; an early rule token applied after distractors; reference chains.
- Evidence: T001 (115 runs, 7.89 h on the 5070); R1's sources; StateBench ([2611.03094](https://arxiv.org/abs/2611.03094), 275M: GDN generalizes perfectly on aperiodic, transformers fail beyond training length, 7B models about 90%).
- Ledger: NEW (own design; extends P-020, P-022, P-024, P-047, P-148). Voltic and StateBench NEW SINCE; LEMA one day before the cutoff.
- Limit: #1, #2. Decides P-022's 30M slot on update evidence, not "learns recall at all".
- NAIRR: about 400 runs become a few hours on one node.
- Test rules: 5 seeds, per-arm LR picks on a selection seed, per-distance STALE/FIRST counts and saved weights (both missing in T001), pre-registered per family.
- Cost: about 10 to 25 H100-h (est.). Confidence: high value; risk is toy-to-chat transfer.

**M2. NAIRR Start-Up plus the calibration package.** Lanes: systems A, B1, B2, D2.
- Mechanism: get access, then measure before deciding anything: bench_micro at 5/20/30/150M (eager vs compile, varlen backend, batch 8-128, K concurrent models); MPS and seed-stacking at K = 1-16; attention backend (varlen Flash vs cuDNN vs Flex+FA4) with the bitwise leak probes; static cu_seqlens plus CUDA graphs; real loader vs synthetic rows; driver and internet checks; E3-N and the cross-hardware check (R6); a 3-point E2 LR check.
- Evidence: Planck compiled 5070 rates (5M 562-617k tok/s); H100 estimated at 3-5x a 5070 at 5M and 8-11x at 150M because tiny GEMMs are bandwidth-bound (est.); HFTA fusion up to 15.1x for repetitive jobs ([2102.02344](https://arxiv.org/abs/2102.02344)); record #92 graph capture worth about 4-5% at 124M-class.
- Ledger: baseline row "Measure throughput on every machine"; the H100 part, packing and crossed seeds are NEW (older but missing).
- Limit: all verdicts read E3. NAIRR: this is the NAIRR item.
- Cost: about 30 to 40 H100-h (est.), plus about 300-500 lines for seed stacking. Confidence: high.

**M3. Long-horizon optimizer bake-off, scored after SFT.** Lanes: optim 1, H1; systems C1 (Porian prior).
- Mechanism: past a pretraining-data threshold, Muon models forget more and adapt worse when fine-tuned (CMU's margin-to-spread explanation). Separately, SOAP and NAdamW beat Muon at 16x Chinchilla.
- Evidence: [2611.03633](https://arxiv.org/abs/2611.03633) (NEW SINCE, 30M-600M, about 600 pretraining and 29,000 fine-tuning runs; 30M crossovers: GSM8K 80, Alpaca 160, StackMathQA 320, MusicPile 640 tok/param); Wen et al. [2509.02046](https://arxiv.org/abs/2509.02046) (130M at 16x Chinchilla: NAdamW and SOAP beat Muon); [2611.02529](https://arxiv.org/abs/2611.02529) (about 90% of Muon's gain vanishes once LR x WD is tuned per method, 1.9-15M MoE); [2610.21555](https://arxiv.org/abs/2610.21555) (17.7M/33.6M LMs plus a 2.70M ViT: Muon wins all 20 pairs at equal data, AdamW wins time-normalized). No paper measures multi-turn state skills. The CMU authors note fine-tuning close to the pretraining data "is simply continued pretraining", and Planck has chat in pretraining from step 0.
- Ledger: P-119 and baseline row 2; the post-SFT reversal is NEW SINCE; SOAP older but missing.
- Limit: #3, #4; the claim is scored after SFT. NAIRR: decisive (5M to 2,000 tok/param is about 1.2 H100-h per arm vs about 5 h compiled on the 5070).
- Test: free MSR read first (R3). Then 5M trunks to 2,000 tok/param: NorMuon, AdamW (cautious WD, seeded from Porian's 5M optimum), SOAP, NorMuon+Hyperball (M13); each with its own 5-point LR x WD design; decay branches at 250/1,000/2,000; one identical short SFT per branch; score bpb, Tier-0 margins, MSR, RC-12 dev after the lock; 2 seeds at each pick.
- Cost: about 40 to 50 H100-h (est.). Confidence: medium-high that it matters, low on direction at 5M.

**M4. LR and weight decay tuned at the real horizon.** Lanes: optim 2, H6; systems C3 (LR law).
- Mechanism: the optimal LR falls as a power of training tokens, and WD's timescale should shrink as a fraction of the run.
- Evidence: Planck E2 Q3 (above); HyperP [2603.28743](https://arxiv.org/abs/2603.28743) (208M, 10.4-166.4B tokens: eta* = 24.27 T^-0.320; Muon at the wrong WD loses 0.021 nats); Power Lines [2505.13738](https://arxiv.org/abs/2505.13738) (111M+, 20-1,280 TPP); the sub-59M recalibration [2609.27581](https://arxiv.org/abs/2609.27581) (posted on the ledger date; AdamW, D/N up to 600: Step Law transfer overestimates LR by about 4x).
- Ledger: baseline rows "Tune LR at 5M" and "WD re-derived for run length"; HyperP older but missing.
- Limit: #4; protects every curve point and P-186's reading. NAIRR: three full-length trunks instead of one trunk plus branches.
- Test: 5M trunks to 2,000 tok/param at g in {0.35, 0.5, 1} x WD factor {0.3, 1, 3} (9 runs or a 5-run design); fit eta*(D) with E2's points; predict 20M at 2,000 and check with 2 runs.
- Cost: about 30 to 40 H100-h (est.). Confidence: high that the optimum moves, medium on how much loss it costs (decay to zero repairs some).

**M5. Multiplicity, blinding and confirmation rules for cheap runs.** Lanes: systems C7, H5.
- Mechanism: Benjamini-Hochberg FDR q=0.1 at stage 1 with strict confirmation on fresh data seeds; blinded result files with a frozen, mutation-tested analyzer; diverged runs count as failures; confirmation seeds derived from the hash of the locked pre-registration commit.
- Evidence: modded-nanogpt's "no leg is discarded" practice; [2606.27687](https://arxiv.org/abs/2606.27687) (6 of 7 hacked configurations failed to carry to the next model); anytime-valid tests [2210.01948](https://arxiv.org/abs/2210.01948).
- Ledger: baseline row "Pre-register decision rules"; the four parts are NEW (older but missing). Limit: measurement. NAIRR: needed because of NAIRR.
- Data-arm rules (data lane idea 7, BabyLM 2026 findings and BabySteps, where validation loss ranked the best configurations poorly): no data arm is adopted on bpb alone (P-004 margins plus Tier 0 decide), and every data arm gets its own LR check.
- Test: simulate the false-WIN rate per 50-arm batch on E3's real null pairs, with and without FDR. Cost: 0 GPU. Confidence: high.

### Tier B: the high-value bets

**M6. Same-tokenizer Planck teacher ladder, then multi-turn on-policy distillation (OPD).** Lanes: data 3 and H5, post-train 3 and H4, optim note.
- Mechanism: train a 300M-1B Planck on the 8k tokenizer and the Planck mix; SFT it, optionally RL it (M7); the student samples assistant turns inside skeleton conversations and minimizes per-token reverse KL to the teacher. Arms: teacher takeover on low-confidence turns (UOPD); a teacher-assistant chain down the curve (150M to 30M to 10M); LFM3-style per-family specialist teachers; a same-size twin that sees the gold state note (privileged view, no capacity gap).
- Evidence: none for chat below 1B. OPD efficiency at 8B ([Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)); LFM3 distilled six specialist RL teachers (Oct 27, NEW SINCE; smallest 1.7B); UOPD [2609.34036](https://arxiv.org/abs/2609.34036) (up to +15.8% relative on WebShop); SynthSelf [2610.13113](https://arxiv.org/abs/2610.13113) (7B/14B aligners for 0.5-3B students, ratios about 2.3x to 28x, beat SDFT and SFT). Guards in M15.
- Ledger: NEW. Related P-068 (not run; no experiment directory), P-069, P-070, P-168. A Planck-family teacher falls outside D6's scope (D6 gave up same-tokenizer KD from Ultra and SmolLM2 only); it does not meet D6's revisit condition, which is about the SmolLM2 OPD pilot.
- Limit: #5, #1, #8, OWN. Also measures the capacity floor per skill (teacher-gap decomposition).
- NAIRR: from impossible to cheap. Test: 400M teacher on 20B tokens; 10M student, 3 seeds: SFT vs OPD vs OPD-then-RL; read RC-12 dev on own history vs golden history, plus loop rate; repeat at 5M and 30M per family.
- Cost: about 35 H100-h teacher plus 10-20 for arms (est.); a 1B teacher about 130. Confidence: medium. Risks: the teacher fails the same cells; claim integrity (needs your ruling).

**M7. RL with checkable rewards on the skeleton generator, with hacking defenses from day one.** Lanes: post-train 1, 2, 9, 12, H1, H2.
- Mechanism: turn-level REINFORCE or GRPO on the model's own 12-turn histories, graded by RC-12-style graders adapted to training templates; user turns are fixed in advance so all G rollouts share them; difficulty settings keep each cell's pass rate near 0.3-0.7. Controls: continued SFT, rejection fine-tuning, shuffled rewards.
- Defenses: training graders stricter than eval graders; reward paid only when every isomorphic twin is right (order, position, wording, alias); fuzz the training grader with the 9 RC-12 cheaters before any GPU time; KL or judge guard on filler turns; a hacking-gap audit the RL never sees (independent grader, human or Claude reading samples, OOD-H); a pre-registered stop rule. D8: Claude is never the reward; any learned judge is Apache-2.0.
- Evidence: Interplay [2512.07783](https://arxiv.org/abs/2512.07783) (100M from scratch: RL lifts pass@128 only with headroom; about 1% pretraining exposure to a context needed for transfer; a third-party reproducibility log reports process verification cut OOD pass@128 from 0.428 to 0.031, so treat process rewards with caution); Saxe lab [2607.07646](https://arxiv.org/abs/2607.07646) (about 38M from scratch: GRPO solves what the base rarely solves; rejection fine-tuning plateaus with shortcut rewrites; this is the only head-to-head RFT vs RL result); [2607.25091](https://arxiv.org/abs/2607.25091) (PPO works at 70M-360M when SFT is fluent; fp32 ratios, weight rollback); Helff [2604.15149](https://arxiv.org/abs/2604.15149) (isomorphic verification removes shortcuts); Ray [2606.01066](https://arxiv.org/abs/2606.01066) (buggy verifiers hacked to near 100% reward). [2611.14912](https://arxiv.org/abs/2611.14912) (NEW SINCE, Qwen2.5-7B/Qwen3-8B) finds SFT on rollouts from the RL-tuned model or Qwen3-14B, with about 98k incorrect added to 39k correct, generalizes like RL; it is not about the starting model's own samples.
- Ledger: NEW (the parked "RL for open chat below 1B" does not cover verifiable state tasks). Closest P-067, P-014, P-077.
- Limit: #1, #2, #5, #6. NAIRR: buys seeds; the real cost is a batched KV-cached sampler with rolling conv state (500-1,000 lines).
- Test: post-SFT 5M and 10M, 3 seeds, arms (a)-(d) above plus per-item vs twin-set reward; metrics: generator pass rate, RC-12 dev, E004 held-out structures, loop rate, OOD-H, shortcut alignment. Then 30M on winners; mix human-sourced user turns (P-154) so RL does not overfit template users. Compute split (post-train 9, Interplay: at fixed compute, mid-training plus light RL beat heavy RL on near-OOD tasks): start the RL arm from both a plain-decay trunk and a P-058 chat-anneal trunk, and keep human-like user turns at about 1% or more of chat tokens.
- Cost: about 20-40 H100-h at 5-10M, about 50 more at 30M (est.). Confidence: medium on dev gains, low on OOD-H transfer.

**M8. Zero-parameter warm-ups: procedural, stack, and update-stream.** Lanes: optim 5, data 6 and H1, memory 7.
- Mechanism: before natural text, train briefly on generated structured sequences. Variants: k-Dyck/Set/Identity (0.1-0.3% of tokens); a 1% stack task (push/pop then final contents); our own update-stream task where the only solution is "read the latest write to this key".
- Evidence: Jiang et al. [2601.21725](https://arxiv.org/abs/2601.21725) (2-layer d16 model: haystack recall about 10% to 98%; at 124M the same loss with 55/67/86% of the data on C4/CodeParrot/DM-Math; shuffled data gives nothing); stack warm-up [2609.38764](https://arxiv.org/abs/2609.38764) (NEW SINCE; SmolLM-135M/360M shapes at about 20 tok/param: MuSiQue F1 38.10 to 41.99, HotpotQA 41.34 to 43.54; a queue task "fails to produce the same gains"; stack training after pretraining hurts four-hop accuracy); Hu et al. [2502.19249](https://arxiv.org/abs/2502.19249) (1B, 33% fewer tokens); against: BabyLM 2026 findings and Veitsman et al. [2609.11505](https://arxiv.org/abs/2609.11505) (lower loss did not consistently give better linguistic scores).
- Ledger: NEW (no procedural, Dyck or formal-language entry); stack paper NEW SINCE.
- Limit: #1, #2, two-hop, LOOKUP. Legal under D1 (no parameters). NAIRR: small effect; cheap anyway.
- Test: toy arms inside M1 first; then 5M on a P-186 trunk: none / k-Dyck 0.3% / stack 1% / update streams 0.2% / queue 1% (control), 2 seeds, scored on bpb, onset, Tier 0, StateBench tiers, then RC-12 dev after one SFT.
- Cost: about 2-4 H100-h (est.). Confidence: medium on the toy, low-medium on chat.

**M9. Dynamic short convolutions on Q, K and V (Canon's successor).** Lanes: arch 2, memory 6.
- Mechanism: per-position depthwise causal filters (width 4) computed from the normed input, added residually to Q, K and V before RoPE.
- Evidence: [2606.03825](https://arxiv.org/abs/2606.03825) (v2 2026-10-26; fla kernel 2026-11): synthetic average 0.709 / 0.793 / 0.822 (plain / static / dynamic), fuzzy recall 0.298 / 0.505 / 0.726; 300M at 15B tokens perplexity 19.12 / 18.66 / 18.01 (low-rank) / 18.22 (per-head), RULER-4k 22.6 / 19.9 / 27.3 / 28.5; static conv gains little with QK-norm while dynamic still helps (Table 3c); at 1.8B low-rank falls below static on RULER. Nothing below 150M except inside the scaling fit.
- Ledger: NEW (older but missing; v2 and kernel NEW SINCE). Family of P-148 (S004) and P-024.
- Limit: #2 (multi-token keys), #1. Parameters at d=192 (est.): per-head about 111k (2.2%), low-rank R=16 about 369k (7.4%); use per-head and let the budget solver shrink the MLP.
- Test: S022 at 5M: BASE vs Canon-AC vs static Canon-B vs per-head dynamic, matched totals, 5 seeds, plus the T002 multi-token-key family.
- Cost: about 2 H100-h (est.). Confidence: medium.

**M10. Teacher pilot at NAIRR scale: larger Apache-2.0 renderers and judges for R0.** Lanes: data 1.
- Mechanism: rerun the pre-registered Phase 1 pilot (same 200 skeletons, checker, frozen banks_v2) with larger teachers on vLLM; pick by accepted Planck tokens per GPU-hour inside the naturalness gate; add a large judge. Keep R1/RH rephrasing on 1-8B models.
- Evidence: Planck's own probe accepts 26 / 11 / 8 per 100 (Gemma 4 12B, Ministral 3 8B, Qwen3.5-9B); judge v2 catches 4 of 10 off-topic lines. Rephrasing quality stops improving above about 1-8B ([2604.13977](https://arxiv.org/abs/2604.13977), [2510.01631](https://arxiv.org/abs/2510.01631)). Licences checked on the HF API: Olmo-3.1-32B-Instruct, Gemma 4 31B, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B and gpt-oss-120b are tagged Apache-2.0 (gpt-oss's usage policy needs your read first).
- Ledger: baseline "Teacher policy" and PLAN Phase 1; no P-id; NEW as affordable now.
- Limit: #3, #8; the pool is the binding constraint after NAIRR. NAIRR: renders no longer limited by the 5070.
- Test: arms Gemma 4 12B (control), Gemma 4 31B, Qwen3.6-35B-A3B, Qwen3.5-122B-A10B, Olmo-3.1-32B; accept rate per event kind and distance bin, naturalness read, AI-ism and dash rates, judge agreement with your labels.
- Cost: tens of GPU-h (est.). Confidence: high that yield rises, medium on cost per accepted token. Risk: one teacher's voice; keep the 30% second-teacher rule.

**M11. Abstention: a learnable sink logit per head.** Lanes: arch 6, H5.
- Mechanism: a phantom zero-value key with a learnable logit gives each head a no-op option, removing the position-0 sink.
- Evidence: [2609.22005](https://arxiv.org/abs/2609.22005) (2026-09-18, older but missing; v2 2026-09-29): 10M loss 4.2119 / 4.1934 (sink) / 4.2006 (projection gate) / 4.1897 (both); at 124M sink mass 6.0% to 0.5%; at 350M (v2, baseline 3.0293) sink logit -0.0103 +/- 0.0084, sink plus norm gate -0.0187, sink plus projection gate -0.0193, so the gain is not monotone in scale. Loss only. fla added GPT-OSS-style sinks (2026-04). Caution: sinks are needed for trigger-conditional tasks without a no-op path ([2603.11487](https://arxiv.org/abs/2603.11487)).
- Ledger: NEW arm. The sink-to-primacy hypothesis itself is already P-021 (2510.10276, SinkProbe).
- Limit: #1 (primacy, first-value-wins), #6. 24 parameters at 5M.
- Test: 5M arms sink / sink+gate / gate (BASE), log sink mass; add as a third arm to P-021's 30M primacy test with FIRST-error counts.
- Cost: under 1 H100-h at 5M (est.). Confidence: medium on loss, low on behavior.

**M12. Gold-note privileged self-distillation.** Lanes: post-train 5, H3.
- Mechanism: the teacher is the same frozen model with the program-written gold_note ("key: value (was old)") inserted before its reply; the bare student matches it by reverse KL on its own rollouts. The privileged view is a distance-1 copy, not in-context learning.
- Evidence: OPSD [2601.18734](https://arxiv.org/abs/2601.18734) (Qwen3-1.7B: base 37.1, SFT 35.8, GRPO 37.7, OPSD 43.4); CCOPD [2605.30251](https://arxiv.org/abs/2605.30251); at 1.7B most of the gain is distillation itself, a reference adds about 1.3 points, not significant after correction ([2609.20612](https://huggingface.co/papers/2609.20612)); SDFT failed at 3B where in-context learning was too weak ([2601.19897](https://arxiv.org/abs/2601.19897)).
- Ledger: NEW arm of P-071, gated by P-005. Limit: #1, #2, #6.
- Test: gate on P-005 (note-in-context well above bare on a 5M/10M SFT checkpoint); then OPSD vs continued SFT, 3 seeds, on CORR, BIND, RECALL, TWOHOP.
- Cost: about 2-5 H100-h per size (est.). Confidence: low-medium.

### Tier C: strong supporting bets

**M13. Weight-norm control (Hyperball/MuonH, MuonGS).** Lanes: optim 3.
- Mechanism: fix each hidden matrix's Frobenius norm at init and normalize the update (WD drops out), or remove only the radial part of each Muon update (MuonGS).
- Evidence: Hyperball [2606.16899](https://arxiv.org/abs/2606.16899) (130M at 1B-128B tokens: asymptote 3.065 vs 3.079; LR drift across depth about 1.4x vs 4x); HyperP (208M: MuonH at WD 0 2.475 vs tuned Muon 2.479); MuonGS [2610.37647](https://arxiv.org/abs/2610.37647) (NEW SINCE, 160M/400M, 7-15% token-equivalent over MuonH). Mixed replication: nanochat PR #498 saw "worse than baseline" at d12. Loss only.
- Ledger: NEW (Hyperball, HyperP older but missing; MuonGS NEW SINCE). Limit: #4, LR transfer from 8 to 29 layers.
- Test: one flag in optim.py plus a mutation test that norms stay fixed; a 5M screen with its own g-check (about 1.4x lower LR), with and without norm_scaling; the long arm in M3.
- Cost: under 1 H100-h plus M3 (est.). Confidence: medium.

**M14. Decay-branch soups across data orders, and moderate decay for the chat anneal.** Lanes: optim 4, H4; data 8.
- Mechanism: run the decay phase K times off one trunk with different data orders and average the weights; when the best data comes last (P-058), decay to about 0.2-0.3x peak plus tail averaging instead of to zero; give every curriculum or ordering bet this control arm.
- Evidence: OLMo 2 [2501.00656](https://arxiv.org/abs/2501.00656) (three 50B-token anneals with different data orders, averaged); Luo et al. [2511.18903](https://arxiv.org/abs/2511.18903) (1.5B, 30B tokens: curriculum gains vanish under aggressive decay; moderate decay plus averaging +1.64%); record #92 TailEMA with LR floor 0.20. None on state skills.
- Ledger: older but missing; touches P-026, P-058, S017 and the PLAN's last-3-5-checkpoint average. Limit: #3, #4; noise.
- Test: 5M, 3 data-order branches off one trunk: each branch vs soup vs last-5 average on F(CHAT), Tier 0, SD; P-058's anneal at decay-to-zero vs 0.2x plus tail EMA.
- Cost: about 0.5-2 H100-h (est.). Confidence: high on lower variance, medium on beating the best branch.

**M15. OPD and RL design guards.** Lanes: post-train 4, 10, 11.
- Rules: train end-of-turn by SFT first; no thinking mode; penalize RUNAWAY; check per-cell pass@16 of the SFT student before OPD (distilled pass@1 never exceeded the starting student's pass@n in 27 of 27 settings, [2609.37326](https://arxiv.org/abs/2609.37326)); stop or restart rollouts at the first wrong turn ([STRIDE 2609.14636](https://arxiv.org/abs/2609.14636)); clip advantages against length gaming; do not soup RL checkpoints by default, run an eval-only interpolation sweep alpha 0-1 ([2606.18521](https://arxiv.org/abs/2606.18521)); keep P-014 as written and add an online regenerated-pairs arm only if it wins.
- Ledger: updates P-068, P-069, P-014 and the baseline soup row. Cost: 0 to minutes. Confidence: high (evidence at 0.6B+).

**M16. Antidoom: preference optimization at the loop-onset token.** Lanes: post-train 6.
- Mechanism: detect loops in own samples; at the loop-starting position, prefer up to 20 of the model's own top-k alternatives over the loop token; stop at chosen_win 0.35; second round for newly exposed loops.
- Evidence: [Liquid AI, 2026-07-07](https://www.liquid.ai/blog/antidoom) (LFM2.5-2.6B loops 10.2% to 1.4%; Qwen3.5-4B greedy 22.9% to 1%). Nothing below 2.6B.
- Ledger: older but missing; child of P-066 and P-153. Limit: #5, LOOP, Level A loop bar.
- Test: SFT 5M/30M, one vs two rounds vs P-066 vs P-153, 3 seeds; loop rate over all replies plus collateral damage on other families. Cost: about 2-5 H100-h (est.). Confidence: medium.

**M17. Binding-capacity probe and a K-curriculum.** Lanes: memory 8, H2.
- Evidence: [2609.30634](https://arxiv.org/abs/2609.30634) (NEW SINCE): zero-shot capacity scales as N^0.82 over 125M-2.8B (R^2 0.73), predicting 0.14 bindings at 5M, yet a 4.98M model trained on the task holds about 3.7; formation time about 766 K^2.58 or 667 K^2.12; a 256-token distractor block halves capacity (K=8 recall 0.54 to 0.20).
- Ledger: NEW; feeds Track K, P-186, P-026. Limit: #2.
- Test: probe K=1-16 with and without distractors on every 5M/30M checkpoint; two P-008 generator schedules (K ramp 1 to 8 vs uniform) at 5M. Cost: inference plus about 1 H100-h (est.). Confidence: medium-high as calibration.

**M18. Data rules from mechanism papers.** Lanes: memory 9, H4; data (Tang).
- Rules: (a) re-introduction traps: after "not Rex, it's Max", later bind "Rex" legitimately to another holder (Tang et al. [2605.30233](https://arxiv.org/abs/2605.30233): re-introduction fails 0.62 at 13B vs 0.01 at 70B; REMOVE uses a global suppression tag); (b) many-entity contexts with targets in middle positions (Gur-Arieh [2510.06182](https://arxiv.org/abs/2510.06182): positional binding governs only first and last positions); (c) large slot pools with no role-position bias, and score P-059 on held-out-value binding, not loss (Li and Webb [2611.13270](https://arxiv.org/abs/2611.13270), NEW SINCE: data diversity drives binding mechanisms); (d) suppression of non-targets matters more than sharpening attention (Gong [2610.21176](https://arxiv.org/abs/2610.21176), NEW SINCE).
- Ledger: updates P-008, P-009, P-025, P-059. Limit: #1, #2. Test: cells in the generators and in T002. Cost: near 0 GPU. Confidence: medium.

**M19. Loop and echo guard on every previous-token arm.** Lanes: memory 13, H3.
- Mechanism: arms that speed induction (smear, Canon, dynamic conv) may raise echo, answering-as-the-user and loops.
- Evidence: Okulska et al. [2609.15045](https://arxiv.org/abs/2609.15045) (1B-13B: ablating 16 copy heads cuts echo, SmolLM2-1.7B 1.97% to 0.04%, random 16 heads 1.47%); Vera Zuniga [2609.29507](https://arxiv.org/abs/2609.29507) (Pythia funnels into repetition, RWKV and Mamba do not).
- Ledger: NEW guard (relates to P-153, P-066). Limit: #5, LOOP. Test: free-running loop and echo rates on RC-12 dev fillers after identical SFT for S004, S007, S018-S020 and M9. Cost: inference only. Confidence: medium.

**M20. Long-horizon batch size and the critical batch size.** Lanes: systems C3, H3; optim 9.
- Evidence: [2609.27581](https://arxiv.org/abs/2609.27581) (below 59M, AdamW, D/N up to 600: B*(D) = 3.6e-4 D^0.931 tokens, independent of N); at D = 0.25B B* is about 24k tokens (near E3's 32,768); at 25.6B (the 5M curve point) about 1.8M tokens, 7-14x PLAN 5's grown batch (extrapolation far outside the fit). Merrill [2505.23971](https://arxiv.org/abs/2505.23971) (CBS grows quickly then plateaus; batch ramp saved 43% of steps at 1B). This applies to the 5M and 10M points (5,000 tok/param); 20M is at 2,000, 30M at 1,000-2,000, 60M and 150M at 500-670.
- Ledger: baseline batch row and P-106; the measurement is missing. Limit: #4; P-186's slope. NAIRR: the large-batch arm only runs there.
- Test: 5M trunk to 2,000 tok/param at 64k / 512k / 2M-token batches, 3 LRs each, WD per Power Lines, 2 seeds; read bpb, Tier 0, IND. Cost: about 22 H100-h (est.). Confidence: medium.

**M21. Restore the budget-cut arms and seeds.** Lanes: systems C5, D3.
- Items: P-186's 10M trunk to 20,000 tok/param (about 37 H100-h per seed, est.); its second 5M seed at 20,000 (about 12); 5 seeds at every curve point (about +200); 150M at 300B tokens (about 280 per seed); P-158 vocab grid and P-151 depth-width sweep (tens each); one GPU type and engine for the whole 3-150M curve (removes the 5070 bf16 vs Titan fp16 confound).
- Ledger: existing P-ids; their cost reasons go away. The chat pool's 25-pass cap now binds first. Confidence: high on feasibility.

**M22. Successive halving on decay branches, gated on rank transfer.** Lanes: systems C6, H4.
- Mechanism: run many arms to 25% of budget, branch-decay all, keep the top quarter, continue. Adopt only if 25%-decayed rankings match full-budget rankings (Spearman at least 0.8, pre-registered).
- Ledger: NEW (older but missing). Test: about 20 arms x 2 seeds at 5M with branches at 25% and 50%. Cost: about 2 H100-h (est.). Confidence: medium; risk is late-blooming arms.

**M23. Compute-multiplier fits with shared exponents.** Lanes: systems C4.
- Mechanism: fit one Chinchilla-style surface across 3-4 sizes and budgets with shared exponents and an arm-specific factor; report "worth X times compute" with a CI ([2602.07712](https://arxiv.org/abs/2602.07712)). Fit bpb and margins, not pass rates.
- Ledger: older but missing; relates to P-055, P-186. Limit: the size-floor claim itself. Cost: about 60-90 H100-h per nominated change (est.). Confidence: medium.

**M24. Grow the chat pool instead of repeating it.** Lanes: data 2 and 7.
- Evidence: Qin et al. [2607.25271](https://arxiv.org/abs/2607.25271) (14M-600M, loss only, at 20 tok/param: paraphrase wins below about 50M, repetition above about 190M, crossover near 94M); BabyLM 2026 rewrites paper (entity tracking rose with simplifying rewrites, fell to 19.0 with harder paraphrases); Chen et al. for many-pass offset targets in Chen's exact form (R10).
- Ledger: updates P-162, P-187, P-160, P-105. Limit: #4, #5.
- Test: inside P-187 at 5M and the 60M point: an R1 paraphrase-expansion arm at matched passes; a "simplify" vs "neutral" R1 prompt; at 100 and 200 passes, Chen's form with slot, role and answer spans protected from noise. Cost: about 2-5 H100-h plus the 60M arm (est.). Confidence: medium for paraphrase at 5-30M, low for the augmentation.

### Tier D: worth a slot, lower expected value (compact)

| # | idea (lanes) | mechanism | best evidence (size, source) | ledger | limit | NAIRR | test | H100-h (est.) | conf. |
|---|---|---|---|---|---|---|---|---|---|
| M25 | Hard-wired match-and-copy operator, ROSA-style (arch 4, H4) | zero-parameter exact suffix match, copy what followed, ties to the most recent, NULL option | 12M attention-free, 10M words: learns markedly faster than matched attention (Taylor, [BabyLM 2026](https://aclanthology.org/2026.babylm-main.3/)) | NEW SINCE | #1, #2 | none needed | T002 arm, then 5M with learned mixing; loop rate as guard | ~1 | med-low |
| M26 | PaTH attention (arch 5) | data-dependent Householder position encoding | flip-flop 1-layer d64: PaTH 0 / 0.0001 error vs RoPE 6.9 / 40.3; 760M PaTH-FoX best ([2505.16381](https://arxiv.org/abs/2505.16381)) | P-020; S021 designed, not run | #1 | fla kernel and 80 GB remove the 19 GB build block | T002 arm, then S021 | ~2 | med |
| M27 | Stick-breaking at 30M with the fla kernel, plus stick-breaking+smear (arch 7) | recency-ordered attention weights | T001's only passing P-020 variant (LWO 0.960, AR32 0.999), slowest learner (5.9k-8.5k steps); fla kernel 2026-10 | P-020 pick; kernel NEW SINCE | #1 | makes 30M cheap | T002 arm, then P-020 at 30M | ~10-20 | med |
| M28 | FoX at paper init plus KV-shift (arch 3) | gate bias 0 (f about 0.5) plus learned previous-key/value mix | 360M: 6.62 vs 6.80 without KV-shift ([2503.02130](https://arxiv.org/abs/2503.02130)); but see R2 | P-020 / S005 | #1 | Flex score_mod or fla kernel | T002 first; 5M KV-shift arm only if toy says so | <1 | med-low |
| M29 | Voltic (memory 3) | delta-rule memory with Kalman-gain vector writes | volatile recall 2-layer w128: Voltic-Q 99.48, GDN-2 70.25 +/- 16.43; but softmax plus warm-up 99.43 in a larger-budget control (Table 8, E.6) ([2610.05700](https://arxiv.org/abs/2610.05700)) | NEW SINCE; P-022 variant | #1 | none | T002 arm only; P-022 arm only if it beats softmax-with-warm-up | ~1 | low-med |
| M30 | One LEMA latest-match head (memory 4) | binarized Q/K read only the most recent exact match | 2-layer d64 extrapolates 8 to 4096 pairs; weak as a full LM at 29M-834M ([2609.25802](https://arxiv.org/abs/2609.25802)) | NEW (one day before cutoff) | #1 | none | T002 arm in layer 2 | <1 | low-med |
| M31 | CoPE contextual position (arch 8) | position counts only gated tokens | 20.8M code model at 100B tokens: test ppl 3.9 vs 4.1 (RoPE model 19.8M, not parameter-matched); flip-flop 0.0 / 4.9 vs 1.8 / 20.3 ([2405.18719](https://arxiv.org/abs/2405.18719)) | NEW (older, missing) | LOOKUP, turn references | none | T002 table and flip-flop families, then 5M with LOOKUP items | ~2 | med |
| M32 | PoPE polar position (arch 9) | content sets magnitude, position sets phase | indirect indexing 11.16% vs 94.82%; 124M ppl 21.55 vs 21.33 ([2509.10534](https://arxiv.org/abs/2509.10534), v3 ICML 2026 camera-ready) | NEW (older, missing) | #2, LOOKUP | none | T002 indirect indexing, then 5M swap | ~2 | med-low |
| M33 | Canon/smear SoftPrior init or isolated taps (memory 5) | bias some channels to lag 1, others to lag 0, or give state writes their own tap | width-128 toys: both fixes raise recall; 79M/318M from scratch: no perplexity change, heads matched earlier on 6/6 seeds, with a near-recall trade-off at 79M ([2609.15545](https://arxiv.org/abs/2609.15545)) | NEW (v3 NEW SINCE) | #2 | none | T002 onset read, then S004 at 5M with SoftPrior | ~1 | med |
| M34 | Looping at two token budgets, with a context-reliance readout (arch 10, memory 10) | shared looped core; looped/grown models rely more on context than weights | Hyperloop 135.7M looped vs 238M plain: 14.40 vs 14.65 at 12.5B tokens, 12.19 vs 12.15 at 100B ([2604.21254](https://arxiv.org/abs/2604.21254)); Kapl [2610.31421](https://arxiv.org/abs/2610.31421) (360M-3B, not parameter-matched) | P-111, P-150 (BabyLM and Kapl NEW SINCE) | #2, #10, LOOKUP | 2-4x FLOPs now affordable | P-111 at 1x and 8x budgets; add P-142 counterfactual and lookup-vs-prior margins | ~10-30 | med |
| M35 | Hybrid "mode" layer: GDN after attention, segment-masked aux loss (arch 12, H2; memory 11) | recurrence carries persona/language mode; attention does exact lookup | 3-4B only ([2609.04434](https://arxiv.org/abs/2609.04434), [2610.06750](https://arxiv.org/abs/2610.06750)); attention-then-recurrent placement wins toys ([2610.18104](https://arxiv.org/abs/2610.18104)) | NEW SINCE; updates P-022 | #6 | none | T002 rule family; 5M 1-GDN-layer arm +/- aux loss; GDN at layer 2 vs 6; do not score on BabyLM entity tracking (28.5% leaked items) | ~2-20 | low |
| M36 | Logit KD from a larger Planck with skill-span routing (data 4, H2) | alpha KL to a 5-10x larger same-tokenizer Planck, dropped at answer spans and lowest-entropy positions | for: Lu and Liu [2605.23857](https://arxiv.org/abs/2605.23857) (1.7B students); against: Goyal [2509.01649](https://arxiv.org/abs/2509.01649) (1B: KD impairs induction heads), repo nulls at 125-200M | NEW variant of parked rows | #8, #2 | online teacher about 2x student FLOPs | 5M student, 30M teacher: CE / KD / KD+entropy routing / KD+span routing, 3 seeds; KD wins only if margins hold | ~3-5 | low |
| M37 | Ensemble-of-seeds KD (data 5) | average K seeds' logits, distill into a fresh same-size model | Kim et al. [2509.14786](https://arxiv.org/abs/2509.14786) (8x300M into 300M kept 83% of the gain; 150M ensembles weak); Baby Llama 58M | NEW | #8 | free teachers from curve seeds | 5M, K=3 and 5 E3 seeds, CE / KD / KD+routing | ~2-3 | low-med |
| M38 | Tied-embedding optimizer, AF-Muon (optim 6, H2) | clipped-row update for the vocab table | 124M NanoGPT: 3.4919 vs 3.5177, 3 seeds, plus long-horizon continuation runs at 0.5B/1B ([2610.01395](https://arxiv.org/abs/2610.01395)) | NEW SINCE | #9 (table is 31% of 5M, 3.5% of 150M) | none | 5M and 20M, own g-check, 2 seeds | ~3-5 | med-low |
| M39 | ScheduleFree+ at 1,000 tok/param (optim 7) | iterate averaging replaces LR decay | 120M at 1,000 TPP matched a 45% longer linear-decay run; AdamC base, not compared with WSD there ([2605.19095](https://arxiv.org/abs/2605.19095)) | older, missing | #4 | horizon-only effect | 5M: NorMuon-WSD vs SF+ vs SF on NorMuon | ~2-4 | low-med |
| M40 | Muon-squared preconditioning (optim 8) | divide momentum by sqrt(v) before Newton-Schulz, keep NorMuon rows | 124M at 3B tokens: 27.95 vs NorMuon 28.40 ppl, apparently 1 seed ([2604.09967](https://arxiv.org/abs/2604.09967)) | older, missing | #4 | none | 5M screen with g-check, 2 seeds | ~0.5 | low-med |
| M41 | Slow momentum rail for the repeated chat pool (optim 10, H3) | second EMA rail at beta about 0.9999 | AdEMAMix forgetting claim; ADANA [2602.05298](https://arxiv.org/abs/2602.05298) (45M+, 20 tok/param); record #92 twin rail | older, missing; record #92 NEW SINCE | #4, #5 | horizon-only effect | 5M x 2 pass counts x 2 seeds at 1,000 tok/param; recompute pass spacing at the real step count first | ~3 | low |
| M42 | Prefix-token auxiliary head (optim H5) | extra head predicts each token's longest-prefix token | modded-nanogpt record #88 (-15 steps, p=0.005, N=10) | older, missing | #2 (retyped names) | none | 5M training-only head, 2 seeds, scored on P-097 retyped-name items | ~0.5 | low |
| M43 | Same-family larger rewriter for SFT targets (post-train 8) | a larger Planck sibling rewrites teacher renders into its own distribution; keep checker-passing rewrites | SynthSelf 0.5-3B students, 7B/14B aligners (2.3-28x; the 6x of 30M-for-5M is inside that range) | update to P-073 | #3, #5 | rewriter runs cheap | 5M SFT on raw vs self-rewrites vs 30M-sibling rewrites, 3 seeds | ~5-10 | med-low |
| M44 | Cross-tokenizer coverage probe (post-train 7, data 10) | share of Planck tokens in strict one-to-one groups vs Gemma 4 / Qwen3.5 tokenizers | strict top-16 alignment kept at least 96% of the gain; smallest student 3B ([2610.08448](https://arxiv.org/abs/2610.08448)) | update to P-074, P-100 | #8 | teacher inference cheap | CPU over 10k pool conversations; go to P-074 only at 80%+ | 0 | high (probe) |
| M45 | Bidirectional attention inside user turns (memory H1) | tokens inside one user turn attend both ways | only on Llama/Qwen fine-tunes ([2412.18487](https://arxiv.org/abs/2412.18487)) | NEW | #2 | none | 5M, chat docs per-turn bidirectional, control drops LM loss on user tokens (P-056) | ~0.5 | low-med |
| M46 | Conversation packed beside its own paraphrase (data H4) | attention allowed across a conversation and its R1 paraphrase only | BabyLM rewrite pairing (encoders) | NEW | #2 (aliases, wording) | none | 5M cross-pair vs docmasked on identical tokens; leakage probe | ~0.5 | low |
| M47 | Mixture search at two sizes (data 9) | include size in the E-sel-1 regression | CAMEL [2603.08022](https://arxiv.org/abs/2603.08022) (mixture optimum shifts with size; up to 3% downstream, 50% cheaper search) | update to E-sel-1 | #4 | about 20 runs at 30M become cheap | add 30M runs; adopt per-size mixes only beyond seed noise | ~6-10 | low-med |
| M48 | ASIDE role rotation (memory 14) | fixed orthogonal rotation of user-token embeddings, 0 parameters | 7-8B only ([2503.10566](https://arxiv.org/abs/2503.10566)) | update to P-023 (S014) | #2 (speaker binding) | none | add beside S014's 768-parameter role table | ~0.3 | low-med |
| M49 | MTA key-query convolution (arch 11) | conv over attention logits across keys, queries, heads | toy error 1.3-78.2 vs 0.0-0.1; 880M ppl 11.25 vs 10.91; about 10x slower unoptimized ([2504.00927](https://arxiv.org/abs/2504.00927)) | NEW (older, missing) | #2 | needs non-flash path | T002 only; 5M only if it wins | ~1-5 | low-med |
| M50 | ChaCAL chain attention (arch 13) | (1-g) A (I-gA)^-1 V follows reference chains in one layer | toy d512: one layer 100% vs 5 standard layers needed ([2410.05565](https://arxiv.org/abs/2410.05565), already cited in E003 AUDIT) | arm NEW | two-hop | none | T002 chain family | <1 | low |
| M51 | Own-replies context pollution (memory 12) | the model's own earlier turns drive derailment | 4B+ only (AC3 [2611.00518](https://arxiv.org/abs/2611.00518); [2611.20071](https://arxiv.org/abs/2611.20071)) | raises P-067's priority | #5 | own-history rollouts cheap | no new test; run P-067 early | 0 | med |

**Checked, low value (kept so nothing is lost).**
- **FP8, MXFP8, NVFP4:** small shapes give speedup below 1 on H100 per torchao; none on A100; adds a numerics variable. Skip.
- **MIG:** unavailable at these sites (Delta says so). **torchtitan / FSDP:** every Planck model fits on one GPU; plain DDP covers the single 150M run.
- **Online vs offline KD cost flip (systems C8):** online KD of a 150M student from SmolLM2-1.7B costs about 4.07x the student (compute.md 6.4); disk, not compute, is now scarce. Moot until a KD result is positive (M36).
- **Deterministic mode for decision runs (systems H2):** expected not to lower paired SD; use it for parity tests only. Test is 16 pairs, about 1 H100-h, if anyone doubts it.
- **XSA, exclusive self attention ([2603.09078](https://arxiv.org/abs/2603.09078)):** zero parameters, no recall evidence, small gains at 0.7B+. A 5M arm is about 0.1 h if a slot is free.
- **Linear-RNN successors (DeltaProduct, Mamba-3, KDA, log-linear, KronLSTM, GSA-2):** StateBench says negative eigenvalues matter only for the general tier, which RC-12 does not need.
- **RLT and PAR looped token recurrence:** permutation tracking, token-sequential training. Watch only.
- **Selective RoPE:** worsened perplexity with FoX at 370M (25.29 to 33.87).
- **TTT and fast-weight memories:** nothing below 340M or inside a 2k window.
- **Muon speed variants (Dion3, BulkBoost, Polar Express):** speed or 0.02-0.15% loss changes. Polar Express only as a free swap.
- **WSM, WSqD, CompleteP:** WSD branching already gives most of it; E2 found transfer holds 5M to 20M; one LR check at 150M's 29 layers is enough.
- **z-loss:** absent from harness/*.py despite training.md's claim; harmless at d=192 with QK-norm. A 1e-4 arm is about 0.03 h if logit growth is ever seen.
- **Mimetic V-O init:** Planck already initializes output projections nonzero.
- **MTP successors beyond P-105's registered curriculum arm:** S006's REJECT stands except R10's Chen-form cell.
- **Delayed WD schedules ([2611.17876](https://arxiv.org/abs/2611.17876)):** a caution, not a bet: judge any WD or regularizer change at the branch's final checkpoint.
- **Repetition ordering ([2610.05591](https://arxiv.org/abs/2610.05591)):** consecutive shard replay costs up to 0.46 bpb; already the reason P-187 spaces passes.
- **Self-generated replay against forgetting ([2605.26097](https://arxiv.org/abs/2605.26097)):** fold into the SFT recipe after long pretraining.
- **LLM-judge consistency rewards, ZPPO, agent OPD papers:** need abilities a 5M model lacks or conflict with D8.
- **DVLM / CSD cross-tokenizer heads:** only after a positive in-family KD result; 1B+ students.
- **Synthetic dialogue too clean ([2610.35029](https://arxiv.org/abs/2610.35029)):** matters for the Level A human test, not RC-12.
- **Tokenizer papers:** nothing reverses D6 or P-158. **Typed intention stores, verbatim compaction:** back P-005; no new bet.
- **The "bind" hobby project, Samata Latch, Britannica entity tracking:** withheld weights, opposite goal, or inflated benchmark.

---

## 4. Hypotheses

### 4a. Lane hypotheses, merged

| id | claim | test | kill condition |
|---|---|---|---|
| HA (arch H1, data H3) | T001 measured how fast the previous-token circuit forms | M1 budget, curriculum, warm-up and attention-distillation arms | base fails at 200k steps, after the warm-up, and returns to chance when distillation is switched off |
| HB (arch H2) | standing rules live in recurrent "mode", not in lookup | M35 | persistence with distractors equal across arms |
| HC (arch H3, revised) | in chat, key formation fails as often as recency; exact-key interference is real too (PI-LLM [2506.08184](https://arxiv.org/abs/2506.08184), DKI [2603.12271](https://arxiv.org/abs/2603.12271)) | T002 paraphrased multi-token-key family vs exact-key family, both lever types | key-formation levers win on neither family |
| HD (arch H4) | an exact-copy prior gives free latest-wins but raises loops | M25 with the loop rate as a pre-registered guard | no correction gain, or loop rate up beyond the guard |
| HE (arch H5) | removing the sink reduces primacy (already P-021; only the sink-logit arm is new) | M11 inside P-021 | FIRST errors unchanged |
| HF (memory H2) | a K-curriculum forms binding faster | M17 | capacity curves equal by end of training |
| HG (memory H3) | previous-token arms trade recall for loops | M19 | loop and echo rates equal across arms |
| HH (memory H4) | corrections that name the old value teach global suppression | M18 re-introduction cells | re-introduced values recalled as well as fresh ones |
| HI (optim H1) | NorMuon's 5M win is tail memorization that costs SFT plasticity at long horizon | M3; MSR ranks arms before SFT | NorMuon equal or better after SFT at 2,000 tok/param |
| HJ (optim H2) | the embedding optimizer is a hidden size-dependent term in the curve | M38 at 5M and 20M | 5M shift below seed SD |
| HK (optim H3, revised) | slow momentum helps the repeated chat pool more than prose | M41 after recomputing step counts at PLAN's grown batch (98k-195k steps for 25.6B tokens, est.) | F(CHAT) and F(PROSE) gains equal |
| HL (optim H4) | a data-order soup beats the mean branch by more than the branch SD | M14 | soup within one branch SD |
| HM (optim H5) | prefix-token prediction improves retyped-name recall (8.5% of names share a token count across four forms at 8k) | M42 | name recall unchanged |
| HN (optim H6) | the long-run LR follows Planck's own drift better than any borrowed exponent | M4 | the 20M/2,000 prediction misses by more than one grid step |
| HO (data H1) | an update-stream warm-up breaks first-value-wins | M8 toy then 5M | all arms within seed noise |
| HP (data H2) | routing KD by skill spans beats routing by entropy | M36 | plain KD does not hurt margins, or span ties entropy |
| HQ (data H4) | packing a conversation beside its paraphrase makes recall wording-robust | M46 | margins flat or leakage rises |
| HR (data H5) | a same-size gold-state twin is at least as good a teacher as a 150M Planck | M6 arm | twin no better than SFT, or below the 150M-teacher arm beyond seed noise |
| HS (data H6) | distillation from a locked-in seed rescues the seed lottery | only where P-026/P-188 shows variable lock-in | lock-in rate unchanged |
| HT (post-train H1) | a twin-set reward kills shortcuts that per-item reward keeps | M7 reward arms, shortcut-alignment metric | shortcut alignment equal |
| HU (post-train H2) | RL's edge over SFT grows as the model shrinks | M7 at 5M, 10M, 30M with equal budgets | gap flat or growing with size |
| HV (post-train H3) | note-view self-distillation works at 5M though demonstration self-distillation fails at 3B | M12 | no gain over continued SFT |
| HW (post-train H4) | the size where OPD students stop approaching the teacher is each skill's capacity floor | M6 across 5/10/30M per family | no size-dependent gap |
| HX (systems H1) | seed stacking gives at least 4x aggregate throughput at 5M with no parity loss | M2 | below 2x MPS at K=8, or parity fails |
| HY (systems H2) | deterministic mode does not lower paired SD for real arms | 8 A/B pairs each mode, plus A vs A+1e-7 pairs | SD_d drops beyond its CI |
| HZ (systems H3) | part of P-186's token scaling at 5-10M is a batch artifact | M20 | 2M-token batch no better than 64k on bpb and margins |
| HAA (systems H4) | 25%-budget decayed rankings transfer at Spearman 0.8+ | M22 | rho below 0.8 |
| HAB (systems H5) | a crossed init x data design cuts seeds needed for a given MDE by about 30% | E3-N re-analysis | no reduction |
| HAC (memory H1) | bidirectional attention inside user turns improves binding | M45 | binding margins flat |

### 4b. New cross-lane hypotheses (this document's own)

**X1. Formation time, not capacity, sets Planck's early skill floor.**
- Reasoning: T001's failures, Voltic's softmax-plus-warm-up result, induction emergence quadratic in context, binding formation growing as K^2.1-2.6, and procedural warm-up all say small models run out of steps before they run out of parameters for circuit formation.
- Prediction: at 5M, the bpb and RC-12 margin advantage of Canon and smear over BASE shrinks between the 250 and 2,000 tok/param branches, while their onset advantage stays; a 0.3% warm-up gives BASE most of the onset advantage.
- Test: S004 and S007 arms crossed with {no warm-up, k-Dyck/update warm-up}, decay branches at 250 / 1,000 / 2,000, 2 seeds (about 10 H100-h, est.).
- Kill (either part): Canon's margin advantage at 2,000 is at least its advantage at 250, or the warm-up closes less than half the onset gap.

**X2. The sink logit and the GDN register fix different correction errors.**
- Reasoning: first-value-wins (E001) looks like primacy, which a position-0 sink can feed; CICM shows several stale values can outvote the current one under RoPE, which a current-value register (StateBench's aperiodic tier) addresses.
- Prediction: in a 2x2 (sink logit x one GDN layer), the sink lowers FIRST errors and barely touches STALE-vote errors; GDN does the reverse; the gains add.
- Test: T002 families with per-distance FIRST/STALE counts, then a 5M 2x2 with 2 seeds (about 3 H100-h, est.).
- Kill: each arm cuts both error types by similar fractions, or neither arm moves its predicted type.

**X3. Long NorMuon pretraining shifts answers from context to weights.**
- Reasoning: Muon's lean toward rote memorization ([2605.06654](https://arxiv.org/abs/2605.06654)) and its fine-tuning deficit at long horizon (2611.03633) predict more prior-driven answers; looping and growth push the other way (2610.31421).
- Prediction: at equal bpb, NorMuon branches at 2,000 tok/param score lower on the P-142 counterfactual open-book probe and the lookup-vs-prior conflict margin than AdamW or Hyperball branches, and the gap widens from 250 to 2,000.
- Test: add both probes to M3's branches (inference only).
- Kill: no difference beyond seed noise at 2,000.

**X4. RL and OPD gains depend on the pretraining optimizer.**
- Reasoning: RL needs edge-of-competence pass rates and plastic weights; OPD only sharpens what the student can already reach (27 of 27 settings); Muon-pretrained models adapt worse after long pretraining.
- Prediction: from M3's 2,000 tok/param branches, M7's REINFORCE arm and M6's OPD arm gain more on RC-12 dev from the AdamW or Hyperball branch than from the NorMuon branch.
- Test: run M7 arm (c) and M6 arm (b) from two optimizer branches, 3 seeds each (about 10-20 H100-h, est.).
- Kill: gains equal within seed noise.

**X5. Data-order noise is mostly onset timing.**
- Reasoning: E3's data-order SD point estimate is larger than its init SD (not established at df 4); binding formation is superlinear in K, so when the first dense correction and binding conversations arrive could decide when the circuit locks in.
- Prediction: in a crossed init x data design, per-run F(CHAT) correlates with the onset step of the induction/binding probe (|r| at least 0.5), and a front-loaded warm-up cuts sigma_data more than sigma_init.
- Test: E3-N crossed families with onset logged, with and without M8's warm-up (about 6 H100-h, est.).
- Kill (either part): |r| below 0.3, or the warm-up leaves sigma_data unchanged. Between 0.3 and 0.5 the correlation part is inconclusive.

**X6. OPD helps only where the student already reaches the answer sometimes, and warm-ups decide which cells qualify.**
- Reasoning: the 27/27 OPD cap plus the edge-of-competence rule for RL.
- Prediction: across RC-12 dev cells, OPD gain at 5M is about zero where SFT pass@16 is 0 and positive elsewhere; M8's warm-up moves cells from the first group to the second.
- Test: M6 at 5M with and without the M8 warm-up trunk, per-cell pass@16 logged before OPD.
- Kill: OPD gain in cells with SFT pass@16 of 0 is not lower than in cells above 0 (beyond seed noise), or the warm-up moves no cell across.

**X7. Fixed LR and fixed batch together flatten P-186's token curve.**
- Reasoning: at long horizon the LR optimum falls (M4) and the optimal batch grows (M20); each alone biases long runs, and PLAN's freeze rules ("flat by 1,000") read the slope.
- Prediction: in a 2x2 (screen LR vs re-tuned LR; 128k vs 1M-token batch) on a 5M trunk to 2,000 tok/param, the re-tuned pair raises the 1,000-to-2,000 slope of RC-12 dev margins more than either change alone.
- Test: 4 trunks x 2 seeds with branches (about 10-12 H100-h, est.).
- Kill: slope differences within seed noise; then P-186's freeze rules stand as written.

---

## 5. Proposed order for the first NAIRR weeks

Estimates only, not caps. Max's rule holds: nothing here is dropped for budget; order is by decisiveness and dependency.

| when | work | H100-h (est.) |
|---|---|---|
| Now (no GPU) | Faculty PI and Start-Up request; dated amendments for anything touching the Dec 4 freeze; pre-register T002, E3-N, M3, M4 and M5's rules; MSR on S003/E3 checkpoints; cross-tokenizer coverage probe (M44, CPU); false-WIN simulation on E3 nulls (M5); Apptainer images (x86 first), job-of-jobs runner with requeue and 30-minute checkpoints, static cu_seqlens, offline packer; RL sampler design | 0 |
| Start-Up week 1: calibration | `nvidia-smi` and driver, compute-node internet test, harness suite and mutation checks inside the image; bench_micro (about 3); packing MPS vs stacking (about 4); attention backends with leak probes (about 3); CUDA graphs (about 2); loader (about 1); E3-N (about 16); cross-hardware 8-seed check (about 1); 3-point E2 check (about 1) | ~30-40 |
| Week 1-2: decisive cheap tests | T002 (M1, about 10-25); rank-transfer calibration (M22, about 2); half-octave re-grids of S002/S004/S005 (R5, about 5); sink logit 5M (M11, under 1); warm-up 5M arms (M8, about 2-4); dynamic conv S022 (M9, about 2); soups (M14, about 1) | ~25-40 |
| Week 2-4: long horizon | LR/WD horizon trunks (M4, about 30-40); optimizer bake-off with SFT (M3, about 40-50); batch test (M20, about 22); X1 and X7 crossed into the same trunks where possible (about 20); binding probe on all checkpoints | ~110-130 |
| Week 3-6: teachers and post-training | 400M Planck teacher (M6, about 35); teacher pilot renders (M10, tens); RL at 5-10M once the sampler passes its tests (M7, about 20-40); gold-note self-distillation if P-005 passes (M12); Antidoom (M16) | ~100-150 |
| Then | Batch 2 screens (S009-S021 plus M9, M11, M25-M33 winners) under M5's FDR rules; P-186 restored arms and seeds (M21); compute-multiplier fits for nominated changes (M23); curve on one GPU type | full request |

Rough total for the first six weeks: about 270-360 H100-hours (est.), inside one 2,000-hour Start-Up. The full curve plus restored seeds belongs to a full request built on the Start-Up's measured numbers.

---

## 6. NAIRR application facts

**Who can apply.**
- US-based researchers and educators at US institutions, institutional email required. The allocations and Start-Up pages include graduate students who submit a support letter from a faculty advisor; undergraduates are not mentioned. Every supported project has its PI name, affiliation, title and abstract posted on the NAIRR Pilot website. Sources: https://nairrpilot.org/opportunities/allocations, https://nairrpilot.org/help/faq.
- So Max needs a faculty member as PI. In NAIRR's own allocations CSV (1,289 rows), only 4 awards are at community colleges, all Classroom awards (CUNY BMCC). Whether a student who is not the PI can hold a provider account on a project is UNVERIFIED; ask the provider.

**Start-Up request** (https://nairrpilot.org/opportunities/startup-project).
- Up to 2,000 GPU-hours on one of PSC Bridges-2 GPU, Purdue Anvil AI, SDSC Expanse AI or TACC Vista (Jetstream2 128,000 SUs; ACES 480,000 SUs). 3 months.
- Decision "within 2 weeks" in one place on the page and "within 3 weeks" in another; recipients must start within 2 weeks.
- Purpose in the page's words: get acquainted with the processes by which resources are allocated, build and test a software environment, perform a scaling study. (The often-quoted "get acquainted with a resource, build your execution environment, scale-test and/or benchmark" is from the Internet2/NAIRR slide deck, 2025-09-29: https://www.depts.ttu.edu/hpcc/downloads/2025-09-29_Requesting-Resources-via-the-NAIRR-Pilot.pdf.)
- Recommended: an x86 H100 site (Anvil AI or Expanse AI). Same software as the PC, no aarch64 porting, and Start-Up exists to benchmark.

**Full research request.** 3-page proposal; monthly review, requests in by the 15th usually decided by the end of the following month; onboarding 1-3 weeks (slide deck); 12-month awards with no defined upper limit; results must be open; reports at 1 and 6 months plus a final; allocated storage lasts only for the project. Reviewers want the math shown: the measured 5070 bench rows, E3's noise table and the token-pass budget are that evidence.

**Typical sizes** (NAIRR allocations CSV, https://nairrpilot.org/pilotallocations/q/awards, recomputed by a verifier). Full 2026 awards on 7 GPU resources: quartiles 4,680 / 10,000 / 20,000 GPU- or node-hours, max 85,000 (n=110). Per-resource medians: DeltaAI 12,500; Anvil AI 10,800; Delta 8,000; Expanse AI 10,000; Vista 10,000 node-h; Bridges-2 5,350; Voltage Park 3,120. CloudBank: 50 awards, median $8,750. Similar funded projects: NAIRR250527 "LLM Scaling Laws" (5,000 DeltaAI GPU-h) and NAIRR250124 (OLMo data interventions, 20,000 Anvil AI GPU-h). The latest award start in the CSV is 2026-11-14, so awards are still being issued.

**Program status.** NSF chose SDSC with TACC to run the permanent NAIRR Operations Center: $35M over 5 years, about 2026-09-01, confirmed through search results (UCSD, fedscoop, the NSF solicitation). The SDSC press release itself refused connections during verification, and the award number circulating (2616251) appears only in a third-party blog. Partners: Voltage Park pledged 1M H100-hours (NSF, 2025-07-16); NVIDIA offers "a minimum of four NVIDIA DGX nodes for at least a month" (HPCwire, 2026-06-22); AWS, GCP and DGX Cloud were "On Pause" in the Sept 2025 slides.

**Hardware.** DeltaAI: 152 quad-GH200 nodes, 608 GPUs, 132 nodes NAIRR-funded (current architecture page; the 2025 slide's 456 is stale). Anvil AI 80 H100; Expanse AI 136 H100; Vista 608 GH200 at 1 GPU per node; Bridges-2 80 H100; Delta A100 and A40 plus an H200x8 partition (2025 slide counts).

**Operations that change the plan.**
- DeltaAI: SSH keys disabled, Duo on every login, so the PLAN's unattended supervisor cannot log in by itself; `ghx4` 48-hour jobs, not requeued unless `--requeue`; HOME 100 GB, PROJECTS 500 GB (1-25 TB on request), node /tmp purged per job; stagger Python-heavy job starts by 3-5 minutes. Source: https://docs.ncsa.illinois.edu/systems/deltaai/en/latest/.
- Delta: preempt partitions at 0.5x charge (10 minutes guaranteed, 5-minute grace); MIG not supported. Vista: 15-minute minimum charge per job, 20 running and 40 submitted jobs, $SCRATCH purged after 10 days, MPS documented. Source: https://docs.tacc.utexas.edu/hpc/vista/.
- So: one node-level job-of-jobs per allocation, several models per GPU, long-lived processes with warm compile caches, checkpoints every 30 minutes, results archived back to the PC.
- Software: CUDA 13 wheels (Planck's torch 2.13.0+cu130) need driver 580+; an older driver forces cu126 (different cuBLAS, different numerics). On GH200, torch 2.11+ is needed for a CUDA wheel by default on aarch64. Outbound internet from compute nodes is UNVERIFIED.

---

## 7. Corrections applied from verification

- Voltic demoted from second to a T002-only arm: its softmax baseline learned standard recall (99.98) and, in a larger-budget control, 10k warm-up steps on a one-observation task then 2k on volatile reached 99.43 (Voltic-Q 99.48 in the main table, so not a matched comparison).
- FoX: S005's trained gates went local anyway (half-life medians 12.6 and 7.1 tokens), so the 5M re-screen case is weaker than the arch lane said; S005's dense bias ran at micro 8 x accum 2 (384 MiB per layer), costing +1.33 GiB and 2.26x time.
- "Real failures are paraphrase-driven" dropped: PI-LLM and DKI document exact-key update failures.
- Sink-logit 350M numbers updated to the paper's v2. The sink-primacy hypothesis is P-021's, not new.
- T001 learning order: smear first; GDN learned before Canon on 4 of 6 seeds.
- Stack warm-up: the queue "fails to produce the same gains" (not harm); stack training after pretraining hurts.
- ChaCAL is already cited in E003's AUDIT; PoPE's ICML 2026 venue is confirmed by its v3.
- Per-arm LR: every screened arm already had a seed-1 g-check with a selection/estimation split; only a finer grid with more selection seeds survives.
- Porian's LR 0.013 / batch 20 / beta2 0.99 optimum is the 5M model (sizes 5M-901M), not 55M.
- Optimizer step counts use PLAN 5's grown batch (98k-195k steps for 25.6B tokens), not 781k.
- E3's data-order vs init gap is a point estimate, not established at df 4.
- Curve budgets: 5,000 tok/param applies to 5M and 10M only; 20,000 only to P-186's continuation and a conditional 3M point.
- No repo file flags FoX for "late benefit"; S005's flag means better CHAT with lower history dependence.
- Quotes fixed: CMU "is simply continued pretraining"; OLMo 2's averaging quote points to Section 4.5; Start-Up purpose wording (section 6).
- P-097: 8.5% of names share a token count across forms, not a tokenization.
- Online KD from a 1.7B teacher is about 4.07x the student, not 3.5x.
- DeltaAI is 608 GPUs; the SDSC award number is unconfirmed.
- AF-Muon has long-horizon runs; 2610.21555's 20 pairs include a ViT; the Procedural Core paper is mainly about ViTs (language at about 16 tok/param).
- 5070 costs used the eager rate; compile halves them. T001-style toy costs exclude LR picks and lock sharing, and warm-up symbols need new token ids.
- 2611.14912: sizes are 7B/8B, and its SFT data come from the RL-tuned model and Qwen3-14B; Interplay never tests rejection fine-tuning, so only the Saxe lab compares RFT and RL head to head.
- Chen et al.: noise plus offset without right-to-left prediction interfered; only Chen's exact form is proposed.
- Interplay's process-verification gain carries a third-party caveat (OOD pass@128 fell).
- D6: a Planck-family teacher is outside D6's scope; it does not meet D6's revisit condition.
- "OPD does not teach stopping" is thinking-mode only; E004's stopping gap came from SFT targets.
- SynthSelf's aligners are 2.3-28x their students; Echo Chamber's pass@64 declined late in training; State-OPD's 26.07 is the strongest baseline, not the pre-OPD student.
- Jiang's 55/67/86% are per dataset, not per model size; Meyer's "top submission" is the author's own; Veitsman 2609.11505 is verified; CAMEL also reports downstream gains; Busbridge's main students span 198M-1.82B; BabyLM's 118 counts Strict plus Strict-Small.
- Dropped as unconfirmed: a Cerebras affiliation for Kang et al., several venue labels (Abdulhai, SDFT, Kim, Goyal), and DVLM's GPU-days per head.

**Independent review pass (2026-11-18, after synthesis).**
- Voltic control restated everywhere: softmax reached 99.43 with 10k warm-up steps plus 2k volatile steps in a 12k-update diagnostic (Table 8), while Voltic-Q's 99.48 and softmax's 2.96 come from the 10k-update main table, so "tying" was not a matched comparison; softmax also fails the stochastic variants, not only volatile. 2508.19029's "always solves" is standard MQAR without updates.
- NAIRR eligibility: the quoted graduate-student sentence could not be found in the saved pages; replaced by the allocations and Start-Up pages' own wording. "Cannot apply alone" softened to what the pages say.
- Kill conditions tightened: X1 and X5 are now killed by either part failing (they needed both), X6 and HR got concrete kill rules.
- Restored two lane items that had been only partly carried: data idea 7's screening rules (no data arm adopted on bpb alone; per-arm LR for data arms; now in M5 and a ledger update) and post-train idea 9's compute split (RL from a plain-decay vs a P-058 anneal trunk; now in M7 and P-197).
- Ledger ids re-checked after a fresh fetch: LEDGER.md and origin/main's LEDGER.md both end at P-188, and no tracked file on origin/main or in the working tree uses P-189 or above.

---

## 9. Gap sweep (2026-11-18)

Asked after sections 1 to 7 (there is no section 8; this one is numbered 9 as requested): is this really everything worth trying? Two more search rounds aimed at venues and sources the six lanes never swept. The answer is no, it was not everything: the rounds found 126 candidates, kept 103 after verification, and merged them into 47 new ledger rows (P-237 to P-283) and 39 update entries for existing bets, proposed in a scratch `LEDGER_GAP.md` and now appended to LEDGER.md under "2026-11 frontier additions" (subsection "2026-11 gap sweep additions"). Most are diagnostics, controls and new arms on bets already listed. Only a handful change what should run first (section 9.4).

**Ledger ids.** origin/main LEDGER.md (commit d71d447, 2026-11-18) now ends at P-236, because the section 3 rows were merged. The header's "next free id P-189" line is stale; the next free id after section 9.6 is P-290. The local checkout is behind origin and still ends at P-188.

### 9.1 What was searched

Per-gap source counts below are the writer's tally from the kept items (the sweep did not log per-gap totals), so treat them as approximate.

**Round 1 (90 found, 70 kept).**
- BabyLM 2026 proceedings, every abstract (about 10 kept: RecGPT, Relational Attention, the Parameter Golf meta-analysis, Miles' multi-epoch chain, MPO, retrieval exemplars, mention priming at 2.9M-33.5M, Attention Residuals).
- ICLR 2027 submissions on OpenReview (about 25 kept). The verifier hit OpenReview's bot check on 2026-11-18, so 20 of these rest wholly on abstracts it could not reopen (marked UNVERIFIED) and about 5 more partly.
- Practitioner leaderboards and repos (about 12): Parameter Golf PRs and records, modded-nanogpt Track 3 and PRs #178, #376, #378, #379, Slowrun PRs #92 and #98, nanochat issue #810.
- Microcontroller ports (3): esp32-ai RESULTS.md, InsiderLLM P4 vs S3.
- arXiv keyword search on the round's terms (about 20).

**Round 2 (36 found, 33 kept), aimed at round 1's critic gaps.**
- NeurIPS 2026 and COLM 2026 main tracks via the OpenReview search API (about 8 kept: entity-frequency phases, readout learning speed, Sieve attention, muLM, two retention papers, strong weight decay under repetition, Inside the Loop).
- EMNLP 2026 main and Findings, BlackboxNLP 2026, CoNLL 2026 from the ACL Anthology (4: copy before suppress, the ChaCAL head cliff, serial position at 14-70M, State Integrity Collapse).
- arXiv full listings cs.CL, cs.LG, cs.NE from 2026-10-15 to 2026-11-18 (about 13: StateWise, GatedV, four looping papers, DAMI, First Passage Attention, rival-value interference, data placement, GateFloor, onona, subliminal KD, perturbed-input KD, mixture-rollout OPD).
- Interpretability write-ups and repos (about 8: gradient routing, patterning, copy suppression, LLC stages, mechanistic data attribution, an induction-lockout toy, Goodfire VPD, the EleutherAI subliminal blog). Two LessWrong posts could not be located and stay UNVERIFIED.

### 9.2 New items ranked into the master list

Each item names its proposed ledger row. "With" means where it sits relative to the existing M items.

**Joins Tier A (free or nearly free, run while waiting for NAIRR).** These are inference or analysis on checkpoints that already exist, and they decide how later results are read.
- **P-271 repeated-lure probe** (copy before suppress, Dahiya and Blondin, Findings EMNLP 2026). The strongest item of the sweep. Name-copying forms before the head that suppresses the repeated name, so accuracy on "A and B ... B gave it to [A]" falls below chance while loss keeps falling: Pythia-160M floor 0.317, nine PolyPythia-160M runs mean floor 27.7%. A below-chance dip that later recovers is direct evidence for R1's "slow, not unable"; a score that stays below chance is not (the paper's zero-injection toy stayed below chance and never learned the rule). Its dose result (3 seeds; 5% and 15% injected override data reach 0.903 and 0.910 on held-out names, 30% reaches 1.000 in distribution but only 0.757 held out) bears on P-009 and P-059. Probe with M1; dose screen in Tier B.
- **P-240 corrective-gain split.** All six instruct models from 135M to 1.7B re-emit a failed value more after it is restated (-23.92 nats at 135M). Splits RC-12 CORR by whether the correction quotes the stale value. With M19.
- **P-272 relapse score** (StateWise, 8B). Chat SFT on add / amend / withdraw dialogues roughly doubled relapse to revoked rules, and DPO on relapse contrasts cut it 8 to 22 points. The free part checks whether Planck's SFT raises relapse. Training part in Tier B.
- **P-263 stale-value twin audit.** Check that RC-12 dev has cells where the stale value is mentioned last without being reasserted, and position-vs-time swaps. Add any missing ones by dated amendment before results exist. With M18.
- **P-276 readout-speed confound** (31M, NeurIPS 2026). Readout LR alone moves loss 2 nats early and 0.06 at the end. Any arm touching the tied table (P-159, P-217, P-245, P-246) needs a long-branch read. With M22.
- **Updates to M5 (P-193, P-209):** an exponent-difference test with reversal probability (UNVERIFIED audits).
- **Updates to M1 (P-189):** the gate-init sweep (P-275), the acquisition fraction (P-255), and a load-bearing ablation on every onset claim (Bi-Induct, 0.13B: planted induction heads were less load-bearing).

**Joins Tier B (high-value bets).**
- **P-238 Copy-Paste LM head (CPLM).** 135M, 3 seeds: ppl 16.33 to 15.83, sliding-window NIAH 14.3 to 99.4. About 25k parameters at d192. It aims at the shared last step of RECALL, BIND, CORR and LOOKUP. Ranked just below M9 (dynamic conv): stronger evidence, but on a different axis.
- **P-271 dose screen** and **P-272 SFT vs SFT+DPO on relapse contrasts.** These are the only new items aimed squarely at RC-12's correction and rule cells.
- **M3 / P-190 update:** add SOAP-Muon (P-190 already has plain SOAP) and NorMuon + EMA-Nesterov, and tune NorMuon's auxiliary LR as its own axis. On Track 3 at 124M, NorMuon only ties tuned Muon (3250 vs 3250 steps) while record #44's SOAP-Muon stack reaches 2750 (a stack of several changes, so SOAP-Muon's own share is not isolated); the UNVERIFIED Qwen3-4B result says the Muon fine-tuning gap is mostly the untuned auxiliary LR. Adds about 25-40 H100-h.
- **P-262 weight decay erasing unexercised skills.** Tier B only if its OpenReview sources check out. Planck stacks four stages under decoupled weight decay, and the reported (UNVERIFIED) 0.98 to 0.01 composition loss would be invisible in bpb.

**Joins Tier C (strong supporting bets).**
- **Toy-first arms:** P-237 NextLat, P-253 gold-pointer attention supervision, P-254 current-value restatement, P-273 imbalanced task curriculum, P-275 gate and decay init, P-260 width vs LR (UNVERIFIED), P-261 removable previous-token scaffold (UNVERIFIED), P-264 short-conv controls for S004 (UNVERIFIED).
- **Cheap 5M screens:** P-278 GatedV (5 of 5 seeds at 2.33M), P-256 head partition, P-257 QK gain init, P-244 small-gamma init.
- **Tokenizer:** P-243 CoBPE-lite. Its stage 0 is CPU-only and must land before the Dec 4 tokenizer freeze.
- **Core-bet measurement:** P-279 gradient routing, the cleanest in-model test of "skills cheap, facts expensive".
- **Post-training:** P-266 own-history error injection, P-267 Adam epsilon on the tied table in SFT (UNVERIFIED), P-274 entity-frequency phases.
- **MCU release only:** P-250 low-bit skill retention and P-251 the MCU port benchmark.
- **Updates:** P-187 (tail averaging, strong WD and masked-input regularization, meta-gradient in high-pass cells), P-196 (top-K support and mixture rollouts against loop-rewarding OPD), P-197 (end-of-turn gradient unit test; RL slope vs pretraining tokens from the 5M-1B chess study), P-022 (DAMI init, load curriculum, entity-keyed addressing), P-111 / P-150 (full BPTT, input injection, loop adapters, late onset), P-123 / P-183 (Linger and Lose: int4 retention 0.581 and ternary 0.090 at 25M; the collapse sits in the head and a WSD cooldown fixes it, which weakens P-183).

**Joins Tier D (worth a slot, lower expected value).**
- **New rows:** P-239 Golf mining, P-241 guided-policy self-distillation, P-242 span-constrained split, P-245 embed_wd 0, P-246 logit softcap, P-247 rare-tail gate, P-248 dual-route split, P-249 vocabulary grow-in, P-252 MCU KV, P-258 mention priming, P-259 Attention Residuals, P-265 recall incentive, P-268 optimizer-state plateaus, P-269 head balancing, P-270 emergence forecast, P-277 serial position, P-280 patterning knockout, P-281 copy suppression, P-282 fixed-offset lockout, P-283 LLC.
- **Decoding, report-only under the locked rule:** P-076 (CFG at 124-160M: LAMBADA +12 to +15), P-077 (vote flip rate), P-078 (temperature-fragility gate), P-095 (NONE as a first-class grammar option).
- **Table variants by size:** P-099, P-113, P-159 (frozen codes, onona, untie at 2/3, Leviathan, GateFloor).
- **Small arms:** S010 (LeakyReLU^2), P-229 (all-layer XSA), P-210 (PR #378 form).

**Checked, low value or watch.** muLM (abstract only), VPD decomposition (cost unknown, tens of GPU-hours), PLE (a loss under total-parameter counting), subliminal KD (only if KD is adopted), Erase-then-Delta (adds nothing beyond GDN-2), and every n-gram, PPM or TTT trick from Parameter Golf (evaluation-time, so a scaffold under the locked rule).

### 9.3 New hypotheses

- **CPLM gains more as size shrinks.** At 5M, a pointer head frees up 3 heads per layer (P-238).
- **H1:** case and space factoring helps cross-form name recall more than bpb (P-243).
- **H2:** rare-tail methods are null at an in-domain 8k vocab (P-247).
- **H3:** a nested 2k-to-8k grow-in front-loads copy practice (P-249).
- **Guided-policy self-distillation** moves CFG/CAD gains into the weights. This is the only route by which decode-time context amplification can count under the locked rule (P-241).
- **Low-bit precision:** int4 keeps context-computed skills better than stored facts (P-250).
- **MCU memory:** the KV cache, not the weights, binds a 12-turn chat on a microcontroller (P-252).
- **Loops:** free-running loops are leftovers of incomplete induction formation and show hidden-state collapse first (P-255).
- **Attention temperature:** part of T001's plain-softmax delay is waiting for attention to sharpen (P-257).
- **Previous-token scaffold:** a previous-token path is needed to learn, not to run, so it can be removed after formation (P-261).
- **Canon and smear** win partly by absorbing fixed-offset template shortcuts that would otherwise lock out induction (P-282).
- **Plain attention** shows a per-layer slot cliff near heads = tracked properties, as ChaCAL does (P-189 update).

### 9.4 What changed in the top recommendations

1. **"The five things", item 1 (T001 measured speed):** strengthened and sharpened.
   - The below-chance repeated-lure window is a new free signal of "slow".
   - An UNVERIFIED OpenReview paper reports that a missing previous-token circuit blocks learning outright (600,000 updates, nothing learned). T002 already gives every arm a previous-token path, so the design stands, but R1's reading does not cover arms without one.
   - The gate-init paper (2609.16540) predicts, from theory and Mamba toys rather than FoX itself, that P-189's "FoX at paper init" arm will plateau. If that carries over, the arm's expected sign flips; run it with the init sweep, not alone.
2. **Item 2 (optimizer regime):** M3's arm list changes. SOAP-Muon and a tuned auxiliary LR join, at about 25-40 extra H100-hours. NorMuon's S003 margin over AdamW is now even less trustworthy.
3. **"What to do first", step 2 (no GPU) grows by about a day of 5070 inference:** P-271 probe, P-240 split, P-272 relapse score, P-263 audit, P-258 mention priming, P-277 serial position, P-281 copy suppression, P-242 span split, the PR #376 copy mixture, PTQ of WSD vs averaged checkpoints (2510.06213), and S001 gate kurtosis. All run on existing checkpoints.
4. **One new Tier B architecture bet:** CPLM (P-238), screened in T002 and at 5M with the loop guard.
5. **Before acting on any UNVERIFIED row:** reopen its OpenReview page in a browser session that gets past the bot check. About 25 items depend on it, including P-260, P-261, P-262 and the M3 auxiliary-LR claim.
6. **Unchanged:** the Tier A order, the NAIRR filing, the freeze rule and the D6 ruling. Of the five things that matter most, only item 4 (RL) gains a guard: the end-of-turn gradient unit test, and logging wrong-mode amplification.

### 9.5 Coverage, honestly

This is still not exhaustive. Round 2's critic listed five gaps that no round has searched yet:
- **Grokking and delayed generalization.** Searched after this list was written; see 9.6.
- **Unswept venues:** ICML 2026 main and all ICML and NeurIPS 2026 workshops, ACL 2026 main and Findings, and TMLR 2026.
- **arXiv full listings from 2026-09-23 to 2026-10-14.**
- **Pre-2022 from-scratch state-tracking work:** partly checked in 9.6 (EntNet-style entity keys, TRADE copy and fast weights are already covered by P-022, P-238 and the delta-rule family). NTM and DNC curricula and SUMBT are still unsearched.
- **Chinese-language optimizer write-ups** (kexue.fm). Lower priority.

Roughly a quarter of what was kept rests on OpenReview abstracts the verifier could not open. Round 2 still found about four items worth a slot, so another round would likely add a few more, mostly diagnostics rather than new levers.

### 9.6 Grokking pass (2026-11-18)

A third, single-topic pass on the largest gap above: grokking and delayed generalization, the regime a chat pool read 100-200 times puts Planck in. It found six items, now ledger rows P-284 to P-289. All are diagnostics or data-design rules; none is a new lever that changes the top recommendations.

- **P-284 grokking ratio rule** (high confidence it applies, medium that it matters). Rule circuits form through grokking, at a speed set by the ratio of inferred examples to memorizable facts, not by data size (Wang et al. 2405.15071). RC-12's BIND, CORR and RULE cells are comparison and retrieval shaped, the kind that generalized there, which supports the core bet. The risk is a generator whose names and values repeat enough to be memorized. Test: three pool diversities at a fixed episode count, 5M.
- **P-285 long branch past the bpb plateau** (medium). Hierarchical generalization arrives long after in-domain loss flattens, with an inverted U in depth (Murty et al. 2305.18741). Nearly free when added to P-187 or P-189.
- **P-286 logit-scale drift readout** (low-medium). Softmax collapse after overfitting (Prieto et al. 2501.04697). Free logging; a projection arm only if drift shows. NorMuon may already blunt it (untested).
- **P-287 Grokfast** (low; mostly hype for language models). One T002 arm only.
- **P-288 grokking diagnostics** (low, watch). Weight-decay regime scores and late "anti-grokking" collapse; both single-group, toy-task results.
- **P-289 first-token vs continuation split** (medium, a scoring change). Two recall mechanisms form at different times (Daniels et al. 2507.01414).

Also checked and already covered: EntNet-style entity-keyed slots (P-022 update), TRADE's copy mechanism (P-238), fast weights (the delta-rule family), Tang et al. 2605.30233 (already in the files). Li et al. 2506.21551 tracks grokking through MoE routing only, so it does not apply to a dense model.

Still unsearched after this pass: ICML 2026 main and the ICML and NeurIPS 2026 workshops, ACL 2026 main, TMLR 2026, a full read of arXiv listings from 2026-09-23 to 2026-10-14, NTM and DNC curricula, SUMBT, and kexue.fm. Yield by pass: round 1 kept 70, round 2 kept 33 (about four worth a slot), this pass six diagnostics. Another pass would most likely add diagnostics, not levers.

---

## 10. Round 3: verification, last sweep, scoop check, triage and NAIRR fit (2026-11-19)

Max asked four things after section 9: reopen the OpenReview pages that section 9 left UNVERIFIED; say whether the list is now good enough to stop looking, and which items will actually make a difference; make sure nobody has already done what Planck is doing; and say whether the NAIRR request covers all of this. The work ran on 2026-11-18 and into the early hours of 2026-11-19 (US Eastern). No model was trained or run for this section.

- **Where the evidence lives:** the session scratchpad, folder `frontier/r3/`. Verification: `verify_g0.md` to `verify_g4.md` and `verify_chrome.md`. Sweep: three rounds of finder reports and their critics. Scoop: `scoop_papers.md`, `scoop_models.md`, `scoop_web.md`, `scoop_benchmarks.md`, `hypA.md`, `hypB.md`, adjudicated in `scoop_verdict.md`. Triage: three independent judges (`triage_evidence.md`, `triage_mechanism.md`, `triage_decision.md`) merged in `triage_consensus.md`. NAIRR: `nairr_bottomup.md` and `nairr_topdown.md`, reconciled in `nairr_fit.md` with the arithmetic in `nairr_fit_calc.py`.
- **Ledger changes:** verification results are applied in place to the rows and update entries they touch (each marked "verified 2026-11-18 via ...", "corrected", or "still UNVERIFIED"). The 260 kept sweep items are in LEDGER.md under "2026-11 round 3 additions": 105 new rows, **P-290 to P-394**, plus one-line update and watch entries. Consensus triage classes for every ledger id are under "2026-11 triage classes". The next free ledger id is **P-395**.
- **Honesty markers:** as before, (est.) means arithmetic, not a measurement. No H100 number here was measured. "Cached OpenReview abstract" means a local copy of OpenReview abstracts pulled through its search API at about 19:16 on 2026-11-18, before the bot check started. The finders read from that same copy, so checking against it confirms our transcription, not the paper.

### 10.1 Verification of the OpenReview-only claims

**What happened in plain words.** Claude could not reopen OpenReview. In the Chrome pass, the one forum URL tried (FjlS13holE) redirected to "Verifying your browser. Complete the check below to continue". Per the rules the pass stopped there: no click, no solve, no second OpenReview page. Max's own Chrome tab on IoYBVmgSfD showed the paper's title, which suggests the page loads when he is the one at it (not tested); that tab was not read or touched. Every result below therefore comes from other routes: arXiv full text and abstracts, Semantic Scholar title match, OpenAlex, conference virtual-site poster pages (NeurIPS, ICLR, ICML), authors' GitHub repos and homepages, and, for transcription checks only, the cached OpenReview abstracts.

**Scope.** Sections 9.1 and 9.4 spoke of "about 25 items" that depend on OpenReview. Those 25 are ledger items; together they cite **55 distinct OpenReview ids** (`frontier/or_ids.json`). All 55 were checked.

| status | count | detail |
|---|---|---|
| VERIFIED | 18 | 9 on an independent full text (arXiv or proceedings); 6 on an independent abstract (a conference poster page or the author's repo); 3 only against the cached OpenReview abstract (0fPxp7a50O, 1v67gG7IE1, DlGhhMUJyv: a transcription check, not an independent copy) |
| CORRECTED | 18 | at least one size, number, attribution, scope or date in our files was wrong; replacement wording applied in the ledger. 15 were corrected against a copy outside OpenReview; two of those (aZr3FJLVF9, TVa97w7bgA) were reached only in the Chrome pass, through author-owned copies found by GitHub repository and code search. 3 (32bXbTKcXx, 9vT48Lms48, BO2R94Ipk5) have no copy outside OpenReview and were corrected against the cached abstract alone (attribution and scope fixes) |
| NOT_FOUND | 19 | no copy found outside OpenReview by any route tried; our claims match the cached abstract except where the cached abstract itself forces a correction, which was applied anyway |
| REFUTED | 0 | no source turned out to say the opposite of what we cited |

**What this changes.** No bet loses its reason to exist. The material changes:
- **P-261 is now backed by a real, verified source** (Kalra and Patlolla, preprint and code on the authors' GitHub). But the "learns nothing in 600,000 updates" result is the arm where the previous-token circuit never formed or was reset, in a 0.69M attention-only model, not the training-only suppression arm. Pythia 160M-6.9B learn 2-18x later by rebuilding the circuit from backup heads. Section 1's item 1 ("slow, not unable") therefore holds only where a previous-token path exists or can be rebuilt; T002 gives every arm a previous-token path, so its design stands, but the reading does not cover arms without one.
- **P-262's numbers rest on one abstract** (DlGhhMUJyv, cached abstract only; single runs; reviewer ratings 1, 2, 2 in the papercopilot mirror of ICLR 2027 reviews). HCM50lrJTb has no erasure measurement at all, and CMfu7lDk2j is about weight-decay timing under Muon, not erasure. A separate verified toy paper from the sweep (2610.05292) puts erasure at the same summed LR x WD of about 1.5. The free dose computation (P-262a) stays needed; the training arms (P-262b) stay B (low).
- **P-263 now has a verified source.** StaleBench (Prause, "When Stale Facts Win", English v1 and the ICLR 2027 draft in the author's public repo; StaleBench covers 0.5B to 70B): in a causal audit of nine models from 0.5B to 14B the correction is the most-attended sentence, yet they answer with the stale value in 7-93% of documents; 0.5B-3B models follow position in 55-70% of position swaps; accuracy falls from 54% to 20% from one to four overwrites; short contexts are harder than long ones. Our "1B+" was wrong.
- **P-190's update needed three fixes:** Everett and Qiu's sizes are 51M-253M, not up to 458M; their momentum cooldown was never tested on Muon-family optimizers and barely helped AdamW, so that arm moves to P-220 (ADANA); and their log-time weight decay harms Muon at high overtraining.
- **P-022's update:** freeze the recurrent register early, for about the first 9% of tokens, not at 50% (aNcHQ9zdZM, corrected against arXiv 2610.26830: a 3.6M toy and 180M-1B hybrids).
- **Attribution fixes:** "GRPO raises repetition" is kNdGD5BTog, not BO2R94Ipk5. The three seed-noise numbers in the P-193 update are not in hAZxes7EMV (arXiv 2608.11859), which instead argues for wide per-arm hyperparameter search at 4M. P-268 loses 9vT48Lms48 (router lock-in, not optimizer state). P-265's "56-69% at 1.4B" rests only on usnA2Fiw5e. P-120's learning-rate claim belongs only to WFIAiSr1dN.
- **Side finding:** P-255 cites "van der Wal et al. 2606.25010 (PolyPythias)", but arXiv 2606.25010 is Baherwani et al., "Emergent Capabilities Arise Randomly from Learning Sparse Attention Patterns". The PolyPythias claim needs its own citation; flagged in the row.

**Every id.** Route "OR abs" means the cached OpenReview abstract only.

| id | title (short) | status | route | what changed |
|---|---|---|---|---|
| 0fPxp7a50O | Where You Put Attention Matters More Than How Much You Use | VERIFIED | OR abs | P-022: motivates layer placement only; the mirror case (one recurrent layer in an attention stack) is untested |
| 1v67gG7IE1 | A Count, Not a Fraction | VERIFIED | OR abs | P-188 unchanged; the 10x invariance is from a pilot; counts hold when background loss is flat (SFT, anneal) |
| 32bXbTKcXx | Same Count, Different Memorisation | CORRECTED | OR abs | P-120: constant LR, 3.2M-1B, canary leakage; the LR and 3M-151M claims belong to WFIAiSr1dN |
| 3eCpXmHlq3 | Muon Learns Facts Better | CORRECTED | arXiv 2610.02798 | P-189: parametric facts, not contextual retrieval; now a counterweight to VB7IbTZPPR |
| 4Dg0z4hCoY | Scaling Properties of Same-Family OPD | VERIFIED | arXiv 2609.32722 | P-196: Qwen2.5 0.5B-14B, math, single runs; bigger teachers stop helping |
| 6d49LBQlIk | The Best Optimizer Depends on Batch Size | VERIFIED | arXiv 2610.08975 | P-190, P-207: 124M, 128K-2M batches, reversals of 0.004-0.021 nats |
| 9vT48Lms48 | When Routing Must Be Right | CORRECTED | OR abs | removed from P-268: router lock-in in hybrids, not optimizer state |
| BO2R94Ipk5 | Prior Reward Hacking Impairs Subsequent Learning | CORRECTED | OR abs | P-197: supports "start from hack-free checkpoints" only; not evidence for P-266 |
| CMfu7lDk2j | Delayed Effects of Temporary WD in Muon Pretraining | VERIFIED | arXiv 2611.17876 | P-262, P-191: a WD window's effect reverses with timing at 162M-1.4B; not an erasure source |
| DlGhhMUJyv | Kept, Hidden, Erased | VERIFIED | OR abs | P-262: numbers match; single runs; the 0.5B result is a pretrained LM |
| DuYY0UuDGb | An Error in the Context Is a Demonstration | CORRECTED | NeurIPS 2026 poster page | P-266: contagion is flat from 0.36B to 3.2B, not "worse when smaller" |
| EAy9tE5Br2 | Mixture of Convolutions | VERIFIED | author repo, Parameter Golf PR #966 | P-264: Transformer gain and init x LR sensitivity confirmed; GDN gain and sizes unreachable |
| FjlS13holE | Associative-Retrieval Regime Entry Varies with Head Partition | NOT_FOUND | | P-256 keeps UNVERIFIED; verified counterweight Adler 2509.22840 added |
| FnT1urzlSH | Stored Sampling Noise Changes Recall Learning in AdamW | NOT_FOUND | | P-268 keeps UNVERIFIED |
| G2HBCKp9Bd | A Deafening Silence | CORRECTED | arXiv 2610.09835 | P-267: 39.4-67.9% over 7 stable settings, including a tied model; only 5.6-14.6% on in-distribution English SFT; Adam-only |
| GPYH0UPouD | Does Muon's Pretraining Advantage Survive Fine-Tuning? | VERIFIED | NeurIPS 2026 OPT poster page | P-190's auxiliary-LR axis (Qwen3-4B, 1.8745 vs 1.8714 tuned) |
| HCM50lrJTb | Scale Weight Decay and Train Better | CORRECTED | arXiv 2607.23777 | P-262: no erasure measurement; Muon-SW 21.7-29.4% faster on 72.7M-932M MoE, single runs; July 2026 |
| HyuEyf9KQQ | Head Dimension Matters for Long-Context Retrieval | VERIFIED | arXiv 2610.32213 | P-256: 0.6B and 3B after 256K-context training; transfer to 5M untested |
| IoYBVmgSfD | Multiple Task-State Descriptions Interfere | NOT_FOUND | | P-253, P-263: "1B+" not stated; verified neighbour CICM 2609.38866 carries the same use |
| Jh8Y1gSYfN | Tied, Not Won | NOT_FOUND | | the P-193 exponent-difference test does not depend on it |
| K45J4B4Yku | Works Where Calibrated, Fails Where Misused | NOT_FOUND | | P-193's three numbers stay UNVERIFIED; E3 independently supports the data-order point |
| NXWLaiiJKm | Delta-Rule Memories Store More Than They Recall | NOT_FOUND | | P-022: "K ramped over 50%" is Planck's design; Boesch and Wee 2609.16183 is the verified neighbour |
| OYQBr0EB84 | Sorted by Date | NOT_FOUND | | P-263: its 11.7-point result is labelled exploratory; "1B+" not stated |
| P2VtelR5KS | Optimizer Memory Schedules | CORRECTED | arXiv 2609.04577 | P-190: 51M-253M; momentum cooldown moves to P-220; log-time WD harms Muon at high overtraining |
| RG4cxb4sab | StateWise | CORRECTED (minor) | arXiv 2610.33766 | P-272, P-008: the relapse doubling is Llama-3.1-8B only; Qwen2.5-7B stayed flat |
| SCKKNDxgUc | SIRRA | NOT_FOUND | | P-008 cell type stands; "1B+" not stated |
| TVa97w7bgA | Structure Before Attention | CORRECTED | author homepage (Chrome pass) | P-264: the gain is mainly windowed (-2.1% vs -0.6% dense at 8k); 81M / 253M and the width-one claim stay OpenReview-only |
| UUldzWxIL4 | Linear Attention lacks Incentive for Associative Recall | NOT_FOUND | | P-265: the methods "narrow", not "close much of", the gap; linear attention only |
| VB7IbTZPPR | Optimizers Select Algorithms | VERIFIED | NeurIPS 2026 OPT poster page | P-189's RP-MQAR family drops UNVERIFIED |
| VJFfO4tLMB | Recent Training is Protected by Adam's Second Moment | VERIFIED | NeurIPS 2026 OPT poster page | P-268: its knobs are beta2 and resetting v, not beta1 |
| WFIAiSr1dN | Acquire, Then Retain | NOT_FOUND | | P-120 keeps UNVERIFIED |
| WJgVOfL8UV | The Ephemeral Cost of Switching Optimizers | VERIFIED | arXiv 2610.22650 | P-190: 1.2B pretraining, SFT up to 14B, little mismatch penalty |
| aNcHQ9zdZM | Competing Learning in Hybrid Architectures | CORRECTED | arXiv 2610.26830 | P-022: freeze the recurrence for the first ~9% of tokens, not at 50% |
| aZr3FJLVF9 | When Stale Facts Win (StaleBench) | CORRECTED | author repo (Chrome pass) | P-263 now verified: 0.5B-70B, positional heuristic at 0.5-3B, overwrite dose |
| dG86lHw2rk | Scaling Width and Scaling Compute Teach Different Algorithms | NOT_FOUND | | P-260: the numbers are for parity, not recall; narrower-and-longer wins all 21 equal-compute comparisons |
| eYOSZZ9Fal | Information Abundance Paradox | VERIFIED | arXiv 2608.12218 | P-265: 20M-750M from scratch; context-rich training moves learning into attention |
| g6ZGK1lKNf | Repeat or Retrieve? | NOT_FOUND | | P-266: the from-scratch reversal used 50% corruption; a 50% arm added |
| gpL5ZAN98s | SF-DPO | NOT_FOUND | | P-008 update: about 0.5B-32B, not 1B+; four preference arms with chosen-response anchoring |
| hAZxes7EMV | Small-Scale Experiments: Are We There Yet? | CORRECTED | arXiv 2608.11859 | P-193: the three numbers are not in it; it argues for wide per-arm HP search at 4M |
| kNdGD5BTog | Tracing Repetitive Degeneration | NOT_FOUND | | P-197: the repetition citation moves here |
| o6XoxYhaYs | The Many Faces of Short Convolutions | NOT_FOUND | | P-264: the LR-range benefit is credited to the filter, not the gain; a depthwise k=1 conv equals gain-only |
| oUgyEOMOLU | KD During Mid-Training Favors Reasoning | VERIFIED | arXiv 2609.01532 | P-215: reverse KL on the 20% lowest-entropy tokens; a mid-training effect at 1B |
| pD2TCZ6Zsq | Entity-Conditioned Addressing | NOT_FOUND | | P-022: needs an explicit entity id per write; the same codes as input tokens fail |
| pkymKR1YiW | Decided While Reading | NOT_FOUND | | P-008 rule-after-fact cell stands; sizes not given |
| q42Wqkwe3q | When Data Can Teach: Prerequisite Circuits | CORRECTED | authors' GitHub preprint | P-261 verified, scope corrected (above) |
| qC88InjePl | What Is a Repeated Token Worth? | CORRECTED | arXiv 2610.05591 | P-187: half value at ~4-5 epochs up to 20 tok/param; "15 epochs at 127M" is a different quantity |
| qjrIQXooV3 | K-Scaling | NOT_FOUND | | P-207 keeps UNVERIFIED; the decay-ratio axis stands on its own merits |
| s5LDNAWhYk | LLMs Can Follow Instructions, But Not Many at Once | VERIFIED | arXiv 2608.12426 | P-008 k-constraint cell; quote v1 (15 models, 41% vs 5.7% at k=8) |
| u5o84zeMEb | Anytime Pretraining | CORRECTED | arXiv 2602.03702 | P-218: 300M only to 16x Chinchilla; AdamW without WD; Feb 2026 |
| uLDuoLNj9H | Pretraining with Masked Backstories in a Toy World | CORRECTED | ICLR 2026 Sci4DL poster page | P-265: named authors, April 2026; the 0.025% figure is in the 2027 abstract only |
| usnA2Fiw5e | RAM: Routed Associative Memories | NOT_FOUND | | P-265: the 56-69% rests on this id alone and bears on P-022 more than P-265 |
| vdMp4RjCed | Strong-to-Weak OPD against GRPO | CORRECTED | arXiv 2611.08453 | P-196: "50x" is the distillation stage at 1 rollout (about 6x at 8); students 1.7B-8B |
| vzn9mvMuJq | You Changed Your Mind, The Model Didn't | VERIFIED | arXiv 2610.06496 | P-008 rejected-change cell confirmed; all models 4B+ |
| yhRk81TgYO | State-Resolving (Sieve) Attention | VERIFIED | NeurIPS 2026 poster page | P-020 recency arm, abstract level |
| yochffTF7t | muLM | VERIFIED | NeurIPS 2026 poster page | watch entry unchanged; no weights or full text |

**NOT_FOUND: what was tried for each of the 19.**
- Exact-title and abstract-keyword queries on the arXiv export API and on arxiv.org search (heavily rate-limited, HTTP 429 for long stretches).
- Semantic Scholar title match ("Title match not found" for all 19 in pass 1; mostly HTTP 429 in the Chrome pass).
- OpenAlex title search; GitHub repository and code search on titles and method names; Hugging Face search for some.
- General web search on exact titles in pass 1. In the later passes the session's shared web-search budget was spent, so no more web searches ran.
- The NeurIPS 2026 virtual site (it lists workshop papers; ICLR 2027 decisions are not out, so there are no ICLR 2027 poster pages yet) and papers.cool ("Not Found").
- In the Chrome pass, one OpenReview forum URL, which hit the bot check.
- All 19 are anonymous ICLR 2027 submissions; none of these routes found a public copy as of 2026-11-19.

**What only Max can do.** Open these in his own browser (completing the check himself, or signed in). In this order:
1. **99EZmxKpJh**, the full text of "Two Kinds of Nothing". It is not one of the 55; it is the only source of the new A-class LOOKUP row P-296, read so far from a workshop abstract.
2. **DlGhhMUJyv**, the only source of P-262's numbers.
3. **IoYBVmgSfD**, already open in his tab.
4. **Heidi-3** (TMLR / CoLLAs 2026), the one open novelty question (10.3).
5. The other 18 NOT_FOUND ids, then the round-3 sweep items that rest on OpenReview-only abstracts (listed in 10.6), as time allows.

### 10.2 The last sweep

Three more rounds, each aimed at the gaps the previous round's critic named. A verifier re-checked every item before it was kept, opening the source wherever it was reachable; where only an abstract, a mirror or nothing could be opened, the ledger row says so (abstract only or UNVERIFIED). Corrections are in the ledger rows.

| round | what was searched | found | kept |
|---|---|---|---|
| 1 | ICML 2026 main (PMLR v306: 6,552 abstracts parsed, about 1,690 titles and 315 abstracts read, about 40 PDFs); all ICML and 32 topical NeurIPS 2026 workshops via the virtual sites (about 6,800 titles, 386 poster abstracts); ACL, EACL, AACL, TACL and CL 2026 (8,694 papers, about 1,800 titles, 160 abstracts); TMLR, JMLR, AISTATS and CoLLAs 2026 (about 1,090 titles, 79 abstracts); the arXiv cs listing for 2026-09-23 to 10-14 via OAI-PMH (11,928 records, about 2,930 titles, 156 abstracts, 26 full texts); pre-2022 from-scratch state tracking (26 full texts, 105 citing titles); Chinese sources (kexue.fm through its feeds and translation mirrors, MiniMind; Zhihu unreadable); lab and industry releases June-November 2026 (Hugging Face API over 42 orgs, 17 model cards, Liquid's blog) | 129 | 105 |
| 2 | NeurIPS 2026 main, read in full this time (7,895 papers; 2,714 never-retrieved poster pages fetched; about 3,600 titles and 275 abstracts); ICLR 2026 main and 8 workshops (about 2,450 titles, 331 abstracts); the arXiv listing for 2026-05-15 to 09-22 (37,497 records, about 2,940 titles, 190 abstracts); OpenAlex forward citations of 16 anchor papers (1,775 citing works, 257 abstracts); Marin's open experiment logs (issue #9451 with 384 comments, 247 registry entries, 538 experiment issues) | 116 | 83 |
| 3 | COLM 2026 workshops (18 workshops, 1,080 abstracts); the 2025 main tracks of ICML, NeurIPS, ICLR and COLM (13,790 entries, about 2,930 titles, 260 abstracts, 28 full texts); BabyLM 2023-2025 (101 abstracts) and the ICML and NeurIPS 2025 workshops on small models, tokenizers, optimization and multi-turn interaction (539 abstracts); the arXiv listing for 2026-01-01 to 05-14 after removing venue papers (28,219 records, about 1,980 titles, 150 abstracts, 25 full texts); the ICLR 2027 submissions the earlier keyword sweep never retrieved (27,386; four topical subsets, about 740 titles, 120 abstracts) | 114 | 72 |

Found minus kept is 99, while the verifiers' own drop log counts 104; the files do not explain the difference of 5, probably merges logged as drops.

**Kept items by consensus class** (classes defined in 10.4):

| round | kept | A | B (strong) | B | B (low) | E | C (needed) | C | D |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 105 | 1 | 2 | 9 | 10 | 4 | 22 | 17 | 40 |
| 2 | 83 | 1 (merged into P-238) | 0 | 10 | 5 | 3 | 11 | 15 | 38 |
| 3 | 72 | 0 | 0 | 6 | 13 | 3 | 12 | 15 | 23 |
| total | 260 | 2 | 2 | 25 | 28 | 10 | 45 | 47 | 101 |

**Where they went.** Every A, B, E and C (needed) item got its own ledger row (P-290 to P-394, 105 rows), except four exact duplicates folded into their twin's row and three items folded into an existing row (CPLM Plus into P-238, the Marin softcap ladder into P-246, MuonIO into P-217). The C (other) and D items are one-line update entries under their parent bet, or watch lines. Each line carries its round id (R1-01 to R3-72) so the triage list in the ledger can point at it.

**Per-round yield.** The finders tagged 29, 22 and 24 items as levers or efficiency gains. By the three-judge consensus, items that are likely to matter (class A) went 1, then 0 new (round 2's one A was evidence for the existing CPLM row), then 0. Diagnostics and parks were 75%, 77% and 69% of each round's kept items. Round 1's critic expected 3-8 more lever-shaped items and round 2's critic expected the same; the rounds did find B-class arms (state lines in training data, dual objectives for the repeated pool, warmup-stable-only before SFT, value-path variants, recovery-turn supervision) but no new A after round 1. The one round-1 A item, the LOOKUP absence fix (P-296), is also the most practical find of the whole pass: a cheap data change, with a measured effect at Planck's own size.

### 10.3 Has anyone done Planck since 2026-10-02?

**Plain answer: no.** As of 2026-11-18, nobody has shown a from-scratch model below 150M passing a strict multi-turn state test (corrections, rule persistence, same-type binding, own-answer consistency, no loops, lookup), and nobody has published a measured minimum-size curve for those skills. The verifier re-opened every ranked candidate at its source except two that exist only on OpenReview (ZeroHero and Routing the Past, UNVERIFIED) and the gated BananaMind-3 previews (metadata only, terms not accepted); a few out-of-lane items (the BabyLM 2026 items, Chess-World-Model, Pearl-Chat, Homebrewed SLMs, Turnslide) rest on the finders' readings. Every new sub-150M multi-turn number found is small-n, self-reported, produced by a harness rather than the naked model, or a failure. Threat levels: none high, one medium, the rest low or none.

**The most important correction: RC-12 is still not locked.** One finder file said three times that Planck's October lock predates the new releases. It does not. `prereg/RC-12.draft.txt` still says "DRAFT, NOT LOCKED" (DRAFT 0.1), PLAN.md says the lock slipped, the repo has no GitHub release, and neither Software Heritage nor the Wayback Machine holds a copy. So no third party holds a dated copy of RC-12's design, and Planck's priority currently rests on git dates, which can be rewritten.

**Ranked threats** (all re-opened by the verifier unless marked):

| # | candidate | first public | threat | what it is |
|---|---|---|---|---|
| 1 | pico-165m (Hugging Face fausap) | 2026-10-14 | medium (framing, not a scoop) | A notebook titled "The Parameter Floor for a Conversational LLM" asking Planck's question with rule-graded multi-turn probes and release gates. It says naked 100-150M models plateau at about 25-35% on its probes, but the notebook has no saved outputs, so that is prose only. Its good scores (0.68 and 0.81 vs SmolLM2-360M's 0.51 and 0.31) come from a two-model note-taking harness. No correction or own-answer families, nothing at 3-30M, no seeds, no sealed split. |
| 2 | Textile Labs Loom-Weave-4 (55.6M) and the Loom-Planck ladder (300K-1.2M) | 2026-10-06 to 11-01 | low | Name collision; owns the loose "how small" headline. Weave 4: 42 of 44 long-conversation turns on target, but only 4 recall items; size is confounded with training time. |
| 3 | KW5-230M Instruct (Kiswahili) | 2026-10-28 | low | Froze its bars before training, trained on changing lists and persistent instructions on purpose, and still missed: persistence 35% (n 20), changed-list answers 5% (n 40). |
| 4 | Laghu 152M and Gargi 384M (Udgam Labs) | 2026-10 | low | Pre-registered, but its test is two-turn follow-ups only, and Falcon-H1-Tiny-90M beats it. |
| 5 | NeuralAI Nae1 (100M) | 2026-10-07 / 11-11 | low | The only new sub-150M model trained on corrections; fails its own memory gate (corrections 3 of 4); 1,024 positions. |
| 6 | ZeroHero (ICLR 2027 submission) | 2026 | low, UNVERIFIED | 5-10M executors from scratch on agent tasks with an LLM oracle; no size sweep in the cached abstract. |
| 7 | r/LocalLLaMA thread and an unexecuted ~9M state-tracking recipe | 2026-10-25 | low | The idea is in the air; nobody trained it. |
| 8 | StateWise (2610.33766) | 2026-10-24 | low | 7-8B; models track the state but do not obey it. Already P-272. |
| 9 | The Capacity of In-Context Binding (NeurIPS 2026 workshop) | 2026 | low | 70M-2.8B; one word of question wording moves Pythia-410M recall from 0.226 to 0.896 (now P-304). |
| 10 | tiny-rl on nanogpt-ladder-v3 (8M-64M SFT and RL checkpoints, no write-up) | 2026-11-11 to 11-15 | low, watch | An academic-grade small chat RL ladder with no card yet. |

Also low or none: nico's unreleased 6.5M harness model (HN), StateLM (no model yet), the ELIZA mini-model wave (nothing trained), Routing the Past (UNVERIFIED), LM Playschool (smallest entry 0.8B), the new multi-turn benchmarks (smallest model 270M; the smallest correction test is Capitulation 2611.02318 at 0.5B), BananaMind-3 (gated, in training, the most likely next public sub-150M multi-turn table), HobbyLM, FCP, SmallTalkLLM (dormant), and MNN-Pocket-0.3B (no multi-turn state numbers).

**Panel candidates for RC-12** (the lock has not happened, so they can still enter the panel properly):
- **Tier 1, add before the lock** (little engine work; each protects a claim sentence): MNN-Pocket-0.3B (protects "no model at or below 0.6B passed"), Loom-Weave-4 (inside Planck's claim range; same engine path as Loom-Spark-3.2), LFM2-2.6B-Exp, Qwen3.5-2B and LFM2.5-1.2B-Instruct (test "everything up to 2.6B fails").
- **Tier 2, only if the engine takes them cleanly**, otherwise list as known and unscored with the reason: Laghu; Nae1 (only if a 2K-position load is shown exact); SmolLM3-3B, granite-4.2-3b, Aion-1.0-Instruct-2.3B, MiniCPM5-2B, gemma-4-E2B-it.
- **Thinking-first models, as labelled diagnostics after Max rules on thinking mode:** LFM3-1.7B, Raon-1 nano, LFM2.5-2.6B.
- **After the lock, as a labelled harness row, never a naked panel member:** pico-165m.
- Panel work must not delay the lock again.

**What Planck should do:**
1. **Get a third-party timestamp now, then lock.** A Software Heritage "save code now" of the public repo, optionally with a GitHub release, dates the RC-12 design today without touching sealed content. Both count as publishing, so each needs Max's explicit yes. Then finish `rc12/LOCK_STEPS.md`.
2. **Cite the neighbours honestly** (pico, KW5, Laghu, Loom, StateWise, the binding poster, Capitulation 2611.02318), use "MaxGPT-Planck" in titles, and never claim the lock predates Loom-Planck.
3. **Treat two results as warnings for the bet, not just citations.** KW5 trained on state data and still failed persistence and list updates, and pico reports a naked-model plateau at 100-150M. Both say SFT data alone may not buy persistence or corrections, which raises the value of relapse training (P-272), RL (P-197) and the update register (P-022). Pre-register the outcome "SFT gets RECALL but not PERSIST or CORR" so it reads as a finding. pico also shows note reading is a trained skill (SmolLM2 given the same notes scores 0.31, 0.42 with perfect notes), so P-005's gate must be measured on a checkpoint SFT'd with notes in context.
4. **Add a post-lock wording check** for BIND and LOOKUP probes (P-304).

**Our hypotheses that turned out to be published** (2 of 44 checked; the verifier re-opened both sources):
- **HL (M14, data-order soups)** is published at 7B: OLMo 2 (2501.00656) Table 14 souped three mid-training runs on different data orders and "consistently equals or outperform[s]" the best single run over six mixes, with one exception (mix E GSM*, 43.0 vs 60.5). M14 / P-203 becomes a replication; read it per RC-12 family.
- **P-241 (guided-policy self-distillation)** is published in another domain: Cideron et al. 2410.06084 (ICLR 2025, MusicLM) distilled CFG-augmented predictions into the plain model, with a diversity reward. P-241 is relabeled tweak-of-known in the ledger; the chat version (guidance from ablating a fact, rule or correction turn) is still untested, and its echo and loop kill rule stays hard.
- **Close calls that change a test design** (from hypA and hypB; the verifier did not re-open these): an SFT-optimizer arm in M3 (already in P-190's 2x2); when the batch grows with tokens the co-tuned LR may need to rise (X7); compute-matched warm-ups never beat natural language and fade by 1B tokens (X1, now verified in the sweep as 2610.40051); a fade-to-zero arm for P-261, since abrupt scaffold removal failed (2609.10287); the induction lockout of P-282 disappears at 4+ layers; OPD evidence is mixed by skill (X6); CPLM's own mechanism predicts a smaller gain at Planck's vocab-to-width ratio (about 43); AdEMAMix showed no test gain under heavy repetition (HK).
- **Coverage gap:** hypotheses HS to HAC (11 items in section 4a) were checked by neither prior-art file.

### 10.4 Triage: what will make a difference and what will not

**Method.** Three judges classed every item independently, each through a different lens: replicated small-scale evidence, whether the mechanism hits a known way a 3-30M model fails RC-12, and decision value per H100-hour. A consensus step merged them into 409 rows (the M, R and X items, P-237 to P-289 split where a judge split them, the update entries, and all 260 sweep items). 313 rows were unanimous; 78 were not (72 split 2-1, 4 three ways, 2 with only two votes). The majority was kept in every 2-1 split and every split is listed with each judge's reason in `triage_consensus.md`. RC-12 has never been run on a Planck checkpoint, so every "effect on RC-12" below is a forecast (est.).

**The consensus top list: the items most likely to make a difference.**

| rank | item | ledger | expected effect (est.) | confidence | cost (est.) | main risk |
|---|---|---|---|---|---|---|
| 1 | RL with program-checked rewards on the skeleton generator, with turn-level rewards, a horizon curriculum and a KL-anchored arm | P-197 (with P-198, P-291, P-325) | +10 to +30 points on RC-12 dev cells where SFT pass@16 is above 0; about 0 where it is 0; could move the passing size down one curve point if many failures are near misses | medium on dev, low on held-out | 20-40 H100-h at 5-10M (or 5070 time), about 50 at 30M, plus a 500-1,000-line batched sampler | reward hacking; a 5M student may start at 0 on 12-turn cells; evidence is from scratch at about 38M-360M on other tasks, never conversation state |
| 2 | LOOKUP absence forms: train both "stated gap" and "silently missing" | P-296 | the LOOKUP X unit could move from near floor to near ceiling. In the source a model trained only on stated gaps read silence correctly 14% of the time; a 3.3% silent minority lifted both forms above 96%. The repo confirms the gap: the generator teaches only explicit "not found" tool results while RC-12's X unit asks for a silently missing key | medium on direction, low-medium on size (one-author workshop abstract; 14.4M, 16 seeds) | about 0.5-1.5 H100-h | narrow (one sub-cell), but one failing family blocks Level R |
| 3 | Correction-data dose with held-out names | P-271b | 0 if Planck's share already sits in the source's 5-15% window; up to 15+ points on held-out-name corrections if not (0% never learned the override rule, 5-15% reached 0.90-0.91 held out, 30% fell to 0.757 held out) | medium | about 2-4 H100-h | one group, from scratch, 3 seeds |
| 4 | Grow the chat pool instead of repeating it, with larger Apache-2.0 renderers | M24 (P-162, P-187), P-201, P-335 | size unknown; it targets the failures Planck has actually measured (E004's wording shortcuts, E005's alias gap). Outside evidence is loss-only: paraphrase beats repetition below about 50M | medium | tens of GPU-h of rendering plus 2-5 H100-h of arms | the pool may not be the binding limit at 5M; renderer voice |
| 5 | Copy-Paste LM head, with the CPLM Plus arms | P-238 | 1.5-4% bpb at 5M if Marin's "gain halves per width doubling" continues below d512, which is outside the fit; possible retrieval gain (135M sliding-window needle 14.3 to 99.4, 3 seeds); RC-12 effect unproven | medium-low | about 1-1.5 H100-h plus T002 minutes; a KV-cache decode path | may copy the stale value or its own text; equal-time gains are much smaller |
| 6 | On-policy distillation, same-size per-family RL specialists merged first, the 400M Planck teacher second, always against teacher-free and shuffled-teacher controls | P-196 (with P-333, P-390, P-391) | 0 to +15 points on cells with headroom; much of OPD's gain may be the student sharpening itself, which the controls will show | low | 15-25 H100-h for the same-size route; about 35 for the teacher plus 10-20 for arms | no chat OPD evidence below 1B; verified evidence says bigger teachers stop helping; needs Max's "from scratch" ruling |

**Near misses** (B strong), each with the trigger that would make it A: loop-onset preference training (P-204) if the first RC-12 dev read shows loops binding at T 0.6; own-history SFT with program-corrected targets (P-067, with P-299's minimal edits) if golden-history and own-history scores differ a lot; new generator cell types (M18 and the P-008 cells) wherever the coverage audit shows RC-12 dev testing a cell the generator never teaches; the update register or dynamic conv (P-022, P-200) if T002 shows, from escape-time evidence (P-350), that plain attention cannot learn latest-wins at 3-5M; LR, weight decay, optimizer and decay at the real horizon (P-190, P-191), near-certain to move but worth 0.5-2% bpb, not a cell flip; relapse DPO on rule contrasts (P-272b) with the squeezing guard (P-376); same-size RL specialists plus MOPD (P-333).

**The classes and how much falls in each** (409 consensus rows):

| class | meaning | rows | share |
|---|---|---|---|
| A | likely to move RC-12 margins or the passing size | 8 (6 distinct items) | 2% |
| B | small but real, or cheap and uncertain; worth a cheap screen. 10 rows are B (strong); 52 are B (low), expected to tie or move bpb by about one 5M seed SD (0.2-0.4%) | 107 | 26% |
| E | saves hours or calendar, not quality | 13 | 3% |
| C | does not improve the model; 67 of these are C (needed), without which some other result gets misread | 143 | 35% |
| D | no measurable difference expected at 3-30M, or direct evidence against (8 matter only for the microcontroller release) | 138 | 34% |

Small gains are real and they add up, but at 5M a 0.3-0.5% bpb gain is about one seed SD. Each B item needs 3-5 seeds and the multiplicity rules (M5) to be seen at all, so bank them in batched 5M screens of 10-20 arms, not one at a time. The free C (needed) items are the cheapest protection the project has: escape-time reads for T002 (P-350), fp32 scoring (P-313), the calibrated untrained-model null (P-307), the wording check (P-304), seed-distribution reporting (P-394), the teacher-free OPD control (P-390).

**Should we stop looking? Yes, for broad sweeps.** All three judges said so independently, and the per-round numbers agree (10.2): items that matter went 1, 0, 0 per round while 69-77% of every round's kept items were diagnostics or parks. The architecture search in particular is done: Planck's own screens, Marin's roughly 250-idea ablation set and the Parameter Golf meta-analysis agree that single architecture or optimizer changes at this scale move loss by 0 to about 1.5%. The biggest measured effect in the whole project is a data fix (E005: alias-correction accuracy at 135M went from .10-.52 per seed to .88 or better once alias corrections were in the training data). The next A item is more likely to come from Planck's own first RC-12 dev read than from another paper.

Keep three narrow searches open:
1. **One failure-driven search after the first RC-12 dev read.** If aliases or references fail: entity-coreference data design. If loops dominate: loop training without decoding tricks. If LOOKUP fails: table-format curricula. If own-history drift dominates: exposure bias and scheduled sampling for small transformers, the one thread the mechanism judge flags as under-searched relative to an A item.
2. **A monthly watch on small-model RL and OPD for multi-turn state** (arXiv cs.CL and cs.LG, the BabyLM interaction track), because item 1 gains most from a new result.
3. **Generator data design for state tracking at small scale:** distractor design, distance distributions, interleaving families.

Not worth funding: more full venue reads, more industry-report reads beyond MNN, Liquid and MiniCPM, inference-time methods (scaffolds under the locked rule), and the rest of the ICLR 2027 pool.

**The most valuable next step** (cheap, on the 5070, as soon as a 5M SFT checkpoint exists): score RC-12 dev on 5M SFT checkpoints, together with the free failure splits (P-328's operation-vs-value split, P-240, P-242, P-271's repeated-lure probe, P-263's stale-twin and frequency-cheater audit, P-206's loop and echo rates, P-272's relapse score). That decides which near misses become A. Caveat found in review: the repo (origin/main) records no 5M Planck SFT checkpoint yet; the 5M checkpoints that exist are starter-mix pretrains without the chat pool (also noted by P-296's verifier), so this step first needs one short identical SFT on an existing 5M trunk, or it runs right after the pool build. Then, before the Nov 22 pool build or by dated amendment after it, add the silent-absence share (P-296) and set the correction dose with held-out names (P-271b): the two A items that live in the pool. Then lock RC-12 with a third-party dated copy.

### 10.5 NAIRR fit

**The reconciled answer.** The two independent estimates (bottom-up and top-down) were reconciled line by line; every disagreement above 25% was re-checked against the filed request, the Ultra runbook, the ledger and the measured 5070 rates.
- **Ultra plus the curve as filed: covered.** Their real cost is about 1,400 of the 1,950 requested GPU-hours in the central case (1,036 to 2,342), leaving about 550 spare. At the request's own conservative rates (the high column) it runs about 390 over, so the week-1 benchmark decides whether there is any spare at all.
- **The new research that truly needs a big GPU: covered by that spare**, about 230 GPU-hours with reruns (105 to 457): H100 calibration, a 400M Planck teacher, data renders with 31B-122B open models, and 30M arms. It fits only if the request's sentence about where saved hours go is widened to allow it.
- **Everything together: not covered.** With the curve extensions (2 more seeds at 30M, 60M and 150M; 150M to 300B tokens; compute-multiplier fits) and conditional items, demand is about 3,200 NAIRR GPU-hours in the central case (2,300 to 6,850), about 1,250 more than the Start-Up allows (350 to 4,900). The 2,000-hour cap rules out a bigger Start-Up, so the rest needs a full research request.
- **Max is right that the small models are cheap: cheap enough that they should not use NAIRR at all.** The 5M-20M research is about 1,500 RTX 5070-hours including 2 extra seeds per size (1,080 to 2,180). The 5070 has about 2,060-2,130 free hours between Dec 4 and Apr 5 once the 5M-20M curve has trained compiled. NAIRR hours go to Ultra (about 700), the 150M point (about 300, plus about 600 if it goes to 300B tokens) and the compute-multiplier fits (about 170 per nominated change).
- **Two limits bind before hours do.** No NAIRR result can exist before the Dec 4 freeze (the request is paused; a decision takes 2-3 weeks and first use can come 2 weeks after that). And Anvil AI's `ai` queue averaged a 119.5-hour wait over the last 30 days with a 48-hour job cap, so Ultra's 4-job chain takes about 6.5 weeks of calendar, not the request's 4.

| NAIRR GPU-hours (est.) | low | central | high |
|---|---|---|---|
| Ultra per its runbook (data builds, proxy A/B, launch test, 100B tokens, post-training) | 537 | 704 | 934 |
| Filed Planck lines (30M / 60M / 150M with 3 seeds each, 150M extension, long budgets, dose-response, scoring, setup) | 450 | 571 | 1,018 |
| Reruns | 49 | 128 | 390 |
| **The Start-Up as filed, real cost** | **1,036** | **1,403** | **2,342** |
| Spare inside 1,950 | 914 | 547 | -392 |
| New research that needs an H100, with reruns | 105 | 226 | 457 |
| Curve extensions, with reruns | 1,155 | 1,528 | 3,067 |
| Conditional items (pool re-render with a 31B+ teacher, recipe-v2 curve, Ultra decay test) | 0 | 55 | 985 |
| **Everything on NAIRR** | **2,296** | **3,211** | **6,852** |
| **Beyond the 1,950 (needs a full request)** | **346** | **1,261** | **4,902** |
| RTX 5070-hours, not NAIRR (5M-20M research, 2 extra seeds at 5M / 10M / 20M, a second dose budget at 20M) | 1,083 | 1,511 | 2,177 |
| RTX 5070 free hours Dec 4 to Apr 5, after the compiled 5-20M curve | 2,128 | 2,092 | 2,056 |

The low, central and high columns are named by demand: fast H100 and few reruns, middle rates and 10% reruns, and the filed request's own conservative rates with 20% reruns. H100 throughput is unmeasured in every column.

**What to do.**
- **On the RTX 5070:** the 5M, 10M and 20M curve from Dec 4 to about Dec 20 (about 357 h compiled); then the long-horizon optimizer and LR trunks first (about 650 h central), T002 and the toys, batch 2 screens and one-change arms, and the 5-10M RL and OPD students. Done around early March in the central case. The 30M points fit here as a fallback (about 21 days).
- **In the Start-Up:** a week-1 benchmark that tells which column applies, then the filed scope, then the spare in this order: the 400M teacher (about 40), teacher pilot renders (about 30), 3x and 10x teachers (about 20), 30M arms (about 45), 30M RL (about 25), then 2 extra seeds at 30M and 60M (about 70). Keep about 10% unspent until Ultra's first job shows its real speed.
- **In a full research request:** about 5,000 GPU-hours on Anvil AI for 12 months (up to 6,000). That is near the lower quartile of 2026 Anvil AI full awards (4,250 to 4,750 depending on the method) and well under the median of 10,800. File by Jan 15 with week-1 measured rates, or by Feb 15 if no job has run by then.
- **Request text to change before the PI resubmits (Max's call):** widen the savings sentence so the spare can go to H100 calibration and H100-only experiments; recost the small-model lines at the compiled 5070 rate (the request used the old eager rate), which frees 171 hours to name Ultra's data and setup steps (about 55) and the H100-only research (about 115) at the same 1,950 total; replace "fully booked" and "cannot finish 30M", which the compiled numbers contradict; move Ultra pretraining to weeks 1-7; raise storage from 400 GB to about 1.5 TB; drop the under-18 checklist item, which no longer applies. PLAN.md Phase 4 is also stale: it still places 30M on the 5070 and 60M and 150M on the Titans, which are gone.
- **Still UNVERIFIED:** whether the Start-Up's 3 months run from the award or from first use; whether several processes may share one GPU on Anvil; whether one project can hold a Start-Up and a full award at once.

**Correction to section 5 (dated 2026-11-19).** Section 5 said its first-six-weeks plan, "about 270-360 H100-hours (est.)", fits "inside one 2,000-hour Start-Up". That is wrong as written. The Start-Up request as filed (paused; the PI will resubmit) asks for 1,950 GPU-hours, and its own lines already give those hours to Ultra and the filed Planck curve; recosted, they use about 1,400 in the central case and more than all of it (about 2,340) in the high case. Of section 5's list, only the part that needs an H100 (calibration, the 400M teacher, big-model renders; about 105-125 hours before reruns) fits in the central-case spare, and only if the savings sentence is widened. The 5M parts of section 5 (T002, the M3 and M4 trunks, warm-ups, soups, the sink logit, dynamic conv, 5-10M RL) belong on the RTX 5070. The rest of the Phase 3 and Phase 4 plan belongs in the full request. Section 1's "one Start-Up allocation is up to 2,000 GPU-hours" is still true of the program; it is not true of the hours left in Planck's own request.

### 10.6 Coverage, honestly

Searched in full or nearly in full across sections 9 and 10: the 2026 main tracks of ICML, NeurIPS, ICLR, COLM (keyword), ACL, EACL, AACL and EMNLP; TMLR, JMLR, AISTATS and CoLLAs 2026; the 2025 main tracks of ICML, NeurIPS, ICLR and COLM; BabyLM 2023-2026; ICML and NeurIPS 2025-2026 workshops in Planck's topics; COLM 2026 workshops; the arXiv cs listing from 2026-01-01 to 11-18 (filtered); OpenAlex forward citations of 16 anchors; Marin, modded-nanogpt, Parameter Golf and nanochat; lab and industry releases since June 2026.

Still not searched, or only partly:
- **OpenReview itself.** The 19 NOT_FOUND ids above, plus the round-3 items that rest on OpenReview-only abstracts: the full text behind P-296 (99EZmxKpJh); the ICLR 2027 items b6XZ4N7a1t (P-394), x69B4FgAMi, IyFVWqSUhl, Ab5IBmIQod and TOCCgXKudX (P-359), efE3h1yYRd, sROOqxHIiw, 1BC0eYN2uS, Rd0dxUJuas, 419qaAMpme; the COLM 2026 workshop papers j9bpNGbHDS, 4jNpOfbjIn, Mok4aix6LQ, 4FYHLEyGpQ (P-367), 8jASKvRXGv (P-368), A7HxTzTM6t, 2osVgHagpq, BNS6PoeEpZ; FdXB4u27LD; the TMLR papers V6X6OuDJ0f and eAMLpW74rV and Heidi-3 (P-322); ZeroHero and Routing the Past. Three more ICLR 2027 ids cited by sweep items were resolved to probable arXiv copies this round (DJOmNDQ2wH as 2611.08016, tA3ABOjQfn as 2609.24432, IkzBak1MYU as 2609.28145).
- **Unread inside searched venues:** about 1,300 local NeurIPS 2026 abstracts and 1,049 fetched ones that a keyword filter dropped; the 2,016 off-topic NeurIPS 2026 papers beyond a title scan; NeurIPS 2026 domain workshops; 711 ICLR 2026 main-track abstracts judged by title only and 12 ICLR 2026 workshops beyond their titles; the ICML 2025 Long-Context Foundation Models workshop posters (OpenReview only); the rest of the ICLR 2027 pool beyond four subsets; cs.AI-only arXiv titles and low-scoring titles in every window; stat.ML-only papers beyond a scan.
- **Sources that could not be read:** Zhihu (403 to every fetch); kexue.fm pages directly (403; read through its feeds and two unofficial translation mirrors); gated Hugging Face cards and data (BananaMind-3), which need Max's OK to accept.
- **Tool limits during the pass:** the shared web-search budget ran out early in several passes, Semantic Scholar answered HTTP 429 most of the time, and the arXiv API throttled for long stretches. Some lookups therefore used arxiv.org pages directly, and a few searches simply did not run.
- **Prior-art gaps:** hypotheses HS to HAC (11 items) never had a prior-art pass; the close calls in 10.3 rest on hypA and hypB without re-opening.
- **What was not done at all:** no model was run for any claim in this section, no RC-12 score exists for any Planck checkpoint, and no H100 number was measured.

Expected yield of more broad searching: low. Three rounds after the gap sweep produced one new A item. The critics after rounds 1 and 2 each forecast 3-8 more lever or efficiency items; the finders tagged 75 items that way, and by consensus 51 of them landed in B or B (low), 10 in E, 10 in C or D, 2 in B (strong) and 2 in A (the LOOKUP absence fix and the CPLM Plus arms folded into P-238). The searches worth keeping are the three narrow ones in 10.4.
