# MaxGPT-Planck plan

Version 2, 2026-09-24. It is built from the draft plan and its 28-item review, `research/REPORT.md`, `LEDGER.md`, `research/followup/*`, `research/lanes/compute.md`, `research/brainstorm/*` and Max's checklist. No model was loaded to write it. "(est.)" marks arithmetic or judgment, not a measurement. Every throughput figure for Planck's shapes, and for the teacher on the 5070, is unmeasured and could be off by 2x either way. `bench_micro` (5070, first weekend) and the teacher pilot (Oct 5-8) replace those figures with measurements.


**Changes from the draft**
- **The 5070 now does almost all GPU work.** That covers scoring the baselines' generations, the teacher pilot, all bulk generation, the 5M screens and the SmolLM2 fine-tunes. The Mac drops from 600-1,000 GPU-hours to about 70-175 and never generates in bulk.
- **One chat pool for every size.** It holds 300M accepted tokens, generated on the 5070 by Nov 22. Each model reads it at most 25 times, and that cap sets each size's chat share.
- **The pre-registration locks Oct 11-14.** That is before any Planck model is scored on any RC-12 split, dev included. A bootstrap over conversations sizes the sealed split, and Level R now reports a human-written out-of-distribution set (OOD-H) beside the sealed score.
- **The curve starts at the Dec 4 freeze, with 3 seeds at every size that could carry the claim.** The 150M defaults to 75B tokens, with seeds 2-3 on the Titans in Jan-Feb. The Level A decision moves to Jan 31.
- **New section 3 on unattended running.** It covers job queues on each machine, heartbeats pushed out to a status repo, disk guards, Windows update windows, and a Titan smoke test on Dec 7-10.
- **D8 now covers every string that reaches training text.** Templates, slot pools and word lists come from human-written open sources or from the teacher, never from Claude.


**Changes requested by Max (Sep 24)**
- **Vocabulary: measure the optimum instead of assuming 8,192.** E7 (P-158) grows from one size to three: 5M, 20M and 60M total parameters, each on a vocabulary grid of 2k, 3k, 4k, 6k, 8k, 12k, 16k, 24k and 32k at equal total parameters (depth adjusted by the budget solver), 2 seeds each, from one nested BPE family trained on the chat mix so every vocabulary is a truncation of the same merges. The grid is centered by the Tao et al. vocabulary law's estimate, then refined around the best point. Each arm is scored on bits per byte, the RC-12 dev battery and rare-name recall (P-096). From the three sizes we fit how the optimal vocabulary moves with model size, and each curve point uses its own measured optimum when it beats 8,192 by at least 1% bits per byte with no battery regression. The result is reported as a valley with its width, since the optimum is a range, not a single exact number.
- **Token budget: measure how much more data buys before spending it.** New experiment E-tok (P-186). As asked (Sep 24): one long WSD run each at 5M and 10M, with decay branches at 500, 2,000, 5,000 and 20,000 tokens per parameter, every branch scored on the RC-12 dev battery. As costed, that is about 1,150-2,380 5070-hours, not the 40-80 first written (CORPUS 5.5), so the design that runs before the freeze is CORPUS 5.5's: a 5M trunk at 2 seeds to 2,000 tok/param with branches at 250 / 500 / 1,000 / 2,000 (35-75 h, est.), each branch scored as a base model and after one identical short SFT, on bits per byte and RC-12 dev margins (after the lock); its freeze rules (flat by 1,000, still rising at 2,000, rising faster than the web fits) set the curve budgets; the 10M curve runs take branches at 500 / 1,000 / 2,000 (and 5,000 on extension), a 3-seed token curve at about 14% overhead. The 20,000 tok/param point (amended 2026-10-10, cuts audit row 45): one 5M seed continues to 20,000 on the 5070 when the spare hours allow (about 109 GPU h eager by arithmetic, not measured), with its branch points fixed in P-186's pre-registration before the trunk starts; the Titan run after the 150M seeds (CORPUS 5.5 step 4) stays as the fallback. Still cut for budget and on Max's list (Appendix B, "for Max"), not dropped: the 10M trunk to 20,000 (about 315 h eager per seed by arithmetic: 199B tokens at bench_micro's 1.58 h per 1B, a full-row lower bound) and a second seed at 20,000. The curve's token budgets follow where multi-turn skills stop improving.
- **No reused data, a corpus built for Planck.** Nothing from the MaxGPT-Ultra build is reused (it was made for a 1.1B general model and spends about 18% on code and math that Planck does not need). Planck gets its own corpus built from scratch: selected for conversation skill and plain language, deliberately light on facts, deduplicated and filtered by classifiers trained for this model, with verified synthetic dialogue from an Apache-2.0 teacher, as large as the token budget requires. The design is in progress (corpus design v1).
- **Final models train as long as the calendar allows.** Short runs are for decisions; every curve model then gets the longest run the machines can give it, because at this size even small gains matter. P-186 decides how the time is split across sizes, and WSD lets the best model keep training and be re-decayed whenever GPU time frees up.
- **Priority: how small can it go (Max, Sep 24).** Models of 30M total parameters and under are now the main target and get most of the GPU time, seeds and experiments; 60M and 150M stay on the curve as upper anchors, with the 150M still on the Titans after mid-December. The curve extends downward (for example 3M, 5M, 10M, 20M, 30M) until the multi-turn skills break, and finding that floor is itself a headline result. Small sizes also make the token budget cheap: tens of thousands of tokens per parameter fit on the 5070.
- **The 360M block of E002 was skipped.** The repaired correction test passed at 135M on 5 of 5 seeds (its later audit found a wording shortcut, so general updating was not shown; see `docs/EXPERIMENTS.md`, E002), which makes the 360M arm unnecessary under its own reading rule. That GPU time goes to E003, the same correction test on much smaller public models (Pythia 14M to 160M, TinyStories 1M to 33M) to find where learning corrections breaks down.
- **Corpus design v1 is in `CORPUS.md`.** Its data-rule defaults are adopted, and its token budgets are rebalanced toward 30M and under (addendum): about 5,000 tokens per parameter at 5-10M and 1,000-2,000 at 20-30M on the 5070, with 60M and 150M on the Titans. (Until 2026-10-10 the Phase 4 and section 5 tables still showed the older 10/30/60M at about 500 tok/param; they now carry these budgets, and CORPUS 7.4's edits are applied. See the 2026-10-10 block below.)


**Status and schedule changes, 2026-09-26** (details and sources in `docs/EXPERIMENTS.md`)
- **Earlier than planned.**
  - *Tokenizer v0* (Phase 1, planned Oct 1-18) was trained on Sep 25 on a sample of the openly licensed starter corpus instead of the planned mix of pilot output, OASST2 and a FineWeb-Edu slice (the pilot has not run). P-098 held exactly, P-101 failed its 5% bar, P-097 was counted; P-100 waits for the teacher pick. v0 is provisional, and the starter corpus is tokenized with it at 8k (2.03B tokens).
  - *OOD-H Part 1* (a lock item) was built on Sep 26: 150 threads, 417 probes, every gating check passing, hashes in `oodh/HASHES.txt`. It is a lock candidate awaiting Max's review.
  - *Core v0 corpus* (CORPUS.md 7.1, planned Oct 19 - Nov 1) has started: its download manifest (790 files, about 130 GB) and tiered build script are written and its OCR filter is calibrated. That code is not committed yet, and the build's progress is not recorded in the repo.
  - *E006*, not in this plan: the three-object follow-up to E005 plus a general-chat replay arm, the first fine-tune on the 5070. Pre-registered and built, not scored.
  - *bench_micro on the Planck harness* (planned Oct 4-5) ran on the 5070 on Sep 25; its measured rates for the 5M-30M shapes replace this plan's throughput estimates. Two speed-ups (batched optimizer step, variable-length attention) are under verification: +16% to +34% on plain rows.
  - *E2 and E3* (Phase 3a, planned Oct 19-24) are pre-registered, critiqued, configured and smoke-tested on the 5070. The three teachers are pinned to exact revisions and 4-bit builds from metadata only; the pilot has not run.
- **Critical path now.**
  1. RC-12: the engine parity check, then the dev baselines (none scored yet; the HF path has never run on a real model).
  2. Headroom, re-anchor, PERSIST drop list and strict-case rates; Max decides the open loop-rule items (F3, the PERSIST statement-filler turns, the near-duplicate report).
  3. Sizing, the sealed build, the OOD-H hashes after Max's review, and the push.
  4. Alongside, on the 5070 (one job at a time under the GPU lock): E2 then E3, E006, and the teacher pilot. On the Mac, E003 finishes (running).
- **Dates.** E2 and E3 can start as soon as the 5070 is free, before Oct 19: they need no RC-12, and their estimates are about 6 h for E2's 5M stages and 9 h for its 20M grid (from the bench before the speed-ups). Not moved: the lock (Oct 11-14; only step 1 of its nine-step checklist is done) and tokenizer v1 (Oct 18).


**GPU and quality policy, 2026-10-11 (Max).** Max: "so the thing is tho all the other chats that needed the pc arent running anymore, so why dont we just drop that gpu hour limit?? you can use it all the time no matter what unless i say otherwise and remember we prefer quality over speed 100 percent of the time ok?" So: the RTX 5070 is Planck's full time until Max says otherwise; no Planck experiment on it has a GPU-hour cap, budget or cut order any more (the SCREENS batch cap, Track K's 15.5 h, the 350 h generation cap and any later one); hour figures in this plan are planning estimates only and never cut or defer a bet. Where a choice trades thoroughness for time (more seeds, more checks, longer runs, a rerun to settle a doubt), the thorough option wins. Each pre-registration that carries a cap gets its own dated amendment removing it, written before any result the cap could have affected. The gpu.lock stays, because Planck's own queues share the card.

**Changes from the cuts audit, 2026-10-10** (`research/CUTS_AUDIT_2026-10.md` item S8, rows 17, 18, 43-46, 73, 84, 85, and rows 35, 37, 53-58 for Appendix B). Text only. No result is involved: nothing here was written after reading a result it could change.
- **No budget or usage rule cuts a bet silently** (Max, 2026-10-06: "if something can be helpful even a little we should give it its proper chance"; his quality-over-usage rule). Every standing rule below that says cut, drop or "only the must-run list" (the E4 no-signal branch, the Nov 22 checkpoint, the bench re-plan rule, the Phase 2 rate check, the generation cap, the Titan seed queue, "too many bets", and the Claude-usage row of section 7) now works one way: when it fires, it slows or defers work, and the list of bets it would cut goes to Max first, each with its cost and what it would decide. Nothing on that list is dropped before he answers. A bet that never runs stays "not run", never "failed".
- **Phase 4 and section 5 carry the adopted budgets:** 5M, 10M, 20M and 30M on the 5070 at the CORPUS addendum's tokens per parameter, 3M conditional, 60M and 150M on the Titans; hours from bench_micro's measured rates where they exist.
- **CORPUS 7.4's eight edits applied:** the strict-open core replaces readability-filtered FineWeb-Edu (mix v0, the recipe, the tokenizer sample, the Lambda prep); FineWeb-Edu moves to excluded (Q5) in the D8 table; the teacher gate measures at realistic request lengths, with the extra arms; the chat pool is R0/R1/R2/RH with a growth rule, P-187 and a 350-hour generation cap; P-186's cost fix (above); new LEDGER rows E-sel-0 to E-sel-6, E-src and P-187, with P-083, P-161, P-164 and P-165 marked absorbed; the 5M and 20M curve points; the PC disk at 250-300 GB.
- **Data bets:** the 30-60 h of 5M screening in Phase 3's compute is an estimate, never a cap on a must-run data bet; P-051's 0.5% density point is restored; the dose-response arm (P-188) is a LEDGER row and has a 3b slot; on an E4 no-signal reading, E8-E10 move to 20M under new pre-registrations instead of dropping.
- **Seeds at 60M and 150M:** 3 seeds each, queued on the Titans; the claim rules (3 seeds at the claimed size) are unchanged.
- **Sealed split:** 600 conversations (Max, Oct 2-4), not the 1,000-1,200 first expected.
- **Appendix B:** every one of the 106 ranked LEDGER bets that PLAN never placed gets a slot, a gate with its trigger, or a stated reason.

---

## 1. The goal and the claim we are trying to earn

MaxGPT-Planck is a size curve of small chat models: Planck-5M, 10M, 20M, 30M, 60M and 150M, with a 3M point if 5M passes a Level A family on dev (the curve as written here until 2026-10-10 was 10M, 30M, 60M and 150M). Sizes count total parameters, and the non-embedding body is always reported beside them. The aim is to find the smallest model that passes a strict, pre-registered multi-turn conversation test. The target is conversational mechanics over the conversation's own context:
- recalling what the user said many turns ago
- resolving references
- updating after a correction
- keeping same-type facts and speakers apart (whose dog; my name vs yours)
- holding instructions
- not looping
- reading from a supplied lookup

Stored world knowledge is out of scope. At roughly 2 bits per parameter, a 150M model cannot win on knowledge. Every knowledge-bearing score is reported separately and never claimed.

The comparison model is Qwen2.5-0.5B-Instruct. It runs in the same harness on the same sealed test, next to a public panel from 90M to 2.6B.

We want a capability-per-parameter result that holds across the curve, not one lucky run. The test is frozen and its hash is published before any Planck model is scored on any split of it. Every seed and every failed idea is published.

**What is new, and what is not** (Max, 2026-10-02: took all recommendations in rc12/DECISIONS_FOR_MAX.md (item 15); evidence in `research/NOVELTY_2026-10.md`, related work in `research/REPORT.md` s2.1). Planck does not claim to be the first to ask how small a chat model can be (SmallTalkLLM), the first from-scratch chat model under 30M (BananaMind-2, Loom, Veyra2, Vertex, Micro Language Models, MiniMind), the first to train tiny models on synthetic multi-turn dialogue (RxT, Micro Language Models), or the first to keep facts outside the weights (LMLM, Co-LMLM, KLLM, Loom). The claim we are trying to earn is the first measured floor for strict multi-turn state skills: the smallest from-scratch model that passes a sealed, mutation-tested 12-turn test on its own conversation history (corrections, binding, rule persistence, own-answer consistency, role, loops), with seeds, against a public panel in one harness. As of 2026-10-02 no published work shows a model under 150M passing such a test or a measured minimum-size curve for these skills. The lock's dated push is the priority claim, so it stays on schedule.

**Level R** (reported along the way). Exact wording (Max, 2026-10-04: took all recommendations in rc12/DECISIONS_LEVEL_R_FOR_MAX.md, decision 1 D; it replaces the original non-inferiority wording, a pre-lock change made after seeing dev, disclosed in the pre-registration with the dev table as its evidence):
> "Planck-{N}M ({N}M total parameters, {B}M non-embedding) clears the state bar on RC-12, a pre-registered, sealed 12-turn conversation test. On its own conversation history over sealed RC-12's seven state families, the lower bound of the 95% CI on Planck-{N}M's state score (bootstrap over conversations and seeds) is {s7lo} of 100, at or above the pre-registered bar of 40 set by the shortcut-rule ceiling; the best public model on our panel scored {y} ({k} training seeds, {c} conversations). On OOD-H, {h} conversations with human-written user turns, the paired difference against Qwen2.5-0.5B-Instruct is {d} points (95% CI {lo} to {hi}). It is the smallest model on our curve that does this."

Level R holds at a size when, on sealed, the 95% CI lower bound of the state score S7 (RECALL, CORR, BIND, TWOHOP, OWN gated, TOPIC, ROLE; 0-100) is at least 40, PERSIST is at least 0.40 (point), T0 is at least 0.90, with at least 3 training seeds. Why: Qwen2.5-0.5B-Instruct scores 2.30 of 100 on the seven state families on dev, so non-inferiority to it at -3 could not tell it from a floor model (a public 5M model met it, mostly by not looping), and every public model sits below a fixed-rule shortcut mix (S7 36.7 on dev); 40 is the cheater bar committed on 2026-09-25, before any baseline was scored. Reported beside, never claimed: R, the original non-inferiority CI against Qwen2.5-0.5B-Instruct, S7 per family, the paired S7 difference against LFM2-2.6B, the sealed shortcut ceiling, the loop rate, and S7 of the generic-recipe 30M control.

If OOD-H's paired point estimate is worse than -5 points, the claim is worded as "clears the state bar on RC-12's format", with the OOD-H result beside it. The unqualified wording is then not used. Level R statements made in December are provisional until OOD-H is complete in January.

**LOOKUP claim** (Max, 2026-10-04; prereg s10b). Reported beside Level R and Level A, never folded into them: on sealed RC-12, at the claimed size, the LOOKUP family score (answer from the table the user supplied, and abstain when the key is only in the other table) is at least 0.60 with at least 3 training seeds and T0 at least 0.90, its 95% CI reported, against every public panel model's LOOKUP rate (0.000 to 0.014 on dev, up to 2.6B). It is the direct test of the skills-in-weights, facts-outside idea on supplied facts; Planck issuing its own lookups is a post-lock diagnostic only (decision 2).

**Level A** (the headline). Exact wording:
> "Planck-{N}M passes a strict multi-turn bar that no model at or below 0.6B passed in our tests. On sealed RC-12, pooled over {k} seeds, it:
> - applies corrections made 4 or more turns earlier (pass rate at least 0.60, with the two-slot and no-update control items also at least 0.60);
> - keeps same-type facts and speakers apart in both mention orders (paired pass rate at least 0.60; chance is 0.25);
> - loops in at most 2% of turns, and no more often than Qwen2.5-0.5B-Instruct.
>
> In a blind pairwise test ({r} raters, 150 conversations, 3 ratings each), raters judged it as good as or better than Qwen2.5-0.5B-Instruct in {w}% of judgments (95% CI {lo} to {hi}), meeting the pre-registered 50%. To our knowledge it is the smallest model shown to pass this bar."

**Rules for both levels**

*Thresholds*
- The thresholds (0.60, 2%, 50%, 40 and 0.40 for Level R, -5) are proposals. (Level A's four content bars were 0.80 until the re-anchor rule below set them to 0.60 on dev, Oct 2; Level R's -3 margin became the S7 bar of 40, Oct 4.) They become final when the pre-registration is pushed (Oct 11-14) and never change after that.
- **Re-anchor rule**, applied mechanically before the lock. LFM2-2.6B is the reference model that should clear the bar. If it scores below 0.80 on dev for a Level A family, first check that family and its grader. If the score is still below 0.80, that family's threshold becomes LFM2-2.6B's dev rate rounded down to 0.05, and never goes below 0.60. If LFM2-2.6B is below 0.60, the threshold stays at 0.60 and the pre-registration says no tested model reached it.

*Seeds*
- Every size that could carry the claim gets 3 training seeds. It gets 5 if E3 or P-026 show seed lock-in, meaning seed pass rates on a pass-or-fail family spread by more than 0.3 at 30M.
- Level A is claimed only at a size with at least 3 seeds. Any curve point with fewer seeds states its count.

*What counts as the model*
- Both levels score the bare model: each reply is generated from the conversation's own history (the scripted user turns, including any table a user turn supplies, and the model's own earlier replies), with no retrieval, memory store, state card, repetition penalty or other help from the harness; the OWN counterfactual twin (prereg s9) edits one assistant turn and can only remove credit. (Max, 2026-10-02: took all recommendations in rc12/DECISIONS_FOR_MAX.md (item 15); the same sentence goes into the pre-registration's claim rules.)
- A note the model writes and reads itself counts as the model.
- A harness-maintained state card is reported only as "Planck-{N}M + state card (scaffold)".

*After the first sealed scoring*
- Any later recipe change is labeled post hoc.

*Odds* (judgment, from REPORT s1.4)
- Level R: not re-estimated since the Oct 4 change; lower than the 40-60% judged for the old non-inferiority bar (both analysts' judgment, rc12/DECISIONS_LEVEL_R_FOR_MAX.md). It now needs a true S7 of about 45-46, an average state family of 0.45, for 80% power (about 47-48 if its runs agree closely unit by unit: the verifier's own simulation, 2026-10-05, rc12/notes.txt STEP 12 VERIFY); the best public model on our panel, LFM2-2.6B, averages 0.18 on dev.
- Level A: 10-25%, or about 30-40% if the repaired `ft_test.py` (P-001) passes with its controls.

---

## 2. Phases

| phase | dates | main machine | what it decides |
|---|---|---|---|
| 0 Measurement, operations and repo | Sep 24 - Oct 14 | 5070 (baseline generations, CUDA probes); Mac (E001, likelihood scoring, harness) | the test, the baselines, whether updating is trainable, unattended running |
| 1 Teacher pilot and tokenizer | Oct 1 - Oct 18 | 5070 (pilot); CPU (tokenizer) | which teacher; tokenizer v1 |
| 2 Data pipeline and chat pool | Oct 9 - Nov 22 | 5070 | whether skill-dense data teaches the skills; the 300M pool |
| 3 Screens and confirmations | Oct 19 - Dec 3 (freeze Dec 4; Dec 14 for 150M) | 5070 | the recipe |
| 4 The curve | Dec 4 - Feb 21, 2027 (Titan seeds to about Mar 2-24) | 5070 (5/10/20/30M, 3M conditional); Titans (60M, 150M) | Level R, then Level A with the human test |
| 5 Writeup and release | Feb 1 - Apr 5, 2027 | little compute | paper, weights, data, arXiv |

**Critical path**
1. RC-12 dev split and graders.
2. Dev baselines on the 5070.
3. Sealed-split sizing, then the pre-registration lock (Oct 11-14). The teacher pick (about Oct 9) runs in parallel.
4. Pilot data.
5. E1-E4.
6. The P-012 / P-053 gate.
7. Bulk generation alongside the data and memory bets.
8. The combined-recipe test.
9. Freeze (Dec 4; Dec 14 for the 150M).
10. The curve.

### Phase 0: measurement, operations and repo (Sep 24 - Oct 14)

**Purpose.** Build the measuring instrument and the plumbing before building models:
- the sealed 12-turn test;
- every baseline scored in one harness;
- queues that keep the GPUs busy without Max;
- the two cheap answers already running (P-001, P-002), which say how hard Level A will be.

**Deliverables**

*Repo and operations*
- **Public repo `MaxwellBrohm/maxgpt-planck`**, created after Max says yes.
  - Contents: README with the claim wording, LEDGER.md, the trimmed public PLAN.md, `tools/`, and `research/` after a scrub.
  - `private/` and `sealed/` are gitignored.
- **Operations layer** (section 3): PC queue runner and startup task, Mac runner under `~/Library/Application Support/`, private status repo with heartbeats, hardened Mac guard, disk guards, Planck's own pinned venvs.

*P-001 and P-002 results*
- Written up with their pre-registered verdicts.
- E001 started at about 00:48 today with dry runs. At 00:55 a Falcon-90M dry chat run on CPU was in progress. STATUS records the actual start and end times.

*RC-12 (P-003)*
- The ledger estimates 40-80 h of Claude work, spread over Sep 28 - Oct 10.
- **Dev split**: about 600 conversations of 12 user turns, scored on the model's own history.
- **Families**: F1-F8 (recall at distance, update and abstain, reference, persistence, override, sharded vs full, persona, social basics). Added to them:
  - composition (P-033);
  - knowledge-free follow-ups with single-turn controls (P-045);
  - perspective and role-capture items;
  - topic return.
- **Also in the split**:
  - Tier 0 likelihood items with shortcut rows.
  - Loop, role-leak, self-copy and runaway diagnostics on every turn, with the loop detector's exact rule pre-registered.
  - Conversations designed to stay at or under about 1,800 Planck tokens at typical reply lengths.
  - A skeleton of the lookup battery LK1-LK10 (P-082).
- **Sealed split size**: 600 conversations (Max, 2026-10-02, rc12/DECISIONS_FOR_MAX.md; kept when Level R moved to the S7 bar on Oct 4, because that bar barely depends on n: a true S7 of about 45.7, 45.1 and 44.6 is needed at 600, 700 and 900). The sizing rule below, which first expected 1,000-1,200, is kept as the record of how the number was first planned.

*Sealed-split isolation (P-049, review 8)*
- The sealed split is built from Claude-written templates and paraphrase banks. D8 allows Claude for gold sets, and no training teacher ever renders them.
- The banks are written in a dedicated session before generator v0 exists. They are encrypted at rest as soon as they are written, so only ciphertext sits in the working tree and no generator session can read them.
- The dev split uses different templates and slot pools.

*OOD-H, the out-of-distribution human set (review 8)*
- **Part 1, fixed at the lock**: 150 held-out, knowledge-light English multi-turn OASST2 threads.
  - Their IDs are hashed into the pre-registration.
  - They are removed by thread ID from every training corpus, the tokenizer's included.
  - Graded probe turns are appended, asking about what the human user said.
- **Part 2, collected in January**: 50-60 short scripts written by the human-test raters before they see any model output.
  - Each script has 6-12 user turns, 2 or more personal facts, one correction and a later question.
  - The rater marks the gold answers.
- Both parts use the same graders as RC-12.

*Graders*
- Each grader is mutation-tested and must fail all of these inputs: an empty reply, a wrong in-pool value, a "shotgun" reply that lists every candidate, an echo of the question, the stale value, a negation, a guess list, a hypothetical, and first person where "you" is meant.

*Scoring the baselines' generations, on the 5070 (review 2)*
- Engines: vLLM in WSL2 for Qwen, SmolLM2, Gemma 3, LFM2/LFM2.5 and Falcon-H1 wherever the architecture is supported. Batched Hugging Face transformers (HF below) for Doge, MaxGPT-3 and anything vLLM rejects.
- Conversations advance in lockstep batches: turn t of every conversation runs in one batch.
- A greedy HF-vs-vLLM parity check runs on 20 conversations per model family. The pre-registration names one engine per model.
- The Mac runs likelihood tiers only.

*Baselines, on dev and then on sealed*
- Qwen2.5-0.5B-Instruct, Qwen3-0.6B (thinking off), Qwen3.5-0.8B
- LFM2.5-230M, LFM2.5-350M
- Falcon-H1-Tiny-90M plus its five checkpoints (P-144)
- SmolLM2-135M/360M-Instruct, plus SFT-only vs Instruct (P-145)
- Gemma 3 270M-it, Doge-160M-Instruct
- LFM2-700M and LFM2-1.2B (P-146)
- LFM2-2.6B, as the reference showing the bar can be reached
- MaxGPT-3 (P-147)

*Decoding, pre-registered (review 24)*
- Each model uses its official chat template with its default system behavior and no added system prompt.
- T = 0.6, top-p 1.0, repetition penalty off, max_new_tokens 256 for every model.
- 3 sampling seeds for every model, Planck and baselines alike. Greedy is reported as a secondary result.
- If a conversation outgrows a model's context, the harness keeps the most recent turns, and the lost items count against that model.

*Sealed-split sizing rule (review 7)*
- From dev runs of Qwen2.5-0.5B-Instruct and the two baselines nearest to it, estimate how strongly turns within one conversation are correlated, and how often the two models' paired outcomes disagree.
- Simulate the pre-registered analysis with those estimates.
- Pick the smallest conversation count, in steps of 100 (minimum 600, cap 1,500), that gives at least 80% power to show non-inferiority at -3 when the true difference is 0.
- If the cap is not enough, the margin stays at -3 and the achieved power is published.

*Level R analysis, pre-registered*
- The unit is the paired per-conversation difference.
- 10,000 bootstrap resamples over conversations and training seeds, with sampling seeds nested inside.
- Percentile 95% CI.
- Since Oct 4 (Max, decision 1 D of rc12/DECISIONS_LEVEL_R_FOR_MAX.md), Level R reads one model's S7 CI with the same resampling (not paired); the paired analysis above is reported beside, against Qwen2.5-0.5B-Instruct on R and against LFM2-2.6B on S7. The sealed size stays 600: the S7 bar barely depends on n (a true S7 of about 45.7, 45.1 and 44.6 is needed at 600, 700 and 900).

*Inference diagnostics*
- P-143: both probes on LFM2.5-230M and Falcon-90M.
- P-005: the state card.
- P-006: re-injecting the system prompt.
- P-142: the counterfactual open-book probe.

*Public pre-registration (review 6)*
- `prereg/RC-12.md` holds:
  - the thresholds and the re-anchor rule;
  - the headroom rule: a family fails if every baseline is at or below 0.05, or every baseline is at or above 0.95;
  - the analysis plan, the sizing rule and its result;
  - seeds per size and the decoding settings;
  - the OOD-H protocol;
  - the human-test protocol and rater allocation;
  - the baseline list and engines.
- Committed with it:
  - the SHA-256 of the sealed files and generator seed;
  - an encrypted archive of the sealed split;
  - the dev split in the clear.
- The commit is timestamped outside git by a GitHub release and a Software Heritage snapshot, because git author dates can be edited.
- The archive key is kept in two places outside any repo, the Mac and the PC. Max may add a third in his password manager.

*Hard rule*
- No Planck model is scored on any RC-12 split, dev included, until the pre-registration is pushed.
- If the lock slips, Planck chat-mix training slips with it. E1-E3 need no RC-12 and may run.

*5070 ready*
- WSL2 with torch cu128 and vLLM in a Planck venv.
- WMI detach verified with a 10-minute dummy job, and the startup task tested with one reboot.
- `bench_micro` on the 5M/20M/60M/150M shapes. It uses Ultra's model code with an 8,192 vocab, because the Planck harness does not exist yet. It is re-run on the Planck harness around Oct 4-5.

**Ledger ids.** P-001, P-002, P-003, P-005, P-006, P-033, P-045, P-049, P-082, P-142, P-143, P-144, P-145, P-146, P-147. Baseline items: eval harness core, guard runner, throughput measurement, pre-registration with an append-only log, all baselines in one harness.

**Machine.**
- 5070: every generative baseline run, P-143 for Falcon-90M (MPS crashed on its Mamba path), P-144, P-145, P-146, and LFM2-2.6B.
- Mac: E001, P-143 for LFM2.5-230M, P-142, likelihood tiers, harness work.

**Compute (est.).** 5070 25-60 h. Mac 15-30 h. Titans 0.

**Max-hours.** About 1 h (section 4).

**Gate**

*Move on when:*
- every grader fails all its mutations;
- every family passes the headroom rule;
- LFM2-2.6B scores at least 0.80 on dev for the correction and binding families, or the re-anchor rule has been applied;
- the baselines are scored on dev;
- the sealed split is sized, rendered and hashed;
- the pre-registration is pushed.

*Change course:*
- **P-001 passes with controls on most seeds at 135M.** Updating is trainable at a 106M body. Level A odds go to about 30-40%, and interference-balanced correction data (P-009) becomes Phase 3's lead bet.
- **P-001 is flat at 135M and 360M.** A few hundred SFT steps do not install updating. Phase 3 shifts weight to:
  - the note (P-010);
  - the recency path (P-020);
  - the delta layer (P-022);
  - the gated distance curriculum (P-026).
- **P-005: corrections pass with the state card and fail without it.** The limit is access or interference, not capacity. The note and recency bets move up.
- **LFM2-2.6B fails a Level A family on dev.** Check the family and grader, then apply the re-anchor rule with its 0.60 floor. This happens before the lock, never after.
- **A public panel model passes Level R** (its S7 point estimate at least 40 and its PERSIST at least 0.40; Max, 2026-10-04). Level R is then new only below that model's size, so Planck's claim moves to the curve points under it. On dev none does at any size.

### Phase 1: teacher pilot and tokenizer (Oct 1 - Oct 18)

**Purpose.**
- Pick the teacher by measured yield on the machine that will do bulk generation.
- Freeze the tokenizer family so data generation can start.

**Deliverables**

*Skeleton generator v0*
- Built in a separate session, after the sealed banks are encrypted.
- It samples:
  - persona and facts;
  - slot pools, including nonce values (P-025);
  - 2-4 skill events S1-S9 per conversation, with distances and same-type distractors;
  - stale traps, plus two-slot and no-update twins;
  - required words and user style.
- It writes the gold notes and thinks (P-011), so the four memory arms share identical rendered text.

*D8 provenance (review 12)*
- Every string that can reach training text comes from an allowed source.
- Slot values come from:
  - public-domain lists (US SSA first names, Census surnames);
  - GeoNames places (CC-BY);
  - word-frequency lists we compute from our own allowed corpora;
  - lists the teacher writes.
- Every natural-language template, think phrasing and note phrasing is written by the Apache-2.0 teacher from a Claude-written prompt, and stored with its provenance.
- Claude writes code, prompts, rubrics, the gold eval sets and the human-test scripts. Nothing Claude writes goes into training text.

*Checker v0*
- The 13 checks in chatdata s3.6, the assistant self-claim rule (P-061), and mutation tests.

*Teacher pilot, on the 5070 under vLLM (review 13)*
- The same 200 skeletons go through three Apache-2.0 teachers at 4-bit, one loaded at a time:
  - Gemma 4 12B: raw completions, thinking off, the prompt format from `tools/gemma_chat.py`. Checked token for token against the LM Studio path on 10 prompts.
  - Ministral 3 8B Instruct.
  - Qwen3.5-9B: thinking off, confirmed from the outputs.
- Batch size goes up to each teacher's KV limit. This matters: Qwen3.5-9B holds about 80 MB per 1k-token sequence against about 336 MB for Gemma 4 12B, so it fits about 4x more sequences in 12 GB.
- Measured for each teacher:
  - yield by reason code;
  - accepted tokens per hour at bulk batch sizes, counted in Planck's tokenizer;
  - diversity (distinct-n, compression ratio, 8-gram caps);
  - a Claude rubric read of 30 conversations;
  - Max's blind skim of 10 (optional).
- The Mac through LM Studio is a fallback only for a teacher vLLM cannot run. Its rate is then planned at 20-55 accepted tok/s (chatdata.verify claim 10), not 100-300.
- Gemma 4 26B-A4B stays parked.

*D8 source audit table, with a provenance column*
- Columns: source, author (human or which model), license, allowed for the headline model or baselines only, and where it enters (pretraining, SFT, templates, slot pools, tokenizer).
- My reading, which Max confirms in one line on Fri 9/25: human-written open-licensed text is allowed, and every model-written token must come from an Apache-2.0 model.
- Baselines only: WildChat, UltraChat, SODA, TinyStories, SimpleStories, TinyDialogues, TinyChat, smol-magpie-ultra.
- Everyday-Conversations is also out. It is Llama-3.1-written, and that license would put "Llama" in the model's name.
- Excluded (Q5, CORPUS 1.4; moved here 2026-10-10 under CORPUS 7.4 edit 2): FineWeb-Edu, because its selector was trained on Llama-3-70B labels. The FineWeb (not -Edu) bucket W stays conditional on Q1, which Max answered no for the headline core (CORPUS addendum, Sep 24).
- Allowed:
  - OASST2 (human, Apache-2.0);
  - the strict-open core: openly licensed Common Pile v0.1 general-language sources after the date gate and Planck's own selection (CORPUS 2.1-3.7);
  - Wikipedia (CC-BY-SA; release rule in section 8);
  - Cosmopedia v2 (Mixtral-written, Apache-2.0);
  - public-domain name lists;
  - our own teacher renders;
  - paraphrases from a small Apache-2.0 model (P-162).

*Tokenizer family v0 (needs the D8 confirmation)*
- One byte-level BPE with byte fallback, digits and punctuation split, a minimal role-token template, and reserved `<note>`, `<think>`, `<lookup>` and `<result>` tokens.
- Trained on the pilot output, OASST2 (with the OOD-H threads removed) and a strict-open-core sample (CORPUS 7.4 edit 1; FineWeb-Edu is excluded), then truncated to 2k/4k/8k/16k/32k (P-098). v0 was in fact trained on the starter corpus sample (Status, Sep 26); this is v1's sample.
- CPU checks: P-097, P-100 (teacher-vocab subset) and P-101 (superword tokens, kept only if they give at least 5% compression).
- Stage-0 health checks: round trip, no glitch tokens, bytes per token on held-out chat.

*Register knobs*
- Pre-registered per size (P-048).

**Ledger ids.** The "teacher policy" and "tokenizer" baseline items, P-011, P-025, P-048, P-061, P-097, P-098, P-100, P-101.

**Machine.** 5070 for the pilot. Mac CPU for the tokenizer. The Mac GPU only if vLLM fails.

**Compute (est.).** 5070 10-25 h. CPU 2-4 h. Mac GPU 0 (10-20 h in the fallback).

**Max-hours.** About 0.3 h, optional.

**Gate**
- *Move on* (about Oct 9): pick the teacher with the most accepted tokens per hour on the 5070 whose naturalness is within 10% of the best (chatdata L12). I expect Gemma 4 12B or Qwen3.5-9B, since batching may reverse the single-stream ranking.
  - CORPUS 7.4 edit 3 (applied 2026-10-10): the gate counts accepted Planck tokens per 5070-hour at realistic request lengths, and the pilot adds the Gemma 4 E4B GGUF, llama.cpp Gemma, an 8B paraphraser and MTP (speculative decoding) arms. Status: fp8 KV, which 7.4 named, was tried and abandoned (no capacity gain, pipeline/notes.txt:398-405); the extra arms have no record of running or of being dropped, so they are named in the real teacher pilot's pre-registration (cuts audit row 116).
- *Change course:*
  - **Every teacher accepts under about 30% on the correction or long-distance cells.** Render those cells from teacher-written templates (P-065) and keep the teacher for naturalness.
  - **No teacher runs under vLLM on the 5070.** Use a llama.cpp server with parallel slots on the PC, then recompute the pool size and chat shares from the measured rate. The Mac is not a bulk fallback: 200M tokens at 20-55 tok/s would take 42-116 days.

### Phase 2: data pipeline and chat pool (Oct 9 - Nov 22)

**Purpose.** Turn the pilot into a verified dataset. Test on public models whether skill-dense data teaches the skills before paying for volume, then scale.

**Deliverables**

*Pipeline at scale*
- Stages: skeleton, render (script mode), verify, exact and MinHash dedup, 8-gram caps, then 13-gram decontamination against RC-12 (dev and sealed), OOD-H and the human-test scripts.
- A coverage matrix steers generation toward low-yield cells, and yield is logged per family, distance and teacher.

*Pilot dataset*
- 20-30M accepted chat tokens, about 100-150k conversations of 200-250 tokens, generated Oct 9-18. It covers every family:
  - correction dialogues with twins and varied original position (P-009)
  - perspective and role-swap traps (P-007)
  - binding-dense dialogues (P-008)
  - back-reference-dense dialogues (P-012)
  - persistence and override (P-018)
  - list state and topic return (P-044)
  - two-sided abstention (P-017)
  - social basics
  - grounded lookup dialogues on a fictional knowledge base (P-080, P-084, P-085)

*Two small slices*
- A live-mode slice for P-064.
- A cross-model user-turn slice (P-154).

*Public-model SFT tests, on the 5070 (moved from the Mac)*
- SmolLM2-135M-Instruct, 3 seeds each, packed per conversation.
- P-012 is the decisive test (its limit #3, chat training signal): generic data vs generic data plus about 5k verified dialogues.
- Also run: P-007, P-008, P-009 stage 1, P-013, P-014, P-016, P-017.

*Tokenizer v1*
- Frozen by Oct 18, retrained on the pilot mix. It never changes after this.

*Mix v0 manifest and general text*
- The manifest lists shard hashes.
- General text for 3-30M (and 60M) is streamed and tokenized on the PC, so raw parquet is never stored: the strict-open core plus Planck selection (CORPUS 7.4 edit 1; core v0 build started Sep 26), simple narrative and human dialogue (OASST2 with the OOD-H threads removed), about 15-25B unique tokens after selection (est.), by Nov 1. FineWeb-Edu and bulk Wikipedia are out (CORPUS 0, item 3).

*Chat pool (reviews 1 and 17; CORPUS 4.1-4.8, applied 2026-10-10 under 7.4 edit 4)*
- Layers: R0, teacher renders of program-written skeletons, 150-300M accepted; R1, 2 re-verified paraphrases per render; R2, slot values re-drawn at load time (CPU); RH, 100-150M grounded rephrasings of open human text by the freeze. That is about 0.5-0.9B surface-unique chat tokens at Dec 4 (est.).
- Target for R0: 300M accepted unique tokens, all generated on the 5070, at least 100M by Nov 8 and all 300M by Nov 22.
- Sizing: every model reads the R0+R1 surface at most 25 times (50 with R2 and spacing if P-187 shows no loss; up to 100 only if P-187 says so), and each size's chat share follows from that (table in section 5).
- Growth after the freeze: only under a growth rule pre-registered at the freeze; new shards from the frozen pipeline become pool v2, v3, each hashed.
- Generation cap: about 350 5070-hours through Nov 22 (CORPUS 7.2). If R0 + R1 + RH would pass it, the cap flags it and the list of what would shrink (R0 first, never below 150M) goes to Max before anything is cut (2026-10-10 rule above).
- Planning rate: 250-500 accepted tok/s in Planck tokens, so 300M takes 170-330 h (est.).

*Rate check*
- After the first full week of bulk generation (about Oct 27 - Nov 1), the measured rate sets the final pool size.
- If it projects under 300M by Nov 22, decide by Nov 8 to shrink the pool, lower the capped chat shares under the pass cap (60M and 150M in the old table; section 5's table since 2026-10-10), and lean on P-162 paraphrases. Anything this would cut (a pool layer, a share, a bet that needs the pool) goes to Max as a list first (2026-10-10 rule).

*Paraphraser for P-162*
- A small Apache-2.0 model, for example Qwen3-1.7B at about 2,300 tok/s on the 5070 (compute.md, est.).
- Per token it is 5-10x cheaper than the teacher, so if P-162 wins it stretches the pool cheaply.

**Ledger ids.** P-007, P-008, P-009, P-012, P-013, P-014, P-016, P-017, P-018, P-044, P-080, P-084, P-085, P-154, P-162. Baseline data items: skeleton/render/verify, hygiene, deflection removal, whole-conversation SFT.

**Machine.** 5070. The Mac does only harness smoke tests.

**Compute (est.).** 5070 180-350 h (generation 170-330 h, SmolLM2 tests 10-20 h). Mac 5-15 h.

**Max-hours.** About 0.2 h (one update window).

**Gate**

*Scale beyond about 100M only if one of these passes:*
- P-012: at least +15 points on recall, deflection and perspective, on every seed (results about Oct 20).
- P-053 wins under its rule. It runs after E2-E4 (Phase 3a) at the smallest size E4 validates, with the win rule set from E3's measured noise. Results about Oct 26-28.

*Middle zone:* the best gain is between +5 and +15, or its CI includes +5.
- Keep generating in the pilot mix up to about 100M.
- Re-test P-053 at 20M by about Nov 8.
- Scale to 300M only if the re-test passes.

*Both null:* gain under +5, with the CI excluding +5, and E3's minimum detectable effect at or below 5.
- Stop at the pilot and move Phase 3's weight to state mechanisms and architecture.
- Record the null publicly. It would echo ufakzeka-1's result at 151M.

*A single family wins:* if P-007 or P-009 wins on SmolLM2, raise that family's share of the pool.

### Phase 3: screens and confirmations (Oct 19 - Dec 3; freeze Dec 4, and Dec 14 for the 150M)

**Purpose.** Find the recipe.
- Each bet is one change against the baseline, pre-registered, with paired seeds (common random numbers).
- Decisions rest on continuous metrics: fixed-window bits per byte, plus the Tier 0 log-prob margin.
- Generative pass rates count only for effects of 5 points or more.

**3a. Calibration first (Oct 19-24, 5070; review 10)**
- E1: engine choice. Packing with document masking vs length buckets, `torch.compile`, and running a 5M job beside vLLM generation. P-141 (MPS bucketing) runs only if the Mac takes screens.
- E2: LR tuning and 5M-to-20M transfer, with muP/CompleteP if the optimum moves 2x or more.
- E3: seed noise and common random numbers. This gives the minimum detectable effect per metric.
- E4: the smallest size with multi-turn signal (P-046). It uses RC-12 dev, so it runs after the lock.
- P-004 (margins vs pass rates), moved here so both Phase 2 gates use a metric we trust.
- P-047: a 1M dialogue-shaped pre-screen.
- Then P-053, at the smallest size E4 validates.
- E7, the vocab sweep at equal total parameters (P-158, P-159), runs Oct 24 - Nov 8.
- A torch 2.7.1 + fp16 + loss-scaling API check. The real Titan test is Dec 7-10 (Phase 4).

**3b. Data bets (Oct 26 - Nov 22; 5M screen, then 30M confirm)**
- Every must-run data bet here (P-053, P-052, P-009, P-007, P-008, P-162) runs at its screening size (5M, or 20M if E4 finds no signal at 5M; P-053 at the smallest size E4 validates, as in 3a) under its own pre-registration once the first pool mix exists, whatever the 5M screening hours already spent. The 30-60 h in Compute below is an estimate, not a cap (2026-10-10, cuts audit row 17); if the bets overrun it, that is a flag to Max, never a cut.
- P-053: skill-dense vs small talk (from 3a).
- P-052: chat share (25/50/75%) and placement, with the 0.1x-LR check P-139. At curve scale the chosen share is capped by the pass rule.
- P-009 from scratch: 0% vs 2-3% correction dialogues.
- P-051: density sweep at 0/0.5/2/8% (the 0.5% point restored 2026-10-10; it had been narrowed to 0/2/8% at 30M with no stated reason, cuts audit row 46).
- P-188: the dose-response arm (NOVELTY_2026-10.md s6.1 (b)): at one size, the smallest that E4 and the first 3b screens leave with signal, 4 increasing doses of verified skill dialogue at 2-3 seeds, so the curve can say whether a skill is missing at a size for lack of capacity or for lack of data (2605.29548 argues data frequency at 4M-4B; mine:o2 argues a capacity ceiling at 124M). Trigger: the first pool mix; its own pre-registration, which fixes the doses and the reading before any run. It can share P-051's runs where the size and doses coincide.
- P-026: accuracy-gated distance curriculum. Report the lock-in rate.
- P-007 and P-008 from scratch.
- P-056: loss mask on user turns. P-015: template.
- P-162: paraphrases vs exact repeats. This sets how far the pool stretches.

**3c. Memory: the four-arm test at 30M, starting by Nov 2 (review 11)**
- Arms: B (baseline), T (capped think), N (note), TN (note then think). Ledger P-010, P-034, P-035.
- Shared stage-1 bases, 3 seeds, about 30-40 GPU-h (est.).
- Diagnostics: forced-empty think (P-155), gold-note injection, note deletion.
- If N wins: run P-032 immediately (N at 10M vs B at 30M), then P-030, P-029 and P-019 as N variants, Nov 9-22.

**3d. Architecture at 30M (Nov 9 - Nov 25)**
- A 2-layer toy (minutes) runs first.
- P-020 recency path.
- P-021 gated attention vs primacy, logging gate openness and sink mass.
- P-022 delta-rule layer, P-148 Canon layers, P-023 role embeddings.
- E8/E9/E10 (P-152, P-151, P-150), Max's skill-over-storage idea: at 5M if E4 finds signal at 5M; if it does not, at 20M under new pre-registrations written before any of their runs (2026-10-10, cuts audit row 18), instead of not running. A 5M bpb screen cannot judge them without signal, which is why they move up a size rather than out.

**3e. Post-training at 30M (Nov 16 - Nov 28)**
- Whole-conversation SFT is the baseline.
- P-014: one epoch of multi-turn DPO.
- P-066 and P-153: anti-loop. P-067: own-history SFT.
- P-068: an on-policy distillation (OPD) pilot on public SmolLM2 models, pre-registered and scheduled before 3e starts (Nov 16), not "if there is time" (2026-10-10, cuts audit row 49): P-069, P-070, P-168 and D6's only revisit trigger wait on it.

**3f. Combined recipe (Nov 26 - Dec 3; must-run; review 11)**
- All adopted winners together vs the baseline at 30M, 3 seeds, 20-30 h.
- If the combination fails its pre-registered rule, drop changes one at a time, weakest evidence first. These are leave-one-out arms, 1 seed each, 10-15 h.
- Freeze the best set that survives.

**Must-run list** (19): E1-E4, P-004, P-158, P-053, P-052, P-009, P-007, P-008, the four-arm test, P-020, P-021, P-022 or P-148 (whichever the toy favors), P-014, P-162, the combined-recipe test. Every other ranked bet has a slot, a gate with its trigger, or a stated reason in Appendix B (2026-10-10; this replaces "everything else runs if time allows"). When time runs short, the must-run list goes first and the rest is deferred, never cut: the list of what would not run goes to Max first. A bet that never runs stays in the ledger as "not run", never as "failed".

**Machine.** 5070 for everything, with 2-3 small jobs at once where E1 shows it pays. The Mac scores small checkpoints on RC-12 dev (batched, minutes each) and runs 5M screens only if the 5070 is down.

**Compute (est.).**
- 5070 400-640 h:
  - 5M screens 30-60 h (est.; not a cap on any must-run data bet, see 3b);
  - 20-30M confirmations, memory, architecture and post-training 350-550 h;
  - combined recipe 20-30 h.
- Mac 20-60 h.
- From Oct 12 to Dec 3, the 5070 carries about 580-990 h of roughly 1,270, including Phase 2 generation.

**Max-hours.** About 10 minutes a week to read STATUS.md, plus the Nov 21-22 remote session (section 4 of Phase 4). About 2.5 h in total.

**Gate.** The recipe freezes Dec 4 for the 5070 curve and Dec 14 for the 150M. The 150M-only settings (token count, T=4096 phase, chat share) are decided between Dec 4 and Dec 14.

*What gets adopted*
- A change is adopted only if it passed its pre-registered rule at 5M and held at 20-30M (or won directly at 30M), and it survived 3f.
- Anything that flips between sizes is marked SCALE-DEPENDENT and left out.

*Level A go signal*
- Measured on dev, pooled over 3 seeds, for the frozen 30M recipe:
  - corrections at 4+ turns reach at least 0.5 in generation;
  - the two-slot and no-update controls also reach at least 0.5;
  - both-order binding has a lower 95% CI bound above 0.40.
- If it does, Level A is attempted at every curve size.
- If not, Level A is attempted only at 60M and 150M with the arm that has the best margin, and STATUS says the odds are low.

*Other decision points*
- **The note wins by 5 points or more.** It joins the recipe. If N at 10M matches B at 30M, the 5M curve point (on the curve since 2026-10-10) also gets the N arm.
- **E4 finds no multi-turn signal at 5M.** All screening moves to 20M. The must-run list goes first; E8-E10 and the other bets follow at 20M under their own pre-registrations. If they cannot all fit before the freeze, the list of what would not run goes to Max first (2026-10-10 rule); nothing is dropped silently.
- **Nov 22 checkpoint.** If Phase 3 is behind, the must-run list takes the GPU first and everything else is deferred; the list of deferred bets, with costs, goes to Max before any is dropped.

### Phase 4: the curve (Dec 4 - Feb 21, 2027)

**Purpose.** Train the frozen recipe at each size, measure Level R on sealed RC-12 and OOD-H, then Level A, including the blind human test.

**5070 runs, from the Dec 4 freeze (review 4)**

In priority order. Rewritten 2026-10-10 (cuts audit rows 43-44) to carry the budgets adopted on Sep 24 (CORPUS addendum; Changes requested by Max above), which the old table (10M 5B, 30M 15B, 60M 30B, all at about 500 tok/param) never did. P-186's freeze rules (CORPUS 5.5) may still move a budget before launch. Hours are bench_micro's eager docmask rates on the Planck harness (harness/notes.txt: 1.06, 1.58, 2.86 and 3.87 h per 1B tokens at 5M, 10M, 20M and 30M), which are lower bounds: they assume full rows, and the curve's larger batches are unmeasured. Dates assume a Dec 4 start and 90% uptime.

| order | model | tok/param | tokens per seed | seeds | 5070 h (lower bound) | done by (est.) |
|---|---|---|---|---|---|---|
| 1 | Planck-5M (5.11M) | 5,000 | 25.6B | 3 | 81 | Dec 8 |
| 2 | Planck-10M (9.96M), with P-186's 500 / 1,000 / 2,000 branches | 5,000 | 49.8B | 3 | 236 (about 270 with branches) | Dec 20 |
| 3 | Planck-20M (19.8M) | 2,000 | 39.6B | 3 | 340 | Jan 5 |
| 4 | Planck-30M (29.7M) seed 1 | 1,000-2,000 | 29.7-59.4B | 1 | 115-230 | Jan 10-15 |
| 5 | Planck-30M seeds 2-3 | 1,000-2,000 | 29.7-59.4B | 2 | 230-460 | Jan 21 - Feb 6 |
| 6 | generic-recipe control, 30M (same size and tokens, baseline data only) | as row 4 | | 1 | 115-230 | Jan 27 - Feb 17 |
| 7 | Planck-3M, only if 5M passes a Level A family on dev (CORPUS 5.3) | 10,000-20,000 | 30-60B | 1-3 | unmeasured | after row 6 |
| | post-training and dev eval, interleaved | | | | 20-40 | |
| | **total, rows 1-6 (with row 2's branches) and post-training** | | | | **about 1,170-1,650** | **Jan 27 - Feb 19** |

Every size also takes a decay branch at 500 tok/param, so an equal-ratio curve is reported beside the budget curve (CORPUS 5.3). The 5070 window (Dec 4 - Feb 21) is about 1,706 h at 90% uptime, so by these lower bounds 50-530 h remain for row 7, CORPUS 5.4's extension queue or lock-in seeds; padding will eat some of it.

60M and 150M run on the Titans (below), each with all of its seeds on one platform, so no size mixes bf16 and fp16 numerics.

**Re-plan rule**, applied after the first curve run measures its real rate (rewritten 2026-10-10 under the rule that no budget rule cuts silently):
- If the measured rate projects rows 1-6 past Feb 21, Claude writes the options with their hours, for example a shorter generic control, 30M at 1,000 instead of 2,000, 20M at 2 seeds unless it is the claimed size (CORPUS 5.3 step 3), or a 30M seed moving to the Titans labeled as the fp16 platform, and Max picks before any is applied.
- No re-plan reaches back into Phase 3: the freeze has passed by then.

**Titans, from about Dec 17**

Dec 17 comes after Ultra's pretraining (ends about Dec 3-7, est.), Ultra's SFT/DPO, and the end of Max's fall finals (Dec 16).

*Smoke test, Dec 7-10 (review 16)*
- Runs in the gap between Ultra's pretraining and its SFT: 1 card, 1-2 h.
- 100 steps of the real 150M config in fp16 with loss scaling, then save and resume, and a heartbeat push.
- It checks the Turing attention path, the driver-470 Triton workaround, and fp16 overflow in Planck's block at d640 x 29 layers.
- Max starts it remotely with one prepared command. It reuses Ultra's keeper and pause/resume watchdog.

*Token count*
- Fixed at launch from the smoke-test rate and the same for every seed (review 20).
- 100B if the measured rate gives 18 days or less per 100B on 4 cards. Otherwise 75B, the default.
- The final 10% of tokens run at T=4096.

*Seeds* (60M added 2026-10-10: PLAN.md:22 and the CORPUS addendum put 60M on the Titans; the addendum's "1 seed each unless time allows more" and this plan's 3 are reconciled as 3 queued, with any shortfall going to Max)
- 150M seed 1 starts about Dec 17 and finishes Dec 31 - Jan 11 (est.).
- Then the three 60M seeds (60.8M, 500 tok/param, 30B tokens each; about 4.0 Titan-days per seed, 2.5-7.3, CORPUS 5.3, unmeasured), then 150M seeds 2 and 3, all queued with no new Max session. At worst, by these estimates, the last 150M seed ends about Mar 24, past the Titans' free window (to about Mar 15): before the queue would run past it, the choice (which seed waits or does not run) goes to Max.
- Max can reclaim the Titans for Ultra at any time. Each size's claim then states its seed count; a Level claim at 60M or 150M still needs 3 seeds there.

*Setup*
- fp16 with loss scaling, torch 2.7.1 cu118. The Titans need no driver upgrade for training.
- The general-text mix for 60M and 150M (about 5B and 15-20B unique tokens, read at most 4 times, CORPUS 5.3) is built beforehand on the box's idle CPUs at nice 19 from the strict-open core plus Planck selection, streamed from Common Pile (CORPUS 7.3 and 7.4 edit 1), with Ultra's thermal log watched. Max kicks it off remotely on Nov 21-22. Ultra's FineWeb-Edu text is not used (Q5).
- Heartbeats every 30 minutes go to the private status repo through a repo-scoped deploy key.

**Sealed scoring at milestones**
- Each curve point is scored once its seeds finish.
- It is scored with and without lookup on the fictional knowledge base (P-094), in the state-card condition labeled as a scaffold (P-005), and on OOD-H.

**Level R table.** Every curve point's S7 with its 95% CI against the bar of 40, its PERSIST and T0, the public panel's S7, and beside them R with the paired CI vs Qwen2.5-0.5B-Instruct and the OOD-H column.

**Level A human test (review 19)**

*Which model is tested*
- The smallest size that passes every generative Level A criterion on sealed.
- The next size down is also tested, and reported but not claimed, if every one of its sealed point estimates is within 0.05 of its threshold.
- The model shown is the seed with the median sealed composite, not the best seed.

*Scripts and display*
- 150 Claude-written, realistic, knowledge-light scripts. Both models run on their own history.
- Shown side by side, with order and left/right randomized.
- Identity strings ("Qwen", "Alibaba", "Planck", "MaxGPT" and similar) are replaced with a neutral token on both sides before display. How often this happened is reported, along with reply lengths per model.

*Raters*
- 6-8 raters, recruited by Max by Jan 10.
- Each pair is rated by 3 different raters, so each rater does 56-75 pairs in sessions of about 20 (1-1.5 h each).
- Total 25-40 rater-hours (est.). Whether to pay raters is Max's call (at $15/h, about $375-600).
- Max does not rate, or his ratings are reported separately.
- Before seeing any model output, each rater writes their 8-10 OOD-H scripts (about 30 minutes).

*Judgments and analysis*
- Each judgment is "Planck better", "tie" or "Qwen better".
- The primary measure is the share of judgments that are "Planck better" or "tie", with a 95% CI from a bootstrap over pairs and raters.
- The claim needs a point estimate of at least 0.50. The CI is always reported beside it.
- Ratings are collected on a page Claude builds, with rater consent and anonymized release. The analysis is pre-registered.

*Window*
- Jan 15 - Feb 21, for whichever size qualifies first.

**Release.** Each size's weights and model card go out once its sealed numbers are final, after Max approves each release. Approvals are batched into STATUS reads.
- **One model per size (Max, Oct 2).** Every curve point that passes Level R (or Level A) is released as its own model, named `planck-{N}m` (for example planck-5m, planck-10m, planck-20m, planck-30m), so users can pick a size. A size that does not pass is still kept and archived (keep-every-model) and can be released as a reported curve point, labelled as not passing.
- **Speed and footprint at every curve point (added Oct 2).** What matters at these sizes is speed and reach: replies that are effectively instant, many conversations running at once on one ordinary machine, and chat on hardware that could never hold a normal model. Each conversation keeps its own persona and memory, with no server. So each released size reports, on a plain CPU and on the Mac: weights size; peak memory per conversation; time to first token and tokens per second at batch 1; and the number of concurrent 12-turn conversations one CPU sustains at a stated latency. Measured with a fixed script on fixed hardware, never estimated. Report only, never a gate.
- **Microcontroller stretch (added Oct 2).** For the smallest released size: quantize (int8, then int4) and try to load and chat with it on a microcontroller-class board (an ESP32-S3 with 8 MB PSRAM is the reference; a 5M model is about 5 MB at int8; a graphing calculator with tens of MB of RAM, such as a TI-Nspire CX II, is a second target). Report whether it fits, tokens per second, and any quality loss from quantization on RC-12 dev. A stretch goal, never a gate.
- **Release demo (added Oct 2).** A small local demo with many characters chatting at once, each with its own persona card and memory, showing the role and identity skills (RC-12 ROLE) and the speed claim together. It is a demo only: Planck is not trained, tuned or marketed as a game-character model. Built after the first size passes, not before.
- **Planck-Assistant, a later track (Max, Oct 4).** The first target is the conversational Planck: casual multi-turn chat that keeps state, answers questions from lookup results directly, and never pads with service boilerplate ("I can certainly help", "happy to help", "anything else I can help with" stay rejected in every track). After a conversational size passes, a second variant is trained on the same base with task-help data (how-to steps, multi-part answers, lookup-heavy question answering) and released as its own model. It does not change the conversational claim or its pre-registration.

**Ledger ids.** P-055 (the curve), P-094, P-005, and the baseline blind-human-test item.

**Compute (est.).**
- 5070 about 1,170-1,650 h (2026-10-10 table, lower bounds; was 740-1,240 for the old 10/30/60M table).
- Titans about 4,620-9,300 card-hours: 1,300-2,400 per 150M seed at 75B, 240-700 per 60M seed (2.5-7.3 Titan-days, est.), plus the smoke test. The top of that range is above the roughly 8,450 free card-hours, which is why the seed queue above ends in a question to Max, not a cut.
- Mac 20-50 h (eval).

**Max-hours.** About 5 h:
- remote data-prep kickoff, Nov 21-22 (30 minutes; counted in Phase 3);
- remote smoke test, Dec 7-10 (20-30 minutes);
- remote launch, Dec 17-18 (30-45 minutes);
- update windows (10 minutes each);
- recruiting and scheduling raters (1-1.5 h);
- reading the Level R and Level A results (about 1 h);
- release approvals (about 40 minutes total);
- a remote check only if the heartbeat stops.

**Gate**
- *Level R* is claimed at the smallest size whose sealed pooled-seed S7 lower bound is 40 or better, with PERSIST at least 0.40, T0 at least 0.90 and 3 seeds at that size, under the OOD-H wording rule (Max, 2026-10-04).
- *Level A* is claimed only if every generative criterion passes on sealed, pooled over at least 3 seeds, and the human win-or-tie point estimate is at least 50%.
- *Level A decision, Jan 31* (review 20). It uses the seeds finished by then. A Level A claim at 150M becomes final only when seeds 2-3 confirm the generative criteria.
- *60M passes Level A.* 150M still runs, because the curve needs it, but the headline is 60M (the 20M point is already on the 5070 curve since 2026-10-10).
- *Nothing passes Level A by Jan 31.* Move to the fallback claims in section 7.

### Phase 5: writeup and release (Feb 1 - Apr 5, 2027)

**Purpose.** A paper that outsiders can check.

**Deliverables**
- Claude drafts the paper. Max writes the motivation and introduction and edits the rest.
- Full release:
  - code, generator, checker, tokenizer;
  - all datasets and every checkpoint;
  - `runs.jsonl`;
  - the ledger, including failed ideas;
  - the sealed split with its key;
  - OOD-H and the anonymized human ratings.
- arXiv submission, target window Mar 1 - Mar 31. The earliest date is Feb 22, if the claimed size is 60M or below and has its 3 seeds. It needs an endorsement (section 8).
- Optional: a short blog post or thread.
- The last 150M seeds finish in this window.

**Compute (est.).** 5070 50-100 h (5-seed lock-in runs at 10M/30M if E3 calls for them, the 20M point, extra evals). Titans: the remaining 150M seeds (counted in Phase 4).

**Max-hours.** 10-15 h.

**Gate.** The arXiv trigger in section 8.

---

## 3. Machines and operations

| machine | job | not used for |
|---|---|---|
| RTX 5070 (PC, WSL2) | all training at 5M and up; scoring every model's generations over RC-12; the teacher pilot; all bulk generation; tokenizing general text for 3-30M (and 60M); the 5/10/20/30M curve (3M conditional; 10/30/60M before 2026-10-10) | nothing that fits in 12 GB is excluded |
| M5 Mac | harness and generator development; likelihood tiers; tokenizer CPU work; smoke tests; scoring small Planck checkpoints on RC-12 dev; 5M screens only when the 5070 is down | bulk generation; anything that needs the laptop at home and awake for days |
| Titans (Lambda, 4 edge cards) | 150M and, since 2026-10-10, 60M seeds (Dec 17 - early Mar, worst case about Mar 24); smoke test Dec 7-10; 150M general-text prep on idle CPUs from late Nov | anything before Ultra finishes |

### Unattended running (review 14)

**Job queues.**
- Each machine runs a queue runner that takes job configs from a `queue/` folder.
- A job starts only if its `prereg.yaml` is committed (the runner checks the hash against git) and the machine's guards pass.
- Claude keeps 3-5 jobs queued per machine, about 1-3 days of work. A sleeping Mac, a capped Claude week, or a session opened on another device (which kills subagents) then does not leave the GPUs idle.

**PC.**
- The runner lives in WSL2.
- A Task Scheduler task starts it at boot, with no stored password. WMI running `wsl.exe` in the foreground can also start it.
- It pulls job configs from the private status repo every 10 minutes, so queueing needs no Tailscale.
- If Windows cannot start WSL from a boot-time task, the task runs at logon instead. A reboot then waits for Max to log in, and the heartbeat shows it.

**Mac.**
- The runner lives in its own clone under `~/Library/Application Support/planck-runner/`, because launchd cannot run scripts from `~/Documents` (macOS privacy block).
- It starts GPU jobs only on AC power.

**Lambda.** The 150M job script runs seeds back to back and reuses Ultra's keeper and pause/resume watchdog.

**Heartbeats.**
- Every job pushes a heartbeat every 30 minutes to a private status repo: step, loss, tok/s, ETA, GPU temperature and throttle state, and free disk.
- The PC and Lambda push outward, so monitoring never depends on Tailscale.
- Lambda uses a repo-scoped deploy key, never a personal token.

**Supervisor.**
- A scheduled Claude task on the Mac runs at about 7:00 and 19:00. It pulls heartbeats and results, runs `analyze.py`, appends to `runs.jsonl`, queues the next pre-registered jobs and updates STATUS.md.
- Two missed heartbeats put a flag at the top of STATUS.md. If the fix needs Max, it also adds one line to his checklist.

**Tailscale** is for setup and debugging only. Leaving it connected is Max's call.

### Guards

**Mac, every job**
- One model process at a time, with a lock file.
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO` / `_LOW_` set to 0.7/0.6, and `PYTORCH_ENABLE_MPS_FALLBACK` off.
- The job stops if swap grows 2 GB, or if step time doubles for 5 minutes.
- The runner will not start, and a running job checkpoints and stops, while LM Studio or Ollama has a model loaded (checked with `lms ps` and `ollama ps`). Both load a model on any API call, and `ollama serve` is resident as a Homebrew service (review 22).
- `caffeinate -i -s`, and atomic, resumable checkpoints.

**Disk, every machine (review 3)**
- Refuse to start below 25 GB free. Checkpoint and stop below 15 GB.
- Keep the last 2 checkpoints per run, plus the final and decay checkpoints.

**PC**
- GPU temperature and throttle state go in the heartbeat.
- After any reboot, the runner resumes from the last checkpoint.

### Environment (review 23)

- **Own venvs.** Planck gets its own venv on each machine, and stops borrowing the video-editor venv (torch 2.13.0).
- **One pinned torch version.** On the Mac it is 2.14.0 if released and it passes the self-tests, otherwise 2.13.x. The PC uses the matching cu128 build, and the Titans use 2.7.1 cu118.
- **Versions logged.** Every run logs torch, CUDA or Metal, driver and package versions.
- **Causal-leak self-test.** It runs at startup twice: with gradients, and under `torch.no_grad`. The review found that on MPS the leaking fused kernel is taken only in no-grad passes, so evaluation is the exposed path.

### Windows and WSL (review 15)

- Windows allows at most a 5-week pause, then forces an install and a reboot.
- Update windows are set at checkpoint boundaries and batched with other Max touchpoints, 10 minutes each (install, reboot, pause 5 weeks):
  - Fri Sep 25 (install, then pause);
  - Oct 24-25;
  - Nov 28-29;
  - Jan 2-3;
  - Feb 6-7.
- Active hours cover the day.
- `.wslconfig` raises WSL's memory limit from the default of half the 32 GB to 24 GB.
- A forced reboot between windows costs only the work since the last checkpoint, because the runner resumes at boot.

### Disk (review 3)

- **Mac today.** About 43 GiB free at 00:55 (LM Studio models 120 GB, HF cache 21 GB).
- **Where the data lives.** General-text mixes live on the PC, streamed and tokenized without keeping raw parquet, and on Lambda for the 150M. The Mac keeps at most a 1 GB screening shard.
- **Downloads.** Teachers and baselines download to the PC, not the Mac.
- **PC needs.** About 250-300 GB free (est.; CORPUS 7.3 and 7.4 edit 8, up from the 150 GB first planned): teachers and paraphrasers about 45 GB, human dialogue under 5 GB, the strict-open core as zstd text about 30-35 GB, tokenized shards for up to 2 vocabularies about 50 GB each, the chat pool with slot spans and provenance 10-20 GB, checkpoints 30-60 GB, streaming scratch about 50 GB.

---

## 5. The baseline recipe (from the ledger)

**Architecture**
- Decoder-only, with full softmax attention in every layer.
- Max's validated block: per-head gated attention, normalized value residual, 1/sqrt(depth) norm scaling, QK-norm.
- SwiGLU MLP with hidden size 8/3 d. RoPE with theta 10k. Tied embeddings.
- A parameter-budget solver matches every arm to within 1-2% of total parameters. Loops count unique parameters, and hash tables count in full.
- Tier 0 recall guard on every checkpoint (review 25): greedy-from-prefix recall of at least 0.95 at 1-2k tokens.
  - Enforced at 20M and up.
  - At 5M and 10M it is logged, and it trips only on a drop of 0.2 or more from the run's own best checkpoint.

Shapes (depth over width). E9 and E7 may change them:

| size | d | layers | heads | vocab | embedding | body | total |
|---|---|---|---|---|---|---|---|
| 5M (screen) | 192 | 8 | 3 | 8,192 | 1.57M | 3.54M | 5.11M |
| 10M | 256 | 10 | 4 | 8,192 | 2.10M | 7.86M | 9.96M |
| 20M (confirm) | 320 | 14 | 5 | 8,192 | 2.62M | 17.2M | 19.8M |
| 30M | 384 | 15 | 6 | 8,192 | 3.15M | 26.5M | 29.7M |
| 60M | 512 | 18 | 8 | 8,192 | 4.19M | 56.6M | 60.8M |
| 150M | 640 | 29 | 10 | 8,192 | 5.24M | 142.5M | 147.8M |

**Tokenizer**
- Planck's own byte-level BPE with byte fallback, digits and punctuation split, trained on the Planck mix.
- A minimal role template: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>` (about 2 tokens of overhead per message), plus the reserved span tokens.
- 8,192 for every size, so the curve measures the model and not the tokenizer. A size moves to a different nested truncation (P-098) only if E7/P-158 shows a gain of at least 1% bits per byte there.

**Data mix**
- **Chat share rule.** Chat share = min(the P-052 winner, the share at which the model reads the R0+R1 chat surface 25 times; 50 with R2 and spacing if P-187 allows). The rule, not a fixed number, is the recipe at every size. (Until 2026-10-10 the cap read "the 300M pool 25 times"; CORPUS 4.1 and 4.5, applied under 7.4 edit 4, count passes over the R0+R1 surface, about 0.5-0.9B tokens at the freeze, est.)

  At a 25% share and the 2026-10-10 budgets (est.; the surface size is unmeasured until the rate check):

  | size | tokens per seed | chat token-passes at 25% | passes over a 0.5-0.9B surface | share at the 25-pass cap |
  |---|---|---|---|---|
  | 5M | 25.6B | 6.4B | 7-13 | 25% |
  | 10M | 49.8B | 12.5B | 14-25 | 25% (at the cap with a 0.5B surface) |
  | 20M | 39.6B | 9.9B | 11-20 | 25% |
  | 30M | 29.7-59.4B | 7.4-14.9B | 8-30 | 21-25% |
  | 60M | 30.4B | 7.6B | 8-15 | 25% |
  | 150M | 75B (100B) | 18.8B (25B) | 21-38 (28-50) | 17-25% (13-23%) |

  - If P-052 picks 50%, every size passes 25 reads at the low end of the surface estimate (5M 14-26, 10M 28-50, 150M 42-75), so there the cap, not P-052, sets the share unless R2 and P-187 allow 50 passes.
  - If the pool ends smaller, the capped shares drop to stay under the cap; the share never goes below the critical share P-051 measures (CORPUS 5.4).
  - If P-162 wins, paraphrases count as new surface and raise the capped shares.
- **Chat content**, in the same template as SFT, from step 0:
  - Skill-dense skeleton renders: S1-S9, 2-4 events per conversation, answers mentioned second as often as first, nonce and real slot values, stale traps and twins.
  - Social basics and OASST2, with the OOD-H threads removed.
  - Grounded lookup dialogues on a fictional knowledge base: 0-5% of chat at 10-30M, 10-20% at 60-150M. BM25 lookup with zero parameters.
- **General text** fills the rest. It is knowledge-light, with 0% code by default (P-060 tests 15%). Since 2026-10-10 (CORPUS 7.4 edits 1-2, applied; the components are CORPUS 2.1's):
  - plain prose, strictly open: Common Pile CCCC, Gutenberg, LoC books, non-encyclopedic Wikimedia, non-technical StackExchange and CC YouTube transcripts, all after the date gate and selected by Planck's own classifiers;
  - simple narrative (Cosmopedia v1 stories, plain public-domain fiction);
  - grounded rephrasings (RH) and human dialogue (OASST2 and other D8-clean sets);
  - not FineWeb-Edu (Q5), not bulk Wikipedia, no code, math, science or law.
  - Note (review 27, now moot): the tension between FineWeb-Edu's knowledge-dense selection and "knowledge-light" ended with its exclusion. Fact density is now set by E-sel-3 (which absorbs P-083), and source tags by E-sel-6 (P-161).
- **Provenance.** Every training string follows D8 and the Phase 1 provenance rule.
- **Hygiene.** Dedup, 8-gram caps, 13-gram decontamination against RC-12, OOD-H and the human-test scripts, and a mutation-tested checker. The chat pool repeats on a spaced schedule, outside the memorization window.

**Tokens per size**

Rewritten 2026-10-10 to the adopted budgets (CORPUS addendum, Sep 24; CORPUS 7.4 edit 7); P-186's freeze rules can still move them (CORPUS 5.5).

| size | tokens per seed | tok/param | where |
|---|---|---|---|
| 3M (only if 5M passes a Level A family on dev) | 30-60B | 10,000-20,000 | 5070 |
| 5M | 25.6B | 5,000 | 5070 |
| 10M | 49.8B | 5,000 | 5070 |
| 20M | 39.6B | 2,000 | 5070 |
| 30M | 29.7-59.4B | 1,000-2,000 | 5070 |
| 60M | 30.4B | 500 | Titans |
| 150M | 75B default, 100B if the Titan smoke-test rate allows | 500-670 | Titans |

- Screening budgets: 5M at 0.25B tokens; 20-30M at 0.4-1.5B.
- Honest note: every chat model that works well at this size used 2,900-83,000 tokens per parameter. The bet is that data density substitutes for token count. The 150M goes to 300B only if Phase 3 or 4 shows skills are token-limited and the Titans are free.

**Optimizer and schedule**
- NorMuon on the matrix parameters. AdamW on embeddings, norms and gains. Cautious weight decay.
- LR tuned at 5M for each arm, with the embedding LR tuned for each vocab.
- WSD schedule: 1% warmup, a stable phase, then linear decay to zero over the last 20%.
- Weight decay re-derived for the run length.
- Batch grows with data: 16x2048 early, 64-128x2048 at scale. Gradient clipping at 1.0.
- Precision: bf16 autocast with fp32 master weights on the Mac and 5070; fp16 with loss scaling on the Titans.
- Common random numbers across arms. Reported numbers average the last 3-5 decay checkpoints.
- Data and post-training ablations run as decay branches off a shared trunk.

**Context length**
- Train at 2048. The 150M adds a final phase of about 10% of its tokens at 4096.
- RC-12 conversations are designed to stay at or under about 1,800 Planck tokens.
- On CUDA: packing with document masking if E1 agrees. On MPS: length-bucketed, unpacked batches.

**Post-training**
- SFT built per whole conversation, with sequences that mirror what the model sees at inference: sequences are cut wherever spans are stripped.
- Short replies, no fixed system prompt, deflection filtered out.
- At most one epoch of multi-turn DPO, and only if P-014 wins.
- Keep the SFT checkpoint; average ("soup") checkpoints.

**Decoding (pre-registered).** T = 0.6, top-p 1.0, repetition penalty off, max_new_tokens 256, 3 sampling seeds. Greedy is secondary.

**Eval at every checkpoint**
- Fixed-window bits per byte (general text, dialogue, answer spans), Tier 0 margins, history dependence and induction score.
- RC-12 dev at the end of each run, but only after the lock. Sealed only at milestones.
- Seeds:
  - 2 for screening;
  - 3 for continuous confirmations;
  - 5 for pass-or-fail lock-in checks;
  - 3 per curve size (5 at 10M and 30M if E3 shows lock-in);
  - at least 3 at the claimed size.

---

## 6. Budget

All figures are estimates, from arithmetic on REPORT s7, loop.md, compute.md and their verify files. `bench_micro` and the pilot replace them.

| phase | dates | Mac GPU-h | 5070 h | Titan card-h | Max h |
|---|---|---|---|---|---|
| 0 | Sep 24 - Oct 14 | 15-30 | 25-60 | 0 | ~1 |
| 1 | Oct 1 - Oct 18 | 0 (10-20 if vLLM fails), +2-4 CPU | 10-25 | 0 | ~0.3 |
| 2 | Oct 9 - Nov 22 | 5-15 | 180-350 | 0 | ~0.2 |
| 3 | Oct 19 - Dec 3 | 20-60 | 400-640 | 0 | ~2.5 |
| 4 | Dec 4 - Feb 21 | 20-50 | 1,170-1,650 (lower bounds, 2026-10-10) | 4,620-9,300 | ~5 |
| 5 | Feb 1 - Apr 5 | 10-20 | 50-100 | (last 150M seeds, counted in 4) | 10-15 |
| **total** | | **~70-175** | **~1,840-2,830** (was ~1,400-2,400 before 2026-10-10) | **4,620-9,300** | **~19-27** |

**Capacity checks**
- 5070: about 580-990 h of roughly 1,270 from Oct 12 to Dec 3, and about 1,170-1,650 h (lower bounds) of roughly 1,706 from Dec 4 to Feb 21 (2026-10-10 table; was 740-1,240 of about 1,200 to Jan 24 for the old one).
- Titans: about 8,450 card-hours free Dec 17 - Mar 15. The plan uses 55-110% of that (46-85% before the 60M seeds moved there on 2026-10-10); above 100%, the seed choice goes to Max (Phase 4, Seeds).

**Max-hours**, including the recurring asks (review 26):

| item | hours |
|---|---|
| setup (9/25) and prereg read (10/3-4) | 0.7 |
| teacher skim (optional) | 0.25 |
| weekly STATUS reads, about 20 weeks | 2-3 |
| 5 Windows update windows | 0.8 |
| Tailscale reconnects and debug asks | 0.5-1 |
| remote sessions (data prep, smoke test, launch, heartbeat checks) | 1.8-2.2 |
| arXiv endorser ask | 0.25 |
| recruiting and scheduling raters | 1-1.5 |
| release approvals (6-8, batched) | 0.5-0.7 |
| reading Level R and Level A results | 1 |
| paper | 10-15 |

**Money**
- $0 of cloud.
- Electricity (review 21): 1,400-2,400 load hours at about 0.35 kW, plus idle power until spring, is about 650-1,000 kWh. That is roughly $130-250 at NJ residential rates (est.). Max tells whoever pays the bill.
- Optional: rater pay, $375-600, if Max chooses to pay.

**Data**

| data | amount | source | ready by |
|---|---|---|---|
| RC-12 dev | ~600 conversations | Claude-written templates (D8 gold set) | Oct 2 |
| RC-12 sealed | 600 conversations (Max, Oct 2-4; the sizing rule's first expectation was 1,000-1,200) | Claude-written banks, separate session, encrypted | Oct 11-14 |
| OOD-H | 150 held-out OASST2 threads + 50-60 rater-written scripts | human-written user turns | lock / Jan 17 |
| Human-test scripts | 150 | Claude-written (D8 gold set) | Jan 10 |
| Teacher pilot | 600 conversations, ~0.2M tokens | 3 Apache-2.0 teachers on the 5070 | Oct 8 |
| Chat pilot | 20-30M accepted tokens | pilot winner, 5070 | ~Oct 18 |
| Chat pool, all sizes | 300M accepted unique (about 500M generated at ~60% yield, est.; at least 100M by Nov 8) | teacher on the 5070 | Nov 22 |
| General text, 3-30M | about 15-25B unique after selection (est.) | strict-open core plus Planck selection, simple narrative, human dialogue, streamed on the PC (CORPUS 7.4 edit 1) | Nov 1 |
| General text, 60M and 150M | about 5B and 15-20B unique, at most 4 reads | strict-open core streamed from Common Pile on the Lambda box's CPUs (CORPUS 7.3) | Dec 14 |

**Training token-passes**
- Phase 3: about 100B, across 50-80 runs at 5M and 40-60 at 20-30M.
- Curve (2026-10-10 budgets):
  - 77B (5M, 3 seeds)
  - 149B (10M, 3 seeds)
  - 119B (20M, 3 seeds)
  - 119-238B (30M, 3 seeds plus the control)
  - 91B (60M, 3 seeds, Titans)
  - 225-300B (150M, 3 seeds, Titans)

**Generation-rate note (review 21).** 170-330 h of generation at 250-500 accepted tok/s yields 150-600M tokens. 300M is the midpoint, so the pool target and the hours now agree. Rates are counted in Planck's tokenizer.

---

## 7. Risks and kill criteria

| risk | early sign | response or kill criterion |
|---|---|---|
| **The strict bar is unreachable at 150M** | P-001 flat; by Nov 22 no 30M arm lifts correction margins at 4+ turns beyond seed noise | Level A is attempted only at 60M and 150M with the best-margin arm. If nothing passes by Jan 31, no Level A claim. **Fallback claims, in order:** (1) Level R at the smallest passing curve size, plus per-skill size floors, which are new because no strict pass/fail floor for multi-turn sub-skills has been published (the published multi-turn numbers below 150M are loose and show failure: BananaMind-2, RxT, Loom; Max, 2026-10-02: took all recommendations in rc12/DECISIONS_FOR_MAX.md (item 15)); (2) the note arm, which counts as the model, if it passes where the plain model does not; (3) "Planck-{N}M + state card (scaffold)", clearly labeled; (4) a negative result: strict corrections not reached at 150M or below with these interventions, and where each one failed. |
| Level R also fails | Sealed S7 lower bound below 40 (or PERSIST below 0.40) at 150M | The claim becomes per-skill floors on the curve plus the negative result (paper path in section 8). Odds: not re-estimated since the Oct 4 change, lower than the 40-60% judged for the old bar (judgment). |
| Someone already clears the bar | LFM2.5-230M or Falcon-90M passes Level R (or A) on RC-12 in Phase 0 | "First" moves to sizes below that model. The curve points under 90M/230M carry the claim, or the claim becomes capability per parameter. |
| The test is invalid | Mutants survive; LFM2-2.6B fails a Level A family; families sit at floor or ceiling | No lock until fixed. The re-anchor rule applies before the lock, with a 0.60 floor, never after. |
| The model learns RC-12's format, not the skill | Sealed scores well above OOD-H | OOD-H is pre-registered, with the -5 wording rule. Sealed banks are written before the generator and kept encrypted. 13-gram decontamination against RC-12, OOD-H and the human scripts. |
| The data hypothesis fails (as ufakzeka-1 found at 151M) | P-012 and P-053 both under +5, with the CI excluding +5 | Stop at the pilot. Phase 3 weight moves to the note, the recency and delta layers, and the curriculum. Middle zone: continue to 100M and re-test at 20M. |
| Generation is slower than planned | Measured accepted tok/s after the first bulk week | Decide by Nov 8: shrink the pool, lower the capped shares under the pass cap, use the P-162 paraphraser; anything this would cut goes to Max as a list first (2026-10-10). The Mac is not a fallback. |
| No multi-turn signal in 5M proxies | E4 | Screen at 20M: the must-run list first, then E8-E10 and the rest at 20M under new pre-registrations; what cannot fit goes to Max as a list first (2026-10-10). |
| Seed lock-in lottery on pass-or-fail skills | E3 and P-026 lock-in rates vary | 5 seeds at 10M and 30M on the 5070 (cheap). 60M and 150M stay at 3 on the Titans, with lock-in rates reported. |
| 5070 unavailable (video-editor or IseMedia use, reboots, WSL issues) | Missed heartbeats; GPU throttling | Boot-time runner, update windows, resumable jobs. If it is out for more than 2 weeks: the Mac runs 5M screens and 20M at 0.4B on the must-run list first, and the 5070's curve list is re-planned with Max (the 60M seeds are already on the Titans since 2026-10-10). |
| Titans late (Ultra overrun, thermal pauses, break access) | Ultra ETA past Dec 14; smoke test fails Dec 7-10 | The smoke test exists so the first Titan failure is not the launch. 150M at 75B. If not started by Jan 10: 150M at 50B on the 5070 after the 60M seeds (about 18-24 days, est.), or pausing Ultra (Max's call). (Written when the 60M seeds ran on the 5070; since 2026-10-10 they queue on the Titans too, so a late Titan start delays both sizes, and the options for each go to Max as a list first.) |
| Another Mac kernel panic | Swap above 2 GB, or a panic log | The runner halts; no retries until the cause is found. The Mac now carries little GPU work. |
| Disk fills | Free space under 25 GB | Disk guards stop jobs; big data lives on the PC and Lambda. |
| Unattended plumbing fails | Two missed heartbeats | STATUS flag; checklist line if Max is needed; queues hold 1-3 days of work. |
| Claude usage caps (weeks 1-3 are mostly Claude-written code) | Weekly cap reached before the lock | Order work by the critical path (RC-12 dev and graders, prereg, generator, harness) and keep GPU queues full. A usage cap slows or defers work; it never cuts a bet, a review, an audit or a test (Max's quality-over-usage rule; rewritten 2026-10-10, cuts audit row 85: this row used to say "Cut to the must-run list"). If deferral would push a bet past its window, the list goes to Max first. Never cut the pre-registration. |
| Contamination or a D8 slip | Decontamination hits; audit gaps | Provenance column, teacher-written templates, 13-gram checks, audit table in the repo. |
| Personal information in public files | Scrub check before each push | Trimmed public PLAN.md; `private/` gitignored; no names of Max's school, professors, colleges or campus access in the repo. |
| Max's time runs over | Missed Max tasks | Claude pauses Max-dependent items, never the compute. No Max tasks Oct 8-17, Oct 26 - Nov 1, Nov 9-15, or during finals. |
| Too many bets | Phase 3 behind at Nov 22 | The must-run list and the combined-recipe test take the GPU first; the list of bets that would miss the freeze goes to Max, with costs, before any is dropped (2026-10-10); unrun bets stay "not run". |

---

## 8. What goes public when

**Sep 25-27 (after Max's yes): the public repo**
- Contents:
  - README (goal, the Level R and Level A wording above, what counts as the model);
  - the trimmed PLAN.md, LEDGER.md, scrubbed `research/`, `tools/`;
  - the probe and battery code, with P-001 and P-002 results as they land.
- Trimmed PLAN.md (review 18) omits:
  - Max's deadlines and test dates;
  - target colleges;
  - professor and school names;
  - campus-access details;
  - machine addresses.
- The full plan stays in the gitignored `private/`.
- Priority is established from day one by commit timestamps.
- Licenses (Max's call on Fri 9/25):
  - Apache-2.0 for code;
  - CC-BY-4.0 for Planck-made data and eval;
  - CC-BY-SA-4.0 for any released slice derived from Wikipedia (review 27), or ship the build recipe instead of that slice.

**Oct 11-14: the pre-registration, before any Planck model is scored on any split**
- `prereg/RC-12.md` with everything listed in Phase 0.
- The SHA-256 of the sealed files and generator seed.
- The encrypted sealed archive; the key is kept outside the repo in two places.
- The dev split in the clear.
- The OOD-H thread-ID hash.
- Timestamped by a GitHub release and a Software Heritage snapshot.

**From then on, weekly**
- STATUS.md. The public copy carries no personal schedule.
- One `prereg.yaml` per experiment, committed before launch. The runner enforces this.
- `results/runs.jsonl`, append-only.
- Ledger updates, with REJECT and TIE rows in "Tried and failed". Negative results are published.


**Nov 8:** generator, checker, tokenizer v1, and the 20-30M-token pilot dataset. It was rendered by an Apache-2.0 teacher, so it is releasable.

**Dec - Feb:** each curve point's weights and model card on Hugging Face, once its sealed numbers are final and Max approves. The card gives total and body parameters, tokens, seeds, RC-12 and OOD-H scores, and known failures.

**At submission:** the paper, the sealed split and its key, all datasets, all checkpoints, OOD-H, and the anonymized human-rating data.

**arXiv trigger (three paths)**
1. At least one curve point meets Level R on sealed RC-12 under the pre-registered analysis, with 3 seeds at the claimed size. The paper is Level R plus the curve, plus Level A if the human test is done and passes.
2. Level A has failed by Jan 31. The paper is Level R plus the curve plus the negative strict result.
3. Level R fails everywhere (review 27). The paper is the per-skill size floors, the negative result and the released test and data. It still goes to arXiv: a measured negative against a public sealed test is publishable. If no endorsement comes through, it goes out as a workshop submission or a blog post with the repo.

**Target window:** Mar 1 - Mar 31, 2027; earliest Feb 22 if the claimed size is 60M or below.

**Endorser.** First-time submitters need one. Max asks an established arXiv author in cs.CL and confirms by Nov 30 that the person can endorse in that category. The name stays out of the repo.

---

## Appendix: Reviewer notes

All 28 review items were applied, except as noted here.

- **3 (partly):** deleting the ~90 GB of unused LM Studio models is optional, not a gate. Teachers and baselines now land on the PC, and the disk guard stops jobs before the Mac fills. Deletion stays Max's own action.
- **4 and 5 (modified):** all three 60M seeds stay on the 5070 by default, so one size never mixes bf16 and fp16. A 60M seed moves to the Titans only under the bench re-plan rule, and is then labeled. The Titans take 150M seeds 2-3 as the review asked. (Superseded 2026-10-10: all three 60M seeds now run on the Titans, per PLAN.md's Sep 24 budgets, so 60M still never mixes bf16 and fp16.)
- **8 (modified):** the rater-written half of OOD-H is collected in January, not before the lock. Only its protocol is pre-registered, which is enough because those scripts cannot have leaked into training.
- **15 (modified):** update windows move to Oct 24-25, Nov 28-29, Jan 2-3 and Feb 6-7. Oct 26 - Nov 1 is a no-Max week, and the first 5-week pause ends Oct 30.
- **16 and 26 (modified):** the Titan smoke test, the data-prep kickoff and the launch all run remotely (Nov 21-22, Dec 7-10, Dec 17-18), so no campus trip is needed.
- **19 (left open):** whether to pay raters is Max's call. The plan budgets the 25-40 rater-hours either way.
- **23 (modified):** torch is pinned to one exact version that passes both leak self-tests (2.14.0 if released), not to 2.14.0 unconditionally. The loop track only establishes the fix in 2.13.0.
- **28 (superseded):** E001 has started. Its logs appear from 00:48, and at 00:55 a Falcon-90M dry chat run on CPU was in progress. STATUS records the actual times.

## Appendix B: triage of the ranked bets PLAN never placed (2026-10-10)

Written 2026-10-10 by Claude for the cuts audit (`research/CUTS_AUDIT_2026-10.md` rows 35, 37, 53-58), before any result of these bets exists. 106 of the ledger's 185 ranked bets were never named in this plan (a script over LEDGER.md's rank table against every P-number in PLAN.md, run 2026-10-10), so "everything else runs if time allows" was their only home, including 10 of the 14 untested-at-this-size bets. Each now has a **slot** (where and when it runs), a **gate** (the result that triggers it, and what happens if the gate does not fire), or a **reason** it does not run. Every bet that runs gets its own pre-registration first, written before any of its results. Max's ledger rule (test as much as possible that is untested at this size) puts the untested bets first inside each group: they are marked (U). Hours are the ledger's estimates.

| bets | decision | trigger or reason | source |
|---|---|---|---|
| P-024 smeared keys | slot: SCREENS batch 1 S007 (stage 2 run); its one-layer-fewer half and 1-vs-2-layer toy: gate | S007's Q2 and the PLAN 3d toy; new pre-registration, about 1.5 GPU h. Fired 2026-10-11: T001 R3 YES, and T001 PREREG lets the toy's half open the gate on its own ("S007's Q2 is reported beside it and does not block it"; S007's Q2 read UNRESOLVED), so the depth-reduced arm is SCREENS2 S018 | SCREENS.txt:40; audit row 10; T001_mqrar_toy PREREG R3 |
| P-105 MTP auxiliary loss | slot: S006, reinstated 2026-10-06, its 4 runs done, verdict pending; curriculum, 2-head and 20M arms: gate | S006's follow-up trigger (written before its verdict, audit S2) | SCREENS.txt:1830-1863; audit row 2 |
| P-110 shared W_q/W_k | slot: the 8-layer sharing contrast in SCREENS batch 2 (S009+); the 9-layer arm and the K = V tie: reason | batch 2 is pre-registered after the stage 2 verdicts; the 9-layer arm moves four things at once, and K = V's one measured cost (3.1% perplexity) points against it | audit rows 4-6 |
| P-112 hyper-connections (non-looped), P-172 ReLU-squared, P-173 multipliers, P-174 TOP, P-175 meta tokens, P-113 output-head rank, P-108 sequence-length curriculum | slot: SCREENS batch 2 at 5M (bpb decides; P-113 also reports its repetition rate, P-108 its wall time), each a default-off flag with tests and mutants | stage 2 verdicts read; about 1.5 GPU h per screen; a NOMINATE goes to the post-lock Tier 0 reading and a 20-30M confirmation | SCREENS.txt:53-54; audit row 23 |
| P-106 small batches and induction | slot: E2 D1 redesigned as a g in {0.5, 1, 2} check at batch 32 x 2048 with the IND hook | before any curve run changes batch size; dated E2 addendum, 3 runs, about 1 GPU h | audit row 20 |
| P-104 partial RoPE / NoPE | gate | a long-context eval past 2k exists; then 30M, 15-25 h | audit row 22 |
| P-109 attention-only SAN | reason | SAN reached parity only at about 105B tokens at 24M, so any screen Planck can afford would reject it for the wrong reason | SCREENS.txt:52; audit row 21 |
| P-137 sharded-vs-full ratio | slot: its own post-lock measurement (F6 left RC-12 to run as P-137) | after the lock, before any Planck score; 3-6 h | audit row 98 |
| P-036, P-037 (U), P-038, P-039, P-040, P-041, P-156 (think variants) | gate | P-034's T arm beats B in the 3c four-arm test; then T variants at 10-30M, 2 seeds each, P-037 first. If T does not win, each is "not run: parent did not win" | LEDGER caveat 3 |
| P-031, P-157, P-043, P-042 (note variants and the split writer) | gate | N wins in 3c; they join P-030, P-029 and P-019 in 3c's Nov 9-22 N-variant slot. P-043 is reported as a scaffold (D4) | LEDGER caveat 3 |
| P-027 errors plus corrections, P-028 belief-state auxiliary loss | slot: 3b after the first pool mix (5M proxy, then 30M for P-028) | both need skeleton data with planted slots; P-028 5 seeds at 30M, 25-40 h | LEDGER |
| P-057 span upweighting, P-059 template subsampling, P-062 narrow register, P-063 vocabulary-breadth knee | slot: 3b after the must-run data bets (5M, then 10-20M for P-063) | the first pool mix; P-062 and P-063 read the register knobs that P-048 pre-registers | LEDGER |
| P-058 chat-shaped anneal | slot: decay branches off a 30M trunk in 3b (part absorbed by E-sel-4) | the first pool mix; 6-10 h | CORPUS 6.3 |
| P-160 BPE-dropout on repeated chat | slot: an arm of P-187, the pool-pass curve | P-187 runs Oct 19 - Nov 1 at 5M | CORPUS 4.5 |
| P-163 instruction-response conversion | reason: absorbed by CORPUS 6.3 row 7 (P-162 + P-165), whose QA-rephrase arm is this bet (this triage's reading: CORPUS 6.3 does not name P-163) | none | CORPUS 6.3 |
| P-164 DCLM in the backbone | reason: excluded by Max's rulings, not budget: DCLM is a web crawl (Q1, Reading B) and its selector's positives are GPT-4-written (Q5) | reopens only if Q1 changes | CORPUS 1.4 |
| P-165 rephrase the backbone | slot: CORPUS 6.3 row 7 (absorbed) | the first pool mix | CORPUS 6.3 |
| P-166 SYNTH reasoning data | gate | a D8 provenance check of SYNTH (every model-written token from an Apache-2.0 model); if it passes, a 10% candidate in E-sel-4's decay branches; if not, reason: D8 | CORPUS 6.3 |
| P-169 MiniPLM selection | gate | E-sel-2 shows selection pays and a reference pair passes D8; then an extra E-sel-2-style arm at 20-30M | CORPUS 6.3 |
| P-054 pure skill floor at 1-10M | slot: the 5070's Phase 4 spare hours (row 7 / CORPUS 5.4 queue); earlier if E4 finds no signal at 5M | its own pre-registration; 30-60 h | audit row 57 |
| P-050 token curve from public checkpoints | slot: once RC-12's likelihood rows run on real models (audit S10), score SmolLM2-135M's public intermediate checkpoints, before P-186's freeze reading | minutes per checkpoint; no training | audit row 56; CORPUS 5.5 |
| P-086 lookup dialogues in pretraining vs SFT | slot: 3b after the first pool mix (5M/20M, about 5% grounded tokens) | the pool's grounded lookup dialogues | LEDGER |
| P-087 (U) lookup placement, P-088 chunk count, P-091 verbatim turns beyond the window, P-092 no own replies, P-093 compressed chunks | slot: eval-only sweeps on the first 20-30M Phase 3 checkpoint trained with lookup data, after the lock and before the freeze, so P-094's curve condition uses the winning placement | P-087 first (untested, 3 h) | LEDGER |
| P-089 retrieval queries | slot: retrieval-only recall@3 on CPU with P-087's sweep; its SFT half at 60M+: gate | the CPU half shows a gain; the SFT half then runs on a 60M checkpoint after the curve | LEDGER |
| P-090 (U) harness retrieval vs model-invoked lookup | slot: the post-lock diagnostic that Max's decision 2 allows for Planck-issued lookups, on one 20-30M checkpoint, report only | after the lock; 10 h; never in a claim | PLAN s1 LOOKUP claim; audit row 58 |
| P-095 lookup-tool traces, P-170 single-call tool traces | gate | P-090 picks the tool route; then P-170 (single calls, SmolLM2-135M, 6-12 h) before P-095 (multi-turn); else "not run: parent picked harness retrieval" (the recipe has no tool calling, LEDGER baseline row 32) | audit row 58 |
| P-107 in-context-learning transience | slot: watched on P-186's 5M trunk (the IND hook and the branch scores; a late fall reads as transience); its L2 arm: gate | a fall is seen; then a 5M L2 arm, 15-30 h | CORPUS 5.2 |
| P-171 RETRO-style retrieval | gate | P-086 shows lookup data in pretraining helps; then the 30M in-context-chunks arm before the freeze; the 60M CCA pair after the curve, report only | LEDGER |
| P-069, P-070, P-168 (OPD arms) | slot: arms inside P-068, now scheduled before Nov 16 | P-068's pre-registration | 3e; audit row 49 |
| P-074 (U) cross-tokenizer distillation | gate | P-068 shows OPD beats SFT on own-history families; then byte-prefix OPD into Planck's tokenizer at 30M, 20-40 h (the only route D6 leaves open, audit row 50) | LEDGER |
| P-075 interactive teacher feedback | gate | P-068 has run (it reuses its rollout code); SmolLM2-135M, 10-20 h | LEDGER |
| P-073 (U) SDFT, P-071 self-distillation | slot: the Phase 2 public-model SFT block on the 5070 (SmolLM2-135M, 3 seeds), after P-012 decides, before Nov 16 | P-073 first (untested); 6-12 h each | LEDGER |
| P-072 persona distillation | gate | P-006 shows persistence decays with distance from the system prompt; else "not run": Planck trains with no fixed system prompt (P-016) and keeps a persona per conversation, so a distilled persona has no use | LEDGER |
| P-167 sequence-level distillation | reason: absorbed by P-053 (teacher-written vs generic text at equal tokens is the same comparison; this triage's reading, not stated in LEDGER or CORPUS) | none | this triage |
| P-076 self-contrast decoding, P-077 best-of-N, P-078 contrastive search and min-p, P-181 MMI rerank | slot: decoding diagnostics on dev, public models any time, Planck checkpoints after the lock; report only, never the pre-registered decoding (T 0.6, no penalty); P-077 labelled a scaffold (D4) | 2-4 h each | audit row 55; NOVELTY s6.1 DRY caveat |
| P-079 answerability probe, P-081 context reliance through SFT, P-136 mechanistic diagnostics | slot: P-079 and P-136 on the first post-lock 20-30M checkpoint (P-136 beside P-021's 30M test); P-081 on every 3e SFT checkpoint | after the lock; minutes to 6 h | LEDGER |
| P-138 correction probes at deployed precision | slot: the release's int8/int4 step (Phase 4, microcontroller stretch), which already reports quantization loss | a size passes | LEDGER |
| P-140 re-score TinyStories and SimpleStories checkpoints | slot: a 5070 gap, with its own short pre-registration and rubric | a GPU gap; 3 h | LEDGER |
| P-099 byte-composition embedding, P-102 hashed-bigram table | slot: E7 arms at 20M (E7 owns embedding-form questions; pre-registered before Oct 24) | E7's pre-registration | LEDGER; PLAN 3a |
| P-103 embedding table as entity store, P-182 sparse lookup memory, P-183 low-bit from scratch | reason: closed by D1 (total parameters); under that count they store per FLOP, not per parameter | reopen only if D1 changes | audit row 51 |
| P-184 SpaceByte byte-level control | slot: Phase 5 (writeup control, one seed on the 5070) | after the curve; 5-10 h | LEDGER |
| P-111 looped wide core with hyper-connections | gate | E10 (P-150) shows looping helps at 5M or 20M; then candidate C at 10M, 40-60 h | LEDGER |
| P-177 MoEUT shared-layer MoE | gate | E10 shows layer sharing helps; then 10M, 20-40 h | LEDGER |
| P-149 hybrid mixer vs pure attention | gate | P-022 or P-148 wins in 3d; then 30M, 20-35 h | LEDGER |
| P-176 progressive growth (G_stack) | gate | the Phase 4 re-plan rule fires (the curve would not fit); G_stack is then the first speed option measured at 30M, before any budget option goes to Max | LEDGER |
| P-185 MoE at fixed total | slot: the Titans after the 150M and 60M seeds, report only (expected negative) | Titan time left; 20-40 h | LEDGER |
| P-114 (U), P-115 (U), P-116 (U), P-117 (U), P-118, P-119 (its bits half; S003 ran only its 5M loss reading, no skill or tail-item measure), P-120, P-121, P-122, P-123, P-124, P-125, P-178, P-179 (the bits-per-parameter rig, Max's idea 3) | slot: after the lock push, on the Mac or 5070 gaps, calibrated first (LEDGER baseline row 45); P-114 first (it decides whether to strip facts from the data), then the untested probes; report only, since stored knowledge is outside the claim (PLAN s1) | 1-20 h per probe at about 1M | audit rows 30, 53 |
| P-126 bleed battery, P-127 per-user LoRA, P-131 (U) sparse per-user update | slot: a 5070 gap on public SmolLM2-135M-Instruct (no RC-12 needed); P-131 runs as an arm of P-127's design because it is untested and costs 1-2 h | P-127 under 1 GPU h; Max's idea 10, outside the within-conversation claim, report only | audit row 54 |
| P-128, P-129, P-132, P-133, P-134, P-180 (per-user stages) | gate | P-127 meets its win line; else "not run: parent missed its line" | LEDGER caveat 3 |
| P-130 per-user delta-rule state | gate | P-127 meets its line and the P-022 delta-rule layer exists (3d); then 5-20M | LEDGER |
| P-135 per-user memory table | gate | P-127 meets its line; 60-150M only, so Titan time after the seeds | LEDGER |

**Placed in Phase 0 but never run (audit row 37):** P-142, the counterfactual open-book probe (likelihood only, 1-3 h), gets a slot with RC-12's first real-model likelihood rows (audit S10), on public models, before the lock.

**For Max (budget cuts still standing, put to him, not dropped):** P-186's 10M trunk to 20,000 tok/param (about 315 h eager per seed by arithmetic) and a second 5M seed at 20,000 (about 109 h); and, if the Titan window cannot hold three seeds at both 60M and 150M, which seed waits.
