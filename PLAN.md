# MaxGPT-Planck plan

Version 2, 2026-09-24. It is built from the draft plan and its 28-item review, `research/REPORT.md`, `LEDGER.md`, `research/followup/*`, `research/lanes/compute.md`, `research/brainstorm/*` and Max's checklist. No model was loaded to write it. "(est.)" marks arithmetic or judgment, not a measurement. Every throughput figure for Planck's shapes, and for the teacher on the 5070, is unmeasured and could be off by 2x either way. `bench_micro` (5070, first weekend) and the teacher pilot (Oct 5-8) replace those figures with measurements.


**Changes from the draft**
- **The 5070 now does almost all GPU work.** That covers scoring the baselines' generations, the teacher pilot, all bulk generation, the 5M screens and the SmolLM2 fine-tunes. The Mac drops from 600-1,000 GPU-hours to about 70-175 and never generates in bulk.
- **One chat pool for every size.** It holds 300M accepted tokens, generated on the 5070 by Nov 22. Each model reads it at most 25 times, and that cap sets each size's chat share.
- **The pre-registration locks Oct 11-14.** That is before any Planck model is scored on any RC-12 split, dev included. A bootstrap over conversations sizes the sealed split, and Level R now reports a human-written out-of-distribution set (OOD-H) beside the sealed score.
- **The curve starts at the Dec 4 freeze, with 3 seeds at every size that could carry the claim.** The 150M defaults to 75B tokens, with seeds 2-3 on the Titans in Jan-Feb. The Level A decision moves to Jan 31.
- **New section 3 on unattended running.** It covers job queues on each machine, heartbeats pushed out to a status repo, disk guards, Windows update windows, and a Titan smoke test on Dec 7-10.
- **D8 now covers every string that reaches training text.** Templates, slot pools and word lists come from human-written open sources or from the teacher, never from Claude.

---

## 1. The goal and the claim we are trying to earn

MaxGPT-Planck is a size curve of small chat models: Planck-10M, 30M, 60M and 150M. Sizes count total parameters, and the non-embedding body is always reported beside them. The aim is to find the smallest model that passes a strict, pre-registered multi-turn conversation test. The target is conversational mechanics over the conversation's own context:
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

**Level R** (reported along the way). Exact wording:
> "Planck-{N}M ({N}M total parameters, {B}M non-embedding) is non-inferior to Qwen2.5-0.5B-Instruct on RC-12, a pre-registered, sealed 12-turn conversation test. On its own conversation history, the lower bound of the paired 95% confidence interval on the RC-12 composite (bootstrap over conversations and seeds) is no worse than -3 points ({k} training seeds, {c} conversations, {n} scored turns). On OOD-H, {h} conversations with human-written user turns, the paired difference is {d} points (95% CI {lo} to {hi}). It is the smallest model on our curve that does this."

If OOD-H's paired point estimate is worse than -5 points, the claim is worded as "non-inferior on RC-12's format", with the OOD-H result beside it. The unqualified wording is then not used. Level R statements made in December are provisional until OOD-H is complete in January.

**Level A** (the headline). Exact wording:
> "Planck-{N}M passes a strict multi-turn bar that no model at or below 0.6B passed in our tests. On sealed RC-12, pooled over {k} seeds, it:
> - applies corrections made 4 or more turns earlier (pass rate at least 0.80, with the two-slot and no-update control items also at least 0.80);
> - keeps same-type facts and speakers apart in both mention orders (paired pass rate at least 0.80; chance is 0.25);
> - loops in at most 2% of turns, and no more often than Qwen2.5-0.5B-Instruct.
>
> In a blind pairwise test ({r} raters, 150 conversations, 3 ratings each), raters judged it as good as or better than Qwen2.5-0.5B-Instruct in {w}% of judgments (95% CI {lo} to {hi}), meeting the pre-registered 50%. To our knowledge it is the smallest model shown to pass this bar."

**Rules for both levels**

*Thresholds*
- The thresholds (0.80, 2%, 50%, -3, -5) are proposals. They become final when the pre-registration is pushed (Oct 11-14) and never change after that.
- **Re-anchor rule**, applied mechanically before the lock. LFM2-2.6B is the reference model that should clear the bar. If it scores below 0.80 on dev for a Level A family, first check that family and its grader. If the score is still below 0.80, that family's threshold becomes LFM2-2.6B's dev rate rounded down to 0.05, and never goes below 0.60. If LFM2-2.6B is below 0.60, the threshold stays at 0.60 and the pre-registration says no tested model reached it.

*Seeds*
- Every size that could carry the claim gets 3 training seeds. It gets 5 if E3 or P-026 show seed lock-in, meaning seed pass rates on a pass-or-fail family spread by more than 0.3 at 30M.
- Level A is claimed only at a size with at least 3 seeds. Any curve point with fewer seeds states its count.

*What counts as the model*
- A note the model writes and reads itself counts as the model.
- A harness-maintained state card is reported only as "Planck-{N}M + state card (scaffold)".

*After the first sealed scoring*
- Any later recipe change is labeled post hoc.

*Odds* (judgment, from REPORT s1.4)
- Level R: 40-60%.
- Level A: 10-25%, or about 30-40% if the repaired `ft_test.py` (P-001) passes with its controls.

---

## 2. Phases

| phase | dates | main machine | what it decides |
|---|---|---|---|
| 0 Measurement, operations and repo | Sep 24 - Oct 14 | 5070 (baseline generations, CUDA probes); Mac (E001, likelihood scoring, harness) | the test, the baselines, whether updating is trainable, unattended running |
| 1 Teacher pilot and tokenizer | Oct 1 - Oct 18 | 5070 (pilot); CPU (tokenizer) | which teacher; tokenizer v1 |
| 2 Data pipeline and chat pool | Oct 9 - Nov 22 | 5070 | whether skill-dense data teaches the skills; the 300M pool |
| 3 Screens and confirmations | Oct 19 - Dec 3 (freeze Dec 4; Dec 14 for 150M) | 5070 | the recipe |
| 4 The curve | Dec 4 - Feb 21, 2027 (150M seeds to about Mar 2) | 5070 (10/30/60M); Titans (150M) | Level R, then Level A with the human test |
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
- **Sealed split size** is set by the sizing rule below: likely 1,000-1,200 conversations, minimum 600, cap 1,500.

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
- **LFM2.5-230M or Falcon-90M passes Level R.** Level R is then new only below that model's size, so Planck's claim moves to the curve points under it.

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
- Allowed:
  - OASST2 (human, Apache-2.0);
  - FineWeb-Edu (human web, ODC-BY);
  - Wikipedia (CC-BY-SA; release rule in section 8);
  - Cosmopedia v2 (Mixtral-written, Apache-2.0);
  - public-domain name lists;
  - our own teacher renders;
  - paraphrases from a small Apache-2.0 model (P-162).

*Tokenizer family v0 (needs the D8 confirmation)*
- One byte-level BPE with byte fallback, digits and punctuation split, a minimal role-token template, and reserved `<note>`, `<think>`, `<lookup>` and `<result>` tokens.
- Trained on the pilot output, OASST2 (with the OOD-H threads removed) and a FineWeb-Edu slice, then truncated to 2k/4k/8k/16k/32k (P-098).
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
- General text for 10-60M is streamed and tokenized on the PC, so raw parquet is never stored: a readability-filtered FineWeb-Edu slice, OASST2 and a Wikipedia slice, 5-15B unique tokens, by Nov 1.

*Chat pool (reviews 1 and 17)*
- Target: 300M accepted unique tokens, all generated on the 5070, at least 100M by Nov 8 and all 300M by Nov 22.
- Sizing: every model reads the pool at most 25 times, and each size's chat share follows from that (table in section 5).
- Planning rate: 250-500 accepted tok/s in Planck tokens, so 300M takes 170-330 h (est.).

*Rate check*
- After the first full week of bulk generation (about Oct 27 - Nov 1), the measured rate sets the final pool size.
- If it projects under 300M by Nov 22, decide by Nov 8 to shrink the pool, lower the 60M and 150M chat shares under the pass cap, and lean on P-162 paraphrases.

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
- P-053: skill-dense vs small talk (from 3a).
- P-052: chat share (25/50/75%) and placement, with the 0.1x-LR check P-139. At curve scale the chosen share is capped by the pass rule.
- P-009 from scratch: 0% vs 2-3% correction dialogues.
- P-051: density sweep, reduced to 0/2/8% at 30M.
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
- E8/E9/E10 (P-152, P-151, P-150) run at 5M, and only if E4 finds signal at 5M.

**3e. Post-training at 30M (Nov 16 - Nov 28)**
- Whole-conversation SFT is the baseline.
- P-014: one epoch of multi-turn DPO.
- P-066 and P-153: anti-loop. P-067: own-history SFT.
- P-068: an on-policy distillation (OPD) pilot on public SmolLM2 models, if there is time.

**3f. Combined recipe (Nov 26 - Dec 3; must-run; review 11)**
- All adopted winners together vs the baseline at 30M, 3 seeds, 20-30 h.
- If the combination fails its pre-registered rule, drop changes one at a time, weakest evidence first. These are leave-one-out arms, 1 seed each, 10-15 h.
- Freeze the best set that survives.

**Must-run list** (19): E1-E4, P-004, P-158, P-053, P-052, P-009, P-007, P-008, the four-arm test, P-020, P-021, P-022 or P-148 (whichever the toy favors), P-014, P-162, the combined-recipe test. Everything else runs if time allows. A bet that never runs stays in the ledger as "not run", never as "failed".

**Machine.** 5070 for everything, with 2-3 small jobs at once where E1 shows it pays. The Mac scores small checkpoints on RC-12 dev (batched, minutes each) and runs 5M screens only if the 5070 is down.

**Compute (est.).**
- 5070 400-640 h:
  - 5M screens 30-60 h;
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
- **The note wins by 5 points or more.** It joins the recipe. If N at 10M matches B at 30M, add a 5M curve point.
- **E4 finds no multi-turn signal at 5M.** All screening moves to 20M, and only the must-run list runs.
- **Nov 22 checkpoint.** If Phase 3 is behind, cut to the must-run list.

### Phase 4: the curve (Dec 4 - Feb 21, 2027)

**Purpose.** Train the frozen recipe at each size, measure Level R on sealed RC-12 and OOD-H, then Level A, including the blind human test.

**5070 runs, from the Dec 4 freeze (review 4)**

In priority order:

| order | model | tokens | seeds | 5070 h (est.) | done by (est.) |
|---|---|---|---|---|---|
| 1 | Planck-10M | 5B | 3 | 30-63 | Dec 5-7 |
| 2 | Planck-30M seed 1 | 15B | 1 | 56-99 | Dec 8-11 |
| 3 | Planck-60M seed 1 | 30B | 1 | 154-245 | Dec 14-21 |
| 4 | Planck-30M seeds 2-3 | 15B | 2 | 112-198 | Dec 19-29 |
| 5 | Planck-60M seeds 2-3 | 30B | 2 | 308-490 | Jan 1-19 |
| 6 | generic-recipe control, 30M (same size and tokens, baseline data only) | 15B | 1 | 56-99 | Jan 3-23 |
| | post-training and dev eval, interleaved | | | 20-40 | |
| | **total** | | | **736-1,234** | **Jan 4-24** |

All three 60M seeds stay on the 5070, so no size mixes bf16 and fp16 numerics.

**Re-plan rule**, applied after `bench_micro` and again after the Planck-harness re-bench:
- If the 5070 measures below about 60% of the estimate, the generic control drops to 5B tokens.
- 60M seed 3 then moves to the Titans in January, labeled as the fp16 platform.
- If the load is still over, Phase 3 cuts to the must-run list.

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

*Seeds*
- Seed 1 starts about Dec 17 and finishes Dec 31 - Jan 11 (est.).
- Seeds 2 and 3 queue automatically after it, with no new Max session. At worst they finish about Feb 5 and Mar 2.
- Max can reclaim the Titans for Ultra at any time. The 150M claim then states its seed count.

*Setup*
- fp16 with loss scaling, torch 2.7.1 cu118. The Titans need no driver upgrade for training.
- The 50-80B-token general-text slice is built beforehand on the box's idle CPUs at nice 19, with Ultra's thermal log watched. Max kicks it off remotely on Nov 21-22. If Ultra's raw FineWeb-Edu text is still on the box, it is re-tokenized instead of downloaded again.
- Heartbeats every 30 minutes go to the private status repo through a repo-scoped deploy key.

**Sealed scoring at milestones**
- Each curve point is scored once its seeds finish.
- It is scored with and without lookup on the fictional knowledge base (P-094), in the state-card condition labeled as a scaffold (P-005), and on OOD-H.

**Level R table.** Every curve point vs Qwen2.5-0.5B-Instruct with the paired CI, the OOD-H column, and the public panel.

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

**Ledger ids.** P-055 (the curve), P-094, P-005, and the baseline blind-human-test item.

**Compute (est.).**
- 5070 740-1,240 h.
- Titans 3,900-7,200 card-hours: 1,300-2,400 per 150M seed at 75B, plus the smoke test.
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
- *Level R* is claimed at the smallest size whose sealed pooled-seed lower bound is -3 or better, with 3 seeds at that size, under the OOD-H wording rule.
- *Level A* is claimed only if every generative criterion passes on sealed, pooled over at least 3 seeds, and the human win-or-tie point estimate is at least 50%.
- *Level A decision, Jan 31* (review 20). It uses the seeds finished by then. A Level A claim at 150M becomes final only when seeds 2-3 confirm the generative criteria.
- *60M passes Level A.* 150M still runs, because the curve needs it, but the headline is 60M, and the 5070 adds a 20M point in Jan-Feb.
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
| RTX 5070 (PC, WSL2) | all training at 5M and up; scoring every model's generations over RC-12; the teacher pilot; all bulk generation; tokenizing general text for 10-60M; the 10/30/60M curve | nothing that fits in 12 GB is excluded |
| M5 Mac | harness and generator development; likelihood tiers; tokenizer CPU work; smoke tests; scoring small Planck checkpoints on RC-12 dev; 5M screens only when the 5070 is down | bulk generation; anything that needs the laptop at home and awake for days |
| Titans (Lambda, 4 edge cards) | 150M seeds (Dec 17 - early Mar); smoke test Dec 7-10; 150M general-text prep on idle CPUs from late Nov | anything before Ultra finishes |

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
- **PC needs.** About 150 GB free (est.): teachers 20 GB, baselines 17 GB, 10-60M general text 10-30 GB, chat pool under 5 GB, checkpoints 30-60 GB. Checked on the first SSH.

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
- **Chat share rule.** Chat share = min(the P-052 winner, the share at which the model reads the 300M pool 25 times). The rule, not a fixed number, is the recipe at every size.

  | size | tokens | chat share | chat token-passes | passes over the 300M pool |
  |---|---|---|---|---|
  | 10M | 5B | 25% | 1.25B | about 4 |
  | 30M | 15B | 25% | 3.75B | about 12.5 |
  | 60M | 30B | 25% | 7.5B | 25 (at the cap) |
  | 150M | 75B (100B) | 10% (7.5%) | 7.5B | 25 (at the cap) |

  - If P-052 picks 50%, 10M reads the pool about 8 times and 30M 25 times.
  - If the pool ends smaller, the 60M and 150M shares drop to stay under the cap.
  - If P-162 wins, paraphrases count as new text and raise the 150M share.
- **Chat content**, in the same template as SFT, from step 0:
  - Skill-dense skeleton renders: S1-S9, 2-4 events per conversation, answers mentioned second as often as first, nonce and real slot values, stale traps and twins.
  - Social basics and OASST2, with the OOD-H threads removed.
  - Grounded lookup dialogues on a fictional knowledge base: 0-5% of chat at 10-30M, 10-20% at 60-150M. BM25 lookup with zero parameters.
- **General text** fills the rest. It is knowledge-light, with 0% code by default (P-060 tests 15%):
  - readability-filtered FineWeb-Edu;
  - a small Wikipedia slice;
  - optionally Cosmopedia v2.
  - Note (review 27): FineWeb-Edu was selected for knowledge-dense educational text, which is in tension with "knowledge-light". The readability filter keeps it simple. P-083 (entity anonymization) and P-161 (source tags) test the mitigation.
- **Provenance.** Every training string follows D8 and the Phase 1 provenance rule.
- **Hygiene.** Dedup, 8-gram caps, 13-gram decontamination against RC-12, OOD-H and the human-test scripts, and a mutation-tested checker. The chat pool repeats on a spaced schedule, outside the memorization window.

**Tokens per size**

| size | tokens | tok/param |
|---|---|---|
| 10M | 5B | ~500 |
| 30M | 15B | ~500 |
| 60M | 30B | ~500 |
| 150M | 75B default, 100B if the Titan smoke-test rate allows | 500-670 |

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
| 4 | Dec 4 - Feb 21 | 20-50 | 740-1,240 | 3,900-7,200 | ~5 |
| 5 | Feb 1 - Apr 5 | 10-20 | 50-100 | (last 150M seeds, counted in 4) | 10-15 |
| **total** | | **~70-175** | **~1,400-2,400** | **3,900-7,200** | **~19-27** |

**Capacity checks**
- 5070: about 580-990 h of roughly 1,270 from Oct 12 to Dec 3, and 740-1,240 h of roughly 1,200 from Dec 4 to Jan 24.
- Titans: about 8,450 card-hours free Dec 17 - Mar 15. The plan uses 46-85% of that.

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
| RC-12 sealed | 600-1,500 conversations by the sizing rule (likely 1,000-1,200) | Claude-written banks, separate session, encrypted | Oct 11-14 |
| OOD-H | 150 held-out OASST2 threads + 50-60 rater-written scripts | human-written user turns | lock / Jan 17 |
| Human-test scripts | 150 | Claude-written (D8 gold set) | Jan 10 |
| Teacher pilot | 600 conversations, ~0.2M tokens | 3 Apache-2.0 teachers on the 5070 | Oct 8 |
| Chat pilot | 20-30M accepted tokens | pilot winner, 5070 | ~Oct 18 |
| Chat pool, all sizes | 300M accepted unique (about 500M generated at ~60% yield, est.; at least 100M by Nov 8) | teacher on the 5070 | Nov 22 |
| General text, 10-60M | 5-15B unique | FineWeb-Edu slice, Wikipedia, OASST2, streamed on the PC | Nov 1 |
| General text, 150M | 50-80B unique | FineWeb-Edu on the Lambda box's CPUs | Dec 14 |

**Training token-passes**
- Phase 3: about 100B, across 50-80 runs at 5M and 40-60 at 20-30M.
- Curve:
  - 15B (10M, 3 seeds)
  - 60B (30M, 3 seeds plus the control)
  - 90B (60M, 3 seeds)
  - 225-300B (150M, 3 seeds)

**Generation-rate note (review 21).** 170-330 h of generation at 250-500 accepted tok/s yields 150-600M tokens. 300M is the midpoint, so the pool target and the hours now agree. Rates are counted in Planck's tokenizer.

---

## 7. Risks and kill criteria

| risk | early sign | response or kill criterion |
|---|---|---|
| **The strict bar is unreachable at 150M** | P-001 flat; by Nov 22 no 30M arm lifts correction margins at 4+ turns beyond seed noise | Level A is attempted only at 60M and 150M with the best-margin arm. If nothing passes by Jan 31, no Level A claim. **Fallback claims, in order:** (1) Level R at the smallest passing curve size, plus per-skill size floors, which are new because nobody has measured multi-turn sub-skills below about 90M; (2) the note arm, which counts as the model, if it passes where the plain model does not; (3) "Planck-{N}M + state card (scaffold)", clearly labeled; (4) a negative result: strict corrections not reached at 150M or below with these interventions, and where each one failed. |
| Level R also fails | Sealed lower bound below -3 at 150M | The claim becomes per-skill floors on the curve plus the negative result (paper path in section 8). Odds: 40-60% (judgment). |
| Someone already clears the bar | LFM2.5-230M or Falcon-90M passes Level R (or A) on RC-12 in Phase 0 | "First" moves to sizes below that model. The curve points under 90M/230M carry the claim, or the claim becomes capability per parameter. |
| The test is invalid | Mutants survive; LFM2-2.6B fails a Level A family; families sit at floor or ceiling | No lock until fixed. The re-anchor rule applies before the lock, with a 0.60 floor, never after. |
| The model learns RC-12's format, not the skill | Sealed scores well above OOD-H | OOD-H is pre-registered, with the -5 wording rule. Sealed banks are written before the generator and kept encrypted. 13-gram decontamination against RC-12, OOD-H and the human scripts. |
| The data hypothesis fails (as ufakzeka-1 found at 151M) | P-012 and P-053 both under +5, with the CI excluding +5 | Stop at the pilot. Phase 3 weight moves to the note, the recency and delta layers, and the curriculum. Middle zone: continue to 100M and re-test at 20M. |
| Generation is slower than planned | Measured accepted tok/s after the first bulk week | Decide by Nov 8: shrink the pool, lower the 60M/150M shares under the pass cap, use the P-162 paraphraser. The Mac is not a fallback. |
| No multi-turn signal in 5M proxies | E4 | Screen at 20M; run only the must-run list. |
| Seed lock-in lottery on pass-or-fail skills | E3 and P-026 lock-in rates vary | 5 seeds at 10M and 30M on the 5070 (cheap). 60M and 150M stay at 3, with lock-in rates reported. |
| 5070 unavailable (video-editor or IseMedia use, reboots, WSL issues) | Missed heartbeats; GPU throttling | Boot-time runner, update windows, resumable jobs. If it is out for more than 2 weeks: the Mac runs 5M screens and 20M at 0.4B on the must-run list, and 60M seeds move to the Titans in January, labeled fp16. |
| Titans late (Ultra overrun, thermal pauses, break access) | Ultra ETA past Dec 14; smoke test fails Dec 7-10 | The smoke test exists so the first Titan failure is not the launch. 150M at 75B. If not started by Jan 10: 150M at 50B on the 5070 after the 60M seeds (about 18-24 days, est.), or pausing Ultra (Max's call). |
| Another Mac kernel panic | Swap above 2 GB, or a panic log | The runner halts; no retries until the cause is found. The Mac now carries little GPU work. |
| Disk fills | Free space under 25 GB | Disk guards stop jobs; big data lives on the PC and Lambda. |
| Unattended plumbing fails | Two missed heartbeats | STATUS flag; checklist line if Max is needed; queues hold 1-3 days of work. |
| Claude usage caps (weeks 1-3 are mostly Claude-written code) | Weekly cap reached before the lock | Order work by the critical path (RC-12 dev and graders, prereg, generator, harness) and keep GPU queues full. Cut to the must-run list; never cut the pre-registration. |
| Contamination or a D8 slip | Decontamination hits; audit gaps | Provenance column, teacher-written templates, 13-gram checks, audit table in the repo. |
| Personal information in public files | Scrub check before each push | Trimmed public PLAN.md; `private/` gitignored; no names of Max's school, professors, colleges or campus access in the repo. |
| Max's time runs over | Missed Max tasks | Claude pauses Max-dependent items, never the compute. No Max tasks Oct 8-17, Oct 26 - Nov 1, Nov 9-15, or during finals. |
| Too many bets | Phase 3 behind at Nov 22 | Freeze on the must-run list plus the combined-recipe test; unrun bets stay "not run". |

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
- **4 and 5 (modified):** all three 60M seeds stay on the 5070 by default, so one size never mixes bf16 and fp16. A 60M seed moves to the Titans only under the bench re-plan rule, and is then labeled. The Titans take 150M seeds 2-3 as the review asked.
- **8 (modified):** the rater-written half of OOD-H is collected in January, not before the lock. Only its protocol is pre-registered, which is enough because those scripts cannot have leaked into training.
- **15 (modified):** update windows move to Oct 24-25, Nov 28-29, Jan 2-3 and Feb 6-7. Oct 26 - Nov 1 is a no-Max week, and the first 5-week pause ends Oct 30.
- **16 and 26 (modified):** the Titan smoke test, the data-prep kickoff and the launch all run remotely (Nov 21-22, Dec 7-10, Dec 17-18), so no campus trip is needed.
- **19 (left open):** whether to pay raters is Max's call. The plan budgets the 25-40 rater-hours either way.
- **23 (modified):** torch is pinned to one exact version that passes both leak self-tests (2.14.0 if released), not to 2.14.0 unconditionally. The loop track only establishes the fix in 2.13.0.
- **28 (superseded):** E001 has started. Its logs appear from 00:48, and at 00:55 a Falcon-90M dry chat run on CPU was in progress. STATUS records the actual times.
