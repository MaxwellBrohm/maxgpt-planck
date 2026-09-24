# Adversarial review of the synthesis draft

Reviewed 2026-09-23 (evening). No model was loaded, run or trained for this review. Work done: read the draft, the three diagnoses, the probe audit and fact-check files, `capacity_probe/` code and saved results (JSON only), Max's A/B README and SFT packer (read-only); targeted web and arXiv searches; plain arithmetic.

**Process note: the draft file does not exist.** The harness blocked the synthesizer's write to `research/REPORT.draft.md` ("Subagents should return findings as text"). The full draft survives only as the synthesizer's returned text in the workflow journal (`(private workflow logs)`, entry 72). I extracted it to `(scratch file)` (512 lines) and reviewed that. I did not write REPORT.draft.md. The final editor has to build from that text.

**Live hazard, seen at 18:06:** LM Studio has `gemma-4-12B-it-QAT` loaded right now (llama-server PID 11931, about 8.9 GB resident, `--load-mode mmap+mlock`, 35.6% of RAM). I did not touch it. Nothing that trains or loads a model should start on this Mac until it is unloaded.

## Bottom line

The draft is careful with numbers: I spot-checked about 35 of its figures against the lanes and fact-checks, and every one traced. Its refuted-claims list is correct. The problems are elsewhere:
1. The experiment it calls decisive (`ft_test.py`) is neither safe as specified nor decisive as designed.
2. Its top-ranked limit (updating after a correction) is stated more strongly than the data allow. For 6 of 8 models the evidence is confounded, and the verdict repeats a claim the probe audit corrected.
3. The premise verdict answers only one direction. By the draft's own evidence, the line for strict multi-turn chat sits above 0.6B, not at 400-600M.
4. Feasibility assumes the 5070 without giving the Mac-only schedule. One "free now" test needs checkpoints that live on the Lambda box.
5. The verdict takes about 5 minutes to read, not 2.

## Blockers

### B1. `ft_test.py` is not safe as run, and a pass would not be decisive

- **Memory.** These are my own rough estimates from the SmolLM2 configs. They assume an fp32 model, AdamW, and an MPS backward pass that stores the attention scores.
  - The script's default batch is 16, with sequences up to 640 tokens. For SmolLM2-135M that needs about 20 GB at 400 tokens and about 35 GB at 640 tokens.
  - The draft suggests batch 8, which still leaves SmolLM2-360M at about 20 to 33 GB.
  - Batch 4 for the 135M (about 6 to 10 GB) and batch 2 for the 360M (about 9 to 13 GB) fit.
  - The earlier crash is consistent with this. The log ends at weight loading, and the script prints its first line only after the first optimizer step.
  - The queue's `memory_pressure` guard only waits for free memory before a job starts. It does not cap peak use.
  - Fix: before any rerun,
    - use batch 4 (135M) or batch 2 (360M) with gradient accumulation to an effective 16;
    - cap length at 512 or turn on gradient checkpointing;
    - unload LM Studio;
    - set `PYTORCH_MPS_HIGH_WATERMARK_RATIO` so PyTorch fails with an out-of-memory error instead of oversubscribing;
    - do a 5-step dry run while watching memory.
- **A shortcut passes it.**
  - In the training generator (`gen`, kind "upd"), the gold answer is always the last weekday mentioned. Distractor turns contain no weekdays, and k=0 items have only one value.
  - The eval items (`items.py` U, `uprobe.py`) have the same property.
  - So "copy the most recent weekday" scores 100% on both.
  - The draft's pass rule ("at least 0.8 latest-wins at d10 on held-out wording = trainable") would count that shortcut as learned updating, and the draft then raises its strict-pass odds to 40-50% on that basis.
  - Fix:
    - add two-slot items (a second event's day is mentioned after the correction, and the question is about the first event) and no-update twins with a later distractor day, to both training and eval;
    - require the pass on those too.
  - The draft prescribes exactly these twins for lever 1 but not for the test that gates everything.
- **One seed.**
  - The script fixes `torch.manual_seed(0)`.
  - Boesch and Wee (Sep 2026) report interference solving as a seed "lock-in lottery": without a curriculum, 1 of 10 seeds locks in; with one, 7 of 10.
  - Their evidence is tiny synthetic recurrent and attention cells, not LMs, so it transfers only as a caution: a flat result on one seed is not evidence of a capacity limit.
  - Fix: 3 or more seeds, and report the lock-in rate.
  - Source: https://arxiv.org/abs/2609.16183 ; also in `followup/floor.verify.md`.
- **A threshold below the test's resolution.** "Closed-book knowledge down no more than 0.05" is judged on `khard`: 40 items, a two-way choice, standard error about 0.06. That test cannot resolve 0.05.

### B2. Limit #1 is overstated, and the verdict repeats a corrected claim

My tabulation of `capacity_probe/results.json` and `out/uprobe__*.jsonl` shows the following.

- **The main U items are lexically confounded.**
  - The question ("What day is my dentist appointment?") and the answer prefix ("Your dentist appointment is on") share the key "dentist appointment" only with the original statement. Every correction reads "Actually, the appointment got moved to X".
  - At distance 0, the correction is the turn right before the question. Even there, 4 of 8 sub-1B models already fail:

    | model | latest-wins accuracy at distance 0, 1 to 3 corrections |
    |---|---|
    | SmolLM2-135M-Instruct | 0.06-0.16 |
    | LFM2.5-350M | 0.00-0.03 |
    | Qwen3-0.6B | 0.00-0.34 |
    | LFM2-350M | 0.19-0.56 |

  - For those models, the zeros at distances 4 and 10 cannot separate primacy from key-matching or from the off-template plain format.
  - "0.00-0.11 in all 8, no gradient from 135M to 0.6B" is therefore not a measured size result.
- **Only one control is clean, and it covers only SmolLM2-135M-I and 360M-I.**
  - `U_neutral` keeps a key confound: the question says "dentist", and only the original does. SmolLM2-360M picks the original in 32 of 32 items for U_neutral with 3 corrections, even at distance 0.
  - `U_same` is the clean control: 32 of 32 correct at distance 0, then 0 of 32 at distances 4 and 10.
  - So the clean primacy evidence covers one model family, one item family, and always puts the original fact in the first turn. The draft's own position control has not been run.
  - The draft's sentence "At distance 0 the 360M gets 0.97-1.00" is true only for U_same. For U_neutral it gets 0.66, 0.03 and 0.00 with 1, 2 and 3 corrections.
- **The generation probe contradicts the verdict.**
  - Each model got 3 correction items. In each, the correction comes 2 user turns before the question, with one exchange in between (`probe/battery.py`, CORR).
  - Qwen3-0.6B passes 2 of 3 greedy, and passes K_time in both sampled seeds (`lanes/probe.verify.md`, claim 9: "fails at every size" is overstated).
  - Yet the verdict says: "once a correction is 4 or more turns back, every model probed from 135M to 0.6B answers with the original value".
- **"4 or more turns back" is an artifact of the grid.** The battery tests only distances 0, 4 and 10. In generation, failures appear with one exchange in between; in the battery, some models fail at 0.
- **Fix.**
  - Restate the finding as: "In SmolLM2-135M and 360M, with identical wording, the first-stated value wins once the correction is 4 or more turns back. For the other models the battery is confounded. In generation, 7 of 8 models fail with one intervening exchange (3 items each; K_color, the lexically clean item, favors the stale value in 6 of 8), and Qwen3-0.6B passes 2 of 3. LFM2-2.6B passes even the confounded items."
  - Keep limit #1 at rank 1 only under the strict bar, labelled "evidence: SmolLM2 family plus 3 generation items per model".
  - Add to step 2: U_same on every model, the position control, and a U item that puts the key in the correction.

## Major

### M1. The premise verdict answers only one direction

- "Not real as a wall" is right for loose chat. The 350M class ties Qwen2.5-0.5B on the probe, and Falcon-90M is the best per parameter.
- Max's stated bar is strict multi-turn coherence. On that bar the draft's own data say no model at or below 0.6B passes, and only LFM2-2.6B does. Nothing between 0.6B and 2.6B was tested.
- So for his actual goal the premise is too generous, not too pessimistic.
- Say both in the first two lines of the verdict:
  - loose chat already works at 300-350M, and at 90M per parameter;
  - strict chat has not been shown below at least 0.6B (probe) or 1B (published).

### M2. The ranking mixes two different questions

- The draft ranks by "most likely to stop a 150M model on a knowledge-light multi-turn battery". Limit #1 also stops Qwen2.5-0.5B.
- Under Level R (non-inferior to Qwen2.5-0.5B), #1 cannot be what binds. The binding limits there are #3 policy, #2 binding, #4 tokens and #5 loops.
- Section 5.2 half-resolves this, but the verdict table does not.
- Give a two-column answer:
  - what separates 150M from 0.5B: #2, #3, #4, #8;
  - what separates every sub-1B model from strict chat: #1, and possibly #6.

### M3. The only sub-300M size gradient is confounded by tokens

- The SmolLM2 135M to 360M jump in binding (owner pair 0.62 to 0.94, perspective 0.09 to 0.91) is the draft's only same-recipe size gradient below 300M of body.
- The 360M saw 4T tokens and the 135M saw 2T (census table).
- Call it a size-plus-2x-tokens gradient, and weaken "the only size gradient below 300M of body" to match.

### M4. The two most relevant models are not on the run list

- LFM2.5-230M (163M body) is called "the closest shipped analogue to a 150M target" and "not probed by us". Step 2 does not add it.
- Falcon-H1-Tiny-90M is missing from the likelihood battery, and step 2 only schedules its chat-probe siblings.
- Running both through both probes is the direct premise test at the target body size. Add both to step 2.
- Also free: `HuggingFaceTB/smollm2-135M-SFT-Only` (verified on the HF API) isolates what DPO does to deflection and binding at 135M.

### M5. Missing prior art for the #1 limit and its levers

- **Boesch and Wee, "Anatomy of Associative Recall in Fixed-State Recurrences" (Sep 2026).** https://arxiv.org/abs/2609.16183
  - Finding: "Interference under sparse supervision, not capacity". A distance curriculum takes recall from 0.021 to 1.000.
  - At length 256, a shaped ramp reopens a boundary that a uniform curriculum cannot (4 of 5 seeds vs 0 of 9).
  - Scale: small synthetic cells, not LMs.
  - This bears directly on lever 1, which specifies "fact-to-question gaps sampled uniformly". The only evidence found says to use an accuracy-gated distance curriculum instead.
  - It is in `followup/floor.verify.md` but absent from the draft.
- **DZ-TiDPO, "Overcoming State Inertia" (Dec 2025).** https://arxiv.org/abs/2512.03704
  - Conflict-aware DPO plus a structural temporal attention bias, for dialogue "state inertia", tested on Multi-Session Chat and IC-Bench.
  - Only Phi-3.5-mini (3.8B) and Qwen2.5-7B [>=1B only]. The smaller model paid a visible stability cost.
  - This is prior art for lever 3 and for the "DPO with a stale-value rejected side" row. Those become tweaks of known work, not new ideas.
- **Selective Attention.** https://arxiv.org/abs/2410.02703
  - Parameter-free; lets later tokens stop attention to superseded ones. Gains at small scale on C4; not significant after Bonferroni correction at 1.2B (`lanes/arch.md`).
  - It belongs in lever 3 next to the Forgetting Transformer and stick-breaking.
- **"In-context superposition" (Apr 2026).** https://arxiv.org/abs/2604.09670
  - A 2-layer transformer trained on a working-memory task solves it perfectly. Trained LLMs show load-dependent interference with a recency bias, and a suppression intervention helps.
  - This supports "#1 is trainable". It also cautions against generalizing the primacy finding beyond the probe's single item family.
- **Entity tracking (Jun 2026).** https://arxiv.org/abs/2608.18083
  - Human-level likelihood tracking at Pythia-410M; generation stays below human level even at 13B-Instruct (`lanes/context.verify.md`).
  - The draft cites it but never uses it, although it is the best outside scale anchor for limit #2.

### M6. Feasibility needs a Mac-only plan

- **The 5070 is uncertain.** The brief says it is lent out; Max's Planck memory note says it is free as of this afternoon. The draft notes the conflict but relies on the 5070 for steps 3 and 4.
- **Mac-only arithmetic.** These use the draft's own Mac rates: about 2.7k tokens/s at 113M, and 10.6k to 11.8k at 18.5M (unverified).
  - At 30M and roughly 6k to 9k tokens/s, 1B tokens takes about 1.3 to 1.9 days.
  - At 60M and roughly 4k to 5k tokens/s, 2B tokens takes about 4.6 to 5.8 days.
  - The step-4 program (mixing sweep of 4 ratios x 3 seeds, plus state-line and gate A/Bs of 2 arms x 3 seeds) is about 18 to 24 serial runs.
  - That is about 1 to 4.5 months serial on the Mac, depending on model size and token count.
  - On a free 5070 at the draft's roughly 32k tokens/s for 150M (faster at 30-60M), each run takes hours and the whole program takes under 2 weeks.
  - The Mac also cannot generate teacher data and train at the same time.
  - The draft should state both schedules and how the plan changes with the 5070's status.
- **The A/B checkpoints are not local.** Max's 124M A/B ran on the "Lambda box" (README title). Only metrics files exist locally, and I found no A/B checkpoints on the Mac.
  - The "free now" test (step 2d, lever 3) needs Max to copy the checkpoints, if they were kept.
  - It is also likely null by construction. The runs saw 1.1B tokens, about 9 tokens per parameter. SmolLM2-135M base at 2T tokens is already at chance on two-hop and at 0.22 on owner pairs.
  - Free local alternative: `maxgpt-3/checkpoints/final.pt` and `final_sft.pt` (235M, Max's own lineage). Running them through the battery costs nothing.
- **Free dialogue data is omitted.**
  - `lanes/data.md` lists WildChat-4.8M (ODC-BY, 3.2M real conversations), SODA (about 300M tokens), UltraChat (about 2.15B tokens), TinyDialogues (MIT) and Everyday-Conversations.
  - The draft's only data-cost line is teacher generation at 20 to 50 Mac-days per 100M tokens, which overstates the bottleneck.
  - Lever 1's interference dialogues can mostly be produced by program, using a teacher only for surface paraphrase (the "script mode" already in Max's Planck notes).

### M7. The verdict is not readable in two minutes

- The evidence-status table plus section 1 run about 1,050 words, with a 12-row table. That is 4 to 5 minutes of dense reading.
- Terms appear before they are defined: "body", "Multi-IF", "tok/param", "pair @10", "uprobe".
- The synthesizer's own SUMMARY (about 400 words, clearer) never made it into the report text.
- Fix: put a box of 200 words or fewer first, with:
  - the two-sided premise answer;
  - the two-column limits answer (M2);
  - the one next action (a safe, repaired `ft_test` plus probes of LFM2.5-230M and Falcon-90M).
- Move the evidence-status table below the box.

## Minor

- **SFT packing line.** "~0.1% of supervised tokens see a clean history" is technically right but misleading.
  - The real defect is that 42.5% of supervised tokens lose the start of their own conversation.
  - The 57.4% that can see an unrelated earlier conversation is ordinary packing without cross-document masking.
  - Lead with the 42.5%. The simulation itself (`recipe_work/pack_sim.py` against `posttrain/sft_data.py`) checks out.
- **Measurement is not a model limit.** Move it out of the ranked limits into "prerequisites".
- **Body gap.** For the wall model itself, the body gap is 2.9 to 3.2x (358/122, 358/112). That is essentially the 3.3x gap in names. The "2.5-3.6x" range spans other models, so the sentence implies a difference that barely exists for Qwen2.5-0.5B.
- **Power.** `lanes/eval.verify.md` row 12 says 80% power at a -3 margin needs roughly double about 1,185 effective items, i.e. about 2,200 to 2,600. The draft says "about 2,000 conversations".
- **Knowledge "0.3x" is a ratio of ceilings.** At about 670 tokens per parameter on web-heavy data, Max's model sits below the 2 bits/param ceiling (`census.verify.md` #15), so the realistic ratio is lower.
- **D6 is presented as open, but Max has already decided.** His Planck note records that Planck gets its own tokenizer. That rules out logit distillation from Ultra or SmolLM2 for his own model. Lever 5's SmolLM2-1.7B to 135M on-policy distillation then applies only to the public-model pilot. Reconcile D6 with that decision.
- **"Active primacy intrusion"** is a label from a 1B-2.5T study (arXiv 2603.00270) and is used in the mechanism column without a [>=1B only] tag.
- **D1's ternary line needs its scale.** Spectra compares Float 190M with Tri 560M at 300B tokens.
- **Landscape table omissions.** It drops census entries that bear on the premise:
  - Doge-160M-Instruct (IFEval 16.8), a same-size generic recipe (`followup/floor.md`);
  - OpenELM-270M-Instruct, ERNIE-4.5-0.3B-PT, Hunyuan-0.5B-Instruct, MiniCPM4-0.5B;
  - MiniMind (26-64M, multi-turn SFT, claims only).
- **3 seeds may be too few** if lock-in is a lottery (M5). For binary lock-in outcomes, report the rate over 5 or more seeds.

## Checks requested

1. **Contradictions with fact-checks.**
   - The verdict re-asserts "every model from 135M to 0.6B" fails corrections, which `probe.verify.md` claim 9 corrected (B2). The draft also mischaracterizes U_neutral as confound-free.
   - Otherwise the numbers trace, and the refuted list matches the verify files.
2. **Evidence from 1B+ models presented as holding at 150M.** Mostly flagged correctly. Exceptions are minor: the "active primacy intrusion" label, and D1's ternary scale. DZ-TiDPO, if added, needs [>=1B only].
3. **Novelty and premise.**
   - Searches for 2025-2026 sub-200M multi-turn chat models and methods found nothing that passes a strict multi-turn test. Beyond the lanes, they turned up only the interference papers in M5.
   - The HF API confirms that the Falcon-H1-Tiny-90M siblings and the SmolLM2-135M intermediate checkpoints exist.
   - The premise verdict survives only in the two-sided form (M1).
   - Novelty claims for levers 1 and 2 hold as far as I could search. Lever 3 and multi-turn DPO are tweaks of known work (DZ-TiDPO, Selective Attention).
4. **Missing items.** See M4, M5 and M6.
5. **Is the limit question clearly answered?**
   - The report gives a ranked answer with hard/soft labels and tests.
   - But the ranking mixes two questions (M2), #1's evidence is narrower than stated (B2), and the "decisive" test is not decisive yet (B1).
   - The hard/soft labels are defensible: knowledge hard; policy soft; updating unknown.
6. **Feasibility.** See M6. Mac-only timelines are missing; one "free now" test needs Lambda files; ft_test needs a memory budget (B1).
7. **Readability and em dashes.** No em or en dashes found (a grep for U+2014 and U+2013 returned 0). The verdict is too long (M7).

## Sources

- Draft text: journal entry 72 (path above); extracted copy in the scratchpad `critic/draft_clean.md`
- Probe data: `capacity_probe/results.json`, `capacity_probe/out/uprobe__*.jsonl`, `capacity_probe/items.py`, `uprobe.py`, `ft_test.py`, `logs/ft__SmolLM2-135M-Instruct.log`, `capq_tail3.sh`, `probe/battery.py`
- Max's files (read-only):
  - `~/Documents/Projects/Max's AI Model/maxgpt-ultra/docs/ab_2026-09-21/README.md`
  - `~/Documents/Projects/Max's AI Model/maxgpt-ultra/posttrain/sft_data.py`
- Model configs:
  - https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/config.json
  - https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/config.json
- HF API:
  - https://huggingface.co/api/models?author=tiiuae&search=Tiny-90M
  - https://huggingface.co/api/models/HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints/refs
- Papers:
  - Boesch and Wee: https://arxiv.org/abs/2609.16183
  - DZ-TiDPO: https://arxiv.org/abs/2512.03704 (PDF read: Phi-3.5-mini 3.8B and Qwen2.5-7B)
  - Selective Attention: https://arxiv.org/abs/2410.02703
  - In-context superposition: https://arxiv.org/abs/2604.09670
  - Entity tracking: https://arxiv.org/abs/2608.18083
  - Dual-process interference: https://arxiv.org/abs/2603.00270
  - ufakzeka-1: https://arxiv.org/abs/2609.25081
  - JugnuLM: https://arxiv.org/abs/2609.14715
  - Micro language models: https://arxiv.org/abs/2604.19642
  - Hy-MultiTurn: https://arxiv.org/abs/2607.29196
- Gemma 4 has no sub-1B size (smallest is E2B): https://ai.google.dev/gemma/docs/core
- Memory and runtime figures in B1 and M6 are my arithmetic. They are estimates, not measurements.
