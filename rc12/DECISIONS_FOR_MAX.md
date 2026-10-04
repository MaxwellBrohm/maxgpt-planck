# RC-12: Max's decisions before the lock (Oct 11-14)

Written 2026-10-02 by the STEP 10e check (rc12/notes.txt). I have applied none of these. Each item gives the question, the options, the evidence and a recommendation. All evidence is from dev (13 public models, template render, 3 sampling seeds) unless an item says otherwise.

**The check itself.** Every number behind the panel and the rules reproduces:
- All 3,640 panel cells and all 117 headroom values, recomputed from the raw rows with separate code: 0 mismatches.
- All 66,560 stored conversations, regraded with the current graders: 0 differences from the stored grades.
- The rules were applied as the prereg text says. The sizing simulation passed its known-answer test and a new case I added.
- One process slip: a STEP 10d search command opened files under sealed/. Its output was filtered; nothing from those files was shown or written.

## Your question: has anyone already done this?
No. The answer is in research/NOVELTY_2026-10.md, written today by a separate search (five search agents plus verifiers, everything up to Oct 2). Nobody has shown a model under 150M passing a strict multi-turn test with corrections, rule persistence and binding, or published a minimum-size curve for those skills.
- Closest: SmallTalkLLM asks the same question with a looser bar and has no results yet; Loom and BananaMind-2 are tiny chat models with weak multi-turn numbers.
- So Planck's novelty is the strict bar and the measured floor, not the idea. The lock's dated push is the priority claim, so keep it on schedule. Items 14 and 15 come from that report.

## A. Results that need a pick

**1. LOOKUP sits at the floor. Rebuild it, or drop it from the composite?**
- Evidence: every core model scores 0.000 to 0.014, on each seed and on greedy. The floor is the X probe (a key only the other table holds): models answer with the other table's value (LFM2-2.6B 83% of the time). Lenient scoring does not lift it.
- Options: (a) drop it, reason written in s11, R becomes the mean of 9 families; (b) rebuild with easier cells: a new dev hash, the full check order and about 104 GPU reruns, with no evidence it clears 0.05.
- Recommend (a). A family where everyone scores about 0 makes non-inferiority easier, which s11 exists to stop. RECALL still tests abstention. The sealed size stays 600 either way.

**2. What standard should the rerun check (verify_dev_runs R1/R2) use, including for Doge?**
- Evidence: the same prompts replayed in a different batch size give identical output on turn 1 only 40% of the time on vLLM (915 of 2,304) and 25% on Doge (47 of 192). An exact-replay check fails for every engine, not just Doge.
- Options: (a) every engine passes if each first difference is a near-tie (at most 0.5 nats, the parity rule's NEAR_TIE); (b) NEAR_TIE for Doge only, exact for vLLM (predicted to fail on drift alone); (c) exact replay of the stored batches (GPU hours); (d) report the identical share only.
- Recommend (a), then `--rerun 5` once per engine family under gpu.lock.

**3. Fix the change-phrase gap in the grader?** (found by the re-anchor audit)
- Evidence: a right answer that recaps the change ("Initially scheduled for Tuesday ... moved to Thursday") fails, though s6 allows a stale value in a change phrase. It decided 1 of 120 audited units and cannot move any threshold.
- Options: (a) add the "initially / previous / prior" frames to the stale-value exemption, re-run the grader checks, regrade the 208 stored runs on CPU, re-run s11; (b) leave it, recorded.
- Recommend (a). It contradicts the prereg's own grader text and needs no GPU.

**4. How big should the sealed split be?**
- Evidence: the s9 rule gives 600 conversations, power 0.91 to 0.93 (0.82 to 0.86 with the size-matched baselines, 0.86 with LOOKUP dropped). The rule cannot see how much Planck's R moves between training seeds. Size needed for 80% power by that seed spread:

  | Seed spread of Planck's R | up to ~1 point | 1.5 points | 2 points | over 2 points |
  |---|---|---|---|---|
  | Size needed | 600 | 700 | about 1,300 | no allowed size |

  E3 measures that seed spread only after the lock.
- Options: (a) 600 as the rule gives, with this table published beside it; (b) a recorded deviation sized for an assumed spread (700 for 1.5 points, about 1,300 for 2); (c) the 1,500 cap.
- Recommend (a). Against a large seed spread, PLAN's 5-seed trigger is what protects, not more conversations.
- Also confirm "the two baselines nearest to it": I used nearest on dev R (SmolLM2-360M, gemma-3-270m); nearest in size would be Qwen3-0.6B and SmolLM2-360M. Both give 600.

**5. Keep the T0 gate at 0.90?** (still a proposal)
- Evidence: the best model, LFM2-2.6B, scores 0.51 (0.76 lenient). Turn 1 asks for a riddle and models stay in the riddle frame; some failures name the right value but fail the guess, echo or voice checks. The re-anchor rule does not cover T0.
- Options: (a) first audit T0's items and grader as the Level A keys were audited (CPU only), then decide; (b) keep 0.90 as is; (c) lower it now.
- Recommend (a), before the lock. If the riddle frame is the cause, fix the items (a dev rebuild), not the bar.

**6. Which render does the PERSIST base-rate rule read?** (new from this check)
- Evidence: on the template render (as applied) no rule is above 0.30, so nothing drops. On the plain render one_sentence is above 0.30 for 4 core models and would drop: LFM2.5-350M 0.60, Qwen3.5-0.8B 0.40, Doge 0.39, LFM2.5-230M 0.37. s11 names the template render in its first sentence but not in the PERSIST sentence.
- Options: (a) template, written into the PERSIST sentence; (b) plain, or either render, so one_sentence drops.
- Recommend (a). PERSIST is scored on the template render, and plain is diagnostic everywhere else.

**7. How should s12's baseline clause be read?** ("If LFM2.5-230M or Falcon-90M passes Level R, Planck's claim moves below its size.")
- Evidence: on dev, released models from 230M up already meet the confidence-interval part against Qwen2.5 (LFM2.5-230M lower bound +0.79). None meets T0 or has 3 training seeds.
- Options: (a) the clause checks the CI part only; (b) full Level R, so it never fires while baselines' T0 is at or below 0.51; (c) also widen it as the novelty report suggests: a claim at size N stands only if no public panel model at or below N passes the same bar.
- Recommend (a) plus (c): the conservative reading, and it covers the new sub-30M models (item 14).

**8. Fix gemma-3-270m-it's extra stop id (`<eos>`, id 1)?**
- Evidence: gemma fails one verifier config check, but no reply stopped on that id: a stop there would have crashed the run, all 16 runs finished, and the stored stops are eos, cap and role only.
- Options: (a) leave it, with the note; (b) add `<eos>` to the HF stop list, redo gemma's parity check and its 16 runs on the GPU.
- Recommend (a). Gemma is an extra, and the check changed no reply.

**9. Revisit strict cases (a)/(b)?** (OD7: only if strict-only failures are common.)
- Evidence: 5 of the strongest model's 120 audited units were true strict-only failures. Even the upper bound adds at most 0.18 to any key and leaves all four below 0.60.
- Recommend no change: strict scoring stays primary, lenient twins reported beside.

## B. OD6 follow-ups still open (s17)

**10. F3, strict case (d): should a right answer that equals an earlier reply still count as a loop?**
- Evidence (this check): Qwen2.5's 62 such probes are one refusal ("I'm sorry, but I can't assist with that request.") repeated on LOOP turns, a real loop under every option. Every other model has at most 18. Option (ii) would change 1 probe across all 13 models, option (iii) at most 29 (mostly PERSIST). LFM2-2.6B has none.
- Options: (i) keep strict; (ii) exempt equality with an earlier reply to a statement turn; (iii) exempt equality with the reply to the probe's source turns.
- Recommend (i). It barely fires on real models, and the training pipeline can avoid teaching Planck terse restatements as acknowledgements.

**11. Should PERSIST P turns that reuse a statement filler (46 dev turns) stay "asking"?**
- Evidence (this check): over 13 models and 3 seeds there are 1,794 such turns. 89 replies equal an earlier reply; only 2 would be exempt if the turns were marked non-asking.
- Options: (a) leave them as asking turns, recorded; (b) mark them non-asking (a dev rebuild, a new hash, the full check order).
- Recommend (a): the change would move 2 replies in 1,794.

**12. Report near-duplicates?** (A parrot that changes one word escapes the loop rule.)
- Evidence (this check, a scratch measure): replies of 12 words or fewer, one word away from an earlier reply and not already a LOOP, are 0.000 to 0.007 of replies per model.
- Options: (a) report a near-duplicate rate beside the loop rate, never gated; (b) leave the rule and say so in the file.
- Recommend (a): CPU only (one function with fixtures and mutants), it cannot hurt a claim, and it is the only way a near-duplicate parrot shows up.

## C. Lock logistics that need you

**13. Review OOD-H Part 1.** Its 150 threads (hashes in oodh/HASHES.txt) are marked LOCK CANDIDATE and are fixed "only after Max's review". Recommend reviewing it this week.

**14. Add sub-30M public chat models to the panel as reported extras?**
- Candidates (novelty report): BananaMind-2 Nano and Mini, Vertex-0.6-15M, Loom-Spark-3.2, Swen-28M, Veyra2 (5M to 30M), cRia-75M.
- s12 is fixed at the lock, so they go in before it. Each takes minutes on the 5070; engine support is unchecked.
- RxT (non-commercial licence, click-through gate) and BananaMind's gated benchmark need your OK to accept their terms.
- Recommend adding the Apache and MIT models, and skipping RxT unless you accept its licence.

**15. Change the public wording?** Claim the first measured floor for strict multi-turn state skills, not the first tiny chat model, and add one line to the claim text that RC-12 scores the bare model with no external state. Recommend yes, before anything goes public.

**16. Confirm the remaining proposal numbers:** the -3 margin, T0 at 0.90 (item 5), the PERSIST 0.30 bar, zero shared word 5-grams between sealed and dev, and the S6 decontamination list. Recommend keeping all of them, except as item 5 decides.

## D. Mechanical once you pick (no choice involved)
- score.py BARS for the four Level A keys: 0.80 to 0.60 (the s10 result), with its mutants.
- After any regrade or rebuild: re-run dev_s11.py and the check order. OWN passes headroom only through LFM2-2.6B's 0.076, so watch it.
- If LOOKUP drops: s9's family list, score.py COMPOSITE and their checks.
- s16 still says "Dev baselines: none scored": update it in the final pass (s19 item 8).
