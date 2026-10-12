# E004 audit (independent, 2026-09-25)

Recomputed from the raw outputs by an auditor agent that loaded no model.

**E004 audit: FAIL confirmed, but the analyzer's explanation for it is wrong**

**Q1. Scores recomputed: CONFIRMED, with small corrections**
- I rebuilt the eval draw and matched all 640 items in each run (base, s1-s5) to the records by prompt hash, family, idx and gold: 0 mismatches. My recomputed `right` flags agree with the stored ones 640/640, and every cell of `tables.txt` reproduces to 2 decimals.
- Failing cells per seed (LIK/GEN):

| Seed | H1 | H2 | H5 |
|---|---|---|---|
| s1 | .672/.656 | .703/.703 | .703/.703 |
| s2 | .703/.719 | .797/.812 | .719/.703 |
| s3 | .703/.766 | .672/.781 | passes (.812/.828) |
| s4 | .672/.656 | .656/.656 | .750/.734 |
| s5 | .656/.656 | .750/.750 | .797/.812 |

- **Verdict: FAIL confirmed.** 0 of 5 seeds pass. H1 fails on every seed at both LIK and GEN, and H1/H2 are not allowed in PARTIAL-X.
- Corrections to the brief:
  - H5 fails on 4 of 5 seeds, not all 5: s3 passes it.
  - Two cells print as 0.80 but are 51/64 = 0.797 (s2 H2 LIK, s5 H5 LIK). The verdict correctly counts them as fails.
  - Some passes are thin: H3 s4 LIK and C_twoslot s1 LIK are both 0.8125. At n=64 the 95% interval is about ±0.10.
- LR pick checked against the dev files: 5e-5 min .594 / mean .743, 1.5e-4 min .781 / mean .854. Choosing 1.5e-4 was correct.

**Q2. H1/H2 diagnosis: answer (b), refined. (a) is REFUTED**
- **Structure.** Each family has 21 alias, 21 ellipsis and 22 pronoun latest corrections. d is fixed at 10.
- **By reference form** (LIK, s1 to s5):
  - H1: alias .19 .33 .33 .19 .10; ellipsis .81 .76 .76 .81 .86; pronoun 1.00 on every seed.
  - H2: alias .24 .52 .38 .33 .38; ellipsis .90 .90 .67 .67 .90; pronoun .95 on every seed.
- **Without the alias items** (n=43): H1 scores .88-.93 at both LIK and GEN; H2 scores .81-.93 LIK and .81-.95 GEN, on every seed.
- **What the model picks when wrong** (LIK, 5 seeds pooled; GEN shows the same pattern):
  - H1 alias, 81 wrong: 76 are A's statement just before the alias correction (46 of those are not echo lures), 3 are a lure that is not that previous statement, 2 are B's value.
  - H2 alias, 66 wrong: 63 previous statement (41 not a lure), 2 original lure, 1 B.
  - Ellipsis: H1 19 of 21 wrong are the previous statement, H2 18 of 20.
- **Items where the rules disagree.** On alias items where O5 (the E002 wording rule) and a "latest statement about A, alias ignored" tracker give different answers (70 item-seeds per family):
  - The model matches the tracker 53 (H1) and 41 (H2) times, matches O5 2 and 2 times, and gets the gold 13 and 26 times.
  - On the 215 non-alias item-seeds where O5 differs, the model gets the gold 194 (H1) and 190 (H2) times, and matches O5 only 13 and 11 times.
- **Which cheap rule the model agrees with.** Mean over seeds, shown as all items | wrong items only:
  - H1: O5 .12|.37, O4 .14|.45, X5 .52|.51, O6 .21|.66, T .78|.37; my no-alias tracker .84|.75; my adjacency tracker .87|.71.
  - H2: O5 .11|.38, T .78|.34, no-alias tracker .79|.69.
  - No O1-O8 or X1-X11 rule exceeds .52 agreement on H1/H2.
  - T (topic tracker) is the top rule on every U family, where it equals IDEAL except in H1/H2. On C_noupd and H5 the wrong picks agree with last mention (O2) .69 and .42, i.e. recency.
- **The wording lure has no effect after fine-tuning:**
  - Pronoun items score 1.00 even though the question frame-echoes a stale statement.
  - Non-alias accuracy is .90 whether or not the original is also echoed.
  - The C_noupd echo cells (question echoes B) score .97 and 1.00.
  - The untouched model does follow the lure (a lure value is its top pick on 42/64 H1 and 56/64 H2 items, which is why it scores below chance). Fine-tuned seeds pick a lure only 4-12 times per 64, almost always where the lure is also A's previous statement.
- **Distance:** no trend. Pronoun items score 1.00 in every bin, and H4 (d20) passes.
- **Diagnosis.** The model learned "latest statement attributable to the asked object": it attributes by name or head noun, or by adjacency (a pronoun or ellipsis directly after). It never binds the alias name to the object.
  - An alias correction scores .73 (H1) and .67 (H2) when it sits directly after A's previous statement.
  - Separated by a filler or by B, it scores .10-.30. One cell with n=1 scores .80; all these cells have n of 1-6.
  - It also scores higher when the template contains generic words ("the whole event", "everything moves to"): city alias .60 and .84, against .11-.16 for colour and weekday.
  - Training contains no names or aliases: 0 Mr./Mrs./Ms./Dr. in seed 1's 6,400 examples, consistent with the purity log. So this is an untrained reference form, not a stale-wording shortcut.
- **Secondary finding (c).** Ellipsis is weak when another object's statements follow (.70 in H1/H2). The same happens in families with no echo (H3 .53, H4 .70, H6 .70), so it is a general ellipsis weakness, not caused by the echo.

**Q3. H5 (three objects)**
- 78 of 320 item-seeds are wrong.
- 52 are another object's value. 49 of those come from statements after the asked object's latest, 38 of them that object's current value. 19 of the 52 are another object's pronoun or ellipsis correction credited to the asked object.
- 26 are the asked object's own stale value (17 an earlier correction, 9 the original).
- Accuracy falls to .67 when another object's indirect correction follows (.84 without). By the asked object's latest form: ellipsis .64 (.53 when an indirect correction follows), pronoun .91, head noun .89, never corrected .69.
- So H5 errors are recency spilling over from other objects' later statements, plus the ellipsis weakness. The asked object's position does not matter (.71-.79).

**Q4. Are the alias items fair? Yes, for a human reader**
- Each alias is defined in the dialogue itself ("with Ms. Petrov", "from Mr. Quintero", "run by Mrs. Ivanova"). No outside facts are needed, only the everyday inference that the named person runs that appointment, order or event.
- 9 of the 12 templates state a clear decision. 3 only state availability ("can only fit me in on", "says only X is in stock now", "can only supply X this season"), where a careful reader might hedge. Those do not explain the failures: "rescheduled me for" scores .07 (H1) and .13 (H2), and "is sending the X version after all" scores .13 and .00.
- Five items with the model's answers:
  1. H1#17: "Mark down Monday for the staff briefing with Ms. Petrov." / "The briefing is going ahead on Friday instead." / B: "I've got the guitar recital Tuesday afternoon." / "On second thought, Ms. Petrov rescheduled me for Sunday." Gold Sunday. LIK 0/5; all five seeds say Friday ("Friday, based on what you told me.").
  2. H2#18: "The math tutoring with Mr. Kowalski is tentatively penciled in for Monday." / "On second thought, they've put it on Wednesday now." / B / "Mr. Kowalski rescheduled me for Saturday." The question echoes the original. Gold Saturday. 0/5; all say Wednesday, the pronoun correction, not the echoed Monday.
  3. H2#60: "...housewarming with Mr. Quintero is expected to happen in March." / "we've moved it to April" / "let's go with May" / "Mr. Quintero can only take our group in December now." Gold December. 0/5; all say "May is the month."
  4. H1#12: "The porch swing from Mr. Quintero ... silver" / "Now the swing ... blue" / B "the desk lamp arrived today, and it's purple" / "Mr. Quintero is sending the green version after all." Gold green. 0/5; all say Blue.
  5. H1#27 (a success): "we've settled on Saturday for the guitar recital now." directly followed by "Ms. Sorensen can only fit me in on Monday now." Gold Monday. 4/5 answer Monday. Likewise H2#53, "Mrs. Ivanova says everything moves to Denver.", is right 5/5.
- Minor issue: alias items are not balanced by value type (H1: colour 7, weekday 7, city 4, month 3). Reference form is balanced only on its own, which contradicts the `items_plan.py` docstring. The effect on H1/H2 is under 0.01.

**Q5. Costs**
- **Closed-book knowledge (kbig, 441 items)**, paired bootstrap with 4,000 resamples and exact McNemar. Base accuracy is .848.

| Seed | Fine-tuned | Change | 95% CI | McNemar p |
|---|---|---|---|---|
| s1 | .753 | -.095 | -.129 to -.061 | 3e-8 |
| s2 | .771 | -.077 | -.111 to -.045 | 1e-5 |
| s3 | .773 | -.075 | -.107 to -.045 | 3e-6 |
| s4 | .757 | -.091 | -.125 to -.059 | 9e-8 |
| s5 | .762 | -.086 | -.118 to -.057 | 1e-7 |

  - This matches the tables (their "p=0.0" is rounding).
  - The drop is not an artifact. It holds with per-token-normalized scores (.871 to .76-.78), in every gold-vs-foil length group, and on the 430 items containing no training-pool value (-.087).
  - Largest drops: currency -.18, symbol -.13, state capitals -.11. Capitals lose only -.04. On khard40 alone the intervals include 0 (n=40).
- **Chat-template generation collapse: CONFIRMED, and it is a stopping failure.**
  - Chat strict GEN is .02 / .27 / .05 / .13 / .46 (s1 to s5), while lenient is .82-.83 on every seed.
  - Replies capped at 48 tokens: 624, 436, 610, 546 and 299 of 640.
  - In the transcripts the model writes a correct answer, then a blank line and more answer templates until the cap: "Gray is the color.\n\nYou chose gray.\n\nYou chose gray...".
  - Grading only the first line puts chat within about .05-.10 of plain GEN (s1: .61 .72 .86 .92 .72 .89 .92 .78 .86).
  - Cause: training targets were "answer\n" in the plain render, where the next line is "User:". An end-of-turn token after an answer was never trained, and the chat render stops only at end of turn.
- **Real chat damage.** In the 53-conversation chat probe, 148-149 of 149 assistant turns hit the length cap on every seed (base: 68). 42-60% of turns contain training answer templates. Example: asked "what did I tell you my name was?", s5 answers "April is the month." on repeat. The probe's "correction 1.0" sits on top of this, and the chat-probe line in `tables.txt` does not show it.

**Q6. Anything that makes the result wrong or overstated**
- **Leakage: none found.** My own check of eval text against seed 1's 6,400 training examples: 0 shared word 5-grams in every family, 1-2% shared 4-grams, 0 alias names or honorifics in training. Five eval head nouns (bottle, refill, napkins, level, block) appear in training fillers; that is incidental vocabulary, not structure.
- **Grader: no false passes or fails found.** On 2,880 plain replies in the pass-rule families, strict equals lenient in every case. GEN and LIK agree on 96%. GEN is in effect "which value did it name".
- **Seeds do differ.** Training streams and losses differ, and pairwise the seeds pick the same answer on only 545-576 of 640 items. But all seeds share the same 64 items per family, so "5 of 5 seeds" says nothing about item sampling: the H1/H2 outcome is set by 21 alias items each.
- **Analyzer.** The numbers are correct, but:
  - Both sub-readings are fixed strings in `rules_e004.sub_readings`, triggered by any H1/H2/control LIK failure without looking at what the model picked. "The E002 shortcut signature" and "a rule that does not transfer" are both contradicted by the item evidence above.
  - The cross/chat continuity comparison was skipped ("items differ (384 vs 184)"), so "0 decision flips" is unverified for that set.
  - Cosmetic: 0.797 is printed as 0.80, and McNemar p as 0.0.
- **Overstated passes:**
  - H6 is partly carried by the training value types (weekday .94-1.00, colour .81-.94). On the two new value types alone (sport + number, n=32): LIK .88 .78 .69 .88 .97.
  - H3's pass hides an ellipsis-latest score of .60.
- **Counterfactual** (post hoc, not a rescue): dropping the 21 alias items from H1 and H2, the registered rule reads PARTIAL-X (H5). Every seed passes everything except H5, and s3 passes all.

**Q7. Tiny ladder running its LR search only: faithful to the pre-registration**
- The pre-registration (d) says: "If SmolLM2-135M-Instruct is a FAIL, the tiny ladder runs its LR search only". `queue_e004.sh` does exactly this (grid plus the one allowed extension, logged "SKIP seeds ... would use"), and `queue_e004_ts1m.sh` reads the same log line.
- Caveats:
  - The second half of that sentence, "E003's deferred easy version becomes the diagnostic", is not queued. E003's own notes trigger it on "E004's tiny models fail", which LR-only runs can never show.
  - The tiny LR picks maximize the dev minimum. For 135M that minimum was H1/H2 at both LRs, so the tiny picks will mostly be driven by alias items.
  - Changing the gate now would need a logged deviation.

**What E004 shows and does not show**
- **Shows:** 135M after 400 steps learned an updating rule that transfers to k=4-5, d=20 (every item over 768 tokens), new fillers, new objects and both controls. It also resists the frame-echo lure that the untouched model follows.
- **Fails on:**
  - linking an alias name back to its object (.10-.57);
  - ellipsis when another object's statements follow (about .70);
  - three objects, where later statements of other objects win (H5 .70-.81).
- **Does not show:** that the E002 wording shortcut persists (the items refute it); that alias linking is beyond 135M (it was never trained); that chat answers are wrong (the model does not stop).
- The pre-registered FAIL stands, but the reason written next to it is wrong.

**Most informative next experiment**
The E004 recipe with one change: add alias corrections to training, using a training-only name pool, different join words and templates, and both adjacent and separated placement. Keep 5 seeds, the same LR and steps, and score on the unchanged E004 eval draw with the registered rule, reporting the alias adjacency split.
- If alias accuracy rises to 0.8 or more, the FAIL was a data gap. The reading would become PARTIAL-X (H5) or PASS, and the tiny ladder should run with the new recipe.
- If it stays at 0.5 or below, linking a name to its object is a real limit at 135M.

Scratch scripts are in `/private/tmp/claude-501/-Users-brohm-Documents/de64e45b-e167-412e-8ebf-84d92eb12068/scratchpad/e004/` (`load.py`, `q1.py` to `q6c.py`, and `q4.out` with all 42 alias items and every seed's replies). Nothing in E004 was modified and no model was loaded.
