# MaxGPT-Planck experiments

A public lab notebook. It records every experiment in order, including the ones that failed and the ones whose reading was overturned by their audit. Last updated 2026-09-25.

## The question

Planck asks how small a language model can be and still hold a real multi-turn conversation: remember what the user said several turns ago, take a correction, keep two people's facts apart, keep following an earlier instruction, and not loop. The bet is that this skill is cheap in parameters and that stored world knowledge is what is expensive (see the [README](../README.md)).

Before training anything from scratch, the first experiments look at one skill that every small model we tested gets wrong: updating after a correction. If a user says "the dentist appointment is on Monday", later says "actually, it moved to Thursday", and ten unrelated exchanges after that asks what day it is, the right answer is Thursday. Small models mostly say Monday. The experiments ask whether that failure is real, whether training can fix it, and what the model has actually learned when it seems fixed.

## How the experiments are run

- **Pre-register before running.** Each experiment's notes file states the question, models, seeds, pass rule and how every outcome will be read, with a timestamp, before any model is scored. Later changes are logged as deviations with their time and reason. Analyses added afterwards are labeled post hoc.
- **Two kinds of score.** *Likelihood* (LIK): the model reads the dialogue and a fixed answer prefix, and an item counts only if the right value gets strictly more probability than every other value mentioned in the dialogue (ties fail). *Free generation* (GEN): the model writes its own reply, graded by a strict program. The reply must name the right value and no other, with no negation, hedge, question, loop, or speaking as the user.
- **Cheater checks on the real items.** Before any model is scored, simple rules that read only the text ("pick the first value mentioned", "pick the last one", "copy the value after the words that match the question", and about twenty more) are run on the actual test items. A test set is accepted only if these rules fail it.
- **Mutation tests.** Graders, item builders and pass rules are deliberately broken in small ways (for example, letting ties count as right), and the test suite must catch every broken version. A check that has never been seen to fail is not trusted.
- **Independent audits.** After a result, a separate AI agent that did not run the experiment, and loads no model, recomputes every number from the raw outputs and tries to break the conclusion: leakage between training and test, shortcuts, grader errors, overstated claims. The audit is published next to the result. Where it disagrees with the result, the audit's reading is the one the [ledger](../LEDGER.md) records.
- **Several seeds.** Fine-tuning results use 3 to 5 training seeds. A model passes only if a strict majority of its seeds pass.
- **Publish failures.** Failed rules, overturned readings and skipped arms are reported here.

The work is done with Claude (Anthropic) agents: they write the code, run the job queues and audit each other's results, and every claim links to its raw output. So far everything ran on one laptop with 24 GB of memory, one model process at a time under a memory guard.

Terms: *d* is the number of unrelated exchanges between the last relevant statement and the question (d10 means ten). *k* is the number of corrections. *Chance* is 1 divided by the number of candidate values in the dialogue. *Body* is the parameter count without the embedding tables.

## E001: is "first value wins" real?

Files: [notes](../experiments/E001_battery_and_probes/notes.txt), [results](../experiments/E001_battery_and_probes/results.json), [battery results](../experiments/E001_battery_and_probes/battery_results.json), [chat results](../experiments/E001_battery_and_probes/chat_results.json)

**Question.** The research phase found that small chat models keep the first value after a correction, but its test items were confounded: the original statement often shared wording with the question, and the text format was not each model's own chat format. Does the failure survive once those confounds are controlled?

**What was run.** New paired test items in two families (a dentist-appointment weekday and a new-car colour), 32 scenarios each at d = 0, 4 and 10, with controls: identical wording in the original and the correction; three unrelated exchanges before the original (position); the question's key words in only the original, or only the correction; a no-update twin where the second statement is about a different object; and a two-slot item where another object's value comes last. Every item was scored in plain "User:/Assistant:" text and in each model's own chat template, 1,728 items per format. Models: 13 public models from 90M to 2.6B parameters (SmolLM2, Falcon-H1-Tiny, LFM2 and LFM2.5, Gemma 3, Qwen2.5, Qwen3) plus MaxGPT-3, a 235M model from earlier in this project. Two models not probed before (LFM2.5-230M, Falcon-H1-Tiny-90M) also went through the 53-conversation chat probe, with graders hardened against the false passes an earlier audit had found.

**Pre-registered rule.** E001 was a measurement, not a pass/fail test. Its checks were fixed before scoring: five text-only oracle scorers (first mention, last mention, key-word matcher, ideal, empty) had to land exactly where the design said, and 12 of 12 broken versions of the item set had to be caught. The hardened chat graders pass 591 self-test assertions and catch 43 of 43 mutants. On 214 adversarial replies built from real false passes, the old graders passed 204 and the hardened ones pass none.

**Result.**
- First value wins is real in every instruct model below 2.6B that was tested (11 models from 90M to 1.2B, five families). With identical wording and 4 to 10 exchanges after the correction, 0.00-0.24 of items are right in the models' own templates. The no-update twin is right 0.92-1.00 of the time, so the models can read the dialogue; they prefer the first matching statement.
- Key-word match and turn-1 position make it worse but do not explain it. With the question's key words only in the correction, the models still pick the original on a third to nine tenths of items (0.09-0.67 right). The chat format mattered at d = 0, where it rescued three LFM models, but not at distance.
- Two-slot items fail even at d = 0 in every instruct model below 2.6B (0.00-0.39): a single later statement about another object is enough to send the model back to the original value.
- The bar is reachable. LFM2-2.6B in its own template scores 0.97 on identical-wording corrections, 0.99 on the no-update twin and 0.93 on two-slot, with all five controls right together on 0.79 of scenarios.
- SmolLM2-135M base and instruct fail alike, so chat tuning neither caused nor fixed the failure. MaxGPT-3 is not better at updating; it is less committed to either value (margins near zero, controls near coin-flip).
- Chat probe: LFM2.5-230M follows instructions at the level of the 350M models, but deflects on 10 of 15 "what did I tell you" questions and fails all 3 corrections.

**Audit.** E001 has no separate audit file. Its checks were built into the run (the oracles and mutants above), the scorer reproduced earlier scores to about 1e-6, and the hardened graders were compared with an earlier hand regrade of 8 models' transcripts (within 0.01 for 7 of 8). A known grader gap: a reply that restates the fact without answering the question still passes. Each cell has 64-128 items from one item draw, so differences under about 0.1 are noise.

**What it changed.** Updating after a correction became the first skill to attack: a real failure below 2.6B, with one model (LFM2-2.6B) showing that the test can be passed. Next question: can a short fine-tune teach it?

## E002: can 400 fine-tuning steps teach a 135M model to update?

Files: [notes](../experiments/E002_ft_test/notes.txt), [audit](../experiments/E002_ft_test/AUDIT.md), [results](../experiments/E002_ft_test/results.json), [post-hoc results](../experiments/E002_ft_test/posthoc_results.json), [tables](../experiments/E002_ft_test/logs/tables.txt)

**Question.** Is updating trainable at 135M, or does it need a larger model?

**What was run.** SmolLM2-135M-Instruct (134.5M total, 106M body), 5 seeds, 400 optimizer steps (AdamW, learning rate 5e-5, effective batch 16, 6,400 examples per seed) on synthetic correction dialogues whose wording was held out from the E001 test items. 40% of the training mix was owner, two-hop and perspective dialogues. A 360M arm with 3 seeds was also pre-registered.

**Pre-registered rule.** A seed passes if, at d = 10 in plain text, it scores at least 0.8 on the identical-wording correction items (n = 192), the two-slot items (n = 64) and the no-update twins (n = 64). The model passes if 3 of 5 seeds pass. The reading of each outcome was also fixed in advance, including "main items pass but controls fail: a shortcut, tells us nothing".

**Result.** Passed on 5 of 5 seeds: correction items 0.95-1.00 (untouched model 0.00), two-slot and no-update 1.00. The switch happened by step 50, the first probe, within about 800 examples. It held in free generation. Closed-book knowledge fell 0.2-3.2 points. A crossed control added post hoc (both objects stated, both corrected) scored 0.89-1.00.

**What the audit found.** The pass reproduces exactly, but it does not show a general skill.
- The wording was held out but the task was not. Every scored structure (item types, value lists, distances 0-10, text format) was in the training data.
- Two cheap text rules pass both the pre-registered rule and the crossed control: "copy the value that follows the last occurrence of the answer prefix's last three words" and "take the latest turn sharing the most words with the question". The crossed control only defeats a rule keyed on the word "Actually".
- Where the correction is worded differently from the question, several seeds fall below chance: 0.00-0.06 on three-correction neutral-wording items for four of five seeds, and 0.00-0.69 on colour items where only the original shares the question's key words (chance 0.5).
- The owner, two-hop and perspective gains came straight from training on those same families.
- Collateral: plain-text replies collapsed to a bare "Value.". The 360M arm was skipped after the 135M pass, which the audit lists as a gap, and the chat-probe check for conversational damage was stopped by the memory guard before it produced a fine-tuned transcript.

**What it changed.** E002 is recorded as "passed the rule, shortcut not excluded: general updating not shown". The size-floor experiment built on its items (E003) was deferred, and E004 was designed so that no known shortcut can pass.

## E003: where does the E002 recipe stop working? (deferred)

Files: [notes](../experiments/E003_correction_floor/notes.txt), [results](../experiments/E003_correction_floor/results.json), [tables](../experiments/E003_correction_floor/logs/tables.txt)

**Question.** At what body size does E002's recipe stop teaching the skill? Models of 30M total parameters and under come first, because they are the main target of the project.

**What was run.** Pre-registered for eight public base models on two ladders: Pythia 14M, 31M, 70M and 160M (same data, tokenizer and context length, so the clean ladder) and TinyStories 1M, 3M, 8M and 33M (narrow-domain text, pretrained on 512-token sequences, a secondary ladder). Bodies run from 0.4M to 85M. Items, pass rule and recipe as in E002, plus a per-model learning-rate search on a separate dev draw with seed 0, then 3 fresh seeds. Only the untouched baselines ran, for six of the eight models.

**Pre-registered rule.** E002's rule per seed, 2 of 3 seeds to pass. A failure at every learning rate was to be read as "not learnable in 400 steps with this recipe at this size", explicitly not as proof of a capacity limit.

**Result.** No fine-tuning ran. The untouched base models fail, as expected: 0.20-0.44 on the correction items at d = 10.

**Audit.** None of its own. E002's audit applies: these items can be passed by the wording shortcut, so a pass here would have meant little, and E003's secondary "per-object tracking" label is satisfied by the prefix-copy rule.

**What it changed.** Deferred on Sep 24. E004 tests the real skill on the same tiny ladder. E003 stays available as the easy-version diagnostic if the tiny models fail E004's harder test.

## E004: a test that only a general updating skill should pass

Files: [notes](../experiments/E004_general_updating/notes.txt), [audit](../experiments/E004_general_updating/AUDIT.md), [results](../experiments/E004_general_updating/results.json), [tables](../experiments/E004_general_updating/logs/tables.txt)

**Question.** Can a 135M model learn "the answer is the latest statement about the asked object", whatever its wording, position, marker or distance, when every scored structure is absent from training?

**What was run.**
- New training data: update dialogues only (no owner, two-hop or perspective families), four value types (weekday, colour, month, city), ten dialogue kinds, separate paraphrase pools for originals, corrections, questions and answers. Corrections refer to the object by full name, head noun, pronoun or ellipsis ("make that Friday"). Markers like "actually" also sit on statements that are not corrections. No question echoes a statement's wording. Answers are short natural sentences, never a bare value.
- Held-out test: nine families of 64 items, each moving one axis that never occurs in training. H1 and H2: the question echoes the wording of a stale correction (H1) or of the original (H2), which is the E002 shortcut made adversarial. H3: 4-5 corrections (training has at most 3). H4: d = 20 (training at most 10). H5: three objects (training at most 2). H6: new object types and two new value types. H7: a new kind of filler conversation. Plus two controls: no update, and two-slot. No test turn, question or answer shares a five-word sequence with the training text.
- Cheater check on the real items: about 20 text-only rules. None reaches 0.80 on any family except the no-update control (the highest is 0.56), and the wording rules score 0.00 on H1 and H2. On the no-update control, rules that pick "the statement naming the asked object" are right by construction; the same rules score at most 0.50 on every family that has a correction of the asked object.
- SmolLM2-135M-Instruct, 400 steps, learning rate 1.5e-4 (picked over 5e-5 on a separate dev draw by a pre-registered rule: worst family 0.78 against 0.59), 5 fresh seeds. The tiny ladder (TinyStories 1M to 8M, Pythia 14M and 31M) was to get full seeds only if 135M did not fail, and a learning-rate search otherwise.

**Pre-registered rule.** A seed passes only if all nine families score at least 0.8 at both LIK and GEN (n = 64 each). The model passes if 3 of 5 seeds pass. Partial readings were defined in advance (for example, passing everything except one or two of H3-H7 reads as "updating with a named limit"). A likelihood failure on H1, H2 or a control can never be partial.

**Result.** FAIL: 0 of 5 seeds pass. Likelihood scores (free generation tracks them closely):

| family | chance | untouched | fine-tuned, 5 seeds |
|---|---|---|---|
| H1 wording lure on a stale correction | 0.20 | 0.05 | 0.66-0.70 |
| H2 wording lure on the original | 0.24 | 0.06 | 0.66-0.797 |
| H3 4-5 corrections | 0.15 | 0.06 | 0.81-0.88 |
| H4 d = 20 | 0.23 | 0.16 | 0.89-0.97 |
| H5 three objects | 0.15 | 0.16 | 0.70-0.81 |
| H6 new object and value types | 0.23 | 0.17 | 0.81-0.94 |
| H7 new filler kind | 0.23 | 0.19 | 0.89-0.95 |
| control: no update | 0.23 | 0.39 | 0.84-0.94 |
| control: two-slot | 0.20 | 0.16 | 0.81-0.88 |

The untouched model is below chance on seven families: it actively picks a wrong value (on H1 and H2, mostly the lure). As pre-registered after a 135M FAIL, the tiny ladder ran learning-rate searches only. On the dev draw (one seed, 32 items per family), TinyStories-8M and Pythia-31M stayed near chance at every learning rate: the best mean over the nine families was 0.34, with chance about 0.2. That is a lead, not a verdict, and TinyStories models were pretrained on 512-token texts while most test items are longer. The remaining searches are paused while E005 runs.

**What the audit found.** The FAIL stands, but the explanation the analysis script printed next to it ("the E002 shortcut signature") was wrong: it was a fixed string triggered by any H1 or H2 failure.
- The wording shortcut is gone. Pronoun corrections score 1.00 even when the question echoes a stale statement. The fine-tuned model picks a lure value on only 4-12 of 64 items, against 42-56 for the untouched model.
- The H1 and H2 failures are nearly all one reference form the model never saw: corrections that name a person instead of the object ("Mark down Monday for the staff briefing with Ms. Petrov" ... "Ms. Petrov rescheduled me for Sunday"). Training contained no names. On these alias items the seeds score 0.10-0.33 (H1) and 0.24-0.52 (H2). Without the 21 alias items per family, H1 is 0.88-0.93 and H2 0.81-0.93 on every seed. When wrong, the model picks the asked object's previous statement: it attributes statements by name, head noun or adjacency, and never links the person's name to the object.
- Two more weaknesses: an ellipsis correction followed by the other object's statements (about 0.70), and three objects (H5), where later statements about other objects win.
- Some passes are thin. At n = 64 the 95% interval is about +/-0.10, and H6 on its two new value types alone is 0.69-0.97.
- Costs: closed-book knowledge fell 7.5-9.5 points on 441 items, every interval excluding zero. In the chat template the model gives a correct answer and then does not stop: training answers ended with a newline in plain text, never with an end-of-turn token. Strict chat GEN was 0.02-0.46 while a lenient grade was 0.82-0.83, and in the 53-conversation chat probe 148-149 of 149 replies ran to the length cap (untouched: 68), repeating training answer templates.
- Counterfactual, post hoc and not a rescue: with the alias items removed, the registered rule would read PARTIAL-X, a named limit on three objects only.

**What it changed.** E004 is recorded as "FAIL, but a transferable updating rule was learned; the failures are untrained forms plus a format gap". The audit's proposed next experiment became E005.

## E005: data gap or real limit?

Files: [notes](../experiments/E005_alias_eot/notes.txt), [audit](../experiments/E005_alias_eot/AUDIT.md), [queue log](../experiments/E005_alias_eot/logs/queue.txt)

**Question.** Was E004's alias failure missing training data, or a limit of a 135M model? And does training an end-of-turn token fix the chat-format stopping failure?

**What was run.** E004's recipe with exactly three changes to the training data. The other 70% of examples are E004's own, in the same order.
1. 20% of examples contain alias corrections, with training-only surnames, joining phrases and templates. The alias correction comes right after the object's previous statement, after filler, or after the other object's statements. In a quarter of these examples the named statement is a lure about the other object, and in a third both objects have a name, so "the object with a name" is not enough.
2. 10% of examples have an ellipsis or pronoun correction followed by the other object's statements (two objects only; three objects stay held out for H5).
3. Half of all examples use the chat template with an end-of-turn token after the answer.

Same model, learning rate 1.5e-4, 400 steps, seeds 1-5, and E004's unchanged test items (checked prompt by prompt, 1,120 of 1,120).

**Pre-registered rule.** E004's pass rule, unchanged. Plus an alias reading on the 42 alias items: DATA GAP if 3 of 5 seeds score at least 0.8 at both LIK and GEN, REAL LIMIT if 3 of 5 score at most 0.5 at both, otherwise INCONCLUSIVE. Scored by the same code, E004's own seeds read REAL LIMIT (4 of 5 low). "Stopping learned" if chat strict GEN is within 0.10 of plain GEN on every family for 3 of 5 seeds. A risk stated in advance: the four titles (Mr., Mrs., Ms., Dr.) are shared with the test, so DATA GAP would mean "new surnames, joins and templates under familiar titles", not names never seen at all.

One gap in that design: every alias correction in the E004 test is also the right answer, so "the latest statement by a named person wins" would score perfectly without linking anything. A 64-item diagnostic set (AL) with new test-only names, where that rule scores 0.25, was built and hashed before E005 started. It will be scored on the untouched model, on E004's seeds and on E005's seeds. No rule reads it, but a DATA GAP with AL below 0.5 would be reported as "resolved by alias recency, not by linking".

**Pre-run checks.** Every check passed before training: cheater rules on each seed's training stream (all at or below 0.63), purity against the test vocabulary, test identity with E004, 47 of 47 generator mutants caught, and a dry run in both formats.

**Result (audited).** All three stored readings reproduce from the raw outputs.
- **Pass rule: PARTIAL-X (H5), barely.** Three seeds (2, 3, 5) pass every family except three objects (H5). No seed passes H5 (0.61-0.72). The margin is 2 items: after four seeds the reading was FAIL, and two more wrong answers on seed 5's two-slot control would make it FAIL again. At 64 items per family the 95% interval is about plus or minus 0.10.
- **Alias reading: DATA GAP, 5 of 5 seeds high.** Alias items score 0.88-0.95 (E004: 0.21-0.43).
- **Name linking (AL).** 0.97-1.00 on every seed, but that number alone is weak evidence: E004 already scores 0.77-0.83, and a name-free rule found after the fact scores 60 of 64. The real evidence is the few items no tested shortcut can solve: 4 of 4 on every seed where the other object is the named one, and 100 of 100 (E004: 41 of 100) where both objects carry two alias corrections under the same title, a structure absent from training. So: consistent with linking through the name, on a small number of items.
- **Stopping: learned.** Every seed stops after one short sentence in the chat format (640 of 640), and chat accuracy tracks plain accuracy within 0.06.
- **Three objects got worse:** 0.66 against E004's 0.76. The model answers with the dialogue's most recent value even when that statement is about another object, and the two controls slipped the same way (two seeds fail a control in free generation). A candidate cause: in E005's training streams the asked object's latest statement comes last more often than in E004's. Untested.
- **Costs.** Closed-book knowledge -7.5 points, the same as E004 within noise. General chat is mixed: far fewer runaway turns than E004 and more of the chat probe's checks passed than the untouched model (29-33 of 73 against 21), but over half of its first sentences are training answer templates (untouched: 1%), used even for unrelated questions.

**What it changed.** Name-based corrections and stopping were data gaps at 135M, not limits. Three objects is still unsolved and is now the main open skill in this line, alongside the damage narrow training does to general chat. The audit proposes a one-change follow-up (balance where the latest statement sits in training) and a larger eval draw before any headline claim.

## What we know so far

- Below 2.6B, "first value wins" after a correction is real: 11 instruct models from 90M to 1.2B, in five families, keep the original value, while LFM2-2.6B passes the same controls (E001).
- A 135M model's first-value habit can be flipped in 400 fine-tuning steps, but passing test items shaped like the training data proves little: a wording-match shortcut passed E002's rule (E002).
- With varied wording and held-out structures, a 135M model learns an updating rule that transfers to 4-5 corrections, twice the training distance, new objects and value types, a new kind of filler and both controls, and it ignores the wording lure the untouched model falls for (E004).
- It did not learn to link a person's name to an object, which it never saw in training, or to keep three objects apart. E004 failed its rule because of these (E004).
- The fine-tunes are not free: 7.5-9.5 points of closed-book knowledge in E004, and a model that does not stop in chat format when end-of-turn is not trained (E004).
- Nothing is known yet about models of 30M and under. Untouched base models are near chance (E003), and the only fine-tunes so far are single-seed learning-rate searches that stayed near chance (E004).
- Name-based corrections and end-of-turn stopping were missing training data at 135M, not limits: E005 fixes both (DATA GAP on 5 of 5 seeds; stops cleanly in chat format). Keeping three objects apart is still unsolved and got worse (E005).
- The audits changed the reading of every fine-tuning result: E002's pass came with a shortcut, E004's printed explanation for its failure was wrong, and E005's high name-linking score rests on a small number of items that shortcuts cannot solve.

## What comes next

- **Three objects.** The E005 audit's one-change follow-up: balance where the asked object's latest statement sits in training, same eval and rule.
- **The tiny ladder.** E005 read DATA GAP, so a new pre-registration runs the E005 recipe on Pythia and TinyStories models of 30M and under. The paused E004 learning-rate searches finish either way, and if the tiny models fail, E003 becomes the easy-version diagnostic.
- **RC-12, the main test.** A battery of 12-turn conversations covering recall, corrections, keeping facts apart, instructions, loops and lookups. Its dev split (640 conversations) and graders are built and mutation-tested, and its pre-registration is drafted. It is locked, with a hash of its sealed split published, before any Planck model is scored on it.
- **Planck models from scratch.** A curve of small models, with most of the effort at 30M total parameters and under, trained on a corpus built for conversation skill and compared with much larger public models on the same sealed test.
