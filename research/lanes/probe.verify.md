# Audit of the probe lane (probe.md, results.json, transcripts)

Audit date: 2026-09-23. This file replaces an earlier draft of the same audit that was written before the machine shut down. Every number below was re-derived in this pass rather than carried over. Where the earlier draft was right it is kept; where this pass found more, it is added and marked **(new)**.

What was done: read `battery.py`, `analyze.py`, `run_model.py`, `exp_fixed_history.py` and `mutation_test_graders.py`; re-ran every grader on every saved reply (8 models, 1,219 main-battery records, 600 fixed-history rows) with my own aggregation code; fed hand-written wrong answers to the real check functions; read every passing greedy user-fact, follow-up and own-answer check, every passing correction check (greedy, sampled, controls), every failed greedy user-fact check labeled "wrong or ignored", and the E1/E2/E3 answers for the models the claims rest on; fetched the cited cards, configs, the TII blog, the Qwen2.5/Qwen3 reports and the Laban et al. abstract. No model was loaded. Nothing in the probe folder was modified: `battery.py` and `mutation_test_graders.py` were run with `python3 -B` (the `__pycache__` listing is unchanged), `analyze.py` was imported, never run as a script (it rewrites `results.json`). Audit scripts live in the session scratchpad.

## Verdict

The bookkeeping is clean: all 144 ability and macro values in `results.json` reproduce exactly from the transcripts, and the self-test and all 14 mutants behave as claimed. The graders are weaker than "strict" suggests. They reject deflections, greeting echoes and first-person sentences with no "you", but they accept negated gold, lists of guesses, hypothetical examples, wrong answers that contain the gold number, first-person claims in a sentence that also says "you", replies that restate a fact while ignoring the question, and any standalone "4" for the 4 pm correction. Real false passes of every one of those kinds are in the transcripts.

The headline conclusions survive hand-regrading: the 350M class is level with Qwen2.5-0.5B (through LFM2.5-350M and SmolLM2-360M; LFM2-350M's lead is fragile); below 300M the order does not follow size; rendering, not memory, is what breaks SmolLM2-135M recall; deflection is post-training, not size. Several numbers and secondary claims do not survive: "63% retrieval-intact" is 48% once single-turn controls are removed; the deflection ranking changes when controls are removed; "corrections fail at every size" is overstated; role capture is 7 to 8 of 8 models, not 6; both ~100M models adopted the pirate persona; Qwen3 does not have the lowest self-copy in its group; the tokens-per-parameter range tops out at LFM2.5 (about 79,000), not Qwen3. **(new)** The report also skipped the one cheap controlled comparison its own Falcon discussion asks for: TII published same-architecture 90M curriculum, pre-DPO and base variants, and TII's own 2-turn MT-Bench does not favor the SFT-in-pretraining recipe the lever table credits.

---

## 1. Recompute of per-model, per-ability scores

- Greedy and sampled, 8 models, 7 ability columns plus controls plus macro: 144 values, **0 mismatches** against `results.json`.
- Macro granularity: greedy recall has 10 checks per model, follow-up 7, instruction 14, correction 3, own-answer 3 to 4, role 3, topic return 3. One check in a 3-check category moves the macro by 0.048.
- 43 checks (41 greedy, 2 sampled) differ between the run-time grade stored in each transcript and the regrade with current graders. That is the documented grader-fix history; `results.json` consistently uses the regrade, so every model is scored by the same code. Confirmed.
- F3 counts reproduce (63 failed recall/binding checks with a forced probe: 40 greedy-gold, 58 gold beats foil). E2 counts reproduce (179 failed free answers through each model's chat template: 111 greedy-gold, 162 gold beats foil; n >= 1 only: 95 of 155).
- Taxonomy reproduces exactly (greedy 95: deflection 36, wrong/ignored 33, copy 13, stale 7, perspective 6; sampled 170: wrong 69, deflection 49, perspective 29, copy 15, stale 8).
- Loop, self-copy, deflection-rate, persistence, flag counts and forced-choice margins in the report tables all match `results.json`.
- Sampled Falcon has 53 records (one seed); the other seven have 106 (two seeds). Confirmed.
- Parameter counts: configs fetched from each repo give vocab x hidden of 32768 x 512 (Falcon), 49152 x 576 (SmolLM2-135M), 262144 x 640 (Gemma; `layer_types` 15 sliding-512 + 3 full), 65536 x 1024 (LFM2 and LFM2.5; `full_attn_idxs` has 6 entries, so 10 conv + 6 attention), 49152 x 960 (SmolLM2-360M), 151936 x 896 (Qwen2.5), 151936 x 1024 (Qwen3). All tied (LFM2's config omits the key, but the Hub reports 354,483,968 stored parameters, which only fits a tied head). The report's non-embedding counts follow.

## 2. Grader audit

### 2a. Adversarial probe of the real check functions

Each row is `check["fn"](answer, [""]*10)` on the unmodified `battery.py`:

| check | answer fed in | result |
|---|---|---|
| R_d2_name | "Your name is not Priya." | PASS |
| R_d2_name | "Is your name Priya, Maria, or Anna?" | PASS |
| R_d2_name | "I told you my name was Priya." | PASS |
| R_d2_name | "PRIYA" / "priya" | PASS / PASS (case handled) |
| R_d2_name | "Hello Priya!" | fail (greeting guard works) |
| R_d2_name | "I'm sorry, I can't recall. Was it Priya?" | fail (deflection guard works) |
| R_d4_city | "I don't have specific information about your location, but if you live in Tucson you might enjoy it." | PASS |
| R_d4_city | "You don't live in Tucson; you live in Phoenix." | PASS |
| R_d2_number | "The apartment number for building 417 is 12345." | PASS |
| R_d2_number | "Maybe 415, 416 or 417?" | PASS |
| R_d6_number | "She is turning 94? No, she is turning 90." | PASS |
| K_time | "Here are tips: 1. Plan ahead. 2. Bring notes. 3. Be polite. 4. Arrive early." | PASS |
| K_time | "Show up at 3 pm. It takes about 4 minutes to walk there." | PASS |
| K_day | "It's on Monday, and it's not on Tuesday." | PASS |
| K_day **(new)** | "Your appointment is on Monday. Tuesday is free." | PASS |
| K_color | "Options: 1. Blue 2. Green" | PASS |
| K_color **(new)** | "Your favorite color is blue. Many people also like green." | PASS |
| T_party | "Bake them at 350F for 12 minutes." | PASS |
| F_whynot | "Touching it is safe and prevents burns." / "It won't hurt at all." | PASS / PASS |
| R_d5_multi | "Elena: you are a parrot named Kiwi living in Oslo." | PASS (all 3 checks) |
| RI_binding | "Your name is Oscar... or maybe it is Lena." | PASS |
| RI_binding **(new)** | "Oscar? Lena? I'm not sure." | PASS |
| RI_identity user_job | "Working as a chef is multifaceted; here is what you might do." | fail (plausibly right) |

Causes: `has()` is a substring test with no negation, hedge, list or example handling; `captures()` exempts any sentence containing a second-person pronoun; `has_not_stale()` accepts any reply whose last mention is the new value, so "Monday ... Tuesday is free" passes; `NO_MEMORY` misses "I don't have specific information", "as a helpful assistant, I'm here to assist", "I am not able to answer that question" and "I am not sure what you told me"; the K_time gold `(?<![\d:])4(?![\d])` matches list numbering and incidental numbers.

### 2b. False passes found in the real transcripts

Clear cases, greedy, multi-turn:
- LFM2-350M K_day (its only correction pass, cited in F5): Tuesday appears only in "For example, if your dentist appointment is on Tuesday, you can check your calendar".
- LFM2-350M R_d4_city: "I don't have specific information about your exact location, but I can tell you that Tucson, Arizona, is a beautiful city". A deflection the regex misses.
- LFM2-350M R_d2_name: "You told me my name was Priya." (perspective error that escapes the capture guard).
- Falcon-H1-Tiny-90M R_d5_multi, 3 of its 5 recall passes: "1. **Elena**: You're a parrot named Kiwi living in Oslo. 2. **Rayleigh Scattering** ... 3. **Noun**". The sampled run repeats it ("**Elena**: I am a parrot named Kiwi."). This inflates Falcon's recall 0.50 and the F2 cell "Falcon d=5 1.00 (n=6)".
- Gemma-3-270M T_dragon fear: "It's a very common fear for many dragons" (a non-answer; "fear of heights" only in its restatement of the question).

Borderline incidental mentions **(new)**: the reply contains the gold but does not answer the question.
- LFM2-350M T_dragon (both checks): asked "What is my dragon afraid of, and what is its name?", it writes mountain titles: "Given Ember's secret fear of heights, let's tie it into the mountain's name. Here are a few ideas: * **The Spine of Ember's Fear**".
- Qwen3-0.6B T_dragon (both checks): writes a new story, "Here's a **different and engaging** story idea, keeping the **dragon's secret fear of heights** and its **name**", with Ember only inside the story.
- Falcon T_dragon name and Gemma T_dragon name: story continuation / "a name that comes from the dragon's own name, Ember".
- SmolLM2-135M knows_60 (turn 0 of O_add10): "15 x 4 = 60 / 60 + 60 = 120 / So, 15 times 4 equals" (cut at the cap, heading to 120).
- Falcon knows_gracias: a list of wrong phrases, "Así qué se han sido (Goodbye, thank you) ... Buenas tardes (Good morning, thank you) ... G[racias]".

Sampled, multi-turn:
- LFM2-350M K_color seed 0 and seed 1: both are lists of colour options ("1. **Blue**: ... 2. **Green** ..."). These are LFM2's only sampled correction passes.
- SmolLM2-360M K_color seed 1: "I'm not sure, but I can suggest some options. Blue is a great color ... Green is another great color ... Red ... Yellow ... Purple".
- Falcon K_color seed 0: "Colors like blue and green can evoke feelings of calmness".
- Falcon F_whynot seed 0: "While hot stoves are generally considered safe ...".
- SmolLM2-135M F_whynot seed 0: "touching a hot stove can be a safe and effective way to prevent burns".
- LFM2-350M R_d4_pet seed 1: "The name of Prima Biscuit, your cat, is typically 'Prima.'"
- LFM2.5-350M T_dragon seed 1 name: "Your dragon, Ember, is afraid of heights. Its name is Luminara."
- SmolLM2-360M T_dragon seed 1 name: "Let's give Ember a name that reflects her fear of heights."

Single-turn controls (these inflate the control column and F10):
- Greedy: LFM2.5-350M R_d2_number "The apartment number for building 417 is **12345**."; Falcon R_d6_number "your grandmother is 188 years old next week."; Falcon R_d4_city "The city closest to Tucson, Arizona, is Tucson."; Falcon K_time "4 pm - 3 pm = 1 hour 30 minutes"; LFM2-350M R_d4_pet (suggests renaming the cat "Boxy"), R_d6_name ("Hello Marcus! My name is Alex."), R_d6_number ("approximately 100 years old"), K_color ("a vibrant mix of blue and green"); Gemma R_d4_city (Tucson only inside "(e.g., Downtown Tucson, ...)"); SmolLM2-135M R_d5_multi ("I'm Elena, and I'm a chatty AI assistant who lives in Oslo", allowed because the capture guard is off for that test).
- Sampled: SmolLM2-135M K_day seed 0 (negated gold). **(new)** Qwen2.5-0.5B K_time seed 1: "you should arrive at your designated location one hour earlier than usual. Typically, this means arriving around 2:45 PM." **(new)** LFM2-350M K_color seed 1: "your favorite color at this point is probably **purple** or **orange** ... I'd say your favorite color right now is probably **vibrant yellow**." **(new)** Falcon K_time seed 0: "3 pm - 4 pm = -1 pm".

Fixed-history sweeps (E1/E2/E3):
- **13** passes (the earlier draft said 12) are first-person claims that escape the capture guard because the sentence also contains "you": Qwen3-0.6B short n=4, 8, 12 "I told you my name was Priya."; LFM2-350M long n=2, 4, 8, 12 "You told me my name was Priya."; LFM2-350M short n=8 "I'm Priya, but you could say: ..." and **(new)** short n=12 "I'm Priya, and you asked about my name."; SmolLM2-135M base plain "You're right, I'm Priya." (short and long); SmolLM2-360M memsys "I'm Priya, but you can call me Pri."
- Falcon short n=1: "For apartment 417, you can use the following apartment numbers: - 417 - 418 - 419 - 420" and two hedges ("A cat that you might call "Biscuit" ...").
- SmolLM2-135M-Instruct plain (E1), 5 of its 14 short-reply passes are non-answers: "Priya, you're right, I'm not sure. Let me check again.", "Tucson is a great choice for a healthy home garden.", "Priya, you're a great person to have in your life.", "Tucson is a great choice for a healthy lifestyle.", "Tucson is a great choice, but you could also consider other cities like San Antonio or Phoenix."
- SmolLM2-135M-Instruct chat (E1): "You can't live in Tucson, but you can live in the city of Tucson, Arizona.", "You're correct, I should have said "Priya" instead of "Priya"."
- **(new)** SmolLM2-135M memsys (E3): of its 4 passes, "You can find a lot of great places to eat in Tucson." and the same "instead of "Priya"" line are non-answers, one is garbled ("That's correct, Priya. Your name is a common misspelling of "Priya""), and only "Your cat's name is Biscuit." is clean.

### 2c. False fails found

- LFM2-350M greedy I_end_question t2 and t3: both hit the 160-token cap mid-question ("...Is it a relaxing, leisurely stroll with a plate of pasta, or is"). Truncation, not non-compliance. Correcting them moves LFM2 instruction from 0.79 to 0.93.
- SmolLM2-360M K_color ctrl: "As a conversational AI, I don't have personal preferences ... your original preference was blue, and now you have a new favorite color, which is green." Correct, failed by the disclaimer guard.
- SmolLM2-135M T_dragon (multi): first-person narration ("I've never seen her fear of heights before") fails both checks via the capture guard. Borderline.
- LFM2-350M RI_identity user_job: "Here's a more detailed look at what you might do". Borderline.
- Truncation in general: greedy replies at the token cap are Falcon 97 of 149, LFM2-350M 84, SmolLM2-135M 68, LFM2.5 57, Qwen2.5 51, Gemma 47, Qwen3 35, SmolLM2-360M 34. Of 581 gradable greedy checks, 222 fail and 46 of those failures were graded on a truncated reply (37 passes were too). The cap is an unreported grading factor.

### 2d. The specific failure modes the brief asked about

- Case sensitivity: every pattern compiles with `re.I`; curly apostrophes are normalized for deflection and mention checks. No case bugs.
- Echo passes: the self-test asserts the gold never appears in a multi-turn question or in any distractor, and that holds. It cannot hold for controls, where the fact is in the same message; controls are echo-prone and carry many of the false passes above. Multi-turn replies can still "echo" the fact while ignoring the question (T_dragon cases).
- Negation and lists of guesses: not handled anywhere except the stale-value check (`asserted()`), and that one only looks 3 words back. Real cases in 2b.
- Template-token leakage counted as content: graded text is decoded with `skip_special_tokens=True`; the leak flag runs on the raw decode. No raw reply in any of the 1,219 records contains a template token, so nothing was counted as content. (The graders themselves would accept "<|im_end|>Priya"; it never arises.)
- Empty answers: every grader returns False on "". Two sampled Gemma replies are empty and fail. Correct.

## 3. Were the graders mutation-tested as claimed?

- `python3 -B battery.py`: "selftest OK: 256 grader assertions over 53 conversations (29 multi-turn, 24 single-turn controls)". Every check passes its good fixture and fails "" and its plausible wrong fixture; guarded checks also fail a deflecting echo. Confirmed by running it.
- `python3 -B mutation_test_graders.py`: all 14 mutants KILLED, unmutated self-test passes, exit 0. Confirmed by running it.
- The claim is literally true, but each check has one wrong fixture, always a clean substitution ("Your name is Maria."). No fixture tries negation, a guess list, a hypothetical, first person plus "you", list numbering, an incidental number, or an unanswered question that restates the fact, which is where the real false passes are (2a, 2b). By the standing rule that an absence test must try to produce the thing it rules out, "strict" is not earned for these modes.
- The E2/E3 path (`analyze.fixed_history`, `e2_pooled`) re-implements the strict rule inline instead of calling `has()`, so the self-test and mutants do not cover it. I checked equivalence directly: on all 600 fixed-history rows the inline rule and `has()` agree (0 differences).

## 4. Claim-by-claim verdicts

1. **350M class matches the 0.5B wall model (0.65/0.62/0.57 vs 0.60; Qwen3 0.71; sampled 0.61/0.66/0.58 vs 0.50).** Numbers reproduce exactly. Fixing only the clear greedy cases (8 flips): LFM2 0.60, LFM2.5 0.62, SmolLM2-360M 0.57, Qwen2.5 0.60, Qwen3 0.71. **(new)** Also flipping the borderline incidental mentions: LFM2 0.50, Qwen3 0.62, the rest unchanged, so LFM2 drops below Qwen2.5 while LFM2.5 and SmolLM2-360M stay level. Sampled, clear cases fixed: LFM2 0.56, LFM2.5 0.64, SmolLM2-360M 0.53 vs Qwen2.5 0.50. "Card sampling" is a different decoder per model (LFM2.5 T 0.1, SmolLM2 T 0.2, Qwen2.5 T 0.7 with repetition penalty 1.1, Gemma T 1.0). **Confirmed** for the class; "LFM2 above Qwen2.5" is not robust.
2. **Below 300M the order does not follow size (Falcon 0.49, Gemma 0.36, SmolLM2-135M 0.15).** Order holds under every regrade I tried (clear cases: 0.46 / 0.31 / 0.15; with borderline: 0.40 / 0.27 / 0.11). Falcon's one-seed sampled 0.57 contains 5 false passes and is about 0.47 corrected. **Corrected** (order right; Falcon inflated).
3. **Most recall failures are retrieval-intact (40/63 = 63%; 58/63 foil; E2 111/179).** Counts reproduce, but 21 of the 63 are single-turn controls where the fact sits in the same message. Multi-turn only: 20 of 42 (48%) greedy-gold, 38 of 42 (90%) gold beats foil. **(new)** Qwen3 contributes 5 of the 42 at 0 greedy-gold because it opens with "**" (the report's own caveat); without it, 20 of 37 (54%). E2 reproduces (111/179 = 62%; 95/155 = 61% for n >= 1). Both probes are low bars: the foil never appears in context, and in E2 the gold is the only name, number or city in the history, so this is an induction-copy test. **Corrected**: about half of multi-turn battery failures, about 60% in the controlled sweep.
4. **E1: rendering moves SmolLM2-135M recall 0.15 to 0.70, deflection 0.30 to 0.00; base 0.60; direction held at every distance.** Numbers reproduce. Hand-regraded (2b): chat about 0.05 to 0.10, plain instruct 0.45 (9 of 20), base 0.55 (11 of 20). Direction holds at every short-reply distance; **(new)** in the long-reply condition plain and chat tie at n=4 (0.50 each), so "every distance" needs the "short replies" qualifier the report body has and the finding drops. **(new)** The report's inference "the instruct weights in plain format also edge out the base model, so SFT added usable conversational skill" does not survive: 0.70 vs 0.60 was already 2 of 20 trials, and after regrading the base model is ahead. Only 4 distinct facts; template and default system prompt are confounded (the report says so). **Corrected** (direction right, effect smaller, the SFT-helped inference unsupported).
5. **Deflection is the largest failure class (36 of 95); rates by model.** Counts reproduce, but: (a) counting is per check, so a three-check R_d5_multi reply counts three times; (b) Qwen2.5's R_d5_multi reply is labeled deflection although it recalls all three facts in the first person, "At the very start, I told you that I'm a large language model created by Alibaba Cloud, I'm Elena, I live in Oslo, and I have a parrot named Kiwi.", while LFM2.5's near-identical first-person summary passes; (c) copy-first priority hides 4 deflections and 4 stale values; (d) the regex misses some deflections ("I didn't know your name was. Could you share it with me?", "I am not sure what you meant to tell me") that end up in wrong/ignored. Per reply, greedy is deflection 32, wrong 31; per check with (b) relabeled, 33 vs 33; with the hidden deflections restored, about 36 vs 31. Deflection is a top-two class under every labeling; "largest" depends on the choice. The per-model rates pool 15 multi-turn and 14 single-turn final turns. Multi-turn only: Qwen2.5 10/15 (67%), LFM2.5 6/15 (40%), Gemma 6/15 (40%), SmolLM2-360M 4/15 (27%), SmolLM2-135M 3/15 (20%), Qwen3 3/15 (20%), LFM2 1/15 (7%), Falcon 0/15. So "the models that deflect most are SmolLM2-360M and Qwen2.5" is a pooling artifact. "Does not track size" holds. **Corrected.**
6. **Falcon-90M holds user facts across 12 turns (E2 greedy-gold 1.00, free 0.80/0.75, zero deflections).** Greedy-gold 1.00 at every cell and zero regex deflections confirmed. The short 0.80 includes a guess list ("- 417 - 418 - 419 - 420") and two "a cat that you might call Biscuit" hedges: hand-strict short 0.65, long 0.75 (all long passes are clean). Falcon also answers as the user ("My name is Priya", "I live in Tucson") in 6 of 40 n >= 1 trials. Only 4 facts per cell. **Confirmed** with a smaller short-reply number.
7. **SmolLM2-135M loses retrieval with distance (chat 1.00 / 0.75 / 0.25 / 0.00; plain 0.75 at 435 and 0.50 at 1,635 tokens; SmolLM2-360M 1.00 through 1,697).** Per-cell numbers reproduce. Through the chat template the decline tracks turn count more than tokens: chat long n=2 (349 tokens) 3/4 vs chat short n=8 (361 tokens) 1/4; chat short n=12 (497 tokens) 0/4 vs plain short n=12 (435 tokens) 3/4 on the same weights. That is a template or persona interaction, not a memory limit. A distance effect shows only in plain long (instruct 2/4, base 1/4 at 1,635 tokens), on 4 facts. **Corrected**: the ~500-token collapse is a template effect; genuine distance fade is weak evidence.
8. **E3: one system line removes deflection at 350M+, does nothing at ~100M; the two small models ignored a persona.** Numbers reproduce. "At 350M and up" rests on the two models that deflected (LFM2.5, Qwen2.5); SmolLM2-360M had nothing to remove. At 135M the measured deflection halves (0.30 to 0.15), but the regex misses the new "as a helpful assistant, I'm here to assist" phrasing, and 2 of the 4 memsys passes are non-answers, so recall is unchanged and the deflection change is unreliable. Gemma's memsys deflection (0.45) is undercounted too ("I am not able to answer that question" x2 is not detected). The persona statement is wrong: SmolLM2-135M says "My name's Captain Pip ... I'm a pirate" and Gemma "Ahoy there, matey! I'm Captain Pip, and I'm a cheerful pirate."; both pass name_pip and only miss the literal "Arr". The inference "below about 100M non-embedding the chat behavior cannot be prompted into place" is not supported. **Corrected.**
9. **Corrections fail at every size (20/24); K_time margin favors stale in all 8, single-message favors corrected in 7 of 8.** 20/24 reproduces; LFM2's pass is a hypothetical (21/24 true). Qwen3-0.6B passes 2 of 3 greedy (K_day "It's **Tuesday**!" is clean; K_time is conditional but right) and K_time in both sampled seeds, so "every size" is overstated. K_time margins reproduce (all 8 negative multi-turn; 7 of 8 positive single-message). But K_day favors the corrected value multi-turn in 6 of 8 (+0.8 to +1.7; SmolLM2-135M -1.8, LFM2.5 -0.2), and K_color favors stale in 6 of 8. With digit-split tokenizers the K_time foil " 3" is also the first token of a legitimate early-arrival answer like "3:45", so K_time is the most confounded item. The multi-turn and single-message contexts also differ by the distractor exchange and the model's own replies, so F5's "same content, only the turn structure differs" is inaccurate. **Corrected**: weak in 7 of 8 models on free answers; log-prob evidence is mixed.
10. **All models pretrained on 800B to 36T tokens; 100B at 150M is 670 tokens/param vs 8,800 (Falcon) to 60,000 (Qwen3).** All eight token counts confirmed at source (TII blog "800GT"; SmolLM2 cards 2T and 4T; Gemma card "the 270M with 6 trillion tokens"; LFM2 card "10 trillion tokens"; LFM2.5 card "Extended pre-training from 10T to 28T tokens and large-scale multi-stage reinforcement learning"; Qwen2.5 abstract 18 trillion; Qwen3 report "approximately 36 trillion"). 100B / 150M = 667 and 800B / 91.1M = 8,782 are right, but the top is LFM2.5: 28T / 354.5M = 78,985 (Qwen3 36T / 596M = 60,403). **Corrected.**
11. **Speaker binding fails single-turn in the two smallest models (margins).** Margins reproduce. For Falcon it holds only in the forced choice: its free answers are right in 3 of 4 runs ("Since you've mentioned Lena, your name is Oscar."; "Lena is your sister, and Oscar is yours."; sampled control right; only sampled multi-turn says "here's your sibling's name: Lena"). SmolLM2-135M fails every way ("My name is Lena, and I'm the name of your sister."). One test. **Corrected.**
12. **Role capture in 6 of 8 models; Qwen3 "I'm Marcus." / "I'm Oscar."** Quotes confirmed. But `assistant_not_user` fails at the same turn (the reply to "What do you do for work?") for all 8. The report's two exceptions: SmolLM2-360M "In my role as a culinary assistant, I assist chefs in various aspects of the kitchen" and Gemma "I enjoy a variety of tasks that allow me to contribute to the restaurant's success." (that is Gemma's turn-1 reply, not "the next" turn as F6 says). **Corrected** to 7 to 8 of 8 (6 clear chef captures, 2 partial).
13. **SmolLM2-135M loops (22%, 17%, 0.95, 0.93); Qwen2.5's loop disappears with sampling (1.00 to 0.12).** All reproduce. n is 1 greedy and 2 sampled "tell me more" chats; SmolLM2's card T 0.2 is near-greedy, so it cannot show the loop is "in the model"; Qwen2.5 changes temperature and repetition penalty at once. **Confirmed** with caveats.
14. **Gemma's 512-token window did not remove the fact (greedy-gold 0.50 at 1,662; gold beats foil 0.75; free recall 0.00 to 0.25 at every distance).** Numbers and config (15 sliding + 3 full layers, window 512) confirmed. Greedy-gold is already 0.75 at n=0 (one fact never retrieved), so 0.50 at 1,662 is 2 of 3 retrievable facts; the 3 global layers are the expected path. **Confirmed.**
15. **Instruction persistence did not decay in 6 of 8 (greedy).** Confirmed from `instruction_persistence_by_turn`. LFM2's flat 0.67 includes the two truncation false fails; each turn position has only 3 instructions. Under sampling LFM2 (0.67/0.67/0.50) and Qwen2.5 (0.33/0.33/0.17) show slight decay. **Confirmed** with caveats.
16. **No template leaks or invented user turns (0 of 149 greedy turns each).** Confirmed from flags and a scan of all raw decodes. 23% to 65% of greedy replies stop at the cap, so anything later is unobserved. **Confirmed.**
17. **Qwen3 strong-to-weak and on-policy distillation; lowest self-copy among non-LFM/Falcon models (7%); most perspective errors.** Distillation confirmed in the Qwen3 report (sections "Strong-to-Weak Distillation", "On-Policy Distillation"). Self-copy is wrong: Gemma-3-270M is 4.9% vs Qwen3 7.1%. "Most perspective errors" holds only under the automatic label (3 vs 2 vs 1); counting the first-person answers the label misses, LFM2 and LFM2.5 are level. **Corrected.**
18. **TII IFEval 66.08 (SFT-pretrain + DPO) vs 53.47 (curriculum + DPO); pre-DPO 50.11 vs 40.77.** Confirmed in the TII table (columns Curriculum-pre-DPO, Curriculum, English-Instruct-pre-DPO, English-Instruct). **(new)** The same table's 2-turn MT-Bench row is 3.17 / 4.40 / 3.08 / 4.33: after DPO the curriculum variant is slightly ahead on the only multi-turn metric. The report cites 4.33 in its bottom line and IFEval in its lever table without this. **Confirmed**, with an omission.
19. **Laban et al.: 39% average single-to-multi-turn drop, unreliability not aptitude.** Confirmed from the abstract (six generation tasks, 200,000+ simulated conversations, top open and closed LLMs). Its setting is underspecified instructions revealed over turns, not state updates after a correction, and all models are far above 1B. **Confirmed**; scope note in "scale misapplications".
20. **Qwen3 E2 recall 0.70 short to 0.20 long, deflection 0.25 to 0.70, gold beats foil in every trial.** Reproduces. 3 of the 14 short passes are "I told you my name was Priya.", so hand-strict short is 0.55. The drop and the deflection rise are real. **Confirmed** with a smaller starting number.

## 5. Report statements outside the findings list that are wrong or overstated

- F11: "No other model, in any of the 21 greedy and sampled runs of the other seven, recalled what the user complained about". Two runs retrieved "bored" but in the first person: Qwen3-0.6B sampled seed 1 "At the start, I was just saying "how's it going?" and then I mentioned being bored." and LFM2-350M sampled seed 1 "At the start, I was feeling a bit stuck and bored." Falcon itself passed 1 of 2 runs.
- F5 lists "LFM2-350M (Monday to Tuesday)" as a correction pass; it is a hypothetical example.
- F6 says SmolLM2-360M and Gemma "passed that turn"; the grader fails both (claim 12).
- F4 "Perspective errors peak in the best model": only under a label that misses first-person answers containing "you".
- F3 E1 paragraph: "the instruct weights in plain format also edge out the base model, so SFT added usable conversational skill" (claim 4).
- F10: controls are echo-prone and carry many of the false passes in 2b (LFM2, LFM2.5, Falcon, Gemma, Qwen2.5), so context-failure counts from 350M up are inflated for more than recall.
- F1 "Falcon recalls 0.50 in the main battery": 3 of the 5 passes are the binding-confused summary.

## 6. Hand-adjusted macro scores

Set A, clear cases (greedy 8 flips: LFM2 K_day, R_d4_city, R_d2_name to fail; LFM2 end-question t2 and t3 to pass; Falcon R_d5_multi name and pet to fail; Gemma T_dragon fear to fail. Sampled 11 flips, listed in 2b). Set B **(new)** adds the greedy borderline incidental mentions (LFM2 and Qwen3 T_dragon both checks, Falcon and Gemma T_dragon name, SmolLM2-135M knows_60, Falcon knows_gracias).

| model | greedy (lane) | greedy set A | greedy set B | sampled (lane) | sampled set A |
|---|---|---|---|---|---|
| Falcon-H1-Tiny-90M | 0.49 | 0.46 | 0.40 | 0.57 | 0.47 |
| SmolLM2-135M | 0.15 | 0.15 | 0.11 | 0.22 | 0.21 |
| Gemma-3-270M | 0.36 | 0.31 | 0.27 | 0.28 | 0.28 |
| LFM2-350M | 0.65 | 0.60 | 0.50 | 0.61 | 0.56 |
| LFM2.5-350M | 0.62 | 0.62 | 0.62 | 0.66 | 0.64 |
| SmolLM2-360M | 0.57 | 0.57 | 0.57 | 0.58 | 0.53 |
| Qwen2.5-0.5B | 0.60 | 0.60 | 0.60 | 0.50 | 0.50 |
| Qwen3-0.6B | 0.71 | 0.71 | 0.62 | 0.65 | 0.65 |

The "followup" and "own_answer" columns include turn-0 knowledge checks (knows_paris, knows_gracias, knows_60) and "instruction" includes arr_t0. Dropping them lowers the greedy macros by 0.004 (Qwen3) to 0.056 (SmolLM2-135M) and does not change the order.

## 7. Omissions (new)

- **The controlled Falcon comparison was available and not run.** TII published `tiiuae/Falcon-H1-Tiny-90M-Instruct-Curriculum`, `-Instruct-pre-DPO`, `-Instruct-Curriculum-pre-DPO` and `-Base` on 2026-01-12 (Hub API), same architecture and size as the probed model. The report says Falcon's recipe "differs on every axis at once ... so the probe cannot say which part helps" and leans on Falcon for the SFT-in-pretraining lever. Running the battery and E2 on those four checkpoints would separate SFT-in-pretraining, DPO and the hybrid architecture, and the base checkpoint would give Falcon an E1-style base-vs-instruct row. At 90M each run is minutes on the Mac.
- **TII's own multi-turn number cuts against the lever.** MT-Bench (2-turn) is 4.40 for curriculum + DPO vs 4.33 for SFT-pretrain + DPO (claim 18). The lever table cites only the single-turn IFEval gap.
- **The current small Qwen was not probed.** `Qwen/Qwen3.5-0.8B` (created 2026-02-28) is the model LFM2.5's card compares against and the current version of Max's "wall" family; the probe's wall reference is Qwen2.5-0.5B from 2024.
- **No truncation handling in grading** (46 greedy failures graded on capped replies) and **no human calibration pass** (the report lists the latter as an open question).

## 8. Methodological limits

- **Decoding.** Greedy is the headline. The sampled pass uses each card's settings, which differ enough (T 0.1 to 1.0, repetition penalty on for three models) that model and decoder are confounded. 2 seeds for seven models, 1 for Falcon (with assumed settings).
- **Prompt wording.** One phrasing per question, one fact per recall test, one binding test, three correction tests. E3 shows one system sentence moves LFM2.5 from 0.70 to 1.00, so single-phrasing scores are fragile. The deflection detector is itself phrasing-sensitive. Several E2 conditions score lower at n=0 than at n=1 (SmolLM2-135M plain instruct and base 0.25 vs 0.75, Gemma 0.00 vs 0.25), so the canned acknowledgement right before the question matters for some models; not investigated.
- **Sample size per cell.** Main battery: 1 greedy trial per recall distance per model; 3 correction and 3 role checks per model; one check is 0.048 of macro. E2: 20 trials per pooled cell but 4 distinct facts, so trials are not independent; per-distance cells are 4 trials. Differences below about 0.2 are noise, as the report says, and several of its per-model comparisons are below that.
- **Graders.** Substring presence plus three guards; no negation, hedge, guess-list, hypothetical or "answered the question" detection; the capture guard exempts sentences with "you"; the K_time gold matches any standalone 4. The taxonomy uses a fixed priority order and counts per check.
- **Forced-prefix probe.** The recall foil never appears in context, and in E2 the gold is the only entity of its type in the history, so greedy-from-prefix is an induction-copy test, not proof the model binds the fact to the user. Qwen3's markdown opener zeroes its greedy-gold. The K_time foil " 3" is also the start of legitimate early-arrival times.
- **Token cap.** 120/160 new tokens (60 in E2) cuts most Falcon and LFM2 replies, shortens every model's own history, and produced at least two false fails and one borderline pass.
- **Runtime.** fp32 on MPS, Falcon's Mamba path on the torch fallback, Gemma from the unsloth mirror. Parity with reference kernels was not checked.
- **Single author.** Graders and taxonomy were written and first audited by the agent that ran the probe.

## 9. What probe.md should change

1. Replace "63% retrieval-intact" with the multi-turn figure (48% greedy-gold, 54% without Qwen3's markdown artifact, 90% gold beats foil) and keep the E2 62%.
2. Report deflection on multi-turn final turns separately from controls; count failures per reply; relabel Qwen2.5's R_d5_multi reply; say deflection is one of the two largest classes.
3. Drop "fails at every size" for corrections; report K_day and K_color margins beside K_time and flag the " 3" confound; remove LFM2 from the correction passes; fix "only the turn structure differs".
4. Reframe SmolLM2-135M's retrieval fade as a chat-template effect; drop "SFT added usable conversational skill" from E1.
5. Fix the persona sentence (both ~100M models adopted Captain Pip), role capture (7 to 8 of 8), Qwen3 self-copy ranking, the Falcon "complained about" claim, and the tokens-per-parameter range (LFM2.5 about 79,000).
6. Add grader fixtures for negation, guess lists, hypotheticals, first person plus "you", incidental numbers and restate-without-answering, then regrade; mark or exclude capped replies.
7. Run the battery on the four Falcon-H1-Tiny-90M sibling checkpoints and on Qwen3.5-0.8B; cite TII's MT-Bench row next to its IFEval row.
