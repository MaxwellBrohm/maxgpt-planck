# E002 audit (independent, 2026-09-24)

Recomputed from the raw outputs by an auditor agent that loaded no model.

**E002 audit (SmolLM2-135M-I, 5 seeds). Recomputed from the raw out/*.jsonl, no model was loaded.**

**Verdict:** the pre-registered pass holds, and it reproduces. The reading "updating is trainable" is not supported beyond dialogues where the correction repeats the question's wording. A cheap rule passes both the pass rule and the crossed control. On the controls where the correction is worded differently, several seeds fall below chance. The owner, two-hop and perspective gains come straight from the training data.

**1. Claims confirmed (results.json matches exactly)**
- **Plain LW10 / TS10 / NU10:**
  - s0 1.000 / 1.000 / 1.000
  - s1 1.000 / 1.000 / 1.000
  - s2 0.953 / 1.000 / 1.000
  - s3 0.995 / 1.000 / 1.000
  - s4 1.000 / 1.000 / 1.000
  - base 0.000 / 0.016 / 1.000
  - 5/5 seeds pass.
- **Chat LW10:** 0.958, 0.995, 0.938, 0.927, 0.969.
- **Lock-in:** step 50 for all seeds, which is the first probe. s1 only just cleared the bar there (0.84 / 0.91 / 1.00).
- **Crossed pair, likelihood, d10:** plain s0r 1.00, s1 0.94, s2 0.89, s3 0.94, s4 1.00; chat 1.00 for every seed; base plain 0.00.
- **Crossed pair, free generation:** plain 1.00, 0.86, 0.86, 0.94, 0.98; chat 1.00, 0.95, 0.98, 0.98, 0.98.
- **Free generation on same_k1 / twoslot / noupd:** 1.00 for every seed in plain. In chat, s2 twoslot is 0.984, so "1.00 for every seed" is true for plain only.
- **Seed 0 rerun (s0 vs s0r):** 0 decision flips over 5,873 shared scorings, largest difference 0.13 nats.
- **Seeds really differ:** s0 vs s1 has 109 flips on new/plain. The only source of difference is the training data sample and order. There is no dropout (attention_dropout 0) and nothing is newly initialized, so `torch.manual_seed` has no effect.
- **Knowledge (kbig, n = 441):** accuracy change −1.36, −0.23, −1.81, −0.91 and −3.17 points. My independent bootstrap reproduces the confidence intervals.
  - All 5 point estimates are negative. s4's interval excludes 0 (5 items gained, 19 lost, exact McNemar p = 0.007).
  - khard40 drops 2.5 to 10 points; s2's interval excludes 0.
  - The log-prob margin rose by 0.40 to 0.62 nats while accuracy fell. That is the model becoming more confident, not gaining knowledge.
  - The "t" interval actually uses z = 1.96. This makes no difference at n = 441 and is about 3% too narrow at n = 40.
- **Owner / two-hop / perspective:** 0.625 → 1.00, 0.094 → 1.00, 0.094 → 0.84 to 0.97. The numbers are right, but see section 3.

**2. Leakage**
- **Wording: held out, as claimed.**
  - No whole sentence frame is shared (validate.py).
  - I compared the 320 scored d10 prompts (value-bearing turns, question and prefix, speaker tags stripped) against all 32,020 training dialogues. They share 15.1% of 3-word sequences, 1.4% of 4-word sequences and none of 5 words.
  - Every shared 4-word sequence is a correction ending such as "is purple now." The longest shared run after replacing values and objects with placeholders is 3 words plus punctuation, for example "switched to <v>."
  - Names, objects and filler exchanges are disjoint. The result is not sentence-level copying.
- **Structure: fully leaked.**
  - Every scored item type is a training type: same_k1/k2/k3 is upd with k = 1..3; twoslot, noupd and noupd_incid have direct twins.
  - The weekday and colour lists are identical in training and eval, and so are the candidate counts.
  - Training distances run uniformly from 0 to 10, and 202 to 250 examples per seed are an exact scored structure at d = 10.
  - The plain User:/Assistant: format is the training format, and the pass rule reads plain only.
  - 38 to 40% of each seed's 6,404 examples are an upd, twoslot or noupd structure. Lock-in happened within 800 examples, about 100 of each type. Final training loss is about 5e-5.
  - Only the crossed item (both objects corrected) is a structure the model never trained on.
- **Conclusion:** the wording is held out but the task is not. Nothing was tested outside the training range: no distance above 10, no more than 3 corrections, no third object, no new value type.

**3. Owner, two-hop and perspective were trained directly.** These three families are 40% of the training mix (bind 14%, twohop 13%, persp 13%), built the same way as the old battery.
- **Owner:** training has "My {pet} is called A and my {rel}'s {pet} is called B" in random order. The eval has "my cat is named A, and my sister's cat is named B".
- **Two-hop:** the layout matches, a names turn, then d//2 fillers, then a jobs turn.
- **Perspective:** training has "Hey, I'm U. My friend O recommended you." The eval has "Hi, I'm U. My sister S told me about you."
- validate.py's own log lists the two shared 5-grams, and both come from these families.

The rise is in-distribution training, not generalization.

**4. Cheap-rule check on the real items at d10**

| Rule | LW10 / TS10 / NU10 | Other E001 controls | cross_A / cross_B / pair |
|---|---|---|---|
| First mention | 0 / 0 / 1 | not listed | 0 / 0 / 0 |
| Last mention | 1 / 0 / 0 | not listed | 0 / 1 / 0 |
| Last "Actually" turn, else first mention | 1 / 1 / 1 | 1.00 on keyorig, keycorr, neutral, pos, incid | 0 / 1 / 0 |
| Copy the value after the last occurrence of the prefix's last 3 words ("appointment is on", "new bike is") | 1 / 1 / 1 | incid 1.00; keyorig 0.00; neutral 0.00 | 1 / 1 / 1 (also at d0 and d4) |
| Turn with most words in common with the question, latest wins | 1 / 1 / 1 | 1.00 except keyorig 0.00 | 1 / 1 / 1 |

- The crossed control does defeat the "Actually" rule. Training contains no "Actually" at all.
- **Refuted:** the claim that the crossed control defeats the cheap rules. Both wording-match rules score 1.00 on it.
- validate.py only ever tested first, last, first-correction and last-correction rules, even though E001 had found that key-phrase matching is a strong driver of the answer.
- cross_B's gold is always the last-mentioned value, so cross_B cannot catch a recency rule; the pair really only tests cross_A. Every fine-tuned cross_A miss (likelihood and generation) picks the other object's correction, which is the last mention. Misses cluster at d0, where cross_A is only 0.55 to 0.83.

**Items that separate wording-matching from real updating, at d10.** These scores are pooled away or missing from tables.txt.

| Item | s0 | s1 | s2 | s3 | s4 | Notes |
|---|---|---|---|---|---|---|
| E001 keyorig, colour, plain | 0.69 | 0.00 | 0.44 | 0.06 | 0.06 | day is 0.81 to 1.00 |
| E001 keyorig, chat render | 0.73 | 0.17 | 0.27 | 0.02 | 0.28 | in results.json, not in tables |
| uprobe U_neutral, all k | 0.43 | 0.54 | 0.26 | 0.44 | 0.74 | never tabled |
| U_neutral, k = 3 | 0.00 | 0.06 | 0.00 | 0.00 | 0.34 | misses pick the previous value |
| Old U, k = 3 | 0.56 | 0.94 | 0.44 | 0.41 | 0.97 | table pools k and d4 to d10 into 0.82 to 0.99 |

- U_same, where the correction repeats the key wording, stays at 0.99 to 1.00.
- On the same seeds, the verbatim same_k3 scores 0.86 to 1.00, while old U k = 3 falls to 0.41 to 0.56 on s0, s2 and s3.
- Under E001's own paired all5 metric at d10 (0.84, 0.50, 0.62, 0.53, 0.53), only 1 of 5 seeds reaches 0.8.

**5. Graders**
- **Likelihood scorer:** sound. Ties, NaN and missing gold all fail. Every candidate in the scored sets is a single token, and no file has a tie. The mutation tests exist and kill ties, first-foil-only, gold-only and empty.
- **Free-generation strict grader:** 4 mutation tests exist (ignores other candidates, ignores negation, passes empty, lenient grader uses last mention). Not covered, all of which pass as correct today:
  - repetition loops at the 40-token cap, e.g. "Sunday. Sunday. …" or "My gray gray gray …": s1 chat 1, s2 chat 7, s4 chat 3;
  - negation after the gold;
  - questions and hedges;
  - guesses outside the candidate set, which inflates only the base;
  - wrong speaker: 14 to 62 chat replies per seed say "My new car is X" or "My X.", as if the model were the user (s2: 62 of 320).
- Plain free generation is unaffected: all 320 plain replies for every seed are a bare "Value.", the training answer format. That collapse from the base's full sentences is itself collateral damage.
- Free generation was never run on keyorig, neutral or U_neutral, so it adds no evidence against wording-matching.

**6. Gaps**
- **Chat probe:** the guard killed it for swap growth of 922 MB. It has only 12 lines of base transcript and nothing for any fine-tuned model. It is re-queued in queue_e002c.sh after E003. Conversational collateral damage is unmeasured.
- **360M arm:** pre-registered with 3 seeds and part of the pre-registered reading, but it never ran and is in no queue.
- **Base crossed chat file is truncated:** the rescore hit its wall-clock ceiling. It has 184 of 384 records, day family only, and d10 n = 28. The tables print these numbers without marking them partial.
- s0 has no crossed scores; s0r stands in, which is acceptable given the 0-flip reproduction.
- The notes say no training example was over 768 tokens. In fact s1 and s3 each rejected 1, and the longest kept example was 763 tokens, not about 733. Immaterial.
- **E003 inherits the flaw:** its "per-object tracking" label (pass AND cross_pair ≥ 0.8) is satisfied by the prefix-copying rule.

**What E002 shows:** within at most 800 examples, the model reliably switches from first-value-wins to latest-value-wins. This holds on dialogues shaped exactly like the training data with new words, across 5 seeds and both renders, in free generation, at a closed-book cost of 0.2 to 3.2 points.

**What it does not show:** a general updating skill, transfer to new task structures, or binding gains beyond direct training. Before drawing that conclusion, add pre-registered controls where the correction and the question share no key wording (keyorig in both families, U_neutral with 1 to 3 corrections), plus structures outside training (4 or more corrections, distance 20, three objects), reported per family and per render.

My check scripts are in /private/tmp/claude-501/-Users-brohm-Documents/de64e45b-e167-412e-8ebf-84d92eb12068/scratchpad/e002/:
- recompute.py
- shortcuts.py
- leak.py
- know.py
- struct.py
