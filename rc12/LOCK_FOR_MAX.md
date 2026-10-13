# The RC-12 lock: what Max has to do

The lock is the moment the test's rules, bars and hidden test set are frozen and pushed with a public timestamp. It has to happen before any Planck model is scored on RC-12. That ordering is what lets anyone check that the bars were not chosen after seeing Planck's results. The window is Oct 11 to 14.

## Already done

**Your OOD-H review.** You reviewed the 30-thread sample on Oct 10 (85 of 85 probes right). You also re-checked the 11 threads the third review changed (11 of 11, 32 of 32). The human-written test set is fixed: 150 threads, 415 probes. Nothing more to open or answer there.

## What you need to do

**1. Say yes to the lock (needed).** The pre-registration says nothing in it binds anyone until you approve it.

Once you say yes, the thresholds that are still marked "proposal" become permanent:

- 0.60 for the Level A content bars and for the LOOKUP claim
- 40 and 0.40 for Level R
- 2% loops
- 50% in the human test
- the -5 OOD-H wording rule

After that, they can never change. If you want to look first, the file is `prereg/RC-12.draft.txt`. Section 19 is the checklist and section 13 is how the hidden set is built.

**2. Say yes to two public timestamps (needed; best answered together with 1).** After the push I would make a GitHub release of the lock commit. I would also ask Software Heritage, the public code archive, to save a snapshot of the repo. Both record the date in places git cannot rewrite, which is how the pre-registration proves when it happened. Both count as publishing, so each needs your explicit yes.

**3. Optional: a third copy of the key.** The hidden test set is encrypted. Its key will be kept on the Mac and on the PC, never in the repo. If you want a third copy in your password manager, I will tell you where the key file is so you can add it yourself. I never print the key.

## Decisions I made under your standing permission

You can reverse any of these before the lock.

- **Hidden set size: 640 records instead of 600.** It reuses the dev test's exact layout (608 conversations plus 32 one-turn controls), which keeps every balance exact with no new generator code. More conversations only add power.
- **The hidden set's seed is drawn until the cheater gates pass.** A dry run on fresh random seeds passed the anti-shortcut gates only 10 times in 42, by chance. So the build keeps drawing seeds until one passes, and logs every draw publicly. It uses only fake answer bots, never a model, so this can only make the test harder to game.

  Heads-up: on fresh seeds, the best fixed-rule "shortcut" bot scores about 38 of 100 on the state score (range 34 to 42), and the draw only accepts a seed where it stays below 40. Level R's bar is 40. The bar still holds, because Level R needs the lower end of the confidence interval to clear 40, which the shortcut mix almost never does. The margin is thinner than dev's 36.7 suggested.
- **Weekday, month and language words are shared with the dev test.** Only 7 weekdays exist, and dev already uses 11 months. Every other value, name, template, filler and question in the hidden set is new, short ones like "What's my name?" too. Only a few labels and word lists the code relies on may repeat (the filler labels "question", "request" and "statement", and the lists that keep each item's object word out of its fillers).
- **LFM2-2.6B gets no likelihood row.** Its bf16 self-check failed under a rule written before the run. Changing the rule after seeing the result would be the thing pre-registration forbids.

## What happens next without you

1. I settle the grader gaps the OOD-H review found (for example "You asked if X" being marked wrong). My recommendation is to fix them before the hidden set is built.
2. A separate session builds and encrypts the hidden set.
3. Then the lock commit, once you say yes.

Exact steps: `rc12/LOCK_STEPS.md`.
