# The RC-12 lock: what Max has to do

The lock is the moment the test's rules, bars and hidden test set are frozen and pushed with a public timestamp. It has to happen before any Planck model is scored on RC-12. That ordering is what lets anyone check that the bars were not chosen after seeing Planck's results. The window is Oct 11 to 14.

## Already done

**Your OOD-H review.** You reviewed the 30-thread sample on Oct 10 (85 of 85 probes right). You also re-checked the 11 threads the third review changed (11 of 11, 32 of 32). The human-written test set is fixed: 150 threads, 415 probes. Nothing more to open or answer there.

## What you need to do

**Status, Nov 19:** you answered 1 and 2 on Oct 19: "Yes, lock it (Recommended)" and "Yes, do both (Recommended)". You also picked how the hidden set is built: "One-click separate session (Recommended)". Only 3 is still yours to decide, and you can reverse any item in the list below it before the lock.

**1. Say yes to the lock (answered Oct 19: yes).** The pre-registration says nothing in it binds anyone until you approve it.

Once you say yes, the thresholds that are still marked "proposal" become permanent:

- 0.60 for the Level A content bars and for the LOOKUP claim
- 40 and 0.40 for Level R
- 2% loops
- 50% in the human test
- the -5 OOD-H wording rule

After that, they can never change. If you want to look first, the file is `prereg/RC-12.draft.txt`. Section 19 is the checklist and section 13 is how the hidden set is built.

**2. Say yes to two public timestamps (answered Oct 19: yes to both).** After the push I would make a GitHub release of the lock commit. I would also ask Software Heritage, the public code archive, to save a snapshot of the repo. Both record the date in places git cannot rewrite, which is how the pre-registration proves when it happened. Both count as publishing, so each needs your explicit yes.

**3. Optional: a third copy of the key.** The hidden test set is encrypted. Its key will be kept on the Mac and on the PC, never in the repo. If you want a third copy in your password manager, I will tell you where the key file is so you can add it yourself. I never print the key.

## Decisions I made under your standing permission

You can reverse any of these before the lock.

- **Hidden set size: 640 records instead of 600.** It reuses the dev test's exact layout (608 conversations plus 32 one-turn controls), which keeps every balance exact with no new generator code. More conversations only add power.
- **The hidden set's seed is drawn until the cheater gates pass.** A dry run on fresh random seeds passed the anti-shortcut gates only 10 times in 42, by chance. So the build keeps drawing seeds until one passes, and logs every draw publicly. It uses only fake answer bots, never a model, so this can only make the test harder to game.

  Heads-up: on fresh seeds, the best fixed-rule "shortcut" bot scores about 38 of 100 on the state score (range 34 to 42), and the draw only accepts a seed where it stays below 40. Level R's bar is 40. The bar still holds, because Level R needs the lower end of the confidence interval to clear 40, which the shortcut mix almost never does. The margin is thinner than dev's 36.7 suggested.
- **Weekday, month and language words are shared with the dev test.** Only 7 weekdays exist, and dev already uses 11 months. Every other value, name, template, filler and question in the hidden set is new, short ones like "What's my name?" too. Only a few labels and word lists the code relies on may repeat (the filler labels "question", "request" and "statement", and the lists that keep each item's object word out of its fillers).
- **LFM2-2.6B gets no likelihood row.** Its bf16 self-check failed under a rule written before the run. Changing the rule after seeing the result would be the thing pre-registration forbids.
- **A reported second option stays as the approved rule grades it (Nov 19).** When a reply gives the right answer first and then repeats the user's question about another option, "Green. You asked if it's blue." fails, the same way "Green. You asked whether it's blue." always did, while "Green. You said it should be blue." passes. Treating these alike would mean a new rule, written after the dev numbers were seen, on top of the answer-first rule you approved on Nov 3. I left it as is (prereg s6 (viii)). Changing it is a grader change, so it would have to happen before the hidden set is built.
- **"I couldn't find it" now counts as abstaining, on the LOOKUP test only (Nov 19).** The training pipeline teaches Planck to say "I couldn't find Thursday on the new rota." or "Nothing came up for that." when a lookup comes back empty, and the test's abstain grader marked every such reply wrong on all 48 dev LOOKUP abstain questions. I fixed it before the lock, but only for the LOOKUP family's abstain question: the other abstain questions and the human-written test keep the grader as it was, because changing it there would change a check the hidden human test runs. It counts the forms the pipeline accepts, with or without "I" ("I couldn't find it.", "Couldn't find it.", "We couldn't find it.", "No results.", "Nothing came up."). A few right replies still fail, for example "I checked, but I couldn't find it." and "No I couldn't find it" without the comma (prereg s6 (x) lists them). Naming the other table's value still fails. It changed no score on the stored dev runs (prereg s6 (x)). A later review (Nov 22) found wrong replies the widened forms let through, such as "Thursday isn't, honestly, missing from the new rota.", "Only the old one has that problem; the new one has Thursday." and "I haven't found any gap."; they fail now, at the cost of a few rare right ones ("We haven't found Thursday on the new rota."). Idioms such as "I don't see why not: Thursday is on the new rota." still pass, as "There's no problem: ..." always has. A second round of that review (Nov 22-23) closed more of the same ("Thursday isn't, as far as I can tell, missing from the new rota.", "Not a single day is missing from the new rota.", "Only the old one has Thursday crossed out.") and passed "None, Thursday is missing from the new rota."; it changed no stored dev result. Because every round of patching opened new wrong passes, on Nov 23 I replaced those rules with a simpler one that fails closed (lock step A2d): any negation word before the cue in its sentence ("not", "nothing", "nobody", "none", "hardly", "isnt" ...) makes it not count, except a short opening such as "I'm afraid not," or "Sorry," at the start; "Only the other one has X." and "X is only on the other one." count only in that plain shape. The price is a few right replies that now fail ("Nobody - Thursday is missing from the new rota.", "None, Thursday is missing ...", "Only the old one has Thursday on it."); they are listed in prereg s6 (x). A review of the built rule the same day found more wrong replies it let through (a "not" glued to the word before it by bold marks, "No-one is missing from the new rota.", "No records on the new rota are missing ...", a question mark or struck-through text the rule missed, "Thursday is missing from (the old rota)."); I closed them the same fail-closed way and recorded what it left open ("Thursday is only on the old one too.", "No - one is missing ...", "Nothing turned up in the new rota as a gap."). It changed no passing stored reply. You can drop it or extend it before the hidden set is built.
- **A name from the other table written in lower case now fails too (Nov 19).** The grader matched a name only with its capital letter, so an abstain reply that named the other table's value in lower case ("the old one says hugo") was not caught, and a fake bot that looks up the named table correctly and did this scored 0.667 on the dev LOOKUP test, above the 0.60 bar. I fixed it before the lock, on the LOOKUP abstain question only: there a name or value from the question's list counts in any case ("hugo", "HUGO", "hugo's"). Everywhere else names are still matched as written, because the other questions include the human-written test, which nobody may check before the lock, and some pet names there are ordinary words ("pumpkin", "biscuit"). It changed no score on the stored dev runs (9 replies that were already wrong now also fail for naming a value), and the fake now scores 0 (prereg s6 (xi)). It brings one rule for the hidden set: its names and cities may not be ordinary words such as Will or Grace, and a test checks what it can of that. You can undo it before the hidden set is built.

## What happens next without you

1. The grader gaps the OOD-H review found (for example "You asked if X" being marked wrong) are settled: five fixed, two recorded as known limits (lock step A2, finished Nov 19), and the "I couldn't find it" gap and the lower-case name above fixed too (A2b, A2c). The PC runs listed in `rc12/LOCK_STEPS.md` A2 are still to do.
2. A separate session builds and encrypts the hidden set.
3. Then the lock commit, once you say yes.

Exact steps: `rc12/LOCK_STEPS.md`.
