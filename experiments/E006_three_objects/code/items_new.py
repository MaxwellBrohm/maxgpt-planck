"""New likelihood-battery items (E001): controls that attribute the correction failure.

REPORT 7.3 step 1 asks for: two-slot and no-update twins, U_same on every model, a position
control, and a key-in-correction item. This file builds them in two item families
(an appointment weekday and a car colour), so no result rests on one item family.

Every scenario draws its values and filler exchanges ONCE and builds all variants from the
same draw, so variants are paired (same values, same fillers, same distance).

Variants (k = number of corrections; all k = 1 unless named):
  same_k1/k2/k3  original and every correction use IDENTICAL key wording, original in turn 1.
                 Only order separates the values: a first-value head picks the original.
  pos            same_k1 with 3 unrelated exchanges BEFORE the original (position control:
                 the original is no longer the first turn).
  keyorig        the answer key ("dentist appointment is on" / "color of my new car is")
                 appears only in the ORIGINAL statement; the correction uses other words.
                 This is the confound in the old main U items, rebuilt cleanly.
  keycorr        mirror of keyorig: the key appears only in the CORRECTION.
                 keycorr - keyorig = how much key overlap, not order, decides the answer.
  neutral        keyorig wording, but a question and answer prefix sharing no key n-gram.
  noupd          twin of same_k1 whose second statement is about a DIFFERENT object
                 (haircut / bike) with the same value type. Gold = the original value.
                 "latest value wins" fails it; "first value wins" passes it.
  twoslot        same_k1, then a later statement about a different object with a third value.
                 Gold = the corrected value. "latest wins" and "first wins" both fail it.

Rendering: items are stored as turns + question + answer prefix, and rendered by lik.py in
plain "User:/Assistant:" form (as the old battery) or in the model's own chat template
(format control).
"""
import random, re

import items as I

DAYS = I.DAYS
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "silver", "gray"]

# Extra filler exchanges: no names, numbers, cities, pets, weekdays, colours, cars, bikes,
# dentists, haircuts or appointments, so they cannot leak or confuse a value.
EXTRA_DISTRACTORS = [
    ("How do I stop my glasses from fogging up?",
     "Wash them with a drop of dish soap, let them air dry, and make sure your mask fits snugly over your nose."),
    ("What's a simple way to make a room feel bigger?",
     "Use a large mirror, keep the floor clear, and choose light curtains that let in daylight."),
    ("How can I remember people's names better?",
     "Repeat the name when you are introduced, and link it to something about the person."),
    ("Is it better to stretch before or after running?",
     "Do gentle movement to warm up before, and save longer stretches for after the run."),
    ("How do I clean a cast iron pan?",
     "Rinse it with hot water, scrub with a brush, dry it well and rub in a thin layer of oil."),
    ("Why do cats purr?",
     "Cats purr when they are relaxed, but also to calm themselves when they are stressed or hurt."),
    ("What's a good way to start journaling?",
     "Write a few lines each evening about one thing that happened and how it made you feel."),
    ("How can I make my coffee less bitter?",
     "Use slightly coarser grounds, water just below boiling, and a shorter brewing time."),
    ("Why do onions make you cry?",
     "Cutting them releases a gas that reacts with the moisture in your eyes and irritates them."),
    ("How do I get better at public speaking?",
     "Practice out loud, record yourself, and start with small groups before larger ones."),
    ("What should I pack for a day hike?",
     "Water, snacks, sunscreen, a light jacket, a small first aid kit and a charged phone."),
    ("How do I keep herbs fresh longer?",
     "Trim the stems, stand them in a glass of water in the fridge, and loosely cover the leaves."),
]


def pool(exclude_words):
    out = []
    for q, a in I.DISTRACTORS + EXTRA_DISTRACTORS:
        t = (q + " " + a).lower()
        if any(re.search(r"\b" + w.lower() + r"\b", t) for w in exclude_words):
            continue
        out.append((q, a))
    return out


FAMILIES = {
    "day": dict(
        values=DAYS,
        key=lambda v: (f"My dentist appointment is on {v}.", f"Okay, your dentist appointment is on {v}."),
        key_corr=lambda v: (f"Actually, my dentist appointment is on {v} now.", f"Got it, your dentist appointment is on {v}."),
        nokey_orig=lambda v: (f"I booked the dentist for {v}.", f"Okay, noted: {v}."),
        nokey_corr=lambda v: (f"Actually, I moved the dentist to {v}.", f"Got it, moved to {v}."),
        other=lambda v: (f"My haircut is on {v}.", f"Okay, your haircut is on {v}."),
        question="What day is my dentist appointment?", prefix="Your dentist appointment is on",
        neutral_q="Which day do I need to go to the dentist?", neutral_prefix="That would be",
        key_words=["dentist appointment", "appointment is on"],
        other_words=["haircut"],
    ),
    "color": dict(
        values=COLORS,
        key=lambda v: (f"The color of my new car is {v}.", f"Okay, the color of your new car is {v}."),
        key_corr=lambda v: (f"Actually, the color of my new car is {v} now.", f"Got it, the color of your new car is {v}."),
        nokey_orig=lambda v: (f"I ordered the car in {v}.", f"Okay, noted: {v}."),
        nokey_corr=lambda v: (f"Actually, I switched the car to {v}.", f"Got it, switched to {v}."),
        other=lambda v: (f"The color of my new bike is {v}.", f"Okay, the color of your new bike is {v}."),
        question="What color is my new car?", prefix="The color of your new car is",
        neutral_q="Which paint should I expect on the one I ordered?", neutral_prefix="That would be",
        key_words=["color of my new car", "color of your new car"],
        other_words=["bike"],
    ),
}
EXCLUDE = DAYS + COLORS + ["grey", "pink", "brown", "car", "bike", "dentist", "haircut", "appointment"]

VARIANTS = ["same_k1", "same_k2", "same_k3", "pos", "keyorig", "keycorr", "neutral", "noupd", "twoslot"]


def build(n_scen=32, distances=(0, 4, 10), seed=2026, families=("day", "color")):
    """Return item dicts: {task, fam, var, d, sid, turns, question, prefix, cands, roles}.
    cands: {label: " value"}; 'gold' is always the right answer. roles maps each candidate
    label to where its value sits (for the oracle checks in validate_items.py)."""
    rng = random.Random(seed)
    P = pool(EXCLUDE)
    items = []
    for fam in families:
        F = FAMILIES[fam]
        for d in distances:
            for s in range(n_scen):
                sid = f"{fam}|{d}|{s}"
                v = rng.sample(F["values"], 4)
                fills = rng.sample(P, 3 + 3 + d)  # 3 between-statement fillers, 3 leading (pos), d tail
                f1, f2, f3 = fills[0], fills[1], fills[2]
                pre3 = fills[3:6]
                tail = fills[6:]
                Q, PRE = F["question"], F["prefix"]

                def add(var, turns, cands, q=Q, pre=PRE, k=None):
                    items.append(dict(task="N_" + var, fam=fam, var=var, d=d, sid=sid, k=k,
                                      turns=turns, question=q, prefix=pre,
                                      cands={lab: " " + val for lab, val in cands.items()}))

                # U_same, k = 1..3
                for k in (1, 2, 3):
                    turns = [F["key"](v[0])]
                    for i in range(1, k + 1):
                        turns += [(f1, f2, f3)[i - 1], F["key_corr"](v[i])]
                    turns += tail
                    c = {"gold": v[k], "orig": v[0]}
                    if k >= 2:
                        c["prev"] = v[k - 1]
                    add(f"same_k{k}", turns, c, k=k)
                # position control
                add("pos", list(pre3) + [F["key"](v[0]), f1, F["key_corr"](v[1])] + tail,
                    {"gold": v[1], "orig": v[0]}, k=1)
                # key placement pair
                add("keyorig", [F["key"](v[0]), f1, F["nokey_corr"](v[1])] + tail, {"gold": v[1], "orig": v[0]}, k=1)
                add("keycorr", [F["nokey_orig"](v[0]), f1, F["key_corr"](v[1])] + tail, {"gold": v[1], "orig": v[0]}, k=1)
                # neutral prefix (keyorig wording)
                add("neutral", [F["key"](v[0]), f1, F["nokey_corr"](v[1])] + tail, {"gold": v[1], "orig": v[0]},
                    q=F["neutral_q"], pre=F["neutral_prefix"], k=1)
                # no-update twin of same_k1: second statement is another object
                add("noupd", [F["key"](v[0]), f1, F["other"](v[1])] + tail, {"gold": v[0], "later": v[1]}, k=0)
                # two-slot: correction, then another object's value comes last
                add("twoslot", [F["key"](v[0]), f1, F["key_corr"](v[1]), f2, F["other"](v[2])] + tail,
                    {"gold": v[1], "orig": v[0], "other": v[2]}, k=1)
    return items


# expected position of each candidate's value in the statement sequence, for validation:
# "first" = mentioned first among candidates, "last" = mentioned last
ROLE_ORDER = {
    "same_k1": ["orig", "gold"], "same_k2": ["orig", "prev", "gold"], "same_k3": ["orig", "prev", "gold"],
    "pos": ["orig", "gold"], "keyorig": ["orig", "gold"], "keycorr": ["orig", "gold"], "neutral": ["orig", "gold"],
    "noupd": ["gold", "later"], "twoslot": ["orig", "gold", "other"],
}

if __name__ == "__main__":
    from collections import Counter
    it = build()
    print(len(it), Counter(x["var"] for x in it))
    for x in it[:9]:
        print("-" * 70)
        print(x["task"], x["sid"], x["cands"])
        print(I.transcript(x["turns"], x["question"], x["prefix"]))
