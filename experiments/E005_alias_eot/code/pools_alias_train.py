"""E005 change (a): TRAINING-only alias vocabulary, exactly the lists of notes.txt (a). An alias is
title + " " + surname. The aliased object's ORIGINAL renders its object phrase as "<phrase> <join>" (the E004
eval's mechanism, with training joins); an alias correction names the alias and never the object.
The four honorifics are shared with the E004 eval by design; every surname (so every honorific-plus-surname
pair), every role title, join and template is disjoint from the E004 eval, dev and probe draws
(checks_e005.check_alias_pools and purity_e005.py check this; heldout_e004.py lists the eval side)."""

SURNAMES = ["Abernathy", "Bergstrom", "Chaudhry", "Delgado", "Eriksen", "Fontaine", "Gallagher", "Hoffmann",
            "Iwasaki", "Jablonski", "Kariuki", "Lombardi", "Moreau", "Nakamura", "Oyelaran", "Rasmussen",
            "Santoro", "Thornbury", "Vasquez", "Wojcik", "Yilmaz", "Zielinski", "Mancini", "Beaumont"]
HONORIFICS_TRAIN = ["Mr.", "Mrs.", "Ms.", "Dr."]           # shared with the eval (heldout_e004.HONORIFICS)
ROLES = ["Pastor", "Professor", "Chef", "Captain"]         # in no E004 draw
P_HONORIFIC = 2 / 3                                        # else a role; uniform within each group
TITLES = HONORIFICS_TRAIN + ROLES

# joins: the object phrase of the aliased object's original becomes "<phrase> <join>"; {a} = alias.
# The eval joins are "with", "from", "run by"; no training join contains with, from, run or by.
JOINS = {
    "weekday": ["arranged through {a}", "that {a} set up", "that {a} looks after"],
    "colour": ["that {a} is handling", "that {a} is taking care of", "that {a} quoted me for"],
    "month": ["that {a} is planning", "that {a} is organizing", "that {a} is coordinating"],
    "city": ["that {a} is putting on", "that {a} is in charge of", "that {a} is hosting"],
}

# alias corrections: no object word, no it / they / them / one; the name is the only link to the object
ALIAS_CORR = {
    "weekday": ["{a} bumped my booking over to {v}.", "{a} texted to say {v} is better.",
                "{a} switched my slot to {v}.", "{a} left a voicemail saying {v}.", "{a} has me down for {v}.",
                "{a} asked if {v} would suit me, and I said yes."],
    "colour": ["{a} tells me I'm getting {v}.", "{a} swapped over to {v} for me.", "{a} emailed to confirm {v}.",
               "{a} ran short and offered {v}, which I took.", "{a} recommends {v} and I agreed.",
               "{a} has {v} ready for me."],
    "month": ["{a} decided on {v} for all of us.", "{a} booked everyone for {v}.", "{a} wrote that {v} is final.",
              "{a} settled on {v} after the vote.", "{a} called around and picked {v}.", "{a} set the date for {v}."],
    "city": ["{a} switched the venue to {v}.", "{a} announced {v} as the location.", "{a} has booked a hall in {v}.",
             "{a} emailed everyone about {v} being the host city.", "{a} picked {v} in the end.",
             "{a} confirmed {v} a minute ago."],
}

# ALIAS block design (notes (a)); the checks keep their own copies of these numbers
CASES = {"latest": 0.60, "earlier": 0.15, "other": 0.25}
PLACEMENTS = ["adjacent", "filler", "other_obj"]           # 1/3 each in every case
P_BOTH_ALIASED = 1 / 3
N_B_CORR = [0, 1, 1, 2]                                    # B's corrections when B is not aliased: 1/4, 1/2, 1/4


def alias_sizes():
    """{(vtype or '*', pool): n} of every alias pool, for coverage."""
    out = {("*", "surname"): len(SURNAMES), ("*", "title"): len(TITLES)}
    for vt in JOINS:
        out[(vt, "join")] = len(JOINS[vt])
        out[(vt, "alias_corr")] = len(ALIAS_CORR[vt])
    return out
