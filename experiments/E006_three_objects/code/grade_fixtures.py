"""Fixtures for the E004 free-generation grader (gen_grade.py). Used by mutation_e004.py and selfcheck_e004.py.
CTX: name -> (gold, pool, phrase, head, alias, in-context candidates). The candidates are NOT used by the real
grader (clause 4 reads the whole pool); they exist so a "candidates only" mutant can be expressed and killed.
FIX: (ctx, reply, stop, expected strict, expected failing clauses)."""
WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
COLOURS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "silver", "gray"]
NUMS = ["2", "3", "4", "5", "6", "7", "8", "9"]

CTX = {
    "wk": ("Friday", WEEK, "piano lesson", "lesson", None, ["Monday", "Friday", "Wednesday"]),
    "mo": ("July", MONTHS, "camping trip", "trip", None, ["March", "July"]),
    "may": ("May", MONTHS, "wedding", "wedding", None, ["May", "June"]),
    "num": ("5", NUMS, "hotel floor", "floor", None, ["3", "5"]),
    "col": ("gray", COLOURS, "tent", "tent", None, ["gray", "red"]),
    "al": ("Friday", WEEK, "tax meeting", "meeting", "Mr. Adeyemi", ["Tuesday", "Friday"]),
}

GOOD = [  # correct natural answers: every one must pass, under every clause
    ("wk", "Your piano lesson is on Friday now.", "newline"),
    ("wk", "It's on Friday now.", "newline"),
    ("wk", "Friday, after the last change.", "newline"),
    ("wk", "You moved it to Friday.", "newline"),
    ("wk", "The latest plan says Friday.", "user"),
    ("wk", "Your lesson is on Friday, as you said.", "eos"),
    ("wk", "Friday.", "newline"),
    ("wk", "friday", "newline"),
    ("wk", "For the piano, it's Friday.", "newline"),          # "no," inside "piano," is not the interjection
    ("wk", "Friday, bring your notes.", "newline"),            # "not" inside "notes"
    ("wk", "Friday, another week away.", "newline"),           # "not" inside "another"
    ("mo", "Your camping trip is in July, so you may want to book early.", "newline"),   # modal "may"
    ("may", "The wedding is in May now.", "newline"),
    ("num", "The hotel floor is number five.", "newline"),
    ("num", "It's on the 5th floor.", "newline"),
    ("num", "Floor 5.", "newline"),
    ("col", "The tent is grey now.", "newline"),
    ("al", "Your meeting with Mr. Adeyemi is on Friday.", "newline"),
]

BAD = [  # (ctx, reply, stop, failing clauses)
    ("wk", "", "newline", [1, 3]),                                                     # empty
    ("wk", "   ", "eos", [1, 3]),                                                      # blank
    ("wk", "Your piano lesson is on Friday and the teacher said the room will be ready early so bring "
           "the sheet music along", "cap", [1]),                                       # capped, otherwise clean
    ("wk", "It's on Friday. It's on Friday. It's on Friday.", "cap", [1, 2]),          # loop at the cap
    ("wk", "It's on Friday. It's on Friday.", "newline", [2]),                         # repeated 3-gram
    ("wk", "Friday friday friday", "newline", [2]),                                    # word 3 times in a row
    ("wk", "Sure, I can help with your lesson.", "newline", [3]),                      # no value
    ("wk", "Fridays are best for your lesson.", "newline", [3]),                       # not a whole-word gold
    ("wk", "It's on Sunday.", "newline", [3, 4]),                                      # wrong value
    ("wk", "Friday or Monday.", "newline", [4]),                                       # two candidates
    ("wk", "Monday, then Friday.", "newline", [4]),                                    # other value before gold
    ("wk", "Friday, or Sunday at the latest.", "newline", [4]),                        # guess outside candidates
    ("wk", "Friday is not the day.", "newline", [5]),                                  # negation after the gold
    ("wk", "It is not on Friday.", "newline", [5]),                                    # negation before the gold
    ("wk", "The lesson isn't on Friday.", "newline", [5]),
    ("wk", "Friday, no longer the plan.", "newline", [5]),
    ("wk", "Friday never works for the lesson.", "newline", [5]),
    ("wk", "No, Friday.", "newline", [5]),
    ("wk", "Friday instead.", "newline", [5]),
    ("wk", "Friday rather than later.", "newline", [5]),
    ("wk", "Is it Friday?", "newline", [6]),
    ("wk", "I think it's Friday.", "newline", [6]),
    ("wk", "Probably Friday.", "newline", [6]),
    ("wk", "Maybe Friday.", "newline", [6]),
    ("wk", "It could be Friday.", "newline", [6]),
    ("wk", "My piano lesson is on Friday.", "newline", [7]),                           # user's voice
    ("wk", "My weekly lesson is on Friday.", "newline", [7]),                          # object 2 words after "my"
    ("al", "It's my Adeyemi slot, Friday.", "newline", [7]),                           # alias word after "my"
    ("wk", "Assistant: Your lesson is on Friday.", "newline", [7]),                    # role tag
    ("wk", "User: Friday works.", "newline", [7]),
    ("may", "You may want to check the date.", "newline", [3]),                        # modal is not the month
    ("num", "It's on floor three.", "newline", [3, 4]),
    ("col", "It's red now.", "newline", [3, 4]),
]

FIX = [(c, r, s, True, []) for c, r, s in GOOD] + [(c, r, s, False, f) for c, r, s, f in BAD]


def run(G, fixtures=FIX):
    """grade every fixture with module G (gen_grade or a patched copy); -> list of mismatches."""
    bad = []
    for ctx, reply, stop, want_s, want_f in fixtures:
        gold, pool, phrase, head, alias, cands = CTX[ctx]
        g = G.grade(reply, stop, gold, pool, G.obj_words_of(phrase, head, alias), cands=cands)
        if g["strict"] != want_s or g["fails"] != want_f:
            bad.append((ctx, reply, stop, want_s, want_f, g["strict"], g["fails"]))
    return bad
