"""Training dialogues for the repaired ft_test (E002, REPORT 7.4).

What changed from research/capacity_probe/ft_test.py gen():
  1. The shortcut is closed. In the old generator every update item had ONE event, so the gold
     was always the most recently mentioned weekday. Here the update family mixes
       upd          one object, k = 1..3 corrections, ask it            gold = latest (also the last mention)
       twoslot      object A + corrections, then object B's value later, ask A
                                                                      gold = A's latest (NOT the last mention)
       twoslot_b    same dialogue, ask B                               gold = B (the last mention)
       noupd        A stated, later B stated, no correction, ask A     gold = A (the FIRST mention)
       noupd_b      same dialogue, ask B                               gold = B (the last mention)
       noupd_incid  A stated, later an incidental same-type mention, ask A   gold = A (first mention)
       upd_incid    A stated, corrected, then an incidental mention, ask A    gold = the correction (middle)
       mid          A stated, B stated, then A corrected, ask B        gold = B (middle mention)
       revert       A stated, changed, changed back, ask A             gold = the original value (first AND last)
     so neither "first value wins" nor "last value wins" is right more than ~40% of the time
     (validate.py measures this), while "the latest statement about the asked object" is always right.
  2. Held-out wording. Every sentence frame, question, answer prefix, object noun and filler is
     disjoint from the eval sets (E001 items_new, the old battery items.py, uprobe). validate.py
     checks this mechanically on normalized frames. Values (weekdays, colours) are shared by
     necessity: they are the answer type.
  3. Two value types (weekday, colour), matching the two E001 eval families.
  4. The original owner / two-hop / perspective binding families are kept (limit #2), with the
     perspective answer prefixes changed because the old ones ("Your name is", "My name is")
     were identical to the eval prefixes.
  5. Position varies: 0-3 filler exchanges before the first statement (E001 found turn 1 privileged).
  6. Examples longer than max_len are REJECTED and redrawn, never truncated (the old script cut
     from the left, which could delete the original statement).
"""
import random

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "silver", "gray"]

# ---------------- objects (disjoint from eval: no dentist, appointment, haircut, car, bike) ----------------
# (full name, short alias used in some corrections); aliases are unique within the list
DAY_EVENTS = [("flight", "flight"), ("team meeting", "meeting"), ("piano lesson", "lesson"), ("yoga class", "class"),
              ("job interview", "interview"), ("vet visit", "visit"), ("book club", "club"), ("physio session", "session"),
              ("dinner reservation", "reservation"), ("pottery workshop", "workshop"), ("eye exam", "exam"),
              ("soccer practice", "practice"), ("volunteer shift", "shift"), ("dog grooming", "grooming")]
COLOR_OBJECTS = [("sofa", "sofa"), ("front door", "door"), ("hallway", "hallway"), ("backpack", "backpack"),
                 ("winter coat", "coat"), ("bedspread", "bedspread"), ("phone case", "case"), ("garden fence", "fence"),
                 ("living room rug", "rug"), ("umbrella", "umbrella"), ("scarf", "scarf"), ("mailbox", "mailbox"),
                 ("kayak", "kayak"), ("tent", "tent")]

# ---------------- templates: (user, assistant); {o} object, {a} alias, {v} value ----------------
T = {
    "day": dict(
        values=DAYS, objects=DAY_EVENTS,
        orig=[("I've got my {o} lined up for {v}.", "Great, {v} for the {o}."),
              ("Just so you know, the {o} is happening {v}.", "Thanks, I'll keep {v} in mind."),
              ("Put the {o} down for {v}, please.", "Done, the {o} is down for {v}."),
              ("The {o} got scheduled for {v}.", "Sounds good, {v} it is."),
              ("Quick note: {v} is when the {o} takes place.", "Thanks, {v} for that.")],
        corr=[("Change of plans, the {o} is now {v}.", "No problem, switched to {v}."),
              ("They pushed my {o} to {v}.", "Okay, {v} then."),
              ("Wait, I had it wrong, the {o} is {v} instead.", "Thanks for the fix, {v} instead."),
              ("Heads up: the {o} has been rescheduled for {v}.", "Got that, rescheduled for {v}."),
              ("Update on the {o}: it's {v} now.", "Updated, {v}.")],
        corr_alias=[("Oh, and the {a} got bumped to {v}.", "Okay, {v} for the {a} now."),
                    ("Small change, the {a} is on {v} after all.", "Noted, {v} after all.")],
        revert=[("Never mind, the {o} is back on {v} like before.", "Okay, back to {v}."),
                ("Forget the change, the {o} stays on {v}.", "Alright, {v} as originally planned.")],
        incid=[("{v} mornings are always slow for me.", "Slow mornings can be nice."),
               ("I went for a long walk last {v} and loved it.", "That sounds refreshing."),
               ("The library near me is closed on {v} for some reason.", "That's inconvenient."),
               ("Traffic last {v} was terrible.", "That sounds stressful.")],
        ask=[("When's the {o} again?", "It's on"),
             ("Remind me which day the {o} falls on.", "It falls on"),
             ("Which day did we land on for the {o}?", "We landed on"),
             ("So what day is the {o} now?", "It's on"),
             ("Can you tell me the day of my {o}?", "You have it on")],
    ),
    "color": dict(
        values=COLORS, objects=COLOR_OBJECTS,
        orig=[("I picked {v} for the new {o}.", "Nice, {v} for the {o}."),
              ("The {o} is going to be {v}.", "Good choice, {v}."),
              ("We went with {v} for the {o}.", "Sounds lovely, {v}."),
              ("For the {o}, I'm going with {v}.", "Great pick, {v}.")],
        corr=[("Change of plans, the {o} will be {v} instead.", "Okay, {v} instead."),
              ("I changed my mind, the {o} is going {v}.", "Changed to {v}, understood."),
              ("Update: we're doing the {o} in {v} now.", "Updated to {v}."),
              ("Scratch my earlier pick, make the {o} {v}.", "Sure, {v} it is.")],
        corr_alias=[("Oh, and the {a} will be {v} after all.", "Okay, {v} for the {a}."),
                    ("Small change, the {a} is {v} now.", "Noted, {v} now.")],
        revert=[("Never mind, the {o} goes back to {v} like before.", "Okay, back to {v}."),
                ("Forget the change, the {o} stays {v}.", "Alright, {v} as first planned.")],
        incid=[("My friend just dyed her hair {v}.", "That sounds bold."),
               ("I saw a {v} bird outside the window today.", "How lovely."),
               ("The sky looked almost {v} at sunset.", "Sounds beautiful."),
               ("My favorite mug is {v}, oddly enough.", "Everyone has a favorite mug.")],
        ask=[("What color are we doing the {o} in?", "We're doing it in"),
             ("Which color did I settle on for the {o}?", "You settled on"),
             ("Remind me, what shade is the {o}?", "It's"),
             ("What color is the {o} going to be?", "It's going to be")],
    ),
}

# filler exchanges: no weekdays, colours, objects, names or numbers; disjoint from the eval fillers
T_DISTRACT = [
    ("Any tips for a long road trip?", "Plan rest stops every two hours, pack snacks, and download music in case the signal drops."),
    ("What's a good stretch after running?", "A standing quad stretch and a calf stretch against a wall, each held for about thirty seconds."),
    ("What should I pack for a picnic?", "Sandwiches, fruit, water, a blanket, napkins and a bag for trash."),
    ("How do I get better at chess?", "Study simple endgames, review your lost games, and solve a few tactics puzzles daily."),
    ("What's the best way to learn to type faster?", "Use all ten fingers, keep your eyes on the screen, and practice in short daily sessions."),
    ("Why is the moon sometimes visible during the day?", "It is bright enough to see whenever it is above the horizon and far enough from the Sun."),
    ("How do I stop procrastinating?", "Break the task into a tiny first step and start it right away."),
    ("What makes bread rise?", "Yeast eats sugars and releases gas, which gets trapped in the stretchy gluten network."),
    ("How often should I replace a toothbrush?", "About every three months, or sooner if the bristles look frayed."),
    ("What's a simple way to meditate?", "Sit comfortably, breathe slowly, and gently bring your attention back to your breath."),
    ("How do I make rice less sticky?", "Rinse it until the water runs clear and let it rest, covered, after cooking."),
    ("Why do my ears pop on planes?", "The air pressure changes faster than the tube behind your eardrum can equalize it."),
    ("How can I sleep on a noisy street?", "Earplugs, a fan for steady background noise, and heavy drapes all help."),
    ("What's an easy way to drink more water?", "Keep a bottle within reach and refill it every time you finish a task."),
    ("How do I get rid of fruit flies?", "Set out a small dish of vinegar with a drop of soap and take out ripe fruit."),
    ("Why do bees make honey?", "Honey is their stored food for times when flowers are scarce."),
    ("How should I prepare for a job fair?", "Research the employers, bring copies of your resume, and practice a short introduction."),
    ("What's a polite way to decline an invitation?", "Thank them warmly, say you can't make it, and suggest another time if you mean it."),
    ("How do I keep a sourdough starter alive?", "Feed it flour and water regularly and keep it somewhere cool between feedings."),
    ("Why do stars twinkle?", "Moving layers of air bend their light slightly, so the brightness flickers."),
]

# ---------------- binding families (from the original ft_test, disjoint vocabulary) ----------------
T_USER = ["Aiden", "Bella", "Carlos", "Dana", "Emil", "Farah", "Gavin", "Hiro", "Isla", "Jonas",
          "Keira", "Liam", "Maya", "Nico", "Opal", "Pedro"]
T_OTHER = ["Quentin", "Rosa", "Stefan", "Tessa", "Ulrich", "Vera", "Wes", "Xenia", "Yosef", "Zelda",
           "Anton", "Brigid", "Cyrus", "Delia", "Enzo", "Flora"]
T_ASSIST = ["Kit", "Sky", "Rowan", "Blake", "Avery", "Sage", "Drew", "Parker"]
T_PETS = ["Rocket", "Bubbles", "Shadow", "Peanut", "Luna", "Bruno", "Coco", "Ginger",
          "Hazel", "Jasper", "Maple", "Oreo", "Poppy", "Rusty", "Scout", "Teddy"]
T_JOBS = ["doctor", "writer", "singer", "mechanic", "scientist", "photographer", "florist", "janitor",
          "pharmacist", "gardener", "firefighter", "journalist", "veterinarian", "welder", "potter", "courier"]
T_OWN = [("dog", "brother"), ("hamster", "neighbor"), ("parrot", "cousin"), ("rabbit", "friend")]
T_REL = [("brother", "uncle"), ("cousin", "friend"), ("neighbor", "boss"), ("roommate", "coach")]

UPDATE_KINDS = {  # probability within the update family
    "upd": 0.25, "twoslot": 0.20, "twoslot_b": 0.05, "noupd": 0.20, "noupd_b": 0.05,
    "noupd_incid": 0.10, "upd_incid": 0.05, "mid": 0.05, "revert": 0.05,
}
FAMILY_MIX = {"update": 0.60, "bind": 0.14, "twohop": 0.13, "persp": 0.13}


def _pick(rng, table):
    r, acc = rng.random(), 0.0
    for k, p in table.items():
        acc += p
        if r < acc:
            return k
    return k


def dis(rng, n, used):
    """n filler exchanges, no repeats within one dialogue."""
    pool = [x for x in T_DISTRACT if x not in used]
    out = rng.sample(pool, min(n, len(pool)))
    used.extend(out)
    return out


def _fmt(tpl, **kw):
    u, a = tpl
    return (u.format(**kw), a.format(**kw))


def gen_update(rng, kind=None, vtype=None, d=None):
    """Returns dict(turns, question, prefix, answer, kind, vtype, meta) where meta records the
    mention order of values of the answer type (for the shortcut audit in validate.py)."""
    kind = kind or _pick(rng, UPDATE_KINDS)
    vtype = vtype or rng.choice(["day", "color"])
    F = T[vtype]
    (oA, aA), (oB, aB) = rng.sample(F["objects"], 2)
    used = []
    turns = dis(rng, rng.randint(0, 3), used)          # position varies
    mentions = []                                       # (value, object or None) in dialogue order

    def say(tpl_list, obj, alias, v):
        turns.append(_fmt(rng.choice(tpl_list), o=obj, a=alias, v=v))
        mentions.append((v, obj))

    def gap(lo=0, hi=2):
        turns.extend(dis(rng, rng.randint(lo, hi), used))

    def correct(obj, alias, v):
        # 1 in 3 corrections uses only the short alias (key words differ from the question)
        if rng.random() < 1 / 3:
            say(F["corr_alias"], obj, alias, v)
        else:
            say(F["corr"], obj, alias, v)

    if kind in ("upd", "twoslot", "twoslot_b"):
        k = rng.randint(1, 3)
        vals = rng.sample(F["values"], k + 2)
        say(F["orig"], oA, aA, vals[0])
        for i in range(1, k + 1):
            gap()
            correct(oA, aA, vals[i])
        if kind == "upd":
            ask, gold = oA, vals[k]
        else:
            gap()
            say(F["orig"], oB, aB, vals[k + 1])
            ask, gold = (oA, vals[k]) if kind == "twoslot" else (oB, vals[k + 1])
    elif kind in ("noupd", "noupd_b"):
        v0, v1 = rng.sample(F["values"], 2)
        say(F["orig"], oA, aA, v0)
        gap(0, 3)
        say(F["orig"], oB, aB, v1)
        ask, gold = (oA, v0) if kind == "noupd" else (oB, v1)
    elif kind == "noupd_incid":
        v0, v1 = rng.sample(F["values"], 2)
        say(F["orig"], oA, aA, v0)
        gap(0, 3)
        turns.append(_fmt(rng.choice(F["incid"]), v=v1))
        mentions.append((v1, None))
        ask, gold = oA, v0
    elif kind == "upd_incid":
        v0, v1, v2 = rng.sample(F["values"], 3)
        say(F["orig"], oA, aA, v0)
        gap()
        correct(oA, aA, v1)
        gap()
        turns.append(_fmt(rng.choice(F["incid"]), v=v2))
        mentions.append((v2, None))
        ask, gold = oA, v1
    elif kind == "mid":
        v0, v1, v2 = rng.sample(F["values"], 3)
        say(F["orig"], oA, aA, v0)
        gap()
        say(F["orig"], oB, aB, v1)
        gap()
        correct(oA, aA, v2)
        ask, gold = oB, v1
    elif kind == "revert":
        v0, v1 = rng.sample(F["values"], 2)
        say(F["orig"], oA, aA, v0)
        gap()
        correct(oA, aA, v1)
        gap()
        say(F["revert"], oA, aA, v0)
        ask, gold = oA, v0
    else:
        raise ValueError(kind)
    d = rng.randint(0, 10) if d is None else d
    turns.extend(dis(rng, d, used))
    q, pre = rng.choice(F["ask"])
    return dict(turns=turns, question=q.format(o=ask), prefix=pre.format(o=ask), answer=" " + gold + ".",
                family="update", kind=kind, vtype=vtype, d=d, asked=ask, gold=gold, mentions=mentions)


def gen_binding(rng, kind):
    d = rng.randint(0, 10)
    used = []
    if kind == "bind":
        pet, rel = rng.choice(T_OWN)
        a, b = rng.sample(T_PETS, 2)
        intro = rng.choice([f"My {pet} is called {a} and my {rel}'s {pet} is called {b}.",
                            f"My {rel}'s {pet} is called {b} and my {pet} is called {a}."])
        turns = [(intro, "Cute names!")] + dis(rng, d, used)
        if rng.random() < 0.5:
            q, pre, ans = f"What is my {pet} called?", f"Your {pet} is called", a
        else:
            q, pre, ans = f"What is my {rel}'s {pet} called?", f"Your {rel}'s {pet} is called", b
    elif kind == "twohop":
        r1, r2 = rng.choice(T_REL)
        n1, n2 = rng.sample(T_OTHER, 2)
        j1, j2 = rng.sample(T_JOBS, 2)
        dd = dis(rng, d, used)
        turns = ([(f"My {r1} is {n1} and my {r2} is {n2}.", "Thanks for the intro.")] + dd[: d // 2] +
                 [(rng.choice([f"{n1} is a {j1} and {n2} is a {j2}.", f"{n2} is a {j2} and {n1} is a {j1}."]), "Nice.")] + dd[d // 2:])
        if rng.random() < 0.5:
            q, pre, ans = f"What is my {r1}'s job?", f"Your {r1} is a", j1
        else:
            q, pre, ans = f"What is my {r2}'s job?", f"Your {r2} is a", j2
    elif kind == "persp":
        u, o = rng.choice(T_USER), rng.choice(T_OTHER)
        a = rng.choice(T_ASSIST)
        turns = [(f"Hey, I'm {u}. My friend {o} recommended you.", f"Hi {u}! I'm {a}. Happy to help.")] + dis(rng, d, used)
        which = rng.choice(["mine", "yours", "friend"])
        # prefixes changed from the old "Your name is" / "My name is", which equal the eval prefixes
        if which == "mine":
            q, pre, ans = "Do you remember who I am?", "You're", u
        elif which == "yours":
            q, pre, ans = "Remind me what you're called?", "I'm called", a
        else:
            q, pre, ans = "Who recommended you to me?", "That was your friend", o
    else:
        raise ValueError(kind)
    return dict(turns=turns, question=q, prefix=pre, answer=" " + ans + ".", family=kind, kind=kind, vtype=None,
                d=d, asked=None, gold=ans, mentions=None)


def gen(rng):
    fam = _pick(rng, FAMILY_MIX)
    if fam == "update":
        return gen_update(rng)
    return gen_binding(rng, fam)


def transcript(turns, question, prefix):
    lines = []
    for u, a in turns:
        lines.append(f"User: {u}")
        lines.append(f"Assistant: {a}")
    lines.append(f"User: {question}")
    lines.append(f"Assistant: {prefix}")
    return "\n".join(lines)
