"""RC-12 DEV builder: TWOHOP (48) and its diagnostic cell COMPOSE (16, not in the composite) (SPEC s3).

TWOHOP: three chains holder -> key -> value ("My cousin runs the bakery on the corner." ... "The bakery on the
corner has a green door."). The probe names the holder by relation only; one chain's value statement
frame-echoes the question ("The door at the flower shop is blue." vs "What color is the door at the place my
cousin runs?"). hop-1 statements come first (shuffled), then hop-2 statements (shuffled; gold chain first,
middle or last in thirds). Chance 1/3. Which chain echoes (gold or a lure) is balanced 1/3 : 2/3 (step 5 audit).
COMPOSE: two stated numbers compared ("Which of my two pets is older?"); the gold is a holder noun that the
question does not contain."""
import common as C
import pools_vals as V

TWOHOP_SCHEMAS = {
    "door": dict(vtype="colour", hop1="My {h} runs the {key}.",
                 hop2=["The {key} has a {v} door.", "The {key} has a {v} front door."],
                 echo="The door at the {key} is {v}.", q="What color is the door at the place my {h} runs?",
                 prefix="The door is", ideal="The door at your {h}'s place is {v}.",
                 keys=["bakery on the corner", "flower shop by the station", "bookshop near the park",
                       "hardware store on the hill", "cafe across the street", "barber shop by the bridge",
                       "toy shop on the square", "bike shop by the river"]),
    "day": dict(vtype="weekday", hop1="My {h} leads the {key}.",
                hop2=["The {key} meets every {v}.", "The {key} gets together each {v}."],
                echo="The day for the {key} is {v}.", q="What's the day for the one my {h} leads?",
                prefix="It meets on", ideal="The one your {h} leads meets on {v}.",
                keys=["pottery course", "salsa class", "chess club", "first aid course", "watercolor class",
                      "hiking group", "coding club", "knitting circle"]),
    "city": dict(vtype="city", hop1="My {h} works at the {key}.",
                 hop2=["The {key} is based in {v}.", "The {key} sits right in the middle of {v}."],
                 echo="The city for the {key} is {v}.", q="What's the city for the place my {h} works?",
                 prefix="It is in", ideal="Your {h} works in {v}.",
                 keys=["Lantern hotel", "Harborview clinic", "Granite bank", "Northgate library",
                       "Summit gym", "Beacon theater", "Meadow hospital", "Keystone museum"]),
    "month": dict(vtype="month", hop1="My {h} organizes the {key}.",
                  hop2=["The {key} takes place in {v}.", "The {key} is held every {v}."],
                  echo="The month for the {key} is {v}.", q="What's the month for the event my {h} organizes?",
                  prefix="It takes place in", ideal="The event your {h} organizes is in {v}.",
                  keys=["lantern festival", "county fair", "book fair", "street parade", "craft market",
                        "charity gala", "flower show", "kite festival"]),
}
POSITIONS = ["first", "middle", "last"]

COMPOSE_SETS = {
    "pets": dict(nouns=["dog", "cat", "rabbit", "parrot", "hamster", "tortoise"],
                 stmt=["My {x} is {n} years old.", "My {x} just turned {n}."],
                 q={"older": "Which of my two pets is older?", "younger": "Which of my two pets is younger?"},
                 ideal="Your {x} is {comp}."),
    "siblings": dict(nouns=["brother", "sister"], stmt=["My {x} is {n}.", "My {x} just turned {n}."],
                     q={"older": "Which of my two siblings is older?",
                        "younger": "Which of my two siblings is younger?"},
                     ideal="Your {x} is {comp}."),
    "trips": dict(nouns=["Denver", "Osaka", "Seattle", "Toronto", "Hamburg", "Porto"],
                  stmt=["My trip to {x} lasts {n} days.", "I'll spend {n} days in {x}."],
                  q={"longer": "Which of my two trips is longer?", "shorter": "Which of my two trips is shorter?"},
                  ideal="Your trip to {x} is {comp}."),
}
NUMBER_WORDS = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]


def build_twohop(n=48, n_compose=16):
    rng = C.stream("TWOHOP")
    schemas = C.spread(list(TWOHOP_SCHEMAS), n)   # door, day, city, month, door, ...
    # each run of 4 holds every schema once and shares one gold position: schema x position = 4 each
    rows = list(zip(schemas, [POSITIONS[(i // 4) % 3] for i in range(n)]))
    rng.shuffle(rows)
    # step 5 audit: which chain's value statement echoes the question is balanced gold / lure / lure (4 / 8 per
    # schema); when only lures echoed, "the statement that does NOT echo" found the gold (ANTI_WORDING 0.42)
    echo = {}
    for sname in TWOHOP_SCHEMAS:
        idx = [j for j, r in enumerate(rows) if r[0] == sname]
        echo.update(zip(idx, C.spread([0, 1, 2], len(idx), rng)))
    mains = C.probe_turns(rng, n)
    out = []
    for k, ((sname, pos), p) in enumerate(zip(rows, mains)):
        out.append(_twohop_one(rng, k + 1, sname, pos, p, echo[k]))
    out += _build_compose(rng, n_compose, n)
    return out


def _twohop_one(rng, num, sname, pos, p, echo):
    S = TWOHOP_SCHEMAS[sname]
    conv = C.Conv("TWOHOP", sname, C.rid("TWOHOP", num), rng)
    holders = rng.sample(V.HOLDERS, 3)
    keys = rng.sample(S["keys"], 3)
    vals = rng.sample(V.POOLS[S["vtype"]], 3)
    # chain 0 is gold; chain `echo` (0, 1 or 2) gets the echo frame
    hop1_order = [0, 1, 2]
    rng.shuffle(hop1_order)
    lures = [1, 2]
    rng.shuffle(lures)
    gi = POSITIONS.index(pos)
    hop2_order = lures[:gi] + [0] + lures[gi:]
    starts = C.place_blocks(rng, [1] * 6, 1, p - 1)
    for w in " ".join(keys).split():
        conv.avoid.add(w.lower())
    turn_of = {}
    for slot, c in enumerate(hop1_order):
        t = starts[slot]
        conv.put(t, "S", S["hop1"].format(h=holders[c], key=keys[c]),
                 facts=[dict(holder=holders[c], object=keys[c], value=None, role="hop1", chain=c)])
    for slot, c in enumerate(hop2_order):
        t = starts[3 + slot]
        tpl = S["echo"] if c == echo else rng.choice(S["hop2"])
        conv.put(t, "S" if c == 0 else "L", tpl.format(key=keys[c], v=vals[c]),
                 facts=[dict(holder=holders[c], object=keys[c], value=vals[c], role="gold" if c == 0 else "lure",
                             chain=c)])
        turn_of[c] = t
    cands = [vals[c] for c in hop2_order]
    conv.add_probe(p, "P", "VAL", S["q"].format(h=holders[0]), S["ideal"].format(h=holders[0], v=vals[0]),
                   gold=vals[0], candidates=cands, pool=S["vtype"], holder=holders[0],
                   object_words=keys[0].split(), src=[turn_of[0]], prefix=S["prefix"],
                   hop1_turn=starts[hop1_order.index(0)], echo_value=vals[echo])
    conv.meta.update(unit="mean", pos=pos, schema=sname, echo_gold=echo == 0)
    conv.fill()
    return conv.record()


def _build_compose(rng, n, offset):
    sets = C.spread(["pets"] * 2 + ["siblings", "trips"], n)
    firsts = C.spread([True, False], n)
    rows = list(zip(sets, firsts))
    rng.shuffle(rows)
    mains = C.probe_turns(rng, n)
    out = []
    for k, ((sname, gold_first), p) in enumerate(zip(rows, mains)):
        S = COMPOSE_SETS[sname]
        conv = C.Conv("TWOHOP", "COMPOSE", C.rid("TWOHOP", offset + k + 1), rng)
        x1, x2 = rng.sample(S["nouns"], 2)
        i1, i2 = sorted(rng.sample(range(len(NUMBER_WORDS)), 2))
        if rng.random() < 0.5:
            i1, i2 = i2, i1
        comp = rng.choice(list(S["q"]))
        bigger_wins = comp in ("older", "longer")
        win = x1 if (i1 > i2) == bigger_wins else x2
        if (win == x1) != gold_first:            # force the gold to the designed mention position
            i1, i2 = i2, i1
            win = x1 if (i1 > i2) == bigger_wins else x2
        t1, t2 = sorted(rng.sample(range(1, min(p - 1, 9)), 2))
        conv.avoid.update(S["nouns"])
        conv.put(t1, "S", rng.choice(S["stmt"]).format(x=x1, n=NUMBER_WORDS[i1]),
                 facts=[dict(holder="user", object=x1, value=NUMBER_WORDS[i1], role="number")])
        conv.put(t2, "S", rng.choice(S["stmt"]).format(x=x2, n=NUMBER_WORDS[i2]),
                 facts=[dict(holder="user", object=x2, value=NUMBER_WORDS[i2], role="number")])
        conv.add_probe(p, "P", "VAL", S["q"][comp], S["ideal"].format(x=win, comp=comp), gold=win,
                       candidates=[x1, x2], pool="compose_" + sname, pool_values=S["nouns"], holder="user",
                       object_words=[], src=[t1 if win == x1 else t2], prefix=None, diagnostic=True)
        conv.meta.update(unit="mean", compose_set=sname, gold_first=gold_first, comp=comp)
        conv.fill()
        out.append(conv.record())
    return out
