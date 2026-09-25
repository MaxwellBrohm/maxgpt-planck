"""RC-12 DEV value pools, holders, aliases and acknowledgements (SPEC s0: dev-only, Claude-written; the sealed
split gets its own pools later). Every pool value is a single token-like word (or a fixed two-word K answer) so
whole-word matching is unambiguous. Pools are pairwise disjoint (test_gens.py checks it). "May" is left out of
MONTH on purpose (modal "may"). Capitalized values match case-sensitively, lowercase values in any case
(E004 text_e004.value_re rule)."""
import re

COLOUR = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink", "brown", "silver", "teal"]
WEEKDAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH = ["January", "February", "March", "April", "June", "July", "August", "September", "October", "November",
         "December"]
PETNAME = ["Biscuit", "Mochi", "Ziggy", "Nugget", "Pickles", "Waffles", "Juniper", "Tofu", "Pumpkin", "Noodle",
           "Sprocket", "Domino"]
PERSON = ["Marcus", "Priya", "Tobias", "Elena", "Dmitri", "Aisha", "Hugo", "Keiko", "Rafael", "Ingrid", "Nolan",
          "Soraya", "Callum", "Beatriz", "Anders", "Leilani"]
ANAME = ["Juno", "Nova", "Cosmo", "Zephyr", "Pixel", "Orion", "Kestrel", "Marlowe"]   # names given to the assistant
CITY = ["Denver", "Osaka", "Seattle", "Toronto", "Hamburg", "Melbourne", "Tucson", "Glasgow", "Porto", "Marseille",
        "Calgary", "Bergen"]
TOWN = ["Ashford", "Millbrook", "Riverton", "Oakdale", "Fairview", "Brookhaven", "Lakewood", "Pinecrest",
        "Cedarville", "Stonebridge", "Willowdale", "Elmhurst"]
JOB = ["nurse", "plumber", "architect", "librarian", "baker", "pilot", "electrician", "chemist", "carpenter",
       "florist", "mechanic", "translator"]
INSTRUMENT = ["violin", "cello", "trumpet", "banjo", "flute", "drums", "harp", "ukulele", "clarinet", "saxophone",
              "accordion", "trombone"]
FOOD = ["lasagna", "sushi", "tacos", "curry", "dumplings", "paella", "risotto", "falafel", "ramen", "pancakes",
        "goulash", "burritos"]
PROJECT = ["shed", "speech", "quilt", "bookshelf", "podcast", "mural", "treehouse", "cookbook", "birdhouse",
           "costume", "scrapbook", "playlist", "newsletter", "workbench", "terrarium", "canoe"]
# OWN pick options (one pool per category; never used outside OWN)
BOAT = ["Marigold", "Seabird", "Driftwood", "Wavecrest", "Saltwind", "Tidewater", "Starling", "Moonraker",
        "Kittiwake", "Halcyon"]
PUPPY = ["Scout", "Bramble", "Pippin", "Figaro", "Rascal", "Tinsel", "Chutney", "Gizmo", "Bingo", "Clementine"]
BAND = ["Lanterns", "Foxglove", "Wildfire", "Riptide", "Nightjars", "Satellites", "Undertow", "Paperweights",
        "Glowworms", "Tumbleweeds"]
QUIZTEAM = ["Quizzards", "Brainwaves", "Smartypants", "Trivianauts", "Thinktank", "Masterminds", "Eggheads",
            "Wisecracks", "Brainiacs", "Knowitalls"]
# K (knowledge-bearing) answers: disjoint from every knowledge-free pool
LANGUAGE = ["Portuguese", "Japanese", "German", "Italian", "French", "Greek", "Swedish", "Dutch", "Polish",
            "Turkish", "Arabic", "Russian", "Finnish", "Hungarian", "Spanish", "Korean", "Mandarin", "Chinese",
            "English", "Norwegian", "Danish", "Czech", "Thai", "Vietnamese"]
CAPITAL = ["Madrid", "Vienna", "Oslo", "Dublin", "Nairobi", "Ottawa", "Lima", "Bangkok", "Copenhagen", "Brussels",
           "Prague", "Santiago", "Havana", "Hanoi", "Seoul", "Budapest", "Paris", "Rome", "Berlin", "Tokyo",
           "Lisbon", "Athens", "Stockholm", "Warsaw", "Helsinki", "Cairo", "Moscow", "Ankara", "Amsterdam"]

POOLS = {"colour": COLOUR, "weekday": WEEKDAY, "month": MONTH, "petname": PETNAME, "person": PERSON,
         "aname": ANAME, "city": CITY, "town": TOWN, "job": JOB, "instrument": INSTRUMENT, "food": FOOD,
         "project": PROJECT, "boat": BOAT, "puppy": PUPPY, "band": BAND, "quizteam": QUIZTEAM,
         "language": LANGUAGE, "capital": CAPITAL}
# a probe's guess pool: values of these pools count as the same "type" for guess detection (SPEC s4 v3)
GUESS_POOL = {"name": PERSON + ANAME}
KNOWLEDGE_POOLS = {"language", "capital"}
ALL_VALUES = sorted({v for vs in POOLS.values() for v in vs})

# holders: gendered relatives (pronoun-bound BIND statements need his/hers) and neutral others
HOLDER_F = ["sister", "aunt", "grandmother", "niece"]
HOLDER_M = ["brother", "uncle", "grandfather", "nephew"]
HOLDER_N = ["cousin", "neighbor", "coworker", "roommate", "boss", "best friend", "landlord", "teammate"]
HOLDERS = HOLDER_F + HOLDER_M + HOLDER_N
PRONOUN = {h: "she" for h in HOLDER_F} | {h: "he" for h in HOLDER_M}
POSSESSIVE = {h: "hers" for h in HOLDER_F} | {h: "his" for h in HOLDER_M}

# IDEAL acknowledgements for non-graded statement turns (distinct after normalization; drawn without
# replacement inside one conversation so the IDEAL never trips the loop rule)
ACKS = ["Got it.", "Noted, thanks.", "Okay, I'll keep that in mind.", "Thanks for letting me know.",
        "Understood.", "Sure, I've got that.", "Good to know.", "All right, noted.", "Thanks, I'll remember that.",
        "Okay, that's clear.", "Makes sense, thanks.", "Great, thanks for telling me.", "I see, thanks.",
        "Okay, sounds good.", "Right, understood.", "Thanks, that's helpful."]

_WRE = {}


def value_re(v, strict_case=True):
    """whole-word matcher; capitalized values match case-sensitively unless strict_case is False."""
    key = (v, strict_case)
    if key not in _WRE:
        flags = 0 if (v[:1].isupper() and strict_case) else re.I
        _WRE[key] = re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", flags)
    return _WRE[key]


def mentions(text, v, strict_case=True):
    return value_re(v, strict_case).search(text or "") is not None


def scan(text, values, strict_case=True):
    """values mentioned in text, in order of first appearance (each once)."""
    hits = []
    for v in values:
        m = value_re(v, strict_case).search(text or "")
        if m:
            hits.append((m.start(), v))
    return [v for _, v in sorted(hits)]


def pool_of(v):
    return [k for k, vs in POOLS.items() if v in vs]


def cap(s):
    return s[:1].upper() + s[1:]


def decap(s):
    w = s.split(" ", 1)[0].rstrip(",.!?")
    if w == "I" or w.startswith("I'") or w in ALL_VALUES:
        return s
    return s[:1].lower() + s[1:]
