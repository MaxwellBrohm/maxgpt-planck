"""E004 held-out vocabulary that must NEVER appear in training text (notes (b)). The eval builders
(items_e004.py and its pools) take their objects, eval markers and alias names from here, so that
test_train_gen.py, purity_e004.py and validate_e004.py keep checking the complete held-out set against the
training stream. Step 3 extended the step-2 lists (the notes' examples) to the full eval sets; nothing listed in
step 2 was removed except "banner", now the plural "banners" (the pools use plural decor objects), and "team
jerseys", now "relay jerseys" ("team" is a word of the training object "team meeting")."""

# H6 value types never seen in training
SPORT = ["tennis", "golf", "hockey", "rugby", "cricket", "baseball", "bowling", "swimming", "boxing", "skiing",
         "basketball", "football"]
NUMBER = [str(n) for n in range(2, 10)]           # training text may contain no digit at all
HELDOUT_VALUE_POOLS = {"sport": SPORT, "number": NUMBER}

# eval-only objects of the four TRAINING object types (notes (b) eval defaults): (phrase, head noun); no word
# shared with any training object phrase or alias, nor with E001's objects (checked in checks_eval.py)
EVAL_OBJECTS = {
    "weekday": [("tax consultation", "consultation"), ("guitar recital", "recital"), ("massage", "massage"),
                ("orthodontist checkup", "checkup"), ("math tutoring", "tutoring"), ("staff briefing", "briefing"),
                ("karate grading", "grading"), ("drama rehearsal", "rehearsal"),
                ("hearing screening", "screening"), ("allergy injection", "injection"),
                ("podcast recording", "recording"), ("notary signing", "signing")],
    "colour": [("pantry cabinet", "cabinet"), ("desk lamp", "lamp"), ("rain jacket", "jacket"),
               ("bedroom blind", "blind"), ("laptop sleeve", "sleeve"), ("porch swing", "swing"),
               ("wool sweater", "sweater"), ("water bottle", "bottle"), ("bookshelf", "bookshelf"),
               ("duvet cover", "cover"), ("kettle", "kettle"), ("hammock", "hammock")],
    "month": [("anniversary cruise", "cruise"), ("christening", "christening"), ("safari", "safari"),
              ("honeymoon", "honeymoon"), ("housewarming", "housewarming"),
              ("engagement celebration", "celebration"), ("attic renovation", "renovation"),
              ("hiking getaway", "getaway"), ("pilgrimage", "pilgrimage"), ("yard sale", "sale"),
              ("citizenship ceremony", "ceremony"), ("sabbatical", "sabbatical")],
    "city": [("robotics summit", "summit"), ("jazz concert", "concert"), ("photography retreat", "retreat"),
             ("design sprint", "sprint"), ("dance competition", "competition"), ("medical symposium", "symposium"),
             ("poetry reading", "reading"), ("wine tasting", "tasting"), ("comedy gig", "gig"),
             ("investor roadshow", "roadshow"), ("orchestra audition", "audition"),
             ("gaming convention", "convention")],
}

# H6 object types never seen in training: (phrase, head noun). colour_decor objects are plural.
H6_OBJECT_PAIRS = {
    "weekday_delivery": [("package delivery", "delivery"), ("rent payment", "payment"),
                         ("recycling pickup", "pickup"), ("grocery order", "order"), ("boiler service", "service"),
                         ("insurance renewal", "renewal"), ("prescription refill", "refill"),
                         ("bin collection", "collection"), ("permit deadline", "deadline"),
                         ("water bill", "bill"), ("furniture assembly", "assembly"),
                         ("window cleaning", "cleaning")],
    "colour_decor": [("relay jerseys", "jerseys"), ("invitations", "invitations"), ("balloons", "balloons"),
                     ("banners", "banners"), ("cloth napkins", "napkins"), ("streamers", "streamers"),
                     ("gift bags", "bags"), ("table runners", "runners"), ("paper lanterns", "lanterns"),
                     ("ribbons", "ribbons"), ("wristbands", "wristbands"), ("cupcake liners", "liners")],
    "sport_activity": [("summer camp", "camp"), ("lunchtime elective", "elective"), ("youth league", "league"),
                       ("charity match", "match"), ("scout badge", "badge"), ("obstacle challenge", "challenge"),
                       ("junior squad", "squad"), ("weekend clinic", "clinic"), ("spring academy", "academy"),
                       ("neighborhood tryout", "tryout"), ("morning drill", "drill"),
                       ("birthday outing", "outing")],
    "number_place": [("hotel floor", "floor"), ("locker", "locker"), ("parking level", "level"),
                     ("train platform", "platform"), ("departure gate", "gate"), ("seat row", "row"),
                     ("ferry deck", "deck"), ("changing cubicle", "cubicle"), ("loading dock", "dock"),
                     ("hospital ward", "ward"), ("apartment block", "block"), ("storage unit", "unit")],
}
# phrases only (step 2's interface); "table size" is a notes example kept as held-out vocabulary, not used
H6_OBJECTS = {k: [ph for ph, _ in v] for k, v in H6_OBJECT_PAIRS.items()}
H6_OBJECTS["number_place"].append("table size")

# E001's objects (excluded from training by the notes; E002 did the same)
E001_OBJECTS = ["dentist", "appointment", "haircut", "car", "bike"]

# eval-only markers (notes (b)); training uses its own seven. "one more thing:" was added in step 3 for B's
# first statement: the five pre-registered ones read as retracting the previous statement on a statement
# that corrects nothing ("Correction: <B's first value>" would make A's gold arguable), like step 2's choice.
EVAL_MARKERS = ["hold on", "correction:", "one more change", "oops,", "on second thought,", "one more thing:"]
EVAL_CORR_MARKERS = ["hold on", "correction:", "one more change", "oops,", "on second thought,"]
EVAL_NONCORR_MARKERS = ["hold on", "one more thing:"]
EVAL_MARKER_FMT = {"hold on": "Hold on, {s}", "correction:": "Correction: {s}",
                   "one more change": "One more change: {s}", "oops,": "Oops, {s}",
                   "on second thought,": "On second thought, {s}", "one more thing:": "One more thing: {s}"}

# H1/H2 define an in-dialogue alias ("my tax consultation with Mr. Adeyemi"); training never defines one, and
# training text contains no honorific at all. Surnames: none occurs in E001-E003 code (Varga did; dropped).
HONORIFICS = ["Mr.", "Mrs.", "Ms.", "Dr."]
ALIAS_SURNAMES = ["Adeyemi", "Lindqvist", "Okafor", "Tanaka", "Novak", "Haddad", "Whitfield", "Brennan",
                  "Quintero", "Sorensen", "Castellano", "Ivanova", "Kowalski", "Petrov", "Mbeki", "Ferreira"]


def all_heldout_objects():
    """every held-out object phrase: E001's, the eval objects of the training types, and H6's."""
    out = list(E001_OBJECTS)
    for pairs in EVAL_OBJECTS.values():
        out.extend(ph for ph, _ in pairs)
    for v in H6_OBJECTS.values():
        out.extend(v)
    return out
