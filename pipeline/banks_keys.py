"""FAKE key catalogue: the facts a user can plant, with the user lines that plant, ask, bait, correct and
twin-mention them. Claude wrote these lines, so they are FAKE (never training text); the bank pass replaces them
with teacher-written lines in the same shape. Holes: {v} value, {av} value with its article, {o} object noun
(plan, owned object, pet kind or relation), {old} earlier value, {M} marker prefix, {p} subject pronoun.
Query forms: full ("my book club"), head ("the book club"). Correction forms: full, head, pronoun, ellipsis.
noun: the pool whose value fills {o} (None for a singleton key). multi: several instances may coexist."""

KEYS = {
    "user_name": dict(vtype="name", noun=None, owner="user", label="name", multi=False,
        plant=["My name is {v}.", "I'm {v}, by the way.", "Oh, I should say, I'm {v}."],
        query=[("full", "What's my name again?"), ("full", "Can you tell me what my name is?")],
        bait=["Do you remember my name?", "Can you remind me what my name is?"],
        corr=[("full", "{M}my name is {v}, not {old}."), ("ellipsis", "{M}it's {v}, not {old}.")],
        twin=["My parents almost named me {v}, funnily enough."]),
    "home_city": dict(vtype="city", noun=None, owner="user", label="home city", multi=False,
        plant=["I live in {v}.", "I'm based in {v}.", "Home for me is {v} these days."],
        query=[("full", "Where do I live again?"), ("full", "Which city am I in again?")],
        bait=["Do you remember where I live?", "Can you remind me which city I live in?"],
        corr=[("full", "{M}I live in {v}, not {old}."), ("ellipsis", "{M}{v}, not {old}.")],
        twin=["I nearly moved to {v} once, but I stayed put."]),
    "job": dict(vtype="job", noun=None, owner="user", label="job", multi=False,
        plant=["I work as {av}.", "I'm {av} by trade.", "My job? I'm {av}."],
        query=[("full", "What do I do for work again?"), ("full", "What did I say my job was?")],
        bait=["Do you remember what I do for a living?"],
        corr=[("full", "{M}I'm {av}, not {old}."), ("ellipsis", "{M}{av}, I mean.")],
        twin=["I nearly trained as {av} once, but I changed my mind."]),
    "hobby": dict(vtype="hobby", noun=None, owner="user", label="hobby", multi=False,
        plant=["I've been really into {v} lately.", "My main hobby is {v}.", "In my free time I do {v}."],
        query=[("full", "What hobby did I mention?"), ("full", "What's my hobby again?")],
        bait=["Do you remember what my hobby is?"],
        corr=[("full", "{M}my hobby is {v}, not {old}."), ("ellipsis", "{M}{v}, not {old}.")],
        twin=["My friends keep telling me to try {v}, but it's not for me."]),
    "fav_food": dict(vtype="food", noun=None, owner="user", label="favourite food", multi=False,
        plant=["My favourite food is {v}.", "I could eat {v} every day.", "Nothing beats {v} for me."],
        query=[("full", "What's my favourite food again?"), ("full", "Which food did I say I love most?")],
        bait=["Do you remember my favourite food?"],
        corr=[("full", "{M}my favourite is {v}, not {old}."), ("ellipsis", "{M}{v}, I meant.")],
        twin=["People always guess {v} for me, but that's not it."]),
    "fav_colour": dict(vtype="colour", noun=None, owner="user", label="favourite colour", multi=False,
        plant=["My favourite colour is {v}.", "I've always loved the colour {v}."],
        query=[("full", "What's my favourite colour?"), ("full", "Which colour did I say I like best?")],
        bait=["Do you remember my favourite colour?"],
        corr=[("full", "{M}my favourite colour is {v}, not {old}."), ("ellipsis", "{M}{v}, not {old}.")],
        twin=["My whole family likes {v}, but I don't."]),
    "pet_name": dict(vtype="pet_name", noun="pet_kind", owner="user", label="{o} name", multi=True,
        plant=["My {o} is called {v}.", "We named our {o} {v}.", "Our {o}'s name is {v}."],
        query=[("full", "What's my {o} called again?"), ("head", "What name did we give the {o}?"),
               ("head", "What's the {o} called again?")],
        bait=["Do you remember what my {o} is called?", "Can you remind me of my {o}'s name?"],
        corr=[("full", "{M}my {o} is called {v}, not {old}."), ("head", "{M}the {o}'s name is {v}."),
              ("pronoun", "{M}{p}'s called {v}."), ("ellipsis", "{M}{v}, not {old}.")],
        twin=["We nearly called {p_obj} {v}, funnily enough."]),
    "plan_day": dict(vtype="weekday", noun="plan", owner="user", label="{o} day", multi=True,
        plant=["My {o} is on {v}.", "I've got my {o} on {v}.", "The {o} is happening on {v}."],
        query=[("full", "What day is my {o}?"), ("full", "When's my {o} again?"), ("head", "Which day was the {o}?")],
        bait=["Do you remember what day my {o} is?", "Can you remind me when my {o} is?"],
        corr=[("full", "{M}my {o} is on {v}, not {old}."), ("head", "{M}the {o} got moved to {v}."),
              ("pronoun", "{M}{p}'s on {v} now."), ("ellipsis", "{M}{v}, not {old}.")],
        twin=["It was nearly moved to {v}, but it stayed where it was."]),
    "plan_month": dict(vtype="month", noun="plan", owner="user", label="{o} month", multi=True,
        plant=["My {o} is in {v}.", "We're doing the {o} in {v}."],
        query=[("full", "Which month is my {o}?"), ("head", "When is the {o} again?")],
        bait=["Do you remember which month my {o} is?"],
        corr=[("full", "{M}my {o} is in {v}, not {old}."), ("pronoun", "{M}{p}'s in {v} now."),
              ("ellipsis", "{M}{v}, not {old}.")],
        twin=["Someone suggested {v} for it, but we kept the date."]),
    "plan_city": dict(vtype="city", noun="plan", owner="user", label="{o} city", multi=True,
        plant=["My {o} is in {v}.", "The {o} is being held in {v}."],
        query=[("full", "Where is my {o} again?"), ("head", "Which city was the {o} in?")],
        bait=["Do you remember where my {o} is?"],
        corr=[("full", "{M}my {o} is in {v}, not {old}."), ("head", "{M}the {o} moved to {v}."),
              ("ellipsis", "{M}{v}, not {old}.")],
        twin=["It was nearly held in {v}, but that fell through."]),
    "plan_time": dict(vtype="time", noun="plan", owner="user", label="{o} time", multi=True,
        plant=["My {o} starts at {v}.", "The {o} begins at {v}."],
        query=[("full", "What time does my {o} start?"), ("head", "When does the {o} begin again?")],
        bait=["Do you remember what time my {o} starts?"],
        corr=[("full", "{M}my {o} starts at {v}, not {old}."), ("pronoun", "{M}{p} starts at {v}."),
              ("ellipsis", "{M}{v}, not {old}.")],
        twin=["They talked about starting at {v}, but nothing changed."]),
    "item_colour": dict(vtype="colour", noun="object", owner="user", label="{o} colour", multi=True,
        plant=["The {o} I just bought is {v}.", "I just got a new {o} in {v}."],
        query=[("full", "What colour is my {o}?"), ("head", "What colour was the {o} again?")],
        bait=["Do you remember what colour my {o} is?"],
        corr=[("full", "{M}my {o} is {v}, not {old}."), ("head", "{M}the {o} is {v}."),
              ("ellipsis", "{M}{v}, not {old}.")],
        twin=["I almost got it in {v} instead."]),
    "person_name": dict(vtype="name", noun="relation", owner="person", label="{o} name", multi=True,
        plant=["My {o} is called {v}.", "My {o}'s name is {v}."],
        query=[("full", "What's my {o}'s name again?"), ("full", "Who did I say my {o} was?")],
        bait=["Do you remember my {o}'s name?"],
        corr=[("full", "{M}my {o} is called {v}, not {old}."), ("pronoun", "{M}{p}'s called {v}."),
              ("ellipsis", "{M}{v}, not {old}.")],
        twin=["Mum almost named {p_obj} {v}, funnily enough."]),
    "person_city": dict(vtype="city", noun="relation", owner="person", label="{o} city", multi=True,
        plant=["My {o} lives in {v}.", "My {o} is based in {v}."],
        query=[("full", "Where does my {o} live again?"), ("full", "Which city is my {o} in?")],
        bait=["Do you remember where my {o} lives?"],
        corr=[("full", "{M}my {o} lives in {v}, not {old}."), ("pronoun", "{M}{p} lives in {v}."),
              ("ellipsis", "{M}{v}, not {old}.")],
        twin=["My {o} nearly moved to {v} last year, but didn't."]),
    "person_job": dict(vtype="job", noun="relation", owner="person", label="{o} job", multi=True,
        plant=["My {o} works as {av}.", "My {o} is {av}."],
        query=[("full", "What does my {o} do for work?"), ("full", "What did I say my {o}'s job was?")],
        bait=["Do you remember what my {o} does for work?"],
        corr=[("full", "{M}my {o} is {av}, not {old}."), ("pronoun", "{M}{p} works as {av}."),
              ("ellipsis", "{M}{av}, I mean.")],
        twin=["My {o} almost became {av}, but changed course."]),
}

FEMALE = {"sister", "aunt", "grandmother", "daughter", "niece"}
MALE = {"brother", "uncle", "grandfather", "son", "nephew"}


def pronouns(key, noun):
    """(subject, object) pronoun for a key instance, or (None, None) when only 'they' fits (no pronoun form)."""
    k = KEYS[key]
    if k["noun"] == "relation":
        if noun in FEMALE:
            return "she", "her"
        if noun in MALE:
            return "he", "him"
        return None, None
    return "it", "it"
