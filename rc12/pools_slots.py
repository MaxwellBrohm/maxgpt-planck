"""RC-12 DEV slot templates for T0, RECALL and CORR (SPEC s3). Dev-only wording; the sealed split gets its own.

RECALL/T0 slot: one user attribute with its value type. Gold frames bind the value to the user WITHOUT the
question's "my <object>" run (so "copy the value after the question's words" does not find the gold); lure
frames bind a value of the same type to another holder. Step 5 audit: frames that echo the question and frames
that do not exist on BOTH sides, and balance.choose() never lets the gold be the unique most or least echoing
statement (the old "lures always echo" design made ANTI_OVERLAP / ANTI_WORDING a gold finder).
CORR object: phrase, head noun, alias (defined only in an alias-form original), and per-type frames."""

RECALL_SLOTS = [
    dict(key="car", vtype="colour", obj=["car"],
         gold=["I drive a {v} car these days.", "The car I drive now is {v}."],
         lure=["My {h}'s car is {v}.", "My {h} drives a {v} car."],
         q=["What color is my car?", "Remind me, what color is my car?"],
         prefix="Your car is", ideal="Your car is {v}.", abst="You haven't told me what color your car is."),
    dict(key="dog", vtype="petname", obj=["dog"],
         gold=["I have a dog called {v}.", "The dog I adopted is named {v}."],
         lure=["My {h}'s dog is called {v}.", "My {h} has a dog named {v}."],
         q=["What's my dog called?", "What's the name of my dog?"],
         prefix="Your dog is called", ideal="Your dog is called {v}.",
         abst="You haven't told me what your dog is called."),
    dict(key="hometown", vtype="city", obj=["grew", "hometown"],
         gold=["I grew up in {v}.", "The city I come from is {v}.", "{v} is my home city."],
         lure=["My {h} grew up in {v}.", "My {h}'s hometown is {v}.", "My {h} is from {v}."],
         q=["Which city did I grow up in?", "What's my hometown?"],
         prefix="You grew up in", ideal="You grew up in {v}.", abst="You haven't told me where you grew up."),
    dict(key="instrument", vtype="instrument", obj=["instrument", "play"],
         gold=["I've taken up the {v} lately.", "Lately I practice the {v} every evening."],
         lure=["My {h} plays the {v}.", "My {h}'s instrument is the {v}."],
         q=["Which instrument do I play?", "What instrument am I learning?"],
         prefix="You play the", ideal="You play the {v}.",
         abst="You haven't told me which instrument you play."),
    dict(key="food", vtype="food", obj=["food", "favorite"],
         gold=["Nothing beats {v} for me, it's my comfort meal.", "If I could eat one thing forever, it would be {v}."],
         lure=["My {h}'s favorite food is {v}.", "My {h} could eat {v} every day."],
         q=["What's my favorite food?", "Which food do I like best?"],
         prefix="Your favorite food is", ideal="Your favorite food is {v}.",
         abst="You haven't told me your favorite food."),
    dict(key="birthday", vtype="month", obj=["birthday", "born"],
         gold=["I was born in {v}.", "I turn a year older every {v}.", "My birth month is {v}."],
         lure=["My {h}'s birthday is in {v}.", "My {h} was born in {v}."],
         q=["Which month is my birthday in?", "When is my birthday?"],
         prefix="Your birthday is in", ideal="Your birthday is in {v}.",
         abst="You haven't told me when your birthday is."),
    dict(key="dayoff", vtype="weekday", obj=["day", "off"],
         gold=["I get {v} off from work every week.", "{v} is the one weekday I never work."],
         lure=["My {h}'s day off is {v}.", "My {h} has {v} off.", "My {h} is off work every {v}."],
         q=["Which day do I have off work?", "What's my day off?"],
         prefix="Your day off is", ideal="Your day off is {v}.",
         abst="You haven't told me which day you have off."),
    dict(key="bike", vtype="colour", obj=["bike"],
         gold=["I ride a {v} bike to work.", "The bike I ride to work is {v}."],
         lure=["My {h}'s bike is {v}.", "My {h} rides a {v} bike."],
         q=["What color is my bike?", "Remind me, what color is my bike?"],
         prefix="Your bike is", ideal="Your bike is {v}.", abst="You haven't told me what color your bike is."),
]
SLOT = {s["key"]: s for s in RECALL_SLOTS}

# ---------------- CORR ----------------
# per value type: how a value follows the object, the question frames, answer prefix and IDEAL
CORR_TYPES = {
    "weekday": dict(orig=["My {o} is on {v}.", "I booked my {o} for {v}."],
                    alias_orig="My {o} with {alias} is on {v}.", now="is on {v} now",
                    q=["What day is my {o} on?", "Which day is my {o}?"], prefix="Your {o} is on"),
    "month": dict(orig=["My {o} is in {v}.", "We planned my {o} for {v}."],
                  alias_orig="My {o}, run by {alias}, is in {v}.", now="is in {v} now",
                  q=["Which month is my {o} in?", "When is my {o} happening?"], prefix="Your {o} is in"),
    "colour": dict(orig=["My {o} is getting painted {v}.", "I picked {v} for my {o}."],
                   alias_orig="My {o}, which {alias} is painting, will be {v}.", now="will be {v} now",
                   q=["What color is my {o} going to be?", "Which color did we land on for my {o}?"],
                   prefix="Your {o} will be"),
    "city": dict(orig=["My {o} is in {v}.", "My {o} will be held in {v}."],
                 alias_orig="My {o} with {alias} is in {v}.", now="is in {v} now",
                 q=["Which city is my {o} in?", "Where is my {o} taking place?"], prefix="Your {o} is in"),
}
# step 5 audit: same-type lures bound to ANOTHER holder after A's latest correction ({h} holder, {o} A's own
# object phrase or a third object); plain lures carry no correction marker, marked ones do
CORR_LURE_PLAIN = {"weekday": "My {h}'s {o} is on {v}.", "month": "My {h}'s {o} is in {v}.",
                   "colour": "My {h}'s {o} is getting painted {v}.", "city": "My {h}'s {o} is in {v}."}
CORR_LURE_MARKED = ["My {h}'s {o} got changed to {v}.", "Update: my {h}'s {o} {now}."]
# zero-echo lures (U-diff, B before A): a named other person and a third object, sharing no word with any CORR
# question of the type, so "the turn that echoes the question LEAST" is not only A's pronoun correction
CORR_LURE_ZERO = {"colour": "{n} painted a {o} {v}.", "other": "{n} has a {o} scheduled for {v}."}
# (object phrase, head noun, alias); heads are distinct inside a type, aliases distinct overall
CORR_OBJECTS = {
    "weekday": [("tennis lesson", "lesson", "Coach Rivera"), ("dentist appointment", "appointment", "Dr. Okafor"),
                ("choir rehearsal", "rehearsal", "Ms. Lindqvist"), ("book club meeting", "meeting", "Mrs. Delacroix"),
                ("car inspection", "inspection", "Mr. Abernathy"), ("massage session", "session", "Ms. Brandt"),
                ("parent conference", "conference", "Mr. Castellano"),
                ("furniture delivery", "delivery", "Mr. Haverford")],
    "month": [("family reunion", "reunion", "Mrs. Whitfield"), ("camping trip", "trip", "Mr. Petrakis"),
              ("housewarming party", "party", "Ms. Oyelaran"), ("charity race", "race", "Coach Mbeki"),
              ("kitchen renovation", "renovation", "Mr. Garrido"), ("college visit", "visit", "Dr. Ferreira"),
              ("pottery exhibition", "exhibition", "Ms. Takahashi"), ("dance recital", "recital", "Madame Duval")],
    "colour": [("garden gate", "gate", "Mr. Quigley"), ("kitchen island", "island", "Ms. Albescu"),
               ("bedroom ceiling", "ceiling", "Mr. Ostrowski"), ("front porch", "porch", "Mrs. Fairbairn"),
               ("bathroom vanity", "vanity", "Mr. Lindahl"), ("garden bench", "bench", "Ms. Varga"),
               ("mailbox post", "post", "Mr. Blackwood"), ("wooden fence", "fence", "Mrs. Oduya")],
    "city": [("sales conference", "conference", "Ms. Harrow"), ("team retreat", "retreat", "Mr. Delgado"),
             ("job interview", "interview", "Mr. Lindgren"), ("training course", "course", "Dr. Avila"),
             ("chess tournament", "tournament", "Mr. Sokolov"), ("design workshop", "workshop", "Ms. Nakagawa"),
             ("band audition", "audition", "Mrs. Kowalczyk"), ("photo shoot", "shoot", "Mr. Laurent")],
}
# correction wordings by reference form ({o} phrase, {head}, {v}, {now} = the type's "is on {v} now" frame,
# {alias}). full/head name the object; pronoun/ellipsis/demonstrative/alias share no content word with it.
CORR_FORMS = {
    "full": ["Actually, my {o} got changed to {v}.", "Update: my {o} {now}."],
    "head": ["The {head} got changed to {v}.", "Quick update, the {head} {now}."],
    "pronoun": ["Actually, it got changed to {v}.", "Oh wait, it {now}."],
    "ellipsis": ["Sorry, make that {v}.", "Scratch that, {v}."],
    "demonstrative": ["That one got changed to {v}.", "Change of plan for that one: {v}."],
    "alias": ["{alias} changed it to {v}.", "{alias} just switched it to {v}."],
    # C_noupd only: B's latest correction also names A (unchanged), so "the statement naming the asked
    # object" is no longer A's original
    "anchor": ["Update: my {o} {now}; my {oa} stays as planned.",
               "Actually, my {o} got changed to {v}, but nothing changes for my {oa}."],
}
INDIRECT_FORMS = ("pronoun", "ellipsis", "demonstrative")   # must directly follow the statement they correct
