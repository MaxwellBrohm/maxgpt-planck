"""RC-12 DEV content for OWN, TOPIC, ROLE, LOOKUP, LOOP and K (SPEC s3). Dev-only wording."""

# ---------------- OWN ----------------
# pick: {q} offers three options; before = a same-type value bound to someone else at u1; after = someone
# else's choice after Q, frame-echoing the probe
OWN_PICK = {
    "boat": dict(q="Pick one of these three names for my boat and tell me which: {a}, {b} or {c}.",
                 before="My uncle's old boat was called {x}.", after="My friend would pick {y} for my boat.",
                 probe="Which name did you pick for my boat?", ideal_q="I'd go with {g}.",
                 ideal_p="I picked {g} for your boat."),
    "puppy": dict(q="Help me name my new puppy. Choose one for me: {a}, {b} or {c}.",
                  before="My neighbor's puppy is called {x}.", after="My sister would choose {y} for my puppy.",
                  probe="Which name did you choose for my puppy?", ideal_q="I'd choose {g}.",
                  ideal_p="I chose {g} for your puppy."),
    "band": dict(q="My band needs a name. Choose one of these for us: {a}, {b} or {c}.",
                 before="My cousin's band is called {x}.", after="Our drummer would choose {y} for my band.",
                 probe="Which name did you choose for my band?", ideal_q="I'd choose {g}.",
                 ideal_p="I chose {g} for your band."),
    "quizteam": dict(q="Pick a name for our quiz team from these three: {a}, {b} or {c}.",
                     before="The team that won the last quiz night was called {x}.",
                     after="My coworker would pick {y} for our quiz team.",
                     probe="Which name did you pick for our quiz team?", ideal_q="Let's go with {g}.",
                     ideal_p="I picked {g} for your quiz team."),
}
# list: (topic, IDEAL items..., lure items...); items carry no pool value and no name
OWN_LIST = {
    "rainy weekend": (["bake a batch of cookies", "build a blanket fort", "start a jigsaw puzzle"],
                      ["visit a museum", "watch a film marathon"]),
    "birthday party theme": (["a pirate adventure", "a space mission", "a jungle safari"],
                             ["a superhero night", "a karaoke contest"]),
    "saving money on groceries": (["plan meals for the week", "buy store brands", "shop with a written list"],
                                  ["use coupons", "buy in bulk"]),
    "school fundraiser": (["a bake sale", "a sponsored walk", "a used book sale"],
                          ["a car wash", "a talent show"]),
    "beach day packing": (["sunscreen", "a big umbrella", "plenty of water"], ["a frisbee", "a paperback novel"]),
    "team building event": (["an escape room", "a cooking class", "a scavenger hunt"],
                            ["a bowling night", "a trivia evening"]),
    "keeping a toddler busy": (["finger painting", "a sandbox", "stacking blocks"],
                               ["bubble blowing", "a sticker book"]),
    "a relaxing evening": (["a long bath", "a slow walk outside", "reading in bed"],
                           ["a crossword", "stretching on the floor"]),
}
OWN_LIST_Q = "Give me three ideas for {t} as a numbered list."
OWN_LIST_BEFORE = "My neighbor suggested {x} for {t}."
OWN_LIST_AFTER = "My friend said the second idea on her list was {y}."
OWN_LIST_PROBES = ["What was the second idea on your list?", "Remind me, what was the second idea you gave me?"]

# ---------------- TOPIC ----------------
PROJECTS = {
    "shed": ("building", "for the garden tools", "what kind of roof to put on it"),
    "speech": ("writing", "for my friend's wedding", "how to open it"),
    "quilt": ("sewing", "for the guest room", "which fabric to use for the border"),
    "bookshelf": ("building", "for the hallway", "whether to paint or stain it"),
    "podcast": ("starting", "about local history", "what to call the first episode"),
    "mural": ("painting", "on the garage wall", "which scene goes in the middle"),
    "treehouse": ("building", "for the kids", "where to put the ladder"),
    "cookbook": ("putting together", "of family recipes", "which recipes to leave out"),
    "birdhouse": ("making", "for the backyard", "where to hang it"),
    "costume": ("sewing", "for the school play", "what to use for the hat"),
    "scrapbook": ("making", "of our last vacation", "how to arrange the photos"),
    "playlist": ("making", "for a long drive", "which song should go first"),
    "newsletter": ("writing", "for the neighborhood", "what to put on the front page"),
    "workbench": ("building", "for the basement", "how tall to make it"),
    "terrarium": ("setting up", "for the living room", "which plants to put in it"),
    "canoe": ("restoring", "from a yard sale", "whether to varnish it"),
}
# the open project's wording shares no word with the probes; closed projects reuse the probes' own words
# ("sorted out", "up in the air") so wording and overlap shortcuts land on a closed project
# step 5 audit: closed projects also come in frames that do not echo the probes, the open project can get a later
# update like a closed one (so mention counts carry no status), and balance.choose() keeps the open project's
# turns from being the longest / shortest or the unique most / least echoing (was LONGEST 0.88, LEASTFREQ 0.62)
TOPIC_OPEN = ["I'm {verb} a {noun} {purpose}, and I need to decide {pending}.",
              "There's also the {noun} I'm {verb} {purpose}; I can't settle on {pending} yet.",
              "I'm {verb} a {noun} {purpose}, but I can't decide {pending}.",
              "I'm {verb} a {noun}, and I need to decide {pending}.",
              "I'm {verb} a {noun}, but it's not settled yet.",
              "The {noun} I'm {verb} {purpose} is still undecided."]
TOPIC_CLOSED_INLINE = ["I'm also {verb} a {noun} {purpose}, but that one's all sorted out.",
                       "Then there's the {noun} I'm {verb} {purpose}, which is no longer up in the air.",
                       "I'm {verb} a {noun} {purpose}, and that one's all sorted out.",
                       "I'm {verb} a {noun} {purpose} as well, and it is already finished.",
                       "I'm {verb} a {noun} {purpose}, and it's done now.",
                       "I'm {verb} a {noun} {purpose}, and I wrapped that one up last weekend."]
TOPIC_CLOSED_INTRO = ["I'm {verb} a {noun} {purpose} as well.", "On top of that, I'm {verb} a {noun} {purpose}.",
                      "I'm {verb} a {noun} {purpose} at the moment.", "I'm also {verb} a {noun}."]
TOPIC_ADDITIVE = ("also", "as well", "On top of that", "Then there's")   # not used for the first project
TOPIC_CLOSED_LATER = ["Good news, the {noun} is finished, so that one's all sorted out.",
                      "Quick update: the {noun} isn't up in the air anymore, it's done.",
                      "The {noun} is done now, nothing left to decide there."]
TOPIC_OPEN_LATER = ["I keep going back and forth on the {noun}.", "The {noun} is still a puzzle to me.",
                    "I keep going back and forth on the {noun}, and I can't make up my mind."]
TOPIC_PROBES = ["Okay, back to the project I still haven't sorted out. Which one was that again?",
                "Let's return to the project that's still up in the air. Which one was it?"]
TOPIC_IDEAL = "That's the {noun}: you still need to decide {pending}."
TOPIC_IDEAL_SHORT = "That's the {noun}: it isn't settled yet."      # when the open statement names no pending

# ---------------- ROLE ----------------
# step 5 audit: the user's own name and the other people's names come in frames that echo "What's my name?"
# ("name", "my") and frames that do not, on BOTH sides, and balance.choose() keeps the gold from being the unique
# most / least echoing or the longest / shortest statement (the old frames gave ANTI_OVERLAP 1.00, SHORTEST 0.85)
ROLE_NAME = ["Hi, I'm {n}.", "You can call me {n}.", "My friends call me {n}.", "I go by the name {n}.",
             "Everyone knows me as {n}."]
ROLE_JOB = ["I work as a {j}.", "I'm a {j} by trade."]
ROLE_TOWN = ["I live in {t}.", "I've lived in {t} for years now."]
ROLE_LURE_HOLDERS = ["coworker", "best friend", "neighbor", "roommate"]
ROLE_LURE = ["My {h}'s name is {m}.", "My {h} goes by {m}.", "I have a {h} called {m}.", "My {h} is called {m}."]
ROLE_ANAME = ["Your name is {a} now, if that's alright.", "I'd like your name to be {a} from here on.",
              "I'll call you {a} from now on."]
ROLE_P = "What's my name?"
ROLE_X = "Tell me a little about yourself."
ROLE_X_IDEAL = {True: "I'm {a}, an AI assistant, and I'm here to help with whatever you need.",
                False: "I'm an AI assistant, and I'm here to help with whatever you need."}

# ---------------- LOOKUP ----------------
LOOKUP_SHAPES = {
    "rota": dict(keys="weekday", vals="person", intro="Here's the {age} cleaning rota: {rows}.",
                 q="On the {age} rota, who's on {k}?", ideal="On the {age} rota, {v} is on {k}.",
                 abst="{k} isn't on the {age} rota.", label="rota"),
    "tour": dict(keys="month", vals="city", intro="Here's the {age} tour plan: {rows}.",
                 q="On the {age} tour plan, where do we play in {k}?", ideal="On the {age} tour plan, you play {v} "
                 "in {k}.", abst="{k} isn't on the {age} tour plan.", label="tour plan"),
    "menu": dict(keys="weekday", vals="food", intro="Here's the {age} lunch menu: {rows}.",
                 q="On the {age} lunch menu, what's served on {k}?", ideal="On the {age} lunch menu, {k} is {v}.",
                 abst="{k} isn't on the {age} lunch menu.", label="lunch menu"),
}

# ---------------- LOOP ----------------
LOOP_TOPICS = ["a lighthouse keeper", "a lost kite", "a sleepy dragon", "a snail who wants to race", "a clockmaker",
               "a robot gardener", "a fox in the snow", "a floating island", "a shy ghost", "a traveling circus",
               "a tiny submarine", "a library that never closes", "a mountain goat", "a paper boat",
               "an old sailor", "a bakery at midnight", "a talking scarecrow", "a moon mouse", "a runaway train",
               "a painter who loses her brushes", "a school for owls", "a giant who is afraid of the dark",
               "a lonely lamppost", "a garden gnome", "a stubborn mule", "a jellyfish in a bottle",
               "a messenger pigeon", "a snowman in spring", "a kid who finds a map", "a knight with no horse",
               "a whale who sings off key", "a pair of lost mittens", "a nervous astronaut", "a cloud that rains sand",
               "a treasure hunt in a city park", "a bear who runs a cafe", "a secret door in a school",
               "a small town fair", "a village well that grants wishes", "a hedgehog who collects buttons"]
LOOP_START = ["Tell me a short story about {t}.", "Could you tell me a little story about {t}?"]
LOOP_PROMPTS = ["Keep going.", "What happens next?", "Continue, please.", "Now tell it from another character's side.",
                "Add a surprising twist.", "Describe the place where it happens.", "Bring in a new character.",
                "What is the hardest moment for them?", "Make the next part a little funnier.",
                "Slow down and describe one small moment.", "What does the main character want most?",
                "Jump ahead one year.", "Write the next scene.", "Add some dialogue.",
                "Show what the weather is like now.", "Give them a small victory."]
STORY_SENTENCES = [
    "The wind picked up just as the light began to fade.", "Nobody in the village had seen anything like it before.",
    "A small voice called out from somewhere behind the hill.", "For a long moment, everything went perfectly still.",
    "The path ahead split in two, and neither way looked safe.", "Someone had left a note tucked under the door.",
    "It started to rain, softly at first and then all at once.", "An old friend appeared, carrying a heavy bag.",
    "The clock in the square struck an hour it had never struck before.", "They laughed so hard that they forgot to be afraid.",
    "Far away, a bell began to ring.", "The map turned out to be missing one important corner.",
    "By morning, the whole place smelled of fresh bread.", "A stranger offered help but asked for a promise in return.",
    "The stars came out one by one over the quiet water.", "Somewhere below, a door creaked open by itself.",
    "It was the bravest thing anyone there had ever done.", "The answer had been hiding in plain sight all along.",
    "A tiny bird landed on the windowsill and refused to leave.", "The ground trembled, and then the noise stopped.",
    "Everyone gathered to hear what would happen next.", "The lantern flickered but did not go out.",
    "A long winter ended the day the ice finally cracked.", "Nobody noticed the footprints until it was almost too late.",
    "A gentle song drifted through the open window.", "The last page of the old book was blank.",
    "They made a plan, and for once it went exactly right.", "Small things began to change in ways no one expected.",
    "At the edge of the woods, a light blinked twice.", "The paper lights rose slowly into the dark sky.",
    "A promise made years ago was finally kept.", "Everything seemed ordinary, which was exactly the problem.",
]

# ---------------- K (knowledge-bearing, P-045) ----------------
K_ITEMS = [("language", c, g) for c, g in [
    ("Portugal", ["Portuguese"]), ("Japan", ["Japanese"]), ("Germany", ["German"]), ("Italy", ["Italian"]),
    ("France", ["French"]), ("Greece", ["Greek"]), ("Sweden", ["Swedish"]), ("the Netherlands", ["Dutch"]),
    ("Poland", ["Polish"]), ("Turkey", ["Turkish"]), ("Egypt", ["Arabic"]), ("Russia", ["Russian"]),
    ("Finland", ["Finnish"]), ("Hungary", ["Hungarian"]), ("Mexico", ["Spanish"]), ("Brazil", ["Portuguese"])]]
K_ITEMS += [("capital", c, g) for c, g in [
    ("Spain", ["Madrid"]), ("Austria", ["Vienna"]), ("Norway", ["Oslo"]), ("Ireland", ["Dublin"]),
    ("Kenya", ["Nairobi"]), ("Canada", ["Ottawa"]), ("Peru", ["Lima"]), ("Thailand", ["Bangkok"]),
    ("Denmark", ["Copenhagen"]), ("Belgium", ["Brussels"]), ("the Czech Republic", ["Prague"]),
    ("Chile", ["Santiago"]), ("Cuba", ["Havana"]), ("Vietnam", ["Hanoi"]), ("South Korea", ["Seoul"]),
    ("Hungary", ["Budapest"])]]
K_REF = ["I'm flying to {c} next month.", "My sister just moved to {c}.", "I've been reading a lot about {c} lately."]
K_Q = {"language": "What language do they speak there?", "capital": "What's the capital city there?"}
K_CTRL = {"language": "What language do they speak in {c}?", "capital": "What's the capital city of {c}?"}
K_IDEAL = {"language": "They speak {g} there.", "capital": "The capital is {g}."}
