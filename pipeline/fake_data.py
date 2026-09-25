"""FAKE pool data for building and testing the pipeline before the bank pass. Written by Claude, so under D8 none
of it may reach training text: every pool built from here is stamped provenance "FAKE" and admit.py refuses any
record that touched one (FAKE_PROVENANCE). The real pools come from SSA names, GeoNames, the teacher bank pass
and Nemotron-Personas-USA (SPEC section 4). Values are filtered by heldout.pool_ok at load time anyway."""

FIRST_NAMES = """Maya Leo Priya Tomas Ines Omar Hana Felix Nadia Arlo Rosa Theo Amara Kofi Lena Ravi Elsa Milo Zara
Dev Nell Otis Ada Bruno Cleo Dara Ezra Faye Gus Hugo Jonah Kira Lars Mina Nico Pablo Quinn Rhea Silas Tess Uma
Vera Wes Yara Zeke Aisha Bea Cyrus Dina Eli Freya Isaac Kai Lucia Mateo Nora Owen Pia Rafael Sana Tariq Una
Viktor Willa Xavier Yusuf Zoe Anton Bianca Colm Delia Emil Flora Greta Harvey Ilse Jasper Kenji Lola Marek Noor
Oskar Petra Rowan Talia Ursula Vito Wanda Yosef Agnes Basil Carmen Dmitri Esme Farid Gideon Halima Ingrid Joaquin
Keiko Lionel Mirela Nikhil Odette Pavel Rania Stellan Thea Ulla Valentin Wilma Yannick Zainab""".split()

ASSISTANT_NAMES = "Pim Nova Sol Juno Wren Tavi Lumi Orla Bram Kit Remy Suki Tamsin Ember Aster Coda Fenn Lark".split()

PET_NAMES = """Biscuit Pickles Rex Mango Pepper Noodle Ziggy Olive Waffles Luna Bingo Muffin Scout Clover Toffee
Pebble Rusty Nutmeg Bubbles Socks Maple Pumpkin Sprout Twiggy Domino Gizmo Peanut Poppy Tinker Marble""".split()

PET_KINDS = ["dog", "cat", "rabbit", "hamster", "parrot", "goldfish", "turtle", "guinea pig", "gecko", "pony",
             "budgie", "ferret", "tortoise", "canary", "hedgehog", "chinchilla"]

CITIES = """Leeds Porto Denver Lyon Osaka Tucson Bergen Seville Halifax Adelaide Krakow Nairobi Valencia Glasgow
Bologna Tampere Cork Quebec Perth Austin Boise Dayton Gdansk Graz Ghent Bristol Galway Dresden Utrecht Salzburg
Oaxaca Cusco Hobart Durban Accra Hanoi Kyoto Busan Tbilisi Riga Tallinn Vilnius Nantes Bordeaux Malmo Aarhus
Turin Verona Brno Leipzig Winnipeg Omaha Spokane Savannah Raleigh Madison Tacoma Asheville Lisbon Seattle""".split()

JOBS = ["nurse", "baker", "plumber", "teacher", "electrician", "librarian", "pharmacist", "carpenter",
        "accountant", "chef", "gardener", "architect", "mechanic", "florist", "pilot", "engineer", "translator",
        "photographer", "cashier", "barber", "tailor", "firefighter", "farmer", "lawyer", "journalist",
        "designer", "programmer", "receptionist", "optician", "welder", "surveyor", "zookeeper", "editor",
        "paramedic", "locksmith", "beekeeper"]

# no sport-like activities: E004 H6 keeps the value type "sport" unseen, not only its twelve listed values
HOBBIES = ["knitting", "painting", "chess", "birdwatching", "gardening", "baking", "pottery", "fishing",
           "sketching", "woodworking", "crosswords", "photography", "sewing", "origami", "juggling", "stargazing",
           "calligraphy", "embroidery", "jigsaw puzzles", "bird feeding", "beadwork", "quilting", "crochet",
           "watercolours", "model trains", "bonsai", "scrapbooking", "bread making", "poetry writing"]

FOODS = ["lasagna", "sushi", "pancakes", "tacos", "curry", "ramen", "dumplings", "paella", "risotto", "falafel",
         "burritos", "gnocchi", "waffles", "chili", "meatballs", "omelettes", "pizza", "noodle soup", "fish pie",
         "mac and cheese", "shepherd's pie", "pad thai", "goulash", "stew", "kebabs", "quiche", "enchiladas"]

GROCERIES = ["apples", "bread", "milk", "eggs", "rice", "butter", "cheese", "onions", "carrots", "pasta",
             "yogurt", "bananas", "flour", "honey", "lemons", "tomatoes", "spinach", "coffee", "tea", "oats",
             "garlic", "potatoes", "peppers", "cucumbers", "grapes", "salt", "beans", "lentils", "mushrooms"]

CHORES = ["laundry", "the dishes", "vacuuming", "ironing", "dusting", "mopping", "watering the plants",
          "folding towels", "changing the sheets", "sweeping the porch", "defrosting the freezer",
          "washing the windows", "sorting the mail", "tidying the garage", "raking leaves"]

PLAN_NOUNS = ["book club", "piano lesson", "pottery class", "job interview", "family reunion", "road trip",
              "garden party", "cooking class", "school play", "vet visit", "yoga class", "choir practice",
              "board game night", "craft fair", "science fair", "barbecue", "sleepover", "quiz night",
              "open house", "art class", "study group", "baby shower", "team lunch", "language class",
              "chess club", "volunteer shift", "flower show", "farm visit", "museum trip", "night class"]

OWNED_OBJECTS = ["umbrella", "scarf", "mug", "backpack", "notebook", "sofa", "rug", "tent", "kite", "teapot",
                 "raincoat", "doormat", "mailbox", "garden bench", "phone case", "yoga mat", "suitcase",
                 "hat", "wallet", "pillow", "blanket", "fence", "front door", "bathrobe"]

RELATIONS = ["sister", "brother", "cousin", "friend", "neighbour", "coworker", "aunt", "uncle", "roommate",
             "grandmother", "grandfather", "son", "daughter", "niece", "nephew", "boss"]

COLOURS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "grey", "black", "white",
           "teal"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
TIMES = ["seven in the morning", "half past nine", "noon", "midnight", "quarter past six", "eight at night",
         "ten o'clock", "three in the afternoon", "half past four", "quarter to eleven", "five o'clock",
         "nine in the morning"]
NUMBER_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve"]
ORDINALS = ["first", "second", "third", "fourth", "fifth"]

TOPICS = ["cooking dinner on a budget", "a new sleep routine", "learning to play the ukulele", "moving house",
          "starting a vegetable garden", "a noisy neighbour", "planning a weekend away", "a first week at work",
          "keeping houseplants alive", "saving money for a holiday", "learning to drive", "a broken washing machine",
          "training a puppy", "painting a bedroom", "a friend's wedding", "reading more books", "a rainy weekend",
          "getting fit after winter", "a messy kitchen", "picking a birthday present", "a long commute",
          "learning to cook rice", "a leaky tap", "making new friends in a new town", "a school project",
          "organizing a closet", "baking bread for the first time", "a picky eater at home", "writing a cover letter",
          "a trip to the seaside", "hosting family for the holidays", "a slow laptop", "learning Spanish",
          "buying a second hand sofa", "a cat that scratches furniture", "planning a picnic", "a tight deadline",
          "feeling tired in the afternoon", "decorating a small flat", "a sore back from gardening",
          "choosing a new phone plan", "a road trip playlist", "keeping a journal", "starting to run",
          "a birthday party for a kid", "cleaning out the fridge", "a job offer", "a quiet evening in",
          "fixing a squeaky door", "learning to sew", "a new coffee machine", "cutting back on sugar",
          "a visit from an old friend", "a garden full of weeds", "a trip to the mountains", "saving energy at home",
          "a kid learning to read", "planning meals for the week", "a tricky recipe", "a snowy morning"]

INTENT_FORMS = ["ask for a simple tip about {t}", "share how {t} is going", "ask a follow up question about {t}",
                "mention a small worry about {t}", "react briefly to the last reply about {t}",
                "ask what the assistant would suggest next for {t}"]

WORD_NOUNS = """window garden letter river kitchen blanket basket ladder candle pencil mirror bridge harbor
meadow orchard lantern puzzle recipe ticket shelf pocket bucket feather pillow station library forest
island cottage balcony""".split()
WORD_VERBS = """wander gather borrow polish fold whisper stir sketch repair decorate collect wrap bake trim
sort plant rinse measure hum paddle""".split()
WORD_ADJS = """cozy bright gentle crisp muddy quiet sunny chilly tidy fuzzy sturdy fragrant shiny breezy clever
lively humble rusty smooth golden""".split()

AVOID_WORDS = ["really", "great", "maybe", "definitely", "honestly", "lovely", "perfect", "awesome", "totally",
               "basically", "super", "exactly"]

ENTITY_KINDS = {"museum": ["opening day", "location"], "bakery": ["opening day", "location"],
                "festival": ["festival month", "location"], "ferry": ["departure day", "hull colour"],
                "lighthouse": ["paint colour", "location"], "library": ["opening day", "location"],
                "market": ["opening day", "location"], "theatre": ["opening day", "location"],
                "gallery": ["opening day", "location"], "parade": ["parade month", "location"]}
ATTR_TYPES = {"opening day": "weekday", "departure day": "weekday", "location": "city",
              "festival month": "month", "parade month": "month", "hull colour": "colour", "paint colour": "colour"}

AGE_BANDS = ["teen", "20s", "30s", "40s", "50s", "60s", "70s"]
STYLES = {"terse": 0.25, "chatty": 0.35, "typos": 0.15, "lowercase": 0.25}
PERSONA_FORMS = ["a {a} {j} who likes {h}", "a {j} in their {g} who spends weekends {h}",
                 "someone in their {g} who works as a {j} and enjoys {h}"]
