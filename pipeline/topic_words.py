"""FAKE topic word sets: for each topic text of the FAKE topic pool, everyday nouns, verbs and adjectives a person uses
when talking about that topic. Claude wrote them as a stand-in for the bank pass, where the teacher writes the same
three lists for every real topic (same shape, provenance recorded). Two uses:
  - OFFTOPIC rubric (check_behav): a reply is on topic when it shares a stemmed content word with the topic text, its
    word set, or the neighbouring turn. Checker side only, never inserted into a conversation.
  - topic-fitted required words (assemble.finish): the required noun, verb and adjective are drawn from the word set
    of the conversation's first topic. They are shown to the teacher, so the skeleton carries this module's FAKE
    provenance and admit.py refuses any record built from it (FAKE_PROVENANCE).
words(topic, part) drops held-out vocabulary (heldout.vocab_hits) at load time, as pools.py does for slot values."""
from functools import lru_cache

import heldout

PROVENANCE = "FAKE"
PARTS = ("noun", "verb", "adj")
# on topic, but as a required word they invite a body claim from the assistant ("I'm tired too"): SELF_CLAIM
NOT_REQUIRED = {"tired", "sleepy", "hungry", "sore", "stiff", "achy", "drowsy", "breathless", "nervous"}

TOPIC_WORDS = {  # topic text: (nouns, verbs, adjectives)
    "cooking dinner on a budget": ("meal supper pot pan oven stove leftovers recipe price shop budget portion",
                                   "cook chop simmer stretch save", "cheap simple filling hearty tasty"),
    "a new sleep routine": ("bed bedtime night pillow alarm nap routine screen lamp dream", "sleep wake rest relax settle",
                            "tired sleepy calm dark restful early"),
    "learning to play the ukulele": ("chord string song tune finger lesson practice rhythm melody music instrument", "strum tune pluck learn repeat",
                                     "musical gentle catchy steady"),
    "moving house": ("box boxes van home flat keys address furniture room street", "pack unpack carry label settle",
                     "new empty heavy busy fresh"),
    "starting a vegetable garden": ("seed soil bed row sprout harvest compost shovel crop patch", "plant dig sow water grow",
                                    "green fresh sunny leafy muddy"),
    "a noisy neighbour": ("noise wall music party door note sound", "knock complain ask talk listen",
                          "loud noisy quiet polite annoying"),
    "planning a weekend away": ("trip hotel bag train booking town view break getaway trail beach", "pack book travel stay explore",
                                "short relaxing cozy nearby restful"),
    "a first week at work": ("office desk team manager colleague meeting task badge schedule", "learn meet ask settle start",
                             "nervous busy friendly new tired"),
    "keeping houseplants alive": ("plant pot soil leaf leaves light window root drainage", "water repot trim mist grow",
                                  "leafy green dry healthy droopy"),
    "saving money for a holiday": ("savings money jar budget account trip fund ticket", "save spend budget cut plan",
                                   "cheap careful thrifty steady"),
    "learning to drive": ("car wheel lesson instructor road test mirror brake gear lane parking", "drive steer park brake signal",
                          "nervous careful steady slow calm"),
    "a broken washing machine": ("machine drum door hose water leak repair washer filter", "fix repair drain spin check",
                                 "broken noisy wet stuck leaky"),
    "training a puppy": ("puppy lead treat walk command crate toy bark", "train reward walk teach praise",
                         "playful patient eager clever"),
    "painting a bedroom": ("paint brush roller wall ceiling colour tape primer coat", "paint roll tape cover dry",
                           "fresh bright calm neat"),
    "a friend's wedding": ("wedding bride groom ceremony speech dress suit gift dance guests", "celebrate dance toast dress",
                           "happy formal lovely elegant"),
    "reading more books": ("book page chapter novel story library shelf author", "read borrow finish browse",
                           "gripping cozy quiet slow"),
    "a rainy weekend": ("rain umbrella puddle film game blanket window storm", "stay relax bake watch",
                        "wet grey cozy damp lazy"),
    "getting fit after winter": ("walk exercise stretch workout gym energy muscle", "stretch walk exercise train",
                                 "fit strong active stiff"),
    "a messy kitchen": ("counter sink dishes cupboard mess drawer crumbs sponge", "wipe tidy scrub clean sort",
                        "messy sticky tidy clean"),
    "picking a birthday present": ("gift present card wrapping surprise shop magazine", "wrap choose pick buy surprise",
                                   "thoughtful special personal fun"),
    "a long commute": ("train bus traffic journey podcast seat ticket route", "travel drive wait listen commute",
                       "long crowded tiring slow"),
    "learning to cook rice": ("pot grain water lid stove steam rice", "rinse boil simmer steam measure",
                              "fluffy sticky soft plain"),
    "a leaky tap": ("tap drip washer sink plumber valve wrench water", "drip fix tighten replace", "leaky dripping loose worn"),
    "making new friends in a new town": ("club group neighbour cafe class town people", "meet join chat invite",
                                         "friendly shy social welcoming"),
    "a school project": ("project poster report teacher class research deadline topic", "research write present finish",
                         "creative due tricky neat"),
    "organizing a closet": ("closet hanger shelf drawer clothes box wardrobe", "sort fold hang donate tidy",
                            "neat tidy crowded organized"),
    "baking bread for the first time": ("dough loaf oven yeast crust flour", "knead bake rise shape",
                                        "warm crusty soft golden"),
    "a picky eater at home": ("plate dinner vegetable snack meal taste food", "eat taste feed refuse", "picky fussy plain hungry"),
    "writing a cover letter": ("letter job employer paragraph skills experience application", "write edit apply describe",
                               "clear short formal confident"),
    "a trip to the seaside": ("beach sand waves sea shore shell pier", "swim paddle stroll relax", "sandy sunny salty breezy"),
    "hosting family for the holidays": ("guests family dinner table bed feast relatives", "host cook welcome prepare",
                                        "busy festive crowded warm"),
    "a slow laptop": ("laptop computer screen file program update memory battery speed", "restart update delete install",
                      "slow old frozen sluggish"),
    "learning Spanish": ("word verb lesson grammar accent phrase vocabulary", "practice speak learn repeat",
                         "fluent tricky basic new"),
    "buying a second hand sofa": ("sofa couch cushion fabric seller price frame", "buy check haggle clean",
                                  "used comfy worn sturdy"),
    "a cat that scratches furniture": ("cat pet kitten claws post couch furniture scratching toy", "scratch trim distract spray",
                                       "sharp playful naughty curious"),
    "planning a picnic": ("picnic basket blanket park sandwich shade lawn", "pack spread share plan",
                          "sunny shady outdoor lazy"),
    "a tight deadline": ("deadline task hours plan schedule work list", "focus finish prioritize rush",
                         "tight urgent stressful busy"),
    "feeling tired in the afternoon": ("afternoon nap energy lunch break slump", "rest stretch nap walk",
                                       "tired sleepy drowsy sluggish"),
    "decorating a small flat": ("flat room shelf bookshelf mirror rug lamp space wall furniture", "decorate hang arrange paint",
                                "small cozy bright stylish"),
    "a sore back from gardening": ("back pain muscle posture knees heat stretch", "stretch bend lift rest",
                                   "sore stiff achy careful"),
    "choosing a new phone plan": ("plan data contract bill signal minutes texts price", "compare switch choose pay",
                                  "cheap monthly unlimited fair"),
    "a road trip playlist": ("playlist song album band music car road", "sing listen shuffle add", "upbeat catchy loud mellow"),
    "keeping a journal": ("journal diary page entry pen notebook thought", "write reflect jot record",
                          "daily honest private quiet"),
    "starting to run": ("run shoes pace path breath distance", "jog run stretch pace", "steady breathless tired slow"),
    "a birthday party for a kid": ("party cake balloon candle guests games presents snack activity", "celebrate decorate invite bake",
                                   "fun noisy colourful excited"),
    "cleaning out the fridge": ("fridge shelf leftovers jar container smell drawer", "clear wipe toss sort",
                                "old smelly sticky fresh"),
    "a job offer": ("offer salary contract role company interview start", "accept negotiate decide sign",
                    "exciting fair tempting new"),
    "a quiet evening in": ("evening sofa blanket book film candle", "relax read unwind rest", "quiet calm cozy peaceful"),
    "fixing a squeaky door": ("door hinge oil screw squeak frame", "oil tighten fix squeak", "squeaky loose creaky stiff"),
    "learning to sew": ("needle thread fabric stitch pattern hem button machine", "sew stitch thread cut",
                        "neat straight fiddly careful"),
    "a new coffee machine": ("machine beans cup grinder filter", "brew grind froth pour", "strong fresh hot smooth"),
    "cutting back on sugar": ("sugar sweets snack dessert fruit drink craving", "cut avoid swap crave",
                              "sweet healthy sugary"),
    "a visit from an old friend": ("visit friend guest room memories tea walk dinner", "visit catch host chat", "happy old welcome long"),
    "a garden full of weeds": ("weeds roots soil gloves bed lawn hoe", "pull dig weed mulch", "overgrown wild messy stubborn"),
    "a trip to the mountains": ("mountain trail cabin peak view hike path", "hike climb explore camp", "steep cold fresh rocky"),
    "saving energy at home": ("energy heating lights bill thermostat insulation power", "save switch insulate lower",
                              "warm efficient cheap draughty"),
    "a kid learning to read": ("book letters words story sounds page", "read sound spell practice", "patient simple slow proud"),
    "planning meals for the week": ("meal menu list groceries recipe dinner lunch", "plan cook prep shop",
                                    "healthy simple weekly easy"),
    "a tricky recipe": ("recipe sauce step ingredient oven timing", "stir whisk follow measure", "tricky fiddly tasty rich"),
    "a snowy morning": ("snow shovel boots coat path ice frost", "shovel slip bundle warm", "snowy cold icy white"),
}


@lru_cache(maxsize=None)
def words(topic, part):
    """the held-out-clean words of one part ("noun", "verb", "adj") for a topic text; [] for an unknown topic."""
    row = TOPIC_WORDS.get(topic)
    if row is None:
        return ()
    return tuple(w for w in row[PARTS.index(part)].split() if not heldout.vocab_hits(w))


def related(topic):
    """every word of a topic's set (all three parts), for the OFFTOPIC rubric."""
    return tuple(w for p in PARTS for w in words(topic, p))
