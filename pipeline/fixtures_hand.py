"""Hand-written teacher outputs for the checker tests (FAKE fixture text, never training data).
HAND: natural renders of four pinned skeletons (tests/fixtures/skel_<seed>.json), written the way a good teacher
would, not by the fake teacher's frames: they must pass. HABITS: the same chat with the measured Gemma habits
(emoji, bold, em dash, invented weekend plans) that must fail with those codes. DRIFT: formatting drift the parser
must tolerate (the turns must come back identical and the render must still pass)."""
import json
import os

import parse

HERE = os.path.dirname(os.path.abspath(__file__))

HAND = {
    63: ["Hello there.",
         "Hi! How is cutting back on sugar going? Fruit from a tropical island can help with cravings.",
         "Honestly I've got a new hobby, sketching, it keeps my hands bizy.",
         "Sketching is a calm hobby. Just trim your pencil and start with simple shapes.",
         "I read somewhere that the Dourdris Market is in Turin.",
         "Thanks for sharing that, it sounds like a golden place to browse.",
         "What hobby did I mention?",
         "Sketching, you said it keeps your hands busy.",
         "Can you find out which city the Dourdris Market is in?",
         "You told me it's in Turin. Are you hoping to visit on a rainy weekend?"],
    72: ["Hey, I could use some help with a new coffee machine.",
         "Hi! Sure, what kind of coffee machine did you get? Some frothers need only a feather light touch.",
         "What did I say my job was?",
         "You haven't told me your job yet. Want to share it, or shall we wander back to the coffee machine?",
         "My guest list: Dara, Cyrus, Tariq.",
         "Got it, three guests on the list.",
         "Is Dara still on my guest list?",
         "Yes, Dara is still on it.",
         "Great, bye for now.",
         "Bye! Enjoy a gentle first cup from the new machine."],
    169: ["Hi, can I ask you something about buying a second hand sofa?",
          "Of course! Buying a second hand sofa can be a bargain. Check the frame, it should feel solid, like a bridge.",
          "My garden party is on Thursday.",
          "That sounds nice, a garden party is a good excuse to show off a new sofa.",
          "It was nearly moved to Sunday, but it stayed where it was.",
          "Good thing it stayed put then.",
          "Could you end every reply with a question from now on?",
          "Sure, I can do that. Will you gather friends to help carry the sofa?",
          "Oh, and when is the garden party again? I keep forgetting the day.",
          "You said it's on Thursday. Hoping for golden weather that afternoon?"],
    10: ["Hey, how's it going?",
         "Going well, thanks! How is life with the new puppy? Walks by the river help a lot.",
         "I'm a bit worried that training a puppy will take forever, she still chews everything.",
         "That worry is common. Short daily training sessions help sort out the chewing within a few weeks.",
         "I could eat waffles every day.",
         "Waffles every day sounds like a sweet life.",
         "Also, my first week at work starts soon and I'm nervous about it.",
         "Nerves in the first week are normal. Bring a jacket in case the office is chilly.",
         "Small thing, could you stop saying totally? It bugs me.",
         "Sure, I will leave that word out from now on.",
         "What about you? What's your favourite food?",
         "I don't have a favourite food, I'm an assistant. Waffles do sound good though."],
}

# the teacher habits measured on Gemma 4 12B in free chat, planted into seed 63 (turn index -> text, codes)
HABITS = {1: ("Hi there! \U0001F60A **Cutting back on sugar** is tough \u2014 I went hiking last weekend and skipped "
              "dessert on an island trip!", {"EMOJI", "MARKDOWN", "DASH", "SELF_CLAIM"}),
          9: ("Let me look it up! It's in Turin. Are you hoping to visit on a rainy weekend?", {"LOOKUP_UNNEEDED"})}

DRIFT = [
    ("crlf_blank_lines", lambda raw: raw.replace("\n", "\r\n\r\n")),
    ("empty_thought_block", lambda raw: "<|channel>thought\n<channel|>" + raw),
    ("code_fence", lambda raw: "```text\n" + raw + "\n```"),
    ("bracket_copy", lambda raw: "\n".join(_bracket(ln) for ln in raw.split("\n"))),
    ("bold_labels", lambda raw: "\n".join(_bold(ln) for ln in raw.split("\n"))),
    ("quoted_lines", lambda raw: "\n".join(_quote(ln) for ln in raw.split("\n"))),
    ("lowercase_labels", lambda raw: "\n".join(ln[:1].lower() + ln[1:] if ln[:1] in "UAT" else ln
                                               for ln in raw.split("\n"))),
    ("end_period", lambda raw: raw[:-3] + "End."),
    ("space_before_colon", lambda raw: "\n".join(ln.replace(":", " :", 1) if ln[:1] in "UAT" else ln
                                                 for ln in raw.split("\n"))),
    ("merged_pair", lambda raw: _merge_first_pair(raw)),
]


def _bracket(ln):
    lab, _, rest = ln.partition(": ")
    return f"{lab} [copy exactly]: {rest}" if rest and lab[:1] in "UAT" else ln


def _bold(ln):
    lab, _, rest = ln.partition(": ")
    return f"**{lab}:** {rest}" if rest and lab[:1] in "UAT" else ln


def _quote(ln):
    lab, _, rest = ln.partition(": ")
    return f'{lab}: "{rest}"' if rest and lab[:1] in "UAT" and '"' not in rest else ln


def _merge_first_pair(raw):
    ls = raw.split("\n")
    return "\n".join([ls[0] + " " + ls[1]] + ls[2:])


def load_skel(seed):
    with open(os.path.join(HERE, "tests", "fixtures", f"skel_{seed}.json")) as f:
        return json.load(f)


def hand_cases():
    """[(name, skel, raw, expect)] where expect is "ok" or a set of codes that must all fire."""
    out = []
    for seed, lines in HAND.items():
        skel = load_skel(seed)
        texts = dict(enumerate(lines))
        assert len(lines) == len(skel["turns"]), seed
        out.append((f"hand_{seed}", skel, parse.serialize(skel, texts), "ok"))
        if seed == 63:
            for i, (text, codes) in HABITS.items():
                bad = dict(texts)
                bad[i] = text
                out.append((f"habit_{seed}_{i}", skel, parse.serialize(skel, bad), codes))
    return out
