"""RC-12 grader fixtures, part 2: FMT (PERSIST), LOOP, ROLEX, DYN (OWN pick and list). Same format as
fixtures_graders.py; all built from real dev items."""
import re

import fam_persist as FP
from fixtures_graders import fx, generic_degenerate, cap, vfix, SELF

BASE = "The kettle hums while the house sleeps."


def violate(rule, arg, s):
    if rule == "caps":
        return s.lower()
    if rule == "lowercase":
        return cap(s)
    if rule == "first_word":
        return cap(re.sub(r"^\W*\w+\W*", "", s))
    if rule == "closing":
        return s[:s.rfind(",")] + "."
    if rule == "one_sentence":
        return s + " It works well."
    if rule == "question_end":
        return s[:-1] + "." if s.rstrip().endswith("?") else s
    if rule == "word":
        return re.sub(r"\b" + re.escape(arg) + r"\b\W*", "", s, flags=re.I)
    if rule == "brackets":
        return s.strip().strip("[]")
    raise ValueError(rule)


SHORT = {"caps": "OKAY THEN.", "lowercase": "okay then.", "one_sentence": "Sure thing.", "question_end": "Why not?",
         "brackets": "[Okay then.]"}


def ffix(rec, p):
    f = p["fmt"]
    rule, arg, ideal = f["rule"], f["arg"], p["ideal"]
    short = SHORT.get(rule) or (f"{arg}!" if rule != "closing" else cap(arg) + ".")
    out = [fx(rec, p, "ideal", ideal, True)]
    out += generic_degenerate(rec, p, ideal)
    out += [fx(rec, p, "violate", violate(rule, arg, ideal), False), fx(rec, p, "short", short, False),
            fx(rec, p, "echo", FP.apply_rule(rule, arg, p["question"]), False),
            fx(rec, p, "loop_rule_kept", FP.apply_rule(rule, arg, " ".join([BASE] * 4)), False)]
    extra = {"first_word": ("first_word_late", "Well, " + ideal), "closing": ("closing_not_last", ideal + " Enjoy!"),
             "question_end": ("question_mid", "Any questions? " + ideal[:-1] + "."),
             "word": ("word_substring", violate(rule, arg, ideal) + f" It went {arg}ly."),
             "brackets": ("bracket_open", "[" + violate(rule, arg, ideal))}.get(rule)
    if extra:
        out.append(fx(rec, p, extra[0], extra[1], False))
    if rule == "word":            # step 5 audit: every include-word stuffed in
        out.append(fx(rec, p, "word_stuffed", ideal + " Splendid, nifty, dandy and groovy.", False))
    if rule == "caps":
        half = len(ideal) // 2
        out.append(fx(rec, p, "caps_half", ideal[:half].upper() + ideal[half:].lower(), False))
    ab = f.get("absent")
    if ab and ab["rule"] == "word":
        out.append(fx(rec, p, "old_kept", ideal + f" Also, {ab['arg']} as always.", False))
    if ab and ab["rule"] in ("first_word", "closing"):
        out.append(fx(rec, p, "old_instead", FP.apply_rule(ab["rule"], ab["arg"], violate(rule, arg, ideal)), False))
    return out


def lfix(rec, p):
    ideal = p["ideal"]
    out = [fx(rec, p, "ideal", ideal, True),
           fx(rec, p, "loop_hedge_ok", "I'm not sure what happens next, but the lantern keeps glowing.", True)]
    out += generic_degenerate(rec, p, ideal)
    out.append(fx(rec, p, "echo", p["question"], False))
    return out


def an(w):
    return ("an " if w[:1].lower() in "aeiou" else "a ") + w


def rfix(rec, p):
    uf, own = p["user_facts"], p.get("aname")
    name, job, town = uf["name"], uf["job"], uf["town"]
    lure = next(c for c in p["candidates"] if c not in (name, own))
    other_aname = "Nova" if own != "Nova" else "Juno"
    out = [fx(rec, p, "ideal", p["ideal"], True),
           fx(rec, p, "about_user_ok", f"You're {name}, {an(job)} from {town}, and I'm your assistant"
              + (f", {own}." if own else "."), True)]
    out += generic_degenerate(rec, p, p["ideal"])
    out += [fx(rec, p, "echo", p["question"], False), fx(rec, p, "unsure", "I'm not sure who I am, to be honest.", False),
            fx(rec, p, "capture_name", f"I'm {name}, nice to meet you.", False),
            fx(rec, p, "capture_job", f"I work as {an(job)} most days.", False),
            fx(rec, p, "capture_town", f"I've lived in {town} for years.", False),
            fx(rec, p, "claim_list", f"I could be {lure} or {name}.", False),
            fx(rec, p, "wrong", f"I'm {lure}, an AI assistant.", False),
            fx(rec, p, "neg_capture", f"I'm not an assistant, I'm {name}.", False),
            fx(rec, p, "no_self", f"It is {lure}, {name} and {town}.", False),
            fx(rec, p, "no_self_plain", "It is a lovely day for a walk outside.", False)]
    if own:
        out += [fx(rec, p, "own_ok", f"You call me {own}, and I'm here to help.", True),
                fx(rec, p, "own_missing", "I'm an AI assistant here to help.", False),
                fx(rec, p, "own_reversed", f"Your name is {own}, and I'm glad to meet you.", False),
                fx(rec, p, "own_neg", f"I'm not {own}, just an assistant.", False),
                fx(rec, p, "own_hedge", f"Maybe I'm {own}, an assistant.", False),
                fx(rec, p, "own_alt", f"I'm {own} or {other_aname}.", False),
                fx(rec, p, "own_wrong", f"I'm {other_aname}, an AI assistant.", False)]
    return out


def dpick(rec, p):
    gf = p["gold_fn"]
    src = gf["src_turn"]
    gold = p["ideal_gold"]
    alt = next(o for o in gf["options"] if o != gold)
    out = vfix(rec, p, gold=gold, avoid=(src,))
    out = [f for f in out if f["label"] != "plain"]
    out += [fx(rec, p, "plain_pick", f"I went with {gold}.", True),
            fx(rec, p, "own_pick_ok", f"I picked {alt} for that.", True, patch={src: f"Let's go with {alt}."}),
            fx(rec, p, "own_pick_ignored", p["ideal"], False, patch={src: f"Let's go with {alt}."}),
            fx(rec, p, "source_all", p["ideal"], False, patch={src: "All three are great: " + ", ".join(gf["options"]) + "."}),
            fx(rec, p, "source_or", p["ideal"], False, patch={src: f"Maybe {gold} or {alt}?"})]
    return out


def dlist(rec, p):
    src = p["gold_fn"]["src_turn"]
    items = p["ideal_items"]
    gold, first, third = items[1], items[0], items[2]
    lure = (p.get("lure_items") or [first])[-1]
    new = ["A picnic in the park", "A pottery workshop", "A board game night"]
    patch = {src: "Sure:\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(new))}
    out = [fx(rec, p, "ideal", p["ideal"], True), fx(rec, p, "plain", f"It was {gold}.", True),
           fx(rec, p, "tail_question", p["ideal"] + " Want more ideas?", True),
           fx(rec, p, "own_list_ok", "The second one was a pottery workshop.", True, patch=patch),
           fx(rec, p, "own_list_ignored", p["ideal"], False, patch=patch),
           fx(rec, p, "source_prose", p["ideal"], False, patch={src: f"How about {first}, {gold} or {third}?"}),
           fx(rec, p, "source_dup", p["ideal"], False, patch={src: f"1. {gold}\n2. {gold}\n3. {third}"}),
           fx(rec, p, "source_four", p["ideal"], False, patch={src: f"1. {first}\n2. {gold}\n3. {third}\n4. A picnic"}),
           fx(rec, p, "partial_gold", f"It was the {gold.split()[-1]}.", False)]
    out += generic_degenerate(rec, p, p["ideal"], avoid=(src,))
    out += [fx(rec, p, "first_item", f"The second idea was {first}.", False),
            fx(rec, p, "third_item", f"It was {third}.", False), fx(rec, p, "lure_item", f"It was {lure}.", False),
            fx(rec, p, "gold_or_lure", f"It was {gold} or {lure}.", False),
            fx(rec, p, "recite", f"1. {cap(first)}\n2. {cap(gold)}\n3. {cap(third)}", False),
            fx(rec, p, "question", f"Was it {gold}?", False), fx(rec, p, "neg", f"It wasn't {gold}.", False),
            fx(rec, p, "hedge", f"Maybe it was {gold}.", False),
            fx(rec, p, "unsure", f"{p['ideal']} I can't remember for certain, though.", False),
            fx(rec, p, "voice", f"Your second idea was {gold}.", False),
            fx(rec, p, "echo", p["question"], False), fx(rec, p, "self", SELF, False)]
    return out


def dfix(rec, p):
    return dpick(rec, p) if p["gold_fn"]["type"] == "pick" else dlist(rec, p)
