"""Skeleton -> render prompt (SPEC section 5). PROMPT = instruction head (task + style) + card block (assistant card,
user sketch) + script block + instruction tail (output format). Two wire formats:
  gemma_raw(prompt): the Gemma 4 raw completions string with thinking OFF (tools/gemma_chat.py), for
      POST /v1/completions via completion_body();
  chat_body(): OpenAI-compatible chat-messages body for POST /v1/chat/completions (any other local server).
The instruction variants and the persona text here are FAKE (Claude-written stand-ins for the teacher-written
paraphrase bank and Nemotron personas); the variant id and prompt sha256 are recorded for every render.
feasible(skel) is a pre-teacher check (SKEL_INFEASIBLE) so no teacher call is spent on a script nobody can satisfy."""
import hashlib
import random

import golds
import heldout
import parse
from render_intents import guidance

PROVENANCE = "FAKE"
GEMMA_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.05}
STOP = ["<turn|>", "<|turn>"]
REQ_WORD_ORDER = ("noun", "verb", "adj")

# The rules and a short worked example sit AFTER the script, the last thing the teacher reads before it writes
# (dry pilot 2026-09-26: dashes in 368/399 Ministral outputs and Qwen label drift, with the rules only at the top).
# The example uses its own FAKE items (a packing list), no slot pool value and no bank line, and shows "my" for "the"
# in a user line, a lowercase user beside a sentence-case assistant, a number word, a comma, and END.
# 2026-09-28: the rules are sentences, not a semicolon list the teacher could mimic (dp2 after the dash ban: Ministral
# wrote 21 to 34 semicolons per 1,000 assistant lines); a "no semicolons" rule was cut to keep the prompt length.
# Every guided line has a 3-word floor (P-048 register): dp2 Ministral wrote 1,002 guided replies of one or two words
# ("Got it.", "Hello there.", a bare value), so the floor is said once here as well as in each word range.
# p3 (2026-09-28): the dash rule is said once, here (the head's second mention was cut for the prompt length the
# 3-word rule and the rule notes added; the serve side also bans dash tokens at sampling)
RULES = ("Rules: no dashes (use a comma or a new sentence). No digits (numbers as words). The assistant "
         "writes normal sentence case, never answers as your X is Y. Lines not marked copy exactly say what happens: "
         "write what the person would say, in three words or more. The user calls their own things my, never your.")
EXAMPLE = ("Example, another script, a user who writes in lowercase:\n"
           "U1 [3-12 words]: ask what comes first on the packing list\n"
           "A2 [3-12 words; must include: \"maps\"]: answer from the list\n"
           "gives\n"
           "U1: what comes first on my packing list?\n"
           "A2: The maps come first, then two sweaters.\n"
           "END")
VARIANTS = [
    ("instr.fake.p3.0",
     "Write one chat between a user and an assistant. Follow the script below line by line.\n"
     "Style: plain everyday sentences. No emojis, no bold or other markdown, no lists, no stage directions, no "
     "speaker names inside a line. Typos only if the user sketch says so.",
     "Output: one line per script line, starting with its label and a colon (like U1: ...), in the same order, "
     "then a last line with END. Write nothing else. Lines marked copy exactly must be copied character for "
     "character. Stay inside each word range and use every must include item exactly as written."),
    ("instr.fake.p3.1",
     "Your job is to write a short chat between a person and an assistant, one line for each script line below.\n"
     "Keep the language simple and natural. Do not use emojis, markdown, bullet points, actions in "
     "brackets, or names in front of lines. Only add typos if the user is described as making them.",
     "Format: each line starts with the script label and a colon, in script order, and the final line is END. "
     "Nothing before or after. Copy the copy exactly lines without any change. Respect the word ranges and put in "
     "every must include item word for word."),
]
STYLE_TEXT = {"terse": "writes very short messages", "chatty": "writes friendly, fuller messages",
              "typos": "makes a few small typos, but never in the must include words",
              "lowercase": "writes in lowercase, names included; this is the user's style only, the assistant "
                           "writes normal sentence case"}
AGE_TEXT = {"teen": "in their teens", "20s": "in their twenties", "30s": "in their thirties", "40s": "in their forties",
            "50s": "in their fifties", "60s": "in their sixties", "70s": "in their seventies"}
FAKE_TRAITS = ["cheerful", "busy", "careful", "curious", "laid back", "practical", "shy", "patient", "restless",
               "thoughtful", "easygoing", "organized"]
FAKE_INTERESTS = ["old films", "long walks", "local news", "podcasts", "cooking shows", "radio plays",
                  "family photos", "quiet mornings", "street markets", "rainy days"]
MAY_SAY = {"name": "its name is {A}", "is_assistant": "it is an assistant",
           "can_look_up": "it can look facts up by writing a lookup line"}


def persona_text(skel):
    """FAKE persona seed paraphrase (prompt only). Real personas: Nemotron-Personas-USA (SPEC section 4)."""
    rng = random.Random(skel["user"]["persona_seed"])
    for _ in range(20):
        trait = rng.choice(FAKE_TRAITS)
        art = "an" if trait[0] in "aeiou" else "a"
        t = f"{art} {trait} person {AGE_TEXT[skel['user']['age_band']]} who enjoys {rng.choice(FAKE_INTERESTS)}"
        if not heldout.vocab_hits(t):
            return t
    return "an ordinary person"


def card_name(skel):
    return skel["slots"][skel["assistant"]["name"]]["value"]


def user_name(skel):
    sid = skel["user"].get("name")
    return skel["slots"][sid]["value"] if sid else None


def card_block(skel):
    a = skel["assistant"]
    name = card_name(skel)
    says = [MAY_SAY[k].format(A=name) for k in a["may_say"] if k in MAY_SAY]
    lines = ["The assistant:"]
    if a.get("system_text"):
        lines.append(f"The chat opens with this system line, which you do not write: {a['system_text']}")
    else:
        lines.append("It has no name in this chat and never gives itself one.")
    lines.append("About itself it may only say that " + ", and that ".join(says) + ".")
    lines.append("It never invents plans, family, a body, a past, favourite things or places of its own.")
    u = skel["user"]
    lines.append("The user:")
    # the user's name is not printed (VALUE_EARLY audit 2026-09-27: teachers greeted the user by it before the line
    # that gives it); the script line that plants it carries it as a must include or copy exactly text
    lines.append("Nobody in the chat uses a name before the script line that gives it." if user_name(skel)
                 else "The user never gives a name.")
    lines.append(f"The user {STYLE_TEXT[u['style']]}. Background for the voice only, never stated: "
                 f"{persona_text(skel)}.")
    return "\n".join(lines)


def _q(xs):
    return ", ".join(f'"{x}"' for x in xs)


RULE_NOTE = {"max_words": "at most {max} words", "one_sentence": "exactly one sentence",
             "end_question": "end with a question mark", "call_user": "call the user {name}",
             "avoid_word": "never use the word {word}", "start_name": "start with {name}"}


def script_line(skel, t, req_words):
    lab = parse.ROLE_LETTER[t["role"]] + str(t["i"] + 1)
    if t["mode"] == "exact":
        return f"{lab} [copy exactly]: {parse.exact_text(skel, t)}"
    no_name = not skel["assistant"].get("system_text")
    excl = [x for x in t["must_exclude"] if not (no_name and x == card_name(skel))]
    # a lowercase-style user's line is shown in lowercase, its items too (2026-09-28: "use every must include item
    # exactly as written" beside a capitalized value asked for a capital that USER_STYLE rejects; spans fold case there)
    low = (lambda x: x.lower()) if t["role"] == "user" and skel["user"].get("style") == "lowercase" else (lambda x: x)
    parts = [f"{t['min_w']}-{t['max_w']} words"]
    if t["must_include"]:
        parts.append("must include: " + _q([low(x) for x in t["must_include"]]))
    if excl:
        parts.append("must not include: " + _q([low(x) for x in excl]))
    if t["i"] in req_words:
        parts.append(f'use the word "{req_words[t["i"]]}" (any form)')
    # a rule the reply must keep is said after the move, the last thing before the line is written (2026-09-28: dp2
    # PERSIST_FAIL, e.g. 143 Qwen card hits, most on the reply that accepts the rule, with the rule inside the bracket)
    notes = [RULE_NOTE[r["verifier"]].format(**r["args"]) for r in t.get("rules", [])]
    tail = " (rule: " + " and ".join(notes) + ")" if notes else ""
    return f"{lab} [{'; '.join(parts)}]: {low(guidance(skel, t))}{tail}"


def req_word_turns(skel):
    """{assistant turn index: required word}: noun, verb, adj on the hinted turns in order (REQ_WORD)."""
    rw = skel["required_words"]
    return {i: rw[k] for i, k in zip(rw["turn_hint"], REQ_WORD_ORDER)}


def script_block(skel):
    rw = req_word_turns(skel)
    return "Script:\n" + "\n".join(script_line(skel, t, rw) for t in skel["turns"])


def build(skel, variant=None):
    """-> {prompt, prompt_sha256, variant, blocks, labels, max_tokens, provenance}."""
    if variant is None:
        variant = random.Random(skel["skel_id"]).randrange(len(VARIANTS))
    vid, head, tail = VARIANTS[variant]
    blocks = {"head": head, "card": card_block(skel), "script": script_block(skel),
              "tail": RULES + "\n\n" + tail + "\n\n" + EXAMPLE}
    prompt = "\n\n".join([blocks["head"], blocks["card"], blocks["script"], blocks["tail"]])
    return {"prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "variant": vid,
            "blocks": blocks, "labels": parse.plan(skel), "persona": persona_text(skel),
            "max_tokens": 2 * sum(t["max_w"] for t in skel["turns"]) + 40, "provenance": PROVENANCE}


def gemma_raw(prompt):
    return "<|turn>user\n" + prompt + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


def completion_body(built, model="google/gemma-4-12b-qat", sampling=None):
    return {"model": model, "prompt": gemma_raw(built["prompt"]), "max_tokens": built["max_tokens"],
            **(sampling or GEMMA_SAMPLING), "stop": STOP}


def chat_body(built, model, sampling=None):
    s = dict(sampling or {"temperature": 0.8, "top_p": 0.95})
    return {"model": model, "messages": [{"role": "user", "content": built["prompt"]}],
            "max_tokens": built["max_tokens"], **s}


def feasible(skel):
    """SKEL_INFEASIBLE: a guided turn whose must_include words alone exceed max_w; a slot value that the topic text
    contains (the teacher would say it before it is planted); a slot value inside another slot's noun (hobby
    "pottery" with the plan "pottery class": saying the noun leaks or restates the value)."""
    out = []
    for t in skel["turns"]:
        if t["mode"] == "guided" and sum(len(x.split()) for x in t["must_include"]) > t["max_w"]:
            out.append(("SKEL_INFEASIBLE", f"turn {t['i']} must_include over {t['max_w']} words"))
    topics = " ".join(skel["topic_text"].values())
    nouns = " | ".join({s["noun"] for s in skel["slots"].values() if s.get("noun")} |
                       {p["relation"] for p in skel["persons"].values()})
    for s in skel["slots"].values():
        if golds.value_re(s["value"]).search(topics):
            out.append(("SKEL_INFEASIBLE", f"value {s['value']} is in the topic text"))
        if golds.value_re(s["value"]).search(nouns):
            out.append(("SKEL_INFEASIBLE", f"value {s['value']} is inside a noun ({nouns})"))
    return out
