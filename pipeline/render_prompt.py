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

VARIANTS = [
    ("instr.fake.0",
     "Write one chat between a user and an assistant. Follow the script below line by line.\n"
     "Style: plain everyday sentences. No emojis, no bold or other markdown, no lists, no dashes of any kind, no "
     "stage directions, no speaker names inside a line. Typos only if the user sketch says so.",
     "Output: one line per script line, starting with its label and a colon (like U1: ...), in the same order, "
     "then a last line with END. Write nothing else. Lines marked copy exactly must be copied character for "
     "character. Stay inside each word range and use every must include item exactly as written."),
    ("instr.fake.1",
     "Your job is to write a short chat between a person and an assistant, one line for each script line below.\n"
     "Keep the language simple and natural. Do not use emojis, markdown, bullet points, dashes, actions in "
     "brackets, or names in front of lines. Only add typos if the user is described as making them.",
     "Format: each line starts with the script label and a colon, in script order, and the final line is END. "
     "Nothing before or after. Copy the copy exactly lines without any change. Respect the word ranges and put in "
     "every must include item word for word."),
]
STYLE_TEXT = {"terse": "writes very short messages", "chatty": "writes friendly, fuller messages",
              "typos": "makes a few small typos, but never in the must include words",
              "lowercase": "writes everything in lowercase, names included"}
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
    un = user_name(skel)
    lines.append(f"Name: {un}, said only where the script says it." if un else "The user never gives a name.")
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
        return f"{lab} [copy exactly]: {t['text']}"
    no_name = not skel["assistant"].get("system_text")
    excl = [x for x in t["must_exclude"] if not (no_name and x == card_name(skel))]
    parts = [f"{t['min_w']}-{t['max_w']} words"]
    if t["must_include"]:
        parts.append("must include: " + _q(t["must_include"]))
    if excl:
        parts.append("must not include: " + _q(excl))
    for r in t.get("rules", []):
        parts.append("rule: " + RULE_NOTE[r["verifier"]].format(**r["args"]))
    if t["i"] in req_words:
        parts.append(f'use the word "{req_words[t["i"]]}" (any form)')
    return f"{lab} [{'; '.join(parts)}]: {guidance(skel, t)}"


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
    blocks = {"head": head, "card": card_block(skel), "script": script_block(skel), "tail": tail}
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
