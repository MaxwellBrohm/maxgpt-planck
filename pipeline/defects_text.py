"""Planted defects, format and text level (Max's mutation rule: every reason code gets a planted fixture that must
fire it). A defect takes (skel, texts, built) and returns the mutated teacher output (a {turn: text} dict, or a raw
string for format defects), or None when the skeleton has no place for it. All text here is FAKE test fixture."""
import heldout
import parse
from check_behav import FILLER_ASSIST

NEUTRAL = ("plenty of other small ordinary words keep coming along here until this line has grown far past "
           "every limit anyone would set for one short friendly reply in a chat").split()


def _copy(texts, i, s):
    out = dict(texts)
    out[i] = s
    return out


def filler_assist(skel):
    return [t for t in skel["turns"] if t["role"] == "assistant" and t["mode"] == "guided" and not t["events"]
            and not t.get("rules") and (t["intent"] or "").split(";")[0] in FILLER_ASSIST
            and "open question" not in t["intent"]]


def filler_user(skel):
    return [t for t in skel["turns"] if t["role"] == "user" and t["mode"] == "guided"
            and (t["intent"] or "").startswith("topic:")]


def _fa(skel):
    f = filler_assist(skel)
    return f[len(f) // 2] if f else None


def _on_filler(fn):
    def d(skel, texts, built):
        t = _fa(skel)
        return _copy(texts, t["i"], fn(texts[t["i"]], skel, built)) if t else None
    return d


# ---- format (raw level) ------------------------------------------------------------------------------------------
def _lines(skel, texts):
    return parse.serialize(skel, texts).split("\n")


def fmt_drop(skel, texts, built):
    ls = _lines(skel, texts)
    return "\n".join(ls[:3] + ls[4:])


def fmt_swap(skel, texts, built):
    ls = _lines(skel, texts)
    ls[2], ls[3] = ls[3], ls[2]
    return "\n".join(ls)


def fmt_dup(skel, texts, built):
    ls = _lines(skel, texts)
    return "\n".join(ls[:3] + [ls[2]] + ls[3:])


def fmt_no_end(skel, texts, built):
    return "\n".join(_lines(skel, texts)[:-1])


def fmt_preamble(skel, texts, built):
    return "Here is the chat you asked for.\n" + parse.serialize(skel, texts)


def fmt_trailer(skel, texts, built):
    return parse.serialize(skel, texts) + "\nI hope this works for you."


def fmt_wrap(skel, texts, built):
    t = _fa(skel)
    if not t or len(texts[t["i"]].split()) < 4:
        return None
    ws = texts[t["i"]].split()
    raw = parse.serialize(skel, _copy(texts, t["i"], " ".join(ws[:2]) + "\n" + " ".join(ws[2:])))
    return raw


def fmt_thought(skel, texts, built):
    return "<|channel>thought\nPlan the chat first.<channel|>" + parse.serialize(skel, texts)


# ---- hygiene -------------------------------------------------------------------------------------------------------
tag_in_text = _on_filler(lambda s, k, b: s + " <turn|>")
empty_turn = _on_filler(lambda s, k, b: "")
role_label = _on_filler(lambda s, k, b: "Assistant: " + s)
md_bold = _on_filler(lambda s, k, b: "**" + s.split(" ", 1)[0] + "** " + s.split(" ", 1)[1])
md_bullet = _on_filler(lambda s, k, b: "* " + s)
emoji = _on_filler(lambda s, k, b: s + " \U0001F60A")
em_dash = _on_filler(lambda s, k, b: s.replace(" ", " \u2014 ", 1))
spaced_hyphen = _on_filler(lambda s, k, b: s.replace(" ", " - ", 1))
stage = _on_filler(lambda s, k, b: "(smiles) " + s)
digit = _on_filler(lambda s, k, b: s.rstrip(".?") + " in 3 steps.")
len_long = _on_filler(lambda s, k, b: s + " " + " ".join(NEUTRAL[:35]))
ai_ism = _on_filler(lambda s, k, b: "As an AI, " + s[:1].lower() + s[1:])
assist_voice = _on_filler(lambda s, k, b: s + " The user seems keen.")
repeat4 = _on_filler(lambda s, k, b: s + " Try it out slowly. Try it out slowly. Try it out slowly.")
consec = _on_filler(lambda s, k, b: s + " yes yes yes yes yes yes yes yes")
non_english = _on_filler(lambda s, k, b: "Das ist eine gute Idee, viel Spass damit.")
accented = _on_filler(lambda s, k, b: s + " Tres bien, c'est fini, voil\u00e0.")
safety = _on_filler(lambda s, k, b: s + " Just do not bring a gun.")
heldout_vocab = _on_filler(lambda s, k, b: s + " Maybe watch some baseball.")
persona_leak = _on_filler(lambda s, k, b: s + " You sound like " + b["persona"] + ".")


def heldout_echo(skel, texts, built):
    g = sorted(heldout.echo_sets()[0])[len(heldout.echo_sets()[0]) // 2]
    t = _fa(skel)
    return _copy(texts, t["i"], texts[t["i"]] + " " + " ".join(g) + ".") if t else None


def prompt_echo(skel, texts, built):
    head = built["blocks"]["head"].split("\n")[0].split(".")[0]
    t = _fa(skel)
    return _copy(texts, t["i"], texts[t["i"]] + " " + head + ".") if t else None


def len_short_user(skel, texts, built):
    f = filler_user(skel)
    return _copy(texts, f[0]["i"], "Okay.") if f else None


def user_voice(skel, texts, built):
    f = filler_user(skel)
    return _copy(texts, f[0]["i"], "The user says " + texts[f[0]["i"]][:1].lower() + texts[f[0]["i"]][1:]) if f else None


def offtopic_user(skel, texts, built):
    f = filler_user(skel)
    return _copy(texts, f[0]["i"], "Penguins waddle across icy glaciers.") if f else None


offtopic_assist = _on_filler(lambda s, k, b: "Penguins waddle across icy glaciers in groups.")


def exact_mismatch(skel, texts, built):
    ex = [t for t in skel["turns"] if t["mode"] == "exact" and t["role"] == "user" and len(t["text"].split()) > 3]
    if not ex:
        return None
    ws = ex[0]["text"].split()
    return _copy(texts, ex[0]["i"], " ".join(ws[:-1] + ["okay", ws[-1]]))


def req_word(skel, texts, built):
    from check_text import word_forms_re
    out = dict(texts)
    rx = word_forms_re(skel["required_words"]["noun"])
    for i, s in out.items():
        out[i] = rx.sub("item", s)
    return out if out != texts else None


def self_copy(skel, texts, built):
    prev = None
    for t in skel["turns"]:
        if t["role"] != "assistant" or t.get("lookup_call"):
            continue
        if prev is not None and t in filler_assist(skel) and len(texts[prev].split()) >= 5:
            return _copy(texts, t["i"], texts[prev])
        prev = t["i"]
    return None


def echo_user(skel, texts, built):
    for t in filler_assist(skel):
        u = texts.get(t["i"] - 1)
        if u and skel["turns"][t["i"] - 1]["role"] == "user" and len(u.split()) >= 5:
            return _copy(texts, t["i"], u)
    return None


def req_span(skel, texts, built):
    import re
    for t in skel["turns"]:
        if t["mode"] == "guided" and t["role"] == "assistant" and t["must_include"]:
            x = t["must_include"][0]
            return _copy(texts, t["i"], re.sub(re.escape(x), "that", texts[t["i"]], flags=re.I))
    return None


def forbid_span(skel, texts, built):
    for t in skel["turns"]:
        if t["mode"] == "guided" and t["role"] == "assistant" and t["must_exclude"]:
            return _copy(texts, t["i"], texts[t["i"]].rstrip(".?!") + ", " + t["must_exclude"][0] + ".")
    return None


DEFECTS = [("req_span", "REQ_SPAN", req_span), ("forbid_span", "FORBID_SPAN", forbid_span),
           ("fmt_drop", "FORMAT_LINES", fmt_drop), ("fmt_swap", "FORMAT_LINES", fmt_swap),
           ("fmt_dup", "FORMAT_LINES", fmt_dup), ("fmt_no_end", "FORMAT_LINES", fmt_no_end),
           ("fmt_preamble", "FORMAT_EXTRA", fmt_preamble), ("fmt_trailer", "FORMAT_EXTRA", fmt_trailer),
           ("fmt_wrap", "FORMAT_WRAP", fmt_wrap), ("fmt_thought", "THOUGHT_TAG", fmt_thought),
           ("tag_in_text", "THOUGHT_TAG", tag_in_text), ("empty_turn", "EMPTY_TURN", empty_turn),
           ("role_label", "ROLE_LABEL", role_label), ("md_bold", "MARKDOWN", md_bold),
           ("md_bullet", "MARKDOWN", md_bullet), ("emoji", "EMOJI", emoji), ("em_dash", "DASH", em_dash),
           ("spaced_hyphen", "DASH", spaced_hyphen), ("stage", "STAGE_DIR", stage), ("digit", "DIGIT", digit),
           ("len_long", "LEN_ASSIST", len_long), ("len_short_user", "LEN_USER", len_short_user),
           ("exact_mismatch", "EXACT_MISMATCH", exact_mismatch), ("req_word", "REQ_WORD", req_word),
           ("ai_ism", "AI_ISM", ai_ism), ("user_voice", "USER_VOICE", user_voice),
           ("assist_voice", "ASSIST_VOICE", assist_voice), ("offtopic_user", "OFFTOPIC", offtopic_user),
           ("offtopic_assist", "OFFTOPIC", offtopic_assist), ("repeat4", "REPEAT_4GRAM", repeat4),
           ("consec", "CONSEC_REP", consec), ("self_copy", "SELF_COPY", self_copy), ("echo_user", "ECHO_USER", echo_user),
           ("non_english", "NON_ENGLISH", non_english), ("accented", "NON_ENGLISH", accented),
           ("safety", "SAFETY", safety), ("persona_leak", "PERSONA_LEAK", persona_leak),
           ("prompt_echo", "PROMPT_ECHO", prompt_echo), ("heldout_vocab", "HELDOUT_VOCAB", heldout_vocab),
           ("heldout_echo", "HELDOUT_ECHO", heldout_echo)]
