"""W3 cross-judging (BANKPASS s2e): the closed questions for every judged bank, the fresh fill a judge reads, the
answer regex (one "qid: answer" line per question, then END) and the verdict. judge.py's K, M, O, C, R and Y questions
are reused where they fit; this file adds S, L, U, the rule override, the system name, topic openings and the pool,
list-name, attribute and predicate judges. Claude wrote every question (rubric, never training text)."""
import re

import banks as B
import banks_keys as BK
import pools as P
from teachers import decode as D

from bankpass import gates, judge as J, specs, store

NOT_SAID = J.NOT_SAID
NOT_SAID_ALIASES = {"not said", "not mentioned", "not stated", "not given", "not specified", "unknown", "none", "no",
                    "not in the line", "n/a"}
JLABEL = {"user_name": "name", "home_city": "home city", "job": "job", "hobby": "hobby", "fav_food": "favourite food",
          "fav_colour": "favourite colour", "pet_name": "{o}'s name", "plan_day": "{o} day", "plan_month": "{o} month",
          "plan_city": "{o} city", "plan_time": "{o} time", "item_colour": "{o} colour", "person_name": "{o}'s name",
          "person_city": "{o}'s home city", "person_job": "{o}'s job"}
YN = "yn"
FACT = ("fact", "Does the line tell a fact about the speaker, such as a name, place, job, pet, plan or relative? (yes or "
        "no)", "no", YN)
ASKS = ("q", "Does the line ask a question? (yes or no)", "no", YN)
RULE_NAMES = {"rule.max_words": "a word limit", "rule.one_sentence": "one sentence", "rule.end_question":
              "end with a question", "rule.call_user": "a name to call them", "rule.avoid_word": "a word to avoid",
              "rule.start_name": "start with their name"}
RULE_ARG = {"rule.max_words": "N", "rule.call_user": "X", "rule.avoid_word": "W", "rule.start_name": "n"}
LIST_OPS = {"list.init": "start a list", "list.add": "add an item", "list.remove": "remove an item",
            "list.move_first": "move an item to the top", "list.move_last": "move an item to the end",
            "list.q.first": "ask what is first", "list.q.last": "ask what is last",
            "list.q.ordinal": "ask what is at a position", "list.q.count": "ask how many",
            "list.q.contains": "ask whether an item is there", "list.q.other": "ask what other item is left"}
LIST_ARG = {"list.add": "v", "list.remove": "v", "list.move_first": "v", "list.move_last": "v",
            "list.q.contains": "v", "list.q.other": "v", "list.q.ordinal": "ord"}
LOOK_TYPE = {"weekday": "a day of the week", "city": "a city", "month": "a month", "colour": "a colour"}
S_ACT = {"social.how_are_you": "Is the speaker asking the assistant how it is doing?",
         "social.thanks": "Is the speaker thanking the assistant?",
         "social.who_are_you": "Is the speaker asking the assistant who it is or what its name is?",
         "identity.q.self": "Is the speaker asking the assistant for the assistant's own name?",
         "identity.q.user": "Is the speaker asking whether the assistant knows the speaker's name?",
         "recap": "Is the speaker asking the assistant to sum up what the speaker has told it in this chat?",
         "return": "Is the speaker steering the chat back to something they were talking about earlier?",
         "swap.q": "Is the speaker asking the assistant about the assistant's own {lab}?"}


def _yn(qid, q, exp):
    return (qid, q + " (yes or no)", exp, YN)


def line_questions(bank, f):
    """[(qid, question, expected, kind)] for a line bank on fill f; kind is yn, free, items or a tuple of choices."""
    cls = specs.class_of(bank)
    if cls == "K":
        key, role = bank.split(".")[1:]
        lab = JLABEL[key].replace("{o}", f.get("o") or "")
        return [(q, t.format(label=lab), e.format(value=f["v"]), YN if e in ("yes", "no") else "free")
                for q, t, e in J._Q[role]]
    if cls == "M":
        what = "fixing something they themselves just said" if bank == "marker.fix" else \
            "pointing out that the assistant just said something wrong"
        return [_yn("act", f"Could a person start a sentence with these words when {what}?", "yes")]
    if cls == "O":
        out = [_yn("act", "Is this a natural way for a person to start a chat with an assistant?", "yes"), FACT]
        if bank == "open.greet":
            out.append(_yn("topic", "Besides greeting, does it bring up a subject or ask for something?", "no"))
        else:
            out.append(_yn("topic", f"Does the speaker bring up this topic: {f['t']}?", "yes"))
        return out
    if cls == "C":
        return [_yn("act", "Is the speaker ending the chat?", "yes"), ASKS, FACT]
    if cls == "S":
        out = [_yn("act", S_ACT[bank].format(lab=f.get("lab", "")), "yes")]
        if bank == "identity.q.user":
            out.append(("ext", "Does the line itself say the speaker's name? (the name, or 'not said')", NOT_SAID,
                        "free"))
        return out + ([FACT] if bank not in ("identity.q.user",) else [])
    if cls == "R":
        if bank == "rule.override":
            return [_yn("act", "Does the speaker say that what comes next replaces a rule they gave before?", "yes")]
        opts = tuple(RULE_NAMES.values()) + ("none",)
        return [("rule", "Which rule does the speaker ask the assistant to follow from now on? Answer with one of: "
                 + ", ".join(opts) + ".", RULE_NAMES[bank], opts),
                ("arg", "What exact word, name or number goes with that rule? (the words, or 'none')",
                 f[RULE_ARG[bank]] if bank in RULE_ARG else "none", "free")] + \
            ([] if bank in ("rule.start_name", "rule.call_user") else [FACT])   # those two name the speaker
    if cls == "L":
        op = "list.init" if bank.startswith("list.init.") else bank
        opts = tuple(LIST_OPS.values())
        out = [("op", "What does the speaker do? Answer with one of: " + ", ".join(opts) + ".", LIST_OPS[op], opts)]
        if op == "list.init":
            out.append(("items", "Which items does the line name, in order? (separated by commas)", f["items"],
                        "items"))
        elif op in LIST_ARG:
            out.append(("arg", "Which item or position does the line name? (the exact words, or 'none')",
                        f[LIST_ARG[op]], "free"))
        return out
    if cls == "U":
        e = f["e"]
        if bank.startswith("lookup.q."):
            opts = tuple(LOOK_TYPE.values()) + ("other",)
            return [_yn("act", f"Is the speaker asking the assistant to find out something about the {e}, without "
                        "stating the answer?", "yes"),
                    ("type", "What kind of answer does the speaker want? Answer with one of: " + ", ".join(opts) + ".",
                     LOOK_TYPE[bank.rsplit(".", 1)[1]], opts), FACT]
        if bank == "lookup.cf":
            return [_yn("act", f"Does the speaker say what they believe about the {e} and ask the assistant to check "
                        "it?", "yes"), FACT]
        return [_yn("act", f"Does the speaker pass on, as a statement, something they heard or read about the {e}?",
                    "yes"), ASKS, FACT]
    if cls == "Y":
        out = [(q, t, e, YN) for q, t, e in J._Q["Y"]]
        out.append(("name", "What name does the text give the assistant? (the name, or 'none')", f["A"], "free"))
        out.append(_yn("lookup", "Does the text say the assistant can look things up?",
                       "yes" if bank == "system.lookup" else "no"))
        return out
    raise KeyError(bank)


def value_questions(kind, d):
    """judges outside the line banks: d holds the item's own values."""
    if kind == "pool":
        qs = J._Q.get(d["vtype"], [])
        return [(q, t.format(value=d["value"]), e, YN) for q, t, e in qs]
    if kind == "listname":
        return [_yn("act", f"Is '{d['value']}' a natural name for {d['desc']}?", "yes")]
    if kind == "attr":
        return [_yn("act", f"Is the {d['attr']} something a person could look up about a {d['kind']}?", "yes"),
                _yn("type", f"For a {d['kind']}, is the {d['attr']} normally {d['type_np']}?", "yes")]
    if kind == "pred":
        return [_yn("act", f"Does 'the {d['name']} {d['pred']}' say that its {d['attr']} is {d['value']}?", "yes")]
    raise KeyError(kind)


# ---- fresh fills ----------------------------------------------------------------------------------------------
def _lookup_entity(rng, vt):
    import fake_data as F
    kinds = sorted(k for k, attrs in F.ENTITY_KINDS.items() if any(F.ATTR_TYPES[a] == vt for a in attrs))
    return f"{P.nonce(rng)} {rng.choice(kinds or sorted(F.ENTITY_KINDS)).capitalize()}"


def fresh_fill(rec, judge_model, tries=10):
    """a fill the author never saw: seeded by (judge, item id); v, items and e differ from the writing fill."""
    bank, old = rec["bank"], (rec.get("seeds") or {}).get("fill") or {}
    spec = specs.line_specs()[bank]
    rng = P.seeded("bankpass-judge", judge_model, rec["id"])
    for _ in range(tries):
        f = gates.sample_fill(bank, spec, rng)
        if bank.startswith("list."):
            lt = bank.rsplit(".", 1)[1] if bank.startswith("list.init.") else rng.choice(sorted(B.LIST_NAMES))
            vals = rng.sample(P.pool(lt).values, rng.choice((3, 4, 5)))
            f.update(L=B.LIST_NAMES[lt], items=vals, v=vals[rng.randrange(len(vals))])
        elif bank.startswith("lookup."):
            vt = bank.rsplit(".", 1)[1] if bank.startswith("lookup.q.") else rng.choice(sorted(B.LOOKUP_PRED))
            f["e"] = _lookup_entity(rng, vt)
            f["pred"] = B.LOOKUP_PRED[vt].format(v=rng.choice(P.pool(vt).values))
        elif bank == "swap.q":
            f["lab"] = rng.choice([d["label"] for d in BK.KEYS.values() if not d["noun"]])
        if all(f.get(h) != old.get(h) for h in ("v", "items", "e", "t", "A", "N", "X", "W", "n") if h in old):
            break
    return f


def fill_text(template, f):
    g = dict(f)
    if isinstance(g.get("items"), list):
        g["items"] = B.join_items(g["items"])
    return B.fill(template, **g)


# ---- prompt, regex, verdict -----------------------------------------------------------------------------------
def res(qs):
    out = []
    for qid, _, _, kind in qs:
        if kind == YN:
            body = "(?:yes|no)"
        elif isinstance(kind, tuple):
            body = "(?:" + "|".join(D.esc(o) for o in kind) + ")"
        else:
            body = D.noend()
        out.append(D.esc(qid) + ": " + body)
    return out


def prompt(line, qs):
    return J.prompt(line, [(q, t, e) for q, t, e, _ in qs]) + "\nAfter the last answer write END on its own line."


_LEAD = re.compile(r"^(?:a|an|the|my|his|her|their|our|your)\s+")


def norm_value(a):
    a = store.straight(str(a)).lower().strip()
    a = re.sub(r"^[\"'(\[]+|[\"'.!?,;:)\]]+$", "", a).strip()
    return _LEAD.sub("", a).strip()


def norm_items(a):
    parts = [p for x in (a if isinstance(a, list) else [a]) for p in re.split(r",|\band\b", str(x))]
    return [norm_value(x) for x in parts if norm_value(x)]


def match(answer, expected, kind):
    if answer is None:
        return False
    if kind == "items":
        return norm_items(answer) == norm_items(expected)
    if kind == "free" and expected == NOT_SAID:
        return norm_value(answer) in NOT_SAID_ALIASES
    return norm_value(answer) == norm_value(expected)


def parse(text, qs):
    got = {}
    for m in J._ANS.finditer(text or ""):
        got.setdefault(m.group(1).lower(), []).append(m.group(2))
    return {qid: (got[qid][0] if len(got.get(qid, [])) == 1 else None) for qid, _, _, _ in qs}


def verdict(answers, qs):
    return "keep" if all(match(answers.get(q), e, k) for q, _, e, k in qs) else "drop"


def record(model, answers, qs, fill, call):
    """the judge block an item carries (s2h judges): judge.vote reads model and verdict."""
    return {"model": model, "answers": answers, "expected": {q: (e if not isinstance(e, list) else ", ".join(e))
                                                             for q, _, e, _ in qs},
            "fill": fill, "verdict": verdict(answers, qs), "call": call}
