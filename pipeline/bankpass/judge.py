"""Cross-judging (BANKPASS s2e). The two teachers that did not write an item read it on a FRESH fill (values the author
never saw) and answer short closed questions: the speech act and an extraction. An item is kept only when both judges
answer every question with the expected answer. Disagreements are logged by (author, judge) pair, so a teacher that
rejects the others' lines far more than its own shows up. Claude wrote the questions (rubric, not training text)."""
import re

from bankpass import store

NOT_SAID = "not said"
RULES = ("max_words", "one_sentence", "end_question", "call_user", "avoid_word", "start_name")
# (class or key role) -> [(qid, question, expected)]; {label}, {value}, {rule} and {arg} are filled per item
_Q = {
    "plant": [("act", "Does the speaker state, as a plain statement, a fact about their own {label}? (yes or no)", "yes"),
              ("ext", "After this line, what is the speaker's {label}? (the exact words, or 'not said')", "{value}")],
    "corr": [("act", "Is the speaker replacing an earlier value of their {label} with a new one? (yes or no)", "yes"),
             ("ext", "After this line, what is the speaker's {label}? (the exact words, or 'not said')", "{value}")],
    "query": [("act", "Is the speaker asking for their own {label} without stating it? (yes or no)", "yes"),
              ("ext", "Does this line itself say what the speaker's {label} is? (the exact words, or 'not said')",
               NOT_SAID)],
    "twin": [("act", "Is the value mentioned as something that is NOT the speaker's {label}? (yes or no)", "yes"),
             ("ext", "After this line, what is the speaker's {label}? (the exact words, or 'not said')", NOT_SAID)],
    "M": [("act", "Could a person start a sentence with this when fixing something they just said? (yes or no)",
           "yes")],
    "O": [("act", "Is this a natural way for a person to start a chat with an assistant? (yes or no)", "yes"),
          ("fact", "Does it tell a fact about the speaker, such as a name, place, job, pet, plan or relative? (yes "
                   "or no)", "no")],
    "C": [("act", "Is the speaker ending the chat? (yes or no)", "yes"),
          ("q", "Does the line ask a question? (yes or no)", "no")],
    "R": [("ext", "Which rule does the speaker ask the assistant to follow from now on? Answer with one of: "
                  + ", ".join(RULES) + ", none.", "{rule}"),
          ("arg", "What exact word, name or number goes with that rule? (the words, or 'none')", "{arg}")],
    "Y": [("act", "Does this text give the assistant any trait beyond its name, being an assistant and looking "
                  "things up (such as plans, family, a body, a past, preferences or places)? (yes or no)", "no")],
    "hobby": [("act", "Is {value} a sport or a competitive game? (yes or no)", "no")],
    "plan": [("act", "Does 'my {value} is on Monday' make sense as something a person says? (yes or no)", "yes")],
    "object": [("act", "Is a {value} something a person buys and whose colour could be named? (yes or no)", "yes")],
}
_Q["bait"] = _Q["query"]
_ANS = re.compile(r"^\s*([a-z_]+)\s*[:.)]\s*(.+?)\s*$", re.I | re.M)


def _norm(a):
    a = store.straight(a).lower().strip().strip(".!'\"")
    return {"y": "yes", "n": "no", "not mentioned": NOT_SAID, "not stated": NOT_SAID, "none": "none"}.get(a, a)


def questions(rec, fill, label=None, rule=None):
    """the closed questions for one item on its fresh fill. key lines use their role; pools their value type."""
    if rec["class"] == "K":
        table = _Q[rec["bank"].split(".")[2]]
    elif rec["class"] == "P":
        table = _Q.get(rec["bank"].split(".", 1)[1], [])
    else:
        table = _Q.get(rec["class"], [])
    sub = {"label": label or "", "value": fill.get("v") or rec["text"], "rule": (rule or ("none", "none"))[0],
           "arg": (rule or ("none", "none"))[1]}
    return [(qid, q.format(**sub), exp.format(**sub)) for qid, q, exp in table]


def prompt(line, qs):
    """the judge prompt: the filled line, then numbered closed questions, answered one per line as 'qid: answer'."""
    head = ("Read the line below and answer each question on its own line, written as the question name, a colon "
            "and a short answer. Write nothing else.\n\nLine: " + line + "\n\n")
    return head + "\n".join(f"{qid}: {q}" for qid, q, _ in qs)


def parse(reply, qs):
    """{qid: answer} for the questions answered once each; a missing or doubled answer is None."""
    got = {}
    for m in _ANS.finditer(reply or ""):
        qid = m.group(1).lower()
        got.setdefault(qid, []).append(_norm(m.group(2)))
    return {qid: (got[qid][0] if len(got.get(qid, [])) == 1 else None) for qid, _, _ in qs}


def verdict(answers, qs):
    """'keep' when every answer equals its expected answer (values compared after straightening, any case)."""
    for qid, _, exp in qs:
        if answers.get(qid) is None or answers[qid] != _norm(exp):
            return "drop"
    return "keep"


def judge_record(model, answers, qs, fill):
    return {"model": model, "answers": answers, "expected": {q: _norm(e) for q, _, e in qs}, "fill": fill,
            "verdict": verdict(answers, qs)}


def vote(rec):
    """kept only when exactly two judges, neither the author, both 'keep' (s2e)."""
    js = rec.get("judges") or []
    author = rec["author"].get("model")
    models = [j["model"] for j in js]
    if len(js) != 2 or len(set(models)) != 2 or author in models:
        return "JUDGE_MISSING"
    if any(m not in store.TEACHERS for m in models):
        return "JUDGE_NOT_PINNED"
    return None if all(j["verdict"] == "keep" for j in js) else "JUDGE_DROP"


def disagreements(recs):
    """{(author, judge): [n judged, n dropped]}, and per judge the drop rate on its own lines vs the others'."""
    pairs = {}
    for r in recs:
        for j in r.get("judges") or []:
            k = (r["author"].get("model"), j["model"])
            n = pairs.setdefault(k, [0, 0])
            n[0] += 1
            n[1] += j["verdict"] != "keep"
    return pairs
