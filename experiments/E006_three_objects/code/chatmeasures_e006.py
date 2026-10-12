"""E006 chat-probe measures (notes.txt Q-CHAT; the E005 audit's Q9 definitions, plus the critique's fixes 4 and 5).
Reads chat-probe transcripts (run_chat.py records: one per conversation, turns with assistant, stopped_eos,
n_gen_tokens, flags.hit_max; checks {name: True/False/None}). No model. Per assistant turn:
  TF      the first sentence (reply.strip() split once at the first "\\n" or at the first whitespace after . ! ?)
          matches a training answer template: any "ans"/"ans_upd" entry of pools_train.POOLS, {v} = one
          alphanumeric word, {o} = [A-Za-z][A-Za-z'\\- ]{0,40} (lazy), case-insensitive, not preceded by a letter
          (the audit's "loose" regexes; strict: {v} a training value)
  TF2     TF, or: the first sentence has 3 words or fewer and the SECOND sentence matches (so an opener such as
          "Sure!" cannot hide a template answer; critique 4)
  WHOLE   the template matches anywhere in the reply (the audit's "template share")
  LOOP    some line occurs 3 or more times, counting only lines outside ``` fences with 3+ alphabetic words that are
          not code or brace lines (critique 5; LOOP_OLD, E005's rule: any stripped non-empty line, is kept too)
  CAP     flags.hit_max;  EOS  stopped at end of turn
Per conversation: CHECKS = graded checks passed (None counts as not passed; 73 per model)."""
import json
import re
from collections import Counter

from pools_train import POOLS

VALS = sorted({v for P in POOLS.values() for v in P["values"]}, key=len, reverse=True)
V_STRICT = "(?:" + "|".join(re.escape(v) for v in VALS) + ")"
V_LOOSE = r"[A-Za-z0-9]+"
OBJ = r"[A-Za-z][A-Za-z'\- ]{0,40}?"
TEMPLATES = sorted({t for P in POOLS.values() for key in ("ans", "ans_upd") for t in P[key]})
SPLIT = re.compile(r"\n|(?<=[.!?])\s")
ALPHA_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
CODE_LINE = re.compile(r"^[\s}\])({\[]*$|[{};]\s*$|^\s*(?:#include|import |def |return |//)")


def _tre(t, vpat):
    out = ""
    for p in re.split(r"(\{v\}|\{o\})", t):
        out += vpat if p == "{v}" else (OBJ if p == "{o}" else re.escape(p))
    return re.compile(r"(?<![A-Za-z])" + out, re.I)


R_LOOSE = [_tre(t, V_LOOSE) for t in TEMPLATES]
R_STRICT = [_tre(t, V_STRICT) for t in TEMPLATES]


def has_tpl(text, strict=False):
    return any(r.search(text) for r in (R_STRICT if strict else R_LOOSE))


def first_two(reply):
    parts = SPLIT.split(reply.strip(), maxsplit=1)
    first = parts[0].strip()
    rest = parts[1].strip() if len(parts) > 1 else ""
    second = SPLIT.split(rest, maxsplit=1)[0].strip() if rest else ""
    return first, second


def loop_old(reply):
    c = Counter(l.strip() for l in reply.split("\n") if l.strip())
    return bool(c) and max(c.values()) >= 3


def loop_lines(reply):
    out, fence = [], False
    for line in reply.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
            continue
        if fence or not s or len(ALPHA_WORD.findall(s)) < 3 or CODE_LINE.search(s):
            continue
        out.append(s)
    return out


def loop_new(reply):
    c = Counter(loop_lines(reply))
    return bool(c) and max(c.values()) >= 3


def turn_row(reply, stopped, hit_max, n_gen):
    first, second = first_two(reply)
    tf = has_tpl(first)
    short = len(first.split()) <= 3
    return {"TF": tf, "TF2": tf or (short and bool(second) and has_tpl(second)), "TFS": has_tpl(first, True),
            "WHOLE": has_tpl(reply), "WHOLE_S": has_tpl(reply, True), "LOOP": loop_new(reply),
            "LOOP_OLD": loop_old(reply), "CAP": bool(hit_max), "EOS": bool(stopped), "n_gen": n_gen, "first": first,
            "latest_plan": "latest plan" in reply.lower()}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def rows(convs):
    """-> per-turn rows with conv id, category, turn index, and the measures."""
    out = []
    for c in convs:
        prev = None
        for i, t in enumerate(c["turns"]):
            r = turn_row(t["assistant"], t["stopped_eos"], t["flags"]["hit_max"], t["n_gen_tokens"])
            r.update(conv=c["id"], cat=c["cat"], turn=i, same_prev=(prev is not None and r["first"] == prev))
            prev = r["first"]
            out.append(r)
    return out


def checks(convs):
    """-> {conv id: (passed, graded total)}; None counts as not passed."""
    return {c["id"]: (sum(v is True for v in c["checks"].values()), len(c["checks"])) for c in convs}


def summary(convs):
    R = rows(convs)
    n = len(R)
    ch = checks(convs)
    stopped = [r["n_gen"] for r in R if r["EOS"]]
    s = {k: sum(r[k] for r in R) / n for k in ("TF", "TF2", "TFS", "WHOLE", "WHOLE_S", "LOOP", "LOOP_OLD", "CAP", "EOS",
                                               "latest_plan")}
    s.update(n_turns=n, n_conv=len(convs), CHECKS=sum(p for p, _ in ch.values()), checks_total=sum(t for _, t in ch.values()),
             CAP_n=sum(r["CAP"] for r in R), EOS_n=sum(r["EOS"] for r in R), same_prev=sum(r["same_prev"] for r in R),
             n_prev=sum(1 for r in R if r["turn"] > 0),
             mean_new_tok_stopped=round(sum(stopped) / len(stopped), 1) if stopped else None,
             prompt_sig=prompt_sig(convs))
    cats = sorted({r["cat"] for r in R})
    s["by_cat"] = {c: {"CAP": sum(r["CAP"] for r in R if r["cat"] == c) / sum(1 for r in R if r["cat"] == c),
                       "CHECKS": sum(ch[x["id"]][0] for x in convs if x["cat"] == c)} for c in cats}
    return s


def prompt_sig(convs):
    import hashlib
    sig = [(c["id"], c["cat"], c["system"], [t["user"] for t in c["turns"]]) for c in convs]
    return hashlib.sha256(json.dumps(sig).encode()).hexdigest()[:16]
