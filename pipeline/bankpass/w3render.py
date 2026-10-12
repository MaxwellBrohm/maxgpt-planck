"""W3: a planned or derived call (any kind) -> its prompt, decode spec and output parse (BANKPASS s2b). Kinds:
  line      a line bank call (prompts.py for K, M, O, C, S, Y; prompts3.py for R, L, U, swap.q)
  pool      a teacher pool call; topic: a topic call (verbalized)
  label, notewas, listname, vote, relfeat, attr, pred      labels and features (prompts3.py)
  pos, verbs, ly, wordset, intent, topen, para, check, group   word labels, topic-dependent calls, paraphrases
  judge     a cross-judge call (judge3.py)
Every author call names a sampling preset (each teacher's D4 shared preset; Gemma has only its card); judge, vote,
label-feature and check calls name none, so the server's --sampling-json (temperature 0) applies: greedy and
reproducible."""
from bankpass import judge3 as J3, prompts as PR, prompts3 as P3, prompts4 as P4

AUTHOR_PRESET = {"qwen3.5-9b": "shared", "ministral-3-8b": "shared", "gemma-4-12b": "card"}
GREEDY_KINDS = {"judge", "vote", "relfeat", "check", "group", "pos", "verbs", "ly"}
PROMPT_VERSIONS = {"prompts": PR.PROMPT_VERSION, "prompts3": P3.PROMPT_VERSION, "prompts4": P4.PROMPT_VERSION}


def _line(c):
    bank = c["bank"]
    if c["class"] == "K" or bank in PR.CLASS_ASK:
        call = dict(c, voice=c.get("voice"), mined=c.get("mined"))
        prompt = PR.prompt_for(call)
    else:
        prompt = P3.line_prompt(bank, c["n"], c.get("fills") or [], c.get("mined"), c.get("voice"))
    from bankpass import gen
    lits, banned = gen.literals_of(c)
    dec4 = c.get("call_id", "").endswith(("v4", "v4r1"))   # amendment 4's decode; amendment 6 took it back out
    if dec4 and c["class"] != "K" and c.get("fills") and bank in LIT4:
        lits = [LIT4[bank](f) for f in c["fills"]]
    verb = bool(c.get("verbalized"))
    question = dec4 and c["class"] == "K" and bank.split(".")[2] in ("query", "bait")
    spec = PR.decode_spec(c["teacher"], lits, verbalized=verb, question=question)
    spec["max_tokens"] = 60 * c["n"] + 60          # amendment 2 (was 40 n + 40: 7.5% of Qwen's calls cut off)
    spec["banned_values_not_sent"] = sorted(set(banned))     # see notes: bad_words cap; checked on CPU instead
    return prompt, spec, ("lines", verb)


def _join(items):
    from banks import join_items
    return join_items(items)


LIT4 = {"rule.max_words": lambda f: f["N"], "swap.q": lambda f: f["lab"], "lookup.cf": lambda f: f"{f['e']} {f['pred']}",
        "lookup.ctx": lambda f: f"{f['e']} {f['pred']}", "list": lambda f: _join(f["items"])}
LIT4.update({f"list.init.{t}": LIT4["list"] for t in ("grocery", "city", "name", "chore")})


def render(c):
    """-> (prompt, spec, parse) where parse is ("lines", verbalized) or ("rows", None)."""
    k, m = c["kind"], c["teacher"]
    if k == "line":
        return _line(c)
    if k in ("pool", "topic"):
        prompt = (P4.pool_prompt2(c["bank"].split(".", 1)[1], c["n"], c.get("voice")) if c.get("v") == 2 and k == "pool"
                  else P4.topic_prompt2(c["n"], c["seed_words"], c.get("voice")) if c.get("v") == 2
                  else PR.prompt_for(dict(c, voice=c.get("voice"))))
        verb = k == "topic"
        spec = PR.decode_spec(m, [None] * c["n"], verbalized=verb)
        spec["max_tokens"] = (40 if verb else 24) * c["n"] + 40
        return prompt, spec, ("lines", verb)
    if k == "label":
        prompt, frags, mt = P3.label_prompt(c["key"], c["n"], c.get("o")), P3.line_res("label", [c.get("o")] * c["n"]), 200
    elif k == "notewas":
        prompt = P3.notewas_prompt2(c["n"]) if c.get("v") == 2 else P3.notewas_prompt(c["n"])
        frags, mt = P3.line_res("x", range(c["n"])), 120
    elif k == "listname":
        prompt, frags, mt = P3.listname_prompt(c["vtype"], c["n"]), P3.line_res("x", range(c["n"])), 160
    elif k == "vote":
        prompt, frags, mt = P3.vote_prompt(c["what"], c["cands"]), P3.line_res("vote", None), 12
    elif k == "relfeat":
        prompt, frags, mt = P3.relfeat_prompt(c["rows"]), P3.line_res("relfeat", c["rows"]), 12 * len(c["rows"]) + 20
    elif k == "attr":
        prompt, frags, mt = P3.attr_prompt(c["value"], c["n"]), P3.line_res("attr", range(c["n"])), 120
    elif k == "pred":
        rows = [tuple(r) for r in c["rows"]]
        prompt, frags, mt = P3.pred_prompt(rows), P3.line_res("pred", rows), 24 * len(rows) + 20
    elif k in ("pos", "verbs", "ly"):
        rows = [tuple(r) for r in c["rows"]] if k == "ly" else c["rows"]
        v2 = k == "pos" and c.get("v") == 2
        prompt = (P4.pos_prompt2 if v2 else {"pos": P4.pos_prompt, "verbs": P4.verbs_prompt, "ly": P4.ly_prompt}[k])(rows)
        frags, mt = P4.word_res("pos2" if v2 else k, rows), {"pos": 8, "verbs": 14, "ly": 12}[k] * len(rows) + 20
    elif k == "wordset":
        prompt, frags, mt = P4.wordset_prompt(c["topic"]), P4.wordset_res(), 260
    elif k == "intent":
        prompt, frags, mt = P4.intent_prompt(c["topic"], c["n"]), P3.line_res("x", range(c["n"])), 30 * c["n"] + 30
    elif k == "topen":
        prompt, frags, mt = (P4.topen_prompt(c["topics"], c["k"]), P3.line_res("x", range(c["n"])),
                             30 * c["n"] + 30)
    elif k == "para":
        prompt, frags, mt = P4.para_prompt(), P3.line_res("para", range(2)), 500
    elif k == "check":
        prompt, frags, mt = P4.check_prompt(c["head"], c["tail"]), P4.check_res(), 130
    elif k == "group":
        prompt, frags, mt = P4.group_prompt(c["topic"]), P4.group_res(), 16
    elif k == "judge":
        qs = [tuple(q) if not isinstance(q[3], list) else (q[0], q[1], q[2], tuple(q[3])) for q in c["qs"]]
        prompt, frags, mt = J3.prompt(c["text"], qs), J3.res(qs), 24 * len(qs) + 20
    else:
        raise KeyError(f"unknown call kind {k!r}")
    return prompt, P3.spec(m, frags, mt), ("rows", None)


def preset(c):
    return None if c["kind"] in GREEDY_KINDS else AUTHOR_PRESET[c["teacher"]]


def parse(c, text, mode, spec):
    """-> (lines, likelihoods, problem)."""
    kind, verb = mode
    if kind == "lines":
        return PR.parse_lines(text, c["n"], verb)
    rows, prob = P3.rows_of(text, spec["n"])
    return rows or [], [None] * len(rows or []), prob
