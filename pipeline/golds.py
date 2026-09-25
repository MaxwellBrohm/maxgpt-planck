"""Gold answers re-derived from the skeleton JSON alone (no sampler state), plus the per-turn state notes (P-011).
derive(skel) -> {event_id: gold}; it replays each event's ops in turn order and checks, on the way, that every
op value is stated in its turn (exact text or must_include), that an exact query does not restate its answer, and
that no exact turn outside the event states the answer between the last mention and the query. A broken
skeleton raises GoldError; test_skeleton compares derive() with the stored golds."""
import re

import banks as B


class GoldError(Exception):
    pass


def value_re(v):
    flags = 0 if v[:1].isupper() else re.I
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?:s|es)?(?![A-Za-z0-9])", flags)


def turn_has(t, v):
    if t.get("text") and value_re(v).search(t["text"]):
        return True
    return any(value_re(v).search(x) for x in t.get("must_include", []))


def _by_i(skel):
    return {t["i"]: t for t in skel["turns"]}


def same_type_values(skel, vtype):
    """every value of vtype the conversation holds: slot values and every op value of a slot of that type."""
    vals = {s["value"] for s in skel["slots"].values() if s["type"] == vtype}
    for e in skel["events"]:
        for op in e["params"].get("ops", []):
            sid = op.get("slot")
            if sid and skel["slots"][sid]["type"] == vtype and op.get("value"):
                vals.add(op["value"])
        if e["kind"] == "S9" and e["params"]["vtype"] == vtype:
            vals |= set(e["params"]["stale"])
    return sorted(vals)


def _ops(skel, e):
    by_i = _by_i(skel)
    out = []
    for op in e["params"].get("ops", []):
        i = e["turns"][op["turn"]]
        vals = op["items"] if op.get("op") == "init" else [op["value"]]
        for v in vals:
            if not turn_has(by_i[i], v):
                raise GoldError(f"{e['id']} op {op['turn']}: {v} not stated in turn {i}")
        out.append((i, op))
    return sorted(out, key=lambda x: x[0])


def _check_window(skel, e, value, last_i, q_i):
    for t in skel["turns"]:
        if last_i < t["i"] < q_i and e["id"] not in t.get("events", []):
            if t.get("text") and value_re(value).search(t["text"]):
                raise GoldError(f"{e['id']}: {value} leaks in turn {t['i']}")
            if t["mode"] == "guided" and value not in t.get("must_exclude", []):
                raise GoldError(f"{e['id']}: turn {t['i']} does not exclude {value}")


def _slot_answer(skel, e, queried):
    ops = _ops(skel, e)
    hist, errs, others, twins, last_i = [], [], [], [], None
    for i, op in ops:
        if op.get("slot") == queried and op["op"] in ("plant", "set", "fix"):
            hist.append(op["value"])
            last_i = i
        elif op.get("slot") == queried and op["op"] == "err":
            errs.append(op["value"])
        elif op["op"] == "twin":
            twins.append(op["value"])
        elif op.get("slot"):
            others.append(op["value"])
    if not hist:
        raise GoldError(f"{e['id']}: queried slot never stated")
    ans = hist[-1]
    stale = []
    for v in hist[:-1] + errs:
        if v != ans and v not in stale:
            stale.append(v)
    cands = []
    for v in others + twins:
        if v != ans and v not in stale and v not in cands:
            cands.append(v)
    q = _by_i(skel)[e["turns"]["query"]]
    if q.get("text") and value_re(ans).search(q["text"]):
        raise GoldError(f"{e['id']}: query restates the answer")
    _check_window(skel, e, ans, last_i, e["turns"]["query"])
    return ans, stale, cands


def _list_answer(skel, e):
    cur, removed = [], []
    for _, op in _ops(skel, e):
        if op["op"] == "init":
            cur = list(op["items"])
        elif op["op"] == "add":
            cur.append(op["value"])
        elif op["op"] == "remove":
            cur.remove(op["value"])
            removed.append(op["value"])
        elif op["op"] == "move_first":
            cur.remove(op["value"])
            cur.insert(0, op["value"])
        elif op["op"] == "move_last":
            cur.remove(op["value"])
            cur.append(op["value"])
    q, arg = e["params"]["query"], e["params"]["query_arg"]
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven"]
    ords = ["first", "second", "third", "fourth", "fifth"]
    ans = {"first": lambda: cur[0], "last": lambda: cur[-1], "count": lambda: words[len(cur)],
           "ordinal": lambda: cur[ords.index(arg)], "other": lambda: [x for x in cur if x != arg][0],
           "contains": lambda: "yes" if arg in cur else "no"}[q]()
    return {"answer": ans, "list": cur, "query": q}


def rule_segments(skel):
    """{event_id: [segment gold]}: each rule holds from its user turn's reply to the next rule start."""
    starts = []
    for e in skel["events"]:
        if e["kind"] == "S4":
            for j, seg in enumerate(e["params"]["segments"]):
                role = "rule" if j == 0 else "override"
                starts.append((e["turns"][role], e["id"], seg))
    starts.sort(key=lambda x: x[0])
    out = {}
    for n, (i, eid, seg) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else 10 ** 9
        turns = [t["i"] for t in skel["turns"] if i < t["i"] < end and t["role"] == "assistant"
                 and not t.get("lookup_call")]
        out.setdefault(eid, []).append({"verifier": seg["verifier"], "args": seg["args"], "turns": turns})
    return out


def derive(skel):
    by_i = _by_i(skel)
    card = skel["slots"][skel["assistant"]["name"]]["value"]
    segs = rule_segments(skel)
    out = {}
    for e in skel["events"]:
        k, p = e["kind"], e["params"]
        slot_kind = k in ("S1", "S3") or (k, p.get("variant")) in (("S6", "two_persons"), ("S7", "given"))
        if slot_kind:
            queried = p.get("queried") or p["slot"]
            ans, stale, cands = _slot_answer(skel, e, queried)
            g = {"answer": ans, "stale": stale, "candidates": cands, "slot": queried}
            if k == "S6":
                g = {"answer": ans, "candidates": cands, "perspective": "your", "slot": queried}
            if k == "S7":
                g = {"answer": ans, "abstain": False, "candidates": cands, "slot": queried}
        elif k == "S2":
            g = _list_answer(skel, e)
        elif k == "S4":
            g = {"segments": segs[e["id"]]}
        elif k == "S5":
            ops = _ops(skel, e)
            g = {"origin": [op["value"] for _, op in ops if op["turn"].startswith("origin")],
                 "digression": [op["value"] for _, op in ops if op["turn"].startswith("digress")]}
        elif k == "S6" and p["variant"] == "user_vs_card":
            uname = _ops(skel, e)[0][1]["value"]
            a, b = (card, uname) if p["ask"] == "self" else (uname, card)
            g = {"answer": a, "candidates": [b], "perspective": "self" if p["ask"] == "self" else "your"}
        elif k == "S6":
            g = {"answer": None, "deny_self": True, "user_value": _ops(skel, e)[0][1]["value"], "perspective": "self"}
        elif k == "S7":
            g = {"answer": None, "abstain": True, "candidates": same_type_values(skel, p["vtype"])}
        elif k == "S8":
            g = {"act": p["act"], "identity": card if p["act"] == "who_are_you" else None, "max_w": 20}
        else:
            g = _lookup_gold(skel, e, by_i)
        out[e["id"]] = g
    return out


def _lookup_gold(skel, e, by_i):
    p = e["params"]
    ent = p["entity"]
    if p["need_kind"] == "context":
        ctx_t = by_i[e["turns"]["context"]]
        vals = [s["value"] for s in skel["slots"].values() if s["type"] == p["vtype"] and turn_has(ctx_t, s["value"])]
        return {"need": "context", "result": "hit", "lookup": False, "answer": vals[0], "stale": []}
    res_text = by_i[e["turns"]["tool"]]["text"]
    m = re.fullmatch(r"<result>" + re.escape(ent) + r" [a-z ]+: (.+)</result>", res_text)
    if not m:
        raise GoldError(f"{e['id']}: bad tool turn")
    val = None if m.group(1) == "not found" else m.group(1)
    q = by_i[e["turns"]["query"]]
    stale = [x for x in q["must_include"] if x != ent]
    res = "empty" if val is None else ("counterfactual" if stale else "hit")
    return {"need": "needed", "result": res, "lookup": True, "answer": val, "stale": stale}


def notes(skel):
    """{assistant turn index: note}: 'label: value (was old)' for every stated slot, and the list state."""
    events = []
    for e in skel["events"]:
        for i, op in _ops(skel, e) if e["params"].get("ops") else []:
            events.append((i, e, op))
    events.sort(key=lambda x: x[0])
    state, prev, lists, out, n = {}, {}, {}, {}, 0
    for t in skel["turns"]:
        while n < len(events) and events[n][0] <= t["i"]:
            _, e, op = events[n]
            n += 1
            if e["kind"] == "S2":
                lists[e["params"]["list_name"]] = _apply_list(lists.get(e["params"]["list_name"]), op)
            elif op["op"] in ("plant", "set", "fix") and op.get("slot"):
                sid = op["slot"]
                if sid in state and state[sid] != op["value"]:
                    prev[sid] = state[sid]
                state[sid] = op["value"]
        if t["role"] == "assistant":
            parts = []
            for sid, v in state.items():
                s = skel["slots"][sid]
                lab = B.label(s["key"], s.get("noun"))
                parts.append(f"{lab}: {v}" + (f" (was {prev[sid]})" if sid in prev else ""))
            for name, lst in lists.items():
                parts.append(f"{name}: " + ", ".join(lst))
            out[t["i"]] = "; ".join(parts) if parts else None
    return out


def _apply_list(lst, op):
    lst = list(lst or [])
    if op["op"] == "init":
        return list(op["items"])
    if op["op"] == "add":
        lst.append(op["value"])
    else:
        lst.remove(op["value"])
        if op["op"] == "move_first":
            lst.insert(0, op["value"])
        elif op["op"] == "move_last":
            lst.append(op["value"])
    return lst
