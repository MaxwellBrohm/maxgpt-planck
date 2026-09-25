"""RC-12 DEV generator tests, part 2: per-family layouts, distances, cells and exact balance of mention orders
(SPEC s3, G9). Called from test_gens.py; run(recs) returns a list of failure messages."""
import os
import re
import sys
from collections import Counter, defaultdict

import pools_vals as V
from fam_persist import QUESTION_TAILS
from pools_slots import INDIRECT_FORMS

F = []


def check(cond, msg):
    if not cond:
        F.append(msg)


def fam(recs, name, cell=None):
    return [r for r in recs if r["family"] == name and (cell is None or r["cell"] == cell)]


def main_p(r):
    return [p for p in r["probes"] if p["kind"] == "P"][0]


def placement(rs, name, early, last=False):
    """3/4 of main probes (last=True: of the later of the probes) at u12, the rest inside `early`."""
    turns = [max(p["turn"] for p in r["probes"]) if last else main_p(r)["turn"] for r in rs]
    n = len(rs)
    check(turns.count(12) == n - n // 4, f"{name}: {turns.count(12)} of {n} main probes at u12")
    check(all(t == 12 or t in early for t in turns), f"{name}: main probe outside u12/{early}")


def _content(text):
    sys.dont_write_bytecode = True
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "E004_general_updating", "code")
    sys.path.insert(0, d)
    import text_e004 as T
    sys.path.pop(0)
    return {w for w in T.words(text) if w not in T.STOPWORDS and w not in T.OBJECT_FREE_VERBS
            and w not in {v.lower() for v in V.ALL_VALUES}}


def fmt_ok(rule, arg, s):
    letters = [c for c in s if c.isalpha()]
    if rule == "caps":
        return len(letters) >= 5 and sum(c.isupper() for c in letters) / len(letters) >= 0.9
    if rule == "lowercase":
        return len(letters) >= 5 and not any(c.isupper() for c in letters)
    if rule == "first_word":
        return re.match(re.escape(arg) + r"\b", s) is not None
    if rule == "closing":
        return s.rstrip(" .!?").lower().endswith(arg.lower())
    if rule == "one_sentence":
        return not re.search(r"[.?!]\s+\S", s.strip())
    if rule == "question_end":
        return s.strip().endswith("?")
    if rule == "word":
        return re.search(r"\b" + re.escape(arg) + r"\b", s, re.I) is not None
    if rule == "brackets":
        return s.startswith("[") and s.endswith("]")
    raise ValueError(rule)


def t_recall(recs):
    main = [r for r in fam(recs, "RECALL") if r["cell"] != "abstain"]
    placement(main, "RECALL", (9, 10, 11))
    placement(fam(recs, "RECALL", "abstain"), "RECALL abstain", (9, 10, 11), last=True)
    placement(fam(recs, "T0"), "T0", (9, 10, 11))
    check(Counter(r["meta"]["pos"] for r in main) == Counter(first=16, middle=16, last=16), "RECALL positions")
    bins = {"d1-3": (1, 3), "d4-7": (4, 7), "d8-11": (8, 11)}
    for r in main:
        p = main_p(r)
        lo, hi = bins[r["cell"]]
        check(lo <= p["d"] <= hi, f"{r['id']} d {p['d']} outside {r['cell']}")
        st = [t for t in r["turns"] if t["kind"] in "SL"]
        check([t["kind"] for t in st].count("S") == 1 and len(st) == 3, f"{r['id']} needs 1 S + 2 L")
        check(all(t["i"] <= 11 for t in st), f"{r['id']} statement after u11")
        pos = ["first", "middle", "last"][[t["kind"] for t in st].index("S")]
        check(pos == r["meta"]["pos"], f"{r['id']} gold position")
        check(all(f["holder"] != "user" for t in st if t["kind"] == "L" for f in t["facts"]), f"{r['id']} lure holder")
    for r in fam(recs, "RECALL", "abstain"):
        g = sorted(p["grader"] for p in r["probes"])
        check(g == ["ABS", "VAL"] and r["meta"]["unit"] == "all", f"{r['id']} abstain layout")
    for r in fam(recs, "T0"):
        check(sorted(p["d"] for p in r["probes"]) in ([1, 1], [1, 2], [2, 2]), f"{r['id']} T0 d")
        check(len({p["pool"] for p in r["probes"]}) == 2, f"{r['id']} T0 value types must differ")
        check(not any(t["kind"] == "L" for t in r["turns"]), f"{r['id']} T0 has a lure")


def t_corr(recs):
    for cell in ("U-diff", "U-same", "C_noupd", "C_twoslot"):
        rs = fam(recs, "CORR", cell)
        placement(rs, f"CORR {cell}", (9, 10, 11))
        for r in rs:
            p = main_p(r)
            vals = [t["facts"][0]["value"] for t in r["turns"] if t["kind"] in "SCL"]
            check(len(vals) == len(set(vals)), f"{r['id']} repeated value")
            if cell != "C_noupd":
                check(p["dc"] is not None and p["dc"] >= 4, f"{r['id']} dc {p['dc']} < 4")
                check(1 <= r["meta"]["k"] <= 3, f"{r['id']} k")
            else:
                check(r["meta"]["k"] == 0 and not p["stale"], f"{r['id']} C_noupd corrected A")
            for t in r["turns"]:
                if t["kind"] == "C" and t["facts"][0]["form"] in INDIRECT_FORMS:
                    prev = r["turns"][t["i"] - 2]
                    check(prev["kind"] in "SC" and prev["facts"][0]["ref"] == t["facts"][0]["ref"],
                          f"{r['id']} u{t['i']} indirect correction not right after its object")
            latest = r["turns"][p["src"][0] - 1]
            if cell == "U-diff":
                check(r["meta"]["latest_form"] in ("pronoun", "ellipsis", "alias", "demonstrative"), f"{r['id']} form")
                check(not _content(p["question"]) & _content(latest["text"]),
                      f"{r['id']} U-diff question shares a content word with the latest correction")
            if cell == "U-same":
                check(r["meta"]["latest_form"] in ("full", "head"), f"{r['id']} form")
            if cell == "C_twoslot":
                b_corr = [t["i"] for t in r["turns"] if t["kind"] == "C" and t["facts"][0]["ref"] == "B"]
                b_orig = [t["i"] for t in r["turns"] if t["kind"] == "S" and t["facts"][0]["ref"] == "B"][0]
                check(min(b_corr) > p["src"][0] and len(b_corr) == r["meta"]["m_b"], f"{r['id']} B not after A's latest")
                check((b_orig > p["src"][0]) == (r["meta"]["layout"] == "blocks"), f"{r['id']} twoslot layout")
        if cell.startswith("U"):
            check(Counter(r["meta"]["b_place"] for r in rs) == Counter(before=8, after=8), f"CORR {cell} B place")
        if cell == "C_noupd":
            check(Counter(r["meta"]["a_pos"] for r in rs) == Counter(first=8, middle=8), "C_noupd A position")
    forms = Counter(r["meta"]["latest_form"] for r in fam(recs, "CORR", "U-diff"))
    check(set(forms.values()) == {4}, f"U-diff forms {forms}")


def t_bind(recs):
    pairs = defaultdict(list)
    for r in fam(recs, "BIND"):
        pairs[r["meta"]["pair_id"]].append(r)
    check(len(pairs) == 30, f"BIND pairs {len(pairs)}")
    orders = Counter()
    for pid, (a, b) in pairs.items():
        pa, pb = main_p(a), main_p(b)
        check(pa["question"] == pb["question"] and pa["gold"] == pb["gold"] and pa["turn"] == pb["turn"],
              f"{pid} twins ask different things")
        check(pa["turn"] in (10, 11, 12), f"{pid} probe outside u10-u12")
        diff = [i for i in range(12) if a["turns"][i]["text"] != b["turns"][i]["text"]]
        check(len(diff) == 2 and a["turns"][diff[0]]["text"] == b["turns"][diff[1]]["text"]
              and a["turns"][diff[1]]["text"] == b["turns"][diff[0]]["text"], f"{pid} twins differ beyond a swap")
        check(all(i + 1 <= 6 for i in diff), f"{pid} statements outside u1-u6")
        check({a["meta"]["order"], b["meta"]["order"]} == {"asked_first", "asked_second"}, f"{pid} orders")
        orders[a["meta"]["order"]] += 1
        orders[b["meta"]["order"]] += 1
    check(orders == Counter(asked_first=30, asked_second=30), f"BIND mention orders {orders}")
    for cell, n in (("owner", 12), ("perspective", 10), ("third-party", 8)):
        rs = [r for r in fam(recs, "BIND", cell) if r["id"].endswith("a")]
        hs = Counter(main_p(r)["holder"] in ("user",) for r in rs)
        if cell == "owner":
            check(hs[True] == 6 and hs[False] == 6, f"BIND owner asks {hs}")
        if cell == "perspective":
            qs = Counter(main_p(r)["question"] for r in rs)
            check(set(qs.values()) == {5}, f"BIND perspective questions {qs}")


def t_twohop(recs):
    rs = [r for r in fam(recs, "TWOHOP") if r["cell"] != "COMPOSE"]
    placement(rs, "TWOHOP", (9, 10, 11))
    check(set(Counter((r["cell"], r["meta"]["pos"]) for r in rs).values()) == {4}, "TWOHOP schema x position")
    for r in rs:
        p = main_p(r)
        q = re.findall(r"[a-z']+", p["question"].lower())
        qg = {tuple(q[i:i + 3]) for i in range(len(q) - 2)}
        for t in r["turns"]:
            if t["kind"] in "SL" and t["facts"][0]["value"]:
                w = re.findall(r"[a-z']+", t["text"].lower())
                shared = bool({tuple(w[i:i + 3]) for i in range(len(w) - 2)} & qg)
                is_echo = t["facts"][0]["value"] == p["echo_value"]
                check(shared == is_echo, f"{r['id']} u{t['i']} echo={is_echo} but shares a 3-run={shared}")
    comp = fam(recs, "TWOHOP", "COMPOSE")
    check(Counter(r["meta"]["gold_first"] for r in comp) == Counter({True: 8, False: 8}), "COMPOSE gold order")


def t_persist(recs):
    hold, over = fam(recs, "PERSIST", "hold"), fam(recs, "PERSIST", "override")
    check(Counter(r["meta"]["i_turn"] for r in hold) == Counter({1: 16, 2: 16}), "PERSIST I turns")
    check(set(Counter(r["meta"]["rules"][0][0] for r in hold).values()) == {4}, "PERSIST rule balance")
    check(Counter(r["meta"]["o_turn"] for r in over) == Counter({4: 8, 5: 8}), "PERSIST O turns")
    for r in hold + over:
        it, ot = r["meta"]["i_turn"], r["meta"]["o_turn"]
        want = [t for t in range(1, 13) if (t - ot >= 6 if ot else 6 <= t - it <= 11)]
        check(sorted(p["turn"] for p in r["probes"]) == want, f"{r['id']} checked turns")
        for p in r["probes"]:
            f = p["fmt"]
            check(fmt_ok(f["rule"], f["arg"], p["ideal"]), f"{r['id']} IDEAL breaks {f['rule']}")
            if f["absent"]:
                check(not fmt_ok(f["absent"]["rule"], f["absent"]["arg"], p["ideal"]), f"{r['id']} old rule kept")


def t_own_topic(recs):
    for cell in ("pick", "list"):
        rs = fam(recs, "OWN", cell)
        placement(rs, f"OWN {cell}", (9, 10, 11))
        for r in rs:
            p, q = main_p(r), r["meta"]["q_turn"]
            check(2 <= q <= 5 and 4 <= p["turn"] - q <= 9, f"{r['id']} Q/P turns")
            check(r["turns"][0]["kind"] == "L", f"{r['id']} no lure at u1")
            ls = [t["i"] for t in r["turns"] if t["kind"] == "L"]
            check(len(ls) == 2 and q < ls[1] < p["turn"], f"{r['id']} after-lure placement")
    check(Counter(main_p(r)["ideal_index"] for r in fam(recs, "OWN", "pick")) == Counter({0: 11, 1: 11, 2: 10}),
          "OWN IDEAL pick index")
    rs = fam(recs, "TOPIC")
    placement(rs, "TOPIC", (9, 10, 11))
    check({r["cell"] for r in rs} == {"topic"}, "TOPIC has one cell (position is a factor, not a cell)")
    check(Counter(r["meta"]["pos"] for r in rs) == Counter(first=16, middle=16, last=16), "TOPIC open position")
    for r in rs:
        p = main_p(r)
        m = r["meta"]
        check(3 <= m["digression"] <= 6, f"{r['id']} digression {m['digression']}")
        check(all(t["kind"] == "D" for t in r["turns"][m["block_start"] + m["block_len"] - 1:p["turn"] - 1]),
              f"{r['id']} digression turns are not all fillers")
        st = [t["i"] for t in r["turns"] if t["facts"] and t["facts"][0]["role"] in ("open", "closed_inline",
                                                                                     "closed_intro")]
        check(len(st) == 3 and max(st) <= 5, f"{r['id']} project statements outside u1-u5")


def run(recs):
    import test_gens_audit as A
    import test_gens_fam_b as B
    F.clear()
    for fn in (t_recall, t_corr, t_bind, t_twohop, t_persist, t_own_topic, B.t_role_lookup, B.t_loop_k, A.t_audit):
        fn(recs)
    check(len(QUESTION_TAILS) >= 12, "question tails")
    return list(F)
