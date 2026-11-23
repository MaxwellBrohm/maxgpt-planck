"""RC-12 DEV generator tests, part 3: ROLE, LOOKUP, LOOP and K layouts (split from test_gens_fam.py to keep
files under 250 lines; failures go to test_gens_fam.F). Lock step A2b's content rule (fix round, 2026-11-19): no
LOOKUP probe question holds a not-found cue (grade_cues.lookup_any: LOOKUP_ABS, LOOKUP_ONLY, LOOKUP_ON, negated or
not), because G-ABS ignores those cues on a probe whose question holds one; it fails every draw of such content at
test_gens, so B4's early stop sees it. Lock step A2c's content rule (2026-11-19, M7): G-ABS reads a pool value in any
case on LOOKUP's X probe, so no capitalized value of a pool the LOOKUP tables read may be an ordinary word; the
mechanical part: none is written in lower case in any content string of any content module (sealed_kit
CONTENT_MODULES: templates, fillers, acks, lists) or in any built turn or IDEAL reply (lookup_case)."""
import re
from collections import Counter, defaultdict

import grade_cues as GC
import grade_text as GT
import pools_vals as V
from test_gens_fam import check, fam, main_p, placement


def t_role_lookup(recs):
    rs = fam(recs, "ROLE")
    placement(rs, "ROLE", (8, 9, 10, 11))
    check(set(Counter((r["meta"]["named"], r["meta"]["name_pos"]) for r in rs).values()) == {8}, "ROLE balance")
    for r in rs:
        p = main_p(r)
        x = [q for q in r["probes"] if q["kind"] == "X"][0]
        st = [t["i"] for t in r["turns"] if t["kind"] in "SL"]
        check(6 <= x["turn"] <= 11 and x["turn"] != p["turn"] and x["turn"] > max(st), f"{r['id']} X turn")
        check(max(st) <= 6, f"{r['id']} ROLE statements past u6")
        names = [v for t in r["turns"] if t["kind"] in "SL" for v in t["vals"] if v in V.PERSON]
        check(len(names) == 3 and ["first", "middle", "last"][names.index(p["gold"])] == r["meta"]["name_pos"],
              f"{r['id']} user name position")
    check(Counter((r["cell"], r["meta"]["len_rank"]) for r in rs) == Counter(
        {(c, t): 8 for c in ("named", "unnamed") for t in ("long", "mid", "short")}), "ROLE len ranks (audit)")
    rs = fam(recs, "LOOKUP")
    placement(rs, "LOOKUP", (9, 10, 11), last=True)
    xf = Counter((r["cell"], [q["turn"] for q in r["probes"] if q["kind"] == "X"][0] < main_p(r)["turn"]) for r in rs)
    check(xf == Counter({(c, f): 8 for c in ("rota", "tour", "menu") for f in (True, False)}),
          f"LOOKUP X/P order (audit: 8/8) {xf}")
    check(set(Counter((r["cell"], r["meta"]["old_first"], r["meta"]["asked"]) for r in rs).values()) == {4},
          "LOOKUP balance")
    for r in rs:
        p = main_p(r)
        x = [q for q in r["probes"] if q["kind"] == "X"][0]
        tt = {k: r["turns"][v - 1]["text"] for k, v in r["meta"]["table_turns"].items()}
        check(all(r["turns"][v - 1]["kind"] == "T" and v <= 4 for v in r["meta"]["table_turns"].values()),
              f"{r['id']} table turns")
        asked = r["meta"]["asked"]
        other = "old" if asked == "new" else "new"
        check(f"{p['key']}: {p['gold']}" in tt[asked], f"{r['id']} gold not in the named table")
        check(x["key"] + ":" not in tt[asked] and x["key"] + ":" in tt[other], f"{r['id']} X key placement")
        check((x["turn"] < p["turn"]) == r["meta"]["x_first"] and min(x["turn"], p["turn"]) >= 6, f"{r['id']} X/P")
        check(not any(GC.lookup_any(GT.norm(q["question"])) for q in r["probes"]),
              f"{r['id']} LOOKUP question holds a not-found cue (A2b)")
    lookup_case(recs)


def leaves(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(k)
            yield from leaves(v)
    elif isinstance(x, (list, tuple, set, frozenset)):
        for v in x:
            yield from leaves(v)


def lookup_case(recs):
    """A2c's content rule (module docstring): every capitalized value of a pool the LOOKUP tables read, as a whole
    word in lower case, in no content string and in no built turn or IDEAL reply (a seed-independent check: the content
    strings hold every filler, template and list whatever the draw)."""
    import importlib
    import pools_misc as M
    import sealed_kit as K
    caps = sorted({v for s in M.LOOKUP_SHAPES.values() for v in V.POOLS[s["vals"]] if v[:1].isupper()})
    texts = [x for m in K.CONTENT_MODULES for n in importlib.import_module(m).CONTENT
             for x in leaves(getattr(importlib.import_module(m), n.rstrip("*")))]
    texts += [t[k] for r in recs for t in r["turns"] for k in ("text", "ideal") if t.get(k)]
    low = [v for v in caps if any(re.search(r"(?<![A-Za-z0-9])" + re.escape(v.lower()) + r"(?![A-Za-z0-9])", x)
                                  for x in texts)]
    check(not low, f"a capitalized LOOKUP value is written in lower case, so it is an ordinary word (A2c): {low}")


def t_loop_k(recs):
    for r in fam(recs, "LOOP"):
        check(len(r["probes"]) == 12 and all(p["grader"] == "LOOP" for p in r["probes"]), f"{r['id']} probes")
        texts = [t["text"] for t in r["turns"]]
        check(len(set(texts)) == 12, f"{r['id']} repeated request")
        check(not any(re.search(r"repeat|summar|again", t, re.I) for t in texts), f"{r['id']} repeat request")
    ks = defaultdict(dict)
    for r in fam(recs, "K"):
        ks[r["meta"]["pair_id"]][r["cell"]] = r
    check(len(ks) == 32 and all(set(v) == {"followup", "control"} for v in ks.values()), "K pairs")
    placement([v["followup"] for v in ks.values()], "K", (9, 10, 11))
    for pid, v in ks.items():
        ref = v["followup"]["meta"]["referent"]
        check(ref not in main_p(v["followup"])["question"] and ref in main_p(v["control"])["question"], f"{pid} K ref")
