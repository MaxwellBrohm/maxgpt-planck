"""Validation of the AL items (items_al.py). No model, no tokenizer. run_all(items) -> failures; prints tables.
Gates:
  G1 IDEAL (builder's gold = latest statement about the asked object) and LINK (text-level true linking: a name
     belongs to the object named in the statement that introduces it; a statement naming an object is that
     object's; the latest statement attributed to the asked object answers) both 1.00
  G2 LN "latest statement that holds a title + name, else last mention" <= 0.30 overall; AR (E005's diagnostic:
     latest alias correction, else O2) <= 0.30 overall
  G3 every E004 oracle and shortcut rule (oracles_e004.ORACLES + shortcuts_x.EXTRA = shortcuts_x.FAKE) <= 0.60
     overall; the topic tracker T (E004/E005 diagnostic, adjacency without names) is reported, not gated
  G4 structure and balance: 16 per cell, value type x placement 4 x 4 per cell, question form 8/8, the cell's
     binary factors 8/8 (6/6, 2/2), the gold's statement kind per cell, alias corrections name no object and
     follow the original that defines their name, the latest name-holding statement is the gold only in AL3
  G5 no value of any pool (the answer word included) in any question or prefix; >= 2 candidates; distinct values
  G6 names: AL surnames disjoint from the E005 training pool and the E004 eval surnames, never a value or an
     object word, in no E004 eval/dev/probe text; two aliases of one item never share a surname
Purity against the E005 training stream (purity_al): see its docstring."""
import re
from collections import Counter, defaultdict

import oracles_e004 as O
import oracles_e005 as O5
import shortcuts_x as SX
import heldout_e004 as H
import pools_alias_train as PA
from pools_eval import ALL_VALUES
from pools_train import POOLS as TRAIN_POOLS
from text_e004 import words, values_in
import items_al as A

NAME_RE = re.compile(r"(?<![A-Za-z])(?:Mr\.|Mrs\.|Ms\.|Dr\.|Pastor|Professor|Chef|Captain) [A-Z][a-z]+")
MAX_LN, MAX_RULE = 0.30, 0.60


class ViewAL(O5.View5):
    """E004's text view; both aliases of the item count as object terms (E005's View5)."""


def ln_latest_name(V):
    return V.latest(lambda ti, t, v: bool(NAME_RE.search(t)), O.o2_last(V))


def link(V):
    objs = V.it["objects"]
    named = lambda t: [i for i, (ph, h) in enumerate(objs) if O.names(t, ph) or O.names(t, h)]
    owner_of_name, owner, prev = {}, [], None
    for ti, t, v in V.stmts:
        n = named(t)
        for m in NAME_RE.finditer(t):
            if len(n) == 1:
                owner_of_name.setdefault(m.group(0), n[0])
        if len(n) == 1:
            cur = n[0]
        else:
            hits = [owner_of_name[m.group(0)] for m in NAME_RE.finditer(t) if m.group(0) in owner_of_name]
            cur = hits[0] if hits else prev
        owner.append(cur)
        prev = cur
    for (ti, t, v), o in zip(reversed(V.stmts), reversed(owner)):
        if o == V.it["asked"]:
            return v
    return None


RULES = SX.FAKE + [("T topic tracker (diagnostic)", SX.topic_tracker), ("AR latest alias correction", O5.ar_alias),
                   ("LN latest statement with a name", ln_latest_name), ("LINK text-level linking", link),
                   ("IDEAL", O.ideal)]


def rule_table(items):
    """{rule: {cell or 'all': share right}}"""
    out = {}
    for name, fn in RULES:
        per = defaultdict(list)
        for it in items:
            r = fn(ViewAL(it)) == it["gold"]
            per[it["cell"]].append(r)
            per["all"].append(r)
            per["pl:" + it["meta"]["placement"]].append(r)
        out[name] = {k: sum(v) / len(v) for k, v in per.items()}
    return out


def print_table(tab):
    cols = ["all"] + list(A.CELLS) + ["pl:adjacent", "pl:filler", "pl:other_obj"]
    head = f"{'rule':42s}" + "".join(f"{c.replace('pl:', ''):>10s}" for c in cols)
    print(head + "\n" + "-" * len(head))
    for name, _ in RULES:
        print(f"{name:42s}" + "".join(f"{tab[name][c]:10.2f}" for c in cols))


def rule_gates(tab):
    fail = []
    for name in ("IDEAL", "LINK text-level linking"):
        if tab[name]["all"] != 1.0:
            fail.append(f"G1 {name} {tab[name]['all']:.3f}")
    for name in ("LN latest statement with a name", "AR latest alias correction"):
        if tab[name]["all"] > MAX_LN:
            fail.append(f"G2 {name} {tab[name]['all']:.3f} > {MAX_LN}")
    fail += [f"G3 {n} {tab[n]['all']:.3f} > {MAX_RULE}" for n, _ in SX.FAKE if tab[n]["all"] > MAX_RULE]
    return fail


def gold_kind(it):
    s = [x for x in it["stmts"] if x["obj"] == it["asked"]][-1]
    return "orig" if s["role"] == "orig" else s["ref"]


EXPECT_KIND = {"AL1": {"orig", "full", "head"}, "AL2": {"alias"}, "AL3": {"alias"}, "AL4": {"head"}}


def structure(items):
    fail = []
    by = defaultdict(list)
    for it in items:
        by[it["cell"]].append(it)
    if len(items) != 64 or sorted(by) != list(A.CELLS) or any(len(v) != A.N_CELL for v in by.values()):
        fail.append(f"G4 counts {len(items)} {dict((k, len(v)) for k, v in by.items())}")
    for cell, its in by.items():
        vp = Counter((it["vtype"], it["meta"]["placement"]) for it in its)
        if any(vp[(vt, p)] != PLACE_N[p] for vt in A.TRAIN_VTYPES for p in PLACE_N):
            fail.append(f"G4 {cell} value type x placement {dict(vp)}")
        if Counter(it["q_form"] for it in its) != Counter({"full": 8, "head": 8}):
            fail.append(f"G4 {cell} question form")
        for key, n in {"AL1": [("kA", 8), ("both", 8)], "AL4": [("both", 8), ("h", 6)], "AL2": [("h", 6)],
                       "AL3": [("h", 2)]}[cell]:
            c = Counter(it["meta"].get(key) for it in its if key in it["meta"])
            if sorted(c.values()) != [n, n]:
                fail.append(f"G4 {cell} factor {key} {dict(c)}")
        kinds = {gold_kind(it) for it in its}
        if not kinds <= EXPECT_KIND[cell]:
            fail.append(f"G4 {cell} gold kinds {kinds}")
    for it in items:
        fail += [f"G4 item {it['idx']}: {m}" for m in item_problems(it)]
    return fail


PLACE_N = {"adjacent": 2, "filler": 1, "other_obj": 1}


def item_problems(it):
    p = []
    objs = it["objects"]
    ow = {w for ph, h in objs for w in words(ph) + [h]}
    st = it["stmts"]
    for i, s in enumerate(st):
        u = it["turns"][s["turn"]][0]
        if s["ref"] == "alias":
            a = it["aliases"][s["obj"]]
            if not a or a not in u or set(words(u)) & ow:
                p.append(f"alias correction {u!r} lacks its alias or names an object")
            orig = [x for x in st[:i] if x["obj"] == s["obj"] and x["role"] == "orig"]
            if not orig or a not in it["turns"][orig[0]["turn"]][0]:
                p.append(f"alias {a} used before its original defines it")
            others = [x for x in it["aliases"] if x and x != a]
            if any(o.split()[1] in u for o in others):
                p.append("an alias correction holds the other alias")
    names = [x for x in it["aliases_all"]]
    if len({n.split()[1] for n in names}) != len(names):
        p.append("two aliases share a surname")
    ln = [s for s in st if NAME_RE.search(it["turns"][s["turn"]][0])][-1]
    if (ln["value"] == it["gold"]) != (it["cell"] == "AL3"):
        p.append(f"latest name statement is gold={ln['value'] == it['gold']} in {it['cell']}")
    for text in (it["question"], it["prefix"]):
        if values_in(text, ALL_VALUES) or re.search(r"\d", text) or it["gold"].lower() in words(text):
            p.append(f"value in question/prefix: {text!r}")
    vals = [s["value"] for s in st]
    if len(set(vals)) != len(vals) or len(it["candidates"]) < 2 or it["gold"] not in it["candidates"]:
        p.append("values not distinct or too few candidates")
    return p


def names_check(items):
    import items_e004 as I
    fail = []
    s = set(A.AL_SURNAMES)
    if len(s) != len(A.AL_SURNAMES):
        fail.append("G6 duplicate AL surname")
    fail += [f"G6 {x} in the E005 training pool" for x in s & set(PA.SURNAMES)]
    fail += [f"G6 {x} in the E004 eval surnames" for x in s & set(H.ALIAS_SURNAMES)]
    used = {a.split()[1] for it in items for a in it["aliases_all"]}
    fail += [f"G6 {x} used but not an AL surname" for x in used - s]
    vals = ALL_VALUES + [v for P in TRAIN_POOLS.values() for v in P["values"]]
    fail += [f"G6 {x} is a value" for x in s if values_in(x, vals)]
    eval_obj = {w for it in items for ph, h in it["objects"] for w in words(ph) + [h]}
    fail += [f"G6 {x} is an object word" for x in s if x.lower() in eval_obj]
    texts = "\n".join(I.prompt(it, with_prefix=True) for name in I.DRAWS for its in I.draw(name).values() for it in its)
    fail += [f"G6 {x} occurs in an E004 draw" for x in s if re.search(r"(?<![A-Za-z])" + x + r"(?![A-Za-z])", texts)]
    print(f"names: {len(s)} AL surnames, {len(used)} used; titles "
          f"{dict(Counter(a.split()[0] for it in items for a in it['aliases_all']))}; aliases per item "
          f"{dict(Counter(len(it['aliases_all']) for it in items))}")
    return fail


def run_all(items):
    import purity_al as PU
    tab = rule_table(items)
    print(f"AL items: {len(items)} (cells {dict(Counter(it['cell'] for it in items))}, value types "
          f"{dict(Counter(it['vtype'] for it in items))}, placements "
          f"{dict(Counter(it['meta']['placement'] for it in items))})\n")
    print_table(tab)
    ch = sum(1 / len(it["candidates"]) for it in items) / len(items)
    print(f"\nchance (1 / in-context candidates): {ch:.3f}; candidates per item "
          f"{dict(sorted(Counter(len(it['candidates']) for it in items).items()))}")
    fails = rule_gates(tab)
    s = structure(items)
    n = names_check(items)
    pu = PU.purity(items)
    for label, f in (("rules G1-G3", fails), ("structure G4/G5", s), ("names G6", n), ("purity", pu)):
        print(f"{'PASS' if not f else 'FAIL'}  {label}: {len(f)} problems {f[:6]}")
    return fails + s + n + pu
