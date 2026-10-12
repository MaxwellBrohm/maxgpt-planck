"""Auditor's own purity / leakage check on the rebuilt KEPT examples (kept_s*.pkl), seeds 1-5.
Eval side: E004 eval/dev/probe draws rebuilt from items_e004.draw (data only), eval pools and held-out lists.
All matching code below is mine; only vocabulary lists and the draws come from the project code."""
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

sys.dont_write_bytecode = True
sys.path.insert(0, "REPO/experiments/E005_alias_eot/code")
import items_e004 as I  # noqa: E402
import pools_eval as PE  # noqa: E402
import heldout_e004 as H  # noqa: E402
import pools_alias_train as PA  # noqa: E402
from pools_train import POOLS  # noqa: E402

TITLES_ALL = ["Mr.", "Mrs.", "Ms.", "Dr.", "Pastor", "Professor", "Chef", "Captain"]
WORD = re.compile(r"<a>|<v>|[a-z0-9]+(?:'[a-z]+)?")


def toks(s):
    return WORD.findall(s.lower())


def ngrams(ws, n):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


# ---------------- eval side ----------------
draws = {name: I.draw(name) for name in ("eval", "dev", "probe")}
items = [it for d in draws.values() for fam in d.values() for it in fam]
draw_aliases = Counter(it["alias"] for it in items if it.get("alias"))
EVAL_SURNAMES = set(H.ALIAS_SURNAMES) | {a.split(" ", 1)[1] for a in draw_aliases}
EVAL_PAIRS = set(PE.ALIASES) | set(draw_aliases)
EVAL_JOINS = sorted({P["alias_join"] for P in PE.POOLS_E.values()})
EVAL_TPLS = [t for P in PE.POOLS_E.values() for t in P["alias_corr"]]


def tpl_words(t):
    return toks(t.replace("{a}", " <a> ").replace("{v}", " <v> "))


EVAL_TPL_G3 = set().union(*(ngrams(tpl_words(t), 3) for t in EVAL_TPLS))
eval_texts = set()
for it in items:
    for u, a in it["turns"]:
        eval_texts |= {u, a}
    eval_texts.add(it["question"])
    eval_texts.add((it.get("prefix") or "") + " " + it["gold"])
EVAL_G5 = set().union(*(ngrams(toks(t), 5) for t in eval_texts))
print(f"eval side: {len(items)} items (eval {sum(map(len, draws['eval'].values()))}, dev "
      f"{sum(map(len, draws['dev'].values()))}, probe {sum(map(len, draws['probe'].values()))}); "
      f"{len(draw_aliases)} distinct aliases used in the draws; surnames {len(EVAL_SURNAMES)}; pairs {len(EVAL_PAIRS)}; "
      f"joins {EVAL_JOINS}; alias templates {len(EVAL_TPLS)} ({len(EVAL_TPL_G3)} word 3-grams); eval texts "
      f"{len(eval_texts)} ({len(EVAL_G5)} word 5-grams)")
print(f"  draw aliases not in the 16-name list: {sorted(set(draw_aliases) - set(PE.ALIASES))}")

# ---------------- pool level ----------------
tr_tpl_g3 = {}
for vt, ts in PA.ALIAS_CORR.items():
    for t in ts:
        tr_tpl_g3[t] = ngrams(tpl_words(t), 3) & EVAL_TPL_G3
for vt, js in PA.JOINS.items():
    for j in js:
        tr_tpl_g3["JOIN " + j] = ngrams(tpl_words(j), 3) & EVAL_TPL_G3
pool_hits = {t: g for t, g in tr_tpl_g3.items() if g}
join_words = {w for js in PA.JOINS.values() for j in js for w in toks(j)}
print(f"pool level: training alias templates + joins sharing a word 3-gram with an eval alias template: "
      f"{len(pool_hits)} {pool_hits}")
print(f"  training join words that are eval join words (with/from/run/by): "
      f"{sorted(join_words & {'with', 'from', 'run', 'by'})}")
print(f"  training surnames in eval surnames: {sorted(set(PA.SURNAMES) & EVAL_SURNAMES)}; roles in any eval text: "
      f"{sorted(r for r in PA.ROLES if any(re.search(r'(?<![A-Za-z])' + r + r'(?![A-Za-z])', t) for t in eval_texts))}")

# ---------------- training side ----------------
SUR_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(sorted(EVAL_SURNAMES)) + r")(?![A-Za-z])", re.I)
PAIR_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(p) for p in sorted(EVAL_PAIRS)) + r")(?![A-Za-z])")
ANYPAIR_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(t) for t in TITLES_ALL) + r")\s+("
                        + "|".join(sorted(EVAL_SURNAMES)) + r")(?![A-Za-z])", re.I)
JOIN_RE = re.compile(r"(?<![A-Za-z])(with|from|run by)\s+(" + "|".join(re.escape(t) for t in TITLES_ALL) + r")",
                     re.I)
TITLE_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(t) for t in TITLES_ALL) + r")(?![A-Za-z])")


def ex_texts(ex):
    out = []
    for u, a in ex["turns"]:
        out += [u, a]
    return out + [ex["question"], ex["answer"]]


def norm_alias_stmt(text, ex):
    t = text
    for a in sorted(ex.get("aliases", {}).values(), key=len, reverse=True):
        t = t.replace(a, " <a> ")
    for v in POOLS[ex["vtype"]]["values"]:
        flags = 0 if v[:1].isupper() else re.I
        t = re.sub(r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])", " <v> ", t, flags=flags)
    return tpl_words(t)


tot = Counter()
per_seed = defaultdict(Counter)
g5_hits, g3_hits, title_outside, extra = Counter(), Counter(), Counter(), Counter()
n_alias_stmts = 0
for seed in range(1, 6):
    D = pickle.load(open(f"kept_s{seed}.pkl", "rb"))
    if seed == 1 and os.environ.get("MUTATE"):
        import copy
        al = [copy.deepcopy(x) for x in D["kept"] if x["block"] == "alias" and x["vtype"] == "weekday"
              and any(s["ref"] == "alias" for s in x["stmts"])][:8]
        def setu(ex, fn):
            s = [s for s in ex["stmts"] if s["ref"] == "alias"][0]
            u, a = ex["turns"][s["turn"]]
            ex["turns"][s["turn"]] = (fn(u, ex), a)
        A = lambda ex: list(ex["aliases"].values())[0]
        setu(al[0], lambda u, ex: u.replace(A(ex).split(" ", 1)[1], "Petrov"))                    # eval surname only
        setu(al[1], lambda u, ex: u.replace(A(ex), "Ms. Petrov"))                                 # eval pair
        setu(al[2], lambda u, ex: u.replace(A(ex), "Captain Novak"))                               # role + eval surname
        o = [s for s in al[3]["stmts"] if s["role"] == "orig" and s["obj"] in al[3]["aliases"]][0]
        u, a = al[3]["turns"][o["turn"]]
        al[3]["turns"][o["turn"]] = (u + " It is with Dr. " + A(al[3]).split(" ", 1)[1] + ".", a)  # eval join
        setu(al[4], lambda u, ex: A(ex) + " bumped me to " + ex["stmts"][[s["ref"] for s in ex["stmts"]].index("alias")]["value"] + ".")  # 3-gram
        al[5]["turns"][0] = ("Mark down that the notary signing got slotted into the schedule on Friday.", al[5]["turns"][0][1])  # 5-gram
        al[6]["turns"][0] = ("Mr. Smith said hello.", al[6]["turns"][0][1])                       # undeclared title
        al[7]["question"] = al[7]["question"] + " 3"                                               # digit
        D["kept"] = al + D["kept"][8:]
    for ex in D["kept"]:
        texts = ex_texts(ex)
        joined = "\n".join(texts)
        flags = set()
        if SUR_RE.search(joined):
            flags.add("eval surname")
        if PAIR_RE.search(joined):
            flags.add("eval pair (16 names)")
        if ANYPAIR_RE.search(joined):
            flags.add("any title + eval surname")
        if JOIN_RE.search(joined):
            flags.add("eval join used as a join")
        # alias template 3-grams: every alias-correction statement
        for s in ex["stmts"]:
            if s.get("ref") == "alias":
                n_alias_stmts += 1
                g = ngrams(norm_alias_stmt(ex["turns"][s["turn"]][0], ex), 3) & EVAL_TPL_G3
                if g:
                    flags.add("alias correction 3-gram with eval alias template")
                    g3_hits.update(g)
        # stricter: any text with the alias placeholder sharing a 3-gram holding <a>
        for t in texts:
            g = {x for x in ngrams(norm_alias_stmt(t, ex), 3) & EVAL_TPL_G3 if "<a>" in x}
            if g:
                flags.add("any text: <a>-3-gram with eval alias template")
                extra.update(g)
        # 5-grams within one text vs eval texts (raw words)
        for t in texts:
            g = ngrams(toks(t), 5) & EVAL_G5
            if g:
                flags.add("raw word 5-gram shared with eval text")
                g5_hits.update(g)
        # titles outside the alias block, or not part of the declared alias
        declared = set(ex.get("aliases", {}).values()) if ex["block"] == "alias" else set()
        for t in texts:
            for m in TITLE_RE.finditer(t):
                nxt = t[m.start():].split(" ")
                al = " ".join(nxt[:2]).rstrip(".,!?;:'")
                if al not in declared:
                    flags.add("title outside a declared training alias")
                    title_outside[al] += 1
        # structural E004 lines, recomputed from the annotation
        objs = {s["obj"] for s in ex["stmts"] if s["obj"] is not None}
        if len(objs) > 2:
            flags.add("> 2 objects")
        if ex["d"] > 10:
            flags.add("d > 10")
        kmax = max([sum(1 for s in ex["stmts"] if s["obj"] == o and s["role"] != "orig") for o in objs] or [0])
        if kmax > 3:
            flags.add("k > 3")
        if re.search(r"[0-9]", joined):
            flags.add("digit")
        for f in flags:
            tot[f] += 1
            per_seed[seed][f] += 1
    per_seed[seed]["n"] = len(D["kept"])

print(f"\ntraining side: {sum(per_seed[s]['n'] for s in per_seed)} kept examples (6,404 x 5), "
      f"{n_alias_stmts} alias-correction statements")
axes = ["eval surname", "eval pair (16 names)", "any title + eval surname", "eval join used as a join",
        "alias correction 3-gram with eval alias template", "any text: <a>-3-gram with eval alias template",
        "raw word 5-gram shared with eval text", "title outside a declared training alias", "> 2 objects", "d > 10",
        "k > 3", "digit"]
for a in axes:
    print(f"  {a:52s} {tot.get(a, 0):6d}   per seed " + " ".join(str(per_seed[s].get(a, 0)) for s in range(1, 6)))
print(f"  3-gram hits: {g3_hits.most_common(5)} | <a> hits: {extra.most_common(5)}")
print(f"  5-gram hits: {g5_hits.most_common(10)}")
print(f"  titles outside declared alias: {title_outside.most_common(10)}")
