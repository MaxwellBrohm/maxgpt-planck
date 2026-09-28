"""SPEC 5 acceptance on a list of skill records: oracle rates, balance rules and purity, from tokens alone.

accept(recs, pools, split) -> report {"pass": bool, "fails": [...], "oracles", "balance", "purity"}. Blocks are read
from record ids ("<fam>.b<block>.i<n>"); the 1/2 rules must hold exactly in every full block of `block` items.
Eval B cube groups (meta grp 0 for the item, 1-3 for its copies, skill.group; K3 round 3) are scored jointly; the
copies count in the per-block shares, not in the per-block intro-rank rules (their ranks follow the item's order).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

import kcommon as K
import oracles as O


def _block(rec) -> str:
    return rec["id"].rsplit(".i", 1)[0]


def item_facts(rec: dict, lex) -> dict:
    st, qs, intro = O.parse(rec, lex)
    q = qs[-1]
    prompt = O.prompt_of(rec, q["turn"])
    ans = O.answers(st, q, prompt, lex)
    S = st[:q["n"]]
    key = [i for i, s in enumerate(S) if s["ent"] == q["name"] and s["attr"] == q["attr"]]
    last = key[-1] if key else -1
    after = [s for s in S[last + 1:] if s["attr"] == q["attr"] and s["ent"] != q["name"]] if last >= 0 else []
    pat, gidx = O.form_pattern(st, q)
    refs = [s for s in S if s["attr"] == q["attr"] and s["kind"] in O.REF_KINDS]
    kref = next((s["kind"] for s in refs if s["ent"] == q["name"]), None)
    all_ok = all(O.answers(st, qq, O.prompt_of(rec, qq["turn"]), lex)["IDEAL"] == qq["gold"] for qq in qs)
    early = [p for p in qs[:-1] if p["attr"] == q["attr"] and (p["name"] != q["name"] or not any(
        s["ent"] == q["name"] and s["attr"] == q["attr"] for s in st[p["n"]:q["n"]]))]      # K3 R-4 (skill._questions)
    hold = {s["ent"] for s in S if s["attr"] == q["attr"]} | {q["name"]}       # K3 round 4 (REVIEW 5a S-1): earlier
    early_name = [p for p in qs[:-1] if p["name"] in hold and not (p["name"] == q["name"] and p["attr"] == q["attr"]
                  and any(s["ent"] == q["name"] and s["attr"] == q["attr"] for s in st[p["n"]:q["n"]]))]   # bystanders
    free = [u for u in intro if u not in hold]                  # R and U reserve one bystander (never asked)
    all_asked = rec["fam"] in "RU" and bool(free) and all(u in {p["name"] for p in qs[:-1]} for u in free)
    tail = S[-1] if S and S[-1]["attr"] == q["attr"] and S[-1]["kind"] in O.REF_KINDS else None
    foils = {s["ent"] for s in S if s["attr"] == q["attr"] and s["ent"] != q["name"]}
    n_foil = sum(s["ent"] in foils for s in S if s["attr"] == q["attr"]) if len(foils) == 1 else -1
    acks = all(t["ids"] == [K.OK] for i, t in enumerate(rec["turns"])
               if t["role"] == "assistant" and rec["turns"][i - 1]["ids"][-1:] != [K.Q])
    als = [s for s in S if s["attr"] == q["attr"] and s["kind"] == "alias"]        # K3 round 3 (REVIEW 4a S-2)
    dx = {u: e for e, u in q["dex"]}
    ents = {s["ent"] for s in als}
    gap = {id(s): s["ex"] - dx[q["name"]] for s in als if q["name"] in dx and s["ex"] - dx[q["name"]] >= 1}
    near = [s for s in als if id(s) in gap and gap[id(s)] == min(gap.values())]
    return {"fam": rec["fam"], "block": _block(rec), "ans": ans,
            "gold": q["gold"], "all_ideal": all_ok, "acks": acks, "form": q["form"], "follow": bool(after),
            "follow_ref": any(s["kind"] in O.REF_KINDS for s in after), "pat": pat, "gidx": gidx,
            "pat_e": O.form_pattern(st, q, "ell")[0], "pat_a": O.form_pattern(st, q, "alias")[0],
            "key_alias": kref == "alias", "same_kind": all(s["kind"] == kref for s in refs),
            "key_ref_latest": last >= 0 and S[last]["kind"] in O.REF_KINDS,
            "follow_alias": any(s["kind"] == "alias" for s in after),
            "key_named": last >= 0 and S[last]["kind"] in ("full", "reord"),
            "n_key": len(key), "n_st": len(S), "is_last": bool(key) and last == len(S) - 1,
            "rank": intro.index(q["name"]) if q["name"] in intro else -1, "k": len(intro),
            "frank": intro.index(next(iter(foils))) if len(foils) == 1 and next(iter(foils)) in intro else -1,
            "has_attr": any(s["attr"] == q["attr"] for s in S), "has_name": q["name"] in intro,
            "early_attr": bool(early), "early_name": bool(early_name), "all_free_asked": all_asked,
            "ref_last": None if tail is None else tail["ent"] == q["name"],
            "n_q": len(qs), "u_cell": (len(key), n_foil, bool(after)), "grp": rec.get("meta", {}).get("grp", -1),
            "alias_win": rec["fam"] != "F" or all(1 <= s["ex"] - dx[u] <= 10 for s in als for u in ents if u in dx),
            "xnear": near[0]["ent"] == q["name"] if len(near) == 1 and len(als) == 2 and last >= 0 and
            S[last]["kind"] == "alias" else None}


def oracle_table(facts: list[dict], surf_fit: dict | None = None) -> dict:
    tab = defaultdict(dict)
    for fam in "RUBFAP":
        F = [f for f in facts if f["fam"] == fam]
        if not F:
            continue
        for o in O.NAMES:
            tab[fam][o] = round(float(np.mean([f["ans"][o] == f["gold"] for f in F])), 4)
        for v, (key, _) in O.VIEWS.items():
            tab[fam][v] = round(surf(F, (surf_fit or {}).get(fam, {}).get(v), key), 4)
        Fr = [f for f in F if f["key_ref_latest"]]
        if Fr:              # SURFr: SURF on the items whose gold is a reference statement (reported)
            tab[fam]["SURFr"] = round(surf(Fr, (surf_fit or {}).get(fam, {}).get("SURF"), "pat", F), 4)
        tab[fam]["n"] = len(F)
    return dict(tab)


def surf_counts(F: list[dict], key: str = "pat") -> dict:
    """Pattern -> {gold index: count}, JSON keys ("NAE" -> {"2": 5}): SURF's fit, kept in the report."""
    out = defaultdict(Counter)
    for f in F:
        out["".join(f[key])][str(f["gidx"])] += 1
    return {k: dict(v) for k, v in out.items()}


def _surf_hit(f: dict, fit: dict, key: str = "pat") -> float:
    c = fit.get("".join(f[key]), {})
    best = max(c.values(), default=0)
    tied = [int(i) for i, v in c.items() if v == best] if best > 0 else [len(f[key]) - 1]
    return (f["gidx"] in tied) / len(tied)


def surf(F: list[dict], fit: dict | None = None, key: str = "pat", within: list[dict] | None = None) -> float:
    """SURF (review R-1): the best rule that sees only the forms of the asked attribute's statements (O.form_pattern):
    each item gets the gold index most often right among fitted items with its pattern (ties share the credit; an
    unseen pattern: the latest statement, as O2). fit: counts from the training stream (surf_counts); None: a 2-fold
    cross-fit on F itself (even items fit the odd ones and back; leave-one-out is biased low on the exactly balanced
    patterns F is built from, an in-sample fit biased high). key: the view (O.VIEWS; SURFe and SURFa see one reference
    kind resolved). within: score only the items of F that are in `within`'s folds (SURFr), the folds drawn on
    `within` so the fit is the one SURF itself uses."""
    if fit is not None:
        return sum(_surf_hit(f, fit, key) for f in F) / max(1, len(F))
    W = within if within is not None else F
    a, b = W[0::2], W[1::2]
    fa, fb = surf_counts(a, key), surf_counts(b, key)
    ids = {id(f) for f in F}
    hits = [_surf_hit(f, fb, key) for f in a if id(f) in ids] + [_surf_hit(f, fa, key) for f in b if id(f) in ids]
    return sum(hits) / max(1, len(hits))


def groups_of(fs: list) -> list[list]:
    """B cube groups: an item with grp 0 and the next three with grp 1, 2, 3 (skill.group's order)."""
    return [fs[i:i + 4] for i in range(len(fs) - 3) if [f["grp"] for f in fs[i:i + 4]] == [0, 1, 2, 3]]


def group_table(facts: list[dict]) -> dict:
    """B cube groups (K3 round 3): share of groups an oracle gets right on all 4 questions."""
    G = groups_of([f for f in facts if f["fam"] == "B"])
    if not G:
        return {}
    return {o: round(float(np.mean([all(f["ans"][o] == f["gold"] for f in g) for g in G])), 4)
            for o in [*O.O3, *O.BCUE, *O.BPART, "O4", "IDEAL"]} | {"n": len(G)}


def balance(facts: list[dict], block: int) -> tuple[dict, list[str]]:
    rep, fails = {}, []
    for fam in "RUBF":
        F = [f for f in facts if f["fam"] == fam and f["n_key"]]
        if not F:
            continue
        got = float(np.mean([f["is_last"] for f in F]))
        exp = float(np.mean([f["n_key"] / f["n_st"] for f in F]))
        rep[f"{fam}.last"] = (round(got, 4), round(exp, 4))
        if abs(got - exp) > max(0.05, 3 * math.sqrt(exp * (1 - exp) / len(F))):     # BD-2: 3 SE on small sets
            fails.append(f"{fam}: last-statement rate {got:.3f} vs chance {exp:.3f}")
    for fam in "RUBFP":
        F = [f for f in facts if f["fam"] == fam and f["rank"] >= 0]
        for k in sorted({f["k"] for f in F}):
            Fk = [f for f in F if f["k"] == k]
            for rk in range(k):
                p = sum(f["rank"] == rk for f in Fk) / len(Fk)
                tol = 3 * math.sqrt((1 / k) * (1 - 1 / k) / len(Fk)) if k > 1 else 0
                if abs(p - 1 / k) > tol + 1e-9:
                    fails.append(f"{fam}: k {k} intro rank {rk} asked at {p:.3f} (1/k {1 / k:.3f}, n {len(Fk)})")
    blocks = defaultdict(list)
    for f in facts:
        blocks[f["block"]].append(f)
    for b, F in blocks.items():         # every block (training blocks hold `block` items; odd counts: floor or ceil)
        fam = F[0]["fam"]
        rules = {"form": ([f["form"] for f in F], 2)}
        if fam in "RUF":
            rules["follow"] = ([f["follow"] for f in F], 2)
        if fam == "F":          # R-1: a foil's reference statement follows in every follow item (1/2); a foil's
            rules["follow_ref"] = ([f["follow_ref"] for f in F], 2)          # alias follows in 1/4; the key's
            rules["follow_alias"] = ([f["follow_alias"] for f in F], 4)      # latest is a named statement in 1/4
            rules["key_named"] = ([f["key_named"] for f in F], 4)
            rules["key_alias"] = ([f["key_alias"] for f in F], 2)          # BD-14: the key's reference kind, 1/2,
            rules["same_kind"] = ([f["same_kind"] for f in F], 1)          # the foil's always the same kind
        if fam == "U":          # the no-update controls (key once, foil twice, the foil's latest after) exactly 1/4,
            rules["no_update"] = ([f["u_cell"] == (1, 2, True) for f in F], 4)      # and their mirror (K3 round 2,
            rules["mirror"] = ([f["u_cell"] == (2, 1, False) for f in F], 4)        # D-1): key twice, foil once
            for k in (2, 3, 4):                             # U's k and intro rank exact per block (BD-17)
                rules[f"k{k}"] = ([f["k"] == k for f in F], 3)
                Fk = [f for f in F if f["k"] == k]
                rules.update({f"k{k}_rank{rk}": ([f["rank"] == rk for f in Fk], k) for rk in range(k)})
                rules.update({f"k{k}_frank{rk}": ([f["frank"] == rk for f in Fk], k) for rk in range(k)})  # round 4
        if fam in "RFP":        # K3 round 4: R's, F's and P's intro rank exact per block within each k (per item,
            for k in sorted({f["k"] for f in F}):       # a K-EVAL-sized R block read k 3 rank 1 at 0.506)
                Fk = [f for f in F if f["k"] == k]
                rules.update({f"k{k}_rank{rk}": ([f["rank"] == rk for f in Fk], k) for rk in range(k)})
        if fam == "B":          # BD-22: 2-4 names in 1/4 (the cube's 8 otherwise), k and intro rank exact (the
            rules["b_small"] = ([f["k"] != 8 for f in F], 4)        # items, not their group copies: BD-26)
            small = [f for f in F if f["k"] != 8]
            rules.update({f"k{k}": ([f["k"] == k for f in small], 3) for k in (2, 3, 4)})
            for k in (2, 3, 4, 8):
                Fk = [f for f in F if f["k"] == k and f["grp"] <= 0]
                rules.update({f"k{k}_rank{rk}": ([f["rank"] == rk for f in Fk], k) for rk in range(k)})
        if fam == "A":
            rules["both_lures"] = ([f["has_attr"] and f["has_name"] for f in F], 2)
        for name, (v, m) in rules.items():      # U's k and rank cells: largest remainder (512 / 3 is no integer)
            exact = len(F) == block and not name.startswith("k")
            if abs(sum(v) * m - len(v)) >= m or (exact and sum(v) * m != len(v)):
                fails.append(f"{b}: {name} {sum(v)} of {len(v)}, not exactly 1/{m}")
    U_ = Counter(f["u_cell"] for f in facts if f["fam"] == "U")
    for (a, b, w), m in sorted(U_.items()):         # U: (key, foil, follow) as often as (foil, key, no follow)
        mm = U_[(b, a, not w)]
        if abs(m - mm) > 3 * math.sqrt(m + mm):
            fails.append(f"U: (key {a}, foil {b}, follow {w}) in {m} items, its mirror in {mm}")
    Fr = [f for f in facts if f["fam"] == "F" and f["key_ref_latest"]]
    if Fr:                  # K3 R-3: the key's and the foil's reference statement end the item equally often
        a, b = sum(f["ref_last"] is True for f in Fr), sum(f["ref_last"] is False for f in Fr)
        rep["F.ref_last"] = (round(a / len(Fr), 4), round(b / len(Fr), 4))
        if abs(a - b) > 3 * math.sqrt(max(1, a + b)):
            fails.append(f"F: the item ends with the key's reference statement in {a}, the foil's in {b}")
    Fx = [f["xnear"] for f in facts if f["fam"] == "F" and f["xnear"] is not None]
    if Fx:          # K3 round 3 (REVIEW 4a S-2, F-XNEAR): on alias items whose gold is the key's alias, the alias
        a, b = sum(Fx), len(Fx) - sum(Fx)   # nearer after the asked name's definition is the key's as often as the
        rep["F.xnear"] = (a, b)             # foil's (in the late quarter the key's alias is earlier by design)
        if abs(a - b) > 3 * math.sqrt(max(1, a + b)):
            fails.append(f"F: the alias nearer after the asked definition is the key's in {a}, the foil's in {b}")
    if any(not f["alias_win"] for f in facts):      # BD-25: both aliases 1-10 exchanges after both definitions
        fails.append(f"{sum(not f['alias_win'] for f in facts)} F items: an alias outside a definition's window")
    if any(f["early_attr"] for f in facts):     # K3 R-4: only U's stale lure may ask the final attribute earlier
        fails.append(f"{sum(f['early_attr'] for f in facts)} items: an earlier question asks the final attribute")
    if any(f["all_free_asked"] for f in facts):     # K3 round 4: R and U keep one bystander unasked, as A does
        fails.append(f"{sum(f['all_free_asked'] for f in facts)} R or U items: every bystander asked")
    if any(f["early_name"] for f in facts):     # K3 round 4 (REVIEW 5a S-1): earlier questions ask bystanders only
        fails.append(f"{sum(f['early_name'] for f in facts)} items: an earlier question names the final name or a "
                     "holder of the final attribute")
    if any(not f["acks"] for f in facts):
        fails.append(f"{sum(not f['acks'] for f in facts)} items: an ack carries a value")
    return rep, fails


def qcount_fails(facts: list[dict], p_range=(2, 4)) -> list[str]:
    """SPEC 4: 1-3 questions per item, P 2-4 (PLONG 5-7: p_range); B eval items carry only the final one (BD-7).
    K3 round 2 (REVIEW 3b T-1): skill.make_item's P redraw had no check."""
    bad = Counter(f["fam"] for f in facts if not (p_range if f["fam"] == "P" else (1, 3))[0] <= f["n_q"] <=
                  (p_range if f["fam"] == "P" else (1, 3))[1])
    return [f"{fam}: {n} items with a question count outside SPEC 4's range" for fam, n in sorted(bad.items())]


def group_count_fails(facts: list[dict], groups: dict) -> list[str]:
    """K3 round 4 (REVIEW 5b O-1): every B cube item (grp 0) must open a scored group (its 3 copies right after it);
    records out of skill.group's order scored no group, and every group bar was skipped silently."""
    n0 = sum(f["fam"] == "B" and f["grp"] == 0 for f in facts)
    got = groups.get("n", 0)
    return [f"B groups: {got} scored of {n0} cube items"] if n0 != got else []


def threshold_fails(tab: dict, groups: dict) -> list[str]:
    fails = []
    for fam, row in tab.items():
        if row["IDEAL"] != 1.0:
            fails.append(f"{fam}: IDEAL {row['IDEAL']}")
        for o, (fams, t) in O.TH.items():
            if fam in fams and not O.rate_ok(row[o], t, row["n"]):
                fails.append(f"{fam}: {o} {row[o]:.3f} > {t}")
    for o in [*O.O3, *O.BCUE, *O.BPART]:
        if o in groups and not O.rate_ok(groups[o], O.GROUP_TH, groups["n"]):
            fails.append(f"B groups: {o} {groups[o]:.3f} > {O.GROUP_TH}")
    return fails


def purity(recs: list[dict], pools, split: str, tok=None) -> tuple[dict, list[str]]:
    own = pools.owner()
    kevs = pools.kevs_set()
    bad = defaultdict(int)
    for r in recs:
        ids = [x for t in r["turns"] for x in t["ids"]]
        cls = {own.get(x, "") for x in ids}
        bad["fb_or_filler_name"] += bool(cls & {"F1", "F2", "F3", "Z1", "Z2", "Z3"})
        if split in ("eval", "evals"):
            bad["skill_train_token"] += split == "eval" and bool(cls & {"S1", "S2", "S3"})
        if split == "train":
            bad["heldout_token"] += bool(cls & {"VH", "AH"})
            bad["kevs_triple"] += any(tuple(ids[i:i + 3]) in kevs for i in range(len(ids) - 2))
        for t in r["turns"]:
            if t["role"] == "assistant" and t["ids"] != [K.OK]:
                bad["candidate_not_one_token"] += own.get(t["ids"][-1], "") not in ("V", "VH") and \
                    t["ids"][-1] != K.NONE
    if tok is not None:
        seqs = [t["ids"] for r in recs for t in r["turns"]]
        texts = tok.decode_batch(seqs, skip_special_tokens=False)
        enc = tok.encode_batch(texts, add_special_tokens=False)
        bad["round_trip"] = sum(e.ids != s for e, s in zip(enc, seqs))
    fails = [f"purity {k}: {v}" for k, v in bad.items() if v]
    return dict(bad), fails


def accept(recs: list[dict], pools, split: str, lex, tok=None, block: int = 512, rules: bool = True,
           surf_fit: dict | None = None, p_questions=(2, 4)) -> dict:
    """rules False (OOD columns: structures absent from training, reported only): oracles, purity and the question
    counts only (p_questions: P's range, PLONG (5, 7)). surf_fit: the training stream's report["surf_counts"], so
    SURF on an eval set is the rule training teaches."""
    facts, bad = [], []
    for r in recs:
        try:
            facts.append(item_facts(r, lex))
        except (O.ParseError, IndexError, KeyError) as e:
            bad.append(f"{r.get('id')}: {type(e).__name__} {e}"[:200])
    tab, groups = oracle_table(facts, surf_fit), group_table(facts)
    bal, bfails = balance(facts, block) if rules else ({}, [])
    pur, pfails = purity(recs, pools, split, tok)
    fails = [f"{len(bad)} items do not parse, first: {bad[0]}"] if bad else []
    if rules:
        fails += threshold_fails(tab, groups)
    else:
        fails += [f"{f}: IDEAL {row['IDEAL']}" for f, row in tab.items() if row["IDEAL"] != 1.0]
    fails += bfails + pfails + qcount_fails(facts, p_questions) + group_count_fails(facts, groups)
    fails += [f"{sum(not f['all_ideal'] for f in facts)} items: a question's gold is not the world rule"] * \
        any(not f["all_ideal"] for f in facts)
    return {"pass": not fails, "fails": fails, "oracles": tab, "groups": groups, "balance": bal, "purity": pur,
            "n": len(recs), "surf_counts": {fam: {v: surf_counts([f for f in facts if f["fam"] == fam], key)
                                                  for v, (key, _) in O.VIEWS.items()} for fam in tab}}
