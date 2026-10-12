"""RC-12 independent re-derivation of recorded dev numbers (rc12/notes.txt STEP 14N): written from the prereg text
(s4 likelihood rows, s5 units, s9 families, R and S7), never from score.py, lik_score.py or any other rc12 module
(it imports none), so a scorer bug and this file would have to agree by accident.
  dev(root, model, render, seeds)  one model and render of a dev_batch.py run tree (<root>/<model>/<render>/<seed>
      and <seed>_owncf, transcripts.jsonl) from the STORED grades: per seed the 9 composite families, LOOKUP, T0, the
      ungated OWN, R, S7 and the loop rate; "mean3" the plain mean over seeds 1-3 (rates pooled over replies).
      Units: the stored conversation unit; BIND one unit per pair_id, 1 iff both twins' units are 1.0; TWOHOP leaves
      the COMPOSE cell out; OWN (gated, OD1 b) 1 iff the own-history unit and its _owncf twin (same id, responder
      and render, cf_unswapped false) are both 1.0. R = 100 x mean of 9 families, S7 = 100 x mean of 7.
  lik(rows)  one render of lik_run rows.jsonl: per key the foils recomputed from ntok (candidates other than the
      gold with the gold's token count), right iff the gold's score is strictly above every foil's and finite,
      n_matched, acc_matched, chance_matched (mean 1 / (1 + foils)), n_equal / acc_equal (every candidate one count,
      judged on the full set), tier0 mean and p10 (nearest rank, ceil(0.1 n)-th smallest; a non-finite margin as
      -inf), BIND pairs (both twins matched and right), and foil_mismatch (rows whose stored foils differ).
  blank_after_role_cut(root, model, render)  the share of replies left blank by the plain render's role cut.
Exact equality with the recorded numbers is what test_numbers14.py asks (rows that the tree's graders would regrade
differently are told apart there)."""
import json
import math
import os
import re

NINE = ["RECALL", "CORR", "BIND", "TWOHOP", "PERSIST", "OWN", "TOPIC", "ROLE", "LOOP"]
SEVEN = ["RECALL", "CORR", "BIND", "TWOHOP", "OWN", "TOPIC", "ROLE"]
EXTRA = ["LOOKUP", "T0", "OWN_ungated"]


def read(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def avg(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def one_run(dev, cf):
    u = {k: [] for k in NINE + EXTRA}
    twins = {r["id"]: r for r in cf if r["family"] == "OWN"}
    pairs = {}
    for r in dev:
        f = r["family"]
        if f == "BIND":
            pairs.setdefault(r["pair_id"], []).append(r["unit"])
        elif f == "TWOHOP" and r["cell"] == "COMPOSE":
            continue
        elif f == "OWN":
            t = twins.get(r["id"])
            if t is None:
                raise ValueError(f"no own-cf twin for {r['id']}")
            same = t["responder"] == r["responder"] and t["render"] == r["render"]
            u["OWN"].append(1.0 if same and not t.get("cf_unswapped") and t["unit"] == 1.0 and r["unit"] == 1.0
                            else 0.0)
            u["OWN_ungated"].append(r["unit"])
        elif f in u:
            u[f].append(r["unit"])
    for pid, us in pairs.items():
        if len(us) != 2:
            raise ValueError(f"BIND pair {pid}: {len(us)} twins")
        u["BIND"].append(1.0 if us[0] == 1.0 and us[1] == 1.0 else 0.0)
    fam = {k: avg(v) for k, v in u.items()}
    flags = [t for r in dev for t in r["flags"]]
    fam["R"] = 100 * avg(fam[k] for k in NINE)
    fam["S7"] = 100 * avg(fam[k] for k in SEVEN)
    return fam, sum("LOOP" in t for t in flags), len(flags)


def dev(root, model, render, seeds=("greedy", "1", "2", "3")):
    d = os.path.join(root, model, render)
    out, loops, replies = {}, 0, 0
    for s in seeds:
        fam, nl, nr = one_run(read(os.path.join(d, s, "transcripts.jsonl")),
                              read(os.path.join(d, s + "_owncf", "transcripts.jsonl")))
        fam["loop"] = nl / nr
        out[s] = fam
        if s != "greedy":
            loops, replies = loops + nl, replies + nr
    m = {k: avg(out[s][k] for s in ("1", "2", "3")) for k in NINE + EXTRA}
    m["R"] = 100 * avg(m[k] for k in NINE)
    m["S7"] = 100 * avg(m[k] for k in SEVEN)
    m["loop"] = loops / replies
    return dict(seeds=out, mean3=m)


def blank_after_role_cut(root, model, render, seeds=("1", "2", "3")):
    """share of all own-history replies (seeds pooled) that are blank and whose raw text starts with a role tag, i.e.
    the model opened a new turn at once and the plain render's role cut left nothing (prereg s12, minimind-3)."""
    n = k = 0
    for s in seeds:
        for r in read(os.path.join(root, model, render, s, "transcripts.jsonl")):
            for t in r["turns"]:
                n += 1
                k += t["reply"].strip() == "" and bool(re.match(r"\s*(user|assistant|human)\s*:", t.get("raw") or "",
                                                                re.I))
    return k / n


def judge(scores, gold, others):
    g = scores[gold]
    if not others:
        return None, None
    best = max(scores[c] for c in others)
    finite = all(math.isfinite(scores[c]) for c in [gold] + others)
    return (finite and g > best), (g - best if finite else None)


def lik(rows):
    by = {}
    for r in rows:
        by.setdefault(r["key"], []).append(r)
    out = {}
    for key, rs in by.items():
        ok = [r for r in rs if not r["over_ctx"]]
        res, mism = [], 0
        for r in ok:
            n = r["ntok"]
            foils = [c for c in r["candidates"] if c != r["gold"] and n[c] == n[r["gold"]]]
            mism += sorted(foils) != sorted(r["foils"])
            right_m, margin_m = judge(r["scores"], r["gold"], foils)
            right_all, _ = judge(r["scores"], r["gold"], [c for c in r["candidates"] if c != r["gold"]])
            res.append(dict(pair=r["pair_id"], foils=foils, right_m=right_m, margin_m=margin_m, right=right_all,
                            equal=len({n[c] for c in r["candidates"]}) == 1))
        mt = [x for x in res if x["foils"]]
        eq = [x for x in res if x["equal"]]
        ms = sorted(-math.inf if x["margin_m"] is None else x["margin_m"] for x in mt)
        s = dict(n_matched=len(mt), acc_matched=avg(float(x["right_m"]) for x in mt),
                 chance_matched=avg(1 / (1 + len(x["foils"])) for x in mt), n_equal=len(eq),
                 acc_equal=avg(float(x["right"]) for x in eq), foil_mismatch=mism,
                 tier0_mean=avg(ms), tier0_p10=ms[math.ceil(0.1 * len(ms)) - 1] if ms else None)
        if key == "BIND":
            tw = {}
            for x in mt:
                tw.setdefault(x["pair"], []).append(x["right_m"])
            full = [v for v in tw.values() if len(v) == 2]
            s["pairs_n"], s["pairs_acc"] = len(full), avg(float(all(v)) for v in full)
        out[key] = s
    return out
