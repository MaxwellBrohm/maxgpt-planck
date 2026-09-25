"""E005 AL tables (descriptive; no rule reads AL). No model, no torch. Reads ../al/al_items.jsonl (sha256 checked) and
every scored model in ../al/out (<tag>__run.json with its LIK and GEN files); writes ../al/tables_al.txt and
../al/al_results.json.
Per model and render: LIK (right = gold beats every other in-context value) and GEN strict (hardened grader), over
all 64, per cell, AL1+AL2+AL4 (the ADDITION's pre-registered reading: "alias recency" scores 0 there), per
placement, on the "needs linking" items (14: the adjacency tracker T AND "latest statement with a name" LN are both
wrong there), and the share of LIK picks that equal LN's value or T's value on the items where they are wrong.
The ADDITION's sentence: if the E005 alias reading is DATA GAP but AL1+AL2+AL4 LIK < 0.5 (plain) on a majority of
E005 seeds, the report says "resolved by alias recency, not by linking"; the check is printed with its inputs.
usage: python3 -B analyze_al.py"""
import glob
import json
import os
import sys

import items_al as A
import checks_al as CA

OUT = os.path.join(A.AL_DIR, "out")
TABLES = os.path.join(A.AL_DIR, "tables_al.txt")
RESULTS = os.path.join(A.AL_DIR, "al_results.json")
E005_RESULTS = os.path.join(os.path.dirname(A.HERE), "results.json")
RENDERS = ("plain", "chat")
ORDER = ["base"] + [f"e005_s{s}" for s in range(1, 6)] + [f"e004_s{s}" for s in range(1, 6)]


def rule_values(items):
    """{idx: {"LN": value, "T": value}}"""
    fns = dict(CA.RULES)
    return {it["idx"]: {"LN": fns["LN latest statement with a name"](CA.ViewAL(it)),
                        "T": fns["T topic tracker (diagnostic)"](CA.ViewAL(it))} for it in items}


def groups(items, rv):
    g = {"all": [], "AL1+AL2+AL4": [], "needs linking": []}
    for it in items:
        i = it["idx"]
        for k in ("all", it["cell"], "pl:" + it["meta"]["placement"]):
            g.setdefault(k, []).append(i)
        if it["cell"] != "AL3":
            g["AL1+AL2+AL4"].append(i)
        if rv[i]["LN"] != it["gold"] and rv[i]["T"] != it["gold"]:
            g["needs linking"].append(i)
    return g


def read_jsonl(p):
    return [json.loads(ln) for ln in open(p)] if os.path.exists(p) else None


def lik_pick(r):
    best = max(r["scores"], key=lambda k: r["scores"][k])
    return r["cand_vals"][best]


def model_table(tag, items, rv, g):
    byidx = {it["idx"]: it for it in items}
    out = {}
    for render in RENDERS:
        L = read_jsonl(os.path.join(OUT, f"{tag}__al__{render}.jsonl"))
        Gn = read_jsonl(os.path.join(OUT, f"{tag}__gen_al__{render}.jsonl"))
        for kind, recs, key in (("LIK", L, "right"), ("GEN", Gn, "strict")):
            if recs is None or len(recs) != 64:
                out[f"{kind} {render}"] = None
                continue
            ok = {r["idx"]: bool(r[key]) for r in recs}
            out[f"{kind} {render}"] = {k: round(sum(ok[i] for i in ids) / len(ids), 4) for k, ids in g.items()}
        if L is not None and len(L) == 64:
            pick = {r["idx"]: lik_pick(r) for r in L}
            for rule in ("LN", "T"):
                wrong = [i for i in byidx if rv[i][rule] != byidx[i]["gold"]]
                out[f"LIK {render} picks {rule}'s value where {rule} is wrong"] = [
                    sum(pick[i] == rv[i][rule] for i in wrong), len(wrong)]
    return out


def main():
    items = A.load()
    rv = rule_values(items)
    g = groups(items, rv)
    tags = sorted({os.path.basename(p)[:-len("__run.json")] for p in glob.glob(os.path.join(OUT, "*__run.json"))} - {"dry"},
                  key=lambda t: (ORDER.index(t) if t in ORDER else 99, t))
    res = {"items_sha256": open(A.SHA_PATH).read().split()[0], "groups_n": {k: len(v) for k, v in g.items()},
           "models": {}}
    lines = [f"E005 AL diagnostic (al/al_items.jsonl, 64 items; groups n: "
             + ", ".join(f"{k} {len(v)}" for k, v in g.items()) + ")", ""]
    cols = ["all", "AL1", "AL2", "AL3", "AL4", "AL1+AL2+AL4", "pl:adjacent", "pl:filler", "pl:other_obj",
            "needs linking"]
    head = f"{'model':10s}{'measure':12s}" + "".join(f"{c.replace('pl:', '')[:11]:>12s}" for c in cols)
    lines += [head, "-" * len(head)]
    for tag in tags:
        run = json.load(open(os.path.join(OUT, f"{tag}__run.json")))
        t = model_table(tag, items, rv, g)
        t["fatal"], t["selftest_fail"], t["model"] = run.get("fatal"), run.get("selftest_fail"), run.get("model")
        t["items_sha256_ok"] = run.get("items_sha256") == open(A.SHA_PATH).read().split()[0]
        res["models"][tag] = t
        for m in ("LIK plain", "GEN plain", "LIK chat", "GEN chat"):
            row = t.get(m)
            cells = "".join(f"{row[c]:12.2f}" for c in cols) if row else "  (missing)"
            lines.append(f"{tag:10s}{m:12s}{cells}")
        for render in RENDERS:
            for rule in ("LN", "T"):
                k = f"LIK {render} picks {rule}'s value where {rule} is wrong"
                if k in t:
                    lines.append(f"{'':10s}{k}: {t[k][0]}/{t[k][1]}")
        if not t["items_sha256_ok"] or t["fatal"] or t["selftest_fail"]:
            lines.append(f"{'':10s}WARNING sha_ok={t['items_sha256_ok']} fatal={t['fatal']} selftest={t['selftest_fail']}")
    e5 = [t for t in tags if t.startswith("e005_s")]
    low = [t for t in e5 if (res["models"][t].get("LIK plain") or {}).get("AL1+AL2+AL4", 1) < 0.5]
    label = None
    if os.path.exists(E005_RESULTS):
        try:
            label = json.load(open(E005_RESULTS))["e005"]["reading"]["alias"]["label"]
        except Exception:
            label = None
    say = label == "DATA GAP" and len(low) >= 3
    res["addition"] = dict(e005_seeds_scored=e5, seeds_al124_lik_below_half=low, alias_reading=label,
                           resolved_by_alias_recency=say)
    lines += ["", f"ADDITION check: E005 alias reading {label!r}; E005 seeds with AL1+AL2+AL4 LIK plain < 0.5: "
              f"{len(low)} of 5 planned ({low}); scored {len(e5)}.",
              "  -> " + ("resolved by alias recency, not by linking" if say else
                         "the ADDITION's sentence does not apply (it needs DATA GAP and >= 3 of 5 seeds < 0.5)")]
    summ = "; ".join(f"{t} " + "/".join("-" if not res["models"][t].get(m) else f"{res['models'][t][m]['all']:.2f}"
                                        for m in ("LIK plain", "GEN plain")) for t in tags)
    lines += ["", f"SUMMARY (LIK/GEN plain, all 64): {summ or 'no model scored'}"]
    open(TABLES, "w").write("\n".join(lines) + "\n")
    json.dump(res, open(RESULTS, "w"), indent=1)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
