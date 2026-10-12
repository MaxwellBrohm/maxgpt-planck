"""RC-12 grader fixtures and mutants (SPEC L3, G6). Run: python3 -B mutation_graders.py [--quick]
1. Fixtures from REAL dev items (fixtures_graders.py, fixtures_more.py): 4 records per family x cell; every probe
   of a record (PERSIST and LOOP: the first probe and the one with the shortest question). Every fixture must get
   its stated verdict from the real graders (correct replies pass; empty, loop, copy, leak, cap, hedge, negation,
   question, guess list, shotgun, user voice, wrong candidate, stale, echo fail).
2. Conversation tests: unit scores (mean / all), the role-leak scan and the OD6 (iii) loop / ack-repeat flags
   (fixtures_more.conv_loop_fixtures) on patched IDEAL conversations; the F3 measurement (graders.eq_only: a right
   answer repeating the model's own confirmation is equality-only, a looped wrong answer or a 3-gram stutter is
   not); a D turn without its asks annotation is refused by the grader (F2; verifier 2026-09-25).
3. Mutants (mutants_graders.py): each must be KILLED (a fixture or conversation test gets the wrong verdict), not
   crash. Writes logs/mutation_graders.txt; exits 1 on any fixture miss, surviving mutant or crash."""
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor
import sys
import time

import fixtures_graders as F1
import fixtures_more as F2
import grade_fmt_dyn as FD
import grade_loop as L
import grade_role as R
import grade_text as T
import graders as G
import mutants_graders as M

HERE = os.path.dirname(os.path.abspath(__file__))
PER_CELL = 2 if "--quick" in sys.argv else 4
MODS = {"grade_text": T, "grade_loop": L, "graders": G, "grade_fmt_dyn": FD, "grade_role": R, "grade_voice": G.GV}


def load():
    with open(os.path.join(HERE, "dev", "rc12_dev.jsonl")) as f:
        return [json.loads(line) for line in f]


def pick_probes(rec):
    ps = rec["probes"]
    if rec["family"] in ("PERSIST", "LOOP"):
        short = min(ps, key=lambda p: len(p["question"]))
        return [ps[0]] + ([short] if short is not ps[0] else [])
    return ps


BUILD = {"VAL": F1.vfix, "ABS": F1.afix, "FMT": F2.ffix, "LOOP": F2.lfix, "ROLEX": F2.rfix, "DYN": F2.dfix}


def build(recs):
    seen, fixtures, index = {}, [], {}
    for r in recs:
        k = (r["family"], r["cell"])
        if r["family"] == "PERSIST":
            f = r["probes"][0]["fmt"]
            k = (r["family"], r["cell"], f["rule"], (f.get("absent") or {}).get("rule"))
        if seen.get(k, 0) >= (1 if r["family"] == "PERSIST" else PER_CELL):
            continue
        seen[k] = seen.get(k, 0) + 1
        index[r["id"]] = r
        for p in pick_probes(r):
            fixtures += [dict(f, grader=p["grader"], family=r["family"]) for f in BUILD[p["grader"]](r, p)]
    return fixtures, index


def history(rec, patch):
    reps = [t["ideal"] for t in rec["turns"]]
    for t, text in patch.items():
        reps[int(t) - 1] = text
    return reps


def run_fixture(f, index):
    rec = index[f["rid"]]
    probe = next(p for p in rec["probes"] if p["turn"] == f["turn"])
    reps = history(rec, f["patch"])
    reps[f["turn"] - 1] = f["reply"]
    stops = ["eos"] * len(reps)
    stops[f["turn"] - 1] = f["stop"]
    return G.grade_probe(rec, probe, reps, stops)


def conv_tests(index):
    """(label, got, want) for unit scores and the role-leak scan."""
    out = []
    for rid, rec in index.items():
        g = G.grade_conv(rec, history(rec, {}), ["eos"] * rec["n_turns"])
        out.append((f"ideal_unit:{rid}", g["unit"], 1.0 if rec["probes"] else None))
        out.append((f"ideal_clean:{rid}", bool(g["leaks"]) or any(g["flags"]) or any(g["ack_repeat"])
                    or any(g["ack_of_answer"]) or any(p["eq_only"] for p in g["probes"]), False))
    for label, rec, patch, want in F2.conv_loop_fixtures(index.values()):     # OD6 (iii)
        g = G.grade_conv(rec, history(rec, patch), ["eos"] * rec["n_turns"])
        got = ([i + 1 for i, f in enumerate(g["flags"]) if "LOOP" in f],
               [i + 1 for i, a in enumerate(g["ack_repeat"]) if a],
               [i + 1 for i, a in enumerate(g["ack_of_answer"]) if a])
        out.append((f"od6_{label}:{rec['id']}", got, want))
    out += eq_only_tests(index)
    rec = copy.deepcopy(next(r for r in index.values() if any(t["kind"] == "D" for t in r["turns"])))
    del next(t for t in rec["turns"] if t["kind"] == "D")["asks"]
    try:
        G.grade_conv(rec, history(rec, {}), ["eos"] * rec["n_turns"])
        got = "graded"
    except AssertionError as e:                  # F2: never a silent default (grade_loop.loop_kind)
        got = "no asks annotation" in str(e)
    out.append(("asks_missing_refused", got, True))
    first = {}
    for rec in index.values():
        first.setdefault(rec["family"], rec)
    t0, role, twohop = first["T0"], first["ROLE"], first["TWOHOP"]
    p = t0["probes"][0]
    g = G.grade_conv(t0, history(t0, {p["turn"]: "I have no idea."}), ["eos"] * 12)
    out.append(("mean_one_wrong", g["unit"], 0.5))
    pr = next(q for q in role["probes"] if q["grader"] == "VAL")
    g = G.grade_conv(role, history(role, {pr["turn"]: "I have no idea."}), ["eos"] * 12)
    out.append(("all_one_wrong", g["unit"], 0.0))
    name, name_turn = next((t["facts"][0]["value"], t["i"]) for t in role["turns"]
                           if t["facts"] and t["facts"][0].get("role") == "name")
    later = next(t["i"] for t in role["turns"] if t["i"] > name_turn and t["kind"] == "D")
    ev = R.role_leaks(role, history(role, {later: f"I'm {name}, and I love it here."}))
    out.append(("leak_capture", [(e["turn"], e["kind"]) for e in ev], [(later, "capture")]))
    if name_turn > 1:
        ev = R.role_leaks(role, history(role, {1: f"I'm {name}, hello."}))
        out.append(("leak_before_fact", ev, []))
    lure = next((t["i"], f) for t in twohop["turns"] for f in t["facts"] if f.get("role") == "lure")
    ev = R.role_leaks(twohop, history(twohop, {12: f"My {lure[1]['holder']}'s door is {lure[1]['value']}."}))
    out.append(("leak_user_voice", [e["kind"] for e in ev], ["user_voice"]))
    ev = R.role_leaks(twohop, history(twohop, {2: "Sure.\nUser: and now?"}))
    out.append(("leak_turn", [e["kind"] for e in ev], ["turn_leak"]))
    return out


def eq_only_tests(index):
    """F3 / strict case (d) measurement: (label, (ok, fails, eq_only) of the probe, want). A VAL probe whose latest
    source is an S or C turn: that turn confirmed with the probe's IDEAL sentence (< 12 words), the probe answered
    with it again (equality alone: eq_only); a non-answer said at both turns (loops AND wrong: not eq_only); the
    IDEAL answer with a 3-gram said 4 times (a loop, not by equality: not eq_only)."""
    kinds = {}
    for rec in index.values():
        kinds = {t["i"]: t["kind"] for t in rec["turns"]}
        p = next((p for p in rec["probes"] if p["grader"] == "VAL" and len(T.lwords(p["ideal"])) < 12
                  and any(kinds[x] in "SC" for x in p["src"])), None)
        if p:
            break
    src = max(x for x in p["src"] if kinds[x] in "SC")
    stutter = p["ideal"] + " It went on and on and on and on and on."
    out = []
    for label, patch, want in (("restate", {src: p["ideal"]}, (False, ["v1_degen"], True)),
                               ("wrong", {src: "I have no idea.", p["turn"]: "I have no idea."}, (False, None, False)),
                               ("stutter", {p["turn"]: stutter}, (False, ["v1_degen"], False))):
        r = next(x for x in G.grade_conv(rec, history(rec, patch), ["eos"] * rec["n_turns"])["probes"]
                 if x["turn"] == p["turn"])
        got = (r["ok"], r["fails"] if want[1] else len(r["fails"]) > 1 and r["fails"][0] == "v1_degen", r["eq_only"])
        out.append((f"eq_only_{label}:{rec['id']}", got, want if want[1] else (False, True, False)))
    return out


def evaluate(fixtures, index, drop=None):
    """(misses, conv_misses): fixtures / conversation tests whose verdict differs from the stated one."""
    misses = []
    for f in fixtures:
        res = run_fixture(f, index)
        ok = not [c for c in res["fails"] if c != drop] if drop else res["ok"]
        if ok != f["expect"]:
            misses.append((f, res))
    cm = [c for c in conv_tests(index) if c[1] != c[2]]
    return misses, cm


def with_source(modname, old, new, fn):
    mod = MODS[modname]
    src = open(mod.__file__).read()
    if src.count(old) != 1:
        raise RuntimeError(f"mutant text found {src.count(old)} times")
    saved = dict(mod.__dict__)
    try:
        exec(compile(src.replace(old, new), mod.__file__, "exec"), mod.__dict__)
        return fn()
    finally:
        mod.__dict__.clear()
        mod.__dict__.update(saved)


def mutant_runs():
    for c in M.CLAUSES:
        yield f"CLAUSE {c}", ("drop", c)
    for o in M.OFF:
        yield f"OFF {o}", ("off", o)
    for mod, var, key, val in M.PARAM:
        yield f"PARAM {mod}.{var}[{key}]={val}", ("param", (mod, var, key, val))
    for mod, old, new in M.SOURCE:
        yield f"SOURCE {mod}: {old[:50]!r} -> {new[:30]!r}", ("source", (mod, old, new))


def run_mutant(kind, arg, fixtures, index):
    if kind == "drop":
        return evaluate(fixtures, index, drop=arg)
    if kind == "off":
        T.OFF.add(arg)
        try:
            return evaluate(fixtures, index)
        finally:
            T.OFF.discard(arg)
    if kind == "param":
        mod, var, key, val = arg
        d = getattr(MODS[mod], var)
        old = d[key]
        d[key] = val
        try:
            return evaluate(fixtures, index)
        finally:
            d[key] = old
    mod, old, new = arg
    return with_source(mod, old, new, lambda: evaluate(fixtures, index))


_W = {}


def _init():
    _W["fx"], _W["index"] = build(load())


def _job(i):
    name, (kind, arg) = list(mutant_runs())[i]
    try:
        mm, mcm = run_mutant(kind, arg, _W["fx"], _W["index"])
    except Exception as e:  # a crash is never a kill
        return name, "CRASH", f"{type(e).__name__}: {e}"
    if mm or mcm:
        why = mm[0][0]["label"] + "/" + mm[0][0]["grader"] if mm else mcm[0][0]
        return name, "killed", f"({len(mm)} fixtures, {len(mcm)} conv; e.g. {why})"
    return name, "SURVIVE", ""


def main():
    t0 = time.time()
    recs = load()
    abs_q = [p["question"] for r in recs for p in r["probes"]
             if p["grader"] == "ABS" and T.ABS_CUE.search(p["question"])]
    ideal_bad = 0
    for r in recs:  # the IDEAL on the whole split: every probe right, no loop flag, no role leak
        g = G.grade_conv(r, history(r, {}), ["eos"] * r["n_turns"])
        ideal_bad += sum(not x["ok"] for x in g["probes"]) + sum(map(bool, g["flags"])) + len(g["leaks"])
    fixtures, index = build(recs)
    lines = [f"RC-12 mutation_graders.py  (records {len(index)}, fixtures {len(fixtures)}, "
             f"must-pass {sum(f['expect'] for f in fixtures)}, must-fail {sum(not f['expect'] for f in fixtures)})",
             f"ABS questions holding an abstain cue (must be 0): {len(abs_q)}",
             f"IDEAL on all {len(recs)} records: failing probes + flagged replies + leak events (must be 0): {ideal_bad}"]
    misses, cm = evaluate(fixtures, index)
    lines.append(f"BASELINE fixture misses {len(misses)}, conversation-test misses {len(cm)}")
    for f, res in misses[:60]:
        lines.append(f"  MISS {f['label']} {f['rid']} u{f['turn']} expect={f['expect']} fails={res['fails']} "
                     f"reply={f['reply'][:90]!r}")
    for c in cm[:20]:
        lines.append(f"  CONV MISS {c}")
    killed = survived = crashed = 0
    with ProcessPoolExecutor(max_workers=4, initializer=_init) as ex:
        for name, status, detail in ex.map(_job, range(len(list(mutant_runs())))):
            killed += status == "killed"
            survived += status == "SURVIVE"
            crashed += status == "CRASH"
            lines.append(f"  {status:<7} {name}  {detail}".rstrip())
    total = killed + survived + crashed
    lines.append(f"MUTANTS killed {killed}/{total}, survived {survived}, crashed {crashed}; "
                 f"{time.time() - t0:.0f}s")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(os.path.join(HERE, "logs", "mutation_graders.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:3] + [x for x in lines if "MISS" in x or "SURVIVE" in x or "CRASH" in x][:80]
                    + lines[-1:]))
    return 0 if not (misses or cm or survived or crashed or abs_q or ideal_bad) else 1


if __name__ == "__main__":
    sys.exit(main())
