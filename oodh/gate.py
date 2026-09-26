"""OOD-H Part 1 gate (oodh/DESIGN.txt s6, s12; prereg draft s2 L2 and s8 G1, G7, G8). No model. Exit 0 when every
GATING check passes, 1 otherwise. G2 and G4 are computed but NOT gating for OOD-H (DESIGN s12: on natural threads a
probe usually asks for the one value of its kind the user gave, so type-guessing cheaters pass structurally; OOD-H is
a paired Planck-vs-comparator check where that helps both sides). They are reported beside the result, with the full
cheater table and the score on the subset of probes where the user gave 2+ values of the asked kind (multi_value).
Each check below lists its failures by record id and probe turn (never item text).
  ids     every tree id is in the OOD-H reserve (corpus/oodh.in_oodh_reserve), not in the leak list, unique, and is the
          thread's first message id. Leak list: --leaked (default sealed/oodh/leaked_reserve_trees.txt), written on the
          PC by oodh/leaks.py (corpus/oodh.leaked_reserve_trees over every OASST2 tree and Dolly instruction).
  struct  probe turns n+1.. (text and ideal = the probe's question and ideal), kinds X..P, 1 to 3 probes, >= 1 VAL,
          <= 1 ABS, a bool asks on every D turn, n_turns.
  G1      IDEAL and TERSE (bare gold) score 1.00 on every thread through rc12 runner.run with the fixed history
          (plain render), with no loop flag and no role-leak event at a probe turn; the history comes back unchanged
          (reply == history_reply, stop eos); every accepted form alone ("<form>.") passes its probe.
  L2      rc12/test_gens.test_l2 (whole word, any case): no question or prefix holds an accepted form or a candidate
          of its probe, and no user turn after the gold's latest src turn (a probe question included) holds the gold;
          plus, for the probe block: no question holds an accepted form of a LATER probe of its thread.
  G8      every VAL gold occurs whole-word in each src turn, and every src turn is a thread turn.
  ABS     no pool value (any case) in any thread message, user or assistant, or in any question of the thread; the
          question holds no abstain cue (grade_text.ABS_CUE, D12); >= 8 pool values; no gold.
  VAL     the gold is a candidate, and >= 8 pool values occur nowhere in the thread (any case).
  G7      every question <= 20 words; proxy tokens (thread messages, questions, 50 words per probe reply) <= 1,800.
  G2      (reported, not gating) rc12's 12 generic and 14 audit value cheaters and play.LATEST_USER_SWAP: family OODH
          <= 0.40 and every cell (probe-kind mix) <= 0.50; ORDER_ABS: cells holding H-ABS <= 0.50. ZERO/NEVER_ABS as
          in rc12. On OOD-H the type matchers pass structurally (s12), so this is recorded, not enforced.
  G4      (reported, not gating) per thread, at most 3 of the 11 generic cheaters (OVERLAP_A and _B as one) score 1.00.
  multi_value (reported) the VAL probes whose thread user turns hold 2+ same-type values (candidates or pool values):
          on that subset the type matchers cannot pass structurally, so each cheater's pass count there is reported.
Reported only: MYONLY, MARKER_OBJ, per probe kind rates, the G4 histogram. Not run: 13-gram decontamination, G11.
Writes sealed/oodh/gate_report.json and a GATE section (counts only) at the end of oodh/HASHES.txt.
Run: python3 -B oodh/gate.py [--data FILE] [--leaked FILE]"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
import select  # noqa: E402,F401  the stdlib module, cached first: oodh/ holds a select.py of its own
REPO = os.path.dirname(HERE)
sys.path[:0] = [HERE]
sys.path.append(os.path.join(REPO, "corpus"))

import argparse  # noqa: E402
import json  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

import build as B  # noqa: E402  (puts rc12/ on sys.path)
import hashes as H  # noqa: E402
import leaks as L  # noqa: E402
import play as P  # noqa: E402
import fakes as F  # noqa: E402
import fakes_audit as FA  # noqa: E402
import graders as G  # noqa: E402
import grade_text as T  # noqa: E402
import pools_vals as V  # noqa: E402
from oodh import in_oodh_reserve  # noqa: E402  (corpus/oodh.py, the ONE reserve rule)

SEALED = B.SEALED
BAR = dict(family=0.40, cell=0.50, item=3)
GATING = ("ids", "struct", "G1", "L2", "G8", "ABS", "VAL", "G7")   # G2 and G4 are reported, not gating (DESIGN s12)
ZERO = ["ECHO", "DEFLECT", "SHOTGUN", "USERVOICE", "ABSTAIN", "ROLECAP", "LATEST_USER"]
ABS_OK = {"ABSTAIN", "DEFLECT", "ORDER_ABS"}
BARRED = F.GENERIC + FA.VALUE_RULES + ["LATEST_USER_SWAP"]
CHEATERS = BARRED + ["LATEST_USER", "ORDER_ABS"] + FA.REFERENCE
REFERENCE = ["IDEAL", "TERSE"]


def has(text, v):
    return bool(v) and V.value_re(v, strict_case=False).search(T.norm(text or "")) is not None


def static_checks(recs, leaked):
    bad = defaultdict(list)
    tids = Counter(r["tree_id"] for r in recs)
    for r in recs:
        rid, tid = r["id"], r["tree_id"]
        if not in_oodh_reserve(tid) or tid in leaked or tids[tid] > 1 or r["message_ids"][0] != tid:
            bad["ids"].append(rid)
        n = r["meta"]["n_thread_user_turns"]
        ps, thread = r["probes"], [t for t in r["turns"] if t["i"] <= n]
        gr = Counter(p["grader"] for p in ps)
        if ([p["turn"] for p in ps] != list(range(n + 1, n + 1 + len(ps))) or r["n_turns"] != len(r["turns"])
                or [p["kind"] for p in ps] != ["X"] * (len(ps) - 1) + ["P"] or not 1 <= len(ps) <= 3
                or gr["VAL"] < 1 or gr["ABS"] > 1 or any(t["kind"] == "D" and type(t.get("asks")) is not bool
                                                          for t in thread)
                or any((r["turns"][p["turn"] - 1]["text"], r["turns"][p["turn"] - 1]["ideal"])
                       != (p["question"], p["ideal"]) for p in ps if p["turn"] <= len(r["turns"]))):
            bad["struct"].append(rid)
        msgs = [t["text"] for t in thread] + [t["history_reply"] for t in thread]
        words = sum(len(m.split()) for m in msgs) + sum(len(p["question"].split()) + 50 for p in ps)
        if 1.35 * words + 4 * 2 * len(r["turns"]) > 1800:
            bad["G7"].append(rid)
        for p in ps:
            at = f"{rid}:u{p['turn']}"
            if len(p["question"].split()) > 20:
                bad["G7"].append(at)
            own = p["accepted"] + p["candidates"]
            later = [f for q in ps if q["turn"] > p["turn"] for f in q["accepted"]]
            after = [t["text"] for t in r["turns"] if p["src"] and t["i"] > max(p["src"])]
            if (any(has(p["question"], f) for f in own + later) or any(has(p.get("prefix"), f) for f in own)
                    or any(has(t, p["gold"]) for t in after)):
                bad["L2"].append(at)
            if p["grader"] == "ABS":
                cue = T.ABS_CUE.search(T.norm(p["question"]))
                inside = [v for v in p["pool_values"] if any(has(m, v) for m in msgs + [q["question"] for q in ps])]
                if inside or cue or len(p["pool_values"]) < 8 or p["gold"] or p["src"]:
                    bad["ABS"].append(at)
                continue
            if not p["src"] or any(s > n or not T.mentioned(T.norm(r["turns"][s - 1]["text"]), p["gold"])
                                   for s in p["src"]):
                bad["G8"].append(at)
            outside = [v for v in p["pool_values"] if not any(has(m, v) for m in msgs)]
            if p["gold"] not in p["candidates"] or len(outside) < 8:
                bad["VAL"].append(at)
    return bad


def reference_checks(recs, rows):
    """G1 on the IDEAL and TERSE rows, and every accepted form alone against the IDEAL transcript."""
    bad = []
    byid = {r["id"]: r for r in recs}
    for name, rs in rows.items():
        for row in rs:
            rec = byid[row["id"]]
            pt = {p["turn"] for p in rec["probes"]}
            if row["unit"] != 1.0 or any(row["flags"][i - 1] for i in pt) or any(e["turn"] in pt for e in row["leaks"]):
                bad.append(f"{name} {row['id']}")
            if any(x["reply"] != rec["turns"][x["i"] - 1]["history_reply"] or x["stop"] != "eos"
                   for x in row["turns"] if x["i"] not in pt):
                bad.append(f"{name} {row['id']} history changed")
    for row in rows["IDEAL"]:
        rec = byid[row["id"]]
        replies, stops = [x["reply"] for x in row["turns"]], [x["stop"] for x in row["turns"]]
        for p in rec["probes"]:
            for form in p["accepted"] if p["grader"] == "VAL" else []:
                alt = replies[:p["turn"] - 1] + [f"{form}."] + replies[p["turn"]:]
                if not G.grade_probe(rec, p, alt, stops)["ok"]:
                    bad.append(f"accepted {row['id']}:u{p['turn']}")
    return bad


def cheater_checks(recs, sc):
    bad = []
    abs_cells = {r["cell"] for r in recs if any(p["grader"] == "ABS" for p in r["probes"])}
    for name, s in sc.items():
        cells = {c: v for c, v in s["cells"].items() if name != "ORDER_ABS" or c in abs_cells}
        if name in BARRED and s["family"] > BAR["family"]:
            bad.append(f"G2 {name}: family {s['family']:.2f}")
        if name in BARRED + ["ORDER_ABS"]:
            bad += [f"G2 {name}: cell {c}={v:.2f} (n={s['cell_n'][c]})" for c, v in sorted(cells.items())
                    if v > BAR["cell"]]
        for g in ("VAL", "ABS"):
            if name in ZERO and (g == "VAL" or name not in ABS_OK) or (g == "ABS" and name not in ABS_OK):
                n = sum(x["ok"] for x in s["probes"] if x["grader"] == g)
                if n:
                    bad.append(f"G2 {name}: passes {n} {g} probes ({'ZERO' if name in ZERO else 'NEVER_ABS'})")
    passes = defaultdict(set)
    for name in F.GENERIC:
        for rid, u in sc[name]["units"].items():
            passes[rid] |= {name.split("_")[0] if name.startswith("OVERLAP") else name} if u == 1.0 else set()
    hist = Counter(len(v) for v in passes.values())
    bad += [f"G4 {rid} passed by {len(v)}: {','.join(sorted(v))}" for rid, v in sorted(passes.items())
            if len(v) > BAR["item"]]
    return bad, dict(sorted(hist.items()))


def multi_value_probes(recs):
    """VAL probes whose THREAD USER turns hold 2+ same-type values (a candidate or pool value each). On these the
    type matchers cannot pass just by knowing the answer's kind. Returns a set of (record id, probe turn)."""
    mv = set()
    for r in recs:
        n = r["meta"]["n_thread_user_turns"]
        users = [t["text"] for t in r["turns"] if t["i"] <= n]
        for p in r["probes"]:
            if p["grader"] != "VAL":
                continue
            vals = set(p["candidates"]) | set(p["pool_values"])
            if sum(1 for v in vals if any(has(u, v) for u in users)) >= 2:
                mv.add((r["id"], p["turn"]))
    return mv


def gate(recs, leaked):
    stat = static_checks(recs, leaked)
    rows = {n: P.play(recs, n) for n in REFERENCE}
    sc = {n: P.scores(P.play(recs, n), recs) for n in CHEATERS}
    g1 = reference_checks(recs, rows)
    g24, hist = cheater_checks(recs, sc)
    found = dict(stat, G1=g1)
    checks = {k: found.get(k, []) for k in GATING}
    reported = dict(G2=[b for b in g24 if b.startswith("G2")], G4=[b for b in g24 if b.startswith("G4")])
    table = {n: dict(family=round(s["family"], 3), kinds={k: round(v, 3) for k, v in sorted(s["kinds"].items())},
                     max_cell=max(s["cells"].values())) for n, s in sc.items()}
    mv = multi_value_probes(recs)
    n_val = sum(1 for r in recs for p in r["probes"] if p["grader"] == "VAL")
    mv_pass = {n: sum(1 for x in sc[n]["probes"] if (x["id"], x["turn"]) in mv and x["ok"]) for n in CHEATERS}
    multi_value = dict(n_probes=len(mv), n_val=n_val, cheater_pass=dict(sorted(mv_pass.items())))
    return dict(checks=checks, reported=reported, table=table, g4_hist=hist, multi_value=multi_value,
                ok=not any(checks.values()))


def report_lines(res, file_sha, n_recs, n_leaked):
    lines = [f"{H.GATE_HEAD} on oodh_part1.jsonl sha256 {file_sha}: {'PASS' if res['ok'] else 'FAIL'} (gating checks)",
             f"  {n_recs} threads; leak list of {n_leaked} tree ids"]
    lines += [f"  {k:6s} {'pass' if not v else f'FAIL {len(v)}'}" for k, v in res["checks"].items()]
    rep = res["reported"]
    lines.append("  reported, NOT gating for OOD-H (DESIGN s12):")
    for k in ("G2", "G4"):
        lines.append(f"    {k:5s} {'clean' if not rep[k] else f'{len(rep[k])} findings (structural, see s12)'}")
    mv = res["multi_value"]
    top = sorted(((n, c) for n, c in mv["cheater_pass"].items() if c), key=lambda x: -x[1])[:6]
    lines.append(f"    2+-value VAL probes: {mv['n_probes']} of {mv['n_val']} VAL; top cheater passes there: "
                 + (", ".join(f"{n} {c}" for n, c in top) if top else "none"))
    lines.append("  G4 histogram (threads passed by k of the 11 generic cheaters): "
                 + ", ".join(f"{k}:{v}" for k, v in res["g4_hist"].items()))
    lines.append("  cheater family scores: " + ", ".join(f"{n} {t['family']:.2f}" for n, t in res["table"].items()))
    verdict = ("LOCK CANDIDATE (150 threads; gating checks pass; G2/G4 reported per DESIGN s12, not gating)"
               if res["ok"] and n_recs >= 150 else
               "not a lock candidate" + ("" if res["ok"] else " (a gating check failed)"))
    lines.append(f"  verdict: {verdict}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(SEALED, "oodh_part1.jsonl"))
    ap.add_argument("--leaked", default=os.path.join(SEALED, "leaked_reserve_trees.txt"))
    ap.add_argument("--hashes", default=os.path.join(HERE, "HASHES.txt"))
    a = ap.parse_args()
    if not os.path.exists(a.leaked):
        print(f"GATE FAIL: no leak list at {a.leaked} (run oodh/leaks.py where the OASST2 file is)")
        return 1
    leaked = L.load(a.leaked)
    recs = B.read_jsonl(a.data)
    res = gate(recs, leaked)
    with open(os.path.join(os.path.dirname(a.data), "gate_report.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    lines = report_lines(res, H.sha(a.data), len(recs), len(leaked))
    H.put_gate(a.hashes, lines)
    print("\n".join(lines))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
