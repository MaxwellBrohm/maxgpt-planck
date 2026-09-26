"""oodh/gate.py and oodh/play.py on the synthetic fixture (tests/fixture.py; no real item text). Loads no model.
Run: python3 -B oodh/tests/test_gate.py"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OODH = os.path.dirname(HERE)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != OODH]
import select  # noqa: E402,F401  the stdlib module first: oodh/ holds a select.py of its own
sys.path[:0] = [OODH, HERE]

import copy  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402

import build as B  # noqa: E402
import fakes as F  # noqa: E402
import fixture as FX  # noqa: E402
import gate as GT  # noqa: E402
import hashes as H  # noqa: E402
import play as P  # noqa: E402
import runner as RN  # noqa: E402
from test_build import TESTS, main, test, world  # noqa: E402

TESTS.clear()


def fixture_recs():
    root, tids = world()
    return root, tids, B.build(B.load_items(root), 9)[0]


def flagged(recs, key, leaked=frozenset()):
    return GT.static_checks(recs, set(leaked))[key]


@test
def fixed_history_only_probes_reach_the_model():
    _, _, recs = fixture_recs()
    seen = []

    class Spy(F.Fake):
        def reply(self, history, i):
            seen.append((self.rec["id"], i, [m["content"] for m in history]))
            return super().reply(history, i)

    rows = RN.run(recs[:2], P.FixedHistory(Spy()), "plain", [None])
    r = recs[0]
    n = r["meta"]["n_thread_user_turns"]
    asked = [(rid, i) for rid, i, _ in seen]
    assert asked == [(x["id"], p["turn"]) for x in recs[:2] for p in x["probes"]], asked
    hist = [h for rid, i, h in seen if rid == r["id"]]
    thread = [m for t in r["turns"][:n] for m in (t["text"], t["history_reply"])]
    assert hist[0] == thread + [r["turns"][n]["text"]]
    assert hist[1] == thread + [r["turns"][n]["text"], r["turns"][n]["ideal"], r["turns"][n + 1]["text"]]
    assert [x["stop"] for x in rows[0]["turns"][:n]] == ["eos"] * n and rows[0]["unit"] == 1.0
    assert all(P.latest_thread_turn(r, p) == r["turns"][n - 1]["text"] for p in r["probes"])
    terse = P.play(recs[:1], "TERSE")[0]["turns"]
    assert [terse[p["turn"] - 1]["reply"] for p in r["probes"]] == [
        f"{p['gold']}." if p["grader"] == "VAL" else p["ideal"] for p in r["probes"]]


@test
def scores_rates():
    recs = [dict(id="a", probes=[dict(turn=3, probe_kind="H-FACT", grader="VAL"),
                                 dict(turn=4, probe_kind="H-ABS", grader="ABS")]),
            dict(id="b", probes=[dict(turn=3, probe_kind="H-FACT", grader="VAL")])]
    rows = [dict(id="a", cell="H-ABS+H-FACT", unit=0.5, probes=[dict(turn=3, ok=True, fails=[]),
                                                              dict(turn=4, ok=False, fails=["a2_cue"])]),
            dict(id="b", cell="H-FACT", unit=0.0, probes=[dict(turn=3, ok=False, fails=["v2_gold"])])]
    s = P.scores(rows, recs)
    assert (s["family"], s["cells"], s["cell_n"], s["kinds"], s["units"]) == (
        0.25, {"H-ABS+H-FACT": 0.5, "H-FACT": 0.0}, {"H-ABS+H-FACT": 1, "H-FACT": 1}, {"H-FACT": 0.5, "H-ABS": 0.0},
        {"a": 0.5, "b": 0.0}), s
    assert [(x["id"], x["grader"], x["ok"]) for x in s["probes"]] == [("a", "VAL", True), ("a", "ABS", False),
                                                                      ("b", "VAL", False)]


@test
def static_clean():
    _, _, recs = fixture_recs()
    bad = GT.static_checks(recs, set())
    assert not any(bad.values()), dict(bad)


def mutated(recs, k, fn):
    out = copy.deepcopy(recs)
    fn(out[k])
    return out


@test
def static_catches_each_defect():
    _, tids, recs = fixture_recs()
    ka = [k for k, r in enumerate(recs) if "H-CORR" in r["cell"]][0]
    kb = [k for k, r in enumerate(recs) if "H-ASK" in r["cell"]][0]
    a, b = recs[ka], recs[kb]
    c0, rel, pet = a["probes"][0]["stale"][0], a["probes"][1]["gold"], b["probes"][1]["gold"]
    rid = a["id"]
    out = FX.OUTSIDE[0]
    assert rid in flagged(mutated(recs, ka, lambda r: r.update(tree_id=out, message_ids=[out])), "ids")
    assert flagged(recs, "ids", {a["tree_id"]}) == [rid]
    assert rid in flagged(recs + [copy.deepcopy(a)], "ids")
    assert rid in flagged(mutated(recs, ka, lambda r: r["message_ids"].reverse()), "ids")
    assert b["id"] in flagged(mutated(recs, kb, lambda r: r["turns"][2].pop("asks")), "struct")
    assert rid in flagged(mutated(recs, ka, lambda r: r["probes"][1].update(grader="ABS")), "struct")
    assert rid in flagged(mutated(recs, ka, lambda r: r["turns"][3].update(text="Who came along?")), "struct")
    assert rid in flagged(mutated(recs, ka, lambda r: r["probes"][2].update(kind="X")), "struct")
    assert rid in flagged(mutated(recs, ka, lambda r: r.update(n_turns=4)), "struct")

    def q(j, text):                          # a probe question lives in the probe and in its turn
        def fn(r):
            r["probes"][j]["question"] = r["turns"][r["probes"][j]["turn"] - 1]["text"] = text
        return fn
    assert f"{rid}:u4" in flagged(mutated(recs, ka, q(1, f"Did I say my {rel} is coming?")), "L2")
    assert f"{rid}:u3" in flagged(mutated(recs, ka, lambda r: r["probes"][0].update(
        prefix=f"You moved from {c0} and settled on")), "L2")
    assert f"{b['id']}:u4" in flagged(mutated(recs, kb, q(0, f"What did I ask you to write for my {pet}?")), "L2")
    assert f"{rid}:u4" in flagged(mutated(recs, ka, q(2, f"How much can my {rel} and I spend?")), "L2")
    assert f"{rid}:u4" in flagged(mutated(recs, ka, lambda r: r["probes"][1].update(src=[2])), "G8")
    assert f"{rid}:u5" in flagged(mutated(recs, ka, lambda r: r["turns"][0].update(
        history_reply=r["turns"][0]["history_reply"] + " Plan on about $500.")), "ABS")
    assert f"{rid}:u5" in flagged(mutated(recs, ka, q(2, "You haven't told me the budget, have you?")), "ABS")
    assert f"{rid}:u5" in flagged(mutated(recs, ka, lambda r: r["probes"][2]["pool_values"].pop()), "ABS")
    assert f"{rid}:u4" in flagged(mutated(recs, ka, lambda r: r["probes"][1].update(candidates=[])), "VAL")
    assert f"{rid}:u3" in flagged(mutated(recs, ka, lambda r: r["probes"][0]["pool_values"].pop()), "VAL")
    assert f"{rid}:u4" in flagged(mutated(recs, ka, q(1, "Who " + "really " * 16 + "did I say I go with?")), "G7")
    assert rid in flagged(mutated(recs, ka, lambda r: r["turns"][1].update(history_reply="word " * 1250)), "G7")
    assert not flagged(mutated(recs, ka, lambda r: r["turns"][1].update(history_reply="word " * 1000)), "G7")

    def verse(r):                            # an earlier probe's accepted form (not its gold) after it: RC-12 allows
        r["probes"][0]["accepted"].append("verse")
        r["probes"][1]["question"] = "Which animal did I want the verse to be about?"
    assert not flagged(mutated(recs, kb, verse), "L2")


@test
def reference_checks():
    _, _, recs = fixture_recs()
    rows = {n: P.play(recs, n) for n in GT.REFERENCE}
    assert GT.reference_checks(recs, rows) == []
    tweaks = [lambda x: x["leaks"].append(dict(turn=x["probes"][0]["turn"], kind="capture", value="v")),
              lambda x: x["flags"].__setitem__(x["probes"][0]["turn"] - 1, ["LOOP"]),
              lambda x: x["turns"][0].update(stop="cap"), lambda x: x["turns"][0].update(reply="Changed.")]
    for tweak in tweaks:                     # a leak event, a loop flag or a changed stop at a probe / history turn
        rs = copy.deepcopy(rows)
        tweak(rs["IDEAL"][0])
        assert any(rs["IDEAL"][0]["id"] in b for b in GT.reference_checks(recs, rs)), tweak
    k = [k for k, r in enumerate(recs) if "H-CORR" in r["cell"]][0]
    rid, c0, c1 = recs[k]["id"], recs[k]["probes"][0]["stale"][0], recs[k]["probes"][0]["gold"]
    wrong = mutated(recs, k, lambda r: r["turns"][2].update(ideal=f"You settled on {c0}."))
    assert f"IDEAL {rid}" in GT.reference_checks(wrong, {n: P.play(wrong, n) for n in GT.REFERENCE})
    hedged = mutated(recs, k, lambda r: r["probes"][0]["accepted"].append(f"maybe {c1}"))
    assert GT.reference_checks(hedged, {n: P.play(hedged, n) for n in GT.REFERENCE}) == [f"accepted {rid}:u3"]
    cut = mutated(recs, k, lambda r: r["turns"][0].update(history_reply="Sure.\nUser: and then?"))
    got = GT.reference_checks(cut, {n: P.play(cut, n) for n in GT.REFERENCE})
    assert f"IDEAL {rid} history changed" in got and f"TERSE {rid} history changed" in got, got


def score(family=0.0, cells=None, units=None, probes=()):
    cells = cells or {"H-ABS+H-FACT": 0.0, "H-FACT": 0.0}
    return dict(family=family, cells=cells, cell_n={c: 4 for c in cells}, kinds={}, units=units or {},
                probes=[dict(grader=g, ok=o) for g, o in probes])


@test
def cheater_bars():
    recs = [dict(cell="H-ABS+H-FACT", probes=[dict(grader="ABS")]), dict(cell="H-FACT", probes=[dict(grader="VAL")])]
    sc = {n: score() for n in GT.CHEATERS}
    assert GT.cheater_checks(recs, sc) == ([], {})
    cases = [("FIRST", score(0.41), "G2 FIRST: family 0.41"), ("FIRST", score(0.40), None),
             ("EXL_MARKER", score(cells={"H-FACT": 0.51}), "G2 EXL_MARKER: cell H-FACT=0.51 (n=4)"),
             ("LAST", score(cells={"H-FACT": 0.50}), None),
             ("LATEST_USER_SWAP", score(0.45), "G2 LATEST_USER_SWAP: family 0.45"),
             ("ORDER_ABS", score(0.9, {"H-FACT": 1.0, "H-ABS+H-FACT": 0.5}), None),
             ("ORDER_ABS", score(0.9, {"H-ABS+H-FACT": 0.6}), "G2 ORDER_ABS: cell H-ABS+H-FACT=0.60 (n=4)"),
             ("SHOTGUN", score(probes=[("VAL", True)]), "G2 SHOTGUN: passes 1 VAL probes (ZERO)"),
             ("LATEST_USER", score(probes=[("VAL", True)]), "G2 LATEST_USER: passes 1 VAL probes (ZERO)"),
             ("ECHO", score(probes=[("ABS", True)]), "G2 ECHO: passes 1 ABS probes (ZERO)"),
             ("ABSTAIN", score(probes=[("ABS", True), ("ABS", True)]), None),
             ("DEFLECT", score(probes=[("VAL", True)]), "G2 DEFLECT: passes 1 VAL probes (ZERO)"),
             ("FIRST", score(probes=[("ABS", True)]), "G2 FIRST: passes 1 ABS probes (NEVER_ABS)"),
             ("ORDER_ABS", score(probes=[("ABS", True)]), None),
             ("MYONLY", score(0.9, {"H-FACT": 0.9}, probes=[("VAL", True)]), None)]
    for name, s, want in cases:
        bad, _ = GT.cheater_checks(recs, dict(sc, **{name: s}))
        assert bad == ([want] if want else []), (name, bad)
    four = {n: score(units={"t1": 1.0}) for n in ("FIRST", "LAST", "OVERLAP_A", "OVERLAP_B", "WORDING")}
    bad, hist = GT.cheater_checks(recs, dict(sc, **four))
    assert bad == ["G4 t1 passed by 4: FIRST,LAST,OVERLAP,WORDING"] and hist == {4: 1}, (bad, hist)
    three = {n: score(units={"t1": 1.0}) for n in ("FIRST", "LAST", "OVERLAP_A", "OVERLAP_B")}
    assert GT.cheater_checks(recs, dict(sc, **three)) == ([], {3: 1})


@test
def gate_end_to_end():
    _, _, recs = fixture_recs()
    res = GT.gate(recs, set())
    c = res["checks"]
    assert not any(c[k] for k in ("ids", "struct", "G1", "L2", "G8", "ABS", "VAL", "G7")), c
    g2 = " | ".join(c["G2"])
    assert "G2 SHOTGUN: passes" in g2 and "G2 LATEST_USER: passes" in g2 and not res["ok"], g2
    assert "ECHO" not in g2 and "ABSTAIN" not in g2 and "USERVOICE" not in g2, g2


@test
def cli():
    root, tids, recs = fixture_recs()
    data, leaked, hp = (os.path.join(root, x) for x in ("oodh_part1.jsonl", "leaked.txt", "HASHES.txt"))
    with open(data, "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in recs))
    H.write_hashes(recs, True, 150, data, hp, [data])
    run = lambda: subprocess.run([sys.executable, "-B", os.path.join(OODH, "gate.py"), "--data", data,  # noqa
                                  "--leaked", leaked, "--hashes", hp], capture_output=True, text=True, timeout=100)
    r = run()
    assert r.returncode == 1 and "no leak list" in r.stdout and H.GATE_HEAD not in open(hp).read(), r.stdout
    with open(leaked, "w") as fh:
        fh.write(tids[0] + "\n")
    r = run()
    text = open(hp).read()
    assert r.returncode == 1 and f"{H.GATE_HEAD} on oodh_part1.jsonl sha256 {H.sha(data)}: FAIL" in text, r.stderr
    assert "  ids    FAIL 1" in text and "  G1     pass" in text and text.startswith("OOD-H Part 1 hashes")
    rid = [x["id"] for x in recs if x["tree_id"] == tids[0]]
    assert json.load(open(os.path.join(root, "gate_report.json")))["checks"]["ids"] == rid


if __name__ == "__main__":
    sys.exit(main())
