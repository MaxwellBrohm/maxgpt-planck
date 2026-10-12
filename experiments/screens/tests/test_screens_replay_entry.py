"""SCREENS AMENDMENT BASE-REPLAY (2026-10-10; cuts audit S7, row 29), the entry (SCREENS.txt; written in experiments/
screens/pending/replay.txt and moved there unchanged): its record and labels, the cap check generated from
measured_hours.tsv through screens_hours and analyze_lib.cap_cut at 28 h, the PC lock hours (tests/replay_record.py),
the readings' constants from E3 results.json, the counts of tests and mutants, and every number and hash in it.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_replay_entry.py
"""
import contextlib, io, json, math, os, re, sys  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze_lib as AL, screens_hours as H, screens_lib as L, replay_lib as R, replay_record as RR  # noqa: E401,E402
sys.path.insert(0, L.E3D)
from e3analyze import t_ppf  # noqa: E402
from test_screens_configs import P  # noqa: E402
from test_screens_replay import BAD, CASES  # noqa: E402

PASSED, PRE, WHOLE, NEW, GUARD = (611, 665), 387, 376, 42, 14  # suite before / after, pre-state, mutants


def test_the_entry_and_its_record():
    sec = " ".join(R.section().split())
    assert R.recorded_replay() == R.RUNS and R.section().count("\n  BASE RUNS ADDED BY AMENDMENT BASE-REPLAY: ") == 1
    assert R.entry_text().count("\n" + R.HEAD) + R.entry_text().startswith(R.HEAD) == 1
    for phrase in ('labels it "BASE replay: stable"', '"BASE replay: sensitive (V at the replay level)"',
                   "the seed 101 shift applied at both seeds", "with the SD_ref, thr and delta of its registered reading",
                   "S005 (its own BASE ran in the same seed set) and every stage 1 contrast", '"not applicable"',
                   "The hook-off run puts no label on any pair", "Report-only: no C7 verdict, C4 noise check, Holm "
                   "reading or onset sigma_seed reads these runs", "never \"no effect\"", "Decided before any stage 2 "
                   "seed result, onset, HD or verdict is read", "eval.induction off stays refused under every other name",
                   "plans/CURRENT names stage2_replay if any differs", "screens.py replay NAME writes them"):
        assert phrase in sec, phrase
    lock = [r for r, _, _ in RR.LOCK]
    plans = {n: [ln.split()[1] for ln in open(os.path.join(L.HERE, "plans", n + ".txt")) if ln.startswith("train ")]
             for n in ("stage2_seeds", "stage2_s006_seeds", "stage2_base_diag")}
    assert lock == [r for r in plans["stage2_seeds"] if not r.startswith("base_")] + plans["stage2_s006_seeds"] + \
        plans["stage2_base_diag"]                   # the queue skipped stage2_seeds' two scored BASE lines


def numbers():
    with contextlib.redirect_stdout(io.StringIO()):
        rep = H.report(P)                           # tally and measured hours only: no cap read
    q, mr, mo = rep["tally"]["queued"], rep["measured_runs"], rep["other"]
    p6, pb = H.run_h(L.STEPS, H.mult("S006", "mtp")), H.run_h(L.STEPS, H.mult(None, None))
    qs, q4 = sum(q.values()), {**q, "S006": q["S006"] + 2 * p6, "BASE": q.get("BASE", 0.0) + 6 * pb}
    c, nar = AL.cap_cut(q4, mr + mo, cap=28.0), AL.cap_cut(q4, mr, cap=28.0)
    held = sum(H.hours_between(a, b) for _, a, b in RR.LOCK)
    e3 = json.load(open(os.path.join(L.E3D, "results.json")))["replay"]
    c101 = json.load(open(os.path.join(L.EXPD, "S001_attn_gate", "results.json")))["base_level_vs_e3_part1"]["metrics"][
        "CHAT"]["base"][0]
    bar = t_ppf(0.975, 3) * math.sqrt(1.5) * math.sqrt(2 / 3) * e3["F(CHAT)"]["sigma_rep_rel"] * c101
    f3 = lambda x: f"{x:.3f}"  # noqa: E731
    return {"mr": f3(mr), "mo": f3(mo), "m": f3(mr + mo), "qs": f3(qs), "ind": f3(2 * p6), "pb3": f3(3 * pb),
            "q4": f3(sum(q4.values())), "tot": f3(c["total_before"]), "room": f3(28.0 - c["total_after"]),
            "narrow": f3(nar["total_before"]), "held": f3(held), "est": f3(qs + 2 * p6 + 3 * pb),
            "act": f3(mr + mo + held + 3 * pb), "room_act": f3(28.0 - mr - mo - held - 3 * pb), "t3": f3(t_ppf(0.975, 3)),
            "e3c": f"{100 * e3['F(CHAT)']['sigma_rep_rel']:.3f}", "e3p": f"{100 * e3['F(PROSE)']['sigma_rep_rel']:.3f}",
            "bar": f"{bar:.5f}", "barp": f"{100 * bar / c101:.2f}", "c101": f"{c101:.5f}", "cut": c["cut"]}


def test_cap_check_constants_and_every_number():
    v, sec = numbers(), " ".join(R.section().split())
    assert v["cut"] == [] and float(v["tot"]) <= 28.0 and float(v["act"]) <= 28.0
    assert f"Measured {v['mr']} h of runs + {v['mo']} h other = {v['m']} h; queued 10 of 90 registered runs {v['qs']} h " \
           f"+ S006's 2 IND runs {v['ind']} h + BASE-DIAG's 3 runs {v['pb3']} h + these 3 runs {v['pb3']} h = {v['q4']} h " \
           f"(unrounded); total {v['tot']} h <= 28 h: nothing cut, {v['room']} h of headroom (runs only, reported: " \
           f"{v['narrow']} h)." in sec
    assert f"held the lock {v['held']} h against their {v['est']} h estimate" in sec
    assert f"the total is {v['act']} h, {v['room_act']} h of headroom" in sec and f"(3 runs, {v['pb3']} h)" in sec
    assert f"|t| > t(0.975, 3) = {v['t3']} (two-sided p < 0.05)" in sec and f"F(CHAT) {v['e3c']}%, F(PROSE) {v['e3p']}%" in sec
    assert f"is {v['bar']} bpb ({v['barp']}% of base_s101's {v['c101']})" in sec and f"Estimate {v['pb3']} h." in sec
    hexes = {L.sha256(L.find(r)[0])[:16] for r in R.RUNS} | {L.sha256(R.REPLAY_PLAN)[:16], R.CODE_DIGEST[:16], *RR.PCSHA}
    hexes |= {"93bc10b", "4ad2be4", "c05d11c", "d21df3d", "3227c60", RR.LAST_START[1][:7], RR.LAST_START[1]}
    text = re.sub(r"\b[a-z]\w*_[\w.<>]*", " ", R.section())
    assert set(re.findall(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b", text)) == hexes
    for r in R.RUNS:
        assert f"{L.sha256(L.find(r)[0])[:16]} {r}" in sec
    text = re.sub(r"\b[0-9a-f]{7,40}\b|\b\d{4}-\d\d-\d\d\b|\b\d\d:\d\d(?::\d\d)?\b", " ", text)
    exp = {x for k, x in v.items() if k != "cut"} | {"0.66", "0.00755", "0.00594", "596.49", "2.13", "0.975", "0.05",
                                                     "0.31", str(R.CODE_N), "101", "103", "105", "253", "611",
                                                     str(PASSED[1]), str(PRE), str(WHOLE)}
    nums = {n for n in re.findall(r"(?<![\w.])(\d[\d,]*\d(?:\.\d+)*|\d\.\d+)(?![\w])", text) if "." in n or len(n) >= 3}
    assert nums <= exp and len(nums) > 20, sorted(nums - exp)
    assert len(L.all_configs()) == 99 and "3 ok, 99 ok over every config" in sec
    import mutation_screens as M
    from mutants_replay import MUTANTS_REPLAY
    assert len(M.MUTANTS) >= WHOLE and len(MUTANTS_REPLAY) == NEW   # later steps add mutants and all(m in M.MUTANTS for m in MUTANTS_REPLAY)
    for phrase in (f"tests {PASSED[1]} passed, 1 skipped ({PASSED[0]} before", f"{NEW} new mutants (tests/mutants_replay.py), "
                   f"{NEW} of {NEW} killed; the whole file {WHOLE} of {WHOLE} killed", f"({PRE} passed)",
                   f"{len(CASES)} check refusals", f"the acceptance refusing {len(BAD)} bad states",
                   f"{GUARD} of {GUARD} killed"):
        assert phrase in sec, phrase
