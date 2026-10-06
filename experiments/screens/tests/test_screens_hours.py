"""SCREENS GPU hours and ORDER's cap (screens_hours.py; SCREENS.txt GPU HOURS, CAP AND CUT RULE, STAGE 2 READINESS):
a measured run counts once (its estimate leaves the queue), the flag factors are the logged eager smoke tok/s, an
S003 search arm costs what stage A measured, and every number is found in the file it is cited from. No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_hours.py
"""
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import screens_hours as H  # noqa: E402
import screens_lib as L  # noqa: E402

P = L.params()
RUNS, OTHER = H.load()
F = 250_019_840 / 255_660 * 1.15 / 3600            # GPU HOURS: one full 5M run, 250.0M slots, +15%


def section(rel: str, head: str) -> str:
    """The entry of a notes file that starts with the column-0 line `head`, up to the next blank line."""
    lines = open(os.path.join(L.ROOT, rel)).read().splitlines()
    i = next(i for i, ln in enumerate(lines) if ln.startswith(head))
    j = next((j for j in range(i, len(lines)) if not lines[j].strip()), len(lines))
    return "\n".join(lines[i:j])


QUEUE = """2026-10-04 22:47:58 waiting for gpu.lock
2026-10-04 22:50:57 gpu.lock held; GPU memory in use 401 MiB
2026-10-04 22:50:59 train s003_adamw_e6_r1_trunk.yaml --require-committed
2026-10-04 23:04:10 done s003_adamw_e6_r1_trunk rc 0
2026-10-04 23:04:11 waiting for gpu.lock
2026-10-04 23:05:40 gpu.lock held; GPU memory in use 540 MiB
2026-10-04 23:05:41 train s003_adamw_e6_r1_b62M.yaml --require-committed
2026-10-04 23:06:42 done s003_adamw_e6_r1_b62M rc 0
2026-10-04 23:06:42 score s003_adamw_e6_r1_b62M/final_00001908
2026-10-04 23:08:26 waiting for gpu.lock
2026-10-05 10:01:07 gpu.lock held; GPU memory in use 474 MiB
2026-10-05 10:01:08 train s003_adamw_e3_r0.5_trunk.yaml --require-committed
"""
STATUS = """{"time": "2026-10-04T23:04:10", "exp": "SCREENS", "run": "s003_adamw_e6_r1_trunk", "outcome": "done"}
{"time": "2026-10-04T23:08:25", "exp": "SCREENS", "run": "s003_adamw_e6_r1_b62M", "outcome": "done and scored"}
"""


def test_queue_parse_is_lock_held_to_status_and_skips_a_running_run():
    """Lines copied from the PC queue log: a lock wait before each run is not GPU time; the run in progress is out."""
    got = H.parse_queue(QUEUE, STATUS)
    assert [g[0] for g in got] == ["s003_adamw_e6_r1_trunk", "s003_adamw_e6_r1_b62M"]       # r0.5 trunk: running
    assert [round(H.hours_between(a, b) * 3600) for _, a, b in got] == [793, 165]
    assert "run\ts003_adamw_e6_r1_trunk\t0.22028\t" in H.tsv_text(QUEUE, STATUS)


STAGE_B = {f"s003_adamw_e3_r{r}_{t}" for r in ("0.5", "2", "4", "8") for t in ("trunk", "b62M", "b125M", "b250M")}
STAGE_C = {f"s003_adamw_e{e}_r2_{t}" for e in ("1.5", "6") for t in ("trunk", "b62M", "b125M", "b250M")}


def test_measured_file_reproduces_the_logged_lock_hours():
    """measured_hours.tsv against SCREENS.txt STAGE 1 SELECTION A RESULT: every lock figure of its table, the
    total 5.483 h and the per-screen sums; its hours column agrees with its own start and end. Since the S003 STAGE
    B RESULT regeneration it also holds stage B's 16 runs, and since S003 STAGE C RESULT stage C's 8 (their figures:
    test_screens_s003_stage_c.py, test_screens_stage1_seeds.py)."""
    sec = section("experiments/SCREENS.txt", "STAGE 1 SELECTION A RESULT")
    table = {m[1]: float(m[2]) for m in re.finditer(r"(s00\d_[a-z]+_[^\s|]+)\s+(\d\.\d{3}) (\d\.\d{3})\s+\d+k", sec)}
    assert len(table) == 32 and len(RUNS) == 56 and set(RUNS) - set(table) == STAGE_B | STAGE_C
    sel = {n: RUNS[n] for n in table}
    assert all(round(sel[n], 3) == pytest.approx(v, abs=1e-9) for n, v in table.items())
    assert round(sum(sel.values()), 3) == 5.483 and "Total 5.483 GPU hours measured" in sec
    per = {s: round(sum(h for n, h in sel.items() if n.startswith(s)), 3) for s in ("s001", "s002", "s003")}
    assert per == {"s001": 0.879, "s002": 2.569, "s003": 2.034} and "(S001 0.879, S002 2.569, S003 search 2.034" in sec
    for ln in open(H.TSV):
        if ln.startswith("run\t"):
            _, n, h, a, b, _ = ln.rstrip("\n").split("\t")
            assert abs(float(h) - H.hours_between(a, b)) < 6e-6, n
    assert OTHER == {"c1a_gate": 2.011, "eager_smokes": 0.109}


def test_other_hours_are_the_logged_gate_and_smoke_hours():
    assert "31 holds, 2.011 h" in section("experiments/SCREENS.txt", "VERIFICATION OF ORDER 0")
    assert "(0.109 h, the failed 10 s hold included)" in section("experiments/SCREENS.txt", "SMOKES RECORDED")


def test_a_measured_run_counts_once():
    """(a): measuring a registered run at exactly its estimate leaves the total unchanged. Before 2026-10-05,
    hours --measured added stage 1's measured hours on top of their still-queued estimates (25.92 h printed)."""
    reg = H.registered(P, RUNS)
    at_est = {n: reg[H.slot(n)[1]][1] for n in RUNS}            # every measured run at its own estimate
    t = H.tally(P, at_est)
    assert sum(at_est.values()) + sum(t["queued"].values()) == pytest.approx(sum(e for _, e in H.registered(P, at_est).values()))
    t = H.tally(P, RUNS)
    assert t["extra"] == {} and len(t["seen"]) == 56 and t["n_registered"] == 90 and sum(t["n_queued"].values()) == 34
    one = H.tally(P, {**RUNS, "s002_novres_g1_s101": 0.3})          # a seed run at its g pick: one slot leaves
    assert sum(one["n_queued"].values()) == 33 and one["extra"] == {}
    assert sum(one["queued"].values()) == pytest.approx(sum(t["queued"].values()) - F)
    ext = H.tally(P, {**RUNS, "s001_nogate_g4_s1": 0.3, "s003_adamw_e3_r16_trunk": 0.2})   # extensions: on top
    assert set(ext["extra"]) == {"s001_nogate_g4_s1", "s003_adamw_e3_r16_trunk"}
    assert H.tally(P, {**RUNS, "s004_canonac_g0.25_s1": 0.4})["queued"] == t["queued"]   # takes no stage 2 slot
    assert ext["queued"] == t["queued"] and ext["measured"]["S001"] == pytest.approx(t["measured"]["S001"] + 0.3)
    no_c = {n: h for n, h in RUNS.items() if n not in STAGE_C}          # r_B 1: stage C reuses stage A runs
    assert sum(H.tally(P, no_c, r_b=1)["n_queued"].values()) == sum(H.tally(P, no_c, r_b=2)["n_queued"].values()) - 8
    assert set(H.tally(P, RUNS, r_b=1)["extra"]) == STAGE_C              # so measured r 2 points would take no slot


def test_factors_are_the_logged_eager_smoke_tok_s():
    """(b): each factor = the shared BASE smoke's tok/s over the arm's, every value on its cited line; the factors
    and per-run hours the notes printed come out; no assumed factor is left in screens_lib."""
    for key, (rel, text) in H.SRC.items():
        sec = section(rel, "SMOKES RECORDED" if key == "BASE" else "EAGER SMOKE")
        assert text in sec and f"{H.SMOKE[key]:,}" in text, key
    logged = {("S004", "canonac"): ("x1.375", "0.429 h"), ("S006", "mtp"): ("x1.177", "0.368 h"),
              ("S007", "smear"): ("x1.027", "0.321 h"), ("S005", "base"): ("x1.321", "0.413 h"),
              ("S005", "forget"): ("x2.981", "0.931 h")}
    for key, (fx, h) in logged.items():
        sec = section(H.SRC[key][0], "EAGER SMOKE")
        assert fx in sec and h in sec and f"x{H.mult(*key):.3f}" == fx, key
        assert f"{H.run_h(L.STEPS, H.mult(*key)):.3f} h" == h, key
    assert H.mult("S005", "forget") / H.mult("S005", "base") == pytest.approx(2.257, abs=5e-4)   # of its own BASE
    assert all(H.mult(sid, a) == 1.0 for sid in ("S001", "S002", "S003") for a in L.SCREENS[sid]["arms"])
    assert H.mult(None, None) == 1.0 and round(H.run_h(L.STEPS), 3) == 0.312
    assert all("mult" not in s for s in L.SCREENS.values())


def test_s003_search_arm_is_the_measured_stage_a_mean():
    """(d): an E2 stage arm scores 9 checkpoints; the +15% rule's 0.359 h was 13% under what stage A measured."""
    tag = H.s003_tag_hours(RUNS)
    arm = sum(tag.values())
    assert 0.406 <= arm <= 0.409 and round(arm, 4) == 0.4068
    assert round(H.run_h(8776), 3) == 0.359 and round(arm / H.run_h(8776) - 1, 2) == 0.13
    reg = H.registered(P, RUNS)
    assert sum(1 for k in reg if k[:2] == ("S003", "search")) == 11 * 4
    assert reg[("S003", "search", "B", 8.0, "b250M")][1] == pytest.approx(tag["b250M"])


def stage2_by_hand(other: bool) -> float:
    """Measured stage 1 selection A, S003 stage B and stage C + every registered run not yet run (the stage 1 seed
    sets, stage 2), from the raw figures (no module call)."""
    base, fac = 253_691, {"canonac": 184_569, "own": 192_092, "forget": 85_113, "mtp": 215_563, "smear": 247_073}
    queued = (2 + 2 + 6 + 2) * F + F * base * (5 / fac["canonac"] + 2 / fac["own"] + 5 / fac["forget"] +
                                               5 / fac["mtp"] + 5 / fac["smear"])
    # stage A, stage B (5,838 s of lock), stage C (2,924 s of lock)
    return 5.4831 + 1.62167 + 0.81222 + queued + (2.011 + 0.109 if other else 0.0)


def test_stage_2_cap_check_with_stage_1_seeds_and_stage_c(capsys, tmp_path):
    """(c): the gate and smoke hours count (the conservative reading); the old double count printed 25.92 h.
    Numbers as of SCREENS.txt S003 STAGE C RESULT (stage C measured, the stage 1 seed sets and stage 2 queued)."""
    r = H.report(P)
    out = capsys.readouterr().out
    assert r["cap"]["total_before"] == pytest.approx(stage2_by_hand(True), abs=2e-4) and r["cap"]["cut"] == []
    assert r["narrow"]["total_before"] == pytest.approx(stage2_by_hand(False), abs=2e-4)
    assert "= 24.856 h against 25 h: fits, nothing cut; headroom 0.144 h" in out
    tsv = tmp_path / "m.tsv"                                        # one queued S003 stage B extension: over 25 h
    tsv.write_text(open(H.TSV).read() + "run\ts003_adamw_e3_r16_trunk\t0.21667\t2026-10-06 10:00:00\t"
                   "2026-10-06 10:13:00\tfixture\n")
    r = H.report(P, str(tsv))
    assert r["cap"]["cut"] == ["S006"] and r["cap"]["fits"]
