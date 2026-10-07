"""SCREENS after S003 stage B (SCREENS.txt S003 STAGE B RESULT, 2026-10-05): the recorded seed-1 values give r_B 2
through the rules (E2's stage B pick, no extension), stage C is exactly g 0.5 and 2 at (3e-3, r 2), its configs and
plan are exactly that and stop at a mark, the entry carries the recorded values and the measured lock hours, and the
cap check before stage C fits. Selection values only; no verdict is read from them (C3, E3 change 3). No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_s003_stage_c.py
"""
import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import analyze  # noqa: E402
import analyze_lib as AL  # noqa: E402
import e2pick  # noqa: E402
import screens_hours as H  # noqa: E402
import screens_lib as L  # noqa: E402
from test_screens_stage1_next import RECORDED as STAGE_1A  # noqa: E402  (stage 1 selection A, same copy rule)

P = L.params()
SETS = ("CHAT", "PROSE", "cccc", "gutenberg", "wikimedia")
# final checkpoint, split all, from ~/planck/runs/SCREENS/<run>/bpb.jsonl (copied 2026-10-05 18:30, sha256 equal on
# both machines): CHAT, PROSE, cccc, gutenberg, wikimedia at the 250M branch
STAGE_B = {
    "s003_adamw_e3_r0.5_b250M": (1.17242, 1.40273, 1.3349, 1.5167, 1.3550),
    "s003_adamw_e3_r2_b250M": (1.16436, 1.39911, 1.3312, 1.5115, 1.3531),
    "s003_adamw_e3_r4_b250M": (1.16589, 1.39750, 1.3288, 1.5118, 1.3502),
    "s003_adamw_e3_r8_b250M": (1.16771, 1.39993, 1.3307, 1.5155, 1.3520),
}
R = ["0.5", "1", "2", "4", "8"]
SHORT = {   # CHAT, PROSE at the 125M and 62.5M branches, r 0.5 / 1 / 2 / 4 / 8 (reported, no rule)
    "b125M": [(1.23815, 1.46098), (1.23343, 1.45233), (1.21782, 1.44858), (1.21898, 1.44329), (1.21628, 1.44457)],
    "b62M": [(1.34294, 1.55257), (1.32681, 1.53543), (1.30704, 1.52087), (1.29834, 1.50888), (1.28942, 1.50755)],
}
POINTS = [("1.5", 0.5), ("6", 2.0)]                 # stage C: (eta x 1e3, g) at r_B 2
TAGS = (("b62M", 1526), ("b125M", 3052), ("b250M", 6104))
STAGE_C_RUNS = {f"s003_adamw_e{e}_r2_{t}" for e, _ in POINTS for t in ("trunk", "b62M", "b125M", "b250M")}
LATER = STAGE_C_RUNS | {ln.split()[1] for n in ("stage1_seeds", "stage2_select", "stage2_s006x") for ln in open(
    os.path.join(L.HERE, "plans", n + ".txt")) if ln.startswith("train ")}   # measured after that entry: stage C, the
# stage 1 seed sets, the stage 2 selection runs, S006's extension
PLAN = os.path.join(L.HERE, "plans", "stage1_s003C.txt")


def section(head: str) -> str:
    lines = open(os.path.join(L.EXPD, "SCREENS.txt")).read().splitlines()
    i = next(i for i, ln in enumerate(lines) if ln.startswith(head))
    j = next((j for j in range(i, len(lines)) if not lines[j].strip()), len(lines))
    return "\n".join(lines[i:j])


@pytest.fixture(scope="module")
def picks(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    for name, vals in {**STAGE_1A, **STAGE_B}.items():
        os.makedirs(runs / name)
        with open(runs / name / "bpb.jsonl", "w") as f:
            for s, v in zip(SETS, vals):
                f.write(json.dumps({"ckpt": "final_00007630.pt", "split": "all", "set": s, "bpb": v}) + "\n")
    return analyze.picks(str(runs))


def b250(r: str) -> tuple:
    return (STAGE_1A if r == "1" else STAGE_B)[f"s003_adamw_e3_r{r}_b250M"]


def test_recorded_stage_b_picks_r2_inside_the_grid(picks):
    a, b = picks["S003"]["A"], picks["S003"]["B"]
    assert (a["pick"], a["decided"]) == (0.003, True)
    assert (b["pick"], b["argmin"], b["at_edge"], b["guard_moved"], b["extend_with"], b["extensions_used"],
            b["decided"]) == (2.0, 2.0, None, False, [], 0, True)
    assert b["axis"] == [0.5, 1.0, 2.0, 4.0, 8.0] and b["runs"]["1"] == "s003_adamw_e3_r1_b250M"
    prose = {r: b250(r)[1] for r in R}
    assert 0.00115 < prose["2"] / min(prose.values()) - 1 < 0.00116 and min(prose, key=prose.get) == "4"
    assert round(b250("4")[0] - b250("2")[0], 5) == 0.00153            # best minus second, under E3's 0.00905


def test_stage_c_is_g_half_and_2_at_eta_a_and_r_b(picks):
    c = picks["S003"]["C"]
    assert c["ready"] is False and c["missing"] == [0.5, 2.0] and c["axis"] == [0.5, 1.0, 2.0]
    assert c["runs"] == {"0.5": "s003_adamw_e1.5_r2_b250M", "1": "s003_adamw_e3_r2_b250M",
                         "2": "s003_adamw_e6_r2_b250M"}           # g 1 is stage B's pick, reused


def test_shorter_branches_are_reported_not_ruled(picks):
    """125M and 62.5M argmins sit at the r 8 edge; stage B's pick and extension read the 250M branch only."""
    for tag in ("b125M", "b62M"):
        p = e2pick.pick({float(r): {"chat": c, "prose": q} for r, (c, q) in zip(R, SHORT[tag])})
        assert (p["argmin"], p["at_edge"]) == (8.0, "high"), tag
    assert picks["S003"]["B"]["extend_with"] == []


def test_entry_carries_the_recorded_values_and_pick():
    sec = section("S003 STAGE B RESULT")
    for i, r in enumerate(R):
        row = [b250(r)[:2]] + [SHORT[t][i] for t in ("b125M", "b62M")]
        want = f"{r:<5} " + "    ".join(f"{c:.5f} / {q:.5f}" for c, q in row)
        assert f"\n    {want}\n" in sec + "\n", want
    assert "Pick r_B = 2: the 250M argmin" in sec and "extend_with [], decided" in sec
    assert "the 125M argmin is r 8 (1.21628; r 2 1.21782) and the 62.5M\n  argmin r 8 (1.28942)" in sec


def test_entry_lock_hours_are_the_measured_file():
    sec = section("S003 STAGE B RESULT")
    table = {m[1]: float(m[2]) for m in re.finditer(r"(s003_[a-z]+_[^\s|]+)\s+(\d\.\d{3}) (\d\.\d{3})\s+\d+k", sec)}
    runs, _ = H.load()
    want = {f"s003_adamw_e3_r{r}_{t}" for r in ("0.5", "2", "4", "8") for t in ("trunk", "b62M", "b125M", "b250M")}
    assert set(table) == want and want <= set(runs)
    assert all(round(runs[n], 3) == pytest.approx(v, abs=1e-9) for n, v in table.items())
    assert round(sum(runs[n] for n in want), 3) == 1.622 and "Total 1.622 GPU hours measured for stage B" in sec
    before = {n: h for n, h in runs.items() if n not in LATER}           # the file as of that entry
    assert len(before) == 48 and round(sum(before.values()), 3) == 7.105 and "48 runs, 7.105 h" in sec


def plan_lines():
    return [ln.rstrip("\n") for ln in open(PLAN)]


def test_stage_c_plan_is_exactly_the_missing_points_then_a_mark(picks):
    lines = plan_lines()
    assert lines[0].startswith("# ") and lines[1] == "wait_mark SCREENS SCREENS S003 STAGE B DONE"
    want = [f"train s003_adamw_e{e}_r2_{t}" for e, _ in POINTS for t in ("trunk", "b62M", "b125M", "b250M")]
    assert lines[2:-1] == want and lines[-1] == "mark SCREENS S003 STAGE C DONE"   # the seed sets need this pick
    c = picks["S003"]["C"]
    assert [c["runs"][L.num(x)] for x in c["missing"]] == [ln.split()[1] for ln in want if ln.endswith("_b250M")]
    for ln in want:
        assert len(L.find(ln.split()[1])) == 1
    # the wait line's mark is the one the stage B plan wrote (queue_screens.sh slug: spaces to underscores)
    stage_b = open(os.path.join(L.HERE, "plans", "stage1_s003B.txt")).read().splitlines()
    assert stage_b[-1] == "mark " + lines[1].split(" ", 2)[2]


@pytest.mark.parametrize("e,g", POINTS)
def test_stage_c_configs_carry_g_times_the_stage_b_lrs_and_their_own_trunk(e, g):
    trunk = L.flat(L.resolve(L.find(f"s003_adamw_e{e}_r2_trunk")[0]))
    assert [trunk[k] for k in L.LRK] == pytest.approx([g * 0.003, g * 0.006, g * 0.006], rel=1e-12)
    assert (trunk["optim.kind"], trunk["schedule.mode"], trunk["seed"], trunk["train.total_steps"]) == \
        ("adamw", "trunk", 1, 6105) and trunk["schedule.branch_points"] == [1526, 3052, 6104]
    assert L.check(L.find(f"s003_adamw_e{e}_r2_trunk")[0], P) == []
    for tag, at in TAGS:
        path = L.find(f"s003_adamw_e{e}_r2_{tag}")[0]
        f = L.flat(L.resolve(path))
        assert [f[k] for k in L.LRK] == [trunk[k] for k in L.LRK] and f["optim.kind"] == "adamw"
        assert f["schedule.init_from"].endswith(f"/runs/SCREENS/s003_adamw_e{e}_r2_trunk/stable_{at:08d}.pt")
        assert L.check(path, P) == []


def test_cap_check_before_stage_c_and_what_an_extension_does(capsys, tmp_path):
    tsv = tmp_path / "before_c.tsv"                 # measured_hours.tsv as of that entry: the LATER lines not yet in
    tsv.write_text("".join(ln for ln in open(H.TSV) if not (ln.startswith("run\t") and ln.split("\t")[1] in LATER)))
    r = H.report(P, str(tsv), r_b=2)
    out = capsys.readouterr().out
    t, cap = r["tally"], r["cap"]
    assert sum(t["n_queued"].values()) == 42 and t["n_queued"]["S003"] == 10 and cap["cut"] == []
    assert round(cap["total_before"], 3) == 24.857 and "= 24.857 h against 25 h: fits, nothing cut" in out
    arm = sum(H.s003_tag_hours(H.load(str(tsv))[0]).values())
    assert sum(1 for k in t["registered"] if k[:3] == ("S003", "search", "C") and k not in t["seen"]) == 8
    ext = AL.cap_cut({**t["queued"], "S003": t["queued"]["S003"] + arm}, r["measured_runs"] + r["other"])
    assert ext["cut"] == ["S006"] and round(ext["total_before"], 3) == 25.264 and round(ext["total_after"], 3) == 23.426
    sec = section("S003 STAGE B RESULT")
    assert "total 24.857 h <= 25 h: nothing cut, 0.143 h of headroom" in sec and "25.264 h" in sec and "23.426 h" in sec
