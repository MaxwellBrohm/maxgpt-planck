"""SCREENS after stage 1 selection A (SCREENS.txt STAGE 1 SELECTION A RESULT, 2026-10-05): the recorded seed-1 values
give the picks the rules read, and the configs and plan written next are exactly what those picks ask for (C3, S003's
E2 search, ORDER 1). Selection values only; no verdict is read from them (C3, E3 change 3). No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_stage1_next.py
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import analyze  # noqa: E402
import screens_lib as L  # noqa: E402

P = L.params()
# final checkpoint, split all, from ~/planck/runs/SCREENS/<run>/bpb.jsonl (copied 2026-10-05; sha256 in SCREENS.txt):
# CHAT, PROSE, cccc, gutenberg, wikimedia
RECORDED = {
    "s001_nogate_g0.5_s1": (1.16307, 1.39242, 1.3244, 1.5042, 1.3471),
    "s001_nogate_g1_s1": (1.15758, 1.38607, 1.3172, 1.5000, 1.3395),
    "s001_nogate_g2_s1": (1.15770, 1.38877, 1.3197, 1.5048, 1.3401),
    "s002_novres_g0.5_s1": (1.17540, 1.39969, 1.3318, 1.5094, 1.3563),
    "s002_novres_g1_s1": (1.16916, 1.39787, 1.3286, 1.5094, 1.3541),
    "s002_novres_g2_s1": (1.17195, 1.40387, 1.3355, 1.5140, 1.3606),
    "s002_noqknorm_g0.5_s1": (1.16212, 1.39728, 1.3262, 1.5137, 1.3502),
    "s002_noqknorm_g1_s1": (1.15240, 1.38838, 1.3176, 1.5047, 1.3411),
    "s002_noqknorm_g2_s1": (1.15428, 1.38775, 1.3183, 1.5041, 1.3392),
    "s002_nonormscale_g0.5_s1": (1.15725, 1.38719, 1.3177, 1.5014, 1.3409),
    "s002_nonormscale_g1_s1": (1.14700, 1.38303, 1.3134, 1.4980, 1.3361),
    "s002_nonormscale_g2_s1": (1.15506, 1.39032, 1.3206, 1.5066, 1.3420),
    "s003_adamw_e0.75_r1_b250M": (1.18025, 1.40702, 1.3394, 1.5196, 1.3604),
    "s003_adamw_e1.5_r1_b250M": (1.18648, 1.40493, 1.3367, 1.5178, 1.3587),
    "s003_adamw_e3_r1_b250M": (1.16891, 1.39837, 1.3296, 1.5132, 1.3508),
    "s003_adamw_e6_r1_b250M": (1.17724, 1.40825, 1.3399, 1.5226, 1.3606),
    "s003_adamw_e12_r1_b250M": (1.17676, 1.40908, 1.3403, 1.5238, 1.3615),
}
STAGE_B_R = ["0.5", "2", "4", "8"]
PLAN = os.path.join(L.HERE, "plans", "stage1_s003B.txt")


@pytest.fixture(scope="module")
def picks(tmp_path_factory):
    runs = tmp_path_factory.mktemp("runs")
    for name, vals in RECORDED.items():
        os.makedirs(runs / name)
        with open(runs / name / "bpb.jsonl", "w") as f:
            for s, v in zip(("CHAT", "PROSE", "cccc", "gutenberg", "wikimedia"), vals):
                f.write(json.dumps({"ckpt": "final_00007630.pt", "split": "all", "set": s, "bpb": v}) + "\n")
    return analyze.picks(str(runs))


def test_recorded_g_checks_pick_g1_inside_the_grid(picks):
    for arm in ("S001.nogate", "S002.novres", "S002.noqknorm", "S002.nonormscale"):
        r = picks[arm]
        assert (r["pick"], r["argmin"], r["at_edge"], r["guard_moved"], r["extend_with"], r["decided"]) == \
            (1.0, 1.0, None, False, [], True), arm
    for arm in ("S004.canonac", "S005.forget", "S006.mtp", "S007.smear"):
        assert picks[arm]["ready"] is False                 # stage 2 has not run


def test_recorded_s003_stage_a_and_what_stage_b_needs(picks):
    a, b = picks["S003"]["A"], picks["S003"]["B"]
    assert (a["pick"], a["at_edge"], a["guard_moved"], a["extend_with"], a["decided"]) == (0.003, None, False, [], True)
    assert "C" not in picks["S003"] and b["ready"] is False
    assert b["missing"] == [float(r) for r in STAGE_B_R]   # r 1 is stage A's pick, reused
    assert b["runs"]["1"] == "s003_adamw_e3_r1_b250M"


def plan_lines():
    return [ln.rstrip("\n") for ln in open(PLAN)]


def test_stage_b_plan_is_exactly_the_missing_points_then_a_mark(picks):
    lines = plan_lines()
    assert lines[0].startswith("# ") and lines[1] == "wait_mark SCREENS SCREENS STAGE 1 SELECTION A DONE"
    want = [f"train s003_adamw_e3_r{r}_{t}" for r in STAGE_B_R for t in ("trunk", "b62M", "b125M", "b250M")]
    assert lines[2:-1] == want and lines[-1] == "mark SCREENS S003 STAGE B DONE"   # nothing runs past the decision
    b = picks["S003"]["B"]
    assert [b["runs"][L.num(x)] for x in b["missing"]] == [ln.split()[1] for ln in want if ln.endswith("_b250M")]
    for ln in want:
        assert len(L.find(ln.split()[1])) == 1


@pytest.mark.parametrize("r", STAGE_B_R)
def test_stage_b_configs_carry_the_stage_b_lrs_and_trunk(r):
    trunk = L.flat(L.resolve(L.find(f"s003_adamw_e3_r{r}_trunk")[0]))
    assert [trunk[k] for k in L.LRK] == pytest.approx([0.003, 0.003 * float(r), 0.003 * float(r)], rel=1e-12)
    assert trunk["optim.kind"] == "adamw" and trunk["schedule.mode"] == "trunk" and trunk["seed"] == 1
    for tag, at in (("b62M", 1526), ("b125M", 3052), ("b250M", 6104)):
        path = L.find(f"s003_adamw_e3_r{r}_{tag}")[0]
        f = L.flat(L.resolve(path))
        assert [f[k] for k in L.LRK] == [trunk[k] for k in L.LRK] and f["optim.kind"] == "adamw"
        assert f["schedule.init_from"].endswith(f"/runs/SCREENS/s003_adamw_e3_r{r}_trunk/stable_{at:08d}.pt")
        assert L.check(path, P) == []
