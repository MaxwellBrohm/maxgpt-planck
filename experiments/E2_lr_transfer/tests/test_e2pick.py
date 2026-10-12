"""e2pick.py on synthetic results (stdlib; runs on the Mac: no torch, no model).
  <the Mac test venv, harness/notes.txt "Run"> -m pytest -q experiments/E2_lr_transfer/tests/test_e2pick.py
"""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import e2pick as P  # noqa: E402

INF = float("inf")


def R(chat, prose=1.0):
    return {"chat": chat, "prose": prose}


def write(runs, name, chat=None, prose=1.0, mark=None):
    d = os.path.join(runs, name)
    os.makedirs(d, exist_ok=True)
    if mark:
        open(os.path.join(d, mark), "w").close()
        return
    with open(os.path.join(d, "bpb.jsonl"), "w") as f:
        for ck, v in (("ckpt_00000100.pt", 9.9), ("final_00007630.pt", None)):
            for s, val in (("CHAT", chat), ("PROSE", prose), ("oasst2", chat)):
                f.write(json.dumps({"ckpt": ck, "set": s, "split": "all", "bpb": v if v is not None else val}) + "\n")


def test_argmin_and_ties_at_four_decimals_go_to_the_lower_lr():
    assert P.pick({1: R(2.0), 2: R(1.5), 4: R(1.7)})["pick"] == 2
    t = P.pick({1: R(2.0), 2: R(1.23452), 4: R(1.23448)})
    assert t["pick"] == 2 and not t["guard_moved"]
    assert P.pick({1: R(2.0), 2: R(1.2346), 4: R(1.2344)})["pick"] == 4


def test_prose_guard_moves_one_step_toward_the_prose_best_only_above_one_percent():
    pts = {1: R(1.6, 1.00), 2: R(1.5, 1.05), 4: R(1.4, 1.02), 8: R(1.7, 1.1)}
    p = P.pick(pts)
    assert (p["argmin"], p["pick"], p["guard_moved"]) == (4, 2, True)
    pts[4] = R(1.4, 1.01)                                  # exactly 1% above: stays
    assert P.pick(pts)["pick"] == 4


def test_diverged_scores_inf_and_gap_next_to_argmin_is_flagged():
    p = P.pick({1: R(INF, INF), 2: R(1.5), 4: "gap", 8: R(1.6)})
    assert p["pick"] == 2 and p["gap_next_to_argmin"]
    assert not P.pick({1: "gap", 2: R(1.9), 4: R(1.5), 8: R(1.6)})["gap_next_to_argmin"]
    assert P.pick({1: R(1.0), 2: None})["ready"] is False


def test_edges():
    assert P.pick({1: R(1.0), 2: R(1.5), 4: R(1.7)})["at_edge"] == "low"
    assert P.pick({1: R(2.0), 2: R(1.5), 4: R(1.4)})["at_edge"] == "high"


def test_vertex_on_a_parabola_and_bounds():
    f = {2.0 ** k: R((k - 1.3) ** 2 + 1) for k in range(-2, 4)}
    v = P.vertex(f)
    assert v["v"] == pytest.approx(1.3) and v["lo"] == v["hi"]
    b = P.vertex({1: R(1.0), 2: R(1.5), 4: R(1.7)})
    assert (b["v"], b["lo"], b["hi"]) == (None, -INF, 0.0)
    b = P.vertex({1: R(2.0), 2: R(1.5), 4: R(1.4)})
    assert (b["lo"], b["hi"]) == (2.0, INF)
    b = P.vertex({1: R(INF), 2: R(1.5), 4: R(1.7)})
    assert (b["v"], b["lo"], b["hi"]) == (None, 0.0, 2.0)


def test_transfer_verdicts():
    ex = lambda v: {"lo": v, "hi": v}                     # noqa: E731
    assert P.transfer(ex(-8.0), ex(-7.2))["verdict"] == "HOLDS"
    assert P.transfer(ex(-8.0), ex(-7.0))["verdict"] == "MOVES"          # |s| = 1 moves
    assert P.transfer(ex(-8.0), ex(-9.5))["verdict"] == "MOVES"
    assert P.transfer(ex(-8.0), {"lo": -7.5, "hi": INF})["verdict"] == "UNRESOLVED"
    assert P.transfer(ex(-8.0), {"lo": -6.5, "hi": INF})["verdict"] == "MOVES"
    assert P.transfer(ex(-8.0), ex(-7.5))["s"] == pytest.approx(0.5)


def test_stage_a_from_runs_extension_then_decided(tmp_path):
    runs = str(tmp_path)
    vals = {0.75e-3: 1.9, 1.5e-3: 1.8, 3e-3: 1.7, 6e-3: 1.6, 12e-3: 1.5}
    for eta, v in vals.items():
        write(runs, P.run_name("5m", eta, 1, "b250M"), v)
    s = P.stage(runs, "A")
    assert s["pick"] == 12e-3 and s["extend_with"] == [24e-3] and not s["decided"]
    write(runs, P.run_name("5m", 24e-3, 1, "b250M"), mark=None, chat=1.55)
    s = P.stage(runs, "A")
    assert math.isclose(s["pick"], 12e-3) and s["decided"] and s["extensions_used"] == 1
    assert s["runs"]["0.024"] == "5m_e24_r1_b250M"


def test_stage_extensions_capped_at_two(tmp_path):
    runs = str(tmp_path)
    for k, eta in enumerate([0.75e-3, 1.5e-3, 3e-3, 6e-3, 12e-3, 24e-3, 48e-3]):
        write(runs, P.run_name("5m", eta, 1, "b250M"), 2.0 - 0.1 * k)
    s = P.stage(runs, "A")
    assert s["pick"] == 48e-3 and s["extensions_used"] == 2 and s["extend_with"] == [] and s["decided"]


def test_stage_c_extends_while_its_125m_argmin_is_at_an_edge(tmp_path):
    runs = str(tmp_path)
    for g, v250, v125 in ((0.5, 1.6, 1.70), (1.0, 1.5, 1.72), (2.0, 1.55, 1.75)):
        write(runs, P.run_name("5m", g * 3e-3, 2.0, "b250M"), v250)
        write(runs, P.run_name("5m", g * 3e-3, 2.0, "b125M"), v125)
    s = P.stage(runs, "C", eta_a=3e-3, r_b=2.0)
    assert s["pick"] == 1.0 and s["argmin_125M"] == 0.5 and s["extend_with"] == [0.25] and not s["decided"]


def test_stage_b_reads_the_r_axis(tmp_path):
    runs = str(tmp_path)
    for r, v in ((0.5, 1.7), (1.0, 1.6), (2.0, 1.5), (4.0, 1.52), (8.0, 1.6)):
        write(runs, P.run_name("5m", 3e-3, r, "b250M"), v)
    write(runs, P.run_name("5m", 6e-3, 1.0, "b250M"), 1.0)        # another axis: ignored
    s = P.stage(runs, "B", eta_a=3e-3)
    assert s["pick"] == 2.0 and s["decided"] and s["axis"] == [0.5, 1.0, 2.0, 4.0, 8.0]


def test_read_run_marks(tmp_path):
    runs = str(tmp_path)
    write(runs, "a", mark="DIVERGED")
    write(runs, "b", mark="GAP")
    write(runs, "c", 1.5, 1.2)
    assert P.read_run(runs, "a")["chat"] == INF and P.read_run(runs, "b") == "gap"
    assert P.read_run(runs, "c") == {"chat": 1.5, "prose": 1.2} and P.read_run(runs, "zz") is None


def test_stage_c_runner_up_is_the_lower_neighbour_and_gives_e3_lrs(tmp_path):
    runs = str(tmp_path)
    for g, v in ((0.5, 1.6), (1.0, 1.5), (2.0, 1.55)):
        write(runs, P.run_name("5m", g * 3e-3, 2.0, "b250M"), v)
        write(runs, P.run_name("5m", g * 3e-3, 2.0, "b125M"), 1.7 if g == 1.0 else 1.8)
    s = P.stage(runs, "C", eta_a=3e-3, r_b=2.0)
    assert s["pick"] == 1.0 and s["runner_up"] == 2.0 and s["decided"]
    assert s["lrs"]["pick"] == [3e-3, 2.0] and s["lrs"]["runner_up"] == pytest.approx([6e-3, 2.0])
    assert P.runner_up({1: R(1.60004), 2: R(1.5), 4: R(1.59996)}, 2) == 1  # tie at 4 decimals: the lower LR
    assert P.runner_up({1: "gap", 2: R(1.5), 4: R(1.9)}, 2) == 4           # a GAP neighbour does not count
    assert P.runner_up({1: R(INF, INF), 2: R(1.5), 4: R(1.9)}, 2) == 4     # a diverged one scores +inf


def test_q2_20m_grid_extends_at_an_edge_and_names_b20(tmp_path, capsys):
    runs, eta5 = str(tmp_path), 3e-3
    for g, v in ((0.5, 1.7), (1.0, 1.6), (2.0, 1.65)):
        for br in ("b62M", "b125M", "b250M"):
            write(runs, P.run_name("5m", g * eta5, 1.0, br), v)

    def q2(**pts):
        for g, v in pts.items():
            for br in ("b250M", "b500M"):
                write(runs, P.run_name("20m", float(g[1:].replace("_", ".")) * eta5, 1.0, br), v)
        P.main(["q2", "--runs", runs, "--eta5", "0.003", "--r5", "1"])
        return json.loads(capsys.readouterr().out)["grid_20m"]
    g = q2(g0_5=1.5, g1=1.45, g2=1.4)
    assert g["argmin_g"] == pytest.approx(2.0) and g["extend_with_g"] == [pytest.approx(4.0)]
    assert g["runner_up_g"] == pytest.approx(1.0) and g["extensions_used"] == 0
    g = q2(g4=1.42)
    assert g["argmin_g"] == pytest.approx(2.0) and g["extend_with_g"] == [] and g["runner_up_g"] == pytest.approx(4.0)
    g = q2(g4=1.35)
    assert g["argmin_g"] == pytest.approx(4.0) and g["extend_with_g"] == [pytest.approx(8.0)]
    g = q2(g8=1.3)
    assert g["argmin_g"] == pytest.approx(8.0) and g["extensions_used"] == 2 and g["extend_with_g"] == []
