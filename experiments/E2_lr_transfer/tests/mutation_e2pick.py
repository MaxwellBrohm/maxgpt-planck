"""Mutation check for tests/test_e2pick.py (stdlib; Mac or PC): one deliberate bug per scratch copy of e2pick.py;
killed = the test file fails. A pattern not found exactly once is INVALID.
  python tests/mutation_e2pick.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.dirname(HERE)
M = [
    ("ties_to_higher_lr", "key=lambda x: (chat[x], x))", "key=lambda x: (chat[x], -x))"),
    ("no_4_decimal_rounding", "round(points[x][\"chat\"], 4)", "points[x][\"chat\"]"),
    ("guard_threshold_2pct", "> 1.01 * best_prose", "> 1.02 * best_prose"),
    ("guard_inclusive", "> 1.01 * best_prose", ">= 1.01 * best_prose"),
    ("guard_two_steps", "(1 if ok.index(target) > ok.index(arg) else -1)", "(2 if ok.index(target) > ok.index(arg) else -2)"),
    ("guard_away_from_prose", "(1 if ok.index(target) > ok.index(arg) else -1)", "(-1 if ok.index(target) > ok.index(arg) else 1)"),
    ("gap_flag_off", "abs(xs.index(x) - xs.index(arg)) == 1)", "abs(xs.index(x) - xs.index(arg)) == 9)"),
    ("three_extensions", "MAX_EXT = 2", "MAX_EXT = 3"),
    ("vertex_sign", "v = x1 - 0.5 * num_ / den", "v = x1 + 0.5 * num_ / den"),
    ("edge_bound_wrong_side", '"lo": -INF, "hi": lx[0]', '"lo": lx[0], "hi": INF'),
    ("moves_needs_more_than_1", "elif s_lo >= 1 or s_hi <= -1:", "elif s_lo > 1 or s_hi < -1:"),
    ("holds_inclusive", "if -1 < s_lo and s_hi < 1:", "if -1 <= s_lo and s_hi <= 1:"),
    ("stage_c_125_rule_off", 'if p125.get("ready") and p125["argmin"] in (xs[0], xs[-1]):', "if False:"),
    ("diverged_not_inf", 'return {"chat": INF, "prose": INF, "diverged": True}', 'return None'),
    ("reads_rolling_ckpt", 'r["ckpt"].startswith("final_")', 'r["ckpt"].startswith("ckpt_")'),
    ("neighbour_inf_unhandled", "if INF in (y0, y2):", "if False:"),
    ("runner_up_higher_chat", "return min(nb, key=key) if nb else None", "return max(nb, key=key) if nb else None"),
    ("runner_up_no_rounding", "round(float(points[x][\"chat\"]), 4)", "float(points[x][\"chat\"])"),
    ("runner_up_ties_higher_lr", "points[x][\"chat\"] < INF else INF, x)", "points[x][\"chat\"] < INF else INF, -x)"),
    ("runner_up_counts_gap", "isinstance(points[xs[j]], dict)]", "points[xs[j]] is not None]"),
    ("runner_up_lrs_unscaled", "[ru * eta_a, r_b]", "[ru, r_b]"),
    ("grid20_extension_off", "if edge is not None and len(g20)", "if False and len(g20)"),
    ("grid20_cap_3", "\"20m\"]) < MAX_EXT else []}", "\"20m\"]) < 3 else []}"),
    ("grid20_edge_wrong_side", "edge = etas[0] / 2 if arg == etas[0] else etas[-1] * 2",
     "edge = etas[-1] * 2 if arg == etas[0] else etas[0] / 2"),
    ("grid20_runner_of_wrong_point", "ru = runner_up(p20, arg)", "ru = runner_up(p20, etas[1])"),
]


def main():
    bad = 0
    for name, old, new in M:
        with tempfile.TemporaryDirectory() as d:
            dst = os.path.join(d, "E2")
            shutil.copytree(E2, dst, ignore=shutil.ignore_patterns("__pycache__"))
            p = os.path.join(dst, "e2pick.py")
            s = open(p).read()
            if s.count(old) != 1:
                res = f"INVALID ({s.count(old)})"
            else:
                open(p, "w").write(s.replace(old, new))
                r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                                    os.path.join(dst, "tests", "test_e2pick.py")], capture_output=True, text=True)
                res = {1: "killed", 0: "SURVIVED"}.get(r.returncode, f"INVALID rc {r.returncode}")
        bad += res != "killed"
        print(f"{name:28s} {res}", flush=True)
    print(f"{len(M) - bad}/{len(M)} killed")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
