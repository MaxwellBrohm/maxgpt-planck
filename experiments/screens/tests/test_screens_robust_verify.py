"""SCREENS.txt AMENDMENT SEEDS-103-105, the corrections made after the independent check (2026-10-12): the C7 line no
longer calls the new-init SD "as registered"; NEW-INIT SD says plainly that keeping sqrt(2) x sigma_seed departs from
C4's new-init row (quoted from C4 here) and leaves the choice to Max before the reading code exists; and its references
and k = 5 thresholds are computed here from E3's results.json (Part 1) and results_part3.json (Part 3). No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_robust_verify.py
"""
import json, math, os, sys  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import robust_lib as RB  # noqa: E402
import screens_lib as L  # noqa: E402
sys.path.insert(0, L.E3D)
from e3analyze import t_ppf  # noqa: E402
from test_screens_integration import flat, lines  # noqa: E402

P1 = json.load(open(os.path.join(L.E3D, "results.json")))["noise_5M_250M"]
P3 = json.load(open(os.path.join(L.E3D, "results_part3.json")))["rule1"]
NI = "same stream, new init"


def para(head):
    """One indented paragraph of the entry, flattened: from the line starting with head to the next line at its indent."""
    ls = lines()
    i = next(i for i, ln in enumerate(ls) if ln.startswith(RB.HEAD))
    j = next(j for j in range(i, len(ls)) if ls[j].startswith(head))
    ind = len(head) - len(head.lstrip())
    k = next(k for k in range(j + 1, len(ls)) if not ls[k].startswith(" " * (ind + 1)))
    return " ".join(" ".join(ls[j:k]).split())


def test_c4_registers_sigma_init_and_part_3_has_run():
    c4 = " ".join(open(os.path.join(L.ROOT, "experiments", "SCREENS.txt")).read().split())
    assert "SD_ref = sqrt(2) * sigma_init (sqrt(2) * sigma_seed if E3 Part 3 did not run)." in c4
    notes = " ".join(open(os.path.join(L.E3D, "notes.txt")).read().split())
    assert "USE AND SCOPE (as registered): from this entry on, rule 1 reads sqrt(2) x sigma_init" in notes
    assert P3["F(CHAT)"][NI]["df"] == 4


def test_the_c7_line_no_longer_says_as_registered_for_the_new_init_sd():
    c7 = para("    C7 as registered with k = 5:")
    assert "SD_ref by the arm's C4 class as registered" not in c7
    assert "SD_ref by the arm's C4 class (E3 Part 1, df_ref 7: same init SD_d; SIA u x SD_d, u = 1.3533; new init " \
           "sqrt(2) x sigma_seed, which departs from C4's text: see NEW-INIT SD below)" in c7


def test_new_init_sd_says_the_departure_and_leaves_the_choice_to_max():
    f = para("    NEW-INIT SD (")
    for phrase in ("this is a departure from C4, not C4 as registered",
                   "C4's new-init row reads sqrt(2) x sigma_init once E3 Part 3 has run, and it has",
                   "The choice was made after the k = 2 verdicts were read and before any robustness value exists",
                   "Both readings are reported wherever this SD is used",
                   "Which one carries the k = 5 labels is Max's call, flagged to him, and is fixed before the reading "
                   "code is written", "the SD a failed C4 check re-reads at"):
        assert phrase in f, phrase
    assert "re-reads at the new-init SD" in para("    C4 at k = 5:")      # why a re-read uses this SD too
    assert "\u2014" not in f and "\u2013" not in f


def test_new_init_sd_numbers():
    f = para("    NEW-INIT SD (")
    pc = lambda x: f"{100 * x:.3f}%"  # noqa: E731
    seed = {m: math.sqrt(2) * P1[m]["sigma_seed_rel"] for m in ("F(CHAT)", "F(PROSE)")}
    init = {m: P3[m][NI]["ref_rel"] for m in ("F(CHAT)", "F(PROSE)")}
    t7, t4 = t_ppf(0.975, 7), t_ppf(0.975, 4)
    assert f"sqrt(2) x sigma_seed {pc(seed['F(CHAT)'])} / {pc(seed['F(PROSE)'])}, sqrt(2) x sigma_init " \
           f"{pc(init['F(CHAT)'])} / {pc(init['F(PROSE)'])};" in f
    ts = [pc(seed[m] * t7 / math.sqrt(5)) for m in ("F(CHAT)", "F(PROSE)")]
    ti = [pc(init[m] * t4 / math.sqrt(5)) for m in ("F(CHAT)", "F(PROSE)")]
    assert f"k = 5 thresholds {ts[0]} / {ts[1]} against {ti[0]} / {ti[1]} (t(0.975, 4) for sigma_init)" in f
    c7 = flat(RB.HEAD)
    assert f"new init {ts[0]} / {ts[1]} (at k = 2:" in c7          # the C7 line's own new-init pair is sigma_seed's
