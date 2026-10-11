"""SCREENS.txt CORRECTIONS AFTER THE W1 VERIFY (2026-10-10): F3, diag_runner.py's budget scales its floors x4 for a 20M
checkpoint (on a frozen clock, tiny CPU models); F4, C6-DIAG's lock-time figures recomputed from its dry run's own job
times; the entry's place, numbers and counts. F1 and F2 are test_screens_onset_diag.py's.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_verify_fixes.py
"""
import collections, os, re, subprocess, sys, types  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import diag_fixtures as FX  # noqa: E402
import diag_runner as DR  # noqa: E402
from mutants_vfix import MUTANTS_VFIX  # noqa: E402
from test_screens_integration import ORDER, flat, lines, section  # noqa: E402

HEAD = "CORRECTIONS AFTER THE W1 VERIFY ("
R5, R20 = ("E3", "e3_5m_A_s2"), ("E3", "e3_20m_A_s1")


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(DR, "time", types.SimpleNamespace(time=lambda: 1000.0))   # elapsed 0, every job 0 s
    root = FX.root_with(tmp_path, {R5: 1})
    FX.save(root, *R20, FX.model("base", 2, d_model=320, n_heads=5, n_kv_heads=5), step=15259)
    return root, FX.evalset(str(tmp_path / "ev"))


def c6(root, run):
    d = os.path.join(root, "E3", run, "c6")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


@pytest.mark.parametrize("budget,rc,scored", [(279, 3, False), (281, 0, True)])
def test_the_budget_counts_a_20m_checkpoint_at_4x_its_floors(world, capsys, budget, rc, scored):
    root, ev = world
    e = DR.EST
    need = 4 * (e["load"] + e["ind"] + e["hd"])
    assert need == 280 and DR.expected(R20[1])[:2] == (15259, 320)
    assert FX.cli(root, ev, "--budget-s", str(budget), only=[R5[1], R20[1]]) == rc
    assert c6(root, R5[1]) == ["hd.json", "ind.json"]
    assert c6(root, R20[1]) == (["hd.json", "ind.json"] if scored else [])
    out = capsys.readouterr().out
    assert ("budget: 0 s spent, e3_20m_A_s1 needs about 280 s: stopping" in out) == (not scored)


def lock_seconds():
    """C6-DIAG's dry run: a 19 s hold for all five jobs of one S005 arm checkpoint (hd 15.7, attn 0.7, halflife 0.4,
    ind 0.3, the rest start-up, load and sha256); every other checkpoint at that rate, by its jobs."""
    hd, attn, half, ind = 15.7, 0.7, 0.4, 0.3
    rest = 19 - (hd + attn + half + ind)
    per = {"ind hd": rest + ind + hd, "+attn": rest + ind + hd + attn, "+half": rest + ind + hd + attn + half}
    grp = {"e3p1": 18 * per["ind hd"], "stage2": 2 * per["+half"] + 8 * per["+attn"], "basediag": 3 * per["+attn"],
           "stage1": 12 * per["+attn"], "replay": per["ind hd"], "e3p2": 6 * per["ind hd"]}
    return rest, per, grp


def test_f4_c6diag_lock_time_recomputed_from_its_dry_run():
    rest, per, grp = lock_seconds()
    rows = [(g, r, j) for g, _, r, j, _ in DR.jobs()]
    gpu = collections.defaultdict(set)
    for g, r, j in rows:
        if j != "modules":
            gpu[(g, r)].add(j)
    kinds = collections.Counter((g, len(js)) for (g, r), js in gpu.items())
    assert kinds == {("e3p1", 2): 18, ("stage2", 4): 2, ("stage2", 3): 8, ("basediag", 3): 3, ("stage1", 3): 12,
                     ("replay", 2): 1, ("e3p2", 2): 6}
    tot = sum(grp.values())
    e3p2_5x = 6 * (rest + 0.3 + 5 * 15.7)
    big = tot - grp["e3p2"] + e3p2_5x
    f = flat(HEAD)
    assert f"{rest:.1f} s of start-up, load and sha256 in a 19 s hold" in f
    assert f"per checkpoint, ind and hd {per['ind hd']:.1f} s, with attn {per['+attn']:.1f} s, with halflife " \
           f"{per['+half']:.1f} s" in f
    assert ", ".join(f"{g} {v:.1f} s" for g, v in grp.items() if g != "e3p2") in f and f"e3p2 {grp['e3p2']:.1f} s:" in f
    assert f"{tot:.1f} s, about {tot / 3600:.2f} h" in f and (f"{tot:.1f}", f"{tot / 3600:.2f}") == ("913.3", "0.25")
    no_rest = tot - 50 * rest
    assert f"({no_rest:.1f} s, {no_rest / 3600:.2f} h, without the start-up)" in f and f"{no_rest:.1f}" == "818.3"
    assert f"({e3p2_5x / 6:.1f} s a checkpoint) the total is {big:,.1f} s, {big / 3600:.2f} h, about " \
           f"{round(big / 60)} min" in f and (f"{big:,.1f}", f"{big / 3600:.2f}") == ("1,290.1", "0.36")
    assert "(0.22 h at the measured rate)" in flat("AMENDMENT C6-DIAG (")     # the moved entry is not edited


def test_the_entry_follows_integration_and_its_f5_numbers():
    at = [next(i for i, ln in enumerate(lines()) if ln.startswith(h)) for h in ORDER + [HEAD]]
    assert at == sorted(at) and sum(ln.startswith(HEAD) for ln in lines()) == 1
    f = flat(HEAD)
    assert "(\"774 passed\", \"541 of 541 killed\") was written at 19:04" in f
    assert "the three mutation shards 19:09:14 to 19:25:30 (100 + 221 + 220 = 541 killed)" in f and 100 + 221 + 220 == 541
    assert "CORRECTION 1 TO ONSET RULE CODE (a)" in f and "every 153 steps below 7,630, and 7,630: 50 steps" in f


def test_the_entry_counts():
    import mutation_screens as M
    f = flat(HEAD)
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider",
                          os.path.join(HERE, "test_screens_onset_diag.py"), __file__], capture_output=True, text=True,
                         cwd=HERE).stdout
    n = collections.Counter(ln.split("::")[0].split("/")[-1] for ln in out.splitlines() if "::" in ln)
    assert f"tests/test_screens_onset_diag.py ({n['test_screens_onset_diag.py']}:" in f
    assert f"tests/test_screens_verify_fixes.py ({n['test_screens_verify_fixes.py']}:" in f
    assert all(m in M.MUTANTS for m in MUTANTS_VFIX)
    by = collections.Counter(os.path.basename(m[1]) for m in MUTANTS_VFIX)
    assert all(f"{k} {v}" in f for k, v in by.items()), by
    assert f"tests/mutants_vfix.py ({len(MUTANTS_VFIX)} mutants" in f


SUITE_AT_ENTRY, MUT_AT_ENTRY = 806, 565      # the suite's tests and mutation_screens.py's list when RUNS was written


def test_the_entry_results_sentence_matches_the_suite_and_the_mutant_list():
    """Written after its runs (F5): passed + failed is the suite's size then (774 at INTEGRATION + this entry's 32), the
    one failure is the test that wants that sentence, and the mutation count is the whole list then. Later steps add
    tests and mutants (2026-10-10: test_registry_c6_inputs.py), so the live counts may only be larger."""
    import mutation_screens as M
    f = flat(HEAD)
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider", HERE],
                         capture_output=True, text=True, cwd=HERE).stdout
    total = sum(1 for ln in out.splitlines() if "::" in ln)
    m = re.search(r"experiments/screens/tests (\d+) passed and (\d+) failed, the failure being "
                  r"test_the_entry_results_sentence_matches_the_suite_and_the_mutant_list", f)
    assert m and int(m[1]) + int(m[2]) == SUITE_AT_ENTRY == 774 + 26 + 6 and int(m[2]) == 1 and total >= SUITE_AT_ENTRY
    assert f"mutation_screens.py {MUT_AT_ENTRY} of {MUT_AT_ENTRY} killed" in f and MUT_AT_ENTRY == 541 + 24
    assert len(M.MUTANTS) >= MUT_AT_ENTRY
