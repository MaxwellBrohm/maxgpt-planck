"""SCREENS.txt INTEGRATION 2026-10-10 and the two entries it moved in: AMENDMENT C6-DIAG and AMENDMENT BASE-REPLAY are
in SCREENS.txt once each, byte for byte the pending files they were written in (their sha256), in date order after the
earlier 2026-10-10 entries; C6-DIAG's job table, queue limits, pins, budget floors, steps and test and mutant counts
against the code; INTEGRATION's cap position generated from the inputs the earlier entries' tests check, its launch
names against the scripts, and its own counts. No model, nothing on the PC.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_integration.py
"""
import collections, hashlib, os, re, subprocess, sys  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze_lib as AL, diag_runner as DR, screens_lib as L  # noqa: E401,E402  (diag_runner: harness on the path)
from mutants_c6diag import MUTANTS_C6DIAG  # noqa: E402
from mutants_integ import MUTANTS_INTEG  # noqa: E402
from mutants_vfix import MUTANTS_VFIX  # noqa: E402
from test_screens_configs import SCREENS_TXT  # noqa: E402
from test_screens_replay_entry import numbers  # noqa: E402
DL = DR.DL

MOVED = {"AMENDMENT C6-DIAG (": "3d91e9bffb033e01e61eff7926f47f9fb7916bbfc9e2306841d37d69ae3fa94e",
         "AMENDMENT BASE-REPLAY (": "8deb8466a81f8b123e448b8488a51307a4708442b0b9d7be2609b79de74773ed"}
ORDER = ["AMENDMENT CAP-FLAG (", "DEVIATION 1 (", "ONSET RULE CODE (", "TESTS FOR THE 2026-10-10 ENTRIES (",
         "AMENDMENT C6-DIAG (", "AMENDMENT BASE-REPLAY (", "INTEGRATION 2026-10-10 ("]
TABLE = {"e3p1": (18, 36), "stage2": (10, 38), "basediag": (3, 9), "stage1": (12, 36), "replay": (1, 2), "e3p2": (6, 12)}
SUITE, MUT_ALL, NEW_TESTS = 774, 541, 22       # INTEGRATION's TESTS paragraph


def lines():
    return open(SCREENS_TXT).read().split("\n")


def section(head):
    ls = lines()
    i = next(i for i, ln in enumerate(ls) if ln.startswith(head))
    j = next((j for j in range(i, len(ls)) if not ls[j].strip()), len(ls))
    return "\n".join(ls[i:j]) + "\n"


def flat(head):
    return " ".join(section(head).split())


def test_the_moved_entries_are_the_pending_files_in_date_order():
    for head, sha in MOVED.items():
        assert sum(ln.startswith(head) for ln in lines()) == 1, head
        assert hashlib.sha256(section(head).encode()).hexdigest() == sha, head
    assert "(sha256 {} and {})".format(*(v[:16] for v in MOVED.values())) in flat("INTEGRATION 2026-10-10 (")
    at = [next(i for i, ln in enumerate(lines()) if ln.startswith(h)) for h in ORDER]
    assert at == sorted(at)
    assert not os.path.exists(os.path.join(L.HERE, "pending"))


def test_c6diag_job_table_is_the_runners_registry():
    rows = [ln.split("\t") for ln in subprocess.run([sys.executable, os.path.join(L.HERE, "diag_runner.py"), "list"],
                                                   capture_output=True, text=True, check=True).stdout.splitlines()]
    runs, jobs = collections.defaultdict(set), collections.Counter()
    for g, run, _job, _f in rows:
        runs[g].add(run)
        jobs[g] += 1
    assert {g: (len(runs[g]), jobs[g]) for g in runs} == TABLE and list(runs) == list(DR.GROUPS) == list(TABLE)
    sec = section("AMENDMENT C6-DIAG (")
    for g, (n, j) in TABLE.items():
        assert re.search(rf"^      {g} +{n} +{j} ", sec, re.M), g
    n_loads, n_runs = sum(len(v) for v in runs.values()), len(set().union(*runs.values()))
    assert (n_loads, n_runs, len(rows)) == (50, 49, 133)
    assert f"({n_runs} checkpoints, {n_loads} loads, {len(rows)} jobs;" in flat("AMENDMENT C6-DIAG (")
    assert re.search(rf"^      total +{n_loads} loads +{len(rows)} +about 22 min, 0\.36 GPU h", sec, re.M)


def test_c6diag_queue_limits_pins_floors_and_steps():
    q, f = open(os.path.join(L.HERE, "diag_queue.sh")).read(), flat("AMENDMENT C6-DIAG (")
    for code in ("HOLD=${C6_HOLD_S:-1980}", "BUDGET=${C6_BUDGET_S:-1500}", "LOCKWAIT=${C6_LOCK_WAIT_S:-7200}",
                 "MINFREE=40", '[ "$used" -gt 3000 ]', '[ "$t" -gt 75 ]', 'timeout -k 30 "$HOLD"'):
        assert code in q, code
    for phrase in ("soft budget of 1,500 s under a hard timeout of 1,980 s plus a 30 s kill", "more than 3,000 MiB",
                   "under 40 GB free", "the GPU above 75 C waits", "flock -w 7200 on ~/planck/locks/gpu.lock",
                   f"IND file {DL.IND_SHA[:8]}...", f"tok_v0_8k {DL.TOK_SHA[:8]}...", f"eval sets v0 {DL.EVALSET_SHA[:8]}...",
                   "the 200 lowest-h oasst2 windows", "(7,630 at 5M, 15,259 at 20M)"):
        assert phrase in f, phrase
    assert DL.WINDOWS == 200 and DR.expected("base_s101")[0] == 7630 and DR.expected("e3_20m_A_s1")[0] == 15259
    e = DR.EST
    assert f"floors (hd {e['hd']:.0f} s, attn {e['attn']:.0f} s, ind {e['ind']:.0f} s, halflife {e['halflife']:.0f} s, " \
           f"load {e['load']:.0f} s, x4 at 20M)" in f
    assert (1980 + 30) / 60 == 33.5 and "so no hold passes 33.5 min" in f


def test_c6diag_test_and_mutant_counts():
    f = flat("AMENDMENT C6-DIAG (")
    by = collections.Counter(os.path.basename(m[1]) for m in MUTANTS_C6DIAG)
    assert by == {"diag_runner.py": 27, "diag_lib.py": 17, "diag_queue.sh": 24, "diag_start.sh": 2, "e3analyze_4d.py": 10}
    assert f"{len(MUTANTS_C6DIAG)} of {len(MUTANTS_C6DIAG)} killed" in f and len(MUTANTS_C6DIAG) == 80
    assert all(f"{k} {v}" in f for k, v in by.items() if k != "diag_start.sh") and "and diag_start.sh 2" in f
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider"] +
                         [os.path.join(HERE, t) for t in ("test_diag_runner.py", "test_diag_readings.py",
                                                          "test_diag_queue.py", "test_e3_part4d.py")],
                         capture_output=True, text=True, cwd=HERE).stdout
    n = collections.Counter(ln.split("::")[0].split("/")[-1] for ln in out.splitlines() if "::" in ln)
    assert n == {"test_diag_runner.py": 19, "test_diag_readings.py": 6, "test_diag_queue.py": 16, "test_e3_part4d.py": 5}
    assert all(f"{t} ({k}" in f for t, k in n.items()) and f"{sum(n.values())} passed" in f


def test_integration_cap_position_is_generated():
    v, f = numbers(), flat("INTEGRATION 2026-10-10 (")
    c6 = float(re.search(r"about 22 min, (\d\.\d\d) GPU h", flat("AMENDMENT C6-DIAG ("))[1])
    now = float(v["act"]) - float(v["pb3"])                   # CAP-FLAG's POSITION NOW: 96 runs and OTHER measured
    assert f"{now:.3f}" == "25.987" and "+ 2.120 h OTHER = 25.987 h; queued 0 of 90" in flat("AMENDMENT CAP-FLAG (")
    tot, tot_a = float(v["act"]) + c6, float(v["tot"]) + c6
    chk = AL.cap_check({"BASE": float(v["pb3"]) + c6}, now, cap=28.0)
    assert chk["fits"] and not chk["flag_to_max"] and f"{chk['total']:.3f}" == f"{tot:.3f}" and AL.CAP == 28.0
    assert f"POSITION NOW's {now:.3f} h (96 runs and OTHER measured, 0 queued) + BASE-REPLAY's 3 runs, {v['pb3']} h, " \
           f"+ the C6 queue's estimate, {c6:.2f} h, counted whole as OTHER" in f
    assert f"= {tot:.3f} h <= 28 h: nothing to flag, {28 - tot:.3f} h of headroom" in f
    assert f"rule input (a) the same addition gives {tot_a:.3f} h, {28 - tot_a:.3f} h of headroom" in f
    assert (f"{tot:.3f}", f"{28 - tot:.3f}", f"{tot_a:.3f}", f"{28 - tot_a:.3f}") == ("27.284", "0.716", "27.600", "0.400")


def test_integration_launch_names_are_the_scripts():
    f = flat("INTEGRATION 2026-10-10 (")
    start, rw = (open(os.path.join(L.HERE, n)).read() for n in ("diag_start.sh", "replay_waiter.sh"))
    assert '--bundle planck_c6diag.bundle "$@"' in start and "planck-kit/planck_c6diag.bundle" in f
    assert "stage2_replay planck_replay.bundle" in rw and "planck-kit/planck_replay.bundle" in f
    assert "MAXR=${CW_MAXR:-24}" in rw and "after a 2 h gpu.lock wait, at most 24 times" in f
    assert "each at most 1,980 s + 30 s" in f and "3 runs, one gpu.lock hold each, about 0.31 h" in f
    assert "about 22 min of holds" in f and "about 22 min, 0.36 GPU h" in flat("AMENDMENT C6-DIAG (")
    assert [ln.strip() for ln in open(os.path.join(L.HERE, "plans", "CURRENT")) if not ln.startswith("#")] == \
        ["stage2_replay"]
    assert "(last written 17:33)" in f and "pending/replay.txt (18:04)" in f


def test_integration_counts():
    import mutation_screens as M
    f = flat("INTEGRATION 2026-10-10 (")
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider",
                          os.path.join(HERE, "test_replay_waiter.py"), __file__], capture_output=True, text=True, cwd=HERE).stdout
    n = collections.Counter(ln.split("::")[0].split("/")[-1] for ln in out.splitlines() if "::" in ln)
    assert f"tests/test_replay_waiter.py ({n['test_replay_waiter.py']}:" in f and sum(n.values()) == NEW_TESTS
    assert f"tests/test_screens_integration.py ({n['test_screens_integration.py']}:" in f
    assert f"tests/mutants_integ.py ({len(MUTANTS_INTEG)} mutants" in f
    assert len(M.MUTANTS) == MUT_ALL + len(MUTANTS_VFIX)        # CORRECTIONS AFTER THE W1 VERIFY adds its own
    assert f"experiments/screens/tests {SUITE} passed" in f and f"{MUT_ALL} of {MUT_ALL} killed" in f
    assert f"{len(MUTANTS_C6DIAG)} and tests/mutants_integ.py's {len(MUTANTS_INTEG)} now in it" in f
    by = collections.Counter(m[0].split("_")[0] for m in MUTANTS_INTEG)
    assert by == {"rw": 14, "cw": 1, "s6w": 1, "ig": 8, "c6e": 9} and "(33 mutants: replay_waiter.sh 14, the older " \
        "waiters' grace 2, this entry 8, C6-DIAG's numbers 9)" in f
    assert all(m[1].endswith("replay_waiter.sh") == (m[0][:3] == "rw_") for m in MUTANTS_INTEG)
