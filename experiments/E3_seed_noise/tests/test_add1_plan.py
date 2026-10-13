"""E3 Part 1b and Part 3 at 20M (notes.txt ADDENDUM 2026-10-12): the 19 configs against Part 1's A1 and Part 2's A20 s1,
plans/part1b_p3at20m.txt and CURRENT, e3plan.py writing them byte for byte with every earlier file unchanged, the pins in
e3add1_lib.py and the waiter, the entry's hashes and statements against the files, and PART 1b PARITY's log. No model,
nothing on the PC. E3_TEST_ROOT (tests/mutation_add1.py) points at a scratch copy of the repo.
  ~/.venvs/planck/bin/python -m pytest -q experiments/E3_seed_noise/tests/test_add1_plan.py
"""
import glob
import hashlib
import importlib.util
import os
import re
import subprocess
import sys

ROOT = os.environ.get("E3_TEST_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
E3 = os.path.join(ROOT, "experiments", "E3_seed_noise")
CONF, PLANS = os.path.join(E3, "configs"), os.path.join(E3, "plans")
P1B = [f"e3_5m_A_s{s}" for s in range(12, 22)]
P20 = ["e3_20m_A_init2", "e3_20m_A_data2", "e3_20m_A_init3", "e3_20m_A_data3", "e3_20m_A_s1_P3", "e3_20m_A_init4",
       "e3_20m_A_data4", "e3_20m_A_init5", "e3_20m_A_data5"]
SEEDS20 = {"e3_20m_A_s1_P3": (1, 1), **{f"e3_20m_A_init{k}": (k, 1) for k in range(2, 6)}, **{f"e3_20m_A_data{k}": (1, k) for k in range(2, 6)}}
# sha256 over "<file> <sha256>" lines of the 33 configs and part1/2/3.txt that existed before this addendum (2026-10-12)
PIN_EARLIER = "d4e11cdf8a923da874a28001686ae958f601ebd44fa13298982d81e828354699"
REGISTERED = {"e3analyze.py": "8392762a3552422ba05df1f01fb38129bed69f458e4d73e1a633b3d1aeac34e3",
              "e3analyze_p2.py": "a657920af9576654198c13d6106930102ff47ffd8d39be493b0cf505423ed406",
              "e3analyze_p3.py": "770f1f877bf8334ca90a975de43a64c5a25a5e6cef44d8d0ccfc903976ad502f"}


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def text(name): return open(os.path.join(CONF, name + ".yaml")).read()


def lib():
    spec = importlib.util.spec_from_file_location("e3t_add1_lib", os.path.join(E3, "e3add1_lib.py"))
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, E3)
    spec.loader.exec_module(m)
    return m


def entry():
    t = open(os.path.join(E3, "notes.txt")).read()
    return t[t.index("\nADDENDUM 2026-10-12: PART 1b AND PART 3 AT 20M"):]


def test_part1b_configs_are_part1s_arm_a_with_their_seed():
    a1 = text("e3_5m_A_s1").splitlines(keepends=True)
    assert a1[0] == "# E3 5M arm A (eta 0.003, r 1), seed 1 (e3plan.py)\n" and a1[4] == "seed: 1\n" and len(a1) == 8
    for n in P1B:
        s = int(n.rsplit("_s", 1)[1])
        want = [f"# E3 5M arm A (eta 0.003, r 1), seed {s}, Part 1b (e3plan.py)\n", a1[1], f"name: {n}\n",
                f"out_dir: ../../../planck_root/runs/E3/{n}\n", f"seed: {s}\n"] + a1[5:]
        assert text(n) == "".join(want), n


def test_20m_configs_are_a20_s1_with_their_seeds():
    a = text("e3_20m_A_s1").splitlines(keepends=True)
    assert a[0] == "# E3 20M arm A (eta 0.003, r 1), seed 1 (e3plan.py)\n" and a[4] == "seed: 1\n" and len(a) == 8
    for n, (s, d) in SEEDS20.items():
        note = ", A20 s1 again at this part's commit" if n == "e3_20m_A_s1_P3" else ""
        want = [f"# E3 20M arm A (eta 0.003, r 1), seed {s}, data_seed {d}{note} (e3plan.py)\n", a[1], f"name: {n}\n",
                f"out_dir: ../../../planck_root/runs/E3/{n}\n", f"seed: {s}\n", f"data_seed: {d}\n"] + a[5:]
        assert text(n) == "".join(want), n
    assert sorted(s for n, (s, d) in SEEDS20.items() if d == 1) == [1, 2, 3, 4, 5]
    assert sorted(d for n, (s, d) in SEEDS20.items() if s == 1) == [1, 2, 3, 4, 5]


def test_plan_and_current():
    lines = open(os.path.join(PLANS, "part1b_p3at20m.txt")).read().splitlines()
    assert lines[:3] == ["# E3 Part 1b, then Part 3 at 20M (e3plan.py; notes.txt ADDENDUM 2026-10-12)",
                         "wait_mark E3 E3 PART 3 DONE", "wait_mark SCREENS SCREENS ROBUSTNESS K5 DONE"]
    assert lines[3:] == [f"train {n}" for n in P1B] + ["mark E3 PART 1B DONE"] + [f"train {n}" for n in P20] + [
        "mark E3 PART 3 AT 20M DONE"]
    assert P20.index("e3_20m_A_s1_P3") == len(P20) // 2 and all(
        P20.index(f"e3_20m_A_data{k}") == P20.index(f"e3_20m_A_init{k}") + 1 for k in range(2, 6))
    assert [ln.strip() for ln in open(os.path.join(PLANS, "CURRENT")) if not ln.startswith("#")] == ["part1b_p3at20m"]


def test_e3plan_writes_them_and_every_earlier_file_is_unchanged(tmp_path):
    r = subprocess.run([sys.executable, os.path.join(E3, "e3plan.py"), "--a5", "0.003,1", "--b5", "0.0015,1", "--a20",
                        "0.003,1", "--b20", "0.0015,1", "--part3", "--addendum1", "--out-dir", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for n in P1B + P20:
        assert (tmp_path / f"{n}.yaml").read_text() == text(n), n
    assert (tmp_path / "part1b_p3at20m.txt").read_text() == open(os.path.join(PLANS, "part1b_p3at20m.txt")).read()
    new = set(P1B + P20)
    earlier = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CONF, "e3_*.yaml")) if os.path.basename(p)[:-5] not in new)
    assert len(earlier) == 33
    rows = "".join(f"{n} {sha(os.path.join(CONF, n))}\n" for n in earlier)
    rows += "".join(f"{f} {sha(os.path.join(PLANS, f))}\n" for f in ("part1.txt", "part2.txt", "part3.txt"))
    assert hashlib.sha256(rows.encode()).hexdigest() == PIN_EARLIER


def test_pins_in_the_lib_and_the_waiter():
    L = lib()
    assert {n: sha(os.path.join(CONF, n + ".yaml")) for n in P1B + P20} == L.CONFIG_SHA and L.P1B == P1B and L.P20 == P20
    assert L.PLAN_SHA == sha(os.path.join(PLANS, L.PLAN)) and L.PLAN == "part1b_p3at20m.txt"
    assert all(sha(os.path.join(E3, f)) == h for f, h in L.PINNED.items()) and len(L.PINNED) == 7
    assert all(sha(os.path.join(E3, f)) == h for f, h in REGISTERED.items())       # imported, unchanged
    w = open(os.path.join(E3, "e3p1b_waiter.sh")).read()
    assert f"PLAN_SHA={L.PLAN_SHA}\n" in w and "PLAN=part1b_p3at20m\n" in w


def test_the_entry_records_the_files():
    e = entry()
    f = " ".join(e.split())
    for n in P1B + P20:
        assert f"{sha(os.path.join(CONF, n + '.yaml'))} {n}.yaml" in e, n
    assert sha(os.path.join(PLANS, "part1b_p3at20m.txt")) in f
    for s in ("Written before any Part 1b or Part 3 at 20M value exists", "seeds 12 to 21",
              "rule 5 reserves 9, 10 and 11 for a refresh", "sigma_A = sqrt((SS_1 + SS_1b) / 16)",
              "only a verdict read after PART 1b RESULT is recorded", "RULE 4 AT 20M applies only to verdicts read after PART 3 "
              "AT 20M RESULT is recorded", "No verdict read before it (SCREENS stages 1 and 2, the robustness stage, any E2 or E3 "
              "reading) is re-read", "init2 data2 init3 data3 s1_P3 init4 data4 init5 data5", "QUEUE_MIN_FREE_GB=41",
              "with fewer than 6 runs left Part 1b gives no SD", "with fewer than 4 runs left the family gives no SD"):
        assert s in f, s
    t = open(os.path.join(E3, "notes.txt")).read()
    assert t.index("\nPART 3 CUDA PARITY (2026-10-11") < t.index("\nADDENDUM 2026-10-12") < t.index("\nPART 1b PARITY (2026-10-12")
    assert chr(0x2014) not in e and chr(0x2013) not in e


def test_parity_record():
    e = " ".join(entry().split())
    d = os.path.join(E3, "logs", "parity_p1b")
    log = open(os.path.join(d, "e3p1b_parity_cuda.log")).read()
    assert f"Log logs/parity_p1b/e3p1b_parity_cuda.log, sha256 {sha(os.path.join(d, 'e3p1b_parity_cuda.log'))}" in e
    sums = dict(reversed(ln.split("  ")) for ln in open(os.path.join(d, "p1bpar.sha256")).read().splitlines())
    assert f"p1bpar_ref.tgz {sums['p1bpar_ref.tgz'][:16]}..." in e and f"p1bpar_new.tgz {sums['p1bpar_new.tgz'][:16]}..." in e
    a = log[log.index("run A:"):log.index("run A rc")]
    assert [ln for ln in a.splitlines() if ln.startswith("[default_parity] ") and ":" in ln[17:]][1:5] == [
        "[default_parity] log: identical", "[default_parity] checkpoint: identical", "[default_parity] start: identical",
        "[default_parity] new_imports: ['durable']"]
    assert "[default_parity] cuda: 24 log records, final loss 3.07388 vs 3.07388" in a
    assert "final loss 3.07388 vs 2.93466" in log[log.index("run B:"):log.index("run B rc")]
    assert "[default_parity] PASS" in log[log.index("run C:"):log.index("run C rc")]
    res = [ln.split(" ", 2)[2] for ln in log.splitlines() if " RESULT " in ln or ln.endswith(" DONE ok")]
    assert res == ["RESULT A rc 1: log, checkpoint and start record identical; new imports ['durable'] only (D4), as wanted",
                   "RESULT B rc 1 FAIL as wanted (log and checkpoint differ: mutant killed)",
                   "RESULT C rc 0 PASS as wanted (one tree, two processes)", "DONE ok"]
    m = re.search(r"ref commit ([0-9a-f]{40}); new commit ([0-9a-f]{40})", log)
    assert m[1].startswith("e39112a") and m[2].startswith("6f478ef")
    assert "train.py sha256 805bf5f85cdfb07c (ref) and aa07b64c7a4e3089 (new)" in e and "tree ref train.py sha256 805bf5f85cdfb07c" in log
    assert "gpu.lock held (GPU 41 C, 650 MiB in use)" in log and "hold 21:14:52 to 21:15:16" in e


def test_no_private_identifier():
    pat = re.compile(r"/(?:home|Users)/[A-Za-z]|\b\d{1,3}(?:\.\d{1,3}){3}\b|[\w.-]+@[\w-]+\.(?:com|net|org|edu)")
    files = [os.path.join(CONF, n + ".yaml") for n in P1B + P20] + [os.path.join(PLANS, f) for f in ("part1b_p3at20m.txt", "CURRENT")]
    files += [os.path.join(E3, f) for f in ("e3p1b_waiter.sh", "e3plan.py", "e3add1_lib.py", "e3analyze_p1b.py", "e3analyze_p3_20m.py",
                                            "e3analyze_add1_mutants.py", "parity_p1b_cuda.sh", "logs/parity_p1b/e3p1b_parity_cuda.log")]
    files += glob.glob(os.path.join(E3, "tests", "*add1*.py")) + glob.glob(os.path.join(E3, "tests", "*p1b*.py")) + [
        os.path.join(E3, "tests", "w1bfixture.py")]
    for p in files:
        t = open(p).read()
        assert not pat.search(t), p
        assert chr(0x2014) not in t and chr(0x2013) not in t, p
    assert not pat.search(entry())
