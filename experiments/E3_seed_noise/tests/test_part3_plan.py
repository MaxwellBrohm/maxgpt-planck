"""E3 Part 3 (notes.txt PART 3 REOPENED): its 9 configs against A1's, the two families, plans/part3.txt and CURRENT,
e3plan.py writing every config and plan byte for byte (Parts 1 and 2 unchanged), the harness read that preflight.py
needs, and the entry's hashes and statements against the files. No model, nothing on the PC. E3_TEST_ROOT (mutation_e3.py)
points at a scratch copy of the repo.
  ~/.venvs/planck/bin/python -m pytest -q experiments/E3_seed_noise/tests/test_part3_plan.py
"""
import glob
import hashlib
import importlib.util
import os
import subprocess
import sys

ROOT = os.environ.get("E3_TEST_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
E3 = os.path.join(ROOT, "experiments", "E3_seed_noise")
CONF, PLANS = os.path.join(E3, "configs"), os.path.join(E3, "plans")
INIT = {"e3_5m_A_s1_P3": (1, 1), **{f"e3_5m_A_init{k}": (k, 1) for k in range(2, 6)}}
DATA = {"e3_5m_A_s1_P3": (1, 1), **{f"e3_5m_A_data{k}": (1, k) for k in range(2, 6)}}
ORDER = ["e3_5m_A_init2", "e3_5m_A_data2", "e3_5m_A_init3", "e3_5m_A_data3", "e3_5m_A_s1_P3", "e3_5m_A_init4",
         "e3_5m_A_data4", "e3_5m_A_init5", "e3_5m_A_data5"]
# sha256 over "<file> <sha256>" lines of the 24 Part 1 and 2 configs and part1/part2.txt at ede95ce, before Part 3
PIN_PART12 = "2a948462b90aa99abd510ece35f6df5db436c2433e0a7a91991ba5e875d0f659"


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def text(name: str) -> str:
    return open(os.path.join(CONF, name + ".yaml")).read()


def expected(name: str, seed: int, dseed: int) -> str:
    """configs/e3_5m_A_s1.yaml with only the comment, name, out_dir and seed lines changed and data_seed added."""
    a1 = text("e3_5m_A_s1").splitlines(keepends=True)
    assert a1[0] == "# E3 5M arm A (eta 0.003, r 1), seed 1 (e3plan.py)\n" and a1[4] == "seed: 1\n"
    note = ", A1 again at Part 3's commit" if name == "e3_5m_A_s1_P3" else ""
    out = [f"# E3 5M arm A (eta 0.003, r 1), seed {seed}, data_seed {dseed}{note} (e3plan.py)\n", a1[1],
           f"name: {name}\n", f"out_dir: ../../../planck_root/runs/E3/{name}\n", f"seed: {seed}\n",
           f"data_seed: {dseed}\n"] + a1[5:]
    return "".join(out)


def test_every_part3_config_is_a1_with_its_seeds():
    for name, (seed, dseed) in {**INIT, **DATA}.items():
        assert text(name) == expected(name, seed, dseed), name


def test_the_families():
    assert sorted(s for s, _ in INIT.values()) == [1, 2, 3, 4, 5] and {d for _, d in INIT.values()} == {1}
    assert sorted(d for _, d in DATA.values()) == [1, 2, 3, 4, 5] and {s for s, _ in DATA.values()} == {1}
    assert set(INIT) & set(DATA) == {"e3_5m_A_s1_P3"} and sorted(set(INIT) | set(DATA)) == sorted(ORDER)
    p3 = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(CONF, "e3_5m_A_*.yaml"))
                if "data_seed:" in open(p).read())
    assert p3 == sorted(ORDER)                       # no other config carries a data_seed
    assert "e3_5m_A_s1_P3" != "e3_5m_A_s1" and "runs/E3/e3_5m_A_s1\n" not in text("e3_5m_A_s1_P3")


def test_plan_and_current():
    lines = open(os.path.join(PLANS, "part3.txt")).read().splitlines()
    assert lines[0].startswith("# E3 Part 3") and lines[1] == "wait_mark E3 E3 PART 2 DONE"
    assert lines[2:-1] == [f"train {n}" for n in ORDER] and lines[-1] == "mark E3 PART 3 DONE"
    assert ORDER.index("e3_5m_A_s1_P3") == len(ORDER) // 2           # the shared member at the time centre
    for k in (2, 3, 4, 5):                                             # each pair (init k, data k) adjacent
        assert ORDER.index(f"e3_5m_A_data{k}") == ORDER.index(f"e3_5m_A_init{k}") + 1
    cur = [ln.strip() for ln in open(os.path.join(PLANS, "CURRENT")) if not ln.startswith("#")]
    assert cur == ["part1b_p3at20m"]     # part3 until it ended at its mark; the next plan since ADDENDUM 2026-10-12


def test_e3plan_writes_every_config_and_plan_byte_for_byte(tmp_path):
    r = subprocess.run([sys.executable, os.path.join(E3, "e3plan.py"), "--a5", "0.003,1", "--b5", "0.0015,1",
                        "--a20", "0.003,1", "--b20", "0.0015,1", "--part3", "--addendum1", "--out-dir", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    made = sorted(os.listdir(tmp_path))
    have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CONF, "*.yaml")) if not p.endswith("prereg.yaml"))
    assert made == sorted(have + ["part1.txt", "part1b_p3at20m.txt", "part2.txt", "part3.txt"]) and len(have) == 52
    for f in have:
        assert (tmp_path / f).read_text() == text(f[:-5]), f
    for f in ("part1.txt", "part1b_p3at20m.txt", "part2.txt", "part3.txt"):
        assert (tmp_path / f).read_text() == open(os.path.join(PLANS, f)).read(), f


def test_parts_1_and_2_are_unchanged():
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CONF, "e3_*.yaml"))
                   if "data_seed:" not in open(p).read() and ", Part 1b (" not in open(p).readline())   # Part 1b: 2026-10-12
    assert len(names) == 24
    rows = "".join(f"{n} {sha(os.path.join(CONF, n))}\n" for n in names)
    rows += "".join(f"{f} {sha(os.path.join(PLANS, f))}\n" for f in ("part1.txt", "part2.txt"))
    assert hashlib.sha256(rows.encode()).hexdigest() == PIN_PART12
    notes = open(os.path.join(E3, "notes.txt")).read()
    for n in names:
        if n.startswith("e3_20m_"):
            assert f"{sha(os.path.join(CONF, n))} {n}" in notes, n         # PART 2 QUEUED's list


def test_the_harness_reads_data_seed_for_preflight():
    pf = os.path.join(ROOT, "experiments", "E2_lr_transfer", "preflight.py")
    spec = importlib.util.spec_from_file_location("e3t_preflight", pf)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert "data_seed" in m.harness_reads(os.path.join(ROOT, "harness"))
    assert 'if "data_seed" in cfg and "data_seed" not in reads:' in open(pf).read()


def entry() -> str:
    t = open(os.path.join(E3, "notes.txt")).read()
    return t[t.index("\nPART 3 REOPENED (2026-10-11"):]


def test_the_entry_records_the_files():
    e = entry()
    f = " ".join(e.split())
    for n in ORDER:
        assert f"{sha(os.path.join(CONF, n + '.yaml'))} {n}.yaml" in e, n
    assert sha(os.path.join(PLANS, "part3.txt")) in f
    w = open(os.path.join(E3, "e3p3_waiter.sh")).read()
    assert f"PLAN_SHA={sha(os.path.join(PLANS, 'part3.txt'))}" in w
    assert "never to stage 1 or stage 2 of SCREENS (S001 to S007)" in f
    assert "only to verdicts read after PART 3 RESULT is recorded" in f
    assert "Written before any Part 3 value exists" in f and "e3analyze_p3.py, written and committed before" in f
    assert "init2 data2 init3 data3 s1_P3 init4 data4 init5 data5" in f
    t = open(os.path.join(E3, "notes.txt")).read()
    assert t.index("\nPART 4d RESULT (2026-10-11") < t.index("\nPART 3 REOPENED (2026-10-11")
    assert "\u2014" not in e and "\u2013" not in e


def test_no_private_identifier():
    import re
    pat = re.compile(r"/(?:home|Users)/[A-Za-z]|\b\d{1,3}(?:\.\d{1,3}){3}\b|[\w.-]+@[\w-]+\.(?:com|net|org|edu)")
    for p in [os.path.join(CONF, n + ".yaml") for n in ORDER] + [os.path.join(PLANS, "part3.txt"),
                                                                 os.path.join(PLANS, "CURRENT"),
                                                                 os.path.join(E3, "e3p3_waiter.sh"),
                                                                 os.path.join(E3, "e3plan.py")]:
        assert not pat.search(open(p).read()), p
    assert not pat.search(entry())
