"""SCREENS.txt AMENDMENT SEEDS-103-105 (ROBUSTNESS), 2026-10-11: the entry's place, dating and words; the registered
verdicts it says stand and the picks it runs, against stage2_verdicts.json; the configs and plan as screens.py robust
writes them (run again here: nothing may change); plans/CURRENT; the acceptance (robust_lib.robust_configs) refusing
bad states; and every threshold, factor, count and hour the entry states, computed from E3's results and the code.
No model, nothing on the PC.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_robust.py
"""
import json, math, os, shutil, sys  # noqa: E401

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import diag_runner as DR  # noqa: E402
import privacy  # noqa: E402
import robust_lib as RB  # noqa: E402
import screens  # noqa: E402
import screens_hours as H  # noqa: E402
import screens_lib as L  # noqa: E402
sys.path.insert(0, L.E3D)
from e3analyze import chi2_ppf, t_ppf  # noqa: E402
from test_screens_integration import flat, lines  # noqa: E402

HEAD = RB.HEAD
V = json.load(open(os.path.join(L.HERE, "stage2_verdicts.json")))
E3 = json.load(open(os.path.join(L.E3D, "results.json")))


def f3(x, n=3):
    return f"{x:.{n}f}"


def test_place_dating_and_words():
    ls = lines()
    i = next(i for i, ln in enumerate(ls) if ln.startswith(HEAD))
    cap = next(i for i, ln in enumerate(ls) if ln.startswith("AMENDMENT CAP-REMOVED ("))
    assert cap < i and ls[i - 1] == "" and sum(ln.startswith(HEAD) for ln in ls) == 1
    f = flat(HEAD)
    assert f.startswith("AMENDMENT SEEDS-103-105 (ROBUSTNESS) (2026-10-11 23:00 EDT, Claude, the SCREENS robustness "
                        "workflow, on Max's quality rule below;")
    assert "Written AFTER the stage 1 verdicts (STAGE 1 VERDICTS, 2026-10-06) and the stage 2 verdicts (STAGE 2 " \
           "VERDICTS, 2026-10-11, ede95ce) were read, and after the values of base_s103 to base_s105 were read there" in f
    assert "STAGE 1 VERDICTS (2026-10-06" in "\n".join(ls) and "STAGE 2 VERDICTS (2026-10-11" in "\n".join(ls)
    q = "remember we prefer quality over speed 100 percent of the time ok?"
    assert f'verbatim (AMENDMENT CAP-REMOVED; PLAN.md b790d85): "{q}"' in f and q in flat("AMENDMENT CAP-REMOVED (")
    plan = " ".join(open(os.path.join(L.ROOT, "PLAN.md")).read().split())
    w = "Where a choice trades thoroughness for time (more seeds, more checks, longer runs, a rerun to settle a doubt), " \
        "the thorough option wins."
    assert w in plan and f'PLAN.md: "{w}"' in f
    for phrase in ("it cannot change them: it adds a reading beside each, and nothing is adopted or dropped on that "
                   "reading alone", "never in place of it", "Nothing is adopted or dropped on the robustness reading "
                   "alone: the k = 2 verdicts and their consequences stand, and each disagreement goes to Max as a flag",
                   "it is not an independent replication; this part is", "DISAGREEMENT, REPORTED PLAINLY IN BOTH "
                   "DIRECTIONS"):
        assert phrase in f, phrase
    sec = "\n".join(ls[i:next(j for j in range(i, len(ls)) if not ls[j].strip())])
    assert "\u2014" not in sec and "\u2013" not in sec and not privacy.hits(sec)


def test_what_stands_is_the_record_and_the_picks_are_the_recorded_ones():
    c = V["contrasts"]
    name = {"S001.nogate": "S001", "S002.novres": "S002 novres", "S002.noqknorm": "noqknorm", "S002.nonormscale":
            "nonormscale", "S003.adamw": "S003", "S004.canonac": "S004", "S005.forget": "S005", "S006.mtp": "S006",
            "S007.smear": "S007"}
    f = flat(HEAD)
    parts = [f"{name[k]} {c[k]['verdict']}" + (" flagged" if k == "S005.forget" else "") for k in name]
    assert "WHAT STANDS: the registered k = 2 verdicts (" + ", ".join(parts) + ")" in f
    picks = {k: v["pick"] for k, v in V["picks"].items() if k != "S003"}
    assert screens.ROBUST_PICKS == picks and all(v["decided"] for k, v in V["picks"].items() if k != "S003")
    assert list(screens.ROBUST_S003) == V["picks"]["S003"]["C"]["lrs"]["pick"] == [0.003, 2.0]
    assert screens.ROBUST_SEEDS == RB.SEEDS == (103, 104, 105)
    assert "S006 mtp g 2 plus its C6 matched-LR IND-only runs at g 1" in f and "S003 adamw at LR5_adamw (eta 3e-3, r 2)" in f


def test_the_generator_writes_exactly_these_files(capsys):
    plan = RB.ROBUST_PLAN
    before = {p: L.sha256(p) for p in [plan] + [L.find(r)[0] for r in RB.SEQUENCE]}
    assert screens.main(["robust"]) == 0
    assert {p: L.sha256(p) for p in before} == before and capsys.readouterr().out.strip() == plan
    assert RB.robust_configs() == set(RB.NEW) and len(RB.NEW) == 33 and len(RB.SEQUENCE) == 36
    lines_ = open(plan).read().splitlines()
    assert lines_[1] == RB.WAIT and lines_[-1] == RB.END and [ln.split()[1] for ln in lines_[2:-1]] == RB.SEQUENCE
    rep = open(os.path.join(L.HERE, "plans", "stage2_replay.txt")).read().splitlines()
    assert RB.WAIT == "wait_mark SCREENS " + rep[-1][len("mark "):]


def test_current_names_the_plan():
    txt = open(os.path.join(L.HERE, "plans", "CURRENT")).read()
    cur = [ln.strip() for ln in txt.splitlines() if not ln.startswith("#")]
    assert cur == RB.ROBUST_PLANS
    for phrase in ("SCREENS_BASE_REPLAY_DONE", "SCREENS_ROBUSTNESS_K5_DONE", "planck-kit/planck_robust.bundle",
                   "experiments/screens/robust_waiter.sh", "tests/test_screens_robust_code.py", "77ab43f"):
        assert phrase in txt, phrase
    assert "plans/CURRENT = robustness_k5" in flat(HEAD)


def variant(tmp_path, edit):
    p = tmp_path / "robustness_k5.txt"
    p.write_text(edit(open(RB.ROBUST_PLAN).read()))
    return str(p)


BAD = {
    "two_runs_swapped": lambda t: t.replace("train s004_canonac_g1_s104\ntrain s007_smear_g1_s104\n",
                                            "train s007_smear_g1_s104\ntrain s004_canonac_g1_s104\n"),
    "a_run_dropped": lambda t: t.replace("train s006_mtp_g1_s105\n", ""),
    "an_extra_run": lambda t: t.replace("train s007_smear_g1_s105\n", "train s007_smear_g1_s105\ntrain s007_smear_g1_s102\n"),
    "seed_102_set": lambda t: t.replace("_s103\n", "_s102\n"),
    "wait_on_diag": lambda t: t.replace(RB.WAIT, "wait_mark SCREENS SCREENS BASE DIAG DONE"),
    "no_wait": lambda t: t.replace(RB.WAIT + "\n", ""),
    "other_end_mark": lambda t: t.replace(RB.END, "mark SCREENS ROBUSTNESS DONE"),
    "no_comment": lambda t: t.split("\n", 1)[1],
    "seed_major_broken": lambda t: t.replace("train base_s104\n", "").replace("train base_s105\n", "train base_s105\n"
                                                                                "train base_s104\n"),
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_the_acceptance_refuses_other_plans(tmp_path, name):
    p = variant(tmp_path, BAD[name])
    assert open(p).read() != open(RB.ROBUST_PLAN).read()
    with pytest.raises(AssertionError):
        RB.robust_configs(p, closed=True)


def test_the_acceptance_refuses_other_configs_and_states(tmp_path, monkeypatch):
    same = variant(tmp_path, lambda t: t)
    assert RB.robust_configs(same, closed=True) == set(RB.NEW) and RB.robust_configs(str(tmp_path / "x.txt")) == set()
    with pytest.raises(AssertionError, match="after BATCH 1 CLOSED"):
        RB.robust_configs(same, closed=False)
    with pytest.raises(AssertionError, match="no AMENDMENT SEEDS-103-105 entry"):
        RB.robust_configs(same, closed=True, text=RB.entry_text().replace(RB.RECORD, "  ROBUSTNESS PLAN: other"))
    bad = tmp_path / "cfg"
    os.makedirs(bad)
    real104 = L.find("s004_canonac_g1_s104")[0]
    lr = bad / "s004_canonac_g1_s104.yaml"
    absx = "extends: " + os.path.join(L.HERE, "configs", "screens_base.yaml")    # resolvable from tmp_path
    lr.write_text(open(real104).read().replace("extends: ../../screens/configs/screens_base.yaml", absx)
                  .replace("optim: {lr: 0.003,", "optim: {lr: 0.006,"))
    for s in (103, 105):                          # S004's other robustness configs beside it, unchanged but extends
        x = f"s004_canonac_g1_s{s}"
        (bad / f"{x}.yaml").write_text(open(L.find(x)[0]).read().replace("extends: ../../screens/configs/screens_base.yaml",
                                                                          absx))
    real = {r: L.find(r) for r in RB.SEQUENCE + [RB.SEED_RE.sub("_s101", r) for r in RB.NEW]}   # before any patch
    find = lambda r: [str(bad / f"{r}.yaml")] if r.startswith("s004_canonac_g1_s10") and r[-3:] != "101" and \
        r[-3:] != "102" else real[r]  # noqa: E731
    with pytest.raises(AssertionError, match="config_dir|/cfg/"):
        RB.robust_configs(same, find=find, check=lambda p: [], closed=True)       # in the wrong directory
    real_dir = L.config_dir
    with monkeypatch.context() as m:              # the directory check passed: the key diff alone must refuse it
        m.setattr(L, "config_dir", lambda sid: str(bad) if sid == "S004" else real_dir(sid))
        with pytest.raises(AssertionError, match="s004_canonac_g1_s104"):
            RB.robust_configs(same, find=find, check=lambda p: [], closed=True)   # differs from seed 101 in optim.lr
    with pytest.raises(AssertionError):
        RB.robust_configs(same, find=find, closed=True)                           # and screens.py check refuses it
    with pytest.raises(AssertionError):
        RB.robust_configs(same, check=lambda p: ["refused"], closed=True)
    sx = tmp_path / "cfg2"
    os.makedirs(sx)
    seed = sx / "s004_canonac_g1_s104.yaml"
    shutil.copy(real104, seed)
    seed.write_text(seed.read_text().replace("extends: ../../screens/configs/screens_base.yaml", absx)
                    .replace("seed: 104\n", "seed: 105\n"))
    for s in (103, 105):
        shutil.copy(bad / f"s004_canonac_g1_s{s}.yaml", sx)
    find2 = lambda r: [str(sx / f"{r}.yaml")] if r.startswith("s004_canonac_g1_s10") and r[-3:] != "101" and \
        r[-3:] != "102" else real[r]  # noqa: E731
    with monkeypatch.context() as m:              # only the seed differs from its name: refused
        m.setattr(L, "config_dir", lambda sid: str(sx) if sid == "S004" else real_dir(sid))
        with pytest.raises(AssertionError, match="s004_canonac_g1_s104"):
            RB.robust_configs(same, find=find2, check=lambda p: [], closed=True)


def test_the_entrys_thresholds_and_factors():
    f, n, q = flat(HEAD), E3["noise_5M_250M"], E3["quantiles"]
    t7, t4, u = t_ppf(0.975, 7), t_ppf(0.975, 4), q["ub80_df7"]
    assert f"t(0.975, 7) x SD_ref / sqrt(5) = {t7 / math.sqrt(5):.4f} x SD_ref (t(0.975, 7) = {t7:.4f})" in f
    assert f"u = {u:.4f}" in f
    rel = lambda m, cls, k: {"same init": n[m]["SD_d_rel"], "SIA": u * n[m]["SD_d_rel"],  # noqa: E731
                             "new init": math.sqrt(2) * n[m]["sigma_seed_rel"]}[cls] * t7 / math.sqrt(k)
    pc = lambda x: f"{100 * x:.3f}%"  # noqa: E731
    k5 = ", ".join(f"{c} {pc(rel('F(CHAT)', c, 5))} / {pc(rel('F(PROSE)', c, 5))}" for c in ("same init", "SIA", "new init"))
    k2 = ", ".join(f"{pc(rel('F(CHAT)', c, 2))} / {pc(rel('F(PROSE)', c, 2))}" for c in ("same init", "SIA", "new init"))
    assert f"Relative thresholds, CHAT / PROSE: {k5} (at k = 2: {k2})." in f
    sia = [f"{s}.{a}" for s, x in L.SCREENS.items() if x["cls"] == "SIA" for a in x["arms"]]
    assert len(sia) == 7 and f"pooled SIA over the {len(sia)} SIA arms" in f
    assert f"df 4, limit sqrt(chi2(0.95, 4) / 4) = {math.sqrt(chi2_ppf(0.95, 4) / 4):.3f} x SD_d" in f
    df = 4 * len(sia)
    assert f"df {df}, limit sqrt(chi2(0.95, {df}) / {df}) = {math.sqrt(chi2_ppf(0.95, df) / df):.3f} x u x SD_d" in f
    assert f"C4's checks at df 2 and {2 * len(sia)}" in f
    sig = V["onsets"]["sigma"]["sigma_seed"]
    fac = t4 * math.sqrt(2) / math.sqrt(5)
    assert f"(df 4; {sig:.3f} steps as recorded)" in f and f"= {fac:.3f} x sigma_seed ({fac * sig:.1f} steps at the " \
           "recorded sigma_seed, under one logged interval of 153" in f and fac * sig < 153
    kn = L.params()["k"]
    assert set(kn.values()) == {2} and "k = 5 is at or above every class's k_needed(1%) (FIXED: 1 or 2)" in f
    p3 = os.path.join(L.E3D, "results_part3.json")
    assert f"results_part3.json sha256 {L.sha256(p3)[:16]}..., df 4" in f
    assert f"the 9 contrasts at k = 5" in f and len(V["holm_family"]) == 9


def test_the_entrys_counts_and_hours():
    f = flat(HEAD)
    rob = DR.jobs(["robust"])
    assert f"the {len({r[2] for r in rob})} new runs other than S006's three g 1 IND-only runs" in f and \
        f"halflife on S005's arm: {len(rob)} jobs" in f
    assert RB.RECORD in "\n".join(lines()) and f"wrote {len(RB.NEW)} configs" in f
    runs, _ = H.load(H.TSV)
    names = ["s001_nogate_g1_s101", "s002_novres_g1_s101", "s002_noqknorm_g1_s101", "s002_nonormscale_g1_s101",
             "s003_adamw_e3_r2_s101", "s005_forget_g1_s1", "s004_canonac_g1_s1", "s007_smear_g1_s1", "s006_mtp_g2_s1",
             "s006_mtp_g1_s1"]
    per = sum(runs[x] for x in names) + 0.393
    assert "s005_base_s101's 0.393 h" in f and "s005_base_s101 held the lock" in flat("AMENDMENT S006-REINSTATE (") and \
        "(0.393 h; estimate 0.413 h)" in flat("AMENDMENT S006-REINSTATE (")
    assert f"about {per:.1f} GPU h (S005's arm {runs['s005_forget_g1_s1']:.2f} h the longest), the plan about " \
           f"{3 * per:.1f} h" in f
    assert max(runs[x] for x in names) == runs["s005_forget_g1_s1"]
