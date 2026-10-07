"""SCREENS AMENDMENT BASE-DIAG (2026-10-06): Max's words; the cap check at 28 h generated from measured_hours.tsv through
screens_hours and analyze_lib.cap_cut (and at 27 h and 25 h); the record; the 3 BASE configs and the plan as screens.py
writes them, byte for byte; plans/CURRENT; the acceptance (basediag_lib.base_diag_configs) on scratch states; the onset
rule's constants; the notes blocks; every number in the entry pinned. No verdict, no model, no stage 2 result read.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_base_diag.py
"""
import contextlib, io, json, os, re, sys, pytest  # noqa: E401


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze_lib as AL, mutation_screens as M, screens_hours as H, screens_lib as L  # noqa: E401,E402
from basediag_lib import DIAG_PLAN, DIAG_PLANS, END, RUNS, base_diag_configs, recorded_diag  # noqa: E402
sys.path.insert(0, L.E3D)
from e3analyze import t_ppf  # noqa: E402
from test_screens_configs import P  # noqa: E402
from test_screens_refusals import scratch  # noqa: E402,F401  (the scratch config tree fixture)
from test_screens_s003_stage_c import section  # noqa: E402


SEC = section("AMENDMENT BASE-DIAG")
FLAT = " ".join(SEC.split())
MAX = ("have you done that for any previous things that couldve been really helpful but you decided to cut them?? if so "
       "add them back if you think they are good and diserved a chance!")
HEADER = ("SCREENS AMENDMENT BASE-DIAG (SCREENS.txt): BASE at seeds 103, 104, 105 for the IND onset sigma_seed and the BASE "
          "level, after S006's seed sets")
PASSED, WHOLE, NEW, BAD_N = (523, 611), 334, 61, 14      # experiments/screens/tests before and after; mutants whole and new
TIMES = {"entry": "23:00", "read": "22:45", "s2start": "20:38:01", "dry": "22:54"}
HELD = [("20:38:03", "21:01:37"), ("21:01:38", "21:53:14"), ("21:53:15", "22:17:43"), ("22:17:44", "22:36:35")]  # PC log
SEED101 = [("S005", "base"), ("S005", "forget"), ("S004", "canonac"), ("S007", "smear")]


@pytest.fixture(scope="module")
def V():
    with contextlib.redirect_stdout(io.StringIO()):
        r = H.report(P)
    q, mr, mo = r["tally"]["queued"], r["measured_runs"], r["other"]
    p6, pb = H.run_h(L.STEPS, H.mult("S006", "mtp")), H.run_h(L.STEPS, H.mult(None, None))
    q3 = {**q, "S006": q["S006"] + 2 * p6, "BASE": q.get("BASE", 0.0) + 3 * pb}
    c = {cap: AL.cap_cut(q3, mr + mo, cap=cap) for cap in (28.0, 27.0, 25.0)}
    narrow, qs = AL.cap_cut(q3, mr, cap=28.0), sum(q.values())
    room = 28.0 - c[28.0]["total_after"]
    held = sum(H.hours_between("2026-10-06 " + a, "2026-10-06 " + b) for a, b in HELD)
    est = sum(H.run_h(L.STEPS, H.mult(s, a)) for s, a in SEED101)
    t4 = t_ppf(0.975, 4)
    f = lambda x, n=3: f"{x:.{n}f}"  # noqa: E731
    fmt = {"pb": f(pb), "three": f(3 * pb), "mr": f(mr), "mo": f(mo), "m": f(mr + mo), "qs": f(qs), "ind": f(2 * p6),
           "qall": f(sum(q3.values()) - 0.0), "tot": f(c[28.0]["total_before"]), "room": f(room),
           "room27": f(27.0 - c[27.0]["total_after"]), "after25": f(c[25.0]["total_after"]), "narrow": f(narrow["total_before"]),
           "s8": f(qs - q["S006"]), "pct8": f(100 * room / (qs - q["S006"]), 1), "pct10": f(100 * room / qs, 1),
           "held": f(held), "est": f(est), "t4": f(t4), "grid": f(153 / t4, 1), "exch": f(1 / 165)}
    return {"fmt": fmt, "c": c, "room": room, "q3": q3, "t4": t4}


def test_max_verbatim_and_the_decision():
    assert f'Max (the PI), 2026-10-06, verbatim as relayed to this step: "{MAX}"' in FLAT
    for phrase in ("three more shared BASE runs, base_s103, base_s104 and base_s105, each differing from base_s101 only "
                   "in name, out_dir and seed", "The batch's cap moves from 27 h to 28 h, only so these runs leave S006's "
                   "headroom intact (CAP CHECK); its reading is unchanged.", "Decided before any stage 2 onset, HD or "
                   "verdict is read", "BASE at seeds 103 to 105 for any k 3 extension of this batch",
                   "research/CUTS_AUDIT_2026-10.md: rows 3, 12 and 29; REINSTATE NOW, step 1"):
        assert phrase in FLAT, phrase
    assert recorded_diag() == RUNS and SEC.count("\n  BASE RUNS ADDED BY AMENDMENT BASE-DIAG: ") == 1


def test_cap_check_at_28_h(V):
    v, c = V["fmt"], V["c"]
    assert c[28.0]["cut"] == [] == c[27.0]["cut"] and c[25.0]["cut"] == ["S006"] and AL.CAP == 25.0
    assert V["room"] >= 1.634 > 27.0 - c[27.0]["total_after"]          # 27 h would eat S006's recorded headroom
    assert f"Measured {v['mr']} h of runs + {v['mo']} h other = {v['m']} h; queued 10 of 90 registered runs {v['qs']} h " \
           f"+ S006's 2 IND runs {v['ind']} h + the 3 BASE runs {v['three']} h = {v['qall']} h (unrounded); total " \
           f"{v['tot']} h <= 28 h: nothing cut, {v['room']} h of headroom, above S006-REINSTATE's 1.634 h (runs only, " \
           f"reported: {v['narrow']} h). At 27 h the same input fits with {v['room27']} h of headroom; at 25 h it cuts " \
           f"S006 ({v['after25']} h)." in FLAT
    assert "(The audit's 26.302 h and 1.698 h add the rounded 25.366 h and 0.936 h.)" in FLAT
    assert (f"{25.366 + 0.936:.3f}", f"{28 - 26.302:.3f}", f"{3 * 0.312:.3f}") == ("26.302", "1.698", "0.936")
    old = " ".join(section("AMENDMENT S006-REINSTATE").split())
    assert "total 25.366 h <= 27 h: nothing cut, 1.634 h of headroom" in old
    assert f"(3 runs, {v['three']} h)" in FLAT and f"3 runs at {v['pb']} h, {v['three']} h" in FLAT
    assert set(H.load()[0]) >= {"s006_mtp_g4_s1"} and len(H.load()[0]) == 81 and not set(RUNS) & set(H.load()[0])


def test_inside_the_plans(V):
    v = V["fmt"]
    assert f"more than {v['room']} h over their {v['s8']} h estimate ({v['pct8']}%)" in FLAT
    assert f"over their {v['qs']} h estimate ({v['pct10']}%)" in FLAT
    assert f"seed set 101's 4 runs held the lock {v['held']} h against their {v['est']} h estimate, each under its own" in FLAT
    for (a, b), (s, arm) in zip(HELD, SEED101):
        assert H.hours_between("2026-10-06 " + a, "2026-10-06 " + b) < H.run_h(L.STEPS, H.mult(s, arm)), (s, arm)


def test_onset_rule_and_base_level_constants(V):
    v = V["fmt"]
    assert v["t4"] == "2.776" and f"thr = t(0.975, 4) x SD_ref / sqrt(2) = {v['t4']} x sigma_seed" in FLAT
    assert f"when sigma_seed < {v['grid']} steps, one logged interval at both seeds in one direction reads" in FLAT
    assert L.flat(L.resolve(L.find("base_s103")[0]))["eval.induction.every"] == 153
    x = json.load(open(os.path.join(L.EXPD, "S001_attn_gate", "results.json")))["base_level_vs_e3_part1"]["metrics"]
    lo, hi = min(x["CHAT"]["e3_arm_a"]), max(x["CHAT"]["e3_arm_a"])
    assert f"(F(CHAT) {lo:.5f} to {hi:.5f})" in FLAT and len(x["CHAT"]["e3_arm_a"]) == 8
    assert f"probability 1/165, {v['exch']})" in FLAT and 165 == 11 * 10 * 9 // 6
    for phrase in ("EARLIER (C7's BETTER) if -dbar > thr and every d_s < 0; LATER (WORSE) if dbar > thr and every d_s > 0; "
                   "else UNRESOLVED; no EQUAL", "SD_ref = sqrt(2) x sigma_seed", "over seeds 101 to 105, df 4",
                   "S005 pairs with its own mask-engine BASE, so its onset stays descriptive", "changes no verdict",
                   "for n 5 (seeds 101 to 105) and for n 3 (seeds 103 to 105 alone"):
        assert phrase in FLAT, phrase


def test_configs_and_plan_are_the_generators(scratch, capsys):
    repo = {r: open(os.path.join(L.HERE, "configs", r + ".yaml")).read() for r in RUNS}   # before the monkeypatch
    plan_txt = open(DIAG_PLAN).read()
    tmp_path, screens = scratch
    for s in (103, 104, 105):
        screens.base(s, P)
    screens.plan("stage2_base_diag", RUNS, HEADER, "SCREENS BASE DIAG DONE", "SCREENS SCREENS STAGE 2 S006 SEEDS DONE")
    for r in RUNS:
        assert open(tmp_path / "screens" / "configs" / f"{r}.yaml").read() == repo[r], r
    assert open(tmp_path / "plans" / "stage2_base_diag.txt").read() == plan_txt


def test_configs_check_and_differ_only_in_seed(capsys):
    import screens
    ref = L.flat(L.resolve(L.find("base_s101")[0]))
    for r in RUNS:
        f = L.flat(L.resolve(L.find(r)[0]))
        assert L.diff(f, ref) == {"name", "out_dir", "seed"} and f["seed"] == int(r[-3:]) and L.check(L.find(r)[0], P) == []
    assert screens.main(["check", *RUNS]) == 0 and capsys.readouterr().out.count("ok      ") == 3
    assert screens.main(["check"]) == 0 and capsys.readouterr().out.count("ok      ") == 96 == len(L.all_configs())
    assert "screens.py check: 3 ok, 96 ok over every config" in FLAT


def test_plan_current_and_acceptance():
    lines = open(DIAG_PLAN).read().splitlines()
    assert lines[1:] == ["wait_mark SCREENS SCREENS STAGE 2 S006 SEEDS DONE"] + [f"train {r}" for r in RUNS] + [END]
    s6 = open(os.path.join(L.HERE, "plans", "stage2_s006_seeds.txt")).read().splitlines()
    assert s6[-1] == "mark SCREENS STAGE 2 S006 SEEDS DONE" and lines[0] == f"# {HEADER} (screens.py). train = screens " \
        "check, preflight, train, score"
    assert f"stage2_base_diag.txt (sha256 {L.sha256(DIAG_PLAN)[:16]}...; written by screens.plan), its 6 lines:" in FLAT
    assert " ".join(lines[0].split()) + " [one line in the file]" in FLAT and all("\n      " + ln + "\n" in SEC for ln in lines[1:])
    cur = [ln.strip() for ln in open(os.path.join(L.HERE, "plans", "CURRENT")) if not ln.startswith("#")]
    assert cur == DIAG_PLANS == ["stage2_base_diag"] and base_diag_configs() == set(RUNS)
    assert "plans/CURRENT = stage2_base_diag (edited here" in FLAT and "Estimate 0.937 h." in FLAT


def variant(tmp_path, name, edit):
    p = tmp_path / f"{name}.txt"
    p.write_text(edit(open(DIAG_PLAN).read()))
    return str(p)


def copy_104(tmp_path, old, new, mp=None):
    """Scratch copies of the 3 configs, base_s104.yaml with one edit, for a find() stub (screens.py check is stubbed by
    the caller). With mp the scratch dir also stands for experiments/screens/configs, so only the content check can
    refuse the state."""
    paths = {"base_s101": L.find("base_s101")[0]}
    for r in RUNS:
        txt = open(L.find(r)[0]).read()
        assert r != "base_s104" or txt.count(old) == 1
        txt = txt.replace(old, new) if r == "base_s104" else txt
        (tmp_path / f"{r}.yaml").write_text(txt.replace("extends: screens_base.yaml", f"extends: {L.BASE_YAML}"))
        paths[r] = str(tmp_path / f"{r}.yaml")
    if mp:
        orig = L.config_dir
        mp.setattr(L, "config_dir", lambda sid: str(tmp_path) if sid is None else orig(sid))
    return lambda r: [paths[r]] if r in paths else L.find(r)


BAD = {   # name: (edit of the plan text, keyword arguments, or a function of tmp_path giving them)
    "runs_out_of_order": (lambda t: t.replace("train base_s103\ntrain base_s104\n", "train base_s104\ntrain base_s103\n"), {}),
    "a_seed_dropped": (lambda t: t.replace("train base_s104\n", ""), {}),
    "seed_106": (lambda t: t.replace("base_s105", "base_s106"), {}),
    "an_arm_run_added": (lambda t: t.replace(END, "train s004_canonac_g1_s101\n" + END), {}),
    "waits_on_the_stage2_seeds_mark": (lambda t: t.replace("S006 SEEDS DONE\n", "SEEDS DONE\n", 1), {}),
    "runs_past_its_mark": (lambda t: t + "train base_s101\n", {}),
    "ends_at_the_s006_mark": (lambda t: t.replace(END, "mark SCREENS STAGE 2 S006 SEEDS DONE"), {}),
    "not_recorded": (lambda t: t, {"recorded": []}),
    "recorded_two_runs": (lambda t: t, {"recorded": RUNS[:2]}),
    "a_config_fails_check": (lambda t: t, {"check": lambda p: ["refused"] if "base_s10" in p else []}),
    "no_s006_plan": (lambda t: t, lambda tmp, mp: {"s006_plan": str(tmp / "absent.txt")}),
    "a_config_at_seed_101": (lambda t: t, lambda tmp, mp: {"find": copy_104(tmp, "seed: 104", "seed: 101", mp),
                                                          "check": lambda p: []}),
    "a_config_at_another_lr": (lambda t: t, lambda tmp, mp: {"find": copy_104(tmp, "{lr: 0.003,", "{lr: 0.006,", mp),
                                                            "check": lambda p: []}),
    "a_config_outside_the_shared_dir": (lambda t: t, lambda tmp, mp: {"find": copy_104(tmp, "seed: 104", "seed: 104"),
                                                                     "check": lambda p: []}),
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_acceptance_refuses(tmp_path, monkeypatch, name):
    edit, kw = BAD[name]
    kw = kw(tmp_path, monkeypatch) if callable(kw) else kw
    with pytest.raises(AssertionError):
        base_diag_configs(variant(tmp_path, name, edit), **kw)


def test_acceptance_takes_the_repo_plan(tmp_path, monkeypatch):
    assert base_diag_configs(variant(tmp_path, "same", lambda t: t)) == set(RUNS) and len(BAD) == BAD_N
    find = copy_104(tmp_path, "seed: 104", "seed: 104", monkeypatch)     # the scratch copies unedited: taken
    assert base_diag_configs(variant(tmp_path, "same", lambda t: t), find=find, check=lambda p: []) == set(RUNS)
    assert base_diag_configs(str(tmp_path / "absent.txt")) == set()
    line = "  BASE RUNS ADDED BY AMENDMENT BASE-DIAG: base_s103, base_s104, base_s105 (3 runs)\n"
    assert recorded_diag(line) == RUNS and recorded_diag(" " + line) == [] and recorded_diag(line.replace(" BY", ",")) == []


def test_notes_blocks():
    head = "\nONSET READING (2026-10-06, SCREENS.txt AMENDMENT BASE-DIAG; decided before any stage 2 onset is read): "
    for d, what in (("S004_canon", "at this screen's pick g 1"), ("S007_smeared_key", "at this screen's pick g 1"),
                    ("S006_mtp_aux", "from this screen's g 1 IND-only runs")):
        notes = open(os.path.join(L.EXPD, d, "notes.txt")).read()
        blk = " ".join(notes[notes.index(head):].split())
        for phrase in (what, "over seeds 101 to 105 (df 4), thr = 2.776 x sigma_seed", "no EQUAL", "changes no verdict"):
            assert phrase in blk, (d, phrase)
        assert ("now means EARLIER here" in blk) == (d != "S004_canon")
    assert "AMENDMENT BASE-DIAG" not in open(os.path.join(L.EXPD, "S005_forget_gate", "notes.txt")).read()


def test_every_number_in_the_entry_is_pinned(V):
    exp = set(V["fmt"].values()) | {"101", "102", "103", "104", "105", "153", "165", "024", "0.975", "0.05", "0.66",
                                    "1.634", "25.0", "26.302", "1.698", "25.366", "0.936"} | {str(x) for x in PASSED + (WHOLE,)}
    x = json.load(open(os.path.join(L.EXPD, "S001_attn_gate", "results.json")))["base_level_vs_e3_part1"]["metrics"]
    exp |= {f"{min(x['CHAT']['e3_arm_a']):.5f}", f"{max(x['CHAT']['e3_arm_a']):.5f}"}
    hexes = {"d21df3d", "3227c60", "93bc10b", L.sha256(DIAG_PLAN)[:16], L.sha256(os.path.join(L.HERE, "chain_waiter.sh"))[:16],
             L.sha256(os.path.join(L.HERE, "base_diag_waiter.sh"))[:16]}
    text = re.sub(r"\b[a-z]\w*_[\w.<>]*", " ", SEC)
    found = re.findall(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,16}\b", text)
    assert set(found) == hexes, set(found) ^ hexes
    text = re.sub(r"\b[0-9a-f]{7,16}\b", " ", text)
    assert set(re.findall(r"\b\d{4}-\d\d-\d\d\b", text)) == {"2026-10-06"}
    assert set(re.findall(r"\b\d\d:\d\d(?::\d\d)?\b", text)) == set(TIMES.values())
    text = re.sub(r"\b\d{4}-\d\d-\d\d\b|\b\d\d:\d\d(?::\d\d)?\b", " ", text)
    nums = re.findall(r"(?<![\w.])(\d[\d,]*\d(?:\.\d+)?|\d\.\d+)(?![\w])", text)
    big = {n for n in nums if "." in n or "," in n or len(n) >= 3}
    assert len(big) > 30 and big <= exp, sorted(big - exp)
    assert len(M.MUTANTS) >= WHOLE and all(m in M.MUTANTS for m in M.MUTANTS_BASEDIAG) and len(M.MUTANTS_BASEDIAG) == NEW
    for phrase in (f"tests {PASSED[1]} passed ({PASSED[0]} before", f"{NEW} new mutants", f"{NEW} of {NEW} killed; the "
                   f"whole file {WHOLE} of {WHOLE} killed", f"the acceptance refusing {BAD_N} bad states",
                   "AMENDMENT BASE-DIAG (2026-10-06 " + TIMES["entry"] + " EDT"):
        assert phrase in FLAT, phrase
