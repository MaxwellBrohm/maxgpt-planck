"""SCREENS AMENDMENT S006-REINSTATE (2026-10-06): Max's words verbatim; the cap check at 27 h generated from
measured_hours.tsv through screens_hours and analyze_lib.cap_cut (nothing cut; the same input at 25 h cuts S006, as the
entry before recorded); the reinstatement record; S006's 4 seed configs are screens.py seeds output at the pick g 2,
byte for byte, and differ from their BASE and their g-check config only in the registered keys; the plan and
plans/CURRENT; the acceptance (reinstate_lib.s006_seed_configs) on scratch plans; the S006 notes block; every number
in the entry pinned. No verdict, no model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_s006_reinstate.py
"""
import contextlib, io, os, re, sys, pytest  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze_lib as AL, mutation_screens as M, screens_hours as H, screens_lib as L  # noqa: E401,E402
from reinstate_lib import END, REINSTATED_PLANS, recorded_reinstated, s006_seed_configs  # noqa: E402
from basediag_lib import DIAG_PLANS, base_diag_configs  # noqa: E402  (AMENDMENT BASE-DIAG, later)
from test_screens_configs import P, STAGE2_SEEDS, recorded_cuts  # noqa: E402
from test_screens_refusals import scratch  # noqa: E402,F401  (the scratch config tree fixture)
from test_screens_s003_stage_c import section  # noqa: E402

SEC = section("AMENDMENT S006-REINSTATE")
FLAT = " ".join(SEC.split())
MAX = ("if it was cut does that mean you are going to add it back later or its just gone? because if something can be "
       "helpful even a little we should give it its proper chance, maybe its finally the big breakthrough that makes "
       "this all work! obv the final decision is by you and if you dont think theres anything special with it than "
       "thats fine.")
RUNS = ["s006_mtp_g2_s101", "s006_mtp_g1_s101", "s006_mtp_g2_s102", "s006_mtp_g1_s102"]
S6DIR = os.path.join(L.EXPD, "S006_mtp_aux", "configs")          # the repo's, fixed before any monkeypatch
PLAN = os.path.join(L.HERE, "plans", "stage2_s006_seeds.txt")
PICKS = ["--pick=S005.forget=1", "--pick=S004.canonac=1", "--pick=S007.smear=1", "--pick=S006.mtp=2"]
PASSED, WHOLE, NEW = (460, 523), 273, 43   # experiments/screens/tests before and after; mutation_screens.py whole, new
TIMES = {"entry": "21:45", "read": "21:23", "start": "20:38:01", "held": "20:38:03", "next": "21:01:38",
         "ap": "20:38:19", "tsv": "19:52:00"}


@pytest.fixture(scope="module")
def V():
    with contextlib.redirect_stdout(io.StringIO()):
        r = H.report(P)
    q, mr, mo = r["tally"]["queued"], r["measured_runs"], r["other"]
    p6 = H.run_h(L.STEPS, H.mult("S006", "mtp"))
    q2 = {**q, "S006": q["S006"] + 2 * p6}
    c27, c25 = AL.cap_cut(q2, mr + mo, cap=27.0), AL.cap_cut(q2, mr + mo)
    narrow = AL.cap_cut(q2, mr, cap=27.0)
    s8, room = sum(q.values()) - q["S006"], 27.0 - c27["total_after"]
    own = H.run_h(L.STEPS, H.mult("S005", "base"))
    held = H.hours_between("2026-10-06 " + TIMES["held"], "2026-10-06 " + TIMES["next"])
    f = lambda x, n=3: f"{x:.{n}f}"  # noqa: E731
    fmt = {"mr": f(mr), "mo": f(mo), "m": f(mr + mo), "qs": f(sum(q.values())), "s8": f(s8), "q6": f(q["S006"]),
           "ind": f(2 * p6), "q2": f(sum(q2.values())), "tot": f(c27["total_before"]), "room": f(room),
           "narrow": f(narrow["total_before"]), "after25": f(c25["total_after"]), "p6": f(p6), "four": f(4 * p6),
           "pct8": f(100 * room / s8, 1), "pct10": f(100 * room / sum(q.values()), 1), "held": f(held), "own": f(own),
           "lr": f"{2 * P['lr5'][0]:g}", "about": f(4 * p6, 1), "mtpw": "1.0"}
    return {"r": r, "q": q, "c27": c27, "c25": c25, "narrow": narrow, "fmt": fmt, "runs": H.load()[0]}


def test_max_verbatim_and_the_decision():
    assert f'Max (the PI), 2026-10-06, verbatim: "{MAX}"' in FLAT
    for phrase in ("S006 is reinstated with its full registered design", "By this amendment the batch's cap is 27 h, "
                   "not 25 h. The cap's reading is unchanged", "Decided before any S006 seed run exists and before any "
                   "stage 2 verdict is read", "only the number moves", "this amendment changes the cap, not its reading",
                   "ORDER CHANGE: S006 runs after the other stage 2 seed sets, not inside ORDER 2's seed-major order"):
        assert phrase in FLAT, phrase
    assert recorded_reinstated() == {"S006"} == recorded_cuts()           # the cut stays recorded; so does this


def test_cap_check_at_27_h(V):
    v, c27, c25 = V["fmt"], V["c27"], V["c25"]
    assert c27["cut"] == [] and c27["fits"] and c25["cut"] == ["S006"] and V["r"]["cap"]["cut"] == []
    assert AL.CAP == 25.0 and c27["total_before"] > 25.0 and V["narrow"]["total_before"] < 25.0  # the cut hinged on OTHER
    assert f"Measured {v['mr']} h of runs + {v['mo']} h other = {v['m']} h; queued 10 of 90 registered runs {v['qs']} h " \
           f"(stage2_seeds' 8 runs {v['s8']} h, S006's 2 seed runs {v['q6']} h) + S006's 2 IND runs {v['ind']} h = " \
           f"{v['q2']} h; total {v['tot']} h <= 27 h: nothing cut, {v['room']} h of headroom (runs only, reported: " \
           f"{v['narrow']} h). The same input at 25 h cuts S006 ({v['after25']} h), as recorded." in FLAT
    old = " ".join(section("S006 EXTENSION RESULT / STAGE 2 SEEDS LAUNCH CHECK").split())
    assert f"= {v['tot']} h, 0.366 h over 25 h" in old and f"= {v['after25']} h <= 25 h" in old
    assert f"without them the same input totals {v['narrow']} h" in FLAT and f"(2.120 h; STAGE 2 READINESS (c)" in FLAT
    assert f"about {v['about']} GPU h: 4 runs at {v['p6']} h, {v['four']} h" in FLAT and v["mo"] == "2.120"
    rows = [ln for ln in open(H.TSV) if ln.startswith("run\t")]
    assert len(rows) == len(V["runs"]) == 81 and max(ln.split("\t")[4] for ln in rows) == "2026-10-06 " + TIMES["tsv"]


def test_inside_the_plans(V):
    v = V["fmt"]
    assert f"more than {v['room']} h over their {v['s8']} h estimate ({v['pct8']}%)" in FLAT
    assert f"more than {v['room']} h over their {v['qs']} h estimate ({v['pct10']}%)" in FLAT
    assert f'"gpu.lock held" {TIMES["held"]} to the next "waiting for gpu.lock" {TIMES["next"]} ({v["held"]} h; ' \
           f'estimate {v["own"]} h)' in FLAT


def test_configs_are_the_generators_at_the_pick(scratch):
    repo = {r: open(os.path.join(S6DIR, r + ".yaml")).read() for r in RUNS}
    tmp_path, screens = scratch
    screens.main(["seeds", "--stage", "2", *PICKS])
    for r in RUNS:
        assert open(tmp_path / "S006_mtp_aux" / "configs" / f"{r}.yaml").read() == repo[r], r
    would = [ln.split()[1] for ln in open(tmp_path / "plans" / "stage2_seeds.txt") if ln.startswith("train ")]
    assert [x for x in would if x.startswith("s006_")] == RUNS
    for s in (101, 102):
        i = would.index(f"s007_smear_g1_s{s}")
        assert would[i + 1:i + 3] == [f"s006_mtp_g2_s{s}", f"s006_mtp_g1_s{s}"]


def test_configs_differ_only_in_registered_keys(V, capsys):
    import screens
    for r in RUNS:
        g, s = r.split("_g")[1].split("_s")
        f = L.flat(L.resolve(L.find(r)[0]))
        b = L.flat(L.resolve(L.find(f"base_s{s}")[0]))
        lr = {k for k in L.LRK} if g == "2" else set()
        assert L.diff(f, b) == {"name", "out_dir", "train.mtp", "train.mtp_weight"} | lr, r
        assert (f["train.mtp"], f["train.mtp_weight"], f["seed"]) == (1, 1.0, int(s))
        assert [f[k] for k in L.LRK] == [float(g) * P["lr5"][0]] * 3 and L.check(L.find(r)[0], P) == []
        assert L.diff(f, L.flat(L.resolve(L.find(f"s006_mtp_g{g}_s1")[0]))) == {"name", "out_dir", "seed"}
    assert screens.main(["check", *RUNS]) == 0 and capsys.readouterr().out.count("ok      ") == 4
    n = 93 + len(base_diag_configs())          # later: AMENDMENT BASE-DIAG's 3 BASE configs, else none
    assert screens.main(["check"]) == 0 and capsys.readouterr().out.count("ok      ") == n == len(L.all_configs())
    assert "screens.py check: 4 ok, 93 ok over every config" in FLAT
    assert f"the three LRs, 2 x LR5 = {V['fmt']['lr']} each" in FLAT


def test_plan_current_and_acceptance(V):
    lines = open(PLAN).read().splitlines()
    assert lines[1:] == ["wait_mark SCREENS SCREENS STAGE 2 SEEDS DONE"] + [f"train {r}" for r in RUNS] + [END]
    assert open(STAGE2_SEEDS).read().splitlines()[-1] == "mark SCREENS STAGE 2 SEEDS DONE"
    assert f"stage2_s006_seeds.txt (sha256 {L.sha256(PLAN)[:16]}...; written by screens.plan), its 7 lines:" in FLAT
    assert " ".join(lines[0].split()) + " [one line in the file]" in FLAT
    assert all("\n      " + ln + "\n" in SEC for ln in lines[1:]) and f"Estimate {V['fmt']['four']} h." in FLAT
    cur = [ln.strip() for ln in open(os.path.join(L.HERE, "plans", "CURRENT")) if not ln.startswith("#")]
    later = DIAG_PLANS if base_diag_configs() else []       # AMENDMENT BASE-DIAG moved CURRENT on, later
    assert cur == (later or REINSTATED_PLANS) and REINSTATED_PLANS == ["stage2_s006_seeds"]
    assert s006_seed_configs() == set(RUNS)
    assert "plans/CURRENT = stage2_s006_seeds (edited here" in FLAT


def variant(tmp_path, name, edit):
    p = tmp_path / f"{name}.txt"
    p.write_text(edit(open(PLAN).read()))
    return str(p)


BAD = {   # name: (edit of the plan text, keyword arguments)
    "ind_before_verdict": (lambda t: t.replace("train s006_mtp_g2_s101\ntrain s006_mtp_g1_s101\n",
                                               "train s006_mtp_g1_s101\ntrain s006_mtp_g2_s101\n"), {}),
    "seed_102_first": (lambda t: t.replace("101", "1xx").replace("102", "101").replace("1xx", "102"), {}),
    "waits_on_the_extension_mark": (lambda t: t.replace("STAGE 2 SEEDS DONE\n", "S006 EXTENSION DONE\n", 1), {}),
    "runs_past_its_mark": (lambda t: t + "train base_s101\n", {}),
    "one_seed_only": (lambda t: t.replace("train s006_mtp_g2_s102\ntrain s006_mtp_g1_s102\n", ""), {}),
    "ind_runs_dropped": (lambda t: t.replace("train s006_mtp_g1_s101\n", "").replace("train s006_mtp_g1_s102\n", ""), {}),
    "not_recorded_reinstated": (lambda t: t, {"reinstated": set()}),
    "not_recorded_cut": (lambda t: t, {"cuts": set()}),
    "an_s006_config_fails_check": (lambda t: t, {"check": lambda p: ["refused"] if "s006_mtp_g" in p else []}),
    "ends_at_the_stage2_seeds_mark": (lambda t: t.replace(END, "mark SCREENS STAGE 2 SEEDS DONE"), {}),
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_acceptance_refuses(tmp_path, name):
    edit, kw = BAD[name]
    with pytest.raises(AssertionError):
        s006_seed_configs(variant(tmp_path, name, edit), **{"cuts": {"S006"}, "reinstated": {"S006"}, **kw})


def test_acceptance_takes_the_repo_plan_and_a_g1_pick_without_ind_runs(tmp_path):
    assert s006_seed_configs(variant(tmp_path, "same", lambda t: t), cuts={"S006"}, reinstated={"S006"}) == set(RUNS)
    g1 = variant(tmp_path, "g1", lambda t: t.replace("train s006_mtp_g2_s101\n", "").replace("train s006_mtp_g2_s102\n", ""))
    assert s006_seed_configs(g1, cuts={"S006"}, reinstated={"S006"}) == {"s006_mtp_g1_s101", "s006_mtp_g1_s102"}
    assert s006_seed_configs(str(tmp_path / "absent.txt")) == set()
    line = "  REINSTATED BY AMENDMENT S006-REINSTATE: S006 (4 runs)\n"
    assert recorded_reinstated(line) == {"S006"} and recorded_reinstated(" " + line) == set()
    assert recorded_reinstated(line.replace("REINSTATED BY", "REINSTATED, BY")) == set()


def test_s006_notes_block(V):
    v, notes = V["fmt"], open(os.path.join(L.EXPD, "S006_mtp_aux", "notes.txt")).read()
    blk = " ".join(notes[notes.index("\nREINSTATED (2026-10-06, SCREENS.txt AMENDMENT S006-REINSTATE"):].split())
    for phrase in (f"the 4 runs cost {v['four']} h ({v['p6']} h each)", f"Cap check at 27 h: {v['tot']} h, nothing cut, "
                   f"{v['room']} h of headroom", "raising the batch's cap from 25 h to 27 h", "plans/stage2_s006_seeds (" +
                   ", ".join(RUNS) + ")", "The CUT block above stays as the record of ORDER's rule at 25 h."):
        assert phrase in blk, phrase


def test_every_number_in_the_entry_is_pinned(V):
    exp = set(V["fmt"].values()) | {"101", "102", "105", "024", "25.0", "4.2"} | {str(x) for x in PASSED + (WHOLE,)}
    hexes = {"d21df3d", "93bc10b", L.sha256(PLAN)[:16], L.sha256(os.path.join(L.HERE, "s006_waiter.sh"))[:16]}
    text = re.sub(r"\b[a-z]\w*_[\w.<>]*", " ", SEC)
    found = re.findall(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,16}\b", text)
    assert set(found) == hexes, set(found) ^ hexes
    text = re.sub(r"\b[0-9a-f]{7,16}\b", " ", text)
    assert set(re.findall(r"\b\d{4}-\d\d-\d\d\b", text)) == {"2026-10-06"}
    assert set(re.findall(r"\b\d\d:\d\d(?::\d\d)?\b", text)) == set(TIMES.values())
    text = re.sub(r"\b\d{4}-\d\d-\d\d\b|\b\d\d:\d\d(?::\d\d)?\b", " ", text)
    nums = re.findall(r"(?<![\w.])(\d[\d,]*\d(?:\.\d+)?|\d\.\d+)(?![\w])", text)
    big = {n for n in nums if "." in n or "," in n or len(n) >= 3}
    assert len(big) > 20 and big <= exp, sorted(big - exp)
    assert len(M.MUTANTS) >= WHOLE and all(x in M.MUTANTS for x in M.MUTANTS_S006R) and len(M.MUTANTS_S006R) == NEW
    for phrase in (f"tests {PASSED[1]} passed ({PASSED[0]} before", f"{NEW} new mutants", f"{NEW} of {NEW} killed; the "
                   f"whole file {WHOLE} of {WHOLE} killed", "AMENDMENT S006-REINSTATE (2026-10-06 " + TIMES["entry"] + " EDT"):
        assert phrase in FLAT, phrase
