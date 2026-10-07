"""SCREENS after the stage 1 seed sets (SCREENS.txt STAGE 1 SEEDS DONE / STAGE 2 LAUNCH CHECK, 2026-10-06): the 12
seed runs are in measured_hours.tsv at the lock hours the entry prints, the cap check before stage 2's first g-check
fits (nothing cut; how the cut rule reads S002 and S003, whose seed sets are done; what one conditional run does; the
in-plan thresholds), stage2_select is exactly ORDER 2's g-checks between the smokes mark and its own, and every number
in the entry is pinned (tables generated from the data; every other decimal, comma or 3+ digit number must be one V
computes; small counts by phrase). No verdict, no model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_stage2_launch.py
"""
import contextlib, datetime as dt, io, os, re, sys, pytest  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze_lib as AL, mutation_screens as M, screens_hours as H, screens_lib as L  # noqa: E401,E402
from test_screens_configs import PLAN_SEQUENCE  # noqa: E402
from test_screens_s003_stage_c import section  # noqa: E402

P = L.params()
PLAN, SEEDS = (os.path.join(L.HERE, "plans", f"{n}.txt") for n in ("stage2_select", "stage1_seeds"))
RUNS1 = [ln.split()[1] for ln in open(SEEDS) if ln.startswith("train ")]
# "train" to "done" (s) and median log.jsonl tok/s per arm (seed 101, seed 102): the PC queue log and log.jsonl
TRAIN = {"base": (994, 994), "s001_nogate_g1": (953, 954), "s002_novres_g1": (953, 953),
         "s002_noqknorm_g1": (841, 841), "s002_nonormscale_g1": (983, 983), "s003_adamw_e3_r2": (974, 973)}
TOKS = {"base": ("254k", "254k"), "s001_nogate_g1": ("263k", "263k"), "s002_novres_g1": ("265k", "265k"),
        "s002_noqknorm_g1": ("301k", "302k"), "s002_nonormscale_g1": ("256k", "256k"), "s003_adamw_e3_r2": ("258k", "259k")}
QSTART, QEND, ENTRY_TIME, MARK_SMOKES = "21:05:42", "00:36:33", "04:30", "10:59"
PCSHA = ("3de087db1b1ac3e9", "88e03ff006879b21", "d30157a57b45e70f")   # PC queue log, status.jsonl, the seed plan
PASSED = (377, 384)             # experiments/screens/tests before and after this entry (Mac CPU), without the tests of
# another step's two new files, test_screens_stage1_verdicts.py and test_screens_stage1_verdicts_numbers.py (STAGE 1
# VERDICTS counts them)
# "waiting for gpu.lock" and "gpu.lock held" per seed run, from the PC queue log (sha256 3de087db1b1ac3e9...)
LOCKREQ = {"base_s101": ("21:05:44", "21:05:44"), "s001_nogate_g1_s101": ("21:24:04", "21:24:04"),
           "s002_novres_g1_s101": ("21:41:41", "21:41:41"), "s002_noqknorm_g1_s101": ("21:59:18", "21:59:18"),
           "s002_nonormscale_g1_s101": ("22:15:01", "22:15:01"), "s003_adamw_e3_r2_s101": ("22:33:09", "22:33:10"),
           "base_s102": ("22:51:08", "22:51:09"), "s001_nogate_g1_s102": ("23:09:27", "23:09:28"),
           "s002_novres_g1_s102": ("23:27:05", "23:27:05"), "s002_noqknorm_g1_s102": ("23:44:41", "23:44:41"),
           "s002_nonormscale_g1_s102": ("00:00:25", "00:00:25"), "s003_adamw_e3_r2_s102": ("00:18:34", "00:18:34")}
WHOLE, NEW = 123, 10            # mutation_screens.py: the whole file and this entry's mutants (mutants_s2launch.py)
SEC = section("STAGE 1 SEEDS DONE / STAGE 2 LAUNCH CHECK")
FLAT = " ".join(SEC.split())
S2 = [f"{sid.lower()}_{a}_g{g}_s1" for sid in L.ORDER[2] for a in L.SCREENS[sid]["arms"] for g in ("0.5", "1", "2")]
EXTS = {ln.split()[1] for ln in open(os.path.join(L.HERE, "plans", "stage2_s006x.txt")) if ln.startswith("train ")}


@pytest.fixture(scope="module")
def V(tmp_path_factory):
    tsv = tmp_path_factory.mktemp("tsv") / "m.tsv"      # measured_hours.tsv as of this entry: the 12 stage 2 g-check
    tsv.write_text("".join(ln for ln in open(H.TSV) if not (ln.startswith("run\t") and ln.split("\t")[1] in {*S2, *EXTS})))
    runs, _ = H.load(str(tsv))                          # lines (STAGE 2 SELECTION RESULT) and S006's extension taken out
    with contextlib.redirect_stdout(io.StringIO()):
        r = H.report(P, str(tsv))
    t, c, f = r["tally"], r["cap"], (lambda x, n=3: f"{x:.{n}f}")
    full = lambda s, a: H.run_h(L.STEPS, H.mult(s, a))  # noqa: E731
    per = {s: full(s, a) for s in L.ORDER[2] for a in L.SCREENS[s]["arms"]}
    g = {s: 3 * per[s] for s in per}
    m, head, run = r["measured_runs"] + r["other"], 25 - c["total_before"], H.run_h(L.STEPS)
    lock = [runs[n] for n in RUNS1]
    train = sum(sum(v) for v in TRAIN.values()) / 3600
    wall = (dt.datetime(2026, 10, 6, *map(int, QEND.split(":"))) - dt.datetime(2026, 10, 5, *map(int, QSTART.split(":"))))
    cum = [g["S005"], g["S005"] + g["S004"], g["S005"] + g["S004"] + g["S007"]]
    one = {}
    for s in ("S007", "S006", "S004", "S005"):
        x = AL.cap_cut({**t["queued"], "S004": t["queued"]["S004"] + per[s]}, m)
        one[s] = (x["total_before"], x["total_after"], x["cut"])
    fmt = {"lock": f(sum(lock)), "s101": f(sum(lock[:6])), "s102": f(sum(lock[6:])), "train": f(train),
           "gaps": f(sum(lock) - train), "est": f(12 * run), "run": f(run), "under": f(12 * run - sum(lock)),
           "wall": f(wall.seconds / 3600, 2), "outside": f(wall.seconds / 3600 - sum(lock)), "lmin": f(min(lock)),
           "lmax": f(max(lock)), "total": f(sum(runs.values())), "mr": f(r["measured_runs"]), "mo": f(r["other"]),
           "mrmo": f(m), "qs": f(sum(t["queued"].values())), "tot": f(c["total_before"]), "room": f(head),
           "narrow": f(r["narrow"]["total_before"]), "g2": f(sum(g.values())),
           "seeds2": f(sum(t["queued"].values()) - sum(g.values())), "ext": f(2 * sum(per.values()), 2),
           "ind": f(2 * (per["S006"] + per["S007"]), 2), "s006q": f(t["queued"]["S006"]),
           "both": f(t["queued"]["S006"] + t["queued"]["S007"]), "c1": f(cum[0] + head), "c1r": f((cum[0] + head) / 3),
           "c2": f(cum[1] + head), "c3": f(cum[2] + head), "c3e": f(cum[2]), "pct": f(100 * head / cum[2], 1),
           "own": f(full("S005", "base"))}
    fmt.update({f"p{s}": f(v) for s, v in per.items()} | {f"g{s}": f(v) for s, v in g.items()})
    fmt.update({f"o{s}": f(v[0]) for s, v in one.items()} | {f"a{s}": f(v[1]) for s, v in one.items()})
    return {"runs": runs, "r": r, "one": one, "head": head, "cum": cum, "m": m, "fmt": fmt, "wall_s": wall.seconds, "sel": {
        n: h for n, h in runs.items() if n not in RUNS1}}


def test_seed_runs_are_measured_once_and_the_earlier_lines_unchanged(V):
    runs = V["runs"]
    assert len(RUNS1) == 12 and set(RUNS1) <= set(runs) and len(runs) == 68 and len(V["sel"]) == 56
    assert round(sum(V["sel"].values()), 3) == 7.917 and V["r"]["tally"]["extra"] == {}
    ends = {ln.split("\t")[1]: ln.split("\t")[4] for ln in open(H.TSV) if ln.startswith("run\t")}
    assert max(ends[n] for n in RUNS1) == "2026-10-06 " + QEND
    now = H.load()[0]                       # later regenerations add lines only (their header: their own test)
    assert ("sha256 " + PCSHA[0] + "... and " + PCSHA[1] in open(H.TSV).read()) if len(now) == 68 else \
        (set(now) - set(runs) - EXTS == set(S2) and len(now) == 80 + len(set(now) & EXTS))


def hours_rows(runs):
    out = []
    for arm, (t1, t2) in TRAIN.items():
        cells = []
        for i, s in enumerate((101, 102)):
            n = f"{arm}_s{s}"
            cells.append(f"{n:<27}{runs[n]:.3f} {(t1, t2)[i] / 3600:.3f}  {TOKS[arm][i]}")
        out.append("    " + "  | ".join(cells))
    return out


def test_entry_health_and_hours(V):
    v = V["fmt"]
    for row in hours_rows(V["runs"]):
        assert "\n" + row + "\n" in SEC, row
    assert f"Total {v['lock']} GPU hours measured for the seed sets (seed 101 {v['s101']}, seed 102 {v['s102']}; " \
           f"{v['train']} h training, {v['gaps']} h scoring and gaps), {v['under']} h under their {v['est']} h " \
           f"estimate (12 x {v['run']} h); {v['wall']} h wall clock (lock waits {v['outside']} h in all)" in FLAT
    # the committed wording; it is the wall clock outside the lock holds (CORRECTIONS 2026-10-06, pinned below)
    assert f"68 runs, {v['total']} h (stage 1 selection A 5.483 + stage B 1.622 + stage C 0.812 + seed sets " \
           f"{v['lock']})" in FLAT and f"held the lock {v['lmin']} to {v['lmax']} h" in FLAT
    for phrase in ('All 12 runs "done ... rc 0" on the first try', 'status.jsonl: 12 "done and scored"',
                   "SCORED, an empty score.err and a train.out with no Traceback, nan or Error in all 12",
                   "136 SCREENS lines (112 + 12 starts, 12 ends)", "its 56 earlier run lines are unchanged"):
        assert phrase in FLAT, phrase


def cap_rows(V):
    t, out = V["r"]["tally"], []
    for scr in ["BASE"] + list(L.SCREENS):
        nm = sum(1 for n in V["runs"] if H.slot(n)[0] == scr)
        m, q = t["measured"].get(scr, 0.0), t["queued"].get(scr, 0.0)
        out.append(f"    {scr:<8}{nm:>2}{'':13}{m:.3f}   {t['n_queued'].get(scr, 0):<12}{q:.3f}  {m + q:.3f}")
    return out


def test_cap_check_before_stage_2(V):
    r, v, t = V["r"], V["fmt"], V["r"]["tally"]
    assert r["cap"]["cut"] == [] and sum(t["n_queued"].values()) == 22 and t["n_registered"] == 90
    assert sorted(t["queued"]) == ["S004", "S005", "S006", "S007"]          # stage 1 has nothing left to run
    for row in cap_rows(V):
        assert "\n" + row + "\n" in SEC, row
    assert f"Measured {v['mr']} h of runs + {v['mo']} h other = {v['mrmo']} h; queued 22 of 90 registered runs (stage " \
           f"2: 12 g-checks {v['g2']} h, 8 arm seed runs and S005's own BASE x 2 {v['seeds2']} h), {v['qs']} h; total " \
           f"{v['tot']} h <= 25 h: nothing cut, {v['room']} h of headroom (runs only, reported: {v['narrow']} h)" in FLAT


def test_the_cut_rule_reads_finished_screens_as_nothing_to_cut(V):
    """S002 and S003 finished their seed sets at k = 2: no hours queued, so cap_cut passes over them; past 25 h the
    rule cuts S006, then S007, then nothing (S001, S004, S005, BASE never)."""
    t, m, h, v = V["r"]["tally"], V["m"], V["head"], V["fmt"]
    assert all(P["seeds"][L.SCREENS[s]["cls"]] == [101, 102] for s in ("S002", "S003"))
    for over, cut, fits in ((h - 1e-4, [], True), (h + 1e-4, ["S006"], True),
                            (h + t["queued"]["S006"] + 1e-4, ["S006", "S007"], True),
                            (h + t["queued"]["S006"] + t["queued"]["S007"] + 1e-4, ["S006", "S007"], False)):
        x = AL.cap_cut(t["queued"], m + over)
        assert (x["cut"], x["fits"]) == (cut, fits), over
    assert f"Past 25 h by up to {v['s006q']} h the rule cuts S006; by up to {v['both']} h S006 and S007" in FLAT
    assert "so for stage 2 the order S006, S003, S007, S002 acts as S006 then S007" in FLAT


def test_one_conditional_run_and_the_in_plan_thresholds(V):
    one, v = V["one"], V["fmt"]
    assert [one[s][2] for s in ("S007", "S006", "S004", "S005")] == [[], [], ["S006"], ["S006"]]
    assert f"an S007 extension or IND run ({v['pS007']} h) gives {v['oS007']} h and an S006 one ({v['pS006']} h) " \
           f"{v['oS006']} h, both fit; an S004 extension ({v['pS004']} h) gives {v['oS004']} h and an S005 extension " \
           f"({v['pS005']} h) {v['oS005']} h, and the rule then cuts S006 ({v['aS004']} h, {v['aS005']} h)" in FLAT
    assert f"up to {v['ext']} h of stage 2 g-check extensions and up to {v['ind']} h of matched-LR IND runs" in FLAT
    assert f"S005's 3 g-checks over {v['c1']} h ({v['c1r']} h a run; estimate {v['pS005']}), S005 and S004's 6 over " \
           f"{v['c2']} h, the 9 before S006 over {v['c3']} h (estimate {v['c3e']} h, {v['pct']}% over)" in FLAT
    assert f"its 3 selection runs ({v['gS006']} h estimated)" in FLAT


def test_stage2_plan_is_order_2_between_the_smokes_mark_and_its_own(V):
    lines = [ln.rstrip("\n") for ln in open(PLAN)]
    assert lines[0].startswith("# ") and lines[1] == "wait_mark SCREENS SCREENS STAGE 2 SMOKES RECORDED"
    assert lines[2:-1] == [f"train {n}" for n in S2] and lines[-1] == "mark SCREENS STAGE 2 SELECTION DONE"
    assert [n[:4] for n in S2[::3]] == ["s005", "s004", "s007", "s006"] and len(lines) == 15
    assert "The mark file is written on the PC as ~/planck/runs/SCREENS/marks/SCREENS_STAGE_2_SMOKES_RECORDED" in \
        " ".join(section("SCREENS_STAGE_2_SMOKES_RECORDED").split())          # the wait line's mark, signed off
    assert PLAN_SEQUENCE[3:5] == ["stage1_seeds", "stage2_select"]
    own, base = (L.flat(L.resolve(L.find(n)[0])) for n in ("s005_base_s101", "base_s101"))
    for n in S2:
        path = L.find(n)
        f = L.flat(L.resolve(path[0]))
        sid = n[:4].upper()
        reg = set(L.SCREENS[sid]["arms"][n.split("_")[1]]) | (set(L.LRK) if "_g1_" not in n else set())
        assert len(path) == 1 and L.check(path[0], P) == [] and f["train.compile"] is False
        assert L.diff(f, own if sid == "S005" else base) - {"name", "out_dir", "seed"} == reg, n
        assert (f["train.doc_attn"], f["train.micro_batch"], f["train.grad_accum"]) == \
            (("mask", 8, 2) if sid == "S005" else ("varlen", 16, 1)), n
    v = V["fmt"]
    assert f"stage2_select.txt as written in BUILT (620c77c; unchanged since, sha256 {L.sha256(PLAN)[:16]}...)" in FLAT
    assert f"Estimate {v['g2']} h (S005 {v['gS005']}, S004 {v['gS004']}, S007 {v['gS007']}, S006 {v['gS006']})" in FLAT
    assert "plans/CURRENT should name stage2_select (not edited here" in FLAT


def test_every_number_in_the_entry_is_pinned(V):
    exp = set(V["fmt"].values()) | {"0.5", "101", "102", "112", "136", "763", "5.483", "1.622", "0.812", "148", "022", "020"}
    exp |= {f"{x:,}" for x in (7344, 7497, L.STEPS)} | {str(x) for x in PASSED + (WHOLE,)}
    exp |= set(re.findall(r"\d+\.\d+", "\n".join(hours_rows(V["runs"]) + cap_rows(V))))
    hexes = set(PCSHA) | {L.sha256(PLAN)[:16], "93bc10b", "620c77c"}
    text = re.sub(r"\b[a-z]\w*_[\w.<>]*", " ", SEC)        # run names, file names, config keys
    found = re.findall(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,16}\b", text)
    assert len(found) >= 6 and set(found) <= hexes, set(found) - hexes
    text = re.sub(r"\b[0-9a-f]{7,16}\b", " ", text)
    assert set(re.findall(r"\b\d{4}-\d\d-\d\d\b", text)) == {"2026-10-05", "2026-10-06"}
    assert set(re.findall(r"\b\d\d:\d\d(?::\d\d)?\b", text)) == {QSTART, QEND, ENTRY_TIME, MARK_SMOKES, "00:36"}
    text = re.sub(r"\b\d{4}-\d\d-\d\d\b|\b\d\d:\d\d(?::\d\d)?\b", " ", text)
    nums = re.findall(r"(?<![\w.])(\d[\d,]*\d(?:\.\d+)?|\d\.\d+)(?![\w])", text)
    big = {n for n in nums if "." in n or "," in n or len(n) >= 3}
    assert len(big) > 50 and big <= exp, sorted(big - exp)
    assert len(M.MUTANTS_S2LAUNCH) == NEW and all(x in M.MUTANTS for x in M.MUTANTS_S2LAUNCH)
    assert len(M.MUTANTS) >= WHOLE                   # later entries add their own mutants
    for phrase in (f"tests {PASSED[1]} passed ({PASSED[0]} before", f"{NEW} new mutants", f"{NEW} of {NEW} killed; "
                   f"the whole file {WHOLE} of {WHOLE} killed", "22 queued slots", "68 measured runs", "<= 25 h",
                   "k (2, E3's k in every class)", "each at g 0.5, 1, 2, seed 1", "14 ok"):
        assert phrase in FLAT, phrase


def test_correction_the_time_outside_the_lock_holds(V):
    """CORRECTIONS (2026-10-06): the entry's "lock waits" figure is the wall clock outside the 12 holds; the queue log
    splits it into waiting for gpu.lock and the gaps before each lock request. LOCKREQ's held times are the
    measured_hours.tsv starts, so the fixture is the committed data's own log."""
    day = lambda t: ("2026-10-06 " if t < "12" else "2026-10-05 ") + t  # noqa: E731
    sec = lambda a, b: round(H.hours_between(day(a), day(b)) * 3600)  # noqa: E731
    rows = {ln.split("\t")[1]: ln.rstrip("\n").split("\t") for ln in open(H.TSV) if ln.startswith("run\t")}
    assert list(LOCKREQ) == RUNS1 and all(rows[n][3] == day(h) for n, (_, h) in LOCKREQ.items())
    waits = [sec(w, h) for w, h in LOCKREQ.values()]
    ends = [QSTART] + [rows[n][4][11:] for n in RUNS1]
    before = [sec(e, w) for e, (w, _) in zip(ends, LOCKREQ.values())]
    assert sum(waits) + sum(before) + sec(ends[-1], QEND) == round(V["wall_s"] - 3600 * sum(V["runs"][n] for n in RUNS1))
    c = " ".join(section("CORRECTIONS (2026-10-06 07:30 EDT, Claude; to STAGE 1 SEEDS DONE").split())
    assert (f'"lock waits {V['fmt']['outside']} h in all" is the wall clock outside the 12 lock holds, '
            f"{sum(waits) + sum(before)} s, not time spent waiting for gpu.lock: \"waiting for gpu.lock\" to \"gpu.lock "
            f"held\" takes {sum(waits)} s over the 12 runs ({min(waits)} or {max(waits)} s each); the other "
            f"{sum(before)} s run from the queue start to the first lock request ({before[0]} s) and from each run's "
            f"status line to the next lock request ({sum(before[1:])} s, {min(before[1:])} or {max(before[1:])} s each)") in c
    assert "now names it s2_s005_config_micro_16x1" in c and [m[0] for m in M.MUTANTS_S2LAUNCH].count("s2_s005_config_micro_16x1") == 1
