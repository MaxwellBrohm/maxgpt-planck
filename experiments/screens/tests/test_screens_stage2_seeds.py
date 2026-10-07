"""SCREENS S006 EXTENSION RESULT / STAGE 2 SEEDS LAUNCH CHECK (2026-10-06): S006's g 4 extension is measured once in
measured_hours.tsv; S006's pick on {0.5, 1, 2, 4} through analyze.picks on the recorded values; C6's two IND runs are
what screens.py seeds writes for that pick; ORDER's cap check before seed set 101 passes 25 h with them and the cut rule
cuts S006 alone; the cut is recorded; plans/stage2_seeds.txt and its configs are screens.py seeds --cut S006 output,
byte for byte; plans/CURRENT; every number in the entry pinned (tables generated from the data; every other decimal,
comma or 3+ digit number must be one V computes). No verdict, no model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_stage2_seeds.py
"""
import contextlib, io, json, os, re, sys, pytest  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze, analyze_lib as AL, mutation_screens as M, screens_hours as H, screens_lib as L  # noqa: E401,E402
from s2select_record import SETS, VALS as VALS12  # noqa: E402
from s2seeds_record import ENTRY_TIME, EXT_VALS, LOCK, PCSHA, QEND, QSTART, TOKS, TRAIN, WAIT_S  # noqa: E402
from test_screens_configs import (LATER_PLANS, PLAN_SEQUENCE, recorded_cuts, seeds2_wait,  # noqa: E402
                                  stage2_seed_configs)
from test_screens_refusals import scratch  # noqa: E402,F401  (the scratch config tree fixture)
from test_screens_s003_stage_c import section  # noqa: E402

P = L.params()
EXT, PLAN = "s006_mtp_g4_s1", os.path.join(L.HERE, "plans", "stage2_seeds.txt")
PASSED = (427, 460)               # experiments/screens/tests before and after this entry (Mac CPU)
WHOLE, NEW = 230, 31              # mutation_screens.py: the whole file and this entry's mutants (mutants_s2seeds.py)
SEC = section("S006 EXTENSION RESULT / STAGE 2 SEEDS LAUNCH CHECK")
FLAT = " ".join(SEC.split())
PICKS = ["--pick=S005.forget=1", "--pick=S004.canonac=1", "--pick=S007.smear=1"]
NEW_CFG = {f"{a}_g1_s{s}" for a in ("s005_forget", "s004_canonac", "s007_smear") for s in (101, 102)}


def write_runs(root, vals):
    for name, v in vals.items():
        os.makedirs(root / name, exist_ok=True)
        with open(root / name / "bpb.jsonl", "w") as f:
            for s, x in zip(SETS, v):
                f.write(json.dumps({"ckpt": "final_00007630.pt", "split": "all", "set": s, "bpb": x}) + "\n")
    return analyze.picks(str(root))


def plan_runs(path=PLAN):
    return [ln.split()[1] for ln in open(path) if ln.startswith("train ")]


@pytest.fixture(scope="module")
def V(tmp_path_factory):
    runs, _ = H.load()
    with contextlib.redirect_stdout(io.StringIO()):
        r = H.report(P)
    t, f = r["tally"], (lambda x, n=3: f"{x:.{n}f}")
    q, m, x = t["queued"], r["measured_runs"] + r["other"], runs[EXT]
    full = lambda s, a: H.run_h(L.STEPS, H.mult(s, a))  # noqa: E731
    per = {"own": full("S005", "base"), "S005": full("S005", "forget"), "S004": full("S004", "canonac"),
           "S007": full("S007", "smear"), "S006": full("S006", "mtp")}
    ind = AL.cap_cut({**q, "S006": q["S006"] + 2 * per["S006"]}, m)
    seed = per["own"] + per["S005"] + per["S004"] + per["S007"]         # one seed set after the cut
    room = 25 - ind["total_after"]
    m6 = [runs[f"s006_mtp_g{g}_s1"] for g in ("0.5", "1", "2", "4")]
    g1, g2 = VALS12["s006_mtp_g1_s1"], VALS12["s006_mtp_g2_s1"]
    assert min(VALS12[f"s006_mtp_g{g}_s1"][1] for g in ("0.5", "1", "2")) == g1[1] < EXT_VALS[1]     # g 1: PROSE best
    fmt = {"x": f(x), "train": f(TRAIN / 3600), "toks": f"{TOKS / 1000:.0f}k", "est": f(per["S006"]),
           "total": f(sum(runs.values())), "earlier": f(sum(runs.values()) - x), "mo": f(r["other"]), "m": f(m),
           "qs": f(sum(q.values())), "ind": f(2 * per["S006"]), "q6": f(q["S006"]), "q6i": f(q["S006"] + 2 * per["S006"]),
           "before": f(ind["total_before"]), "after": f(ind["total_after"]), "room": f(room),
           "over": f(ind["total_before"] - 25), "tot": f(r["cap"]["total_before"]), "narrow": f(r["narrow"]["total_before"]),
           "s6m": f(sum(m6)), "seed": f(seed), "pct": f(100 * room / seed, 1), "queued8": f(sum(q.values()) - q["S006"]),
           "lo6": f(min(m6[:3])), "hi6": f(max(m6[:3])), "c4": f(EXT_VALS[0], 5), "c4r": f(round(EXT_VALS[0], 4), 4),
           "c2": f(g2[0], 5), "d42": f(EXT_VALS[0] - g2[0], 5), "rel": f(100 * (g2[1] / g1[1] - 1)), "pbest": f(g1[1], 5),
           "d21": f(g1[0] - g2[0], 5), "held": LOCK}
    fmt.update({f"p{k}": f(v) for k, v in per.items()})
    return {"runs": runs, "r": r, "q": q, "m": m, "x": x, "per": per, "ind": ind, "fmt": fmt}


def test_the_extension_gives_s006_an_interior_pick_and_the_others_stand(tmp_path):
    p = write_runs(tmp_path, {**VALS12, EXT: EXT_VALS})
    for k in ("S005.forget", "S004.canonac", "S007.smear"):
        assert (p[k]["pick"], p[k]["decided"], p[k]["extend_with"]) == (1.0, True, []), k
    r = p["S006.mtp"]
    assert (r["pick"], r["argmin"], r["guard_moved"], r["at_edge"], r["extend_with"], r["decided"], r["axis"],
            r["extensions_used"]) == (2.0, 2.0, False, None, [], True, [0.5, 1.0, 2.0, 4.0], 1)
    assert round(EXT_VALS[0], 4) > round(VALS12["s006_mtp_g2_s1"][0], 4)        # g 4 above g 2 at 4 decimals


def pick_row():
    cells = [f"{v[0]:.5f} / {v[1]:.5f}" for v in [VALS12[f"s006_mtp_g{g}_s1"] for g in ("0.5", "1", "2")] + [EXT_VALS]]
    return "    S006 mtp     " + "  ".join(cells) + "  g 2"


def test_entry_pick_and_extension_run(V):
    v = V["fmt"]
    assert "\n" + pick_row() + "\n" in SEC
    assert f"s006_mtp_g4_s1 held the lock {v['x']} h (train {v['train']} h, {v['toks']} tok/s), against its {v['est']} h " \
           f"estimate; S006's g-checks held it {v['lo6']} to {v['hi6']} h" in FLAT
    assert f"81 runs, {v['total']} h (the 80 earlier {v['earlier']} + the extension {v['x']})" in FLAT
    for phrase in ('"done s006_mtp_g4_s1 rc 0" on the first try', 'status.jsonl: "done and scored"',
                   "(763 records, the last at step 7,630)", "diag.jsonl holds 50 IND readings",
                   "its 80 earlier run lines are unchanged (diff: the header's two sha256 and 1 new line)",
                   "at_edge None, extend_with [], extensions_used 1, decided True"):
        assert phrase in FLAT, phrase


def test_the_extension_is_measured_once(V):
    runs, rows = V["runs"], {ln.split("\t")[1]: ln.rstrip("\n").split("\t") for ln in open(H.TSV) if ln.startswith("run\t")}
    assert len(runs) == 81 and set(V["r"]["tally"]["extra"]) == {EXT} and H.slot(EXT) == ("S006", None)
    assert rows[EXT][4] == "2026-10-06 " + QEND and rows[EXT][3] > "2026-10-06 " + QSTART
    assert f"(sha256 {PCSHA[0]}... and {PCSHA[1]}...)" in open(H.TSV).read()
    assert round(H.hours_between(rows[EXT][3], rows[EXT][4]) * 3600) == round(V["x"] * 3600)


def cap_rows(V):
    t, out = V["r"]["tally"], []
    for scr in ["BASE"] + list(L.SCREENS):
        nm = sum(1 for n in V["runs"] if H.slot(n)[0] == scr)
        m, q = t["measured"].get(scr, 0.0), t["queued"].get(scr, 0.0)
        out.append(f"    {scr:<8}{nm:>2}{'':13}{m:.3f}   {t['n_queued'].get(scr, 0):<12}{q:.3f}  {m + q:.3f}")
    return out


def test_ind_runs_are_the_generators_and_the_cap_cuts_s006_alone(V, scratch):
    """The would-be plan (S006 at its pick g 2) holds C6's two IND runs at g 1; with them queued the total passes 25 h,
    ORDER's order cuts S006 first and the rest fits, so nothing else is cut; with S006 queued but no IND run it fits."""
    tmp_path, screens = scratch
    screens.main(["seeds", "--stage", "2", *PICKS, "--pick=S006.mtp=2"])
    would = plan_runs(str(tmp_path / "plans" / "stage2_seeds.txt"))
    assert [r for r in would if r.startswith("s006_")] == [f"s006_mtp_g{g}_s{s}" for s in (101, 102) for g in (2, 1)]
    extra = [r for r in would if r not in V["runs"] and H.slot(r)[1] not in V["r"]["tally"]["registered"]]
    keys = [H.slot(r)[1] for r in would if r not in V["runs"]]
    assert extra == [] and len(keys) == 12 and len(set(keys)) == 10       # two runs share a seed slot: the IND runs
    ind = V["ind"]
    assert ind["total_before"] > 25 and ind["cut"] == ["S006"] and ind["fits"] and V["r"]["cap"]["cut"] == []
    # not vacuous: had the rest still passed 25 h, the rule would have gone on to S007 (S003 has nothing queued)
    assert AL.cap_cut({**V["q"], "S006": V["q"]["S006"] + 2 * V["per"]["S006"]}, V["m"], cap=V["ind"]["total_after"] - 1e-6)[
        "cut"] == ["S006", "S007"]
    assert recorded_cuts() == {"S006"}


def test_entry_cap_check_and_cut(V):
    v = V["fmt"]
    for row in cap_rows(V):
        assert "\n" + row + "\n" in SEC, row
    assert f"Measured {v['total']} h of runs + {v['mo']} h other = {v['m']} h; queued 10 of 90 registered runs " \
           f"{v['qs']} h; total {v['tot']} h (runs only, reported: {v['narrow']} h)" in FLAT
    assert f"{v['m']} + {v['qs']} + {v['ind']} = {v['before']} h, {v['over']} h over 25 h" in FLAT
    assert f"CUT BY ORDER'S CAP RULE BEFORE THE STAGE 2 SEED SETS: S006 (its queued {v['q6i']} h: 2 seed runs " \
           f"{v['q6']} h and 2 IND runs {v['ind']} h)" in FLAT
    assert f"{v['before']} - {v['q6i']} = {v['after']} h <= 25 h, {v['room']} h of headroom" in FLAT
    assert f"4 seed-1 selection runs ({v['s6m']} h measured)" in FLAT


def test_plan_is_the_generators_with_the_cut_and_current_names_it(V, scratch):
    tmp_path, screens = scratch
    screens.main(["seeds", "--stage", "2", *PICKS, "--cut", "S006", "--wait", seeds2_wait()[10:]])
    assert open(tmp_path / "plans" / "stage2_seeds.txt").read() == open(PLAN).read()
    for n in NEW_CFG:
        assert open(L.find(n)[0]).read() == open(tmp_path / os.path.basename(os.path.dirname(os.path.dirname(L.find(n)[0])))
                                                 / "configs" / f"{n}.yaml").read(), n
    lines = open(PLAN).read().splitlines()
    want = [f"{a}_s{s}" for s in (101, 102) for a in ("base", "s005_base", "s005_forget_g1", "s004_canonac_g1",
                                                         "s007_smear_g1")]
    assert plan_runs() == want and lines[1] == "wait_mark SCREENS SCREENS S006 EXTENSION DONE"
    assert lines[-1] == "mark SCREENS STAGE 2 SEEDS DONE" and len(lines) == 13
    assert stage2_seed_configs() == NEW_CFG and PLAN_SEQUENCE[-1] == "stage2_s006x" and LATER_PLANS == ["stage2_seeds"]
    cur = [ln.strip() for ln in open(os.path.join(L.HERE, "plans", "CURRENT")) if not ln.startswith("#")]
    assert cur == ["stage2_seeds"]
    v = V["fmt"]
    assert f"stage2_seeds.txt (sha256 {L.sha256(PLAN)[:16]}...; written by screens.py seeds), its 13 lines:" in FLAT
    assert " ".join(lines[0].split()) + " [one line in the file]" in FLAT
    assert all("\n      " + ln + "\n" in SEC for ln in lines[1:])
    assert f"Estimate {v['queued8']} h (8 runs; the 2 BASE lines are skipped by the queue" in FLAT


def test_entry_pick_c6_and_inside_the_plan(V):
    v = V["fmt"]
    assert f"g 4 reads F(CHAT) {v['c4']} ({v['c4r']} at 4 decimals), {v['d42']} above g 2's {v['c2']}, so the argmin " \
           f"stays g 2; g 2's PROSE is {v['rel']}% above the axis best ({v['pbest']} at g 1), inside the 1% guard" in FLAT
    assert f"best minus second is still g 2 against g 1, {v['d21']}, \"not resolved at one seed\"" in FLAT
    assert f"s006_mtp_g1_s101 and s006_mtp_g1_s102, {v['pS006']} h each, {v['ind']} h" in FLAT
    assert "(run in a scratch export of the index: 14 train lines, 12 not yet run; the 2 IND runs share S006's 2 seed " \
           "slots, so they take no slot and count on top)" in FLAT
    assert f"(S005's own BASE {v['pown']} h, S005 forget {v['pS005']} h, S004 canonac {v['pS004']} h, S007 smear " \
           f"{v['pS007']} h: {v['seed']} h estimated) held the lock over their estimates by more than the {v['room']} h " \
           f"of headroom ({v['pct']}%)" in FLAT
    assert f'free ("waiting for gpu.lock" and "gpu.lock held" at {v["held"]})' in FLAT and WAIT_S == 0


def test_new_configs_differ_from_their_base_only_in_registered_keys(capsys):
    import screens
    for n in sorted(NEW_CFG):
        sid = n.split("_")[0].upper()
        f = L.flat(L.resolve(L.find(n)[0]))
        b = L.flat(L.resolve(L.find(f"{sid.lower()}_base_s{n[-3:]}" if L.engine(sid) else f"base_s{n[-3:]}")[0]))
        reg = L.SCREENS[sid]["arms"][n.split("_")[1]]
        assert L.diff(f, b) == {"name", "out_dir"} | set(reg) and all(f[k] == v for k, v in reg.items()), n
        assert [f[k] for k in L.LRK] == [P["lr5"][0]] * 3 and f["seed"] == int(n[-3:]) and L.check(L.find(n)[0], P) == []
    assert screens.main(["check", *plan_runs()]) == 0 and capsys.readouterr().out.count("ok      ") == 10
    assert screens.main(["check"]) == 0 and capsys.readouterr().out.count("ok      ") == 89 == len(L.all_configs())
    assert not [p for p in L.all_configs() if os.path.basename(p).startswith("s006_") and "_s10" in p]
    assert "10 ok on the plan's runs, 89 ok over every config" in FLAT


def test_s006_notes_record_the_cut(V):
    v, notes = V["fmt"], open(os.path.join(L.EXPD, "S006_mtp_aux", "notes.txt")).read()
    blk = " ".join(notes[notes.index("\nCUT (2026-10-06, SCREENS.txt S006 EXTENSION RESULT"):].split())
    for phrase in (f"(F(CHAT) {v['c4']})", f"read {v['before']} h, over 25 h", f"(its {v['q6i']} h queued: 2 seed runs, "
                   f"2 IND runs), leaving {v['after']} h", f"selection runs ({v['s6m']} h measured)",
                   'P-105\'s bet stays "not run" in the ledger (ORDER)'):
        assert phrase in blk, phrase


def test_every_number_in_the_entry_is_pinned(V):
    exp = set(V["fmt"].values()) | {"0.5", "101", "102", "105", "148", "022", "763", "7630", "7,630", "7,344", "7,497"}
    exp |= {str(x) for x in PASSED + (WHOLE,)}
    exp |= set(re.findall(r"\d+\.\d+", "\n".join(cap_rows(V) + [pick_row()])))
    hexes = set(PCSHA) | {L.sha256(PLAN)[:16], "53da0f7", "93bc10b"}
    text = re.sub(r"\b[a-z]\w*_[\w.<>]*", " ", SEC)        # run names, file names, config keys
    found = re.findall(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,16}\b", text)
    assert set(found) == hexes, set(found) ^ hexes
    text = re.sub(r"\b[0-9a-f]{7,16}\b", " ", text)
    assert set(re.findall(r"\b\d{4}-\d\d-\d\d\b", text)) == {"2026-10-06"}
    assert set(re.findall(r"\b\d\d:\d\d(?::\d\d)?\b", text)) == {QSTART, QEND, ENTRY_TIME, V["fmt"]["held"]}
    text = re.sub(r"\b\d{4}-\d\d-\d\d\b|\b\d\d:\d\d(?::\d\d)?\b", " ", text)
    nums = re.findall(r"(?<![\w.])(\d[\d,]*\d(?:\.\d+)?|\d\.\d+)(?![\w])", text)
    big = {n for n in nums if "." in n or "," in n or len(n) >= 3}
    assert len(big) > 50 and big <= exp, sorted(big - exp)
    assert len(M.MUTANTS_S2SEEDS) == NEW and all(x in M.MUTANTS for x in M.MUTANTS_S2SEEDS)
    assert len(M.MUTANTS) >= WHOLE                   # later entries add their own mutants
    for phrase in (f"tests {PASSED[1]} passed ({PASSED[0]} before", f"{NEW} new mutants", f"{NEW} of {NEW} killed; the "
                   f"whole file {WHOLE} of {WHOLE} killed", "queued 10 of 90 registered runs", "<= 25 h", "81 runs",
                   "its 13 lines", "seed set 101", "STAGE 2 SEEDS LAUNCH CHECK (2026-10-06 " + ENTRY_TIME + " EDT"):
        assert phrase in FLAT, phrase
