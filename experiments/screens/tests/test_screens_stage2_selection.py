"""SCREENS STAGE 2 SELECTION RESULT (2026-10-06): the 12 stage 2 g-checks give S005, S004 and S007 g 1 and S006 an edge
pick (g 2, one extension asked: g 4) through analyze.picks on the recorded values; their lock hours are the 12 new lines
of measured_hours.tsv; ORDER's cap check before S006's extension fits; the extension config and the plan stage2_s006x;
the arithmetic of what C6's IND runs and a second extension would do; every number in the entry pinned (tables generated
from the data; every other decimal, comma or 3+ digit number must be one V computes). No verdict, no model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_stage2_selection.py
"""
import contextlib, io, json, os, re, sys, pytest  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze, analyze_lib as AL, mutation_screens as M, screens, screens_hours as H, screens_lib as L  # noqa: E401,E402
from test_screens_configs import EXT2, LATER_PLANS, PLAN_SEQUENCE, stage2_seed_configs  # noqa: E402
from test_screens_s003_stage_c import section  # noqa: E402

P = L.params()
from s2select_record import (ENTRY_TIME, PCSHA, QEND, QSTART, S2, SETS, TOKS, TRAIN,  # noqa: E402
                             VALS, WAITS)
ARMS = {"S005": ("forget", "S005 forget"), "S004": ("canonac", "S004 canonac"), "S007": ("smear", "S007 smear"),
        "S006": ("mtp", "S006 mtp")}
EXT, PLAN = "s006_mtp_g4_s1", os.path.join(L.HERE, "plans", "stage2_s006x.txt")
PASSED = (415, 427)             # experiments/screens/tests before and after this entry (Mac CPU)
WHOLE, NEW = 199, 13            # mutation_screens.py: the whole file and this entry's mutants (mutants_s2select.py)
SEC = section("STAGE 2 SELECTION RESULT")
FLAT = " ".join(SEC.split())


def write_runs(root, vals):
    for name, v in vals.items():
        os.makedirs(root / name, exist_ok=True)
        with open(root / name / "bpb.jsonl", "w") as f:
            for s, x in zip(SETS, v):
                f.write(json.dumps({"ckpt": "final_00007630.pt", "split": "all", "set": s, "bpb": x}) + "\n")
    return analyze.picks(str(root))


@pytest.fixture(scope="module")
def V(tmp_path_factory):
    runs, _ = H.load()
    with contextlib.redirect_stdout(io.StringIO()):
        r = H.report(P)
    t, f = r["tally"], (lambda x, n=3: f"{x:.{n}f}")
    q, m = t["queued"], r["measured_runs"] + r["other"]
    per = {s: H.run_h(L.STEPS, H.mult(s, a)) for s, (a, _) in ARMS.items()}
    by = {s: [runs[n] for n in S2 if n.startswith(s.lower())] for s in ARMS}
    est, lock, train = per["S006"], sum(runs[n] for n in S2), sum(TRAIN.values()) / 3600
    ext = AL.cap_cut({**q, "S006": q["S006"] + est}, m)
    ind = AL.cap_cut({**q, "S006": q["S006"] + 2 * est}, m)
    ind1 = AL.cap_cut({**q, "S006": q["S006"] + 2 * est}, m + est)
    own2 = 2 * H.run_h(L.STEPS, H.mult("S005", "base"))
    chat = {n: v[0] for n, v in VALS.items()}
    marg = {s: sorted(x for n, x in chat.items() if n.startswith(s.lower()))[:2] for s in ARMS}
    p6 = {g: VALS[f"s006_mtp_g{g}_s1"][1] for g in ("0.5", "1", "2")}
    mde = json.load(open(os.path.join(L.E3D, "results.json")))["noise_5M_250M"]["F(CHAT)"]["MDE"]["1"]["paired"]
    fmt = {"lock": f(lock), "train": f(train), "gaps": f(lock - train), "est": f(3 * sum(per.values())),
           "under": f(3 * sum(per.values()) - lock), "wall": f(H.hours_between("2026-10-06 " + QSTART, "2026-10-06 " + QEND), 2),
           "total": f(sum(runs.values())), "earlier": f(sum(runs.values()) - lock), "mo": f(r["other"]), "m": f(m),
           "qs": f(sum(q.values())), "seed8": f(sum(q.values()) - own2), "own2": f(own2), "tot": f(r["cap"]["total_before"]),
           "room": f(25 - r["cap"]["total_after"]), "xt": f(ext["total_before"]), "xroom": f(25 - ext["total_after"]),
           "xnarrow": f(r["narrow"]["total_before"] + est), "ind": f(2 * est), "ind0": f(ind["total_before"]),
           "q6": f(q["S006"] + 2 * est), "after0": f(ind["total_after"]), "after1": f(ind1["total_after"]),
           "g8": f(25 - ext["total_before"]), "fac": f(H.mult("S006", "mtp")), "mde": f(mde, 5),
           "rel": f(100 * (p6["2"] / min(p6.values()) - 1)), "c2": f(chat["s006_mtp_g2_s1"], 5),
           "c1": f(chat["s006_mtp_g1_s1"], 5), "c2r": f(round(chat["s006_mtp_g2_s1"], 4), 4),
           "c1r": f(round(chat["s006_mtp_g1_s1"], 4), 4), "p2": f(p6["2"], 5), "pbest": f(min(p6.values()), 5),
           "lr4": f"{4 * P['lr5'][0]:.3f}"}
    fmt.update({f"m{s}": f(sum(v)) for s, v in by.items()} | {f"e{s}": f(3 * v) for s, v in per.items()})
    fmt.update({f"p{s}": f(v) for s, v in per.items()} | {f"lo{s}": f(min(v)) for s, v in by.items()})
    fmt.update({f"hi{s}": f(max(v)) for s, v in by.items()} | {f"d{s}": f(b - a, 5) for s, (a, b) in marg.items()})
    return {"runs": runs, "r": r, "q": q, "m": m, "est": est, "ext": ext, "ind": ind, "fmt": fmt}


def test_recorded_values_give_three_settled_picks_and_one_edge(tmp_path):
    p = write_runs(tmp_path, VALS)
    for k in ("S005.forget", "S004.canonac", "S007.smear"):
        r = p[k]
        assert (r["pick"], r["argmin"], r["guard_moved"], r["at_edge"], r["extend_with"], r["decided"]) == \
            (1.0, 1.0, False, None, [], True), k
    r = p["S006.mtp"]
    assert (r["pick"], r["argmin"], r["guard_moved"], r["at_edge"], r["extend_with"], r["decided"], r["axis"],
            r["extensions_used"]) == (2.0, 2.0, False, "high", [4.0], False, [0.5, 1.0, 2.0], 0)
    for s, (a, _) in ARMS.items():                  # the g 1 picks are each axis's PROSE best (guard not moved)
        pr = {g: VALS[f"{s.lower()}_{a}_g{g}_s1"][1] for g in ("0.5", "1", "2")}
        assert min(pr, key=pr.get) == "1"


def test_the_edge_and_its_extensions_follow_e2s_rule(tmp_path):
    """Not vacuous: S006's edge reading is the 4-decimal comparison (g 2 at g 1's value ties to the lower LR, g 1); with
    g 4 present the axis grows; a g 4 argmin asks for g 8, the last; whatever g 4 gives, the pick is never g 1."""
    tie = dict(VALS, s006_mtp_g2_s1=(1.16349,) + VALS["s006_mtp_g2_s1"][1:])
    r = write_runs(tmp_path / "tie", tie)["S006.mtp"]
    assert (r["pick"], r["extend_with"], r["decided"]) == (1.0, [], True)
    for i, (c4, p4) in enumerate((c, pr) for c in (1.1500, 1.16335, 1.16345, 1.1700, 1.3000) for pr in (1.38, 1.403, 1.43)):
        r = write_runs(tmp_path / f"g4_{i}", {**VALS, EXT: (c4, p4, 1.3, 1.5, 1.3)})["S006.mtp"]
        assert r["axis"] == [0.5, 1.0, 2.0, 4.0] and r["extensions_used"] == 1 and r["pick"] != 1.0, (c4, p4)
        assert r["extend_with"] == ([8.0] if r["pick"] == 4.0 else []), (c4, p4, r)
    two = {**VALS, EXT: (1.150, 1.40, 1.3, 1.5, 1.3), "s006_mtp_g8_s1": (1.140, 1.40, 1.3, 1.5, 1.3)}
    r = write_runs(tmp_path / "g8", two)["S006.mtp"]
    assert (r["pick"], r["at_edge"], r["extensions_used"], r["extend_with"], r["decided"]) == (8.0, "high", 2, [], True)


def hours_rows(runs):
    cell = lambda n: f"{n:<27}{runs[n]:.3f} {TRAIN[n] / 3600:.3f}  {TOKS[n] / 1000:.0f}k"  # noqa: E731
    return ["    " + "  | ".join(cell(n) for n in S2[i:i + 2]) for i in range(0, 12, 2)]


def test_the_12_runs_are_measured_once_at_their_lock_hours(V):
    runs, v = V["runs"], V["fmt"]
    assert set(S2) <= set(runs) and len(runs) == 80 and V["r"]["tally"]["extra"] == {}
    rows = {ln.split("\t")[1]: ln.rstrip("\n").split("\t") for ln in open(H.TSV) if ln.startswith("run\t")}
    assert all(rows[n][3] == "2026-10-06 " + held for n, (_, held) in WAITS.items())     # the lock hours start after
    assert rows[S2[-1]][4] == "2026-10-06 " + QEND and min(rows[n][3] for n in S2) > "2026-10-06 " + QSTART
    assert f"(sha256 {PCSHA[0]}... and {PCSHA[1]}...)" in open(H.TSV).read()
    waits = [round(H.hours_between("2026-10-06 " + a, "2026-10-06 " + b) * 3600) for a, b in WAITS.values()]
    assert f"waited for gpu.lock {waits[0]} s before s005_forget_g0.5_s1, {waits[1]} s before s005_forget_g1_s1 and " \
           f"{waits[2]} s before s006_mtp_g2_s1 (outside the lock hours)" in FLAT
    for row in hours_rows(runs):
        assert "\n" + row + "\n" in SEC, row
    assert f"Total {v['lock']} GPU hours measured for the 12 g-checks (S005 {v['mS005']}, S004 {v['mS004']}, S007 " \
           f"{v['mS007']}, S006 {v['mS006']}; {v['train']} h training, {v['gaps']} h scoring and gaps), {v['under']} h " \
           f"under their {v['est']} h estimate (S005 {v['eS005']}, S004 {v['eS004']}, S007 {v['eS007']}, S006 " \
           f"{v['eS006']}); {v['wall']} h wall clock" in FLAT
    assert f"S005 forget {v['loS005']} h ({v['pS005']}), S004 canonac {v['loS004']} to {v['hiS004']} h ({v['pS004']}), " \
           f"S007 smear {v['hiS007']} h ({v['pS007']}), S006 mtp {v['loS006']} to {v['hiS006']} h ({v['pS006']})" in FLAT
    assert v["loS005"] == v["hiS005"] and v["loS007"] == v["hiS007"]
    assert f"80 runs, {v['total']} h (the 68 earlier {v['earlier']} + the stage 2 g-checks {v['lock']})" in FLAT
    for phrase in ('All 12 runs "done ... rc 0" on the first try', 'status.jsonl: 12 "done and scored"',
                   "(763 records a run, the last at step 7,630;", "each diag.jsonl holds 50 IND readings",
                   "its 68 earlier run lines are unchanged (diff: the header's two sha256 and 12 new lines)"):
        assert phrase in FLAT, phrase


def pick_rows():
    out = []
    for s, (a, lab) in ARMS.items():
        cells = [f"{VALS[f'{s.lower()}_{a}_g{g}_s1'][0]:.5f} / {VALS[f'{s.lower()}_{a}_g{g}_s1'][1]:.5f}" for g in ("0.5", "1", "2")]
        out.append(f"    {lab:<18}" + "    ".join(cells) + f"    {'g 2, high edge' if s == 'S006' else 'g 1'}")
    return out


def test_entry_pick_table_and_readings(V):
    v = V["fmt"]
    for row in pick_rows():
        assert "\n" + row + "\n" in SEC, row
    assert f"S006: the argmin is g 2, F(CHAT) {v['c2']} against g 1's {v['c1']} ({v['c2r']} and {v['c1r']} at 4 decimals); " \
           f"its PROSE {v['p2']} is {v['rel']}% above the axis best ({v['pbest']} at g 1), inside the 1% guard" in FLAT
    assert "e2pick at_edge high, extend_with [4.0], extensions_used 0, decided False" in FLAT
    assert f"MDE at k = 1, {v['mde']}): S005 {v['dS005']}, S004 {v['dS004']}, S007 {v['dS007']}, S006 {v['dS006']}" in FLAT
    assert "(at_edge None, extend_with [], decided): settled at g = 1" in FLAT


def test_extension_config_is_s006_at_4_x_lr5(V, capsys):
    path = L.find(EXT)
    assert len(path) == 1 and os.sep + os.path.join("S006_mtp_aux", "configs") + os.sep in path[0] and EXT2 == {EXT}
    g2 = open(L.find("s006_mtp_g2_s1")[0]).read().replace("g 2 x LR5, seed 1: C3 g-check (", "g 4 x LR5, seed 1: C3 "
                                                          "g-check extension (").replace("s006_mtp_g2_s1", EXT)
    assert open(path[0]).read() == g2.replace("{lr: 0.006, embed_lr: 0.006, scalar_lr: 0.006}", "{lr: 0.012, embed_lr: "
                                              "0.012, scalar_lr: 0.012}")
    f, b = L.flat(L.resolve(path[0])), L.flat(L.resolve(L.find("base_s101")[0]))
    assert L.diff(f, b) == {"name", "out_dir", "seed", "train.mtp", "train.mtp_weight", *L.LRK}
    assert [f[k] for k in L.LRK] == pytest.approx([4 * P["lr5"][0]] * 3) and P["lr5"][1] == 1.0 and f["seed"] == 1
    assert screens.main(["check", EXT]) == 0 and capsys.readouterr().out.count("ok      ") == 1
    assert len(L.all_configs()) == 83 + len(stage2_seed_configs()) and H.slot(EXT) == ("S006", None)
    v = V["fmt"]
    assert f"s006_mtp_g4_s1.yaml (sha256 {L.sha256(path[0])[:16]}...)" in FLAT and "1 ok (83 ok over every config)" in FLAT
    assert f"4 x LR5 = {v['lr4']} each" in FLAT and f"at its estimate, {v['pS006']} h (S006's smoke factor x{v['fac']})" in FLAT


def cap_rows(V):
    t, out = V["r"]["tally"], []
    for scr in ["BASE"] + list(L.SCREENS):
        nm = sum(1 for n in V["runs"] if H.slot(n)[0] == scr)
        m, q = t["measured"].get(scr, 0.0), t["queued"].get(scr, 0.0)
        out.append(f"    {scr:<8}{nm:>2}{'':13}{m:.3f}   {t['n_queued'].get(scr, 0):<12}{q:.3f}  {m + q:.3f}")
    return out


def test_cap_check_before_the_extension_fits(V):
    r, t, v, x = V["r"], V["r"]["tally"], V["fmt"], V["ext"]
    assert r["cap"]["cut"] == [] and sum(t["n_queued"].values()) == 10 and t["n_registered"] == 90
    assert sorted(t["queued"]) == ["S004", "S005", "S006", "S007"] and x["cut"] == [] and x["fits"]
    for row in cap_rows(V):
        assert "\n" + row + "\n" in SEC, row
    assert f"Measured {v['total']} h of runs + {v['mo']} h other = {v['m']} h; queued 10 of 90 registered runs (stage " \
           f"2's 8 arm seed runs {v['seed8']} h, S005's own BASE x 2 {v['own2']} h), {v['qs']} h; total {v['tot']} h, " \
           f"{v['room']} h of headroom. With the extension queued (+{v['pS006']} h, S006): {v['xt']} h <= 25 h: nothing " \
           f"cut, {v['xroom']} h of headroom (runs only, reported: {v['xnarrow']} h)" in FLAT


def test_what_follows_as_arithmetic(V):
    """(1) two IND runs at g 1 pass 25 h for every x >= 0 and S006 is cut first; (2) a g 8 point fits only to x = g8."""
    q, m, est, v = V["q"], V["m"], V["est"], V["fmt"]
    for x in (0.0, 0.353, est, 1.0):
        y = AL.cap_cut({**q, "S006": q["S006"] + 2 * est}, m + x)
        assert y["total_before"] > 25 and y["cut"] == ["S006"] and y["fits"]
        assert y["total_after"] == pytest.approx(V["ind"]["total_after"] + x)
    lim = 25 - V["ext"]["total_before"]
    for x, cut in ((lim - 1e-4, []), (lim + 1e-4, ["S006"])):
        assert AL.cap_cut({**q, "S006": q["S006"] + est}, m + x)["cut"] == cut
    assert f"s006_mtp_g1_s102, {v['pS006']} h each, {v['ind']} h), queued before its seed set" in FLAT
    assert f"{v['m']} + x + {v['qs']} + {v['ind']} = {v['ind0']} h + x, over 25 h for every x >= 0, and S006 is first " \
           f"in the cut order: its queued {v['q6']} h (2 seed runs, 2 IND runs) out gives {v['after0']} h + x " \
           f"({v['after1']} h at x = {v['pS006']})" in FLAT
    assert f"(its {v['c1r']} stays above g 2's {v['c2r']}), and the guard moves a pick one grid step and only when its " \
           f"PROSE is over 1% above the axis best (g 2's is {v['rel']}% above g 1's)" in FLAT
    assert f"{v['m']} + x + {v['qs']} + {v['pS006']} = {v['xt']} h + x, under 25 h only while x <= {v['g8']} h" in FLAT
    assert f"held the lock {v['loS006']} to {v['hiS006']} h each" in FLAT


def test_plan_waits_on_the_selection_mark_and_runs_only_the_extension(V):
    lines = [ln.rstrip("\n") for ln in open(PLAN)]
    assert lines[1:] == ["wait_mark SCREENS SCREENS STAGE 2 SELECTION DONE", f"train {EXT}", "mark SCREENS S006 EXTENSION DONE"]
    assert lines[0].startswith("# SCREENS stage 2 (ORDER 2): C3 extension of S006 mtp, g 4 at seed 1")
    sel = open(os.path.join(L.HERE, "plans", "stage2_select.txt")).read().splitlines()
    assert sel[-1] == "mark " + lines[1].split(" ", 2)[2]          # the wait line is stage2_select's own mark
    assert PLAN_SEQUENCE.index("stage2_s006x") == PLAN_SEQUENCE.index("stage2_select") + 1
    cur = [ln.strip() for ln in open(os.path.join(L.HERE, "plans", "CURRENT")) if not ln.startswith("#")]
    seq = PLAN_SEQUENCE + LATER_PLANS
    assert len(cur) == 1 and seq.index(cur[0]) >= seq.index("stage2_s006x")
    assert f"stage2_s006x.txt (sha256 {L.sha256(PLAN)[:16]}...; written by screens.plan), its 4 lines:" in FLAT
    assert " ".join(lines[0].split()) + " [one line in the file]" in FLAT
    assert all("\n      " + ln + "\n" in SEC for ln in lines[1:])
    assert f"Estimate {V['fmt']['pS006']} h. plans/CURRENT = stage2_s006x (edited here" in FLAT


def test_every_number_in_the_entry_is_pinned(V):
    exp = set(V["fmt"].values()) | {"0.5", "1.0", "4.0", "101", "763", "7630", "7,630", "7,344", "7,497"}
    exp |= {str(x) for x in PASSED + (WHOLE,)} | {"0.00905"}
    exp |= {str(round(H.hours_between("2026-10-06 " + a, "2026-10-06 " + b) * 3600)) for a, b in WAITS.values()}
    exp |= set(re.findall(r"\d+\.\d+", "\n".join(hours_rows(V["runs"]) + cap_rows(V) + pick_rows())))
    hexes = set(PCSHA) | {L.sha256(L.find(EXT)[0])[:16], L.sha256(PLAN)[:16], "3ca1216", "b0d2011"}
    text = re.sub(r"\b[a-z]\w*_[\w.<>]*", " ", SEC)        # run names, file names, config keys
    found = re.findall(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,16}\b", text)
    assert set(found) == hexes, set(found) ^ hexes
    text = re.sub(r"\b[0-9a-f]{7,16}\b", " ", text)
    assert set(re.findall(r"\b\d{4}-\d\d-\d\d\b", text)) == {"2026-10-06"}
    assert set(re.findall(r"\b\d\d:\d\d(?::\d\d)?\b", text)) == {QSTART, QEND, ENTRY_TIME}
    text = re.sub(r"\b\d{4}-\d\d-\d\d\b|\b\d\d:\d\d(?::\d\d)?\b", " ", text)
    nums = re.findall(r"(?<![\w.])(\d[\d,]*\d(?:\.\d+)?|\d\.\d+)(?![\w])", text)
    big = {n for n in nums if "." in n or "," in n or len(n) >= 3}
    assert len(big) > 60 and big <= exp, sorted(big - exp)
    assert len(M.MUTANTS_S2SELECT) == NEW and all(x in M.MUTANTS for x in M.MUTANTS_S2SELECT)
    assert len(M.MUTANTS) >= WHOLE                   # later entries add their own mutants
    for phrase in (f"tests {PASSED[1]} passed ({PASSED[0]} before", f"{NEW} new mutants", f"{NEW} of {NEW} killed; the "
                   f"whole file {WHOLE} of {WHOLE} killed", "queued 10 of 90 registered runs", "<= 25 h", "80 runs",
                   "1 ok (83 ok over every config)", "its 4 lines", "seed set 101"):
        assert phrase in FLAT, phrase
