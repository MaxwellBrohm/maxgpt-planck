"""SCREENS after S003 stage C (SCREENS.txt S003 STAGE C RESULT, 2026-10-05): the recorded seed-1 values give g_C 1
through E2's stage C rule with no extension (neither the pick nor the 125M argmin at an edge; the rule is shown to
fire when either is), LR5_adamw = (3e-3, 2), the stage 1 seed configs and plan are exactly ORDER 1's seed sets and stop
at a mark, the cap check before them fits, and every number in the entry is pinned (tables generated from the data;
every other decimal, comma or 3+ digit number must be one V computes; small counts by phrase). No verdict. No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_stage1_seeds.py
"""
import contextlib, datetime as dt, io, json, os, re, sys, pytest  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze  # noqa: E402
import analyze_lib as AL  # noqa: E402
import e2plan  # noqa: E402
import mutation_screens as M  # noqa: E402
import screens_hours as H  # noqa: E402
import screens_lib as L  # noqa: E402
from test_screens_s003_stage_c import SHORT, STAGE_B, section  # noqa: E402
from test_screens_stage1_next import RECORDED as STAGE_1A  # noqa: E402

P = L.params()
SETS = ("CHAT", "PROSE", "cccc", "gutenberg", "wikimedia")
# final checkpoint, split all, from ~/planck/runs/SCREENS/<run>/bpb.jsonl (copied 2026-10-05 20:25; the 6 stage C
# branch files sha256 equal on both machines): CHAT, PROSE, cccc, gutenberg, wikimedia
STAGE_C = {
    "s003_adamw_e1.5_r2_b250M": (1.16606, 1.39615, 1.3285, 1.5089, 1.3495),
    "s003_adamw_e1.5_r2_b125M": (1.22013, 1.44486, 1.3797, 1.5543, 1.3990),
    "s003_adamw_e1.5_r2_b62M": (1.30418, 1.51528, 1.4524, 1.6232, 1.4687),
    "s003_adamw_e6_r2_b250M": (1.16727, 1.40363, 1.3332, 1.5199, 1.3562),
    "s003_adamw_e6_r2_b125M": (1.21830, 1.45061, 1.3822, 1.5632, 1.4049),
    "s003_adamw_e6_r2_b62M": (1.30091, 1.52200, 1.4549, 1.6334, 1.4762),
    "s003_adamw_e3_r2_b125M": (1.21782, 1.44858, 1.3820, 1.5589, 1.4033),     # stage B's g 1, reused
    "s003_adamw_e3_r2_b62M": (1.30704, 1.52087, 1.4557, 1.6313, 1.4741),
}
G = {"0.5": "1.5", "1": "3", "2": "6"}                 # g -> eta x 1e3 at r_B 2
NAMES = {f"s003_adamw_e{e}_r2_{t}" for e in ("1.5", "6") for t in H.TAGS}
# train ("train" to "done", s) and median log.jsonl tok/s per run (g 0.5, g 2), from the PC queue log and log.jsonl
TRAIN = {"trunk": (781, 781), "b62M": (61, 61), "b125M": (101, 102), "b250M": (203, 203)}
TOKS = {"trunk": ("258k", "258k"), "b62M": ("258k", "258k"), "b125M": ("258k", "257k"), "b250M": ("258k", "258k")}
QSTART, QEND, ENTRY_TIME = "18:55:59", "20:20:36", "20:45"   # queue start (log), plan finished, entry written
PLAN = os.path.join(L.HERE, "plans", "stage1_seeds.txt")
PASSED = (338, 377)             # experiments/screens/tests before and after this entry (Mac CPU)
SEC = section("S003 STAGE C RESULT")
FLAT = " ".join(SEC.split())


def write_runs(root, vals):
    for name, v in vals.items():
        os.makedirs(root / name, exist_ok=True)
        with open(root / name / "bpb.jsonl", "w") as f:
            for s, x in zip(SETS, v):
                f.write(json.dumps({"ckpt": "final_00007630.pt", "split": "all", "set": s, "bpb": x}) + "\n")
    return analyze.picks(str(root))


@pytest.fixture(scope="module")
def picks(tmp_path_factory):
    return write_runs(tmp_path_factory.mktemp("runs"), {**STAGE_1A, **STAGE_B, **STAGE_C})


def bpb(g, tag):
    return {**STAGE_B, **STAGE_C}[f"s003_adamw_e{G[g]}_r2_{tag}"][:2]


@pytest.fixture(scope="module")
def V():
    """Every number the entry states, computed from the recorded values, measured_hours.tsv, the hours report and
    E3 results.json, formatted as the entry writes it (fmt)."""
    runs, _ = H.load()
    with contextlib.redirect_stdout(io.StringIO()):
        r = H.report(P, r_b=2)
    t, c, f = r["tally"], r["cap"], (lambda x, n=3: f"{x:.{n}f}")
    c250 = {g: bpb(g, "b250M") for g in G}
    best = min(v[1] for v in c250.values())
    mde = json.load(open(os.path.join(L.E3D, "results.json")))["noise_5M_250M"]["F(CHAT)"]["MDE"]["1"]["paired"]
    lock, train = sum(runs[n] for n in NAMES), sum(sum(v) for v in TRAIN.values()) / 3600
    arms = {round(sum(runs[f"s003_adamw_e{e}_r2_{tg}"] for tg in H.TAGS), 3) for e in ("1.5", "6")}
    wall = (dt.datetime.strptime(QEND, "%H:%M:%S") - dt.datetime.strptime(QSTART, "%H:%M:%S")).seconds / 3600
    full = {n: h for n, h in runs.items() if n.startswith(("s001", "s002"))}
    sel = {n: h for n, h in runs.items() if n in full or "_r1_" in n}
    stb = {n: h for n, h in runs.items() if n.startswith("s003_adamw_e3_r") and "_r1_" not in n}
    mr, mo, qs = r["measured_runs"], r["other"], sum(t["queued"].values())
    run, head = H.run_h(L.STEPS), 25 - c["total_before"]
    one = 6 * run + head                          # seed set 101's lock h at which the cap is passed; two: still after S006's cut
    two = one + t["queued"]["S006"]
    ext = sum(2 * H.run_h(L.STEPS, H.mult(s, a)) for s in L.ORDER[2] for a in L.SCREENS[s]["arms"])
    ind = 2 * (H.run_h(L.STEPS, H.mult("S006", "mtp")) + H.run_h(L.STEPS, H.mult("S007", "smear")))
    fmt = {"d05": f(c250["0.5"][0] - c250["1"][0], 5), "d2": f(c250["2"][0] - c250["1"][0], 5), "best": f(best, 5),
           "rel": f(100 * (c250["1"][1] / best - 1)), "mde": f(mde, 5), "lock": f(lock), "arm": f(min(arms)),
           "train": f(train), "gaps": f(lock - train), "wall": f(wall, 2), "total": f(sum(runs.values())),
           "sel": f(sum(sel.values())), "stb": f(sum(stb.values())), "a": f(sum(h for n, h in sel.items() if "_r1_" in n)),
           "s003": f(sum(h for n, h in runs.items() if n.startswith("s003"))), "mr": f(mr), "mo": f(mo),
           "mrmo": f(mr + mo), "qs": f(qs), "tot": f(c["total_before"]), "room": f(25 - c["total_after"]),
           "narrow": f(r["narrow"]["total_before"]), "run": f(run), "six": f(6 * run), "one": f(one),
           "one6": f(one / 6), "two": f(two), "two6": f(two / 6), "ext": f(ext, 2), "ind": f(ind, 2),
           "est12": f(12 * run), "fmin": f(min(full.values())), "fmax": f(max(full.values()))}
    return {"runs": runs, "r": r, "arms": arms, "run": run, "head": head, "two": two, "fmt": fmt}


def test_recorded_stage_c_picks_g1_with_no_extension(picks):
    assert [bpb("1", t) for t in ("b125M", "b62M")] == [SHORT[t][2] for t in ("b125M", "b62M")]   # stage B record
    a, b, c = picks["S003"]["A"], picks["S003"]["B"], picks["S003"]["C"]
    assert (a["pick"], b["pick"]) == (0.003, 2.0) and a["decided"] and b["decided"]
    assert (c["pick"], c["argmin"], c["at_edge"], c["argmin_125M"], c["guard_moved"], c["extend_with"],
            c["extensions_used"], c["decided"], c["runner_up"]) == (1.0, 1.0, None, 1.0, False, [], 0, True, 0.5)
    assert c["axis"] == [0.5, 1.0, 2.0] and c["runs"] == {g: f"s003_adamw_e{e}_r2_b250M" for g, e in G.items()}
    assert c["lrs"] == {"pick": [0.003, 2.0], "runner_up": [0.0015, 2.0]}
    assert L.lrs(*c["lrs"]["pick"]) == {"optim.lr": 0.003, "optim.embed_lr": 0.006, "optim.scalar_lr": 0.006}


def test_the_edge_rules_would_have_fired(tmp_path):
    """Not vacuous: a 125M argmin at g 2, or a 250M pick at g 0.5, asks for one more factor-2 point."""
    low = dict(STAGE_C, **{"s003_adamw_e6_r2_b125M": (1.2170,) + STAGE_C["s003_adamw_e6_r2_b125M"][1:]})
    c = write_runs(tmp_path / "a", {**STAGE_1A, **STAGE_B, **low})["S003"]["C"]
    assert (c["pick"], c["argmin_125M"], c["extend_with"], c["decided"]) == (1.0, 2.0, [4.0], False)
    low = dict(STAGE_C, **{"s003_adamw_e1.5_r2_b250M": (1.1600,) + STAGE_C["s003_adamw_e1.5_r2_b250M"][1:]})
    c = write_runs(tmp_path / "b", {**STAGE_1A, **STAGE_B, **low})["S003"]["C"]
    assert (c["pick"], c["at_edge"], c["extend_with"], c["decided"]) == (0.5, "low", [0.25], False)


def test_entry_pick_table_and_its_readings(V):
    for g, eta in (("0.5", "1.5e-3"), ("1", "3e-3"), ("2", "6e-3")):
        row = [bpb(g, t) for t in ("b250M", "b125M", "b62M")]
        assert f"\n    {g:<5} {eta:<7} " + "    ".join(f"{x:.5f} / {y:.5f}" for x, y in row) + "\n" in SEC + "\n"
    v, c125 = V["fmt"], {g: bpb(g, "b125M")[0] for g in G}
    assert f"(g 0.5 is {v['d05']} higher, g 2 {v['d2']}); its PROSE {bpb('1', 'b250M')[1]:.5f} is {v['rel']}% above " \
           f"the axis best ({v['best']} at g 0.5), under the 1% guard" in FLAT
    assert f"best minus second {v['d05']} against E3's paired F(CHAT) MDE at k = 1, {v['mde']}:" in FLAT
    assert f"125M argmin is g 1 too ({c125['1']:.5f}; g 2 {c125['2']:.5f}, g 0.5 {c125['0.5']:.5f})" in FLAT
    assert f"the 62.5M argmin is g 2 ({bpb('2', 'b62M')[0]:.5f}), at the high edge" in FLAT
    assert min(G, key=lambda g: bpb(g, "b62M")[0]) == "2" and "Pick g_C = 1: the 250M argmin" in FLAT
    assert "extend_with [], extensions_used 0, decided; runner-up g 0.5 (eta 1.5e-3, r 2)" in FLAT
    assert "present in the runs directory: 0, so 2 are left" in FLAT and "EXTENSION: none." in FLAT
    assert "optim.lr 3e-3, optim.embed_lr = optim.scalar_lr 6e-3, optim.kind adamw" in FLAT
    assert "LR5_adamw = (g_C x eta_A, r_B) = (eta 3e-3, r 2)" in FLAT and "LR5 = (3e-3, 1): 3e-3 for all three" in FLAT


def hours_rows(runs):
    out = []
    for i, e in enumerate(("1.5", "6")):
        for a, b in (("trunk", "b62M"), ("b125M", "b250M")):
            out.append("    " + "  | ".join(f"{f's003_adamw_e{e}_r2_{t}':<27}{runs[f's003_adamw_e{e}_r2_{t}']:.3f} "
                                            f"{TRAIN[t][i] / 3600:.3f}  {TOKS[t][i]}" for t in (a, b)))
    return out


def test_entry_hours_are_the_measured_file(V):
    runs, v = V["runs"], V["fmt"]
    for row in hours_rows(runs):
        assert "\n" + row + "\n" in SEC, row
    assert NAMES <= set(runs) and len(runs) == 56 and len(V["arms"]) == 1
    assert f"Total {v['lock']} GPU hours measured for stage C (each arm {v['arm']} h; {v['train']} h training, " \
           f"{v['gaps']} h scoring and gaps); {v['wall']} h wall clock" in FLAT
    assert f"56 runs, {v['total']} h (stage 1 selection A {v['sel']} + stage B {v['stb']} + stage C {v['lock']})" in FLAT
    assert f"11 arms, 44 runs, {v['s003']} h measured (stage A {v['a']}, stage B {v['stb']}, stage C {v['lock']})" in FLAT
    assert (v["sel"], v["stb"], v["a"]) == ("5.483", "1.622", "2.034")      # the earlier entries' figures
    ends = {ln.split("\t")[1]: ln.split("\t")[4] for ln in open(H.TSV) if ln.startswith("run\t")}
    assert max(ends[n] for n in NAMES) == "2026-10-05 " + QEND


def cap_rows(V):
    t, out = V["r"]["tally"], []
    for scr in ["BASE"] + list(L.SCREENS):
        nm = sum(1 for n in V["runs"] if H.slot(n)[0] == scr)
        m, q = t["measured"].get(scr, 0.0), t["queued"].get(scr, 0.0)
        out.append(f"    {scr:<8}{nm:>2}{'':13}{m:.3f}   {t['n_queued'].get(scr, 0):<12}{q:.3f}  {m + q:.3f}")
    return out


def test_cap_check_before_the_seed_sets(V):
    r, v = V["r"], V["fmt"]
    t, c = r["tally"], r["cap"]
    assert sum(t["n_queued"].values()) == 34 and t["n_registered"] == 90 and c["cut"] == [] and t["extra"] == {}
    open_ = [k for k in t["registered"] if k not in t["seen"]]
    assert sorted({k[0] for k in open_ if k[-1] in (101, 102) and k[0] in ("BASE", "S001", "S002", "S003")}) == \
        ["BASE", "S001", "S002", "S003"] and sum(1 for k in open_ if k[0] in ("BASE", "S001", "S002", "S003")) == 12
    for row in cap_rows(V):
        assert "\n" + row + "\n" in SEC, row
    assert f"Measured {v['mr']} h of runs + {v['mo']} h other = {v['mrmo']} h; queued 34 of 90 registered runs (the " \
           f"stage 1 seed sets 12, stage 2's 22), {v['qs']} h; total {v['tot']} h <= 25 h: nothing cut, {v['room']} h " \
           f"of headroom (runs only, reported: {v['narrow']} h)" in FLAT
    q101 = dict(t["queued"])                              # after seed set 101: its estimates leave the queue
    for scr, n in (("BASE", 1), ("S001", 1), ("S002", 3), ("S003", 1)):
        q101[scr] -= n * V["run"]
    m = r["measured_runs"] + r["other"] + 6 * V["run"]
    for over, cut in ((V["head"] - 1e-4, []), (V["head"] + 1e-4, ["S006"]), (V["two"] - 6 * V["run"] + 1e-4, ["S006", "S003"])):
        assert AL.cap_cut(q101, m + over)["cut"] == cut
    assert f"(6 x {v['run']} = {v['six']} h)" in FLAT and f"over {v['one']} h ({v['one6']} h a run;" in FLAT
    assert f"seed set 101 over {v['two']} h ({v['two6']} h a run)" in FLAT and f"held it {v['fmin']} to {v['fmax']} h" in FLAT
    assert f"up to {v['ext']} h of stage 2 g-check extensions and up to {v['ind']} h of matched-LR IND runs" in FLAT
    assert f"Estimate 12 x {v['run']} h = {v['est12']} h" in FLAT


def test_seed_plan_is_exactly_order_1s_seed_sets_then_a_mark():
    lines = [ln.rstrip("\n") for ln in open(PLAN)]
    stage_c = open(os.path.join(L.HERE, "plans", "stage1_s003C.txt")).read().splitlines()
    assert lines[1] == "wait_mark SCREENS " + stage_c[-1].split(" ", 1)[1]    # waits on stage C's own mark
    arms = ["s001_nogate_g1_s{}"] + [f"s002_{a}_g1_s{{}}" for a in L.SCREENS["S002"]["arms"]]
    want = [f"train {n.format(s)}" for s in P["seeds"]["same init"] for n in ["base_s{}"] + arms + ["s003_adamw_e3_r2_s{}"]]
    assert lines[2:-1] == want and lines[-1] == "mark SCREENS STAGE 1 SEEDS DONE" and len(lines) == 15
    assert all(P["seeds"][L.SCREENS[s]["cls"]] == [101, 102] for s in L.ORDER[1])
    head = " ".join(SEC.split("Its 15 lines:")[1].split("[one line in the file]")[0].split())
    assert lines[0] == head and all(f"\n      {ln}\n" in SEC + "\n" for ln in lines[1:])
    assert f"stage1_seeds.txt (sha256 {L.sha256(PLAN)[:16]}...)" in FLAT


@pytest.mark.parametrize("seed", (101, 102))
def test_seed_configs_are_their_base_plus_the_registered_keys(seed):
    base = L.flat(L.resolve(L.find(f"base_s{seed}")[0]))
    want = {"s001_nogate_g1": {"model.attn_gate": False}, "s002_novres_g1": {"model.value_residual": False},
            "s002_noqknorm_g1": {"model.qk_norm": False}, "s002_nonormscale_g1": {"model.norm_scaling": False},
            "s003_adamw_e3_r2": {"optim.kind": "adamw", "optim.embed_lr": 0.006, "optim.scalar_lr": 0.006}}
    for stem, keys in want.items():
        path = L.find(f"{stem}_s{seed}")[0]
        f = L.flat(L.resolve(path))
        assert L.diff(f, base) == {"name", "out_dir"} | set(keys) and all(f[k] == v for k, v in keys.items())
        assert f["seed"] == seed and f["optim.lr"] == 0.003 and f["schedule.mode"] == "full" and L.check(path, P) == []
        assert f"{L.sha256(path)[:16]} {stem}_s{seed}" in SEC
    assert f"{L.sha256(L.find(f'base_s{seed}')[0])[:16]} base_s{seed}" in SEC
    assert len(L.all_configs()) == 82 and "12 ok (82 ok over every config)" in FLAT


def test_every_number_in_the_entry_is_pinned(V):
    exp = {f"{x:.5f}" for vals in STAGE_C.values() for x in vals[:2]} | {f"{x:.5f}" for x in bpb("1", "b250M")}
    exp |= set(V["fmt"].values()) | {"0.5"} | {f"{e}e-3" for e in G.values()} | {"101", "102", "112", str(len(M.MUTANTS))}
    exp |= {str(PASSED[0]), str(PASSED[1])}
    exp |= {f"{x:,}" for x in [at for _, at in e2plan.SIZES["5m"]["branches"]] + [L.STEPS]}
    exp |= set(re.findall(r"\d+\.\d+", "\n".join(hours_rows(V["runs"]) + cap_rows(V))))
    hexes = {L.sha256(p)[:16] for p in L.all_configs()} | {L.sha256(PLAN)[:16], "03ec0d236cc72a36"}
    hexes |= set(re.findall(r"sha256 ([0-9a-f]{16})\.\.\. and ([0-9a-f]{16})", open(H.TSV).read())[0])
    text = re.sub(r"\b[a-z]\w*_[\w.<>]*", " ", SEC)        # run names, file names, config keys
    found = re.findall(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{16}\b", text)
    assert len(found) == 16 and set(found) <= hexes, set(found) - hexes
    text = re.sub(r"\b[0-9a-f]{16}\b", " ", text)
    assert set(re.findall(r"\b\d{4}-\d\d-\d\d\b", text)) == {"2026-10-05"}
    assert set(re.findall(r"\b\d\d:\d\d(?::\d\d)?\b", text)) == {QSTART, QEND, ENTRY_TIME}
    text = re.sub(r"\b\d{4}-\d\d-\d\d\b|\b\d\d:\d\d(?::\d\d)?\b", " ", text)
    nums = re.findall(r"(?<![\w.])(\d[\d,]*\d(?:\.\d+)?(?:e-\d+)?|\d\.\d+(?:e-\d+)?|\d+e-\d+)(?![\w])", text)
    big = {n for n in nums if "." in n or "," in n or "e-" in n or len(n) >= 3}
    assert len(big) > 60 and big <= exp, sorted(big - exp)
    for phrase in ("All 8 runs", '6 "done and scored" (branches), 2 "done" (trunks', "all 6 scored runs",
                   "log.jsonl of all 8", "112 SCREENS lines (96 + 8 starts, 8 ends). The 6 branch bpb.jsonl and the 8 "
                   "log.jsonl", "its 48 earlier run lines", "8 new lines", "and 2 rolling checkpoints", "<= 25 h",
                   "k = 2 in every class", "seeds 101, 102", "wrote 10 configs (S001_attn_gate 2, S002_block_ablations 6,"
                   " S003_adamw 2)", "{0.5, 1, 2}", "at most 2 as e2pick", "1% guard", "first 16 hex", "Its 15 lines",
                   f"tests {PASSED[1]} passed ({PASSED[0]} before", "smoke, 30; the new file, 9)", "82 configs, the seed",
                   f"{len(M.MUTANTS_S003C)} new mutants", f"{len(M.MUTANTS_S003C)} of {len(M.MUTANTS_S003C)} killed; the "
                   f"whole file {len(M.MUTANTS)} of {len(M.MUTANTS)} killed", "56 measured runs", "34 queued slots"):
        assert phrase in FLAT, phrase
