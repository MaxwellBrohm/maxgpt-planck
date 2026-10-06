"""SCREENS STAGE 1 VERDICTS (SCREENS.txt, 2026-10-06): the C7 verdicts of S001, S002 and S003 at seeds 101 and 102 follow
from the recorded final bpb through analyze.py verdicts (C4 classes and noise checks, CCCC-ONLY, Holm); each screen's
results.json and tables.txt carry exactly those readings; the rule could have read either direction at these SDs (and
its every-seed, delta and noise-check clauses fire on nearby values); the C6 readings in results.json agree with their
own curves and counts; each notes.txt RESULT block and the SCREENS entry state the verdict and its registered
consequence. A 5M loss reading only (C8). Seed-1 runs only name the picks. No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_stage1_verdicts.py
"""
import contextlib, copy, io, json, os, re, statistics as st, sys, pytest  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze  # noqa: E402
import analyze_lib as AL  # noqa: E402
import screens_lib as L  # noqa: E402
from test_screens_s003_stage_c import STAGE_B, section  # noqa: E402
from test_screens_stage1_next import RECORDED as STAGE_1A  # noqa: E402
from test_screens_stage1_seeds import STAGE_C, write_runs  # noqa: E402

SETS = ("CHAT", "PROSE", "cccc", "gutenberg", "wikimedia")
# final checkpoint, split all, from ~/planck/runs/SCREENS/<run>/bpb.jsonl (plan stage1_seeds, commit 93bc10b; copied
# 2026-10-06 after the mark, sha256 equal on both machines): CHAT, PROSE, cccc, gutenberg, wikimedia
SEED = {
    "base_s101": (1.1525046041114577, 1.386213915309061, 1.316926106921712, 1.500800122104768, 1.3392976247836352),
    "base_s102": (1.1560899136419525, 1.3884848494244255, 1.320740294136035, 1.4993517856381693, 1.3437992119728759),
    "s001_nogate_g1_s101": (1.1561644240594138, 1.386184744394854, 1.3179063579612862, 1.4985795642594508, 1.340482308973933),
    "s001_nogate_g1_s102": (1.1602369811297133, 1.3903294656370027, 1.3216481949047583, 1.5033485192733793, 1.3443969433489045),
    "s002_novres_g1_s101": (1.1681628869467313, 1.3995204337318128, 1.3305236555321718, 1.5116938212060376, 1.3547635112329837),
    "s002_novres_g1_s102": (1.17485869776289, 1.4031701961630063, 1.3356724526706323, 1.5103964340091247, 1.3619357261899945),
    "s002_noqknorm_g1_s101": (1.1544925898871108, 1.3896275006881784, 1.318698858823117, 1.5054809543548646, 1.343069533179633),
    "s002_noqknorm_g1_s102": (1.155191716149333, 1.3880458026667117, 1.3184698417260636, 1.503097638829953, 1.3409454807363899),
    "s002_nonormscale_g1_s101": (1.150206548454015, 1.3849332142672182, 1.3159611616651155, 1.4982648639745781, 1.3389747700297394),
    "s002_nonormscale_g1_s102": (1.1572805036260072, 1.388003817630094, 1.3188590403227496, 1.5005862678781128, 1.3429797570442685),
    "s003_adamw_e3_r2_s101": (1.1711180801405827, 1.399537316319161, 1.3316367726429361, 1.513112044983797, 1.352257188854294),
    "s003_adamw_e3_r2_s102": (1.1728911655363063, 1.4005206604363314, 1.331565783579991, 1.5134132275663508, 1.3549910691856806),
}
VERDICT = {"S001.nogate": "TIE", "S002.novres": "REJECT", "S002.noqknorm": "TIE", "S002.nonormscale": "TIE",
           "S003.adamw": "REJECT"}
HOLM = {"S001.nogate": "does not hold", "S002.novres": "holds", "S002.noqknorm": "does not hold",
        "S002.nonormscale": "does not hold", "S003.adamw": "holds"}
NAME = {"S001.nogate": "s001_nogate_g1_s{}", "S002.novres": "s002_novres_g1_s{}", "S002.noqknorm": "s002_noqknorm_g1_s{}",
        "S002.nonormscale": "s002_nonormscale_g1_s{}", "S003.adamw": "s003_adamw_e3_r2_s{}"}
SIDS = ("S001", "S002", "S003")
EXP = os.path.dirname(L.HERE)
DATE = "2026-10-06"


def res(sid):
    return json.load(open(os.path.join(EXP, L.SCREENS[sid]["dir"], "results.json")))


def text(sid, f):
    return open(os.path.join(EXP, L.SCREENS[sid]["dir"], f)).read()


def flat(s):
    return " ".join(s.split())


def verdicts(root, seed_vals):
    write_runs(root, {**STAGE_1A, **STAGE_B, **STAGE_C, **seed_vals})
    with contextlib.redirect_stdout(io.StringIO()):
        assert analyze.main(["verdicts", "--runs", str(root), "--out", str(root / "v.json")]) == 0
    return json.load(open(root / "v.json"))


@pytest.fixture(scope="module")
def V(tmp_path_factory):
    return verdicts(tmp_path_factory.mktemp("runs"), SEED)


def test_recorded_values_give_the_verdicts(V):
    c = V["contrasts"]
    assert {k: x["verdict"] for k, x in c.items()} == VERDICT and {k: x["holm"] for k, x in c.items()} == HOLM
    assert all(x["labels"] == [] and x["seeds"] == [101, 102] and x["dropped_base_diverged"] == [] for x in c.values())
    assert {k: x["class"] for k, x in c.items()} == {k: L.SCREENS[k[:4]]["cls"] for k in VERDICT}   # no re-read
    assert V["holm_family"] == sorted(VERDICT) and V["picks"]["S003"]["C"]["lrs"]["pick"] == [0.003, 2.0]
    p = V["pooled_sia_check"]
    assert [(m, p[m]["df"], p[m]["arms"], p[m]["ok"]) for m in ("CHAT", "PROSE")] == [("CHAT", 3, 3, True), ("PROSE", 3, 3, True)]
    assert all(x["ok"] and x["df"] == 1 for x in c["S003.adamw"]["noise_check"].values())
    assert c["S002.noqknorm"]["readings"]["gutenberg"]["label"] == "WORSE"      # reported; PROSE (the guard) is EQUAL


def test_results_json_carry_the_analyzer_readings(V):
    for sid in SIDS:
        r = res(sid)
        assert r["class"] == L.SCREENS[sid]["cls"] and r["seeds"] == [101, 102] and r["k"] == 2
        assert sorted(r["contrasts"]) == sorted(k for k in VERDICT if k.startswith(sid))
        for k, c in r["contrasts"].items():
            assert r["verdicts"][k] == c["verdict"] == VERDICT[k] and c["holm"] == HOLM[k]
            assert c == V["contrasts"][k]           # exact: the recorded floats at full precision, JSON round trip
        assert r["holm"]["family"] == V["holm_family"]
        assert r["holm"]["p_chat"] == {k: c["readings"]["CHAT"]["p_two_sided"] for k, c in V["contrasts"].items()}
    assert res("S002")["pooled_sia_check"] == V["pooled_sia_check"]
    assert res("S003")["same_init_check"] == V["contrasts"]["S003.adamw"]["noise_check"]


def moved(key, f):
    """The arm's recorded values with d(seed, set) replaced by f(d, seed); BASE as recorded."""
    out = {}
    for s in (101, 102):
        b, a = SEED[f"base_s{s}"], SEED[NAME[key].format(s)]
        out[NAME[key].format(s)] = tuple(bi + f(ai - bi, s) for ai, bi in zip(a, b))
    return out


def contrast(key, vals):
    p, e3 = L.params(), json.load(open(os.path.join(L.E3D, "results.json")))["noise_5M_250M"]
    pick = lambda n: {x: dict(zip(SETS, vals.get(n.format(x), SEED.get(n.format(x))))) for x in (101, 102)}  # noqa: E731
    return AL.contrast(pick(NAME[key]), pick("base_s{}"), L.SCREENS[key[:4]]["cls"], e3, p["u"])


def test_the_rule_reads_both_directions_at_these_sds():
    """Not vacuous: at the recorded SDs a mirrored REJECT nominates, the S001 TIE doubled rejects and mirrored nominates."""
    for k in ("S002.novres", "S003.adamw"):
        assert contrast(k, moved(k, lambda d, s: -d))["verdict"] == "NOMINATE"
    assert contrast("S001.nogate", moved("S001.nogate", lambda d, s: 2 * d))["verdict"] == "REJECT"
    assert contrast("S001.nogate", moved("S001.nogate", lambda d, s: -2 * d))["verdict"] == "NOMINATE"
    # the delta clause: 1.5 x S001's d is under thr but its interval is not inside +-1% -> INCONCLUSIVE, not TIE
    c = contrast("S001.nogate", moved("S001.nogate", lambda d, s: 1.5 * d))
    assert (c["verdict"], c["readings"]["CHAT"]["label"]) == ("INCONCLUSIVE", "UNRESOLVED")
    # the every-seed clause, both signs: one seed on the other side keeps a large mean INCONCLUSIVE
    assert contrast("S002.novres", moved("S002.novres", lambda d, s: 2 * d if s == 101 else -0.001))["verdict"] == "INCONCLUSIVE"
    assert contrast("S003.adamw", moved("S003.adamw", lambda d, s: -2 * d if s == 101 else 0.001))["verdict"] == "INCONCLUSIVE"


def test_the_noise_checks_would_fire(tmp_path):
    """C4: a spread seed in one S002 arm fails the pooled SIA check and re-reads every SIA arm at the new-init SD;
    a spread S003 seed fails the same-init check. Both re-reads are labelled underpowered."""
    v = verdicts(tmp_path / "a", {**SEED, **moved("S002.noqknorm", lambda d, s: d + (0.02 if s == 101 else 0))})
    assert not v["pooled_sia_check"]["CHAT"]["ok"]
    for k in ("S002.novres", "S002.noqknorm", "S002.nonormscale"):
        c = v["contrasts"][k]
        assert c["class"] == "new init" and c["class_registered"] == "SIA" and "underpowered" in c["labels"]
    v = verdicts(tmp_path / "b", {**SEED, **moved("S003.adamw", lambda d, s: d + (0.02 if s == 101 else 0))})
    c = v["contrasts"]["S003.adamw"]
    assert not c["noise_check"]["CHAT"]["ok"] and c["class"] == "new init" and "underpowered" in c["labels"]


ROW = re.compile(r"^(\w+)\s+(CHAT|PROSE|cccc|gutenberg|wikimedia)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\w+)$")


def test_tables_txt_rows_equal_results_json():
    for sid in SIDS:
        r, t = res(sid), text(sid, "tables.txt")
        rows = [m.groups() for m in map(ROW.match, t.splitlines()) if m]
        assert len(rows) == 5 * len(r["contrasts"])
        for a, m, d1, d2, dbar, thr, delta, sd, df, p, label in rows:
            x = r["contrasts"][f"{sid}.{a}"]["readings"][m]
            assert (d1, d2, dbar) == tuple(f"{v:+.5f}" for v in (*x["d"], x["dbar"])) and label == x["label"]
            assert (thr, delta, sd, df, p) == (f"{x['thr']:.5f}", f"{x['delta']:.5f}", f"{x['sd_ref']:.5f}", str(x["df"]),
                                               f"{x['p_two_sided']:.3g}")      # 3 significant digits: 4.39e-05, not 0.0000
        for k, c in r["contrasts"].items():
            assert f"VERDICT {k}: {c['verdict']}; labels none; Holm (family of 5 so far): {c['holm']}" in t
        assert "A 5M loss reading only (C8)" in t


def test_s003_tables_state_the_registered_asymmetry_beside_the_verdict():
    """S003 notes: the NorMuon-favouring asymmetry is stated "beside every S003 verdict", and under REJECT "The asymmetry
    above could have produced a small REJECT, so the report states it": tables.txt carries both, verbatim."""
    r, lines = res("S003"), text("S003", "tables.txt").splitlines()
    head = notes_result("S003")[0]
    asym, reg = flat(r["asymmetry_registered"]), flat(r["consequence_registered"])
    assert asym in head and reg in head and "it favours NorMuon" in head and reg.startswith("REJECT ")
    i = next(n for n, x in enumerate(lines) if x.startswith("VERDICT S003.adamw: "))
    assert lines[i + 1] == f'ASYMMETRY (registered; it favours NorMuon; stated beside every S003 verdict): "{asym}"'
    j = next(n for n, x in enumerate(lines) if x.startswith("CONSEQUENCE: "))
    assert lines[j] == f"CONSEQUENCE: {r['consequence']}"
    assert lines[j + 1] == f'REGISTERED REJECT (verbatim): "{reg.removeprefix("REJECT ")}"' and "small REJECT" in lines[j + 1]
    for sid in ("S001", "S002"):
        assert "ASYMMETRY" not in text(sid, "tables.txt")


def test_c5_and_c6_agree_with_the_recorded_values_and_their_own_curves():
    for sid in SIDS:
        r = res(sid)
        c6 = r["c6"]
        for a, ms in r["c5_reported"].items():
            k = f"{sid}.{a}"
            assert ms["F(CHAT)"]["arm"] == [SEED[NAME[k].format(s)][0] for s in (101, 102)]
            assert ms["F(PROSE)"]["base"] == [SEED[f"base_s{s}"][1] for s in (101, 102)]
            for x in ms.values():
                assert x["d"] == pytest.approx([p - q for p, q in zip(x["arm"], x["base"])]) and x["dbar"] == pytest.approx(st.fmean(x["d"]))
            ind = c6["ind_final"][a]
            assert ind["arm"] == [c6["ind_curves"][NAME[k].format(s)][-1][1] for s in (101, 102)]
            assert ind["d"] == pytest.approx([p - q for p, q in zip(ind["arm"], ind["base"])])
        for n, cv in c6["ind_curves"].items():
            assert [s for s, _ in cv] == list(range(153, 7630, 153)) + [7630]
            bf = None if n.startswith("base") else c6["ind_curves"]["base_s" + n[-3:]][-1][1]
            assert c6["ind_onsets"][n] == pytest.approx(AL.onsets([tuple(x) for x in cv], bf))
            x = c6["clip"][n]
            assert x["logged"] == 763 and x["clip_rate"] == x["clipped"] / 763
        assert c6["ind_onset"].startswith("LR-confounded, not read" if sid == "S003" else "read")
        assert set(c6["not_computed"]) == {"S001": {"HD", "SINK", "GATE_200_windows_p10_p90"}, "S002": {"HD", "SINK"},
                                           "S003": {"HD"}}[sid]
    g = res("S001")["c6"]["gate_base"]
    assert all(v["shape"] == [8, 3] and v["final_step"] == 7630 for v in g.values())


def notes_result(sid):
    t = text(sid, "notes.txt")
    head, sep, block = t.partition(f"\nRESULT ({DATE}")
    assert sep and f"\nRESULT ({DATE}" not in block
    return flat(head), flat(block)


def test_notes_result_blocks_state_verdict_and_registered_consequence():
    for sid in SIDS:
        r, (head, block) = res(sid), notes_result(sid)
        cr = r["consequence_registered"]
        for k, v in r["contrasts"].items():
            assert f"VERDICT {k}: {v['verdict']}" in block
            reg = cr[v["verdict"]] if isinstance(cr, dict) else cr
            assert flat(reg) in head and flat(reg) in block          # verbatim from the pre-registration
        assert "5M loss reading only" in block and "C8" in block and "does NOT show" in block
        assert flat(r["consequence"]) in block
    assert flat(res("S003")["asymmetry_registered"]) in notes_result("S003")[1]


def entry_line(k, c):
    ch, pr = c["readings"]["CHAT"], c["readings"]["PROSE"]
    return (f"{k:<17} {c['class']:<10} {c['verdict']:<7} {c['holm']:<14} {ch['d'][0]:+.5f} / {ch['d'][1]:+.5f}  "
            f"{ch['dbar']:+.5f}  {ch['thr']:.5f}  {pr['dbar']:+.5f}  {pr['thr']:.5f}  {ch['label']} / {pr['label']}")


def test_screens_entry_carries_every_verdict():
    sec = section("STAGE 1 VERDICTS")
    assert sec.count("STAGE 1 VERDICTS (") == 1 and DATE in sec.splitlines()[0]
    for sid in SIDS:
        for k, c in res(sid)["contrasts"].items():
            assert entry_line(k, c) in sec
    assert "5M loss reading only" in sec and "underpowered" in sec and "not resolved" in sec


def test_no_private_identifiers():
    """C9: no machine address, user name or key path (tests/privacy.py: committed shapes, runtime values, the git-ignored
    local list; no name is written in the repo). Its own probes: test_screens_stage1_corrections.py."""
    import privacy
    pats = privacy.patterns()
    for sid in SIDS:
        for f in ("results.json", "tables.txt", "notes.txt"):
            assert not privacy.hits(text(sid, f), pats), (sid, f, privacy.hits(text(sid, f), pats))
    assert not privacy.hits(section("STAGE 1 VERDICTS"), pats)
