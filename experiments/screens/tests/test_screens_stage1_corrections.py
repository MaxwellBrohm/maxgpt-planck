"""SCREENS STAGE 1 VERDICTS, corrected 2026-10-06 (SCREENS.txt entry; each S00x notes.txt CORRECTIONS block): C4's
"noise check failed later" is the label analyze.py writes for an earlier-read verdict when a later SIA arm fails the
pooled check (run through the tool); k_needed per class as E3 gives it; C9's out/<run>/ copies equal the recorded
sha256 and hold no private identifier; the privacy check itself can fail. No model.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_stage1_corrections.py
"""
import contextlib, getpass, glob, hashlib, io, json, math, os, re, sys  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import analyze  # noqa: E402
import privacy  # noqa: E402
import screens_lib as L  # noqa: E402
from test_screens_s003_stage_c import STAGE_B, section  # noqa: E402
from test_screens_stage1_next import RECORDED as STAGE_1A  # noqa: E402
from test_screens_stage1_seeds import STAGE_C, write_runs  # noqa: E402
from test_screens_stage1_verdicts import EXP, SEED, SIDS, flat, res, text  # noqa: E402

S002_KEYS = ("S002.nonormscale", "S002.noqknorm", "S002.novres")
S002_RES = os.path.join(EXP, "S002_block_ablations", "results.json")


def s007_runs(spread):
    """A stage 2 SIA arm (S007 smear): g-checks that pick g 1 inside the grid (S001's recorded shape), and seed runs at
    BASE + 0.001 with one seed moved by `spread` bpb on every metric."""
    out = {f"s007_smear_g{g}_s1": STAGE_1A[f"s001_nogate_g{g}_s1"] for g in ("0.5", "1", "2")}
    for s, extra in ((101, spread), (102, 0.0)):
        out[f"s007_smear_g1_s{s}"] = tuple(x + 0.001 + extra for x in SEED[f"base_s{s}"])
    return out


def verdicts(root, spread, earlier):
    write_runs(root, {**STAGE_1A, **STAGE_B, **STAGE_C, **SEED, **s007_runs(spread)})
    argv = ["verdicts", "--runs", str(root), "--out", str(root / "v.json")] + [x for e in earlier for x in ("--earlier", e)]
    with contextlib.redirect_stdout(io.StringIO()):
        assert analyze.main(argv) == 0
    return json.load(open(root / "v.json"))


def test_a_later_pooled_failure_relabels_earlier_verdicts_with_c4s_text(tmp_path):
    reg = re.findall(r're-labelled "([^"]+)"', flat(section("C4 SEEDS")))
    assert reg == ["noise check failed later"] == [analyze.LATER]          # C4's own words, read from the file
    v = verdicts(tmp_path / "a", 0.03, [S002_RES])
    assert v["read_earlier"] == sorted(res("S002")["contrasts"]) == list(S002_KEYS)
    assert not v["pooled_sia_check"]["CHAT"]["ok"] and v["pooled_sia_check"]["CHAT"]["arms"] == 4
    later = res("S002")["if_pooled_check_fails_later"]["contrasts"]
    for k in S002_KEYS:
        c = v["contrasts"][k]
        assert c["labels"] == [analyze.LATER, "underpowered"] and c["class"] == "new init", k
        assert (c["verdict"], c["labels"], c["readings"]["CHAT"]["thr"]) == \
            (later[k]["verdict"], later[k]["labels"], later[k]["chat_thr"]), k
    assert [v["contrasts"][k]["verdict"] for k in S002_KEYS] == ["TIE", "TIE", "REJECT"]
    assert v["contrasts"]["S007.smear"]["labels"] == [analyze.POOLED, "underpowered"]    # read for the first time
    v = verdicts(tmp_path / "b", 0.03, [])                       # not told what was read earlier: no "later"
    assert all(v["contrasts"][k]["labels"] == [analyze.POOLED, "underpowered"] for k in S002_KEYS)
    v = verdicts(tmp_path / "c", 0.0, [S002_RES])                # the check passes: nothing re-read, no label
    assert v["pooled_sia_check"]["CHAT"]["ok"] and all(v["contrasts"][k]["labels"] == [] for k in S002_KEYS)
    sec = flat(section("STAGE 1 VERDICTS"))
    assert "(checked)" not in sec and '"noise check failed later" (C4\'s text) and "underpowered" when it is given ' \
        "--earlier experiments/S002_block_ablations/results.json; without --earlier it writes \"noise check failed " \
        '(pooled SIA)"' in sec


def k_needed(e3):
    """screens_lib.params()'s rule per class and metric, written out again here."""
    q, out = e3["quantiles"], {}
    for cls in ("same init", "SIA", "new init"):
        d = {}
        for m in ("CHAT", "PROSE"):
            n = e3["noise_5M_250M"][f"F({m})"]
            d[m] = {"same init": n["k_needed_paired"]["1.000%"], "new init": n["k_needed_unpaired"]["1.000%"]}.get(cls) or \
                next(k for k in range(1, 99) if q["factor_df7"] * q["ub80_df7"] * n["SD_d_rel"] / math.sqrt(k) <= 0.01)
        out[cls] = d
    return out


def test_k_needed_per_class_and_the_corrected_wording():
    kn = k_needed(json.load(open(os.path.join(L.E3D, "results.json"))))
    assert kn == {"same init": {"CHAT": 1, "PROSE": 1}, "SIA": {"CHAT": 2, "PROSE": 1}, "new init": {"CHAT": 2, "PROSE": 1}}
    for sid in SIDS:
        r = res(sid)
        x, want = r["k_needed_1pct"], kn[r["class"]]
        assert (x["CHAT"], x["PROSE"], x["class"], x["k"]) == (want["CHAT"], want["PROSE"], max(want.values()), r["k"])
        assert r["k"] == L.params()["k"][r["class"]] == max(2, x["class"])
    sec = flat(section("STAGE 1 VERDICTS"))
    assert "k 2 = k_needed in every class" not in sec
    assert "new init 2 / 1 and SIA 2 / 1, so k equals the class's k_needed there; same init 1 / 1, so S003's k is " \
           "above it" in sec
    fix = text("S003", "notes.txt").partition("\nCORRECTIONS (2026-10-06")[2]
    assert flat("k_needed(1%) for same init is 1 on CHAT and 1 on PROSE (E3 results.json k_needed_paired), and k = max(2, "
                "1) = 2: k is above k_needed") in flat(fix)


def test_out_copies_are_the_recorded_files():
    """C9 out/<run>/: exactly the files results.json hashes, each byte-equal (sha256) to the PC copy it recorded."""
    for sid in SIDS:
        r, d = res(sid), os.path.join(EXP, L.SCREENS[sid]["dir"])
        want = r["source"]["files_sha256"]
        got = sorted(os.path.relpath(p, os.path.join(d, "out")) for p in glob.glob(os.path.join(d, "out", "*", "*")))
        assert got == sorted(want) and r["out"]["files"] == ["out/" + f for f in sorted(want)]
        assert len(want) == 3 * 2 * (1 + len(r["contrasts"]))          # 3 files x 2 seeds x (BASE + each arm)
        for f, sha in want.items():
            assert hashlib.sha256(open(os.path.join(d, "out", f), "rb").read()).hexdigest() == sha, (sid, f)
        assert "out/<run>/ now holds log.jsonl, bpb.jsonl and diag.jsonl" in flat(text(sid, "notes.txt"))


def test_privacy_check_flags_what_it_must_and_passes_the_outputs():
    pats = privacy.patterns()
    assert {s for s, _, _ in pats} >= {"generic", "runtime"}
    for probe in ("/home/someone/planck", "/Users/someone/x", "C:\\Users\\someone", "~/.ssh/key", "id_ed25519",
                  "10.11.12.13", "PLANCK_PC_HOST", "%USERNAME%", "someone@example.org", "/x/" + getpass.getuser() + "/y"):
        assert privacy.hits(probe, pats), probe
    assert not privacy.hits("bpb 1.15250 step 7,630 ~/planck/runs/SCREENS ~/.venvs/planck torch 2.13.0+cu130 "
                            "2026-10-05T21:05:49-0400 tok_v0_8k 250.0M", pats)
    if os.path.isfile(privacy.LOCAL):                             # git-ignored; absent on a fresh clone
        assert any(s == "local" for s, _, _ in pats)
    for sid in SIDS:
        for p in glob.glob(os.path.join(EXP, L.SCREENS[sid]["dir"], "out", "*", "*")):
            assert not privacy.hits(open(p).read(), pats), (sid, os.path.basename(os.path.dirname(p)), privacy.hits(open(p).read(), pats))
    src = open(os.path.join(HERE, "test_screens_stage1_verdicts.py")).read() + open(os.path.join(HERE, "privacy.py")).read()
    assert not [x for s, x, rx in pats if s != "generic" and rx.search(src)]        # no name or address in the tests
