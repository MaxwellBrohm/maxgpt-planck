"""k0read.py (K3 round 16, 2026-11-21): K0-g's K-DEV objective and its two-seed pick, on hand-built scorer rows."""
import json
import math

import pytest

import k0read as R
import kconfig as KC

SHA = "ef" * 32


def rows(skill_nll, fb_nll, n_sk=1024, n_fb=1024, sha=SHA, extra=()):
    fams = "RUBFAP"
    out = [{"id": f"s{i}", "set": "K-DEV", "fam": fams[i % 6], "ce_full": skill_nll, "item_sha256": sha}
           for i in range(n_sk)]
    out += [{"id": f"b{i}", "set": "K-DEV", "fam": "bio", "ce_full": fb_nll, "item_sha256": sha} for i in range(n_fb)]
    out += [{"id": f"e{i}", "set": "K-EVAL-ID", "fam": "R", "ce_full": 99.0, "item_sha256": "00"} for i in extra]
    return out


def run(tmp_path, name, rs=None, diverged=False):
    d = tmp_path / name
    d.mkdir()
    if diverged:
        (d / "DIVERGED").write_text("")
    if rs is not None:
        (d / "k_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    return str(d)


def test_objective_weights_skill_and_fb_equally_and_reads_only_k_dev(tmp_path):
    """Skill NLL 2.0 over 1,024 items and FB NLL 4.0 over 1,024 prompts: 3.0, whatever the counts' ratio would give;
    K-EVAL-ID rows in the same file are not read."""
    o = R.objective(run(tmp_path, "a", rows(2.0, 4.0, extra=range(50))))
    assert o["objective"] == pytest.approx(3.0) and (o["n_skill"], o["n_fb"]) == (1024, 1024)
    assert o["skill_nll"] == 2.0 and o["fb_nll"] == 4.0 and o["item_sha256"] == SHA
    rs = rows(2.0, 4.0)
    rs[0]["ce_full"] = 2.0 + 1024.0           # one skill row moves the skill mean by 1.0: objective + 0.5
    assert R.objective(run(tmp_path, "b", rs))["objective"] == pytest.approx(3.5)


def test_objective_refuses_partial_mixed_or_missing_scores(tmp_path):
    with pytest.raises(ValueError, match="rows 1000 skill, 1024 FB"):
        R.objective(run(tmp_path, "a", rows(2.0, 4.0, n_sk=1000)))
    with pytest.raises(ValueError, match="rows 1024 skill, 512 FB"):
        R.objective(run(tmp_path, "b", rows(2.0, 4.0, n_fb=512)))
    mixed = rows(2.0, 4.0)
    mixed[5]["item_sha256"] = "12" * 32
    with pytest.raises(ValueError, match="item files"):
        R.objective(run(tmp_path, "c", mixed))
    with pytest.raises(ValueError, match="neither scored"):
        R.objective(run(tmp_path, "d"))
    assert R.objective(run(tmp_path, "e", diverged=True))["objective"] is None
    assert R.objective(run(tmp_path, "f", rows(float("nan"), 4.0)))["objective"] is None


def test_read_picks_by_the_mean_over_both_seeds_and_appends(tmp_path):
    """Objectives (skill, FB) per g and seed: g 1 has the least mean (seed 199 alone picks 0.5, seed 200 alone 2:
    SEED-SPLIT); g 2 at seed 200 diverged, so g 2's mean is inf. The reading is appended, never rewritten, and kconfig
    reads g_K from it."""
    runs = tmp_path / "runs"
    runs.mkdir()
    obj = {(0.5, 199): 2.9, (0.5, 200): 3.2, (1.0, 199): 2.95, (1.0, 200): 3.0, (2.0, 199): 3.3}
    for (g, s), v in obj.items():
        run(runs, f"k0g_{KC.gname(g)}_s{s}", rows(v - 0.5, v + 0.5))
    run(runs, "k0g_g2_s200", diverged=True)
    rd = tmp_path / "readings.jsonl"
    rec = R.read(str(runs), str(rd))
    assert (rec["g"], rec["final"], rec["extend"]) == (1.0, True, None)
    assert rec["per_seed"] == {"199": 0.5, "200": 1.0} and rec["labels"] == ["SEED-SPLIT"]
    assert rec["mean"]["2"] == "inf" and rec["mean"]["1"] == pytest.approx(2.975)
    assert rec["table"]["2"] == {"199": pytest.approx(3.3), "200": None}
    assert KC.final_pick(str(rd)) == 1.0
    R.read(str(runs), str(rd))
    assert len(rd.read_text().splitlines()) == 2


def test_read_asks_the_extension_and_refuses_a_missing_run(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for (g, s), v in {(0.5, 199): 2.0, (0.5, 200): 2.1, (1.0, 199): 2.5, (1.0, 200): 2.6, (2.0, 199): 3.0}.items():
        run(runs, f"k0g_{KC.gname(g)}_s{s}", rows(v, v))
    with pytest.raises(ValueError, match="seeds"):           # g 2 at seed 200 neither run nor diverged: no pick
        R.read(str(runs), str(tmp_path / "r.jsonl"))
    run(runs, "k0g_g2_s200", rows(3.1, 3.1))
    rec = R.read(str(runs), str(tmp_path / "r.jsonl"))
    assert (rec["g"], rec["final"], rec["extend"]) == (0.5, False, 0.25)
    assert KC.final_pick(str(tmp_path / "r.jsonl")) is None
    for s in (199, 200):
        run(runs, f"k0g_g0p25_s{s}", rows(2.2, 2.2))
    rec = R.read(str(runs), str(tmp_path / "r.jsonl"))
    assert (rec["g"], rec["final"]) == (0.5, True) and math.isclose(rec["mean"]["0.25"], 2.2)


def test_runs_scored_on_two_k_dev_files_are_refused(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    for g in (0.5, 1.0, 2.0):
        for s in (199, 200):
            run(runs, f"k0g_{KC.gname(g)}_s{s}", rows(2.0, 2.0, sha=SHA if (g, s) != (2.0, 200) else "12" * 32))
    with pytest.raises(ValueError, match="2 different K-DEV files"):
        R.read(str(runs), str(tmp_path / "r.jsonl"))
    assert not (tmp_path / "r.jsonl").exists()
