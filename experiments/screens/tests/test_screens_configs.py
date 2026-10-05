"""SCREENS configs, no model: E2/E3 parameters, C1 (BASE = E3 arm A) and C1-b, C2 (every arm differs from its BASE
in exactly its registered keys, LRs g x LR5), the RC-12 GUARD, the 2% rule, harness resolution at full size,
E2's preflight checks, the plans' ORDER, the queue's config lookup (bash 3.2) and the C6 IND file (the hours:
test_screens_hours.py).
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_configs.py
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import screens_lib as L  # noqa: E402

P = L.params()
CONFIGS = L.all_configs()
FIX = {"name", "out_dir", "seed"}
HOOK = set(L.hook_keys())


def flat(path):
    return L.flat(L.resolve(path))


def test_parameters_from_e2_and_e3():
    import schedule as S
    assert P["lr5"] == (0.003, 1.0) and P["k"] == {"same init": 2, "SIA": 2, "new init": 2}
    assert all(v == [101, 102] for v in P["seeds"].values()) and abs(P["u"] - 1.3533) < 1e-4
    e2 = flat(os.path.join(L.E2D, "configs", "5m_e3_r1_b250M.yaml"))
    e3 = flat(os.path.join(L.E3D, "configs", "e3_5m_A_s1.yaml"))
    assert e2["train.total_steps"] == e3["train.total_steps"] == 7630 and 7630 * 32768 == 250_019_840
    w = S.full(7630, 0.01, 0.2, 76)
    assert (w.warmup_steps, w.decay_start) == (76, 6104)


def test_base_is_e3_arm_a_key_for_key_eager_plus_the_ind_hook():
    b, e3 = flat(os.path.join(L.config_dir(None), "base_s101.yaml")), flat(os.path.join(L.E3D, "configs", "e3_5m_A_s1.yaml"))
    assert L.diff(b, e3) == FIX | {"engine_fixed"} | HOOK and len(HOOK) == 7
    assert (b["train.compile"], b["train.compile_dynamic"], b["train.ce_chunk_rows"], e3["train.compile"]) == (False, None, 0, False)
    assert b["engine_fixed"] == "2026-10-03 f314eb7 SCREENS C1-b" and b["eval.induction.every"] == 153
    assert b["seed"] == 101 and b["train.doc_attn"] == "varlen" and b["optim.batched"] is True


LATER = {f"s003_adamw_e3_r{r}_{t}" for r in ("0.5", "2", "4", "8") for t in ("trunk", "b62M", "b125M", "b250M")}
# written after a pick (SCREENS.txt STAGE 1 SELECTION A RESULT, 2026-10-05): S003 stage B at eta_A 3e-3
LATER |= {f"s003_adamw_e{e}_r2_{t}" for e in ("1.5", "6") for t in ("trunk", "b62M", "b125M", "b250M")}
# SCREENS.txt S003 STAGE B RESULT (2026-10-05): S003 stage C, g 0.5 and 2 at (eta_A 3e-3, r_B 2)


def test_config_set_is_complete():
    names = sorted(os.path.basename(c)[:-5] for c in CONFIGS)
    assert LATER <= set(names) and len(set(names) - LATER) == 48 and len(names) == 72
    names = sorted(set(names) - LATER)
    for sid, s in L.SCREENS.items():
        for a in s["arms"]:
            if sid != "S003":
                assert {f"{sid.lower()}_{a}_g{g}_s1" for g in ("0.5", "1", "2")} <= set(names)
    assert {"base_s101", "base_s102", "s005_base_s101", "s005_base_s102"} <= set(names)
    assert sum(n.startswith("s003_adamw_e") for n in names) == 20


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: os.path.basename(p)[:-5])
def test_every_config_passes_check(path):
    assert L.check(path, P) == []


def own_base(sid):
    return flat(os.path.join(L.config_dir(sid if L.engine(sid) else None),
                             (f"{sid.lower()}_" if L.engine(sid) else "") + "base_s101.yaml"))


def test_every_arm_differs_from_its_base_only_in_its_registered_keys(capsys):
    rows = []
    for path in CONFIGS:
        m = L.NAME_RE.match(os.path.basename(path)[:-5])
        sid = m["sid"].upper() if m["sid"] else None
        f = flat(path)
        if m["tag"]:                                    # S003 search: against E2's own stage config
            ref = L.s003_e2_reference(float(m["e"]) * 1e-3, float(m["r"]), m["tag"])
            want = {"optim.kind", "engine_fixed"} | HOOK | ({"schedule.init_from"} if m["tag"] != "trunk" else set())
            assert L.diff(f, ref) - FIX == want and f["optim.kind"] == "adamw"
            rows.append(("S003", f"adamw search {m['tag']}", sorted(want)))
        elif m["arm"] == "base":
            d = L.diff(f, own_base(None)) - FIX
            assert d == set(L.engine(sid)) and all(f[k] == v for k, v in L.engine(sid).items())
            rows.append((sid or "BASE", "base", sorted(d)))
        else:
            d = L.diff(f, own_base(sid)) - FIX
            reg = L.SCREENS[sid]["arms"][m["arm"]]
            g = float(m["g"])
            assert d - set(L.LRK) == set(reg) and all(f[k] == v for k, v in reg.items())
            assert (set(L.LRK) <= d) == (g != 1.0)
            rows.append((sid, f"{m['arm']} g{m['g']}", sorted(d)))
    with capsys.disabled():
        for r in sorted(set((a, b.split(" g")[0], tuple(c for c in k if not c.startswith("optim.") or c == "optim.kind"))
                            for a, b, k in rows)):
            print(f"\n  {r[0]:<5} {r[1]:<24} {', '.join(r[2]) or '(none: the BASE)'}", end="")


def test_lrs_are_g_times_lr5_with_r_kept():
    for name, want in (("s001_nogate_g2_s1", 0.006), ("s002_novres_g0.5_s1", 0.0015), ("s007_smear_g1_s1", 0.003)):
        f = flat(L.find(name)[0])
        assert [f[k] for k in L.LRK] == [want] * 3
    f = flat(L.find("s003_adamw_e12_r1_b62M")[0])
    assert [f[k] for k in L.LRK] == [0.012] * 3 and f["schedule.decay_steps"] == 382 and f["train.total_steps"] == 1908


def test_stage_a_reference_is_e2s_own_config_text():
    import e2plan
    for eta in L.S003_STAGE_A:
        for name, txt in e2plan.runs_for("5m", eta, 1.0):
            assert open(os.path.join(L.E2D, "configs", name + ".yaml")).read() == txt


def test_ind_file_regenerates_and_is_pinned_in_base():
    import hashlib
    import make_ind
    assert make_ind.main(["--check"]) == 0
    raw = open(make_ind.OUT, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == L.hook_keys()["eval.induction.sha256"]
    seqs = json.loads(raw)["seqs"]
    assert len(seqs) == 64 and {len(x) for x in seqs} == {128} and min(min(x) for x in seqs) >= 7
    assert max(max(x) for x in seqs) < 8192
    base = os.path.dirname(L.find("s001_nogate_g1_s1")[0])
    assert os.path.normpath(os.path.join(base, L.hook_keys()["eval.induction.file"])) == make_ind.OUT


def test_param_counts_within_2_percent():
    from budget import count_analytic, count_mtp
    from config import PlanckConfig
    want = {"base": 5_010_133, "nogate": 5_005_525, "novres": 5_010_112, "noqknorm": 5_009_109, "nonormscale": 5_010_133,
            "adamw": 5_010_133, "canonac": 5_022_421, "forget": 5_014_765, "mtp": 5_010_133, "smear": 5_010_157}
    for path in CONFIGS:
        mc = PlanckConfig.from_dict(L.resolve(path)["model"])
        assert count_analytic(mc)["total"] == want[L.NAME_RE.match(os.path.basename(path)[:-5])["arm"]]
    assert count_mtp(mc, 1) == 37_056


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: os.path.basename(p)[:-5])
def test_resolves_in_the_harness_at_full_size(path):
    import train
    from budget import count_analytic
    from config import PlanckConfig
    from model import ce_chunk_rows, compile_mode
    from mtp import mtp_settings
    cfg = L.resolve(path)
    tc, sc = cfg["train"], cfg["schedule"]
    mc = PlanckConfig.from_dict(cfg["model"])
    assert mtp_settings(tc) == ((1, 1.0) if "s006_mtp" in cfg["name"] else (0, 1.0))
    assert compile_mode(tc["compile"]) is None and ce_chunk_rows(tc["ce_chunk_rows"]) == 0
    train.check_compile(compile_mode(tc["compile"]), tc.get("compile_dynamic"), tc["doc_attn"])
    assert tc["doc_attn"] == ("mask" if mc.forget_gate or cfg["name"].startswith("s005") else "varlen")
    assert tc["micro_batch"] * tc["grad_accum"] * mc.seq_len == 32768 and cfg["optim"]["kind"] in ("normuon", "adamw")
    if sc["mode"] != "branch":
        w, pts = train.build_schedule(sc, tc["total_steps"], count_analytic(mc)["total"], 32768, None)
        assert (w.warmup_steps, w.total_steps) == (76, tc["total_steps"])
        assert (w.decay_start, pts) == ((6104, set()) if sc["mode"] == "full" else (6105, {1526, 3052, 6104}))
    else:
        assert tc["total_steps"] == int(sc["init_from"][-11:-3]) + sc["decay_steps"]


def test_e2_preflight_refuses_only_the_missing_pc_data():
    import preflight
    reads = preflight.harness_reads(os.path.join(L.ROOT, "harness"))
    for path in CONFIGS:
        rep = preflight.check_one(path, L.ROOT, False, reads, init_check=False)
        other = [r for r in rep["refusals"] if not r.startswith(("provenance", "source "))]
        assert other == [], (path, other)
        assert rep["engine"]["train.compile"] is False and rep["n_params"] > 4_900_000


def plan_runs(name):
    return [ln.split()[1] for ln in open(os.path.join(L.HERE, "plans", name + ".txt")) if ln.startswith("train ")]


def test_plans_follow_the_registered_order():
    s1 = plan_runs("stage1_select")
    want = [f"s001_nogate_g{g}_s1" for g in ("0.5", "1", "2")]
    want += [f"s002_{a}_g{g}_s1" for a in ("novres", "noqknorm", "nonormscale") for g in ("0.5", "1", "2")]
    want += [f"s003_adamw_e{e}_r1_{t}" for e in ("0.75", "1.5", "3", "6", "12") for t in ("trunk", "b62M", "b125M", "b250M")]
    assert s1 == want
    s2 = plan_runs("stage2_select")
    assert [r.split("_")[0] for r in s2[::3]] == ["s005", "s004", "s007", "s006"] and len(s2) == 12
    for r in s1 + s2:
        assert len(L.find(r)) == 1
    lines = open(os.path.join(L.HERE, "plans", "stage1_select.txt")).read().splitlines()
    assert lines[1] == "wait_mark SCREENS SCREENS PRECONDITIONS OK" and lines[-1].startswith("mark ")
    assert open(os.path.join(L.HERE, "plans", "stage2_select.txt")).read().splitlines()[1] == \
        "wait_mark SCREENS SCREENS STAGE 2 SMOKES RECORDED"
    cur = [ln for ln in open(os.path.join(L.HERE, "plans", "CURRENT")) if not ln.startswith("#")]
    assert len(cur) == 1 and cur[0].strip() in PLAN_SEQUENCE    # ORDER's plans so far, in the order they run
    for r in plan_runs(cur[0].strip()):
        assert len(L.find(r)) == 1


PLAN_SEQUENCE = ["stage1_select", "stage1_s003B", "stage1_s003C"]   # stage1_s003B: SCREENS.txt STAGE 1 SELECTION A
# RESULT; stage1_s003C: S003 STAGE B RESULT


def cfg_for(code, name):
    q = os.path.join(L.HERE, "qlib_screens.sh")
    return subprocess.run(["/bin/bash", "-c", f'CODE="{code}"; . "{q}"; cfg_for "{name}"'], capture_output=True, text=True)


def test_queue_finds_exactly_one_config(tmp_path):
    r = cfg_for(L.ROOT, "s005_forget_g1_s1")
    assert r.returncode == 0 and r.stdout.strip() == L.find("s005_forget_g1_s1")[0]
    assert cfg_for(L.ROOT, "base_s101").stdout.strip() == L.find("base_s101")[0]
    assert cfg_for(L.ROOT, "s001_nogate_g3_s1").returncode == 1
    for d in ("screens", "S001_attn_gate", "S002_block_ablations"):
        os.makedirs(tmp_path / "experiments" / d / "configs")
    shutil.copy(L.find("base_s101")[0], tmp_path / "experiments" / "screens" / "configs")
    assert cfg_for(tmp_path, "base_s101").returncode == 0
    shutil.copy(L.find("base_s101")[0], tmp_path / "experiments" / "S002_block_ablations" / "configs")
    r = cfg_for(tmp_path, "base_s101")
    assert r.returncode == 1 and r.stdout == ""


def test_shell_scripts_parse():
    for s in ("queue_screens.sh", "launch_screens.sh", "qlib_screens.sh"):
        assert subprocess.run(["/bin/bash", "-n", os.path.join(L.HERE, s)]).returncode == 0
