"""SCREENS check refusals (no model): each config is copied into a scratch tree with the screen's directory name,
changed in one way, and screens_lib.check must refuse it (C2 extra / missing / wrong keys, LRs, g, the RC-12
GUARD on BASE and arms, C1-b, the C6 hook, seeds, the 2% rule, the S003 search references). Also the generator: the seed sets
for every pick, written into a scratch tree, pass check and come out seed-major.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_screens_refusals.py
"""
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import screens_lib as L  # noqa: E402

P = L.params()


def variant(tmp_path, name, edit):
    """The run's config in tmp/<its dir>/configs/, extends made absolute, then edit(text) -> text."""
    src = L.find(name)[0]
    d = tmp_path / os.path.basename(os.path.dirname(os.path.dirname(src))) / "configs"
    d.mkdir(parents=True, exist_ok=True)
    txt = open(src).read()
    txt = re.sub(r"^extends: .*$", f"extends: {L.BASE_YAML}", txt, flags=re.M)
    txt = edit(txt)
    p = d / (re.search(r"^name: (\S+)$", txt, re.M)[1] + ".yaml")   # the file is named after the edited name
    p.write_text(txt)
    return str(p)


def refusals(tmp_path, name, edit):
    return " | ".join(L.check(variant(tmp_path, name, edit), P))


def test_unchanged_copies_pass(tmp_path):
    for n in ("base_s101", "s005_base_s101", "s001_nogate_g1_s1", "s004_canonac_g2_s1", "s003_adamw_e3_r1_b125M"):
        assert refusals(tmp_path, n, lambda t: t) == ""


CASES = [
    ("s001_nogate_g1_s1", lambda t: t + "train: {grad_clip: 0.5}\n", "differs from its BASE in train.grad_clip"),
    ("s002_novres_g2_s1", lambda t: t.replace("value_residual: false", "value_residual: false, qk_norm: false"),
     "differs from its BASE in model.qk_norm"),
    ("s001_nogate_g1_s1", lambda t: t.replace("model: {attn_gate: false}\n", ""), "must set model.attn_gate"),
    ("s004_canonac_g1_s1", lambda t: t.replace("canon: AC", "canon: A"), "must set model.canon"),
    ("s006_mtp_g1_s1", lambda t: t.replace("mtp_weight: 1.0", "mtp_weight: 0.5"), "must set train.mtp_weight"),
    ("s005_forget_g1_s1", lambda t: t.replace("micro_batch: 8, grad_accum: 2", "micro_batch: 16, grad_accum: 1"),
     "differs from its BASE in train.micro_batch"),
    ("s005_base_s101", lambda t: t.replace("doc_attn: mask, ", ""), "must set train.doc_attn"),
    ("s002_noqknorm_g2_s1", lambda t: t.replace("embed_lr: 0.006", "embed_lr: 0.003"), "optim.embed_lr = 0.003 but C3 gives 0.006"),
    ("s007_smear_g1_s1", lambda t: t.replace("_g1_", "_g3_").replace("lr: 0.003", "lr: 0.009"), "g 3 is not a power of 2"),
    ("base_s101", lambda t: t + "eval: {rc12: {every: 0}}\n", "RC-12 GUARD"),
    ("s001_nogate_g1_s1", lambda t: t + "eval: {rc12: {every: 153}}\n", "RC-12 GUARD"),
    ("base_s101", lambda t: t.replace("optim: {lr: 0.003", "optim: {lr: 0.006"), "differs from its BASE in optim.lr"),
    ("base_s101", lambda t: t + "train: {compile: default}\n", "C1-b"),
    ("s007_smear_g1_s1", lambda t: t + "train: {compile: default}\n", "C1-b"),
    ("s003_adamw_e3_r1_b62M", lambda t: t.replace("train: {total_steps", "train: {compile: default, total_steps"), "C1-b"),
    ("s004_canonac_g1_s1", lambda t: t + "train: {ce_chunk_rows: 2048}\n", "C1-b"),
    ("s001_nogate_g1_s1", lambda t: t + "eval: {induction: {every: 0}}\n", "C6"),
    ("base_s101", lambda t: t + "eval: {induction: {sha256: abc}}\n", "C6"),
    ("s003_adamw_e6_r1_trunk", lambda t: t + "eval: {induction: {gate_windows: 8}}\n", "C6"),
    ("s007_smear_g1_s1", lambda t: t + "engine_fixed: \"2026-09-26 93ea41c\"\n", "engine_fixed"),
    ("base_s102", lambda t: t.replace("seed: 102", "seed: 7").replace("_s102", "_s7"), "seed 7"),
    ("s002_novres_g1_s1", lambda t: t.replace("value_residual: false", "value_residual: false, d_model: 256"),
     "outside 2% of BASE"),
    ("s003_adamw_e3_r1_trunk", lambda t: t.replace("kind: adamw", "kind: adamw, weight_decay: 0.05"),
     "differs from its BASE in optim.weight_decay"),
    ("s003_adamw_e3_r1_b62M", lambda t: t.replace("runs/SCREENS/s003_adamw_e3_r1_trunk", "runs/E2/5m_e3_r1_trunk"),
     "is not this arm's SCREENS trunk"),
    ("s003_adamw_e3_r1_b62M", lambda t: t.replace("stable_00001526", "stable_00003052"), "is not this arm's SCREENS trunk"),
    ("s003_adamw_e3_r1_trunk", lambda t: t.replace("schedule: {mode: trunk", "schedule: {init_from: x.pt, mode: trunk"),
     "is not this arm's SCREENS trunk"),
    ("s003_adamw_e3_r1_b250M", lambda t: t.replace("decay_steps: 1526", "decay_steps: 1500"),
     "differs from its BASE in schedule.decay_steps"),
    ("s003_adamw_e6_r1_trunk", lambda t: t.replace(", kind: adamw", ""), "must set optim.kind"),
    ("s001_nogate_g1_s1", lambda t: t.replace("out_dir: ../../../planck_root/runs/SCREENS/", "out_dir: ../../../planck_root/runs/E3/"),
     "out_dir do not match"),
]


@pytest.mark.parametrize("name,edit,want", CASES, ids=[f"{c[0]}:{c[2][:24]}" for c in CASES])
def test_check_refuses(tmp_path, name, edit, want):
    assert want in refusals(tmp_path, name, edit)


def test_a_screen_config_outside_its_directory_is_refused(tmp_path):
    p = variant(tmp_path, "s001_nogate_g1_s1", lambda t: t)
    q = tmp_path / "S002_block_ablations" / "configs"
    q.mkdir(parents=True)
    os.rename(p, q / "s001_nogate_g1_s1.yaml")
    assert "not in experiments/S001_attn_gate/configs" in " ".join(L.check(str(q / "s001_nogate_g1_s1.yaml"), P))
    p = variant(tmp_path, "base_s101", lambda t: t)
    os.rename(p, q / "base_s101.yaml")
    assert "not in experiments/screens/configs" in " ".join(L.check(str(q / "base_s101.yaml"), P))


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """Empty config dirs with the real names and a screens_base.yaml whose extends is absolute."""
    for d in ["screens"] + [s["dir"] for s in L.SCREENS.values()]:
        (tmp_path / d / "configs").mkdir(parents=True)
    base = open(L.BASE_YAML).read().replace("extends: ../../E2_lr_transfer", f"extends: {L.E2D}")
    (tmp_path / "screens" / "configs" / "screens_base.yaml").write_text(base)
    import screens
    monkeypatch.setattr(L, "EXPD", str(tmp_path))
    monkeypatch.setattr(screens, "PLANS", str(tmp_path / "plans"))
    return tmp_path, screens


def runs(tmp_path, name):
    return [ln.split()[1] for ln in open(tmp_path / "plans" / f"{name}.txt") if ln.startswith("train ")]


def test_seed_sets_for_every_pick_pass_check_and_are_seed_major(scratch):
    tmp_path, screens = scratch
    for g in (0.25, 0.5, 1.0, 2.0, 4.0):           # each pick from empty config dirs
        for d in ["screens"] + [s["dir"] for s in L.SCREENS.values()]:
            for f in (tmp_path / d / "configs").iterdir():
                if f.name != "screens_base.yaml":
                    f.unlink()
        screens.main(["seeds", "--stage", "1", "--pick", f"S001.nogate={g}", "--pick", "S002.novres=1",
                      "--pick", "S002.noqknorm=0.5", "--pick", "S002.nonormscale=2", "--s003", "0.0015,2"])
        gs = L.num(g)
        assert runs(tmp_path, "stage1_seeds") == [r for s in (101, 102) for r in (
            f"base_s{s}", f"s001_nogate_g{gs}_s{s}", f"s002_novres_g1_s{s}", f"s002_noqknorm_g0.5_s{s}",
            f"s002_nonormscale_g2_s{s}", f"s003_adamw_e1.5_r2_s{s}")]
        f = L.flat(L.resolve(L.find(f"s001_nogate_g{gs}_s101")[0]))
        assert [f[k] for k in L.LRK] == [g * 0.003] * 3 and f["seed"] == 101
        f = L.flat(L.resolve(L.find("s003_adamw_e1.5_r2_s102")[0]))
        assert [f[k] for k in L.LRK] == [0.0015, 0.003, 0.003] and f["optim.kind"] == "adamw"


def test_stage_2_seed_sets_add_matched_lr_ind_runs(scratch):
    tmp_path, screens = scratch
    screens.main(["seeds", "--stage", "2", "--pick", "S005.forget=1", "--pick", "S004.canonac=0.5",
                  "--pick", "S007.smear=2", "--pick", "S006.mtp=1"])
    assert runs(tmp_path, "stage2_seeds") == [r for s in (101, 102) for r in (
        f"base_s{s}", f"s005_base_s{s}", f"s005_forget_g1_s{s}", f"s004_canonac_g0.5_s{s}", f"s007_smear_g2_s{s}",
        f"s007_smear_g1_s{s}", f"s006_mtp_g1_s{s}")]
    for p in L.all_configs():
        assert L.check(p, P) == []


def test_generator_refuses_missing_picks(scratch):
    _, screens = scratch
    with pytest.raises(SystemExit):
        screens.main(["seeds", "--stage", "1", "--pick", "S001.nogate=1", "--s003", "0.003,1"])
    with pytest.raises(SystemExit):
        screens.main(["config", "S002.novres", "--g", "3"])
