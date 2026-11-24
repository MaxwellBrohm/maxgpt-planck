"""kconfig.py (K3 round 16, 2026-11-21): the STAGE 0 and K1 run configs, their registry check and the queue plans.
Everything is written under tmp_path laid out as the code root (experiments/K/configs, with experiments/E2_lr_transfer
linked in), never into the tree."""
import json
import os
import shutil

import pytest

import kcommon as K
import kconfig as KC
import ktrain
import runio

SHA = "ab" * 32


@pytest.fixture
def root(tmp_path, monkeypatch):
    return _mkroot(tmp_path, monkeypatch)

def _mkroot(tmp_path, monkeypatch):
    ex = tmp_path / "experiments"
    (ex / "K" / "configs").mkdir(parents=True)
    os.symlink(os.path.join(K.ROOT, "experiments", "E2_lr_transfer"), ex / "E2_lr_transfer")
    # the code root's prereg.yaml (K.ROOT, K_CODE_ROOT under the mutant runner), not KC.HERE's: a scratch copy of
    # code/ has no ../configs, so every test here errored at setup and mutation_k16's baseline exited 1 (2026-11-22)
    shutil.copy(os.path.join(K.ROOT, "experiments", "K", "configs", "prereg.yaml"), ex / "K" / "configs" / "prereg.yaml")
    ev = tmp_path / "evals"
    ev.mkdir()
    (ev / "manifest.json").write_text(json.dumps({"sets": {"k_probe": {"n": 2048, "sha256": SHA}}}))
    monkeypatch.setattr(KC, "CONF", str(ex / "K" / "configs"))
    monkeypatch.setattr(KC, "PLANS", str(ex / "K" / "plans"))
    monkeypatch.setattr(KC, "READINGS", str(ex / "K" / "results" / "readings.jsonl"))
    assert KC.main(["base"]) == 0
    return {"conf": ex / "K" / "configs", "plans": ex / "K" / "plans", "evals": str(ev),
            "readings": ex / "K" / "results" / "readings.jsonl"}


def cfg(root, name):
    return runio.load_yaml(str(root["conf"] / f"{name}.yaml"))


def pick(root, g, final=True):
    root["readings"].parent.mkdir(parents=True, exist_ok=True)
    with open(root["readings"], "a") as f:
        f.write(json.dumps({"reading": "K0-g", "g": g, "final": final}) + "\n")


def test_lr5_is_e2s_pick_as_e3_confirmed_it():
    assert KC.lr5() == (0.003, 1.0)
    assert KC.lrs(0.5) == {"lr": 0.0015, "embed_lr": 0.0015, "scalar_lr": 0.0015}


def test_k0_writes_smoke_and_the_g_grid_at_both_selection_seeds(root):
    assert KC.main(["k0", "--evals", root["evals"]]) == 0
    plan = (root["plans"] / "k0.txt").read_text().splitlines()
    runs = [x.split()[1] for x in plan if x.startswith("train ")]
    assert runs == ["k0_smoke_s200", "k0_smoke_off_s200", "k0_smoke_rep_s200", "k0g_g0p5_s199", "k0g_g1_s199", "k0g_g2_s199",
                    "k0g_g0p5_s200", "k0g_g1_s200", "k0g_g2_s200"] and plan[-1].startswith("mark ")
    # 2026-11-22 (REVIEW 17b): the grid waits for the smoke's pass mark (SPEC 13: the K smoke run passed first)
    assert plan[3] == "wait_mark K K0-smoke passed" and plan[:3] == [f"train {n}" for n in runs[:3]]
    for n in runs:
        c = cfg(root, n)
        assert KC.check(str(root["conf"] / f"{n}.yaml")) == [], n
        g = 1.0 if "smoke" in n else KC.g_of(n.split("_")[1][1:])
        assert c["optim"]["lr"] == pytest.approx(0.003 * g) and c["optim"]["embed_lr"] == pytest.approx(0.003 * g)
        assert c["seed"] == int(n.rsplit("_s", 1)[1]) and c["train"]["compile"] is False
        assert c["train"]["total_steps"] == (306 if "smoke" in n else 7630) and c["schedule"]["warmup_steps"] == 76
        assert [s["name"] for s in c["data"]["sources"]][:3] == ["skill", "fact_bio", "fact_qa"]
        assert c["out_dir"] == f"../../../planck_root/runs/K/{n}" and ktrain.check(c) == (c.get("k") or {})
    assert "k" not in cfg(root, "k0_smoke_off_s200") and "k" not in cfg(root, "k0_smoke_rep_s200")
    assert cfg(root, "k0g_g1_s199")["k"]["probe"] == {
        "every": 153, "files": [{"file": "../../../planck_root/data/k/evals/k_probe.jsonl", "sha256": SHA}]}


def test_the_check_refuses_every_registry_breach(root):
    KC.main(["k0", "--evals", root["evals"]])
    p = root["conf"] / "k0g_g1_s199.yaml"
    good = p.read_text()
    cases = [("seed: 199", "seed: 201", "not among"), ('"lr": 0.003,', '"lr": 0.0031,', "not LR5"),
             ('"share": 0.25', '"share": 0.3', "data_block"), ("data:\n", "train: {compile: true}\ndata:\n", "compile"),
             ('"every": 153', '"every": 100', "K-PROBE every"), ("k:\n", "k:\n  wd_mult: {mlp: 4.0}\n", "k block"),
             ("k:\n", "train: {total_steps: 3815}\nk:\n", "total_steps"),
             ("extends: k_base.yaml", "extends: k_base.yaml\nmodel: {d_model: 256}", "5,010,133"),
             ("name: k0g_g1_s199", "name: k0g_g3_s199", "neither")]
    for a, b, msg in cases:
        assert good.count(a) == 1, a
        p.write_text(good.replace(a, b))
        bad = KC.check(str(p))
        assert any(msg in x for x in bad), (a, b, bad)
    p.write_text(good)
    # 2026-11-22 (mutant k16_seed_any survived): a registered name at a seed of another kind, here K0-g at the paired
    # seed 201, with name, file, out_dir and seed all agreeing, is refused too
    q = root["conf"] / "k0g_g1_s201.yaml"
    q.write_text(good.replace("k0g_g1_s199", "k0g_g1_s201").replace("seed: 199", "seed: 201"))
    assert runio.load_yaml(str(q))["seed"] == 201 and runio.load_yaml(str(q))["name"] == "k0g_g1_s201"
    assert any("not among" in x for x in KC.check(str(q))), KC.check(str(q))
    q.unlink()
    shutil.copy(p, root["conf"] / "k0g_g2_s199_copy.yaml")
    assert any("file name" in x for x in KC.check(str(root["conf"] / "k0g_g2_s199_copy.yaml")))
    os.unlink(root["conf"] / "prereg.yaml")
    assert any("prereg" in x for x in KC.check(str(p)))


def test_smoke_off_differs_from_smoke_only_in_its_k_block(root):
    KC.main(["k0", "--evals", root["evals"]])
    on, off, rep = cfg(root, "k0_smoke_s200"), cfg(root, "k0_smoke_off_s200"), cfg(root, "k0_smoke_rep_s200")
    assert {k: v for k, v in on.items() if k not in ("k", "name", "out_dir")} == \
        {k: v for k, v in off.items() if k not in ("name", "out_dir")} == \
        {k: v for k, v in rep.items() if k not in ("name", "out_dir")}
    p = root["conf"] / "k0_smoke_off_s200.yaml"
    p.write_text(p.read_text() + "k:\n  probe: " + json.dumps(on["k"]["probe"]) + "\n")
    assert any("k block" in x for x in KC.check(str(p)))


def test_k0x_writes_one_registered_extension(root):
    assert KC.main(["k0x", "4", "--evals", root["evals"]]) == 0
    assert [x.split()[1] for x in (root["plans"] / "k0x.txt").read_text().splitlines()[:2]] == [
        "k0g_g4_s199", "k0g_g4_s200"]
    assert cfg(root, "k0g_g4_s200")["optim"]["lr"] == pytest.approx(0.012)
    with pytest.raises(SystemExit):
        KC.main(["k0x", "3", "--evals", root["evals"]])
    assert not (root["conf"] / "k0g_g3_s199.yaml").exists()


def test_k1_takes_g_k_only_from_a_final_k0g_pick(root):
    with pytest.raises(SystemExit, match="no final K0-g pick"):
        KC.main(["k1", "--evals", root["evals"]])
    pick(root, 0.5, final=False)
    with pytest.raises(SystemExit, match="no final K0-g pick"):
        KC.main(["k1", "--evals", root["evals"]])
    pick(root, 2.0)
    assert KC.main(["k1", "--evals", root["evals"]]) == 0
    a = (root["plans"] / "k1a.txt").read_text().splitlines()
    assert a == ["train k0cap_s200", "train k1_fh_s201", "mark K1-b: read LOADED(FH) at seed 201", "stop"]
    b = [x.split()[1] for x in (root["plans"] / "k1b.txt").read_text().splitlines() if x.startswith("train")]
    assert b == ["k1_f0_s201", "k1_fl_s201", "k1_f0_s202", "k1_fl_s202", "k1_fh_s202", "k1_f0_s203", "k1_fl_s203",
                 "k1_fh_s203"]
    cap = cfg(root, "k0cap_s200")
    assert [s["name"] for s in cap["data"]["sources"]] == ["fact_bio", "fact_qa"]
    assert cap["optim"]["lr"] == pytest.approx(0.006) and cfg(root, "k1_f0_s203")["optim"]["lr"] == pytest.approx(0.006)
    assert KC.check(str(root["conf"] / "k1_fl_s202.yaml"), 2.0) == []
    assert any("not LR5" in x for x in KC.check(str(root["conf"] / "k1_fl_s202.yaml"), 1.0))
    pick(root, 1.0, final=False)                 # a later provisional pick leaves no final g_K
    assert KC.final_pick(str(root["readings"])) is None


def test_an_existing_config_is_never_overwritten_and_a_refused_one_is_deleted(root):
    KC.main(["k0", "--evals", root["evals"]])
    p = root["conf"] / "k0g_g1_s200.yaml"
    before = p.read_text()
    (root["conf"].parent.parent.parent / "evals" / "manifest.json").write_text(
        json.dumps({"sets": {"k_probe": {"sha256": "cd" * 32}}}))
    with pytest.raises(SystemExit, match="overwrite"):
        KC.main(["k0", "--evals", root["evals"]])
    assert p.read_text() == before
    for f in root["conf"].glob("k0*.yaml"):
        f.unlink()
    orig = KC.text
    KC.text = lambda n, g, pr, h: orig(n, g, pr, h).replace("seed: 200", "seed: 202")
    try:
        with pytest.raises(SystemExit, match="refused"):
            KC.main(["k0", "--evals", root["evals"]])
    finally:
        KC.text = orig
    assert not (root["conf"] / "k0_smoke_s200.yaml").exists()


PLAN_NAMES = ("k0", "k0x", "k1a", "k1b")

def _pin(tree):
    return runio.load_yaml(os.path.join(tree, "configs", "k0g_g1_s199.yaml"))["k"]["probe"]["files"][0]["sha256"]

def _set_pin(evals, pin):
    with open(os.path.join(evals, "manifest.json"), "w") as f:
        json.dump({"sets": {"k_probe": {"sha256": pin}}}, f)

def _matches_kconfig(tree, tmp, monkeypatch) -> list[str]:
    """2026-11-23 (K review finding 2): every config and plan in `tree` (an experiments/K directory) against what
    kconfig.py writes, in a fresh code root under tmp, for the plans the tree holds: base and k0 always, k0x at the
    one extension g its K0-g configs hold, k1 (k1a.txt and k1b.txt) from the tree's own results/readings.jsonl when it
    holds either; plans/CURRENT names one of the plans written. [] when all of it matches, else what does not."""
    r, bad = _mkroot(tmp, monkeypatch), []
    cf, pl = os.path.join(tree, "configs"), os.path.join(tree, "plans")
    _set_pin(r["evals"], _pin(tree))
    have = set(os.listdir(pl))
    ext = sorted({KC.g_of(m["g"]) for f in os.listdir(cf) if f.endswith(".yaml")
                  for m in [KC.NAME_RE.match(f[:-5])] if m and m["g"] and KC.g_of(m["g"]) in KC.G_EXT})
    try:
        KC.main(["k0", "--evals", r["evals"]])
        if len(ext) > 1:
            bad.append(f"more than one K0-g extension: {ext}")
        for g in ext:
            KC.main(["k0x", f"{g:g}", "--evals", r["evals"]])
        if have & {"k1a.txt", "k1b.txt"}:
            rd = os.path.join(tree, "results", "readings.jsonl")
            if os.path.exists(rd):
                r["readings"].parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(rd, r["readings"])
            KC.main(["k1", "--evals", r["evals"]])
    except SystemExit as e:
        bad.append(f"kconfig.py refused: {e}")
    mine = {f for f in os.listdir(r["conf"]) if f.endswith(".yaml")}
    theirs = {f for f in os.listdir(cf) if f.endswith(".yaml")}
    if mine != theirs:
        bad.append(f"configs only kconfig writes {sorted(mine - theirs)}, only the tree holds {sorted(theirs - mine)}")
    bad += [f"config {f} is not kconfig's output" for f in sorted(mine & theirs)
            if (r["conf"] / f).read_text() != open(os.path.join(cf, f)).read()]
    plans = set(os.listdir(r["plans"]))
    if have - {"CURRENT"} != plans:
        bad.append(f"plans: the tree holds {sorted(have - {'CURRENT'})}, kconfig writes {sorted(plans)}")
    bad += [f"plan {f} is not kconfig's output" for f in sorted(plans & have)
            if (r["plans"] / f).read_text() != open(os.path.join(pl, f)).read()]
    cur = [x.strip() for x in open(os.path.join(pl, "CURRENT")) if x.strip() and not x.startswith("#")] \
        if "CURRENT" in have else []
    if len(cur) != 1 or cur[0] not in PLAN_NAMES or cur[0] + ".txt" not in plans:
        bad.append(f"plans/CURRENT names {cur}, not one plan kconfig wrote")
    return bad

def test_the_committed_configs_and_plans_are_kconfigs_output(tmp_path, monkeypatch):
    """2026-11-22 (REVIEW 17b): experiments/K/configs and plans as committed (at the code root, K.ROOT) are byte for
    byte what kconfig.py writes with the K-PROBE pin they hold; a plan written before a kconfig change (plans/k0.txt
    without the smoke gate, until 2026-11-22) cannot ship. [2026-11-23, K review finding 2: this test pinned the k0
    state (plans exactly CURRENT and k0.txt, CURRENT naming k0), so K OPEN step 6's k0x commit, and k1's later, would
    have turned the suite red. It now accepts every plan kconfig writes (k0, k0x, k1a, k1b), CURRENT naming any one.]"""
    assert _matches_kconfig(os.path.join(K.ROOT, "experiments", "K"), tmp_path / "regen", monkeypatch) == []

def test_a_tree_at_k0x_or_k1_passes_and_a_hand_edit_does_not(tmp_path, monkeypatch):
    """2026-11-23 (K review finding 2): K OPEN step 6's edge branch (kconfig.py k0x 4, plans/CURRENT k0x) and K1's
    (kconfig.py k1 from a final K0-g pick, CURRENT k1a, later k1b), done on a copy of the committed tree as the plan
    says, pass the committed-tree check. A hand-edited plan or config, an extra config, CURRENT naming no plan or two,
    k0x configs without their plan, and k1 plans without the readings they came from do not."""
    src = os.path.join(K.ROOT, "experiments", "K")
    ex = tmp_path / "copy" / "experiments"
    t = ex / "K"
    for d in ("configs", "plans"):
        shutil.copytree(os.path.join(src, d), t / d)
    os.symlink(os.path.join(K.ROOT, "experiments", "E2_lr_transfer"), ex / "E2_lr_transfer")
    ev = tmp_path / "copy" / "evals"
    ev.mkdir()
    _set_pin(str(ev), _pin(str(t)))
    n = iter(range(100))

    def ok():
        return _matches_kconfig(str(t), tmp_path / f"r{next(n)}", monkeypatch)

    def on_copy(*argv):
        monkeypatch.setattr(KC, "CONF", str(t / "configs"))
        monkeypatch.setattr(KC, "PLANS", str(t / "plans"))
        monkeypatch.setattr(KC, "READINGS", str(t / "results" / "readings.jsonl"))
        assert KC.main(list(argv) + ["--evals", str(ev)]) == 0

    def bad_when(path, text, why):
        keep = path.read_text() if path.exists() else None
        path.write_text(text)
        assert ok(), why
        path.unlink() if keep is None else path.write_text(keep)

    cur = t / "plans" / "CURRENT"
    on_copy("k0x", "4")
    cur.write_text(cur.read_text().replace("\nk0\n", "\nk0x\n"))
    assert cur.read_text().splitlines()[-1] == "k0x" and ok() == []
    bad_when(cur, "k9\n", "CURRENT naming no plan")
    bad_when(cur, "k0\nk0x\n", "CURRENT naming two plans")
    x = t / "plans" / "k0x.txt"
    bad_when(x, x.read_text() + "train k0g_g4_s201\n", "a hand-edited k0x plan")
    c4 = t / "configs" / "k0g_g4_s200.yaml"
    bad_when(c4, c4.read_text().replace("seed: 200", "seed: 201"), "a hand-edited k0x config")
    bad_when(t / "configs" / "k0g_g4_s201.yaml", c4.read_text(), "an extra config")
    keep = x.read_text()
    x.unlink()
    assert ok(), "k0x configs without plans/k0x.txt"
    x.write_text(keep)
    (t / "results").mkdir()
    (t / "results" / "readings.jsonl").write_text(json.dumps({"reading": "K0-g", "g": 4.0, "final": True}) + "\n")
    on_copy("k1")
    for name in ("k1a", "k1b"):
        cur.write_text(f"# the plan launch_k.sh runs\n{name}\n")
        assert ok() == [], name
    k1 = t / "configs" / "k1_fl_s202.yaml"
    bad_when(k1, k1.read_text().replace("seed: 202", "seed: 203"), "a hand-edited k1 config")
    keep = (t / "results" / "readings.jsonl").read_text()
    (t / "results" / "readings.jsonl").unlink()
    assert ok(), "k1 plans without the readings they came from"
    (t / "results" / "readings.jsonl").write_text(keep)
    assert ok() == []
