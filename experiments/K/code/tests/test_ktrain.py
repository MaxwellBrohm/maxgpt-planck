"""ktrain.py (SPEC 14's harness pieces, K-side): off is harness train.py bit for bit; the K-PROBE hook changes nothing;
the weights-only start; K4b's weight-decay groups; refusals. CPU, tiny models, fake data (harness make_fake_data).
Runs: 12 steps, micro 4 x accum 2, seq 256, vocab 8192 (K's token ids), log every step."""
import hashlib
import json
import os

os.environ.setdefault("PLANCK_TEST_THREADS", os.environ.get("K_TEST_THREADS", "4"))

import pytest                                          # noqa: E402
import torch                                           # noqa: E402

import gen                                             # noqa: E402
import kopt                                            # noqa: E402
import kprobe                                          # noqa: E402
import ktrain                                          # noqa: E402
import probes as PR                                    # noqa: E402
import prune                                           # noqa: E402
import runio                                           # noqa: E402
import train                                           # noqa: E402
from config import PlanckConfig                        # noqa: E402
from make_fake_data import make                        # noqa: E402
from model import build_model                          # noqa: E402
from testutil import read_jsonl, write_run             # noqa: E402

MODEL = {"vocab_size": 8192, "d_model": 32, "n_layers": 2, "n_heads": 2, "n_kv_heads": 2, "head_dim": 16,
         "mlp_hidden": 24, "seq_len": 256}
SRC = lambda d: [{"name": "text", "kind": "tokens", "paths": [f"{d}/text_*.bin"], "eot_id": 1, "share": 0.7},  # noqa
                 {"name": "chat", "kind": "chat", "paths": [f"{d}/chat_*.jsonl"], "share": 0.3}]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(root, name, k=None, steps=12, extra=(), **over):
    p = write_run(str(root / name), str(root / "data"), model=MODEL, data={"sources": SRC(str(root / "data"))},
                  train={"total_steps": steps, "ckpt_every": 6}, **({"k": k} if k is not None else {}), **over)
    return p, os.path.join(os.path.dirname(p), "out")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("ktrain")
    make(str(root / "data"), vocab=8192, docs=120, chats=60)
    assert gen.main(["evals", str(root / "ev"), "--small"]) == 0
    pf = str(root / "ev" / "evals" / "k_probe.jsonl")
    probe = {"every": 5, "files": [{"file": "../ev/evals/k_probe.jsonl", "sha256": sha(pf)}]}
    out = {}
    p, out["plain"] = run(root, "plain")
    assert train.main([p, "--device", "cpu"]) == 0
    for name, k in (("knone", {}), ("probe", {"probe": probe}), ("wd1", {"wd_mult": {"mlp": 1.0}}),
                    ("wd4", {"wd_mult": {"mlp": 4.0}})):
        p, out[name] = run(root, name, k)
        assert ktrain.main([p, "--device", "cpu"]) == 0, name
    return {"root": root, "out": out, "probe": probe, "pf": pf}


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def final(d):
    return runio.load_checkpoint(runio.latest_checkpoint(d))


def logs(d):
    return [{k: v for k, v in r.items() if k not in ("time", "tok_per_s")} for r in read_jsonl(os.path.join(d, "log.jsonl"))]


@pytest.mark.parametrize("name", ["knone", "probe", "wd1"])
def test_off_probe_and_unit_multiplier_train_bitwise_as_harness(runs, name):
    a, b = final(runs["out"]["plain"]), final(runs["out"][name])
    assert a["step"] == b["step"] == 12
    for k in ("model", "data_state", "rng_torch", "tokens", "sup_tokens"):
        assert same(a[k], b[k]), (name, k)
    if name != "wd1":                     # wd1 holds the same values in one more group
        assert same(a["optimizer"], b["optimizer"]), name
    assert logs(runs["out"]["plain"]) == logs(runs["out"][name])
    assert ("k_sha256" in b) == (name != "knone") and not os.path.exists(os.path.join(runs["out"]["plain"],
                                                                                         "k_start.jsonl"))


def test_probe_readings_at_every_and_last_step(runs):
    d = runs["out"]["probe"]
    log = read_jsonl(os.path.join(d, "k_probe.jsonl"))
    assert [r["step"] for r in log] == [5, 10, 12] and not any("error" in r for r in log)
    recs, _ = kprobe.load_records(runs["probe"]["files"], os.path.join(str(runs["root"]), "probe"))
    ck = final(d)
    m = build_model(PlanckConfig.from_dict(ck["model_cfg"]))
    m.load_state_dict(ck["model"])
    import pools as P
    ref = PR.score(PR.model_fn(m, "cpu"), recs, P.get()["V"])          # the scorer's own path, independently
    assert {o["set"] for o in ref} == {"K-PROBE"}
    sk = [o["lik"] for r, o in zip(recs, ref) if "qs" in r]
    fa = [o["gen"] for r, o in zip(recs, ref) if "qs" not in r]
    assert len(sk) == 16 and len(fa) == 16
    assert log[-1]["skill_lik"]["K-PROBE"] == pytest.approx(sum(sk) / len(sk), abs=1e-12)
    assert log[-1]["fact_acc"]["K-PROBE"] == pytest.approx(sum(fa) / len(fa), abs=1e-12)
    assert log[-1]["summary"] == PR.summarize(ref)
    st = read_jsonl(os.path.join(d, "k_start.jsonl"))
    assert len(st) == 1 and st[0]["installed"]["probe"]["n"] == 32 and st[0]["k_sha256"] == ktrain.k_sha(st[0]["k"])


def test_probe_file_pinned_and_sets_taken(runs, tmp_path):
    f = dict(runs["probe"]["files"][0])
    base = os.path.join(str(runs["root"]), "probe")
    with pytest.raises(ValueError, match="sha256"):
        kprobe.load_records([{**f, "sha256": "0" * 64}], base)
    recs, _ = kprobe.load_records([{**f, "sets": ["K-PROBE"], "take": 5}], base)
    assert len(recs) == 5
    # K3 round 15 (REVIEW 16b T-5): take is the FIRST n of the set in file order (K5's "first 256"; mutant
    # r16_probe_take_last), and take without sets is refused, not ignored (mutant r16_probe_take_without_sets_ok)
    rows = [json.loads(x) for x in open(runs["pf"])]
    assert len([r for r in rows if r.get("set") == "K-PROBE"]) > 5
    assert recs == [r for r in rows if r.get("set") == "K-PROBE"][:5]
    with pytest.raises(ValueError, match="take needs sets"):
        kprobe.load_records([{**f, "take": 5}], base)
    # K3 round 14 (REVIEW 15b T-5): a set with no record beside a real one is refused by name, not dropped (mutant
    # r15_probe_missing_set_ok; "no record" alone also matched the empty-list refusal "k.probe: no records")
    with pytest.raises(ValueError, match="holds no record of set 'FB-HIT'"):
        kprobe.load_records([{**f, "sets": ["K-PROBE", "FB-HIT"]}], base)


def test_a_probe_reading_that_draws_from_the_rng_stops_the_run(runs, monkeypatch):
    real = kprobe.reading
    monkeypatch.setattr(kprobe, "reading", lambda *a, **kw: (torch.rand(1), real(*a, **kw))[1])
    p, _ = run(runs["root"], "probe_rng", {"probe": runs["probe"]}, steps=6)
    with pytest.raises(RuntimeError, match="RNG"):
        ktrain.main([p, "--device", "cpu"])


def test_wd_groups_and_their_step(runs):
    ck = final(runs["out"]["wd4"])
    groups = ck["optimizer"]["param_groups"]
    assert [g["name"] for g in groups] == ["matrix", "matrix_mlp", "embed", "scalar"]
    assert [g["weight_decay"] for g in groups] == [pytest.approx(0.1), pytest.approx(0.4), pytest.approx(0.1), 0.0]
    m = build_model(PlanckConfig(**MODEL))
    names = kopt.mlp_param_names(m)
    assert sorted(names) == sorted(f"blocks.{i}.mlp.{w}_proj.weight" for i in range(2) for w in ("gate", "up", "down"))
    assert len(groups[1]["params"]) == 6
    # one optimizer step from equal weights and gradients: non-MLP parameters move identically; MLP ones decay more
    from optim import make_optimizer
    out = {}
    for mult in (1.0, 4.0):
        torch.manual_seed(0)
        m = build_model(PlanckConfig(**MODEL))
        p0 = {n: p.detach().clone() for n, p in m.named_parameters()}
        opt = make_optimizer(m, {"lr": 3e-2, "weight_decay": 0.1})
        kopt.split_mlp_wd(opt, m, mult)
        g = torch.Generator().manual_seed(1)
        for p in m.parameters():
            p.grad = torch.randn(p.shape, generator=g)
        opt.step()
        out[mult] = {n: p.detach().clone() for n, p in m.named_parameters()}
    for n in out[1.0]:
        if n in names:                    # the extra decay pulls toward 0: the difference opposes the weight's sign
            diff = out[4.0][n] - out[1.0][n]
            assert (diff != 0).float().mean() > 0.2 and bool((diff * p0[n] <= 0).all()), n
        else:
            assert torch.equal(out[1.0][n], out[4.0][n]), n
    # K3 round 15 (REVIEW 16b T-6): the split refuses an optimizer that has stepped (its state would sit under the
    # old groups; mutant r16_wd_split_after_step) and one whose MLP matrices are not all in "matrix" (a stray one
    # would keep the base decay; mutant r16_wd_partial_move_ok)
    with pytest.raises(RuntimeError, match="before the first step"):
        kopt.split_mlp_wd(opt, m, 4.0)
    torch.manual_seed(0)
    m = build_model(PlanckConfig(**MODEL))
    opt = make_optimizer(m, {"lr": 3e-2, "weight_decay": 0.1})
    pid = {id(p): n for n, p in m.named_parameters()}
    gm, ge = (next(g for g in opt.param_groups if g["name"] == x) for x in ("matrix", "embed"))
    stray = next(p for p in gm["params"] if pid[id(p)] == "blocks.1.mlp.up_proj.weight")
    gm["params"] = [p for p in gm["params"] if p is not stray]
    ge["params"].append(stray)
    with pytest.raises(RuntimeError, match="5 of 6 MLP matrices"):
        kopt.split_mlp_wd(opt, m, 4.0)


def test_weights_only_start(runs, tmp_path):
    src = runio.latest_checkpoint(runs["out"]["plain"])
    w = str(tmp_path / "w.pt")
    r = prune.prune(src, w, block_last=4, seed=7901)
    assert r["n_units"] == 8 and r["units"] == {"0": [20, 21, 22, 23], "1": [20, 21, 22, 23]}
    p, d = run(runs["root"], "init", {"init_weights": {"file": w, "sha256": r["sha256"]}})
    assert ktrain.main([p, "--device", "cpu", "--max-steps", "0"]) == 0
    ck = final(d)
    wf = torch.load(w, weights_only=False)
    assert ck["step"] == 0 and same(ck["model"], wf["model"]) and ck["optimizer"]["state"] == {}
    assert ck["k_init_weights_sha256"] == r["sha256"] and ck["data_state"]["window"] == -1
    plain = final(runs["out"]["plain"])["model"]
    for k, v in wf["model"].items():
        if ".mlp.down_proj" in k:
            assert torch.equal(v[:, 20:], torch.zeros_like(v[:, 20:])) and torch.equal(v[:, :20], plain[k][:, :20])
        elif ".mlp." in k:
            assert not torch.equal(v[20:], plain[k][20:]) and torch.equal(v[:20], plain[k][:20])
        else:
            assert torch.equal(v, plain[k]), k
    assert ktrain.main([p, "--device", "cpu", "--max-steps", "3"]) == 0        # resumes from its own step 0
    assert final(d)["step"] == 3
    with pytest.raises(ValueError, match="sha256"):
        ktrain.load_init(w, "0" * 64)
    # K3 round 15 (REVIEW 16b T-5): a training checkpoint (optimizer, data state, RNG) pinned as k.init_weights is
    # refused: P-F would otherwise run from unreset weights, as P-0 (mutant r16_init_accepts_full_ckpt)
    assert {"model", "model_cfg", "optimizer"} <= set(torch.load(src, weights_only=False))
    with pytest.raises(ValueError, match="expected a prune.py weights file"):
        ktrain.load_init(src, sha(src))


def test_refusals(runs):
    bad = [({"k": {"nope": 1}}, "unknown"),
           ({"k": {"grad_route": {"sources": ["chat"], "reserved": 4}}, "train": {"compile": "default"}}, "eager"),
           ({"k": {"probe": {"every": 5}}, "train": {"compile": "default"}}, "eager"),
           ({"k": {"grad_route": {"sources": ["fact"], "reserved": 4}}}, "not among"),
           ({"k": {"grad_route": {"sources": [], "reserved": 4}}}, "non-empty"),    # round 14 (REVIEW 15b T-4)
           ({"k": {"grad_route": {"sources": ["chat"], "reserved": 4}}, "data": {"mode": "bucket"}}, "pack"),
           ({"k": {"init_weights": {"file": "w", "sha256": "0"}}, "schedule": {"init_from": "x.pt"}}, "both set"),
           ({"k": {"wd_mult": {"attn": 2.0}}}, "unknown")]
    for over, msg in bad:
        cfg = {**over, "data": {"sources": SRC("d"), **over.get("data", {})}}
        with pytest.raises(ValueError, match=msg):
            ktrain.check(cfg)
    assert ktrain.check({"k": {}}) == {} and ktrain.check({}) == {}
    import trainer
    import data as harness_data
    import optim
    assert train.Trainer is trainer.Trainer and train.build_loader is harness_data.build_loader   # nothing leaks
    assert train.make_optimizer is optim.make_optimizer


def test_prune_units_file(runs, tmp_path):
    src = runio.latest_checkpoint(runs["out"]["plain"])
    u = tmp_path / "u.json"
    u.write_text(json.dumps({"0": [5, 1], "1": []}))
    r = prune.prune(src, str(tmp_path / "w.pt"), units_json=str(u), seed=7901)
    assert r["units"] == {"0": [1, 5]} and r["n_units"] == 2 and r["weights_after"] != r["weights_before"]
    for bad in ({"0": [1, 1]}, {"0": [24]}, {"5": [1]}):
        u.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            prune.prune(src, str(tmp_path / "x.pt"), units_json=str(u), seed=7901)
    with pytest.raises(ValueError, match="seed"):
        prune.prune(src, str(tmp_path / "y.pt"), block_last=2)
    u.write_text(json.dumps({"0": [1]}))        # K3 round 14 (REVIEW 15b T-8): both selections refused, not the block
    with pytest.raises(ValueError, match="exclusive"):      # taken silently (mutant r15_prune_units_and_block_ok)
        prune.prune(src, str(tmp_path / "v.pt"), units_json=str(u), block_last=2, seed=7901)
    same_w = prune.prune(src, str(tmp_path / "z.pt"))
    assert same_w["n_units"] == 0 and same_w["weights_after"] == same_w["weights_before"]


def test_a_resume_mid_continuation_wins_over_the_init_file(runs, tmp_path):
    """REVIEW 14c D1: a weights-only continuation run as 6 steps, a resume and 6 more equals the uninterrupted 12 steps
    bit for bit (weights, optimizer, data state, RNG), and its weights are not the init file's: a resume from the
    run's own checkpoint wins over k.init_weights (mutant r14c_init_clobbers_resume reloads the file after a resume;
    test_weights_only_start resumes from step 0, where the two are equal)."""
    w = str(tmp_path / "w.pt")
    r = prune.prune(runio.latest_checkpoint(runs["out"]["plain"]), w, block_last=4, seed=7901)
    k = {"init_weights": {"file": w, "sha256": r["sha256"]}}
    pa, da = run(runs["root"], "init_whole", k)
    assert ktrain.main([pa, "--device", "cpu"]) == 0
    pb, db = run(runs["root"], "init_split", k)
    assert ktrain.main([pb, "--device", "cpu", "--max-steps", "6"]) == 0 and final(db)["step"] == 6
    assert ktrain.main([pb, "--device", "cpu"]) == 0
    a, b = final(da), final(db)
    assert a["step"] == b["step"] == 12
    for key in ("model", "optimizer", "data_state", "rng_torch", "tokens"):
        assert same(a[key], b[key]), key
    wf = torch.load(w, weights_only=False)["model"]
    assert not any(torch.equal(wf[n], a["model"][n]) for n in wf if n.endswith("proj.weight"))
    assert len(read_jsonl(os.path.join(db, "k_start.jsonl"))) == 2          # one record per start


def test_k5_probe_records_read_as_fact_accuracy(runs):
    """REVIEW 14c D2: K5's FB-MISS and FB-HIT chats (K5 notes EXTRA 2; chats without "qs") are scored as fact accuracy
    per set, never as skill LIK, through the hook with the K5 files entry (sets, take) on a 6-step run, against
    probes.score on the final checkpoint (mutant r14c_probe_k5_as_skill files them as skill)."""
    k5 = str(runs["root"] / "ev" / "evals" / "k5.jsonl")
    probe = {"every": 3, "files": [{"file": "../ev/evals/k5.jsonl", "sha256": sha(k5), "sets": ["FB-MISS", "FB-HIT"],
                                    "take": 4}]}
    p, d = run(runs["root"], "probe_k5", {"probe": probe}, steps=6)
    assert ktrain.main([p, "--device", "cpu"]) == 0
    log = read_jsonl(os.path.join(d, "k_probe.jsonl"))
    assert [r["step"] for r in log] == [3, 6] and not any("error" in r for r in log)
    recs, _ = kprobe.load_records(probe["files"], os.path.dirname(p))
    assert [r["set"] for r in recs] == ["FB-MISS"] * 4 + ["FB-HIT"] * 4 and not any("qs" in r for r in recs)
    assert all(r["turns"][0]["role"] == "tool" for r in recs)
    ck = final(d)
    m = build_model(PlanckConfig.from_dict(ck["model_cfg"]))
    m.load_state_dict(ck["model"])
    import pools as P
    ref = PR.score(PR.model_fn(m, "cpu"), recs, P.get()["V"])
    assert log[-1]["skill_lik"] == {} and sorted(log[-1]["fact_acc"]) == ["FB-HIT", "FB-MISS"]
    for s in ("FB-MISS", "FB-HIT"):
        g = [o["gen"] for r, o in zip(recs, ref) if r["set"] == s]
        assert len(g) == 4 and log[-1]["fact_acc"][s] == pytest.approx(sum(g) / 4, abs=1e-12), s


def test_an_init_file_of_another_model_config_is_refused(runs, tmp_path):
    """REVIEW 14c D4: an init file whose model_cfg differs only where parameter names and shapes stay (rope_theta)
    loads into the model without complaint, so ktrain must refuse it by model_cfg (mutant
    r14c_init_shape_unchecked)."""
    w, w2 = str(tmp_path / "w.pt"), str(tmp_path / "w2.pt")
    prune.prune(runio.latest_checkpoint(runs["out"]["plain"]), w)
    ck = torch.load(w, weights_only=False)
    ck["model_cfg"] = {**ck["model_cfg"], "rope_theta": 500000.0}
    torch.save(ck, w2)
    m = build_model(PlanckConfig(**MODEL))
    m.load_state_dict(ck["model"])                       # names and shapes fit: load_state_dict alone takes it
    p, d = run(runs["root"], "init_cfg", {"init_weights": {"file": w2, "sha256": sha(w2)}})
    with pytest.raises(ValueError, match="another model shape"):
        ktrain.main([p, "--device", "cpu", "--max-steps", "0"])
    assert runio.latest_checkpoint(d) is None


def test_compile_and_duplicate_source_names_are_refused(runs):
    """REVIEW 14c D3: ktrain refuses train.compile whatever the k: block holds (none, or only null keys: K0-cap and
    K0-g run through ktrain with no k key), before anything runs (mutant s14c_compile_needs_k). D6: grad_route
    refuses two data sources of one name, which names.index would route by the first (mutant r14c_dup_names_ok)."""
    for over in ({"train": {"compile": "default"}}, {"train": {"compile": True}},
                 {"k": {"probe": None}, "train": {"compile": "default"}}, {"k": {}, "train": {"compile": True}}):
        with pytest.raises(ValueError, match="eager"):
            ktrain.check({**over, "data": {"sources": SRC("d")}})
    for off in (False, None, "off", "eager"):
        assert ktrain.check({"train": {"compile": off}, "data": {"sources": SRC("d")}}) == {}
    p = write_run(str(runs["root"] / "compiled"), str(runs["root"] / "data"), model=MODEL,
                  data={"sources": SRC(str(runs["root"] / "data"))}, train={"total_steps": 2, "compile": "default"})
    assert ktrain.main([p, "--device", "cpu"]) == 2 and not os.path.exists(os.path.join(os.path.dirname(p), "out"))
    dup = [{**s, "name": "chat"} for s in SRC("d")]
    with pytest.raises(ValueError, match="unique"):
        ktrain.check({"k": {"grad_route": {"sources": ["chat"], "reserved": 4}}, "data": {"sources": dup}})
    assert ktrain.check({"k": {"grad_route": {"sources": ["chat"], "reserved": 4}}, "data": {"sources": SRC("d")}})


def test_k_block_values_are_refused_before_any_run(tmp_path):
    """K3 round 16 (2026-11-21, REVIEW 17b): the k: block's values, not only its keys, are checked before anything
    runs. Each case survived every test until now: a negative, NaN or boolean wd_mult (r17_wd_value_unchecked,
    r17_wd_bool_ok: true ran as x1.0), init_weights without its sha256 or with an extra key
    (r17_init_keys_unchecked), a probe block with an unknown key or every 0 (r17_probe_unknown_key_ok,
    r17_probe_every_zero_ok: a ZeroDivisionError at the first step), a probe file entry with a misspelt key, "set"
    for "sets", which read every record of the file (r17_probe_file_keys_unchecked), and a negative unit index,
    a unit counted from the end (r17_prune_negative_unit_ok)."""
    import types
    src = {"data": {"sources": SRC("d")}}
    for v in (-1.0, float("nan"), True):
        with pytest.raises(ValueError, match="must be a number >= 0"):
            ktrain.check({**src, "k": {"wd_mult": {"mlp": v}}})
    assert ktrain.check({**src, "k": {"wd_mult": {"mlp": 0}}})["wd_mult"] == {"mlp": 0}
    for iw in ({"file": "w.pt"}, {"file": "w.pt", "sha256": "0" * 64, "seed": 1}):
        with pytest.raises(ValueError, match="exactly file and sha256"):
            ktrain.check({**src, "k": {"init_weights": iw}})
    with pytest.raises(ValueError, match="unknown keys"):
        kprobe.KProbe({"every": 5, "files": [], "evry": 5}, str(tmp_path), str(tmp_path), "cpu")
    with pytest.raises(ValueError, match="positive step count"):
        kprobe.KProbe({"every": 0, "files": []}, str(tmp_path), str(tmp_path), "cpu")
    p = tmp_path / "probe.jsonl"
    p.write_text("".join(json.dumps({"id": f"r{i}", "set": s}) + "\n" for i, s in enumerate(("K-PROBE", "FB-HIT"))))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    recs, _ = kprobe.load_records([{"file": str(p), "sha256": sha, "sets": ["K-PROBE"]}], str(tmp_path))
    assert [r["id"] for r in recs] == ["r0"]
    with pytest.raises(ValueError, match="needs file and sha256"):
        kprobe.load_records([{"file": str(p), "sha256": sha, "set": ["K-PROBE"]}], str(tmp_path))
    lin = types.SimpleNamespace(out_features=24)
    m = types.SimpleNamespace(blocks=[types.SimpleNamespace(mlp=types.SimpleNamespace(gate_proj=lin))] * 2)
    u = tmp_path / "u.json"
    u.write_text(json.dumps({"0": [0, 23]}))
    assert prune.units_of(m, str(u), None) == {0: [0, 23]}
    for bad in ([-1, 3], [24], [2, 2]):
        u.write_text(json.dumps({"0": bad}))
        with pytest.raises(ValueError, match="do not fit the model"):
            prune.units_of(m, str(u), None)
