"""diag_lib.py readings (SCREENS C6, S004 / S005 / S007 DIAGNOSTICS, E3 Part 4d) on tiny CPU models.

The runner's JSON equals a direct call of the registered scorer (bpb.score with --hd, attn_diag.read, induction
.ind_score) on the same checkpoint. S005's half-life has a closed form when the gate weight is 0 (f = sigmoid(b) for
every token), and with a learned weight it equals ln 2 over the mean of -log f taken by this test's own hook (not the
mean of per-token half-lives), with the percentiles of the per-token half-lives over real tokens only. The per-module
readings equal the checkpoint's own parameters.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_diag_readings.py
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

import diag_fixtures as FX
import diag_lib as DL

import attn_diag as D  # noqa: E402  (diag_fixtures put harness/ and data_prep/ on sys.path)
import bpb  # noqa: E402
import induction  # noqa: E402
from blocks import Attention  # noqa: E402


@pytest.fixture(scope="module")
def done(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("diag")
    root, ev = FX.root_with(tmp, FX.SMALL), FX.evalset(str(tmp / "ev"))
    assert FX.cli(root, ev) == 3 and FX.cli(root, ev) == 0          # the replay takes a second process
    return root, ev, FX.outputs(root)


def test_every_registered_job_is_written_with_its_provenance(done):
    root, ev, out = done
    want = {("SCREENS", "s004_canonac_g1_s101"): {"modules", "ind", "hd", "attn"},
            ("SCREENS", "s005_forget_g1_s101"): {"modules", "ind", "hd", "attn", "halflife"},
            ("SCREENS", "s007_smear_g1_s101"): {"modules", "ind", "hd", "attn"},
            ("SCREENS", "base_s101"): {"ind", "hd", "attn"}, ("SCREENS", "s001_nogate_g1_s101"): {"ind", "hd", "attn"},
            ("E3", "e3_5m_A_s1"): {"ind", "hd", "ind.replay", "hd.replay"}}
    got = {}
    for (exp, run, f), j in out.items():
        got.setdefault((exp, run), set()).add(f[:-5])
        ck = os.path.join(root, exp, run, "final_00007630.pt")
        assert j["run"] == run and j["exp"] == exp and j["checkpoint"] == "final_00007630.pt" and j["step"] == 7630
        assert j["checkpoint_sha256"] == DL_sha(ck) and j["evalset_sha256"] == DL.EVALSET_SHA
        assert j["tokenizer_sha256"] == DL.TOK_SHA and set(j["code_sha256"]) >= {"data_prep/bpb_hd.py", "harness/attn_diag.py"}
        assert j["device"] == "cpu"
    assert got == want


def DL_sha(p):
    import diag_runner as DR
    return DR.sha256(p)


def test_hd_ind_and_attn_equal_the_registered_scorers(done):
    root, ev, out = done
    seqs = induction.load_seqs(DL.IND_FILE, DL.IND_SHA)
    for (exp, run, f), j in out.items():
        if f not in ("hd.json", "ind.json", "attn.json", "hd.replay.json"):
            continue
        m = FX.load(os.path.join(root, exp, run, "final_00007630.pt"))
        if f.startswith("hd"):
            want = bpb.score(m, m.cfg, ev, DL.TOK_FILE, ["oasst2"], "cpu", "fp32", 16384, None, hd=True)
            hd = j["result"]["sets"]["oasst2"]["hd"]
            assert hd["kept"] == 18 and hd["first_turn"] == 6 and set(hd["by_turn"]) == {"1", "2", "3"}
            assert set(hd["by_role"]) == {"user", "assistant"}
        elif f == "ind.json":
            want = {**induction.ind_score(m, seqs, "cpu"), "ind_file_sha256": DL.IND_SHA}
        else:
            rows, meta = D.eval_rows(ev, "oasst2", DL.TOK_FILE, 200, m.cfg.seq_len)
            want = {**D.read(m, rows, "cpu", meta["pad_id"], 8192), **meta, "set": "oasst2", "sink_from": D.SINK_FROM}
            assert j["result"]["check_ok"] and len(j["result"]["sink"]) == m.cfg.depth
            assert (j["result"]["gate_mean"] is None) == (run.startswith("s001_nogate"))
            assert (("gate_p10" in j["result"]) and ("gate_p90" in j["result"])) == m.cfg.attn_gate
        assert same(j["result"], want), (run, f)


def same(a, b) -> bool:
    """Equal JSON trees, floats to 1e-9 relative (the JSON round trip keeps them exact; this only allows for it)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def captured_nlf(m, rows):
    """-log f per effective layer, from this test's own pre-hooks (real tokens only), one window per forward."""
    vals = {}
    def pre(mod, args, kwargs):
        x = args[0] if args else kwargs["x"]
        i = len(vals.setdefault("order", []))
        vals["order"].append(i)
        vals.setdefault(i % m.cfg.depth, []).append((-mod.forget_logf(x))[0].T.double())
    hs = [mod.register_forward_pre_hook(pre, with_kwargs=True) for mod in m.modules() if isinstance(mod, Attention)]
    with torch.no_grad():
        for r in rows:
            m(torch.tensor([r]))
    for h in hs:
        h.remove()
    return [torch.cat(vals[i]) for i in range(m.cfg.depth)]


def test_halflife_is_ln2_over_mean_neg_log_f_with_token_percentiles(done):
    root, ev, out = done
    j = out[("SCREENS", "s005_forget_g1_s101", "halflife.json")]["result"]
    m = FX.load(os.path.join(root, "SCREENS", "s005_forget_g1_s101", "final_00007630.pt"))
    rows, _ = D.eval_rows(ev, "oasst2", DL.TOK_FILE, 200, m.cfg.seq_len)
    nlf = captured_nlf(m, rows)
    assert j["layers"] == m.cfg.depth == len(j["by_layer"])
    for i, layer in enumerate(j["by_layer"]):
        v = nlf[i].numpy()
        assert layer["tokens"] == sum(len(r) for r in rows) == v.shape[0]
        np.testing.assert_allclose(layer["mean_neg_log_f"], v.mean(0), rtol=1e-4)
        np.testing.assert_allclose(layer["half_life"], math.log(2) / v.mean(0), rtol=1e-4)
        hl = math.log(2) / v
        np.testing.assert_allclose(layer["half_life_p10"], np.quantile(hl, 0.1, axis=0), rtol=1e-4)
        np.testing.assert_allclose(layer["half_life_p90"], np.quantile(hl, 0.9, axis=0), rtol=1e-4)
        np.testing.assert_allclose(layer["neg_log_f_p10"], np.quantile(v, 0.1, axis=0), rtol=1e-4)
        np.testing.assert_allclose(layer["neg_log_f_p90"], np.quantile(v, 0.9, axis=0), rtol=1e-4)
        assert not np.allclose(layer["half_life"], np.mean(hl, axis=0), rtol=1e-3)     # the two readings differ here


def test_halflife_closed_form_with_a_constant_gate(tmp_path):
    m = FX.model("s005_forget", seed=7)
    b = torch.tensor([7.99, 3.0, -1.0])
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, Attention):
                mod.forget_w.zero_()
                mod.forget_b.copy_(b)
    rows = [[11, 12, 13, 14, 15], [21, 22, 23]]
    r = DL.halflife(m, rows, "cpu", pad_id=0, batch_tokens=64)          # one padded batch: pads must not count
    want = [math.log(2) / math.log1p(math.exp(-float(x))) for x in b]
    for layer in r["by_layer"]:
        assert layer["tokens"] == 8 and layer["f_exactly_1_tokens"] == 0
        for k in ("half_life", "half_life_p10", "half_life_p90"):
            np.testing.assert_allclose(layer[k], want, rtol=1e-5)
    assert abs(want[0] - 2046.03) < 0.01                                 # S005's init: about 2,046 tokens


def test_halflife_none_without_a_forget_gate():
    r = DL.halflife(FX.model("base", seed=1), [[5, 6, 7, 8]], "cpu")
    assert r["layers"] == 2 and r["by_layer"] == [None, None]


def test_modules_equal_the_checkpoint_parameters(done):
    root, ev, out = done
    for run in ("s004_canonac_g1_s101", "s005_forget_g1_s101", "s007_smear_g1_s101"):
        sd = torch.load(os.path.join(root, "SCREENS", run, "final_00007630.pt"), weights_only=False)["model"]
        res = out[("SCREENS", run, "modules.json")]["result"]
        if run.startswith("s004"):
            assert [(x["layer"], x["site"]) for x in res["canon"]] == [(0, "A"), (0, "C"), (1, "A"), (1, "C")]
            for x in res["canon"]:
                w = sd[f"blocks.{x['layer']}.canon_{x['site'].lower()}.weight"].abs()
                assert x["mean_abs"] == pytest.approx(float(w.mean()), rel=1e-6)
                assert x["max_abs"] == pytest.approx(float(w.max()), rel=1e-6)
                np.testing.assert_allclose(x["mean_abs_by_tap"], w.mean(0).numpy(), rtol=1e-6)
            assert set(res) == {"canon"}
        if run.startswith("s007"):
            assert res == {"smear_alpha": [{"layer": i, "alpha": sd[f"blocks.{i}.attn.smear_alpha"].tolist()}
                                           for i in (0, 1)]}
        if run.startswith("s005"):
            assert set(res) == {"forget"} and [x["layer"] for x in res["forget"]] == [0, 1]
            for x in res["forget"]:
                assert x["b"] == pytest.approx(sd[f"blocks.{x['layer']}.attn.forget_b"].tolist(), rel=1e-6)
                assert x["w_norm"] == pytest.approx(sd[f"blocks.{x['layer']}.attn.forget_w"].norm(dim=1).tolist(), rel=1e-6)
    assert DL.modules(FX.model("base").state_dict()) == {}
