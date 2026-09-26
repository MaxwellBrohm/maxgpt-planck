"""bench_micro on CPU with models under 1M parameters (the CUDA run happens on the PC)."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("PLANCK_TEST_THREADS", "4")))

import bench_micro  # noqa: E402
from count_params import build_and_count  # noqa: E402  (harness, on sys.path via bench_micro)

TINY = ["--device", "cpu", "--vocab", "512", "--seq-len", "64", "--steps", "2", "--warmup", "1"]


def rows(path):
    return [json.loads(x) for x in open(path)]


def test_cpu_smoke_records_every_measurement(tmp_path):
    out = tmp_path / "b.jsonl"
    assert bench_micro.main(TINY + ["--targets", "3e5,6e5", "--batches", "2,3",
                                    "--out", str(out)]) == 0
    rs = rows(out)
    assert len(rs) == 2 * 2 * 2                       # targets x modes x batches
    for r in rs:
        assert r["status"] == "ok" and r["tok_per_s"] > 0 and r["n_params"] < 1_000_000
        assert r["env"]["torch"] == torch.__version__ and r["precision"] == "fp32"
        sol = bench_micro.budget.solve(r["target"], bench_micro.budget.Constraints(vocab_size=512))
        assert (r["shape"]["d"], r["shape"]["layers"]) == (sol.cfg.d_model, sol.cfg.n_layers)
        assert r["n_params"] == build_and_count(sol.cfg, "meta")["total"]   # exact, built
        assert r["n_params_built"] == r["n_params"]       # the benchmarked model IS that shape
        assert abs(r["n_params"] / r["target"] - 1) <= 0.02
    assert {r["mode"] for r in rs} == {"causal", "docmask"}


def test_docmask_mode_really_passes_document_ids(tmp_path, monkeypatch):
    seen = []
    real = bench_micro.build_model

    def spy(cfg, device):
        m = real(cfg, device)
        orig = m.forward

        def fwd(idx, tgt=None, doc=None, pos=None, reduction="mean"):
            seen.append(None if doc is None else int(doc.max()) + 1)
            return orig(idx, tgt, doc, pos, reduction=reduction)
        m.forward = fwd
        return m
    monkeypatch.setattr(bench_micro, "build_model", spy)
    bench_micro.main(TINY + ["--targets", "3e5", "--batches", "2", "--modes", "causal",
                             "--out", str(tmp_path / "a.jsonl")])
    assert seen and all(s is None for s in seen)
    seen.clear()
    bench_micro.main(TINY + ["--targets", "3e5", "--batches", "2", "--modes", "docmask",
                             "--docs-per-row", "4", "--out", str(tmp_path / "b.jsonl")])
    assert seen and all(s is not None and s >= 2 for s in seen)


def test_random_docs_shape_and_counts():
    g = torch.Generator().manual_seed(0)
    d = bench_micro.random_docs(64, 2048, 8.0, g)
    assert d.shape == (64, 2048) and (d[:, 0] == 0).all()
    assert (d[:, 1:] - d[:, :-1]).ge(0).all() and (d[:, 1:] - d[:, :-1]).le(1).all()
    per_row = (d[:, -1] + 1).float().mean().item()
    assert 6 < per_row < 11


def test_oom_is_recorded_and_larger_batches_skipped(tmp_path, monkeypatch):
    class Boom:
        def step(self):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        def zero_grad(self, set_to_none=True):
            pass
    monkeypatch.setattr(bench_micro, "make_optimizer", lambda *a, **k: Boom())
    out = tmp_path / "o.jsonl"
    bench_micro.main(TINY + ["--targets", "3e5", "--batches", "2,4,8", "--out", str(out)])
    rs = rows(out)
    assert [(r["mode"], r["micro_batch"], r["status"]) for r in rs] == \
        [("causal", 2, "oom"), ("docmask", 2, "oom")]


def test_refuses_cuda_when_absent(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA")
    with pytest.raises(SystemExit):
        bench_micro.main(["--targets", "3e5", "--out", str(tmp_path / "x.jsonl")])


def test_loss_first_is_warmup_step_zero_and_step_times_recorded(tmp_path):
    """loss_first is the loss of warmup step 0 whatever --warmup is (CPU is deterministic);
    it used to hold the LAST warmup step's loss."""
    out = tmp_path / "w.jsonl"
    for w in ("1", "3"):
        bench_micro.main(TINY[:-1] + [w] + ["--targets", "3e5", "--batches", "2",
                                            "--modes", "causal", "--out", str(out)])
    r1, r3 = rows(out)
    assert r1["loss_first"] == r3["loss_first"] and r3["loss_first"] != r3["loss_last"]
    for r in (r1, r3):
        assert r["first_step_s"] >= 0 and r["first_step_s"] <= r["warmup_s"] + 0.01
        assert 0 < r["step_ms_median"] <= r["step_ms_max"] and r["step_max_over_median"] >= 1


G = 2**30


@pytest.mark.parametrize("free_before,res_before,alloc,reserved,free_after,fit", [
    (10.76, 0.0, 14.45, 15.41, 0.0, "over"),   # 20M causal B16 eager, 1st cell: 4x slower/token
    (10.18, 0.53, 11.07, 12.17, 0.0, "over"),  # 10M docmask B16 eager, a later cell: 30% slower
    (10.0, 0.0, 9.50, 10.40, 0.0, "over"),     # allocated alone looks fine; the reservation is not
    (10.18, 0.53, 9.92, 10.50, 0.09, "edge"),  # 30M docmask B8, a later cell: the 0.53 GiB left
                                               # reserved by the previous cell is reusable
    (10.76, 0.0, 7.41, 7.85, 2.86, "ok"),      # 20M causal B8 eager
])
def test_mem_report_uses_reservation_and_free_memory(monkeypatch, free_before, res_before, alloc,
                                                     reserved, free_after, fit):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (int(free_after * G), int(11.94 * G)))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: int(reserved * G))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: int(alloc * G))
    r = bench_micro.mem_report((int(free_before * G), int(res_before * G)), 0.5)
    assert r["mem_fit"] == fit
    assert (r["peak_mem_gib"], r["peak_reserved_gib"], r["free_gib"], r["reserved_before_gib"]) == \
        (round(alloc, 2), round(reserved, 2), round(free_after, 2), round(res_before, 2))


def test_optimizer_switches_reach_the_step(tmp_path, monkeypatch):
    """--batched-optim builds the batched optimizer, --lazy-loss and --phase-times change
    only when the loss is read and what is timed: the same losses as the reference step."""
    built = []
    real = bench_micro.make_optimizer

    def spy(*a, **k):
        built.append(real(*a, **k))
        return built[-1]
    monkeypatch.setattr(bench_micro, "make_optimizer", spy)
    out = tmp_path / "s.jsonl"
    base = TINY[:-3] + ["4", "--warmup", "1", "--targets", "3e5", "--batches", "2",
                        "--modes", "causal", "--out", str(out)]
    bench_micro.main(base)
    bench_micro.main(base + ["--batched-optim", "--lazy-loss", "--phase-times"])
    ref, new = rows(out)
    assert [o.batched for o in built] == [False, True]
    assert (ref["batched_optim"], ref["lazy_loss"], new["batched_optim"], new["lazy_loss"]) == \
        (False, False, True, True)
    assert "opt_cpu_ms_median" not in ref and new["opt_cpu_ms_median"] > 0
    assert new["opt_ms_median"] is None                 # CUDA events only on CUDA
    assert new["loss_first"] == ref["loss_first"] and new["loss_first"] != new["loss_last"]
    assert new["loss_last"] == pytest.approx(ref["loss_last"], rel=1e-5)


def test_doc_attn_flag_recorded_and_varlen_refused_off_cuda(tmp_path, monkeypatch):
    out = tmp_path / "d.jsonl"
    bench_micro.main(TINY + ["--targets", "3e5", "--batches", "2", "--modes", "docmask",
                             "--out", str(out)])
    assert [r["doc_attn"] for r in rows(out)] == ["mask"]           # default: the reference
    seen = []
    real = bench_micro.set_doc_attn
    monkeypatch.setattr(bench_micro, "set_doc_attn",
                        lambda m, impl, dev: seen.append(impl) or real(m, impl, dev))
    with pytest.raises(ValueError, match="cuda"):
        bench_micro.main(TINY + ["--targets", "3e5", "--batches", "2", "--doc-attn", "varlen",
                                 "--out", str(tmp_path / "v.jsonl")])
    assert seen == ["varlen"]


def test_batched_optim_flag_maps_to_the_harness_key(tmp_path, monkeypatch):
    """--batched-optim: absent = the harness default (auto: on for cuda), bare or on = True,
    off = False (so off really forces the reference step on cuda too)."""
    seen = []
    real = bench_micro.resolve_batched
    monkeypatch.setattr(bench_micro, "resolve_batched", lambda v, d: seen.append(v) or real(v, d))
    base = TINY + ["--targets", "3e5", "--batches", "2", "--modes", "causal",
                   "--out", str(tmp_path / "f.jsonl")]
    for extra in ([], ["--batched-optim"], ["--batched-optim", "on"], ["--batched-optim", "off"]):
        bench_micro.main(base + extra)
    assert seen == ["auto", True, True, False]
    assert [r["batched_optim"] for r in rows(tmp_path / "f.jsonl")] == [False, True, True, False]


def test_tokens_per_step_is_rows_times_seq_len_times_accum(tmp_path):
    """tok_per_s counts B x T x accum tokens per optimizer step (a B4 x accum 2 vs B8 comparison
    depends on it); docs_per_row is recorded on docmask rows only."""
    out = tmp_path / "t.jsonl"
    for acc in ("1", "2"):
        bench_micro.main(TINY[:-3] + ["3", "--warmup", "1", "--targets", "3e5", "--batches", "3",
                                      "--accum", acc, "--docs-per-row", "2.5", "--out", str(out)])
    rs = rows(out)
    assert [(r["mode"], r["accum"]) for r in rs] == \
        [("causal", 1), ("docmask", 1), ("causal", 2), ("docmask", 2)]
    for r in rs:
        per_step = 3 * 64 * r["accum"]
        # tok_per_s is rounded to 0.1 and step_s to 1e-4: bound the product's rounding error
        err = r["tok_per_s"] * 5e-5 + r["step_s"] * 0.05 + 1e-4
        assert r["step_s"] > 0 and abs(r["tok_per_s"] * r["step_s"] - per_step) <= err
        assert r["docs_per_row"] == (2.5 if r["mode"] == "docmask" else None)
