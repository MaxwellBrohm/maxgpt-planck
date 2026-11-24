"""k0smoke.py (K3 round 16, 2026-11-21): K0-smoke's checks on hand-built run directories (logs, probe readings, tiny
checkpoints); no model is trained."""
import copy
import json

import pytest
import torch

import gen
import k0smoke as S

SRC = gen.data_block("FH")["sources"]
CFG = {"train": {"total_steps": 306}, "data": {"sources": SRC, "max_item_len": 2048}}
PER_STEP = 32_000


def logs(noise=0.0, shift=None):
    tot_share = sum(s["share"] for s in SRC)
    out = []
    for step in list(range(10, 306, 10)) + [306]:
        tot = step * PER_STEP
        r = {"step": step, "loss": round(5.0 - step / 100 + noise, 5), "lr_factor": 1.0, "gnorm": 0.5,
             "tokens": tot, "sup_tokens": tot // 3, "tok_per_s": 400_000.0, "phase": "stable"}
        r.update({f"drawn_{s['name']}": int(s["share"] / tot_share * tot) for s in SRC})
        if shift and step == 306:
            r[f"drawn_{shift[0]}"] += shift[1]
        out.append(r)
    return out


def make(tmp_path, on=None, off=None, rep=None, probe=None, w=(1.0, 1.0, 1.0), mem=None):
    for name, L, wv in ((S.ON, on, w[0]), (S.OFF, off, w[1]), (S.REP, rep, w[2])):
        d = tmp_path / name
        d.mkdir()
        (d / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (L if L is not None else logs())))
        torch.save({"model": {"w": torch.full((3,), wv)}}, d / "final_00000306.pt")
    pr = probe if probe is not None else [{"step": 153, "seconds": 2.0}, {"step": 306, "seconds": 2.0}]
    (tmp_path / S.ON / "k_probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in pr))
    if mem is not None:
        (tmp_path / S.ON / "gpu_mem_max_mib.txt").write_text(f"{mem}\n")
    return str(tmp_path)


def written(fact_share_bio, fact_share_qa, frac=1.0):
    """Written tokens such that the projected 7,630-step draw is `frac` of the written length."""
    out = {}
    for s in SRC:
        if s["name"].startswith("fact_"):
            L = logs()[-1]
            proj = L[f"drawn_{s['name']}"] / 306 * 7630
            out[s["name"]] = int(proj / frac)
    return out


def test_a_clean_smoke_passes_and_reports_tok_s_and_hours(tmp_path):
    r = S.read(make(tmp_path), "", CFG, written(0, 0, 0.97))
    assert r["pass"] and all(r["checks"].values()), r["checks"]
    assert r["hook_off"] == {"same_data": True, "bitwise": True, "repeatable": True, "H": 0.0, "R": 0.0}
    assert r["tok_per_s"] == 400_000.0
    assert r["hours_7630"] == pytest.approx((PER_STEP * 7630 / 400_000 + 2.0 * 50) / 3600)


def test_replay_noise_decides_when_the_pair_is_not_bitwise(tmp_path):
    """H 0.002 against R 0.004: within 2R, passes though not bitwise. H 0.009 against R 0.004 fails; so does any H
    when the replays agree exactly but the probe run does not, in its losses or only in its final weights."""
    t1, t2, t3 = (tmp_path / x for x in "abc")
    for t in (t1, t2, t3):
        t.mkdir()
    r = S.read(make(t1, on=logs(0.002), rep=logs(0.004), w=(1.1, 1.0, 1.2)), "", CFG, written(0, 0, 0.97))
    assert r["checks"]["hook_off"] and not r["hook_off"]["bitwise"]
    assert r["hook_off"]["H"] == pytest.approx(0.002) and r["hook_off"]["R"] == pytest.approx(0.004)
    r = S.read(make(t2, on=logs(0.009), rep=logs(0.004), w=(1.1, 1.0, 1.2)), "", CFG, written(0, 0, 0.97))
    assert not r["checks"]["hook_off"] and not r["pass"]
    r = S.read(make(t3, on=logs(0.00001)), "", CFG, written(0, 0, 0.97))
    assert r["hook_off"]["R"] == 0.0 and not r["checks"]["hook_off"]
    t4 = tmp_path / "d"
    t4.mkdir()
    r = S.read(make(t4, w=(1.5, 1.0, 1.0)), "", CFG, written(0, 0, 0.97))      # equal logs, other weights,
    assert r["hook_off"]["repeatable"] and not r["hook_off"]["bitwise"] and not r["checks"]["hook_off"]  # replays equal


def test_a_data_difference_fails_whatever_the_losses(tmp_path):
    t1, t2 = tmp_path / "a", tmp_path / "b"
    t1.mkdir(), t2.mkdir()
    r = S.read(make(t1, rep=logs(shift=("skill", 1))), "", CFG, written(0, 0, 0.97))
    assert not r["hook_off"]["same_data"] and not r["checks"]["hook_off"]
    on = logs()
    on[3]["lr_factor"] = 0.9
    assert not S.read(make(t2, on=on), "", CFG, written(0, 0, 0.97))["checks"]["hook_off"]



def _stamp(L, i):
    """A log in the harness's own format: runio.append_jsonl puts the wall clock ("time", to the second) first in
    every record; run i of three ran later than run i - 1, and its tok/s differs too."""
    return [{"time": f"2026-11-24T0{i}:00:{j % 60:02d}+0000", **r, "tok_per_s": r["tok_per_s"] + i}
            for j, r in enumerate(L)]


def test_a_clean_smoke_in_the_harness_log_format_passes(tmp_path):
    """2026-11-23 (K review finding 1): the three smoke runs run one after another, so their "time" stamps always
    differ. The frozen check compared every field but loss, gnorm and tok_per_s, so same_data, bitwise and repeatable
    were False on every real smoke and no pass mark could be written. A clean smoke in the harness's format passes,
    bitwise and repeatable (the review's probe test_k0smoke_time.py, which failed on the frozen code)."""
    r = S.read(make(tmp_path, on=_stamp(logs(), 0), off=_stamp(logs(), 1), rep=_stamp(logs(), 2)), "", CFG,
               written(0, 0, 0.97))
    assert r["pass"] and all(r["checks"].values()), r["checks"]
    assert r["hook_off"] == {"same_data": True, "bitwise": True, "repeatable": True, "H": 0.0, "R": 0.0}


def _with_dropped(L):
    return [{**r, "dropped_long_skill": 0} for r in L]


@pytest.mark.parametrize("field,new", [("step", 11), ("tokens", 1), ("sup_tokens", 1), ("lr_factor", 0.999),
                                       ("phase", "decay"), ("drawn_skill", 1), ("dropped_long_skill", 1)])
def test_hook_off_compares_each_k_d1_field(tmp_path, field, new):
    """2026-11-23 (K review finding 1): same_data reads the K-D1 fields by name (step, tokens, sup_tokens, lr_factor,
    phase, every drawn_<source> and dropped_<...>). One record of the replay off in any one of them fails the check,
    though the losses, the weights and every other field agree. The review's mutant rv_smoke_phase_ignored (phase,
    sup_tokens and tokens not compared) survived the suite before this test."""
    rep = _with_dropped(logs())
    rep[5][field] = new if isinstance(new, str) or field == "lr_factor" else rep[5][field] + new
    r = S.read(make(tmp_path, on=_with_dropped(logs()), off=_with_dropped(logs()), rep=rep), "", CFG,
               written(0, 0, 0.97))
    assert not r["hook_off"]["same_data"] and not r["checks"]["hook_off"] and not r["pass"], (field, r["hook_off"])


@pytest.mark.parametrize("field", ["sup_tokens", "lr_factor", "phase", "drawn_"])
def test_a_log_without_a_named_field_fails_same_data(tmp_path, field):
    """2026-11-23 (K review finding 1): every record must carry step, tokens, sup_tokens, lr_factor, phase and a
    drawn_ count; three logs that all lack one agree by absence, which is no evidence of equal data. (Without step
    or tokens the reading itself cannot be taken.)"""
    L = [{k: v for k, v in r.items() if not k.startswith(field)} for r in logs()]
    r = S.read(make(tmp_path, on=copy.deepcopy(L), off=copy.deepcopy(L), rep=L), "", CFG, written(0, 0, 0.97))
    assert not r["hook_off"]["same_data"] and not r["checks"]["hook_off"], (field, r["hook_off"])

def test_probe_readings_must_be_every_153_and_the_last_without_errors(tmp_path):
    t = [tmp_path / x for x in "abc"]
    for x in t:
        x.mkdir()
    assert not S.read(make(t[0], probe=[{"step": 306, "seconds": 1.0}]), "", CFG, written(0, 0, .97))["checks"]["probe"]
    bad = [{"step": 153, "seconds": 1.0}, {"step": 306, "error": "RuntimeError: x", "seconds": 1.0}]
    assert not S.read(make(t[1], probe=bad), "", CFG, written(0, 0, .97))["checks"]["probe"]
    extra = [{"step": 153, "seconds": 1.0}, {"step": 200, "seconds": 1.0}, {"step": 306, "seconds": 1.0}]
    assert not S.read(make(t[2], probe=extra), "", CFG, written(0, 0, .97))["checks"]["probe"]


def test_shares_fact_draw_and_memory(tmp_path):
    t = [tmp_path / x for x in "abcde"]
    for x in t:
        x.mkdir()
    off = logs(shift=("fact_qa", 2400))         # one source about 2,210 tokens over its share: more than one item
    r = S.read(make(t[0], on=copy.deepcopy(off), off=copy.deepcopy(off), rep=off), "", CFG, written(0, 0, .97))
    assert not r["checks"]["shares"] and r["checks"]["hook_off"]
    ok = logs(shift=("fact_qa", 2000))
    assert S.read(make(t[1], on=copy.deepcopy(ok), off=copy.deepcopy(ok), rep=ok), "", CFG,
                  written(0, 0, .97))["checks"]["shares"]
    assert not S.read(make(t[2]), "", CFG, written(0, 0, 1.01))["checks"]["fact_draw"]     # draw past the stream
    assert not S.read(make(t[3]), "", CFG, written(0, 0, 0.94))["checks"]["fact_draw"]     # FB region not reached
    r = S.read(make(t[4], mem=11_500), "", CFG, written(0, 0, 0.97))
    assert not r["checks"]["memory"] and r["gpu_mem_max_mib"] == 11_500 and not r["pass"]
    assert not S.read(str(t[4]), "", CFG, {})["checks"]["fact_draw"]                      # nothing counted: no pass


def test_main_writes_the_grid_mark_only_on_a_pass(tmp_path, monkeypatch):
    """2026-11-22 (REVIEW 17b): plans/k0.txt's K0-g grid waits for RUNS_DIR/marks/K0_smoke_passed, so main() writes
    it after a passing reading and never after a failing one; the reading is appended either way."""
    import count_k
    import kconfig as KC
    import runio
    monkeypatch.setattr(runio, "load_yaml", lambda p: CFG)
    monkeypatch.setattr(count_k, "count_arm", lambda root, arm: {"sources": {"fact_bio": {"total": {"tokens": 10}}}})
    for ok in (False, True):
        runs = tmp_path / f"runs_{ok}"
        runs.mkdir()
        rec = {"reading": "K0-smoke", "pass": ok, "checks": {"x": ok}, "tok_per_s": 1.0, "hours_7630": 1.0}
        monkeypatch.setattr(S, "read", lambda *a, r=rec: dict(r))
        rd = tmp_path / f"readings_{ok}.jsonl"
        assert S.main([str(runs), "data", "--readings", str(rd)]) == (0 if ok else 1)
        assert len(rd.read_text().splitlines()) == 1
        assert (runs / "marks" / "K0_smoke_passed").exists() == ok
    assert KC.SMOKE_MARK == "K0-smoke passed"


def test_a_run_without_final_weights_fails_finished(tmp_path):
    """2026-11-22 (port review finding 3): a reading taken before all three smoke runs finished must not pass (a pass
    writes the mark that releases the K0-g grid). Each run in turn lacks its final_*.pt."""
    for name in (S.ON, S.OFF, S.REP):
        t = tmp_path / name
        t.mkdir()
        make(t)
        (t / name / "final_00000306.pt").unlink()
        r = S.read(str(t), "", CFG, written(0, 0, 0.97))
        assert not r["checks"]["finished"] and not r["pass"], (name, r["checks"])
        assert all(v for k, v in r["checks"].items() if k != "finished"), (name, r["checks"])
