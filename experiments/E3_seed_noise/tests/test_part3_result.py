"""E3 Part 3's recorded numbers (notes.txt PART 3 RESULT): results_part3.json and tables_part3.txt regenerate from the
copied records byte for byte; their sha256s are the entry's; the checks are the registered 27 of 32; every number the entry
quotes is results_part3.json's at the printed digits; the 9 registry lines match the records; the independent re-check
(audit/p3_recheck.py, a Python with scipy) passes. E3P3_ROOT (e3analyze_p3_mutants.py --result) points at a scratch copy
of the repo (experiments/E3_seed_noise and models/registry.jsonl).
  ~/.venvs/planck/bin/python -m pytest -q experiments/E3_seed_noise/tests/test_part3_result.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.environ.get("E3P3_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
E3 = os.path.join(ROOT, "experiments", "E3_seed_noise")
R = json.load(open(os.path.join(E3, "results_part3.json")))
SEEDS = {"e3_5m_A_s1_P3": (1, 1), **{f"e3_5m_A_init{k}": (k, 1) for k in range(2, 6)}, **{f"e3_5m_A_data{k}": (1, k) for k in range(2, 6)}}
pc = lambda x: f"{x:.3%}"   # noqa: E731


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def entry() -> str:
    t = open(os.path.join(E3, "notes.txt")).read()
    return " ".join(t[t.index("\nPART 3 RESULT (2026-10-11"):].split())


def test_regenerates_byte_for_byte(tmp_path):
    d = tmp_path / "E3"
    os.makedirs(d / "out")
    for f in ("e3analyze.py", "e3analyze_p3.py", "e3analyze_p3_tables.py", "results.json", "tables.txt", "results_part2.json",
              "tables_part2.txt", "results_part4d.json"):
        shutil.copy(os.path.join(E3, f), d)
    for sub in ("logs", "configs", "plans"):
        shutil.copytree(os.path.join(E3, sub), d / sub)
    for n in list(SEEDS) + ["e3_5m_A_s1", "e3_5m_A_s1_R2", "e3_5m_A_s1_R3"]:
        shutil.copytree(os.path.join(E3, "out", n), d / "out" / n, ignore=shutil.ignore_patterns("c6"))
    r = subprocess.run([sys.executable, str(d / "e3analyze_p3.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (d / "tables_part3.txt").read_bytes() == open(os.path.join(E3, "tables_part3.txt"), "rb").read()
    new = json.load(open(d / "results_part3.json"))
    new["analyzer"]["python"] = R["analyzer"]["python"]          # the record was made under the venv's 3.12.13
    assert new == R


def test_shas_are_the_entrys_and_the_checks_as_registered():
    e = entry()
    assert f"results_part3.json sha256 {sha(os.path.join(E3, 'results_part3.json'))}" in e
    assert f"tables_part3.txt sha256 {sha(os.path.join(E3, 'tables_part3.txt'))}" in e
    ck = R["checks"]
    assert len(ck) == 32 and sum(c["ok"] for c in ck) == 27 and R["unexpected_failures"] == [] and R["gaps"] == []
    assert sorted(c["check"] for c in ck if not c["ok"]) == sorted(R["expected_failures"])
    assert R["replay_stream_equal_A1"] is True and "27 of 32 pass" in e


def quoted():
    F, U, C, V, X = R["families"], R["rule1"], R["compare_part1_sigma_seed"], R["values"], R["replay_vs_part1_A1"]
    q = []
    for m in ("F(CHAT)", "F(PROSE)"):
        i, d = F[m]["INIT"], F[m]["DATA"]
        q += [f"{m} sigma_init {i['sd']:.5f} bpb, {pc(i['rel'])} [{pc(i['rel_ci95'][0])}, {pc(i['rel_ci95'][1])}]; "
              f"sigma_data {d['sd']:.5f} bpb, {pc(d['rel'])} [{pc(d['rel_ci95'][0])}, {pc(d['rel_ci95'][1])}]."]
        for cls in ("same stream, new init", "same init, new stream"):
            u = U[m][cls]
            q.append(f"{pc(u['ref_rel'])} ({u['ref_bpb']:.5f} bpb)")
            q.append(f"MDE{'_u' if m == 'F(CHAT)' else ''} k1 {pc(u['MDE_rel']['1'])}, k2 {pc(u['MDE_rel']['2'])}")
            q.append("k_needed " + ("0.5%: {}, 1%: {}, 2%: {}" if m == "F(CHAT)" else "{}, {}, {}").format(*u["k_needed"].values()))
        q.append(f"{m} {C[m]['INIT']['ratio']:.2f} [{C[m]['INIT']['ratio_95ci_F'][0]:.2f}, {C[m]['INIT']['ratio_95ci_F'][1]:.2f}]"
                 if m == "F(PROSE)" else f"F(CHAT) sigma_init / sigma_seed {C[m]['INIT']['ratio']:.2f}")
        q.append(f"{C[m]['DATA']['ratio']:.2f} [{C[m]['DATA']['ratio_95ci_F'][0]:.2f}, {C[m]['DATA']['ratio_95ci_F'][1]:.2f}]")
        q.append(f"{C[m]['sqrt_init2_plus_data2_over_seed']:.2f} for {m}")
        x = X[m]
        lead = "minus their mean " if m == "F(CHAT)" else ""
        q.append(f"{m} {x['s1_P3']:.5f} against {', '.join(f'{y:.5f}' for y in x['A1_R2_R3'])} ({lead}{x['minus_rep_mean']:+.5f}, {x['in_sigma_rep']:+.2f}")
    for k in ("1", "2", "3", "5"):
        q.append(f"k{k} {pc(U['F(CHAT)']['same stream, new init']['MDE_rel'][k])}")
        q.append(f"k{k} {pc(U['F(CHAT)']['same init, new stream']['MDE_rel'][k])}")
    q.append(f"[k2 at the 80% upper-bound SD {pc(U['F(CHAT)']['same stream, new init']['MDE_rel_ub80']['2'])}]")
    q.append(f"[k2 {pc(U['F(CHAT)']['same init, new stream']['MDE_rel_ub80']['2'])}]")
    q.append(f"M(CHAT) {pc(F['M(CHAT)']['INIT']['rel'])} and {pc(F['M(CHAT)']['DATA']['rel'])}; M(PROSE) {pc(F['M(PROSE)']['INIT']['rel'])} and {pc(F['M(PROSE)']['DATA']['rel'])}")
    for m, lab in (("CHAT user", "CHAT user"), ("CHAT assistant", "assistant"), ("dolly", "dolly"), ("irc", "irc"), ("stackexchange", "stackexchange")):
        q.append(f"{lab} {pc(F[m]['INIT']['rel'])}, {pc(F[m]['DATA']['rel'])}")
    src = [F[f"PROSE {s}"] for s in ("cccc", "gutenberg", "wikimedia")]
    q.append(f"sources {pc(min(z['INIT']['rel'] for z in src))} to {pc(max(z['INIT']['rel'] for z in src))}, "
             f"{pc(min(z['DATA']['rel'] for z in src))} to {pc(max(z['DATA']['rel'] for z in src))}")
    t = F["train loss last 2%"]
    q.append(f"INIT {pc(t['INIT']['rel'])}, DATA {pc(t['DATA']['rel'])}")
    tl = V["train loss last 2%"]["DATA"]
    q.append(f"data2 ({tl['e3_5m_A_data2']:.4f} against {min(v for n, v in tl.items() if n != 'e3_5m_A_data2'):.4f} to {max(tl.values()):.4f})")
    rat = {m: F[m]["DATA"]["rel"] / F[m]["INIT"]["rel"] for m in F}
    q.append(f"(F(CHAT) by x{rat['F(CHAT)']:.2f}, F(PROSE) x{rat['F(PROSE)']:.2f})")
    q.append(f"variance ratio {rat['F(CHAT)'] ** 2:.2f}")
    allr = [C[m]["sqrt_init2_plus_data2_over_seed"] for m in C]
    q.append(f"({min(allr):.2f} to {max(allr):.2f} over the 13 metrics)")
    q.append(f"F(CHAT) INIT {F['F(CHAT)']['INIT']['mean']:.5f}, DATA {F['F(CHAT)']['DATA']['mean']:.5f}")
    q.append(f"F(PROSE) {F['F(PROSE)']['INIT']['mean']:.5f} and {F['F(PROSE)']['DATA']['mean']:.5f}")
    return q, rat


def test_every_quoted_number_is_the_results():
    e = entry()
    q, rat = quoted()
    missing = [s for s in q if s not in e]
    assert missing == [], missing
    assert all(r > 1 for r in rat.values()) and len(rat) == 13 and "above sigma_init on all 13 metrics" in e
    assert "—" not in e and "–" not in e


def test_the_nine_registry_lines():
    reg = [json.loads(x) for x in open(os.path.join(ROOT, "models", "registry.jsonl"))]
    got = {e["id"]: e for e in reg if e["id"].startswith("E3/5m_A/") and e["code"]["git_commit"].startswith("4962426")}
    assert sorted(got) == sorted(f"E3/5m_A/{n.replace('e3_5m_A_', '')}/final" for n in SEEDS)
    for n, (seed, dseed) in SEEDS.items():
        e = got[f"E3/5m_A/{n.replace('e3_5m_A_', '')}/final"]
        st = json.load(open(os.path.join(E3, "out", n, "run_records.json")))[0]
        fam = "INIT" if "init" in n or n.endswith("P3") else "DATA"
        assert (e["seed"], e["hparams"]["data_seed"], e["config_sha256"]) == (seed, dseed, st["config_sha256"])
        assert e["params_total"] == st["n_params"] and e["locations"] == [f"pc:~/planck/runs/E3/{n}"]
        assert f"CHAT {R['values']['F(CHAT)'][fam][n]:.4f}, PROSE {R['values']['F(PROSE)'][fam][n]:.4f}" in e["result"]
        assert e["code"]["digest_sha256"] == sha(os.path.join(E3, "logs", "part3", "code_sha256_20261011_131228.txt"))


def test_independent_recheck_passes():
    py = shutil.which("python3")
    if not py or subprocess.run([py, "-c", "import scipy"], capture_output=True).returncode:
        pytest.skip("no python3 with scipy")
    r = subprocess.run([py, "-I", os.path.join(E3, "audit", "p3_recheck.py")], capture_output=True, text=True)
    assert r.returncode == 0 and "RECHECK PASS" in r.stdout and "810 comparisons, 0 disagree" in r.stdout, r.stdout[-2000:]
