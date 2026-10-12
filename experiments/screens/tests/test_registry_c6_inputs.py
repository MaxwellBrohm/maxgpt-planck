"""models/registry.jsonl against the C6 diagnostics runner (SCREENS AMENDMENT C6-DIAG): every checkpoint the runner
scores is a registered final, at the run folder the runner reads (pc:~/planck/runs/<EXP>/<run>), with the step the
runner wants and a sha256. E3 Part 2's six 20M finals were registered 2026-10-10 (models/MODELS.md section 11); before
that, group e3p2 read six unregistered checkpoints. No model, nothing on the PC.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_registry_c6_inputs.py
"""
import json, os, re, sys  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import diag_runner as DR  # noqa: E402
import screens_lib as L  # noqa: E402

REG = os.path.join(L.ROOT, "models", "registry.jsonl")


def by_location():
    out = {}
    for ln in open(REG):
        if ln.strip():
            e = json.loads(ln)
            for loc in e["locations"]:
                out.setdefault(loc, []).append(e)
    return out


def test_every_checkpoint_the_c6_runner_scores_is_a_registered_final():
    loc = by_location()
    seen = set()
    for _g, (exp, runs, _js, _suf) in DR.GROUPS.items():
        for r in runs:
            if (exp, r) in seen:
                continue
            seen.add((exp, r))
            es = [e for e in loc.get(f"pc:~/planck/runs/{exp}/{r}", []) if e["kind"] == "from_scratch"]
            assert len(es) == 1, (exp, r, len(es))
            e = es[0]
            step = DR.expected(r)[0]
            assert [f["path"] for f in e["files"]] == [f"final_{step:08d}.pt"], r
            assert re.fullmatch(r"[0-9a-f]{64}", e["files"][0]["sha256"] or ""), r
            assert e["id"].endswith("/final") and e["experiment"] == exp and e["status"] in ("success", "diagnostic")
    assert len(seen) == 49


def test_e3_part2_lines_match_their_configs_and_results():
    sys.path.insert(0, os.path.join(L.ROOT, "harness"))
    import runio
    res = json.load(open(os.path.join(L.ROOT, "experiments", "E3_seed_noise", "results_part2.json")))["values"]
    loc = by_location()
    for arm in "AB":
        for s in (1, 2, 3):
            r = f"e3_20m_{arm}_s{s}"
            (e,) = loc[f"pc:~/planck/runs/E3/{r}"]
            c = runio.load_yaml(os.path.join(L.ROOT, "experiments", "E3_seed_noise", "configs", f"{r}.yaml"))
            assert e["id"] == f"E3/20m_{arm}/s{s}/final" and (e["arm"], e["seed"], e["role"]) == (f"20m_{arm}", s, "scored")
            assert e["hparams"]["lr"] == c["optim"]["lr"] and e["hparams"]["schedule"]["total_steps"] == 15259
            assert e["code"]["git_commit"].startswith("3c102f7") and e["params_total"] == 20001511
            assert f"CHAT {res['F(CHAT)'][arm][s - 1]:.4f}, PROSE {res['F(PROSE)'][arm][s - 1]:.4f}" in e["result"]
            assert e["tokenizer"]["sha256"] == c["provenance"]["tokenizer_sha256"]
            assert e["data"]["sha256"] == c["provenance"]["shard_manifest_sha256"]
