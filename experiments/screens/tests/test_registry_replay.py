"""models/registry.jsonl's lines for the three runs of plan stage2_replay (SCREENS.txt AMENDMENT BASE-REPLAY; registered
2026-10-11, models/MODELS.md section 12): one saved final each, at the run folder, with train.py's hash of the committed
config, the queue's commit, times inside the run's gpu.lock hold, and a result that says it was not read. No model,
nothing on the PC.
  ~/.venvs/planck/bin/python -m pytest -q experiments/screens/tests/test_registry_replay.py
"""
import datetime as dt, hashlib, json, os, re, sys  # noqa: E401

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]
import screens_lib as L  # noqa: E402
from test_screens_cap31 import REPLAY  # noqa: E402

REG = os.path.join(L.ROOT, "models", "registry.jsonl")
ROLE = {"base_s101_rep1": "base_replay", "base_s101_hookoff": "base_hookoff", "base_s101_rep2": "base_replay"}


def lines():
    out = {}
    for ln in open(REG):
        if ln.strip():
            e = json.loads(ln)
            if e["locations"] and e["locations"][0].startswith("pc:~/planck/runs/SCREENS/base_s101_"):
                out.setdefault(e["locations"][0].rsplit("/", 1)[1], []).append(e)
    return out


def test_each_replay_run_has_one_saved_final():
    got = lines()
    plan = [ln.split()[1] for ln in open(os.path.join(L.HERE, "plans", "stage2_replay.txt")) if ln.startswith("train ")]
    assert sorted(got) == sorted(plan) == sorted(ROLE) and all(len(v) == 1 for v in got.values())
    for run, (e,) in got.items():
        assert e["id"] == f"SCREENS/BASE/{run[5:]}/final" and (e["arm"], e["role"], e["seed"]) == ("BASE", ROLE[run], 101)
        assert e["kind"] == "from_scratch" and e["status"] == "success" and e["release"] == "never"
        assert e["locations"] == [f"pc:~/planck/runs/SCREENS/{run}"] and [f["path"] for f in e["files"]] == \
            ["final_00007630.pt"] and re.fullmatch(r"[0-9a-f]{64}", e["files"][0]["sha256"])
        assert e["result"].startswith("not read: registered 2026-10-11 before the stage 2 verdict step reads it") and \
            e["result"].endswith("; no value of it was read for this entry")
    shas = [e["files"][0]["sha256"] for (e,) in got.values()]
    assert len(set(shas)) == 3


def test_config_hash_code_and_times():
    sys.path.insert(0, os.path.join(L.ROOT, "harness"))
    import runio
    got = lines()
    for run, a, b in REPLAY:
        (e,) = got[run]
        cfg = runio.load_yaml(os.path.join(L.HERE, "configs", f"{run}.yaml"))
        assert e["config_sha256"] == hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
        assert e["code"]["git_commit"].startswith("2aa3787") and e["config_source"].endswith(f"{run}.yaml at 2aa3787)")
        t = lambda s: dt.datetime.fromisoformat(s).replace(tzinfo=None)  # noqa: E731
        hold = (dt.datetime.fromisoformat(a), dt.datetime.fromisoformat(b))
        assert hold[0] <= t(e["run"]["started"]) < t(e["run"]["ended"]) <= hold[1] and e["created"] == e["run"]["ended"]
        assert e["hparams"]["lr"] == cfg["optim"]["lr"] == 0.003 and e["seed"] == cfg["seed"] == 101
    assert got["base_s101_hookoff"][0]["notes"].count("IND hook off") == 1
