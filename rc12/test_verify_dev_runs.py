"""RC-12 tests of verify_dev_runs.py and of the run config dev_batch.py records (notes STEP 9 VERIFY), no model.
Builds fake dev runs (fake:SAMPLER, plain, greedy / 1 / 2, with their --own-cf twins) in a temp dir with dev_batch.py,
then corrupts one copy per case and requires the verifier to report the named failure (exit 1); the untouched copy
fails only C3 (SAMPLER's sampled replies ignore the seed value, which is exactly what C3 exists to catch), and a copy
whose seed-2 turn-1 replies differ passes. meta: every run records decode = hf_responder.DECODE, an audit, and the
dtype engines.json names for the model (it overrides --dtype).
Run: python3 -B test_verify_dev_runs.py   (exit 1 on any failure)"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import hf_responder as HR

HERE = os.path.dirname(os.path.abspath(__file__))
P = "SAMPLER/plain/"


def jl(p):
    return [json.loads(line) for line in open(p)]


def wl(p, rows):
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))


def edit_meta(d, f):
    m = json.load(open(d + "/meta.json"))
    f(m)
    json.dump(m, open(d + "/meta.json", "w"))


def edit_rows(d, f):
    rows = jl(d + "/transcripts.jsonl")
    f(rows)
    wl(d + "/transcripts.jsonl", rows)


def m_reply(rows):
    rows[0]["turns"][1]["reply"] += " changed"


def m_role(rows):
    rows[0]["turns"][1]["reply"] += "\nUser: hi"


def m_seed2(rows):
    for r in rows:
        r["turns"][0]["reply"] += " s2"


def vllm_meta(extra):
    def f(m):
        m["engine"] = "vllm"
        m["audit"] = dict(stop_ids=[2], sampled=dict(top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=256,
                          stop_token_ids=[2], skip_special_tokens=True, temperature=0.6),
                          greedy=dict(temperature=0.0), vllm_default_sampling={}, gen_config_stops_not_in_hf=[])
        m["audit"].update(extra)
    return f


CASES = [  # (name, run dir, corruption, text the verifier must print, --rerun?)
    ("twin missing", P + "1_owncf", lambda d: os.remove(d + "/DONE"), "C2 1: no finished _owncf twin", False),
    ("decode", P + "1", lambda d: edit_meta(d, lambda m: m["decode"].update(temperature=0.7)), "FAIL C1 1: decode",
     False),
    ("role left in reply", P + "greedy", lambda d: edit_rows(d, m_role), "FAIL C4 greedy", False),
    ("unknown stop", P + "1", lambda d: edit_rows(d, lambda r: r[3]["turns"][2].update(stop="length")), "FAIL C4 1:",
     False),
    ("unstripped reply", P + "2_owncf", lambda d: edit_rows(d, lambda r: r[0]["turns"][0].update(reply=" x ")),
     "FAIL C4 2_owncf", False),
    ("twin ids", P + "1_owncf", lambda d: edit_rows(d, lambda r: r.pop()), "FAIL C2 1: twin ids", False),
    ("row seed", P + "2", lambda d: edit_rows(d, lambda r: r[0].update(seed=9)), "FAIL C3 2: 1 rows", False),
    ("row count", P + "2", lambda d: edit_rows(d, lambda r: r.pop()), "FAIL C1 2: 639 rows", False),
    ("stored sampled reply", P + "1", lambda d: edit_rows(d, m_reply), "FAIL R1 seed 1", True),
    ("stored greedy reply", P + "greedy", lambda d: edit_rows(d, m_reply),
     "FAIL R2 greedy lockstep replay vs stored", True),
    ("vllm defaults leak", P + "1", lambda d: edit_meta(d, vllm_meta(dict(vllm_default_sampling={"top_k": 20}))),
     "FAIL C1 1: vLLM sampling defaults", False),
    ("vllm gen-config stop", P + "1", lambda d: edit_meta(d, vllm_meta(dict(gen_config_stops_not_in_hf=[1]))),
     "FAIL C1 1: generation_config.json adds", False),
    ("vllm top_k", P + "1", lambda d: edit_meta(d, vllm_meta(dict(sampled=dict(top_k=20)))),
     "FAIL C1 1: vLLM kwargs", False),
    ("hf gen-config field", P + "1", lambda d: edit_meta(d, lambda m: m.update(
        engine="hf", audit=dict(hf_generation_config={"no_repeat_ngram_size": 3}))),
     "FAIL C1 1: generation_config fields", False),
]


def verify(root, rerun=False):
    cmd = [sys.executable, "-B", "verify_dev_runs.py", "--root", root, "--model", "SAMPLER", "--render", "plain",
           "--no-prompt", "--engines", os.path.join(root, "engines.json")] + (["--rerun", "3"] if rerun else [])
    return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=300)


def main():
    fails = []
    base = tempfile.mkdtemp()
    src = os.path.join(base, "runs")
    os.makedirs(src)
    json.dump({"models": {"SAMPLER": {"engine": "fake", "dtype": "float32"}}}, open(os.path.join(src, "engines.json"), "w"))
    p = subprocess.run([sys.executable, "-B", "dev_batch.py", "--responder", "fake:SAMPLER", "--render", "plain",
                        "--seeds", "greedy,1,2", "--root", src, "--engines", os.path.join(src, "engines.json")],
                       cwd=HERE, capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
        sys.exit("dev_batch failed")
    for d in ("greedy", "greedy_owncf", "1", "1_owncf", "2", "2_owncf"):
        m = json.load(open(os.path.join(src, P, d, "meta.json")))
        if m.get("decode") != HR.DECODE or "audit" not in m or m.get("dtype") != "float32":
            fails.append(f"meta {d}: decode {m.get('decode')} audit {'audit' in m} dtype {m.get('dtype')}")
    p = verify(src, True)
    lines = [x for x in p.stdout.splitlines() if x.startswith("FAIL")]
    if p.returncode != 1 or len(lines) != 1 or "C3 turn-1 replies identical, 1 vs 2" not in lines[0] \
            or p.stdout.count("PASS R") != 3:
        fails.append(f"baseline: exit {p.returncode}, fails {lines}")
    for name, d, act, want, rr in CASES:
        root = os.path.join(base, "case")
        shutil.rmtree(root, ignore_errors=True)
        shutil.copytree(src, root)
        act(os.path.join(root, d))
        p = verify(root, rr)
        if p.returncode != 1 or want not in p.stdout:
            fails.append(f"{name}: exit {p.returncode}, {want!r} not reported\n{p.stdout[-600:]}{p.stderr[-600:]}")
    root = os.path.join(base, "case")
    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(src, root)
    edit_rows(os.path.join(root, P + "2"), m_seed2)
    p = verify(root)
    if p.returncode != 0 or "INFO C3 turn-1 replies identical, 1 vs 2: 0 / 640" not in p.stdout:
        fails.append(f"seed-dependent copy: exit {p.returncode}\n{p.stdout[-600:]}")
    shutil.rmtree(base, ignore_errors=True)
    for f in fails:
        print("FAIL", f)
    print("ALL VERIFY_DEV_RUNS CHECKS PASS (fake:SAMPLER, no model)" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
