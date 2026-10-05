"""RC-12 tests of verify_dev_runs.py and of the run config dev_batch.py records (notes STEP 9 VERIFY), no model.
Builds fake dev runs (fake:SAMPLER, plain, greedy / 1 / 2, with their --own-cf twins) in a temp dir with dev_batch.py,
then corrupts one copy per case and requires the verifier to report the named failure (exit 1); the untouched copy
fails only C3 (SAMPLER's sampled replies ignore the seed value, which is exactly what C3 exists to catch), and a copy
whose seed-2 turn-1 replies differ passes. meta: every run records decode = hf_responder.DECODE, an audit, and the
dtype engines.json names for the model (it overrides --dtype).
Rerun standard (rerun_tie.py; Max, 2026-10-02: took all recommendations in rc12/DECISIONS_FOR_MAX.md (item 2)): a
stored reply changed at turn 2 is a first divergence; with a stub margin of 0.3 or 0.5 R2 PASSES (near-tie <= 0.5),
with 0.8 or none it FAILS. R1 is reported, never a gate (Max, 2026-10-04: took all recommendations in
rc12/DECISIONS_LEVEL_R_FOR_MAX.md (decision 3, a)): at 0.3, 0.5 or 0.8 it prints an INFO line, never PASS or FAIL,
with the identical share and, for the changed conversation, its id, turn, shared prefix and margin (c_r1_report);
a stub is refused for a vLLM run before anything loads, and by the margins stage itself;
hf_margins on a stub tokenizer: the first differing token, abs of the margin, a stopped prefix against the stop id,
none for a capped prefix or equal ids; history() drops the fitted-out pairs.
Run: python3 -B test_verify_dev_runs.py   (exit 1 on any failure)"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

import hf_responder as HR
import rerun_tie as RT

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
    ("stored sampled reply", P + "1", lambda d: edit_rows(d, m_reply),
     "INFO R1 seed 1 lockstep replay vs stored (reported, not gated): 2 / 3", True),
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


R1_INFO = "INFO R1 seed 1 lockstep replay vs stored (reported, not gated): 2 / 3 conversations identical"
TIE_CASES = [  # (name, run dir, corruption, --tie-margin-stub, text that must appear, text that must not)
    ("near-tie sampled", P + "1", m_reply, 0.3, R1_INFO, "FAIL R1"),
    ("near-tie boundary", P + "1", m_reply, 0.5, R1_INFO, "FAIL R1"),
    ("past the tie, R1 reported", P + "1", m_reply, 0.8, R1_INFO, "FAIL R1"),
    ("R1 never PASS", P + "1", m_reply, 0.3, R1_INFO, "PASS R1"),
    ("near-tie greedy", P + "greedy", m_reply, 0.3, "PASS R2 greedy lockstep replay vs stored greedy run: 2 / 3",
     "FAIL R2"),
    ("near-tie greedy boundary", P + "greedy", m_reply, 0.5,
     "PASS R2 greedy lockstep replay vs stored greedy run: 2 / 3", "FAIL R2"),
    ("past the tie, R2 gates", P + "greedy", m_reply, 0.8,
     "FAIL R2 greedy lockstep replay vs stored greedy run: 2 / 3", "PASS R2 greedy lockstep replay vs stored"),
    ("stub on vLLM", P + "1", None, 0.3, "FAIL R: --tie-margin-stub is for fake engines only", "R replay stage"),
]


def verify(root, rerun=False, extra=()):
    cmd = [sys.executable, "-B", "verify_dev_runs.py", "--root", root, "--model", "SAMPLER", "--render", "plain",
           "--no-prompt", "--engines", os.path.join(root, "engines.json")] + (["--rerun", "3"] if rerun else [])
    cmd += list(extra)
    return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=300)


def c_r1_report(src, base):
    """decision 3 (a): the R1 INFO line names the changed conversation's id, turn, shared prefix (characters of the
    two generations, raw when stored) and its margin; R1 never adds a FAIL line."""
    root = os.path.join(base, "r1")
    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(src, root)
    row = jl(os.path.join(root, P + "1", "transcripts.jsonl"))[0]
    a = RT.gen(row["turns"][1])
    t = dict(row["turns"][1])
    t["reply"] += " changed"
    want = f"{row['id']} t2 shared {len(os.path.commonprefix([a, RT.gen(t)]))} chars margin 0.800"
    edit_rows(os.path.join(root, P + "1"), m_reply)
    p = verify(root, True, ["--tie-margin-stub", "0.8"])
    fails = [x for x in p.stdout.splitlines() if x.startswith("FAIL")]
    ok = any(x.startswith("INFO R1 ") and x.endswith(want) for x in p.stdout.splitlines()) and len(fails) == 1 \
        and "C3 turn-1 replies identical" in fails[0]
    return [] if ok else [f"R1 report: want an INFO R1 line ending {want!r} and only the C3 FAIL\n{p.stdout[-700:]}"]


def c_stage_refusal(base):
    """the margins stage itself refuses a stub for a non-fake replay (defense behind check()'s refusal)."""
    d = os.path.join(base, "stage")
    os.makedirs(d, exist_ok=True)
    json.dump(dict(engine="vllm", hf={}, comparisons=[], info=[]), open(os.path.join(d, "replay.json"), "w"))
    p = subprocess.run([sys.executable, "-B", "rerun_tie.py", "--stage", "margins", "--dir", d, "--model", "x",
                        "--render", "plain", "--tie-margin-stub", "0.3"], cwd=HERE, capture_output=True, text=True)
    ok = p.returncode != 0 and "fake engines only" in p.stderr and not os.path.exists(os.path.join(d, "margins.json"))
    return [] if ok else [f"margins stage took a stub for vLLM: exit {p.returncode} {p.stderr[-200:]}"]


def c_margins():
    """hf_margins and history() on a stub tokenizer and margin function (no model)."""
    voc = {}

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [voc.setdefault(w, len(voc) + 10) for w in text.split()]
    h = types.SimpleNamespace(tok=Tok(), encode=lambda hist: [1] * len(hist), stop_ids=[2])
    calls = []

    def margin(h_, prompt, prefix, a, b):
        calls.append((len(prompt), list(prefix), a, b))
        return (-0.4 if b != 2 and a != 2 else 0.2), 0
    hist = [{"role": "user", "content": "u"}]
    divs = [dict(history=hist, a="x y z", b="x q z", stop_a="eos", stop_b="eos"),
            dict(history=hist, a="x y", b="x y w", stop_a="eos", stop_b="cap"),
            dict(history=hist, a="x y", b="x y w", stop_a="cap", stop_b="eos"),
            dict(history=hist, a="x y", b="x y", stop_a="eos", stop_b="cap")]
    got = RT.hf_margins(h, divs, margin)
    fails = []
    v = voc
    want_calls = [(1, [v["x"]], v["y"], v["q"]), (1, [v["x"], v["y"]], 2, v["w"])]
    if got != [0.4, 0.2, None, None] or calls != want_calls:
        fails.append(f"hf_margins: {got} calls {calls} want [0.4, 0.2, None, None] {want_calls}")
    row = dict(turns=[dict(user=f"u{i}", reply=f"a{i}", dropped=0 if i < 3 else 1) for i in range(1, 5)])
    hs = RT.history(row, 3)
    if [m["content"] for m in hs] != ["u2", "a2", "u3", "a3", "u4"]:
        fails.append(f"history with one dropped pair: {[m['content'] for m in hs]}")
    return fails


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
            or p.stdout.count("PASS R") != 2 or "INFO R1 seed 1 lockstep replay vs stored (reported, not gated): 3 / 3" \
            not in p.stdout:
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
    for name, d, act, stub, want, never in TIE_CASES:
        root = os.path.join(base, "case")
        shutil.rmtree(root, ignore_errors=True)
        shutil.copytree(src, root)
        if act is None:
            for x in os.listdir(os.path.join(root, P)):
                edit_meta(os.path.join(root, P, x), lambda m: m.update(engine="vllm"))
        else:
            edit_rows(os.path.join(root, d), act)
        p = verify(root, True, ["--tie-margin-stub", str(stub)])
        if want not in p.stdout or never in p.stdout:
            fails.append(f"{name}: {want!r} missing or {never!r} printed\n{p.stdout[-700:]}{p.stderr[-300:]}")
    fails += c_r1_report(src, base) + c_stage_refusal(base) + c_margins()
    shutil.rmtree(base, ignore_errors=True)
    for f in fails:
        print("FAIL", f)
    print("ALL VERIFY_DEV_RUNS CHECKS PASS (fake:SAMPLER, no model)" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
