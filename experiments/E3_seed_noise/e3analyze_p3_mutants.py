"""Mutation test of e3analyze_p3.py on scratch copies (never the repo).
  python e3analyze_p3_mutants.py            DATA mutants on a copy of this folder's Part 3 inputs (after the copy-back)
  python e3analyze_p3_mutants.py --fixture  DATA mutants on tests/p3fixture.py's synthetic record set (before any value)
  python e3analyze_p3_mutants.py --code     CODE mutants of the statistics: tests/test_e3analyze_p3.py must fail on each
  python e3analyze_p3_mutants.py --result   RESULT mutants (records, results, notes, registry): tests/test_part3_result.py
A DATA mutant is killed when the set of failing checks differs from the unmutated copy's (expected: the five
EXPECTED_FAIL checks). Writes nothing here."""
import json, os, shutil, subprocess, sys, tempfile
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SRC, "tests"))
PY, WD = sys.executable, tempfile.mkdtemp(prefix="e3p3_mut_")
R = {n: f"out/e3_5m_A_{n}" for n in ("s1_P3", "init2", "init3", "init4", "init5", "data2", "data3", "data4", "data5")}
QL, CL = "logs/part3/queue_e3.txt", "logs/part3/code_sha256_20261011_131228.txt"


def fresh(fixture):
    d = os.path.join(WD, "E3"); shutil.rmtree(d, ignore_errors=True)
    if fixture:
        import p3fixture
        global _CL
        try: _CL
        except NameError: _CL = p3fixture.code_list()
        return p3fixture.build(d, codelist=_CL)
    os.makedirs(os.path.join(d, "out"))
    for f in ("e3analyze.py", "e3analyze_p3.py", "e3analyze_p3_tables.py", "results.json", "tables.txt", "results_part2.json",
              "tables_part2.txt", "results_part4d.json"): shutil.copy(os.path.join(SRC, f), d)
    for sub in ("logs", "configs", "plans"): shutil.copytree(os.path.join(SRC, sub), os.path.join(d, sub))
    for n in os.listdir(os.path.join(SRC, "out")):
        if n.startswith("e3_5m_A_"): shutil.copytree(os.path.join(SRC, "out", n), os.path.join(d, "out", n), ignore=shutil.ignore_patterns("c6"))
    return d


def fails(d):
    r = subprocess.run([PY, os.path.join(d, "e3analyze_p3.py"), "--checks-only"], capture_output=True, text=True)
    if r.returncode: return {"CRASH: " + r.stderr.strip().splitlines()[-1]}
    return {c["check"] for c in json.loads(r.stdout) if not c["ok"]}


def jl(p): return [json.loads(x) for x in open(p)]
def wjl(p, rows): open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
def ej(p, fn): x = json.load(open(p)); fn(x); json.dump(x, open(p, "w"), indent=1)
def sub(p, a, b):
    t = open(p).read(); assert t.count(a) >= 1, (p, a); open(p, "w").write(t.replace(a, b, 1))
def tail(d, run, i, k, v):
    p = f"{d}/{R[run]}/train_tail.jsonl"; r = jl(p); r[i][k] = v(r[i][k]); wjl(p, r)
def bpb(d, run, fn):
    p = f"{d}/{R[run]}/bpb.jsonl"; r = jl(p); fn(r); wjl(p, r)
def pf(d, run, fn): ej(f"{d}/{R[run]}/preflight.json", lambda x: fn(x["runs"][0]))
def rec(d, run, i, fn): ej(f"{d}/{R[run]}/run_records.json", lambda x: fn(x[i]))
def qswap(d, a, b):
    t = open(f"{d}/{QL}").read().replace(f"train e3_5m_A_{a}.yaml", "XX").replace(f"train e3_5m_A_{b}.yaml", f"train e3_5m_A_{a}.yaml")
    open(f"{d}/{QL}", "w").write(t.replace("XX", f"train e3_5m_A_{b}.yaml"))


DATA = {
    "init_tail_drawn": lambda d: tail(d, "init3", 5, "drawn_cccc", lambda v: v + 1),
    "init_end_drawn": lambda d: rec(d, "init2", 1, lambda x: x.__setitem__("drawn_oasst2", x["drawn_oasst2"] + 1)),
    "data_same_stream": lambda d: rec(d, "data4", 1, lambda x: x.update(
        {k: v for k, v in json.load(open(f"{d}/{R['data2']}/run_records.json"))[1].items() if k.startswith(("drawn_", "dropped_"))})),
    "start_data_seed": lambda d: rec(d, "data3", 0, lambda x: x.__setitem__("data_seed", 1)),
    "start_no_data_seed": lambda d: rec(d, "s1_P3", 0, lambda x: x.pop("data_seed")),
    "schedule": lambda d: rec(d, "init5", 0, lambda x: x["schedule"].__setitem__("decay_start", 6103)),
    "resumed": lambda d: rec(d, "data5", 0, lambda x: x.__setitem__("resumed_from", "ckpt_00007344.pt")),
    "n_params": lambda d: rec(d, "init4", 0, lambda x: x.__setitem__("n_params", 5010134)),
    "bpb_step": lambda d: bpb(d, "data2", lambda r: r.pop(next(i for i, x in enumerate(r) if x["step"] == 7497))),
    "evalset": lambda d: bpb(d, "init2", lambda r: r[3].__setitem__("evalset_sha256", "0" * 64)),
    "max_windows": lambda d: bpb(d, "s1_P3", lambda r: r[0].__setitem__("max_windows", 100)),
    "bpb_run_field": lambda d: bpb(d, "data5", lambda r: r[2].__setitem__("run", "e3_5m_A_s1")),
    "tail_missing": lambda d: wjl(f"{d}/{R['init3']}/train_tail.jsonl", jl(f"{d}/{R['init3']}/train_tail.jsonl")[:-1]),
    "tail_nonfinite": lambda d: tail(d, "data3", 4, "loss", lambda v: float("nan")),
    "compile_on": lambda d: pf(d, "init4", lambda x: x["engine"].__setitem__("train.compile", True)),
    "preflight_refusal": lambda d: pf(d, "data4", lambda x: x["refusals"].append("x")),
    "cfg_edited": lambda d: sub(f"{d}/configs/e3_5m_A_init3.yaml", "data_seed: 1\n", "data_seed: 1 \n"),
    "prereg_edited": lambda d: sub(f"{d}/configs/prereg.yaml", "id: E3", "id: E3 "),
    "order": lambda d: qswap(d, "init2", "data2"),
    "two_trains": lambda d: sub(f"{d}/{QL}", "train e3_5m_A_init4.yaml --require-committed\n", "train e3_5m_A_init4.yaml --require-committed\n2026-10-11 13:12:29 train e3_5m_A_init4.yaml --require-committed\n"),
    "no_mark": lambda d: sub(f"{d}/{QL}", " MARK E3 PART 3 DONE\n", " plan note\n"),
    "queue_commit": lambda d: sub(f"{d}/{QL}", "code commit 4962426b8f0d", "code commit 4962426b8f0e"),
    "queue_plan_sha": lambda d: sub(f"{d}/{QL}", "plan part3.txt (sha256 07018f0f", "plan part3.txt (sha256 07018f0e"),
    "part2_prefix_edited": lambda d: sub(f"{d}/{QL}", "MARK E3 PART 2 DONE", "MARK E3 PART 2 DONE "),
    "second_start_no_list": lambda d: open(f"{d}/{QL}", "a").write(
        "2026-10-11 20:00:00 queue start: exp E3, plan part3.txt (sha256 07018f0fec572994), code commit 4962426b8f0d17738051bd499f9141a7cbc18c40\n"),
    "codelist_row": lambda d: sub(f"{d}/{CL}", "  harness/blocks.py", "  harness/blocks2.py"),
    "codelist_train": lambda d: sub(f"{d}/{CL}", "aa07b64c7a4e3089", "aa07b64c7a4e3088"),
    "codelist_analyzer": lambda d: sub(f"{d}/{CL}", "8392762a3552422b", "8392762a3552422c"),
    "p1_results": lambda d: open(f"{d}/results.json", "a").write(" "),
    "p4d_results": lambda d: open(f"{d}/results_part4d.json", "a").write(" "),
    "plan_edited": lambda d: open(f"{d}/plans/part3.txt", "a").write("# x\n"),
}
CODE = [  # (name, old, new): statistics mutants of e3analyze_p3.py
    ("df_is_n", "n = len(vals); df = n - 1;", "n = len(vals); df = n;"),
    ("no_sqrt2", "f, ref = E.factor(fz[\"df\"]), math.sqrt(2) * fz[\"rel\"]", "f, ref = E.factor(fz[\"df\"]), fz[\"rel\"]"),
    ("factor_df7", "f, ref = E.factor(fz[\"df\"]),", "f, ref = E.factor(7),"),
    ("rel_over_p1_mean", "\"rel\": sd / mean,", "\"rel\": sd / 1.2,"),
    ("chi2_swapped", "lo, hi = math.sqrt(df / E.chi2_ppf(0.975, df)), math.sqrt(df / E.chi2_ppf(0.025, df))",
     "lo, hi = math.sqrt(df / E.chi2_ppf(0.025, df)), math.sqrt(df / E.chi2_ppf(0.975, df))"),
    ("k_needed_from_2", "next(k for k in range(1, 10**6) if f * ref", "next(k for k in range(2, 10**6) if f * ref"),
    ("k_needed_loose", "if f * ref / math.sqrt(k) <= dl)", "if f * ref / math.sqrt(k) <= dl * 1.1)"),
    ("rule2_no_floor", "\"k_rule2\": {dl: max(2, k)", "\"k_rule2\": {dl: max(1, k)"),
    ("families_swapped", "CLASS = {\"INIT\": \"same stream, new init\", \"DATA\": \"same init, new stream\"}",
     "CLASS = {\"DATA\": \"same stream, new init\", \"INIT\": \"same init, new stream\"}"),
    ("shared_left_out", "FAM = {\"INIT\": [SHARED] + [f\"e3_5m_A_init{k}\"", "FAM = {\"INIT\": [f\"e3_5m_A_init{k}\""),
    ("gap_min_3", "if len(v) >= 4 else None", "if len(v) >= 3 else None"),
    ("F_swapped", "[math.sqrt(vr / f_ppf(0.975, z[\"df\"], 7)), math.sqrt(vr / f_ppf(0.025, z[\"df\"], 7))]",
     "[math.sqrt(vr / f_ppf(0.975, 7, z[\"df\"])), math.sqrt(vr / f_ppf(0.025, 7, z[\"df\"]))]"),
    ("replay_vs_A1_only", "\"minus_rep_mean\": v - st.fmean(rp),", "\"minus_rep_mean\": v - rp[0],"),
    ("ub_df7", "\"ub80\": sd * E.ub(df), \"rel_ub80\": sd * E.ub(df) / mean}", "\"ub80\": sd * E.ub(7), \"rel_ub80\": sd * E.ub(7) / mean}"),
]


def code_mutants():
    test = os.path.join(SRC, "tests", "test_e3analyze_p3.py"); bad = []
    for name, old, new in CODE:
        c = os.path.join(WD, "code_" + name); os.makedirs(c)
        for f in ("e3analyze.py", "e3analyze_p3.py", "e3analyze_p3_tables.py"): shutil.copy(os.path.join(SRC, f), c)
        p = os.path.join(c, "e3analyze_p3.py"); t = open(p).read()
        if t.count(old) != 1: print(f"INVALID  {name} (old text occurs {t.count(old)} times)"); bad.append(name); continue
        open(p, "w").write(t.replace(old, new))
        r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", test], capture_output=True, text=True,
                           env={**os.environ, "E3P3_CODE": c, "PYTHONDONTWRITEBYTECODE": "1"})
        k = r.returncode == 1; bad += [] if k else [name]
        print(f"{'KILLED ' if k else 'SURVIVED'} {name} (pytest rc {r.returncode})", flush=True)
    print(f"{len(CODE) - len(bad)}/{len(CODE)} killed"); return 1 if bad else 0


RESULT = {  # (name, file under the scratch root, old, new): tests/test_part3_result.py must fail on each
    "r_bpb_value": None,
    "r_results_json": ("experiments/E3_seed_noise/results_part3.json", '"gaps": []', '"gaps": ["e3_5m_A_data2"]'),
    "r_tables": ("experiments/E3_seed_noise/tables_part3.txt", "GAPS: none", "GAPS: none "),
    "r_notes_number": ("experiments/E3_seed_noise/notes.txt", "0.375% [0.225%, 1.078%]", "0.376% [0.225%, 1.078%]"),
    "r_notes_k": ("experiments/E3_seed_noise/notes.txt", "k_needed 0.5%: 16, 1%: 4", "k_needed 0.5%: 16, 1%: 3"),
    "r_notes_sha": ("experiments/E3_seed_noise/notes.txt", "results_part3.json sha256\n17b1614d", "results_part3.json sha256\n17b1614e"),
    "r_registry_cfg": ("models/registry.jsonl", '"config_sha256": "ef61c789f25cb71d', '"config_sha256": "ef61c789f25cb71e'),
    "r_registry_seed": ("models/registry.jsonl", '"id": "E3/5m_A/init3/final", "experiment": "E3", "arm": "5m_A", "role": "scored", "seed": 3',
                        '"id": "E3/5m_A/init3/final", "experiment": "E3", "arm": "5m_A", "role": "scored", "seed": 1'),
}


def result_mutants():
    root = os.path.dirname(os.path.dirname(SRC)); test = os.path.join(SRC, "tests", "test_part3_result.py"); bad = []
    for name, spec in RESULT.items():
        d = os.path.join(WD, name)
        shutil.copytree(SRC, os.path.join(d, "experiments", "E3_seed_noise"), ignore=shutil.ignore_patterns("__pycache__"))
        os.makedirs(os.path.join(d, "models")); shutil.copy(os.path.join(root, "models", "registry.jsonl"), os.path.join(d, "models"))
        if spec is None:   # one copied bpb value moves in its 7th digit
            p = os.path.join(d, "experiments", "E3_seed_noise", "out", "e3_5m_A_data3", "bpb.jsonl"); r = jl(p)
            i = next(i for i, x in enumerate(r) if x["set"] == "CHAT" and x["split"] == "all" and x["step"] == 7630)
            r[i]["bpb"] += 1e-6; wjl(p, r)
        else:
            p = os.path.join(d, spec[0]); t = open(p).read()
            if t.count(spec[1]) != 1: print(f"INVALID  {name}"); bad.append(name); continue
            open(p, "w").write(t.replace(spec[1], spec[2]))
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", test], capture_output=True, text=True,
                           env={**os.environ, "E3P3_ROOT": d, "PYTHONDONTWRITEBYTECODE": "1"})
        k = r.returncode == 1; bad += [] if k else [name]
        print(f"{'KILLED ' if k else 'SURVIVED'} {name} (pytest rc {r.returncode})", flush=True)
    print(f"{len(RESULT) - len(bad)}/{len(RESULT)} killed"); return 1 if bad else 0


def main():
    if "--code" in sys.argv: rc = code_mutants(); shutil.rmtree(WD); return rc
    if "--result" in sys.argv: rc = result_mutants(); shutil.rmtree(WD); return rc
    fx = "--fixture" in sys.argv
    base = fails(fresh(fx)); print("baseline failing:", len(base), sorted(base)); killed = 0
    for name, m in DATA.items():
        d = fresh(fx); m(d); f = fails(d); k = f != base; killed += k
        print(f"{'KILLED ' if k else 'SURVIVED'} {name:22s} new failures: {sorted(f - base)}", flush=True)
    print(f"{killed}/{len(DATA)} killed"); shutil.rmtree(WD); return 0 if killed == len(DATA) else 1


if __name__ == "__main__":
    sys.exit(main())
