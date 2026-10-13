"""Mutation test of e3analyze_p1b.py and e3analyze_p3_20m.py on scratch copies (never the repo).
  python e3analyze_add1_mutants.py --fixture   DATA mutants on tests/add1fixture.py's synthetic record set (before any value)
  python e3analyze_add1_mutants.py --real      DATA mutants on a copy of this folder's copied records (after the copy-back)
    add --p1b or --p20 to either to run one analyzer's mutants (--real --p1b once logs/part1b/ is copied back)
  python e3analyze_add1_mutants.py --code      CODE mutants of the statistics: tests/test_e3analyze_add1.py must fail on each
A DATA mutant is killed when the set of failing checks of its analyzer differs from the unmutated copy's (expected: Part 3's
five FIXED 2 checks). Every mutant's old text must occur where it is applied. Writes nothing here."""
import json, os, shutil, subprocess, sys, tempfile
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SRC, "tests"))
PY, WD = sys.executable, tempfile.mkdtemp(prefix="e3a1_mut_")
A1B, A20 = "e3analyze_p1b.py", "e3analyze_p3_20m.py"
Q1, Q2 = "logs/part1b/queue_e3.txt", "logs/part3_20m/queue_e3.txt"


def fresh(fixture):
    root = os.path.join(WD, "root"); shutil.rmtree(root, ignore_errors=True)
    if fixture:
        import add1fixture
        return add1fixture.build(root)
    d = os.path.join(root, "experiments", "E3_seed_noise")
    shutil.copytree(SRC, d, ignore=shutil.ignore_patterns("__pycache__", "tests", "audit", "c6"))
    import add1fixture
    for p in add1fixture.BASE.values():
        os.makedirs(os.path.dirname(os.path.join(root, p)), exist_ok=True)
        shutil.copy(os.path.join(add1fixture.REPO, p), os.path.join(root, p))
    return d


def fails(d, script):
    r = subprocess.run([PY, os.path.join(d, script), "--checks-only"], capture_output=True, text=True)
    if r.returncode: return {"CRASH: " + r.stderr.strip().splitlines()[-1]}
    return {c["check"] for c in json.loads(r.stdout) if not c["ok"]}


def jl(p): return [json.loads(x) for x in open(p)]
def wjl(p, rows): open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
def ej(p, fn): x = json.load(open(p)); fn(x); json.dump(x, open(p, "w"), indent=1)
def sub(p, a, b):
    t = open(p).read(); assert t.count(a) >= 1, (p, a); open(p, "w").write(t.replace(a, b, 1))
def O(d, r, f): return f"{d}/out/{r}/{f}"
def tail(d, r, i, k, fn): p = O(d, r, "train_tail.jsonl"); x = jl(p); x[i][k] = fn(x[i][k]); wjl(p, x)
def end(d, r, fn): ej(O(d, r, "run_records.json"), lambda x: fn(x[1]))
def start(d, r, fn): ej(O(d, r, "run_records.json"), lambda x: fn(x[0]))
def bpb(d, r, fn): p = O(d, r, "bpb.jsonl"); x = jl(p); fn(x); wjl(p, x)
def pf(d, r, fn): ej(O(d, r, "preflight.json"), lambda x: fn(x["runs"][0]))
def both(d, r, k, fn): end(d, r, lambda x: x.__setitem__(k, fn(x[k]))); tail(d, r, -1, k, fn)
def copy_stream(d, r, frm):
    e = {k: v for k, v in json.load(open(O(d, frm, "run_records.json")))[1].items() if k.startswith(("drawn_", "dropped_"))}
    end(d, r, lambda x: x.update(e)); p = O(d, r, "train_tail.jsonl"); x = jl(p); x[-1].update(e); wjl(p, x)
def swap(p, a, b): t = open(p).read().replace(f"train {a}.yaml", "XX").replace(f"train {b}.yaml", f"train {a}.yaml"); open(p, "w").write(t.replace("XX", f"train {b}.yaml"))
def both_q(d, a, b): sub(f"{d}/{Q1}", a, b); sub(f"{d}/{Q2}", a, b)
def code_lists(d, a, b):
    for s in ("part1b", "part3_20m"):
        g = [f for f in os.listdir(f"{d}/logs/{s}") if f.startswith("code_sha256_")][0]; sub(f"{d}/logs/{s}/{g}", a, b)
def second_start(d):
    line = "2026-10-13 00:30:00 queue start: exp E3, plan part1b_p3at20m.txt (sha256 17c939a5e86f3d63), code commit " + "f" * 40 + "\n"
    for s, q in (("part1b", Q1), ("part3_20m", Q2)):
        g = sorted(f for f in os.listdir(f"{d}/logs/{s}") if f.startswith("code_sha256_"))[0]
        shutil.copy(f"{d}/logs/{s}/{g}", f"{d}/logs/{s}/code_sha256_20261013_003000.txt"); open(f"{d}/{q}", "a").write(line)


S12, S13, S14, S15 = "e3_5m_A_s12", "e3_5m_A_s13", "e3_5m_A_s14", "e3_5m_A_s15"
DATA = {  # name: (analyzer, mutation)
    "p1b_tail_end": (A1B, lambda d: tail(d, S14, -1, "drawn_cccc", lambda v: v + 1)),
    "p1b_same_stream": (A1B, lambda d: copy_stream(d, S15, S14)),
    "p1b_stream_is_part1s": (A1B, lambda d: copy_stream(d, S12, "e3_5m_A_s3")),
    "p1b_start_data_seed": (A1B, lambda d: start(d, "e3_5m_A_s16", lambda x: x.__setitem__("data_seed", 16))),
    "p1b_schedule": (A1B, lambda d: start(d, "e3_5m_A_s17", lambda x: x["schedule"].__setitem__("decay_start", 6103))),
    "p1b_resumed": (A1B, lambda d: start(d, "e3_5m_A_s18", lambda x: x.__setitem__("resumed_from", "ckpt_00007344.pt"))),
    "p1b_n_params": (A1B, lambda d: start(d, "e3_5m_A_s19", lambda x: x.__setitem__("n_params", 5010134))),
    "p1b_bpb_step": (A1B, lambda d: bpb(d, "e3_5m_A_s20", lambda r: r.pop(next(i for i, x in enumerate(r) if x["step"] == 7497)))),
    "p1b_evalset": (A1B, lambda d: bpb(d, "e3_5m_A_s21", lambda r: r[3].__setitem__("evalset_sha256", "0" * 64))),
    "p1b_run_field": (A1B, lambda d: bpb(d, S12, lambda r: r[2].__setitem__("run", "e3_5m_A_s1"))),
    "p1b_tail_missing": (A1B, lambda d: wjl(O(d, S13, "train_tail.jsonl"), jl(O(d, S13, "train_tail.jsonl"))[1:])),
    "p1b_compile_on": (A1B, lambda d: pf(d, S14, lambda x: x["engine"].__setitem__("train.compile", True))),
    "p1b_refusal": (A1B, lambda d: pf(d, S15, lambda x: x["refusals"].append("x"))),
    "p1b_cfg_edited": (A1B, lambda d: sub(f"{d}/configs/e3_5m_A_s13.yaml", "seed: 13\n", "seed: 13\ndata_seed: 13\n")),
    "p1b_prereg_edited": (A1B, lambda d: sub(f"{d}/configs/prereg.yaml", "id: E3", "id: E3 ")),
    "p1b_order": (A1B, lambda d: swap(f"{d}/{Q1}", S12, S13)),
    "p1b_two_trains": (A1B, lambda d: sub(f"{d}/{Q1}", f"train {S14}.yaml --require-committed\n",
                                          f"train {S14}.yaml --require-committed\n2026-10-13 00:00:01 train {S14}.yaml --require-committed\n")),
    "p1b_no_mark": (A1B, lambda d: sub(f"{d}/{Q1}", " MARK E3 PART 1B DONE\n", " plan note\n")),
    "p1b_mark_early": (A1B, lambda d: sub(f"{d}/{Q1}", f" done e3_5m_A_s21 rc 0\n", " MARK E3 PART 1B DONE\n")),
    "p1b_other_commit": (A1B, second_start),
    "p1b_plan_sha": (A1B, lambda d: sub(f"{d}/{Q1}", "(sha256 17c939a5", "(sha256 17c939a4")),
    "p1b_part3_prefix": (A1B, lambda d: sub(f"{d}/{Q1}", "MARK E3 PART 3 DONE", "MARK E3 PART 3 DONE ")),
    "p1b_list_harness_row": (A1B, lambda d: code_lists(d, "  harness/blocks.py", "  harness/blocks2.py")),
    "p1b_list_config": (A1B, lambda d: code_lists(d, "  experiments/E3_seed_noise/configs/e3_5m_A_s12.yaml", "  experiments/E3_seed_noise/configs/e3_5m_A_s12.yml")),
    "p1b_list_analyzer": (A1B, lambda d: code_lists(d, "  experiments/E3_seed_noise/e3analyze_p1b.py", "  experiments/E3_seed_noise/e3analyze_p1b.pyc")),
    "p1b_list_missing": (A1B, lambda d: [os.remove(f"{d}/logs/part1b/{f}") for f in os.listdir(f"{d}/logs/part1b") if f.startswith("code_")]),
    "p1b_base_edited": (A1B, lambda d: open(f"{d}/../screens/base_out/base_s104/bpb.jsonl", "a").write("\n")),
    "p1b_p3_results": (A1B, lambda d: open(f"{d}/results_part3.json", "a").write(" ")),
    "p1b_plan_edited": (A1B, lambda d: open(f"{d}/plans/part1b_p3at20m.txt", "a").write("# x\n")),
    "p20_init_tail": (A20, lambda d: tail(d, "e3_20m_A_init3", 5, "drawn_cccc", lambda v: v + 1)),
    "p20_init_end": (A20, lambda d: both(d, "e3_20m_A_init2", "drawn_oasst2", lambda v: v + 1)),
    "p20_data_same": (A20, lambda d: copy_stream(d, "e3_20m_A_data4", "e3_20m_A_data2")),
    "p20_s1p3_not_p2": (A20, lambda d: tail(d, "e3_20m_A_s1", 3, "drawn_irc", lambda v: v + 1)),
    "p20_start_data_seed": (A20, lambda d: start(d, "e3_20m_A_data3", lambda x: x.__setitem__("data_seed", 1))),
    "p20_start_no_data_seed": (A20, lambda d: start(d, "e3_20m_A_s1_P3", lambda x: x.pop("data_seed"))),
    "p20_schedule": (A20, lambda d: start(d, "e3_20m_A_init5", lambda x: x["schedule"].__setitem__("decay_start", 12206))),
    "p20_n_params": (A20, lambda d: start(d, "e3_20m_A_init4", lambda x: x.__setitem__("n_params", 5010133))),
    "p20_bpb_step": (A20, lambda d: bpb(d, "e3_20m_A_data5", lambda r: r.pop(next(i for i, x in enumerate(r) if x["step"] == 15250)))),
    "p20_tail_missing": (A20, lambda d: wjl(O(d, "e3_20m_A_data2", "train_tail.jsonl"), jl(O(d, "e3_20m_A_data2", "train_tail.jsonl"))[:-1])),
    "p20_cfg_edited": (A20, lambda d: sub(f"{d}/configs/e3_20m_A_init3.yaml", "data_seed: 1\n", "data_seed: 3\n")),
    "p20_order": (A20, lambda d: swap(f"{d}/{Q2}", "e3_20m_A_init2", "e3_20m_A_data2")),
    "p20_no_end_mark": (A20, lambda d: sub(f"{d}/{Q2}", " MARK E3 PART 3 AT 20M DONE\n", " plan note\n")),
    "p20_no_plan_end": (A20, lambda d: sub(f"{d}/{Q2}", " plan finished: part1b_p3at20m.txt\n", " plan note\n")),
    "p20_part1b_copy_edited": (A20, lambda d: sub(f"{d}/{Q1}", " MARK E3 PART 1B DONE\n", " MARK E3 PART 1B DONE \n")),
    "p20_list_config": (A20, lambda d: code_lists(d, "  experiments/E3_seed_noise/configs/e3_20m_A_data5.yaml", "  experiments/E3_seed_noise/configs/e3_20m_A_data6.yaml")),
    "p20_p2_results": (A20, lambda d: open(f"{d}/results_part2.json", "a").write(" ")),
}
CODE = [  # (file, name, old, new): tests/test_e3analyze_add1.py must fail on each
    (A1B, "pooled_df_total", "df = sum(len(g) - 1 for g in groups)", "df = sum(len(g) for g in groups) - 1"),
    (A1B, "pooled_grand_mean", "ss = sum(sum((x - st.fmean(g)) ** 2 for x in g) for g in groups)",
     "ss = sum(sum((x - st.fmean([y for h in groups for y in h])) ** 2 for x in g) for g in groups)"),
    (A1B, "pooled_rel_last_mean", '"rel": sd / mean,', '"rel": sd / st.fmean(groups[-1]),'),
    (A1B, "unpaired_no_sqrt2", "f, ref = E.factor(df), math.sqrt(2) * rel", "f, ref = E.factor(df), rel"),
    (A1B, "unpaired_factor_df7", "f, ref = E.factor(df), math.sqrt(2) * rel", "f, ref = E.factor(7), math.sqrt(2) * rel"),
    (A1B, "rule2_no_floor", '"k_rule2": {dl: max(2, k)', '"k_rule2": {dl: max(1, k)'),
    (A1B, "level_se_one_group", "sd * math.sqrt(1 / len(a) + 1 / len(b))", "sd * math.sqrt(1 / len(b))"),
    (A1B, "level_sign", "diff, se = st.fmean(b) - st.fmean(a),", "diff, se = st.fmean(a) - st.fmean(b),"),
    (A1B, "level_ci_normal", "t, tq = diff / se, E.t_ppf(0.975, df)", "t, tq = diff / se, 1.96"),
    (A1B, "level_p_one_sided", '"p_two_sided": 2 * (1 - E.t_cdf(abs(t), df))', '"p_two_sided": (1 - E.t_cdf(abs(t), df))'),
    (A1B, "base_sd_from_part1", "bz = pooled([b, list(vb.values())])", "bz = pz"),
    (A1B, "min_runs_5", 'SUB, RUNS, MIN_RUNS = "part1b", L.P1B, 6', 'SUB, RUNS, MIN_RUNS = "part1b", L.P1B, 5'),
    (A20, "r4_smaller", "use20 = rel20 >= rel5", "use20 = rel20 <= rel5"),
    (A20, "r4_df_20m_always", 'ref_from(rel20 if use20 else rel5, z["df"] if use20 else z5["df"])', 'ref_from(rel20 if use20 else rel5, z["df"])'),
    (A20, "r4_5m_no_sqrt2", 'rel5 = math.sqrt(2) * z["rel"], math.sqrt(2) * z5["rel"]', 'rel5 = math.sqrt(2) * z["rel"], z5["rel"]'),
    (A20, "fci_df_swapped", 'fci(z["rel"] / s2, z["df"], 2)', 'fci(z["rel"] / s2, 2, z["df"])'),
    (A20, "fci_bounds_swapped", "[math.sqrt(vr / P3.f_ppf(0.975, d1, d2)), math.sqrt(vr / P3.f_ppf(0.025, d1, d2))]",
     "[math.sqrt(vr / P3.f_ppf(0.025, d1, d2)), math.sqrt(vr / P3.f_ppf(0.975, d1, d2))]"),
    (A20, "hypot_is_sum", 'math.hypot(fz["INIT"]["rel"], fz["DATA"]["rel"]) / s2', '(fz["INIT"]["rel"] + fz["DATA"]["rel"]) / s2'),
    (A20, "shared_left_out", 'FAM = {"INIT": [SHARED] + [f"e3_20m_A_init{k}"', 'FAM = {"INIT": [f"e3_20m_A_init{k}"'),
    (A20, "classes_swapped", "r4[CLASS[fam]] = {", 'r4[{"INIT": "same init, new stream", "DATA": "same stream, new init"}[fam]] = {'),
    (A20, "min_family_3", "if len(v) >= 4 else None", "if len(v) >= 3 else None"),
    (A20, "replay_sign", '"diff": v - w,', '"diff": w - v,'),
    (A20, "values_from_5m", "vals = {fam: {n: P2.value(runs[n], m)", "vals = {fam: {n: E.value(runs[n], m)"),
]


def code_mutants():
    test = os.path.join(SRC, "tests", "test_e3analyze_add1.py"); bad = []
    for f, name, old, new in CODE:
        c = os.path.join(WD, "code_" + name); os.makedirs(c)
        import add1fixture
        for g in add1fixture.PY: shutil.copy(os.path.join(SRC, g), c)
        p = os.path.join(c, f); t = open(p).read()
        if t.count(old) != 1: print(f"INVALID  {name} (old text occurs {t.count(old)} times)"); bad.append(name); continue
        open(p, "w").write(t.replace(old, new))
        r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", test], capture_output=True, text=True,
                           env={**os.environ, "E3A1_CODE": c, "PYTHONDONTWRITEBYTECODE": "1"})
        k = r.returncode == 1; bad += [] if k else [name]
        print(f"{'KILLED ' if k else 'SURVIVED'} {name} (pytest rc {r.returncode})", flush=True)
    print(f"{len(CODE) - len(bad)}/{len(CODE)} killed"); return 1 if bad else 0


def main():
    if "--code" in sys.argv: rc = code_mutants(); shutil.rmtree(WD); return rc
    fx = "--fixture" in sys.argv
    only = A1B if "--p1b" in sys.argv else A20 if "--p20" in sys.argv else None
    base = {s: fails(fresh(fx), s) for s in (A1B, A20) if only in (None, s)}
    for s, b in base.items(): print(f"baseline failing: {len(b)} ({s}) {sorted(b)}")
    killed = 0
    todo = {n: v for n, v in DATA.items() if v[0] in base}
    for name, (s, m) in todo.items():
        d = fresh(fx)
        try: m(d)
        except (AssertionError, StopIteration, KeyError, IndexError) as e: print(f"INVALID  {name}: {e!r}"); continue
        f = fails(d, s); k = f != base[s]; killed += k
        print(f"{'KILLED ' if k else 'SURVIVED'} {name:24s} new failures: {sorted(f - base[s])}", flush=True)
    print(f"{killed}/{len(todo)} killed"); shutil.rmtree(WD); return 0 if killed == len(todo) else 1


if __name__ == "__main__":
    sys.exit(main())
