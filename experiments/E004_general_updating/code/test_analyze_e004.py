"""No-model end-to-end check of analyze_e004.py (+ analyze_e004_b.py) and pick_lr_e004.py on synthetic output files
written to a temporary directory (the real ../out is never touched). Exit 0 iff every check holds.
Writes ../logs/test_analyze_e004.txt.   usage: python3 -B test_analyze_e004.py"""
import json, os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_e004 as A
import pick_lr_e004 as PL
import rules_e004 as RU

M135, TS8, P31 = "HuggingFaceTB/SmolLM2-135M-Instruct", "roneneldan/TinyStories-8M", "EleutherAI/pythia-31m"
REFS = ["pron", "ell", "alias"]


def lik_recs(accs, n=64, seq_len=600):
    out = []
    for f in A.FAMILIES:
        k = round(accs.get(f, 0.9) * n)
        for i in range(n):
            sc = {"gold": -1.0, "c1": -2.0, "c2": -3.0} if i < k else {"gold": -2.0, "c1": -1.0, "c2": -3.0}
            out.append({"family": f, "scores": sc, "seq_len": seq_len + (300 if i % 4 == 0 else 0), "id": len(out),
                        "h": f"{f}{i}", "latest_ref": REFS[i % 3] if f in ("H1", "H2") else None})
    return out


def gen_recs(accs, n=64, bare=False):
    out = []
    for f in A.FAMILIES:
        k = round(accs.get(f, 0.9) * n)
        for i in range(n):
            out.append({"family": f, "strict": i < k, "lenient": i < k, "bare": bare, "long": not bare,
                        "n_words": 1 if bare else 6, "voice": False, "capped": False, "set": "e004",
                        "latest_ref": REFS[i % 3] if f in ("H1", "H2") else None})
    return out


def write(d, model, tag, lik=None, gen=None, gen_bare=False, cut=0):
    stem = os.path.join(d, f"{A.slug(model)}__{tag}")
    L, G = lik_recs(lik or {}), gen_recs(gen or {}, bare=gen_bare)
    with open(stem + "__e004__plain.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in L[:len(L) - cut])
    with open(stem + "__gen_e004__plain.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in G)
    json.dump({"lr": 5e-5, "steps": 400, "lock_in_step": 150}, open(stem + "__run.json", "w"))


def write_lr(d, model, lr, fam_acc, bad_loss=False):
    stem = os.path.join(d, f"{A.slug(model)}__lr{lr}_s0")
    L = [1.0] * 400
    if bad_loss:
        L[7] = float("nan")
    json.dump({"steps": 400, "losses": L, "eval_counts": {"dev/plain": 320}, "lock_in_step": None},
              open(stem + "__run.json", "w"))
    with open(stem + "__dev__plain.jsonl", "w") as f:
        for r in lik_recs(fam_acc, n=32):
            f.write(json.dumps(r) + "\n")


def checks(d, logs):
    lo = {f: 0.3 for f in A.FAMILIES}
    write(d, M135, "base", lo, lo)
    for s in (1, 2, 3):
        write(d, M135, f"s{s}", gen_bare=True)
    write(d, M135, "s4", {"H1": 0.5})
    write(d, M135, "s5", cut=1)                       # incomplete: counts as a failing seed
    write(d, TS8, "base")                            # an untouched pass
    for s in (1, 2, 3):
        write(d, TS8, f"s{s}", {"H4": 0.5})
    for s in (1, 2, 3):
        write(d, P31, f"s{s}", {"H1": 0.3, "ID": 0.9})
    res, lines = A.analyze(d, logs)
    M = res["models"]
    out = []
    ok = lambda cond, msg: out.append(("ok  " if cond else "FAIL") + " " + msg)
    ok(M["135m"]["reading"]["label"] == "PASS", f"135M PASS with 3 of 5 seeds ({M['135m']['reading']})")
    ok(M["135m"]["missing"] == ["s5"], f"incomplete s5 counted missing ({M['135m']['missing']})")
    ok(M["135m"]["untouched_pass"] is False, "135M untouched fails")
    ok(set(M["135m"]["below_chance_flags"]) == set(), "no below-chance flag at 0.9")
    ok(M["135m"]["base"]["below_chance"] == A.FAMILIES, "untouched at 0.3 with 3 candidates is below chance (1/3)")
    ok(M["135m"]["collateral"]["bare_flag"] == ["s1", "s2", "s3"], f"bare flag {M['135m']['collateral']['bare_flag']}")
    ok(M["ts8m"]["reading"]["label"] == "PARTIAL-X" and M["ts8m"]["reading"]["limit"] == ["H4"]
       and "positional" in M["ts8m"]["reading"]["detail"], f"TinyStories H4 -> PARTIAL-X ({M['ts8m']['reading']})")
    ok(M["ts8m"]["untouched_pass"] is True, "TinyStories fixture base passes untouched")
    r = M["p31m"]["reading"]
    ok(r["label"] == "FAIL" and any("shortcut" in s for s in r["sub"]) and any("not transfer" in s for s in r["sub"]),
       f"pythia H1 -> FAIL with sub-readings ({r})")
    ok(M["ts3m"]["reading"]["label"] == "NOT_SCORED", "a model with no runs is NOT_SCORED")
    ls = M["135m"]["collateral"]["length_split"]["s1"]["H1"]
    ok(ls[">768"][1] == 16 and ls["<=768"][1] == 48 and ls[">512"][1] == 64 and ls["<=512"][1] == 0,
       f"length split counts {ls}")
    ok(M["135m"]["seeds"]["s4"]["ref_split"]["H1 LIK"]["pron"] is not None, "H1 reference-form split present")
    ok(any("reading PASS" in x for x in lines), "tables carry the reading line")
    json.dumps(res)  # results must serialize
    # gates (the queue reads these lines)
    g = lambda *argv: __import__("subprocess").run([sys.executable, "-B", os.path.join(HERE, "analyze_e004.py"), *argv,
                                                    "--out-dir", d, "--logs-dir", logs], capture_output=True, text=True).stdout.strip()
    ok(g("--gate", M135) == "PASS", "gate 135M prints PASS")
    ok(g("--gate", P31) == "FAIL", "gate pythia-31m prints FAIL")
    ok(g("--gate-base", TS8) == "UNTOUCHED_PASS" and g("--gate-base", M135) == "UNTOUCHED_FAIL"
       and g("--gate-base", "EleutherAI/pythia-14m") == "UNTOUCHED_MISSING", "base gates")
    # LR picks
    base = {f: 0.5 for f in A.FAMILIES}
    write_lr(d, M135, "5e-05", base)
    write_lr(d, M135, "1.5e-04", {f: 0.625 for f in A.FAMILIES})
    p = PL.pick(M135, "135m", ["5e-05", "1.5e-04"], out_dir=d)
    ok(p["line"] == "CHOSEN 1.5e-04", f"135M LR check picks 1.5e-04 at +0.125 ({p['line']})")
    write_lr(d, M135, "1.5e-04", {f: 0.625 for f in A.FAMILIES}, bad_loss=True)
    p = PL.pick(M135, "135m", ["5e-05", "1.5e-04"], out_dir=d)
    ok(p["line"] == "CHOSEN 5e-05" and "1.5e-04" in p["ineligible"], f"a NaN loss makes a run ineligible ({p})")
    for lr, a in (("5e-05", 0.2), ("3e-04", 0.4), ("1e-03", 0.6)):
        write_lr(d, TS8, lr, {f: a for f in A.FAMILIES})
    p = PL.pick(TS8, "grid", ["5e-05", "3e-04", "1e-03"], out_dir=d)
    ok(p["line"] == "EXTEND 3e-03", f"grid extends ({p['line']})")
    write_lr(d, TS8, "3e-03", {f: 0.5 for f in A.FAMILIES})
    p = PL.pick(TS8, "grid", ["5e-05", "3e-04", "1e-03"], ext="3e-03", out_dir=d)
    ok(p["line"] == "CHOSEN 1e-03" and p["final"], f"final pick after the extension ({p['line']})")
    p = PL.pick(P31, "grid", ["5e-05", "3e-04", "1e-03"], out_dir=d)
    ok(p["line"].startswith("NONE"), "no LR runs -> NONE")
    return out


def main():
    with tempfile.TemporaryDirectory() as d:
        logs = os.path.join(d, "logs")
        os.makedirs(logs)
        out = checks(d, logs)
    bad = [x for x in out if x.startswith("FAIL")]
    txt = "\n".join(out + [f"{len(out) - len(bad)}/{len(out)} checks hold", "ALL PASS" if not bad else "FAILED"])
    print(txt)
    open(os.path.join(os.path.dirname(HERE), "logs", "test_analyze_e004.txt"), "w").write(txt + "\n")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
