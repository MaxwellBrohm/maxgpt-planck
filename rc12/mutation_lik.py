"""Mutation test of the likelihood rows (Max's rule: every test claim is watched failing). Each mutant replaces one
exact text in a scratch copy of rc12 (the *.py files and dev/); then test_lik.py or test_lik_more.py (STEP 11 FIX
ROUND) or, for the adapter arithmetic, test_lik_adapters.py under the Python named by LIK_TORCH_PY runs there.
KILLED = the test exits 1 (its own FAIL lines; an exception inside a test is reported as a FAIL). Exit 3 from
test_lik_adapters (no torch) is NOT RUN, not a kill; any other exit code is a problem. The unmutated copy must pass; malformed = the text is not found exactly
once. No model: stubs and a fixed bigram table only.
  LIK_TORCH_PY=<python with torch> python3 -B mutation_lik.py     (writes logs/mutation_lik.txt)"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TORCH_PY = os.environ.get("LIK_TORCH_PY")
S, R, P, U, A = "lik_score.py", "lik_rows.py", "pools_lik_prefix.py", "lik_run.py", "lik_adapters.py"
CM = "lik_compare.py"
T, T2, TA = "test_lik.py", "test_lik_more.py", "test_lik_adapters.py"
MUTANTS = [
    (S, "return all(scores[gold] > v for", "return all(scores[gold] >= v for", T, "a tie counts as right"),
    (S, "return scores[gold] - max(v for", "return scores[gold] - min(v for", T, "margin against the worst other"),
    (S, "ms = [guard_margin(r) for r in mt]", 'ms = [r["margin"] for r in ok if r["margin"] is not None]', T,
     "Tier 0 margin from the full candidate set on every row"),
    (S, 'acc_equal=rate([r["right"] for r in eq])', 'acc_equal=rate([r["right"] for r in ok])', T,
     "acc_equal over unequal rows"),
    (S, "equal=len(set(ntok.values())) == 1,", "equal=True,", T, "token counts never compared"),
    (S, "math.ceil(0.1 * len(xs))", "math.floor(0.1 * len(xs))", T, "p10 rank"),
    (S, 'return pre, list(tok.encode(cont, False)), "split"', 'return pre, full[len(pre):], "split"', T,
     "split path takes the joint tail"),
    (S, "while pre and pre[-1] == eos and full and full[-1] == eos:", "while False:", T, "trailing eos kept"),
    (S, 'return [(f[:lcp], f[lcp:], "canonical") for f in fulls]',
     'return [(f[:lcp - 1], f[lcp - 1:], "canonical") for f in fulls]', T, "canonical prefix one short"),
    (S, "if ctx is not None and max(len(p[0]) + len(p[1]) for p in parts) > ctx:", "if False:", T, "ctx ignored"),
    (S, 'acc=rate([all(t["right_m"] for t in tw) for tw in both])', 'acc=rate([any(t["right_m"] for t in tw) for '
     'tw in both])', T, "BIND pair right if either twin is"),
    (S, 'if len(tw) == 2 and all(t["foils"] for t in tw)]', "if len(tw) == 2]", T, "BIND pairs with unmatched twins"),
    (S, 'return [], RD.plain(messages) + " " + lead, True', 'return [], RD.plain(messages) + " " + lead, False', T,
     "plain render without special tokens"),
    (S, "return [], tok.template(messages) + lead, False", "return [], tok.template(messages), False", T,
     "template render without the lead-in"),
    (S, "if many is not None and all(p[0] == parts[0][0] for p in parts):", "if False:", T, "many() never used"),
    (R, 'msgs.append({"role": "assistant", "content": t["ideal"]})',
     'msgs.append({"role": "assistant", "content": t["text"]})', T, "history is not the IDEAL replies"),
    (R, 'if probe["grader"] == "DYN" and (probe.get("gold_fn") or {}).get("type") == "pick":', "if False:", T,
     "OWN pick dropped"),
    (R, "if len(cands) < 2:", "if len(cands) < 1:", T, "one-candidate probes kept"),
    (R, 'if rec["knowledge"]:', "if False:", T, "knowledge rows not excluded as such"),
    (R, 'return "COMPOSE" if (rec["family"], rec["cell"]) in S.DIAG else rec["family"]', 'return rec["family"]', T,
     "COMPOSE pooled into TWOHOP"),
    (R, 'lead = LP.lead_for(lead_key(rec), p["turn"])', 'lead = LP.lead_for(lead_key(rec), 1)', T,
     "lead-in drawn per record"),
    (P, "SEED = 4747\nLEADS", "SEED = 1212\nLEADS", T, "prefix seed = dev seed"),
    (P, "return stream(key, turn, seed).choice(LEADS)", "return LEADS[0]", T, "one lead-in for every row"),
    (P, '"The answer would be")', '"The answer would be", "That is Friday")', T, "a lead-in names a value"),
    (U, "if not args.prereg_pushed:", "if False:", T, "planck: scored before the push"),
    (U, 'if kind in ("hf", "planck") and not (LA.LIK_MODEL_TESTED or args.untested_ok):', "if False:", T,
     "untested adapters not refused"),
    (A, "return list(self.tok(text, add_special_tokens=special).input_ids)",
     "return list(self.tok(text, add_special_tokens=False).input_ids)", T2, "HFTok drops special tokens"),
    (A, "return list(self.r.encode(messages))", "return list(self.r.encode(messages))[:-1]", T2,
     "PlanckTok template without <|assistant|>"),
    (A, "lp = torch.log_softmax(logits[npre - 1:npre + nc - 1].float(), -1)",
     "lp = torch.log_softmax(logits[npre:npre + nc].float(), -1)", TA, "log-prob positions off by one"),
    (A, "logits[i], seqs[i], len(pre), len(c))", "logits[i], seqs[i], len(pre), len(conts[0]))", TA,
     "batched rows use the first continuation's length"),
    (A, "if self.ctx is not None and len(seq) > self.ctx:", "if False:", TA, "Planck ctx not enforced"),
    # STEP 11 FIX ROUND
    (R, 'return rec["meta"].get("pair_id") or rec["id"]', 'return rec["id"]', T2, "BIND twins' lead-ins per record"),
    (S, 'foils = [c for c in row["candidates"] if c != gold and ntok[c] == ntok[gold]]',
     'foils = [c for c in row["candidates"] if c != gold]', T2, "foils of any token count"),
    (S, "sub = {c: scores[c] for c in [gold] + foils}", "sub = scores", T2, "matched verdict over every candidate"),
    (S, "chance=1 / (1 + len(foils)) if foils else None", 'chance=1 / len(row["candidates"]) if foils else None', T2,
     "chance over the full candidate set"),
    (S, 'mt = [r for r in ok if r["foils"]]', 'mt = [r for r in ok if r["equal"]]', T2,
     "matched = all-equal rows only (the old filter)"),
    (S, 'return -math.inf if r["margin_m"] is None else r["margin_m"]', 'return 0.0 if r["margin_m"] is None else '
     'r["margin_m"]', T2, "a non-finite row enters the guard as 0"),
    (S, "ms = [guard_margin(r) for r in mt]", 'ms = [r["margin_m"] for r in mt if r["margin_m"] is not None]', T2,
     "non-finite rows dropped from the guard (the reviewer's finding)"),
    (S, 'n_nonfinite=sum(1 for r in mt if r["margin_m"] is None),', "n_nonfinite=0,", T2,
     "non-finite rows not counted"),
    (S, 'n_ties=sum(1 for r in mt if r["margin_m"] == 0.0),', "n_ties=0,", T2, "ties not counted"),
    (S, 's["note"] = OWN_NOTE', "pass", T2, "OWN copying note dropped"),
    (CM, 'both = set(a["foils"]) & set(b["foils"])', 'both = set(a["foils"])', T2, "compare on model a's foils"),
    (CM, "bad = [k for k in SAME if A[i][k] != B[i][k]]", "bad = []", T2, "rows of different builds compared"),
    (A, 'if str(model_type or "").startswith("gemma3"):', "if False:", T2, "Gemma 3 on the default sdpa path"),
    (A, 'if requested not in (None, "eager"):', "if False:", T2, "sdpa accepted for Gemma 3"),
    (A, 'kw["attn_implementation"] = attn', "pass", T2, "attention not passed to from_pretrained"),
    (A, 'def load(cls, model_id, dtype="float32", device="cuda", attn=None):',
     'def load(cls, model_id, dtype="bfloat16", device="cuda", attn=None):', T2, "HFLik.load defaults to bf16"),
    (U, 'ap.add_argument("--dtype", default="float32")', 'ap.add_argument("--dtype", default="bfloat16")', T2,
     "lik_run defaults to bf16"),
    (U, 'ap.add_argument("--planck-precision", default="fp32")',
     'ap.add_argument("--planck-precision", default="auto")', T2, "Planck precision auto (bf16 off cpu)"),
    (U, 'engine = getattr(logprob, "info", None) or dict(kind="stub", tok=args.tok)',
     'engine = dict(kind="stub", tok=args.tok)', T2, "engine info (attention, dtype) not in summary.json"),
    (U, " nonfinite {s['n_nonfinite']} ties {s['n_ties']}", "", T2, "console line without n_nonfinite / n_ties"),
]


def copy_tree(dst):
    for f in os.listdir(HERE):
        if f.endswith(".py"):
            shutil.copy(os.path.join(HERE, f), dst)
    shutil.copytree(os.path.join(HERE, "dev"), os.path.join(dst, "dev"))


def run(test, src=None, old=None, new=None):
    """-> (exit code or 'malformed' / 'not run', output tail)"""
    py = TORCH_PY if test == TA else sys.executable
    if not py:
        return "not run", "LIK_TORCH_PY unset"
    d = tempfile.mkdtemp(prefix="rc12_lik_mut_")
    try:
        copy_tree(d)
        if src:
            p = os.path.join(d, src)
            text = open(p).read()
            if text.count(old) != 1:
                return "malformed", f"{text.count(old)} matches"
            open(p, "w").write(text.replace(old, new))
        e = subprocess.run([py, "-B", test], cwd=d, capture_output=True, text=True, timeout=300)
        fails = [ln for ln in (e.stdout + e.stderr).splitlines() if ln.startswith("FAIL") or "Error" in ln]
        return e.returncode, (fails[0] if fails else (e.stdout + e.stderr).strip()[-160:])[:160]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    t0, lines, bad, not_run = time.time(), [], 0, 0
    for test in (T, T2, TA):
        rc, tail = run(test)
        lines.append(f"baseline {test}: {rc}" + ("" if rc == 0 else f"  {tail}"))
        bad += rc not in (0, "not run")
        not_run += rc == "not run"
    for k, (src, old, new, test, what) in enumerate(MUTANTS, 1):
        rc, tail = run(test, src, old, new)
        if rc == 1:
            lines.append(f"KILLED   #{k} {src}: {what}  [{tail}]")
        elif rc == "not run" or (test == TA and rc == 3):
            lines.append(f"NOT RUN  #{k} {src}: {what} ({tail})")
            not_run += 1
        else:
            lines.append(f"PROBLEM  #{k} {src}: {what} (exit {rc}) [{tail}]")
            bad += 1
    lines.append(f"{len(MUTANTS)} mutants, {bad} problems, {not_run} not run, {time.time() - t0:.0f} s")
    lines.append("ALL MUTANTS KILLED" if not bad and not not_run else "NOT ALL KILLED")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    open(os.path.join(HERE, "logs", "mutation_lik.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if bad or not_run else 0


if __name__ == "__main__":
    sys.exit(main())
