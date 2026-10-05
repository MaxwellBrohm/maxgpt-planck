"""W3 preflight (PC CPU, no model loaded, before any hold): every call kind, for every teacher, is rendered and its
regex compiled by xgrammar on the teacher's own tokenizer (the TokenizerInfo vLLM builds, as teachers/test_decode_pc
does); a well-formed answer (the test fake's, with Ministral's blank lines between lines too) must be accepted and a
broken one refused; the rendered prompt plus max_tokens must fit max_model_len. The calls come from a mini stage run
on a sample of the real plan with the FAKE teacher in a temporary directory, so derived and judge calls are covered.

    python -m bankpass.preflight3 --plan STAGE/plan.json --wordlist WORDLIST.tsv [--per-kind 4]"""
import argparse
import collections
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "teachers"))


class Recorder:
    def __init__(self, fake):
        self.fake, self.seen = fake, []

    def generate(self, prompt, spec, seed, preset=None, call=None):
        row = self.fake.generate(prompt, spec, seed, preset, call)
        self.seen.append((call, prompt, spec, row["text"]))
        return row


def sample_plan(plan, per_bank=2):
    """per teacher: up to per_bank base calls of each bank, so every kind and class is present but small."""
    out, n = [], collections.Counter()
    for c in plan["calls"]:
        k = (c["teacher"], c["bank"])
        if n[k] < per_bank:
            n[k] += 1
            out.append(c)
    return dict(plan, calls=out)


def collect(plan, wl):
    import _w3fake as FK
    from bankpass import hold, plan as PL, w3state as WS
    seen = []
    with tempfile.TemporaryDirectory() as root:
        for t in PL.TEACHER_ORDER + PL.TEACHER_ORDER[:2]:
            rec = Recorder(FK.Fake(t))
            for _ in range(10):
                b, d, j, _ = hold.owed(root, t, plan, wl)
                if not b + d + j:
                    break
                hold.run_calls(rec, b + d + j, WS.calls_path(root, t), 1e12, 8, {"by": "Max", "date": "x"}, "pf",
                               lambda m: None, set(), {})
            seen += rec.seen
            WS.refresh(root)
    return seen


def check(seen, per_kind):
    import serve
    import xgrammar as xgr
    from test_decode_pc import xgr_info
    out = {}
    by = collections.defaultdict(list)
    for call, prompt, spec, text in seen:
        by[(call["teacher"], call["kind"])].append((call, prompt, spec, text))
    for t in sorted({k[0] for k in by}):
        comp = xgr.GrammarCompiler(xgr_info(t), max_threads=4)
        tok = serve.tokenizer_only(t)
        for (tt, kind), rows in sorted(by.items()):
            if tt != t:
                continue
            r = {"n": 0, "accepted": 0, "blank_ok": 0, "refused_broken": 0, "compile_s_max": 0.0, "prompt_tok_max": 0,
                 "over_len": 0}
            for call, prompt, spec, text in rows[:per_kind]:
                r["n"] += 1
                t0 = time.time()
                ctx = comp.compile_regex(spec["regex"])
                r["compile_s_max"] = round(max(r["compile_s_max"], time.time() - t0), 3)
                broken = "\n".join(text.split("\n")[:-1] + ["one more line here", "END"])
                for txt, key, want in ((text, "accepted", True), (text.replace("\n", "\n\n"), "blank_ok", True),
                                       (broken, "refused_broken", False)):
                    m = xgr.GrammarMatcher(ctx)
                    got = bool(m.accept_string(txt) and m.is_completed())
                    r[key] += got == want if key == "refused_broken" else got
                ids, _ = serve.render(tok, prompt)
                r["prompt_tok_max"] = max(r["prompt_tok_max"], len(ids))
                r["over_len"] += len(ids) + spec["max_tokens"] > serve.ENGINE["max_model_len"]
            out[f"{t}:{kind}"] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--wordlist", required=True)
    ap.add_argument("--per-kind", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from bankpass import w3state, wordload
    w3state.install_human()
    with open(a.plan, encoding="utf-8") as f:
        plan = json.load(f)
    seen = collect(sample_plan(plan), wordload.read(a.wordlist))
    res = check(seen, a.per_kind)
    bad = {k: v for k, v in res.items() if v["accepted"] < v["n"] or v["refused_broken"] < v["n"] or v["over_len"]
           or (k.startswith("ministral") and v["blank_ok"] < v["n"])}
    rep = {"kinds": res, "bad": sorted(bad), "calls_seen": len(seen)}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1, sort_keys=True)
    print(json.dumps(rep, indent=1, sort_keys=True))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
