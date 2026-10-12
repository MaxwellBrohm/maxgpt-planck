"""RC-12 likelihood rows, CLI (prereg draft s4, P-004; diagnostic, never in a composite or a claim). runner.py is not
used or changed: the rows are scored on GOLDEN history (lik_rows.py), not played.

  python3 -B lik_run.py --responder stub:IDEAL --tok chunk --render both --out runs/lik_stub_IDEAL
  <venv python> -B lik_run.py --responder hf:Qwen/Qwen2.5-0.5B-Instruct --render both --device mps \\
      --out runs/lik_qwen05 --untested-ok
  <venv python> -B lik_run.py --responder planck:<ckpt.pt> --planck-config <config.yaml> --render both \\
      --out runs/lik_planck_x --untested-ok --prereg-pushed

Responders: stub:IDEAL | stub:LENGTH | stub:RECENCY | stub:MENTIONS (lik_stubs.py; --tok word | chunk | merge |
canonical | eos), hf:<model id or path> and planck:<checkpoint> (lik_adapters.py, UNTESTED on a model:
--untested-ok; planck: also --prereg-pushed, because no Planck model is scored on any RC-12 split before the
pre-registration is pushed).
Output (--out DIR): excluded.json (probes without a closed candidate set, per key and reason) and, per render,
<render>/rows.jsonl (one scored row each: scores, ntok, how, equal, foils, chance, right, margin, right_m,
margin_m, over_ctx) and
<render>/summary.json (lik_score.summarize: per family n_matched, acc_matched, chance_matched, n_equal,
acc_equal, n_unequal, acc_unequal, n_split, n_nonfinite, n_ties, tier0 {n, mean, p10}, BIND pairs; plus "engine":
the scorer's info, e.g. model_type, dtype and attention for hf:). The Tier 0 margin is over the matched rows: the gold
against the candidates of its own token count (lik_score docstring). Precision (STEP 11 FIX ROUND): --dtype
defaults to float32 and --planck-precision to fp32 (E004 lik.load's default; every panel model up to 0.5B fits);
a bf16 run is allowed and its ties are counted (n_ties). Two models are compared with lik_compare.py.
hf: engine options (STEP 14S): trust_remote_code, attn_implementation and chat_template come from the model's
--engines entry (lik_adapters.engine_opts), so a row is scored as its generation runs ran; an --attn that differs
from the entry's is refused. Real-model self-check (--selfcheck N, STEP 14S): the first N scored rows of each render
are scored again one candidate at a time (no padded batch) and compared with the batched scores; summary.json gets
selfcheck {n_rows, n_scores, max_abs_diff, flips, tol, ok}, and the run exits 4 if the largest gap passes the tol
(SELFCHECK_TOL per dtype) or a matched verdict flips. Run it on every model's first run."""
import argparse
import json
import os
import sys
import time

import lik_rows as LR
import lik_score as LS

HERE = os.path.dirname(os.path.abspath(__file__))
SELFCHECK_TOL = {"float32": 0.01, "bfloat16": 0.25, "float16": 0.25}   # nats, summed over a candidate's tokens


class Single:
    """a scorer without many(): lik_score.score_row then scores each candidate alone (a batch of one, no padding)."""

    def __init__(self, f):
        self.f = f

    def __call__(self, pre, cont):
        return self.f(pre, cont)


def selfcheck(scored, rows, logprob, tok, render, ctx, n, tol):
    """scores the first n scored (not over_ctx) rows again through Single(logprob) and compares them."""
    by = {r["id"]: r for r in rows}
    todo = [r for r in scored if r["scores"] is not None][:n]
    gaps, flips = [], 0
    for r in todo:
        again = LS.score_row(by[r["id"]], Single(logprob), tok, render, ctx)
        gaps += [abs(again["scores"][c] - r["scores"][c]) for c in r["candidates"]]
        flips += again["right_m"] != r["right_m"]
    worst = max(gaps) if gaps else None
    ok = bool(todo) and worst is not None and worst <= tol and flips == 0
    return dict(n_rows=len(todo), n_scores=len(gaps), max_abs_diff=worst, flips=flips, tol=tol, ok=ok)


def stub_prompts(rows, tok, render):
    out = []
    for r in rows:
        head, text, _ = LS.prompt_parts(tok, r["messages"], r["lead"], render)
        assert not head, "stub renders are text"
        out.append((text, r["gold"], r["candidates"]))
    return out


def make(args, rows, render, cache):
    """-> (logprob, tok, ctx); a model is loaded once and reused for the second render (cache)."""
    kind, _, name = args.responder.partition(":")
    if kind in cache:
        return cache[kind]
    if kind == "stub":
        import lik_stubs as ST
        tok = ST.make_tok(args.tok)
        if name == "IDEAL":
            return ST.Ideal(tok, stub_prompts(rows, tok, render)), tok, None
        if name == "MENTIONS":
            return ST.Mentions(tok), tok, None
        if name == "LENGTH":
            return ST.length, tok, None
        if name == "RECENCY":
            return ST.Recency(tok), tok, None
        sys.exit(f"unknown stub {name}")
    import lik_adapters as LA
    if kind in ("hf", "planck") and not (LA.LIK_MODEL_TESTED or args.untested_ok):
        sys.exit("the hf / planck likelihood adapters are UNTESTED on a model; rerun with --untested-ok")
    if kind == "hf":
        opts = LA.engine_opts(name, args.engines)
        want = opts.get("attn_implementation")
        if args.attn is not None and want is not None and args.attn != want:
            sys.exit(f"{name}: --attn {args.attn} refused; engines.json runs it with {want}")
        scorer, tok = LA.HFLik.load(name, args.dtype, args.device, args.attn or want,
                                    opts.get("trust_remote_code", False), opts.get("chat_template"))
        cache[kind] = scorer, tok, scorer.ctx
        return cache[kind]
    if kind == "planck":
        if not args.prereg_pushed:
            sys.exit("no Planck model is scored on any RC-12 split before the pre-registration is pushed "
                     "(prereg draft s0); after the push, rerun with --prereg-pushed")
        scorer, tok = LA.PlanckLik.load(name, args.planck_config, args.planck_tokenizer, args.planck_device,
                                        args.planck_precision)
        cache[kind] = scorer, tok, scorer.ctx
        return cache[kind]
    sys.exit(f"unknown responder {args.responder}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--responder", required=True)
    ap.add_argument("--render", choices=["template", "plain", "both"], default="both")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=LR.DEV)
    ap.add_argument("--families", default=None, help="comma list of keys (families; COMPOSE for TWOHOP:COMPOSE)")
    ap.add_argument("--tok", default="chunk", help="stub tokenizer: word | chunk | merge | canonical | eos")
    ap.add_argument("--untested-ok", action="store_true")
    ap.add_argument("--prereg-pushed", action="store_true")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--attn", default=None, help="hf: attn_implementation (default: the library's; Gemma 3: eager)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--planck-config", default=None)
    ap.add_argument("--planck-tokenizer", default=None)
    ap.add_argument("--planck-device", default="cpu")
    ap.add_argument("--planck-precision", default="fp32")
    ap.add_argument("--engines", default=os.path.join(HERE, "engines.json"))
    ap.add_argument("--selfcheck", type=int, default=0, help="rows re-scored one candidate at a time per render")
    args = ap.parse_args(argv)
    rows, excluded = LR.build(LR.load(args.data))
    if args.families:
        keep = set(args.families.split(","))
        rows = [r for r in rows if r["key"] in keep]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "excluded.json"), "w") as f:
        json.dump({f"{k}:{w}": n for (k, w), n in sorted(excluded.items())}, f, indent=1)
    renders = ["template", "plain"] if args.render == "both" else [args.render]
    cache, bad = {}, []
    for render in renders:
        t0 = time.time()
        logprob, tok, ctx = make(args, rows, render, cache)
        scored = LS.score_rows(rows, logprob, tok, render, ctx)
        engine = getattr(logprob, "info", None) or dict(kind="stub", tok=args.tok)
        summary = dict(responder=args.responder, render=render, rows=len(scored), engine=engine,
                       families=LS.summarize(scored))
        if args.selfcheck:
            tol = SELFCHECK_TOL.get(args.dtype, SELFCHECK_TOL["float32"])
            summary["selfcheck"] = sc = selfcheck(scored, rows, logprob, tok, render, ctx, args.selfcheck, tol)
            print(f"{render}: selfcheck {sc['n_rows']} rows, {sc['n_scores']} scores, max gap "
                  f"{fmt(sc['max_abs_diff'], 6)} nats (tol {tol}), flips {sc['flips']}: {'ok' if sc['ok'] else 'FAIL'}")
            if not sc["ok"]:
                bad.append(render)
        d = os.path.join(args.out, render)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "rows.jsonl"), "w") as f:
            for r in scored:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        for key, s in summary["families"].items():
            t = s["tier0"]
            print(f"{render:8s} {key:8s} n {s['n']:3d} matched {s['n_matched']:3d} acc {fmt(s['acc_matched'])} "
                  f"chance {fmt(s['chance_matched'])} equal {s['n_equal']:3d} acc_equal {fmt(s['acc_equal'])} "
                  f"tier0 mean {fmt(t['mean'])} p10 {fmt(t['p10'])} nonfinite {s['n_nonfinite']} ties {s['n_ties']}")
        print(f"{render}: {len(scored)} rows in {time.time() - t0:.1f} s -> {d}")
    if bad:
        print(f"SELFCHECK FAILED: {', '.join(bad)}")
        return 4
    return 0


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


if __name__ == "__main__":
    sys.exit(main())
