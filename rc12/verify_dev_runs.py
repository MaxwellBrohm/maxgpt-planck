"""RC-12 dev-baseline run verifier (notes STEP 9 VERIFY; draft s3, s4, s9 OD1): reads the finished dev_batch.py runs of
one model and render and checks them against the draft. With --rerun N it reloads the run's engine (on the PC, under
the caller's gpu.lock) and replays N conversations.
  python -B verify_dev_runs.py --root ~/planck/runs/rc12_dev --model Qwen/Qwen2.5-0.5B-Instruct --render template
      [--rerun 5] [--data dev/rc12_dev.jsonl] [--tokenizer-only]
No model: C1 run config per run (meta.json: decode = hf_responder.DECODE; for vLLM the kwargs it passes, its sampling
defaults off, no generation_config.json stop id outside HF's list; for HF the generation_config fields generate would
still pick up), C2 own-cf pairing (every seed has its twin: same seed, render, responder, train seed, and the twin's
ids are the dev run's OWN ids; R is not None), C3 seeds (every row carries its run's seed; turn-1 replies, identical
prompts, differ across sampling seeds), C4 stops (template: eos / eot / cap; plain adds role, and reply = strip of
the role cut of raw), C5 the turn-3 prompt of one conversation rebuilt from its own history with the model's
tokenizer (printed; no system turn passed; Qwen3-family prompts end in an empty think block).
--rerun N: R1 the first N conversations of the first sampled run, replayed in lockstep, equal the stored rows; R2 the
same N greedy in lockstep equal the same N played sequentially (runner.play, one request per engine call), and both
are compared with the stored greedy rows. Prints PASS / FAIL / INFO lines; exit 1 on any FAIL."""
import argparse
import json
import os
import sys

import dev_batch as DB
import hf_responder as HR
import render as RD
import runner as R
import score as S
import vllm_responder as VR

OK_STOPS = {"template": {"eos", "eot", "cap"}, "plain": {"eos", "eot", "cap", "role"}}
HF_OVERRIDDEN = {"bos_token_id", "eos_token_id", "pad_token_id", "do_sample", "temperature", "top_p", "top_k",
                 "repetition_penalty", "max_length", "max_new_tokens", "use_cache", "cache_implementation",
                 "output_attentions", "output_hidden_states", "transformers_version", "_from_model_config"}
FAILS = []


def say(kind, msg):
    print(f"{kind:4s} {msg}", flush=True)
    if kind == "FAIL":
        FAILS.append(msg)


def load_runs(base):
    out = {}
    for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        p = os.path.join(base, d)
        if d.endswith(".partial") or not os.path.exists(os.path.join(p, "DONE")):
            continue
        out[d] = dict(meta=json.load(open(os.path.join(p, "meta.json"))),
                      rows=[json.loads(line) for line in open(os.path.join(p, "transcripts.jsonl"))])
    return out


def c1_config(name, run, engines, expect):
    m, a = run["meta"], run["meta"].get("audit") or {}
    if m.get("decode") != HR.DECODE:
        say("FAIL", f"C1 {name}: decode {m.get('decode')} != {HR.DECODE}")
    if m["engine"] == "vllm":
        want = VR.sampling_kwargs(0, "x", 1, a.get("stop_ids") or [])
        want.pop("seed")
        if a.get("sampled") != want or (a.get("greedy") or {}).get("temperature") != 0.0:
            say("FAIL", f"C1 {name}: vLLM kwargs sampled {a.get('sampled')} greedy {a.get('greedy')}")
        if a.get("vllm_default_sampling") not in ({}, None) or "vllm_config_error" in a:
            say("FAIL", f"C1 {name}: vLLM sampling defaults {a.get('vllm_default_sampling')} "
                        f"error {a.get('vllm_config_error')}")
        if a.get("gen_config_stops_not_in_hf"):
            say("FAIL", f"C1 {name}: generation_config.json adds stop ids {a['gen_config_stops_not_in_hf']} "
                        f"(vLLM stops {a.get('vllm_stop_ids')}, HF list {a.get('stop_ids')})")
    elif m["engine"] in ("hf", "hfb"):
        gc = a.get("hf_generation_config")
        leak = sorted(set(gc) - HF_OVERRIDDEN) if isinstance(gc, dict) else [f"unreadable: {gc}"]
        if leak:
            say("FAIL", f"C1 {name}: generation_config fields generate still applies: {leak}")
    chosen = DB.chosen_engine(m["model"], engines)
    if chosen is not None and chosen != m["engine"]:
        say("FAIL", f"C1 {name}: engine {m['engine']} but engines.json names {chosen}")
    exp = expect[1] if m["own_cf"] else expect[0]           # (records, OWN records) of --data: 640, 48 on dev
    if len(run["rows"]) != m["conversations"] or (m.get("limit") is None and m["conversations"] != exp):
        say("FAIL", f"C1 {name}: {len(run['rows'])} rows, meta {m['conversations']} conversations, expected {exp}")
    say("INFO", f"C1 {name}: engine {m['engine']} dtype {m.get('dtype')} parity {m.get('parity')} decision "
                f"{m.get('engine_decision')} gate_skipped {m.get('parity_gate_skipped')} stops "
                f"{a.get('vllm_stop_ids') or a.get('stop_ids')} qwen3 {a.get('qwen3')} ctx {m.get('ctx')} "
                f"{m['seconds']} s versions {m.get('versions')}")


def c2_pairing(runs):
    for name, run in runs.items():
        if name.endswith("_owncf"):
            continue
        twin = runs.get(name + "_owncf")
        if twin is None:
            say("FAIL", f"C2 {name}: no finished _owncf twin (R would be None)")
            continue
        m, t = run["meta"], twin["meta"]
        for k in ("seed", "render", "model", "train_seed", "engine", "dtype"):
            if m.get(k) != t.get(k):
                say("FAIL", f"C2 {name}: {k} {m.get(k)} vs twin {t.get(k)}")
        own = sorted(r["id"] for r in run["rows"] if r["family"] == "OWN")
        tw = sorted(r["id"] for r in twin["rows"])
        flags = {(r["own_cf"], r["seed"], r["responder"], r["render"]) for r in twin["rows"]}
        if own != tw or flags != {(True, m["seed"], m["model"], m["render"])} \
                or any(r["own_cf"] for r in run["rows"]):
            say("FAIL", f"C2 {name}: twin ids / flags do not pair ({len(own)} OWN vs {len(tw)} twin, {flags})")
        summ = S.summarize(run["rows"] + twin["rows"])
        say("INFO" if summ["R"] is not None else "FAIL", f"C2 {name}: R {summ['R']} R_ungated {summ['R_ungated']} "
            f"own_gate {summ['own_gate']}")


def c3_seeds(runs):
    for name, run in runs.items():
        bad = [r["id"] for r in run["rows"] if r["seed"] != run["meta"]["seed"]]
        if bad:
            say("FAIL", f"C3 {name}: {len(bad)} rows carry another seed")
    first = {n: {r["id"]: r["turns"][0]["reply"] for r in run["rows"]} for n, run in runs.items()
             if not n.endswith("_owncf")}
    names = sorted(first)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = set(first[a]) & set(first[b])
            same = sum(first[a][k] == first[b][k] for k in common)
            kind = "FAIL" if common and same == len(common) and "greedy" not in (a, b) else "INFO"
            say(kind, f"C3 turn-1 replies identical, {a} vs {b}: {same} / {len(common)}")


def c4_stops(name, run, render):
    seen, bad = {}, []
    for r in run["rows"]:
        for t in r["turns"]:
            seen[t["stop"]] = seen.get(t["stop"], 0) + 1
            if t["stop"] not in OK_STOPS[render] or t["reply"] != t["reply"].strip():
                bad.append((r["id"], t["i"], t["stop"]))
            elif render == "plain" and not run["meta"]["own_cf"]:
                raw = t["raw"] if t.get("raw") is not None else t["reply"]
                text, cut = RD.cut_plain(raw)
                if text.strip() != t["reply"] or cut != (t["stop"] == "role"):
                    bad.append((r["id"], t["i"], "cut"))
    say("FAIL" if bad else "INFO", f"C4 {name}: stops {seen}" + (f"; bad {len(bad)} e.g. {bad[:3]}" if bad else ""))


def c5_prompt(model, render, run):
    from transformers import AutoConfig, AutoTokenizer
    tok, cfg = AutoTokenizer.from_pretrained(model), AutoConfig.from_pretrained(model)
    shim = VR.VLLMResponder(model, render=render, tokenizer=tok, model_type=getattr(cfg, "model_type", ""),
                            native=VR.native_len(cfg), llm=object(), sampling_params=dict)
    row = next(r for r in run["rows"] if r["family"] != "T0")
    hist = []
    for t in row["turns"][:3]:
        hist += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["reply"]}]
    hist = hist[:-1]
    s = tok.decode(shim.encode(hist), skip_special_tokens=False)
    print(f"C5 {model} {render} {row['id']} turn 3, qwen3 {shim.qwen3} rule_differs {shim.thinking_rule_differs}:\n"
          f"{s!r}")
    missing = [m["content"] for m in hist if m["content"] not in s]
    if missing or (render == "template" and shim.qwen3 and not s.endswith("<think>\n\n</think>\n\n")):
        say("FAIL", f"C5 {model}: history text missing {missing[:2]} or thinking not off")


def rerun(args, runs, recs):
    sampled = [n for n in runs if n.isdigit()]
    ref = runs[sampled[0]] if sampled else None
    greedy = runs.get("greedy")
    m = (ref or greedy)["meta"]
    ns = argparse.Namespace(render=args.render, dtype=m.get("dtype") or "bfloat16", device="cuda",
                            gpu_mem=m.get("gpu_mem", 0.85), max_model_len=m.get("max_model_len"),
                            batch_invariant=bool(m.get("batch_invariant")), lockstep=True, hf_untested_ok=True,
                            vllm_untested_ok=True)
    spec = f"{m['engine']}:{args.model}"
    eng, name = R.make_responder(spec, ns)
    ctx = getattr(eng, "ctx", None)
    seq = eng if hasattr(eng, "start") else R.make_responder(spec, argparse.Namespace(**{**vars(ns), "lockstep": False}))[0]

    def cmp(tag, a, b):
        diff = [(x["id"], ta["i"], ta["reply"][:80], tb["reply"][:80]) for x, y in zip(a, b)
                for ta, tb in zip(x["turns"], y["turns"]) if (ta["reply"], ta["stop"]) != (tb["reply"], tb["stop"])]
        convs = sum(all((ta["reply"], ta["stop"]) == (tb["reply"], tb["stop"]) for ta, tb in zip(x["turns"], y["turns"]))
                    for x, y in zip(a, b))
        say("FAIL" if diff else "PASS", f"{tag}: {convs} / {len(a)} conversations identical, {len(diff)} turns differ"
            + (f"; first {diff[0]}" if diff else ""))
    if ref is not None:
        seed = ref["meta"]["seed"]
        stored = ref["rows"][:args.rerun]
        again = R.run([recs[r["id"]] for r in stored], eng, args.render, [seed], ctx, None, name, 0, False, True)
        cmp(f"R1 seed {seed} lockstep replay vs stored", again, stored)
        other = R.run([recs[r["id"]] for r in stored], eng, args.render, [seed + 100], ctx, None, name, 0, False, True)
        same = sum(a["turns"][0]["reply"] == b["turns"][0]["reply"] for a, b in zip(other, stored))
        say("INFO", f"R1 seed {seed + 100} vs seed {seed}: turn-1 replies identical {same} / {len(stored)}")
    ids = [r["id"] for r in (greedy or ref)["rows"][:args.rerun]]
    lock = R.run([recs[i] for i in ids], eng, args.render, [None], ctx, None, name, 0, False, True)
    serial = R.run([recs[i] for i in ids], seq, args.render, [None], ctx, None, name, 0, False, False)
    cmp("R2 greedy lockstep vs sequential", lock, serial)
    if greedy is not None:
        cmp("R2 greedy lockstep replay vs stored greedy run", lock, greedy["rows"][:args.rerun])
    say("INFO", f"R batches {getattr(eng, 'batches', [])[:30]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--render", choices=["template", "plain"], required=True)
    ap.add_argument("--data", default=R.DEV)
    ap.add_argument("--engines", default=DB.ENGINES)
    ap.add_argument("--rerun", type=int, default=0)
    ap.add_argument("--no-prompt", action="store_true", help="skip C5 (no tokenizer: fakes)")
    args = ap.parse_args()
    base = os.path.join(os.path.expanduser(args.root), DB.slug(args.model), args.render)
    runs = load_runs(base)
    if not runs:
        say("FAIL", f"no finished run under {base}")
        return 1
    say("INFO", f"runs {sorted(runs)}")
    data = R.load(args.data)
    expect = (len(data), sum(r["family"] == "OWN" for r in data))
    for n, run in runs.items():
        c1_config(n, run, args.engines, expect)
        c4_stops(n, run, args.render)
    c2_pairing(runs)
    c3_seeds(runs)
    if not args.no_prompt:
        c5_prompt(args.model, args.render, next(r for n, r in sorted(runs.items()) if not n.endswith("_owncf")))
    if args.rerun:
        rerun(args, runs, {r["id"]: r for r in data})
    print(f"VERIFY {'FAIL' if FAILS else 'PASS'} ({len(FAILS)} failures) {args.model} {args.render}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
