"""RC-12 greedy HF-vs-vLLM parity check (draft s4, s16, s19 item 2; notes STEP 9). One model per call, template
render, the first --n (20) dev conversations (--pick spread: round-robin over families). Runs on the 5070 only (each stage is its own process, so one engine's
GPU memory is freed before the next loads; hold the GPU lock around the whole call, pc_jobs.py does).

  stage vllm   vllm_responder plays the conversations greedy on its OWN history through the real lockstep runner
               (runner.run lockstep=True; transcripts in <out>/vllm_run/) and traces every reply: history, prompt
               ids, reply ids (HF convention: stop token last), text, stop reason  -> <out>/vllm.jsonl
  stage hf     hf_responder.HFResponder replies greedy to EXACTLY the same histories (the vLLM run's), so every turn
               is compared on the same prompt. Per turn: prompt ids equal?, reply ids, text, stop; where the ids
               differ, the first differing token position j and HF's log-prob margin there: log p(HF token) -
               log p(vLLM token) under one HF forward on prompt + the common prefix (a near-tie shows as a small
               margin)                                                                 -> <out>/hf.jsonl
  stage report per-turn lines (match, first differing token, margin, both texts where they differ) and the
               verdict                                            -> <out>/parity.txt, <out>/parity.json
  stage all    vllm, hf (subprocesses), then report.   --table DIR [DIR ...]: one line per model from parity.json.

Verdict (proposal, mechanical; the engine per model is named from it, draft s16):
  FAIL_PROMPT  any turn whose HF and vLLM prompt ids differ (a render or tokenizer difference, not the engine)
  FAIL_STOP    equal reply ids but a different stop reason or text
  IDENTICAL    every turn: equal ids and stop reason
  NEAR_TIE     every differing turn's first difference is at an HF near-tie: margin <= NEAR_TIE nats
  DIFFERS      anything else
Run (PC): python -B parity_hf_vllm.py --model Qwen/Qwen2.5-0.5B-Instruct --out ~/planck/runs/rc12_dev/<m>/parity"""
import argparse
import json
import os
import subprocess
import sys

import runner as R

NEAR_TIE = 0.5
ENGINES = {"vllm": "vLLM", "hfb": "batched HF (hf_batched.py)"}
STAGES = ("vllm", "hf", "report", "all")


def read(path):
    return [json.loads(line) for line in open(path)]


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def first_diff(a, b):
    for j, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return j
    return None if len(a) == len(b) else min(len(a), len(b))


def pick(recs, n, how="first"):
    """first: the first n dev records (the task's rule; in dev order these are all T0). spread: round-robin over
    the families in dev order, the first unused record of each in turn."""
    if how == "first":
        return recs[:n]
    queues, out = {}, []
    for r in recs:
        queues.setdefault(r["family"], []).append(r)
    while len(out) < n and any(queues.values()):
        for q in queues.values():
            if q and len(out) < n:
                out.append(q.pop(0))
    return out


def stage_vllm(args):
    engine = getattr(args, "engine", "vllm")
    if engine == "hfb":                          # Doge and anything vLLM rejects: batched HF vs serial HF
        import hf_batched as HB
        import transformers as vllm              # its version goes in the meta's "vllm" field
        v = HB.HFBatched(args.model, "template", args.dtype, args.device)
        v.thinking_rule_differs = False
    else:
        import vllm
        import vllm_responder as VR
        v = VR.VLLMResponder(args.model, "template", args.dtype, args.gpu_mem, args.max_model_len)
    v.trace = []
    recs = pick(R.load(args.data), args.n, args.pick)
    R.run(recs, v, "template", [None], v.ctx, os.path.join(args.out, "vllm_run"), args.model, lockstep=True)
    write(os.path.join(args.out, "vllm.jsonl"), v.trace)
    meta = dict(model=args.model, engine=engine, vllm=vllm.__version__, ctx=v.ctx, stop_ids=v.stop_ids, eot=v.eot, qwen3=v.qwen3,
                thinking_rule_differs=v.thinking_rule_differs, batches=v.batches, n=len(recs), pick=args.pick,
                ids=[r["id"] for r in recs])
    json.dump(meta, open(os.path.join(args.out, "vllm_meta.json"), "w"), indent=1)


def hf_margin(h, prompt, prefix, a, b):
    torch = h.torch
    x = torch.tensor([prompt + prefix], device=h.device)
    with torch.no_grad():
        try:
            logits = h.model(x, logits_to_keep=1).logits[0, -1].float()
        except TypeError:
            logits = h.model(x).logits[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    return float(lp[a] - lp[b]), int((lp > lp[b]).sum())


def stage_hf(args):
    import hf_responder as HR
    import transformers
    h = HR.HFResponder(args.model, "template", args.dtype, args.device)
    seen, dec = {}, h.tok.decode

    def capture(ids, **kw):
        seen["ids"] = [int(x) for x in ids]
        return dec(ids, **kw)
    h.tok.decode = capture
    out = []
    for v in read(os.path.join(args.out, "vllm.jsonl")):
        h.start({"id": v["rid"]}, None, "template")
        prompt = h.encode(v["history"])
        text, stop = h.reply(v["history"], v["turn"])
        ids = seen.pop("ids")
        j = first_diff(ids, v["ids"])
        margin = rank = None
        if j is not None and j < len(ids) and j < len(v["ids"]):
            margin, rank = hf_margin(h, prompt, ids[:j], ids[j], v["ids"][j])
        out.append(dict(rid=v["rid"], turn=v["turn"], prompt_equal=prompt == v["prompt_ids"], prompt_len=len(prompt),
                        ids=ids, text=text, stop=stop, first_diff=j, margin=margin, vllm_rank_under_hf=rank))
    write(os.path.join(args.out, "hf.jsonl"), out)
    meta = dict(model=args.model, transformers=transformers.__version__, torch=h.torch.__version__, qwen3=h.qwen3,
                stop_ids=h.stop_ids, eot=h.eot, ctx=h.ctx)
    json.dump(meta, open(os.path.join(args.out, "hf_meta.json"), "w"), indent=1)


def verdict(turns):
    if any(not t["prompt_equal"] for t in turns):
        return "FAIL_PROMPT"
    if any(t["ids_equal"] and not t["match"] for t in turns):
        return "FAIL_STOP"
    bad = [t for t in turns if not t["match"]]
    if not bad:
        return "IDENTICAL"
    if all(t["margin"] is not None and t["margin"] <= NEAR_TIE for t in bad):
        return "NEAR_TIE"
    return "DIFFERS"


def compare(vrows, hrows, model="?"):
    hf = {(r["rid"], r["turn"]): r for r in hrows}
    turns = []
    for v in vrows:
        h = hf[(v["rid"], v["turn"])]
        eq = h["ids"] == v["ids"]
        turns.append(dict(rid=v["rid"], turn=v["turn"], prompt_equal=h["prompt_equal"], ids_equal=eq,
                          match=eq and h["stop"] == v["stop"] and h["text"] == v["text"], first_diff=h["first_diff"],
                          margin=h["margin"], vllm_rank_under_hf=h["vllm_rank_under_hf"], stop_hf=h["stop"],
                          stop_vllm=v["stop"], text_hf=h["text"], text_vllm=v["text"]))
    convs = {}
    for t in turns:
        convs[t["rid"]] = convs.get(t["rid"], True) and t["match"]
    diffs = [t for t in turns if not t["match"]]
    return dict(model=model, verdict=verdict(turns), near_tie=NEAR_TIE, turns=len(turns),
                identical=len(turns) - len(diffs), conversations=len(convs), conversations_identical=sum(convs.values()),
                prompt_mismatch=sum(not t["prompt_equal"] for t in turns),
                stop_disagree=sum(t["stop_hf"] != t["stop_vllm"] for t in turns),
                first_diffs=[t["first_diff"] for t in diffs], margins=[t["margin"] for t in diffs], per_turn=turns)


def lines_of(res):
    out = [f"RC-12 parity HF vs {ENGINES[res.get('engine', 'vllm')]}, greedy, template render: {res['model']}",
           f"VERDICT {res['verdict']}  ({res['identical']} / {res['turns']} turns identical; "
           f"{res['conversations_identical']} / {res['conversations']} conversations; prompt mismatches "
           f"{res['prompt_mismatch']}; stop disagreements {res['stop_disagree']}; near-tie = margin <= {NEAR_TIE})"]
    for t in res["per_turn"]:
        head = f"{t['rid']} t{t['turn']:02d} " + ("same" if t["match"] else "DIFF")
        if t["match"]:
            out.append(f"{head} stop {t['stop_hf']}")
            continue
        m = "-" if t["margin"] is None else f"{t['margin']:.3f}"
        out.append(f"{head} first differing token {t['first_diff']} margin {m} rank {t['vllm_rank_under_hf']} "
                   f"stop {t['stop_hf']}/{t['stop_vllm']} prompt_equal {t['prompt_equal']}")
        out.append(f"    HF   {t['text_hf']!r}")
        out.append(f"    vLLM {t['text_vllm']!r}")
    return out


def stage_report(args):
    meta = json.load(open(os.path.join(args.out, "vllm_meta.json")))
    res = compare(read(os.path.join(args.out, "vllm.jsonl")), read(os.path.join(args.out, "hf.jsonl")), meta["model"])
    res["engine"] = meta.get("engine", "vllm")
    for k in ("vllm_meta", "hf_meta"):
        res[k] = json.load(open(os.path.join(args.out, k + ".json")))
    json.dump(res, open(os.path.join(args.out, "parity.json"), "w"), indent=1)
    text = "\n".join(lines_of(res)) + "\n"
    open(os.path.join(args.out, "parity.txt"), "w").write(text)
    print("\n".join(text.splitlines()[:2]))
    return 0


def table(dirs):
    for d in dirs:
        p = os.path.join(d, "parity.json")
        if not os.path.exists(p):
            print(f"{d}: no parity.json")
            continue
        r = json.load(open(p))
        print(f"{r['model']:40s} {r['verdict']:12s} {r['identical']}/{r['turns']} turns, "
              f"{r['conversations_identical']}/{r['conversations']} convs, first diffs {sorted(r['first_diffs'])[:8]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--stage", choices=STAGES, default="all")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pick", choices=["first", "spread"], default="first")
    ap.add_argument("--data", default=R.DEV)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--table", nargs="+", default=None)
    ap.add_argument("--engine", choices=sorted(ENGINES), default="vllm",
                    help="what stage vllm plays with: vllm, or hfb (hf_batched.HFBatched, for a model vLLM rejects)")
    args = ap.parse_args()
    if args.table:
        return table(args.table)
    os.makedirs(args.out, exist_ok=True)
    if args.stage == "all":
        base = [sys.executable, "-B", os.path.abspath(__file__), "--model", args.model, "--out", args.out, "--n",
                str(args.n), "--pick", args.pick, "--data", args.data, "--dtype", args.dtype, "--device", args.device, "--gpu-mem",
                str(args.gpu_mem), "--engine", args.engine] + (["--max-model-len", str(args.max_model_len)] if args.max_model_len else [])
        for st in ("vllm", "hf"):
            cmd = base + ["--stage", st]
            print(f"stage {st}", flush=True)
            if subprocess.run(cmd).returncode:
                sys.exit(f"stage {st} failed")
        return stage_report(args)
    return {"vllm": stage_vllm, "hf": stage_hf, "report": stage_report}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
