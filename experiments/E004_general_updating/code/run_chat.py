"""Run the multi-turn battery against ONE model in ONE process, then exit (E001 copy of
research/probe/run_model.py: hardened graders, truncation-aware grading, E001 output paths,
LFM2.5-230M sampling settings, low_cpu_mem_usage). Run only through guard.py.

usage: python run_model.py <hf_model_id> --mode greedy|sampled [--seeds 0 1] [--only ID,ID]

Writes one jsonl record per conversation (per seed) to transcripts/<slug>__<mode>.jsonl.
Memory rule for this Mac: never run two of these at once.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, math, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery as B

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)

# Card-recommended sampling (source: each model card / generation_config.json on huggingface.co)
SAMPLING = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": dict(temperature=0.2, top_p=0.9),                     # card example
    "HuggingFaceTB/SmolLM2-360M-Instruct": dict(temperature=0.2, top_p=0.9),                     # card example
    "unsloth/gemma-3-270m-it": dict(temperature=1.0, top_k=64, top_p=0.95),                      # generation_config.json
    "LiquidAI/LFM2-350M": dict(temperature=0.3, min_p=0.15, repetition_penalty=1.05),            # card
    "LiquidAI/LFM2.5-350M": dict(temperature=0.1, top_k=50, repetition_penalty=1.05),            # card
    "Qwen/Qwen2.5-0.5B-Instruct": dict(temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1),  # generation_config.json
    "Qwen/Qwen3-0.6B": dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0),                    # card, non-thinking mode
    "tiiuae/Falcon-H1-Tiny-90M-Instruct": dict(temperature=0.7, top_p=0.9),                      # no card recommendation (assumed)
    "LiquidAI/LFM2.5-230M": dict(temperature=0.1, top_k=50, repetition_penalty=1.05),            # card + generation_config.json
}

TEMPLATE_MARKERS = re.compile(r"<\|im_start\|>|<\|im_end\|>|<start_of_turn>|<end_of_turn>|<\|startoftext\|>|<\|endoftext\|>|<\|end_of_text\|>|<\|begin_of_text\|>|<\|user\|>|<\|assistant\|>|<bos>|<eos>")
USER_TURN = re.compile(r"(<\|im_start\|>\s*user|<start_of_turn>\s*user|(^|\n)\s*(user|human|### user|### human)\s*[:\n])", re.I)


def slug(m):
    return m.replace("/", "__")


def build_prompt(tok, model_id, messages, add_gen=True):
    kw = {"enable_thinking": False} if "Qwen3" in model_id else {}
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen, **kw)


@torch.no_grad()
def seq_logprob(model, tok, prompt, cont, device):
    a = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    ab = tok(prompt + cont, add_special_tokens=False, return_tensors="pt").input_ids
    n = ab.shape[1] - a.shape[1]
    if n <= 0 or not torch.equal(ab[0, : a.shape[1]], a[0]):
        return None, None
    logits = model(ab.to(device)).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    tgt = ab[0, a.shape[1]:].to(device)
    pos = torch.arange(a.shape[1] - 1, ab.shape[1] - 1, device=device)
    toks = lp[pos, tgt]
    ranks = [int((lp[p] > lp[p, t]).sum().item()) + 1 for p, t in zip(pos.tolist(), tgt.tolist())]
    return float(toks.sum().item()), ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mode", default="greedy", choices=["greedy", "sampled"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--only", default="")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True)
    model.to(args.device).eval()
    nparams = sum(p.numel() for p in model.parameters())
    gc0 = model.generation_config
    eos = gc0.eos_token_id
    eos_list = eos if isinstance(eos, list) else [eos]
    pad = gc0.pad_token_id if gc0.pad_token_id is not None else eos_list[0]
    print(f"loaded {args.model} params={nparams/1e6:.1f}M device={args.device} eos={eos_list} in {time.time()-t0:.1f}s", flush=True)

    tests = B.TESTS
    if args.only:
        keep = set(args.only.split(","))
        tests = [t for t in tests if t["id"] in keep]
    os.makedirs(os.path.join(EXP, "transcripts"), exist_ok=True)
    out = args.out or os.path.join(EXP, "transcripts", f"{slug(args.model)}__{args.mode}.jsonl")
    fout = open(out, "w")

    for seed in (args.seeds if args.mode == "sampled" else [None]):
        for ti, t in enumerate(tests):
            if seed is not None:
                torch.manual_seed(seed * 1000 + ti)
            msgs = [{"role": "system", "content": t["system"]}] if t["system"] else []
            replies, turns = [], []
            for i, u in enumerate(t["turns"]):
                msgs.append({"role": "user", "content": u})
                prompt = build_prompt(tok, args.model, msgs)
                ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(args.device)
                is_distractor = u in B.D
                max_new = t["max_new"].get(i, 120 if is_distractor else 160)
                if args.mode == "greedy":
                    gcfg = GenerationConfig(do_sample=False, max_new_tokens=max_new, eos_token_id=eos_list,
                                            pad_token_id=pad, repetition_penalty=1.0)
                else:
                    gcfg = GenerationConfig(do_sample=True, max_new_tokens=max_new, eos_token_id=eos_list,
                                            pad_token_id=pad, **SAMPLING[args.model])
                try:
                    g = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=gcfg)
                except RuntimeError as e:  # MPS allocator cap hit: finish this model on CPU
                    if args.device != "mps":
                        raise
                    print(f"MPS error ({str(e)[:80]}); switching to CPU", flush=True)
                    torch.mps.empty_cache()
                    args.device = "cpu"
                    model.to("cpu")
                    ids = ids.to("cpu")
                    g = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=gcfg)
                new = g[0, ids.shape[1]:].tolist()
                stopped = len(new) > 0 and new[-1] in eos_list
                body = new[:-1] if stopped else new
                raw = tok.decode(body, skip_special_tokens=False)
                text = tok.decode(body, skip_special_tokens=True).strip()
                flags = {
                    "hit_max": (not stopped) and len(new) >= max_new,
                    "leaked_template": bool(TEMPLATE_MARKERS.search(raw)),
                    "invented_user_turn": bool(USER_TURN.search(raw)),
                    "empty": len(B.words(text)) == 0,
                    "no_memory_claim": bool(B.NO_MEMORY.search(text)),
                }
                turns.append(dict(user=u, assistant=text, raw=raw, n_prompt_tokens=int(ids.shape[1]),
                                  n_gen_tokens=len(new), stopped_eos=stopped, flags=flags))
                replies.append(text)
                msgs.append({"role": "assistant", "content": text})

            checks, lenient, trunc = {}, {}, []
            for c in t["checks"]:
                r, tr = B.grade(c, replies, hit_max=turns[c["turn"]]["flags"]["hit_max"])
                checks[c["name"]] = r
                if tr:
                    trunc.append(c["name"])
                if "gold" in c:  # gold mentioned at all, ignoring the deflection guard
                    lenient[c["name"]] = B.mentions(c["gold"], replies[c["turn"]])

            forced = []
            if t["forced"]:
                ctx = msgs[:-1]  # up to and including the final user turn
                base = build_prompt(tok, args.model, ctx)
                for prefix, gold, foil in t["forced"]:
                    lg, rg = seq_logprob(model, tok, base + prefix, gold, args.device)
                    lf, rf = seq_logprob(model, tok, base + prefix, foil, args.device)
                    # greedy_gold: every gold token is the argmax, i.e. greedy decoding from
                    # the forced prefix would emit the gold answer verbatim.
                    forced.append(dict(prefix=prefix, gold=gold, foil=foil, lp_gold=lg, lp_foil=lf,
                                       ranks_gold=rg, ranks_foil=rf,
                                       greedy_gold=(None if rg is None else all(x == 1 for x in rg)),
                                       margin=(None if lg is None or lf is None else lg - lf)))

            rec = dict(model=args.model, params_m=round(nparams / 1e6, 1), mode=args.mode, seed=seed,
                       gen=("greedy, repetition_penalty=1.0" if args.mode == "greedy" else SAMPLING[args.model]),
                       id=t["id"], cat=t["cat"], distance=t["distance"], control_for=t["control_for"],
                       system=t["system"], turns=turns, checks=checks, checks_lenient=lenient, forced=forced,
                       ungraded_truncated=trunc, device=args.device, dtype=args.dtype)
            if args.device == "mps":
                torch.mps.empty_cache()  # do not let the MPS allocator hoard memory between conversations
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            ok = [k for k, v in checks.items() if v]
            print(f"[{time.time()-t0:6.0f}s] seed={seed} {t['id']:<26} {len(ok)}/{len(checks)} "
                  f"tok={turns[-1]['n_prompt_tokens']}", flush=True)
    fout.close()
    print(f"done {args.model} in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
