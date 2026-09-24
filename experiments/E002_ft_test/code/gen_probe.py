"""POST-HOC generation check (added 2026-09-24 after seed 0; not part of the pre-registered pass rule).
The pass rule is a likelihood read-out with a forced answer prefix. This asks whether the fine-tuned
model also SAYS the right value when it answers freely: greedy decoding, no answer prefix, on E001
items at d10 (same_k1, twoslot, noupd) and the crossed items (cross_A, cross_B), in two renders:
  plain : "User: ... / Assistant:" transcript (the training format), generation stops at a newline
  chat  : the model's own chat template, generation stops at end of turn
Grader (strict): right iff the reply mentions the gold value, mentions NO other candidate value, and
the gold is not negated (no "not", "n't", "no longer", "never", "instead of" in the 4 words before it).
Lenient: the first candidate value mentioned is the gold. Empty or wrong replies fail both.
usage: gen_probe.py <model_id> <weights_dir|hf> --tag TAG [--max-new 40]   (run only through guard.py)
Writes ../out/<slug>__<tag>__gen__<render>.jsonl
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NEG = re.compile(r"\b(not|no longer|never|instead of|rather than|isn't|wasn't|aren't|won't)\b|n't\b", re.I)


def mentions(text, val):
    return [m.start() for m in re.finditer(r"(?<![A-Za-z])" + re.escape(val) + r"(?![A-Za-z])", text, re.I)]


def grade(reply, cands):
    """cands: {label: ' value'} with 'gold'. Returns (strict, lenient)."""
    vals = {lab: v.strip() for lab, v in cands.items()}
    if not reply or not reply.strip() or "gold" not in vals:
        return False, False
    pos = {lab: mentions(reply, v) for lab, v in vals.items()}
    first = min(((p[0], lab) for lab, p in pos.items() if p), default=(None, None))[1]
    lenient = first == "gold"
    if not pos["gold"] or any(pos[lab] for lab in vals if lab != "gold"):
        return False, lenient
    for p in pos["gold"]:
        before = " ".join(reply[:p].split()[-4:])
        if NEG.search(before):
            return False, lenient
    return True, lenient


def self_test():
    c = {"gold": " Tuesday", "orig": " Monday"}
    cases = [("Your dentist appointment is on Tuesday.", True, True), ("", False, False), ("   ", False, False),
             ("Your dentist appointment is on Monday.", False, False), ("It moved from Monday to Tuesday.", False, False),
             ("Tuesday, not Monday.", False, True), ("It's not on Tuesday.", False, True),
             ("It isn't Tuesday, it's Monday.", False, True), ("I don't know.", False, False),
             ("tuesday", True, True), ("Tuesdays are busy.", False, False)]
    bad = [(r, s, l, grade(r, c)) for r, s, l in cases if grade(r, c) != (s, l)]
    return bad


def items():
    import items_new as N
    import eval_extra as X
    its = [x for x in N.build() if x["d"] == 10 and x["var"] in ("same_k1", "twoslot", "noupd")]
    its += [x for x in X.build_crossed() if x["d"] == 10]
    return its


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import items as I
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("weights")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.device == "mps" and (os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") != "0.7" or
                              os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO") != "0.6"):
        sys.exit("refusing: MPS watermarks must be 0.7/0.6; run through guard.py")
    bad = self_test()
    if bad:
        sys.exit(f"grader self-test failed: {bad}")
    src = a.model if a.weights == "hf" else a.weights
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32, low_cpu_mem_usage=True).to(a.device).eval()
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    nl_ids = {i for t, i in tok.get_vocab().items() if "\n" in tok.convert_tokens_to_string([t])}
    its = items()[: a.limit] if a.limit else items()
    stem = os.path.join(os.path.dirname(HERE), "out", f"{a.model.replace('/', '__')}__{a.tag}__gen")
    t0 = time.time()
    for render in ("plain", "chat"):
        with open(f"{stem}__{render}.jsonl", "w") as f:
            n_s = n_l = 0
            for i, it in enumerate(its):
                if render == "plain":
                    prompt = I.transcript(it["turns"], it["question"], "").rstrip()  # ends with "Assistant:"
                    ids = tok(prompt, return_tensors="pt").input_ids.to(a.device)
                else:
                    msgs = []
                    for u, r in it["turns"]:
                        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": r}]
                    msgs.append({"role": "user", "content": it["question"]})
                    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(a.device)
                out_ids = []
                with torch.no_grad():
                    past, cur = None, ids
                    for _ in range(a.max_new):
                        o = model(input_ids=cur, past_key_values=past, use_cache=True)
                        past = o.past_key_values
                        nxt = int(o.logits[0, -1].argmax())
                        if nxt == im_end or nxt == tok.eos_token_id:
                            break
                        if render == "plain" and nxt in nl_ids and out_ids:
                            break
                        out_ids.append(nxt)
                        cur = torch.tensor([[nxt]], device=a.device)
                reply = tok.decode(out_ids, skip_special_tokens=True).strip()
                s, l = grade(reply, it["cands"])
                n_s += s; n_l += l
                f.write(json.dumps({"render": render, "id": i, "var": it["var"], "fam": it["fam"], "d": it["d"], "sid": it["sid"],
                                    "cands": it["cands"], "reply": reply, "strict": s, "lenient": l}) + "\n")
                if a.device == "mps" and i % 100 == 0:
                    torch.mps.empty_cache()
            print(f"{render}: strict {n_s}/{len(its)} lenient {n_l}/{len(its)} {time.time()-t0:.0f}s", flush=True)
    print("DONE", a.model, a.tag, flush=True)


if __name__ == "__main__":
    main()
