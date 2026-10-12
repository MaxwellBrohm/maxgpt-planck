"""E004 free generation (notes.txt (c)): greedy, no answer prefix, at most MAX_NEW new tokens.
Plain render: stop at the first newline or at "User:" (the reply is cut there); a newline before any text gives
an empty reply, which fails grader clause 1 (the training target starts on the "Assistant:" line). Chat render
(SmolLM2 only, reported, not ruled): stop at end of turn / EOS only. A reply that reaches MAX_NEW tokens without
a stop is "cap" and fails clause 1. Same KV-cache loop as E002 gen_probe.py. torch is imported lazily so the
stop logic (find_stop) is testable without it; mutation_e004.py drives generate() with a scripted stub model.
"""
import json

import gen_grade as G

MAX_NEW = 48
STOP_TEXT = {"newline": "\n", "user": "User:"}


def find_stop(text, newline_stop=True):
    """-> (reply cut at the earliest stop marker, stop name) or (None, None) when no marker is present."""
    best = None
    for name, mark in STOP_TEXT.items():
        if name == "newline" and not newline_stop:
            continue
        i = text.find(mark)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, name)
    if best is None:
        return None, None
    return text[:best[0]], best[1]


def generate(model, tok, prompt, device, max_new=MAX_NEW, add_special=True, end_ids=(), newline_stop=True):
    """-> (reply, stop, n_new_tokens). stop is "newline" | "user" | "eos" | "cap"."""
    import torch
    ids = tok(prompt, add_special_tokens=add_special, return_tensors="pt").input_ids.to(device)
    end = {i for i in end_ids if i is not None}
    out, past, cur = [], None, ids
    with torch.no_grad():
        for _ in range(max_new):
            o = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
            if nxt in end:
                return tok.decode(out, skip_special_tokens=True).strip(), "eos", len(out)
            out.append(nxt)
            cut, stop = find_stop(tok.decode(out, skip_special_tokens=True), newline_stop)
            if stop:
                return cut.strip(), stop, len(out)
            cur = torch.tensor([[nxt]], device=device)
    return tok.decode(out, skip_special_tokens=True).strip(), "cap", len(out)


def end_ids_of(tok):
    ids = [getattr(tok, "eos_token_id", None)]
    vocab = tok.get_vocab() if hasattr(tok, "get_vocab") else {}
    if "<|im_end|>" in vocab:
        ids.append(vocab["<|im_end|>"])
    return tuple(i for i in ids if i is not None)


def run_gen(model, tok, gen_items, render, device, path, max_new=MAX_NEW, empty_cache=None):
    """gen_items: dicts with prompt (plain, ends "Assistant:") or chat msgs, gold, pool, obj, and report keys.
    Writes one jsonl record per item; returns the records."""
    recs = []
    end = end_ids_of(tok)
    with open(path, "w") as f:
        for i, it in enumerate(gen_items):
            if render == "plain":
                reply, stop, n = generate(model, tok, it["prompt"], device, max_new, True, end, True)
            else:
                prompt = tok.apply_chat_template(it["msgs"], tokenize=False, add_generation_prompt=True)
                reply, stop, n = generate(model, tok, prompt, device, max_new, False, end, False)
            g = G.grade(reply, stop, it["gold"], it["pool"], it["obj"])
            rec = {"set": it["set"], "render": render, "id": i, "reply": reply, "stop": stop, "n_new": n, **g}
            for k in ("family", "cell", "vtype", "k", "d", "var", "fam", "task", "idx", "latest_ref", "h"):
                if k in it:
                    rec[k] = it[k]
            recs.append(rec)
            f.write(json.dumps(rec) + "\n")
            if empty_cache is not None and i % 100 == 0:
                empty_cache()
    return recs
