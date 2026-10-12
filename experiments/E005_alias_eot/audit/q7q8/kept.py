"""Auditor's own rebuild of each seed's KEPT training examples (no model, no torch, no transformers).
Generator: E005 code/train_e005.stream (data only, imported read-only with bytecode off).
Length: my own re-implementation of the trainer's encoder with the `tokenizers` library on the HF snapshot's
tokenizer.json, and the SmolLM2 chat template written out by hand.
usage: python -B kept.py SEED  -> kept_s{SEED}.pkl (kept examples, drawn count, drawn/kept counters)"""
import pickle
import sys
from collections import Counter

sys.dont_write_bytecode = True
CODE = "REPO/experiments/E005_alias_eot/code"
sys.path.insert(0, CODE)
import train_e005 as T5  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

TOKJ = ("HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/"
        "12fd25f77366fa6b3b4b768ec3050bf629380bac/tokenizer.json")
tok = Tokenizer.from_file(TOKJ)
EOT = "<|im_end|>"
EOS_ID = tok.token_to_id(EOT)
SYS = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
MAX_LEN, N_KEPT = 768, 6404
KEYS = ("kind", "block", "render", "case", "placement")


def ids(text, special):
    return tok.encode(text, add_special_tokens=special).ids


def plain_prompt(ex):
    lines = []
    for u, a in ex["turns"]:
        lines += [f"User: {u}", f"Assistant: {a}"]
    lines += [f"User: {ex['question']}", "Assistant:"]
    return "\n".join(lines)


def chat_prompt(ex):
    out = SYS
    for u, a in ex["turns"]:
        out += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
    out += f"<|im_start|>user\n{ex['question']}<|im_end|>\n<|im_start|>assistant\n"
    return out


def length(ex):
    if ex["render"] == "plain":
        p, c = plain_prompt(ex), ex["answer"]
        pre, full = ids(p, True), ids(p + c, True)
        while pre and pre[-1] == EOS_ID and full and full[-1] == EOS_ID:
            pre, full = pre[:-1], full[:-1]
        if full[:len(pre)] == pre and len(full) > len(pre):
            return len(full)
        return len(pre) + len(ids(c, False))
    p = chat_prompt(ex)
    sent = ex["answer"][1:-1]
    pre, full = ids(p, False), ids(p + sent + EOT, False)
    if full[:len(pre)] == pre and len(full) > len(pre):
        return len(full)
    return len(pre) + len(ids(sent, False)) + 1


def main(seed):
    drawn, kept, lens = 0, [], []
    cd, ck = {k: Counter() for k in KEYS}, {k: Counter() for k in KEYS}
    rejected = 0
    for ex in T5.stream(seed):
        drawn += 1
        for k in KEYS:
            cd[k][str(ex.get(k))] += 1
        n = length(ex)
        if n > MAX_LEN:
            rejected += 1
            continue
        for k in KEYS:
            ck[k][str(ex.get(k))] += 1
        kept.append(ex)
        lens.append(n)
        if len(kept) == N_KEPT:
            break
    with open(f"kept_s{seed}.pkl", "wb") as f:
        pickle.dump(dict(seed=seed, drawn=drawn, rejected=rejected, kept=kept, lens=lens, cd=cd, ck=ck), f)
    trained = kept[4:]
    print(f"seed {seed}: drawn {drawn} rejected {rejected} kept {len(kept)} blocks "
          f"{dict(Counter(x['block'] for x in kept))} render kept {dict(Counter(x['render'] for x in kept))} "
          f"trained(4:6404) render {dict(Counter(x['render'] for x in trained))} mean_len {sum(lens)/len(lens):.1f} "
          f"max_len plain {max(l for l, x in zip(lens, kept) if x['render']=='plain')} "
          f"chat {max(l for l, x in zip(lens, kept) if x['render']=='chat')}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
