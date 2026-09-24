"""Where does the size gap live? Per-token loss on real multi-turn chat, split by
whether the target token could have been copied from earlier turns.

Data: OASST1 validation (English), one root-to-leaf thread per tree following the
top-ranked reply, >= 4 messages, rendered as a plain "User:/Assistant:" transcript,
truncated to 1024 tokens. Scored tokens: assistant messages after the first exchange.

Classes for each scored token t_i (tokenizer-specific, so compare within a family):
  hist_bigram : the bigram (t_{i-1}, t_i) occurs in an EARLIER message (induction-copyable)
  hist_token  : t_i occurs in an earlier message, but not as that bigram
  novel       : t_i does not occur anywhere earlier in the conversation (incl. this reply)
  self_only   : t_i occurs earlier only inside the current reply

usage: python copysplit.py <hf_model_id> [--dtype fp32|bf16] [--n 250]
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def threads(n):
    # extracted from the OASST1 validation parquet (English), up to 3 root-to-leaf paths
    # per tree with >= 4 messages; see oasst_threads.json
    th = json.load(open(os.path.join(HERE, "oasst_threads.json")))
    return th[:n]


@torch.no_grad()
def run(model_id, dtype, n, device="mps"):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    sums = {}
    def add(key, v):
        a = sums.setdefault(key, [0.0, 0]); a[0] += v; a[1] += 1
    vocab_str = {}
    def is_content(t):
        if t not in vocab_str:
            s_ = tok.decode([t]).strip()
            vocab_str[t] = len(s_) >= 4 and s_.isalpha()
        return vocab_str[t]
    probe = tok(" ", add_special_tokens=True).input_ids
    bos = [tok.bos_token_id] if (tok.bos_token_id is not None and probe and probe[0] == tok.bos_token_id) else []
    for th in threads(n):
        ids, msg_of = list(bos), [-1] * len(bos)
        for j, (role, text) in enumerate(th):
            pre = ("User: " if role == "prompter" else "Assistant: ")
            piece = tok(("\n" if j else "") + pre, add_special_tokens=False).input_ids
            ids += piece; msg_of += [-1] * len(piece)
            body = tok(text, add_special_tokens=False).input_ids
            ids += body; msg_of += [j] * len(body)
        ids, msg_of = ids[:1024], msg_of[:1024]
        x = torch.tensor([ids], device=device)
        lp = torch.log_softmax(model(x).logits[0].float(), -1)
        roles = [r for r, _ in th]
        for j in range(2, len(th)):
            if roles[j] != "assistant":
                continue
            pos = [i for i in range(1, len(ids)) if msg_of[i] == j]
            if not pos:
                continue
            earlier = [k for k in range(len(ids)) if msg_of[k] != -1 and msg_of[k] < j]
            earlier_tok = {ids[k] for k in earlier}
            earlier_big = {(ids[k - 1], ids[k]) for k in earlier if k > 0}
            seen_self = set()
            for i in pos:
                t, p = ids[i], ids[i - 1]
                if (p, t) in earlier_big:
                    c = "hist_bigram"
                elif t in earlier_tok:
                    c = "hist_token"
                elif t in seen_self:
                    c = "self_only"
                else:
                    c = "novel"
                seen_self.add(t)
                nll = -float(lp[i - 1, t].item())
                ct = "content" if is_content(t) else "other"
                for key in (c, "all", c + "/" + ct, "all/" + ct):
                    add(key, nll)
    return {k: {"nll": round(v[0] / max(v[1], 1), 4), "n": v[1]} for k, v in sums.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    res = run(a.model, dt, a.n, a.device)
    json.dump(res, open(os.path.join(HERE, "out", "copysplit__" + a.model.replace("/", "__") + ".json"), "w"), indent=1)
    print(a.model, json.dumps(res))
