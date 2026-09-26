"""RC-12 hf_batched.py tests (notes STEP 9).
  split    split_reply: cut after the first stop id; eot / eos / cap by HFResponder.reply's rule (pure Python, any
           machine)
  engine   needs torch + transformers and runs ON THE PC ONLY (refused on macOS: nothing is loaded on the Mac). A random
           2-layer Llama in float32 on the CPU, saved with the tokenizer of --tok (a local HF id) to a temp dir, its
           eos and one end-of-turn row of lm_head scaled so replies stop at varied lengths. 12 requests with prompt
           lengths 1 to 12 turns, greedy and sampled (seeds 1 and 2) mixed, go through HFBatched.reply_batch (batch 12,
           then chunks of 5) and each must equal HFResponder.reply on the same history (text and stop): the batch,
           the padding and the per-row generators change nothing. The stop mix must include eot, eos and cap, and a
           greedy row must differ from a sampled one (the seeds are used).
Run (PC): <venv python> -B test_hf_batched.py --tok HuggingFaceTB/SmolLM2-135M-Instruct   (exit 1 on any failure)
Run (Mac): python3 -B test_hf_batched.py --split-only"""
import argparse
import platform
import sys
import tempfile

import hf_batched as HB
import hf_responder as HR

SCALE_EOS, SCALE_EOT = 1.8, 1.5                  # lm_head row scales: tuned so the 12 serial replies mix eot, eos, cap


def c_split():
    stop, eot, cap = [2, 7], [7], 5
    cases = [([4, 5, 7, 2, 2], ([4, 5, 7], "eot")), ([4, 2, 9, 9, 9], ([4, 2], "eos")),
             ([4, 5, 6, 8, 9], ([4, 5, 6, 8, 9], "cap")), ([4, 5, 6, 8, 2], ([4, 5, 6, 8, 2], "eos")),
             ([7], ([7], "eot")), ([], ([], "eos"))]
    out = []
    for row, want in cases:
        got = HB.split_reply(row, stop, eot, cap)
        if got != want:
            out.append(f"split {row}: {got} != {want}")
    return out


def toy(tok_id, d):
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
    tok = AutoTokenizer.from_pretrained(tok_id)
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, tie_word_embeddings=False)
    m = LlamaForCausalLM(cfg).float().eval()
    eot = [i for i in (tok.get_vocab().get(t) for t in HR.EOT_TOKENS) if i is not None and i != tok.eos_token_id]
    with torch.no_grad():
        m.model.norm.weight.fill_(1.0)
        w = m.lm_head.weight
        w.mul_(40.0)
        for t, c in [(tok.eos_token_id, SCALE_EOS)] + [(t, SCALE_EOT) for t in eot[:1]]:
            w[t] = w[t] * c
    m.save_pretrained(d)
    tok.save_pretrained(d)
    return eot


def reqs_for():
    import runner as R
    recs = R.load(R.DEV, None, 12)
    out = []
    for k, rec in enumerate(recs):
        msgs = []
        for t in sorted(rec["turns"], key=lambda x: x["i"])[:k + 1]:
            msgs += [{"role": "user", "content": t["text"]}, {"role": "assistant", "content": t["ideal"]}]
        out.append((k, rec, [None, 1, 2][k % 3], msgs[:-1], k + 1))
    return out


def c_engine(tok_id):
    out = []
    with tempfile.TemporaryDirectory() as d:
        toy(tok_id, d)
        serial = HR.HFResponder(d, "template", "float32", "cpu")
        want = []
        for _, rec, seed, hist, i in reqs_for():
            serial.start(rec, seed, "template")
            want.append(serial.reply(hist, i))
        for mb in (12, 5):
            b = HB.HFBatched(d, "template", "float32", "cpu", max_batch=mb)
            got = b.reply_batch(reqs_for())
            for k, (g, w) in enumerate(zip(got, want)):
                if g != w:
                    out.append(f"batch {mb} request {k}: {g!r} != serial {w!r}")
            if b.batches != [12]:
                out.append(f"batch sizes {b.batches}")
        stops = {s for _, s in want}
        if not {"eot", "eos", "cap"} <= stops:
            out.append(f"stop mix {sorted(stops)} lacks one of eot, eos, cap (the fixture does not exercise them)")
        lens = sorted(len(t) for t, _ in want)
        print(f"engine: 12 requests, stops {[s for _, s in want]}, reply chars {lens}")
        b = HB.HFBatched(d, "template", "float32", "cpu")
        r = reqs_for()[:1]
        g0 = b.reply_batch([r[0][:2] + (None,) + r[0][3:]])[0]
        g1 = b.reply_batch([r[0][:2] + (1,) + r[0][3:]])[0]
        g2 = b.reply_batch([r[0][:2] + (2,) + r[0][3:]])[0]
        if len({g0, g1, g2}) < 3:
            out.append("greedy and seeds 1, 2 gave equal replies on request 0: the seeds are not used")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--split-only", action="store_true")
    a = ap.parse_args()
    fails = c_split()
    if not a.split_only:
        if platform.system() == "Darwin":
            sys.exit("the engine check builds a model: PC only (use --split-only here)")
        fails += c_engine(a.tok)
    for f in fails:
        print("FAIL", f)
    print("ALL HF_BATCHED CHECKS PASS" if not fails else f"{len(fails)} FAILURES")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
