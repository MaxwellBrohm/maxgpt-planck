"""RC-12 hf_batched.py tests (notes STEP 9).
  split    split_reply: cut after the first stop id; eot / eos / cap by HFResponder.reply's rule (pure Python, any
           machine)
  chunk    the STEP 9f chunker (pure Python, any machine; no model): plan_chunks on fixed and 400 random cases (every
           request exactly once; at most max_batch rows; rows x width^2 <= budget unless one row; longest first,
           ties in request order; a cut only where one more row breaks a limit; budget None = consecutive runs of
           max_batch), and HFBatched.reply_batch built without a model (fake encode, _chunk and torch): replies and
           the trace in request order, each _chunk call within max_batch and the budget, the default budget is
           TOKEN_BUDGET, CUDA's cache emptied right before every chunk on a cuda device and never on the CPU
  engine   needs torch + transformers and runs ON THE PC ONLY (refused on macOS: nothing is loaded on the Mac). A random
           2-layer Llama in float32 on the CPU, saved with the tokenizer of --tok (a local HF id) to a temp dir, its
           eos and one end-of-turn row of lm_head scaled so replies stop at varied lengths. 12 requests with prompt
           lengths 1 to 12 turns, greedy and sampled (seeds 1 and 2) mixed, go through HFBatched.reply_batch (batch 12,
           then chunks of 5) and each must equal HFResponder.reply on the same history (text and stop): the batch,
           the padding and the per-row generators change nothing. The stop mix must include eot, eos and cap, and a
           greedy row must differ from a sampled one (the seeds are used). Also with token_budget None (the pre-9f
           chunker) and with a budget of 3 rows at the widest prompt (length-sorted chunks of 3 and fewer).
Run (PC): <venv python> -B test_hf_batched.py --tok HuggingFaceTB/SmolLM2-135M-Instruct   (exit 1 on any failure)
Run (Mac): python3 -B test_hf_batched.py --split-only   (split and chunk)"""
import argparse
import platform
import random
import sys
import tempfile
import types

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


def check_plan(lengths, mb, budget):
    """every rule plan_chunks promises, for one case; returns failure strings."""
    plan, out, n = HB.plan_chunks(lengths, mb, budget), [], len(lengths)
    flat = [k for c in plan for k in c]
    if sorted(flat) != list(range(n)) or any(not c for c in plan):
        return [f"plan {lengths} mb {mb} budget {budget}: not every request exactly once: {plan}"]
    if budget is None:
        want = [list(range(lo, min(lo + mb, n))) for lo in range(0, n, mb)]
        return [] if plan == want else [f"budget None {lengths} mb {mb}: {plan} != consecutive runs {want}"]
    if flat != sorted(range(n), key=lambda k: (-lengths[k], k)):
        out.append(f"plan {lengths}: order {flat} is not longest first with ties in request order")
    for j, c in enumerate(plan):
        width = max(lengths[k] for k in c)
        if len(c) > mb:
            out.append(f"plan {lengths} mb {mb}: chunk {c} has {len(c)} rows")
        if len(c) > 1 and len(c) * width ** 2 > budget:
            out.append(f"plan {lengths} budget {budget}: chunk {c} costs {len(c) * width ** 2}")
        if j + 1 < len(plan) and len(c) + 1 <= mb and (len(c) + 1) * width ** 2 <= budget:
            out.append(f"plan {lengths} mb {mb} budget {budget}: chunk {c} cut early (room for {plan[j + 1][0]})")
    return out


class FakeCuda:
    def __init__(self, log):
        self.log = log

    def empty_cache(self):
        self.log.append("empty_cache")

    def reset_peak_memory_stats(self):
        pass

    def max_memory_allocated(self):
        return 0


def fake_engine(mb, budget, device, log):
    """an HFBatched without a model: encode = one id per character of the last message, _chunk = records its call."""
    b = HB.HFBatched.__new__(HB.HFBatched)
    b.max_batch, b.batches, b.trace, b.token_budget, b.chunks, b.progress = mb, [], [], budget, [], False
    b.device, b.torch = device, types.SimpleNamespace(cuda=FakeCuda(log))
    b.encode = lambda history: [7] * len(history[-1]["content"])

    def chunk(reqs, enc):
        log.append(("chunk", [r[0] for r in reqs], [len(e) for e in enc]))
        return [(f"reply-{r[1]['id']}-{r[4]}", "eos", [len(e)]) for r, e in zip(reqs, enc)]
    b._chunk = chunk
    return b


def c_chunk():
    out = []
    import inspect
    if inspect.signature(HB.HFBatched.__init__).parameters["token_budget"].default != HB.TOKEN_BUDGET:
        out.append("HFBatched's default token_budget is not TOKEN_BUDGET")
    fixed = [([5, 9, 9, 2, 7], 2, 10 ** 9, [[1, 2], [4, 0], [3]]), ([5, 9, 9, 2, 7], 8, 2 * 81, [[1, 2], [4, 0, 3]]),
             ([5, 9, 9, 2, 7], 8, 150, [[1], [2], [4, 0, 3]]), ([3] * 12, 5, 10 ** 9, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9],
                                                                                    [10, 11]]),
             ([4, 1, 3], 2, None, [[0, 1], [2]]), ([], 4, 100, [])]
    for lengths, mb, budget, want in fixed:
        got = HB.plan_chunks(lengths, mb, budget)
        if got != want:
            out.append(f"plan {lengths} mb {mb} budget {budget}: {got} != {want}")
    rng = random.Random(9)
    for _ in range(400):
        n, mb = rng.randint(0, 70), rng.randint(1, 40)
        lengths = [rng.randint(1, 1800) for _ in range(n)]
        budget = rng.choice([None, rng.randint(1, 60) * 1800 ** 2, rng.randint(1, 5) * 10 ** 6])
        out += check_plan(lengths, mb, budget)[:1]
    reqs = [(k, {"id": f"c{k}"}, None, [{"role": "user", "content": "x" * n}], 3) for k, n in
            enumerate([40, 5, 90, 90, 12, 60, 7, 33])]
    for mb, budget, device in ((3, 2 * 90 ** 2, "cuda"), (8, None, "cpu"), (4, 10 ** 9, "cuda:0")):
        log = []
        b = fake_engine(mb, budget, device, log)
        try:
            got = b.reply_batch(reqs) + b.reply_batch(reqs[:2])
        except Exception as e:  # noqa: BLE001  (a lost or doubled request shows up here; a failure, not a crash)
            out.append(f"reply_batch mb {mb} budget {budget} raised {type(e).__name__}: {e}")
            continue
        want = [(f"reply-c{k}-3", "eos") for k in range(8)] + [(f"reply-c{k}-3", "eos") for k in range(2)]
        if got != want:
            out.append(f"reply_batch mb {mb} budget {budget}: replies out of request order: {got}")
        if [t["rid"] for t in b.trace] != [f"c{k}" for k in list(range(8)) + [0, 1]] or \
                any(t["ids"] != [len(t["prompt_ids"])] for t in b.trace):
            out.append(f"reply_batch mb {mb} budget {budget}: trace not in request order")
        calls = [x for x in log if x != "empty_cache"]
        if sorted(k for x in calls[:-1] for k in x[1]) != list(range(8)):
            out.append(f"reply_batch mb {mb} budget {budget}: requests not sent exactly once: {calls}")
        for _, ks, lens in calls:
            if len(ks) > mb or (budget is not None and len(ks) > 1 and len(ks) * max(lens) ** 2 > budget):
                out.append(f"reply_batch mb {mb} budget {budget}: _chunk got {ks} widths {lens}")
        if [x[1] for x in calls[:-1]] != HB.plan_chunks([len(r[3][-1]["content"]) for r in reqs], mb, budget):
            out.append(f"reply_batch mb {mb} budget {budget}: chunks are not plan_chunks' plan")
        empties = [j for j, x in enumerate(log) if x == "empty_cache"]
        if device != "cpu" and (len(empties) != len(calls) or any(log[j + 1][0] != "chunk" for j in empties)):
            out.append(f"reply_batch on {device}: CUDA cache not emptied right before every chunk: {log}")
        if device == "cpu" and empties:
            out.append("reply_batch on the CPU called torch.cuda")
        if b.batches != [8, 2] or len(b.chunks) != len(calls) or b.reply_batch([]) != []:
            out.append(f"reply_batch mb {mb}: batches {b.batches} chunks {b.chunks}")
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
        widest = max(len(serial.encode(r[3])) for r in reqs_for())
        for mb, budget in ((12, HB.TOKEN_BUDGET), (5, HB.TOKEN_BUDGET), (5, None), (12, 3 * widest ** 2)):
            b = HB.HFBatched(d, "template", "float32", "cpu", max_batch=mb, token_budget=budget)
            got = b.reply_batch(reqs_for())
            print(f"engine: max_batch {mb} budget {budget}: chunks (rows, width) {b.chunks}")
            for k, (g, w) in enumerate(zip(got, want)):
                if g != w:
                    out.append(f"batch {mb} budget {budget} request {k}: {g!r} != serial {w!r}")
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
    fails = c_split() + c_chunk()
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
