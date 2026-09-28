"""RC-12 batched Hugging Face engine (draft s4: "batched HF transformers for Doge, MaxGPT-3 and anything vLLM
rejects"; notes STEP 9). HFBatched IS an HFResponder (same load, prompt, encode, count, reply, stop list, ctx) plus
reply_batch for lockstep.py: the pending turn of every live conversation goes through model.generate in chunks
(Chunking, below), each left-padded with an attention mask.

Per-reply seeds inside one batch: generate runs with do_sample=False and a logits processor (RowSampler). For a
sampled row it draws that row's token from softmax(scores / T) with the row's own torch.Generator, seeded
hf_responder.conv_seed(seed, conversation id, turn), and leaves only that token finite, so argmax takes it; a greedy
row passes through untouched. With equal logits this is the draw HFResponder.reply makes (torch.manual_seed(the same
seed), one multinomial per step on softmax(scores / T)), whatever else shares the batch. Top-p 1.0, top-k off and
repetition penalty 1.0 add no processor on either path.
Reply: the row's new tokens cut after its first stop id (generate pads a finished row), stop reason by HFResponder's
rule (split_reply). Padding can still move the numerics: parity_hf_vllm.py --engine hfb compares this engine with
the serial HFResponder on the same histories (greedy), and test_hf_batched.py (PC, toy model, CPU) checks greedy
and sampled rows against it.
Importing this module imports nothing heavy; torch and transformers load in HFResponder.__init__ only.
trace: set to a list to record every reply in parity_hf_vllm.py's format (finish / stop_reason None).
Chunking (notes STEP 9f): reply_batch sorts the requests by prompt length, longest first, and cuts them into chunks of
at most max_batch rows whose cost rows x width^2 (width = the chunk's longest prompt; left padding gives every row
that width) stays within token_budget, so no chunk's eager-attention prefill (several [rows, heads, width, width]
tensors) outgrows the card; replies come back in request order. Before each chunk it empties CUDA's cache, so blocks
cached for earlier shapes do not pile up: on WSL the driver spills them to shared memory instead of failing (the
pre-9f run reserved 21.9 GiB on the 12 GB card and slowed 27-fold). token_budget None = the pre-9f chunker
(consecutive runs of max_batch, request order)."""
import time

import hf_responder as HR

MAX_BATCH = 64   # engines.json "max_batch" overrides it per model (Doge: 32; 64 rows of 1708-token prompts
                 # under eager attention peaked at 17.9 GiB on the 12 GB card, notes STEP 9c)
TOKEN_BUDGET = 53 * 2 ** 20    # rows x width^2 per chunk = 17.3 rows at 1792 (notes STEP 9f): Doge-160M's measured
                               # peak GiB = 0.327 + 6.907e-8 rows w^2 + 4.314e-5 rows w stays <= 6.0 at every width


def plan_chunks(lengths, max_batch, budget=None):
    """[[request index, ...], ...] covering every index once. budget None: consecutive runs of max_batch in request
    order (the pre-9f chunker). Else longest prompt first (ties in request order), a new chunk whenever one more row
    would pass max_batch rows or rows x width^2 > budget (width = the chunk's first, longest, prompt); a prompt over
    the budget on its own is a chunk of one."""
    if budget is None:
        return [list(range(lo, min(lo + max_batch, len(lengths)))) for lo in range(0, len(lengths), max_batch)]
    chunks, cur = [], []
    for k in sorted(range(len(lengths)), key=lambda k: (-lengths[k], k)):
        if cur and (len(cur) + 1 > max_batch or (len(cur) + 1) * lengths[cur[0]] ** 2 > budget):
            chunks.append(cur)
            cur = []
        cur.append(k)
    if cur:
        chunks.append(cur)
    return chunks


def split_reply(out, stop_ids, eot, cap=HR.DECODE["max_new_tokens"]):
    """(ids, stop) for one row of generate's new tokens: cut after the first stop id, then HFResponder.reply's
    rule (eot = ends on an end-of-turn id; cap = cap tokens and no stop id last; eos otherwise)."""
    out = list(out)
    for j, t in enumerate(out):
        if t in stop_ids:
            out = out[:j + 1]
            break
    if out and out[-1] in eot:
        stop = "eot"
    elif len(out) >= cap and (not out or out[-1] not in stop_ids):
        stop = "cap"
    else:
        stop = "eos"
    return out, stop


def left_pad(torch, enc, pad):
    width = max(len(e) for e in enc)
    ids = torch.full((len(enc), width), pad, dtype=torch.long)
    mask = torch.zeros((len(enc), width), dtype=torch.long)
    for r, e in enumerate(enc):
        if e:
            ids[r, width - len(e):] = torch.tensor(e, dtype=torch.long)
            mask[r, width - len(e):] = 1
    return ids, mask, width


class RowSampler:
    """logits processor: row r samples with its own generator (seed not None) or stays greedy (seed None)."""

    def __init__(self, torch, seeds, temperature, device):
        self.torch, self.temperature = torch, temperature
        self.gens = [None if s is None else torch.Generator(device=device).manual_seed(s) for s in seeds]
        self.steps = 0

    def __call__(self, input_ids, scores):
        torch = self.torch
        self.steps += 1
        for r, g in enumerate(self.gens):
            if g is None:
                continue
            probs = torch.nn.functional.softmax(scores[r:r + 1] / self.temperature, dim=-1)
            tok = int(torch.multinomial(probs, num_samples=1, generator=g)[0, 0])
            scores[r] = float("-inf")
            scores[r, tok] = 0.0
        return scores


class HFBatched(HR.HFResponder):
    def __init__(self, model_id, render="template", dtype="bfloat16", device="cuda", max_batch=MAX_BATCH,
                 trust_remote_code=False, attn_implementation=None, token_budget=TOKEN_BUDGET):
        super().__init__(model_id, render, dtype, device, trust_remote_code, attn_implementation)
        self.max_batch, self.batches, self.trace = max_batch, [], None
        self.token_budget, self.chunks, self.progress = token_budget, [], True
        self.rec = None

    def start(self, rec, seed=None, render=None):
        super().start(rec, seed, render)
        self.rec = rec

    def reply(self, history, i):
        """one request through the batched path (runner.play without --lockstep)."""
        return self.reply_batch([(0, self.rec, self.seed, history, i)])[0]

    def reply_batch(self, reqs):
        self.batches.append(len(reqs))
        if not reqs:
            return []
        cuda = str(self.device).startswith("cuda")
        if cuda:
            self.torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        enc = [self.encode(history) for _, _, _, history, _ in reqs]
        plan = plan_chunks([len(e) for e in enc], self.max_batch, self.token_budget)
        out = [None] * len(reqs)
        for idx in plan:
            if cuda:                               # chunks of equal cost differ slightly in shape, so cached blocks
                self.torch.cuda.empty_cache()      # rarely fit the next one (STEP 9f: reserved 12.5 GiB, spilled)
            self.chunks.append((len(idx), max(len(enc[k]) for k in idx)))
            for k, res in zip(idx, self._chunk([reqs[k] for k in idx], [enc[k] for k in idx])):
                out[k] = res
        if self.trace is not None:
            for (_, rec, seed, history, i), e, (text, stop, got) in zip(reqs, enc, out):
                self.trace.append(dict(rid=rec["id"], turn=i, seed=seed, history=[dict(m) for m in history],
                                       prompt_ids=list(e), ids=got, text=text, stop=stop, finish=None,
                                       stop_reason=None, vllm_text=None))
        if self.progress:
            peak = f", peak {self.torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB" if cuda else ""
            print(f"hfb: {len(reqs)} requests, {len(plan)} chunks, widths {min(len(e) for e in enc)}-"
                  f"{max(len(e) for e in enc)}, {time.time() - t0:.1f} s{peak}", flush=True)
        return [(text, stop) for text, stop, _ in out]

    def _chunk(self, reqs, enc):
        torch = self.torch
        from transformers import LogitsProcessorList
        ids, mask, width = left_pad(torch, enc, self.pad)
        seeds = [None if seed is None else HR.conv_seed(seed, rec["id"], i) for _, rec, seed, _, i in reqs]
        sampler = RowSampler(torch, seeds, HR.DECODE["temperature"], self.device)
        kw = dict(max_new_tokens=HR.DECODE["max_new_tokens"], eos_token_id=self.stop_ids, pad_token_id=self.pad,
                  repetition_penalty=HR.DECODE["repetition_penalty"], do_sample=False, temperature=None, top_p=None,
                  top_k=None, logits_processor=LogitsProcessorList([sampler]))
        with torch.no_grad():
            gen = self.model.generate(ids.to(self.device), attention_mask=mask.to(self.device), **kw)
        res = []
        for row in gen[:, width:].tolist():
            got, stop = split_reply(row, self.stop_ids, self.eot)
            res.append((self.tok.decode(got, skip_special_tokens=True), stop, got))
        return res
