"""RC-12 batched Hugging Face engine (draft s4: "batched HF transformers for Doge, MaxGPT-3 and anything vLLM
rejects"; notes STEP 9). HFBatched IS an HFResponder (same load, prompt, encode, count, reply, stop list, ctx) plus
reply_batch for lockstep.py: the pending turn of every live conversation goes through ONE model.generate call
(chunks of max_batch rows), left-padded with an attention mask.

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
trace: set to a list to record every reply in parity_hf_vllm.py's format (finish / stop_reason None)."""
import hf_responder as HR

MAX_BATCH = 64


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
    def __init__(self, model_id, render="template", dtype="bfloat16", device="cuda", max_batch=MAX_BATCH):
        super().__init__(model_id, render, dtype, device)
        self.max_batch, self.batches, self.trace = max_batch, [], None
        self.rec = None

    def start(self, rec, seed=None, render=None):
        super().start(rec, seed, render)
        self.rec = rec

    def reply(self, history, i):
        """one request through the batched path (runner.play without --lockstep)."""
        return self.reply_batch([(0, self.rec, self.seed, history, i)])[0]

    def reply_batch(self, reqs):
        self.batches.append(len(reqs))
        out = []
        for lo in range(0, len(reqs), self.max_batch):
            out += self._chunk(reqs[lo:lo + self.max_batch])
        return out

    def _chunk(self, reqs):
        torch = self.torch
        from transformers import LogitsProcessorList
        enc = [self.encode(history) for _, _, _, history, _ in reqs]
        ids, mask, width = left_pad(torch, enc, self.pad)
        seeds = [None if seed is None else HR.conv_seed(seed, rec["id"], i) for _, rec, seed, _, i in reqs]
        sampler = RowSampler(torch, seeds, HR.DECODE["temperature"], self.device)
        kw = dict(max_new_tokens=HR.DECODE["max_new_tokens"], eos_token_id=self.stop_ids, pad_token_id=self.pad,
                  repetition_penalty=HR.DECODE["repetition_penalty"], do_sample=False, temperature=None, top_p=None,
                  top_k=None, logits_processor=LogitsProcessorList([sampler]))
        with torch.no_grad():
            gen = self.model.generate(ids.to(self.device), attention_mask=mask.to(self.device), **kw)
        rows = gen[:, width:].tolist()
        res = []
        for (_, rec, seed, history, i), e, row in zip(reqs, enc, rows):
            got, stop = split_reply(row, self.stop_ids, self.eot)
            text = self.tok.decode(got, skip_special_tokens=True)
            res.append((text, stop))
            if self.trace is not None:
                self.trace.append(dict(rid=rec["id"], turn=i, seed=seed, history=[dict(m) for m in history],
                                       prompt_ids=list(e), ids=got, text=text, stop=stop, finish=None,
                                       stop_reason=None, vllm_text=None))
        return res
