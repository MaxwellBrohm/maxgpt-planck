"""RC-12 Hugging Face causal-LM responder for runner.py (SPEC s2).

UNTESTED. HF_TESTED = False: no model has ever been loaded through this file (it was written while the Mac's GPU
ran another experiment, and baselines generate on the 5070). runner.py refuses to use it without
--hf-untested-ok. Before the first real run: greedy on 20 conversations per family, compare with a vLLM run
(SPEC s2 parity), read the stop reasons and a few transcripts by eye, then set HF_TESTED = True in a logged step.

Importing this module imports nothing heavy; torch and transformers load in HFResponder.__init__ only.
Decoding: greedy (seed None) or sampling with T = 0.6, top-p 1.0, top-k OFF, repetition penalty 1.0, max 256 new
tokens (SPEC s2). Every knob is passed explicitly because a model's generation_config.json can carry its own
defaults (Qwen ships top_p 0.8, top_k 20, repetition_penalty 1.05), which would silently change the protocol.
Seeding: torch.manual_seed(md5(seed, conversation id, turn)) before each sampled reply, so a reply does not depend
on batch order or on how many conversations ran before it.
Render: template = tokenizer.apply_chat_template (no system prompt; enable_thinking=False for Qwen3-family
config.model_type), tokenized without extra special tokens; plain = render.plain(), tokenized WITH the tokenizer's
BOS if it adds one. Stops: eos_token_id plus the end-of-turn tokens below that exist in the vocabulary.
Stop reason: "eot" (an end-of-turn token), "eos" (the eos token, or generation ended early), "cap" (256 tokens)."""
import hashlib

import render as RD

HF_TESTED = False
DECODE = dict(temperature=0.6, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=RD.MAX_NEW_TOKENS)
EOT_TOKENS = ["<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>"]


def conv_seed(seed, rid, turn):
    return int(hashlib.md5(f"{seed}:{rid}:{turn}".encode()).hexdigest()[:8], 16)


class HFResponder:
    def __init__(self, model_id, render="template", dtype="bfloat16", device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device, self.render = torch, device, render
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, dtype))
        self.model.to(device).eval()
        self.has_template = bool(getattr(self.tok, "chat_template", None))
        if render == "template" and not self.has_template:
            raise ValueError(f"{model_id} ships no chat template: use --render plain")
        self.qwen3 = str(getattr(self.model.config, "model_type", "")).startswith("qwen3")
        vocab = self.tok.get_vocab()
        self.eos = self.tok.eos_token_id
        self.eot = sorted({vocab[t] for t in EOT_TOKENS if t in vocab} - {self.eos})
        self.stop_ids = ([self.eos] if self.eos is not None else []) + self.eot
        self.ctx = getattr(self.model.config, "max_position_embeddings", None)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos

    def start(self, rec, seed=None, render=None):
        self.rid, self.seed = rec["id"], seed
        if render and render != self.render:
            raise ValueError("render changed after loading")

    def prompt(self, messages):
        return RD.template(self.tok, messages, self.qwen3) if self.render == "template" else RD.plain(messages)

    def encode(self, messages):
        return self.tok(self.prompt(messages), add_special_tokens=self.render == "plain").input_ids

    def count(self, messages, render=None):
        return len(self.encode(messages))

    def reply(self, history, i):
        torch = self.torch
        ids = torch.tensor([self.encode(history)], device=self.device)
        kw = dict(max_new_tokens=DECODE["max_new_tokens"], eos_token_id=self.stop_ids, pad_token_id=self.pad,
                  repetition_penalty=DECODE["repetition_penalty"])
        if self.seed is None:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        else:
            torch.manual_seed(conv_seed(self.seed, self.rid, i))
            kw.update(do_sample=True, temperature=DECODE["temperature"], top_p=DECODE["top_p"],
                      top_k=DECODE["top_k"])
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), **kw)[0, ids.shape[1]:].tolist()
        if out and out[-1] in self.eot:
            stop = "eot"
        elif len(out) >= DECODE["max_new_tokens"] and (not out or out[-1] not in self.stop_ids):
            stop = "cap"
        else:
            stop = "eos"
        return self.tok.decode(out, skip_special_tokens=True), stop
