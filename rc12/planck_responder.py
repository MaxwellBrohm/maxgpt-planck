"""RC-12 responder for MaxGPT-Planck checkpoints (harness/ models with Planck's own role tokens).

runner.py drives it like any responder (start, count, reply): the history it gets is the user turns plus the
model's OWN earlier replies (stop-cut, stripped), fitted by the runner to ctx - 256 tokens, where ctx is the
checkpoint's seq_len (a larger --ctx is refused by runner.py). Importing this module loads no torch; torch and the
harness modules load in PlanckResponder / from_checkpoint only.

Render "template" (the scored protocol for Planck): harness chat_template.ChatTemplate.render, the SAME code and
encode call (tokenizer.encode(text, add_special_tokens=False)) the trainer uses on chat records, then the
<|assistant|> role token: <|user|> u1 <|end|> <|assistant|> a1 <|end|> <|user|> u2 <|end|> <|assistant|>.
No system turn, no BOS (RC-12 SPEC s2; Planck's template has no BOS). Render "plain": the tokenizer on
render.plain(messages) (diagnostic; the runner cuts at role tags).

Decoding (hf_responder.DECODE, SPEC s2): greedy (seed None) or sampling at T 0.6, top-p 1.0, top-k off, repetition
penalty 1.0 (this file implements temperature only and refuses other DECODE values). Each sampled reply draws from
a private CPU torch.Generator seeded with hf_responder.conv_seed(seed, conversation id, turn), so replies do not
depend on run order and the global torch RNG is never touched (the trainer's eval hook relies on that).
KV cache: harness/decode.py (an adapter over the model's own modules, equal to the full forward on every arm,
test_decode.py); cache=False re-runs the full forward per token (the reference path, compared in the tests).
Stops: <|end|> -> "eot" (not in the text); an EOS id (the text EOT of the config's token sources and
<|endoftext|> if the tokenizer has it) -> "eos" (not in the text); a role token (<|user|>, <|assistant|>,
<|system|>, <|tool|>) -> "role", and the token STAYS in the text, so the LEAK rule (grade_loop.MARKER) charges
the model for opening a turn without closing its own; 256 new tokens -> "cap". Other special tokens are decoded
as their markers (skip_special_tokens=False), so they are visible to the LEAK rule too.
Context: new tokens are also capped at seq_len - prompt length. That only binds when the runner could not fit
the history (the current user turn alone leaves < 256 tokens); hitting it is stop "cap" and is counted in
self.short. A prompt that fills seq_len gets an empty reply, stop "cap", without running the model."""
import os
import sys
from contextlib import nullcontext

import hf_responder as HR
import render as RD

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(os.path.dirname(HERE), "harness")
EOS_STRINGS = ("<|endoftext|>",)


def use_harness():
    if HARNESS not in sys.path:
        sys.path.append(HARNESS)


def template_for(dcfg, tok):
    """the chat template the trainer builds from the config's data block (data.build_loader), checked against the
    tokenizer: a role token the tokenizer knows must have the configured id."""
    use_harness()
    from chat_template import END_TOKEN, ROLE_TOKENS, ChatTemplate
    chat = (dcfg or {}).get("chat", {})
    if "role_ids" in chat:
        tmpl = ChatTemplate({r: int(i) for r, i in chat["role_ids"].items()}, int(chat["end_id"]),
                            chat.get("loss", "assistant"), chat.get("tool_role", "tool"))
    else:
        tmpl = ChatTemplate.from_tokenizer(tok, chat.get("loss", "assistant"))
    for role, i in list(tmpl.role_ids.items()) + [("end", tmpl.end_id)]:
        want = tok.token_to_id(END_TOKEN if role == "end" else ROLE_TOKENS[role])
        if want is not None and want != i:
            raise ValueError(f"config gives {role} id {i}, tokenizer has {want}")
    return tmpl


def eos_for(dcfg, tok):
    ids = {int(s["eot_id"]) for s in (dcfg or {}).get("sources", []) if s.get("kind") == "tokens" and "eot_id" in s}
    ids |= {tok.token_to_id(s) for s in EOS_STRINGS if tok.token_to_id(s) is not None}
    return sorted(ids)


class PlanckResponder:
    def __init__(self, model, tok, template, seq_len, eos_ids=(), device="cpu", amp=None, render="template",
                 max_new=RD.MAX_NEW_TOKENS, cache=True):
        import torch
        use_harness()
        from decode import KVDecoder
        self.KVDecoder, self.cache = KVDecoder, cache
        assert (HR.DECODE["top_p"], HR.DECODE["top_k"], HR.DECODE["repetition_penalty"]) == (1.0, 0, 1.0), \
            "planck_responder implements temperature sampling only; DECODE changed"
        assert render in ("template", "plain"), render
        self.torch, self.model, self.tok, self.tmpl = torch, model, tok, template
        self.ctx = self.seq_len = int(seq_len)
        self.device, self.amp, self.render, self.max_new = device, amp, render, int(max_new)
        self.end_id = template.end_id
        self.asst_id = template.role_id("assistant")
        self.role_ids = set(template.role_ids.values())
        self.eos_ids = set(eos_ids) - self.role_ids - {self.end_id}
        self.rid, self.seed = None, None
        self.short = 0          # replies whose new-token cap was cut below max_new by seq_len
        self.log = []           # per reply: rid, turn, seed, prompt_len, n_new, room, stop, new_ids (+ history)
        self.keep_history = False   # tests: also log the messages each reply was generated from

    def start(self, rec, seed=None, render=None):
        self.rid, self.seed = rec["id"], seed
        if render and render != self.render:
            raise ValueError("render changed after loading")

    def encode_text(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode(self, messages):
        if self.render == "plain":
            return self.encode_text(RD.plain(messages))
        rec = {"turns": [{"role": m["role"], "text": m["content"]} for m in messages]}
        ids, _ = self.tmpl.render(rec, self.encode_text)
        return [int(x) for x in ids] + [self.asst_id]

    def count(self, messages, render=None):
        return len(self.encode(messages))

    def pick(self, logits, gen):
        torch = self.torch
        if gen is None:
            return int(torch.argmax(logits))
        probs = torch.softmax(logits / HR.DECODE["temperature"], dim=-1).cpu()
        return int(torch.multinomial(probs, 1, generator=gen))

    def next_logits(self, ids, out, dec):
        """logits for the next token: KV-cached (harness decode.py) or the full forward (cache=False)."""
        with (self.amp() if self.amp else nullcontext()):
            if dec is not None:
                return dec.forward(out[-1:] if out else ids, self.device)[-1].float()
            x = self.torch.tensor([ids + out], dtype=self.torch.long, device=self.device)
            return self.model(x)[0][0, -1].float()

    def generate(self, ids, gen, max_new):
        out, stop = [], "cap"
        dec = self.KVDecoder(self.model) if self.cache else None
        for _ in range(max_new):
            nxt = self.pick(self.next_logits(ids, out, dec), gen)
            if nxt == self.end_id:
                return out, "eot"
            if nxt in self.eos_ids:
                return out, "eos"
            out.append(nxt)
            if nxt in self.role_ids:
                return out, "role"
        return out, stop

    def reply(self, history, i):
        torch = self.torch
        ids = self.encode(history)
        room = self.seq_len - len(ids)
        max_new = max(0, min(self.max_new, room))
        self.short += max_new < self.max_new
        gen = None if self.seed is None else torch.Generator().manual_seed(HR.conv_seed(self.seed, self.rid, i))
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out, stop = self.generate(ids, gen, max_new)
        finally:
            self.model.train(was_training)
        self.log.append(dict(rid=self.rid, turn=i, seed=self.seed, prompt_len=len(ids), n_new=len(out), room=room,
                             stop=stop, new_ids=out, history=[dict(m) for m in history] if self.keep_history else None))
        return self.tok.decode(out, skip_special_tokens=False), stop


def resolve(path, base):
    return path if os.path.isabs(path) or base is None else os.path.normpath(os.path.join(base, path))


def from_checkpoint(ckpt, config=None, tokenizer=None, device="cpu", precision="auto", render="template"):
    """load a harness checkpoint (runio.save_checkpoint payload: model, model_cfg, config). config: the run's
    config.yaml (its data block gives the tokenizer path, relative to it, and the chat ids; its model block must
    match the checkpoint); else the config stored in the checkpoint. tokenizer: an explicit tokenizer.json."""
    use_harness()
    import runio
    from config import PlanckConfig
    from data import load_tokenizer
    from device import amp_factory, resolve_precision
    from model import build_model
    ck = runio.load_checkpoint(ckpt)
    mcfg = PlanckConfig.from_dict(ck["model_cfg"])
    run_cfg, base = ck.get("config") or {}, None
    if config:
        run_cfg, base = runio.load_yaml(config), os.path.dirname(os.path.abspath(config))
        if PlanckConfig.from_dict(run_cfg["model"]).to_dict() != mcfg.to_dict():
            raise ValueError(f"{config} and {ckpt} describe different models")
    dcfg = run_cfg.get("data", {})
    tok_path = tokenizer or dcfg.get("tokenizer")
    if not tok_path:
        raise ValueError("no tokenizer: pass --planck-tokenizer (the run config names none)")
    if not tokenizer and not os.path.isabs(tok_path) and base is None:
        raise ValueError(f"tokenizer path {tok_path!r} is relative: pass --planck-config or --planck-tokenizer")
    tok = load_tokenizer(resolve(tok_path, None if tokenizer else base))
    if tok.get_vocab_size() > mcfg.vocab_size:
        raise ValueError(f"tokenizer has {tok.get_vocab_size()} ids, the model {mcfg.vocab_size}")
    model = build_model(mcfg, "cpu")
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    prec = resolve_precision(precision, device)
    r = PlanckResponder(model, tok, template_for(dcfg, tok), mcfg.seq_len, eos_for(dcfg, tok), device,
                        amp_factory(prec, device), render)
    r.step, r.precision, r.path = ck.get("step"), prec, ckpt
    return r
