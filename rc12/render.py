"""RC-12 renders, stop rule and history fitting (SPEC s1, s2). Pure Python; the tokenizer path is used only by the
HF responder (hf_responder.py, UNTESTED) and never imports torch here.

messages: [{"role": "user"|"assistant", "content": str}, ...], always ending with the current user turn.
  plain(messages)          "User: <u1>\nAssistant: <a1>\nUser: <u2>\nAssistant:" (generation starts after the
                           final "Assistant:"; a leading space in the generation is stripped by the runner)
  cut_plain(raw)           the plain-render stop rule: cut at the first newline followed by a role tag (User:,
                           Assistant:, Human:, case-insensitive); returns (text, cut?)
  template(tok, messages)  tokenizer.apply_chat_template(messages, add_generation_prompt=True), no system prompt,
                           enable_thinking=False for Qwen3-family templates (SPEC s2)
  template_stub(messages)  a ChatML-shaped string for fakes (token counting and logs only; no tokenizer)
  proxy_tokens(messages)   SPEC s1 proxy: 1.35 x words + 4 per message (until a Planck tokenizer exists)
  fit(messages, budget, count)  drop the oldest whole user/assistant pairs while count(kept) > budget; returns
                           (kept, n_dropped_pairs). User turns 1..n_dropped are then missing from the history."""
import re

import common as C

MAX_NEW_TOKENS = 256
ROLE_CUT = re.compile(r"\n[ \t]*(?:user|assistant|human)[ \t]*:", re.I)
TAG = {"user": "User", "assistant": "Assistant"}


def plain(messages):
    lines = [f"{TAG[m['role']]}: {m['content']}" for m in messages]
    return "\n".join(lines) + "\nAssistant:"


def cut_plain(raw):
    m = ROLE_CUT.search(raw or "")
    if m is None:
        return raw or "", False
    return raw[:m.start()], True


def template_stub(messages):
    body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return body + "<|im_start|>assistant\n"


def template(tokenizer, messages, qwen3=False):
    kw = dict(add_generation_prompt=True, tokenize=False)
    if qwen3:
        kw["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kw)


def proxy_tokens(messages):
    words = sum(len(m["content"].split()) for m in messages)
    return C.TOKENS_PER_WORD * words + C.TOKENS_PER_MSG * len(messages)


def fit(messages, budget, count):
    """keep the newest history that fits budget tokens; the current user turn is always kept."""
    if budget is None:
        return list(messages), 0
    kept, dropped = list(messages), 0
    while len(kept) > 1 and count(kept) > budget:
        kept = kept[2:]
        dropped += 1
    return kept, dropped


def first_turn(history, i):
    """user-turn number of history[0], given that history ends with user turn i."""
    return i - (len(history) - 1) // 2


def own_reply(history, i, q):
    """the assistant's own reply to user turn q as it sits in the history for turn i (None if dropped)."""
    k = q - first_turn(history, i)
    if k < 0 or 2 * k + 1 >= len(history) - 1:
        return None
    m = history[2 * k + 1]
    return m["content"] if m["role"] == "assistant" else None
