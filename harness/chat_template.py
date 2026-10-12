"""Minimal role-token chat template and per-token loss flags.

PLAN.md (tokenizer, D6): one token per role, `<|system|>`, `<|user|>`, `<|assistant|>`,
`<|end|>`, about 2 tokens of overhead per message. A rendered conversation is

    <|role|> content... <|end|>  <|role|> content... <|end|>  ...

At inference the prompt ends with `<|assistant|>` and generation stops at `<|end|>`, so
in "assistant" loss mode the supervised targets are each assistant turn's content tokens
plus its `<|end|>`; the role token itself is prompt. "all" supervises every token.

Records follow pipeline/SPEC.txt section 11 (accepted/*.jsonl): {"turns": [{"role",
"text", ...}], "system": optional text}. A turn may carry pre-tokenized "ids" instead of
(or besides) "text"; ids win. A turn with "loss": false is never supervised (the open
user-turn loss switch in SPEC.txt section 12 can use it). Tool turns (lookup results)
use the "tool" role id when the template has one; they are never supervised in
"assistant" mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ROLE_TOKENS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>",
               "tool": "<|tool|>"}
END_TOKEN = "<|end|>"


@dataclass
class ChatTemplate:
    role_ids: dict[str, int]          # role name -> token id; "tool" may be absent
    end_id: int
    loss: str = "assistant"           # "assistant" | "all"
    tool_role: str = "tool"           # role whose token marks tool turns (e.g. "user")
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        assert self.loss in ("assistant", "all"), self.loss
        for r in ("user", "assistant"):
            assert r in self.role_ids, f"template needs a {r} role id"

    @classmethod
    def from_tokenizer(cls, tok, loss: str = "assistant") -> "ChatTemplate":
        """Look the role tokens up by string in a HF `tokenizers.Tokenizer`."""
        ids = {r: tok.token_to_id(s) for r, s in ROLE_TOKENS.items()}
        ids = {r: i for r, i in ids.items() if i is not None}
        end = tok.token_to_id(END_TOKEN)
        assert end is not None, f"tokenizer has no {END_TOKEN}"
        return cls(ids, end, loss, "tool" if "tool" in ids else "user")

    def role_id(self, role: str) -> int:
        if role == "tool" and "tool" not in self.role_ids:
            role = self.tool_role
        assert role in self.role_ids, f"no token for role {role!r}"
        return self.role_ids[role]

    def render(self, record: dict, encode=None) -> tuple[np.ndarray, np.ndarray]:
        """-> (ids int64, flags bool). flags[i] True = token i is a supervised TARGET
        (predicted from the tokens before it). flags[0] is always False."""
        turns = list(record["turns"])
        if record.get("system"):
            turns = [{"role": "system", "text": record["system"], "loss": False}] + turns
        ids: list[int] = []
        flags: list[bool] = []
        for t in turns:
            role = t["role"]
            content = t.get("ids")
            if content is None:
                assert encode is not None, "turn has text but no tokenizer was given"
                content = encode(t.get("text", ""))
            content = [int(c) for c in content]
            sup_turn = t.get("loss", True) and (self.loss == "all" or role == "assistant")
            ids.append(self.role_id(role))
            flags.append(self.loss == "all" and t.get("loss", True))
            ids.extend(content)
            flags.extend([sup_turn] * len(content))
            ids.append(self.end_id)
            flags.append(sup_turn)
        if flags:
            flags[0] = False
        return np.asarray(ids, dtype=np.int64), np.asarray(flags, dtype=bool)
