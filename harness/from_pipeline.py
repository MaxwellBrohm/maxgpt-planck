"""Pipeline -> harness adapter: accepted pipeline records (pipeline/SPEC.txt section 11) become harness
chat records that sources.ChatJsonlSource and chat_template.ChatTemplate read unchanged.

  python from_pipeline.py OUT_DIR INPUT... [--allow-nontrainable] [--tokenizer tok.json]
                          [--shard-size 5000] [--max-len N]

INPUT is a pipeline run directory (its accepted/ shards are read), an accepted/ directory, or one
accepted-NNNNN.jsonl file. Files are read in sorted order, lines in file order, so output is
deterministic. OUT_DIR must not exist or be empty; the run is written to OUT_DIR.partial and renamed.

Per record:
  refused unless trainable is true and blocked is empty (NOT_TRAINABLE); --allow-nontrainable admits it
    and stamps adapter.admitted_nontrainable. Every record the FAKE pipeline writes today is refused.
  refused if any text holds a role-token string such as <|end|> (ROLE_TOKEN_IN_TEXT): a tokenizer with
    added specials turns it into the real control token. Also BAD_JSON, NO_ID, NO_TURNS, BAD_TURN,
    BAD_MASK (mask 1 on a non-assistant turn), SPECIAL_ID_IN_TEXT, EMPTY_IDS (token mode).
  turns: the pipeline system text becomes turn 0 {"role": "system", "loss": false} (so a token-mode
    record renders with no tokenizer), then every pipeline turn with src_i (its pipeline index; events,
    gold notes and spans use it), role, text, author, mask and spans kept, and "loss":
      assistant: true unless mask 1 (S3 assistant_err, SPEC 2: no loss)
      user:      true, so the harness chat.loss setting decides ("assistant" = no user loss; "all" =
                 the user-turn loss switch L8)
      tool:      false (program-written lookup results are never a target, in either loss mode)
    With --tokenizer each turn also carries "ids" (the template uses ids when present): token shards.
  pipeline: every other field of the input record verbatim (ids, provenance, trainable, blocked, rc12,
    heldout_gate, check, events, golds, gold notes, slots, system, ...).
  source: {file, line, sha256 of the line}. adapter: {version, admitted_nontrainable, tokenizer_sha256,
    n_tokens}.
OUT_DIR gets chat-00000.jsonl ... (rotating at --shard-size) and manifest.json (inputs with sha256,
counts, refusals by code with examples, tokenizer, a suggested harness data block). Exit 0 when at least
one record was admitted, 3 when none was (no shard is written then, only the manifest).
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys

from chat_template import ChatTemplate

VERSION = "from_pipeline/1"
PIPE_ROLES = ("user", "assistant", "tool")
SPECIAL_RE = re.compile(r"<\|[^|<>\s]{1,40}\|>")
N_EXAMPLES = 20


class Refused(ValueError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_files(inputs: list[str]) -> list[str]:
    out: list[str] = []
    for p in inputs:
        if os.path.isdir(os.path.join(p, "accepted")):
            p = os.path.join(p, "accepted")
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "accepted-[0-9][0-9][0-9][0-9][0-9].jsonl")))
            if not found:
                raise SystemExit(f"{p}: no accepted-NNNNN.jsonl shards")
            out.extend(found)
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise SystemExit(f"{p}: not found")
    out = [os.path.abspath(x) for x in out]
    if len(out) != len(set(out)):
        raise SystemExit("an input shard is listed twice")
    return out


def _text(text, where: str) -> str:
    if not isinstance(text, str) or not text:
        raise Refused("BAD_TURN", f"{where}: text must be a non-empty string")
    m = SPECIAL_RE.search(text)
    if m:
        raise Refused("ROLE_TOKEN_IN_TEXT", f"{where}: {m.group(0)}")
    return text


def _ids(text: str, encode, special_ids, where: str) -> list[int]:
    ids = [int(i) for i in encode(text)]
    if not ids:
        raise Refused("EMPTY_IDS", where)
    bad = sorted(set(ids) & special_ids)
    if bad:
        raise Refused("SPECIAL_ID_IN_TEXT", f"{where}: ids {bad}")
    return ids


def harness_turns(rec: dict, encode=None, special_ids=frozenset()) -> list[dict]:
    turns = rec.get("turns")
    if not isinstance(turns, list) or not turns:
        raise Refused("NO_TURNS")
    out = []
    system = rec.get("system")
    if system is not None and system != "":
        out.append({"role": "system", "text": _text(system, "system"), "loss": False, "src_i": None,
                    "author": "pipeline:system"})
    for i, t in enumerate(turns):
        if not isinstance(t, dict) or t.get("role") not in PIPE_ROLES:
            raise Refused("BAD_TURN", f"turn {i}: role {t.get('role') if isinstance(t, dict) else t!r}")
        role, mask = t["role"], t.get("mask", 0)
        if mask not in (0, 1) or isinstance(mask, bool):
            raise Refused("BAD_TURN", f"turn {i}: mask {mask!r}")
        if mask == 1 and role != "assistant":
            raise Refused("BAD_MASK", f"turn {i}: mask 1 on a {role} turn")
        loss = role == "user" or (role == "assistant" and mask == 0)
        out.append({"role": role, "text": _text(t.get("text"), f"turn {i}"), "loss": loss, "src_i": i,
                    "author": t.get("author"), "mask": mask, "spans": t.get("spans", [])})
    if encode is not None:
        for t in out:
            t["ids"] = _ids(t["text"], encode, special_ids, "system" if t["src_i"] is None else f"turn {t['src_i']}")
    return out


def convert(rec, *, allow_nontrainable: bool, encode=None, special_ids=frozenset(),
            source=None, tokenizer_sha256=None) -> dict:
    """One pipeline accepted record -> one harness chat record; raises Refused."""
    if not isinstance(rec, dict):
        raise Refused("BAD_JSON", "record is not an object")
    trainable = rec.get("trainable") is True and not rec.get("blocked")
    if not trainable and not allow_nontrainable:
        raise Refused("NOT_TRAINABLE", f"trainable={rec.get('trainable')!r} blocked={rec.get('blocked')!r}")
    if not isinstance(rec.get("conv_id"), str) or not rec["conv_id"]:
        raise Refused("NO_ID", "conv_id missing")
    turns = harness_turns(rec, encode, special_ids)
    return {"id": rec["conv_id"], "turns": turns,
            "pipeline": {k: v for k, v in rec.items() if k != "turns"}, "source": source,
            "adapter": {"version": VERSION, "admitted_nontrainable": not trainable,
                        "tokenizer_sha256": tokenizer_sha256, "n_tokens": None}}


class ShardWriter:
    """chat-NNNNN.jsonl, rotating every `size` records; a file is only created by its first record."""

    def __init__(self, d: str, size: int):
        self.d, self.size, self.n, self.f, self.paths = d, size, 0, None, []

    def write(self, rec: dict) -> None:
        if self.f is None or self.n % self.size == 0 and self.n:
            if self.f:
                self.f.close()
            self.paths.append(os.path.join(self.d, f"chat-{len(self.paths):05d}.jsonl"))
            self.f = open(self.paths[-1], "w", encoding="utf-8")
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n += 1

    def close(self) -> None:
        if self.f:
            self.f.close()


def _tokenizer(path: str):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    tmpl = ChatTemplate.from_tokenizer(tok)
    info = {"path": os.path.abspath(path), "sha256": sha256_file(path), "vocab": tok.get_vocab_size(),
            "role_ids": tmpl.role_ids, "end_id": tmpl.end_id, "pad_id": tok.token_to_id("<|pad|>")}
    special = frozenset(int(i) for i in tok.get_added_tokens_decoder())
    return (lambda s: tok.encode(s, add_special_tokens=False).ids), special, tmpl, info


def run(inputs: list[str], out: str, *, allow_nontrainable: bool = False, tokenizer: str | None = None,
        shard_size: int = 5000, max_len: int | None = None) -> dict:
    files = input_files(inputs)
    out = os.path.abspath(out)
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{out} exists and is not empty")
    tmp = out + ".partial"
    if os.path.exists(tmp):
        raise SystemExit(f"{tmp} exists (an interrupted run?); remove it first")
    os.makedirs(tmp)
    encode, special, tmpl, tok_info = (None, frozenset(), None, None)
    if tokenizer:
        encode, special, tmpl, tok_info = _tokenizer(tokenizer)
    c = collections.Counter()
    refused, examples, lengths = collections.Counter(), [], []
    w = ShardWriter(tmp, shard_size)
    for path in files:
        with open(path, "rb") as f:
            for line_no, raw in enumerate(f, 1):
                if not raw.strip():
                    continue
                c["read"] += 1
                src = {"file": path, "line": line_no, "sha256": hashlib.sha256(raw.rstrip(b"\n")).hexdigest()}
                try:
                    try:
                        rec = json.loads(raw)
                    except ValueError as e:
                        raise Refused("BAD_JSON", str(e)[:120])
                    o = convert(rec, allow_nontrainable=allow_nontrainable, encode=encode, special_ids=special,
                                source=src, tokenizer_sha256=tok_info and tok_info["sha256"])
                except Refused as e:
                    refused[e.code] += 1
                    if len(examples) < N_EXAMPLES:
                        examples.append({"file": path, "line": line_no, "code": e.code, "detail": e.detail[:200]})
                    continue
                if tmpl is not None:
                    ids, flags = tmpl.render(o)
                    o["adapter"]["n_tokens"] = len(ids)
                    lengths.append(len(ids))
                    c["supervised_targets"] += int(flags.sum())
                c["admitted"] += 1
                c["admitted_nontrainable"] += o["adapter"]["admitted_nontrainable"]
                for t in o["turns"]:
                    c[f"turns_{t['role']}"] += 1
                    c["assistant_no_loss"] += t["role"] == "assistant" and not t["loss"]
                w.write(o)
    w.close()
    manifest = {
        "version": VERSION, "allow_nontrainable": allow_nontrainable,
        "inputs": [{"file": p, "sha256": sha256_file(p)} for p in files],
        "counts": dict(c), "refused": dict(refused), "refused_examples": examples,
        "shards": [os.path.join(out, os.path.basename(p)) for p in w.paths], "tokenizer": tok_info,
        "tokens": None if not lengths else {
            "total": sum(lengths), "max": max(lengths), "mean": round(sum(lengths) / len(lengths), 1),
            "over_max_len": None if max_len is None else sum(n > max_len for n in lengths), "max_len": max_len},
        "harness_data": None if not w.paths else {
            "mode": "pack", "pad_id": tok_info["pad_id"] if tok_info else None,
            "chat": ({"role_ids": tok_info["role_ids"], "end_id": tok_info["end_id"], "loss": "assistant"}
                     if tok_info else {"loss": "assistant", "note": "text records: set data.tokenizer"}),
            "sources": [{"name": "pipeline", "kind": "chat", "paths": [os.path.join(out, "chat-*.jsonl")]}]},
    }
    with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    if os.path.exists(out):
        os.rmdir(out)
    os.rename(tmp, out)
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--allow-nontrainable", action="store_true")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--max-len", type=int, default=None)
    a = ap.parse_args(argv)
    m = run(a.inputs, a.out, allow_nontrainable=a.allow_nontrainable, tokenizer=a.tokenizer,
            shard_size=a.shard_size, max_len=a.max_len)
    print(json.dumps({"out": os.path.abspath(a.out), "counts": m["counts"], "refused": m["refused"],
                      "tokens": m["tokens"]}))
    return 0 if m["counts"].get("admitted") else 3


if __name__ == "__main__":
    sys.exit(main())
