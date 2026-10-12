"""E006 data for the entry point and the checks (notes.txt ARMS). No torch at import, no model.
Update examples (arms C, P, and G's update part) go through E005's encoder unchanged: e005_sets.example_stream
(E005's two renders, the 768-token redraw, the drawn/kept stats), fed
  C and G  train_e005.stream(seed)        (source=None inside e005_sets: byte for byte E005's stream)
  P        train_e006p.stream_p(seed)     (the one change, notes ARMS P)
kept(...) taps the source to return the kept example dicts with their ids/labels (validate_e006, the tests).
Replay threads (arm G, notes ARMS G "render"): SmolLM2's chat template over the thread's turns (user first, ending
on an assistant turn), no system message (the template inserts its default one), the "\\n" after the final
<|im_end|> not in the input (as E005's chat render); labels = every assistant turn's content tokens plus its
<|im_end|>, -100 elsewhere; spans from joint tokenization of turn prefixes (a thread whose spans do not tokenize
jointly is dropped, reason "split"). Over max_len: cut after the longest prefix of whole exchanges that fits; if the
first exchange does not fit, dropped ("first_exchange_too_long"). A turn containing a special-token string is
dropped ("special_token_text"). A thread whose tokenized input does not decode back to its template text is dropped
("lossy_text"; SmolLM2's vocabulary has no token for some control bytes, e.g. U+0006, and the tokenizer drops them
silently)."""
import hashlib
import json
import random

import e005_sets as S5
import train_e005 as T5
import train_e006p as TP

EOT = "<|im_end|>"
SPECIAL = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
ARMS = ("C", "P", "G")
REPLAY_ORDER = 48        # per seed: pool sorted by id, shuffled with Random(1000 * seed + 48)


def cand_ids_notorch(tok, prompt, cont, add_special=True):
    """lik.cand_ids, copied verbatim (lik.py imports torch at module level). test_train_e006p checks the source
    text equals lik.py's; the PC's validate_e006 uses lik.cand_ids itself."""
    pre = tok(prompt, add_special_tokens=add_special).input_ids
    full = tok(prompt + cont, add_special_tokens=add_special).input_ids
    eos = getattr(tok, "eos_token_id", None)
    if eos is not None:
        while pre and pre[-1] == eos and full and full[-1] == eos:
            pre, full = pre[:-1], full[:-1]
    if full[: len(pre)] == pre and len(full) > len(pre):
        return pre, full[len(pre):], "joint"
    return pre, tok(cont, add_special_tokens=False).input_ids, "split"


def source(arm, seed):
    """the drawn update stream of an arm; None = e005_sets' own default (train_e005.stream(seed))."""
    if arm in ("C", "G"):
        return None
    if arm == "P":
        return TP.stream_p(seed)
    raise ValueError(f"unknown arm {arm!r}")


def example_stream(tok, arm, seed, max_len, stats, cand_ids):
    """(ids, labels, render) of the arm's kept update examples: E005's encoder on the arm's source."""
    return S5.example_stream(tok, seed, max_len, stats, cand_ids, source=source(arm, seed))


def kept(tok, arm, seed, n, cand_ids, max_len=768):
    """-> (list of (example, ids, labels, render) for the first n kept, stats, n drawn)."""
    drawn, out, stats = [], [], S5.new_stats()
    base = source(arm, seed)

    def tap():
        for ex in (base if base is not None else T5.stream(seed)):
            drawn.append(ex)
            yield ex

    for ids, labels, r in S5.example_stream(tok, seed, max_len, stats, cand_ids, source=tap()):
        if drawn[-1]["render"] != r:
            raise ValueError(f"arm {arm} seed {seed}: kept example and yielded render disagree")
        out.append((drawn[-1], ids, labels, r))
        if len(out) == n:
            break
    return out, stats, len(drawn)


def digest(pairs):
    """sha256 over a sequence of (ids, labels): the identity checks compare arms with it."""
    h = hashlib.sha256()
    for ids, labels in pairs:
        h.update(json.dumps([list(ids), list(labels)]).encode())
    return h.hexdigest()


# ---------------- replay (arm G) ----------------
class ReplayDrop(Exception):
    pass


def _msgs(turns):
    return [{"role": t["role"], "content": t["text"]} for t in turns]


def check_turns(turns):
    if len(turns) < 2 or len(turns) % 2:
        raise ReplayDrop("odd_or_short")
    for i, t in enumerate(turns):
        if t["role"] != ("user" if i % 2 == 0 else "assistant") or not t["text"].strip():
            raise ReplayDrop("roles")
        if any(s in t["text"] for s in SPECIAL):
            raise ReplayDrop("special_token_text")


def encode_prefix(tok, turns):
    """whole thread -> (ids, labels); raises ReplayDrop("split") if a span does not tokenize jointly."""
    msgs = _msgs(turns)
    eot = tok.convert_tokens_to_ids(EOT)
    full = tok.apply_chat_template(msgs, tokenize=False)
    if not full.endswith(EOT + "\n"):
        raise ReplayDrop("template_end")
    full = full[:-1]
    ids = tok(full, add_special_tokens=False).input_ids
    if tok.decode(ids, skip_special_tokens=False) != full:
        raise ReplayDrop("lossy_text")
    labels = [-100] * len(ids)
    for i in range(1, len(msgs), 2):
        pre = tok.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True)
        end = pre + msgs[i]["content"] + EOT
        if not full.startswith(end):
            raise ReplayDrop("template_prefix")
        a = tok(pre, add_special_tokens=False).input_ids
        b = tok(end, add_special_tokens=False).input_ids
        if ids[:len(a)] != a or ids[:len(b)] != b or len(b) <= len(a) or b[-1] != eot or eot in b[len(a):-1]:
            raise ReplayDrop("split")
        labels[len(a):len(b)] = ids[len(a):len(b)]
    return ids, labels


def encode_replay(tok, turns, max_len):
    """-> (ids, labels, info); the longest prefix of whole exchanges that fits max_len."""
    check_turns(turns)
    n_ex = len(turns) // 2
    for m in range(n_ex, 0, -1):
        ids, labels = encode_prefix(tok, turns[:2 * m])
        if len(ids) <= max_len:
            return ids, labels, {"exchanges": m, "cut": m < n_ex, "n_tok": len(ids),
                                 "n_lab": sum(x != -100 for x in labels)}
    raise ReplayDrop("first_exchange_too_long")


def label_spans(labels):
    spans, start = [], None
    for i, x in enumerate(list(labels) + [-100]):
        if x != -100 and start is None:
            start = i
        elif x == -100 and start is not None:
            spans.append((start, i))
            start = None
    return spans


def check_replay_labels(tok, ids, labels, turns):
    """the pre-registered label checks -> list of problems (empty = ok): one labelled span per used assistant turn,
    decoding exactly to its text + <|im_end|>, label == id on it, nothing else labelled; and the whole input decodes
    exactly to the chat template text of the used turns (no character lost anywhere, user turns included)."""
    p = []
    spans = label_spans(labels)
    used = [t["text"] for t in turns if t["role"] == "assistant"][:len(spans)]
    n_ass = sum(t["role"] == "assistant" for t in turns)
    if not spans or len(spans) > n_ass:
        p.append(f"{len(spans)} labelled spans for {n_ass} assistant turns")
    for (s, e), text in zip(spans, used):
        if tok.decode(ids[s:e], skip_special_tokens=False) != text + EOT:
            p.append(f"span {s}:{e} does not decode to its assistant turn + {EOT}")
        if list(labels[s:e]) != list(ids[s:e]):
            p.append(f"span {s}:{e}: labels differ from ids")
    tpl = tok.apply_chat_template(_msgs(turns[:2 * len(spans)]), tokenize=False)[:-1] if spans else None
    if spans and tok.decode(ids, skip_special_tokens=False) != tpl:
        p.append("the input does not decode to the chat template text of the used turns")
    pre_all = tok.decode(ids[:spans[0][0]] if spans else ids, skip_special_tokens=False)
    if spans and not pre_all.endswith("<|im_start|>assistant\n"):
        p.append("the first labelled span does not start right after an assistant header")
    return p


def replay_order(pool, seed):
    """pool: list of thread dicts with "id" -> the seed's order (sorted by id, then shuffled)."""
    xs = sorted(pool, key=lambda t: t["id"])
    random.Random(1000 * seed + REPLAY_ORDER).shuffle(xs)
    return xs


def new_replay_stats():
    return {"used": 0, "dropped": {}, "cut": 0, "exchanges": 0, "tokens": 0, "labelled": 0}


def replay_stream(tok, pool, seed, max_len, stats):
    """yields (ids, labels, thread id) in the seed's order; each thread at most once; drops counted by reason."""
    for t in replay_order(pool, seed):
        try:
            ids, labels, info = encode_replay(tok, t["turns"], max_len)
        except ReplayDrop as e:
            stats["dropped"][str(e)] = stats["dropped"].get(str(e), 0) + 1
            continue
        stats["used"] += 1
        stats["cut"] += info["cut"]
        stats["exchanges"] += info["exchanges"]
        stats["tokens"] += info["n_tok"]
        stats["labelled"] += info["n_lab"]
        yield ids, labels, t["id"]
