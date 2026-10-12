"""E005 data for the entry point (e005_ft_test.py): the E005 training stream in both renders (notes.txt (c)).
No torch at import, no model. The eval, continuity, knowledge, dev and probe sets are E004's (e004_sets, unchanged).
Training render (ex["render"], drawn per example in train_e005.stream):
  plain  E004's, unchanged: prompt = train_e004.prompt(ex) (the "User:/Assistant:" transcript ending "Assistant:"),
         target = ex["answer"] (" <sentence>\\n"), tokenized by e004_sets.encode (lik.cand_ids, special tokens on)
  chat   the eval's chat render: msgs = every (user, assistant) exchange, then the question as a user message;
         prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), no system message
         (SmolLM2's template inserts its default one, as lik.chat_prompt and gen_run), tokenized with
         add_special_tokens=False (the template carries its own role tokens, as lik.render's chat mode);
         target = the sentence (no leading space, no newline) + "<|im_end|>" (SmolLM2's eos = end of turn);
         labels on the sentence tokens and the end-of-turn token only. The "\\n" the template writes after
         <|im_end|> is neither in the input nor trained.
Examples over max_len tokens are rejected and the next one is drawn (E004's redraw, never truncated). Drawn and kept
counts are recorded per kind, block, render, case and placement (the notes' <= 2 point share check reads them)."""
import e004_sets as S4
import train_e005 as T5
from e004_sets import SET_NAMES, lik_sets, gen_sets, probe_items  # noqa: F401  (E004's sets, re-exported)

EOT = "<|im_end|>"
RENDERS = ("plain", "chat")
STAT_KEYS = ("kind", "block", "render", "case", "placement")


def chat_msgs(ex):
    msgs = []
    for u, a in ex["turns"]:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": ex["question"]})
    return msgs


def sentence(ex):
    """the answer sentence of a training example: ex["answer"] without its leading space and final newline."""
    a = ex["answer"]
    s = a[1:-1]
    if not (a.startswith(" ") and a.endswith("\n")) or not s or s != s.strip() or "\n" in s:
        raise ValueError(f"unexpected training answer format {a!r}")
    return s


def chat_prompt(tok, ex):
    return tok.apply_chat_template(chat_msgs(ex), tokenize=False, add_generation_prompt=True)


def eot_id(tok):
    """the end-of-turn token id; refuses a tokenizer whose eos is not <|im_end|> (the chat render is SmolLM2's)."""
    i = tok.convert_tokens_to_ids(EOT)
    if tok.eos_token != EOT or i is None or i == getattr(tok, "unk_token_id", None) or i != tok.eos_token_id:
        raise ValueError(f"tokenizer eos {tok.eos_token!r} is not {EOT}: no chat render for this model")
    return i


def encode_chat(tok, ex):
    """-> (ids, labels, how): labels -100 on the whole chat prompt, the ids on the sentence and <|im_end|>."""
    prompt, sent, eot = chat_prompt(tok, ex), sentence(ex), eot_id(tok)
    pre = tok(prompt, add_special_tokens=False).input_ids
    full = tok(prompt + sent + EOT, add_special_tokens=False).input_ids
    if full[:len(pre)] == pre and len(full) > len(pre):
        ans, how = full[len(pre):], "joint"
    else:
        ans, how = tok(sent, add_special_tokens=False).input_ids + [eot], "split"
    if not ans or ans[-1] != eot or eot in ans[:-1] or eot in pre[-1:]:
        raise ValueError(f"chat target must end with exactly one {EOT}: {ans}")
    return list(pre) + list(ans), [-100] * len(pre) + list(ans), how


def encode(tok, ex, cand_ids):
    """-> (ids, labels, how) in the example's render."""
    if ex["render"] == "plain":
        return S4.encode(tok, ex, cand_ids)
    if ex["render"] == "chat":
        return encode_chat(tok, ex)
    raise ValueError(f"unknown render {ex['render']!r}")


def new_stats():
    return {"n": 0, "rejected_long": 0, "drawn": {k: {} for k in STAT_KEYS}, "kept": {k: {} for k in STAT_KEYS},
            "len_sum": 0, "len_max": {r: 0 for r in RENDERS}, "split": {r: 0 for r in RENDERS}}


def _count(d, ex):
    for k in STAT_KEYS:
        v = str(ex.get(k))
        d[k][v] = d[k].get(v, 0) + 1


def example_stream(tok, seed, max_len, stats, cand_ids, source=None):
    """yields (ids, labels, render) for the kept examples of train_e005.stream(seed) (or of source, for tests)."""
    for ex in (source if source is not None else T5.stream(seed)):
        _count(stats["drawn"], ex)
        ids, labels, how = encode(tok, ex, cand_ids)
        if len(ids) > max_len:
            stats["rejected_long"] += 1
            continue
        _count(stats["kept"], ex)
        r = ex["render"]
        stats["n"] += 1
        stats["len_sum"] += len(ids)
        stats["len_max"][r] = max(stats["len_max"][r], len(ids))
        stats["split"][r] += how == "split"
        yield ids, labels, r


def stream_summary(stats):
    """shares drawn vs kept per key (kind, block, render, case, placement) and the largest shift (notes: <= 2 pts)."""
    out, worst = {}, 0.0
    for k in STAT_KEYS:
        nd, nk = sum(stats["drawn"][k].values()), sum(stats["kept"][k].values())
        sh = {v: {"drawn": round(stats["drawn"][k][v] / nd, 4) if nd else None,
                  "kept": round(stats["kept"][k].get(v, 0) / nk, 4) if nk else None} for v in sorted(stats["drawn"][k])}
        if nd and nk:
            worst = max([worst] + [abs(x["drawn"] - x["kept"]) for x in sh.values()])
        out[k] = sh
    n = stats["n"]
    return dict(n=n, rejected_long=stats["rejected_long"], mean_len=round(stats["len_sum"] / n, 1) if n else None,
                len_max=stats["len_max"], split=stats["split"], shares=out,
                max_share_shift=round(worst, 4) if n else None)
