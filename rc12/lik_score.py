"""RC-12 likelihood-row scoring (prereg draft s4, P-004): diagnostic only, never in a composite or a claim.

score_rows(rows, logprob, tok, render, ctx=None) scores lik_rows.build() rows with a scorer
  logprob(prefix_ids, continuation_ids) -> float (summed natural-log probability of the continuation after the
  prefix; an optional logprob.many(prefix_ids, [continuation_ids, ...]) -> [float, ...] scores one row's candidates
  in one call when they share the prefix) and a tokenizer tok with encode(text, special) -> [int], optional eos
  (an id the tokenizer appends), canonical (bool), and either template(messages) -> str (a chat-template string)
  or template_ids(messages) -> [int] (Planck: its role-token template as ids, ending with <|assistant|>).
Renders (as the scored protocol, s4): plain = render.plain(messages) + " " + lead, encoded with special tokens
  (BOS if the tokenizer adds one); template = the chat template with the generation prompt, then the lead-in, no
  added special tokens. Each candidate is the continuation " " + candidate.
Candidate ids (E004 lik.cand_ids, same rule): prefix = encode(prompt); if encode(prompt + cont) starts with it,
  the rest is the candidate ("joint"), else the candidate encoded alone ("split", counted per family as n_split).
  A trailing eos on both is dropped first. canonical tokenizers (E004 canonical_scoring: byte BPE that merges
  across spaces): every candidate's full encoding is cut after the longest prefix all candidates share.
Rule (E004 metrics_ft.right / margin): a row is right iff the gold's summed log-prob is STRICTLY above every other
  candidate's (a tie is wrong; a non-finite score is wrong); margin = gold minus the best other candidate, in nats
  (None when not finite).
Token counts, per tokenizer (decided 2026-09-26, notes STEP 11 and its FIX ROUND): a summed log-prob falls with every
  extra token, so a candidate with more tokens than the gold can lose on length alone (the LENGTH stub, -1 per
  token, is right on every row whose gold is the unique shortest). Real tokenizers leave few rows whose candidates
  ALL have one count (Qwen2.5: OWN 2 of 32, ROLE 7 of 48), so each row is scored on the gold against its FOILS: the
  other candidates with the gold's token count under THIS tokenizer (at least one; a row with none is "unmatched",
  counted, not scored). right_m / margin_m apply the rule above to {gold} + foils; chance = 1 / (1 + len(foils)).
  The full-set verdict is kept beside it: acc_equal (rows whose candidates all have one count) and acc_unequal (the
  rest, length-confounded), never pooled with acc_matched. Two tokenizers can give a row different foils:
  lik_compare.py compares two models on the foils both tokenizers share.
Context: with ctx, a row whose longest prefix + candidate exceeds ctx is not scored (over_ctx) and counted; the
  history is never truncated here.
summarize(scored) per key (family; COMPOSE apart): n, n_over_ctx, n_matched, n_unmatched, acc_matched,
  chance_matched (mean per-row chance), n_equal, acc_equal, n_unequal, acc_unequal, n_split, n_nonfinite and n_ties
  (matched rows with a non-finite score among the gold and its foils / whose gold exactly equals the best foil: a
  tie is wrong, and bf16 logits make ties, so they are counted), tier0 = {n, mean, p10} of the matched rows'
  margins (the per-checkpoint guard; a row with a non-finite score enters as -inf, so the mean shows it; p10
  nearest rank, the ceil(0.1 n)-th smallest), BIND pairs = {n, acc} (twins joined by pair_id, both matched, both
  right_m), and for OWN the copying note."""
import math
from collections import defaultdict

import render as RD

ORDER = ["RECALL", "CORR", "BIND", "TWOHOP", "OWN", "TOPIC", "ROLE", "LOOKUP", "COMPOSE"]
OWN_NOTE = ("copying diagnostic: on golden history the gold is the only candidate named twice (the user's offer and "
            "the IDEAL pick), so a most-mentioned heuristic is right on every OWN row; not a test of own-commitment "
            "memory (notes STEP 11 FIX ROUND)")


def prompt_parts(tok, messages, lead, render):
    """-> (head ids, text, special): the prompt is head + encode(text, special)."""
    if render == "plain":
        return [], RD.plain(messages) + " " + lead, True
    if render != "template":
        raise ValueError(render)
    if hasattr(tok, "template_ids"):
        return list(tok.template_ids(messages)), lead, False
    return [], tok.template(messages) + lead, False


def cand_ids(tok, head, text, cont, special):
    pre = head + list(tok.encode(text, special))
    full = head + list(tok.encode(text + cont, special))
    eos = getattr(tok, "eos", None)
    if eos is not None:
        while pre and pre[-1] == eos and full and full[-1] == eos:
            pre, full = pre[:-1], full[:-1]
    if full[:len(pre)] == pre and len(full) > len(pre):
        return pre, full[len(pre):], "joint"
    return pre, list(tok.encode(cont, False)), "split"


def canonical_ids(tok, head, text, conts, special):
    fulls = [head + list(tok.encode(text + c, special)) for c in conts]
    lcp = 0
    while all(len(f) > lcp for f in fulls) and all(f[lcp] == fulls[0][lcp] for f in fulls):
        lcp += 1
    return [(f[:lcp], f[lcp:], "canonical") for f in fulls]


def finite(x):
    return isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x)


def right(scores, gold):
    if gold not in scores or len(scores) < 2 or not all(finite(v) for v in scores.values()):
        return False
    return all(scores[gold] > v for c, v in scores.items() if c != gold)


def margin(scores, gold):
    if gold not in scores or len(scores) < 2 or not all(finite(v) for v in scores.values()):
        return None
    return scores[gold] - max(v for c, v in scores.items() if c != gold)


def score_row(row, logprob, tok, render, ctx=None):
    head, text, special = prompt_parts(tok, row["messages"], row["lead"], render)
    conts = [" " + c for c in row["candidates"]]
    if getattr(tok, "canonical", False):
        parts = canonical_ids(tok, head, text, conts, special)
    else:
        parts = [cand_ids(tok, head, text, c, special) for c in conts]
    out = {k: row[k] for k in ("id", "rid", "key", "family", "cell", "pair_id", "turn", "lead", "gold", "candidates")}
    ntok = {c: len(p[1]) for c, p in zip(row["candidates"], parts)}
    gold = row["gold"]
    foils = [c for c in row["candidates"] if c != gold and ntok[c] == ntok[gold]]
    out.update(render=render, ntok=ntok, how=sorted({p[2] for p in parts}), equal=len(set(ntok.values())) == 1,
               foils=foils, chance=1 / (1 + len(foils)) if foils else None, n_prompt=len(parts[0][0]),
               over_ctx=False, scores=None, right=None, margin=None, right_m=None, margin_m=None)
    if ctx is not None and max(len(p[0]) + len(p[1]) for p in parts) > ctx:
        out["over_ctx"] = True
        return out
    many = getattr(logprob, "many", None)
    if many is not None and all(p[0] == parts[0][0] for p in parts):
        vals = many(parts[0][0], [p[1] for p in parts])
    else:
        vals = [logprob(p[0], p[1]) for p in parts]
    scores = {c: float(v) for c, v in zip(row["candidates"], vals)}
    out.update(scores=scores, right=right(scores, gold), margin=margin(scores, gold))
    if foils:
        sub = {c: scores[c] for c in [gold] + foils}
        out.update(right_m=right(sub, gold), margin_m=margin(sub, gold))
    return out


def score_rows(rows, logprob, tok, render, ctx=None):
    return [score_row(r, logprob, tok, render, ctx) for r in rows]


def p10(xs):
    xs = sorted(xs)
    return xs[max(1, math.ceil(0.1 * len(xs))) - 1] if xs else None


def rate(flags):
    return sum(flags) / len(flags) if flags else None


def guard_margin(r):
    """a matched row's margin for the Tier 0 guard: -inf when a score is not finite (margin_m None)."""
    return -math.inf if r["margin_m"] is None else r["margin_m"]


def pairs(rows):
    by = defaultdict(list)
    for r in rows:
        if r["pair_id"] is not None:
            by[r["pair_id"]].append(r)
    both = [tw for tw in by.values() if len(tw) == 2 and all(t["foils"] for t in tw)]
    return dict(n=len(both), acc=rate([all(t["right_m"] for t in tw) for tw in both]))


def summarize(scored):
    by = defaultdict(list)
    for r in scored:
        by[r["key"]].append(r)
    out = {}
    for key in sorted(by, key=lambda k: (ORDER.index(k) if k in ORDER else len(ORDER), k)):
        rows = by[key]
        ok = [r for r in rows if not r["over_ctx"]]
        mt = [r for r in ok if r["foils"]]
        eq = [r for r in ok if r["equal"]]
        uneq = [r for r in ok if not r["equal"]]
        ms = [guard_margin(r) for r in mt]
        s = dict(n=len(rows), n_over_ctx=len(rows) - len(ok), n_matched=len(mt), n_unmatched=len(ok) - len(mt),
                 acc_matched=rate([r["right_m"] for r in mt]), chance_matched=rate([r["chance"] for r in mt]),
                 n_equal=len(eq), acc_equal=rate([r["right"] for r in eq]), n_unequal=len(uneq),
                 acc_unequal=rate([r["right"] for r in uneq]), n_split=sum(1 for r in ok if "split" in r["how"]),
                 n_nonfinite=sum(1 for r in mt if r["margin_m"] is None),
                 n_ties=sum(1 for r in mt if r["margin_m"] == 0.0),
                 tier0=dict(n=len(ms), mean=sum(ms) / len(ms) if ms else None, p10=p10(ms)))
        if key == "BIND":
            s["pairs"] = pairs(ok)
        if key == "OWN":
            s["note"] = OWN_NOTE
        out[key] = s
    return out
