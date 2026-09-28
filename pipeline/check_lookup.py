"""Lookup checks (SPEC section 6, S9 golds): LOOKUP_FORMAT, LOOKUP_UNNEEDED, LOOKUP_MISSING, LOOKUP_COPY,
LOOKUP_EMPTY and a stale value in a lookup answer (ANSWER_STALE). Moved out of check_behav unchanged on 2026-09-28
(that file passed 250 lines)."""
import lexicons as L
import parse
from check_base import mentioned
from check_behav import _pool_guess


def chk_lookup(ctx):
    out, sched = [], set()
    for e in ctx.skel["events"]:
        if e["kind"] != "S9":
            continue
        g, tr = e["gold"], e["turns"]
        i = tr.get("answer")
        if g["need"] == "context":
            if i in ctx.text and L.LOOKUP_TALK_RE.search(ctx.text[i]):
                out.append(("LOOKUP_UNNEEDED", i, L.LOOKUP_TALK_RE.search(ctx.text[i]).group(0)))
            continue
        sched |= {tr["call"], tr["tool"]}
        for j in (tr["call"], tr["tool"]):
            if j not in ctx.text:
                continue
            if j == tr["call"] and not L.TAG_RE.search(ctx.text[j]):
                out.append(("LOOKUP_MISSING", j, ctx.text[j][:30]))
            elif parse.normalize(ctx.text[j]) != parse.normalize(ctx.by_i[j]["text"]):
                out.append(("LOOKUP_FORMAT", j, ctx.text[j][:30]))
        if i not in ctx.text:
            continue
        if g["result"] == "empty":
            guess = _pool_guess(ctx, i, e["params"]["vtype"])
            if not L.NOT_FOUND_RE.search(ctx.text[i]) or guess:
                out.append(("LOOKUP_EMPTY", i, f"guess {guess}"))
        elif not ctx.has(i, g["answer"]):
            out.append(("LOOKUP_COPY", i, f"lacks {g['answer']}"))
        if g.get("stale") and mentioned(ctx, i, g["stale"]):
            out.append(("ANSWER_STALE", i, f"{e['id']} {g['stale']}"))
    for t, s in ctx.turns():
        if t["i"] not in sched and L.TAG_RE.search(s):
            out.append(("LOOKUP_UNNEEDED", t["i"], "unscheduled tag"))
    return out


CHECKS = [chk_lookup]
