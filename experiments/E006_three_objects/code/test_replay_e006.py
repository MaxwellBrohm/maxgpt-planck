"""E006 arm G tests (notes.txt CODE TO WRITE, test_replay_e006). No model; the SmolLM2 tokenizer is loaded
(tokshim_e006.load: E006_TOK=hf on the PC, shim elsewhere).
  drops      every pool drop rule of replay_e006 fires on an injected case, and a clean thread is kept
  exposure   the kbig exposure rules (strict, loose) on injected turns
  encode     labels decode exactly to each assistant turn + <|im_end|>; a long thread is cut at whole exchanges; a first
             exchange over max_len and a special-token string are dropped; five label-mask mutants are all caught by
             check_replay_labels (header newline labelled, <|im_end|> unlabelled, a user turn labelled, the system
             prompt labelled, a wrong label id); a thread with U+0006 in a user turn (the tokenizer drops it) is
             dropped as lossy_text, and an input that lost a user-turn character fails check_replay_labels
  order      replay_order is deterministic, differs per seed, and replay_stream never repeats a thread
  schedule   e006_train.g_schedule gives [u, u, u, u, r] per step, update items in stream order; G's update source is
             C's, and per seed 1-5 G's check batch plus 5 steps of update micro-batches are C's first 84 kept examples,
             its replay micro-batches the seed's replay order (the one-change property of G); patch_stream routes
             e005_sets.example_stream to the arm's source and restores the original
usage: python -B test_replay_e006.py   (exit 0 = all pass)"""
import os
import sys
sys.dont_write_bytecode = True

import e005_sets as S5
import e006_sets as S6
import e006_train as TR
import replay_e006 as RP
import tokshim_e006 as TK

FAIL = []


def ok(c, m):
    print(("PASS  " if c else "FAIL  ") + m)
    if not c:
        FAIL.append(m)


def thread(tid, turns):
    return {"id": f"oasst2:{tid}", "meta": {"tree_id": tid},
            "turns": [{"role": "user" if i % 2 == 0 else "assistant", "text": t} for i, t in enumerate(turns)]}


def tree_ids(n_free=8):
    ids = [f"test-tree-{i}" for i in range(1000)]
    res = next(t for t in ids if RP.oodh.in_oodh_reserve(t))
    return res, [t for t in ids if not RP.oodh.in_oodh_reserve(t)][:n_free]


def drops_suite(R=RP):
    F = []
    res, free = tree_ids()
    base = "Could you give me a few ideas for a small vegetable garden on a sunny balcony this spring?"
    ans = "Tomatoes, peppers and herbs such as basil do well in pots with plenty of sun and regular watering."
    long_user = "Please explain in simple words how bread dough rises when yeast is added and left in a warm place."
    rows = [thread(res, [base, ans]), thread("held-1", [base, ans]), thread(free[0], [long_user, ans]),
            thread(free[1], [base, "As an AI language model I cannot garden."]),
            thread(free[2], [base, "The quick brown fox jumps over one lazy dog today, said my neighbour."]),
            thread(free[3], ["hey, why is the sky blue?", ans]), thread(free[4], ["Go on.", ans]),
            thread(free[5], [base, "Ask Mr. tanaka about the balcony."]), thread(free[6], [base, ans])]
    rows[1]["meta"]["tree_id"] = "held-1"
    want = ["reserve", "heldout", "reserve_leak", "aiism", "eval_8gram", "probe_turn", "probe_turn", "eval_surname",
            "keep"]
    evt = ["Yesterday a quick brown fox jumps over one lazy dog again."]
    kept, reasons = R.build(rows, {"held-1"}, {R.oodh.prompt_key(long_user)}, evt, ["Why is the sky blue?", "Go on."],
                            ["Tanaka"])
    got = dict(reasons)
    for r, w in zip(rows, want):
        if got.get(r["meta"]["tree_id"]) != w:
            F.append(f"drop rule: {r['meta']['tree_id']} got {got.get(r['meta']['tree_id'])}, want {w}")
    if len(kept) != 1:
        F.append(f"{len(kept)} threads kept, want 1")
    exp = R.exposure([{"turns": [{"role": "assistant", "text": "The capital of Bhutan is Thimphu, of course."},
                                 {"role": "assistant", "text": "Vientiane is lovely; Laos has great food."}]}])
    import kbig_items as KB
    q = {it["qid"]: it for it in KB.build()}
    fr = next(i for i, it in q.items() if it["question"] == "What is the capital of Bhutan?")
    it_ = next(i for i, it in q.items() if it["question"] == "What is the capital of Laos?")
    if fr not in exp["strict"] or it_ in exp["strict"] or it_ not in exp["loose"]:
        F.append(f"exposure: Bhutan strict {fr in exp['strict']}, Laos strict {it_ in exp['strict']} loose "
                 f"{it_ in exp['loose']} (want True, False, True)")
    return F


def encode_suite(tok, S=S6):
    F = []
    turns = [{"role": "user", "text": "Hi! Can you write a haiku about rain?\nKeep it short."},
             {"role": "assistant", "text": "Soft rain on the roof,\nquiet streets shine like silver,\nthe city exhales."},
             {"role": "user", "text": "Now one about snow, with an emoji at the end."},
             {"role": "assistant", "text": "White hush on the pines \u2744\ufe0f\n```\ncode block stays as it is {}\n```"}]
    try:
        ids, labels, info = S.encode_replay(tok, turns, 768)
        n1 = len(S.encode_replay(tok, turns[:2], 10_000)[0])
        ids2, lab2, info2 = S.encode_replay(tok, turns, n1 + 2)
    except S.ReplayDrop as e:
        return [f"a valid thread was dropped ({e})"]
    p = S.check_replay_labels(tok, ids, labels, turns)
    if p or info["exchanges"] != 2 or info["cut"]:
        F.append(f"encode: {p} {info}")
    spans = S.label_spans(labels)
    mutants = {"header newline labelled": lambda L: _set(L, spans[0][0] - 1, ids),
               "<|im_end|> unlabelled": lambda L: _unset(L, spans[0][1] - 1),
               "a user turn labelled": lambda L: _set(L, spans[0][0] - 8, ids),
               "system prompt labelled": lambda L: _set(L, 3, ids),
               "wrong label id": lambda L: _swap(L, spans[1][0])}
    for name, mut in mutants.items():
        if not S.check_replay_labels(tok, ids, mut(list(labels)), turns):
            F.append(f"label-mask mutant not caught: {name}")
    if not info2["cut"] or info2["exchanges"] != 1 or S.check_replay_labels(tok, ids2, lab2, turns):
        F.append(f"cut at whole exchanges: {info2}")
    for bad, why in ((turns[:2], "first_exchange_too_long"),
                     ([turns[0], {"role": "assistant", "text": "ok <|im_end|> sneaky"}], "special_token_text")):
        try:
            S.encode_replay(tok, bad, 20 if why == "first_exchange_too_long" else 768)
            F.append(f"not dropped: {why}")
        except S.ReplayDrop as e:
            if str(e) != why:
                F.append(f"dropped as {e}, want {why}")
    lossy = [{"role": "user", "text": "Please fix this line: a\x06b"}, turns[1]]
    if tok.decode(tok(lossy[0]["text"], add_special_tokens=False).input_ids) == lossy[0]["text"]:
        F.append("fixture: the tokenizer keeps U+0006, so the lossy_text drop is not exercised")
    try:
        S.encode_replay(tok, lossy, 768)
        F.append("not dropped: lossy_text")
    except S.ReplayDrop as e:
        if str(e) != "lossy_text":
            F.append(f"dropped as {e}, want lossy_text")
    lost = [dict(turns[0], text=turns[0]["text"] + "\x06")] + turns[1:]
    if not S.check_replay_labels(tok, ids, labels, lost):
        F.append("whole-input check: an input that lost a user-turn character passed check_replay_labels")
    pool = [{"id": f"t{i}", "turns": turns} for i in range(30)]
    o1, o1b, o2 = S.replay_order(pool, 1), S.replay_order(pool, 1), S.replay_order(pool, 2)
    if [x["id"] for x in o1] != [x["id"] for x in o1b] or [x["id"] for x in o1] == [x["id"] for x in o2]:
        F.append("replay_order: not deterministic per seed, or equal across seeds")
    used = [t for _, _, t in S.replay_stream(tok, pool, 3, 768, S.new_replay_stats())]
    if len(used) != 30 or len(set(used)) != 30:
        F.append(f"replay_stream: {len(used)} threads, {len(set(used))} distinct")
    return F


def _set(L, i, ids):
    L[i] = ids[i]
    return L


def _unset(L, i):
    L[i] = -100
    return L


def _swap(L, i):
    L[i] = (L[i] + 1) % 49152
    return L


def schedule_suite(tok, TR=TR):
    F = []
    u, r = iter(range(1000)), iter(range(5000, 6000))
    steps = [list(TR.g_schedule(u, r, 4, 4)) for _ in range(3)]
    if [[k for k, _ in s] for s in steps] != [["u", "u", "u", "u", "r"]] * 3:
        F.append("g_schedule: not [u, u, u, u, r] per step")
    if [x for s in steps for k, b in s if k == "u" for x in b] != list(range(48)):
        F.append("g_schedule: update items not in stream order")
    if S6.source("G", 1) is not None or S6.source("C", 1) is not None:
        F.append("G's or C's update source is not train_e005.stream")
    pool = [{"id": f"r{i}", "turns": [{"role": "user", "text": f"Question number {i}?"},
                                      {"role": "assistant", "text": f"Answer number {i}."}]} for i in range(40)]
    for seed in (1, 2, 3, 4, 5):
        c_ref = S5.example_stream(tok, seed, 768, S5.new_stats(), S6.cand_ids_notorch)
        c_items = [next(c_ref) for _ in range(4 + 5 * 16)]
        reg = []
        restore = TR.patch_stream("G", reg)
        try:
            g_upd = S5.example_stream(tok, seed, 768, S5.new_stats(), S6.cand_ids_notorch)
            first = [next(g_upd) for _ in range(4)]
            rep = S6.replay_stream(tok, pool, seed, 768, S6.new_replay_stats())
            sched = [x for _ in range(5) for x in TR.g_schedule(g_upd, rep, 4, 4)]
        finally:
            restore()
        u_items = first + [e for k, b in sched if k == "u" for e in b]
        r_ids = [t for k, b in sched if k == "r" for _, _, t in b]
        if u_items != c_items:
            F.append(f"s{seed}: G's update micro-batches (check batch + 5 steps) are not C's kept examples in order")
        if r_ids != [t["id"] for t in S6.replay_order(pool, seed)][:20]:
            F.append(f"s{seed}: G's replay micro-batches are not the seed's replay order")
    orig = S5.example_stream
    for arm, ref in (("C", lambda: S5.example_stream(tok, 1, 768, S5.new_stats(), S6.cand_ids_notorch)),
                     ("P", lambda: S6.example_stream(tok, "P", 1, 768, S5.new_stats(), S6.cand_ids_notorch))):
        want = [next(it) for it in [ref()] for _ in range(60)]
        reg = []
        restore = TR.patch_stream(arm, reg)
        it = S5.example_stream(tok, 1, 768, S5.new_stats(), S6.cand_ids_notorch)
        got = [next(it) for _ in range(60)]
        restore()
        if got != want or reg[0]["n"] != 60 or S5.example_stream is not orig:
            F.append(f"patch_stream {arm}: routed stream differs, tap count {reg[0]['n']}, or not restored")
    return F


def main():
    tok = TK.load()
    for name, fn in (("drops and exposure", drops_suite), ("encode and order", lambda: encode_suite(tok)),
                     ("schedule and routing", lambda: schedule_suite(tok))):
        f = fn()
        ok(not f, f"{name}: {f[:4]}")
    print("RESULT: " + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
