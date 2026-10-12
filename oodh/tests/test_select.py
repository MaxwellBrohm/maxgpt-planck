"""oodh/select.py on synthetic OASST2-shaped trees (no real item text). Run: python oodh/tests/test_select.py"""
import importlib.util
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("OODH_SELECT_SRC") or os.path.join(os.path.dirname(HERE), "select.py")   # env: mutants
spec = importlib.util.spec_from_file_location("oodh_select", SRC)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
from oodh import in_oodh_reserve  # noqa: E402  (corpus/ is on sys.path via select.py)

_ids = itertools.count()
RES = [t for t in (f"tree-{i}" for i in range(400)) if in_oodh_reserve(t)]
OUT = [t for t in (f"tree-{i}" for i in range(400)) if not in_oodh_reserve(t)]

U1 = "I am planning a small birthday dinner for my sister next month and she really loves spicy food."
A1 = "That sounds like a lovely plan. A spicy menu could start with a warm soup and end with something cool."
U2 = "Good idea, but I forgot to say that she cannot eat nuts, so please keep that in mind for the dessert."
A2 = "Of course. A mango sorbet or a simple lemon cake would be safe and it would cool things down after dinner."
AI_ISM = "As an AI language model, I would suggest a mango sorbet, which is safe and cools things down."


def m(role, text, rank=None, replies=(), **kw):
    d = dict(message_id=f"m{next(_ids)}", role=role, text=text, lang="en", rank=rank, deleted=False,
             synthetic=False, review_result=True, replies=list(replies))
    d.update(kw)
    return d


def chain(*texts, **kw):
    """u, a, u, a ... as one straight path."""
    node = None
    for i, t in reversed(list(enumerate(texts))):
        node = m("prompter" if i % 2 == 0 else "assistant", t, rank=None if i % 2 == 0 else 0,
                 replies=[node] if node else [])
    return node


def tree(tid, prompt, state="ready_for_export"):
    return dict(message_tree_id=tid, tree_state=state, prompt=prompt)


def run(trees, dolly=()):
    return S.select(lambda: iter(trees), dolly)


def test_keeps_a_reserved_two_user_turn_thread_and_drops_the_rest():
    trees = [tree(RES[0], chain(U1, A1, U2, A2)),
             tree(OUT[0], chain("My cat is old and sleeps all day, should I worry about her?", A1, U2 + " n", A2)),
             tree(RES[1], chain(U1, A1, U2, A2) | {"lang": "de"}),                          # not English
             tree(RES[2], chain(U1 + " x", A1, U2, A2), state="aborted_low_grade"),         # not ready
             tree(RES[3], chain(U1 + " y", A1))]                                              # one user turn
    cands, st = run(trees)
    assert [c["tree_id"] for c in cands] == [RES[0]], cands
    assert st["reserve"] == 4 and st["drop_lang_en"] == 1 and st["drop_ready"] == 1, st
    assert st["thread"] == 2 and st["user2"] == 1, st
    c = cands[0]
    assert [t["role"] for t in c["turns"]] == ["user", "assistant", "user", "assistant"]
    assert c["kl"]["pass"] and c["kl"]["reason"] == "personal", c["kl"]


def test_highest_ranked_reply_and_ai_ism_skipped_for_a_sibling():
    a_good = m("assistant", A1, rank=1, replies=[m("prompter", U2, replies=[m("assistant", A2, rank=0)])])
    a_top_ai = m("assistant", AI_ISM, rank=0, replies=[m("prompter", U2 + " z", replies=[m("assistant", A2, rank=0)])])
    a_worse = m("assistant", A1 + " w", rank=2, replies=[])
    p = m("prompter", U1, replies=[a_worse, a_top_ai, a_good])
    cands, _ = run([tree(RES[0], p)])
    assert len(cands) == 1 and cands[0]["message_ids"][1] == a_good["message_id"], cands
    a_ok = m("assistant", A1, rank=1, replies=[])
    a_top = m("assistant", A1 + " v", rank=0, replies=[m("prompter", U2, replies=[m("assistant", A2, rank=0)])])
    cands, _ = run([tree(RES[0], m("prompter", U1, replies=[a_ok, a_top]))])
    assert cands and cands[0]["message_ids"][1] == a_top["message_id"], cands


def test_deepest_path_fallback_takes_the_most_user_turns_and_ties_go_to_rank():
    a_top = m("assistant", A1 + " " + A2, rank=0, replies=[m("prompter", U2 + " t")])   # its user turn: no reply
    a_mid = m("assistant", A1, rank=1, replies=[m("prompter", U2, replies=[m("assistant", A2, rank=0)])])
    a_low = m("assistant", A1 + " k", rank=2, replies=[m("prompter", U2 + " k", replies=[m("assistant", A2, rank=0)])])
    trees = [tree(RES[0], m("prompter", U1, replies=[a_low, a_top, a_mid]))]
    assert run(trees)[0] == []                                   # the best path has one user turn
    cands, st = S.select(lambda: iter(trees), (), "deepest")
    assert len(cands) == 1 and cands[0]["message_ids"][1] == a_mid["message_id"], cands
    assert cands[0]["path_rule"] == "deepest" and cands[0]["n_user_turns"] == 2, cands


def test_ai_ism_cuts_the_path_and_the_trailing_user_turn_is_counted_as_a_fallback():
    cands, st = run([tree(RES[0], chain(U1, A1 + " " + A2, U2, AI_ISM))])     # u, a is over 200 bytes
    assert cands == [] and st["thread"] == 1 and st["fb_trailing_user"] == 1, st


def test_leaked_trees_are_dropped_by_other_trees_and_by_dolly():
    trees = [tree(RES[0], chain(U1, A1, U2, A2)), tree(OUT[0], chain(U2, A1)),
             tree(RES[1], chain(U1 + " Then again, my brother is coming too.", A1, U2, A2))]
    cands, st = run(trees)
    assert cands == [] and st["leaked_all"] == 2 and st["not_leaked"] == 0, st
    trees = [tree(RES[0], chain(U1, A1, U2, A2))]
    cands, st = run(trees, dolly=[U1.upper()])
    assert cands == [] and st["leaked_all"] == 1, st
    cands, st = run(trees, dolly=["What is the capital of a country?"])
    assert len(cands) == 1 and st["leaked_all"] == 0, st


def test_kl_rule_reasons():
    cases = [
        (["Write a python function that sorts a list.", "Now make it faster."], "code"),
        (["Why does this fail?\n```\nx = [1, 2]\n```", "Thanks, and the other one?"], "code"),
        (["Create a SwiftUI view for a login form.", "Add a dark mode."], "code"),
        (["What is 12 * 7?", "And if you divide that by two?"], "math"),
        (["Solve for x: 3x + 2 = 11.", "Show the steps please."], "math"),
        (["What is the tallest mountain in Europe?", "How tall is it?"], "trivia"),
        (["Who wrote the first modern novel?", "When was it published?"], "trivia"),
        (["My dog keeps barking at night.", "She is a beagle, if that helps."], "personal"),
        (["I'm moving to a new city soon and I have no friends there.", "Any tips?"], "personal"),
        (["I\u2019m new in town and I\u2019ve got a job interview.", "Any advice?"], "personal"),
        (["Write a short poem about autumn leaves.", "Make it rhyme."], "task"),
        (["Can you help me plan a weekend trip?", "Somewhere near the sea."], "task"),
        (["Hi! Could you please draft a thank-you note?", "Shorter please."], "task"),
        (["Draw a small house in ASCII art.", "Add a chimney."], "task"),
    ]
    for users, want in cases:
        ok, reason, _ = S.kl_rule(users)
        assert reason == want, (users, reason, want)
        assert ok == (want in ("personal", "task")), (users, ok)


def test_kl_rule_does_not_call_everyday_words_code_or_math():
    for users in (["I live in a zip code area near the coast.", "My dress code at work is strict."],
                  ["I have 2-3 hours free and my knee hurts after 5 km.", "We usually run on 1/2 days."]):
        ok, reason, _ = S.kl_rule(users)
        assert ok and reason == "personal", (users, reason)


def test_extractor_hygiene_drops_a_tiny_thread():
    cands, st = run([tree(RES[0], chain("My cat is sad.", "Oh no.", "I moved.", "Okay."))])
    assert cands == [] and st["drop_hygiene"] == 1, st


def test_budget_counts_only_short_threads():
    long_a = " ".join([A1] * 60)
    cands, st = run([tree(RES[0], chain(U1, A1, U2, A2)), tree(RES[1], chain(U1 + " p", long_a, U2 + " p", A2)),
                     tree(RES[2], chain(U1 + " r", A1, U2 + " r", A2))])
    assert st["kl"] == 3 and st["budget"] == 2, st


def test_refuses_output_outside_sealed():
    argv, bad = sys.argv, os.path.join(HERE, "not_the_vault")
    sys.argv = ["select.py", "--out", bad, "--trees", os.path.join(bad, "none.gz"), "--dolly", ""]
    got = None
    try:
        S.main()
    except BaseException as e:            # noqa: BLE001  (the guard must be what stops it)
        got = e
    finally:
        sys.argv = argv
        made = os.path.exists(bad)
        if made:
            os.rmdir(bad)
    assert isinstance(got, SystemExit) and "refusing" in str(got), repr(got)
    assert not made, "main() created a directory outside sealed/"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(f"{len(fns)} passed")
