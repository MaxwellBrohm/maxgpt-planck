"""leaked_reserve_trees: reserved OASST2 trees whose user turns also occur outside the reserve."""
from fixtures_chat import msg, tree, tree_ids
from oodh import in_oodh_reserve, leaked_reserve_trees, prompt_key

LINUX = "I want you to act as a Linux terminal for me."
WOOD = "How much wood would a woodchuck chuck if it could?"
JOKE = "Tell me a joke about a cat who learns to cook."


def test_prompt_key_normalizes_case_space_and_nfc():
    assert prompt_key("  How much WOOD\n would\ta  woodchuck ") == "how much wood would a woodchuck"
    assert prompt_key("café") == prompt_key("café")
    assert prompt_key(None) == ""


def test_leaked_reserve_trees():
    ok, res = tree_ids(False, 3), tree_ids(True, 4)
    assert all(in_oodh_reserve(t) for t in res) and not any(in_oodh_reserve(t) for t in ok)

    def a(i):
        return [msg(f"a{i}", "assistant", "an answer", rank=0)]
    trees = [
        tree(res[0], msg(res[0], "prompter", LINUX, replies=a(0))),
        tree(ok[0], msg(ok[0], "prompter", "  " + LINUX.upper().replace(" ", "  "), replies=a(1))),
        tree(res[1], msg(res[1], "prompter", WOOD, replies=a(2))),
        tree(res[2], msg(res[2], "prompter", "A fresh prompt that nobody else ever wrote down.",
                         replies=a(3))),
        tree(res[3], msg(res[3], "prompter", "A first turn that is long enough to count here.",
                         replies=[msg("x", "assistant", "ok", rank=0,
                                      replies=[msg("y", "prompter", JOKE)])])),
        tree(ok[1], msg(ok[1], "prompter", JOKE, replies=a(4))),
        tree(ok[2], msg(ok[2], "prompter", "Something else entirely, and long enough too.",
                        replies=a(5))),
    ]
    dolly = [WOOD, "", "Thanks!"]
    assert leaked_reserve_trees(trees, dolly) == {res[0], res[1], res[3]}
    assert leaked_reserve_trees(trees) == {res[0], res[3]}
    assert leaked_reserve_trees([t for t in trees if t["message_tree_id"] in res]) == set()


def test_short_turns_are_not_leaks():
    ok, res = tree_ids(False, 1), tree_ids(True, 1)
    trees = [tree(res[0], msg(res[0], "prompter", "Thanks!", replies=[msg("a", "assistant", "ok")])),
             tree(ok[0], msg(ok[0], "prompter", "thanks!", replies=[msg("b", "assistant", "ok")]))]
    assert leaked_reserve_trees(trees) == set()
    assert leaked_reserve_trees(trees, min_chars=1) == {res[0]}
