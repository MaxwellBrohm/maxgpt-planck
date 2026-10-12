"""OASST2 fixture trees (split out of fixtures.py). Every planted case carries a MARKER string."""
from oodh import in_oodh_reserve


def prose(tag, n=5, sep=" "):
    from fixtures import prose as _prose
    return _prose(tag, n, sep)


def tree_ids(reserved: bool, n: int):
    out, i = [], 0
    while len(out) < n:
        t = f"fixture-tree-{i:04d}"
        if in_oodh_reserve(t) == reserved:
            out.append(t)
        i += 1
    return out


def msg(mid, role, text, rank=None, replies=(), lang="en", **kw):
    m = {"message_id": mid, "parent_id": None, "user_id": "u", "created_date":
         "2023-02-05T22:44:05+00:00", "text": text, "role": role, "lang": lang,
         "review_count": 3, "review_result": True, "deleted": False, "rank": rank,
         "synthetic": False, "model_name": None, "emojis": {}, "replies": list(replies),
         "labels": {"spam": {"value": 0.0, "count": 3}}}
    m.update(kw)
    return m


def tree(tid, prompt, state="ready_for_export"):
    return {"message_tree_id": tid, "tree_state": state, "prompt": prompt, "origin": "fixture"}


def oasst_trees():
    ok, res = tree_ids(False, 40), tree_ids(True, 3)
    a3 = msg("A3", "assistant", prose("a3", 4), rank=0)
    p3 = msg("P3", "prompter", prose("p3", 3), rank=1, replies=[a3])
    p2 = msg("P2", "prompter", "SPAM_MARKER " + prose("p2", 3), rank=0,
             labels={"spam": {"value": 1.0, "count": 3}})
    a1 = msg("A1", "assistant", prose("a1", 4), rank=1, replies=[p2, p3])
    trees = [tree(ok[0], msg(ok[0], "prompter", prose("p1", 3), replies=[
        msg("A0", "assistant", "DELETED_MARKER " + prose("a0", 4), rank=0, deleted=True), a1,
        msg("A2", "assistant", "LOWER_RANK_MARKER " + prose("a2", 4), rank=2)]))]
    for i, t in enumerate(ok[1:34]):
        a = msg(f"{t}-a", "assistant", prose(f"oa{i}", 5), rank=0)
        trees.append(tree(t, msg(t, "prompter", prose(f"oq{i}", 3), replies=[a])))
    for k, t in enumerate(res):
        a = msg(f"{t}-a", "assistant", f"OODH_RESERVED_MARKER_{k} " + prose(f"r{k}", 5), rank=0)
        trees.append(tree(t, msg(t, "prompter", f"OODH_RESERVED_MARKER_{k} " + prose(k), replies=[a])))
    es = msg(ok[34], "prompter", "SPANISH_MARKER " + prose("es"), lang="es",
             replies=[msg("E1", "assistant", prose("es1"), lang="es", rank=0)])
    trees.append(tree(ok[34], es))
    trees.append(tree(ok[35], msg(ok[35], "prompter", "STATE_MARKER " + prose("st"), replies=[
        msg("S1", "assistant", prose("st1"), rank=0)]), state="aborted_low_grade"))
    trees.append(tree(ok[36], msg(ok[36], "prompter", "NOREPLY_MARKER " + prose("nr"), replies=[
        msg("N1", "assistant", prose("nr1"), rank=0, deleted=True)])))
    trees.append(tree(ok[37], msg(ok[37], "prompter", prose("ai-q", 3), replies=[
        msg("AI1", "assistant", "AIISM_MARKER As an AI language model, I do not have opinions. "
            + prose("ai1", 4), rank=0),
        msg("AI2", "assistant", prose("ai2", 4), rank=1)])))
    trees.append(tree(ok[38], msg(ok[38], "prompter", "AIISM_PROMPT_MARKER What would ChatGPT say? "
                                  + prose("ai-p", 3), replies=[
        msg("AI3", "assistant", prose("ai3", 4), rank=0)])))
    return trees, ok, res
