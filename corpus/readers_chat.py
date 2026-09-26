"""Chat readers for the starter extract: OASST2 message trees and Dolly-15k. Same event protocol as
readers_cp.py.

OASST2 (2023-11-05_oasst2_all.trees.jsonl.gz; one tree per line: message_tree_id, tree_state,
prompt, origin; each message has message_id, parent_id, role 'prompter'|'assistant', lang, text,
deleted, synthetic, model_name, review_result, rank, labels{spam{value,count}}, created_date,
replies):
1. The OOD-H reserve rule (oodh.py) runs first, before anything else is read from the tree.
   Reserved trees are counted and skipped; their bytes are not measured.
2. English trees only (prompt.lang == 'en') in tree_state 'ready_for_export' (the curated export;
   the 'all' file also holds prompt-only lottery trees and halted or aborted trees).
3. A message is unusable when deleted, synthetic or model-written (synthetic / model_name),
   failed review (review_result False), flagged spam (labels.spam.value >= 0.5), not English,
   empty, or when it trips the AI-ism filter (hygiene.aiism, CORPUS 2.2's condition for OASST2).
   An unusable reply is skipped in favour of its best usable sibling (step 4), so an AI-ism drops
   that turn, not the whole tree, unless it is the prompt.
4. Flattening rule: ONE thread per tree, the highest-ranked path. From the root, repeatedly take
   the usable reply with the lowest rank (0 is best; unranked replies come after ranked ones,
   ties go to file order), and stop at a node with no usable reply. Trailing user turns are cut,
   so a thread ends on an assistant turn. One thread per tree means no prompt text is repeated
   and a tree is one document, so train/held-out splits by tree are exact. It leaves out the
   lower-ranked sibling replies (counted in the stats as messages_unused).
5. Rendered the way harness/chat_template.py encodes a turn: each turn's text on its own, turns
   joined by one newline, with no 'User:' / 'Assistant:' prefix (the template marks roles with
   atomic role tokens, never with role-name text, and a turn's first word has no leading space).
   The roles are kept in the record as "turns": [{"role": "user"|"assistant", "text": ...}].
OASST2 is dated 2023, after the Q6 cutoff; CORPUS 2.2 admits it (human, Apache-2.0) with the
date gate marked 'exempt'.

Dolly: instruction, context, response, category; no date or per-row license (dataset
CC-BY-SA-3.0). One doc per row, two turns: user = instruction [blank line context], assistant =
response, rendered as above. Rows that trip the AI-ism filter are dropped (reason aiism). Many rows
share one Wikipedia context, so meta.split_key is a hash of the context (the row id when there is
none): make_tok_sample.py splits train/held-out by it, and rows sharing a context share a side.
"""
import json
from collections import Counter

import hygiene as H
from oodh import in_oodh_reserve
from readers_cp import open_text

OASST_DATASET = "OpenAssistant/oasst2"
DOLLY_DATASET = "databricks/databricks-dolly-15k"
ROLE = {"prompter": "user", "assistant": "assistant"}


def message_problem(m):
    if m.get("deleted"):
        return "deleted"
    if m.get("synthetic") or m.get("model_name"):
        return "synthetic"
    if m.get("review_result") is False:
        return "review_failed"
    spam = ((m.get("labels") or {}).get("spam") or {}).get("value")
    if spam is not None and spam >= 0.5:
        return "spam"
    if m.get("lang") != "en":
        return "lang_mismatch"
    if not (m.get("text") or "").strip():
        return "empty"
    if H.aiism(m["text"]):
        return "aiism"
    return None


def census(m, counts: Counter):
    """Count every message in the tree by problem ('ok' when usable)."""
    counts[message_problem(m) or "ok"] += 1
    for r in m.get("replies") or []:
        census(r, counts)


def tree_bytes(m) -> int:
    return H.nbytes(m.get("text") or "") + sum(tree_bytes(r) for r in m.get("replies") or [])


def best_path(root):
    if message_problem(root) or root.get("role") != "prompter":
        return []
    path, node = [], root
    while node is not None:
        path.append(node)
        kids = [(c.get("rank") is None, c.get("rank") or 0, i, c)
                for i, c in enumerate(node.get("replies") or []) if not message_problem(c)]
        node = min(kids, key=lambda k: k[:3])[3] if kids else None
    while path and path[-1].get("role") != "assistant":
        path.pop()
    return path


def render_turns(turns) -> str:
    """Chat text for the tokenizer sample: turn texts joined by one newline, no role-name text."""
    return "\n".join(t["text"] for t in turns)


def thread_turns(path):
    return [{"role": ROLE[m["role"]], "text": H.normalize(m["text"])} for m in path]


def oasst_tree(t, rel, o):
    """One OASST2 tree -> list of events."""
    tid = t["message_tree_id"]
    if in_oodh_reserve(tid):                         # step 1, before anything else
        return [("drop", "oodh_reserve", 0), ("note", "oodh_reserved_trees", 1)]
    p = t["prompt"]
    tb = tree_bytes(p)
    if p.get("lang") != "en":
        return [("drop", "lang", tb)]
    if t.get("tree_state") not in o["oasst_states"]:
        return [("drop", "tree_state", tb)]
    counts = Counter()
    census(p, counts)
    ev = [("note", f"messages_{k}", v) for k, v in counts.items()]
    path = best_path(p)
    if len(path) < 2:
        return ev + [("drop", "aiism" if message_problem(p) == "aiism" else "no_reply", tb)]
    ev.append(("note", "messages_unused", counts["ok"] - len(path)))
    turns = thread_turns(path)
    text = render_turns(turns)
    reason = H.hygiene_reason(text, o["min_bytes"], o["max_non_ascii"], o["min_stopwords"])
    if reason:
        return ev + [("drop", reason, tb)]
    meta = {"dataset": OASST_DATASET, "file": rel, "tree_id": tid,
            "message_ids": [m["message_id"] for m in path], "n_turns": len(path),
            "n_user_turns": sum(m["role"] == "prompter" for m in path),
            "created": p.get("created_date"), "date_status": "exempt", "license": "apache-2.0",
            "license_basis": "dataset", "tree_state": t.get("tree_state"),
            "path_rule": "best_rank", "lang": "en", "split_key": tid}
    return ev + [("keep", {"id": f"oasst2:{tid}", "source": "oasst2", "text": text,
                           "turns": turns, "meta": meta}, tb)]


def read_oasst(source, path, rel, o):
    with open_text(path) as fh:
        for i, line in enumerate(fh):
            if o["max_docs"] and i >= o["max_docs"]:
                break
            if line.strip():
                yield from oasst_tree(json.loads(line), rel, o)


def dolly_row(i, d, rel, o):
    ins, ctx, resp = ((d.get(k) or "").strip() for k in ("instruction", "context", "response"))
    rb = H.nbytes(ins) + H.nbytes(ctx) + H.nbytes(resp)
    if not ins or not resp:
        return ("drop", "empty", rb)
    turns = [{"role": "user", "text": H.normalize(ins + ("\n\n" + ctx if ctx else ""))},
             {"role": "assistant", "text": H.normalize(resp)}]
    text = render_turns(turns)
    reason = H.hygiene_reason(text, o["min_bytes"], o["max_non_ascii"], o["min_stopwords"])
    if reason:
        return ("drop", reason, rb)
    if H.aiism(text):
        return ("drop", "aiism", rb)
    split = "ctx:" + H.dedup_key(H.normalize(ctx)) if ctx else f"dolly:{i}"
    meta = {"dataset": DOLLY_DATASET, "file": rel, "orig_id": str(i),
            "category": d.get("category"), "created": None, "date_status": "undated",
            "license": "cc-by-sa-3.0", "license_basis": "dataset", "split_key": split}
    return ("keep", {"id": f"dolly:{i}", "source": "dolly", "text": text, "turns": turns,
                     "meta": meta}, rb)


def read_dolly(source, path, rel, o):
    with open_text(path) as fh:
        for i, line in enumerate(fh):
            if o["max_docs"] and i >= o["max_docs"]:
                break
            if line.strip():
                yield dolly_row(i, json.loads(line), rel, o)
