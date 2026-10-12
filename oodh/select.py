"""OOD-H Part 1, step 1: candidate threads from the OOD-H reserve (prereg/RC-12.draft.txt s14; oodh/DESIGN.txt).

Runs on the PC (or anywhere with the OASST2 file); loads no model. Writes item content ONLY under --out, which must be
a sealed/ directory, and prints counts only.

Stages, in order, each counted in select_stats.json:
  reserve       corpus/oodh.in_oodh_reserve (the ONE reserve rule; same salt, nothing re-implemented)
  not_leaked    minus corpus/oodh.leaked_reserve_trees(all trees, every Dolly instruction)
  lang_en       prompt.lang == 'en'
  ready         tree_state == 'ready_for_export'
  path          --path best (default): corpus/readers_chat.best_path, the highest-ranked usable path, the extractor's
                own rule; --path deepest (fallback F1, DESIGN.txt): the usable path with the most user turns. A message
                is unusable when deleted, synthetic, review-failed, spam, not English, empty, or it trips the AI-ism
                filter (hygiene.aiism); trailing user turns are cut, so a thread ends on an assistant turn.
  hygiene       hygiene.hygiene_reason on the rendered thread (the extractor's check: size, non-ASCII, English)
  user2         >= 2 user turns -> candidates.jsonl (candidates_deepest.jsonl under --path deepest), every one with
                its knowledge-light verdict
  kl            the first knowledge-light rule, kl_rule() below (DESIGN.txt KL1)
  budget        thread proxy tokens <= THREAD_BUDGET, so the thread plus 3 probes fits RC-12's 1,800 (DESIGN.txt)
Fallback counts (reported, never used for selection): fb_trailing_user (2+ user turns if the cut trailing user turn
were kept, F2), fb_mod4_* (the same pipeline, leak check skipped, on sha256(SALT + id) % 4 == 0 and not in the
reserve: an ESTIMATE of what a wider reserve adds, F3). Run as: python oodh/select.py [--path deepest].
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0] or ".") == _HERE:
    sys.path.pop(0)       # this file shares the stdlib module name 'select'; never let its directory shadow it

import argparse  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from collections import Counter  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "corpus"))
import hygiene as H  # noqa: E402
from oodh import SALT, in_oodh_reserve, leaked_reserve_trees  # noqa: E402
from readers_chat import best_path, message_problem, render_turns, thread_turns  # noqa: E402

TREES = "~/planck/data/raw/starter/OpenAssistant__oasst2/2023-11-05_oasst2_all.trees.jsonl.gz"
DOLLY = "~/planck/data/raw/starter/databricks__databricks-dolly-15k/databricks-dolly-15k.jsonl"
OUT = "~/planck/sealed/oodh"
EXTRACT_OPTS = dict(min_bytes=H.MIN_BYTES, max_non_ascii=H.MAX_NON_ASCII, min_stopwords=H.MIN_EN_STOPWORDS)
TOKENS_PER_WORD, TOKENS_PER_MSG = 1.35, 4          # rc12/common.py proxy
PROBE_ALLOWANCE = 3 * (TOKENS_PER_WORD * 70 + 2 * TOKENS_PER_MSG)   # 3 probes: <= 20-word turn + 50-word reply
THREAD_BUDGET = 1800 - PROBE_ALLOWANCE

# ---- knowledge-light rule KL1 (user turns only; classifier-free; DESIGN.txt) ----
_I = re.I | re.M
CODE = re.compile(
    r"```|\bdef \w+\(|^\s*(?:import|from) [\w.]+|#include\b|console\.log|\bprint\(|\bfunction\s*\w*\s*\(|=>"
    r"|</?(?:div|span|html|body|script|p|a|br|td|tr|table)\b[^>]*>|\bselect\b[^.?!\n]*\bfrom\b\s+\w+\s+where\b"
    r"|;\s*$|\{\s*$|^\s*\}", _I)
CODE_WORDS = re.compile(
    r"(?<!\w)(?:python|javascript|typescript|java|c\+\+|c#|rust(?:lang| code| program)|golang|sql|html|css"
    r"|regex|regular expression|bash|powershell|json|yaml|api|compiler|github|numpy|pandas|reactjs|react\.js"
    r"|django|arduino|programming|programmer|algorithm|source code|stack trace|swiftui|kotlin|php|npm|docker"
    r"|kubernetes|nginx|jquery|node\.js)(?!\w)"
    r"|(?<!dress )(?<!zip )(?<!postal )(?<!area )(?<!morse )(?<!promo )(?<!discount )(?<!bar )\bcode\b", re.I)
MATH = re.compile(
    r"\\(?:frac|int|sum|sqrt|cdot|times|begin|alpha|beta|theta|pi)\b|\$[^$\n]*\\[^$\n]*\$"
    r"|\d\s*[=+*^×÷]\s*\(?-?\d|\d\s+/\s+\d|(?<![\w'])[xyz]\s*(?:[=^]|\+\s*\d)"
    r"|\b(?:equation|integral|derivative|polynomial|theorem|prove that|solve for"
    r"|logarithm|calculus|algebra|matrices)\b", re.I)
MY_THING = re.compile(r"\bmy\s+(?!own\b|question\b|previous\b|last\b|first\b|next\b)[a-z]{3,}", re.I)
I_STATE = re.compile(
    r"\b(?:i|we)(?:'m|'ve|'d|'ll| am| have| had| was| were| want| need| plan| planned| would like| like| love"
    r"| hate| prefer| enjoy| live| lived| work| worked| study| feel| felt| got| bought| started| tried| own"
    r"| just| recently| currently| usually| will| can't| cannot| don't| didn't| decided| moved| met| found)\b", re.I)
TASK = re.compile(
    r"^\W*(?:(?:hi|hello|hey|ok|okay|thanks|thank you|great|nice|cool|now|also|and|please)\W+)*"
    r"(?:(?:can|could|would|will) you\s+(?:please\s+)?|please\s+|i (?:want|need|would like|'d like) you to\s+)?"
    r"(?:write|compose|draft|create|make|generate|give me|suggest|recommend|plan|design|invent|imagine|pretend|draw"
    r"|act as|role-?play|let's|let us|tell me a (?:story|joke|poem)|rewrite|rephrase|reword|edit|proofread"
    r"|brainstorm|come up with|help me)\b", re.I)


def kl_rule(user_texts):
    """-> (passed, reason, signals). reason: 'code' | 'math' | 'personal' | 'task' | 'trivia'."""
    user_texts = [u.replace("\u2019", "'") for u in user_texts]
    joined = "\n".join(user_texts)
    sig = dict(code=bool(CODE.search(joined) or CODE_WORDS.search(joined)), math=bool(MATH.search(joined)),
               my=len(MY_THING.findall(joined)), i_state=len(I_STATE.findall(joined)),
               task=sum(bool(TASK.search(s)) for u in user_texts for s in re.split(r"(?<=[.!?])\s+|\n+", u)))
    if sig["code"]:
        return False, "code", sig
    if sig["math"]:
        return False, "math", sig
    if sig["my"] >= 1 or sig["i_state"] >= 2:
        return True, "personal", sig
    if sig["task"] >= 1:
        return True, "task", sig
    return False, "trivia", sig


# ---- selection ----
def iter_trees(path):
    with gzip.open(os.path.expanduser(path), "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def dolly_instructions(path):
    if not path:
        return []
    with open(os.path.expanduser(path), encoding="utf-8") as fh:
        return [json.loads(line)["instruction"] for line in fh if line.strip()]


def usable_kids(node):
    return [c for c in node.get("replies") or [] if not message_problem(c)]


def n_users(path):
    return sum(m.get("role") == "prompter" for m in path)


def deepest_path(node):
    """F1: from a usable node, the usable path with the most user turns that ends on an assistant turn; ties go to
    the better-ranked reply (readers_chat's order: rank 0 first, unranked after ranked, then file order). [] if none."""
    kids = sorted(((c.get("rank") is None, c.get("rank") or 0, i), c) for i, c in enumerate(usable_kids(node)))
    best = []
    for _, c in kids:
        p = deepest_path(c)
        if p and (not best or n_users(p) > n_users(best)):
            best = p
    return [node] + best if best or node.get("role") == "assistant" else []


def root_path(p, rule):
    if rule == "best":
        return best_path(p)
    return deepest_path(p) if not message_problem(p) and p.get("role") == "prompter" else []


def proxy_tokens(turns):
    return TOKENS_PER_WORD * sum(len(t["text"].split()) for t in turns) + TOKENS_PER_MSG * len(turns)


def thread_of(t, rule="best"):
    """-> (drop_stage, None) or (None, record) for a tree already known to be reserved and not leaked."""
    p = t["prompt"]
    if p.get("lang") != "en":
        return "lang_en", None
    if t.get("tree_state") != "ready_for_export":
        return "ready", None
    path = root_path(p, rule)
    if len(path) < 2:
        return "path", None
    turns = thread_turns(path)
    if H.hygiene_reason(render_turns(turns), **EXTRACT_OPTS):
        return "hygiene", None
    users = [x["text"] for x in turns if x["role"] == "user"]
    ok, reason, sig = kl_rule(users)
    return None, {"tree_id": t["message_tree_id"], "message_ids": [m["message_id"] for m in path],
                  "turns": turns, "n_user_turns": len(users), "proxy_tokens": round(proxy_tokens(turns), 1),
                  "trailing_user_cut": bool(usable_kids(path[-1])),
                  "path_rule": rule,
                  "kl": {"pass": ok, "reason": reason, "signals": sig}}


def in_mod4(tid):
    return int.from_bytes(hashlib.sha256((SALT + tid).encode("utf-8")).digest(), "big") % 4 == 0


def select(trees_factory, other_texts=(), rule="best"):
    """trees_factory() -> a fresh iterable of OASST2 trees (read twice). -> (candidates, stats Counter)."""
    leaked = leaked_reserve_trees(trees_factory(), other_texts)
    st, cands = Counter(leaked_all=len(leaked)), []
    for t in trees_factory():
        tid = t["message_tree_id"]
        st["trees"] += 1
        res = in_oodh_reserve(tid)
        if not res and in_mod4(tid):
            stage, rec = thread_of(t, rule)
            if rec and rec["n_user_turns"] >= 2:
                st["fb_mod4_extra_user2"] += 1
                st["fb_mod4_extra_kl"] += rec["kl"]["pass"]
        if not res:
            continue
        st["reserve"] += 1
        if tid in leaked:
            continue
        st["not_leaked"] += 1
        stage, rec = thread_of(t, rule)
        if stage:
            st[f"drop_{stage}"] += 1
            continue
        st["thread"] += 1
        n = rec["n_user_turns"]
        st["fb_trailing_user"] += n + rec["trailing_user_cut"] >= 2
        if n < 2:
            continue
        st["user2"] += 1
        st["user3"] += n >= 3
        cands.append(rec)
        kl = rec["kl"]
        st[f"kl_{kl['reason']}"] += 1
        if kl["pass"]:
            st["kl"] += 1
            st["kl_user3"] += n >= 3
            st["budget"] += rec["proxy_tokens"] <= THREAD_BUDGET
    cands.sort(key=lambda r: r["tree_id"])
    st["candidates_ids_sha256"] = hashlib.sha256("\n".join(r["tree_id"] for r in cands).encode()).hexdigest()
    return cands, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", default=TREES)
    ap.add_argument("--dolly", default=DOLLY)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--path", choices=("best", "deepest"), default="best")
    a = ap.parse_args()
    out = os.path.abspath(os.path.expanduser(a.out))
    if "sealed" not in out.split(os.sep):
        sys.exit(f"refusing --out outside a sealed/ directory: {out}")
    os.makedirs(out, exist_ok=True)
    cands, st = select(lambda: iter_trees(a.trees), dolly_instructions(a.dolly), a.path)
    tag = "" if a.path == "best" else "_" + a.path
    with open(os.path.join(out, f"candidates{tag}.jsonl"), "w", encoding="utf-8") as fh:
        for r in cands:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    st = dict(st, thread_budget=round(THREAD_BUDGET, 1), path_rule=a.path)
    with open(os.path.join(out, f"select_stats{tag}.json"), "w") as fh:
        json.dump(st, fh, indent=1, sort_keys=True)
    print(json.dumps(st, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
