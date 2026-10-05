"""W4 amendment 7, the label re-vote (BANKPASS s3 N). Stage P's vote answer regex was "[1-9][0-9]?", so a teacher
could run a valid answer on into a second digit or answer past the list (Qwen 20 and 88 on lists of 12 and 11,
Gemma 67, 77, 50): 9 of 16 label banks had no 2 of 3 majority among in-range answers. The re-vote asks the same
question (prompts3.vote_prompt) on the candidate list the stage view gives now (w3deps.vote_calls), with the answer
forced to one of 1..n, greedy (bankserve's judge sampling), every label bank, all three teachers. Records go to
<out>/<q|m|g>.jsonl (the stage's call files are not touched); w4freeze reads them as kind "vote2".

    python -m bankpass.w4vote --root STAGE --out W4VOTE --teacher T --port P"""
import argparse
import json
import os
import sys
import time

from bankpass import client as CL, prompts3 as P3, store, w3amend, w3deps as WD, w3state as WS

MAX_TOKENS = 12


def answer_re(n):
    return "(?:" + "|".join(str(i) for i in range(n, 0, -1)) + ")"


def calls(root, teacher):
    recs = WS.records(root)
    its = WS.view(root, recs)
    with open(os.path.join(root, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    out = []
    for c in WD.vote_calls(teacher, its, recs, w3amend.base_calls(plan)):
        out.append(dict(c, call_id=c["call_id"].replace("vote.", "vote2.", 1), kind="vote2"))
    return out


def render(c):
    prompt = P3.vote_prompt(c["what"], c["cands"])
    return prompt, P3.spec(c["teacher"], [answer_re(len(c["cands"]))], MAX_TOKENS)


def parse(text, n):
    rows, problem = P3.rows_of(text, 1)
    if problem:
        return [], problem
    if not rows[0].isdigit() or not 1 <= int(rows[0]) <= n:
        return [], "OUT_OF_RANGE"
    return rows, None


def record(c, prompt, spec, row, t0):
    lines, problem = parse(row.get("text"), len(c["cands"]))
    return {"schema": store.CALL_SCHEMA, "call_id": c["call_id"], "kind": "vote2", "bank": c["bank"], "class": "N",
            "author": store.teacher_author(c["teacher"]), "plan": c, "prompt": prompt,
            "prompt_sha256": store.sha256_text(prompt), "decode_sha256": spec["sha256"], "regex": spec["regex"],
            "seed": c["seed"], "server_decode": row.get("decode"), "raw": row.get("text"),
            "raw_sha256": store.sha256_text(row.get("text") or ""), "finish": row.get("finish"),
            "n_out": row.get("n_out"), "lines": lines, "problem": problem, "time": round(t0, 3),
            "seconds": round(time.time() - t0, 3), "mode": "real", "amendment": 7}


def run(cl, todo, path):
    done = store.done_ids(path)
    n = 0
    for c in todo:
        if c["call_id"] in done:
            continue
        t0 = time.time()
        prompt, spec = render(c)
        row = cl.generate(prompt, spec, c["seed"], None, call=c)
        store.append_jsonl(path, record(c, prompt, spec, row, t0))
        n += 1
    return n


def path_of(out, teacher):
    return os.path.join(out, store.SHORT[teacher] + ".jsonl")


def records(out):
    recs = []
    for t in store.SHORT:
        recs += store.read_jsonl(path_of(out, t))
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args(argv)
    WS.install_human()
    todo = calls(a.root, a.teacher)
    cl = CL.Client(f"http://127.0.0.1:{a.port}", a.teacher)
    cl.check()
    n = run(cl, todo, path_of(a.out, a.teacher))
    print(json.dumps({"teacher": a.teacher, "calls": len(todo), "ran": n}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
