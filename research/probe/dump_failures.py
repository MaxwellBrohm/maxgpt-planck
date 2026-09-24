"""Print every failed check with the conversation, for hand-picking FAILURES.md entries."""
import json, glob, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
mode = sys.argv[1] if len(sys.argv) > 1 else "greedy"
only = sys.argv[2] if len(sys.argv) > 2 else ""
for f in sorted(glob.glob(os.path.join(HERE, "transcripts", f"*__{mode}.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if only and only not in r["id"]:
            continue
        failed = [k for k, v in r["checks"].items() if v is False]
        if not failed and not only:
            continue
        print(f"\n######## {r['model']} | {r['id']} | seed={r['seed']} | failed={failed} | lenient={r.get('checks_lenient')}")
        if r["system"]:
            print("SYSTEM:", r["system"])
        for i, t in enumerate(r["turns"]):
            print(f"--- U{i}: {t['user']}")
            fl = {k: v for k, v in t['flags'].items() if v}
            print(f"--- A{i} ({t['n_gen_tokens']} tok, prompt {t['n_prompt_tokens']}) {fl if fl else ''}:\n{t['assistant'][:700]}")
        for fz in r["forced"]:
            print("FORCED:", fz)
