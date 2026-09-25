"""E005 eval identity and copy integrity (notes.txt WHAT IS RE-CHECKED, "Eval identity"). No model, no tokenizer.
  1 every copied E004 file in this directory still has the sha256 logged in step 1
    (../logs/copied_from_e004_sha256.txt), and so does its E004 source (E004 is read, never written)
  2 E005's eval, dev and probe draws equal E004's item by item: sha256 of (draw, family, index, the LIK prompt
    with its prefix, gold, candidates), built in two separate processes, one per code directory (python3 -B,
    so no bytecode is written into E004); 1,120 of 1,120 must match
usage: python3 -B eval_identity_e005.py > ../logs/eval_identity_e005.txt"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E004 = os.path.abspath(os.path.join(HERE, "..", "..", "E004_general_updating", "code"))
LIST = os.path.join(HERE, "..", "logs", "copied_from_e004_sha256.txt")
SNIPPET = r"""
import hashlib, json, items_e004 as I
out = []
for name in I.DRAWS:
    for fam, items in I.draw(name).items():
        for i, it in enumerate(items):
            key = json.dumps([name, fam, i, I.prompt(it, with_prefix=True), it["gold"], it["candidates"]])
            out.append([name, fam, i, hashlib.sha256(key.encode()).hexdigest()])
print(json.dumps(out))
"""


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def hashes(code_dir):
    r = subprocess.run([sys.executable, "-B", "-c", SNIPPET], cwd=code_dir, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=110)
    if r.returncode:
        raise SystemExit(f"draw failed in {code_dir}: {r.stderr[-500:]}")
    return json.loads(r.stdout)


def main():
    fail = []
    rows = [ln.split() for ln in open(LIST).read().splitlines()[3:] if ln.strip()]
    bad_copy = [f for h, f in rows if sha(os.path.join(HERE, f)) != h]
    bad_src = [f for h, f in rows if sha(os.path.join(E004, f)) != h]
    print(f"1 copied files: {len(rows)} listed; E005 copies changed: {len(bad_copy)} {bad_copy[:5]}; "
          f"E004 sources changed: {len(bad_src)} {bad_src[:5]}")
    if len(rows) != 64 or bad_copy or bad_src:
        fail.append("copy integrity")
    extra = sorted(f for f in os.listdir(HERE) if f.endswith((".py", ".sh")) and f not in {r[1] for r in rows})
    print(f"  files added in E005 (not copies): {extra}")
    a, b = hashes(HERE), hashes(E004)
    same = sum(1 for x, y in zip(a, b) if x == y)
    per = {}
    for x, y in zip(a, b):
        per.setdefault(x[0], [0, 0])
        per[x[0]][0] += x == y
        per[x[0]][1] += 1
    print(f"2 eval/dev/probe items identical to E004's: {same} of {len(b)} (E005 built {len(a)}) "
          + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in per.items()))
    if len(a) != len(b) or same != len(b) or len(b) != 1120:
        fail.append("eval identity")
    print("\nRESULT: " + ("ALL PASS" if not fail else f"FAILED: {fail}"))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
