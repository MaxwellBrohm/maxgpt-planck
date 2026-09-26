from common import *
import textwrap
READ = ["F_ellipsis_capital", "F_translate", "F_whynot", "F_shorter", "I_one_sentence", "O_add10", "RI_identity",
        "T_dragon", "L_chitchat", "R_d2_name"]
tag = sys.argv[1]
exp = sys.argv[2] if len(sys.argv) > 2 else "E005"
W = int(sys.argv[3]) if len(sys.argv) > 3 else 170
root = E5 if exp == "E005" else E4
C = {c["id"]: c for c in (json.loads(l) for l in open(f"{root}/transcripts/{SLUG}__{tag}__greedy.jsonl"))}
for cid in READ:
    c = C[cid]
    print(f"## {exp} {tag} {cid} checks={c['checks']}")
    for i, t in enumerate(c["turns"]):
        a = t["assistant"].replace("\n", " / ")
        print(f"  U{i}: {t['user'][:110]}")
        print(f"  A{i} [{t['n_gen_tokens']}{' CAP' if t['flags']['hit_max'] else ''}]: {a[:W]}")
