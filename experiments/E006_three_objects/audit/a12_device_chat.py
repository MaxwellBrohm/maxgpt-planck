"""A12. Q-device on the chat probe: C (CUDA) minus E005's Mac transcripts, TF / TF2 / LOOP / CHECKS, with A4's
measures and bootstrap; and e005w (CUDA) vs E005 (Mac) transcripts of the same weights, turn by turn."""
import json
from common import *
import a4_qchat as Q


def conv_rows(path):
    convs = sorted((json.loads(l) for l in open(path)), key=lambda c: c["id"])
    return [dict(id=c["id"], n=len(c["turns"]), tf=sum(Q.tf(t["assistant"]) for t in c["turns"]),
                 tf2=sum(Q.tf2(t["assistant"]) for t in c["turns"]), loop=sum(Q.loop(t["assistant"]) for t in c["turns"]),
                 checks=sum(v is True for v in c["checks"].values()), turns=[t["assistant"] for t in c["turns"]])
            for c in convs]


E6T = lambda t: os.path.join(E6, "transcripts", f"{PFX}{t}__greedy.jsonl")
E5T = lambda s: os.path.join(E5, "transcripts", f"{PFX}s{s}__greedy.jsonl")
C = [conv_rows(E6T(f"C{s}")) for s in SEEDS]
M = [conv_rows(E5T(s)) for s in SEEDS]
assert all([r["id"] for r in a] == [r["id"] for r in b] for a, b in zip(C, M))
NT = np.array([[r["n"] for r in x] for x in C])
for key, kind in (("tf", "rate"), ("tf2", "rate"), ("loop", "rate"), ("checks", "checks")):
    A = np.array([[r[key] for r in x] for x in C]); B = np.array([[r[key] for r in x] for x in M])
    print(f"C - E005(Mac) chat {key.upper()}: {fmt(boot_conv(A, B, NT, kind=kind))}")
for s in SEEDS:
    W = conv_rows(E6T(f"e005w{s}"))
    same = sum(a == b for x, y in zip(W, M[s - 1]) for a, b in zip(x["turns"], y["turns"]))
    cs = sum(a == b for x, y in zip(C[s - 1], M[s - 1]) for a, b in zip(x["turns"], y["turns"]))
    print(f"s{s}: e005w (CUDA) turns identical to E005 (Mac): {same}/149; C{s} turns identical to E005 s{s}: {cs}/149")
