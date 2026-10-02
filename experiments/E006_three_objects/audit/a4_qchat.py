"""A4. Q-chat (G vs C) from the transcripts with my own TF / TF2 / LOOP / CHECKS written from the notes' text
(chatmeasures_e006.py is not imported; pools_train is imported for the template strings only), my own two-way
bootstrap over seeds x conversations, conditions (a)-(d), the label, and the margin of condition (a)."""
import json, re
from collections import Counter
from common import *

code_path()
import pools_train

TPLS = sorted({t for p in pools_train.POOLS.values() for key in ("ans", "ans_upd") for t in p[key]})
V, O = r"[A-Za-z0-9]+", r"[A-Za-z][A-Za-z'\- ]{0,40}?"


def tre(t):
    parts = re.split(r"(\{v\}|\{o\})", t)
    return re.compile(r"(?<![A-Za-z])" + "".join(V if p == "{v}" else O if p == "{o}" else re.escape(p)
                                                    for p in parts), re.I)


RX = [tre(t) for t in TPLS]
CUT = re.compile(r"\n|(?<=[.!?])\s")


def sentences(reply, n=2):
    s, out = reply.strip(), []
    for _ in range(n):
        if not s:
            break
        m = CUT.search(s)
        out.append(s[:m.start()] if m else s)
        s = s[m.end():].strip() if m else ""
    return out + [""] * (n - len(out))


def is_tpl(text):
    return any(r.search(text) for r in RX)


def tf(reply):
    return is_tpl(sentences(reply, 1)[0])


def tf2(reply):
    a, b = sentences(reply, 2)
    return is_tpl(a) or (len(a.split()) <= 3 and is_tpl(b))


CODEISH = re.compile(r"[{};]\s*$|^\s*[{}\[\]()]+\s*$|^\s*(def |import |return |#include|//|class |for \(|if \()")


def loop(reply):
    fence, lines = False, []
    for ln in reply.split("\n"):
        if ln.strip().startswith("```"):
            fence = not fence
            continue
        if fence or CODEISH.search(ln) or len(re.findall(r"[A-Za-z]+", ln)) < 3:
            continue
        lines.append(ln.strip())
    return bool(lines) and max(Counter(lines).values()) >= 3


def per_conv(tag):
    convs = [json.loads(l) for l in open(os.path.join(E6, "transcripts", f"{PFX}{tag}__greedy.jsonl"))]
    convs.sort(key=lambda c: c["id"])
    rows = []
    for c in convs:
        ts = c["turns"]
        rows.append(dict(id=c["id"], n=len(ts), tf=sum(tf(t["assistant"]) for t in ts),
                         tf2=sum(tf2(t["assistant"]) for t in ts), loop=sum(loop(t["assistant"]) for t in ts),
                         cap=sum(bool(t["flags"]["hit_max"]) for t in ts), eos=sum(bool(t["stopped_eos"]) for t in ts),
                         checks=sum(v is True for v in c["checks"].values()), nchecks=len(c["checks"])))
    return rows


def arm_arrays(arm, key):
    R = [per_conv(tag_of(arm, s)) for s in SEEDS]
    ids = [r["id"] for r in R[0]]
    assert all([r["id"] for r in x] == ids for x in R)
    return np.array([[r[key] for r in x] for x in R]), np.array([[r["n"] for r in x] for x in R])


if __name__ == "__main__":
    tot = {}
    for t in ["base"] + [f"{a}{s}" for a in ("C", "G", "P", "e005w") for s in SEEDS]:
        R = per_conv(t)
        n = sum(r["n"] for r in R)
        tot[t] = {k: sum(r[k] for r in R) for k in ("tf", "tf2", "loop", "cap", "eos", "checks")}
        print(f"{t}: turns {n} TF {tot[t]['tf'] / n:.2f} TF2 {tot[t]['tf2'] / n:.2f} LOOP {tot[t]['loop'] / n:.3f}"
              f" CAP {tot[t]['cap']} EOS {tot[t]['eos']} CHECKS {tot[t]['checks']}/{sum(r['nchecks'] for r in R)}")
    res = {}
    for key, kind in (("tf", "rate"), ("tf2", "rate"), ("loop", "rate"), ("checks", "checks")):
        G, NT = arm_arrays("G", key)
        C, NT2 = arm_arrays("C", key)
        assert (NT == NT2).all()
        res[key] = boot_conv(G, C, NT, kind=kind)
        print(f"G - C {key.upper()}: {fmt(res[key])}  (G total {G.sum()}, C total {C.sum()}, turns {NT.sum()})")
    G_tf, NT = arm_arrays("G", "tf")
    C_tf, _ = arm_arrays("C", "tf")
    G_tf2, _ = arm_arrays("G", "tf2")
    C_tf2, _ = arm_arrays("C", "tf2")
    a = (G_tf.sum() <= 0.5 * C_tf.sum() and G_tf2.sum() <= 0.5 * C_tf2.sum() and res["tf"][2] < 0 and res["tf2"][2] < 0)
    a_ci = res["tf"][2] < 0 and res["tf2"][2] < 0
    b = res["loop"][2] < 0.10
    c = res["checks"][1] > -3
    d = True  # from a2_passrule.py: G stopping learned 5 of 5 seeds
    worse = res["tf"][1] > 0 or res["tf2"][1] > 0
    lab = ("PROTECTS" if a and b and c and d else "PARTLY PROTECTS" if a else "SMALL EFFECT" if a_ci
           else "WORSE" if worse else "NO EFFECT")
    print(f"(a) {a} [TF {G_tf.sum()} <= {0.5 * C_tf.sum()}; TF2 {G_tf2.sum()} <= {0.5 * C_tf2.sum()}] (b) {b} (c) {c}"
          f" (d) {d} -> {lab}")
    print(f"margin: G may gain {int(np.floor(0.5 * C_tf.sum() - G_tf.sum()))} TF turns and"
          f" {int(np.floor(0.5 * C_tf2.sum() - G_tf2.sum()))} TF2 turns before (a) fails")
    # per seed TF ratio and the P arm (not ruled) as a second same-data comparison
    for s in range(5):
        print(f"  s{s + 1}: TF C {C_tf[s].sum()} G {G_tf[s].sum()} ratio {G_tf[s].sum() / C_tf[s].sum():.2f}")
    P_tf, _ = arm_arrays("P", "tf")
    print(f"P (not ruled): TF {P_tf.sum()}/{NT.sum()} = {P_tf.sum() / NT.sum():.3f}; P - C {fmt(boot_conv(P_tf, C_tf, NT))}")
    # leave one conversation out: does (a) survive dropping any single conversation?
    flips = []
    for j in range(NT.shape[1]):
        keep = [i for i in range(NT.shape[1]) if i != j]
        g, cc = G_tf[:, keep].sum(), C_tf[:, keep].sum()
        g2, c2 = G_tf2[:, keep].sum(), C_tf2[:, keep].sum()
        if not (g <= 0.5 * cc and g2 <= 0.5 * c2):
            flips.append(j)
    ids = [r["id"] for r in per_conv("G1")]
    print("ratio condition fails when dropping conversation:", [ids[j] for j in flips])
