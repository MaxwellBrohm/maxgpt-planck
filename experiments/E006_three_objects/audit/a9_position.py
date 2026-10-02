"""A9. P's manipulation, recomputed on the DRAWN streams (first 6,404 per seed, no tokenizer) with my own GL code:
GL = 1 if the dialogue's last value statement is about the asked object and is not incidental; GL2 / GL1 = mean GL
over 2-object examples whose asked object is introduced second / first; GLX2 = mean over all 2-object examples.
The notes' drawn, pooled s1-5 values: P .4369 / .0832 / .2080, C .6658 / .0690 / .2665, E004 .5586 / .0384 / .2189."""
import itertools
from collections import Counter
from common import *

code_path()
import train_e004, train_e005, train_e006p


def stats(src, n=6404, block=None):
    g1, g2, roles = [], [], Counter()
    for s in SEEDS:
        for x in itertools.islice(src(s), n):
            if block and x.get("block") != block:
                continue
            if int(x["n_obj"]) != 2:
                continue
            st = x["stmts"]
            order = []
            for z in st:
                if z["obj"] not in order:
                    order.append(z["obj"])
            last = st[-1]
            roles[last["role"]] += 1
            gl = int(last["obj"] == x["asked"] and last["role"] != "incid")
            (g2 if order.index(x["asked"]) == 1 else g1).append(gl)
    m = lambda v: sum(v) / len(v)
    return m(g2), m(g1), m(g1 + g2), len(g1 + g2), dict(roles)


if __name__ == "__main__":
    for name, src in (("P", train_e006p.stream_p), ("C", train_e005.stream), ("E004", train_e004.stream)):
        gl2, gl1, glx2, n, roles = stats(src)
        print(f"{name}: GL2 {gl2:.4f} GL1 {gl1:.4f} GLX2 {glx2:.4f} (n2 {n}; last-statement roles {roles})")
    for blk in ("alias", "ind"):
        for name, src in (("P", train_e006p.stream_p), ("C", train_e005.stream)):
            gl2, gl1, glx2, n, _ = stats(src, block=blk)
            print(f"  block {blk} {name}: GL2 {gl2:.3f} GL1 {gl1:.3f} GLX2 {glx2:.3f} (n2 {n})")
