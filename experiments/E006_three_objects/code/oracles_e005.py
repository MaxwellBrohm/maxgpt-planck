"""E005 stream acceptance, text level (notes.txt WHAT IS RE-CHECKED): every cheap rule on each seed's E005 stream,
overall and per block. Rules: E004's O1-O8 with the O5 variants (oracles_e004.ORACLES) and X1-X11
(shortcuts_x.EXTRA), IDEAL, and two diagnostics: AR (latest alias correction, else O2) and T (topic tracker).
Gate (E004's thresholds, applied to every O and X rule): <= 0.70 overall, O1 and O2 <= 0.50, IDEAL = 1.00.
Step 2 runs it on the first 6,404 DRAWN examples of seeds 1-5 (4 self-check + 6,400; no tokenizer, no chat
render yet); validate_e005 (a later step) repeats it on the examples kept after the 768-token rejection.
usage: python3 -B oracles_e005.py [--n 6404] [--seeds 1 2 3 4 5] > ../logs/oracles_e005.txt"""
import argparse
import re
import sys

import oracles_e004 as O
import train_e005 as T5
from shortcuts_x import EXTRA, DIAGNOSTIC
from pools_train import POOLS

MAX_RULE, MAX_O12 = 0.70, 0.50
TITLE_RE = re.compile(r"(?<![A-Za-z])(Mr\.|Mrs\.|Ms\.|Dr\.|Pastor|Professor|Chef|Captain) [A-Z][a-z]+")


class View5(O.View):
    """E004's text view; every alias of the example counts as an object term (as the eval view counts its one)."""

    def terms(self):
        return [t for ph, h in self.it["objects"] for t in (ph, h)] + list(self.it.get("aliases_all", []))


def from_train(ex):
    it = O.from_train(ex)
    it["aliases_all"] = list(ex.get("aliases", {}).values())
    return it


def ar_alias(V):
    """latest statement that holds a title + name and names no object (an alias correction), else O2."""
    return V.latest(lambda ti, t, v: bool(TITLE_RE.search(t)) and not any(
        O.names(t, ph) or O.names(t, h) for ph, h in V.it["objects"]), O.o2_last(V))


FAKE = O.ORACLES + EXTRA
DIAG = [("AR latest alias correction", ar_alias)] + DIAGNOSTIC + [("IDEAL", O.ideal)]


def score(exs):
    """{rule: share right}, {block: {rule: share right}}"""
    rules = FAKE + DIAG
    tot, per, nb = {n: 0 for n, _ in rules}, {}, {}
    for ex in exs:
        V = View5(from_train(ex))
        b = ex["block"]
        nb[b] = nb.get(b, 0) + 1
        pb = per.setdefault(b, {n: 0 for n, _ in rules})
        for name, fn in rules:
            r = fn(V) == ex["gold"]
            tot[name] += r
            pb[name] += r
    n = len(exs)
    return ({k: v / n for k, v in tot.items()},
            {b: {k: v / nb[b] for k, v in d.items()} for b, d in per.items()}, nb)


def gate(overall):
    bad = []
    for name, _ in FAKE:
        v = overall[name]
        if v > MAX_RULE or (name.startswith(("O1", "O2")) and v > MAX_O12):
            bad.append(f"{name} {v:.3f}")
    if overall["IDEAL"] != 1.0:
        bad.append(f"IDEAL {overall['IDEAL']:.4f}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6404)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    a = ap.parse_args(argv)
    fails, tables, streams = [], {}, {}
    for s in a.seeds:
        exs = streams[s] = T5.take(s, a.n)
        overall, per, nb = score(exs)
        tables[s] = (overall, per, nb)
        bad = gate(overall)
        fails += [f"seed {s}: {b}" for b in bad]
        mx = max(overall[n] for n, _ in FAKE)
        print(f"seed {s}: n={len(exs)} blocks {nb} max(rule)={mx:.3f} O1={overall['O1 first mention']:.3f} "
              f"O2={overall['O2 last mention']:.3f} IDEAL={overall['IDEAL']:.2f} "
              f"{'OK' if not bad else 'FAIL ' + '; '.join(bad)}")
    print()
    head = f"{'rule (mean over seeds; overall, then per block)':48s}{'all':>7s}{'e004':>7s}{'alias':>7s}{'ind':>7s}" \
           f"{'max all':>9s}"
    print(head + "\n" + "-" * len(head))
    for name, _ in FAKE + DIAG:
        m = lambda get: sum(get(tables[s]) for s in a.seeds) / len(a.seeds)
        row = [m(lambda t: t[0][name])] + [m(lambda t, b=b: t[1][b][name]) for b in ("e004", "alias", "ind")]
        mx = max(tables[s][0][name] for s in a.seeds)
        flag = "  <- diagnostic" if name in dict(DIAG) else ""
        print(f"{name:48s}" + "".join(f"{x:7.3f}" for x in row) + f"{mx:9.3f}{flag}")
    print("\nalias block by case (mean over seeds): AR, T, O2, O6, O7")
    for case in ("latest", "earlier", "other"):
        xs = [ex for s in a.seeds for ex in streams[s] if ex.get("case") == case and ex["block"] == "alias"]
        o, _, _ = score(xs)
        print(f"  {case:8s} n={len(xs):5d} " + " ".join(f"{k.split()[0]}={o[k]:.3f}" for k in (
            "AR latest alias correction", "T topic tracker (diagnostic)", "O2 last mention", "O6 latest naming asked",
            "O7 latest naming no other")))
    print("\nRESULT: " + ("ALL PASS (every O and X rule <= 0.70, O1/O2 <= 0.50, IDEAL 1.00 on every seed)" if not fails
                          else f"{len(fails)} FAILED: {fails}"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
