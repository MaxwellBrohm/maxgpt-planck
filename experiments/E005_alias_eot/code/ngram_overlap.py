"""E004 n-gram overlap of eval text with training text (step 3; notes (b): "0 whole-sentence frames and 0 word
5-grams shared between any eval text and any training text (4-grams reported)").
Training text: the 20,000 dialogues of purity_e004.sample() (every turn, question and answer).
Eval text: every turn, question, forced prefix and prefix + gold of the eval, dev and probe draws.
  1 word n-grams (n = 3, 4, 5) within one text (a turn, a question, a prefix), by eval text source; 5-grams
    gated at 0.
  2 whole-sentence frames: each statement, acknowledgement, question and prefix + <v> normalized (object terms
    -> <o>, values -> <v>) against the normalized training statements, acknowledgements, questions and answers;
    gated at 0.
  3 template level (sound for every object/value combination, including ones the sample never drew): 5-grams of
    the normalized templates, <v> matching any value and <o> never matching (eval and training objects share no
    word); gated at 0.
  4 whole rendered prompts, role tags included (grams across turn boundaries): reported, not gated.
Usage: python3 -B ngram_overlap.py > ../logs/ngram_overlap.txt"""
import sys
from collections import Counter

import items_e004 as I
import train_e004 as T
from purity_e004 import sample
from text_e004 import words, normalize
from pools_train import POOLS, ACKS, ELL_COMMON, REV_ELL_COMMON
from pools_eval import ALL_EVAL_POOLS, ELL_COMMON_EVAL, ACKS_E, filler_pool


def grams(ws, n):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def train_texts(exs):
    texts, frames = set(), set()
    for ex in exs:
        terms = [t for ph, h in ex["objects"] for t in (ph, h)]
        pool = POOLS[ex["vtype"]]["values"]
        st = {s["turn"] for s in ex["stmts"]}
        for i, (u, a) in enumerate(ex["turns"]):
            texts |= {u, a}
            if i in st:
                frames |= {tuple(normalize(u, terms, pool)), tuple(normalize(a, terms, pool))}
        texts |= {ex["question"], ex["answer"].strip()}
        frames |= {tuple(normalize(ex["question"], terms, pool)), tuple(normalize(ex["answer"], terms, pool))}
    return texts, frames


def eval_texts(draws):
    """{source: set of texts} and the set of normalized eval sentences (frames)."""
    src, frames = {}, set()
    fill = {"default": set(filler_pool("default")), "h7": set(filler_pool("h7"))}
    for D in draws.values():
        for items in D.values():
            for it in items:
                terms = [t for ph, h in it["objects"] for t in (ph, h)] + ([it["alias"]] if it["alias"] else [])
                st = {s["turn"] for s in it["stmts"]}
                for i, (u, a) in enumerate(it["turns"]):
                    if i in st:
                        src.setdefault("statement", set()).add(u)
                        src.setdefault("ack", set()).add(a)
                        frames |= {tuple(normalize(u, terms, it["values"])), tuple(normalize(a, terms, it["values"]))}
                    else:
                        key = "filler h7" if (u, a) in fill["h7"] else "filler default"
                        src.setdefault(key, set()).update((u, a))
                src.setdefault("question", set()).add(it["question"])
                src.setdefault("prefix + gold", set()).add(it["prefix"] + " " + it["gold"])
                frames |= {tuple(normalize(it["question"], terms, it["values"])),
                           tuple(normalize(it["prefix"] + " {v}", terms, it["values"]))}
    return src, frames


def template_grams(side):
    """normalized templates of one side; <o> gets a side-specific token so it can never match."""
    out = set()
    if side == "train":
        tpls = [t for P in POOLS.values() for r in ("orig", "corr", "pron", "ell", "rev", "rev_pron", "rev_ell",
                                                     "incid", "ask", "ans", "ans_upd") for t in P[r]]
        tpls += ELL_COMMON + REV_ELL_COMMON + [t for pool in ACKS.values() for t in pool]
    else:
        tpls = [t for P in ALL_EVAL_POOLS.values() for r in ("orig", "corr", "pron", "ell", "ask", "pre",
                                                              "alias_corr") for t in P.get(r, [])]
        tpls += [f[x] for P in ALL_EVAL_POOLS.values() for f in P.get("frames", []) for x in f]
        tpls += ELL_COMMON_EVAL + [t for pool in ACKS_E.values() for t in pool]
    for t in tpls:
        ws = ["<o_" + side + ">" if w == "<o>" else w for w in normalize(t.replace("{a}", "{o}"))]
        out |= {(g, t) for g in grams(ws, 5)}
    return out


def main():
    exs = sample()
    ttexts, tframes = train_texts(exs)
    draws = {name: I.draw(name) for name in I.DRAWS}
    src, eframes = eval_texts(draws)
    tg = {n: set().union(*(grams(words(t), n) for t in ttexts)) for n in (3, 4, 5)}
    print(f"E004 n-gram overlap: {len(ttexts)} distinct training texts from {len(exs)} dialogues; eval texts from"
          f" the eval, dev and probe draws\n")
    print("1 word n-grams within one text: shared / distinct eval n-grams (share)")
    fail = 0
    for s, texts in sorted(src.items()):
        row = []
        for n in (3, 4, 5):
            eg = set().union(*(grams(words(t), n) for t in texts))
            sh = eg & tg[n]
            row.append(f"{n}-grams {len(sh)}/{len(eg)} ({len(sh) / max(1, len(eg)):.3f})")
            if n == 5 and sh:
                fail += 1
                row.append(f"SHARED 5-GRAMS {sorted(sh)[:5]}")
        print(f"  {s:15s} " + "  ".join(row))
    common4 = Counter()
    for texts in src.values():
        for t in texts:
            for g in grams(words(t), 4) & tg[4]:
                common4[" ".join(g)] += 1
    print("  most frequent shared 4-grams:", ", ".join(f"'{g}' x{c}" for g, c in common4.most_common(12)))
    shf = eframes & tframes
    fail += bool(shf)
    print(f"\n2 whole-sentence frames shared: {len(shf)} {sorted(shf)[:5]}")
    te, tt = template_grams("eval"), template_grams("train")
    sht = {g for g, _ in te} & {g for g, _ in tt}
    fail += bool(sht)
    print(f"3 template-level 5-grams shared (<v> = any value): {len(sht)} "
          f"{[(g, [t for x, t in te if x == g][:1]) for g in sorted(sht)[:5]]}")
    tr_prompt = set()
    for ex in exs:
        tr_prompt |= grams(words(T.prompt(ex) + ex["answer"]), 5)
    ev_prompt = set()
    for it in draws["eval"].values():
        for x in it:
            ev_prompt |= grams(words(I.prompt(x, with_prefix=True) + " " + x["gold"]), 5)
    shp = ev_prompt & tr_prompt
    print(f"4 whole rendered prompts (role tags, across turns), eval draw: {len(shp)}/{len(ev_prompt)} 5-grams "
          f"shared ({len(shp) / len(ev_prompt):.4f}), e.g. {[' '.join(g) for g in sorted(shp)[:6]]}")
    print("\nRESULT: " + ("ALL PASS (1: 0 shared 5-grams, 2: 0 frames, 3: 0 template 5-grams)" if not fail
                          else f"{fail} FAILED"))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
