"""E005 n-gram overlap of E004's eval text with the E005 training text (notes.txt: "0 word 5-grams shared between
training text and eval text (E004's rule, new pools included)"). Same sections and gates as ngram_overlap.py:
  1 word 5-grams within one text (a turn, question, prefix, answer), gated at 0 (3/4-grams reported)
  2 whole-sentence frames (object terms AND aliases -> <o>, values -> <v>), gated at 0
  3 template level, gated at 0: E004's training templates plus the alias corrections and every original template
    with each join substituted in ({a} -> <o>, which never matches across sides)
  4 whole rendered prompts, reported only.
Training text: purity_e005.sample() (seeds 1-5, first 6,404 draws each); eval text: the eval, dev, probe draws.
usage: python3 -B ngram_overlap_e005.py [--seeds 1 2 3 4 5] > ../logs/ngram_overlap_e005.txt"""
import sys
from collections import Counter

import items_e004 as I
import train_e005 as T5
import purity_e005 as P5
import ngram_overlap as N4
import pools_alias_train as PA
from text_e004 import words, normalize
from pools_train import POOLS


def train_texts(exs):
    texts, frames = set(), set()
    for ex in exs:
        terms = [t for ph, h in ex["objects"] for t in (ph, h)] + list(ex.get("aliases", {}).values())
        pool = POOLS[ex["vtype"]]["values"]
        st = {s["turn"] for s in ex["stmts"]}
        for i, (u, a) in enumerate(ex["turns"]):
            texts |= {u, a}
            if i in st:
                frames |= {tuple(normalize(u, terms, pool)), tuple(normalize(a, terms, pool))}
        texts |= {ex["question"], ex["answer"].strip()}
        frames |= {tuple(normalize(ex["question"], terms, pool)), tuple(normalize(ex["answer"], terms, pool))}
    return texts, frames


def alias_templates():
    out = [t.replace("{a}", "{o}") for vt in PA.ALIAS_CORR for t in PA.ALIAS_CORR[vt]]
    for vt, P in POOLS.items():
        for j in PA.JOINS[vt]:
            out += [t.replace("{o}", "{o} " + j.replace("{a}", "{o}")) for t in P["orig"]]
    return out


def template_train_grams():
    out = N4.template_grams("train")
    for t in alias_templates():
        ws = ["<o_train>" if w == "<o>" else w for w in normalize(t)]
        out |= {(g, t) for g in N4.grams(ws, 5)}
    return out


def main():
    seeds = [int(x) for x in sys.argv[sys.argv.index("--seeds") + 1:]] if "--seeds" in sys.argv else list(P5.SEEDS)
    exs = [x for s in seeds for x in T5.take(s, P5.PER_SEED)]
    ttexts, tframes = train_texts(exs)
    draws = {name: I.draw(name) for name in I.DRAWS}
    src, eframes = N4.eval_texts(draws)
    tg = {n: set().union(*(N4.grams(words(t), n) for t in ttexts)) for n in (3, 4, 5)}
    print(f"E005 n-gram overlap: {len(ttexts)} distinct training texts from {len(exs)} examples (seeds {seeds}, "
          f"first {P5.PER_SEED} draws; {sum(1 for x in exs if x['block'] == 'alias')} ALIAS); eval texts from the "
          f"eval, dev and probe draws\n")
    print("1 word n-grams within one text: shared / distinct eval n-grams (share)")
    fail = 0
    for s, texts in sorted(src.items()):
        row = []
        for n in (3, 4, 5):
            eg = set().union(*(N4.grams(words(t), n) for t in texts))
            sh = eg & tg[n]
            row.append(f"{n}-grams {len(sh)}/{len(eg)} ({len(sh) / max(1, len(eg)):.3f})")
            if n == 5 and sh:
                fail += 1
                row.append(f"SHARED 5-GRAMS {sorted(sh)[:5]}")
        print(f"  {s:15s} " + "  ".join(row))
    common4 = Counter()
    for texts in src.values():
        for t in texts:
            for g in N4.grams(words(t), 4) & tg[4]:
                common4[" ".join(g)] += 1
    print("  most frequent shared 4-grams:", ", ".join(f"'{g}' x{c}" for g, c in common4.most_common(12)))
    shf = eframes & tframes
    fail += bool(shf)
    print(f"\n2 whole-sentence frames shared: {len(shf)} {sorted(shf)[:5]}")
    te, tt = N4.template_grams("eval"), template_train_grams()
    sht = {g for g, _ in te} & {g for g, _ in tt}
    fail += bool(sht)
    print(f"3 template-level 5-grams shared (<v> = any value; {len(alias_templates())} alias/join templates added): "
          f"{len(sht)} {[(g, [t for x, t in te if x == g][:1]) for g in sorted(sht)[:5]]}")
    tr_prompt = set()
    for ex in exs:
        tr_prompt |= N4.grams(words(T5.prompt(ex) + ex["answer"]), 5)
    ev_prompt = set()
    for it in draws["eval"].values():
        for x in it:
            ev_prompt |= N4.grams(words(I.prompt(x, with_prefix=True) + " " + x["gold"]), 5)
    shp = ev_prompt & tr_prompt
    print(f"4 whole rendered prompts (role tags, across turns), eval draw: {len(shp)}/{len(ev_prompt)} 5-grams "
          f"shared ({len(shp) / len(ev_prompt):.4f}), e.g. {[' '.join(g) for g in sorted(shp)[:6]]}")
    print("\nRESULT: " + ("ALL PASS (1: 0 shared 5-grams, 2: 0 frames, 3: 0 template 5-grams)" if not fail
                          else f"{fail} FAILED"))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
