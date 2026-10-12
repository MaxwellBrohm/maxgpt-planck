"""Purity of the AL items (items_al.py) against the E005 training stream. No model, no tokenizer.
Training text: train_e005.take(s, 7000) for seeds 1-5, E004 block included. The trainer consumes 6,400 examples
plus 12 self-check examples per seed and rejects 101-112 over 768 tokens (logs/validate_e005.txt), all within the
first 7,000 draws, so this covers every example E005 can train on (and more).
  P1 names     no AL surname in any training text; no training surname or role title in any AL text; no training
               join (with its title + name) in any AL text
  P2 templates no AL alias correction (marker stripped; aliases -> <o>, values -> <v>) shares a word 3-gram with a
               training alias correction template ({a} -> <o>, {v} -> <v>); no AL statement is a training template
  P3 5-grams   no word 5-gram of any AL text (user turn, assistant turn, question, prefix) occurs in a training text
  P4 frames    no AL statement, acknowledgement or question normalized (object terms and aliases -> <o>, values ->
               <v>) equals a normalized training sentence (ngram_overlap_e005.train_texts)
  P5 held out  no AL object is a training object; no AL filler is a training filler"""
import re

import train_e005 as T5
import ngram_overlap_e005 as NG
import checks_e005 as CE
import pools_alias_train as PA
from fillers_train import FILLERS_TRAIN
from oracles_e004 import strip_marker
from pools_train import POOLS
from text_e004 import words, normalize, grams

SEEDS, PER_SEED = (1, 2, 3, 4, 5), 7000
TITLE_ANY = r"(?:Mr\.|Mrs\.|Ms\.|Dr\.|Pastor|Professor|Chef|Captain) [A-Z][a-z]+"


def al_texts(it):
    return [t for pair in it["turns"] for t in pair] + [it["question"], it["prefix"]]


def join_res():
    out = []
    for vt, js in PA.JOINS.items():
        for j in js:
            out.append(re.compile(re.escape(j).replace(re.escape("{a}"), TITLE_ANY)))
    return out


def purity(items, seeds=SEEDS, per_seed=PER_SEED):
    import items_al as A
    exs = [x for s in seeds for x in T5.take(s, per_seed)]
    ttexts, tframes = NG.train_texts(exs)
    big = "\n".join(sorted(ttexts))
    fail = []
    # P1 names
    for s in A.AL_SURNAMES:
        if re.search(r"(?<![A-Za-z])" + s + r"(?![A-Za-z])", big):
            fail.append(f"P1 AL surname {s} in training text")
    al_all = "\n".join(t for it in items for t in al_texts(it))
    for s in list(PA.SURNAMES) + list(PA.ROLES):
        if re.search(r"(?<![A-Za-z])" + re.escape(s) + r"(?![A-Za-z])", al_all):
            fail.append(f"P1 training name/title {s} in AL text")
    for r in join_res():
        if r.search(al_all):
            fail.append(f"P1 training join {r.pattern} in AL text")
    # P2 templates
    tg = set().union(*(CE._tgrams(t) for vt in PA.ALIAS_CORR for t in PA.ALIAS_CORR[vt]))
    train_tpl = {tuple(normalize(t.replace("{a}", "{o}"))) for vt in PA.ALIAS_CORR for t in PA.ALIAS_CORR[vt]}
    n_alias = 0
    for it in items:
        for s in it["stmts"]:
            u = strip_marker(it["turns"][s["turn"]][0])
            n = normalize(u, [t for o in it["objects"] for t in o] + it["aliases_all"], it["values"])
            if s["ref"] == "alias":
                n_alias += 1
                if grams(n, 3) & tg:
                    fail.append(f"P2 alias 3-gram shared: {u!r} {sorted(grams(n, 3) & tg)[:2]}")
            if tuple(n) in train_tpl:
                fail.append(f"P2 statement is a training alias template: {u!r}")
    # P3 5-grams
    t5 = set().union(*(grams(words(t), 5) for t in ttexts))
    shared = {}
    for it in items:
        for t in al_texts(it):
            for g in grams(words(t), 5) & t5:
                shared.setdefault(" ".join(g), t)
    fail += [f"P3 5-gram {g!r} (in {t!r})" for g, t in sorted(shared.items())[:10]]
    # P4 frames
    fr = set()
    for it in items:
        terms = [t for o in it["objects"] for t in o] + it["aliases_all"]
        st = {s["turn"] for s in it["stmts"]}
        for i, (u, a) in enumerate(it["turns"]):
            if i in st:
                fr |= {tuple(normalize(u, terms, it["values"])), tuple(normalize(a, terms, it["values"]))}
        fr |= {tuple(normalize(it["question"], terms, it["values"])), tuple(normalize(it["prefix"], terms, it["values"]))}
    fail += [f"P4 frame shared: {' '.join(f)}" for f in sorted(fr & tframes)[:10]]
    # P5 held out
    tobj = {ph for P in POOLS.values() for ph, _ in P["objects"]}
    fail += [f"P5 training object {ph}" for it in items for ph, _ in it["objects"] if ph in tobj]
    tf = {tuple(f) for f in FILLERS_TRAIN}
    st_turns = lambda it: {s["turn"] for s in it["stmts"]}
    n_fill = 0
    for it in items:
        for i, pair in enumerate(it["turns"]):
            if i not in st_turns(it):
                n_fill += 1
                if tuple(pair) in tf:
                    fail.append(f"P5 training filler {pair[0]!r}")
    print(f"purity: {len(exs)} training examples (seeds {seeds}, first {per_seed} draws each; blocks "
          f"{ {b: sum(1 for x in exs if x['block'] == b) for b in ('e004', 'alias', 'ind')} }), {len(ttexts)} distinct "
          f"training texts, {len(t5)} training 5-grams; AL: {n_alias} alias corrections, {len(fr)} frames, "
          f"{n_fill} filler turns; shared 5-grams {len(shared)}, shared frames {len(fr & tframes)}")
    return fail
