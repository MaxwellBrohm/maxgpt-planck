"""E005 AL diagnostic items (notes.txt ADDITION, deviation 2 follow-up). No model, no tokenizer.
In the E004 eval every alias correction is the asked object's latest statement, so "the latest statement that holds
a name wins" scores 1.00 on the alias items without linking a name to its object. AL separates the two:
  AL1  B aliased; B's alias correction comes after A's latest; ask A.       gold A's latest (the alias is a lure)
  AL2  both aliased; A's alias correction, then B's; ask A.                 gold A's alias correction (B's is a lure)
  AL3  as AL2, ask B.                                                       gold B's alias correction (the latest)
  AL4  A aliased; A's alias correction, then a head-noun correction of A.   gold the head-noun correction
"Latest statement with a name" is right on AL3 only (0.25); true linking is right on all four.
Wording is E004's held-out eval wording (eval objects, templates, joins "with" / "from" / "run by", eval alias
corrections, acknowledgements, markers, questions, prefixes, fillers; d = 10, pre 0-3). Names: 20 NEW eval-only
surnames (none in the E005 training pool, none in the E004 eval list, none in any E001-E005 file when chosen)
under the four honorifics, drawn per item; two aliases in one item never share a surname.
Balance per cell of 16: value type x placement jointly (4 x 4: adjacent twice, filler, other_obj), so every value
type has 2 adjacent, 1 filler and 1 other-object item; question form full/head 8/8. Placement = the key alias
correction (AL1: B's lure; AL2, AL4: A's; AL3: B's) relative to its own object's previous statement: adjacent (the
very next turn), filler (1-2 fillers, no statement between) or other_obj (the other object's statement between).
Binary factors balanced within a cell: AL1 A corrected 0/1 times and A also aliased (never used) 8/8; AL2 B's
original before A's or after A's alias correction (6/6 over adjacent + filler); AL3 B's original first or second
(2/2 over other_obj); AL4 B's original before A's or after the head-noun correction (6/6), B also aliased 8/8.
Draw: Random("E005:4005:AL"). Validation: checks_al.py (rules, structure, purity against the E005 stream).
usage: python3 -B items_al.py [--write] > ../logs/items_al.txt"""
import hashlib
import json
import os
import random
import sys

from heldout_e004 import EVAL_CORR_MARKERS, EVAL_NONCORR_MARKERS, HONORIFICS
from pools_eval import POOLS_E, TRAIN_VTYPES, ACKS_E, cap, with_marker, filler_pool
from items_render import pick_objects, object_words, _Templates, _filler_ok
from items_plan import joint, bal
from text_e004 import words

HERE = os.path.dirname(os.path.abspath(__file__))
AL_DIR = os.path.join(os.path.dirname(HERE), "al")
ITEMS_PATH = os.path.join(AL_DIR, "al_items.jsonl")
SHA_PATH = os.path.join(AL_DIR, "al_items.sha256")
SEED = "E005:4005:AL"
CELLS = ("AL1", "AL2", "AL3", "AL4")
N_CELL, D = 16, 10
PLACE4 = ("adjacent", "adjacent", "filler", "other_obj")
AL_SURNAMES = ["Albrecht", "Balogun", "Carvalho", "Donnelly", "Dubois", "Esposito", "Figueroa", "Halvorsen",
               "Ishikawa", "Janssen", "Kovacs", "Magnusson", "Marchetti", "Nwosu", "Okonkwo", "Pellegrini",
               "Rinaldi", "Sandoval", "Szabo", "Whitaker"]


def ev(obj, role, ref, gap=None, marker=None):
    """one statement: gap = fillers before it (None: 0-2 at random); the first statement follows the pre fillers."""
    return dict(obj=obj, role=role, ref=ref, gap=gap, marker=marker)


def cm(rng):
    """E004's correction marker: an eval correction marker with probability 1/2."""
    return rng.choice(EVAL_CORR_MARKERS) if rng.random() < 0.5 else None


def b_orig(rng, gap=None):
    """B's original; an eval non-correction marker in 1/3 (E004's rule for B's first statement)."""
    return ev(1, "orig", "full", gap, rng.choice(EVAL_NONCORR_MARKERS) if rng.random() < 1 / 3 else None)


def a_orig():
    return ev(0, "orig", "full")


def alias_corr(rng, obj, gap=None):
    return ev(obj, "corr", "alias", gap, cm(rng))


def gap_of(rng, place):
    return 0 if place == "adjacent" else rng.randint(1, 2)


def plan_al1(rng, place, f):
    a = [a_orig()] + ([ev(0, "corr", rng.choice(["full", "head"]), None, cm(rng))] if f["kA"] else [])
    if place == "other_obj":        # B's original first, then A's chain, then B's alias right after A's latest
        events = [b_orig(rng)] + a + [alias_corr(rng, 1, rng.randint(0, 1))]
    else:
        events = a + [b_orig(rng), alias_corr(rng, 1, gap_of(rng, place))]
    return events, 0, [0, 1] if f["both"] else [1]


def plan_al2(rng, place, f):
    if place == "other_obj":        # B's original sits between A's original and A's alias correction
        events = [a_orig(), b_orig(rng, rng.randint(0, 1)), alias_corr(rng, 0, rng.randint(0, 1)), alias_corr(rng, 1)]
    else:
        a = [a_orig(), alias_corr(rng, 0, gap_of(rng, place))]
        events = ([b_orig(rng)] + a + [alias_corr(rng, 1)] if f["h"] == "before"
                  else a + [b_orig(rng), alias_corr(rng, 1)])
    return events, 0, [0, 1]


def plan_al3(rng, place, f):
    if place == "other_obj":        # A's alias correction sits between B's original and B's alias correction
        first = [b_orig(rng), a_orig()] if f["h"] == "before" else [a_orig(), b_orig(rng)]
        events = first + [alias_corr(rng, 0), alias_corr(rng, 1, rng.randint(0, 1))]
    else:
        events = [a_orig(), alias_corr(rng, 0), b_orig(rng), alias_corr(rng, 1, gap_of(rng, place))]
    return events, 1, [0, 1]


def plan_al4(rng, place, f):
    head = ev(0, "corr", "head", None, cm(rng))
    if place == "other_obj":
        events = [a_orig(), b_orig(rng, rng.randint(0, 1)), alias_corr(rng, 0, rng.randint(0, 1)), head]
    else:
        a = [a_orig(), alias_corr(rng, 0, gap_of(rng, place)), head]
        events = [b_orig(rng)] + a if f["h"] == "before" else a + [b_orig(rng)]
    return events, 0, [0, 1] if f["both"] else [0]


PLANS = {"AL1": plan_al1, "AL2": plan_al2, "AL3": plan_al3, "AL4": plan_al4}
H_GROUP = {"AL2": ("adjacent", "filler"), "AL3": ("other_obj",), "AL4": ("adjacent", "filler")}


def cell_specs(rng, cell):
    J = joint(rng, N_CELL, vtype=list(TRAIN_VTYPES), place=list(PLACE4))
    q = bal(rng, N_CELL, ["full", "head"])
    kA, both = bal(rng, N_CELL, [0, 1]), bal(rng, N_CELL, [True, False])
    grp = [i for i, j in enumerate(J) if j["place"] in H_GROUP.get(cell, ())]
    h = dict(zip(grp, bal(rng, len(grp), ["before", "after"]))) if grp else {}
    out = []
    for i, j in enumerate(J):
        f = dict(kA=kA[i] if cell == "AL1" else None, both=both[i] if cell in ("AL1", "AL4") else True,
                 h=h.get(i))
        events, asked, aliased = PLANS[cell](rng, j["place"], f)
        out.append(dict(cell=cell, vtype=j["vtype"], placement=j["place"], q_form=q[i], events=events, asked=asked,
                        aliased=aliased, factors=f, pre=rng.randint(0, 3)))
    return out


def render_al(rng, spec):
    P = POOLS_E[spec["vtype"]]
    objs = pick_objects(rng, P["objects"], 2)
    names = rng.sample(AL_SURNAMES, len(spec["aliased"]))
    aliases = [None, None]
    for o, s in zip(spec["aliased"], names):
        aliases[o] = f"{rng.choice(HONORIFICS)} {s}"
    banned = object_words(objs) | {w for a in aliases if a for w in words(a)}
    fillers = [f for f in filler_pool("default") if _filler_ok(f, banned)]
    rng.shuffle(fillers)
    events = spec["events"]
    vals = rng.sample(P["values"], len(events))
    T = _Templates(rng)
    turns, stmts = [], []

    def fill(n):
        if n > len(fillers):
            raise RuntimeError("filler pool exhausted")
        for _ in range(n):
            turns.append(tuple(fillers.pop()))

    fill(spec["pre"])
    for i, (e, v) in enumerate(zip(events, vals)):
        if i > 0:
            fill(e["gap"] if e["gap"] is not None else rng.randint(0, 2))
        phrase, head = objs[e["obj"]]
        a = aliases[e["obj"]]
        if e["role"] == "orig":
            term, key, pool = phrase + (f" {P['alias_join']} {a}" if a else ""), "orig", P["orig"]
        elif e["ref"] in ("full", "head"):
            term, key, pool = (phrase if e["ref"] == "full" else head), "corr", P["corr"]
        elif e["ref"] == "alias":
            term, key, pool = None, "alias_corr", P["alias_corr"]
        else:
            raise ValueError(e["ref"])
        j, tpl = T.take(key, pool)
        user = with_marker(cap(tpl.format(o=term, v=v, a=a)), e["marker"])
        _, atpl = T.take("ack_" + e["role"], ACKS_E[e["role"]])
        turns.append((user, cap(atpl.format(v=v))))
        stmts.append(dict(turn=len(turns) - 1, obj=e["obj"], value=v, role=e["role"], ref=e["ref"],
                          marker=e["marker"], lure=False, tpl=[key, j], alias=a if e["ref"] == "alias" else None))
    fill(D)
    asked = spec["asked"]
    term = objs[asked][0] if spec["q_form"] == "full" else objs[asked][1]
    question = cap(T.take("ask", P["ask"])[1].format(o=term))
    prefix = cap(T.take("pre", P["pre"])[1].format(o=term))
    about = [s for s in stmts if s["obj"] == asked]
    cands = []
    for s in stmts:
        if s["value"] not in cands:
            cands.append(s["value"])
    last = about[-1]
    meta = dict(placement=spec["placement"], latest_ref="orig" if last["role"] == "orig" else last["ref"],
                **{k: v for k, v in spec["factors"].items() if v is not None})
    return dict(family="AL", cell=spec["cell"], seed=SEED, pool_key=spec["vtype"], vtype=spec["vtype"],
                objects=[list(o) for o in objs], asked=asked, aliases=aliases,
                aliases_all=[a for a in aliases if a], alias=aliases[asked],
                alias_obj=asked if aliases[asked] else None, turns=[list(t) for t in turns], stmts=stmts,
                question=question, prefix=prefix, gold=last["value"], candidates=cands, values=list(P["values"]),
                k=sum(1 for s in about if s["role"] == "corr"), d=D, pre=spec["pre"], n_obj=2, q_form=spec["q_form"],
                meta=meta)


def build():
    rng = random.Random(SEED)
    items = []
    for cell in CELLS:
        for spec in cell_specs(rng, cell):
            items.append(render_al(rng, spec))
    for i, it in enumerate(items):
        it["idx"] = i
    return items


def dumps(items):
    return "".join(json.dumps(it, sort_keys=True) + "\n" for it in items)


def load(path=ITEMS_PATH, check_sha=True):
    """the written items; refuses a file whose sha256 differs from al_items.sha256."""
    data = open(path, "rb").read()
    if check_sha:
        want = open(SHA_PATH).read().split()[0]
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            raise SystemExit(f"refusing: {path} sha256 {got} != recorded {want}")
    return [json.loads(ln) for ln in data.decode().splitlines() if ln.strip()]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    import checks_al as CA
    items = build()
    again = build()
    if dumps(items) != dumps(again):
        print("FAIL: the draw is not deterministic")
        return 1
    fails = CA.run_all(items)
    if fails:
        print(f"\nRESULT: {len(fails)} FAILED: {fails}")
        return 1
    if "--write" in argv:
        os.makedirs(AL_DIR, exist_ok=True)
        if os.path.exists(ITEMS_PATH):
            print(f"REFUSED: {ITEMS_PATH} exists (the items are written once)")
            return 1
        text = dumps(items)
        with open(ITEMS_PATH, "w") as f:
            f.write(text)
        h = hashlib.sha256(text.encode()).hexdigest()
        with open(SHA_PATH, "w") as f:
            f.write(f"{h}  al_items.jsonl\n")
        print(f"WROTE {ITEMS_PATH}: {len(items)} items, sha256 {h}")
    print("\nRESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
