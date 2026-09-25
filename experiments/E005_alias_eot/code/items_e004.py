"""E004 held-out eval families (notes (b)): the public entry point. build(seed, family, n) plans and renders n
items of one family; draw(seed) builds every family at the notes' size for that draw.
  eval  seed 4004, 64 per family (576 pass-rule items)
  dev   seed 4104, 32 per family (LR choice only)
  probe seed 4204, 16 per family (in-training likelihood probe)
Each family has its own random stream, Random("E004:<seed>:<family>"), so adding a family never moves another.
Plans: items_plan.py (U families), items_plan_b.py (H5, ID, controls); rendering: items_render.py; pools:
pools_eval*.py, fillers_eval.py; held-out vocabulary: heldout_e004.py."""
from items_plan import plan_u
from items_plan_b import plan_h5, plan_id, plan_c_twoslot, plan_c_noupd
from items_render import render, seeded, prompt

PASS_FAMILIES = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "C_noupd", "C_twoslot"]
FAMILIES = PASS_FAMILIES + ["ID"]
DRAWS = {"eval": (4004, 64), "dev": (4104, 32), "probe": (4204, 16)}
AXIS = {"H1": "frame echo on a stale correction (training: 0 echoes); alias",
        "H2": "frame echo on the original only; alias", "H3": "k = 4-5 (training k <= 3)",
        "H4": "d = 20 (training d <= 10)", "H5": "3 objects (training <= 2)",
        "H6": "new object types; value types sport and number", "H7": "small talk / creative fillers",
        "C_noupd": "control, one cell per axis + in range", "C_twoslot": "control, one cell per axis + in range",
        "ID": "diagnostic: training structures, eval wording"}


def _plans(rng, family, n):
    if family == "H5":
        return plan_h5(rng, n)
    if family == "ID":
        return plan_id(rng, n)
    if family == "C_twoslot":
        return plan_c_twoslot(rng, n)
    if family == "C_noupd":
        return plan_c_noupd(rng, n)
    return plan_u(rng, family, n)


def build(seed, family, n):
    rng = seeded(seed, family)
    items = []
    for i, spec in enumerate(_plans(rng, family, n)):
        spec["pre"] = rng.randint(0, 3)
        item = render(rng, spec)
        item["idx"], item["seed"] = i, seed
        items.append(item)
    return items


def draw(name="eval", families=FAMILIES):
    seed, n = DRAWS[name]
    return {f: build(seed, f, n) for f in families}


__all__ = ["build", "draw", "prompt", "FAMILIES", "PASS_FAMILIES", "DRAWS", "AXIS"]
