"""E006 per-arm readings (notes.txt PER-ARM READINGS). CPU only, no model. The pass rule and the alias and stopping
readings are E004's and E005's (rules_e004 / rules_e005 / evidence_e005 / analyze_e004, as copied, unedited):
  reading     rules_e005.reading on draw 4004 plain (E004 (c): 9 families, LIK and GEN >= 0.8, 3 of 5 planned seeds,
              an incomplete seed fails; PASS, PARTIAL-G, PARTIAL-X, FAIL in that order)
  any_seed    the variant in which a LIK failure on H1, H2 or a control on ANY seed makes the reading not partial
              (printed next to the reading when it differs)
  loo         the label with each seed left out (4 planned seeds, as the E005 audit did)
  flips       the fewest item flips (right <-> wrong, in at most 3 cells) that would change the label
  alias       rules_e005.alias_reading on the 42 H1+H2 alias items; stopping: rules_e005.stopping"""
from itertools import combinations

import analyze_e004 as A
import evidence_e005 as V
import rules_e004 as RU
import rules_e005 as R5

N_ITEMS, N_PLANNED = 64, 5
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def run_block(out_dir, tag):
    """analyze_e005.run_block's content for one run (cells, evidence, alias, chat), or None."""
    cells = A.run_cells(out_dir, MODEL, tag)
    if cells is None:
        return None
    stem = f"{A.slug(MODEL)}__{tag}__"
    lik, gen = A.load(f"{out_dir}/{stem}e004__plain.jsonl"), A.load(f"{out_dir}/{stem}gen_e004__plain.jsonl")
    ev = V.evidence(V.join(lik, gen))
    chat_gen = A.load(f"{out_dir}/{stem}gen_e004__chat.jsonl")
    chat = V.chat_cells(chat_gen, gen) if chat_gen and len(chat_gen) == A.N_EVAL else None
    al = ev["alias"]["all"]
    return {"cells": cells, "evidence": ev, "alias": (al["LIK"]["acc"], al["GEN"]["acc"]), "chat": chat}


def label(seed_cells, n_planned=N_PLANNED):
    return RU.reading(list(seed_cells), n_planned)["label"]


def any_seed(seed_cells, lab):
    if lab not in ("PARTIAL-G", "PARTIAL-X"):
        return lab
    hit = [i + 1 for i, c in enumerate(seed_cells) if c is not None and
           not all(RU.ok((c.get("LIK") or {}).get(f)) for f in ["H1", "H2"] + RU.CONTROLS)]
    return f"FAIL (any-seed variant: LIK failure on H1, H2 or a control on seed(s) {hit})" if hit else lab


def loo(seed_cells):
    return {f"without s{i + 1}": label([c for j, c in enumerate(seed_cells) if j != i], N_PLANNED - 1)
            for i in range(len(seed_cells))}


def _margin(acc):
    k = round(acc * N_ITEMS)
    return (k - 51) if RU.ok(acc) else (52 - k)


def flips(seed_cells, max_cells=3, max_margin=8):
    """-> (fewest item flips, [(seed, part, family)]) that change the label, or (None, []) beyond the search."""
    base = label(seed_cells)
    cand = []
    for i, c in enumerate(seed_cells):
        if c is None:
            continue
        for part in ("LIK", "GEN"):
            for f in RU.PASS:
                x = c[part].get(f)
                if x is not None and _margin(x) <= max_margin:
                    cand.append((_margin(x), i, part, f))
    best = (None, [])
    for size in range(1, max_cells + 1):
        for combo in combinations(cand, size):
            cost = sum(m for m, *_ in combo)
            if best[0] is not None and cost >= best[0]:
                continue
            mod = [None if c is None else {p: dict(c[p]) for p in ("LIK", "GEN")} for c in seed_cells]
            for _, i, part, f in combo:
                mod[i][part][f] = 51 / N_ITEMS if RU.ok(seed_cells[i][part][f]) else 52 / N_ITEMS
            if label(mod) != base:
                best = (cost, [(i + 1, part, f) for _, i, part, f in combo])
    return best


def arm_readings(out_dir, tags):
    """tags: [tag of seed 1, ..., seed 5] -> dict of every per-arm reading (NOT_SCORED if no seed is complete)."""
    runs = [run_block(out_dir, t) for t in tags]
    cells = [r["cells"] if r else None for r in runs]
    if all(c is None for c in cells):
        return {"label": "NOT_SCORED", "runs": runs}
    rd = R5.reading(cells, N_PLANNED, [c["ID_LIK"] if c else None for c in cells], MODEL,
                    [r["evidence"] if r else None for r in runs])
    alias = R5.alias_reading([r["alias"] if r else None for r in runs], N_PLANNED)
    stop = R5.stopping([(r["chat"] and {f: v.get("strict") for f, v in r["chat"].items()}, r["cells"]["GEN"])
                        if r and r["chat"] else None for r in runs], N_PLANNED)
    return {"label": rd["label"], "detail": rd.get("detail"), "sub": rd.get("sub", []),
            "any_seed": any_seed(cells, rd["label"]), "loo": loo(cells), "flips": flips(cells),
            "alias": alias, "stopping": stop, "cells": cells, "runs": runs}
