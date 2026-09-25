"""E005 decision rules (notes.txt PASS RULE, ALIAS READING, EXTRA REPORTS 3), pure functions: no files, no model.
The pass rule and the reading labels are E004's (rules_e004.py, unchanged, called here). What changes is what a
FAIL prints: E004's sub-readings were fixed interpretive strings triggered by any H1/H2/control LIK failure without
looking at a single item (the audit found both wrong). Here each E004 trigger is printed as the fact it checks,
followed by item evidence (evidence_e005): H1/H2 by reference form, the alias adjacency split, and what the wrong
answers picked. No interpretation is attached by the code.
  alias reading   per seed alias LIK and GEN over the 42 H1 + H2 alias items; HIGH = both >= 0.8, LOW = both
                  <= 0.5; DATA GAP if a strict majority of the PLANNED seeds is HIGH, REAL LIMIT if a majority is
                  LOW, else INCONCLUSIVE (a missing seed is neither)
  AL note         (the step-1 addition) with DATA GAP, if AL1+AL2+AL4 LIK < 0.5 on a majority of planned seeds:
                  "resolved by alias recency, not by linking"
  stopping        a seed "stops" if its chat strict GEN is within 0.10 of its plain GEN (|chat - plain| <= 0.10)
                  on every pass family; "stopping learned" if a majority of the planned seeds stop"""
import rules_e004 as RU

PASS, CONTROLS, THRESH = RU.PASS, RU.CONTROLS, RU.THRESH
HIGH, LOW = 0.8, 0.5
STOP_TOL = 0.10


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def rng(xs):
    xs = [x for x in xs if x is not None]
    return "-" if not xs else (f3(xs[0]) if len(xs) == 1 else f"{f3(min(xs))}-{f3(max(xs))}")


def lik_fail_counts(seed_cells, fams):
    return {f: sum(1 for c in seed_cells if c is not None and not RU.ok((c.get("LIK") or {}).get(f))) for f in fams}


def trigger_facts(seed_cells, n_planned, id_accs):
    """E004's four sub-reading triggers, each printed as the fact it checks (no interpretation)."""
    out = []
    if RU.lik_majority(seed_cells, n_planned, ["H1", "H2"] + CONTROLS, lambda xs: not all(RU.ok(x) for x in xs)):
        cnt = lik_fail_counts(seed_cells, ["H1", "H2"] + CONTROLS)
        out.append("LIK < 0.8 on H1, H2 or a control on a majority of seeds (seeds failing LIK: "
                   + ", ".join(f"{f} {n}" for f, n in cnt.items()) + ")")
    id_hi = RU.majority([RU.ok(x) for x in id_accs], n_planned)
    if id_hi and RU.lik_majority(seed_cells, n_planned, ["H1", "H2"], lambda xs: not all(RU.ok(x) for x in xs)):
        out.append(f"ID LIK >= 0.8 ({rng(id_accs)}) while H1 or H2 LIK < 0.8 on a majority of seeds")
    if RU.majority([not RU.ok(x) for x in id_accs], n_planned):
        out.append(f"ID LIK < 0.8 on a majority of seeds ({rng(id_accs)})")
    if RU.lik_majority(seed_cells, n_planned, RU.U_FAMS, lambda xs: all(RU.ok(x) for x in xs)) and \
            RU.lik_majority(seed_cells, n_planned, CONTROLS, lambda xs: not all(RU.ok(x) for x in xs)):
        out.append("every U family passes LIK but a control fails, on a majority of seeds")
    return out


def pooled_roles(evs, group, part):
    tot, lure, n = {}, 0, 0
    for ev in evs:
        w = (ev or {}).get("wrong", {}).get(group)
        if w:
            for k, v in w[part].items():
                tot[k] = tot.get(k, 0) + v
            lure += w[part + "_lure"]
            n += w[part + "_n_wrong"]
    return tot, lure, n


def evidence_lines(evs):
    """evs: one evidence_e005.evidence dict (or None) per seed -> printable item-evidence lines."""
    evs = [e for e in evs if e]
    if not evs:
        return ["no item evidence (no scored seed)"]
    out = []
    for f in ("H1", "H2"):
        parts = []
        for form in ("alias", "ell", "pron"):
            xs = [e["form"].get(f, {}).get(form, {}).get("LIK", {}).get("acc") for e in evs]
            parts.append(f"{form} {rng(xs)}")
        out.append(f"{f} LIK by the latest statement's reference form (range over seeds): " + ", ".join(parts))
    pl = []
    for p in ("adjacent", "filler", "B_sep"):
        xs = [e["alias"]["place"][p]["LIK"]["acc"] for e in evs]
        pl.append(f"{p} {rng(xs)} (n {evs[0]['alias']['place'][p]['LIK']['n']})")
    out.append("H1+H2 alias items by placement, LIK (range over seeds): " + ", ".join(pl))
    for g in ("H1/H2 alias", "H1/H2 not alias", "H5", "C_noupd", "C_twoslot"):
        for part in ("LIK", "GEN"):
            tot, lure, n = pooled_roles(evs, g, part)
            if n:
                out.append(f"wrong {part} picks, {g}, pooled over seeds: {n} wrong: "
                           + ", ".join(f"{k} {v}" for k, v in sorted(tot.items(), key=lambda t: -t[1]))
                           + f"; of them lures {lure}")
    return out


def reading(seed_cells, n_planned, id_accs=(), model_id="", evs=()):
    """E004's reading (labels unchanged); a FAIL's sub-readings are trigger facts + item evidence."""
    r = RU.reading(seed_cells, n_planned, id_accs, model_id)
    if r["label"] == "FAIL":
        r["sub"] = trigger_facts(list(seed_cells), n_planned, list(id_accs)) + evidence_lines(list(evs))
    return r


def alias_class(lik, gen):
    if lik is None or gen is None:
        return None
    if lik >= HIGH and gen >= HIGH:
        return "HIGH"
    if lik <= LOW and gen <= LOW:
        return "LOW"
    return "MID"


def alias_reading(per_seed, n_planned, al_per_seed=None):
    """per_seed: one (alias LIK, alias GEN) or None per planned seed; al_per_seed: one AL1+AL2+AL4 LIK or None."""
    cls = [alias_class(*x) if x else None for x in per_seed]
    if RU.majority([c == "HIGH" for c in cls], n_planned):
        label = "DATA GAP"
    elif RU.majority([c == "LOW" for c in cls], n_planned):
        label = "REAL LIMIT"
    else:
        label = "INCONCLUSIVE"
    note = None
    if label == "DATA GAP" and al_per_seed is not None and \
            RU.majority([x is not None and x < 0.5 for x in al_per_seed], n_planned):
        note = "resolved by alias recency, not by linking"
    return {"label": label, "classes": cls, "n_high": cls.count("HIGH"), "n_low": cls.count("LOW"), "note": note}


def seed_stops(chat_gen, plain_gen):
    """chat_gen, plain_gen: {family: strict GEN} of one seed -> True iff |chat - plain| <= 0.10 on every pass family."""
    if not chat_gen or not plain_gen:
        return False
    xs = [(chat_gen.get(f), plain_gen.get(f)) for f in PASS]
    return all(c is not None and p is not None and abs(c - p) <= STOP_TOL + 1e-9 for c, p in xs)


def stopping(per_seed, n_planned):
    """per_seed: one (chat GEN dict, plain GEN dict) or None per planned seed."""
    flags = [bool(x) and seed_stops(*x) for x in per_seed]
    return {"learned": RU.majority(flags, n_planned), "seeds_stop": flags}
