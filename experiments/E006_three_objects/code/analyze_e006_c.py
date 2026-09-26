"""E006 analysis: Q-H5 and the cost tables (notes.txt Q-H5, Q-CHAT "Cost"). CPU only, no model.
q_h5(out_dir) reads BIG H5 (plain, LIK and GEN) for arms P and C, the references e004w (L4) and e005w, and H5L, and
applies rules_e006.q_h5; everything "reported next to it" is computed here too. cost(out_dir, arm) marks every
family of draw 4004 and BIG where the arm is below C (rules_e006.cost_mark)."""
import analyze_e006_b as B
import rules_e006 as RL
import rules_e004 as RU

D4004_INTRO = {1: 27, 2: 20, 3: 17}          # the eval draw's H5 introduction-order mix (E005 audit Q5)
E004_STORED_H5, E005_STORED_H5 = 0.756, 0.656
U_FAMS = ["H1", "H2", "H3", "H4", "H6", "H7"]


def tags(prefix):
    return {s: f"{prefix}{s}" for s in B.SEEDS}


def get(out_dir, prefix, set_name, fam, part, render="plain"):
    return B.arm(out_dir, tags(prefix), set_name, fam, render, part)


def rows_of(out_dir, prefix, set_name):
    return {s: B.rows(out_dir, t, set_name) for s, t in tags(prefix).items()}


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


def q_h5(out_dir):
    L, res = [], {}
    X = {(a, p): get(out_dir, a, "big", "H5", p) for a in ("C", "P", "e004w", "e005w") for p in ("LIK", "GEN")}
    H = {(a, p): get(out_dir, a, "h5l", "H5L", p) for a in ("C", "P") for p in ("LIK", "GEN")}
    ok = all(B.complete(X[(a, p)]) for a in ("C", "P") for p in ("LIK", "GEN"))
    d = {p: RL.paired(X[("P", p)], X[("C", p)]) if ok else None for p in ("LIK", "GEN")}
    l4ok = all(B.complete(X[("e004w", p)]) for p in ("LIK", "GEN"))
    p_l4 = {p: RL.paired(X[("P", p)], X[("e004w", p)]) for p in ("LIK", "GEN")} if ok and l4ok else None
    gap = B.pooled(X[("e004w", "LIK")]) - B.pooled(X[("C", "LIK")]) if ok and l4ok else None
    dev = (B.pooled(X[("C", "LIK")]) - B.pooled(X[("e005w", "LIK")])) if ok and B.complete(X[("e005w", "LIK")]) else None
    h5l = {p: RL.paired(H[("P", p)], H[("C", p)]) for p in ("LIK", "GEN")
           if B.complete(H[("P", p)]) and B.complete(H[("C", p)])}
    lab = RL.q_h5(d["LIK"], d["GEN"], p_l4, gap, dev, h5l)
    res.update(label=lab, d=d, p_minus_l4=p_l4, l4_minus_c_lik=gap, dev_shift_lik=dev, h5l=h5l)
    L.append(f"Q-H5 (BIG H5, plain, P - C): {lab['label']}" + (f" [{', '.join(lab['flags'])}]" if lab["flags"] else "")
             + (f"; {lab['suffix']}" if lab["suffix"] else ""))
    for p in ("LIK", "GEN"):
        L.append("   " + RL.sentence(f"P - C, BIG H5 {p}", d[p]))
        if p_l4:
            L.append("   " + RL.sentence(f"P - e004w (L4), BIG H5 {p}", p_l4[p]))
        if p in h5l:
            L.append("   " + RL.sentence(f"P - C, H5L {p}", h5l[p]))
    L.append(f"   L4 - C (BIG H5 LIK) {fmt(gap)}; training-device shift C - e005w {fmt(dev)}")
    for a in ("C", "P", "e004w", "e005w"):
        L.append(f"   BIG H5 pooled {a}: LIK {fmt(B.pooled(X[(a, 'LIK')]))} GEN {fmt(B.pooled(X[(a, 'GEN')]))} "
                 f"(seeds {sorted(X[(a, 'LIK')])})")
    for p in ("LIK", "GEN"):
        per = {s: sum(X[("P", p)][s].values()) / len(X[("P", p)][s]) - sum(X[("C", p)][s].values()) /
               len(X[("C", p)][s]) for s in B.SEEDS if s in X[("P", p)] and s in X[("C", p)]}
        L.append(f"   per seed P_s - C_s {p}: " + ", ".join(f"s{s} {v:+.3f}" for s, v in per.items())
                 + f" ({sum(v > 0 for v in per.values())} of {len(per)} > 0)")
    for a in ("C", "P", "e004w", "e005w"):
        e = get(out_dir, a, "e004", "H5", "LIK")
        L.append(f"   D4004 H5 LIK pooled {a}: {fmt(B.pooled(e))} (>= E004's stored {E004_STORED_H5}: "
                 f"{B.pooled(e) is not None and B.pooled(e) >= E004_STORED_H5}); E005 Mac stored {E005_STORED_H5}")
    res["reported"] = reported(out_dir, L)
    return res, L


def reported(out_dir, L):
    rep = {}
    for a in ("C", "P", "e004w", "e005w"):
        rb, re_ = rows_of(out_dir, a, "big"), rows_of(out_dir, a, "e004")
        for name, rs in (("BIG", rb), ("D4004", re_)):
            sp = B.split(rs, "H5", "intro")
            oi = B.split(rs, "H5", "other_ind")
            L.append(f"   {a} {name} H5 LIK by intro order " + ", ".join(f"{k}: {fmt(r / n)} ({n})" for k, (r, n) in sp.items())
                     + "; other object's indirect after the asked latest " +
                     ", ".join(f"{k}: {fmt(r / n)} ({n})" for k, (r, n) in oi.items()))
            if name == "BIG" and sp:
                rep[f"{a} BIG H5 reweighted to the D4004 intro mix"] = B.reweighted_intro(sp, D4004_INTRO)
                nl = B.split(rs, "H5", "nl_right")
                L.append(f"   {a} BIG H5 LIK reweighted to D4004's intro mix {fmt(rep[f'{a} BIG H5 reweighted to the D4004 intro mix'])}"
                         "; by rule NL right/wrong " + ", ".join(f"{k}: {fmt(r / n)} ({n})" for k, (r, n) in nl.items()))
            for part in ("LIK", "GEN"):
                n, roles, last = B.wrong(rs, "H5", part)
                L.append(f"   {a} {name} H5 wrong {part} picks: {n}, {roles}; the dialogue's last statement {last}")
        rh = rows_of(out_dir, a, "h5l")
        for set_name, fam, rs in (("big", "H5", rb), ("big", "C_noupd", rb), ("big", "C_twoslot", rb),
                                  ("h5l", "H5L", rh)):
            rc = B.recency(rs, fam)
            if rc:
                L.append(f"   {a} recency index {set_name} {fam}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float)
                                                                            else f"{k} {v}" for k, v in rc.items()))
        gl = {f: B.split(re_, f, "gold_last") for f in U_FAMS}
        tot = {k: [sum(gl[f].get(k, (0, 0))[i] for f in U_FAMS) for i in (0, 1)] for k in (True, False)}
        L.append(f"   {a} D4004 U families LIK by gold-last: " + ", ".join(f"{k}: {fmt(r / n) if n else '-'} ({n})"
                                                                         for k, (r, n) in tot.items()))
    for fam in ("C_noupd", "C_twoslot"):
        for p in ("LIK", "GEN"):
            x, y = get(out_dir, "P", "big", fam, p), get(out_dir, "C", "big", fam, p)
            if B.complete(x) and B.complete(y):
                rep[f"P - C BIG {fam} {p}"] = RL.paired(x, y)
                L.append("   control (secondary) " + RL.sentence(f"P - C, BIG {fam} {p}", rep[f"P - C BIG {fam} {p}"]))
            for a in ("C", "P"):
                n, roles, last = B.wrong(rows_of(out_dir, a, "big"), fam, p)
                L.append(f"   {a} BIG {fam} wrong {p} picks: {n}, {roles}; the dialogue's last statement {last}")
    L.append("   note: on H5, C_noupd and C_twoslot the gold is never the dialogue's last statement, so a weaker recency "
             "prior alone raises all three; rule NL scores .94 on both controls (they cannot discriminate), H5L can")
    return rep


def cost(out_dir, arm):
    """-> ({(set, family, part, render): (d, lo, hi, k, n)}, lines) for arm - C; the plain rows are the
    pre-registered cost marks, the chat rows are reported (EXTRA REPORT 1)."""
    out, L = {}, []
    for render in ("plain", "chat"):
        for set_name, fams in (("e004", RU.PASS + ["ID"]), ("big", ["H5", "C_noupd", "C_twoslot"]), ("h5l", ["H5L"])):
            for f in fams:
                for p in ("LIK", "GEN"):
                    x, y = get(out_dir, arm, set_name, f, p, render), get(out_dir, "C", set_name, f, p, render)
                    r = RL.paired(x, y) if B.complete(x) and B.complete(y) else None
                    out[(set_name, f, p, render)] = r
                    L.append(f"   {arm} - C {set_name} {f} {p} {render}: " + ("not read" if r is None else
                             f"{r[0]:+.3f} CI {r[1]:+.3f} to {r[2]:+.3f} [{RL.cost_mark(r)}]"))
    return out, L


def companion(out_dir, arm):
    """the pass rule's H5 and control cells read on BIG (critique SHOULD 11; reported, not ruled): per seed, LIK and
    GEN >= 0.8 on BIG H5, C_noupd, C_twoslot."""
    L = []
    for f in ("H5", "C_noupd", "C_twoslot"):
        cells = {p: get(out_dir, arm, "big", f, p) for p in ("LIK", "GEN")}
        row = []
        for s in B.SEEDS:
            v = [sum(cells[p][s].values()) / len(cells[p][s]) if s in cells[p] else None for p in ("LIK", "GEN")]
            row.append(f"s{s} " + "/".join(fmt(x) for x in v) + (" pass" if all(x is not None and x >= RU.THRESH
                                                                              for x in v) else ""))
        L.append(f"   {arm} BIG {f} (LIK/GEN, bar 0.8): " + ", ".join(row))
    return L
