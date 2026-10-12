"""RC-12 DEV generator tests, part 4: the step 5 audit's balance guarantees (notes.txt STEP 5), read from the TURNS
where possible, not only from the design metadata (mutation_audit.py showed a metadata-only check survives a
generator that writes the wrong turns). Called from test_gens_fam.run()."""
import re
from collections import Counter

import fam_corr as FC
from test_gens_fam import check, fam, main_p

MARKS = ("actually", "update", "changed", "switched", "make that", "scratch", "oh wait", "sorry", "change of plan")


def tail_sig(r):
    """the events after A's latest correction, read from the turns: Lm / Lp lure (marked / plain), B0, B1."""
    p = main_p(r)
    out = []
    for t in r["turns"]:
        if t["kind"] in "SCL" and p["src"][0] < t["i"] < p["turn"]:
            marked = any(w in t["text"].lower() for w in MARKS)
            out.append(("Lm" if marked else "Lp") if t["kind"] == "L" else "B0" if t["kind"] == "S" else "B1")
    return tuple(out)


def len_rank(r):
    """long / short / mid: where the gold statement's word count sits among the probe's value turns."""
    p = main_p(r)
    vt = [t for t in r["turns"] if t["kind"] in "SCL" and t["facts"] and t["i"] < p["turn"]]
    n = {t["i"]: len(re.findall(r"[A-Za-z0-9']+", t["text"])) for t in vt}
    g, others = n[p["src"][0]], [v for k, v in n.items() if k != p["src"][0]]
    return "long" if g > max(others) else "short" if g < min(others) else "mid"


def x_turn(r):
    return [p["turn"] for p in r["probes"] if p["kind"] == "X"][0]


def t_audit(recs):
    main = [r for r in fam(recs, "RECALL") if r["cell"] != "abstain"]
    for b in ("d1-3", "d4-7", "d8-11"):          # gold length rank balanced inside each bin
        got = Counter(r["meta"]["len_rank"] for r in main if r["cell"] == b)
        check(sorted(got.values()) == [5, 5, 6] and set(got) == {"long", "mid", "short"}, f"RECALL {b} len {got}")
    ab = fam(recs, "RECALL", "abstain")
    for r in ab:
        check((x_turn(r) < main_p(r)["turn"]) == r["meta"]["x_first"], f"{r['id']} x_first")
    xf = Counter(x_turn(r) < main_p(r)["turn"] for r in ab)
    check(xf == Counter({True: 6, False: 6}), f"RECALL abstain X/P order (audit: 6/6) {xf}")
    for cell in ("U-diff", "U-same", "C_noupd", "C_twoslot"):
        rs = fam(recs, "CORR", cell)
        for r in rs:
            p = main_p(r)
            lures = [t for t in r["turns"] if t["kind"] == "L"]
            check(all(t["facts"][0]["holder"] != "user" for t in lures), f"{r['id']} CORR lure holder")
            if cell == "C_noupd":
                continue
            pre = [t for t in lures if t["i"] < p["src"][0]]
            check(len(pre) <= 1 and all(t["i"] < min(x["i"] for x in r["turns"] if x["kind"] in "SC") for t in pre),
                  f"{r['id']} CORR lure placement")
            after = [t for t in r["turns"] if t["kind"] in "SCL" and p["src"][0] < t["i"] < p["turn"]]
            check(len(after) == r["meta"]["t_after"], f"{r['id']} t_after")
        if cell.startswith("U"):                 # the tails as written equal the searched plan (fam_corr.U_TAILS)
            got = Counter((r["meta"]["b_place"], tail_sig(r)) for r in rs)
            want = Counter({(b, t): n for b, rows in FC.U_TAILS[cell].items() for t, n in rows})
            check(got == want, f"CORR {cell} tails after A's latest {sorted(got.items())}")
            ends = sum(not tail_sig(r) for r in rs)  # ending on A's latest: U-same never (G4), U-diff <= 2
            check(ends <= (0 if cell == "U-same" else 2), f"CORR {cell} ends on A's latest in {ends} items")
        if cell == "C_twoslot":
            check(set(Counter((r["meta"]["layout"], r["meta"]["m_b"]) for r in rs).values()) == {4},
                  "C_twoslot layout x m (audit)")
        if cell == "C_noupd":                    # another holder's statement about A's object, never the last value
            for r in rs:                         # turn; an anchor is never the last value turn either
                vt = [t for t in r["turns"] if t["kind"] in "SCL" and t["facts"]]
                el = [t for t in vt if t["kind"] == "L" and t["facts"][0]["object"] == r["meta"]["objA"]]
                check(any(t is not vt[-1] for t in el), f"{r['id']} C_noupd echo lure")
                check(not r["meta"]["anchor"] or vt[-1]["kind"] == "L", f"{r['id']} C_noupd anchor is the last turn")
        ranks = Counter(len_rank(r) for r in rs)  # the gold statement's length rank, read from the turns
        check(ranks["short"] <= 6 and ranks["long"] <= 6, f"CORR {cell} gold length ranks {dict(ranks)}")
    rs = [r for r in fam(recs, "TWOHOP") if r["cell"] != "COMPOSE"]
    got = Counter((r["cell"], any(t["kind"] == "S" and t["facts"][0]["value"] == main_p(r)["echo_value"]
                                  for t in r["turns"] if t["facts"])) for r in rs)
    check(got == Counter({(c, g): 4 if g else 8 for c in ("door", "day", "city", "month") for g in (True, False)}),
          f"TWOHOP echo chain gold / lure 4 / 8 per schema (audit) {got}")
    rs = fam(recs, "TOPIC")
    for r in rs:
        p, m = main_p(r), r["meta"]
        st = [t["i"] for t in r["turns"] if t["facts"] and t["facts"][0]["role"] in ("open", "closed_inline",
                                                                                     "closed_intro")]
        ment = [t["facts"][0]["value"] for t in r["turns"] if t["facts"] and t["kind"] in "SC"]
        check(not (ment[0] == p["gold"] == ment[-1]), f"{r['id']} gold is both the first and the last mention")
        upd = [t for t in r["turns"] if t["facts"] and t["facts"][0]["role"].endswith("_later")]
        check(len(upd) == m["n_later"] and all(t["i"] > max(st) for t in upd), f"{r['id']} later updates")
        check(all((t["facts"][0]["value"] == p["gold"]) == (t["facts"][0]["role"] == "open_later") for t in upd),
              f"{r['id']} later update roles")   # an update of the open project never closes it
    check(Counter(r["meta"]["len_rank"] for r in rs) == Counter(long=16, mid=16, short=16), "TOPIC len ranks")
    ol = Counter((sum(t["facts"][0]["role"].endswith("_later") for t in r["turns"] if t["facts"]),
                  any(t["facts"] and t["facts"][0]["role"] == "open_later" for t in r["turns"])) for r in rs)
    check(ol == Counter({(0, False): 16, (1, True): 4, (1, False): 12, (2, True): 10, (2, False): 6}),
          f"TOPIC open-project later updates {ol} (audit plan)")
    check(Counter(r["meta"]["n_later"] for r in rs) == Counter({0: 16, 1: 16, 2: 16}), "TOPIC updates 16/16/16")
