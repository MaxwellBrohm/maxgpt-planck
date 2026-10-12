"""v3_shotgun on OOD-H (rc12/notes.txt STEP 14S, cuts audit S11; prereg s14: OOD-H is graded with "the same graders
as RC-12"). Until 2026-10-10 graders.g_val skipped the clause on any probe carrying pool_values, which only COMPOSE
carries on dev and every OOD-H probe carries (build.norm_probe), so the SHOTGUN cheater scored 0.59 on OOD-H and 0.00
on RC-12. graders.shotgun_applies now exempts COMPOSE only (pool compose_*), and (review D1) a stale value named only in
a change phrase, the gold asserted, is not counted (graders.shotgun_named). Synthetic OOD-H world (tests/fixture.py)
and the real dev records; no sealed file is read and no model is loaded.
Run: python3 -B oodh/tests/test_shotgun.py   (mutants: oodh/tests/mutation_shotgun.py)"""
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
OODH = os.path.dirname(HERE)
REPO = os.path.dirname(OODH)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != OODH]
import select  # noqa: E402,F401  the stdlib module first: oodh/ holds a select.py of its own
sys.path[:0] = [OODH, HERE]

import build as B  # noqa: E402  (puts rc12/ on the path)
import gate as GT  # noqa: E402
import graders as G  # noqa: E402
import play as P  # noqa: E402
from test_build import TESTS, main, test, world  # noqa: E402

TESTS.clear()
DEV = os.path.join(REPO, "rc12", "dev", "rc12_dev.jsonl")


def oodh_recs():
    root, _ = world()
    return B.build(B.load_items(root), 9)[0]


def dev_recs():
    with open(DEV) as fh:
        return [json.loads(line) for line in fh]


def ideal_replies(rec):
    """the IDEAL transcript: history replies on thread turns, the probe ideals on probe turns."""
    by = {p["turn"]: p for p in rec["probes"]}
    return [by[t["i"]]["ideal"] if t["i"] in by else t["history_reply"] for t in rec["turns"]]


def grade_with(rec, p, reply):
    replies = ideal_replies(rec)
    replies[p["turn"] - 1] = reply
    return G.grade_probe(rec, p, replies, ["eos"] * len(replies))


@test
def fixture_has_both_kinds():
    """the fixture must hold VAL probes with 2+ candidates (where the clause can fire) and with 1 (where it cannot),
    each carrying pool_values, so the assertions below are not vacuous."""
    ps = [p for r in oodh_recs() for p in r["probes"] if p["grader"] == "VAL"]
    multi = [p for p in ps if len(p["candidates"]) >= 2]
    single = [p for p in ps if len(p["candidates"]) == 1]
    assert len(multi) >= 6 and len(single) >= 6, (len(multi), len(single))
    assert all(p["pool_values"] and p["pool"] is None for p in ps)


@test
def shotgun_fails_on_oodh():
    """a reply naming every candidate fails v3_shotgun on each 2+-candidate OOD-H probe; the ideal does not."""
    n = 0
    for rec in oodh_recs():
        for p in rec["probes"]:
            if p["grader"] != "VAL" or len(p["candidates"]) < 2:
                continue
            gold, others = p["gold"], [c for c in p["candidates"] if c != p["gold"]]
            listing = p["ideal"] + " " + " ".join(f"Not {c}." for c in others)
            res = grade_with(rec, p, listing)
            assert "v3_shotgun" in res["fails"] and not res["ok"], (rec["id"], p["turn"], res)
            ideal = grade_with(rec, p, p["ideal"])
            assert ideal["ok"] and "v3_shotgun" not in ideal["fails"], (rec["id"], p["turn"], ideal)
            assert gold in p["candidates"]
            n += 1
    assert n >= 6, n


def stale_probes():
    out = [(rec, p) for rec in oodh_recs() for p in rec["probes"]
           if p["grader"] == "VAL" and p.get("stale") and len(p["candidates"]) == 2]
    assert len(out) >= 5, len(out)                   # H-CORR: the gold and the one stale value
    return out


@test
def change_phrase_is_not_a_shotgun():
    """review D1 (notes STEP 14S): a stale value named only in a change phrase, the gold asserted, is not counted, so
    the right H-CORR answer passes on OOD-H as the same shape passes on RC-12 (shotgun_named)."""
    for rec, p in stale_probes():
        s, g = p["stale"][0], p["gold"]
        for txt in (f"You changed it from {s} to {g}.", f"You settled on {g}; {s} was the first plan."):
            res = grade_with(rec, p, txt)
            assert res["ok"] and not res["fails"], (rec["id"], p["turn"], txt, res)


@test
def stale_still_counts_outside_a_change_phrase():
    """the exemption is narrow: the stale value named bare (alone, or once more beside a change phrase), the gold not
    asserted, or the stale value bare and only the gold in a change phrase: each still names every candidate."""
    for rec, p in stale_probes():
        s, g = p["stale"][0], p["gold"]
        for txt in (f"It is {g} or {s}.", f"{s} is nice too. You changed it from {s} to {g}.",
                    f"Not {g}. You changed it from {s}.", f"{s}. You changed it from {g}."):
            assert "v3_shotgun" in grade_with(rec, p, txt)["fails"], (rec["id"], p["turn"], txt)


@test
def single_candidate_never_fires():
    for rec in oodh_recs():
        for p in rec["probes"]:
            if p["grader"] == "VAL" and len(p["candidates"]) == 1:
                res = grade_with(rec, p, p["ideal"] + " " + p["ideal"])
                assert "v3_shotgun" not in res["fails"], (rec["id"], p["turn"], res)


@test
def shotgun_cheater_on_oodh():
    """end to end through play.py and rc12/runner.py: the SHOTGUN cheater passes no 2+-candidate probe; IDEAL and
    TERSE still pass every probe (G1)."""
    recs = oodh_recs()
    sc = P.scores(P.play(recs, "SHOTGUN"), recs)
    ncand = {(r["id"], p["turn"]): len(p["candidates"]) for r in recs for p in r["probes"]}
    multi = [x for x in sc["probes"] if x["grader"] == "VAL" and ncand[(x["id"], x["turn"])] >= 2]
    assert multi and not any(x["ok"] for x in multi), multi
    assert all("v3_shotgun" in x["fails"] for x in multi), multi
    rows = {n: P.play(recs, n) for n in GT.REFERENCE}
    assert GT.reference_checks(recs, rows) == []


@test
def compose_keeps_its_exemption():
    """dev COMPOSE: a right answer names both holders and must pass; a non-COMPOSE dev probe with the same listing
    shape fails (the clause is on there)."""
    recs = dev_recs()
    seen_c = seen_v = 0
    for rec in recs:
        for p in rec["probes"]:
            if p["grader"] != "VAL" or len(p.get("candidates") or []) < 2 or p["gold"] not in p["candidates"]:
                continue
            o = [c for c in p["candidates"] if c != p["gold"]][0]
            if str(p.get("pool") or "").startswith("compose_") and seen_c < 16:
                adj = p["question"].rstrip("?").split()[-1]
                res = G.g_val(f"Your {p['gold']} is {adj} than your {o}.", "eos", [], rec, p)
                assert res["ok"] and "v3_shotgun" not in res["fails"], (rec["id"], p["turn"], res)
                seen_c += 1
            elif not p.get("pool_values") and seen_v < 40:
                res = G.g_val(f"{p['ideal']} " + " ".join(f"Not {c}." for c in p["candidates"] if c != p["gold"]),
                              "eos", [], rec, p)
                assert "v3_shotgun" in res["fails"], (rec["id"], p["turn"], res)
                seen_v += 1
    assert seen_c == 16 and seen_v == 40, (seen_c, seen_v)


@test
def dev_scope_unchanged():
    """on every dev probe the new scope equals the old one (no pool_values), so no dev verdict moves."""
    n = 0
    for rec in dev_recs():
        for p in rec["probes"]:
            assert G.shotgun_applies(p) == (not p.get("pool_values")), (rec["id"], p["turn"])
            n += 1
    assert n >= 1404, n                              # every dev probe (640 records)


@test
def scope_rule():
    assert not G.shotgun_applies(dict(pool="compose_pets", pool_values=["cat"]))
    assert G.shotgun_applies(dict(pool=None, pool_values=["Porto", "Lisbon"]))
    assert G.shotgun_applies(dict(pool="city"))


if __name__ == "__main__":
    sys.exit(main())
