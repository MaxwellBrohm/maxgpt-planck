"""RC-12 grader fixtures for lock step A2c (2026-11-19, rc12/LOCK_STEPS.md M7; rc12/notes.txt STEP 16A2C, prereg s6
(xi)): on the LOOKUP family's X probe G-ABS's a3_value counts a pool value named in any case (graders.g_abs, anycase:
pools_vals.value_re(v, False)), so the lure in lower case ("the old one says hugo"), in capitals, in mixed case or as a
possessive fails as the lure as written always did. Built from REAL dev items with fixtures_graders.fx over EVERY
LOOKUP record (all 48 X probes and their P probes) and every ABS probe of another family; mutation_graders.build adds
the records it did not pick (records_and_fixtures).
  X fail   a not-found reply (A2b's cues and the older ABS_CUE) beside the lure in lower case, in capitals, in mixed
           case, as a possessive, at the reply's start, in quotes, in a reply written all in capitals; another value of
           the tables in lower case; a pool value outside the conversation in lower case. On the graders before A2c each
           passed on the rota and tour cells (capitalized values) and failed on menu (lower-case dishes, matched in any
           case since the T0 audit)
  X pass   right not-found replies in capitals and in lower case (the key and the table words are no pool value), and
           one whose words hold the lure inside a longer word ("prehugox", built from the record's own lure, so it
           holds on any content; a first version used "wanders" and "scurry", which hold only dev's Anders and curry,
           and the renamed-content dry run caught it): the any-case match keeps whole-word boundaries
  P        pins: G-VAL is unchanged by A2c. A gold in lower case or capitals fails on a capitalized value (v2_gold,
           prereg s6 v2: capitalized values as written) and passes on the menu's lower-case values; the lure in lower
           case beside the right gold passes on rota and tour (no gain: P needs the gold as written) and fails on menu
  scope    every ABS probe of another family (RECALL abstain) whose pool holds a capitalized value: that value in
           lower case beside an abstain cue still passes there, as before A2c (prereg s6 (xi): only LOOKUP's X probe
           reads values in any case; the shared a3 elsewhere is a recorded limit), so a widened scope is caught; and,
           whatever a draw's RECALL pools hold, a copy of the first LOOKUP record of each cell relabelled RECALL
           (marked synthetic; the conversation tests skip it) whose X reply names the lure in lower case beside a cue
           passes (a2c_scope_relabelled); and a second copy relabelled OODH, OOD-H's family (sealed Part 1 is graded
           by these graders and no one may inspect it before the lock), passes the same reply, so a scope widened to
           OOD-H alone is caught too (a2c_scope_relabelled_oodh; the A2c finish, 2026-11-21)"""
import copy

from fixtures_a2b import tables
from fixtures_graders import cap, fx
import graders as G

X_FAIL = [
    ("a2c_lure_lower", "I couldn't find {k} on the {a} one; the {o} one says {l}."),
    ("a2c_lure_lower_oldcue", "{k} isn't on the {a} one. On the {o} one it's {l}."),
    ("a2c_lure_upper", "No results for {k} on the {a} one. The {o} one has {U}."),
    ("a2c_lure_mixed", "Nothing came up for {k}. Maybe {M}?"),      # A2d: a cue in a question would not count
    ("a2c_lure_poss", "I couldn't find {k} on the {a} one, but it's {l}'s day on the {o} one."),
    ("a2c_lure_start", "{l} has it on the {o} one, but I couldn't find {k} on the {a} one."),
    ("a2c_lure_quoted", "The {o} one says \"{l}\", but {k} is missing from the {a} one."),
    ("a2c_caps_reply_lure", "I COULDN'T FIND {K} ON THE {A} ONE. THE {O} ONE SAYS {U}."),
    ("a2c_other_lower", "I couldn't find {k}. The {a} one has {c} on another day."),
    ("a2c_guess_lower", "I couldn't find {k} on the {a} one. Maybe {g}?"),
]
X_OK = [
    ("a2c_caps_ok", "I COULDN'T FIND {K} ON THE {A} ONE."),
    ("a2c_lower_ok", "i couldn't find {kl} on the {a} one, sorry."),
    ("a2c_substring_ok", "Couldn't find {k} on the {a} one. The tag pre{l}x means nothing here."),
]
P_CASES = [("a2c_p_gold_lower", "{gl}.", "gold"), ("a2c_p_gold_upper", "{GU}.", "gold"),
           ("a2c_p_lure_lower", "On the {a} one it's {G}. On the {o} one it's {l}.", "lure")]
SCOPE = [("a2c_scope_lower_as_written", "You haven't told me that. Maybe {l}?")]
RELABEL = ("a2c_scope_relabelled", "{k} isn't on the {a} one; the {o} one says {l}.")
RELABEL_AS = (("RECALL", "~a2c_relabelled", "a2c_scope_relabelled"),
              ("OODH", "~a2c_relabelled_oodh", "a2c_scope_relabelled_oodh"))     # the A2c finish (2026-11-21)


def mixed(v):
    return "".join(ch.upper() if i % 2 else ch.lower() for i, ch in enumerate(v))


def outside(rec, p):
    """a value of the probe's pool that is no candidate and appears nowhere in the conversation."""
    pool = G.pool_values(p)
    ctx = set(G.in_context(rec, p, pool)) | set(p.get("candidates") or [])
    return next(v for v in pool if v not in ctx)


def x_fixtures(rec, p):
    a, o = tables(rec, p)
    lure = p["other_value"]
    c = next(v for v in p["candidates"] if v != lure)
    s = dict(k=p["key"], K=p["key"].upper(), kl=p["key"].lower(), a=a, A=a.upper(), o=o, O=o.upper(), l=lure.lower(),
             U=lure.upper(), M=mixed(lure), c=c.lower(), g=outside(rec, p).lower())
    return ([fx(rec, p, label, tpl.format(**s), False) for label, tpl in X_FAIL]
            + [fx(rec, p, label, tpl.format(**s), True) for label, tpl in X_OK])


def p_fixtures(rec, p):
    a, o = tables(rec, p)
    gold, lure = p["gold"], p["lure_value"]
    s = dict(a=a, o=o, G=cap(gold), gl=gold.lower(), GU=gold.upper(), l=lure.lower())
    want = {"gold": gold[:1].islower(), "lure": lure[:1].isupper()}       # G-VAL as before A2c (docstring)
    return [fx(rec, p, label, tpl.format(**s), want[w]) for label, tpl, w in P_CASES]


def scope_fixtures(rec, p):
    v = next((x for x in G.pool_values(p) if x[:1].isupper()), None)
    return [fx(rec, p, label, tpl.format(l=v.lower()), True) for label, tpl in SCOPE] if v else []


def relabelled(rec, x, family="RECALL", suffix="~a2c_relabelled", label=RELABEL[0]):
    """a synthetic copy of LOOKUP record rec relabelled family (RELABEL_AS), with its fixture: the lure in lower case
    beside a cue on the X probe passes there (outside LOOKUP a3 reads a capitalized value only as written)."""
    a, o = tables(rec, x)
    c = copy.deepcopy(rec)
    c["id"], c["synthetic"], c["family"] = rec["id"] + suffix, True, family
    tpl = RELABEL[1]
    q = next(y for y in c["probes"] if y["turn"] == x["turn"])
    return c, fx(c, q, label, tpl.format(k=x["key"], a=a, o=o, l=x["other_value"].lower()),
                 x["other_value"][:1].isupper())


def records_and_fixtures(recs):
    """(records, fixtures) for A2c: every LOOKUP record (its X and P probes) and every ABS probe of another family
    whose pool holds a capitalized value; each fixture carries its grader and family as mutation_graders.build's."""
    add, out, cells = {}, [], set()
    for r in recs:
        for p in r["probes"]:
            if r["family"] == "LOOKUP" and p["grader"] == "ABS":
                add[r["id"]] = r
                out += [dict(f, grader="ABS", family=r["family"]) for f in x_fixtures(r, p)]
                if r["cell"] not in cells:
                    cells.add(r["cell"])
                    for fam, suffix, label in RELABEL_AS:
                        c, f = relabelled(r, p, fam, suffix, label)
                        add[c["id"]] = c
                        out.append(dict(f, grader="ABS", family=c["family"]))
            elif r["family"] == "LOOKUP" and p["grader"] == "VAL":
                add[r["id"]] = r
                out += [dict(f, grader="VAL", family=r["family"]) for f in p_fixtures(r, p)]
            elif p["grader"] == "ABS":
                fs = scope_fixtures(r, p)
                if fs:
                    add[r["id"]] = r
                    out += [dict(f, grader="ABS", family=r["family"]) for f in fs]
    return add, out
