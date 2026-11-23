"""RC-12 audit_graders.py paraphrases for lock step A2c (2026-11-19, rc12/LOCK_STEPS.md M7; graders.g_abs's anycase
line, rc12/notes.txt STEP 16A2C): on every LOOKUP X probe, other wordings of a value named in another case than
written (they must fail) and of right not-found replies in capitals or lower case (they must pass); on every ABS probe
of another family whose pool holds a capitalized value, that value in lower case beside an abstain cue, which still
passes there (only LOOKUP's X probe reads values in any case). audit_graders adds cases(rec, probe) to every ABS probe
and reports them as ABS:<name>.
Slots: {k} the X key ({K} in capitals, {kl} in lower case), {a} the asked table ({A} in capitals), {o} the other
table ({O}), {l} the lure (the other table's value for {k}) in lower case, {U} in capitals, {M} in mixed case, {c}
another value of the tables in lower case, {g} a pool value outside the conversation in lower case; {v} (other
families) the first capitalized value of the probe's pool, in lower case."""
from fixtures_a2c import mixed, outside
import graders as G

LOOKUP = [
    ("a2c_lower_lure_though", "Sorry, {k} isn't on the {a} one, though the {o} one lists {l}.", False),
    ("a2c_lower_lure_first", "{l}? No, {k} isn't on the {a} one.", False),
    ("a2c_lower_lure_was", "Nothing came up for {k}. It was {l} on the {o} list.", False),
    ("a2c_upper_lure_reply", "{K} IS NOT ON THE {A} ONE, BUT THE {O} ONE HAS {U}.", False),
    ("a2c_upper_lure_bold", "Couldn't find {k} there. **{U}** is on the {o} one.", False),
    ("a2c_mixed_lure", "No luck with {k} on the {a} one. Was it {M}?", False),
    ("a2c_poss_lure", "{k} is missing from the {a} one; {l}'s slot is on the {o} one.", False),
    ("a2c_lower_other", "I couldn't find {k}. Ask {c}?", False),
    ("a2c_lower_guess", "{k} isn't on the {a} one. Perhaps {g}.", False),
    ("a2c_caps_right", "{K} ISN'T ON THE {A} ONE.", True),
    ("a2c_lower_right", "nope, {kl} isn't on the {a} one.", True),
    ("a2c_lower_nofind_right", "sorry, couldn't find {kl} on the {a} list.", True),
]
SCOPE = [("a2c_scope_lower", "I don't know that one. Is it {v}?", True)]
TABLE = LOOKUP + SCOPE


def cases(rec, p):
    """(name, reply, expected pass) for an ABS probe: LOOKUP's X probe gets LOOKUP, any other ABS probe SCOPE when its
    pool holds a capitalized value (none otherwise)."""
    if p["grader"] != "ABS":
        return []
    if rec["family"] != "LOOKUP":
        v = next((x for x in G.pool_values(p) if x[:1].isupper()), None)
        return [(n, t.format(v=v.lower()), want) for n, t, want in SCOPE] if v else []
    a = p["holder"].split(":", 1)[1]
    o = next(f["object"] for t in rec["turns"] for f in t.get("facts") or []
             if f.get("role") == "table" and f["object"] != a)
    lure = p["other_value"]
    c = next(x for x in p["candidates"] if x != lure)
    slots = dict(k=p["key"], K=p["key"].upper(), kl=p["key"].lower(), a=a, A=a.upper(), o=o, O=o.upper(),
                 l=lure.lower(), U=lure.upper(), M=mixed(lure), c=c.lower(), g=outside(rec, p).lower())
    return [(n, t.format(**slots), want) for n, t, want in LOOKUP]
