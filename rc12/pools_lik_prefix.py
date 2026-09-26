"""RC-12 likelihood rows (prereg draft s4): the forced answer prefix pool, separate from every generator pool.

A likelihood row ends with the model's turn opened by one neutral lead-in from LEADS (template render: right after
the generation prompt; plain render: after "Assistant: "), and each candidate is scored as the continuation
" <candidate>". The lead-ins are neutral: they fit every closed-set probe the same way (a name, a day, an
instrument, a project), name no value, holder or object, and so give no candidate an advantage over another.
They are NOT the record's own answer prefix (the probe's "prefix" field is the IDEAL reply's frame, e.g. "Your bike
is", and is None on BIND, TOPIC, LOOKUP and OWN probes).
Draw: lead_for(key, turn) is a pure function of the key and the probe turn, from its own stream
"RC12:likprefix:<key>:<turn>:<SEED>" with SEED = 4747 (not the dev seed 1212, not a "RC12:dev:" stream), so a row's
lead-in does not depend on which records or families are loaded. key = the record id, or the pair_id for BIND twins
(lik_rows.lead_key; STEP 11 FIX ROUND): the twins are a minimal pair (only the order of the two value statements
differs), so they share one lead-in, and a position heuristic whose preference depends on the lead-in cannot split
them. test_lik.py checks: no lead-in mentions any value
of any pools_vals pool or any dev candidate (whole word, any case), SEED differs from the dev seed, the draw is
deterministic, and every lead-in is used on the dev rows."""
import random

SEED = 4747
LEADS = ("It is", "It's", "That would be", "That's", "The answer is", "That is", "It would be",
         "The answer would be")


def stream(key, turn, seed=SEED):
    return random.Random(f"RC12:likprefix:{key}:{turn}:{seed}")


def lead_for(key, turn, seed=SEED):
    return stream(key, turn, seed).choice(LEADS)
