"""Mutation test for the new likelihood items and their scorer: break each part on purpose and
confirm validate_items.run() goes red. Exits non-zero if any mutant survives."""
import importlib, sys

import items_new as N
import metrics as M
import validate_items as V

ORIG_BUILD = N.build
ORIG_RIGHT = M.right
ORIG_MARGIN = M.margin
ORIG_PAIRS = dict(M.PAIRS)


def reset():
    importlib.reload(N)
    importlib.reload(M)
    importlib.reload(V)


def mutate_items(fn):
    """Wrap build(): fn(items) edits the built items in place."""
    def patch():
        orig = N.build
        def build(*a, **k):
            its = orig(*a, **k)
            fn(its)
            return its
        N.build = build
    return patch


def m_ties_pass():
    M.right = lambda s: all(s["gold"] >= v for k, v in s.items() if k != "gold")


def m_only_vs_orig():
    M.right = lambda s: s["gold"] > s.get("orig", s.get("later", float("-inf")))


def m_margin_sign():
    M.margin = lambda s: max(v for k, v in s.items() if k != "gold") - s["gold"]


def m_pair_drop_noupd():
    M.PAIRS["upd_bind"] = ("same_k1",)


def _swap_noupd(its):
    for it in its:
        if it["var"] == "noupd":
            c = it["cands"]
            c["gold"], c["later"] = c["later"], c["gold"]


def _keycorr_key_in_both(its):
    for it in its:
        if it["var"] == "keycorr":
            F = N.FAMILIES[it["fam"]]
            v0 = it["cands"]["orig"].strip()
            it["turns"][0] = F["key"](v0)


def _leak_gold(its):
    for it in its:
        if it["var"] == "same_k1":
            it["question"] = it["question"][:-1] + " (" + it["cands"]["gold"].strip() + "?)"


def _pos_no_lead(its):
    for it in its:
        if it["var"] == "pos":
            del it["turns"][:3]


def _filler_value(its):
    for it in its:
        if it["var"] == "twoslot" and it["d"] > 0:
            q, a = it["turns"][-1]
            it["turns"][-1] = (q, a + " Maybe " + it["cands"]["orig"].strip() + ".")


def _twoslot_other_keyed(its):
    for it in its:
        if it["var"] == "twoslot":
            F = N.FAMILIES[it["fam"]]
            i = [k for k, (u, a) in enumerate(it["turns"]) if it["cands"]["other"].strip() in u][0]
            it["turns"][i] = F["key_corr"](it["cands"]["other"].strip())


def _short_tail(its):
    for it in its:
        if it["d"] == 4 and it["var"] == "keyorig":
            it["turns"].pop()


def _gold_is_orig_same(its):
    for it in its:
        if it["var"] == "same_k1":
            c = it["cands"]
            c["gold"], c["orig"] = c["orig"], c["gold"]


MUTANTS = {
    "scorer: ties count as right": m_ties_pass,
    "scorer: gold compared only with the original": m_only_vs_orig,
    "scorer: margin sign flipped": m_margin_sign,
    "pair metric: upd_bind drops the no-update twin": m_pair_drop_noupd,
    "items: noupd gold/later swapped": mutate_items(_swap_noupd),
    "items: keycorr has the key in the original too": mutate_items(_keycorr_key_in_both),
    "items: gold leaked into the question": mutate_items(_leak_gold),
    "items: position control without leading fillers": mutate_items(_pos_no_lead),
    "items: a filler mentions a candidate value": mutate_items(_filler_value),
    "items: twoslot's other object uses the key wording": mutate_items(_twoslot_other_keyed),
    "items: tail shorter than d": mutate_items(_short_tail),
    "items: same_k1 gold set to the original value": mutate_items(_gold_is_orig_same),
}

if __name__ == "__main__":
    survivors = []
    for name, patch in MUTANTS.items():
        reset()
        patch()
        errs = V.run(verbose=False)
        killed = bool(errs)
        print(("KILLED  " if killed else "SURVIVED"), name, f"({len(errs)} errors, e.g. {errs[0][:90] if errs else '-'})")
        if not killed:
            survivors.append(name)
    reset()
    assert not V.run(verbose=False), "unmutated validation fails"
    print("unmutated validation passes;", "all mutants killed" if not survivors else f"SURVIVORS: {survivors}")
    sys.exit(1 if survivors else 0)
