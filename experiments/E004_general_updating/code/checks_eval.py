"""E004 eval item checks, per item (notes (b) item-set acceptance, the text-level parts). Each check returns a list
of problem strings (empty = pass). No model, no tokenizer. Used by test_eval_items.py and mutation_eval.py."""
import re

from text_e004 import normalize, echo_runs, values_in, words, content_words, count_value
from pools_eval import ALL_VALUES, filler_pool
from oracles_e004 import strip_marker, names

K_RANGE = {"H1": (2, 3), "H2": (1, 3), "H3": (4, 5), "H4": (1, 3), "H6": (1, 3), "H7": (1, 3), "H5": (0, 3),
           "ID": (0, 3), "C_noupd": (0, 0)}
C_TWOSLOT_K = {"H1": (2, 3), "H2": (1, 3), "H3": (4, 5)}
MARKER_WORDS = ["hold on", "correction", "one more change", "oops", "on second thought", "one more thing",
                "actually", "wait", "change of plans", "sorry, i meant", "update", "never mind"]


def _terms(it):
    return [t for ph, h in it["objects"] for t in (ph, h)] + ([it["alias"]] if it["alias"] else [])


def stmt_turns(it):
    return {s["turn"] for s in it["stmts"]}


def check_text(it):
    """annotation vs text: one value per statement, no value in any other turn, acks repeat only their value,
    candidates = the values mentioned, gold unique to its statement, question and prefix value-free."""
    p = []
    st = {s["turn"]: s for s in it["stmts"]}
    for ti, (u, a) in enumerate(it["turns"]):
        vu, va = values_in(u, ALL_VALUES), values_in(a, ALL_VALUES)
        if ti in st:
            if vu != [st[ti]["value"]] or count_value(u, st[ti]["value"]) != 1:
                p.append(f"statement turn {ti} values {vu} != [{st[ti]['value']}]")
            if va and va != [st[ti]["value"]]:
                p.append(f"ack turn {ti} mentions {va}")
        elif vu or va or re.search(r"\d", u + a):
            p.append(f"non-statement turn {ti} holds a value or digit: {vu + va}")
    vals = []
    for s in it["stmts"]:
        if s["value"] not in vals:
            vals.append(s["value"])
    if vals != it["candidates"] or len(vals) < 2 or it["gold"] not in vals:
        p.append(f"candidates {it['candidates']} vs {vals}")
    about = [s for s in it["stmts"] if s["obj"] == it["asked"]]
    if not about or about[-1]["value"] != it["gold"]:
        p.append("gold is not the latest statement about the asked object")
    if sum(s["value"] == it["gold"] for s in it["stmts"]) != 1:
        p.append("gold value used by more than one statement")
    for name in ("question", "prefix"):
        if values_in(it[name], ALL_VALUES) or re.search(r"\d", it[name]):
            p.append(f"{name} contains a value")
    return p


def check_refs(it):
    """reference forms: named statements name exactly their object; pronoun/ellipsis directly after a statement
    about the same object; aliases defined in the object's original and used only by it; no statement names
    two objects; the question and prefix name the asked object."""
    p, prev = [], None
    for s in it["stmts"]:
        u = strip_marker(it["turns"][s["turn"]][0])
        named = [i for i, (ph, h) in enumerate(it["objects"]) if names(u, ph) or names(u, h)]
        if s["ref"] in ("full", "head"):
            ph, h = it["objects"][s["obj"]]
            if named != [s["obj"]] or not names(u, ph if s["ref"] == "full" else h):
                p.append(f"turn {s['turn']} {s['ref']} names {named}")
        elif named:
            p.append(f"turn {s['turn']} {s['ref']} names object {named}")
        if s["ref"] in ("pron", "ell"):
            if prev is None or prev["obj"] != s["obj"] or prev["turn"] != s["turn"] - 1:
                p.append(f"turn {s['turn']} {s['ref']} is not directly after its object's statement")
        if it["alias"] and (it["alias"] in u) != (s["ref"] == "alias" or (s["role"] == "orig"
                                                                         and s["obj"] == it["alias_obj"])):
            p.append(f"turn {s['turn']} alias use wrong")
        if s["ref"] == "alias" and (not it["alias"] or s["obj"] != it["alias_obj"]):
            p.append(f"turn {s['turn']} alias without a definition")
        prev = s
    ph, h = it["objects"][it["asked"]]
    for name in ("question", "prefix"):
        other = [o for i, o in enumerate(it["objects"]) if i != it["asked"] and (names(it[name], o[0]) or
                                                                                  names(it[name], o[1]))]
        if other or (name == "question" and not names(it[name], ph if it["q_form"] == "full" else h)):
            p.append(f"{name} names the wrong object")
    return p


def check_echo(it):
    """frame echoes: with a frame, the question AND prefix echo every lure statement and nothing else (no other
    statement, no acknowledgement); without one, nothing echoes the question or prefix."""
    n = lambda s: normalize(s, _terms(it), it["values"])
    q, pre = n(it["question"]), n(it["prefix"])
    p = []
    for s in it["stmts"]:
        u, a = it["turns"][s["turn"]]
        for side, txt in (("statement", u), ("ack", a)):
            e_q, e_p = echo_runs(q, n(txt)), echo_runs(pre, n(txt))
            want = s["lure"] and side == "statement"
            if want and not (e_q and e_p):
                p.append(f"lure turn {s['turn']} not echoed by question {bool(e_q)} / prefix {bool(e_p)}")
            if not want and (e_q or e_p):
                p.append(f"unintended echo {side} turn {s['turn']}: {e_q or e_p}")
    if (it["frame"] is not None) != any(s["lure"] for s in it["stmts"]):
        p.append("frame without lure or lure without frame")
    return p


def check_content(it):
    """H1 (and the C_twoslot H1 cell): the latest correction shares no content word with the question or
    prefix; H2 (and its cell): no correction of the asked object does."""
    cell = it["family"] if it["family"] in ("H1", "H2") else it.get("cell")
    if cell not in ("H1", "H2") or it["family"] == "C_noupd":
        return []
    ex = [v.lower() for v in it["values"]]
    qs = set(content_words(it["question"], ex)) | set(content_words(it["prefix"], ex))
    corr = [s for s in it["stmts"] if s["obj"] == it["asked"] and s["role"] == "corr"]
    check = corr[-1:] if cell == "H1" else corr
    return [f"correction turn {s['turn']} shares content words {sorted(shared)}" for s in check
            if (shared := set(content_words(strip_marker(it["turns"][s["turn"]][0]), ex)) & qs)]


def check_fillers(it):
    """fillers come from the declared pool, name no object or alias of the item, carry no marker; d fillers
    after the last statement."""
    p = []
    pool = set(filler_pool(it["fillers"]))
    st = stmt_turns(it)
    banned = {w for ph, h in it["objects"] for w in words(ph) + [h]} | (set(words(it["alias"])) if it["alias"]
                                                                        else set())
    for ti, t in enumerate(it["turns"]):
        if ti in st:
            continue
        if t not in pool:
            p.append(f"turn {ti} is not from the {it['fillers']} filler pool")
        ws = set(words(t[0] + " " + t[1]))
        if ws & (banned | {b + "s" for b in banned}):
            p.append(f"filler turn {ti} names an object word {sorted(ws & banned)}")
        low = (t[0] + " " + t[1]).lower()
        if any(re.search(r"(?<![a-z])" + re.escape(m) + r"(?![a-z])", low) for m in MARKER_WORDS):
            p.append(f"filler turn {ti} holds a marker word")
    d = len(it["turns"]) - 1 - max(st)
    if d != it["d"]:
        p.append(f"d {d} != {it['d']}")
    if len(set(it["turns"])) != len(it["turns"]):
        p.append("a turn repeats")
    return p


def check_structure(it):
    p = []
    fam, cell = it["family"], it.get("cell")
    lo, hi = (C_TWOSLOT_K.get(cell, (1, 3)) if fam == "C_twoslot" else K_RANGE[fam])
    if not lo <= it["k"] <= hi:
        p.append(f"k {it['k']} outside {lo}-{hi}")
    want_d = 20 if (fam == "H4" or cell == "H4") else 10
    if it["d"] != want_d:
        p.append(f"d {it['d']}")
    want_obj = 3 if (fam == "H5" or cell == "H5") else ((1, 2) if fam == "ID" else 2)
    if it["n_obj"] not in (want_obj if isinstance(want_obj, tuple) else (want_obj,)) or \
            len(it["objects"]) != it["n_obj"] or {s["obj"] for s in it["stmts"]} != set(range(it["n_obj"])):
        p.append(f"objects {it['n_obj']}")
    h6 = fam == "H6" or cell == "H6"
    if h6 != (it["pool_key"] in ("weekday_delivery", "colour_decor", "sport_activity", "number_place")):
        p.append(f"pool {it['pool_key']}")
    if (fam == "H7" or cell == "H7") != (it["fillers"] == "h7"):
        p.append("filler genre")
    if fam == "C_noupd" and [s["obj"] for s in it["stmts"]].count(it["asked"]) != 1:
        p.append("C_noupd asked object stated more than once")
    if fam == "C_twoslot":
        a_last = max(i for i, s in enumerate(it["stmts"]) if s["obj"] == 0)
        if not any(s["obj"] == 1 for s in it["stmts"][a_last:]) or any(s["obj"] == 1 for s in it["stmts"][:a_last]):
            p.append("C_twoslot: B is not stated after A's latest")
    return p


ITEM_CHECKS = [("text", check_text), ("refs", check_refs), ("echo", check_echo), ("content", check_content),
               ("fillers", check_fillers), ("structure", check_structure)]
