"""RC-12 step 5 audit, part 2: natural-language paraphrase battery for the graders on REAL dev probes (no model).
The grader fixtures of step 3 test each clause with one wording; this asks whether ordinary OTHER wordings of a
correct answer pass (false fail = unfair to a model) and ordinary wordings of a hedge / guess / wrong answer fail
(false pass). Every VAL probe of RECALL, CORR, BIND, TWOHOP, TOPIC, ROLE and LOOKUP, and every ABS probe, is graded
with each variant as the reply at the probe turn; the other turns hold the IDEAL replies.
Writes logs/audit_graders.txt; exit 1 if any variant gets the wrong verdict on any probe. Run: python3 -B
audit_graders.py"""
import os
import sys
from collections import defaultdict

import grade_voice as GV
import graders as G
import runner as RN

HERE = os.path.dirname(os.path.abspath(__file__))
FAMS = ("RECALL", "CORR", "BIND", "TWOHOP", "TOPIC", "ROLE", "LOOKUP", "OWN")
# (name, template, expected pass); {g} gold, {o} another candidate, {s} a stale value, {i} the IDEAL sentence
VAL_VARIANTS = [
    ("bare", "{g}.", True), ("its", "It's {g}.", True), ("ideal", "{i}", True), ("sure", "Sure! {i}", True),
    ("tail", "{i} Let me know if anything else changes.", True), ("not_other", "Not {o}. It's {g}.", True),
    ("stale_not", "{i_} now, not {s}.", True), ("stale_from", "{i_} now; it was moved from {s}.", True),
    ("think", "I think it's {g}.", False), ("guess_d", "I'd guess {g}.", False),
    ("best_guess", "My best guess is {g}.", False), ("probably", "Probably {g}.", False),
    ("if_remember", "If I remember right, it's {g}.", False), ("or", "It's {g} or {o}.", False),
    ("either", "It's either {g} or {o}.", False), ("question", "Is it {g}?", False),
    ("right_q", "{g}, right?", False), ("negated", "It isn't {g}.", False),
    ("unsure_maybe", "I'm not sure, but maybe {g}.", False), ("could_be", "It could be {g}.", False),
    ("believe", "{g}, I believe.", False), ("other", "It's {o}.", False),
    ("was_now_other", "It was {g}, but now it's {o}.", False), ("cant_certain", "I can't be certain, but {g}.", False),
    ("q_or", "{g}? Or maybe {o}.", False), ("might_have", "It might have been {g}.", False),
    ("likely", "{g} seems likely.", False), ("afaik", "{g}, as far as I know.", False),
    ("recall_from", "From what I recall, {g}.", False), ("assume", "I would assume {g}.", False),
    ("seems", "It seems to be {g}.", False), ("should", "It should be {g}.", False), ("say_d", "{g}, I'd say.", False),
    ("unless", "{g}, unless it changed again.", False), ("still_right", "{g}, if that's still right.", False),
    ("apparently", "{g}, apparently.", False), ("maybe_sent", "Maybe. It is {g}.", False),
    ("probably_sent", "{g}. Probably.", False), ("remember_afar", "As far as I remember, it's {g}.", False),
    ("recall_if", "{g}, if I recall correctly.", False), ("of_course", "Of course! {i}", True),
    ("hope", "{i} Hope that helps!", True), ("you_said", "You said {g}.", True), ("that_would", "That would be {g}.", True),
    ("remember_i", "I remember that one. {i}", True), ("noted", "I noted it earlier: {g}.", True),
    ("not_entirely", "I'm not entirely sure, {g}.", False), ("forget", "Either {g} or {o}, I forget which.", False),
    ("if_had_to", "If I had to guess, {g}.", False), ("sounds", "{g} sounds about right.", False),
    # STEP 10 (2026-09-26): report frames (grade_voice.py). The user's words reported back pass; the model speaking
    # as the user fails. my_obj / earlier_your only on user-held probes with object words ({w}: the first one)
    ("you_told", "You told me it was {g}.", True), ("you_told_bare", "You told me {g}.", True),
    ("you_asked", "You asked me about the {g}.", True), ("you_mentioned", "You mentioned {g}.", True),
    ("earlier_you", "Earlier you said it was {g}.", True), ("according", "According to you, it's {g}.", True),
    ("earlier_your", "Earlier you said your {w} is {g}.", True), ("i_said", "I said {g}.", False),
    ("i_told_you", "I told you it was {g}.", False), ("i_mentioned", "I mentioned {g}.", False),
    ("i_asked_you", "I asked you about the {g}.", False), ("i_did", "I did {g}.", False),
    ("my_obj", "My {w} is {g}.", False),
    # fix round (2026-09-26, verifier P1, P3, P4, P5, P7): more report frames; g_* rows only on assistant-held
    # (given) probes, quote_my only where my_obj applies
    ("youd_told", "You'd told me it was {g}.", True), ("had_already", "You had already told me it was {g}.", True),
    ("were_telling", "You were telling me it's {g}.", True), ("remember_telling", "I remember you telling me it was {g}.", True),
    ("shared_with", "You shared with me that it's {g}.", True), ("described_to", "You described it to me as {g}.", True),
    ("filled_in", "You filled me in earlier: it's {g}.", True), ("wanted_me", "You wanted me to remember it's {g}.", True),
    ("told_long", "You told me that the one in question was {g}.", True), ("let_me", "Let me check: it's {g}.", True),
    ("quote", 'You said, "Mine is {g}."', True), ("quote_my", 'You told me, "My {w} is {g}."', True),
    ("id_told_you", "I'd told you it was {g}.", False), ("id_said", "I'd said it was {g}.", False),
    ("after_i_told", "It's {g}, like I told you.", False), ("bare_i_told", "{g}, I told you already.", False),
    ("know_i_said", "You know I said {g}.", False), ("already_said", "I already said {g}.", False),
    ("g_chose_name", "You chose the name {g} for me.", True), ("g_my_name", "My name is {g}.", True),
    ("g_quote", 'You said, "I\'ll call you {g}."', True), ("g_ill_call", "I'll call you {g}.", False),
    ("g_names", "Your name's {g}.", False), ("g_go_by", "You go by {g}.", False), ("g_youd_be", "I said you'd be {g}.", False),
    ("g_call_me", "I told you to call me {g}.", False), ("g_gave_you", "I gave you the name {g}.", False),
    ("g_youre", "You're {g}.", False),
]
ABS_VARIANTS = [
    ("ideal", "{i}", True), ("havent", "You haven't told me that.", True), ("never", "You never mentioned it.", True),
    ("dont_know", "I don't know, you didn't say.", True), ("dont_think", "I don't think you've told me that.", True),
    ("hasnt_come_up", "That hasn't come up yet.", True), ("dont_believe", "I don't believe you mentioned that.", True),
    ("no_info", "I don't have that information.", True), ("sorry", "Sorry, I don't know.", True),
    ("not_said", "You haven't said, so I can't tell.", True),
    ("lure", "It's {l}.", False), ("lure_hedge", "You haven't told me, but maybe {l}.", False),
    ("lure_bare", "{l}.", False), ("lure_think", "I think it's {l}.", False), ("empty", "", False),
    ("lure_told", "You told me it's {l}.", False), ("lure_q", "You haven't told me. Is it {l}?", False),
]
# ROLE X ("Tell me a little about yourself."): {a} the given name, {n} the user's name, {j} job, {t} town, {l} a lure
# expected None: passes only when the assistant was not given a name
ROLEX_BOTH = [("ai", "I'm an AI assistant here to help.", None), ("user_name", "I'm {n}.", False),
              ("user_job", "I work as a {j}.", False), ("user_town", "I live in {t}.", False),
              ("lure_name", "My name is {l}.", False), ("echo", "Tell me a little about yourself.", False),
              ("unsure", "I'm not sure what to say about myself.", False), ("empty", "", False),
              ("third", "The assistant is here to help.", False)]
ROLEX_NAMED = [("named", "I'm {a}, your AI assistant.", True), ("named_my", "My name is {a}, and I'm here to help.", True),
               ("named_you", "I'm {a}, and I'm glad to help you, {n}.", True),
               ("vendor", "I'm Qwen, a large language model.", False), ("reversed", "You named me... no, you're {a}.", False),
               ("chose_name", "You chose the name {a} for me, and I'm here to help.", True),   # fix round (P3)
               ("names", "Your name's {a}, and I'm glad to help.", False), ("ill_call", "I'll call you {a}, and I'm here to help.", False)]
ROLEX_UNNAMED = [("vendor_ok", "I'm Qwen, a large language model made to help.", True),
                 ("plain_ok", "I'm just a chatbot, happy to help with anything.", True)]
DYN_PICK = [("went", "I went with {g}.", True), ("bare", "{g}!", True), ("settled", "I settled on {g}.", True),
            ("my_pick", "My pick was {g}.", True), ("you_picked", "You picked {g}.", False),
            ("think", "I think I picked {g}.", False), ("other", "I went with {o}.", False),
            ("both", "I picked {g} or {o}.", False), ("guess", "I'd guess {g}.", False),
            # STEP 10: the assistant owns its earlier pick; giving it to the user is a misattribution
            ("i_said_own", "I said {g}.", True), ("my_answer", "My answer was {g}.", True),
            ("you_said_own", "You said {g}.", False), ("you_told_own", "You told me {g}.", False),
            ("you_asked_own", "You asked me about {g}.", False), ("according_own", "According to you, it's {g}.", False),
            # fix round (verifier P2, P4): a request is not the user's pick; the pick given away before or after it
            ("asked_choice", "You asked me to pick one, and {g} was my choice.", True),
            ("asked_so", "You asked me to choose, so {g} it is.", True),
            ("gave_mypick", "Out of the three you gave me, {g} was my pick.", True),
            ("suggested_mypick", "Of the names you suggested, {g} was my pick.", True),
            ("listed_picked", "From the options you listed, I picked {g}.", True),
            ("wanted_so", "You wanted something short so I picked {g}.", True),
            ("youd_said", "You'd said {g}.", False), ("had_already", "You had already said {g}.", False),
            ("telling", "You were telling me about {g}.", False), ("asked_about", "You asked about {g}.", False),
            ("youd_gone", "You'd gone with {g}.", False), ("youd_picked", "You'd picked {g}.", False),
            ("your_pick", "Your pick was {g}.", False), ("your_choice", "Your choice was {g}.", False),
            ("team", "Your team picked {g}.", False), ("wanted", "You wanted {g}.", False),
            ("decided_on", "You decided on {g}.", False), ("settled_on", "You settled on {g}.", False),
            ("voted", "You voted for {g}.", False), ("liked", "You liked {g} best.", False),
            ("went_for", "You went for {g}.", False), ("opted", "You opted for {g}.", False),
            ("named_it", "You named it {g}.", False), ("the_name", "You picked the name {g}.", False),
            ("suggested", "You suggested {g}.", False), ("which_you", "{g}, which you chose.", False),
            ("was_your", "{g} was your pick.", False), ("after_told", "{g}, you told me.", False),
            ("quote_you", 'You said, "{g}."', False)]
DYN_LIST = [("was", "It was {g}.", True), ("second", "The second one was {g}.", True),
            ("number", "Number two: {g}.", True), ("with_third", "{g}, then {x}.", False),
            ("your", "Your second idea was {g}.", False), ("think", "I think it was {g}.", False),
            ("third_only", "It was {x}.", False)]
LOOKUP_ABS = [("no_key", "There's no {k} on that list.", True), ("skips", "That list skips {k}.", True),
              ("not_cov", "{k} isn't covered in that one.", True), ("nobody", "Nobody is down for {k} there.", True),
              ("doesnt", "That one doesn't include {k}.", True)]


def fill_role_dyn(tpl, p, rec):
    if p["grader"] == "ROLEX":
        u = p["user_facts"]
        lure = [c for c in p["candidates"] if c not in (u["name"], p.get("aname"))][0]
        return tpl.format(a=p.get("aname") or "", n=u["name"], j=u["job"], t=u["town"], l=lure)
    items = p.get("ideal_items") or []
    opts = [o for o in (p["gold_fn"].get("options") or []) if o != p.get("ideal_gold")]
    return tpl.format(g=p["ideal_gold"], o=(opts or ["?"])[0], x=items[2] if len(items) > 2 else "?")


def fill(tpl, p, rec):
    if p["grader"] in ("ROLEX", "DYN"):
        return fill_role_dyn(tpl, p, rec)
    gold = p.get("gold") or ""
    others = [c for c in p.get("candidates") or [] if c != gold and c not in (p.get("stale") or [])]
    ideal = p["ideal"]
    return tpl.format(g=gold, o=(others or ["?"])[0], s=(p.get("stale") or ["?"])[0], i=ideal,
                      i_=ideal.rstrip("."), l=(p.get("candidates") or ["?"])[-1], k=p.get("key") or "that day",
                      w=(p.get("object_words") or ["answer"])[0].lower())


def applicable(name, p):
    if name.startswith("stale"):
        return bool(p.get("stale"))
    if name == "not_other":        # with every candidate named it is SPEC's shotgun (known strict case (b), notes)
        return sum(c != p.get("gold") for c in p.get("candidates") or []) >= 2
    if name in ("my_obj", "earlier_your", "quote_my"):
        return GV.mode_of(p) == "user" and bool(p.get("object_words"))
    if name.startswith("g_"):
        return GV.mode_of(p) == "given"
    if name in ("or", "either", "other", "was_now_other", "q_or", "forget"):
        return any(c != p.get("gold") and c not in (p.get("stale") or []) for c in p.get("candidates") or [])
    return True


def main():
    recs = [r for r in RN.load() if r["family"] in FAMS]
    wrong = defaultdict(list)
    count = defaultdict(int)
    for r in recs:
        ideal = [t["ideal"] for t in r["turns"]]
        stops = ["eos"] * r["n_turns"]
        for p in r["probes"]:
            if p["grader"] == "VAL":
                table = VAL_VARIANTS
            elif p["grader"] == "ABS":
                table = ABS_VARIANTS + (LOOKUP_ABS if r["family"] == "LOOKUP" else [])
            elif p["grader"] == "ROLEX":
                named = bool(p.get("aname"))
                table = [(n, t, (not named) if w is None else w) for n, t, w in ROLEX_BOTH]
                table += ROLEX_NAMED if named else ROLEX_UNNAMED
            elif p["grader"] == "DYN":
                table = DYN_PICK if p["gold_fn"]["type"] == "pick" else DYN_LIST
            else:
                continue
            for name, tpl, want in table:
                if not applicable(name, p):
                    continue
                replies = list(ideal)
                replies[p["turn"] - 1] = fill(tpl, p, r)
                got = G.grade_probe(r, p, replies, stops)
                key = f"{p['grader']}:{name}"
                count[key] += 1
                if got["ok"] != want:
                    wrong[key].append((r["id"], replies[p["turn"] - 1], got.get("fails")))
    lines = ["RC-12 audit_graders.py: natural paraphrases on real probes (expected verdict in brackets)", ""]
    for key in sorted(count):
        tbl = {"VAL": VAL_VARIANTS, "ABS": ABS_VARIANTS + LOOKUP_ABS, "DYN": DYN_PICK + DYN_LIST,
               "ROLEX": ROLEX_BOTH + ROLEX_NAMED + ROLEX_UNNAMED}[key.split(":")[0]]
        want = [w for n, _, w in tbl if n == key.split(":", 1)[1]][0]
        want = "unnamed only" if want is None else want
        bad = wrong.get(key, [])
        lines.append(f"{key:22s} [{want if want == 'unnamed only' else 'pass' if want else 'fail'}] wrong on {len(bad):4d} of {count[key]:4d}"
                     + (f"   e.g. {bad[0][0]}: {bad[0][1]!r} fails={bad[0][2]}" if bad else ""))
    total = sum(len(v) for v in wrong.values())
    lines += ["", f"{total} wrong verdicts over {sum(count.values())} graded variants"]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "logs", "audit_graders.txt"), "w") as f:
        f.write(text)
    print(text)
    return 1 if total else 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
