"""RC-12 fake scripted responders, part 2: family-specific cheaters (SPEC s5) and machinery-test fakes. No model.
Each is IDEAL outside its own family, so it isolates one failure mode.
  PERSIST (base reply = the filler's plain IDEAL answer; I/O turns = the plain acknowledgement):
    P_NEVER    never applies the rule
    P_ONCE     applies the latest rule only in the reply to it and the next reply
    P_FIRST5   applies the latest rule for 5 replies after it, then drops it
    P_UNIVERSAL  reads no rule: every reply is ONE all-caps sentence that starts with "Greetings", holds all four
               include-words and ends with "that is all" (step 5 audit: 14 of 32 hold items blind before G-FMT
               failed include-word stuffing, 10 after)
    P_OBEYALL  applies every rule ever given, the OLDEST outermost (it keeps the old rule where the two conflict
               and adds the new one where they do not): on hold it equals IDEAL, so it is gated on override only
  LOOP (turn 1 = IDEAL; from turn 2 it reads its OWN previous reply from the history):
    L_REPEAT   repeats its previous reply verbatim (loop rule: equality)
    L_SELFCOPY its previous reply plus the first sentence of the IDEAL reply (loop rule: self-copy)
    L_STUTTER  the IDEAL reply plus a 3-gram said 4 times (loop rule: within-reply repetition), every turn
  Machinery tests (validate_e2e.py):
    CONTINUER  the IDEAL reply, then a fake "User:" / "Assistant:" continuation (plain render must cut it; the
               template render must flag it as a LEAK)
    HISTCHECK  the IDEAL reply padded with whitespace; records every history that is not exactly the user turns
               plus its own earlier replies (stripped), in order, ending with the current user turn
    CAPPER     the IDEAL reply with stop reason "cap" (RUNAWAY, never LOOP)
    SAMPLER    the IDEAL reply on greedy (seed None) and FIRST's reply on sampling seeds (seed plumbing test)
    FORGETFUL  IDEAL, but a probe whose needed turns (S C I O Q T before it, plus its src; defined HERE, not taken
               from runner.py, so it is an independent oracle) are missing from the history it was given gets
               "I don't remember that." (records the probes it forgot)
    T0WRONG    IDEAL, but deflects on T0, K and COMPOSE probes (R must not move)
    BINDHALF   IDEAL, but gives the other candidate on the "asked_first" BIND twin (pair rate must be 0)
    ACKER      IDEAL, but "Got it." on every statement turn (S L C I O T): OD6 (iii), no LOOP, ack-repeats only
    PARROT     IDEAL, but on a statement turn after a short (< 12 words) reply to an asking turn it says that reply
               again (verifier 2026-09-25): every repeat is a LOOP under F1 (decided 2026-09-25) and an answer repeat.
               Asking is read from the record (asks_rec: P X Q, or D with asks: true; F2), not from grade_loop
    SMALLTALK  IDEAL, but one fixed reply (SMALL_ACK) on every small-talk D turn (asks: false): F2 (decided
               2026-09-25), no LOOP, ack-repeats only (one fixed reply on those turns was a LOOP before F2)
  F3 / strict case (d) measurement (verifier 2026-09-25; validate_machinery c_ackrep). restated(rec) picks, for
  each VAL probe whose latest source turn is an S or C turn, that source turn (one probe per turn):
    RESTATE    IDEAL, but confirms each such source turn with its probe's IDEAL sentence ("Your bike is green."),
               then gives the same IDEAL sentence at the probe: a right answer charged as a LOOP (equality; and
               self-copy too when the sentence has 12+ words)
    TERSE_VAL  the bare gold ("Green.") as that confirmation and as the probe answer: LOOP by equality alone
    TERSE_VAL_CTRL  the bare gold as the probe answer, IDEAL acknowledgements: no LOOP, every probe right
  OD1 (b) gate tests (validate_machinery c_owncf, c_owngate; verifier 2026-09-25). None reads its own Q reply:
    NEXT       CONSIST off by one: picks option 1 at Q, later names option 2 of the USER's list; lists: names item
               3 of the annotated list. Wrong on its own history, right on the --own-cf one (gated OWN must be 0)
    LISTPAD    CONSIST, its list Q reply ends with a marker line G-DYN drops ("4. (or mix them)")
    LISTBLANK  CONSIST, a blank bullet line ("- ") between list items 1 and 2
    LISTLS     CONSIST, the list lines joined by U+2028 (str.splitlines splits there, split("\n") does not)"""
import re

import common as C
import fam_persist as FP
import fakes as F
import fakes_audit as FA
import grade_fmt_dyn as FD
import grade_text as T
import pools_vals as V
import render as RD

BASE = {q: a for q, a, _ in C.FILLERS}
STUTTER = "and then it went on and then it went on and then it went on and then it went on."
SMALL_ACK = "That sounds lovely."


def asks_rec(t):
    """whether user turn t asks something, read from the record itself (kind P, X or Q, or a D turn with asks: true;
    F2): the fakes' and the machinery checks' own reading, independent of grade_loop."""
    return t["kind"] in "PXQ" or (t["kind"] == "D" and t["asks"])


class Persist(F.Fake):
    def __init__(self, policy):
        self.policy, self.name = policy, "P_" + policy.upper()

    def start(self, rec, seed=None, render="plain"):
        super().start(rec, seed, render)
        self.rules = [(t["i"], f["rule"], f["arg"]) for t in sorted(rec["turns"], key=lambda x: x["i"])
                      for f in t["facts"] if f.get("role") in ("instruction", "override")]

    def respond(self, history, t, p):
        if self.rec["family"] != "PERSIST":
            return super().respond(history, t, p)
        i = t["i"]
        base = FP.ACK_I if t["kind"] == "I" else FP.ACK_O if t["kind"] == "O" else BASE[t["text"]]
        if self.policy == "universal":           # step 5 audit: obey as many rules as possible without reading any
            return f"GREETINGS, SPLENDID NIFTY DANDY GROOVY: {base.rstrip('.!?').upper()}, THAT IS ALL."
        active = [r for r in self.rules if r[0] <= i]
        if not active or self.policy == "never":
            return base
        g, rule, arg = active[-1]
        if self.policy == "once":
            return FP.apply_rule(rule, arg, base, k=i) if i - g <= 1 else base
        if self.policy == "first5":
            return FP.apply_rule(rule, arg, base, k=i) if i - g <= 5 else base
        s = base
        for _, rule, arg in reversed(active):
            s = FP.apply_rule(rule, arg, s, k=i)
        return s


class Loop(F.Fake):
    def __init__(self, kind):
        self.kind, self.name = kind, "L_" + kind.upper()

    def respond(self, history, t, p):
        if self.rec["family"] != "LOOP":
            return super().respond(history, t, p)
        if self.kind == "stutter":
            return t["ideal"] + " " + STUTTER
        prev = RD.own_reply(history, t["i"], t["i"] - 1)
        if prev is None:
            return t["ideal"]
        if self.kind == "repeat":
            return prev
        return prev + " " + re.split(r"(?<=[.!?])\s+", t["ideal"])[0]


class Continuer(F.Fake):
    name = "CONTINUER"

    def respond(self, history, t, p):
        return super().respond(history, t, p) + "\nUser: Thanks, that helps a lot.\nAssistant: Happy to help!"


class HistCheck(F.Fake):
    name = "HISTCHECK"

    def start(self, rec, seed=None, render="plain"):
        super().start(rec, seed, render)
        self.said, self.violations = {}, []

    def respond(self, history, t, p):
        i = t["i"]
        first = RD.first_turn(history, i)
        want = []
        for j in range(first, i + 1):
            want.append({"role": "user", "content": self.turns[j]["text"]})
            if j < i:
                want.append({"role": "assistant", "content": self.said[j].strip()})
        if history != want or first < 1:
            self.violations.append((self.rec["id"], i))
        out = "  " + super().respond(history, t, p) + " \n"
        self.said[i] = out
        return out


class Capper(F.Fake):
    name = "CAPPER"

    def reply(self, history, i):
        return super().reply(history, i)[0], "cap"


class Sampler(F.Fake):
    name = "SAMPLER"

    def start(self, rec, seed=None, render="plain"):
        super().start(rec, seed, render)
        self.first = F.make("FIRST")
        self.first.start(rec, seed, render)

    def respond(self, history, t, p):
        if self.seed is None:
            return super().respond(history, t, p)
        return self.first.respond(history, t, p)


class Forgetful(F.Fake):
    name = "FORGETFUL"

    def start(self, rec, seed=None, render="plain"):
        super().start(rec, seed, render)
        self.forgot = set()

    def probe(self, history, t, p):
        need = {x["i"] for x in self.rec["turns"] if x["i"] < t["i"] and x["kind"] in "SCIOQT"} | set(p["src"])
        if min(need, default=t["i"]) < RD.first_turn(history, t["i"]):
            self.forgot.add(t["i"])
            return "I don't remember that."
        return super().probe(history, t, p)


class T0Wrong(F.Fake):
    name = "T0WRONG"

    def probe(self, history, t, p):
        if self.rec["family"] in ("T0", "K") or self.rec["cell"] == "COMPOSE":
            return F.DEFLECT
        return super().probe(history, t, p)


class BindHalf(F.Fake):
    name = "BINDHALF"

    def probe(self, history, t, p):
        if self.rec["family"] == "BIND" and self.rec["meta"]["order"] == "asked_first":
            other = [c for c in p["candidates"] if c != p["gold"]][0]
            return F.sub_value(p["ideal"], p["gold"], other)
        return super().probe(history, t, p)


class Acker(F.Fake):
    name = "ACKER"

    def nonprobe(self, history, t):
        return "Got it." if t["kind"] in ("S", "L", "C", "I", "O", "T") else super().nonprobe(history, t)


class Parrot(F.Fake):
    name = "PARROT"

    def nonprobe(self, history, t):
        if t["kind"] not in ("S", "L", "C", "I", "O", "T"):
            return super().nonprobe(history, t)
        mine = [m["content"] for m in history if m["role"] == "assistant"]        # no truncation in its tests
        short = [mine[i - 1] for i in range(1, t["i"]) if asks_rec(self.turns[i])
                 and len(T.lwords(T.norm(mine[i - 1]))) < 12]
        return short[-1] if short else super().nonprobe(history, t)


class SmallTalk(F.Fake):
    name = "SMALLTALK"

    def nonprobe(self, history, t):
        return SMALL_ACK if t["kind"] == "D" and t["asks"] is False else super().nonprobe(history, t)


def restated(rec):
    """{source turn: VAL probe} for the VAL probes whose latest source turn is an S or C turn (the last such probe
    wins a shared turn): the turns RESTATE and TERSE_VAL confirm, read from the record."""
    kinds = {t["i"]: t["kind"] for t in rec["turns"]}
    out = {}
    for p in rec["probes"]:
        src = [s for s in p["src"] if kinds[s] in "SC"] if p["grader"] == "VAL" else []
        if src:
            out[max(src)] = p
    return out


def bare(p):
    return p["gold"][:1].upper() + p["gold"][1:] + "."


class Restate(F.Fake):
    def __init__(self, name="RESTATE"):
        self.name = name

    def start(self, rec, seed=None, render="plain"):
        super().start(rec, seed, render)
        self.said = restated(rec)
        self.asked = [p["turn"] for p in self.said.values()]

    def nonprobe(self, history, t):
        p = self.said.get(t["i"])
        if p is None or self.name == "TERSE_VAL_CTRL":
            return super().nonprobe(history, t)
        return p["ideal"] if self.name == "RESTATE" else bare(p)

    def probe(self, history, t, p):
        if self.name != "RESTATE" and t["i"] in self.asked:
            return bare(p)
        return super().probe(history, t, p)


class Next(FA.Consist):
    name = "NEXT"

    def value_probe(self, history, t, p):
        if p["grader"] != "DYN":
            return super().value_probe(history, t, p)
        gf = p["gold_fn"]
        if gf["type"] == "pick":
            hits = V.scan(self.turns[gf["src_turn"]]["text"], gf["options"])
            return F.own_sentence(p, hits[1]) if len(hits) > 1 else F.ACK
        it3 = FD.parse_list(self.turns[gf["src_turn"]]["ideal"])[2]
        return p["ideal"].replace(p["ideal_gold"], it3[:1].lower() + it3[1:])


class ListPad(FA.Consist):
    name = "LISTPAD"
    LIST = {"LISTPAD": lambda s: s + "\n4. (or mix them)",
            "LISTBLANK": lambda s: s.replace("\n2.", "\n- \n2.", 1),
            "LISTLS": lambda s: s.replace("\n", "\u2028")}

    def __init__(self, name="LISTPAD"):
        self.name = name

    def q_reply(self, t):
        if self.q_probe[t["i"]]["gold_fn"]["type"] != "pick":
            return self.LIST[self.name](t["ideal"])
        return super().q_reply(t)


FAMILY = {"P_NEVER": ("PERSIST", None), "P_ONCE": ("PERSIST", None), "P_FIRST5": ("PERSIST", None),
          "P_UNIVERSAL": ("PERSIST", None),
          "P_OBEYALL": ("PERSIST", "override"), "L_REPEAT": ("LOOP", None), "L_SELFCOPY": ("LOOP", None),
          "L_STUTTER": ("LOOP", None)}
TESTS = ["CONTINUER", "HISTCHECK", "CAPPER", "SAMPLER", "FORGETFUL", "T0WRONG", "BINDHALF", "ACKER", "PARROT",
         "SMALLTALK", "RESTATE", "TERSE_VAL", "TERSE_VAL_CTRL", "NEXT", "LISTPAD", "LISTBLANK", "LISTLS"]


def make(name):
    if name.startswith("P_"):
        return Persist(name[2:].lower())
    if name.startswith("L_"):
        return Loop(name[2:].lower())
    if name in ListPad.LIST:
        return ListPad(name)
    if name in ("RESTATE", "TERSE_VAL", "TERSE_VAL_CTRL"):
        return Restate(name)
    extra = {"CONTINUER": Continuer, "HISTCHECK": HistCheck, "CAPPER": Capper, "SAMPLER": Sampler,
             "FORGETFUL": Forgetful, "T0WRONG": T0Wrong, "BINDHALF": BindHalf, "ACKER": Acker, "PARROT": Parrot,
             "SMALLTALK": SmallTalk, "NEXT": Next}
    return extra[name]() if name in extra else F.make(name)
