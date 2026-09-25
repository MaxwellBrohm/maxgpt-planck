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
    BINDHALF   IDEAL, but gives the other candidate on the "asked_first" BIND twin (pair rate must be 0)"""
import re

import common as C
import fam_persist as FP
import fakes as F
import render as RD

BASE = dict(C.FILLERS)
STUTTER = "and then it went on and then it went on and then it went on and then it went on."


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


FAMILY = {"P_NEVER": ("PERSIST", None), "P_ONCE": ("PERSIST", None), "P_FIRST5": ("PERSIST", None),
          "P_UNIVERSAL": ("PERSIST", None),
          "P_OBEYALL": ("PERSIST", "override"), "L_REPEAT": ("LOOP", None), "L_SELFCOPY": ("LOOP", None),
          "L_STUTTER": ("LOOP", None)}
TESTS = ["CONTINUER", "HISTCHECK", "CAPPER", "SAMPLER", "FORGETFUL", "T0WRONG", "BINDHALF"]


def make(name):
    if name.startswith("P_"):
        return Persist(name[2:].lower())
    if name.startswith("L_"):
        return Loop(name[2:].lower())
    extra = {"CONTINUER": Continuer, "HISTCHECK": HistCheck, "CAPPER": Capper, "SAMPLER": Sampler,
             "FORGETFUL": Forgetful, "T0WRONG": T0Wrong, "BINDHALF": BindHalf}
    return extra[name]() if name in extra else F.make(name)
