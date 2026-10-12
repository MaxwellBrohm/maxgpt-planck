"""RC-12 fake scripted responders (SPEC s5), part 1: IDEAL, IDEAL_ALT and the generic cheaters. No model.
runner.py drives a fake through the REAL 12-turn loop: reply(history, i) sees only the history the runner built (its
OWN earlier replies, maybe truncated) plus the record's annotations (a script may read them; a model never does).
  IDEAL      turns[i].ideal, except OWN probes: it reads its OWN Q-turn reply from the history and answers with what
             it picked / listed there (so it only scores if the runner fed its replies back).
  IDEAL_ALT  IDEAL, but at OWN Q turns it picks another option / swaps list items 2 and 3 (G-DYN gold must follow
             the model's own reply, not the annotation).
  Generic cheaters: S L C D T turns "Okay, noted."; I, O turns the IDEAL ack; Q turns IDEAL (FIRST picks option 1,
  LAST option 3); ROLE X probes IDEAL; PERSIST and LOOP probes "Okay, noted.". Value probes (VAL, ABS, DYN) get
  the IDEAL sentence with the gold replaced by the value the rule picks ("It's <v>." on ABS probes):
    FIRST, LAST, WORDING, OVERLAP_A, OVERLAP_B, MARKER  quick_shortcuts.picks (E004 O3-O5)
    ECHO / DEFLECT  every probe (any grader): the user turn restated / "I'm not sure, could you remind me?"
    SHOTGUN    every candidate listed ("Tuesday, Monday and Wednesday")
    USERVOICE  the IDEAL sentence with I/you swapped ("My massage session is on Wednesday.", "You picked
               Quizzards."); if nothing swaps "My <object> is <gold>." (own list: "Your second idea was <item>.")
    ABSTAIN    "You haven't told me that." on every value probe (LOOKUP always-answer is WORDING)
    ROLECAP    speaks as the user: parrots user statements, USERVOICE on value probes, and on ROLE X it says
               I'm <name>. I work as a <job>, and I live in <town>."""
import re

import grade_fmt_dyn as FD
import pools_vals as V
import quick_shortcuts as QS
import render as RD

ACK = "Okay, noted."
DEFLECT = "I'm not sure, could you remind me?"
ABSTAIN = "You haven't told me that."
SWAP = {"you are": "I am", "i am": "you are", "you're": "I'm", "i'm": "you're", "you've": "I've", "i've": "you've",
        "you'd": "I'd", "i'd": "you'd", "you'll": "I'll", "i'll": "you'll", "yourself": "myself", "myself": "yourself",
        "yours": "mine", "mine": "yours", "your": "my", "my": "your", "you": "I", "me": "you", "i": "you"}
SWAP_RX = re.compile(r"\b(" + "|".join(re.escape(k) for k in sorted(SWAP, key=len, reverse=True)) + r")\b", re.I)
OBJ_YOU = re.compile(r"\b(?:to|for|with|told|tell|help|give|gave|about|at|thank|thanks|let|show)\s*$", re.I)


def swap_person(text):
    def rep(m):
        k = m.group(0).lower()
        out = SWAP[k]
        if k == "you" and OBJ_YOU.search(text[:m.start()]):
            out = "me"
        start = m.start() == 0 or re.search(r"[.!?:]\s*$", text[:m.start()])
        return out[:1].upper() + out[1:] if (start or out.startswith("I")) else out
    return SWAP_RX.sub(rep, text)


def sub_value(text, old, new):
    return V.value_re(old).sub(new, text, count=1)


def joined(vals):
    vals = list(dict.fromkeys(vals))
    return vals[0] if len(vals) == 1 else ", ".join(vals[:-1]) + " and " + vals[-1]


class Fake:
    name = "IDEAL"

    def start(self, rec, seed=None, render="plain"):
        self.rec, self.seed, self.render = rec, seed, render
        self.turns = {t["i"]: t for t in rec["turns"]}
        self.probes = {}
        for p in rec["probes"]:
            assert p["turn"] not in self.probes, (rec["id"], p["turn"])
            self.probes[p["turn"]] = p
        self.q_probe = {p["gold_fn"]["src_turn"]: p for p in rec["probes"] if p.get("gold_fn")}

    def count(self, messages, render):
        return RD.proxy_tokens(messages)

    def reply(self, history, i):
        text = self.respond(history, self.turns[i], self.probes.get(i))
        return text, ("eot" if self.render == "template" else "eos")

    def respond(self, history, t, p):
        return self.nonprobe(history, t) if p is None else self.probe(history, t, p)

    # IDEAL policy
    def nonprobe(self, history, t):
        return self.q_reply(t) if t["kind"] == "Q" else t["ideal"]

    def q_reply(self, t):
        return t["ideal"]

    def probe(self, history, t, p):
        if p["grader"] == "DYN":
            v = self.own_value(history, p)
            return own_sentence(p, v) if v else "I don't remember what I said."
        return t["ideal"]

    def own_value(self, history, p):
        gf = p["gold_fn"]
        src = RD.own_reply(history, p["turn"], gf["src_turn"])
        if src is None:
            return None
        if gf["type"] == "pick":
            hits = V.scan(src, gf["options"])
            return hits[0] if hits else None
        items = FD.parse_list(src)
        return items[1][:1].lower() + items[1][1:] if len(items) >= 2 else None

    def own_items(self, history, p):
        src = RD.own_reply(history, p["turn"], p["gold_fn"]["src_turn"]) or ""
        return [x[:1].lower() + x[1:] for x in FD.parse_list(src)]


def own_sentence(p, v):
    if p["gold_fn"]["type"] == "pick":
        return sub_value(p["ideal"], p["ideal_gold"], v)
    return p["ideal"].replace(p["ideal_gold"], v)


class IdealAlt(Fake):
    name = "IDEAL_ALT"

    def q_reply(self, t):
        p = self.q_probe[t["i"]]
        if p["gold_fn"]["type"] == "pick":
            opts = p["gold_fn"]["options"]
            return sub_value(t["ideal"], p["ideal_gold"], opts[(opts.index(p["ideal_gold"]) + 1) % len(opts)])
        lines = t["ideal"].split("\n")
        nums = [k for k, s in enumerate(lines) if re.match(r"\s*\d+[.)]\s", s)]
        a, b = nums[1], nums[2]
        la, lb = lines[a], lines[b]
        lines[a] = la.split(" ", 1)[0] + " " + lb.split(" ", 1)[1]
        lines[b] = lb.split(" ", 1)[0] + " " + la.split(" ", 1)[1]
        return "\n".join(lines)


class Cheater(Fake):
    """generic cheater skeleton: fixed ack off-probe, IDEAL on ROLE X, ack on PERSIST / LOOP probes."""

    def nonprobe(self, history, t):
        if t["kind"] in ("I", "O"):
            return t["ideal"]
        if t["kind"] == "Q":
            return self.q_reply(t)
        return ACK

    def probe(self, history, t, p):
        if p["grader"] == "ROLEX":
            return t["ideal"]
        if p["grader"] in ("FMT", "LOOP"):
            return ACK
        return self.value_probe(history, t, p)

    def gold_sentence(self, history, p, v):
        """the IDEAL sentence of a value probe with its gold replaced by v (ABS: "It's v.")."""
        if p["grader"] == "ABS":
            return f"It's {v}."
        if p["grader"] == "DYN":
            return own_sentence(p, v)
        return sub_value(p["ideal"], p["gold"], v)


class Picker(Cheater):
    def __init__(self, rule):
        self.name = self.rule = rule

    def q_reply(self, t):
        p = self.q_probe[t["i"]]
        if p["gold_fn"]["type"] != "pick" or self.rule not in ("FIRST", "LAST"):
            return t["ideal"]
        opts = p["gold_fn"]["options"]
        return sub_value(t["ideal"], p["ideal_gold"], opts[0] if self.rule == "FIRST" else opts[-1])

    def value_probe(self, history, t, p):
        is_list = p["grader"] == "DYN" and p["gold_fn"]["type"] == "list2"
        v = QS.picks(self.rec, dict(p, pool_values=p["lure_items"]) if is_list else p).get(self.rule)
        return ACK if v is None else self.gold_sentence(history, p, v)


class Echo(Cheater):
    name = "ECHO"

    def probe(self, history, t, p):
        return t["text"]


class Deflect(Cheater):
    name = "DEFLECT"

    def probe(self, history, t, p):
        return DEFLECT


class Shotgun(Cheater):
    name = "SHOTGUN"

    def value_probe(self, history, t, p):
        if p["grader"] == "DYN" and p["gold_fn"]["type"] == "list2":
            return own_sentence(p, joined(p["lure_items"] + self.own_items(history, p)))
        cands = p.get("candidates") or p.get("pool_values") or []
        return self.gold_sentence(history, p, joined(cands)) if cands else ACK


class UserVoice(Cheater):
    name = "USERVOICE"

    def value_probe(self, history, t, p):
        obj = " ".join(p.get("object_words") or []) or "answer"
        if p["grader"] == "ABS":
            s = swap_person(p["ideal"])
            return s if s != p["ideal"] else f"I don't know my {obj}."
        gold = self.own_value(history, p) if p["grader"] == "DYN" else p["gold"]
        if gold is None:
            return ACK
        ideal = self.gold_sentence(history, p, gold)
        s = swap_person(ideal)
        if s != ideal:
            return s
        return f"Your second idea was {gold}." if p["holder"] == "assistant" else f"My {obj} is {gold}."


class Abstain(Cheater):
    name = "ABSTAIN"

    def value_probe(self, history, t, p):
        return ABSTAIN


class RoleCap(UserVoice):
    name = "ROLECAP"

    def nonprobe(self, history, t):
        if t["kind"] in ("S", "L", "C"):
            return t["text"]
        return super().nonprobe(history, t)

    def probe(self, history, t, p):
        if p["grader"] == "ROLEX":
            u = p["user_facts"]
            art = "an" if u["job"][:1].lower() in "aeiou" else "a"
            return f"I'm {u['name']}. I work as {art} {u['job']}, and I live in {u['town']}."
        return super().probe(history, t, p)


GENERIC = ["FIRST", "LAST", "WORDING", "OVERLAP_A", "OVERLAP_B", "MARKER", "ECHO", "DEFLECT", "SHOTGUN",
           "USERVOICE", "ABSTAIN", "ROLECAP"]
CLASSES = {"IDEAL": Fake, "IDEAL_ALT": IdealAlt, "ECHO": Echo, "DEFLECT": Deflect, "SHOTGUN": Shotgun,
           "USERVOICE": UserVoice, "ABSTAIN": Abstain, "ROLECAP": RoleCap}


def make(name):
    if name in CLASSES:
        return CLASSES[name]()
    if name in QS.NAMES:
        return Picker(name)
    import fakes_audit  # step 5 audit cheaters (fakes_audit.py); raises KeyError for an unknown name
    return fakes_audit.make(name)
