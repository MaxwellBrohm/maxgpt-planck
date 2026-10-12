"""RC-12 fake scripted responders, part 3: the step 5 AUDIT cheaters (adversarial; no model). They follow the
generic cheater skeleton of fakes.py (fixed ack off-probe, IDEAL on I / O / Q / ROLE X, the IDEAL sentence with
the picked value on value probes), so a unit fails or passes because of the rule alone.
  Value rules (audit_rules.rule_picks; user turns only):
    ANTI_OVERLAP   turn sharing the FEWEST content words with the question (latest on ties)
    ANTI_WORDING   turn with the SHORTEST word run shared with the question (latest on ties)
    LONGEST / SHORTEST   value of the longest / shortest value-bearing turn
    LEASTFREQ / MOSTFREQ value mentioned least / most often
    POS2 / PENULT / ANTEPENULT   2nd distinct value / 2nd and 3rd distinct value from the end
    PENULT_MARKER  value of the 2nd-to-last correction-marker turn
    PREV_FRESH     value of the turn just before the latest unmarked statement ("the value before the last new thing")
    EXL_OVERLAP / EXL_WORDING / EXL_MARKER   OVERLAP / WORDING / MARKER after dropping the newest value turn
  Reference rules, reported but NOT gated (they are the skill a cell measures, not a shortcut past it):
    MYONLY         latest value in a turn with no holder noun, no "'s", no he/she (RECALL / ROLE binding)
    MARKER_OBJ     latest correction naming the asked object (the U-same skill by the cell's definition)
  Structure cheaters:
    ORDER_ABS  positional abstain: with a table in the conversation it abstains on the FIRST table question and
               answers the second by an exact table lookup (the key's value in the table the question names);
               without one it answers the first value probe (FIRST) and abstains on every later one
    CONSIST    OWN without reading its own reply: picks option 1 at Q and later answers option 1 of the user's Q
               list; lists: re-derives the same list (the annotation) and answers its 2nd item
    VARIED     LOOP: a different canned sentence every turn, never about the story (non-degenerate, contentless)"""
import re

import audit_rules as AR
import fakes as F
import pools_vals as V
import quick_shortcuts as QS

VALUE_RULES = ["ANTI_OVERLAP", "ANTI_WORDING", "LONGEST", "SHORTEST", "LEASTFREQ", "MOSTFREQ", "POS2", "PENULT",
               "ANTEPENULT", "PENULT_MARKER", "PREV_FRESH", "EXL_OVERLAP", "EXL_WORDING", "EXL_MARKER"]
REFERENCE = ["MYONLY", "MARKER_OBJ"]
STRUCT = ["ORDER_ABS", "CONSIST", "VARIED"]
AUDIT = VALUE_RULES + REFERENCE + STRUCT
ALIAS = {"ANTI_OVERLAP": "ANTI_OVERLAP_L"}
CANNED = ["Sure, here is a little more.", "That is a lovely direction to take it.", "Happy to keep it going.",
          "Here comes the next part.", "What a fun thing to picture.", "I like where this is heading.",
          "Let me add another piece.", "That works nicely for the tale.", "Good idea, continuing now.",
          "This part is my favorite so far.", "Onward with the next bit.", "Thanks, here is what happens."]


class RulePicker(F.Picker):
    def value_probe(self, history, t, p):
        is_list = p["grader"] == "DYN" and p["gold_fn"]["type"] == "list2"
        q = dict(p, pool_values=p["lure_items"]) if is_list else p
        v = AR.rule_picks(self.rec, q).get(ALIAS.get(self.rule, self.rule))
        return F.ACK if v is None else self.gold_sentence(history, p, v)


def table_lookup(rec, p):
    """the value right after '<key>:' in the table turn named by the question (old / new), else None."""
    age = "new" if re.search(r"\bnew\b", p["question"], re.I) else "old"
    key = [w for w in re.findall(r"[A-Z][a-z]+", p["question"]) if w in V.WEEKDAY + V.MONTH]
    for t in rec["turns"]:
        if t["i"] < p["turn"] and t["kind"] == "T" and re.search(rf"\b{age}\b", t["text"]) and key:
            m = re.search(rf"\b{key[0]}: ([A-Za-z]+)", t["text"])
            return m.group(1) if m else None
    return None


class OrderAbs(F.Cheater):
    name = "ORDER_ABS"

    def value_probe(self, history, t, p):
        order = sorted(x["turn"] for x in self.rec["probes"] if x["grader"] in ("VAL", "ABS", "DYN"))
        first = p["turn"] == order[0]
        if any(x["kind"] == "T" for x in self.rec["turns"]):
            if first:
                return F.ABSTAIN
            v = table_lookup(self.rec, p)
            return self.gold_sentence(history, p, v) if v else F.ACK
        if not first:
            return F.ABSTAIN
        v = QS.picks(self.rec, p).get("FIRST")
        return self.gold_sentence(history, p, v) if v else F.ACK


class Consist(F.Cheater):
    name = "CONSIST"

    def q_reply(self, t):
        p = self.q_probe[t["i"]]
        if p["gold_fn"]["type"] != "pick":
            return t["ideal"]
        return F.sub_value(t["ideal"], p["ideal_gold"], p["gold_fn"]["options"][0])

    def value_probe(self, history, t, p):
        if p["grader"] != "DYN":
            v = QS.picks(self.rec, p).get("FIRST")
            return self.gold_sentence(history, p, v) if v else F.ACK
        gf = p["gold_fn"]
        qtext = self.turns[gf["src_turn"]]["text"]            # the USER's Q turn, never its own reply
        if gf["type"] == "pick":
            hits = V.scan(qtext, gf["options"])
            return F.own_sentence(p, hits[0]) if hits else F.ACK
        return p["ideal"]                                     # re-derives the same list: item 2 of the annotation


class Varied(F.Fake):
    name = "VARIED"

    def respond(self, history, t, p):
        if self.rec["family"] != "LOOP":
            return super().respond(history, t, p)
        return CANNED[(t["i"] - 1) % len(CANNED)]


def make(name):
    if name in VALUE_RULES + REFERENCE:
        return RulePicker(name)
    return {"ORDER_ABS": OrderAbs, "CONSIST": Consist, "VARIED": Varied}[name]()
