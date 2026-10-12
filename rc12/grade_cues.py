"""RC-12 grader cue additions from the step 5 audit (audit_graders.py found them on real probes). Kept apart so
grade_text.py stays under the file limit; grade_text.HEDGE and grade_text.ABS_CUE splice these in.
  HEDGE_MORE  epistemic hedges the first list let through as confident assertions: "I'd guess X", "If I remember
              right, X", "X, as far as I know", "It should be X", "X, unless it changed", "I can't be certain,
              but X" (each passed all 392 value probes before the fix)
  ABS_MORE    ordinary abstentions the first list failed: "I don't think you've told me that", "That hasn't come
              up yet", "That list skips Thursday", "Nobody is down for Thursday there" (failed every ABS probe)
  near_hedge  a hedge that stands as its own short sentence next to the answer ("Maybe. It is X." / "X.
              Probably."): the sentence-scoped hedge rule could not see it"""
import re

HEDGE_MORE = (r"guess|can't be (?:sure|certain)|cannot be (?:sure|certain)|if i (?:remember|recall)|if i'm not mistaken"
              r"|if memory serves|sounds (?:about )?right|as far as i (?:know|remember|recall|can tell)"
              r"|from what i (?:remember|recall|know)|i (?:would |'d )?assume|seems?|apparently|should be"
              r"|i'd say|i would say|if that's (?:still )?(?:right|correct)|unless")
ABS_MORE = (r"\bi (?:don't|do not) (?:think|believe) you(?:'ve| have)? (?:ever )?(?:told|mentioned|said|shared)\b"
            r"|\b(?:hasn't|has not|didn't|did not|never) (?:come up|been mentioned)\b|\b(?:isn't|is not|not) covered\b"
            r"|\bskips\b|\b(?:nobody|no one)(?: is|'s)? (?:down|scheduled|assigned)\b")
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def near_hedge(text, s0, s1, hedge_rx, max_words=3):
    """True when the sentence right before [s0, s1) or right after it is short (<= max_words) and a hedge."""
    before = [x for x in SPLIT.split(text[:s0]) if x.strip()]
    after = [x for x in SPLIT.split(text[s1:]) if x.strip()]
    for sent in before[-1:] + after[:1]:
        if len(re.findall(r"[A-Za-z0-9']+", sent)) <= max_words and hedge_rx.search(sent):
            return True
    return False
