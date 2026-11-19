"""dev_regrade.py --expect a2 (lock step A2, 2026-10-19, rc12/notes.txt STEP 16): which regrade changes the A2 grader
fixes may make. Synthetic change records (the shape dev_regrade.regrade writes), no stored run, no model. Mutants:
mutation_regrade.py (the a2 block).
  allowed   a probe losing v2_gold, a2_cue or d4_assert, gaining or losing v3_other (each alone), with its unit;
            and v3_shotgun, which the answer-first waiver (graders.answer_first, 2026-11-03) ties to the assertion
            rule: an A2 change to what the first sentence asserts grants or withdraws the waiver of v3_other and
            v3_shotgun
  refused   any other clause moving (v1_degen, v5_echo, v3_guess, v4_voice), a loop flag, an ack mark, the
            role-leak scan; and (review r1, 2026-11-19) a probe GAINING v2_gold, a2_cue or d4_assert: A2 only
            widens what counts as a mention, an assertion or an abstain cue, so those three can only go (a gain is
            a regression such as review r1's capitalized negation value, which the gold no longer matched)
  source    a probe whose OWN source parse (d0_source) flips may move any clause, as in t0gaps
  no probe  (review r2, 2026-11-19) a change with no changed probe (the unit alone moved) is refused, as first and
            cap refuse it: A2 changes only probe grading, so a unit that moves alone means something else moved
Since 2026-11-19 also --expect first and --expect cap (the Nov 3 amendments; untested until then): first lets a probe
only LOSE v3_other and/or v3_shotgun, cap only LOSE degenerate clauses (dev_regrade.CAP_CLAUSES); each gains nothing,
moves only probes and units, and refuses a change with no probe.
Run: python3 -B test_regrade_expect.py   (exit 1 on any failure)"""
import sys

import dev_regrade as DR


ALLOWED = ("v3_other", "v3_shotgun")                 # either way
LOST_ONLY = ("v2_gold", "a2_cue", "d4_assert")        # review r1: only lost


def change(old, new, fields=("probes", "unit")):
    return dict(id="x", family="RECALL", cell="c", fields=list(fields), unit=[0.0, 1.0],
                probes=[dict(turn=3, old=list(old), new=list(new))])


def cases():
    out = []
    for c in ALLOWED:                                # written out here, never read from DR.A2_MOVED
        out += [(f"gain {c}", change([], [c]), True), (f"lose {c}", change([c], []), True)]
    for c in LOST_ONLY:
        out += [(f"lose {c}", change([c], []), True), (f"refuse gain {c}", change([], [c]), False),
                (f"refuse gain {c} beside v3_other", change(["v3_other"], [c]), False),
                (f"source flip may gain {c}", change(["d0_source"], [c]), True)]
    out += [("lose v2_gold and v3_other", change(["v2_gold", "v3_other"], []), True),
            ("refuse unit only, no probe", dict(change([], [], ("unit",)), probes=[]), False),  # review r2
            ("gain v3_other and v3_shotgun (waiver withdrawn)", change([], ["v3_other", "v3_shotgun"]), True)]
    for c in ("v1_degen", "v5_echo", "v3_guess", "v4_voice", "a3_value"):
        out += [(f"refuse {c}", change([], [c]), False), (f"refuse {c} beside v2_gold", change(["v2_gold"], [c]), False)]
    for f in ("flags", "ack_repeat", "ack_of_answer", "leaks"):
        out.append((f"refuse field {f}", change(["v2_gold"], [], ("probes", "unit", f)), False))
    out += [("source flip moves anything", change(["d0_source"], ["v5_echo", "v1_degen"]), True),
            ("source flip on one probe only", dict(change(["d0_source"], ["v5_echo"]), probes=[
                dict(turn=3, old=["d0_source"], new=["v5_echo"]), dict(turn=5, old=[], new=["v1_degen"])]), False)]
    return out


def cases_first():
    out = [(f"first: lose {c}", change([c], []), True) for c in ("v3_other", "v3_shotgun")]
    out += [("first: lose both", change(["v3_other", "v3_shotgun", "v2_gold"], ["v2_gold"]), True),
            ("first: gain v3_other", change([], ["v3_other"]), False),
            ("first: lose v2_gold", change(["v2_gold"], []), False),
            ("first: lose v3_other, gain v4_voice", change(["v3_other"], ["v4_voice"]), False),
            ("first: field flags", change(["v3_other"], [], ("probes", "unit", "flags")), False),
            ("first: unit only, no probe", dict(change([], [], ("unit",)), probes=[]), False)]
    return out


def cases_cap():
    out = [(f"cap: lose {c}", change([c, "v2_gold"], ["v2_gold"]), True)    # written out, never read from CAP_CLAUSES
           for c in ("v1_degen", "a1_degen", "f1_degen", "d1_degen", "r1_degen", "loop_RUNAWAY")]
    out += [("cap: gain v1_degen", change([], ["v1_degen"]), False),
            ("cap: lose v2_gold", change(["v2_gold"], []), False),
            ("cap: lose v3_other", change(["v1_degen", "v3_other"], []), False),
            ("cap: field flags", change(["v1_degen"], [], ("probes", "unit", "flags")), False),
            ("cap: unit only, no probe", dict(change([], [], ("unit",)), probes=[]), False)]
    return out


def main():
    fails = 0
    if "a2" not in DR.EXPECT or DR.EXPECT["a2"] is not DR.expected_a2:
        print("FAIL a2 is not an --expect choice")
        return 1
    for mode, fn in (("first", DR.expected_first), ("cap", DR.expected_cap)):
        if DR.EXPECT.get(mode) is not fn:
            print(f"FAIL {mode} is not an --expect choice running its own rule")
            return 1
    every = [("a2", x) for x in cases()] + [("first", x) for x in cases_first()] + [("cap", x) for x in cases_cap()]
    for mode, (name, c, want) in every:
        got = bool(DR.EXPECT[mode](c))
        if got != want:
            fails += 1
            print(f"FAIL {name}: got {got}, want {want}")
    print(f"test_regrade_expect: {len(every) - fails} of {len(every)} pass")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
