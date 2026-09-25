"""RC-12 step 5 audit, part 1 (DIAGNOSTIC, value level): a wide sweep of cheap value-picking rules on the REAL dev
VAL probes, beyond quick_shortcuts.py's six. A rule reads USER turns only (never an annotation, never a kind) and
picks a VALUE; the rules that come near a bar get a real responder in fakes_audit.py and go through the runner.
  positions      POS2 / POS3 (2nd / 3rd distinct value), PENULT / ANTEPENULT (2nd / 3rd distinct from the end)
  frequency      MOSTFREQ / LEASTFREQ (mention count; ties: latest)
  anti-echo      ANTI_OVERLAP_E / _L (turn sharing FEWEST content words with the question; earliest / latest)
                 ANTI_WORDING (turn with the SHORTEST longest word run shared with the question; latest)
  markers        PENULT_MARKER (value of the 2nd-to-last marker turn), NOMARK_LAST (latest turn with no marker),
                 MARKER_OBJ (latest marker turn sharing a content word with the question, else OVERLAP_A)
                 PREV_FRESH (value of the turn just before the latest UNMARKED value turn, a fresh statement)
  anti-recency   EXL_LAST / EXL_OVERLAP / EXL_WORDING / EXL_MARKER: drop the LAST value turn, then LAST, OVERLAP
                 (latest wins), WORDING and MARKER on the rest ("the answer is never the newest value")
  turn shape     LONGEST / SHORTEST (value in the longest / shortest value-bearing turn, latest on ties)
  binding        MYONLY (latest value in a turn with no holder noun, no "'s" and no pronoun he/she/his/her)
                 PRONOUN (latest value in a turn that starts with a pronoun or demonstrative)
BIND is scored per pair (both twins). GATE (step 5): every rule here and quick_shortcuts' six, except the reference
rules MYONLY / PRONOUN (holder binding, the skill of RECALL, ROLE and BIND) and MARKER_OBJ (U-same's skill), must
stay <= 0.50 on every cell and <= 0.40 pooled per composite family, at the value level (T0, COMPOSE and the
single-candidate X probes of RECALL:abstain left out). Writes logs/audit_rules.txt; exit 1 on a violation.
Run: python3 -B audit_rules.py"""
import os
import re
import sys
from collections import defaultdict

import pools_vals as V
import quick_shortcuts as QS
import test_gens as T

MARK = QS.MARKERS
HOLD = [h.split()[-1] for h in V.HOLDERS] + ["friend", "partner", "wife", "husband", "kids", "mom", "dad"]
HOLD_RX = re.compile(r"\b(" + "|".join(HOLD) + r")\b|'s\b|\b(he|she|his|her|hers|him)\b", re.I)
PRON_RX = re.compile(r"^\s*(it|that|this|they|he|she|the one|same|make that|scratch that)\b", re.I)
RULES = ["POS2", "POS3", "PENULT", "ANTEPENULT", "MOSTFREQ", "LEASTFREQ", "ANTI_OVERLAP_E", "ANTI_OVERLAP_L",
         "ANTI_WORDING", "PENULT_MARKER", "NOMARK_LAST", "MARKER_OBJ", "PREV_FRESH", "LONGEST", "SHORTEST", "MYONLY", "PRONOUN", "EXL_LAST", "EXL_OVERLAP",
         "EXL_WORDING", "EXL_MARKER"]


def words(s):
    return re.findall(r"[a-z0-9']+", s.lower())


def vturns(r, p):
    vals = QS.pool_values(p)
    out = []
    for t in r["turns"][:p["turn"] - 1]:
        if t["kind"] == "D":
            continue
        ms = QS.mentions(t["text"], vals)
        if ms:
            out.append((t, [m[1] for m in ms]))
    return out


def has_mark(text):
    low = text.lower()
    return any(m in low for m in MARK)


def rule_picks(r, p):
    vt = vturns(r, p)
    if not vt:
        return {}
    seq = [v for _, ms in vt for v in ms]
    dist = list(dict.fromkeys(seq))
    out = {}
    for name, k in (("POS2", 1), ("POS3", 2), ("PENULT", -2), ("ANTEPENULT", -3)):
        if -len(dist) <= k < len(dist):
            out[name] = dist[k]
    cnt = {v: seq.count(v) for v in dist}
    last_at = {v: max(i for i, x in enumerate(seq) if x == v) for v in dist}
    out["MOSTFREQ"] = max(dist, key=lambda v: (cnt[v], last_at[v]))
    out["LEASTFREQ"] = min(dist, key=lambda v: (cnt[v], -last_at[v]))
    qc = QS.content(p["question"])
    qw = words(p["question"])
    ov = [(len(QS.content(t["text"]) & qc), t["i"], ms) for t, ms in vt]
    lo = min(x[0] for x in ov)
    out["ANTI_OVERLAP_E"] = [x for x in ov if x[0] == lo][0][2][0]
    out["ANTI_OVERLAP_L"] = [x for x in ov if x[0] == lo][-1][2][-1]
    runs = [(QS.longest_run(words(t["text"]), qw)[0], t["i"], ms) for t, ms in vt]
    lr = min(x[0] for x in runs)
    out["ANTI_WORDING"] = [x for x in runs if x[0] == lr][-1][2][-1]
    mk = [ms for t, ms in vt if has_mark(t["text"])]
    if len(mk) >= 2:
        out["PENULT_MARKER"] = mk[-2][-1]
    nm = [ms for t, ms in vt if not has_mark(t["text"])]
    if nm:
        out["NOMARK_LAST"] = nm[-1][-1]
    mo = [ms for t, ms in vt if has_mark(t["text"]) and QS.content(t["text"]) & qc]
    out["MARKER_OBJ"] = mo[-1][-1] if mo else max(ov, key=lambda x: (x[0], x[1]))[2][0]
    fresh = [j for j, (t, _) in enumerate(vt) if not has_mark(t["text"])]
    out["PREV_FRESH"] = vt[fresh[-1] - 1][1][-1] if fresh and fresh[-1] > 0 else seq[-1]
    ln = [(len(words(t["text"])), t["i"], ms) for t, ms in vt]
    out["LONGEST"] = max(ln, key=lambda x: (x[0], x[1]))[2][-1]
    out["SHORTEST"] = min(ln, key=lambda x: (x[0], -x[1]))[2][-1]
    my = [ms for t, ms in vt if not HOLD_RX.search(t["text"])]
    if my:
        out["MYONLY"] = my[-1][-1]
    pr = [ms for t, ms in vt if PRON_RX.search(t["text"])]
    if pr:
        out["PRONOUN"] = pr[-1][-1]
    ex = vt[:-1] or vt                            # anti-recency: the newest value turn is never the answer
    out["EXL_LAST"] = ex[-1][1][-1]
    out["EXL_OVERLAP"] = max(ex, key=lambda x: (len(QS.content(x[0]["text"]) & qc), x[0]["i"]))[1][0]
    out["EXL_WORDING"] = max(ex, key=lambda x: (QS.longest_run(words(x[0]["text"]), qw)[0], x[0]["i"]))[1][-1]
    em = [ms for t, ms in ex if has_mark(t["text"])]
    out["EXL_MARKER"] = em[-1][-1] if em else ex[0][1][0]
    return out


def table(recs, rules, picker):
    score = defaultdict(lambda: defaultdict(list))
    pairs = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["knowledge"] or r["family"] == "T0":
            continue
        for p in r["probes"]:
            if p["grader"] != "VAL" or not p["gold"]:
                continue
            if r["cell"] == "abstain":
                continue
            got = picker(r, p)
            key = f"{r['family']}:{r['cell']}"
            for n in rules:
                right = got.get(n) == p["gold"]
                if r["family"] == "BIND":
                    pairs[key][(n, r["meta"]["pair_id"])].append(right)
                else:
                    score[key][n].append(right)
    for key, d in pairs.items():
        for (n, _), v in d.items():
            score[key][n].append(all(v))
    return score


def render(score, rules, title):
    lines = [title, f"{'family:cell':20s}" + "".join(f"{n[:9]:>10s}" for n in rules) + "  units"]
    fam = defaultdict(lambda: defaultdict(list))
    for key in sorted(score):
        row = score[key]
        lines.append(f"{key:20s}" + "".join(f"{sum(row[n]) / len(row[n]):10.2f}" for n in rules)
                     + f"  {len(row[rules[0]])}")
        if key != "TWOHOP:COMPOSE":
            for n in rules:
                fam[key.split(":")[0]][n] += row[n]
    for f in sorted(fam):
        lines.append(f"{f + ' (pool)':20s}" + "".join(f"{sum(fam[f][n]) / len(fam[f][n]):10.2f}" for n in rules))
    flag = [f"{k}:{n}={sum(v[n]) / len(v[n]):.2f}" for k, v in sorted(score.items()) for n in rules
            if k != "TWOHOP:COMPOSE" and sum(v[n]) / len(v[n]) > 0.5]
    lines.append("cells above 0.50 (value level): " + (", ".join(flag) or "none"))
    return lines


REFERENCE = {"MYONLY", "PRONOUN", "MARKER_OBJ"}


def all_picks(r, p):
    return dict(QS.picks(r, p), **rule_picks(r, p))


def violations(recs):
    """(rule, where, rate) above the bars for every non-reference rule (value level)."""
    names = [n for n in QS.NAMES + RULES if n not in REFERENCE]
    score = table(recs, names, all_picks)
    out, fam = [], defaultdict(lambda: defaultdict(list))
    for key, row in score.items():
        if key == "TWOHOP:COMPOSE":
            continue
        for n in names:
            fam[key.split(":")[0]][n] += row[n]
            if sum(row[n]) / len(row[n]) > 0.50:
                out.append((n, key, sum(row[n]) / len(row[n])))
    for f, row in fam.items():
        out += [(n, f, sum(v) / len(v)) for n, v in row.items() if sum(v) / len(v) > 0.40]
    return sorted(out)


def main():
    recs = T.load()
    lines = ["RC-12 audit_rules.py (diagnostic): value picks of cheap rules on the real dev VAL probes.",
             "RECALL:abstain X probes (one candidate) and T0 / K are left out; BIND per pair.", ""]
    half = len(RULES) // 2
    for chunk in (RULES[:half], RULES[half:]):
        lines += render(table(recs, chunk, rule_picks), chunk, "") + [""]
    bad = violations(recs)
    lines += ["GATE (non-reference rules incl. quick_shortcuts: cell <= 0.50, family <= 0.40): "
              + ("PASS" if not bad else "FAIL " + ", ".join(f"{n} {k}={v:.2f}" for n, k, v in bad))]
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.join(T.HERE, "logs"), exist_ok=True)
    with open(os.path.join(T.HERE, "logs", "audit_rules.txt"), "w") as f:
        f.write(text)
    print(text)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main())
