"""OOD-H paired difference and the Level R wording rule (prereg draft s1, s9, s14; oodh/DESIGN.txt s7). Pure
Python, no model. score_stats imports it as ST.OW. It says how Level R is worded IF Level R holds on sealed RC-12
(s9: CI lower bound >= -3, T0 >= 0.90, >= 3 training seeds); it never decides that Level R holds.

Input: each model's OOD-H transcript rows (rc12 runner rows played through oodh/play.FixedHistory and graded by the
RC-12 graders; family OODH; unit = the thread's mean probe result, strict). score.select keeps the sampling seeds
(greedy only when a model has no sampled run), as for RC-12.
  thread_table(rows)  {train_seed: {seed: {thread_id: unit}}}. Refused (ValueError): no rows, a row that is not
      OODH, an --own-cf row, a unit that is not a number in [0, 1], a thread twice in one run, two renders.
  paired_diff(rows_a, rows_b, n=10000, seed=0, expected=None, part1=None)  D = M(a) - M(b) in points (a =
      Planck, b = the comparator); M = 100 x the mean over threads, then over sampling seeds, then over training
      seeds (as score.nested). Paired bootstrap: each resample draws the threads with replacement, the SAME draw for
      both models; training seeds are resampled per model and sampling seeds inside each drawn training seed
      (nested); percentile 95% CI. Refused: thread sets that differ between any two runs (of one model or across
      the two), or from expected (the item file's ids); renders that differ between the models; two responders in
      one model's rows; comparator rows whose responder is not Qwen2.5-0.5B-Instruct (s1 names it), or Planck rows
      that are; one model's selected rows greedy and the other's sampled (s9: D is sampling T 0.6). A plain-render
      or greedy pair is computed but labeled ci["diagnostic"] (s9: D is the template render, sampled), and decide()
      words no claim from it. part1 (the Part 1 item ids) gives ci["part2_threads"], the scored threads outside it.
  decide(ci, part2_complete=False, fill=None)  the s1 rule; refused (ValueError) for a diagnostic ci, and for
      part2_complete unless ci["part2_threads"] > 0 (Part 2 threads were scored). D worse than -5 points (D < -5;
      exactly -5 is not worse) -> "qualified": the claim says "non-inferior ... on RC-12's format", the OOD-H
      sentence stays beside it, and the unqualified wording is refused. Otherwise "unqualified" (the qualified
      wording is then allowed too). D is compared after rounding to 1e-6 points, so float noise cannot cross the bar (a real nonzero
      difference is about 0.004 points or more: 150 threads of 1 to 3 probes, up to 10 x 3 seeds). provisional =
      not part2_complete (s1: provisional until OOD-H is complete, Part 2 in January); the note is then in the text.
      fill: the other s1 numbers {N, B, k, c, n}; a missing one stays a placeholder.
  refusals(text, dec)  why a claim text breaks the rule: while qualified, any "non-inferior" (any case, with or
      without the hyphen, a Unicode dash or a space) that does not start the exact QUALIFIED phrase; the OOD-H
      sentence (with these numbers) missing; the provisional note missing.
CLI: python3 -B oodh_wording.py --planck T.jsonl [...] --comparator T.jsonl [...] [--data ITEMS.jsonl]
     [--part2-complete] [--part1 PART1.jsonl] [--n 10000] [--fill N=30,B=18,k=3,c=900,n=5400]
     prints the CI and the decision as JSON (exit 0; decision null for a diagnostic CI); a refused input exits 2.
     --part2-complete needs --data (the scored thread set) and the Part 1 item file (default
     sealed/oodh/oodh_part1.jsonl): --data must hold threads outside Part 1, and the rows must cover them."""
import argparse
import json
import os
import random
import re
import sys

import score as S

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PART1 = os.path.join(REPO, "sealed", "oodh", "oodh_part1.jsonl")
COMPARATOR = "Qwen/Qwen2.5-0.5B-Instruct"   # s1 and s9 name it; runner rows carry the bare model id as responder
OODH_MARGIN = -5.0             # s1: "worse than -5 points"
DECIMALS = 6                   # D is compared after rounding to this many decimals (float noise only)
LEVEL_R = ("Planck-{N}M ({N}M total parameters, {B}M non-embedding) is non-inferior to Qwen2.5-0.5B-Instruct on "
           "RC-12, a pre-registered, sealed 12-turn conversation test. On its own conversation history, the lower "
           "bound of the paired 95% confidence interval on the RC-12 composite (bootstrap over conversations and "
           "seeds) is no worse than -3 points ({k} training seeds, {c} conversations, {n} scored turns). {oodh} It "
           "is the smallest model on our curve that does this.")
OODH_SENTENCE = ("On OOD-H, {h} conversations with human-written user turns, the paired difference is {d} points "
                 "(95% CI {lo} to {hi}).")
UNQUALIFIED = "non-inferior to Qwen2.5-0.5B-Instruct on RC-12,"
QUALIFIED = "non-inferior to Qwen2.5-0.5B-Instruct on RC-12's format,"
NON_INFERIOR = r"non[\s\-\u2010-\u2015]?inferior"   # any case; hyphen, Unicode dash, space or none
PROVISIONAL = "Provisional until OOD-H is complete: Part 2 (the raters' scripts, collected in January) is not scored."


class _Keep(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _unit(r):
    u = r.get("unit")
    if isinstance(u, bool) or not isinstance(u, (int, float)) or not 0 <= u <= 1:     # NaN fails the range too
        raise ValueError(f"thread {r.get('id')}: unit {u!r} is not a number in [0, 1]")
    return float(u)


def thread_table(rows):
    if not rows:
        raise ValueError("no OOD-H rows")
    for r in rows:
        if r.get("family") != "OODH":
            raise ValueError(f"row {r.get('id')} is family {r.get('family')}, not OODH")
        if r.get("own_cf"):
            raise ValueError(f"row {r.get('id')} is an --own-cf row (OOD-H has none)")
    renders = {r.get("render") for r in rows}
    if len(renders) != 1:
        raise ValueError(f"two renders in one model's rows: {sorted(map(str, renders))}")
    out = {}
    for r in S.select(rows):
        run = out.setdefault(r.get("train_seed", 0), {}).setdefault(r["seed"], {})
        if r["id"] in run:
            raise ValueError(f"thread {r['id']} twice in run {r.get('train_seed', 0)}/{r['seed']}")
        run[r["id"]] = _unit(r)
    return out


def _threads(tables, expected):
    sets = {(m, tr, sd): frozenset(run) for m, t in tables.items() for tr, by in t.items() for sd, run in by.items()}
    union = frozenset().union(*sets.values())
    for k, s in sets.items():
        if s != union:
            raise ValueError(f"run {k} lacks {len(union - s)} of {len(union)} threads (e.g. {sorted(union - s)[0]}): "
                             "unequal thread sets, no paired difference")
    if expected is not None and union != set(expected):
        raise ValueError(f"{len(union ^ set(expected))} threads differ from the item file's {len(set(expected))}")
    return sorted(union)


def _mean(xs):
    return sum(xs) / len(xs)


def _level(arrs, idx, rng=None):
    """100 x nested mean over the threads idx, sampling seeds, training seeds; rng None = the runs as observed."""
    trains = list(arrs)
    if rng is not None:
        trains = [rng.choice(trains) for _ in trains]
    per_train = []
    for tr in trains:
        seeds = list(arrs[tr])
        if rng is not None:
            seeds = [rng.choice(seeds) for _ in seeds]
        per_train.append(_mean([_mean([arrs[tr][sd][i] for i in idx]) for sd in seeds]))
    return 100 * _mean(per_train)


def _responder(rows, side):
    names = {r.get("responder") for r in rows}
    if len(names) != 1:
        raise ValueError(f"{side}: {len(names)} responders in one model's rows: {sorted(map(str, names))[:3]}")
    return names.pop()


def is_comparator(name):
    return isinstance(name, str) and (name == COMPARATOR or name.endswith(":" + COMPARATOR))


def _greedy(table):
    return all(sd is None for by in table.values() for sd in by)


def paired_diff(rows_a, rows_b, n=10000, seed=0, expected=None, part1=None):
    ta, tb = thread_table(rows_a), thread_table(rows_b)
    ra, rb = rows_a[0].get("render"), rows_b[0].get("render")
    if ra != rb:
        raise ValueError(f"renders differ between the models: {ra} vs {rb}")
    na, nb = _responder(rows_a, "Planck"), _responder(rows_b, "comparator")
    if not is_comparator(nb):
        raise ValueError(f"comparator responder {nb!r} is not {COMPARATOR} (s1 names it)")
    if is_comparator(na):
        raise ValueError(f"the Planck rows' responder {na!r} is the comparator")
    if _greedy(ta) != _greedy(tb):
        raise ValueError("one model's selected rows are greedy and the other's sampled: no paired difference")
    diagnostic = ([] if ra == "template" else [f"{ra} render (s9: D is the template render)"]) + \
        (["greedy rows (s9: D is sampling T 0.6)"] if _greedy(ta) else [])
    ids = _threads({"a": ta, "b": tb}, expected)
    aa, ab = ({tr: {sd: [run[t] for t in ids] for sd, run in by.items()} for tr, by in t.items()} for t in (ta, tb))
    m = len(ids)
    idx = list(range(m))
    point = _level(aa, idx) - _level(ab, idx)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        draw = [rng.randrange(m) for _ in range(m)]
        diffs.append(_level(aa, draw, rng) - _level(ab, draw, rng))
    diffs.sort()
    return dict(D=point, lo=diffs[int(0.025 * n)], hi=diffs[min(n - 1, int(0.975 * n))], n=n, h=m, render=ra,
                runs_a=sum(len(by) for by in ta.values()), runs_b=sum(len(by) for by in tb.values()),
                train_seeds_a=len(ta), train_seeds_b=len(tb), items_checked=expected is not None,
                responder_a=na, responder_b=nb, greedy=_greedy(ta), diagnostic=diagnostic,
                part2_threads=None if part1 is None else len(set(ids) - set(part1)))


def fmt_d(d, qualified):
    """D to 2 decimals, more when 2 would show a number on the other side of the -5 bar."""
    for places in range(2, DECIMALS + 1):
        s = f"{d:.{places}f}"
        if (float(s) < OODH_MARGIN) == qualified:
            return s
    return f"{d:.{DECIMALS}f}"


def decide(ci, part2_complete=False, fill=None):
    if ci.get("diagnostic"):
        raise ValueError(f"diagnostic CI ({'; '.join(ci['diagnostic'])}): no claim is worded from it")
    if part2_complete and not ci.get("part2_threads"):
        raise ValueError("Part 2 complete, but no Part 2 thread is in the scored set (part2_threads "
                         f"{ci.get('part2_threads')})")
    d = round(ci["D"], DECIMALS)
    qualified = d < OODH_MARGIN
    sentence = OODH_SENTENCE.format(h=ci["h"], d=fmt_d(d, qualified), lo=f"{ci['lo']:.2f}", hi=f"{ci['hi']:.2f}")
    text = LEVEL_R.format_map(_Keep(fill or {}, oodh=sentence))
    if qualified:
        text = text.replace(UNQUALIFIED, QUALIFIED)
    provisional = not part2_complete
    if provisional:
        text += " " + PROVISIONAL
    return dict(D=ci["D"], D_compared=d, lo=ci["lo"], hi=ci["hi"], h=ci["h"], margin=OODH_MARGIN,
                decision="qualified" if qualified else "unqualified", provisional=provisional,
                allowed=[QUALIFIED] if qualified else [UNQUALIFIED, QUALIFIED],
                refused=[UNQUALIFIED] if qualified else [], oodh_sentence=sentence, text=text,
                note=PROVISIONAL if provisional else None)


def refusals(text, dec):
    out = []
    bare = [m.start() for m in re.finditer(NON_INFERIOR, text, re.I)
            if text[m.start():m.start() + len(QUALIFIED)] != QUALIFIED]
    if dec["decision"] == "qualified" and bare:
        out.append(f"unqualified 'non-inferior' at {bare}: OOD-H's D is worse than -5 points (s1)")
    if dec["oodh_sentence"] not in text:
        out.append("the OOD-H result is not beside the claim")
    if dec["provisional"] and PROVISIONAL not in text:
        out.append("the provisional note is missing while OOD-H Part 2 is not scored")
    return out


def _load(paths):
    return [json.loads(line) for p in paths for line in open(p) if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description="OOD-H paired difference and the s1 Level R wording rule (no model)")
    ap.add_argument("--planck", nargs="+", required=True, help="the Planck model's OOD-H transcripts.jsonl files")
    ap.add_argument("--comparator", nargs="+", required=True, help="Qwen2.5-0.5B-Instruct's OOD-H transcripts")
    ap.add_argument("--data", default=None, help="the OOD-H item file(s) as one JSONL: its ids are the thread set")
    ap.add_argument("--part2-complete", action="store_true", help="OOD-H Part 2 is scored (ends 'provisional')")
    ap.add_argument("--part1", default=PART1, help="the OOD-H Part 1 item file (its ids are Part 1)")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fill", default="", help="the other s1 numbers, e.g. N=30,B=18,k=3,c=900,n=5400")
    a = ap.parse_args(argv)
    fill = dict(kv.split("=", 1) for kv in a.fill.split(",")) if a.fill else {}
    try:
        if a.part2_complete and not a.data:
            raise ValueError("--part2-complete needs --data (the scored thread set, Part 2 included)")
        expected = [r["id"] for r in _load([a.data])] if a.data else None
        part1 = [r["id"] for r in _load([a.part1])] if os.path.exists(a.part1) else None   # None: decide refuses
        ci = paired_diff(_load(a.planck), _load(a.comparator), a.n, a.seed, expected, part1)
        dec = None if ci["diagnostic"] else decide(ci, a.part2_complete, fill)
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(dict(ci=ci, decision=dec), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
