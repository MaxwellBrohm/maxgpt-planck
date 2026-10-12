"""RC-12 STEP 10c: the re-anchor rule's audit (prereg draft s10) and the strict-case rates (s6, s19 item 4). No model.
  python -B reanchor_audit.py --root <copy of ~/planck/runs/rc12_dev> --out runs/dev_panel/reanchor
1. LFM2-2.6B, template, sampling seeds 1-3: each Level A key's rate by score.summarize (must equal dev_panel's
   reanchor_input), its failing units (CORR: a conversation, unit < 1; BIND: a pair whose twins are not both right),
   the failing clauses, and a sample of 30 failing units per key, drawn by Random("RC12:s10c:<key>"): shuffled, one
   unit per distinct item first, then the rest. Each sampled unit is dumped (every user turn, the probe reply in
   full, earlier replies cut to 200 characters, every value mention in the probe reply with the grader's reason
   when it is not asserted) to <out>/<key>.txt for reading. Labels written by the reader go in <out>/labels.json
   ({key: {unit: class}}); this script only tallies them.
2. The rule (s10), mechanical: rate >= 0.80 keeps 0.80; else floor(rate / 0.05) x 0.05, never below 0.60, and
   below 0.60 the threshold is 0.60 with "no tested model reached it" checked over the template panel.
3. Strict cases, every panel model, template, seeds 1-3: (a) / (b) candidates = VAL probes whose gold is asserted
   and whose only failing clauses are v3_other / v3_shotgun (an upper bound: a reader decides whether the other
   value carried its right owner), per Level A key with the key rate those would give if they passed; the lenient
   twin's rate on each key; (c) ack-repeat and (d) equality_only are dev_panel's and are copied from it."""
import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict

import grade_text as T
import graders as G
import score as S
from dev_panel import CORE, EXTRAS, load

REF = "LFM2-2.6B"
SEEDS = ["1", "2", "3"]
LA = list(S.LEVEL_A)
AB = {"v3_other", "v3_shotgun"}


def rows_of(root, model, render="template"):
    out = []
    for s in SEEDS:
        for d in (s, s + "_owncf"):
            out += load(os.path.join(root, model, render, d, "transcripts.jsonl"))
    return out


def in_key(r, key):
    fam, cells = S.LEVEL_A[key]
    return r["family"] == fam and not r.get("own_cf") and (cells is None or r["cell"] in cells)


def key_units(rows, key):
    """[(unit name, seed, [rows])] of one key; BIND units are pairs joined by pair_id within a seed."""
    rr = [r for r in rows if in_key(r, key)]
    if key != "BIND":
        return [(f"{r['id']}@s{r['seed']}", r["seed"], [r]) for r in rr]
    pairs = defaultdict(list)
    for r in rr:
        pairs[(r["seed"], r["pair_id"])].append(r)
    return [(f"{pid}@s{sd}", sd, sorted(tw, key=lambda r: r["id"])) for (sd, pid), tw in sorted(pairs.items())]


def unit_ok(rows):
    return all(r["unit"] == 1.0 for r in rows)


def sample(fails, key, n=30):
    rng = random.Random(f"RC12:s10c:{key}")
    pool = sorted(fails, key=lambda u: u[0])
    rng.shuffle(pool)
    seen, first, rest = set(), [], []
    for u in pool:
        item = u[0].split("@")[0]
        (rest if item in seen else first).append(u)
        seen.add(item)
    return (first + rest)[:n]


def mentions(reply, probe, rec):
    text, pool = T.norm(reply), G.pool_values(probe)
    out = []
    for v in pool:
        for m in T.value_hits(text, v):
            why = T.why_not_asserted(text, m, pool)
            tag = "GOLD" if v == probe["gold"] else "stale" if v in (probe.get("stale") or []) else "cand" \
                if v in (probe.get("candidates") or []) else "ctx" if v in G.in_context(rec, probe, pool) else "OUT"
            out.append(f"{v}[{tag}]:{why or 'ASSERTED'}")
    return out


def dump(unit, dev):
    name, _, rows = unit
    lines = [f"===== {name}"]
    for r in rows:
        rec = dev[r["id"]]
        p, res = rec["probes"][0], r["probes"][0]
        meta = json.dumps({k: v for k, v in rec["meta"].items() if v is not None})
        lines.append(f"--- {r['id']} cell={r['cell']} meta={meta}")
        lines.append(f"Q t{p['turn']}: {p['question']} | gold={p['gold']} cands={p.get('candidates')} "
                     f"stale={p.get('stale')} b={p.get('b_values')} lures={p.get('lure_values')} src={p.get('src')}")
        lines.append(f"GRADER ok={res['ok']} fails={res['fails']} lenient={res['lenient']}")
        for t in r["turns"]:
            full = t["i"] == p["turn"]
            more = f" [+{len(t['reply']) - 200}]" if len(t["reply"]) > 200 else ""
            rep = t["reply"] if full else t["reply"][:200] + more
            lines.append(f"  u{t['i']} {t['kind']}: {t['user']}")
            lines.append(f"  a{t['i']} ({t['stop']}){' PROBE' if full else ''}: {rep}")
        tp = r["turns"][p["turn"] - 1]
        lines.append(f"MENTIONS: {mentions(tp['reply'], p, rec)}  stop={tp['stop']} flags={r['flags'][p['turn'] - 1]}")
    return "\n".join(lines) + "\n"


def reanchor(rate, bar=0.80, floor=0.60, step=0.05):
    if rate >= bar:
        return bar, "kept"
    if rate < floor:
        return floor, "below 0.60: threshold 0.60, no tested model reached it"
    return round(max(floor, math.floor(round(rate / step, 9)) * step), 2), "re-anchored"


def seed_rate(us, credited=()):
    """the key rate (mean over seeds of the unit mean) with the units named in credited counted right."""
    by_seed = defaultdict(list)
    for u in us:
        by_seed[u[1]].append(unit_ok(u[2]) or u[0] in credited)
    return S.mean(S.mean(v) for v in by_seed.values())


def cp_upper(k, n, alpha=0.05):
    """exact (Clopper-Pearson) two-sided upper limit for k successes in n."""
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        cdf = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        lo, hi = (mid, hi) if cdf > alpha / 2 else (lo, mid)
    return hi


def gold_named(u, dev):
    """every failing twin's probe reply names its gold somewhere (any form): the loosest bound on what a grader or
    rule change could credit."""
    def named(r):
        p = dev[r["id"]]["probes"][0]
        text = T.norm(r["turns"][p["turn"] - 1]["reply"])
        return any(T.mentioned(text, g) for g in G.golds(p))
    return all(r["unit"] == 1.0 or named(r) for r in u[2])


def strict_cases(rows, key):
    """(a)/(b) candidates on one key: units, failing units, failing units whose every failing probe has its gold
    asserted and only v3_other / v3_shotgun, the key rate with those passed, and the lenient rate."""
    us = key_units(rows, key)
    bad = [u for u in us if not unit_ok(u[2])]
    ab = [u for u in bad if all(not r["probes"][0]["fails"] or set(r["probes"][0]["fails"]) <= AB for r in u[2])]
    len_ok = {u[0] for u in us if all(r["probes"][0]["lenient"] for r in u[2])}
    return dict(units=len(us), failing=len(bad), ab_only=len(ab), rate=seed_rate(us),
                rate_if_ab_passed=seed_rate(us, {u[0] for u in ab}),
                lenient=seed_rate([(n, sd, [dict(unit=0.0)]) for n, sd, _ in us], len_ok))


def family_ab(rows):
    """per family, VAL probes failing ONLY by v3_other / v3_shotgun with the gold asserted, over VAL probes."""
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("own_cf"):
            continue
        for p in r["probes"]:
            if p["grader"] == "VAL":
                out[r["family"]][1] += 1
                out[r["family"]][0] += bool(p["fails"]) and set(p["fails"]) <= AB
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel", default="runs/dev_panel")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = {r["id"]: r for r in load("dev/rc12_dev.jsonl")}
    rules = json.load(open(os.path.join(args.panel, "rules.json")))
    pj = json.load(open(os.path.join(args.panel, "panel.json")))
    panel = {f"{x['model']}|{x['render']}|{x['seed']}": x for x in pj["per_seed"] + pj["mean3"]}
    rows = rows_of(args.root, REF)
    summ = S.summarize(rows)
    labels_p = os.path.join(args.out, "labels.json")
    labels = json.load(open(labels_p)) if os.path.exists(labels_p) else {}
    res = dict(reference=REF, keys={})
    for key in LA:
        rate = summ["keys"][key]
        assert abs(rate - rules["reanchor_input"][key]) < 1e-12, (key, rate)
        us = key_units(rows, key)
        bad = [u for u in us if not unit_ok(u[2])]
        clauses = Counter(f for u in bad for r in u[2] for f in r["probes"][0]["fails"])
        combos = Counter(" ".join(sorted(r["probes"][0]["fails"])) for u in bad for r in u[2] if r["unit"] < 1)
        pick = sample(bad, key)
        with open(os.path.join(args.out, key.replace(":", "_") + ".txt"), "w") as f:
            for u in pick:
                f.write(dump(u, dev))
        lab = labels.get(key, {})
        missing = [u[0] for u in pick if u[0] not in lab]
        tally = Counter(lab[u[0]].split(":")[0] for u in pick if u[0] in lab)
        ge = sorted(u[0] for u in pick if lab.get(u[0], "").startswith("GRADER_ERROR"))
        k, n, nb = len(ge), len(pick), len(bad)
        audit = dict(grader_errors=ge, rate_after_audit=seed_rate(us, ge),
                     rate_if_share_extrapolated=(len(us) - nb + nb * k / n) / len(us),
                     rate_at_cp_upper=(len(us) - nb + nb * cp_upper(k, n)) / len(us),
                     rate_if_every_gold_named_unit_passed=seed_rate(us, {u[0] for u in bad if gold_named(u, dev)}))
        thr, how = reanchor(max(rate, audit["rate_after_audit"]))
        best = max(((m, panel[f"{m}|template|mean3"][key]) for m in CORE + EXTRAS), key=lambda x: x[1])
        res["keys"][key] = dict(rate=rate, units=len(us), failing=len(bad), clauses=dict(clauses),
                                fail_combos=dict(combos.most_common()), sample=[u[0] for u in pick],
                                labels=dict(tally), unlabeled=len(missing), audit=audit, threshold=thr, how=how,
                                best_panel_template=best)
    strict = {}
    for m in CORE + EXTRAS:
        mr = rows_of(args.root, m)
        strict[m] = dict(keys={k: strict_cases(mr, k) for k in LA}, families_ab=family_ab(mr),
                         equality_only=rules["equality_only"][f"{m}|template"],
                         ack_repeat=panel[f"{m}|template|mean3"]["ack_repeat"],
                         ack_repeat_of_statements=panel[f"{m}|template|mean3"]["ack_repeat_of_statements"])
    res["strict_cases"] = strict
    c = "Qwen2.5-0.5B-Instruct|"
    res["comparator_loop"] = {f"{r}|{s}": panel[f"{c}{r}|{s}"]["loop"] for r in ("template", "plain")
                              for s in ["mean3", "greedy"] + SEEDS}
    with open(os.path.join(args.out, "reanchor.json"), "w") as f:
        json.dump(res, f, indent=1)
    for key, v in res["keys"].items():
        print(f"{key:15s} rate {v['rate']:.3f} failing {v['failing']}/{v['units']} labels {v['labels']} "
              f"unlabeled {v['unlabeled']} -> {v['threshold']:.2f} ({v['how']}); best {v['best_panel_template']}")
        print("   ", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v["audit"].items()})


if __name__ == "__main__":
    main()
