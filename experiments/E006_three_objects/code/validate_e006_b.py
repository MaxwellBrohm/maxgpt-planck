"""E006 stream acceptance, helpers for validate_e006.py (notes.txt STREAM ACCEPTANCE AND PURITY). No model.
  e004_kept          E004's first 6,404 kept examples per seed (E004's plain encoder): the reference position rates
  gates_e005         E005's gates on one seed's kept examples: every O and X rule <= .70, O1/O2 <= .50, IDEAL 1.00,
                     drawn-vs-kept share shift <= 2 points, purity_e005.violations empty on every kept example
  fivegram_overlap   word 5-grams shared between the kept examples' texts and every eval text (draws 4004/4104/4204,
                     BIG, H5L, AL): must be 0 over EVERY kept example (the E005 audit's coverage fix)
  replay_checks      the replay pool's build counts, the per-seed usable/distinct threads, the label checks"""
import json
import os
from collections import Counter

import e004_sets as S4
import e005_sets as S5
import e006_sets as S6
import oracles_e005 as OR
import purity_e005 as PU
import ngram_overlap as N4
import ngram_overlap_e005 as NO5
import train_e004 as T4
from text_e004 import words

MAX_SHIFT = 0.02


def e004_kept(tok, seed, n, cand_ids, max_len=768):
    out = []
    for ex in T4.stream(seed):
        ids, labels, how = S4.encode(tok, ex, cand_ids)
        if len(ids) <= max_len:
            out.append(ex)
            if len(out) == n:
                return out
    return out


def gates_e005(kept_exs, stats, eg):
    """-> (failures, info line)."""
    fails = []
    overall, per, nb = OR.score(kept_exs)
    fails += [f"oracle: {b}" for b in OR.gate(overall)]
    shift = S5.stream_summary(stats)["max_share_shift"]
    if shift is None or shift > MAX_SHIFT:
        fails.append(f"drawn-vs-kept shift {shift} > {MAX_SHIFT}")
    pv = Counter(axis for ex in kept_exs for axis in PU.violations(ex, eg))
    if pv:
        fails.append(f"purity {dict(pv)}")
    mx = max(overall[n] for n, _ in OR.FAKE)
    info = (f"max rule {mx:.3f} O1 {overall['O1 first mention']:.3f} O2 {overall['O2 last mention']:.3f} "
            f"IDEAL {overall['IDEAL']:.2f}; shift {shift}; purity violations {sum(pv.values())}")
    return fails, info


_EVAL5 = []


def eval_fivegrams():
    if not _EVAL5:
        import items_e004 as I
        import items_al as AL
        import items_big_e006 as BG
        draws = {name: I.draw(name) for name in I.DRAWS}
        src, _ = N4.eval_texts(draws)
        texts = {t for ts in src.values() for t in ts}
        for its in list(BG.load().values()) + [AL.load()]:
            for it in its:
                texts |= {x for u, a in it["turns"] for x in (u, a)} | {it["question"], it.get("prefix") or ""}
        _EVAL5.append(set().union(*(N4.grams(words(t), 5) for t in texts if t)))
    return _EVAL5[0]


def fivegram_overlap(kept_exs):
    texts, _ = NO5.train_texts(kept_exs)
    tg = set().union(*(N4.grams(words(t), 5) for t in texts))
    return sorted(tg & eval_fivegrams())


def replay_checks(tok, pool, build_json, seeds, n_need, max_len=768):
    """-> (failures, info lines)."""
    fails, L = [], []
    c = build_json["counts"]
    for k in ("reserve", "heldout", "aiism", "eval_8gram", "probe_turn", "eval_surname", "reserve_leak"):
        L.append(f"pool drop {k}: {c.get(k)}")
    if any(c.get(k) for k in ("reserve", "heldout", "aiism")):
        fails.append(f"pool: reserve/heldout/aiism threads present {c}")
    ids = {r["id"] for r in pool}
    if len(ids) != len(pool) or len(pool) != c.get("keep"):
        fails.append(f"pool: {len(pool)} rows, {len(ids)} distinct, build says {c.get('keep')}")
    byid = {r["id"]: r for r in pool}
    for s in seeds:
        st, used, bad = S6.new_replay_stats(), [], 0
        for x, y, tid in S6.replay_stream(tok, pool, s, max_len, st):
            used.append(tid)
            if S6.check_replay_labels(tok, x, y, byid[tid]["turns"]):
                bad += 1
            if len(used) == n_need:
                break
        if len(used) < n_need or len(set(used)) != len(used) or bad:
            fails.append(f"replay seed {s}: {len(used)} usable (need {n_need}), {len(set(used))} distinct, "
                         f"{bad} with label problems")
        L.append(f"replay seed {s}: first {len(used)} usable threads, drops before them {st['dropped']}, cut {st['cut']}, "
                 f"exchanges {st['exchanges']}, tokens {st['tokens']}, labelled {st['labelled']}, label problems {bad}")
    return fails, L


def load_pool(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    bj = json.load(open(os.path.join(os.path.dirname(path), "replay_build.json")))
    return rows, bj
