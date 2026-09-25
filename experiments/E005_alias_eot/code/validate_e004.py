"""E004 stream-level acceptance (notes.txt (a)), checked on each seed's ACTUAL training examples, per tokenizer,
BEFORE any training run. Tokenizers only (no model). Run with the venv python (transformers), HF_HUB_OFFLINE=1.

For every model's tokenizer and every seed it trains (SmolLM2-135M: 0-5; tiny models: 0-3), the trainer's own path
is replayed: e004_sets.encode + the --max-len 768 rejection (never truncation), the first 4 kept examples go to the
loss self-check batch, the next 400 x 16 = 6,400 kept examples are the ones trained on. Acceptance, all must hold:
  every cheap oracle O1-O8 (all variants in oracles_e004.ORACLES) <= 0.70 on the 6,400; O1 and O2 <= 0.50;
  IDEAL = 1.00; no kind's kept share differs from its drawn share by more than 0.02; the first-mention-gold share
  at d >= 8 moves by <= 0.02 (drawn vs kept).
Tokenizers with an identical vocabulary (TinyStories-1M/3M/8M; pythia-14m/31m) share one replay (vocab sha256).
Writes ../logs/validate_e004.txt and ../logs/validate_e004.json; exit 0 iff every check holds.
usage: validate_e004.py [--models M ...] [--max-len 768] [--n 6400]"""
import argparse, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import lik
import e004_sets as S
import oracles_e004 as O
import train_e004 as T

SEEDS = {"HuggingFaceTB/SmolLM2-135M-Instruct": [0, 1, 2, 3, 4, 5]}
TINY = ["roneneldan/TinyStories-8M", "EleutherAI/pythia-31m", "roneneldan/TinyStories-3M", "EleutherAI/pythia-14m",
        "roneneldan/TinyStories-1M"]
MAX_ORACLE, MAX_O12, MAX_SHIFT = 0.70, 0.50, 0.02
SELFCHECK_BATCH = 4


def vocab_hash(tok):
    v = sorted(tok.get_vocab().items())
    return hashlib.sha256((type(tok).__name__ + json.dumps(v)).encode()).hexdigest()[:16]


def first_mention_gold(ex):
    V = O.View(O.from_train(ex))
    return O.o1_first(V) == ex["gold"]


def replay(tok, seed, max_len, n):
    """-> (trained examples, drawn examples up to the last one kept)."""
    drawn, kept = [], []
    for ex in T.stream(seed):
        drawn.append(ex)
        ids, _, _ = S.encode(tok, ex, lik.cand_ids)
        if len(ids) > max_len:
            continue
        kept.append(ex)
        if len(kept) == SELFCHECK_BATCH + n:
            break
    return kept[SELFCHECK_BATCH:], drawn, kept


def shares(exs):
    c = {}
    for ex in exs:
        c[ex["kind"]] = c.get(ex["kind"], 0) + 1
    return {k: v / len(exs) for k, v in c.items()}


def check(tok, seed, max_len, n):
    trained, drawn, kept = replay(tok, seed, max_len, n)
    res = {"seed": seed, "n_trained": len(trained), "n_drawn": len(drawn), "rejected": len(drawn) - len(kept)}
    acc = {name: 0 for name, _ in O.ORACLES}
    acc["IDEAL"] = 0
    for ex in trained:
        V = O.View(O.from_train(ex))
        for name, fn in O.ORACLES:
            acc[name] += fn(V) == ex["gold"]
        acc["IDEAL"] += O.ideal(V) == ex["gold"]
    res["oracles"] = {k: round(v / len(trained), 4) for k, v in acc.items()}
    sd, sk = shares(drawn), shares(kept)
    res["max_share_shift"] = round(max(abs(sd[k] - sk.get(k, 0.0)) for k in sd), 4)
    fd = [first_mention_gold(e) for e in drawn if e["d"] >= 8]
    fk = [first_mention_gold(e) for e in kept if e["d"] >= 8]
    share = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
    res["fm_gold_d8"] = {"drawn": share(fd), "kept": share(fk), "n_kept_d8": len(fk)}
    both = None not in (res["fm_gold_d8"]["drawn"], res["fm_gold_d8"]["kept"])
    res["fm_gold_shift"] = round(abs(res["fm_gold_d8"]["drawn"] - res["fm_gold_d8"]["kept"]), 4) if both else 1.0
    bad = []
    if len(trained) != n:
        bad.append(f"only {len(trained)} trained examples")
    for k, v in res["oracles"].items():
        if k == "IDEAL":
            if v != 1.0:
                bad.append(f"IDEAL {v} != 1.00")
        elif v > MAX_ORACLE or (k.startswith(("O1", "O2")) and v > MAX_O12):
            bad.append(f"{k} {v}")
    if res["max_share_shift"] > MAX_SHIFT:
        bad.append(f"kind share shift {res['max_share_shift']}")
    if res["fm_gold_shift"] > MAX_SHIFT:
        bad.append(f"first-mention-gold shift {res['fm_gold_shift']}")
    res["fails"] = bad
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(SEEDS) + TINY)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--n", type=int, default=6400)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    out, lines, memo, ok = {}, [], {}, True
    for m in a.models:
        tok = AutoTokenizer.from_pretrained(m)
        vh = vocab_hash(tok)
        for seed in SEEDS.get(m, [0, 1, 2, 3]):
            t0 = time.time()
            if (vh, seed) in memo:
                r = dict(memo[(vh, seed)], shared_with=memo[(vh, seed)]["model"])
            else:
                r = dict(check(tok, seed, a.max_len, a.n), model=m, vocab=vh)
                memo[(vh, seed)] = r
            out[f"{m}|{seed}"] = r
            ok &= not r["fails"]
            o = r["oracles"]
            lines.append(f"{m:<38} seed {seed} n={r['n_trained']} rejected={r['rejected']} "
                         f"max(O)={max(v for k, v in o.items() if k != 'IDEAL'):.3f} O1={o['O1 first mention']:.3f} "
                         f"O2={o['O2 last mention']:.3f} IDEAL={o['IDEAL']:.2f} shift={r['max_share_shift']:.4f} "
                         f"fm_d8={r['fm_gold_shift']:.4f} {'OK' if not r['fails'] else 'FAIL ' + '; '.join(r['fails'])}"
                         + (f" (replay shared with {r['shared_with']})" if r.get("shared_with") else "")
                         + f" {time.time() - t0:.1f}s")
            print(lines[-1], flush=True)
    worst = {}
    for r in out.values():
        for k, v in r["oracles"].items():
            worst[k] = max(worst.get(k, 0.0), v)
    lines.append("worst per oracle over all (model, seed): " + ", ".join(f"{k}={v:.3f}" for k, v in worst.items()))
    lines.append("ALL PASS" if ok else "FAILED")
    print(lines[-2] + "\n" + lines[-1])
    sfx = f"_{a.tag}" if a.tag else ""
    open(os.path.join(EXP, "logs", f"validate_e004{sfx}.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(EXP, "logs", f"validate_e004{sfx}.json"), "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
