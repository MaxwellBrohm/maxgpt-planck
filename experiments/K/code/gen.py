"""Track K shard and item-file writer (SPEC 1-8, 14). Output formats are data_prep/pretokenize.py's: headerless
uint16 .bin token shards (TokenShardSource, eot 1) and id-carrying chat .jsonl (ChatJsonlSource). Deterministic by
the world seed and the tags in kcommon; the same command writes the same bytes (tests/test_gen.py).

  python gen.py skill  OUT [--workers 8]            SKILL stream (S names), 1.03 x its expected draw
  python gen.py fact   OUT --arm F0|FL|FH|CAP       FACT streams: fact_bio (tokens) and fact_qa (chat)
  python gen.py k5     OUT --arm L0|L50|L100        K5 FACT stream: fact_k5 (chat)
  python gen.py evals  OUT                          every SPEC 8 file, with sha256s in evals/manifest.json
  python gen.py check  OUT                          SPEC 5 acceptance on OUT/skill and OUT/evals -> check.json
  python gen.py data   OUT --arm ARM                the run config's data: block (yaml) for an arm
Scale keys for tests only: --slots, --n-high, --n-low, --e-bio, --e-qa, --n5, --e5, --block.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from multiprocessing import get_context

import numpy as np

import items as I
import kcommon as K
import lookup as L
import pools as P
import world as Wd

CODE = ["kcommon.py", "pools.py", "world.py", "lookup.py", "items.py", "fams.py", "skill.py", "oracles.py",
        "purity.py", "evalsets.py", "gen.py"]
LANG = [("cccc", "tokens", 0.4142), ("dolly", "chat", 0.0160), ("gutenberg", "tokens", 0.1479),
        ("irc", "tokens", 0.0400), ("oasst2", "chat", 0.0120), ("stackexchange", "tokens", 0.2219),
        ("wikimedia", "tokens", 0.1479)]                      # E2 base5m.yaml pinned shares (sum 0.9999)


def code_sha() -> dict:
    return {c: K.sha256_file(os.path.join(K.HERE, c)) for c in CODE}


def _manifest(d: str, extra: dict) -> dict:
    files = sorted(glob.glob(os.path.join(d, "*.bin")) + glob.glob(os.path.join(d, "*.jsonl")))
    man = {"files": {os.path.basename(f): K.sha256_file(f) for f in files}, "code_sha256": code_sha(),
           "tokenizer_sha256": K.TOK_SHA256, "world_seed": K.W, "tags": K.TAGS, **extra}
    K.write_json(os.path.join(d, "manifest.json"), man)
    return man


def _skill_round(args):
    ri, per_block = args
    import skill as S
    g = S.Gen(P.get(), "train")
    out = []
    for fam, nb in S.ROUND.items():
        for b in range(nb):
            bi = ri * nb + b
            for i, it in enumerate(S.block(g, fam, per_block, (K.TAGS["skill_train"], ord(fam), bi))):
                out.append(I.render(it, f"{fam}.b{bi}.i{i}", full=False))
    return out


def write_skill(out: str, slots: int = K.B_SLOTS, workers: int = 8, per_block: int = 512,
                shard_items: int = 1 << 16) -> dict:
    target = K.WRITE_MARGIN * K.expected_draw(K.SHARES["skill"], slots)
    d = os.path.join(out, "skill")
    os.makedirs(d, exist_ok=True)
    recs, tokens, ri = [], 0, 0
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        while tokens < target:
            for got in pool.map(_skill_round, [(ri + j, per_block) for j in range(workers)]):
                recs += got
                tokens += sum(I.n_tokens(x) for x in got)
            ri += workers
    order = K.rng("skill_train", 1).permutation(len(recs))
    n_sh = 0
    for s, lo in enumerate(range(0, len(recs), shard_items)):
        K.write_jsonl(os.path.join(d, f"skill-{s:05d}.jsonl"), (recs[i] for i in order[lo:lo + shard_items]))
        n_sh += 1
    return _manifest(d, {"kind": "chat", "items": len(recs), "tokens": tokens, "target_tokens": target,
                         "rounds": ri, "per_block": per_block, "shards": n_sh})


def write_fact(out: str, c: Wd.FactCfg) -> dict:
    pl = P.get()
    w = Wd.make_world(pl)
    lay = Wd.layout(c)
    pln = Wd.plan(w, c)
    db, dq = os.path.join(out, "fact_bio"), os.path.join(out, "fact_qa")
    for x in (db, dq):
        os.makedirs(x, exist_ok=True)
    for s, lo, hi in Wd.shards(lay["bio_slots"], c.shard_docs):
        Wd.bio_block(w, pl, pln["bio"][lo:hi], s).tofile(os.path.join(db, f"fact_bio-{s:05d}.bin"))
    for s, lo, hi in Wd.shards(lay["qa_slots"], c.shard_docs):
        K.write_jsonl(os.path.join(dq, f"fact_qa-{s:05d}.jsonl"),
                      Wd.qa_block(w, pl, pln["qa_ent"][lo:hi], pln["qa_attr"][lo:hi], s, lo))
    exp = np.bincount(pln["bio"][pln["bio"] >= 0], minlength=c.n_plan)
    info = {"arm": c.arm, "cfg": vars(c), "layout": {k: v for k, v in lay.items()},
            "fb_bio_exposures_min_max": [int(exp.min()), int(exp.max())] if c.n_fb else [0, 0],
            "plan_sha256": K.sha256_bytes(pln["bio"].tobytes() + pln["qa_ent"].tobytes() + pln["qa_attr"].tobytes())}
    return {"bio": _manifest(db, {"kind": "tokens", **info}), "qa": _manifest(dq, {"kind": "chat", **info})}


def write_k5(out: str, c: L.K5Cfg) -> dict:
    pl = P.get()
    w = Wd.make_world(pl)
    ch = L.chats(c)
    lay = L.layout5(c, ch)
    slots = L.plan5(c, ch, lay)
    d = os.path.join(out, "fact_k5")
    os.makedirs(d, exist_ok=True)
    for s, lo, hi in Wd.shards(len(slots), c.shard_docs):
        K.write_jsonl(os.path.join(d, f"fact_k5-{s:05d}.jsonl"),
                      L.k5_block(w, pl, ch, lay["hit"], slots[lo:hi], s, lo))
    info = {k: v for k, v in lay.items() if k != "hit"}
    return _manifest(d, {"kind": "chat", "arm": c.arm, "cfg": vars(c), "layout": info,
                         "hits": int(lay["hit"].sum()), "chats": int(len(lay["hit"]))})


def write_evals(out: str, small: bool = False) -> dict:
    import evalsets as E
    pl = P.get()
    lex = P.Lex(pl)
    w = Wd.make_world(pl)
    d = os.path.join(out, "evals")
    os.makedirs(d, exist_ok=True)
    al = E.allocate(w)
    n = (lambda x: max(8, x // 64)) if small else (lambda x: x)
    sets = {"k_eval_id": E.skill_set(pl, lex, "eval", K.SEEDS["eval"], n(512), tag="K-EVAL-ID"),
            "k_eval_s": E.skill_set(pl, lex, "evals", K.SEEDS["evals"], n(256), tag="K-EVAL-S")}
    import skill as S
    sets["k_eval_ood"] = [r for o in S.OOD for fam in S.OOD_FAMS[o] for r in
                          E.skill_set(pl, lex, "eval", K.SEEDS["ood"], n(256) // len(S.OOD_FAMS[o]), fams=fam,
                                      ood=o, groups=False, tag=f"OOD-{o}")]
    cs = E.mean_cands(sets["k_eval_id"])
    fb = al["fbprobe"]
    low = fb["lt"][:n(2048)] + fb["lh"][:n(2048)]
    sets["fb_probe"] = (E.bio_prompts(w, pl, fb["lt"][:n(2048)], K.SEEDS["eval"], "FB-PROBE", csize=cs,
                                      block="low_qa")
                        + E.bio_prompts(w, pl, fb["lh"][:n(2048)], K.SEEDS["eval"], "FB-PROBE", csize=cs,
                                        block="low_held")
                        + E.bio_prompts(w, pl, fb["ho"][:n(4096)], K.SEEDS["eval"], "FB-PROBE", csize=cs,
                                        block="high"))
    sets["conflict"] = E.conflict(w, pl, low, K.SEEDS["conflict"], n(512))
    sets["k5"] = E.k5_sets(w, pl, low)
    for tag, key, seed in (("K-DEV", "dev", K.SEEDS["dev"]), ("K-PROBE", "kprobe", K.SEEDS["probe"])):
        ents = al[key]["lt"] + al[key]["lh"] + al[key]["ho"]
        sets[tag.lower().replace("-", "_")] = (E.mix_set(pl, lex, seed, n(1024), tag)
                                               + E.one_per_entity(E.bio_prompts(w, pl, ents[:n(1024)], seed, tag,
                                                                                orders=1, csize=cs, qa=False), seed))
    for split, key in (("K2-A", "k2a"), ("K2-B", "k2b")):
        ents = al[key]["lt"] + al[key]["lh"] + al[key]["ho"]
        seed = K.SEEDS["k2"] + (split == "K2-B")
        sets[split.lower().replace("-", "_")] = (E.mix_set(pl, lex, seed, n(512), split)
                                                 + E.bio_prompts(w, pl, ents, seed, split, csize=cs, qa=False)
                                                 + E.conflict(w, pl, al[key]["lt"] + al[key]["lh"], seed, n(128),
                                                              tag=split + "-CONFLICT"))
    man = {"sets": {}, "fact_cand_size": cs}
    for name, recs in sets.items():
        p = os.path.join(d, f"{name}.jsonl")
        K.write_jsonl(p, recs)
        man["sets"][name] = {"n": len(recs), "sha256": K.sha256_file(p)}
    return _manifest(d, man)


def data_block(arm: str, root: str = "../../../planck_root/data/k") -> dict:
    """The run config's data: block (paths relative to a run config three levels below the code root)."""
    src = [{"name": "skill", "kind": "chat", "share": K.SHARES["skill"], "shuffle": False,
            "paths": [f"{root}/skill/skill-*.jsonl"]}]
    if arm == "CONT":           # K3 continuation: SKILL and LANG only (the loader renormalizes to 0.5 / 0.5)
        pass
    elif arm in L.K5_ARMS:
        src.append({"name": "fact_k5", "kind": "chat", "share": K.SHARES["fact"], "shuffle": False,
                    "paths": [f"{root}/{arm}/fact_k5/fact_k5-*.jsonl"]})
    else:
        lay = Wd.layout(Wd.FactCfg(arm=arm))
        src += [{"name": "fact_bio", "kind": "tokens", "eot_id": K.EOT, "share": round(lay["share_bio"], 6),
                 "shuffle": False, "paths": [f"{root}/{arm}/fact_bio/fact_bio-*.bin"]},
                {"name": "fact_qa", "kind": "chat", "share": round(lay["share_qa"], 6), "shuffle": False,
                 "paths": [f"{root}/{arm}/fact_qa/fact_qa-*.jsonl"]}]
    lang = 0.0 if arm == "CAP" else K.SHARES["lang"]
    if arm == "CAP":
        src = src[1:]
    tot = sum(s for _, _, s in LANG)
    for name, kind, s in LANG if lang else []:
        ext = "jsonl" if kind == "chat" else "bin"
        e = {"name": name, "kind": kind, "share": round(lang * s / tot, 6),
             "paths": [f"../../../planck_root/data/shards/starter_v0_8k/{name}/{name}-*.{ext}"]}
        src.append({**e, "eot_id": K.EOT} if kind == "tokens" else e)
    return {"pad_id": K.PAD, "chat": {"loss": "assistant", "role_ids": K.ROLE_IDS, "end_id": K.END}, "sources": src}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["skill", "fact", "k5", "evals", "check", "data"])
    ap.add_argument("out")
    ap.add_argument("--arm", default="FH")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--slots", type=int, default=K.B_SLOTS)
    ap.add_argument("--n-high", type=int, default=K.N_HIGH)
    ap.add_argument("--n-low", type=int, default=K.N_LOW)
    ap.add_argument("--e-bio", type=int, default=K.E_BIO)
    ap.add_argument("--e-qa", type=int, default=K.E_QA)
    ap.add_argument("--n5", type=int, default=K.N_LOW)
    ap.add_argument("--e5", type=int, default=K.E5)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--small", action="store_true", help="tests: 1/64-size eval sets")
    a = ap.parse_args(argv)
    if a.cmd == "skill":
        man = write_skill(a.out, a.slots, a.workers, a.block)
    elif a.cmd == "fact":
        man = write_fact(os.path.join(a.out, a.arm), Wd.FactCfg(a.arm, a.slots, a.e_bio, a.e_qa, a.n_high, a.n_low,
                                                                max(a.n_high, min(K.N_CAP, 2 * a.n_high))))
    elif a.cmd == "k5":
        man = write_k5(os.path.join(a.out, a.arm), L.K5Cfg(a.arm, a.slots, a.n5, a.e5))
    elif a.cmd == "evals":
        man = write_evals(a.out, a.small)
    elif a.cmd == "check":
        import check_k
        man = check_k.run(a.out, a.block)
    else:
        import yaml
        yaml.safe_dump({"data": data_block(a.arm)}, sys.stdout, sort_keys=False)
        return 0
    print(json.dumps({k: v for k, v in man.items() if k not in ("files", "code_sha256")}, default=str)[:2000])
    return 0 if man.get("pass", True) else 1


if __name__ == "__main__":
    sys.exit(main())
