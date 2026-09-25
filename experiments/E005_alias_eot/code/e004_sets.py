"""E004 data for the training / scoring entry point (e004_ft_test.py). No torch at import, no model.
Training: train_e004.stream(seed) (Random(1000 * seed + 44)); the target is the answer sentence after "Assistant:"
  (loss on its tokens only, ending in "\\n"); examples longer than max_len tokens are redrawn, never truncated;
  drawn and kept counts per kind are recorded (the notes' stream-level acceptance reads them).
Likelihood items carry turns / question / prefix / cands, so lik.render gives the plain render (checked equal to
  items_e004.prompt(item, True) on all 640 eval items) or SmolLM2's chat render. cands = {"gold": " <gold>",
  "c<i>": " <other in-context value>"}; the value of each label is in cand_vals.
Free-generation items: plain prompt ending "Assistant:" (no prefix), chat msgs, gold, pool, obj words (gen_grade).
Sets (--sets):  eval  E004 eval draw 4004: LIK plain (+ chat) and GEN plain (+ chat), 9 pass families + ID
                cont  E002 continuity: new, extra, old, uprobe, cross (LIK plain; new/extra/cross also chat)
                      + continuity GEN (E001 same_k1/twoslot/noupd/keyorig/neutral d10, uprobe U_neutral k1-3 d10)
                know  closed-book knowledge: khard (40), kbig (441)
                dev   E004 dev draw 4104 (32 per family), LIK plain only: the LR choice, never the eval draw
Probe: E004 probe draw 4204 (16 per family), LIK plain, every --probe-every steps (lock-in step)."""
import items as I
import items_new as N
import eval_extra as X
import uprobe_items as UP
import khard_items as KH
import kbig_items as KB
import items_e004 as E
import train_e004 as T
import gen_grade as G

SET_NAMES = ("eval", "cont", "know", "dev")
CONT_GEN_VARS = ("same_k1", "twoslot", "noupd", "keyorig", "neutral")
UPROBE_GEN_TASKS = ("Uneutral_k1", "Uneutral_k2", "Uneutral_k3")


# ---------------- training ----------------
def encode(tok, ex, cand_ids):
    """-> (ids, labels, how); labels are -100 on the prompt and the token ids on the answer."""
    pre, ans, how = cand_ids(tok, T.prompt(ex), ex["answer"], True)
    return list(pre) + list(ans), [-100] * len(pre) + list(ans), how


def new_stats():
    return {"n": 0, "rejected_long": 0, "drawn": {}, "kept": {}, "len_sum": 0, "len_max": 0, "split": 0}


def example_stream(tok, seed, max_len, stats, cand_ids):
    for ex in T.stream(seed):
        stats["drawn"][ex["kind"]] = stats["drawn"].get(ex["kind"], 0) + 1
        ids, labels, how = encode(tok, ex, cand_ids)
        if len(ids) > max_len:
            stats["rejected_long"] += 1
            continue
        stats["kept"][ex["kind"]] = stats["kept"].get(ex["kind"], 0) + 1
        stats["n"] += 1
        stats["len_sum"] += len(ids)
        stats["len_max"] = max(stats["len_max"], len(ids))
        stats["split"] += how == "split"
        yield ids, labels


# ---------------- E004 items ----------------
def _meta(it):
    out = {"family": it["family"], "cell": it.get("cell"), "vtype": it["vtype"], "k": it["k"], "d": it["d"],
           "idx": it["idx"], "n_obj": it["n_obj"]}
    if it.get("meta", {}).get("latest_ref"):
        out["latest_ref"] = it["meta"]["latest_ref"]
    return out


def lik_item(it):
    others = [v for v in it["candidates"] if v != it["gold"]]
    cands = {"gold": " " + it["gold"], **{f"c{i}": " " + v for i, v in enumerate(others, 1)}}
    return dict(turns=it["turns"], question=it["question"], prefix=it["prefix"], cands=cands,
                cand_vals={"gold": it["gold"], **{f"c{i}": v for i, v in enumerate(others, 1)}}, **_meta(it))


def gen_item(it, set_name):
    gold, pool, obj = G.spec_e004(it)
    msgs = []
    for u, a in it["turns"]:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": it["question"]})
    return dict(set=set_name, prompt=E.prompt(it, False), msgs=msgs, gold=gold, pool=pool, obj=sorted(obj),
                cands=list(it["candidates"]), **_meta(it))


def draw_items(name, families=E.FAMILIES, per_family=None):
    d = E.draw(name, families)
    return {f: (its[:per_family] if per_family else its) for f, its in d.items()}


def e004_lik(name, per_family=None, families=E.FAMILIES):
    return [lik_item(it) for its in draw_items(name, families, per_family).values() for it in its]


def e004_gen(name, per_family=None, families=E.FAMILIES):
    return [gen_item(it, name) for its in draw_items(name, families, per_family).values() for it in its]


def probe_items(dry):
    return e004_lik("probe", 1 if dry else None, E.PASS_FAMILIES)


# ---------------- continuity ----------------
def cont_gen():
    out = []
    for it in N.build():
        if it["d"] == 10 and it["var"] in CONT_GEN_VARS:
            gold, pool, obj = G.spec_cont(it, it["fam"])
            prompt = I.transcript(it["turns"], it["question"], "").rstrip()
            out.append(dict(set="cont_new", prompt=prompt, gold=gold, pool=pool, obj=sorted(obj),
                            cands=[v.strip() for v in it["cands"].values()], var=it["var"], fam=it["fam"],
                            d=it["d"], k=it.get("k")))
    for it in UP.build():
        if it["d"] == 10 and it["task"] in UPROBE_GEN_TASKS:
            gold, pool, obj = G.spec_cont(it, "day")
            head, tail = it["prompt"].rsplit("\nAssistant:", 1)
            out.append(dict(set="cont_uprobe", prompt=head + "\nAssistant:", gold=gold, pool=pool, obj=sorted(obj),
                            cands=[v.strip() for v in it["cands"].values()], task=it["task"], d=it["d"],
                            fam="day", prefix_cut=tail.strip()))
    return out


def lik_sets(names, chat, dry):
    """-> [(set name, render, items)] in a fixed order."""
    sets = []
    if "eval" in names:
        sets.append(("e004", "plain", e004_lik("eval")))
        if chat:
            sets.append(("e004", "chat", e004_lik("eval")))
    if "cont" in names:
        sets += [("new", "plain", N.build()), ("extra", "plain", X.build()), ("old", "plain", I.build()),
                 ("uprobe", "plain", UP.build()), ("cross", "plain", X.build_crossed())]
        if chat:
            sets += [("new", "chat", N.build()), ("extra", "chat", X.build()), ("cross", "chat", X.build_crossed())]
    if "know" in names:
        sets += [("khard", "plain", KH.build()), ("kbig", "plain", KB.build())]
    if "dev" in names:
        sets.append(("dev", "plain", e004_lik("dev")))
    if dry:
        sets = [(s, r, its[:2]) for s, r, its in sets]
    return sets


def gen_sets(names, chat, dry):
    """-> [(set name, render, gen items)]; the dev draw is likelihood only."""
    sets = []
    if "eval" in names:
        sets.append(("e004", "plain", e004_gen("eval")))
        if chat:
            sets.append(("e004", "chat", e004_gen("eval")))
    if "cont" in names:
        sets.append(("cont", "plain", cont_gen()))
    if dry:
        sets = [(s, r, its[:2]) for s, r, its in sets]
    return sets
