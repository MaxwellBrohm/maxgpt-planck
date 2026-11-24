"""SPEC 5 acceptance over a generated data root (gen.py check OUT): the full kept SKILL stream, every eval file and
the FACT streams' purity. Writes OUT/check.json; pass only if every rule holds (E005 AUDIT Q7: full streams)."""
from __future__ import annotations

import glob
import os

import numpy as np

import kcommon as K
import pools as P
import purity as U

EVAL_SPLIT = {"k_eval_id": "eval", "k_eval_s": "evals", "k_dev": "eval", "k_probe": "eval", "k2_a": "eval",
              "k2_b": "eval"}
CHECK_CODE = ("check_k.py", "cuegate.py", "cues.py", "purity.py", "oracles.py", "pools.py", "world.py", "kcommon.py")
#   K3 round 15 (2026-10-12, REVIEW 16b N-1): check.json records the sha256 of every module the check runs, so a pass
#   names the gate that produced it (check_k, cuegate and cues were in no manifest); no rule reads it


def fact_purity(root: str, pl) -> dict:
    """Fact streams: no skill-pool (S, T), held-out (VH, AH) or alias/marker token; F0 holds no F-pool name;
    FL holds F-pool names of FB_low entities only (checked on the bio docs' and the fact QA records' name triples;
    review 2026-09-27: a QA-only leak passed when only bio docs were read)."""
    import json
    import world as Wd
    own = pl.owner()
    w = Wd.make_world(pl)
    low = {tuple(int(x) for x in w.names[i]) for i in range(K.N_LOW)}
    f1 = set(pl["F1"].tolist())
    rep = {}
    for arm_dir in sorted(glob.glob(os.path.join(root, "*", "fact_bio"))):
        arm = os.path.basename(os.path.dirname(arm_dir))
        bad = {"skill_or_heldout_token": 0, "fb_name_in_F0": 0, "non_low_name_in_FL": 0, "qa_fb_name_in_F0": 0,
               "qa_non_low_name_in_FL": 0}
        forbidden = {"S1", "S2", "S3", "T1", "T2", "T3", "VH", "AH", "AL", "MK"}
        for f in sorted(glob.glob(os.path.join(arm_dir, "*.bin"))):
            docs = np.fromfile(f, dtype=np.uint16).reshape(-1, K.BIO_LEN)
            cls = {own.get(int(t), "") for t in np.unique(docs)}
            bad["skill_or_heldout_token"] += len(cls & forbidden)
            fb = np.isin(docs[:, 0], pl["F1"])
            if arm == "F0":
                bad["fb_name_in_F0"] += int(fb.sum())
            if arm == "FL":
                bad["non_low_name_in_FL"] += sum(tuple(int(x) for x in d[:3]) not in low for d in docs[fb])
        for f in sorted(glob.glob(os.path.join(os.path.dirname(arm_dir), "fact_qa", "*.jsonl"))):
            with open(f) as fh:
                for line in fh:
                    turns = json.loads(line)["turns"]
                    bad["skill_or_heldout_token"] += bool({own.get(x, "") for t in turns for x in t["ids"]} & forbidden)
                    name = tuple(turns[0]["ids"][:3])
                    if arm == "F0":
                        bad["qa_fb_name_in_F0"] += name[0] in f1
                    if arm == "FL":
                        bad["qa_non_low_name_in_FL"] += name[0] in f1 and name not in low
        rep[arm] = bad
    for arm_dir in sorted(glob.glob(os.path.join(root, "*", "fact_k5"))):
        arm = os.path.basename(os.path.dirname(arm_dir))
        bad = {"skill_or_heldout_token": 0, "hit_in_L0": 0, "miss_in_L100": 0, "non_low_name": 0}
        for f in sorted(glob.glob(os.path.join(arm_dir, "*.jsonl"))):
            for r in K.read_jsonl(f):
                ids = {x for t in r["turns"] for x in t["ids"]}
                bad["skill_or_heldout_token"] += bool({own.get(x, "") for x in ids} & {
                    "S1", "S2", "S3", "T1", "T2", "T3", "VH", "AH", "AL", "MK"})
                tool = r["turns"][0]["ids"] if r["turns"][0]["role"] == "tool" else None
                if tool is None:
                    continue
                bad["hit_in_L0"] += arm == "L0" and K.NONE not in tool
                bad["miss_in_L100"] += arm == "L100" and K.NONE in tool
                bad["non_low_name"] += tuple(tool[1:4]) not in low
        rep[arm] = bad
    return rep


MARKS = ("O13e", "O13a", "O14e", "O14a", "SURFe", "SURFa")     # F's one-kind rules: skill marks, not cheap readers


def ceilings(out: dict) -> dict:
    """K3 round 7 (REVIEW 8a G-1): SPEC 9 reads each family's LIK against its ceiling, the largest of the fitted models
    (the gate's name-blind ceiling and the attribute-blind TA model's) and of the cheap rules barred on that family
    (oracles.TH, but MARKS, which resolve one of F's reference kinds) on the same set. U-TIGHTSPAN read about 0.60 on
    U against the gate's 0.555 (REVIEW 8a), so a fitted ceiling alone could sit under a rule's reading. -> {fam: {set:
    {"value", "from", "gate", "ta", "rule", "rule_value"}}} for the training stream ("train") and every eval set."""
    import oracles as O
    gate = out["cue_gate"]
    res = {}
    for fam, row in gate["ceiling"].items():
        for sn, g in row.items():
            tab = out["sets"].get("skill_train" if sn == "train" else sn, {}).get("oracles", {}).get(fam, {})
            rules = [o for o, (fs, _) in O.TH.items() if fam in fs and o not in MARKS and o in tab]
            best = max(rules, key=lambda o: tab[o], default=None)
            ta = gate.get("ta_model", {}).get("ceiling", {}).get(fam, {}).get(sn)
            cands = [("gate", g)] + ([("ta", ta)] if ta is not None else []) + ([(best, tab[best])] if best else [])
            top = max(cands, key=lambda c: c[1])
            res.setdefault(fam, {})[sn] = {"value": top[1], "from": top[0], "gate": g, "ta": ta, "rule": best,
                                           "rule_value": tab.get(best)}
    return res


def run(root: str, block: int = 512, cues: bool = True) -> dict:
    pl = P.get()
    lex = P.Lex(pl)
    tok = K.load_tokenizer()
    out = {"sets": {}, "fails": []}
    skill = [r for f in sorted(glob.glob(os.path.join(root, "skill", "*.jsonl"))) for r in K.read_jsonl(f)]
    fit = None
    if skill:
        rep = U.accept(skill, pl, "train", lex, tok, block)
        out["sets"]["skill_train"] = rep
        out["fails"] += [f"skill_train: {x}" for x in rep["fails"]]
        fit = rep["surf_counts"]            # SURF on the eval sets: the surface rule the training stream teaches
    evsets = {}
    for name, split in EVAL_SPLIT.items():
        p = os.path.join(root, "evals", f"{name}.jsonl")
        if not os.path.exists(p):
            continue
        recs = [r for r in K.read_jsonl(p) if r.get("qs") and r.get("fam") in tuple("RUBFAP")]
        rep = U.accept(recs, pl, split, lex, tok, block, surf_fit=fit)
        out["sets"][name] = rep
        out["fails"] += [f"{name}: {x}" for x in rep["fails"]]
        nb = sum(r["fam"] == "B" for r in recs)
        if name in ("k_eval_id", "k_eval_s") and rep["groups"].get("n", 0) != 3 * nb // 16:      # K3 round 4 (O-1):
            out["fails"].append(f"{name}: B groups {rep['groups'].get('n', 0)} of {nb} records, not 3n/16")  # BD-26
        evsets[name] = recs
    if skill and cues:          # the fitted cue model (K3): training fit, every eval set scored with it
        import cuegate as G
        out["cue_gate"] = G.run(skill, evsets, lex)
        out["fails"] += [f"cue_gate: {x}" for x in out["cue_gate"]["fails"]]
        out["ceilings"] = ceilings(out)         # K3 round 7 (REVIEW 8a G-1): SPEC 9's ceiling per family and set
    p = os.path.join(root, "evals", "k_eval_ood.jsonl")
    if os.path.exists(p):
        recs = K.read_jsonl(p)
        for tag in sorted({r["set"] for r in recs}):
            rep = U.accept([r for r in recs if r["set"] == tag], pl, "eval", lex, tok, block, rules=False,
                           surf_fit=fit, p_questions=(5, 7) if tag == "OOD-PLONG" else (2, 4))
            out["sets"][tag] = rep
            out["fails"] += [f"{tag}: {x}" for x in rep["fails"]]
    out["fact_purity"] = fact_purity(os.path.join(root), pl)
    out["fails"] += [f"fact {arm}: {k} {v}" for arm, b in out["fact_purity"].items() for k, v in b.items() if v]
    out["pass"] = not out["fails"]
    out["code_sha256"] = {c: K.sha256_file(os.path.join(K.HERE, c)) for c in CHECK_CODE}
    K.write_json(os.path.join(root, "check.json"), out)
    return out
