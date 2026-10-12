"""E006 load / save-and-hash / scoring helpers for e006_ft_test.py (notes.txt PC QUEUE, Records). The scoring calls
are E005's (e004_core.score_set, gen_run.run_gen, unedited); only the file writer changes: every record gets the
run's weights_sha256, tag and arm, and every output file is created exclusively (an existing name stops the job).
Paths written into run.json are relative to the experiment folder (no home directory in any record)."""
import argparse
import hashlib
import json
import os
import sys
import time

import e004_ft_test as M4
import e005_sets as S
import gen_run as R

LIK_SUMMARY = ("e004", "big", "h5l", "al")


def write_json(path, obj):
    with open(path, "x") as f:
        json.dump(obj, f, indent=1, default=str)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(a):
    """-> (model, tok, info, the weights file this model came from)."""
    import e005_al_ft_test as AL
    import params as PR
    AL.patch_local_config()
    src = a.weights or a.model
    model, tok, info = M4.load(argparse.Namespace(model=src, device=a.device))
    wfile = os.path.join(src, "model.safetensors") if a.weights else \
        os.path.join(PR.snapshot_dir(a.model), "model.safetensors")
    return model, tok, info, wfile


def save_and_hash(model, tok, a, wfile, exp):
    """trained: save to ../weights/<slug>__<tag> (never over an existing directory), hash the saved file.
    -> (weights_sha256, path relative to the experiment folder or a label)."""
    if a.steps > 0:
        if a.dry or not a.save:
            return "unsaved", None
        wdir = os.path.join(exp, "weights", f"{a.model.replace('/', '__')}__{a.tag}")
        if os.path.exists(wdir):
            sys.exit(f"refusing: {wdir} exists (weights are never overwritten)")
        model.save_pretrained(wdir, safe_serialization=True)
        tok.save_pretrained(wdir)
        return sha_file(os.path.join(wdir, "model.safetensors")), os.path.relpath(wdir, exp)
    label = os.path.relpath(os.path.abspath(a.weights), exp) if a.weights else "hf-snapshot " + a.model
    return sha_file(wfile), label


def _fam_acc(recs, key):
    out = {}
    for r in recs:
        f = r.get("family") or r.get("cell") or "all"
        out.setdefault(f, []).append(bool(r[key]) if key == "strict" else r["right"])
    return {f: round(sum(v) / len(v), 4) for f, v in sorted(out.items())}


def score_all(model, tok, a, stem, extra, lik2, gen2, fails, writer):
    import e004_core as C
    names = a.set_names & {"eval", "cont", "know"}
    counts, summ, t2 = {}, {}, time.time()
    for set_name, render, its in list(S.lik_sets(names, a.chat, a.dry)) + list(lik2):
        with writer(f"{stem}__{set_name}__{render}.jsonl", extra) as f:
            recs, dt = C.score_set(model, tok, a.model, set_name, render, its, a.device, f, None)
        counts[f"{set_name}/{render}"] = len(recs)
        if set_name in LIK_SUMMARY:
            summ[f"LIK {set_name}/{render}"] = _fam_acc(recs, "right")
        print(f"scored {set_name}/{render}: {len(recs)} items in {dt:.0f}s", flush=True)
        if a.dry and not C.finite_scores(recs):
            fails.append(f"non-finite score in {set_name}/{render}")
    R.open = lambda path, mode="r", *k, **kw: writer(path, extra) if mode == "w" else open(path, mode, *k, **kw)
    try:
        for set_name, render, its in list(S.gen_sets(names, a.chat, a.dry)) + list(gen2):
            t3 = time.time()
            recs = R.run_gen(model, tok, its, render, a.device, f"{stem}__gen_{set_name}__{render}.jsonl",
                             R.MAX_NEW, None)
            counts[f"gen_{set_name}/{render}"] = len(recs)
            summ[f"GEN {set_name}/{render}"] = dict(_fam_acc(recs, "strict"), capped=sum(r["capped"] for r in recs))
            print(f"generated {set_name}/{render}: {len(recs)} in {time.time() - t3:.0f}s", flush=True)
    finally:
        del R.open
    return dict(eval_s=round(time.time() - t2, 1), eval_counts=counts, summary=summ)
