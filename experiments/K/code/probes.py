"""SPEC 9 scorer (code/k_eval in SPEC 14): skill items and fact probes scored by likelihood and greedy decoding on a
harness checkpoint, fp32, one causal pass per prompt (plus one more for GEN's <|end|> check and P's marker).

A scorer is a function fn(list of prompts) -> (N, vocab) next-token logits; model_fn wraps a harness PlanckLM (plain
causal attention, doc=None, prompts right-padded so padding never precedes a scored position). Per record:
  lik    candidate-set argmax equals gold; a tie over k candidates that holds gold earns 1/k (a uniform model scores
         exactly chance = 1/|C|)
  ce_c   -log softmax over C of gold (nats); ce_full over the vocabulary; ce_v renormalized over V (latent)
  gen    greedy equals gold and, for chat prompts, the next greedy token is <|end|> (strict); bio prompts: greedy only
  marker P items only: the greedy first answer token is the system marker y
  python probes.py --ckpt CK.pt --items evals/k_eval_id.jsonl [...] --out scores.jsonl [--device cuda]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

import kcommon as K

if K.HARNESS not in sys.path:
    sys.path.insert(0, K.HARNESS)

ROLE = K.ROLE_IDS


def chat_ids(turns: list[dict]) -> list[int]:
    return [x for t in turns for x in (ROLE[t["role"]], *t["ids"], K.END)]


def prompts_of(rec: dict) -> dict:
    """-> {"main": the prompt ending where gold is predicted, "gold", "chat": bool, "marker": (P's marker
    prompt, the marker) or None}"""
    if "ids" in rec:
        return {"main": list(rec["ids"]), "gold": rec["gold"], "chat": False, "marker": None}
    turns = rec["turns"]
    if "qs" not in rec:                                   # a fact chat prompt: every turn is context
        return {"main": chat_ids(turns) + [K.ASST], "gold": rec["gold"], "chat": True, "marker": None}
    q = rec["qs"][-1]
    ans = turns[q["turn"]]["ids"]
    base = chat_ids(turns[:q["turn"]]) + [K.ASST]
    if len(ans) == 2:                                     # P: " y v", scored after the marker
        return {"main": base + [ans[0]], "gold": q["gold"], "chat": True, "marker": (base, ans[0])}
    return {"main": base, "gold": q["gold"], "chat": True, "marker": None}


def model_fn(model, device: str = "cpu"):
    model = model.to(device).eval()

    @torch.no_grad()
    def fn(prompts: list[list[int]]) -> torch.Tensor:
        T = max(len(p) for p in prompts)
        idx = torch.zeros(len(prompts), T, dtype=torch.long)
        for i, p in enumerate(prompts):
            idx[i, :len(p)] = torch.tensor(p)
        logits, _ = model(idx.to(device))
        last = torch.tensor([len(p) - 1 for p in prompts], device=device)
        return logits[torch.arange(len(prompts), device=device), last].float().cpu()
    return fn


def load(ckpt: str, device: str = "cpu"):
    import runio
    from config import PlanckConfig
    from model import build_model
    ck = runio.load_checkpoint(ckpt)
    model = build_model(PlanckConfig.from_dict(ck["model_cfg"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), weights_sha(model)


def weights_sha(model) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def _chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def score(fn, recs: list[dict], vpool, batch: int = 64) -> list[dict]:
    V = torch.as_tensor(np.asarray(vpool, dtype=np.int64))
    pr = [prompts_of(r) for r in recs]
    out = []
    for idx in _chunks(list(range(len(recs))), batch):
        logits = fn([pr[i]["main"] for i in idx])
        logp = torch.log_softmax(logits.float(), -1)
        for row, i in enumerate(idx):
            r, p, lp = recs[i], pr[i], logp[row]
            g = int(p["gold"])
            c = torch.as_tensor(r["cands"])
            lc = lp[c]
            ties = lc == lc.max()
            gi = int((c == g).nonzero()[0, 0])
            lv = lp[V]
            out.append({"id": r["id"], "set": r.get("set", ""), "fam": r.get("fam", r.get("kind", "")),
                        "ent": r.get("ent"), "attr": r.get("attr"),
                        "gold": g, "n_c": len(c), "lik": float(ties[gi]) / float(ties.sum()),
                        "ce_c": float(-(lc[gi] - torch.logsumexp(lc, 0))), "greedy": int(lp.argmax()),
                        "ce_full": float(-lp[g]), "ce_v": float(-(lp[g] - torch.logsumexp(lv, 0))) if g in
                        set(V.tolist()) else float("nan"), "argmax_v": int(V[lv.argmax()])})
    need = [j for j, o in enumerate(out) if o["greedy"] == o["gold"] and pr[j]["chat"]]
    ends = {}
    for idx in _chunks(need, batch):
        g = fn([pr[j]["main"] + [out[j]["gold"]] for j in idx]).argmax(-1)
        ends.update({j: int(g[k]) == K.END for k, j in enumerate(idx)})
    for j, o in enumerate(out):
        o["gen"] = o["greedy"] == o["gold"] and (ends.get(j, False) or not pr[j]["chat"])
    mk = [j for j in range(len(out)) if pr[j]["marker"] is not None]
    for idx in _chunks(mk, batch):
        g = fn([pr[j]["marker"][0] for j in idx]).argmax(-1)
        for k, j in enumerate(idx):
            out[j]["marker"] = int(g[k]) == pr[j]["marker"][1]
    return out


def summarize(res: list[dict]) -> dict:
    """Per (set, family): LIK, CE_C, GEN, chance = mean 1/|C|, n."""
    groups = {}
    for o in res:
        groups.setdefault((o["set"], o["fam"]), []).append(o)
    out = {}
    for (s, f), g in sorted(groups.items()):
        out[f"{s}/{f}"] = {"n": len(g), "lik": float(np.mean([o["lik"] for o in g])),
                           "chance": float(np.mean([1 / o["n_c"] for o in g])),
                           "ce_c": float(np.mean([o["ce_c"] for o in g])),
                           "gen": float(np.mean([o["gen"] for o in g]))}
    return out


def stored_bits(res: list[dict], n_entities: int, key: str = "ce_full", vbits: float = 11.0) -> float:
    """Physics 3.3 Def. 4.3 value term: N x 6 x (log2 |V| - mean CE / ln 2), CE averaged over a key's prompts."""
    per = {}
    for o in res:
        per.setdefault((o["ent"], o["attr"]), []).append(o[key])
    ce = np.mean([np.mean(v) for v in per.values()])
    return float(n_entities * K.N_ATTR * (vbits - ce / math.log(2)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args(argv)
    import pools as P
    model, wsha = load(a.ckpt, a.device)
    fn = model_fn(model, a.device)
    rows, summ = [], {}
    for path in a.items:
        recs = K.read_jsonl(path)
        isha = K.sha256_file(path)
        res = score(fn, recs, P.get()["V"], a.batch)
        for o in res:
            o.update(item_sha256=isha, weights_sha256=wsha)
        rows += res
        summ[os.path.basename(path)] = summarize(res)
    K.write_jsonl(a.out, rows)
    K.write_json(a.out + ".summary.json", {"weights_sha256": wsha, "summary": summ})
    print(json.dumps(summ)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
