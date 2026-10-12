"""E004 model-side helpers for e004_ft_test.py, adapted from E003 e003_ft_test.py (same code path): memory
telemetry, the architecture-generic answer-only loss and its two wrong-loss mutants, load checks (parameter count
from config via params.py, no uninitialised parameter, head tie state, sanity NLL), likelihood scoring through
lik.py (E002 byte copy), and the in-training probe over the 9 pass-rule families. Nothing is loaded at import."""
import hashlib, json, math, time

import torch
import torch.nn.functional as Fnn

import lik
import metrics_ft as MF
import params as PR
import items_e004 as E

SANITY_TEXT = ("Once upon a time, there was a little girl named Lily. She liked to play with her red ball in the park. "
               "One day, she saw a big dog near the tree.")
SANITY_MAX_NLL = 8.0
TRAINED_CTX = {"roneneldan/TinyStories": 512, "EleutherAI/pythia": 2048, "HuggingFaceTB/SmolLM2": 8192}
REC_KEYS = ("task", "cond", "d", "fam", "var", "sid", "k", "cat", "family", "cell", "vtype", "idx", "n_obj",
            "latest_ref", "cand_vals")


def slug(m):
    return m.replace("/", "__")


def phash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def trained_ctx(model_id):
    for k, v in TRAINED_CTX.items():
        if model_id.startswith(k):
            return v
    return None


class Mem:
    def __init__(self, device):
        self.device, self.peak_driver, self.peak_alloc = device, 0, 0

    def sample(self):
        if self.device != "mps":
            return 0, 0
        a, d = torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
        self.peak_alloc, self.peak_driver = max(self.peak_alloc, a), max(self.peak_driver, d)
        return a, d

    def s(self):
        a, d = self.sample()
        return f"mps_alloc={a/2**30:.2f}G driver={d/2**30:.2f}G peak_driver={self.peak_driver/2**30:.2f}G"


def make_batch(exs, pad_id, pad_to=None):
    L = pad_to or max(len(x) for x, _ in exs)
    B = len(exs)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    lab = torch.full((B, L), -100, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for i, (x, y) in enumerate(exs):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        att[i, :len(x)] = 1
    return ids, lab, att


def answer_loss(model, ids, att, lab):
    """mean next-token cross-entropy over labelled (answer) positions only; architecture-generic (E003)."""
    h = model.base_model(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = lab[:, 1:]
    m = tgt != -100
    logits = model.get_output_embeddings()(h[:, :-1][m])
    return Fnn.cross_entropy(logits.float(), tgt[m])


def close(a, b):
    return abs(a - b) <= 1e-3 * max(1.0, abs(a))


def loss_mutants(model, ids, att, lab):
    """two WRONG answer-only losses; the self-check comparator must reject both (dry-run self-test)."""
    with torch.no_grad():
        h = model.base_model(input_ids=ids, attention_mask=att).last_hidden_state
        head = model.get_output_embeddings()
        m = lab != -100
        no_shift = Fnn.cross_entropy(head(h[m]).float(), lab[m]).item()
        lab2 = ids.clone()
        lab2[att == 0] = -100
        with_prompt = answer_loss(model, ids, att, lab2).item()
    return {"no_shift": no_shift, "prompt_labelled": with_prompt}


@torch.no_grad()
def sanity_nll(model, tok, device):
    ids = torch.tensor([tok(SANITY_TEXT).input_ids], device=device)
    return float(model(input_ids=ids, labels=ids).loss.item())


def load_checks(model, info, model_id, device, tok):
    """-> (dict, fatal message or None). Same checks as E003."""
    cfg_count = PR.count(PR.load_config(model_id))
    n_params = sum(p.numel() for p in model.parameters())
    names = dict(model.named_parameters())
    missing = [k for k in (info or {}).get("missing_keys", []) if k in names]
    tied_head = cfg_count["out_head"] == 0
    shares = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    if tied_head:
        missing = [k for k in missing if not k.startswith("lm_head")]
    nll = sanity_nll(model, tok, device)
    out = {"params_config": cfg_count, "params_model": n_params, "params_match": n_params == cfg_count["total"],
           "missing_param_keys": missing, "unexpected_keys": sorted(map(str, (info or {}).get("unexpected_keys", [])))[:10],
           "mismatched_keys": [str(x) for x in (info or {}).get("mismatched_keys", [])][:10],
           "head_tied_to_input_embedding": shares, "config_says_tied": tied_head, "sanity_nll": round(nll, 4)}
    fatal = None
    if missing:
        fatal = f"parameters not in the checkpoint (would be random): {missing[:5]}"
    elif out["mismatched_keys"]:
        fatal = f"shape-mismatched keys: {out['mismatched_keys']}"
    elif shares != tied_head:
        fatal = f"output head tie state {shares} differs from config ({tied_head})"
    elif not (nll < SANITY_MAX_NLL):
        fatal = f"sanity NLL {nll:.2f} >= {SANITY_MAX_NLL} (random or broken head?)"
    return out, fatal


@torch.no_grad()
def score_set(model, tok, model_id, set_name, render, its, device, f=None, mem=None):
    """likelihood of every candidate after the forced prefix; items carry turns (lik.render) or a prompt."""
    recs, t0 = [], time.time()
    for i, it in enumerate(its):
        if "turns" in it:
            prompt, add_sp = lik.render(it, render, tok, model_id)
        else:
            prompt, add_sp = it["prompt"], True
        sc, L = lik.score_item(model, tok, prompt, it["cands"], device, add_sp)
        rec = {"set": set_name, "render": render, "id": i, "h": phash(prompt), "seq_len": L}
        rec.update({k: it[k] for k in REC_KEYS if k in it})
        rec["scores"] = {lab: v[0] for lab, v in sc.items()}
        rec["ntok"] = {lab: v[1] for lab, v in sc.items()}
        rec["right"] = MF.right(rec["scores"])
        recs.append(rec)
        if f is not None:
            f.write(json.dumps(rec) + "\n")
        if device == "mps" and i % 200 == 0:
            torch.mps.empty_cache()
            if mem is not None:
                mem.sample()
    return recs, time.time() - t0


def family_acc(recs, families=E.PASS_FAMILIES):
    out = {}
    for fam in families:
        xs = [MF.right(r["scores"]) for r in recs if r.get("family") == fam]
        out[fam] = round(sum(xs) / len(xs), 4) if xs else None
    return out


def probe(model, tok, model_id, its, device):
    """likelihood accuracy per pass-rule family on the probe draw (4204); lock-in reads min over the 9."""
    model.eval()
    recs, _ = score_set(model, tok, model_id, "probe", "plain", its, device)
    model.train()
    return family_acc(recs)


def lock_in(traj):
    return MF.lock_in_step(traj, keys=tuple(E.PASS_FAMILIES)) if traj else None


def finite_scores(recs):
    return all(isinstance(v, float) and math.isfinite(v) for r in recs for v in r["scores"].values())
