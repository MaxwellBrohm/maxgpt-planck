"""Token-count check of the likelihood rows under the panel's own tokenizers (prereg draft s4; STEP 11 FIX ROUND).
TOKENIZER ONLY: it loads AutoTokenizer and AutoConfig (config.json, for model_type) from the local Hugging Face cache
with HF_HUB_OFFLINE=1, never a model's weights, and scores every dev row with a constant scorer, so only the token
counts matter. Per model and render, per key: n, n_equal (every candidate one count), n_matched (the gold has >= 1
foil of its own count; lik_score), chance (mean 1 / (1 + foils)); against the comparator (Qwen2.5-0.5B-Instruct,
template): rows with >= 1 foil common to both tokenizers (lik_compare) and rows equal under both. A model whose
tokenizer is not in the cache is listed as not measured, with the error.
Engine options (STEP 14S): trust_remote_code and chat_template come from the model's engines.json entry
(lik_adapters.engine_opts), as in its generation runs; a model whose tokenizer has no chat template is measured on
the plain render alone (its template render listed under "skipped").
  <python with transformers> -B lik_tokcheck.py [--models id,id,...] [--out logs/lik_tokcheck.json]
(default models: engines.json's models and not_run)"""
import argparse
import json
import os
import sys
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hf_responder as HR  # noqa: E402
import lik_adapters as LA  # noqa: E402
import lik_compare as LC  # noqa: E402
import lik_rows as LR  # noqa: E402
import lik_score as LS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARATOR = "Qwen/Qwen2.5-0.5B-Instruct"


def panel():
    e = json.load(open(os.path.join(HERE, "engines.json")))
    extra = e.get("not_run") or []
    return list(e["models"]) + (list(extra) if isinstance(extra, (list, dict)) else [extra])


def zero(pre, cont):
    return 0.0


def score(model_id, rows, renders, engines=None):
    """-> (model_type, {render: scored rows}, skipped renders, engine options used)"""
    from transformers import AutoConfig, AutoTokenizer
    opts = LA.engine_opts(model_id, engines if engines is not None else os.path.join(HERE, "engines.json"))
    trc = {"trust_remote_code": True} if opts.get("trust_remote_code") is True else {}
    model_type = getattr(AutoConfig.from_pretrained(model_id, **trc), "model_type", "")
    hf = AutoTokenizer.from_pretrained(model_id, **trc)
    HR.set_template(hf, opts.get("chat_template"))
    tok = LA.HFTok(hf, str(model_type).startswith("qwen3"))
    has_template = bool(getattr(hf, "chat_template", None))
    out = {r: LS.score_rows(rows, zero, tok, r) for r in renders if r != "template" or has_template}
    if not out:
        raise ValueError("no render to measure (no chat template and no plain render asked)")
    return model_type, out, [r for r in renders if r not in out], sorted(opts)


def per_key(scored):
    out = {}
    for key, s in LS.summarize(scored).items():
        out[key] = dict(n=s["n"], n_equal=s["n_equal"], n_matched=s["n_matched"], chance=s["chance_matched"])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None)
    ap.add_argument("--renders", default="template,plain")
    ap.add_argument("--out", default=os.path.join(HERE, "logs", "lik_tokcheck.json"))
    a = ap.parse_args(argv)
    models = a.models.split(",") if a.models else panel()
    renders = a.renders.split(",")
    rows, _ = LR.build(LR.load())
    res, scored = {}, {}
    for m in models:
        try:
            mt, sc, skipped, used = score(m, rows, renders)
        except Exception as e:                      # not in the local cache (offline): reported, never guessed
            res[m] = dict(measured=False, error=f"{type(e).__name__}: {str(e)[:160]}")
            print(f"{m:40s} NOT MEASURED ({type(e).__name__})")
            continue
        scored[m] = sc
        rs = list(sc)
        res[m] = dict(measured=True, model_type=mt, renders={r: per_key(sc[r]) for r in rs},
                      renders_agree=all(per_key(sc[r]) == per_key(sc[rs[0]]) for r in rs), skipped=skipped,
                      engine_opts=used)
        t = res[m]["renders"][rs[0]]
        print(f"{m:40s} " + " ".join(f"{k} {v['n_matched']}/{v['n_equal']}/{v['n']}" for k, v in t.items())
              + (f" (render {rs[0]}; skipped {','.join(skipped)})" if skipped else ""))
    base = scored.get(COMPARATOR, {}).get("template")
    for m, sc in scored.items():
        if base is None or m == COMPARATOR or "template" not in sc:
            continue
        cmp_ = LC.compare(sc["template"], base)
        res[m]["vs_comparator"] = {k: dict(n_common=v["n_common"], n_equal_both=v["n_equal_both"])
                                   for k, v in cmp_.items()}
        print(f"{m:40s} vs comparator: " + " ".join(f"{k} {v['n_common']}/{v['n_equal_both']}"
                                                     for k, v in cmp_.items()))
    tot = Counter(r["key"] for r in rows)
    json.dump(dict(rows=dict(tot), renders=renders, models=res,
                   legend="per key n_matched/n_equal/n; vs comparator n_common/n_equal_both (template)"),
              open(a.out, "w"), indent=1)
    print(f"legend: key matched/equal/n (first render {renders[0]}); vs comparator: common-foil rows / equal under "
          f"both -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
