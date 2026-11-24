"""RC-12 extras' configs (notes STEP 13; research/CUTS_AUDIT_2026-10.md step 2): rc12/pins.json against
rc12/engines.json. No model, no network: reads the two files only.
  pairs      every pinned model has an engines.json entry whose dtype is the pinned checkpoint dtype and whose
             trust_remote_code is set exactly when the checkpoint ships custom code
  family     every "family verdict (X)" entry names a model X that engines.json lists with the same family and engine
  licence    every pinned model's licence is one this file allows (Apache-2.0, MIT, or the TII Falcon License that the
             core panel's Falcon-H1-Tiny-90M-Instruct carries); skipped models are not in engines.json, so the queue
             writes NOENGINE for them
  P-144      the four Falcon-H1-Tiny-90M siblings: vLLM, family Falcon-H1, bfloat16, the Instruct's verdict; the Base
             has no chat template (plain render only) and the other three have one
  minimind-3 (STEP 13a, 14) pinned and out of "skipped"; vLLM at its float16 with its own parity (a family of one)
  shards     every "shards" an int >= 1; cRia's 10 with max_batch 32 (STEP 14X)
  tier1      (STEP 16P, 2026-11-19) the five tier-1 rows named before the lock are pinned in pins.json "tier1" (full
             revision, licence allowed here, every weights file with its sha256 and bytes) and in neither "models"
             nor "skipped" (a row joins "models" with its dev numbers); the three with a family verdict have it in
             engines.json (vLLM, the pinned dtype); the two that needed their own parity carry exactly its verdict
             (STEP 16P2, 2026-11-23: their engines.json rows came after it; a row named None here has no entry yet, so
             the queue writes NOENGINE for it and dev_batch cannot run it before its parity). The LFM
             Open License v1.0 is allowed as the TII Falcon License is: the core panel's LFM2 and LFM2.5 models
             carry it (Apache-2.0-based, royalty-free, not gated; commercial use above a revenue threshold is not
             licensed, which an evaluation is not); since review r1 (2026-11-19) only on a LiquidAI/ repo, so it
             widens no other pin
  python -B test_pins_engines.py [--pins F] [--engines F]   (exit 1 on any failure)"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LICENCES = {"apache-2.0", "mit", "other (falcon-llm-license)"}
LFM = ("other (lfm1.0)", "LiquidAI/")      # the LFM Open License only on Liquid AI's own repos (STEP 16P; review r1)


def licence_ok(m, lic):
    """lic is allowed for model m: a LICENCES entry, or the LFM Open License v1.0 on a LiquidAI/ repo."""
    return lic in LICENCES or (lic == LFM[0] and m.startswith(LFM[1]))


TIER1 = {"taobao-mnn/MNN-Pocket-0.3B": "bf16 NEAR_TIE",      # own parity (C2)
         "textilelabs/Loom-Weave-4": "fp32 IDENTICAL",
         "LiquidAI/LFM2-2.6B-Exp": "family verdict (LFM2-700M)",
         "LiquidAI/LFM2.5-1.2B-Instruct": "family verdict (LFM2.5-230M)",
         "Qwen/Qwen3.5-2B": "family verdict (Qwen3.5-0.8B)"}
SIBS = ["tiiuae/Falcon-H1-Tiny-90M-Instruct-Curriculum", "tiiuae/Falcon-H1-Tiny-90M-Instruct-pre-DPO",
        "tiiuae/Falcon-H1-Tiny-90M-Instruct-Curriculum-pre-DPO", "tiiuae/Falcon-H1-Tiny-90M-Base"]
NOT_RUN = ["HuggingFaceTB/smollm2-135M-SFT-Only", "SupraLabs/Supra2-Medium-Instruct"]
MINIMIND, CRIA = "jingyaogong/minimind-3", "sz14/cRia-LM-75M-Instruct"


def check(pins, eng):
    fails = []
    em = eng["models"]
    by_base = {m.split("/")[-1]: m for m in em}
    for m, p in pins["models"].items():
        e = em.get(m)
        if e is None:
            fails.append(f"pairs: {m} pinned but not in engines.json")
            continue
        if e.get("dtype") != p["checkpoint_dtype"]:
            fails.append(f"pairs: {m} engines dtype {e.get('dtype')} != pinned {p['checkpoint_dtype']}")
        if bool(e.get("trust_remote_code")) != bool(p["custom_code"]):
            fails.append(f"pairs: {m} trust_remote_code {e.get('trust_remote_code')} vs custom_code {p['custom_code']}")
        if not licence_ok(m, p["license"]):
            fails.append(f"licence: {m} {p['license']!r} is not an allowed licence")
    for m, e in em.items():
        par = e.get("parity", "")
        if par.startswith("family verdict ("):
            ref = by_base.get(par[len("family verdict ("):].rstrip(")"))
            if ref is None:
                fails.append(f"family: {m} names {par!r}, not in engines.json")
            elif (em[ref].get("family"), em[ref].get("engine")) != (e.get("family"), e.get("engine")):
                fails.append(f"family: {m} {e.get('family')}/{e.get('engine')} vs {ref} "
                             f"{em[ref].get('family')}/{em[ref].get('engine')}")
    for m in NOT_RUN:
        if m not in pins.get("skipped", {}):
            fails.append(f"licence: {m} missing from pins.json skipped")
        if m in em:
            fails.append(f"licence: {m} is in engines.json but was not added")
    for m in SIBS:
        e, p = em.get(m, {}), pins["models"].get(m)
        if (e.get("engine"), e.get("family"), e.get("dtype")) != ("vllm", "Falcon-H1", "bfloat16"):
            fails.append(f"P-144: {m} engine/family/dtype {e.get('engine')}/{e.get('family')}/{e.get('dtype')}")
        if e.get("parity") != "family verdict (Falcon-H1-Tiny-90M-Instruct)":
            fails.append(f"P-144: {m} parity {e.get('parity')!r}")
        if p is None:
            fails.append(f"P-144: {m} not pinned")
            continue
        plain_only = p["chat_template"].startswith("none")
        if plain_only != m.endswith("-Base"):
            fails.append(f"P-144: {m} chat_template {p['chat_template'][:40]!r}")
    p, e = pins["models"].get(MINIMIND), em.get(MINIMIND, {})         # STEP 13a / STEP 14X
    if p is None or MINIMIND in pins.get("skipped", {}):
        fails.append("minimind-3: not pinned, or still in skipped")
    if (e.get("engine"), e.get("dtype"), e.get("parity")) != ("vllm", "float16", "fp16 NEAR_TIE") \
            or sum(x.get("family") == e.get("family") for x in em.values()) != 1:
        fails.append(f"minimind-3: engine/dtype/parity {e.get('engine')}/{e.get('dtype')}/{e.get('parity')} or its "
                     f"family {e.get('family')!r} is shared (its own parity, not a family verdict)")
    for m, x in em.items():
        v = x.get("shards", 1)
        if type(v) is not int or v < 1:
            fails.append(f"shards: {m} {v!r}")
    c = em.get(CRIA, {})
    if (c.get("shards"), c.get("max_batch")) != (10, 32):
        fails.append(f"shards: cRia shards {c.get('shards')} max_batch {c.get('max_batch')} (STEP 14X: 10 and 32)")
    t1 = pins.get("tier1", {})                                       # STEP 16P
    if sorted(t1) != sorted(TIER1):
        fails.append(f"tier1: pinned {sorted(t1)} vs {sorted(TIER1)}")
    for m, p in t1.items():
        if m in pins["models"] or m in pins.get("skipped", {}):
            fails.append(f"tier1: {m} is also in models or skipped (it joins models with its dev numbers)")
        if not licence_ok(m, p.get("license")):
            fails.append(f"tier1: {m} licence {p.get('license')!r} is not an allowed licence")
        w = p.get("weights") or []
        if not re.fullmatch(r"[0-9a-f]{40}", str(p.get("revision"))) or not w or any(
                not re.fullmatch(r"[0-9a-f]{64}", str(x.get("sha256"))) or type(x.get("bytes")) is not int for x in w):
            fails.append(f"tier1: {m} revision or weights not pinned in full")
        e, want = em.get(m), TIER1.get(m)
        if want is None and e is not None:
            fails.append(f"tier1: {m} is in engines.json before its own parity (families are per recipe, STEP 13a)")
        if want is not None and (e is None or (e.get("engine"), e.get("parity"), e.get("dtype")) !=
                                 ("vllm", want, p.get("checkpoint_dtype"))
                                 or bool(e.get("trust_remote_code")) != bool(p.get("custom_code"))):
            fails.append(f"tier1: {m} engines entry {e and (e.get('engine'), e.get('parity'), e.get('dtype'))} vs "
                         f"vllm, {want}, {p.get('checkpoint_dtype')}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pins", default=os.path.join(HERE, "pins.json"))
    ap.add_argument("--engines", default=os.path.join(HERE, "engines.json"))
    a = ap.parse_args()
    fails = check(json.load(open(a.pins)), json.load(open(a.engines)))
    for f in fails:
        print("FAIL", f)
    print("test_pins_engines:", "PASS" if not fails else f"{len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
