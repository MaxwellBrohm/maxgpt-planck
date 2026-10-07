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
  python -B test_pins_engines.py [--pins F] [--engines F]   (exit 1 on any failure)"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LICENCES = {"apache-2.0", "mit", "other (falcon-llm-license)"}
SIBS = ["tiiuae/Falcon-H1-Tiny-90M-Instruct-Curriculum", "tiiuae/Falcon-H1-Tiny-90M-Instruct-pre-DPO",
        "tiiuae/Falcon-H1-Tiny-90M-Instruct-Curriculum-pre-DPO", "tiiuae/Falcon-H1-Tiny-90M-Base"]
NOT_RUN = ["HuggingFaceTB/smollm2-135M-SFT-Only", "jingyaogong/minimind-3", "SupraLabs/Supra2-Medium-Instruct"]


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
        if p["license"] not in LICENCES:
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
