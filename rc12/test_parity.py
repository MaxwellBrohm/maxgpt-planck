"""RC-12 parity_hf_vllm.py tests without a model (notes STEP 9).
  pick       --pick first = the first n records; spread = n distinct records over every family, reproducible
  diff       first_diff: the first differing index, the shorter length for a strict prefix, None when equal
  verdict    IDENTICAL, NEAR_TIE (every first difference at margin <= NEAR_TIE), DIFFERS (a larger or missing
             margin), FAIL_PROMPT (prompt ids differ), FAIL_STOP (equal ids, different stop reason)
  engine     --engine hfb plays stage vllm with hf_batched.HFBatched (vLLM never loaded) and the report and both metas
             name the engine; the default names vLLM
  stages     stage_vllm -> stage_hf -> stage_report with the stub LLM / toy tokenizer of test_vllm_responder and a
             stub HF responder that decodes through its tokenizer (the capture path of the real one): the same
             replies give IDENTICAL; one changed token on every turn 3 gives NEAR_TIE with a 0.2 margin and DIFFERS
             with 3.0; the HF stage replies to the vLLM run's histories (prompt ids equal); the report names the
             first differing token and prints both texts
Run: python3 -B test_parity.py   (exit 1 on any failure)"""
import argparse
import json
import os
import sys
import tempfile
import types

import hf_batched as HB
import hf_responder as HR
import parity_hf_vllm as PH
import runner as R
import test_vllm_responder as TV
import vllm_responder as VR


def turn(rid="a", t=1, prompt=True, ids=(1, 2), vids=(1, 2), stop="eot", vstop="eot", margin=None):
    h = dict(rid=rid, turn=t, prompt_equal=prompt, ids=list(ids), text=str(ids), stop=stop,
             first_diff=PH.first_diff(list(ids), list(vids)), margin=margin, vllm_rank_under_hf=None)
    v = dict(rid=rid, turn=t, ids=list(vids), text=str(vids), stop=vstop)
    return v, h


def verdict_of(*pairs):
    return PH.compare([p[0] for p in pairs], [p[1] for p in pairs])["verdict"]


def c_diff():
    got = [PH.first_diff([1, 2, 3], [1, 2, 4]), PH.first_diff([1, 2], [1, 2, 3]), PH.first_diff([4], [4]),
           PH.first_diff([], [7])]
    return [] if got == [2, 2, None, 0] else [f"diff: {got}"]


def c_verdict():
    fails = []
    same, near = turn(), turn(t=2, ids=(1, 5), vids=(1, 6), margin=0.2)
    cases = [((same,), "IDENTICAL"), ((same, near), "NEAR_TIE"),
             ((same, turn(t=2, ids=(1, 5), vids=(1, 6), margin=2.0)), "DIFFERS"),
             ((same, turn(t=2, ids=(1, 5), vids=(1, 5, 6), margin=None)), "DIFFERS"),
             ((same, near, turn(t=3, prompt=False)), "FAIL_PROMPT"),
             ((same, turn(t=2, stop="eos", vstop="eot")), "FAIL_STOP")]
    for pairs, want in cases:
        if verdict_of(*pairs) != want:
            fails.append(f"verdict: {verdict_of(*pairs)} != {want}")
    return fails


class StubHF(HR.HFResponder):
    flip, think, torch = False, False, types.SimpleNamespace(__version__="stub")

    def __init__(self, model_id, render="template", dtype=None, device=None):
        self.v = TV.make()
        self.tok, self.render, self.qwen3 = self.v.tok, render, self.v.qwen3 or self.think
        self.eot, self.stop_ids, self.eos, self.ctx, self.device = self.v.eot, self.v.stop_ids, self.v.eos, None, device

    def reply(self, history, i):
        c = TV.StubLLM().one(self.encode(history), TV.StubSP(**VR.sampling_kwargs(None, self.rid, i, self.stop_ids)))
        ids, stop = self.v.stop_of(c.token_ids, c.finish_reason, c.stop_reason)
        if self.flip and i == 3:
            ids = [TV.wid("bravo" if ids[0] != TV.wid("bravo") else "alpha")] + ids[1:]
        return self.tok.decode(ids, skip_special_tokens=True), stop


def refuse(*a, **k):
    raise AssertionError("the other engine was loaded")


def run_stages(d, flip, margin, think=False, engine="vllm"):
    args = argparse.Namespace(model="toy", out=d, n=12, pick="first", data=R.DEV, dtype="bfloat16", device="cpu", gpu_mem=0.85,
                              max_model_len=None, engine=engine)
    saved = (VR.VLLMResponder, HR.HFResponder, PH.hf_margin, sys.modules.get("vllm"), sys.modules.get("transformers"),
             HB.HFBatched)
    sys.modules["vllm"] = types.SimpleNamespace(__version__="stub")
    sys.modules["transformers"] = types.SimpleNamespace(__version__="stub")
    stub_v = TV.make()
    VR.VLLMResponder = (lambda *a, **k: stub_v) if engine == "vllm" else refuse
    HB.HFBatched = (lambda *a, **k: stub_v) if engine == "hfb" else refuse
    StubHF.flip, StubHF.think = flip, think
    HR.HFResponder = StubHF
    PH.hf_margin = lambda h, prompt, prefix, a, b: (margin, 1)
    try:
        PH.stage_vllm(args)
        VR.VLLMResponder = saved[0]              # StubHF builds its own toy responder
        PH.stage_hf(args)
        PH.stage_report(args)
    finally:
        VR.VLLMResponder, HR.HFResponder, PH.hf_margin = saved[:3]
        HB.HFBatched = saved[5]
        for name, mod in (("vllm", saved[3]), ("transformers", saved[4])):
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    return json.load(open(os.path.join(d, "parity.json"))), open(os.path.join(d, "parity.txt")).read()


def c_stages():
    fails = []
    for flip, margin, want in ((False, None, "IDENTICAL"), (True, 0.2, "NEAR_TIE"), (True, 3.0, "DIFFERS")):
        with tempfile.TemporaryDirectory() as d:
            res, text = run_stages(d, flip, margin)
            n_t3 = sum(t["turn"] == 3 for t in res["per_turn"])
            if res["verdict"] != want:
                fails.append(f"stages: flip {flip} margin {margin}: {res['verdict']} != {want}")
            if res["turns"] != sum(len(r["turns"]) for r in R.load()[:12]) or res["prompt_mismatch"]:
                fails.append(f"stages: {res['turns']} turns, {res['prompt_mismatch']} prompt mismatches")
            if flip and (res["turns"] - res["identical"] != n_t3 or "first differing token 0" not in text
                         or text.count("    HF   ") != n_t3 or text.count("    vLLM ") != n_t3):
                fails.append(f"stages: flip {flip}: {res['turns'] - res['identical']} differing turns, want {n_t3}")
            if not os.path.exists(os.path.join(d, "vllm_run", "transcripts.jsonl")):
                fails.append("stages: the vLLM stage wrote no lockstep transcripts")
            if not flip and (res.get("engine") != "vllm" or "HF vs vLLM," not in text):
                fails.append(f"stages: engine {res.get('engine')!r} or the header does not name vLLM")
    with tempfile.TemporaryDirectory() as d:
        res, text = run_stages(d, False, None, engine="hfb")
        meta = json.load(open(os.path.join(d, "vllm_meta.json")))
        if res["verdict"] != "IDENTICAL" or res.get("engine") != "hfb" or meta.get("engine") != "hfb" \
                or "HF vs batched HF" not in text:
            fails.append(f"stages: --engine hfb: {res['verdict']}, engine {res.get('engine')!r}, meta "
                         f"{meta.get('engine')!r}, header {text.splitlines()[0]!r}")
    with tempfile.TemporaryDirectory() as d:
        res, _ = run_stages(d, False, None, think=True)
        if res["verdict"] != "FAIL_PROMPT" or res["prompt_mismatch"] != res["turns"]:
            fails.append(f"stages: HF renders thinking off, vLLM not: {res['verdict']}, {res['prompt_mismatch']}")
    return fails


def c_pick():
    recs = R.load()
    first, spread = PH.pick(recs, 20, "first"), PH.pick(recs, 20, "spread")
    fams = [r["family"] for r in spread]
    if first != recs[:20] or len(spread) != 20 or len(set(fams)) != len({r["family"] for r in recs}) \
            or len({r["id"] for r in spread}) != 20 or spread != PH.pick(recs, 20, "spread"):
        return [f"pick: first {len(first)}, spread {len(spread)} over {len(set(fams))} families"]
    return []


def main():
    fails = []
    for fn in (c_diff, c_verdict, c_stages, c_pick):
        try:
            fails += fn()
        except Exception as e:  # noqa: BLE001  (a raising check is a failure line, not a crash)
            fails.append(f"{fn.__name__}: raised {type(e).__name__}: {str(e)[:120]}")
    print("\n".join(f"FAIL {f}" for f in fails) or "ALL PARITY TOOL CHECKS PASS (stubs, no model)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
