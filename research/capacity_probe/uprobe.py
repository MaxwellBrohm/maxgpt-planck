"""State-update control variants, to separate surface-form copying from updating.

The main U items phrase the original as "I have a dentist appointment on D0" and the
update as "the appointment got moved to D1", then prime the answer with
"Your dentist appointment is on". Only the ORIGINAL shares the n-gram "appointment on",
so a pure copy (induction) circuit picks D0. Two controls:

  U_same    : every update restates the same surface form ("my dentist appointment is on Di now"),
              so the latest matching key is the right one; a first-match head still picks D0.
  U_neutral : main-item surface forms, but a neutral answer prefix ("That would be") that
              shares no n-gram with either statement.

usage: python uprobe.py <hf_model_id> [--dtype fp32|bf16]
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, sys, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import items as I
from run_capacity import score_item


def build(n_scen=32, distances=(0, 4, 10), seed=99):
    rng = random.Random(seed)
    out = []
    for d in distances:
        for s in range(n_scen):
            for k in (1, 2, 3):
                days = rng.sample(I.DAYS, k + 1)
                fills = [I.distractors(rng, 1) for _ in range(k)] + [I.distractors(rng, d)]
                cands = {"gold": " " + days[k], "orig": " " + days[0]}
                if k >= 2:
                    cands["prev"] = " " + days[k - 1]
                # U_same
                turns = [(f"My dentist appointment is on {days[0]}.", f"Okay, your dentist appointment is on {days[0]}.")]
                for i in range(1, k + 1):
                    turns += fills[i - 1]
                    turns.append((f"Actually, my dentist appointment is on {days[i]} now.",
                                  f"Got it, your dentist appointment is on {days[i]}."))
                turns += fills[k]
                out.append(dict(task=f"Usame_k{k}", d=d, cands=cands,
                                prompt=I.transcript(turns, "What day is my dentist appointment?", "Your dentist appointment is on")))
                # U_neutral
                turns = [(f"I have a dentist appointment on {days[0]}.", f"Okay, noted: {days[0]}.")]
                for i in range(1, k + 1):
                    turns += fills[i - 1]
                    turns.append((f"Actually, the appointment got moved to {days[i]}.", f"Got it, moved to {days[i]}."))
                turns += fills[k]
                out.append(dict(task=f"Uneutral_k{k}", d=d, cands=cands,
                                prompt=I.transcript(turns, "Which day do I need to go to the dentist?", "That would be")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).to(a.device).eval()
    path = os.path.join(HERE, "out", "uprobe__" + a.model.replace("/", "__") + ".jsonl")
    with open(path, "w") as f:
        for it in build():
            sc = score_item(model, tok, it["prompt"], it["cands"], a.device)
            f.write(json.dumps({"task": it["task"], "d": it["d"], "scores": {k: v[0] for k, v in sc.items()}}) + "\n")
    print("done", a.model, path)


if __name__ == "__main__":
    main()
