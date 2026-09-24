"""Reproducibility check: the untouched-model baseline of E002 must reproduce E001's scores for the same
model on the same E001 items (same builder, same seed, same renders). Prints max |diff| and decision flips.
usage: python check_repro.py HuggingFaceTB/SmolLM2-135M-Instruct
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
E001 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "E001_battery_and_probes", "out")
E002 = os.path.join(os.path.dirname(HERE), "out")
sys.path.insert(0, HERE)
import metrics_ft as MF


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main(model):
    slug = model.replace("/", "__")
    out = {}
    for render in ("plain", "chat"):
        a = [r for r in load(os.path.join(E001, f"{slug}__new__{render}.jsonl")) if not r.get("meta")]
        b = load(os.path.join(E002, f"{slug}__base__new__{render}.jsonl"))
        assert len(a) == len(b), (len(a), len(b))
        mx, flips = 0.0, 0
        for x, y in zip(a, b):
            assert (x["var"], x["sid"], x["d"]) == (y["var"], y["sid"], y["d"])
            for k in x["scores"]:
                mx = max(mx, abs(x["scores"][k] - y["scores"][k]))
            flips += MF.right(x["scores"]) != MF.right(y["scores"])
        out[render] = {"n": len(a), "max_abs_diff": mx, "decision_flips": flips}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    main(sys.argv[1])
