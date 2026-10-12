"""Score likelihood item sets for ONE model in ONE process, then exit. Run only through guard.py.

usage: run_battery.py <model_id | maxgpt3:final | maxgpt3:final_sft>
          [--sets new,old,khard] [--renders plain,chat] [--dtype fp32|bf16] [--device mps|cpu]
          [--dry N] [--tag TAG]

  new   : E001 control items (items_new.py), in every requested render
  old   : the original capacity_probe battery (items.py), plain only, identical text and seed
  khard : long-tail knowledge closed/open book (khard_items.py), plain only
Writes ../out/<slug>__<set>__<render>[__TAG].jsonl (first line = meta).
"""
import argparse, json, os, re, sys, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
sys.path.insert(0, HERE)
import items as I
import items_new as N
import khard_items as KH
import lik


def slug(m):
    return m.replace("/", "__").replace(":", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sets", default="new")
    ap.add_argument("--renders", default="plain,chat")
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dry", type=int, default=0, help="score only the first N items of each set/render")
    ap.add_argument("--tag", default="")
    ap.add_argument("--shared", action="store_true", help="batch-1 shared-prompt scorer (verified by check_shared.py)")
    a = ap.parse_args()
    t0 = time.time()
    model, tok, info = lik.load(a.model, a.dtype, a.device)
    print(f"loaded {a.model} {info} in {time.time()-t0:.1f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    suffix = ("__" + a.tag) if a.tag else ""
    for set_name in a.sets.split(","):
        renders = a.renders.split(",") if set_name == "new" else ["plain"]
        if set_name != "new" and hasattr(tok, "own_format") and "chat" in a.renders:
            renders = ["plain", "chat"]  # MaxGPT-3: old items also in its USER:/ASSISTANT: format
        for render in renders:
            if render == "chat" and not hasattr(tok, "own_format") and not getattr(tok, "chat_template", None):
                print(f"skip {set_name}/{render}: tokenizer has no chat template", flush=True)
                continue
            if set_name == "new":
                its = N.build()
            elif set_name == "old":
                its = I.build()
            elif set_name == "khard":
                its = KH.build()
            else:
                raise ValueError(set_name)
            if a.dry:
                its = its[: a.dry]
            path = os.path.join(OUT, f"{slug(a.model)}__{set_name}__{render}{suffix}.jsonl")
            n_skip = 0
            t1 = time.time()
            with open(path, "w") as f:
                f.write(json.dumps({"meta": True, "model": a.model, "set": set_name, "render": render,
                                    "n_items": len(its), "dry": a.dry, "scorer": "shared" if a.shared else "batched", **info}) + "\n")
                for k, it in enumerate(its):
                    if set_name == "new":
                        prompt, add_sp = lik.render(it, render, tok, a.model)
                    else:
                        prompt, add_sp = it["prompt"], True
                        if render == "chat":
                            prompt = re.sub(r"(?m)^User: ", "USER: ", re.sub(r"(?m)^Assistant: ", "ASSISTANT: ", prompt))
                    try:
                        scorer = lik.score_item_shared if a.shared else lik.score_item
                        sc, L = scorer(model, tok, prompt, it["cands"], a.device, add_sp)
                    except ValueError as e:  # longer than the model's context (MaxGPT-3 only)
                        if "exceeds" not in str(e):
                            raise
                        n_skip += 1
                        f.write(json.dumps({"skipped": str(e)[:120], "task": it["task"]}) + "\n")
                        continue
                    rec = {kk: it.get(kk) for kk in ("task", "cond", "d", "fam", "var", "sid", "k")}
                    rec.update(render=render, scores={lab: v[0] for lab, v in sc.items()},
                               ntok={lab: v[1] for lab, v in sc.items()},
                               split=any(v[2] == "split" for v in sc.values()), how=sorted(set(v[2] for v in sc.values())), seq_len=L)
                    f.write(json.dumps(rec) + "\n")
                    if k % 250 == 0:
                        print(f"{set_name}/{render} {k}/{len(its)} {time.time()-t1:.0f}s len={L}", flush=True)
                        if a.device == "mps":
                            torch.mps.empty_cache()
            print(f"done {set_name}/{render}: {len(its)-n_skip} scored, {n_skip} skipped, {time.time()-t1:.0f}s -> {path}", flush=True)
    print(f"ALL DONE {a.model} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
