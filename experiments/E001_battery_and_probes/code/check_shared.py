"""Equivalence check: score the same items with lik.score_item (padded batch) and
lik.score_item_shared (batch-1 shared prompt) in ONE process; report max |diff| and whether
the right/wrong decision ever differs. Run through guard.py.
usage: check_shared.py <model> [--device cpu] [--n 40]"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import items as I, items_new as N, lik, metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--device", default="cpu"); ap.add_argument("--n", type=int, default=40)
a = ap.parse_args()
model, tok, info = lik.load(a.model, "fp32", a.device)
new = N.build()
step = max(1, len(new) // a.n)
sel = [("new", "plain", it) for it in new[::step][: a.n]] + [("new", "chat", it) for it in new[step // 2::step][: a.n // 2]]
old = I.build()
sel += [("old", "plain", it) for it in old[:: max(1, len(old) // (a.n // 2))][: a.n // 2]]
maxd, flips, t_b, t_s, n = 0.0, 0, 0.0, 0.0, 0
for st, rd, it in sel:
    prompt, sp = lik.render(it, rd, tok, a.model) if st == "new" else (it["prompt"], True)
    t0 = time.time(); b, _ = lik.score_item(model, tok, prompt, it["cands"], a.device, sp); t_b += time.time() - t0
    t0 = time.time(); s, _ = lik.score_item_shared(model, tok, prompt, it["cands"], a.device, sp); t_s += time.time() - t0
    bs = {k: v[0] for k, v in b.items()}; ss = {k: v[0] for k, v in s.items()}
    maxd = max(maxd, max(abs(bs[k] - ss[k]) for k in bs))
    flips += int(M.right(bs) != M.right(ss)); n += 1
res = dict(model=a.model, device=a.device, n_items=n, max_abs_diff_nats=maxd, decision_flips=flips,
           sec_batched=round(t_b, 1), sec_shared=round(t_s, 1))
print(json.dumps(res))
json.dump(res, open(os.path.join(os.path.dirname(HERE), "logs", "check_shared.json"), "w"), indent=1)
