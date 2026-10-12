"""E004 static self-check, part B: data loaders with the REAL tokenizers (tokenizers only, no model weights) and the
likelihood scoring path with a STUB model (CPU logits from a rule; no language model).
  1 per tokenizer (the 6 E004 models): 400 training examples of seed 1 encode with labels -100 on the whole prompt
    and the answer tokens on the answer; the answer tokens decode to exactly the target; the length rejection and
    make_batch behave (shapes, padding, labels aligned); kind counts are recorded drawn vs kept
  2 per tokenizer: every E004 likelihood item (eval, dev, probe: 1,120 items) has candidates with EQUAL token
    counts, joint tokenization, and the longest prompt + candidate per family is logged (H4 d20 especially)
  3 stub model through e004_core.score_set / family_acc / probe on SmolLM2's tokenizer: a stub that prefers the
    gold's tokens is right on every item, a uniform stub (a tie) and a stub that prefers a foil are right on none
usage: <venv>/python -B selfcheck_e004_b.py   (writes ../logs/selfcheck_e004_b.txt; exit 0 only if ALL PASS)"""
import os, sys, time, types
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoTokenizer
import lik
import e004_core as C
import e004_sets as S
import train_e004 as T

LOG = os.path.join(os.path.dirname(HERE), "logs", "selfcheck_e004_b.txt")
MODELS = ["HuggingFaceTB/SmolLM2-135M-Instruct", "roneneldan/TinyStories-8M", "EleutherAI/pythia-31m",
          "roneneldan/TinyStories-3M", "EleutherAI/pythia-14m", "roneneldan/TinyStories-1M"]
N_TRAIN, MAX_LEN = 400, 768
out, fails = [], []


def check(ok, msg):
    out.append(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def label_bad(tok, ex, ids, lab, how):
    """True if the labels are not exactly: -100 on every prompt token, the token ids on every answer token."""
    pre = tok(T.prompt(ex)).input_ids
    n_ans = len(ids) - len(pre) if how == "joint" else len(lab) - lab.count(-100)
    ans = [t for t in lab if t != -100]
    return lab[:len(ids) - n_ans] != [-100] * (len(ids) - n_ans) or ans != ids[len(ids) - n_ans:]


def part_train(tok, name):
    exs = T.take(1, N_TRAIN)
    bad_lab = bad_dec = split = 0
    enc = []
    for ex in exs:
        ids, lab, how = S.encode(tok, ex, lik.cand_ids)
        ans = [t for t in lab if t != -100]
        bad_lab += label_bad(tok, ex, ids, lab, how)
        bad_dec += tok.decode(ans) != ex["answer"]
        split += how == "split"
        enc.append((ids, lab))
    ex, (ids, lab) = exs[0], enc[0]
    n_pre = lab.count(-100)
    neg = [lab[:n_pre - 1] + [ids[n_pre - 1]] + lab[n_pre:],        # last prompt token labelled
           lab[:n_pre] + [-100] + lab[n_pre + 1:],                   # first answer token unlabelled
           [-100] + lab[:-1]]                                        # labels shifted by one
    check(all(label_bad(tok, ex, ids, m, "joint") for m in neg), f"{name}: label check flags 3 of 3 wrong labellings")
    check(bad_lab == 0, f"{name}: labels = -100 on the prompt, answer ids on the answer ({bad_lab} bad of {N_TRAIN})")
    check(bad_dec == 0, f"{name}: answer tokens decode to exactly the target ({bad_dec} bad; split {split})")
    stats = S.new_stats()
    st = S.example_stream(tok, 1, MAX_LEN, stats, lik.cand_ids)
    kept = [next(st) for _ in range(N_TRAIN)]
    long_ = sum(len(i) > MAX_LEN for i, _ in enc)
    check(all(len(i) <= MAX_LEN for i, _ in kept) and stats["rejected_long"] == long_ and
          sum(stats["kept"].values()) == N_TRAIN, f"{name}: stream keeps {N_TRAIN} <= {MAX_LEN} tokens, rejected "
                                                  f"{stats['rejected_long']} (= {long_} over-long in the first {N_TRAIN} drawn)")
    # the rejection path itself, at a short limit so that it fires: kept + rejected = drawn, and the rejected
    # count equals the number of over-limit examples among the drawn ones
    st2, lim = S.new_stats(), 350
    s2 = S.example_stream(tok, 1, lim, st2, lik.cand_ids)
    kept2 = [next(s2) for _ in range(100)]
    n_drawn = sum(st2["drawn"].values())
    over = sum(len(S.encode(tok, ex, lik.cand_ids)[0]) > lim for ex in T.take(1, n_drawn))
    check(st2["rejected_long"] > 0 and st2["rejected_long"] == over and all(len(i) <= lim for i, _ in kept2)
          and n_drawn == 100 + over, f"{name}: rejection at {lim} tokens: drawn {n_drawn}, rejected "
                                     f"{st2['rejected_long']} (= {over} over-limit), kept all <= {lim}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids, lab, att = C.make_batch(kept[:4], pad)
    ok = ids.shape == lab.shape == att.shape and all(
        int(att[i].sum()) == len(kept[i][0]) and lab[i, :len(kept[i][1])].tolist() == kept[i][1] and
        (lab[i, len(kept[i][1]):] == -100).all() for i in range(4))
    ids2, _, _ = C.make_batch(kept[:4], pad, pad_to=MAX_LEN)
    check(ok and ids2.shape[1] == MAX_LEN, f"{name}: make_batch shapes, padding and label alignment")


def part_items(tok, name):
    unequal, notjoint, n = [], 0, 0
    longest, lens_f = defaultdict(int), defaultdict(list)
    for draw in ("eval", "dev", "probe"):
        for it in S.e004_lik(draw):
            prompt, add_sp = lik.render(it, "plain", tok, name)
            lens = {}
            for lab, c in it["cands"].items():
                pre, cid, how = lik.cand_ids(tok, prompt, c, add_sp)
                lens[lab] = len(cid)
                notjoint += how != "joint"
                longest[it["family"]] = max(longest[it["family"]], len(pre) + len(cid))
            if draw == "eval":
                lens_f[it["family"]].append(len(pre) + max(lens.values()))
            n += 1
            if len(set(lens.values())) != 1:
                unequal.append((draw, it["family"], it["idx"], it["cand_vals"], lens))
    check(not unequal, f"{name}: equal candidate token counts on {n - len(unequal)}/{n} items {unequal[:2]}")
    it = S.e004_lik("eval", 1, ["H1"])[0]                            # negative control: a 3+ token foil
    prompt, add_sp = lik.render(it, "plain", tok, name)
    ctl = {len(lik.cand_ids(tok, prompt, c, add_sp)[1]) for c in (it["cands"]["gold"], " Kalamazooville")}
    check(len(ctl) == 2, f"{name}: the count comparison separates a 1-token gold from a long foil {sorted(ctl)}")
    check(notjoint == 0, f"{name}: joint tokenization for every candidate ({notjoint} split)")
    out.append(f"INFO {name}: longest prompt+candidate tokens per family " +
               " ".join(f"{f}={v}" for f, v in sorted(longest.items())))
    out.append(f"INFO {name}: eval draw, share of items over 512 / over 768 tokens per family " +
               " ".join(f"{f}={sum(x > 512 for x in v) / len(v):.2f}/{sum(x > 768 for x in v) / len(v):.2f}"
                        for f, v in sorted(lens_f.items())))


class Stub:
    """logits from a rule: 'gold' prefers the gold's token ids, 'foil' prefers c1's, 'flat' is uniform."""

    def __init__(self, vocab, mode):
        self.V, self.mode, self.pref = vocab, mode, []

    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, input_ids=None, attention_mask=None, logits_to_keep=None, **k):
        B, L = input_ids.shape
        keep = logits_to_keep or L
        lg = torch.zeros((B, keep, self.V))
        for t in self.pref:
            lg[:, :, t] = 5.0
        return types.SimpleNamespace(logits=lg)


def part_stub(tok, name):
    its = [it for fam_its in S.draw_items("eval").values() for it in fam_its[:6]]
    items = [S.lik_item(it) for it in its]
    res = {}
    for mode in ("gold", "flat", "foil"):
        right = 0
        for it in items:
            stub = Stub(len(tok), mode)
            if mode != "flat":
                prompt, add_sp = lik.render(it, "plain", tok, name)
                lab = "gold" if mode == "gold" else "c1"
                stub.pref = lik.cand_ids(tok, prompt, it["cands"][lab], add_sp)[1][-1:]
            recs, _ = C.score_set(stub, tok, name, "e004", "plain", [it], "cpu")
            right += recs[0]["right"]
        res[mode] = right
    n = len(items)
    check(res == {"gold": n, "flat": 0, "foil": 0}, f"{name}: stub scoring right on gold-preferring {res['gold']}/{n}, "
                                                    f"uniform {res['flat']}/{n}, foil-preferring {res['foil']}/{n}")
    acc = C.probe(Stub(len(tok), "flat"), tok, name, S.probe_items(True), "cpu")
    check(sorted(acc) == sorted(S.E.PASS_FAMILIES) and all(v == 0.0 for v in acc.values()),
          f"{name}: probe returns the 9 pass-rule families (uniform stub: all 0.0)")
    traj = [{"step": 0, **{f: 0.5 for f in acc}}] + [{"step": s, **{f: 0.9 for f in acc}} for s in (50, 100)]
    check(C.lock_in(traj) == 50 and C.lock_in(traj[:1]) is None, "lock-in step reads the min over the 9 families")


def main():
    t0 = time.time()
    for m in MODELS:
        tok = AutoTokenizer.from_pretrained(m)
        out.append(f"-- {m} ({type(tok).__name__}, vocab {len(tok)})")
        part_train(tok, m)
        part_items(tok, m)
        if m == MODELS[0]:
            part_stub(tok, m)
    out.append(f"{len(fails)} failures in {time.time()-t0:.0f}s; {'ALL PASS' if not fails else 'FAIL'}")
    with open(LOG, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
