"""E005 change (c) test: the training renders, their loss mask and the loss self-check, with SmolLM2's REAL tokenizer
(no model weights) and a STUB model on the CPU (an embedding and a linear head; no language model).
  1 encode, seeds 0-5 x 400 examples: plain = E004's labels (decode to " answer\\n", no end-of-turn); chat = the
    eval's chat prompt (== lik.chat_prompt and gen_run's template call) with labels -100 on all of it and the ids
    on the sentence + <|im_end|> only; the input ends at that <|im_end|> (the template's "\\n" is not there)
  2 the loss mask by gradients: through e004_core.answer_loss on the stub, the positions that receive gradient
    predict exactly the sentence tokens + <|im_end|> (chat) or the answer tokens (plain), nothing else
  3 e005_train.loss_selfcheck on the stub: the first / plain / chat batches match the HF-style loss; a stub whose
    HF loss also covers the prompt is caught on each batch; with dry, both wrong-loss mutants differ
  4 the stream: both renders kept, the rejection path, stream_summary, check_batches (the training iterator moves
    by exactly bs examples; the plain and chat check batches hold only their render)
  5 mutants: 7 wrong label layouts and 6 wrong encoders must each be flagged by checks 1-2
usage: <venv>/python -B test_render_e005.py   (writes ../logs/test_render_e005.txt; exit 0 only if ALL PASS)"""
import os, sys, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import lik
import gen_run as R
import e004_core as C
import e005_sets as S
import e005_train as TR
import train_e005 as T5
import train_e004 as T4

LOG = os.path.join(os.path.dirname(HERE), "logs", "test_render_e005.txt")
SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
out, fails = [], []


def check(ok, msg):
    out.append(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def problems(tok, ex, ids, lab):
    """-> list of problems with one encoded example (empty = right)."""
    p, eot = [], S.eot_id(tok)
    if len(ids) != len(lab):
        return ["ids and labels differ in length"]
    if ex["render"] == "chat":
        prompt = S.chat_prompt(tok, ex)
        if prompt != lik.chat_prompt(tok, SMOL, ex["turns"], ex["question"], ""):
            p.append("chat prompt differs from the eval's lik.chat_prompt")
        if prompt != tok.apply_chat_template(S.chat_msgs(ex), tokenize=False, add_generation_prompt=True):
            p.append("chat prompt differs from gen_run's template call")
        pre, want = tok(prompt, add_special_tokens=False).input_ids, S.sentence(ex) + S.EOT
        if ids[-1] != eot or lab[-1] != eot:
            p.append("the input does not end at a labelled <|im_end|>")
    else:
        pre, want = tok(T4.prompt(ex)).input_ids, ex["answer"]
        if S.eot_id(tok) in [t for t in lab if t != -100]:
            p.append("an end-of-turn token is labelled in a plain example")
    n = len(pre)
    if ids[:n] != pre:
        p.append("the input does not start with the prompt tokens")
    if lab[:n] != [-100] * n:
        p.append("a prompt position is labelled")
    if lab[n:] != ids[n:] or len(ids) == n:
        p.append("an answer position is not labelled with its own token")
    if tok.decode(ids[n:]) != want:
        p.append(f"answer tokens decode to {tok.decode(ids[n:])!r}, not {want!r}")
    return p


class Stub(torch.nn.Module):
    """embedding + linear head; HF-style loss (shifted, -100 ignored, mean); keeps the hidden states for grads.
    prompt_loss=True: a WRONG HF loss that also scores every attended position (the self-check must catch it)."""

    def __init__(self, V, prompt_loss=False):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = torch.nn.Embedding(V, 8)
        self.head = torch.nn.Linear(8, V, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(V, 8, generator=g))
            self.head.weight.copy_(torch.randn(V, 8, generator=g))
        self.prompt_loss, self.h, self.config = prompt_loss, None, types.SimpleNamespace(use_cache=False)

    def base_model(self, input_ids=None, attention_mask=None):
        self.h = self.emb(input_ids)
        if self.h.requires_grad:
            self.h.retain_grad()
        return types.SimpleNamespace(last_hidden_state=self.h)

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        logits = self.head(self.base_model(input_ids=input_ids).last_hidden_state)
        lab = labels.clone()
        if self.prompt_loss:
            lab = input_ids.clone()
            lab[attention_mask == 0] = -100
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
        return types.SimpleNamespace(loss=loss, logits=logits)


def trained_targets(stub, ids, lab):
    """token ids the answer-only loss trains, read from which positions get a gradient (position p -> ids[p + 1])."""
    I, L, A = C.make_batch([(ids, lab)], 0)
    stub.zero_grad()
    C.answer_loss(stub, I, A, L).backward()
    g = stub.h.grad[0].abs().sum(-1)
    return [ids[p + 1] for p in range(len(ids) - 1) if g[p] > 0]


def part_encode(tok):
    bad, nbad, n, how = [], 0, {"plain": 0, "chat": 0}, {"joint": 0, "split": 0}
    for seed in range(6):
        for ex in T5.take(seed, 400):
            ids, lab, h = S.encode(tok, ex, lik.cand_ids)
            n[ex["render"]] += 1
            how[h] += 1
            pr = problems(tok, ex, ids, lab)
            nbad += bool(pr)
            if pr and len(bad) < 3:
                bad.append((seed, ex["render"], ex["block"], pr))
    check(nbad == 0, f"labels right on {2400 - nbad}/2400 examples of seeds 0-5 (plain {n['plain']}, chat {n['chat']}, "
                     f"tokenization {how}) {bad}")
    check(min(n.values()) > 1000 and how["split"] == 0, "both renders present; every chat target tokenized jointly")
    exs = [e for e in T5.take(3, 60)]
    ok = all(tok.decode(S.encode(tok, e, lik.cand_ids)[0]).endswith(S.EOT) and
             not tok.decode(S.encode(tok, e, lik.cand_ids)[0]).endswith(S.EOT + "\n") for e in exs if e["render"] == "chat")
    check(ok, "every chat input ends at <|im_end|>; the template's trailing newline is not in the input")
    check(S.eot_id(tok) in R.end_ids_of(tok), "the trained end-of-turn token is one that gen_run stops on")


def part_grad(tok, stub):
    bad, n = 0, 0
    for ex in T5.take(1, 40):
        ids, lab, _ = S.encode(tok, ex, lik.cand_ids)
        want = S.sentence(ex) + S.EOT if ex["render"] == "chat" else ex["answer"]
        got = tok.decode(trained_targets(stub, ids, lab))
        bad += got != want
        n += 1
    check(bad == 0, f"loss mask by gradients: trained tokens decode to exactly the target on {n - bad}/{n} examples")


def part_selfcheck(tok):
    a = types.SimpleNamespace(bs=2, seed=1, max_len=768)
    st = S.new_stats()
    stream = S.example_stream(tok, 1, 768, st, lik.cand_ids)
    batches = TR.check_batches(tok, a, stream)
    fresh = S.example_stream(tok, 1, 768, S.new_stats(), lik.cand_ids)
    first = [next(fresh) for _ in range(3)]
    nxt = next(stream)
    check(nxt[0] == first[2][0] and [x for x, _ in batches["first"]] == [first[0][0], first[1][0]],
          "check_batches consumes exactly bs examples of the training iterator (E004's first batch)")
    start = tok.convert_tokens_to_ids("<|im_start|>")
    rend = {r: ["chat" if x[0] == start else "plain" for x, _ in batches[r]] for r in ("plain", "chat")}
    check(len(batches["plain"]) == 2 and len(batches["chat"]) == 2 and set(rend["plain"]) == {"plain"}
          and set(rend["chat"]) == {"chat"}, f"plain and chat check batches hold only their render {rend}")
    meta, sf = {}, []
    good = TR.loss_selfcheck(Stub(len(tok)), batches, 0, "cpu", True, meta, sf)
    check(good == [] and not sf and set(meta["loss_selfcheck"]) == {"first", "plain", "chat"},
          f"self-check passes on first/plain/chat with a correct HF loss; mutants differ {sf}")
    check(meta["loss_selfcheck"]["chat"]["n_labelled"] > 0 and meta["loss_selfcheck"]["plain"]["n_labelled"] > 0,
          "the plain and chat check batches have labelled tokens")
    wrong = TR.loss_selfcheck(Stub(len(tok), prompt_loss=True), batches, 0, "cpu", False, {}, [])
    check(wrong == ["first", "plain", "chat"], f"a prompt-scoring HF loss is caught on every batch ({wrong})")


def part_stream(tok):
    st = S.new_stats()
    it = S.example_stream(tok, 1, 768, st, lik.cand_ids)
    kept = [next(it) for _ in range(600)]
    rs = {r: sum(k[2] == r for k in kept) for r in S.RENDERS}
    summ = S.stream_summary(st)
    check(all(len(x) <= 768 for x, _, _ in kept) and min(rs.values()) > 200 and summ["n"] == 600,
          f"stream keeps <= 768 tokens, both renders {rs}")
    check(all(abs(sum(v["drawn"] for v in summ["shares"][k].values()) - 1) < 1e-3 for k in S.STAT_KEYS),
          "stream_summary shares sum to 1 per key")
    st2, lim = S.new_stats(), 350
    src = T5.take(1, 2000)
    s2 = S.example_stream(tok, 1, lim, st2, lik.cand_ids, source=iter(src))
    kept2 = [next(s2) for _ in range(100)]
    nd = st2["n"] + st2["rejected_long"]
    over = sum(len(S.encode(tok, e, lik.cand_ids)[0]) > lim for e in src[:nd])
    check(st2["rejected_long"] > 0 and st2["rejected_long"] == over and all(len(x) <= lim for x, _, _ in kept2),
          f"rejection at {lim} tokens: drawn {nd}, rejected {st2['rejected_long']} (= {over} over-limit), kept <= {lim}")


def enc_mutants(tok):
    """wrong encoders; each must be flagged by problems() or by the gradient check on some example."""
    eot = S.eot_id(tok)

    def m_no_eot(ex, c):
        ids, lab, h = S.encode(tok, ex, c)
        return (ids[:-1], lab[:-1]) if ex["render"] == "chat" else (ids, lab)

    def m_newline(ex, c):
        ids, lab, h = S.encode(tok, ex, c)
        nl = tok("\n", add_special_tokens=False).input_ids
        return (ids + nl, lab + nl) if ex["render"] == "chat" else (ids, lab)

    def m_space(ex, c):
        if ex["render"] != "chat":
            return S.encode(tok, ex, c)[:2]
        pre = tok(S.chat_prompt(tok, ex), add_special_tokens=False).input_ids
        ans = tok(" " + S.sentence(ex), add_special_tokens=False).input_ids + [eot]
        return pre + ans, [-100] * len(pre) + ans

    def m_all_plain(ex, c):
        return S.S4.encode(tok, ex, c)[:2]

    def m_no_genprompt(ex, c):
        if ex["render"] != "chat":
            return S.encode(tok, ex, c)[:2]
        pre = tok(tok.apply_chat_template(S.chat_msgs(ex), tokenize=False), add_special_tokens=False).input_ids
        ans = tok(S.sentence(ex), add_special_tokens=False).input_ids + [eot]
        return pre + ans, [-100] * len(pre) + ans

    def m_prompt_labelled(ex, c):
        ids, lab, h = S.encode(tok, ex, c)
        return ids, list(ids)

    return dict(no_eot=m_no_eot, trailing_newline=m_newline, leading_space=m_space, all_plain=m_all_plain,
                no_generation_prompt=m_no_genprompt, prompt_labelled=m_prompt_labelled)


def part_mutants(tok, stub):
    exs = T5.take(2, 30)
    chat = next(e for e in exs if e["render"] == "chat")
    ids, lab, _ = S.encode(tok, chat, lik.cand_ids)
    n = lab.count(-100)
    lays = {"eot_unlabelled": lab[:-1] + [-100], "last_prompt_labelled": lab[:n - 1] + [ids[n - 1]] + lab[n:],
            "first_answer_unlabelled": lab[:n] + [-100] + lab[n + 1:], "shifted": [-100] + lab[:-1],
            "all_labelled": list(ids), "none_labelled": [-100] * len(ids), "eot_only": [-100] * (len(ids) - 1) + [ids[-1]]}
    want = S.sentence(chat) + S.EOT
    caught = {k: bool(problems(tok, chat, ids, m)) or tok.decode(trained_targets(stub, ids, m)) != want
              for k, m in lays.items()}
    check(all(caught.values()), f"label-layout mutants flagged: {sum(caught.values())}/{len(caught)} {caught}")
    res = {}
    for name, fn in enc_mutants(tok).items():
        res[name] = any(problems(tok, e, *fn(e, lik.cand_ids)) for e in exs)
    check(all(res.values()), f"encoder mutants flagged: {sum(res.values())}/{len(res)} {res}")


def main():
    tok = AutoTokenizer.from_pretrained(SMOL)
    stub = Stub(len(tok))
    out.append(f"-- {SMOL}: eos {tok.eos_token!r} id {tok.eos_token_id}, <|im_end|> id {S.eot_id(tok)}, pad {tok.pad_token_id}")
    for part in (part_encode, lambda t: part_grad(t, stub), part_selfcheck, part_stream, lambda t: part_mutants(t, stub)):
        part(tok)
    out.append(f"{len(fails)} failures; {'ALL PASS' if not fails else 'FAIL'}")
    with open(LOG, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
