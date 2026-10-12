"""E004 mutation test, part B: the free-generation loop (gen_run.generate / find_stop) driven by a SCRIPTED STUB
(a fixed token sequence; no language model, no weights, CPU only) and a stub tokenizer with a 15-string vocab.
Checks the stop rules (newline, "User:", end of turn, the 48-token cap, leading newline = empty reply, chat
render ignores newlines) and that run_gen() grades the reply with gen_grade (a correct natural answer passes,
a capped loop fails). Mutants of the loop must each be killed (a crash is not a kill).
usage: <venv>/python -B mutation_e004_b.py   (needs torch; writes ../logs/mutation_e004_b.txt)"""
import contextlib, os, sys, tempfile, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import gen_run as R

LOG = os.path.join(os.path.dirname(HERE), "logs", "mutation_e004_b.txt")
VOCAB = ["<p>", "<eos>", "<|im_end|>", " Your", " lesson", " is", " on", " Friday", ".", "\n", " User", ":",
         ".\n", " More", " a"]
IDX = {s: i for i, s in enumerate(VOCAB)}
EOS, IM_END = IDX["<eos>"], IDX["<|im_end|>"]


class StubTok:
    eos_token_id = EOS

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        return types.SimpleNamespace(input_ids=torch.zeros((1, 3), dtype=torch.long))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(VOCAB[i] for i in ids if not (skip_special_tokens and VOCAB[i].startswith("<")))

    def get_vocab(self):
        return dict(IDX)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "chat prompt"


class StubModel:
    """returns logits whose argmax is script[t] at call t; past_key_values counts calls."""

    def __init__(self, script):
        self.script = [IDX[s] for s in script]

    def __call__(self, input_ids=None, past_key_values=None, use_cache=True):
        t = past_key_values or 0
        tok = self.script[t] if t < len(self.script) else IDX[" a"]
        logits = torch.zeros((1, input_ids.shape[1], len(VOCAB)))
        logits[0, -1, tok] = 1.0
        return types.SimpleNamespace(logits=logits, past_key_values=t + 1)


W = [" Your", " lesson", " is", " on", " Friday", "."]
CASES = [  # (name, script, newline_stop, expected (reply, stop, n_new))
    ("newline stop", W + ["\n", " More"], True, ("Your lesson is on Friday.", "newline", 7)),
    ("merged '.\\n' token", W[:-1] + [".\n", " More"], True, ("Your lesson is on Friday.", "newline", 6)),
    ("User: stop", W + [" User", ":", " More"], True, ("Your lesson is on Friday.", "user", 8)),
    ("end-of-turn stop", W + ["<|im_end|>", " More"], True, ("Your lesson is on Friday.", "eos", 6)),
    ("eos stop", W + ["<eos>", " More"], True, ("Your lesson is on Friday.", "eos", 6)),
    ("leading newline = empty reply", ["\n"] + W + ["\n"], True, ("", "newline", 1)),
    ("cap at 48", [" Friday"] * 60, True, (" ".join(["Friday"] * 48), "cap", 48)),
    ("newline as the 48th token", [" a"] * 47 + ["\n"], True, (" ".join(["a"] * 47), "newline", 48)),
    ("chat render keeps newlines", W + ["\n", " More", "<|im_end|>"], False,
     ("Your lesson is on Friday.\n More", "eos", 8)),
]


def run_cases():
    bad = []
    for name, script, nl, want in CASES:
        got = R.generate(StubModel(script), StubTok(), "prompt", "cpu", R.MAX_NEW, True, R.end_ids_of(StubTok()), nl)
        if got != want:
            bad.append((name, want, got))
    return bad


def run_gen_check():
    """run_gen end to end on two stub items: a correct natural answer passes, a capped loop fails."""
    items = [dict(set="t", prompt="p", gold="Friday", pool=["Monday", "Friday"], obj={"lesson"}, family="H1"),
             dict(set="t", prompt="p", gold="Friday", pool=["Monday", "Friday"], obj={"lesson"}, family="H1")]
    scripts = iter([W + ["\n"], [" Friday"] * 60])
    real = R.generate
    with tempfile.TemporaryDirectory() as d:
        def gen(model, *a, **k):
            return real(StubModel(next(scripts)), *a, **k)
        with patched(R, generate=gen):
            recs = R.run_gen(None, StubTok(), items, "plain", "cpu", os.path.join(d, "g.jsonl"))
    want = [(True, []), (False, [1, 2])]
    got = [(r["strict"], r["fails"]) for r in recs]
    return [] if got == want else [("run_gen", want, got)]


@contextlib.contextmanager
def patched(mod, **attrs):
    old = {k: getattr(mod, k) for k in attrs}
    for k, v in attrs.items():
        setattr(mod, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(mod, k, v)


_real_find, _real_gen, _real_end = R.find_stop, R.generate, R.end_ids_of


def _skip_leading_nl(text, newline_stop=True):
    """E002 gen_probe: a newline before any text is kept and generation goes on to the next newline."""
    body = text.lstrip()
    return _real_find(body, newline_stop) if body else (None, None)


def _cap_as_newline(*a, **k):
    r = _real_gen(*a, **k)
    return (r[0], "newline", r[2]) if r[1] == "cap" else r


MUTANTS = [
    ("no 'User:' stop", dict(STOP_TEXT={"newline": "\n"})),
    ("no newline stop", dict(STOP_TEXT={"user": "User:"})),
    ("newline stop ignores the render", dict(find_stop=lambda t, nl=True: _real_find(t, True))),
    ("leading newline skipped (E002)", dict(find_stop=_skip_leading_nl)),
    ("cap reported as a stop", dict(generate=_cap_as_newline)),
    ("end-of-turn / eos ignored", dict(end_ids_of=lambda tok: ())),
    ("only eos, not <|im_end|>", dict(end_ids_of=lambda tok: (tok.eos_token_id,))),
    ("cap at 47", dict(MAX_NEW=47)),
    ("cap at 49", dict(MAX_NEW=49)),
]


def main():
    lines, ok = [], True
    base, bg = run_cases(), run_gen_check()
    lines.append(f"baseline generate(): {len(CASES)} scripted cases, mismatches {len(base)}; run_gen + grader: "
                 f"mismatches {len(bg)}")
    for b in base + bg:
        lines.append(f"  BASELINE MISMATCH {b}")
    ok &= not base and not bg
    killed = 0
    for name, attrs in MUTANTS:
        try:
            with patched(R, **attrs):
                bad = run_cases()
            k, detail = bool(bad), "; ".join(f"{b[0]}: want {b[1]!r} got {b[2]!r}" for b in bad[:1])
        except Exception as e:  # a crash is not a kill
            k, detail = False, f"CRASH {e!r}"
        killed += k
        ok &= k
        lines.append(f"  {'KILLED' if k else 'SURVIVED':8s} generate | {name:34s} | {detail[:160]}")
    lines.append(f"mutants killed {killed}/{len(MUTANTS)} (none by a crash); {'ALL PASS' if ok else 'FAIL'}")
    with open(LOG, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
