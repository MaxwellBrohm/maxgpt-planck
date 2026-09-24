"""Experiment E2: recall vs distance with a FIXED, model-independent history.

In the main battery every model sees its own replies, so context length and
fact re-mentions differ per model. Here every model gets the identical history:
  user fact  ->  canned ack (never repeats the fact)
  N x (distractor question -> canned reply)
  final question
with N in {0,1,2,4,8,12} and canned replies either SHORT (~15 words, hand-written)
or LONG (~100 words, fixed text). Separates turn distance from token distance.
Grades the free answer (same strict grader as the battery) and runs the
forced-prefix probe (does greedy decoding from "Your name is" emit the gold?).

usage: python exp_fixed_history.py <model_id> [--plain] [--memsys] [--cpu]   (one model per process)
--memsys adds a system prompt telling the model the user's earlier messages are part of the
conversation (E3: is the deflection steerable by one instruction, i.e. not a capacity limit?).
--plain renders a plain "User:/Assistant:" transcript instead of the chat template
(used for the base-vs-instruct comparison, run on both the base and instruct model).
"""
import os, sys, json, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery as B
from run_model import build_prompt, seq_logprob, slug

HERE = os.path.dirname(os.path.abspath(__file__))

FACTS = [  # (id, fact, question, gold regex, forced prefix, gold, foil)
    ("name", "Hi! My name is Priya, and I'm trying to get more organized this year.",
     "Quick check: what did I tell you my name was?", r"\bpriya\b", "Your name is", " Priya", " Maria"),
    ("number", "I just moved into apartment 417 in a new building downtown.",
     "What's my apartment number again?", B.num_pat(417), "Your apartment number is", " 417", " 714"),
    ("pet", "I have a cat named Biscuit who loves sitting in cardboard boxes.",
     "What's my cat called?", r"\bbiscuit", "Your cat is called", " Biscuit", " Whiskers"),
    ("city", "I live in Tucson and I work at a small bakery.",
     "Which city do I live in?", r"\btucson\b", "You live in", " Tucson", " Phoenix"),
]
ACK = "Thanks for sharing that! How can I help you today?"
SHORT = [
    "Give them bright indirect light and water only when the top inch of soil is dry.",
    "Keep a consistent bedtime and avoid screens for an hour before sleep.",
    "Sunlight scatters off air molecules, and blue light scatters the most.",
    "Try oatmeal with fruit, Greek yogurt with nuts, or eggs on whole-grain toast.",
    "Use a coarser grind, cooler water, and a shorter brew time.",
    "Weather is day-to-day conditions; climate is the long-term average.",
    "A noun is a word for a person, place, thing, or idea.",
    "Work in short timed blocks, put your phone away, and take regular breaks.",
    "Build a blanket fort, bake cookies, or start a jigsaw puzzle.",
    "Chlorophyll breaks down in autumn, revealing yellow and orange pigments.",
    "Fold it with a sharp nose, keep the wings level, and throw it gently.",
    "A librarian organizes books and helps people find information.",
]
LONG_PAD = (" To expand on that a little: small, consistent habits usually matter more than any single big change, "
            "so it helps to start with one simple step, notice what works for you, and adjust over time. "
            "Different people find different approaches useful, and there is rarely one perfect answer. "
            "If something does not seem to help after a week or two, try a different variation, and keep "
            "whatever makes the routine easier to stick with. Let me know if you would like more detail on any part of this.")
LONG = [s + LONG_PAD for s in SHORT]
DISTANCES = [0, 1, 2, 4, 8, 12]


MEMSYS = ("You are a helpful assistant chatting with a user. Everything the user said earlier in this "
          "conversation is visible to you; when the user asks about something they told you, answer from the conversation.")
PLAIN_HEAD = "The following is a conversation between a user and an AI assistant.\n\n"


def plain_prompt(msgs):
    s = PLAIN_HEAD
    for m in msgs:
        s += ("User: " if m["role"] == "user" else "Assistant: ") + m["content"] + "\n"
    return s + "Assistant:"


def main():
    mid = sys.argv[1]
    plain = "--plain" in sys.argv
    memsys = "--memsys" in sys.argv
    t0 = time.time()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if "--cpu" in sys.argv:
        dev = "cpu"
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(dev).eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    pad = model.generation_config.pad_token_id if model.generation_config.pad_token_id is not None else eos[0]
    os.makedirs(os.path.join(HERE, "exp"), exist_ok=True)
    out = open(os.path.join(HERE, "exp", f"fixed_history__{slug(mid)}{'__plain' if plain else ''}{'__memsys' if memsys else ''}.jsonl"), "w")
    styles = (("short", SHORT),) if memsys else (("short", SHORT), ("long", LONG))
    for style, replies in styles:
        for n in DISTANCES:
            for fid, fact, q, gold, prefix, g, foil in FACTS:
                msgs = ([{"role": "system", "content": MEMSYS}] if memsys else []) + \
                       [{"role": "user", "content": fact}, {"role": "assistant", "content": ACK}]
                for i in range(n):
                    msgs += [{"role": "user", "content": B.D[i]}, {"role": "assistant", "content": replies[i]}]
                msgs.append({"role": "user", "content": q})
                prompt = plain_prompt(msgs) if plain else build_prompt(tok, mid, msgs)
                ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
                gcfg = GenerationConfig(do_sample=False, max_new_tokens=60, eos_token_id=eos, pad_token_id=pad, repetition_penalty=1.0)
                try:
                    gen = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=gcfg)[0, ids.shape[1]:].tolist()
                except RuntimeError as e:  # MPS allocator cap hit: finish on CPU
                    if dev != "mps":
                        raise
                    print(f"MPS error ({str(e)[:80]}); switching to CPU", flush=True)
                    torch.mps.empty_cache(); dev = "cpu"; model.to("cpu"); ids = ids.to("cpu")
                    gen = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=gcfg)[0, ids.shape[1]:].tolist()
                if gen and gen[-1] in eos:
                    gen = gen[:-1]
                ans = tok.decode(gen, skip_special_tokens=True).strip()
                if plain:  # a base model keeps writing the transcript; cut at the next speaker
                    ans = ans.split("\nUser")[0].split("User:")[0].strip()
                fp = prompt + (" " + prefix if plain else prefix)
                lg, rg = seq_logprob(model, tok, fp, g, dev)
                lf, rf = seq_logprob(model, tok, fp, foil, dev)
                rec = dict(model=mid, format=("plain" if plain else "chat_template"), memsys=memsys, style=style, n_distractors=n, fact=fid, prompt_tokens=int(ids.shape[1]), answer=ans,
                           strict=B.mentions(gold, ans) and not B.deflects(ans), lenient=B.mentions(gold, ans),
                           deflect=B.deflects(ans), lp_gold=lg, lp_foil=lf, ranks_gold=rg,
                           greedy_gold=(None if rg is None else all(x == 1 for x in rg)),
                           margin=(None if lg is None or lf is None else lg - lf))
                out.write(json.dumps(rec) + "\n"); out.flush()
                if dev == "mps":
                    torch.mps.empty_cache()
            print(f"[{time.time()-t0:5.0f}s] {style} n={n} done", flush=True)
    print(f"done {mid} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    # guard: no gold answer may appear in any canned reply
    for fid, fact, q, gold, *_ in FACTS:
        for r in SHORT + LONG + [ACK] + B.D:
            assert not B.mentions(gold, r), (fid, r)
    main()
