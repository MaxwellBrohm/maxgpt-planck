"""Decisive test: are the in-context failures at 135M capacity-bound, or trainable?

Fine-tune a small chat model for a few hundred steps on programmatically generated
dialogues that exercise updates, same-type binding, two-hop reference and speaker
perspective, using templates, names, events, jobs and distractors that are DISJOINT
from the eval battery in items.py / uprobe.py. Then re-score the eval battery.
If held-out-template accuracy jumps to near ceiling, the failure was a training-signal
gap, not a parameter limit at that size.

usage: python ft_test.py <hf_model_id> [--steps 400] [--lr 5e-5] [--bs 16]
Writes out/ft__<slug>.jsonl (same format as run_capacity) and out/ftu__<slug>.jsonl (uprobe format).
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import argparse, json, random, sys, time, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import items as I
import uprobe as UP
import khard as KH
from run_capacity import score_item

# ---------------- disjoint training vocabulary ----------------
T_USER = ["Aiden", "Bella", "Carlos", "Dana", "Emil", "Farah", "Gavin", "Hiro", "Isla", "Jonas",
          "Keira", "Liam", "Maya", "Nico", "Opal", "Pedro"]
T_OTHER = ["Quentin", "Rosa", "Stefan", "Tessa", "Ulrich", "Vera", "Wes", "Xenia", "Yosef", "Zelda",
           "Anton", "Brigid", "Cyrus", "Delia", "Enzo", "Flora"]
T_ASSIST = ["Kit", "Sky", "Rowan", "Blake", "Avery", "Sage", "Drew", "Parker"]
T_PETS = ["Rocket", "Bubbles", "Shadow", "Peanut", "Luna", "Bruno", "Coco", "Ginger",
          "Hazel", "Jasper", "Maple", "Oreo", "Poppy", "Rusty", "Scout", "Teddy"]
T_JOBS = ["doctor", "writer", "singer", "mechanic", "scientist", "photographer", "florist", "janitor",
          "pharmacist", "gardener", "firefighter", "journalist", "veterinarian", "welder", "potter", "courier"]
T_EVENTS = ["haircut", "flight", "team meeting", "piano lesson", "car service", "yoga class", "job interview", "vet visit"]
T_OWN = [("dog", "brother"), ("hamster", "neighbor"), ("parrot", "cousin"), ("rabbit", "friend")]
T_REL = [("brother", "uncle"), ("cousin", "friend"), ("neighbor", "boss"), ("roommate", "coach")]
T_DISTRACT = [
    ("Any tips for a long road trip?", "Plan rest stops every two hours, pack snacks, and download music in case the signal drops."),
    ("How do I clean a cast iron pan?", "Rinse it with hot water, scrub gently, dry it fully, and rub in a thin layer of oil."),
    ("What's a good stretch after running?", "A standing quad stretch and a calf stretch against a wall, each held for about thirty seconds."),
    ("How can I remember names better?", "Repeat the name back when you hear it and link it to something about the person."),
    ("Why do cats purr?", "Cats purr when relaxed, but also to soothe themselves when stressed or hurt."),
    ("What should I pack for a picnic?", "Sandwiches, fruit, water, a blanket, napkins and a bag for trash."),
    ("How do I get better at chess?", "Study simple endgames, review your lost games, and solve a few tactics puzzles daily."),
    ("What's the best way to learn to type faster?", "Use all ten fingers, keep your eyes on the screen, and practice in short daily sessions."),
    ("How do I keep herbs fresh?", "Trim the stems and keep soft herbs in a glass of water in the fridge, loosely covered."),
    ("Why is the moon sometimes visible during the day?", "It is bright enough to see whenever it is above the horizon and far enough from the Sun."),
    ("How do I stop procrastinating?", "Break the task into a five-minute first step and start it right away."),
    ("What makes bread rise?", "Yeast eats sugars and releases gas, which gets trapped in the stretchy gluten network."),
    ("How often should I replace a toothbrush?", "About every three months, or sooner if the bristles look frayed."),
    ("What's a simple way to meditate?", "Sit comfortably, breathe slowly, and gently bring your attention back to your breath."),
]
UPD = ["Change of plans: my {e} is now on {d}.", "Actually, make that {d} for the {e}.",
       "Scratch that, my {e} was moved to {d}.", "Update: the {e} switched to {d}."]
UPD_ACK = ["Okay, {d} it is.", "Noted, {d} now.", "Got it, updated to {d}.", "Sure, {d}."]


def dis(rng, n):
    return [rng.choice(T_DISTRACT) for _ in range(n)]


def gen(rng):
    kind = rng.choice(["upd", "upd", "bind", "twohop", "persp"])
    d = rng.randint(0, 8)
    if kind == "upd":
        e = rng.choice(T_EVENTS)
        k = rng.randint(0, 3)
        days = rng.sample(I.DAYS, k + 1)
        turns = [(f"My {e} is on {days[0]}.", f"Okay, your {e} is on {days[0]}.")]
        for i in range(1, k + 1):
            turns += dis(rng, rng.randint(0, 2))
            turns.append((rng.choice(UPD).format(e=e, d=days[i]), rng.choice(UPD_ACK).format(d=days[i])))
        turns += dis(rng, d)
        return turns, f"When is my {e}?", f"Your {e} is on", " " + days[k] + "."
    if kind == "bind":
        pet, rel = rng.choice(T_OWN)
        a, b = rng.sample(T_PETS, 2)
        intro = rng.choice([f"My {pet} is called {a} and my {rel}'s {pet} is called {b}.",
                            f"My {rel}'s {pet} is called {b} and my {pet} is called {a}."])
        turns = [(intro, "Cute names!")] + dis(rng, d)
        if rng.random() < 0.5:
            return turns, f"What is my {pet} called?", f"Your {pet} is called", " " + a + "."
        return turns, f"What is my {rel}'s {pet} called?", f"Your {rel}'s {pet} is called", " " + b + "."
    if kind == "twohop":
        r1, r2 = rng.choice(T_REL)
        n1, n2 = rng.sample(T_OTHER, 2)
        j1, j2 = rng.sample(T_JOBS, 2)
        dd = dis(rng, d)
        turns = [(f"My {r1} is {n1} and my {r2} is {n2}.", "Thanks for the intro.")] + dd[: d // 2] + \
                [(rng.choice([f"{n1} is a {j1} and {n2} is a {j2}.", f"{n2} is a {j2} and {n1} is a {j1}."]), "Nice.")] + dd[d // 2:]
        if rng.random() < 0.5:
            return turns, f"What is my {r1}'s job?", f"Your {r1} is a", " " + j1 + "."
        return turns, f"What is my {r2}'s job?", f"Your {r2} is a", " " + j2 + "."
    u, o = rng.sample(T_USER, 1)[0], rng.choice(T_OTHER)
    a = rng.choice(T_ASSIST)
    turns = [(f"Hey, I'm {u}. My friend {o} recommended you.", f"Hi {u}! I'm {a}. Happy to help.")] + dis(rng, d)
    q = rng.choice(["mine", "yours", "friend"])
    if q == "mine":
        return turns, "What is my name?", "Your name is", " " + u + "."
    if q == "yours":
        return turns, "What is your name?", "My name is", " " + a + "."
    return turns, "What is my friend's name?", "Your friend's name is", " " + o + "."


def encode(tok, rng, max_len=640):
    turns, q, pre, ans = gen(rng)
    prompt = I.transcript(turns, q, pre)
    p = tok(prompt, add_special_tokens=True).input_ids
    full = tok(prompt + ans, add_special_tokens=True).input_ids
    if full[: len(p)] != p:
        full = p + tok(ans, add_special_tokens=False).input_ids
    labels = [-100] * len(p) + full[len(p):]
    return full[-max_len:], labels[-max_len:]


def batches(tok, rng, bs):
    while True:
        exs = [encode(tok, rng) for _ in range(bs)]
        L = max(len(x) for x, _ in exs)
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        ids = torch.full((bs, L), pad); lab = torch.full((bs, L), -100); att = torch.zeros((bs, L), dtype=torch.long)
        for i, (x, y) in enumerate(exs):
            ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
        yield ids, lab, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(a.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    rng = random.Random(2026)
    it = batches(tok, rng, a.bs)
    t0 = time.time()
    for step in range(a.steps):
        ids, lab, att = next(it)
        out = model(input_ids=ids.to(a.device), attention_mask=att.to(a.device), labels=lab.to(a.device))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 25 == 0:
            print(f"step {step} loss {out.loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    model.eval()
    slug = a.model.replace("/", "__")
    with open(os.path.join(HERE, "out", f"ft__{slug}.jsonl"), "w") as f:
        n_params = sum(p.numel() for p in model.parameters()); emb = model.get_input_embeddings().weight.numel()
        f.write(json.dumps({"meta": True, "model": a.model + "+FT", "params": n_params, "emb": emb, "dtype": "fp32",
                            "steps": a.steps, "lr": a.lr, "bs": a.bs}) + "\n")
        for x in I.build():
            sc = score_item(model, tok, x["prompt"], x["cands"], a.device)
            rec = {kk: x[kk] for kk in ("task", "cond", "d")}
            rec["scores"] = {k: v[0] for k, v in sc.items()}; rec["ntok"] = {k: v[1] for k, v in sc.items()}
            rec["split"] = any(v[2] == "split" for v in sc.values()); rec["prompt_tokens"] = len(tok(x["prompt"]).input_ids)
            f.write(json.dumps(rec) + "\n")
    with open(os.path.join(HERE, "out", f"ftu__{slug}.jsonl"), "w") as f:
        for x in UP.build():
            sc = score_item(model, tok, x["prompt"], x["cands"], a.device)
            f.write(json.dumps({"task": x["task"], "d": x["d"], "scores": {k: v[0] for k, v in sc.items()}}) + "\n")
    with open(os.path.join(HERE, "out", f"ftk__{slug}.jsonl"), "w") as f:
        for x in KH.build():
            sc = score_item(model, tok, x["prompt"], x["cands"], a.device)
            f.write(json.dumps({"task": x["task"], "scores": {k: v[0] for k, v in sc.items()}}) + "\n")
    print("done", a.model, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
