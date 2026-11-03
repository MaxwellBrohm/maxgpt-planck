"""RC-12 frontier calibration: a Claude model through the same runner, graders and scorer as the panel.

Plays the dev split through runner.run_one (the real feedback loop: the model's OWN replies are its history) with a
responder that calls the Anthropic Messages API, and writes the panel's layout, <out>/<model>/template/<seed>/ and
<seed>_owncf/ (OD1 b twin, OWN family only), so dev_panel-style scoring reads it unchanged.

Protocol differences from the local panel, all forced by the API, all recorded in meta.json:
- render is "template" only: the API takes chat messages, there is no plain-text render to give it.
- no seeds and no temperature (current models 400 on a non-default temperature): seeds 1, 2, 3 are labels for three
  independent samples at the model's default sampling, not seeded draws. There is no greedy run.
- max_tokens 256, the panel's MAX_NEW_TOKENS, counted in Claude's tokenizer instead of the model's own.
- thinking off where the model allows it (Sonnet 5.5: between_tools, Haiku 5.5: disabled), so a reply is a reply.
- no system prompt, as in the panel. No refusal fallback: a refusal is scored as the reply it is (an empty one).
- an empty reply goes into the history as "(no reply)", because the API refuses an empty assistant turn; counted.

Usage: api_run.py --model claude-sonnet-5-5 --seeds 1,2,3 [--max-usd 40] [--limit-per-family 2] [--workers 16]
       [--env-file <.env>] [--out runs/frontier]
The key is read from ANTHROPIC_API_KEY, or from an ANTHROPIC_API_KEY= line in --env-file; it is never printed."""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic

import render as RD
import runner as RN
import score as S

THINKING = {"claude-sonnet-5-5": {"type": "between_tools"}, "claude-haiku-5-5": {"type": "disabled"}}
PRICE = {"claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50), "claude-opus-5-5": (4.00, 20.00)}
EMPTY = "(no reply)"


def api_key(env_file=None):
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    m = env_file and re.search(r"^\s*(?:export\s+)?ANTHROPIC_API_KEY\s*=\s*[\"']?([^\"'\s#]+)", open(env_file).read(), re.M)
    if not m:
        sys.exit("no ANTHROPIC_API_KEY in the environment or --env-file")
    return m.group(1)


class ClaudeResponder:
    """runner's responder interface (start, count, reply); stateless, so one instance serves every thread."""

    def __init__(self, model, env_file=None):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key(env_file), max_retries=8)
        self.lock = threading.Lock()
        self.usage = dict(calls=0, input=0, output=0, cache_read=0, cache_write=0, empty=0, cap=0, refusal=0)

    def start(self, rec, seed=None, render=None):
        pass

    def count(self, messages, render=None):
        return sum(len(m["content"]) for m in messages) // 4    # only used when --ctx trims history; it is not set

    def reply(self, history, i):
        msgs = [dict(m, content=m["content"] or EMPTY) if m["role"] == "assistant" else m for m in history]
        # cache the growing history (each turn's request is the last one plus two messages); it changes no output
        kw = dict(model=self.model, max_tokens=RD.MAX_NEW_TOKENS, messages=msgs, cache_control={"type": "ephemeral"})
        if self.model in THINKING:
            kw["thinking"] = THINKING[self.model]
        r = self.client.messages.create(**kw)
        text = "".join(b.text for b in r.content if b.type == "text")
        stop = {"end_turn": "eot", "max_tokens": "cap"}.get(r.stop_reason, r.stop_reason)
        u = r.usage
        with self.lock:
            self.usage["calls"] += 1
            self.usage["input"] += u.input_tokens
            self.usage["output"] += u.output_tokens
            self.usage["cache_read"] += u.cache_read_input_tokens or 0
            self.usage["cache_write"] += u.cache_creation_input_tokens or 0
            self.usage["empty"] += not text.strip()
            self.usage["cap"] += stop == "cap"
            self.usage["refusal"] += stop == "refusal"
        return text, stop

    def cost(self):
        pin, pout = PRICE[self.model]
        u = self.usage
        return (u["input"] * pin + u["cache_write"] * pin * 1.25 + u["cache_read"] * pin * 0.1 + u["output"] * pout) / 1e6


def write(out_dir, rows):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "transcripts.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(out_dir, "scores.jsonl"), "w") as f:
        for line in S.score_lines(rows):
            f.write(json.dumps(line) + "\n")


def play_all(pool, todo, resp, seed, cf, name, out_dir, max_usd):
    """every conversation of one run dir, each row appended to transcripts.partial.jsonl as it finishes (a rerun skips
    the ids already there), so a stop loses nothing paid for. Stops starting conversations once the spend passes
    max_usd or the API refuses the key (a usage limit); returns the rows in record order, or None if any is missing.
    (2026-11-03: the first Sonnet run died at the account's spend limit after ~10 minutes and kept nothing.)"""
    os.makedirs(out_dir, exist_ok=True)
    part = os.path.join(out_dir, "transcripts.partial.jsonl")
    done = {}
    if os.path.exists(part):
        for line in open(part):
            row = json.loads(line)
            done[row["id"]] = row
    lock, halt = threading.Lock(), []

    def one(rec):
        if rec["id"] in done:
            return done[rec["id"]]
        if halt or resp.cost() > max_usd:
            return None
        try:
            row = RN.run_one(rec, resp, "template", seed, None, name, 0, cf)
        except anthropic.APIStatusError as e:      # retries are spent (the SDK retries 429 and 5xx): stop cleanly
            halt.append(str(e)[:300])
            return None
        with lock, open(part, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    rows = list(pool.map(one, todo))
    if halt:
        print("stopped:", halt[0], flush=True)
    if any(r is None for r in rows):
        return None
    write(out_dir, rows)
    os.remove(part)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--limit-per-family", type=int, default=None, help="a pilot: the first N conversations per family")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-usd", type=float, default=40.0, help="stop starting conversations past this spend (this run)")
    ap.add_argument("--env-file", default=None, help="a .env holding ANTHROPIC_API_KEY, if it is not in the environment")
    ap.add_argument("--out", default=os.path.join(RN.HERE, "runs", "frontier"))
    args = ap.parse_args()
    recs = RN.load()
    if args.limit_per_family:
        seen = {}
        recs = [r for r in recs if seen.setdefault(r["family"], []).append(r) or len(seen[r["family"]]) <= args.limit_per_family]
    own = [r for r in recs if r["family"] == "OWN"]
    resp = ClaudeResponder(args.model, args.env_file)
    base = os.path.join(args.out, args.model, "template")
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for seed in args.seeds.split(","):
            s = int(seed)
            for label, todo, cf in ((seed, recs, False), (seed + "_owncf", own, True)):
                if os.path.exists(os.path.join(base, label, "scores.jsonl")):
                    continue                          # finished by an earlier run
                rows = play_all(pool, todo, resp, s, cf, args.model, os.path.join(base, label), args.max_usd)
                print(f"{label}: {'incomplete' if rows is None else len(rows)} conversations, {time.time() - t0:.0f}s,"
                      f" ${resp.cost():.2f} this run", json.dumps(resp.usage), flush=True)
                if rows is None:
                    sys.exit("run incomplete: rerun the same command to finish it")
    meta = dict(model=args.model, render="template", seeds=args.seeds.split(","), n_conversations=len(recs),
                limit_per_family=args.limit_per_family, max_tokens=RD.MAX_NEW_TOKENS,
                thinking=THINKING.get(args.model), temperature="model default (not settable)", system_prompt=None,
                fallbacks=None, usage=resp.usage, cost_usd=round(resp.cost(), 2), seconds=round(time.time() - t0),
                anthropic_sdk=anthropic.__version__)
    with open(os.path.join(args.out, args.model, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("done", json.dumps(meta))


if __name__ == "__main__":
    main()
