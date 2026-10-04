"""RC-12 rerun-check standard for verify_dev_runs --rerun (R1 / R2). Max, 2026-10-02: took all recommendations in
rc12/DECISIONS_FOR_MAX.md (item 2): EVERY engine passes a replay when each conversation's first differing turn differs
at a near-tie, margin <= parity_hf_vllm.NEAR_TIE (0.5 nats) under one HF forward (parity_hf_vllm.hf_margin) on that
turn's prompt plus the tokens both replies share; the identical share is reported beside it. Exact replay is not the
bar: the stored runs drift with the batch on every engine (notes STEP 10a: identical turn-1 output 915 / 2304 on vLLM,
47 / 192 on Doge).
Comparisons: R1 the first N conversations of the first sampled run replayed in lockstep (same seed) vs the stored
  rows; R2 the same N greedy in lockstep vs played sequentially, and vs the stored greedy rows.
divergences(a, b): per conversation pair the FIRST turn whose (reply, stop) differ. Every earlier reply is equal, so
  the prompt there is the same in both runs; later turns sit on different histories and are not compared. The
  history is rebuilt from the row (the user turns and replies before it, minus the pairs the fit dropped).
Stages, each its own process so one model is on the GPU at a time (as in parity); check() runs both for verify:
  replay   the run's engine (meta.json: engine, dtype, trust_remote_code, attn, max_batch) replays -> <dir>/replay.json
  margins  HFResponder(model, render) encodes each history; both generations (raw text, else the reply; re-tokenized,
           the ids are not stored) are compared token by token; at the first differing token j the margin is
           |log p(a_j) - log p(b_j)| on prompt + a[:j]. A strict prefix whose turn stopped on eos / eot competes with
           HF's first stop id; equal token ids or a capped prefix give no margin (None). No divergence: no model is
           loaded. A fake engine (tests, no model) takes --tie-margin-stub X as every margin; a real engine refuses it.
judge(): PASS when every first divergence has a margin <= NEAR_TIE; a missing margin fails, as in parity.
Caveat for the GPU run: under sampling (R1) one seed's draw can flip between two tokens that are not near-tied when
the logits move slightly, so R1 can FAIL on drift alone; its margins are printed. R2 (greedy) is the parity case."""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import parity_hf_vllm as PH
import runner as R

NEAR_TIE = PH.NEAR_TIE
HERE = os.path.dirname(os.path.abspath(__file__))


def history(row, k):
    """the messages the runner sent for the reply to turn index k: earlier user turns and replies, minus the pairs
    render.fit dropped (turn "dropped"), ending with user turn k."""
    msgs = []
    for t in row["turns"][:k]:
        msgs += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["reply"]}]
    msgs.append({"role": "user", "content": row["turns"][k]["user"]})
    return msgs[2 * row["turns"][k]["dropped"]:]


def gen(t):
    return t["raw"] if t.get("raw") is not None else t["reply"]


def divergences(a_rows, b_rows):
    out, same = [], 0
    for x, y in zip(a_rows, b_rows):
        assert x["id"] == y["id"], (x["id"], y["id"])
        k = next((i for i, (ta, tb) in enumerate(zip(x["turns"], y["turns"]))
                  if (ta["reply"], ta["stop"]) != (tb["reply"], tb["stop"])), None)
        if k is None:
            same += 1
            continue
        ta, tb = x["turns"][k], y["turns"][k]
        out.append(dict(id=x["id"], turn=ta["i"], history=history(x, k), a=gen(ta), b=gen(tb), stop_a=ta["stop"],
                        stop_b=tb["stop"]))
    return dict(conversations=len(a_rows), identical=same, divergences=out)


def replay(a):
    import dev_batch as DB
    import verify_dev_runs as V
    runs = V.load_runs(os.path.join(os.path.expanduser(a.root), DB.slug(a.model), a.render))
    recs = {r["id"]: r for r in R.load(a.data)}
    json.dump(replay_runs(a, runs, recs), open(os.path.join(a.dir, "replay.json"), "w"))


def replay_runs(a, runs, recs):
    """the replays of stage replay on loaded runs (test_hf_trust drives it with stub libraries)."""
    sampled = [n for n in runs if n.isdigit()]
    ref = runs[sampled[0]] if sampled else None
    greedy = runs.get("greedy")
    m = (ref or greedy)["meta"]
    ns = argparse.Namespace(render=a.render, dtype=m.get("dtype") or "bfloat16", device="cuda",
                            gpu_mem=m.get("gpu_mem", 0.85), max_model_len=m.get("max_model_len"),
                            batch_invariant=bool(m.get("batch_invariant")), lockstep=True, hf_untested_ok=True,
                            vllm_untested_ok=True, trust_remote_code=m.get("trust_remote_code") is True,
                            attn_implementation=m.get("attn_implementation"),   # the run's HF options (STEP 9c)
                            max_batch=m.get("max_batch"), token_budget=m.get("token_budget"),
                            chat_template=m.get("chat_template"),       # item 14's engine options, when recorded
                            nan_guard=m.get("nan_guard"), use_cache=m.get("use_cache"))
    spec = f"{m['engine']}:{a.model}"
    eng, name = R.make_responder(spec, ns)
    ctx = getattr(eng, "ctx", None)
    seq = eng if hasattr(eng, "start") else R.make_responder(spec, argparse.Namespace(**{**vars(ns), "lockstep": False}))[0]
    comps, info = [], []
    if ref is not None:
        seed = ref["meta"]["seed"]
        stored = ref["rows"][:a.rerun]
        again = R.run([recs[r["id"]] for r in stored], eng, a.render, [seed], ctx, None, name, 0, False, True)
        comps.append(dict(tag=f"R1 seed {seed} lockstep replay vs stored", **divergences(again, stored)))
        other = R.run([recs[r["id"]] for r in stored], eng, a.render, [seed + 100], ctx, None, name, 0, False, True)
        same = sum(x["turns"][0]["reply"] == y["turns"][0]["reply"] for x, y in zip(other, stored))
        info.append(f"R1 seed {seed + 100} vs seed {seed}: turn-1 replies identical {same} / {len(stored)}")
    ids = [r["id"] for r in (greedy or ref)["rows"][:a.rerun]]
    lock = R.run([recs[i] for i in ids], eng, a.render, [None], ctx, None, name, 0, False, True)
    serial = R.run([recs[i] for i in ids], seq, a.render, [None], ctx, None, name, 0, False, False)
    comps.append(dict(tag="R2 greedy lockstep vs sequential", **divergences(lock, serial)))
    if greedy is not None:
        comps.append(dict(tag="R2 greedy lockstep replay vs stored greedy run",
                          **divergences(lock, greedy["rows"][:a.rerun])))
    info.append(f"R batches {getattr(eng, 'batches', [])[:30]}")
    hf = dict(dtype=ns.dtype, trust_remote_code=ns.trust_remote_code, attn_implementation=ns.attn_implementation,
              chat_template=ns.chat_template)
    return dict(engine=m["engine"], hf=hf, comparisons=comps, info=info)


def hf_margins(h, divs, margin=None):
    """one margin per divergence (None where none exists), h an HFResponder-like object (tok, encode, stop_ids)."""
    margin = margin or PH.hf_margin
    stop = h.stop_ids[0] if h.stop_ids else None
    out = []
    for d in divs:
        prompt = h.encode(d["history"])
        ta, tb = (h.tok.encode(x, add_special_tokens=False) for x in (d["a"], d["b"]))
        j = PH.first_diff(ta, tb)
        if j is None:
            out.append(None)
        elif j < len(ta) and j < len(tb):
            out.append(abs(margin(h, prompt, ta[:j], ta[j], tb[j])[0]))
        else:
            short, longer, st = (ta, tb, d["stop_a"]) if len(ta) == j else (tb, ta, d["stop_b"])
            ok = st in ("eos", "eot") and stop is not None
            out.append(abs(margin(h, prompt, short, stop, longer[j])[0]) if ok else None)
    return out


def margins(a):
    rep = json.load(open(os.path.join(a.dir, "replay.json")))
    divs = [d for c in rep["comparisons"] for d in c["divergences"]]
    if rep["engine"] != "fake" and a.tie_margin_stub is not None:
        sys.exit("--tie-margin-stub is for fake engines only (a real engine gets HF margins)")
    if rep["engine"] == "fake":
        out = [a.tie_margin_stub] * len(divs)
    elif not divs:
        out = []
    else:
        import hf_responder as HR
        o = rep["hf"]
        tmpl = {"chat_template": o["chat_template"]} if o.get("chat_template") else {}
        h = HR.HFResponder(a.model, a.render, o["dtype"], "cuda", trust_remote_code=o["trust_remote_code"],
                           attn_implementation=o["attn_implementation"], **tmpl)
        out = hf_margins(h, divs)
    json.dump(out, open(os.path.join(a.dir, "margins.json"), "w"))


def judge(c, ms, say):
    ok = all(m is not None and m <= NEAR_TIE for m in ms)
    known = [m for m in ms if m is not None]
    first = c["divergences"][0] if c["divergences"] else None
    say("PASS" if ok else "FAIL", f"{c['tag']}: {c['identical']} / {c['conversations']} conversations identical, "
        f"{len(ms)} first divergences, {len(ms) - len(known)} without a margin, max margin "
        f"{f'{max(known):.3f}' if known else '-'} (near-tie <= {NEAR_TIE})"
        + (f"; first {first['id']} t{first['turn']} {first['a'][:60]!r} / {first['b'][:60]!r}" if first else ""))


def check(args, say, engine):
    """verify_dev_runs --rerun N: the replay and margins stages as subprocesses, then one verdict line each. engine:
    the runs' meta engine (a stub margin is refused before anything loads unless it is "fake")."""
    if args.tie_margin_stub is not None and engine != "fake":
        return say("FAIL", f"R: --tie-margin-stub is for fake engines only, not {engine} (it gets HF margins)")
    d = tempfile.mkdtemp(prefix="rc12_rerun_")
    base = [sys.executable, "-B", os.path.join(HERE, "rerun_tie.py"), "--dir", d, "--model", args.model, "--render",
            args.render]
    p = subprocess.run(base + ["--stage", "replay", "--root", args.root, "--data", args.data, "--rerun", str(args.rerun)])
    if p.returncode:
        return say("FAIL", f"R replay stage exit {p.returncode} (work dir {d})")
    stub = [] if args.tie_margin_stub is None else ["--tie-margin-stub", repr(args.tie_margin_stub)]
    p = subprocess.run(base + ["--stage", "margins"] + stub, capture_output=True, text=True)
    if p.returncode:
        return say("FAIL", f"R margins stage exit {p.returncode}: {(p.stdout + p.stderr).strip()[-200:]}")
    rep = json.load(open(os.path.join(d, "replay.json")))
    ms = json.load(open(os.path.join(d, "margins.json")))
    i = 0
    for c in rep["comparisons"]:
        judge(c, ms[i:i + len(c["divergences"])], say)
        i += len(c["divergences"])
    for line in rep["info"]:
        say("INFO", line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["replay", "margins"], required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--render", choices=["template", "plain"], required=True)
    ap.add_argument("--root")
    ap.add_argument("--data", default=R.DEV)
    ap.add_argument("--rerun", type=int, default=5)
    ap.add_argument("--tie-margin-stub", type=float, default=None)
    a = ap.parse_args()
    return (replay if a.stage == "replay" else margins)(a)


if __name__ == "__main__":
    sys.exit(main())
