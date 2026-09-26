"""RC-12 runner (SPEC s1, s2): plays every 12-turn conversation against a responder, feeding the responder's OWN
replies back as history, grades it with the real graders, and writes per-turn transcripts and per-family scores.

Responders: fake:<NAME> (fakes.py / fakes_family.py, no model), hf:<model id or path> (hf_responder.py,
UNTESTED: no model has been loaded through it; it runs only with --hf-untested-ok) or planck:<checkpoint.pt>
(planck_responder.py: a harness checkpoint rendered with Planck's own role tokens; ctx = its seq_len; flags
--planck-config, --planck-tokenizer, --planck-device (default cpu), --planck-precision; needs torch, so run it with
a Python that has torch and tokenizers).
Loop per conversation: send u1, generate a1, append a1 as the assistant message, send u2, ... to a12. Nothing from a
gold, IDEAL reply or annotation is ever put in the history. The fed-back reply is the decoded text after the stop
rule, stripped; it is stored verbatim with its stop reason (eos | eot | role | cap).
Render: template (tokenizer chat template; fakes use a ChatML-shaped stub) or plain ("User: ...\nAssistant:";
the runner cuts a reply at the first newline followed by a role tag and records stop "role").
Seeds: "greedy" (seed None) or integers (sampling, T = 0.6, top-p 1.0, per-conversation seed in the responder).
Context: with --ctx N, the history is fitted to N - 256 tokens by dropping the oldest whole user/assistant pairs
(logged per turn as "dropped"); a probe fails ("trunc") if any S, C, I, O, Q or T turn before it, or any of its
src turns, was dropped when its reply was generated (SPEC s1; notes D13: every such turn, not only the probe's own).

Output (--out DIR): transcripts.jsonl (one line per conversation x seed: turns [{i, kind, user, reply, stop,
dropped, raw}], probes (grader results), unit, flags, leaks, ack_repeat and ack_of_answer (OD6 iii reports), own_cf
and cf_unswapped (a --own-cf rewrite that did not take: own_cf.py)) and scores.jsonl (per seed: every family and cell;
score.py computes the composites). CLI:
  python3 -B runner.py --responder fake:IDEAL --render plain --seeds greedy --out runs/fake_IDEAL
  python3 -B runner.py --responder hf:Qwen/Qwen2.5-0.5B-Instruct --render template --seeds 1,2,3 \\
      --ctx 32768 --out runs/qwen05 --hf-untested-ok
  <venv python> -B runner.py --responder planck:../runs/x/out/final_00001000.pt --planck-config ../runs/x/config.yaml \\
      --render template --seeds greedy,1,2,3 --out runs/planck_x
OD1 (b): R needs the OWN counterfactual run too (same responder, render, seeds, --train-seed, plus --own-cf
--families OWN --out <dir>_owncf); score both: score.py <dir>/transcripts.jsonl <dir>_owncf/transcripts.jsonl.
A run alone prints R null and R_ungated (the composite with the own-history OWN)."""
import argparse
import json
import os
import sys
import time

import graders as G
import own_cf as OC
import render as RD
import score as S

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.join(HERE, "dev", "rc12_dev.jsonl")
NEEDED_KINDS = ("S", "C", "I", "O", "Q", "T")


def load(path=DEV, families=None, limit=None):
    recs = [json.loads(line) for line in open(path)]
    if families:
        recs = [r for r in recs if r["family"] in families]
    return recs[:limit] if limit else recs


def play(rec, responder, render="plain", seed=None, ctx=None, own_cf=False):
    """one conversation through the real feedback loop; returns the per-turn transcript. own_cf: the OWN
    counterfactual-history run (own_cf.py); score.py uses its rows only to gate OWN (OD1 b), never as a history."""
    responder.start(rec, seed, render)
    budget = None if ctx is None else ctx - RD.MAX_NEW_TOKENS
    msgs, out = [], []
    for t in sorted(rec["turns"], key=lambda x: x["i"]):
        msgs.append({"role": "user", "content": t["text"]})
        hist, dropped = RD.fit(msgs, budget, lambda m: responder.count(m, render))
        raw, stop = responder.reply([dict(m) for m in hist], t["i"])
        text = raw
        if render == "plain":
            text, cut = RD.cut_plain(raw)
            if cut:
                stop = "role"
        text = text.strip()
        alt = OC.swap(rec, t["i"], text) if own_cf else None
        failed = alt is OC.UNSWAPPED               # G-DYN parses it, but the rewrite did not take (own_cf.py)
        if failed:
            alt = None
        if alt is not None:
            raw, text = raw, alt                     # the model's own text stays in raw
        msgs.append({"role": "assistant", "content": text})
        out.append(dict(i=t["i"], kind=t["kind"], user=t["text"], reply=text, stop=stop, dropped=dropped,
                        raw=None if raw == text else raw))
        if failed:
            out[-1]["cf_unswapped"] = True           # score.py: this OWN unit counts 0 in OWN_GATED
    return out


def needed_turns(rec, probe):
    need = {t["i"] for t in rec["turns"] if t["i"] < probe["turn"] and t["kind"] in NEEDED_KINDS}
    return sorted(need | set(probe.get("src") or []))


def grade(rec, played):
    replies = [x["reply"] for x in played]
    stops = [x["stop"] for x in played]
    g = G.grade_conv(rec, replies, stops)
    for res, probe in zip(g["probes"], rec["probes"]):
        lost = [i for i in needed_turns(rec, probe) if i <= played[probe["turn"] - 1]["dropped"]]
        if lost:
            res["ok"] = False
            res["fails"] = res["fails"] + ["trunc"]
            res["lost"] = lost
    g["unit"] = G.unit_score(rec, g["probes"])
    return g


def run_one(rec, responder, render, seed, ctx, name="?", train_seed=0, own_cf=False):
    played = play(rec, responder, render, seed, ctx, own_cf)
    g = grade(rec, played)
    return dict(id=rec["id"], family=rec["family"], cell=rec["cell"], knowledge=rec["knowledge"],
                pair_id=rec["meta"].get("pair_id"), responder=name, render=render, seed=seed,
                train_seed=train_seed, unit=g["unit"], probes=g["probes"], flags=g["flags"], leaks=g["leaks"],
                ack_repeat=g["ack_repeat"], ack_of_answer=g["ack_of_answer"], turns=played, own_cf=own_cf,
                cf_unswapped=any(x.get("cf_unswapped") for x in played))


def run(recs, responder, render="plain", seeds=(None,), ctx=None, out_dir=None, name="?", train_seed=0, own_cf=False):
    rows = [run_one(r, responder, render, s, ctx, name, train_seed, own_cf) for s in seeds for r in recs]
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "transcripts.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        with open(os.path.join(out_dir, "scores.jsonl"), "w") as f:
            for line in S.score_lines(rows):
                f.write(json.dumps(line) + "\n")
    return rows


def make_responder(spec, args):
    kind, _, name = spec.partition(":")
    if kind == "fake":
        import fakes_family as FF
        return FF.make(name), name
    if kind == "hf":
        if not args.hf_untested_ok:
            sys.exit("hf responder is UNTESTED (no model was ever loaded through it); rerun with --hf-untested-ok")
        import hf_responder as H
        return H.HFResponder(name, render=args.render, dtype=args.dtype, device=args.device), name
    if kind == "planck":
        import planck_responder as PR
        r = PR.from_checkpoint(name, args.planck_config, args.planck_tokenizer, args.planck_device,
                               args.planck_precision, args.render)
        if args.ctx is not None and args.ctx > r.ctx:
            sys.exit(f"--ctx {args.ctx} exceeds the checkpoint's seq_len {r.ctx}")
        return r, name
    sys.exit(f"unknown responder {spec}")


def parse_seeds(text):
    return [None if s == "greedy" else int(s) for s in text.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responder", required=True)
    ap.add_argument("--render", choices=["template", "plain"], default="template")
    ap.add_argument("--seeds", default="greedy")
    ap.add_argument("--ctx", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=DEV)
    ap.add_argument("--families", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--hf-untested-ok", action="store_true")
    ap.add_argument("--own-cf", action="store_true", help="OWN counterfactual-history diagnostic (own_cf.py)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--planck-config", default=None, help="planck: the run's config.yaml (tokenizer path, chat ids)")
    ap.add_argument("--planck-tokenizer", default=None, help="planck: tokenizer.json (overrides the config's)")
    ap.add_argument("--planck-device", default="cpu", help="planck: cpu | mps | cuda (default cpu)")
    ap.add_argument("--planck-precision", default="auto", help="planck: auto | fp32 | bf16 (auto: bf16 off cpu)")
    args = ap.parse_args()
    recs = load(args.data, args.families.split(",") if args.families else None, args.limit)
    responder, name = make_responder(args.responder, args)
    t0 = time.time()
    ctx = args.ctx if args.ctx is not None else getattr(responder, "ctx", None)
    rows = run(recs, responder, args.render, parse_seeds(args.seeds), ctx, args.out, name, args.train_seed,
               args.own_cf)
    summary = S.summarize(rows)
    print(json.dumps({k: summary[k] for k in ("R", "R_ungated", "own_gate", "families", "level_a_met", "loop_rate",
                                              "ack_repeat", "t0")}, indent=1))
    print(f"{len(rows)} conversations in {time.time() - t0:.1f} s -> {args.out}")


if __name__ == "__main__":
    main()
