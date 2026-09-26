"""Yield table for one teacher's raw outputs (DRY: FAKE skeletons, nothing here is training data). CPU only.

    python3 -B check_outputs.py --skels SKELS.jsonl --outputs TEACHER.dry.outputs.jsonl [--wall-s S]
        [--planck-tok tokenizer/v0/tok_v0_8k.json] [--json OUT.json]

Runs every output through the pipeline's own path: render_prompt.build -> checker.run (parse + all checks) ->
checker.record_turns -> records.accepted, exactly as driver.finish does, minus the dedup index and the retry (so
this is FIRST-ATTEMPT yield). Reports accepted share, rejects by primary code, every code's hit count, diversity of the
accepted chats (stats.text_stats and stats.repetition on teacher-written turns) and, with --planck-tok (needs the
tokenizers package), accepted tokens counted in Planck's tokenizer; with --wall-s (the bench's yield-run wall time)
also accepted Planck tokens per hour."""
import argparse
import collections
import json
import os
import sys

sys.dont_write_bytecode = True
PIPE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PIPE)
import checker  # noqa: E402
import records  # noqa: E402
import render_prompt as R  # noqa: E402
import stats  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def planck_counter(path):
    if not path:
        return None
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    if hasattr(tok, "encode_special_tokens"):
        tok.encode_special_tokens = True        # as harness/data.load_tokenizer does
    return lambda s: len(tok.encode(s).ids)


def run(skels, outputs, teacher, count=None):
    by_id = {s["skel_id"]: s for s in skels}
    prim, codes, fin, only = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    acc, examples = [], {}
    for o in outputs:
        sk = by_id[o["skel_id"]]
        built = R.build(sk)
        fin[o.get("finish")] += 1
        res = checker.run(sk, o["text"], built)
        codes.update(res["codes"])
        if not res["ok"]:
            prim[res["primary"]] += 1
            failing = [c for c in res["codes"] if c not in checker.REPORT_ONLY]
            if len(failing) == 1:
                only[failing[0]] += 1
            examples.setdefault(res["primary"], {"skel_id": sk["skel_id"],
                                                 "hit": [str(x)[:160] for x in res["hits"][:2]]})
            continue
        turns = checker.record_turns(sk, res)
        meta = {"model": teacher, "license": "Apache-2.0", "mode": "offline", "stub": False,
                "prompt_variant": built["variant"], "prompt_sha256": built["prompt_sha256"], "render_seed": None}
        acc.append(records.accepted(sk, built, res, {"finish": o.get("finish")}, 0, meta, turns))
    n = len(outputs)
    out = {"status": "dry", "teacher": teacher, "n": n, "accepted": len(acc),
           "yield": round(len(acc) / n, 4) if n else None, "finish": dict(fin),
           "rejects_by_primary": dict(prim.most_common()), "all_codes": dict(codes.most_common()),
           "fails_on_exactly_one_code": dict(only.most_common()),
           "first_example_per_primary": examples, "dedup": "not applied (first-attempt checker yield only)"}
    if acc:
        out["diversity"] = {"text": stats.text_stats(acc), "repetition": stats.repetition(acc, top=3)}
    if count and acc:
        allt = sum(count(t["text"]) for r in acc for t in r["turns"])
        teach = sum(count(t["text"]) for r in acc for t in r["turns"] if str(t["author"]).startswith("teacher:"))
        out["planck_tokens"] = {"accepted_all_turns": allt, "accepted_teacher_turns": teach,
                                "per_accepted_chat": round(allt / len(acc), 1),
                                "counted": "turn text only, no role or template tokens"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skels", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--planck-tok", default=None)
    ap.add_argument("--wall-s", type=float, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    outputs = load_jsonl(a.outputs)
    teacher = outputs[0].get("teacher", "?") if outputs else "?"
    out = run(load_jsonl(a.skels), outputs, teacher, planck_counter(a.planck_tok))
    if a.wall_s and "planck_tokens" in out:
        out["accepted_planck_tokens_per_hour"] = round(out["planck_tokens"]["accepted_all_turns"] / a.wall_s * 3600)
        out["wall_s"] = a.wall_s
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    brief = {k: out.get(k) for k in ("teacher", "n", "accepted", "yield", "rejects_by_primary",
                                     "fails_on_exactly_one_code", "planck_tokens",
                                     "accepted_planck_tokens_per_hour")}
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
