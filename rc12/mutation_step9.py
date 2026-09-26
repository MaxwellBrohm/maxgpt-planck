"""RC-12 mutation test of the STEP 9 code (Max's rule: every test claim is watched failing). Each mutant replaces
one text in lockstep.py, runner.py, vllm_responder.py, parity_hf_vllm.py, dev_batch.py or hf_batched.py, is
installed in a FRESH process (spawn) before anything imports it, and runs the suites it names: lock =
test_lockstep.checks on QUICK fakes (no CLI), vllm = test_vllm_responder.checks (no import-light subprocess), parity =
test_parity, batch = test_dev_batch, hfb = test_hf_batched.c_split (its engine check builds a toy model: PC only,
mutation_hf_batched.py). KILLED = a suite reports a failure (the unmutated baseline reports none); crashed = an
exception escaped (counted, not a kill); malformed = the text is not found exactly once. No model.
Usage: python3 -B mutation_step9.py [--part i/n]   (writes logs/mutation_step9[_part<i>].txt; exit 1 unless the
baseline is clean and every mutant is killed)"""
import argparse
import multiprocessing as mp
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = ["IDEAL", "IDEAL_ALT", "FIRST", "SAMPLER", "HISTCHECK", "FORGETFUL", "CONSIST", "NEXT", "LISTPAD", "CAPPER",
         "CONTINUER"]
MUTANTS = [
    ("lockstep", "outs = eng.reply_batch(reqs)", "outs = eng.reply_batch(reqs)[::-1]", "lock",
     "replies go back to the wrong conversations"),
    ("lockstep", "(k, convs[k][0], convs[k][1])", "(k, convs[k][0], convs[0][1])", "lock vllm",
     "every conversation gets the first conversation's seed"),
    ("lockstep", "R.play_steps(rec, eng, render, ctx, own_cf)", "R.play_steps(rec, eng, render, ctx, False)",
     "lock vllm", "--own-cf rewrite skipped in lockstep"),
    ("lockstep", "R.play_steps(rec, eng, render, ctx, own_cf)", "R.play_steps(rec, eng, render, None, own_cf)",
     "lock", "history not fitted to the context in lockstep"),
    ("lockstep", "self.live[k].start(rec, seed, self.render)", "self.live[k].start(rec, None, self.render)", "lock",
     "PerConv starts every conversation greedy"),
    ("lockstep", "self.r.start(rec, seed, self.render)", "self.r.start(rec, None, self.render)", "lock",
     "Serial drops the seed"),
    ("lockstep", "R.make_row(rec, pl, render, seed, name, train_seed, own_cf)",
     "R.make_row(rec, pl, render, seed, name, train_seed, False)", "lock", "lockstep rows lose the own_cf flag"),
    ("lockstep", "convs = [(r, s) for s in seeds for r in recs]", "convs = [(r, s) for r in recs for s in seeds]",
     "lock vllm", "lockstep row order differs from runner.run"),
    ("lockstep", "if isinstance(responder, (HR.HFResponder, PR.PlanckResponder)):", "if True:", "lock",
     "a bare fake runs through Serial"),
    ("runner", "req = steps.send(responder.reply(*req))", "req = steps.send(responder.reply(req[0], 1))", "lock",
     "sequential play sends the wrong turn index"),
    ("runner", "    if lockstep:  ", "    if False:  ", "lock vllm", "--lockstep ignored by runner.run"),
    ("runner", 'if getattr(args, "lockstep", False):', "if False:", "lock",
     "runner --lockstep shares one fake instance"),
    ("runner", "if not (VR.VLLM_TESTED or args.vllm_untested_ok):", "if False:", "vllm",
     "untested vLLM path runs without the flag"),
    ("vllm_responder", "seed=HR.conv_seed(seed, rid, i)", "seed=HR.conv_seed(seed, rid, 0)", "vllm",
     "per-reply seed ignores the turn"),
    ("vllm_responder", "top_k=TOP_K_OFF", "top_k=20", "vllm", "top-k on"),
    ("vllm_responder", 'kw["temperature"] = 0.0', 'kw["temperature"] = 0.6', "vllm", "greedy samples"),
    ("vllm_responder", 'max_tokens=HR.DECODE["max_new_tokens"]', "max_tokens=512", "vllm", "512 new tokens"),
    ("vllm_responder", "self.eot = sorted({vocab[t] for t in HR.EOT_TOKENS if t in vocab} - {self.eos})",
     "self.eot = []", "vllm", "end-of-turn tokens do not stop"),
    ("vllm_responder", "if not ids or ids[-1] != tok:", "if False:", "vllm",
     "stop token not appended when vLLM leaves it out"),
    ("vllm_responder", 'return ids, "eot" if tok in self.eot else "eos"', 'return ids, "eos"', "vllm",
     "eot reported as eos"),
    ("vllm_responder", 'if finish == "length":', 'if finish == "cap":', "vllm", "length finish not a cap"),
    ("vllm_responder", "text = self.tok.decode(ids, skip_special_tokens=True)",
     "text = self.tok.decode(ids, skip_special_tokens=False)", "vllm", "special tokens in the text"),
    ("vllm_responder", "= hf_rule or takes, takes and not hf_rule", "= hf_rule, False", "vllm",
     "a template that takes enable_thinking renders thinking on"),
    ("vllm_responder", 'prompts.append({"prompt_token_ids": ids})', 'prompts.append({"prompt": self.prompt(history)})',
     "vllm", "vLLM re-tokenizes a text prompt"),
    ("vllm_responder", "min(native or MAX_LEN_CAP, MAX_LEN_CAP)", "(native or MAX_LEN_CAP)", "vllm",
     "context not capped at 32768"),
    ("vllm_responder", "(0, self.rec, self.seed, history, i)", "(0, self.rec, None, history, i)", "vllm",
     "sequential reply drops the seed"),
    ("parity_hf_vllm", 'if all(t["margin"] is not None and t["margin"] <= NEAR_TIE for t in bad):', "if True:",
     "parity", "any difference passes as a near-tie"),
    ("parity_hf_vllm", 'if any(not t["prompt_equal"] for t in turns):', "if False:", "parity",
     "prompt mismatch not failed"),
    ("parity_hf_vllm", 'match=eq and h["stop"] == v["stop"] and h["text"] == v["text"]', "match=eq", "parity",
     "stop reasons not compared"),
    ("parity_hf_vllm", "return None if len(a) == len(b) else min(len(a), len(b))", "return None", "parity",
     "a strict prefix counts as equal"),
    ("parity_hf_vllm", 'prompt_equal=prompt == v["prompt_ids"]', "prompt_equal=True", "parity",
     "HF prompt never compared with vLLM's"),
    ("parity_hf_vllm", 'if j is not None and j < len(ids) and j < len(v["ids"]):', "if False:", "parity",
     "margins never computed"),
    ("parity_hf_vllm", "        return recs[:n]\n", "        return recs[1:n + 1]\n", "parity",
     "--pick first skips the first record"),
    ("parity_hf_vllm", "out.append(q.pop(0))", "out.append(q[0])", "parity", "--pick spread repeats records"),
    ("dev_batch", 'if os.path.exists(os.path.join(d, "DONE")):', "if False:", "batch", "finished runs not skipped"),
    ("dev_batch", "if gated and verdict not in PASSING and chosen != kind and not args.no_parity_gate:", "if False:",
     "batch", "parity gate off"),
    ("dev_batch", "if gated and verdict not in PASSING and chosen != kind and not", "if gated and verdict not in "
     "PASSING and not", "batch", "the engines.json decision is ignored"),
    ("dev_batch", "if chosen is not None and chosen != kind and not args.no_parity_gate:", "if False:", "batch",
     "a model runs on an engine engines.json did not choose"),
    ("dev_batch", 'return json.load(open(path))["models"].get(model_id, {}).get("engine")', "return None", "batch",
     "engines.json never read"),
    ("dev_batch", 'gated = kind in ("vllm", "hfb")', 'gated = kind == "vllm"', "batch", "hfb runs without parity"),
    ("parity_hf_vllm", "res[\"engine\"] = meta.get(\"engine\", \"vllm\")", "pass", "parity",
     "the report does not say which engine was compared"),
    ("parity_hf_vllm", '    if engine == "hfb":', "    if False:", "parity", "--engine hfb loads vLLM anyway"),
    ("hf_batched", "            out = out[:j + 1]\n", "            out = out[:j]\n", "hfb",
     "the stop token is cut off a batched reply"),
    ("hf_batched", "    elif len(out) >= cap and", "    elif len(out) > cap and", "hfb", "a capped reply reads as eos"),
    ("dev_batch", 'open(os.path.join(tmp, "DONE"), "w").close()', "pass", "batch", "a run is never marked DONE"),
    ("dev_batch", "own if own_cf else recs", "recs", "batch", "the cf twin runs every family"),
    ("dev_batch", "shutil.rmtree(tmp, ignore_errors=True)", "pass", "batch", "a stale .partial is kept"),
]


def suites(names):
    import runner as R
    fails = []
    if "lock" in names:
        import test_lockstep as TL
        fails += TL.checks(R.load(), QUICK, cli=False)
    if "vllm" in names:
        import test_vllm_responder as TV
        fails += TV.checks(R.load(), light=False)
    if "parity" in names:
        import test_parity as TP
        for fn in (TP.c_diff, TP.c_verdict, TP.c_stages, TP.c_pick):
            try:
                fails += fn()
            except Exception as e:  # noqa: BLE001
                fails.append(f"{fn.__name__} raised {type(e).__name__}")
    if "hfb" in names:
        import test_hf_batched as TH
        fails += TH.c_split()
    if "batch" in names:
        import test_dev_batch as TD
        fails += TD.checks()
    return fails


def job(k):
    sys.path.insert(0, HERE)
    sys.dont_write_bytecode = True
    names = "lock vllm parity batch hfb"
    if k >= 0:
        mod, old, new, names, _ = MUTANTS[k]
        path = os.path.join(HERE, mod + ".py")
        src = open(path).read()
        if src.count(old) != 1:
            return k, "malformed", [f"text found {src.count(old)} times"]
        m = types.ModuleType(mod)
        m.__file__ = path
        sys.modules[mod] = m
        exec(compile(src.replace(old, new), path, "exec"), m.__dict__)
    devnull = open(os.devnull, "w")
    saved, sys.stdout = sys.stdout, devnull
    try:
        return k, "ran", suites(names.split())
    except Exception as e:  # noqa: BLE001
        return k, "crashed", [f"{type(e).__name__}: {e}"[:200]]
    finally:
        sys.stdout = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="1/1")
    i, n = (int(x) for x in ap.parse_args().part.split("/"))
    ks = [k for k in range(len(MUTANTS)) if k % n == i - 1]
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(8, len(ks) + 1), maxtasksperchild=1) as pool:
        results = {k: (v, f) for k, v, f in pool.map(job, [-1] + ks, chunksize=1)}
    verdict, base = results.pop(-1)
    lines = [f"RC-12 mutation_step9.py part {i}/{n}: {len(ks)} of {len(MUTANTS)} mutants, fresh process each, "
             f"{time.time() - t0:.0f} s. No model.", f"baseline: {verdict}, failures: {base[:3] or 'none'}"]
    counts = {"killed": 0, "survived": 0, "crashed": 0, "malformed": 0}
    for k in ks:
        mod, _, _, names, claim = MUTANTS[k]
        v, fails = results[k]
        v = ("killed" if fails else "survived") if v == "ran" else v
        counts[v] += 1
        lines.append(f"{v.upper():9s} #{k:02d} {mod} [{names}]: {claim}" + (f"  <- {fails[0][:140]}" if fails else ""))
    lines.append(" | ".join(f"{a} {b}" for a, b in counts.items()))
    ok = verdict == "ran" and not base and counts["killed"] == len(ks)
    lines.append("ALL MUTANTS KILLED" if ok else "NOT ALL KILLED (or the baseline is not clean)")
    text = "\n".join(lines) + "\n"
    name = "mutation_step9.txt" if n == 1 else f"mutation_step9_part{i}.txt"
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    with open(os.path.join(HERE, "logs", name), "w") as f:
        f.write(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
