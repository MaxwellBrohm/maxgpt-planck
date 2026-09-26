"""Deliberate bugs in data_prep, each applied to a scratch copy of the tree; the test suite must fail.

    python data_prep/mutation_check.py [--list] [--id X] [--part i/n] [--python PY]

killed = pytest exit 1 (test failures). A collection error (exit 2 or 4) is INVALID, never killed: it
proves nothing about the tests. The bpb mutants need torch (run where torch is; bpb tests build tiny
CPU models only). The ngram mutants need only numpy and tokenizers (--id or a filtered list runs them alone).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COPY = ["data_prep", "harness", "corpus/oodh.py", "corpus/sample_plan.py", "tokenizer/spec.py", "tokenizer/v0"]
D = "data_prep/"
MUTANTS = [
    ("eot_to_pad", D + "pretok_worker.py", "arr[-1] = eot", "arr[-1] = 0"),
    ("no_special_clean", D + "prep_common.py", 't = SPECIAL_RE.sub(" ", s)', "t = s"),
    ("exclude_by_id_only", D + "pretok_worker.py", "if any(k in exclude for k in C.record_keys(rec)):",
     'if rec["id"] in exclude:'),
    ("heldout_ids_only", D + "pretokenize.py", "keys.update(C.record_keys(json.loads(line)))",
     'keys.update([json.loads(line)["id"]])'),
    ("record_keys_no_split_key", D + "prep_common.py", 'for k in ("split_key", "tree_id"):', 'for k in ("tree_id",):'),
    ("shard_input_order", D + "pretok_worker.py", "entries.sort(key=lambda e: (e[0], e[1]))",
     "entries.sort(key=lambda e: (e[2], e[3]))"),
    ("bucket_depends_on_chunk", D + "pretok_worker.py", "rows.append((key % nb,",
     'rows.append(((key + job["index"]) % nb,'),
    ("key_ignores_seed", D + "prep_common.py", 'f"{FORMAT_VERSION}:{seed}:{source}:{doc_id}"',
     'f"{FORMAT_VERSION}:{source}:{doc_id}"'),
    ("assistant_loss_off", D + "pretok_worker.py", 't = {"role": role, "ids": ids}',
     't = {"role": role, "ids": ids, "loss": role != "assistant"}'),
    ("system_dropped", D + "pretok_worker.py", 'if rec.get("system"):\n            turns.append',
     'if False:\n            turns.append'),
    ("empty_doc_kept", D + "pretok_worker.py", 'cnt["empty_docs"] += 1\n                    continue',
     'cnt["empty_docs"] += 1'),
    ("shard_sha_wrong", D + "pretok_worker.py", "h.update(data)", "h.update(data[:-1])"),
    ("top_tokens_wrong", D + "pretokenize.py", '"tokens": m["counts"]["tokens"],',
     '"tokens": m["counts"]["tokens"] - m["counts"]["docs"],'),
    ("harness_eot_wrong", D + "pretokenize.py", '{"eot_id": ti["eot_id"]}', '{"eot_id": ti["pad_id"]}'),
    ("oodh_check_off", D + "prep_common.py", 'meta = rec.get("meta") or {}\n    tid = meta.get("tree_id")',
     'return\n    meta = rec.get("meta") or {}\n    tid = meta.get("tree_id")'),
    ("encode_specials_on", D + "prep_common.py", "tok.encode_special_tokens = True",
     "tok.encode_special_tokens = False"),
    ("first_word_scored", D + "evalwin.py", "for s, e in cut(b, int(bnd[0]), W, lookback, bnd):",
     "for s, e in cut(b, 1, W, lookback, bnd):"),
    ("context_too_long", D + "evalwin.py", "lo = s - C\n", "lo = s - 2 * C\n"),
    ("cut_ignores_boundaries", D + "evalwin.py", "e = int(bnd[i])", "e = char_floor(b, x)"),
    ("chat_budget_not_spent", D + "evalwin.py", "ct, cc, rem = j, 0, rem - L", "ct, cc, rem = j, 0, rem"),
    ("system_turn_scored", D + "evalwin.py", 'if turns[t]["role"] not in roles:', "if False:"),
    ("window_bytes_wrong", D + "evalwin.py", '{"c": c, "s": s, "e": e, "b": e - s}', '{"c": c, "s": s, "e": e, "b": e - c}'),
    ("chat_no_target_role", D + "evalwin.py", "    ids.append(role_ids[roles[t]])\n", "\n"),
    ("chat_no_end_token", D + "evalwin.py", "        ids.append(end_id)\n", "\n"),
    ("fit_drops_right", D + "evalwin.py", "return ids[drop:], n_ctx - drop, True", "return ids[:max_len], n_ctx, True"),
    ("eval_oodh_off", D + "eval_sets.py", "        C.check_oodh(r)\n", "\n"),
    ("eval_overlap_ignored", D + "eval_sets.py", "if any(ov.values()):", "if False:"),
    ("eval_split_key_unchecked", D + "eval_sets.py", 'ov["split_key"] += any(k in keys or k in ids for k in ks)',
     'ov["split_key"] += 0'),
    ("eval_hash_unchecked", D + "eval_sets.py", 'ov["text_sha1"] += C.text_hash', 'ov["text_sha1"] += 0 * len'),
    ("eval_text_not_cleaned", D + "eval_sets.py", 'text, changed = C.clean_text(r.get("text") or "")',
     'text, changed = r.get("text") or "", False'),
    ("bpb_target_shift", D + "bpb.py", "tgt[r, n_ctx - 1:len(ids) - 1]", "tgt[r, n_ctx:len(ids)]"),
    ("bpb_nats", D + "bpb.py", "bits = float(nll[sel].sum() / math.log(2))", "bits = float(nll[sel].sum())"),
    ("bpb_per_token", D + "bpb.py", '"bpb": bits / nb if nb else None', '"bpb": bits / nt if nt else None'),
    ("bpb_pad_scored", D + "bpb.py", "tgt = torch.full((len(rows), T), -100,", "tgt = torch.full((len(rows), T), pad_id,"),
    ("bpb_role_flip", D + "bpb.py", "if it[3] == r]))", "if it[3] != r]))"),
    ("bpb_subsample_first", D + "bpb.py", 'wins = sorted(wins, key=lambda w: (w["h"], w["d"], w["k"]))[:max_windows]',
     "wins = wins[:max_windows]"),
    ("bpb_no_prefix_strip", D + "bpb.py", 'k.removeprefix("_orig_mod.")', "k"),
    ("ngram_ignores_eot", D + "ngram_overlap.py", 'return src, _found(np.memmap(path, dtype="<u2", mode="r"))',
     'a = np.memmap(path, dtype="<u2", mode="r")\n        return src, _found(np.asarray(a)[np.asarray(a) != 1])'),
    ("ngram_chat_no_separators", D + "ngram_overlap.py", 'buf += [G["role_ids"][t["role"]], *t["ids"], G["end_id"]]',
     'buf += [*t["ids"]]'),
    ("ngram_stride_ignored", D + "ngram_overlap.py", "hashes(np.array(e.ids, dtype=np.int64), n)[::stride])",
     "hashes(np.array(e.ids, dtype=np.int64), n))"),
    ("ngram_once_seen_off", D + "ngram_overlap.py", "once = cnt[np.searchsorted(uq, allh)] == 1",
     "once = cnt[np.searchsorted(uq, allh)] >= 1"),
    ("ngram_system_sampled", D + "ngram_overlap.py", 'for t in d["turns"] if t["role"] in ("user", "assistant")]',
     'for t in d["turns"]]'),
    ("ngram_hash_short", D + "ngram_overlap.py", "        for j in range(n):\n", "        for j in range(n - 1):\n"),
    ("ngram_source_misattributed", D + "ngram_overlap.py", "got.setdefault(src, []).append(found)",
     'got.setdefault("web" if src == "chat" else src, []).append(found)'),
    ("ngram_prefilter_misses", D + "ngram_overlap.py", "            k = np.minimum(np.searchsorted(S, c), len(S) - 1)",
     "            c = c[:-1]\n            k = np.minimum(np.searchsorted(S, c), len(S) - 1)"),
]


def run_one(mid: str, rel: str, old: str, new: str, python: str) -> tuple[str, float]:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for c in COPY:
            src, dst = os.path.join(ROOT, c), os.path.join(tmp, c)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            (shutil.copytree if os.path.isdir(src) else shutil.copy2)(
                src, dst, **({"ignore": shutil.ignore_patterns("__pycache__", "runs")} if os.path.isdir(src) else {}))
        p = os.path.join(tmp, rel)
        s = open(p, encoding="utf-8").read()
        assert s.count(old) == 1, f"{mid}: pattern found {s.count(old)} times in {rel}"
        open(p, "w", encoding="utf-8").write(s.replace(old, new))
        r = subprocess.run([python, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "tests"],
                           cwd=os.path.join(tmp, "data_prep"), capture_output=True, text=True)
    verdict = {0: "SURVIVED", 1: "killed"}.get(r.returncode, f"INVALID(exit {r.returncode})")
    return verdict, time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--id", default=None)
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)
    ms = [m for m in MUTANTS if a.id in (None, m[0])]
    i, n = map(int, a.part.split("/"))
    ms = ms[i - 1::n]
    if a.list:
        print("\n".join(m[0] for m in ms))
        return 0
    bad = 0
    for mid, rel, old, new in ms:
        v, dt = run_one(mid, rel, old, new, a.python)
        bad += v != "killed"
        print(f"{mid:28s} {v:18s} {dt:5.1f}s", flush=True)
    print(f"{len(ms) - bad}/{len(ms)} killed", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
