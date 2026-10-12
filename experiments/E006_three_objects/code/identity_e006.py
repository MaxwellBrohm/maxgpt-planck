"""E006 identity checks before any model job (notes.txt WHAT STAYS IDENTICAL 1, 5, 6; queue step 0). No model, no
torch import (versions are read with importlib.metadata). Every line prints PASS or FAIL; exit 0 only if all pass.
  copies     every file in ../copies/copied_from_e005_sha256.txt: E006's copy, the listed hash and E005's source
             (read-only) agree; the AL items and their .sha256 likewise
  snapshot   the pinned SmolLM2 snapshot (revision 12fd25f7...) under HF_HOME holds every file the loaders read, and
             model.safetensors hashes to 5af571cb... (E005's)
  pins       torch 2.13.0 (any +cu build), transformers 5.15.1, tokenizers 0.22.2, safetensors 0.8.0
  items      ../big/big_items.jsonl and ../al/al_items.jsonl equal their .sha256 files
  replay     ../replay/replay_pool.jsonl hashes to ../replay_ref/replay_pool.sha256 (the pool fixed at writing), and
             replay_pool_ids.txt and exposed_kbig.json equal the reference copies
  refs       E005's and E004's saved weights (s1-s5) hash to the values E005's AL runs recorded (al/out/*__run.json)
usage: python -B identity_e006.py [--no-refs] [--no-replay]"""
import glob
import hashlib
import json
import os
import sys
from importlib import metadata

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
EXPS = os.path.dirname(EXP)
E5 = os.path.join(EXPS, "E005_alias_eot")
SLUG = "HuggingFaceTB__SmolLM2-135M-Instruct"
REV = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
BASE_SHA = "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
SNAP_FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "vocab.json", "merges.txt")
PINS = {"torch": "2.13.0", "transformers": "5.15.1", "tokenizers": "0.22.2", "safetensors": "0.8.0"}
FAIL = []


def ok(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def listed(path):
    return [(l.split()[0], l.split()[1]) for l in open(path) if l.strip()]


def check_copies():
    rows = listed(os.path.join(EXP, "copies", "copied_from_e005_sha256.txt"))
    bad = [n for h, n in rows if not (sha(os.path.join(HERE, n)) == h == sha(os.path.join(E5, "code", n)))]
    ok(len(rows) == 98 and not bad, f"{len(rows)} copied E005 code files equal the list and their E005 source {bad[:5]}")
    for n in ("al_items.jsonl", "al_items.sha256"):
        ok(sha(os.path.join(EXP, "al", n)) == sha(os.path.join(E5, "al", n)), f"al/{n} equals E005's")


def check_snapshot():
    root = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub",
                        "models--" + SLUG.replace("__", "--"), "snapshots", REV)
    miss = [f for f in SNAP_FILES if not os.path.isfile(os.path.join(root, f))]
    ok(not miss, f"snapshot {REV[:8]} holds {', '.join(SNAP_FILES)} (missing: {miss})")
    if not miss:
        ok(sha(os.path.join(root, "model.safetensors")) == BASE_SHA, "snapshot model.safetensors sha256 5af571cb...")


def check_pins():
    for m, want in PINS.items():
        try:
            v = metadata.version(m)
        except metadata.PackageNotFoundError:
            v = None
        ok(v is not None and v.split("+")[0] == want, f"{m} {v} (pinned {want})")


def check_items():
    for d, n in (("big", "big_items"), ("al", "al_items")):
        p = os.path.join(EXP, d, n + ".jsonl")
        ok(sha(p) == open(os.path.join(EXP, d, n + ".sha256")).read().split()[0], f"{d}/{n}.jsonl equals its sha256")


def check_replay():
    ref, got = os.path.join(EXP, "replay_ref"), os.path.join(EXP, "replay")
    ok(os.path.isfile(os.path.join(got, "replay_pool.jsonl")) and
       sha(os.path.join(got, "replay_pool.jsonl")) == open(os.path.join(ref, "replay_pool.sha256")).read().split()[0],
       "replay pool equals the pool fixed at writing (replay_ref/replay_pool.sha256)")
    for n in ("replay_pool_ids.txt", "exposed_kbig.json"):
        p = os.path.join(got, n)
        ok(os.path.isfile(p) and open(p).read() == open(os.path.join(ref, n)).read(), f"replay/{n} equals replay_ref's")


def check_refs():
    want = {}
    for f in glob.glob(os.path.join(E5, "al", "out", "e00[45]_s[1-5]__run.json")):
        tag = os.path.basename(f).split("__")[0]
        want[tag] = json.load(open(f))["weights_sha256"]
    ok(len(want) == 10, f"10 reference hashes in E005's al/out ({len(want)})")
    for tag, h in sorted(want.items()):
        exp_dir = "E005_alias_eot" if tag.startswith("e005") else "E004_general_updating"
        p = os.path.join(EXPS, exp_dir, "weights", f"{SLUG}__{tag.split('_')[1]}", "model.safetensors")
        ok(os.path.isfile(p) and sha(p) == h, f"reference {tag} weights hash {h[:8]}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    check_copies()
    check_snapshot()
    check_pins()
    check_items()
    if "--no-replay" not in argv:
        check_replay()
    if "--no-refs" not in argv:
        check_refs()
    print("RESULT: " + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILED"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
