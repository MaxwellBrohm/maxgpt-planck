"""Train tokenizer family v0 (PLAN Phase 1, P-098): one byte-level BPE at the top size, nested truncations.

  python train_bpe.py nested   --sample SAMPLE --out OUT [--top 32768] [--sizes 2048,4096,8192,16384]
                               [--weight oasst2=3 ...] [--min-frequency 2]
      trains ONCE at --top and writes OUT/tok_v0_{2k,4k,8k,16k,32k}.json (each a prefix of the next)
  python train_bpe.py separate --sample SAMPLE --out OUT --size 8192 [--prefix tok_v0_sep]
      trains a separate tokenizer at one size (P-098's comparison arm), OUT/tok_v0_sep_8k.json
  python train_bpe.py truncate FULL.json --size 8192 --out OUT.json

SAMPLE is a directory written by corpus/make_tok_sample.py (layout in sample_io.py). spec.py fixes the special tokens,
the pre-tokenizer and the decoder. Every output file is re-loaded with Tokenizer.from_file and checked
(ids 0..V-1 contiguous, merge order, special ids). OUT/<prefix>_manifest.json records the sample
manifest hash, weights, trainer settings, tokenizers version and each file's sha256.
Heavy runs on the PC: nice -n 10, RAYON_NUM_THREADS=24, under ~/planck/locks/pc_heavy.lock.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sample_io  # noqa: E402
import spec  # noqa: E402
from truncate import check_merge_order, is_prefix, truncate_json, write_json  # noqa: E402

DEFAULT_MIN_FREQUENCY = 2


def _pieces(texts):
    for t in texts:
        t = sample_io.strip_specials(t)
        if t:
            yield t


def finalize(d: dict) -> dict:
    """Set the added-token flags (control tokens special, tags not) and assert the fixed layout."""
    want = {s: i for i, s in enumerate(spec.SPECIALS)}
    got = {a["content"]: a["id"] for a in d["added_tokens"]}
    assert got == want, f"added tokens {got} != {want}"
    for a in d["added_tokens"]:
        a["special"] = a["content"] in spec.CONTROL_TOKENS
        a["normalized"] = False
    vocab = d["model"]["vocab"]
    by_id = sorted(vocab, key=vocab.get)
    assert by_id[: spec.N_SPECIAL] == spec.SPECIALS, "special tokens are not ids 0..14 in order"
    assert by_id[spec.N_SPECIAL: spec.N_BASE] == spec.byte_alphabet(), "ids 15..270 are not the byte alphabet"
    assert d["normalizer"] is None and d["post_processor"] is None
    check_merge_order(d["model"])
    return d


def train_json(texts, vocab_size: int, min_frequency: int = DEFAULT_MIN_FREQUENCY,
               show_progress: bool = False) -> dict:
    """Train one BPE of exactly vocab_size ids; returns the tokenizer JSON as a dict."""
    from tokenizers import pre_tokenizers, trainers
    assert vocab_size > spec.N_BASE, vocab_size
    tok = spec.empty_tokenizer()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                  special_tokens=spec.added_tokens(), show_progress=show_progress,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(_pieces(texts), trainer)
    d = finalize(json.loads(tok.to_str()))
    n = len(d["model"]["vocab"])
    if n != vocab_size:
        raise RuntimeError(f"training stopped at {n} ids, asked for {vocab_size}: the sample has too few "
                           f"distinct pairs above min_frequency={min_frequency}")
    return d


def build_nested(texts, out_dir: str, top: int = spec.TOP_VOCAB, sizes=None, prefix: str = spec.FAMILY,
                 min_frequency: int = DEFAULT_MIN_FREQUENCY, show_progress: bool = False) -> dict:
    """Train once at `top`, write the top file and one truncation per size. Returns {size: path}."""
    sizes = sorted(set(sizes or [s for s in spec.NESTED_SIZES if s < top]) | {top})
    assert all(spec.N_BASE < s <= top for s in sizes), sizes
    full = train_json(texts, top, min_frequency, show_progress)
    os.makedirs(out_dir, exist_ok=True)
    out, members = {}, {}
    for s in sizes:
        d = full if s == top else truncate_json(full, s)
        path = os.path.join(out_dir, spec.file_name(s, prefix))
        write_json(d, path)
        out[s], members[s] = path, d
    for a, b in zip(sizes, sizes[1:]):
        assert is_prefix(members[a], members[b]), f"size {a} is not a prefix of size {b}"
    return out


def build_separate(texts, out_dir: str, size: int, prefix: str = spec.FAMILY + "_sep",
                   min_frequency: int = DEFAULT_MIN_FREQUENCY, show_progress: bool = False) -> str:
    d = train_json(texts, size, min_frequency, show_progress)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, spec.file_name(size, prefix))
    write_json(d, path)
    return path


def _weights(sample: str, pairs) -> dict[str, int]:
    w = sample_io.load_weights(sample)
    for p in pairs or []:
        k, v = p.split("=")
        w[k] = int(v)
    return w


def write_manifest(out_dir: str, prefix: str, paths: dict, sample: str, weights: dict, args: dict,
                   seconds: float) -> str:
    import tokenizers
    sm = os.path.join(sample, "manifest.json")
    m = {"family": spec.FAMILY, "prefix": prefix, "tokenizers_version": tokenizers.__version__,
         "specials": spec.SPECIALS, "control_tokens": spec.CONTROL_TOKENS, "tag_tokens": spec.TAG_TOKENS,
         "pretokenize_regex": spec.PRETOKENIZE_REGEX, "normalizer": None, "args": args,
         "weights": weights, "train_seconds": round(seconds, 1),
         "sample_manifest_sha256": sample_io.sha256_file(sm) if os.path.exists(sm) else None,
         "files": {os.path.basename(p): {"size": s, "sha256": sample_io.sha256_file(p)}
                   for s, p in sorted(paths.items())}}
    path = os.path.join(out_dir, f"{prefix}_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1, ensure_ascii=False)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("nested", "separate"):
        p = sub.add_parser(name)
        p.add_argument("--sample", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--weight", action="append", help="source=repeats, overrides the sample manifest")
        p.add_argument("--min-frequency", type=int, default=DEFAULT_MIN_FREQUENCY)
        p.add_argument("--progress", action="store_true")
    sub.choices["nested"].add_argument("--top", type=int, default=spec.TOP_VOCAB)
    sub.choices["nested"].add_argument("--sizes", default=",".join(str(s) for s in spec.NESTED_SIZES[:-1]))
    sub.choices["nested"].add_argument("--prefix", default=spec.FAMILY)
    sub.choices["separate"].add_argument("--size", type=int, required=True)
    sub.choices["separate"].add_argument("--prefix", default=spec.FAMILY + "_sep")
    t = sub.add_parser("truncate")
    t.add_argument("full")
    t.add_argument("--size", type=int, required=True)
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "truncate":
        with open(a.full, encoding="utf-8") as f:
            write_json(truncate_json(json.load(f), a.size), a.out)
        print(a.out)
        return 0
    spec.check_harness_spelling()
    w = _weights(a.sample, a.weight)
    t0 = time.time()
    texts = sample_io.iter_texts(a.sample, "train", w)
    if a.cmd == "nested":
        sizes = [int(x) for x in a.sizes.split(",") if x]
        paths = build_nested(texts, a.out, a.top, sizes, a.prefix, a.min_frequency, a.progress)
    else:
        paths = {a.size: build_separate(texts, a.out, a.size, a.prefix, a.min_frequency, a.progress)}
    # directory arguments are recorded by basename only: the manifest may be published with the files
    args = {k: os.path.basename(os.path.normpath(v)) if k in ("sample", "out") else v for k, v in vars(a).items()}
    man = write_manifest(a.out, a.prefix, paths, a.sample, w, args, time.time() - t0)
    for s, p in sorted(paths.items()):
        print(f"{s:>6} {p}")
    print(man)
    return 0


if __name__ == "__main__":
    sys.exit(main())
