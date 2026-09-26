"""Tokenizer v0 CPU measurements (PLAN Phase 1): bytes/token table, P-098, P-101, P-097. P-100 is out.

  python tokenizer/checks.py --family DIR --sample SAMPLE --out OUT [--prefix tok_v0] [--base-size 8192]
        [--superword-n 100,300,500] [--chat-sources oasst2,dolly,irc] [--mine-mb 10] [--max-docs N]

DIR holds train_bpe.py's output: the nested family <prefix>_<size>.json and, for P-098, the separately
trained <prefix>_sep_<size>.json (each file's vocab size is read from the file, not the name).
SAMPLE is make_tok_sample.py's directory: heldout/<source>.jsonl is what every measurement scores,
train/<source>.jsonl (chat sources only, first --mine-mb MB of each) is where P-101 mines phrases.
Writes OUT/<prefix>_checks.json and OUT/<prefix>_checks.md. Exact definitions: checks_bpt.py (table,
P-098), checks_superword.py (P-101), checks_names.py (P-097).
P-100 (8k as a subset of the teacher's vocabulary) needs the teacher, which the Phase 1 pilot has not
picked; it is recorded as out of scope for v0.
CPU only, no model is loaded. Real-sample runs go on the PC under the heavy lock (see train_bpe.py).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks_bpt  # noqa: E402
import checks_names  # noqa: E402
import checks_superword  # noqa: E402
import sample_io  # noqa: E402
import spec  # noqa: E402
from checks_report import to_markdown  # noqa: E402
from health import load_heldout  # noqa: E402

CHAT_SOURCES = ("oasst2", "dolly", "irc")
P100 = {"status": "out_of_scope",
        "reason": "P-100 compares the free 8k BPE with an 8k built from the teacher's vocabulary; the "
                  "teacher is picked by the Phase 1 pilot, which has not run, so v0 cannot measure it"}


def _vocab(path: str) -> int:
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path).get_vocab_size()


def find_files(d: str, prefix: str) -> tuple[dict[int, str], dict[int, str]]:
    """({vocab: path} nested family, {vocab: path} separately trained), by the train_bpe.py names."""
    fam, sep = {}, {}
    for p in sorted(glob.glob(os.path.join(d, f"{prefix}_*.json"))):
        name = os.path.basename(p)
        if name.endswith("_manifest.json"):
            continue
        target = sep if name.startswith(f"{prefix}_sep_") else fam
        v = _vocab(p)
        assert v not in target, f"two files of vocab {v}: {target[v]} and {p}"
        target[v] = p
    return fam, sep


def load_train(sample: str, sources, max_mb: float) -> dict[str, list[str]]:
    """The first max_mb MB (whole docs) of each chat source's training file."""
    files = sample_io.source_files(sample, "train")
    out = {}
    for s in sources:
        if s not in files:
            continue
        docs, nb = [], 0
        for t in sample_io.iter_file(files[s]):
            if nb >= max_mb * 1e6:
                break
            docs.append(t)
            nb += len(t.encode("utf-8"))
        out[s] = docs
    return out


def run(fam: dict[int, str], sep: dict[int, str], heldout: dict[str, list[str]], train_chat: dict[str, list[str]],
        base_size: int = 8192, ns=checks_superword.DEFAULT_NS) -> dict:
    from tokenizers import Tokenizer
    toks = {v: Tokenizer.from_file(p) for v, p in fam.items()}
    res, t0 = {"sizes": sorted(fam), "sources": sorted(heldout), "seconds": {}}, time.time()
    res["bytes_per_token"] = checks_bpt.bpt_table(toks, heldout)
    res["seconds"]["table"] = round(time.time() - t0, 1)

    t0 = time.time()
    both = sorted(set(fam) & set(sep))
    if both:
        v = both[0]
        sep_stats = checks_bpt.source_stats(Tokenizer.from_file(sep[v]), heldout)
        res["P-098"] = dict(size=v, **checks_bpt.p098(res["bytes_per_token"][v], sep_stats,
                                                      checks_bpt.same_model(fam[v], sep[v])))
    else:
        res["P-098"] = {"status": "not_run", "reason": "no separately trained file with a family size"}
    res["seconds"]["P-098"] = round(time.time() - t0, 1)

    t0 = time.time()
    held_chat = {s: heldout[s] for s in train_chat if s in heldout}
    if base_size not in fam:
        res["P-101"] = {"status": "not_run", "reason": f"no family member of size {base_size}"}
    elif not held_chat:
        res["P-101"] = {"status": "not_run", "reason": "no chat source in both train and heldout"}
    else:
        with open(fam[base_size], encoding="utf-8") as f:
            base_json = json.load(f)
        texts = [t for s in sorted(train_chat) for t in train_chat[s]]
        res["P-101"] = dict(chat_sources=sorted(held_chat), mined_from={s: len(d) for s, d in sorted(train_chat.items())},
                            **checks_superword.p101(base_json, texts, held_chat, ns))
    res["seconds"]["P-101"] = round(time.time() - t0, 1)

    t0 = time.time()
    res["P-097"] = checks_names.p097(toks)
    res["seconds"]["P-097"] = round(time.time() - t0, 1)
    res["P-100"] = P100
    return res


def provenance(fam: dict[int, str], sep: dict[int, str], sample: str, args: dict) -> dict:
    """Basenames and hashes only: the results may be committed to the public repo."""
    import tokenizers
    held = sample_io.source_files(sample, "heldout")
    sm = os.path.join(sample, "manifest.json")
    return {"tokenizers_version": tokenizers.__version__, "family": spec.FAMILY, "args": args,
            "tokenizer_files": {os.path.basename(p): sample_io.sha256_file(p)
                                for p in sorted(list(fam.values()) + list(sep.values()))},
            "heldout_files": {s: sample_io.sha256_file(p) for s, p in held.items()},
            "sample_manifest_sha256": sample_io.sha256_file(sm) if os.path.exists(sm) else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default=spec.FAMILY)
    ap.add_argument("--base-size", type=int, default=8192)
    ap.add_argument("--superword-n", default=",".join(str(n) for n in checks_superword.DEFAULT_NS))
    ap.add_argument("--chat-sources", default=",".join(CHAT_SOURCES))
    ap.add_argument("--mine-mb", type=float, default=10.0, help="per chat source, from its training file")
    ap.add_argument("--max-docs", type=int, default=None, help="cap held-out docs per source")
    a = ap.parse_args(argv)
    fam, sep = find_files(a.family, a.prefix)
    if not fam:
        print(f"no {a.prefix}_*.json in the family directory", file=sys.stderr)
        return 2
    heldout = load_heldout(os.path.join(a.sample, "heldout"), a.max_docs)
    train_chat = load_train(a.sample, [s for s in a.chat_sources.split(",") if s], a.mine_mb)
    ns = [int(x) for x in a.superword_n.split(",") if x]
    res = run(fam, sep, heldout, train_chat, a.base_size, ns)
    args = {k: os.path.basename(os.path.normpath(v)) if k in ("family", "sample", "out") else v
            for k, v in vars(a).items()}
    res["provenance"] = provenance(fam, sep, a.sample, args)
    os.makedirs(a.out, exist_ok=True)
    jp, mp = (os.path.join(a.out, f"{a.prefix}_checks.{ext}") for ext in ("json", "md"))
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(to_markdown(res))
    print(jp)
    print(mp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
