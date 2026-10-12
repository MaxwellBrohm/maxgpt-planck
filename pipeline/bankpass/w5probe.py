"""W5 bank probe (BANKPASS s6: 60 skeletons per teacher on the stage P banks, dry, own dir, not pilot data), run on
the FROZEN banks (w4freeze's banks_v1) instead of W3's provisional snapshot (probe.py, which never ran: its master
stopped on a STOP file). DRY: nothing here is training or pilot data.

  build  (PC CPU) load and install the frozen set, build 60 RM skeletons (shard bankprobe-1005), and map every topic
         text to its kept stage P intents (aux bank "intent") -> <out>/skels60.jsonl, intents.json, build.json
  run    (inside one gpu.lock hold, against bankserve) the same frozen set installed in the driver process (so the
         checker reads the real topic words and pools), topic guidance from the intents (probe.patch_guidance:
         forms 0 to 3 and 5; "react briefly" and digressions keep their program wording), teachers/drive.py with one
         attempt per skeleton and probe.py's presets and concurrency -> <out>/dry/<teacher>/run

    python -m bankpass.w5probe build --banks BANKS_V1 --out PROBE
    python -m bankpass.w5probe run --banks BANKS_V1 --out PROBE --teacher T --port P"""
import argparse
import collections
import json
import os
import sys

from bankpass import load, probe, store

SHARD = "bankprobe-1005"


def intents(bank_dir):
    """{topic text: [kept intents]} from the frozen aux intent bank."""
    out = collections.defaultdict(list)
    path = os.path.join(bank_dir, "intent.jsonl")
    for r in store.read_jsonl(path, tolerate_torn_tail=False) if os.path.exists(path) else []:
        if r["status"] == "kept":
            out[r["features"]["topic"]].append(r["text"])
    return dict(out)


def build(bank_dir, out, n=probe.N):
    import skeleton_shard as SS
    bs = load.from_dir(bank_dir)
    undo = load.install(bs)
    try:
        sks = SS.shard(SHARD, n, "RM")
    finally:
        undo()
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "skels60.jsonl"), "w", encoding="utf-8") as f:
        for sk in sks:
            f.write(json.dumps(sk, sort_keys=True) + "\n")
    ints = intents(bank_dir)
    with open(os.path.join(out, "intents.json"), "w", encoding="utf-8") as f:
        json.dump(ints, f, sort_keys=True)
    info = {"skeletons": len(sks), "shard": SHARD, "bank_manifest_sha256": store.sha256_file(
        os.path.join(bank_dir, "manifest.json")), "skels_sha256": store.sha256_file(os.path.join(out, "skels60.jsonl")),
        "intents_topics": len(ints), "fake_banks": bs.fake_banks, "skeleton_fake": sum(s["provenance"]["fake"]
                                                                                       for s in sks)}
    with open(os.path.join(out, "build.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, sort_keys=True, indent=1)
    return info


def run(bank_dir, out, teacher, port):
    with open(os.path.join(out, "build.json"), encoding="utf-8") as f:
        info = json.load(f)
    if store.sha256_file(os.path.join(bank_dir, "manifest.json")) != info["bank_manifest_sha256"]:
        raise SystemExit("the bank set changed since the probe was built; rebuild it")
    load.install(load.from_dir(bank_dir))
    with open(os.path.join(out, "intents.json"), encoding="utf-8") as f:
        probe.patch_guidance(json.load(f))
    d = os.path.join(out, "dry", teacher)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "DRY.txt"), "w") as f:
        f.write("dry: bank pass W5 probe renders on the frozen banks (2026-10-05); not training or pilot data\n")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "teachers"))
    import drive
    return drive.main(["--serve", "--allow-real-teacher", "--endpoint", f"http://127.0.0.1:{port}", "--out",
                       os.path.join(d, "run"), "--skeletons", os.path.join(out, "skels60.jsonl"), "--n",
                       str(info["skeletons"]), "--max-attempts", "1", "--structured", "labels_exact", "--ban-dashes",
                       "--preset", probe.PRESET[teacher], "--concurrency", str(probe.CONC[teacher]), "--timeout",
                       "1800", "--http-tries", "3", "--log-every", "20"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run"])
    ap.add_argument("--banks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--port", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        print(json.dumps(build(a.banks, a.out)))
        return 0
    return run(a.banks, a.out, a.teacher, a.port)


if __name__ == "__main__":
    sys.exit(main())
