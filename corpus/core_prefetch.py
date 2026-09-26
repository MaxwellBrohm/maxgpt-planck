"""core_prefetch.py: download (never extract) the manifest files of later tiers while
run_core_v0.sh works on an earlier one, so a heavy-lock wait does not also stall the network.

    python corpus/core_prefetch.py --out STAGE1 --raw RAW --starter STARTER --tiers 2 3 4 5 \\
        [--window-gb 100] [--dl-threads 4] [--url-rewrite FROM=TO]

Same fetch code as stream_core.py (stream_fetch.Prefetcher: resume, checksum, disk guard), same
raw layout, so stream_core.py later finds each file, verifies it again and uses it. At most
--window-gb of these files are kept on disk (downloaded or downloading). Only one prefetcher runs
at a time (flock on STAGE1/../prefetch.lock). It must never run on a tier stream_core.py is
processing, or two downloaders could write one file: run_core_v0.sh stops it (SIGTERM, which
terminates curl and leaves a resumable .part) before each stream_core.py run and restarts it with
the tiers after that one. It exits by itself when every file is ready or failed, or on STAGE1/STOP.
No heavy lock: downloads and checksums only, at nice 15.
"""
import argparse
import fcntl
import json
import os
import signal
import sys
import time

import stream_core as C
import stream_fetch as F


def pending(cfg):
    manifest, _, _ = C._load(cfg)
    done, lp = set(), os.path.join(cfg["out"], "LEDGER.jsonl")
    if os.path.exists(lp):
        for line in open(lp, encoding="utf-8"):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                break
            if ev.get("event") == "file_done":
                done.add(ev["fid"])
    return C.select(cfg, manifest, done)[0]      # starter-local files are only verified


def run(cfg, log=print, poll=5.0):
    cfg = C.check_config(cfg)
    os.makedirs(cfg["out"], exist_ok=True)
    lock = open(os.path.join(os.path.dirname(cfg["out"]), "prefetch.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("core_prefetch: another prefetcher is running")
        return "busy"
    items = pending(cfg)
    log(f"core_prefetch: {len(items)} files of tiers {cfg['tiers']}, "
        f"{sum(i['entry']['size'] for i in items) / 1e9:.1f} GB, window {cfg['window_gb']} GB")
    stop = {"why": None}

    def on_signal(*_):
        stop["why"] = "signal"

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_signal)
    pf = F.Prefetcher(items, cfg, log)
    pf.start()
    try:
        while stop["why"] is None:
            if os.path.exists(os.path.join(cfg["out"], "STOP")):
                stop["why"] = "stop_file"
            elif all(it["state"] in ("ready", "failed") for it in items):
                stop["why"] = "all_fetched"
            else:
                time.sleep(poll)
    finally:
        pf.stop()
    n = {}
    for it in items:
        n[it["state"]] = n.get(it["state"], 0) + 1
    for it in items:
        if it["state"] == "failed":
            log(f"core_prefetch: failed {it['fid']}: {it.get('error')}")
    log(f"core_prefetch: exit ({stop['why']}), states {n}")
    return stop["why"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--starter")
    ap.add_argument("--manifest", default=C.DEFAULTS["manifest"])
    ap.add_argument("--tiers", type=int, nargs="+", required=True)
    ap.add_argument("--window-gb", type=float, default=100)
    ap.add_argument("--dl-threads", type=int, default=4)
    ap.add_argument("--url-rewrite", action="append", default=[], metavar="FROM=TO")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    cur = os.nice(0)
    if cur < 15:
        os.nice(15 - cur)
    cfg = {k: v for k, v in vars(a).items() if v is not None and k != "url_rewrite"}
    cfg["url_rewrite"] = dict(x.split("=", 1) for x in a.url_rewrite)
    return 0 if run(cfg) in ("all_fetched", "signal", "stop_file") else 1


if __name__ == "__main__":
    sys.exit(main())
