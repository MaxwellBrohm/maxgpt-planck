"""core_status.py: one-screen progress of the core v0 job (run_core_v0.sh), read-only.

    python corpus/core_status.py [--stage1 DIR] [--final DIR] [--json]

Per source: manifest files done / total (pass A, stage1 LEDGER.jsonl), documents and text bytes in
and kept, the main drop reasons (date_gate among them), and, once the tier is finalized
(FINAL/_tiers/tier<K>.json), the near-dedup, boilerplate and pass D drops and the final size.
Throughput is raw (compressed) manifest bytes committed per hour of wall clock over the last
--window-h hours, and the ETA divides the bytes left by it; lock waits are included.
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def load_events(stage1):
    out = []
    p = os.path.join(stage1, "LEDGER.jsonl")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out


def ts(s):
    return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%S"))


def summary(stage1, final, manifest, window_h=2.0):
    man = json.load(open(manifest))["files"]
    total, size, tier = {}, {}, {}
    for e in man:
        if e.get("role", "core") == "core":
            total[e["source"]] = total.get(e["source"], 0) + 1
            size[e["source"]] = size.get(e["source"], 0) + e["size"]
            tier[e["source"]] = e["tier"]
    ev = load_events(stage1)
    done = {e["fid"]: e for e in ev if e["event"] == "file_done"}
    failed = {e["fid"] for e in ev if e["event"] == "file_failed"} - set(done)
    by = {s: {"tier": tier[s], "files": 0, "files_total": total[s], "raw_gb_total":
              round(size[s] / 1e9, 2), "raw_gb_done": 0.0, "docs_in": 0, "docs_kept": 0,
              "text_gb_in": 0.0, "text_gb_kept": 0.0, "dropped": {}} for s in total}
    for e in done.values():
        b, st = by[e["source"]], e["stats"]
        b["files"] += 1
        b["raw_gb_done"] = round(b["raw_gb_done"] + e["size"] / 1e9, 3)
        b["docs_in"] += st["docs_in"]
        b["docs_kept"] += st["docs_kept"]
        b["text_gb_in"] = round(b["text_gb_in"] + st["bytes_in"] / 1e9, 3)
        b["text_gb_kept"] = round(b["text_gb_kept"] + st["bytes_kept"] / 1e9, 3)
        for r, d in st["dropped"].items():
            b["dropped"][r] = b["dropped"].get(r, 0) + d["docs"]
    fsrc = {f"{e['dataset']}/{e['path']}": e["source"] for e in man}
    for f in failed:
        if fsrc.get(f) in by:
            by[fsrc[f]].setdefault("failed", []).append(f)
    tdir = os.path.join(final, "_tiers")
    for f in sorted(os.listdir(tdir)) if os.path.isdir(tdir) else []:
        if f.endswith(".json") and f.startswith("tier"):
            r = json.load(open(os.path.join(tdir, f)))
            for src, st in r["sources"].items():
                if src in by:
                    by[src]["final"] = {
                        "docs_in": st["docs_in"], "docs_kept": st["docs_kept"],
                        "text_gb_kept": round(st["bytes_kept"] / 1e9, 3),
                        "zst_gb": round(st["bytes_compressed"] / 1e9, 3),
                        "dropped": {k: v["docs"] for k, v in st["dropped"].items()},
                        "stripped_mb": round(st["stripped"]["bytes"] / 1e6, 1),
                        "near_dup_rate": round(st["dropped"].get("dup_near", {}).get("docs", 0)
                                               / max(1, st["docs_in"]), 4),
                        "near_dup_lead": st.get("near_dup_lead"), "finished": r["finished"]}
    now = time.time()
    recent = [e for e in done.values() if now - ts(e["ts"]) <= window_h * 3600]
    starts = [ts(e["ts"]) for e in ev if e["event"] == "run_start"]
    span = min(window_h * 3600, now - min(starts)) if starts else 0
    rate = sum(e["size"] for e in recent) / span * 3600 / 1e9 if span > 0 else 0.0
    left = sum(size.values()) / 1e9 - sum(b["raw_gb_done"] for b in by.values())
    st = {}
    sp = os.path.join(stage1, "STATUS.json")
    if os.path.exists(sp):
        st = json.load(open(sp))
    return {"now": time.strftime("%Y-%m-%dT%H:%M:%S"), "state": st.get("state"),
            "status_updated": st.get("updated"), "this_run": st.get("this_run"),
            "free_gib": round(shutil.disk_usage(stage1 if os.path.isdir(stage1) else "/").free
                              / 2**30, 1),
            "raw_gb_left": round(left, 2), "raw_gb_per_hour": round(rate, 2),
            "eta_hours": round(left / rate, 1) if rate else None,
            "last_event": ev[-1]["event"] + " " + ev[-1]["ts"] if ev else None,
            "failed_now": sorted(failed), "sources": by}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage1", default="~/planck/data/core_v0_work/stage1")
    ap.add_argument("--final", default="~/planck/data/core_v0")
    ap.add_argument("--manifest", default=os.path.join(HERE, "fetch", "core_v0_manifest.json"))
    ap.add_argument("--window-h", type=float, default=2.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    s = summary(os.path.expanduser(a.stage1), os.path.expanduser(a.final), a.manifest, a.window_h)
    if a.json:
        print(json.dumps(s, indent=1))
        return 0
    print({k: v for k, v in s.items() if k != "sources"})
    for src, b in sorted(s["sources"].items(), key=lambda x: (x[1]["tier"], x[0])):
        top = dict(sorted(b["dropped"].items(), key=lambda x: -x[1])[:5])
        print(f"T{b['tier']} {src:13s} files {b['files']:3d}/{b['files_total']:3d} raw "
              f"{b['raw_gb_done']:6.2f}/{b['raw_gb_total']:6.2f} GB  docs {b['docs_kept']:>10,}/"
              f"{b['docs_in']:>10,}  text {b['text_gb_kept']:7.2f}/{b['text_gb_in']:7.2f} GB  "
              f"drop {top}")
        if "final" in b:
            f = b["final"]
            print(f"   final: docs {f['docs_kept']:,}  text {f['text_gb_kept']} GB  zst "
                  f"{f['zst_gb']} GB  near-dup {f['near_dup_rate']:.2%}  drop {f['dropped']}  "
                  f"stripped {f['stripped_mb']} MB  lead {f['near_dup_lead']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
