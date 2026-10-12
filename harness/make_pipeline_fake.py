"""Produce a small FAKE pipeline run for the from_pipeline adapter tests: the real pipeline driver
(pipeline/driver.py) renders N shard skeletons through its FAKE teacher stub (pipeline/fake_server.py,
in-process, loopback only) into OUT/. Nothing is downloaded, no model is loaded, no real server is
contacted: the pipeline's client refuses any endpoint that is not the stub.

  python3 -B make_pipeline_fake.py OUT [--n 200] [--seed adapter] [--register RM]

Stdlib only (the pipeline is stdlib python3). OUT gets the driver's usual run directory:
accepted/accepted-00000.jsonl, rejects/, skeletons/, run.json, yield.jsonl. Every accepted record is
FAKE and non-trainable (blocked FAKE_PROVENANCE + DECON_PENDING). Prints one JSON summary line.
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
PIPE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline")


def make(out: str, n: int = 200, seed: str = "adapter", register: str = "RM") -> dict:
    if PIPE not in sys.path:
        sys.path.insert(0, PIPE)
    import driver
    import fake_server
    import skeleton
    import stub_plan
    import teacher_client as TC

    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{out} exists and is not empty")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    skels = skeleton.shard(seed, n, register)
    skel_file = os.path.join(out, "input_skeletons.jsonl")
    with open(skel_file, "w", encoding="utf-8") as f:
        for s in skels:
            f.write(json.dumps(s) + "\n")
    plan = stub_plan.plan(skels, http_tries=4)
    stub, server, url = fake_server.start(plan)
    try:
        client = TC.TeacherClient(url, plan["model"], http_tries=4, backoff=0, timeout=30)
        snap = driver.Driver({"out": os.path.join(out, "run"), "n": n, "skeletons": skel_file,
                              "concurrency": 8, "quiet": True, "fsync": False, "log_every": 5.0},
                             client).run()
    finally:
        fake_server.stop(server)
    return {"out": os.path.join(out, "run"), "skeletons": n, "attempts": snap["attempts"],
            "accepted": snap["accepted"], "rejected": snap["rejected"],
            "expected_accepted": plan["expected"]["accepted"], "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", default="adapter")
    ap.add_argument("--register", default="RM")
    a = ap.parse_args()
    print(json.dumps(make(a.out, a.n, a.seed, a.register)))
