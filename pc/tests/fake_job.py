"""A stand-in training job for the runner tests.

Behaves like harness/train.py where the runner cares: writes log.jsonl progress lines,
checkpoints and exits 0 on SIGTERM (marker file 'checkpointed'). It can also heat or fill
the fake GPU mid-run by editing $FAKE_SMI_STATE, ignore SIGTERM, or leave a child process.

  python fake_job.py OUT_DIR [--steps 40] [--rc 0] [--ignore-term] [--child]
                             [--smi-at 5 --smi '{"temp_c": 95}']
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time


def touch(out, name, text=""):
    with open(os.path.join(out, name), "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--rc", type=int, default=0)
    ap.add_argument("--ignore-term", action="store_true")
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--smi-at", type=int, default=-1)
    ap.add_argument("--smi", default="{}")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.ignore_term:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        def on_term(*_):
            touch(a.out, "checkpointed")
            sys.exit(0)
        signal.signal(signal.SIGTERM, on_term)
    touch(a.out, "started", str(os.getpid()))
    print(f"fake job started pid {os.getpid()}", flush=True)
    if a.child:
        c = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        touch(a.out, "child.pid", str(c.pid))
    for i in range(a.steps):
        with open(os.path.join(a.out, "log.jsonl"), "a") as f:
            f.write(json.dumps({"step": i + 1, "loss": round(5.0 - 0.01 * i, 4),
                                "tok_per_s": 1000.0, "tokens": 100 * (i + 1)}) + "\n")
        if i == a.smi_at:
            path = os.environ["FAKE_SMI_STATE"]
            with open(path) as f:
                st = json.load(f)
            st.update(json.loads(a.smi))
            with open(path + ".tmp", "w") as f:
                json.dump(st, f)
            os.replace(path + ".tmp", path)
        time.sleep(0.05)
    touch(a.out, "finished")
    sys.exit(a.rc)


if __name__ == "__main__":
    main()
