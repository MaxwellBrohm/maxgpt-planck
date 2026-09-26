"""Fake interpreter for test_queue_e006.sh (never loads a model, never touches a GPU). queue_e006.sh calls it as
$E006_PY; `-c CODE` is passed to the real python3 (the queue's json reads); `-B script args` is logged to $E006T_CALLS
and simulated:
  guard_e006_pc.py --name N ... -- cmd   writes ../logs/N.log and N.guard.json; exit 1 if N is in $E006T_FAIL,
                                         killed "temp" (exit 9) if N is in $E006T_VERDICT, exit 3 with no files for
                                         the first k calls if "N:k" is in $E006T_REFUSE; a successful e006_ft_test
                                         job with --save 1 creates ../weights/<slug>__<tag>/
  any other script                       exit 1 if its name (without .py) is in $E006T_FAIL, else prints "ok"."""
import json
import os
import sys


def main():
    a = sys.argv[1:]
    if a and a[0] == "-c":
        os.execv(os.environ["E006T_REAL_PY"], [os.environ["E006T_REAL_PY"]] + a)
    if a and a[0] == "-B":
        a = a[1:]
    script = os.path.basename(a[0]) if a else ""
    env = lambda k: os.environ.get(k, "").split()
    with open(os.environ["E006T_CALLS"], "a") as f:
        f.write(("GUARD " + a[a.index("--name") + 1] if script == "guard_e006_pc.py" else "RUN " + script) + "\n")
    if script != "guard_e006_pc.py":
        if script.replace(".py", "") in env("E006T_FAIL"):
            print("fake failure")
            return 1
        print("ok")
        return 0
    name = a[a.index("--name") + 1]
    refuse = dict(x.split(":") for x in env("E006T_REFUSE"))
    cnt_file = os.environ["E006T_CALLS"] + f".refuse_{name}"
    n = int(open(cnt_file).read()) if os.path.exists(cnt_file) else 0
    if n < int(refuse.get(name, 0)):
        open(cnt_file, "w").write(str(n + 1))
        return 3
    cmd = a[a.index("--") + 1:]
    rc, killed = (1, None) if name in env("E006T_FAIL") else ((9, "temp") if name in env("E006T_VERDICT") else (0, None))
    open(f"../logs/{name}.log", "x").write(f"fake run of {' '.join(os.path.basename(c) for c in cmd[:3])}\n")
    json.dump({"exit": rc if rc != 9 else -15, "killed": killed, "elapsed_s": 1.0, "peak_gpu_mem_mib": 1.0,
               "weights_sha256": None}, open(f"../logs/{name}.guard.json", "x"))
    if rc == 0 and any("e006_ft_test.py" in c for c in cmd) and "--save" in cmd and cmd[cmd.index("--save") + 1] == "1":
        os.makedirs(f"../weights/HuggingFaceTB__SmolLM2-135M-Instruct__{cmd[cmd.index('--tag') + 1]}", exist_ok=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
