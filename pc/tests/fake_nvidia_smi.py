"""A stand-in for nvidia-smi, driven by a JSON state file ($FAKE_SMI_STATE).

Answers `--query-gpu=a,b,c --format=csv,noheader,nounits` with the state's values in the
requested order, the way the real tool does. Tests edit the state file while a job runs to
heat the GPU, fill its memory or make the tool fail.

State keys: temp_c, mem_used_mib, mem_total_mib, util_pct, power_w, throttle (hex string),
fail (exit 9, no output), hang (sleep 30 s), temp_na (print [N/A] for temperature),
old_driver (reject clocks_event_reasons.active like a pre-rename driver).
"""
import json
import os
import sys
import time

MAP = {"temperature.gpu": "temp_c", "memory.used": "mem_used_mib",
       "memory.total": "mem_total_mib", "utilization.gpu": "util_pct", "power.draw": "power_w",
       "clocks_event_reasons.active": "throttle", "clocks_throttle_reasons.active": "throttle"}


def main() -> int:
    with open(os.environ["FAKE_SMI_STATE"]) as f:
        st = json.load(f)
    if st.get("fail"):
        return 9
    if st.get("hang"):
        time.sleep(30)
    q = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--query-gpu=")), None)
    if q is None:
        print("fake nvidia-smi: only --query-gpu is supported", file=sys.stderr)
        return 2
    out = []
    for field in q.split(","):
        if field not in MAP or (field == "clocks_event_reasons.active" and st.get("old_driver")):
            print(f'Field "{field}" is not a valid field to query.', file=sys.stderr)
            return 2
        if field == "temperature.gpu" and st.get("temp_na"):
            out.append("[N/A]")
            continue
        out.append(str(st.get(MAP[field], "[N/A]")))
    print(", ".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
