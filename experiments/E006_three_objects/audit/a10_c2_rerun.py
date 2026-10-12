"""A10. C2's power-cut attempt vs its rerun: the PC manifest (pc_hashes.out, '== crashed' section) against the Mac
copies of the rerun's files. Usage: python a10_c2_rerun.py <pc_hashes.out>"""
import re, sys
from common import *

txt = open(sys.argv[1]).read().split("== crashed")[1].split("== weights")[0]
same = diff = 0
for h, n, path in re.findall(r"^([0-9a-f]{64})\s+(\d+)\s+\d+\s+\S+ \S+\s+(\S+)$", txt, re.M):
    if not path.startswith("out/"):
        if path.endswith("model.safetensors"):
            print("crashed attempt weights", h[:16])
        continue
    mac = os.path.join(E6, path)
    ok = os.path.exists(mac) and sha256_file(mac) == h
    same += ok
    diff += not ok
    if not ok:
        print("differs:", os.path.basename(path), "crashed bytes", n)
print(f"crashed C2 out files byte-identical to the rerun's: {same}; differing: {diff}")
