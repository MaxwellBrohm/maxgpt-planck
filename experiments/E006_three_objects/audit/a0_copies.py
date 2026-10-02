"""A0. The Mac copies against the PC: every file under ~/planck/e006_run/{out,logs,transcripts} hashed on the PC
(pc_hashes.sh, read-only) against the same path here. Usage: python a0_copies.py <pc_hashes.out>"""
import re, sys
from common import *

txt = open(sys.argv[1]).read().split("== files")[1].split("== crashed")[0]
pc = {p: h for h, p in re.findall(r"^([0-9a-f]{64})\s+(\S+)$", txt, re.M)}
mac = {}
for d in ("out", "logs", "transcripts"):
    for root, _, files in os.walk(os.path.join(E6, d)):
        for f in files:
            p = os.path.relpath(os.path.join(root, f), E6)
            mac[p] = sha256_file(os.path.join(root, f))
same = sum(pc[p] == mac.get(p) for p in pc)
print(f"PC files {len(pc)}; identical on the Mac {same}")
print("differ:", [p for p in pc if p in mac and pc[p] != mac[p]])
print("PC only:", [p for p in pc if p not in mac])
print("Mac only:", sorted(p for p in mac if p not in pc))
