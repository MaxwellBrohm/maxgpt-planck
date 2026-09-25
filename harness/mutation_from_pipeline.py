"""Mutation check for from_pipeline.py. Each mutant replaces ONE exact string (it must occur once) in a
scratch copy of harness/*.py and runs test_from_pipeline.py + test_from_pipeline_refuse.py there against
one FAKE pipeline run made up front (PLANCK_PIPELINE_FAKE). Killed = pytest exit 1; exit 2-5 = INVALID.
The unmutated copy runs first and must be green.
  python mutation_from_pipeline.py [--workers 4]      (about 30-60 s)
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = ["test_from_pipeline.py", "test_from_pipeline_refuse.py"]
F = "from_pipeline.py"
M = [
    ('loss = role == "user" or (role == "assistant" and mask == 0)', 'loss = role in ("user", "assistant")',
     "pipeline mask ignored"),
    ('loss = role == "user" or (role == "assistant" and mask == 0)',
     'loss = role in ("user", "tool") or (role == "assistant" and mask == 0)', "tool turns get loss"),
    ('"loss": False, "src_i": None', '"loss": True, "src_i": None', "system turn gets loss"),
    ('trainable = rec.get("trainable") is True and not rec.get("blocked")',
     'trainable = bool(rec.get("trainable")) and not rec.get("blocked")', "truthy trainable accepted"),
    ('trainable = rec.get("trainable") is True and not rec.get("blocked")',
     'trainable = rec.get("trainable") is True', "blocked reasons ignored"),
    ("if not trainable and not allow_nontrainable:", "if not trainable and allow_nontrainable is None:",
     "non-trainable never refused"),
    ('"admitted_nontrainable": not trainable', '"admitted_nontrainable": False', "admission not stamped"),
    ("m = SPECIAL_RE.search(text)", "m = None", "role-token strings pass"),
    (r'SPECIAL_RE = re.compile(r"<\|[^|<>\s]{1,40}\|>")',
     r'SPECIAL_RE = re.compile(r"<\|(end|user|assistant|system)\|>")', "only four role strings checked"),
    ("bad = sorted(set(ids) & special_ids)", "bad = []", "special ids pass"),
    ("if not ids:\n        raise", "if False:\n        raise", "empty ids pass"),
    ('if system is not None and system != "":', "if False:", "system text dropped"),
    ('if mask == 1 and role != "assistant":', "if False:", "mask on a user/tool turn accepted"),
    ("if mask not in (0, 1) or isinstance(mask, bool):", "if mask not in (0, 1):", "boolean mask accepted"),
    ('if not isinstance(text, str) or not text:', "if not isinstance(text, str):", "empty text accepted"),
    ('"pipeline": {k: v for k, v in rec.items() if k != "turns"}',
     '"pipeline": {k: v for k, v in rec.items() if k not in ("turns", "events")}', "provenance field lost"),
    ('hashlib.sha256(raw.rstrip(b"\\n"))', "hashlib.sha256(raw)", "line hash includes the newline"),
    ("for line_no, raw in enumerate(f, 1):", "for line_no, raw in enumerate(f):", "line numbers off by one"),
    ('"spans": t.get("spans", [])', '"spans": []', "spans dropped"),
    ('"author": t.get("author")', '"author": None', "author dropped"),
    ('"src_i": i,', '"src_i": i + 1,', "src_i off by one"),
    ("for t in out:\n            t[\"ids\"]", "for t in out[1:]:\n            t[\"ids\"]", "first turn not tokenized"),
    ('o["adapter"]["n_tokens"] = len(ids)', 'o["adapter"]["n_tokens"] = len(ids) - 1', "n_tokens wrong"),
    ('"role_ids": tok_info["role_ids"]', '"role_ids": {**tok_info["role_ids"], "tool": 3}',
     "manifest maps tool to the user id"),
    ("self.n % self.size == 0 and self.n", "self.n % (self.size + 1) == 0 and self.n", "rotation off by one"),
    ('return 0 if m["counts"].get("admitted") else 3', "return 0", "exit 0 with nothing admitted"),
    ("if os.path.exists(tmp):\n        raise", "if False:\n        raise", "stale .partial reused"),
    ('raise SystemExit("an input shard is listed twice")', "pass", "duplicate input read twice"),
    ('raise Refused("BAD_JSON", "record is not an object")', 'return {}', "non-object line admitted"),
    ('if len(examples) < N_EXAMPLES:', 'if len(examples) < 1:', "refusal examples truncated"),
    ("                w.write(o)", '                if c["admitted"] != 3:\n                    w.write(o)',
     "one admitted record not written"),
    ("        raise Refused(\"ROLE_TOKEN_IN_TEXT\", f\"{where}: {m.group(0)}\")\n    return text",
     "        raise Refused(\"ROLE_TOKEN_IN_TEXT\", f\"{where}: {m.group(0)}\")\n    return text.replace(\"'\", \"\\u2019\")",
     "turn text altered"),
]


def run_copy(root: str, env: dict) -> int:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TESTS], cwd=root,
                       env=env, capture_output=True, text=True, timeout=110)
    return r.returncode


def one(k, old, new, env) -> tuple[int, str]:
    root = tempfile.mkdtemp(prefix="mut_fp_")
    try:
        for p in glob.glob(os.path.join(HERE, "*.py")):
            shutil.copy(p, root)
        if k is not None:
            path = os.path.join(root, F)
            src = open(path).read()
            if src.count(old) != 1:
                return k, f"INVALID (pattern occurs {src.count(old)} times)"
            open(path, "w").write(src.replace(old, new))
        rc = run_copy(root, env)
        return k, {0: "SURVIVED", 1: "killed"}.get(rc, f"INVALID (pytest exit {rc})")
    finally:
        shutil.rmtree(root, True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="mut_fp_run_")
    try:
        subprocess.run([sys.executable, "-B", os.path.join(HERE, "make_pipeline_fake.py"), os.path.join(tmp, "p")],
                       check=True, capture_output=True, timeout=110)
        env = {**os.environ, "PLANCK_PIPELINE_FAKE": os.path.join(tmp, "p", "run"), "PYTHONDONTWRITEBYTECODE": "1"}
        base = one(None, "", "", env)[1]
        print("unmutated:", "green" if base == "SURVIVED" else base, flush=True)
        if base != "SURVIVED":
            return 2
        with ThreadPoolExecutor(a.workers) as ex:
            res = dict(ex.map(lambda km: one(km[0], km[1][0], km[1][1], env), enumerate(M)))
    finally:
        shutil.rmtree(tmp, True)
    for k, (_, _, why) in enumerate(M):
        print(f"{k:2d} {res[k]:10s} {why}")
    killed = sum(v == "killed" for v in res.values())
    print(f"{killed}/{len(M)} killed")
    return 0 if killed == len(M) else 1


if __name__ == "__main__":
    sys.exit(main())
