"""Mutation test of the step 4 driver tests (Max's rule: a test never watched failing is not evidence).

Each mutant is one source edit. It is applied to a COPY of pipeline/ in a temp directory (the real files are never
touched, so a killed run cannot leave a mutant behind), then tests/test_driver.py and tests/test_driver_units.py run
in that copy. A mutant is killed when the tests fail. Exit 0 only if every mutant is killed and the unmutated copy
passes.

  python3 -B mutation_driver.py [--jobs 4] [--only NAME ...] [--list]"""
import argparse
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = ["test_driver", "test_driver_units"]

# (name, file, old, new): old must occur exactly once in file
MUTANTS = [
    ("no_dedup", "driver.py", 'dup, keys = self.state.index.check(turns, sk["slots"])',
     'dup, keys = None, self.state.index.check(turns, sk["slots"])[1]'),
    ("no_retry", "driver.py", 'self.queue.appendleft((sk["skel_id"], attempt + 1))', 'pass'),
    ("infeasible_retried", "driver_state.py", 'TERMINAL = {"SKEL_INFEASIBLE"}', 'TERMINAL = set()'),
    ("three_attempts", "driver_state.py", "len(d) >= self.max_attempts", "len(d) > self.max_attempts"),
    ("resume_forgets_accepts", "driver_state.py", 'self.mark(r["skel_id"], r["attempt"], None)', 'pass'),
    ("resume_forgets_rejects", "driver_state.py", 'self.mark(r["skel_id"], r["attempt"], r["primary"])', 'pass'),
    ("resume_no_dedup_index", "driver_state.py",
     'self.index.add(r["conv_id"], self.index.keys(r["turns"], r["slots"]))', 'pass'),
    ("resume_no_tally", "driver_state.py",
     'tally.add(sk, ok, r.get("primary"), r.get("codes", ()), r.get("est_tokens", 0), session=False)', 'pass'),
    ("no_repair", "shards.py", '    size = os.path.getsize(path)\n', '    return 0\n'),
    ("rotation_off_by_one", "shards.py", "if self.count >= self.size:", "if self.count > self.size:"),
    ("no_http_retry", "teacher_client.py", "TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}",
     "TRANSIENT_STATUS = set()"),
    ("drop_not_retried", "teacher_client.py",
     "except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,\n"
     "                    http.client.HTTPException, ValueError, KeyError, IndexError, TypeError) as e:",
     "except (urllib.error.URLError, socket.timeout, TimeoutError, ValueError, KeyError) as e:"),
    ("no_loopback_check", "teacher_client.py", "if u.hostname not in LOOPBACK:", "if False:"),
    ("no_stub_handshake", "teacher_client.py", "ident = self._get(STUB_PATH)", 'ident = {"planck_fake_stub": True}'),
    ("seed_ignores_attempt", "teacher_client.py", 'f"{skel_id}:{attempt}"', 'f"{skel_id}"'),
    ("tally_double_count", "yieldlog.py", "self.by_primary[primary] += 1", "self.by_primary[primary] += 2"),
    ("not_blocked", "records.py", "    out = []\n    if skel.get(\"provenance\"", "    return []\n    if skel.get(\"provenance\""),
    ("span_offset", "records.py", '"start": m.start()', '"start": m.start() + 1'),
    ("one_worker", "driver.py", 'max_workers=c["concurrency"]', "max_workers=1"),
    ("stop_after_ignored", "driver.py", 'self.stopped = "stop_after"', "pass"),
    ("near_threshold_high", "dedup.py", "NEAR_T = 0.7", "NEAR_T = 0.95"),
    ("no_mask", "dedup.py", "    for value, typ in sorted(slots, key=lambda s: -len(s[0])):", "    for value, typ in []:"),
    ("distinct_counts_tokens", "stats.py", "seen.add(tuple(toks[i:i + n]))", "seen.add(tuple(toks[i:i + 1]))"),
    ("ngram_alarm_ignores_min_n", "stats.py", '"alarm": bool(n >= MIN_N and worst > cap)', '"alarm": bool(worst > cap)'),
    ("hist_edges", "stats.py", "if edges[k] <= v < edges[k + 1]:", "if edges[k] < v <= edges[k + 1]:"),
    ("checker_crash_swallowed_as_accept", "driver.py", 'rec = records.reject(sk, attempt, "CHECKER_ERROR"',
     'raise\n            rec = records.reject(sk, attempt, "CHECKER_ERROR"'),
    ("manifest_after_dispatch", "driver.py", 'self.w_man.write({"j": j, "skel": sk})', 'pass'),
]


def copy_tree(dst):
    for name in os.listdir(HERE):
        src = os.path.join(HERE, name)
        if name.endswith(".py") or name in ("SPEC.txt",):
            shutil.copy2(src, dst)
    shutil.copytree(os.path.join(HERE, "tests"), os.path.join(dst, "tests"),
                    ignore=shutil.ignore_patterns("__pycache__"))


def run_one(mut, timeout=150):
    name = mut[0] if mut else "unmutated"
    root = tempfile.mkdtemp(prefix="planck-mut-")
    tmp = os.path.join(root, "pipeline")
    try:
        os.makedirs(tmp)
        copy_tree(tmp)
        # heldout.py imports the E004 lists from ../experiments (read-only, no bytecode written): link it
        os.symlink(os.path.join(os.path.dirname(HERE), "experiments"), os.path.join(root, "experiments"))
        if mut:
            _, fname, old, new = mut
            p = os.path.join(tmp, fname)
            with open(p) as f:
                src = f.read()
            if src.count(old) != 1:
                return name, "BAD", f"pattern occurs {src.count(old)} times in {fname}"
            with open(p, "w") as f:
                f.write(src.replace(old, new))
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, "-B", "-W", "ignore", "-m", "unittest", *TESTS],
                               cwd=os.path.join(tmp, "tests"), capture_output=True, text=True, timeout=timeout)
            failed, tail = r.returncode != 0, (r.stderr.strip().splitlines() or [""])[-1]
        except subprocess.TimeoutExpired:
            failed, tail = True, "timeout"
        return name, "killed" if failed else "SURVIVED", f"{tail} ({time.time() - t0:.0f}s)"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(m[0] for m in MUTANTS))
        return 0
    muts = [m for m in MUTANTS if not a.only or m[0] in a.only]
    base = run_one(None)
    print(f"{base[0]}: {'pass' if base[1] == 'SURVIVED' else 'FAIL'} {base[2]}", flush=True)
    if base[1] != "SURVIVED":
        print("the unmutated copy fails its tests; no mutant result would mean anything", flush=True)
        return 1
    bad = False
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for name, status, detail in ex.map(run_one, muts):
            print(f"{name}: {status} {detail}", flush=True)
            bad |= status != "killed"
    print("ALL MUTANTS KILLED" if not bad else "MUTATION RUN FAILED", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
