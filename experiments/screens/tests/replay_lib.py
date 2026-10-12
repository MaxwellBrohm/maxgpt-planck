"""The BASE-REPLAY state (SCREENS.txt AMENDMENT BASE-REPLAY, 2026-10-10; cuts audit S7, row 29): plans/stage2_replay.txt
runs base_s101_rep1, base_s101_hookoff and base_s101_rep2 after BASE-DIAG, from its own bundle. The tests that pin the
config set and plans/CURRENT's sequence accept that state only through replay_configs(), which is empty while the plan
does not exist. When it exists: the BASE-DIAG state holds (basediag_lib.base_diag_configs is not empty); the entry
records exactly these three runs; screens_lib.BASE_REPLAYS registers exactly them; the plan waits on stage2_base_diag's
end mark, trains them in this order and nothing else, and ends at its own mark; each config sits in experiments/screens/
configs, passes screens.py check and differs from base_s101 only in name and out_dir (the hook-off run also in
eval.induction, null, with every hook key gone). Also the same-code check: the files under harness/, data_prep/,
tokenizer/, corpus/ and the 5 E2 files a SCREENS run reads must equal 93bc10b's (git blob ids) while plans/CURRENT
names stage2_replay. The entry is read from SCREENS.txt (written in experiments/screens/pending/replay.txt, moved there
unchanged before the commit). Data and functions; no test here.
"""
import hashlib
import os
import re
import subprocess

import screens_lib as L

REPLAY_PLANS = ["stage2_replay"]          # after DIAG_PLANS (basediag_lib)
REPLAY_PLAN = os.path.join(L.HERE, "plans", "stage2_replay.txt")
END = "mark SCREENS BASE REPLAY DONE"
RUNS = ["base_s101_rep1", "base_s101_hookoff", "base_s101_rep2"]
KIND = {"base_s101_rep1": "replay", "base_s101_hookoff": "hook-off", "base_s101_rep2": "replay"}
HEAD = "AMENDMENT BASE-REPLAY ("
REC_RE = re.compile(r"^  BASE RUNS ADDED BY AMENDMENT BASE-REPLAY: (base_s\w+(?:, base_s\w+)*) \(", re.M)
CODE_REF = "93bc10b9128e296d497ba2c08a71981431879702"
CODE_DIRS = ("harness", "data_prep", "tokenizer", "corpus")
CODE_E2 = tuple(f"experiments/E2_lr_transfer/{x}" for x in ("configs/base5m.yaml", "configs/engine.yaml", "preflight.py",
                                                            "queue_lib.sh", "bpb_lines.py"))
CODE_N, CODE_DIGEST = 253, "d627284410e7ecb025821086e018e27bb685c660acdac01de7233f89a585babb"   # git ls-tree -r 93bc10b
JUNK = ("__pycache__", ".pytest_cache")


def entry_text() -> str:
    """SCREENS.txt, which holds the amendment's entry (no pending copy is read any more)."""
    from test_screens_configs import SCREENS_TXT
    return open(SCREENS_TXT).read()


def section() -> str:
    lines = entry_text().splitlines()
    i = next(i for i, ln in enumerate(lines) if ln.startswith(HEAD))
    j = next((j for j in range(i, len(lines)) if not lines[j].strip()), len(lines))
    return "\n".join(lines[i:j])


def recorded_replay(text: str | None = None) -> list:
    """The runs the entry records as added by AMENDMENT BASE-REPLAY (its line at the entry's 2-space indent), in order."""
    text = entry_text() if text is None else text
    return [r for m in REC_RE.finditer(text) for r in m[1].split(", ")]


def replay_configs(plan: str = REPLAY_PLAN, find=None, check=None, recorded=None, diag_plan=None, registry=None,
                   diag=None) -> set:
    """The configs plans/stage2_replay.txt adds (empty while that plan does not exist); asserts the state above. diag:
    base_diag_configs(diag_plan), computed here unless a test that stubs the config dirs passes it."""
    if not os.path.exists(plan):
        return set()
    from basediag_lib import DIAG_PLAN, base_diag_configs
    from test_screens_configs import HOOK, P
    find, check = find or L.find, check or (lambda p: L.check(p, P))
    diag_plan = diag_plan or DIAG_PLAN
    diag = base_diag_configs(diag_plan) if diag is None else diag
    assert diag, "BASE-REPLAY runs after AMENDMENT BASE-DIAG's plan: no BASE-DIAG plan"
    rec = recorded_replay() if recorded is None else list(recorded)
    reg = L.BASE_REPLAYS if registry is None else registry
    assert rec == RUNS and reg == KIND, (rec, reg)
    wait = "wait_mark SCREENS " + open(diag_plan).read().splitlines()[-1][len("mark "):]
    lines = open(plan).read().splitlines()
    assert lines[0].startswith("# ") and lines[1] == wait and lines[-1] == END, lines
    runs = [ln.split()[1] for ln in lines[2:-1] if ln.startswith("train ")]
    assert len(runs) == len(lines) - 3 and runs == RUNS, lines
    ref = L.flat(L.resolve(find("base_s101")[0]))
    for r in runs:
        assert len(find(r)) == 1 and check(find(r)[0]) == [], r
        assert os.path.dirname(find(r)[0]) == L.config_dir(None), find(r)
        f = L.flat(L.resolve(find(r)[0]))
        want = {"name", "out_dir"} | ({"eval.induction"} | HOOK if KIND[r] == "hook-off" else set())
        assert L.diff(f, ref) == want and f["seed"] == 101, (r, L.diff(f, ref) ^ want)
        assert (KIND[r] == "hook-off") == ("eval.induction" in f and f["eval.induction"] is None), r
    return set(runs)


def git_blob(path: str) -> str:
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def code_files(root: str = L.ROOT) -> list:
    """The tracked files of the code paths (git ls-files in a checkout; every file but caches in an index export)."""
    if os.path.isdir(os.path.join(root, ".git")):
        env = dict(os.environ)
        if os.path.isdir("/Library/Developer/CommandLineTools"):
            env.setdefault("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
        out = subprocess.run(["git", "ls-files", "-z", "--", *CODE_DIRS, *CODE_E2], cwd=root, env=env,
                             capture_output=True, check=True).stdout.decode()
        return sorted(x for x in out.split("\0") if x)
    files = [x for x in CODE_E2 if os.path.isfile(os.path.join(root, x))]
    for top in CODE_DIRS:
        for d, dirs, fs in os.walk(os.path.join(root, top)):
            dirs[:] = [x for x in dirs if x not in JUNK]
            files += [os.path.relpath(os.path.join(d, x), root) for x in fs if not x.endswith(".pyc")]
    return sorted(files)


def code_digest(root: str = L.ROOT, files: list | None = None) -> tuple:
    """(file count, sha256 of the "path blob" lines): what 93bc10b gives is CODE_N, CODE_DIGEST."""
    files = code_files(root) if files is None else files
    txt = "".join(f"{p} {git_blob(os.path.join(root, p))}\n" for p in files)
    return len(files), hashlib.sha256(txt.encode()).hexdigest()
