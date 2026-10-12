"""Queue a job for the PC runner: validate a job JSON and copy it into queue/pending/.

  python3 add_job.py jobs/020_bench_micro.json            # keeps the file name (order prefix)
  python3 add_job.py my_job.json --as 015_etok_5m.json    # choose the name / position
  python3 add_job.py my_job.json --dry-run                # validate only

Checks what the runner will check before a start: the file parses, cmd is a list, cwd
exists, and the prereg is committed (or no_prereg_reason is given). Refuses to overwrite a
job of the same name in pending/, running/ or done/. The job format is in jobq.py.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jobq  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="queue a Planck PC job")
    ap.add_argument("job")
    ap.add_argument("--home", default=os.environ.get("PLANCK_HOME", "~/planck"))
    ap.add_argument("--as", dest="name", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-require-prereg", action="store_true")
    a = ap.parse_args(argv)
    home = os.path.abspath(jobq.expand(a.home))
    os.environ["PLANCK_HOME"] = home
    try:
        job = jobq.load(a.job)
        note = jobq.check_prereg(job, not a.no_require_prereg)
    except jobq.JobError as e:
        print(f"add_job: REFUSED: {e}")
        return 1
    name = a.name or os.path.basename(a.job)
    if not name.endswith(".json"):
        name += ".json"
    r = jobq.resolved(job)
    print(f"add_job: {name}: {note}\n  cmd {r['cmd']}\n  cwd {r['cwd']}")
    if a.dry_run:
        return 0
    jobq.ensure_layout(home)
    for d in ("pending", "running", "done"):
        if os.path.exists(os.path.join(jobq.qdir(home, d), name)):
            print(f"add_job: REFUSED: {name} already exists in queue/{d}/")
            return 1
    job.pop("attempts", None)
    job.pop("history", None)
    jobq.save(job, os.path.join(jobq.qdir(home, "pending"), name))
    print(f"add_job: queued in {jobq.qdir(home, 'pending')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
