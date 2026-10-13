"""RC-12 lock: the lock-day edits of the prereg (s19 step 8), as a fixed list. Stdlib, no model; never opens sealed/.

Everything that could be settled before the lock is already written in the draft (rc12/notes.txt STEP 15). What is
left changes only words that are true at the lock and not before it: the status header, "proposal until the push"
on the thresholds, the s16 title, s19's open steps, and the lock record (s20) copied from the sealed session's
public S5 file. Each edit must match its draft text exactly once, so a draft that drifted fails loudly instead of
locking half-edited.

  git mv prereg/RC-12.draft.txt prereg/RC-12.md
  python3 -B rc12/lock_finalize.py --date YYYY-MM-DD --sealed-date YYYY-MM-DD [--root R]
then rc12/lock_check.py manifest and rc12/lock_check.py check --final."""
import argparse
import os
import textwrap
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL = "prereg/RC-12.md"
S5 = "rc12/lock/SEALED_HASHES.txt"
DRAWS = "rc12/lock/SEED_ATTEMPTS.txt"


def edits(date, sealed_date):
    return [
        ("RC-12 PRE-REGISTRATION: DRAFT, NOT LOCKED", f"RC-12 PRE-REGISTRATION: LOCKED {date}"),
        ("Status: DRAFT 0.1, written 2026-09-25. Nothing here binds anyone until Max approves it and it is pushed as "
         "the locked\npre-registration (PLAN Phase 0, lock window Oct 11-14). Every number marked (proposal) may "
         "change before the lock and\nnever after it. Items marked OPEN are listed again in s16; s17 holds the "
         "decisions that needed Max.",
         f"Status: LOCKED {date} (first written 2026-09-25 as draft 0.1). Max approved it and it was pushed as the "
         "locked\npre-registration (PLAN Phase 0, lock window Oct 11-14); the push and, with Max's yes, the outside "
         "timestamps of\ns13 S9 are recorded in rc12/notes.txt after the lock commit. Every number in it is fixed and "
         "never changes;\ns16 lists what was settled before the lock, s17 the decisions that needed Max, s20 the lock "
         "record."),
        ("Its 0.60 is a proposal until the push and never changes after it.",
         "Its 0.60 was fixed at the lock and never changes."),
        ("and -5 are proposals until the push and\n    never change after it",
         "and -5 were fixed at the lock and\n    never change"),
        ("(proposal thresholds):", "(thresholds fixed at the lock):"),
        ("The bar (proposal until the push):", "The bar (fixed at the lock):"),
        ("its bar (0.60, a proposal until the push)", "its bar (0.60, fixed at the lock)"),
        ("16. STILL OPEN, SETTLED BY A MEASUREMENT OR A MECHANICAL RULE (no judgment needed)",
         "16. SETTLED BEFORE THE LOCK BY A MEASUREMENT OR A MECHANICAL RULE (no judgment needed)"),
        ("  The sealed split (s13) and every hash of S5 and S9.",
         f"  The sealed split (s13): built {sealed_date} in the sealed session; S5's hashes are in s20, S9's code "
         "hashes in\n    rc12/lock/MANIFEST.txt."),
        ("its kit built and dry-run on 2026-10-11 and 12 without sealed content (s13 S3;\n"
         "    rc12/notes.txt STEP 15; the steps in rc12/LOCK_STEPS.md); the session itself not yet run.",
         "its kit built and dry-run on 2026-10-11 and 12 without sealed content (s13 S3;\n"
         f"    rc12/notes.txt STEP 15); DONE {sealed_date} (s20; rc12/lock/SEALED_GATES.txt)."),
        ("8 Final pass on this file: every (proposal) resolved, every OPEN closed, DRAFT removed: the text settled "
         "before\n    the lock is written (2026-10-11 and 12, STEP 15); the lock-day edits are the fixed list in "
         "rc12/lock_finalize.py,\n    checked by rc12/lock_check.py check --final.",
         f"8 Final pass on this file: DONE {date}: the text settled before the lock written 2026-10-11 and 12\n"
         "    (STEP 15), the lock-day edits made by rc12/lock_finalize.py, rc12/lock_check.py check --final passing."),
    ]


def record(root, date):
    s5 = open(os.path.join(root, S5)).read().rstrip("\n").split("\n")
    draws = open(os.path.join(root, DRAWS)).read()
    live = draws.rsplit("\nsuperseded:", 1)[-1]                    # restarts void earlier acceptances (s13 S3)
    acc = re.findall(r"^accepted draw (\d+)(?:;|$)", live, re.M)
    if len(acc) != 1:
        sys.exit(f"{DRAWS} must name exactly one accepted draw after its last superseded line")
    total, sup = len(re.findall(r"^draw \d+:", draws, re.M)), draws.count("\nsuperseded:")
    ceil = re.search(rf"^draw {acc[0]}: PASS;.*?ceiling S7 ([0-9.]+)", live, re.M)
    if not ceil:
        sys.exit(f"{DRAWS}: the accepted draw's line carries no shortcut ceiling")
    draws_line = (f"Seed draws (s13 S3 (iv)): the seed of draw {acc[0]} was accepted, of {total} draws in all, {sup} "
                  f"restart(s) after a fix ({DRAWS}); the sealed shortcut ceiling (S7, reported beside Level R, s8 G3) "
                  f"is {ceil.group(1)}.")
    lines = [f"20. LOCK RECORD ({date}; copied from {S5}, written by the sealed session, s13 S5)"]
    lines += textwrap.wrap(draws_line, 118, initial_indent="  ", subsequent_indent="    ")
    lines.append("  OOD-H Part 1 (s14): sha256 and counts in oodh/HASHES.txt, the file itself in the archive.")
    lines += ["  " + ln for ln in s5]
    lines.append("  Code, configs and splits at the lock: rc12/lock/MANIFEST.txt (s13 S9).")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--sealed-date", required=True)
    ap.add_argument("--root", default=os.path.dirname(HERE))
    a = ap.parse_args()
    for d in (a.date, a.sealed_date):
        if not re.fullmatch(r"2026-1[0-2]-(0[1-9]|[12][0-9]|3[01])", d):
            sys.exit(f"bad date {d}")
    path = os.path.join(a.root, FINAL)
    if not os.path.isfile(path) or os.path.exists(os.path.join(a.root, "prereg/RC-12.draft.txt")):
        sys.exit(f"git mv prereg/RC-12.draft.txt {FINAL} first")
    text = open(path, encoding="utf-8").read()
    for old, new in edits(a.date, a.sealed_date):
        if text.count(old) != 1:
            sys.exit(f"lock_finalize: expected exactly one {old[:60]!r}, found {text.count(old)}; nothing written")
        text = text.replace(old, new)
    text = text.rstrip("\n") + "\n\n" + record(a.root, a.date)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"{FINAL}: {len(edits(a.date, a.sealed_date))} edits and the s20 lock record written")


if __name__ == "__main__":
    main()
