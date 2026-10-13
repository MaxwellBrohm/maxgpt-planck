# RC-12 lock steps, Oct 11-14 2026

The lock is s19 of the prereg (`prereg/RC-12.draft.txt`). Steps 1 to 5, 4b and 7 are done; Max's OOD-H review (step 7) finished on 2026-10-10 (s14 REVIEWED). Left: step 6 (the sealed session), step 8 (the final pass) and step 9 (push, release, snapshot, key copies). The machinery was built and dry-run on 2026-10-11 and 12 without any sealed content (`rc12/notes.txt` STEP 15): the content hook, the work-copy kit, the seed draws, encryption, the lock checks and the lock-day edits.

Rules for every step: **never read, list or build anything under `sealed/` except in track B**, and **no Planck model is scored on any RC-12 split before the push**. On the Mac, at most 4 worker processes. Git runs with `DEVELOPER_DIR=/Library/Developer/CommandLineTools`; commit explicit paths only, after checking `git diff --cached` is empty; push, never force. No machine paths, user names, keys or emails go in committed files.

Order: A, then B, then M (Max), then D (lock day), then E (release and snapshot, only on Max's yes).

## A. Claude in this repo, before the sealed session

Every code change (`rc12/`, the E001 and E004 code it imports, `oodh/*.py`, `corpus/oodh.py`) is committed before B starts: B gates the sealed items with the code of one commit `C`, and `lock_check.py check --final` refuses any code change between `C` and the lock commit. While B runs, the main session does not touch `rc12/` or `oodh/`.

**A1. dev_regrade durability fix.** Done in STEP 15: two fresh reviewer sessions approved it (their minor findings fixed), `mutation_regrade.py` 17 of 17, committed with STEP 15.

`<owner>/<repo>` below is `gh repo view --json nameWithOwner -q .nameWithOwner`.

**A2. Grader findings from the OOD-H reviews** (`oodh/DESIGN.txt` s14 and s15, "Grader findings for the rc12 workflow"; counts on OOD-H Part 1 unless stated). Each is a natural, right reply the graders fail:

- "You asked if X." fails every VAL probe (s15: 0 of 323 pass; s14 counted 97 of 97 H-ASK before the review). "You asked whether X." passes.
- Abstain cues: "You never gave me that.", "You didn't specify.", "You never specified that.", "You didn't provide that." fail 92 of 92 ABS probes and **60 of 60 RC-12 dev ABS probes**; "You never picked one.", "You haven't chosen one yet." and the bare "No, you didn't." fail 92 of 92; the reviews also saw "You didn't name ...", "You didn't rate it ..." and "You haven't ordered anything yet." fail.
- Hedge inside a report: "You said it should be X." and "You said it seems to be X." fail 323 of 323 VAL probes.
- A gold inside a quoted question ending in "?" reads as not asserted; a contraction of the user's own negation ("isn't X" for "is not X") fails unless listed.
- Also in s15: the loop clauses compare a reply with the fixed OASST2 history replies (1 probe seen); the shotgun clause fails "new X instead of old Y" on 2-candidate probes (2 seen; a policy cost, STEP 14S).

Decide each one before `C`; the graders are hashed at the lock. Either **fix it** the way the T0 audit's gaps were fixed (fixtures that fail before and pass after, a mutant per clause, the paraphrases in `audit_graders`, `mutation_graders`; then `dev_regrade.py` over every stored dev run, `dev_panel.py`, `dev_s11.py`, `dev_extras14.py`; then the numbers in s6 (Evidence counts), s8 (cheater table), s9, s10b, s11, s12 and `test_numbers14.py`, and the full check order), or **record it** in s6 as a known limit with its counts. Recommendation: fix the cue gaps ("You asked if", the abstain cues, the hedge inside a report, the quoted question); record the loop and shotgun items. The OOD-H gate re-run after a grader change reads `sealed/oodh`, so it is B7.

**A3. Full check order** on an export of the commit B will use:

```
git archive <C> rc12 experiments/E001_battery_and_probes/code experiments/E004_general_updating/code prereg oodh corpus pc harness | tar -x -C <tmp>
```

In `<tmp>/rc12`: `build_dev.py --stdout` must equal `dev/rc12_dev.jsonl` (sha256 3e543f8d); then, each alone: `test_gens`, `audit_rules`, `audit_graders`, `validate_e2e`, `validate_machinery`; every `test_*.py` (among them `test_lock`, `test_lock_b`, `test_lock_c`, `test_lock_d`; and `extras/test_rc12x_extras.py`); `mutation_graders`, `mutation_gens`, `mutation_audit --part 1/2` and `2/2`, `mutation_e2e --part i/8` with `RC12_MAX_WORKERS=4`, `mutation_lock`, `mutation_regrade`, `mutation_numbers14`, `mutation_oodh_wording`, `mutation_lik` (with `LIK_TORCH_PY`), `mutation_step9`, `mutation_verify`, `test_near_dup --mutate`, `mutation_queue`, `mutation_s14`, `mutation_step14x`, `mutation_rerun_ws`, `mutation_hf_batched --chunk-only`; `oodh/tests`; and the pc suite from `<tmp>/pc`. `C` is recorded in `rc12/notes.txt` in the lock commit (D6), not before, so B0 holds.

**A4. Handoff to W2 (pipeline)**, through the FOR OTHER WORKFLOWS line: the S6 list hashes word 13-grams with words `re.findall(r"[a-z0-9']+", text.lower())` joined by one space; the pipeline's decontamination must normalize the same way. Do not edit `pipeline/`.

## B. The sealed session (a separate Claude session)

**Reading list.** This section and the rules line at the top; prereg s3 (the filler and length rules), s5, s6 (Inflected forms), s8, s13 and s14; `rc12/notes.txt` STEP 15. No other context is needed. It may read and write under `sealed/` and `private/` (both gitignored).

**It must not** read `pipeline/` banks, templates, teacher renders or lookup-line formats (s13 S2), load a model, commit, change any tracked file except `rc12/lock/*` and the GATE section `oodh/gate.py` rewrites in `oodh/HASHES.txt`, or let item text leave the session: only gate tables, the draw log, counts and hashes leave it.

**Variables** (repo root): `P` = the planck venv's python (the one every rc12 step uses); `C` = the commit from A3; `RC12_KEY` = a key path outside every repository (for example a `planck-keys` folder under the Mac's Application Support). The PC helpers are `pcwsl.sh` (in the Mac's `planck-e2chain` support folder; it copies a script to the PC and runs it in WSL) and the gitignored `pc/local.env` (`PLANCK_PC_KEY`, `PLANCK_PC_HOST`), as rc12/notes.txt STEP 15 records.

**B0. Pin the code.** `git diff --quiet "$C" HEAD -- rc12 oodh corpus/oodh.py experiments/E001_battery_and_probes/code experiments/E004_general_updating/code ':!*.txt' ':!*.md'` and `git status --porcelain -- rc12 oodh corpus/oodh.py experiments/E001_battery_and_probes/code experiments/E004_general_updating/code` showing no code file (text files such as `rc12/notes.txt` may move; code may not). Otherwise stop. Repeat B0 before B6 and B9.

**B1.** `$P -B rc12/sealed_kit.py prepare sealed/rc12_work --commit $C`

**B2.** `$P -B rc12/sealed_overlap.py skeleton sealed/rc12_content` writes one `content_<module>.py` per content module (11) at the dev values. Rewrite every value as new Claude-written content under S1 and s3. A content file is data, never code: only its docstring and `NAME = <literal>` lines are accepted (`split_content.load_literals`).

- **New:** the value pools; the BIND intros and the neutral statement tails (`content_balance.py`); HOLDER_F (female), HOLDER_M (male) and HOLDER_N (pronouns depend on the first two); the PERSIST rule words; the templates; the fillers with their hand labels ("question", "request", "statement"; no digit, name-like word, pool value, holder word or rule word; 0 word 5-grams with the E001 and E004 fillers; one-sentence IDEAL replies); the LOOKUP table format (`ROW`, `ROW_SEP`, OD5). User turns stay at 35 words or fewer (tables 70).
- **Shared with dev:** only WEEKDAY, MONTH (still without "May") and LANGUAGE. Every other string is new, short ones too: objects, aliases, topics, nouns, frames and questions such as "What's my name?" (S1 (d)).
- **May repeat dev's words (S1 (d) exempts them; `sealed_overlap.D_SKIP`, `D_SKIP_NAMES`):** the fillers' hand labels; the avoid lists `BIND_AVOID`, `OWN_AVOID` and `LOOKUP_AVOID`; BIND_NAME's `objw`. Each avoid list names the object words its own items use (no filler of the item may contain them): `OWN_AVOID` the object word that runs through each new OWN_PICK item's `q`, `before`, `after` and `probe` (dev: boat, puppy, band, quiz, team), `LOOKUP_AVOID` the one through each LOOKUP item's `intro`, `q`, `ideal` and `abst` (dev: rota, tour, menu), `BIND_AVOID` new BIND object words (`obj` or BIND_NAME `objw`; dev: bike, kitten, instrument, trip, name). Rewrite each list to match your new templates; keep a dev word only where your template keeps that object. OWN_AVOID has 5 slots for 4 items: the spare holds a second object word of one item (dev: quiz and team) or repeats one, never a common word. test_gens refuses an avoid word no template uses (for BIND: no object word), and an OWN or LOOKUP item whose four texts share no word of its list and an avoid word that runs through several items (an object word is its own item's) (`{placeholders}` are not words; case is ignored, as the build ignores it; an avoid word is letters only, as test_gens reads words).
- **Unchanged:** every name, type, dict key, list length and `{placeholder}` (`split_content.shape_diff` refuses anything else), and the identifier fields of `split_content.PINS` (RECALL_SLOTS `key`/`vtype`, TWOHOP `vtype`, LOOKUP `keys`/`vals`, the K kind); keys are free only where CONTENT marks a name with `*` (OWN_LIST, PROJECTS; PROJECTS keys equal the new PROJECT pool). No weekday, instrument, food or project value whose plural or singular is another common word, and at least one instrument, food or project value ends in s, as dev's "drums" does (test_gens checks it: the grader's singular-form rule needs one).
- OWN pick values (the BOAT, PUPPY, BAND and QUIZTEAM pools) hold no hedge or alternative cue ("guess", "seems", "maybe", "or", "/", "?"): a name the grader reads as hedged, an alternative or a question is never asserted, and a G6 source fixture then gets the wrong verdict on any seed whose chosen OWN pick records use it.
- Count your labels as q, r, s.

**B3.** `$P -B rc12/sealed_kit.py install sealed/rc12_work sealed/rc12_content`, then `$P -B rc12/sealed_kit.py check sealed/rc12_work` (must print that the copy equals its commit).

**B4. Seed draws (s13 S3 (iv)).** `$P -B rc12/sealed_search.py sealed/rc12_work --seed-file sealed/rc12_seed.txt --log sealed/rc12_draws.txt --expect-labels q,r,s --max 200`. The draw log lives outside the work copy and keeps every draw of every run. Each draw builds, runs test_gens, G10, G1-G4, mutation_graders and mutation_audit, in that order (their detectors and fixtures depend on the items), stops at the first failure and discards a failing seed; the first passing seed is kept. The cap is 200 draws per run (a new run on a log that has draws, a restart or a run after 200 failures, needs `--supersede`; the log counts every draw). The dry run's searches took 1 to 20 draws (each 1 to 3 minutes). If it exits 1 (no seed in 200), stop and write the report: what happens next is the main session's and Max's decision. Watch the first draws: a content rule (test_gens `test_pools`, the fillers, the avoid lists) fails every seed the same way, so if the first 3 draws all fail test_gens, press Ctrl-C right after the third draw line (that draw's gate logs stay until the next build ends), read `sealed/rc12_work/gate_logs/test_gens.txt`, fix the content, reinstall (B3) and restart with `--supersede "test_gens content rule"` (the interrupted draw gets no line; the supersede line says why). test_gens' G9 balance checks also fail by chance, so read the log before deciding it is the content.

**B5. S1 separation**, right after the items exist: `$P -B rc12/sealed_kit.py overlap sealed/rc12_work` (run from the repo; must print `S1 overall: PASS`; it writes the counts, never text, to `sealed/rc12_work/OVERLAP.txt`, which the pack archives; on dev content it prints FAIL on (a) to (d), the positive control; on renamed values with dev templates, FAIL on (b) to (d)).

**Restart after a fix.** If B5 or B6 fails on content: fix `sealed/rc12_content`; `mv sealed/rc12_seed.txt sealed/rc12_seed.rejected.<n>.txt`; `mv sealed/rc12_work sealed/rc12_work.old<n>`; B1 again (a fresh `sealed/rc12_work`); B3; then B4 with `--supersede "<step and module, e.g. S1 (b) pools_fill_a>"` (the reason is published, so it must match `sealed_search.REASON`: a step and a module, never item text; the same `sealed/rc12_draws.txt` continues and the earlier acceptance is void). If B6 fails on a seed-sensitive gate (the content is not the cause), the same restart without a content change redraws the seed. If a failure needs a code change: stop, write the gate table to `private/rc12_sealed_report.txt`, and hand back to the main session, which fixes, commits and reruns A3 for a new `C`; then restart from B0 with the new `C` the same way.

**B6. Every S4 gate.** `$P -B rc12/sealed_kit.py gates sealed/rc12_work --seed-file sealed/rc12_seed.txt --expect sealed/rc12_work/expect.json` must end with `failing steps: 0` in `sealed/rc12_work/GATES.txt` (about 20 to 30 minutes). In the dry run one B6 run stalled in `mutation_e2e` part 2 for over two hours with its parent idle; if a step uses no CPU for 10 minutes and has no worker processes, kill the gates run and run B6 again from the start (the table must come from one run).

**B7. OOD-H gate (s14 (d)).** `$P -B oodh/gate.py` must exit 0; keep the GATE section it writes in `oodh/HASHES.txt` and re-report its G2 rates; the `oodh_part1.jsonl` sha256 line must stay `40d2a76d1315d3cb...`.

**B8. S6 list.** `$P -B rc12/sealed_crypto.py ngrams sealed/rc12_work --oodh sealed/oodh/oodh_part1.jsonl --out private/rc12_decon13_sha256.txt.gz`

**B9. S5 pack.** `$P -B rc12/sealed_pack.py sealed/rc12_work --seed-file sealed/rc12_seed.txt --log sealed/rc12_draws.txt --oodh sealed/oodh/oodh_part1.jsonl --ngrams private/rc12_decon13_sha256.txt.gz --out sealed/rc12_pack` writes the deterministic `RC-12-sealed.tar` (mode 600) and `SEALED_HASHES.txt` (with the S6 count and the S1 verdict). It refuses unless the copy equals `C`, GATES.txt is a full clean table whose recorded inputs are the files packed, the seed rebuilds the items byte for byte, the draw log's last run accepted this seed, and OVERLAP.txt passed for these items. Read `sealed/rc12_work/GATES.txt`, `sealed/rc12_draws.txt` and `sealed/rc12_pack/SEALED_HASHES.txt` and confirm they hold no item text before B11 (the kit withholds quoted lines and the content's own words, but read them anyway).

**B10. Key (S7, S8).** `$P -B rc12/sealed_crypto.py keygen "$RC12_KEY"` prints only a fingerprint. Copy it straight into WSL, never through the Windows file system (source `pc/local.env` first):

```
ssh -i "$PLANCK_PC_KEY" -o IdentitiesOnly=yes -o BatchMode=yes "$PLANCK_PC_HOST" "wsl -d Ubuntu --exec bash -c \"umask 077; mkdir -p ~/.planck_keys; cat > ~/.planck_keys/rc12_sealed.key; sha256sum ~/.planck_keys/rc12_sealed.key; stat -c %a ~/.planck_keys/rc12_sealed.key\"" < "$RC12_KEY" | tr -d '\r' | cut -c1-16
```

The first printed line (the fingerprint) must equal the Mac's and the second must be `600` (this exact command passed the dry run with a dummy key, STEP 15). Max may add a third copy to his password manager. The key never goes into a repo, a notes file, MEMORY.md or a transcript.

**B11. Encrypt and publish the public outputs.**

```
$P -B rc12/sealed_crypto.py encrypt sealed/rc12_pack/RC-12-sealed.tar --key "$RC12_KEY" --out rc12/lock/RC-12-sealed.tar.enc --hashes sealed/rc12_pack/SEALED_HASHES.txt
cp sealed/rc12_pack/SEALED_HASHES.txt rc12/lock/
cp sealed/rc12_work/GATES.txt rc12/lock/SEALED_GATES.txt
cp sealed/rc12_draws.txt rc12/lock/SEED_ATTEMPTS.txt
```

**B12. Verify on both machines** (s19 step 9's key check). Mac: `$P -B rc12/sealed_crypto.py verify rc12/lock/RC-12-sealed.tar.enc --key "$RC12_KEY" --hashes rc12/lock/SEALED_HASHES.txt` prints PASS. PC: scp `rc12/sealed_crypto.py`, `rc12/sealed_kit.py`, `rc12/sealed_public.py` (sealed_kit imports it), the `.enc` and `SEALED_HASHES.txt` to `planck-kit`, then through `pcwsl.sh` run `cd "$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' | tr -d '\r')")/planck-kit" && python3 -B sealed_crypto.py verify RC-12-sealed.tar.enc --key ~/.planck_keys/rc12_sealed.key --hashes SEALED_HASHES.txt`, which prints PASS. The dry run passed both (STEP 15).

**B13. Remove the plaintext (S7)**, only after both verifies pass: `sealed/rc12_work*`, `sealed/rc12_pack`, `sealed/rc12_content`, `sealed/rc12_seed*.txt`, any `sealed/rc12_seed.txt.try*` and `sealed/rc12_draws.txt` (its public copy is in `rc12/lock`). The archive holds every file that counts. OOD-H's files in `sealed/oodh` stay.

**B14. Report**, counts only, in `private/rc12_sealed_report.txt` and in the reply: the `rc12/lock/` file list, the B6 gate table, the draw count and restarts, the sealed shortcut ceiling, the S1 counts (0, 0, 0), the OOD-H gate verdict and its G2 rates, the S6 count, both key fingerprints (matching), and the date (the sealed date for D).

## M. Max (`rc12/LOCK_FOR_MAX.md`)

M1 OOD-H review: done 2026-10-10. M2 a yes to the lock: the prereg binds nothing until Max approves it and it is pushed. M3 a separate yes to the GitHub release and the Software Heritage snapshot (publishing); asking M2 and M3 together is best, because S9 makes both part of the lock. M4 optional: a third key copy in his password manager.

## D. Lock day (Claude in this repo, after B14 and M2)

1. `git diff --cached` empty; `git status --porcelain -- rc12 oodh corpus/oodh.py experiments/E001_battery_and_probes/code experiments/E004_general_updating/code` shows only `rc12/lock/*`, the `oodh/HASHES.txt` GATE section, `rc12/notes.txt` and `rc12/LOCK_STEPS.md`; `git diff --stat C HEAD -- rc12 oodh corpus/oodh.py experiments/E001_battery_and_probes/code experiments/E004_general_updating/code` shows no code file.
2. `git mv prereg/RC-12.draft.txt prereg/RC-12.md`
3. `$P -B rc12/lock_finalize.py --date <today> --sealed-date <B14 date>`: fixed edits, each matching once, and s20, the lock record copied from `rc12/lock/SEALED_HASHES.txt` and the draw log.
4. Nothing else changes for the rename: the four tests that read the prereg read `RC-12.md` once it exists, and `test_lock_c` checks the locked file instead of re-running the finalize.
5. `$P -B rc12/lock_check.py manifest`, then `$P -B rc12/lock_check.py check --final` (PASS).
6. Stage exactly: `git add prereg/RC-12.md rc12/lock/RC-12-sealed.tar.enc rc12/lock/SEALED_HASHES.txt rc12/lock/SEALED_GATES.txt rc12/lock/SEED_ATTEMPTS.txt rc12/lock/MANIFEST.txt oodh/HASHES.txt rc12/notes.txt rc12/LOCK_STEPS.md`; `git diff --cached --name-status` must show those that changed plus `D prereg/RC-12.draft.txt` (or `R`), nothing else; then `git commit` (no pathspec) with a "RC-12 LOCKED" message.
7. Before the push: `git worktree add <tmp> HEAD`; in it `lock_check.py check --final --root <tmp>` and the suites (the rc12 tests, `oodh/tests`, the pc suite). Push only if all pass; remove the worktree.
8. Push. Record in `rc12/notes.txt` the commit sha, `C`, and the push time from the PushEvent's `created_at` in `gh api repos/<owner>/<repo>/events` (a commit date can be edited; the event time cannot); commit `rc12/notes.txt` alone and push.

## E. Release and snapshot (only after M3)

**GitHub release:** `gh release create rc12-lock-<date> --target <lock sha> --title "RC-12 pre-registration, locked <date>" --notes "Locked RC-12 pre-registration (prereg/RC-12.md), the encrypted sealed split (rc12/lock/RC-12-sealed.tar.enc) and its hashes. The key and the sealed split are published at submission."` It records a tag on the lock commit and a release whose creation time GitHub stamps; check with `gh release view rc12-lock-<date> --json createdAt,tagName,targetCommitish`.

**Software Heritage, Save Code Now:** `curl -s -X POST https://archive.softwareheritage.org/api/1/origin/save/git/url/https://github.com/<owner>/<repo>/` (or the form at https://archive.softwareheritage.org/save/, origin type git). It returns the request id, `save_request_status` and `save_task_status`. Poll the same URL with GET until `save_task_status` is `succeeded`, then record its `visit_date` and `snapshot_swhid` (swh:1:snp:...) in `rc12/notes.txt`. The snapshot is an independent, timestamped copy of every branch and tag at that visit, the lock commit included. Re-check the two key fingerprints (B10) the same day, then commit `rc12/notes.txt` alone and push.

## After the lock

Scoring on sealed (S10) decrypts into a work copy of the lock commit with the archived content installed, because the graders read the sealed pools. Sealed transcripts are never committed or pushed before submission. The OOD-H wording rule needs both models' OOD-H runs; OOD-H Part 2 comes in January.
