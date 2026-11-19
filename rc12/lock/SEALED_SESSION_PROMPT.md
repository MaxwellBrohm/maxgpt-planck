# RC-12 sealed session: the prompt (lock track B)

This file is the whole prompt for the separate sealed Claude session of `rc12/LOCK_STEPS.md` section B (prereg s13 S1 to S8). It is committed with placeholders only. The main session fills in the placeholders, gives the filled text to a fresh session, and never commits the filled copy or pastes it anywhere else.

## For the main session, before launching

- Track A is finished: every code change is committed, A3 passed on an export of `C`, and nobody touches `rc12/` or `oodh/` on the lock branch until the sealed session's B14 report is back.
- Max is at hand. The PC must answer over Tailscale for B10 and B12 (Max turns the PC on and clicks Connect himself), and he may add the optional third key copy (LOCK_STEPS M4).
- Launch it the way Max chose on 2026-10-19 ("One-click separate session (Recommended)"): a separate session started from a task card. Such a session opens in a fresh worktree of its own; no step runs there (Rule 1 below).
- **Its transcript counts as `sealed/`.** The session writes every content value and template through file-write calls, so its transcript holds the item text. No other session, agent, workflow or tool reads, searches, exports, summarizes or resumes that transcript, before or after the lock (LOCK_STEPS rules line). Max answers its questions in the session without opening its file writes.
- Fill in:

| Placeholder | What it is |
|---|---|
| `<C>` | the full sha of commit `C` from A3 |
| `<MAIN>` | the main checkout. The session only runs `git worktree add` against its repository and copies `sealed/oodh` out of it; env.sh sources its `pc/local.env` read only |
| `<SEALED_WT>` | a new path for the session's own worktree: durable (not a temp or scratch folder), not inside any git work tree, not in a cloud-synced folder |
| `<SESSION_DIR>` | an existing folder outside every repository, for `env.sh`, the B12 helper script and the B14 copy of the S6 list (paths and hashes only: no secret, no item text) |
| `<P>` | the planck venv's python, the one every rc12 step uses |
| `<RC12_KEY>` | the key path: outside every repository, not existing yet (for example a `planck-keys` folder under the Mac's Application Support) |
| `<PCWSL>` | `pcwsl.sh` in the Mac's `planck-e2chain` support folder (it copies a script to the PC and runs it in WSL) |

Everything below the line is the prompt.

---

You are the RC-12 sealed session.

## What you are doing

RC-12 is the pre-registered 12-turn chat test of the MaxGPT-Planck project. Its sealed split must be written fresh, by you, before the pre-registration is locked. You write new content for every family, build the 640 sealed records from it with a drawn seed, run every gate on them with fake responders only, encrypt them, and hand back only public outputs: gate tables, the draw log, counts and hashes. Nobody else ever sees the item text before submission.

The protocol is `rc12/LOCK_STEPS.md` section B as it stands at commit `<C>`. This prompt adds only two things: where you work (your own worktree at `<C>`, not the main checkout) and how to run the steps in this environment. If a command below differs from section B at `<C>` in anything other than those two things, follow LOCK_STEPS at `<C>` and say so in your report.

You are the only session that may read or write under `sealed/` and `private/`.

## Rules

1. **Work only in your worktree `<SEALED_WT>`.** The main checkout `<MAIN>` is diverged from origin and holds other sessions' uncommitted work: never edit, build or run a step there. From `<MAIN>` you only (a) read commit `<C>` from its repository with `git rev-parse`, `git cat-file` and `git show` (W2) and run `git worktree add` against it (W4), (b) copy `sealed/oodh` and compare your copy with it (W6, and step 5 of the code-change restart), and (c) source `pc/local.env`, read only, through env.sh. Never list or open anything else under `<MAIN>/sealed/`, and nothing under `<MAIN>/private/`. If you were started in a worktree of your own (a task card opens one), run no step there either: W1 to W4 name their paths, and from W5 on env.sh moves you into `<SEALED_WT>`.
2. **Do not read `pipeline/`** (s13 S2 forbids its banks, templates, teacher renders and lookup-line formats; nothing in B needs the rest of it) **or `research/`.** Do not load a model. Do not commit or push. In your worktree never run `git checkout`, `git switch`, `git reset`, `git stash`, `git clean` or `git worktree remove`. Never set `GIT_DIR`, `GIT_WORK_TREE` or `GIT_INDEX_FILE`.
3. **Tracked files you may change:** only the four files B11 writes in `rc12/lock/` and the GATE section that `oodh/gate.py` rewrites in `oodh/HASHES.txt` (B7). Never this prompt's own file, `rc12/lock/SEALED_SESSION_PROMPT.md`. Everything else you write goes under `sealed/` or `private/` (both gitignored) or into `<SESSION_DIR>`.
4. **Item text never leaves this session.** Only gate tables, the draw log, counts and hashes leave it. Your transcript holds the content you write, and no other session may read it (it counts as `sealed/`). No item text, content value, seed or key goes into your replies, `<SESSION_DIR>`, a memory file, a notes file, a checklist or a message. No artifact, upload, file send, web search or fetch carries it. Use no subagents or workflows: s13 S1 has one session write the content.
5. **The content is yours** (s13 S1; D8 provenance: a Claude-written gold set, never training text). Write every value, template and filler yourself. Never copy text from the web, a dataset, another repository or the training pipeline.
6. **The key** is made only by `sealed_crypto.py keygen`. Never print it, `cat` it, copy it by hand or put it on a command line. It never goes into a repository, a notes file, MEMORY.md or a transcript. Only its fingerprint (the first 16 hex of the key file's sha256) may appear.
7. **Git** runs with `DEVELOPER_DIR=/Library/Developer/CommandLineTools` (env.sh sets it; the kit's scripts call git too). Python runs only as `$P -B`. At most 4 worker processes (the kit sets `RC12_MAX_WORKERS=4`), and one heavy step at a time.
8. **Report what is true.** A failed or skipped step is reported with its output, never smoothed over.

## Reading list

At `<C>`: the rules line at the top of `rc12/LOCK_STEPS.md` and its section B; prereg `prereg/RC-12.draft.txt` s3 (the filler and length rules), s5, s6 (Inflected forms), s8, s13 and s14; `rc12/notes.txt` STEP 15. Beyond that, read only what a step needs: the skeleton files you rewrite, the kit's messages, the gate logs, and the code a failure points to.

## Running commands here

- Shell state does not persist between commands. From W5 on, start every command with `. <SESSION_DIR>/env.sh &&`. It sets the variables, sources `pc/local.env` and changes into the worktree.
- B4, B6, B7 and B9 can outlast one foreground command. Run them in the background, with their output in a log under `sealed/rc12_work_logs/` (`mkdir -p` it first). That folder is gitignored and may hold item text; B13's `sealed/rc12_work*` removes it. Wait on a condition (for example a Monitor until-loop), not on fixed sleeps.
- "Press Ctrl-C" in B4 means `pkill -INT -f 'rc12/sealed_search.py'`. Only this session ever runs that script, so the pattern hits nothing else. On the interrupt, Python's subprocess handling kills the step the search was running. Then `pgrep -fl 'rc12/sealed_search.py'` must print nothing.
- Kill only processes you started. Other sessions may be running rc12 tests in other trees at the same time.

## Preflight, W1 to W6 (before B0)

Stop at the first failure and report it. Create nothing before W3 passes.

**W1. Tools.**
- `openssl version` must print OpenSSL 3, because s13 S7 names OpenSSL 3. If the default `openssl` is another major version, find an OpenSSL 3 (Homebrew keeps one at `$(brew --prefix openssl@3)/bin`). Put that folder first on PATH in env.sh (W5) and check again. If there is none, stop.
- Also check: `<P> --version`; `DEVELOPER_DIR=/Library/Developer/CommandLineTools git --version`; `test -x <PCWSL>`; `test -f <MAIN>/pc/local.env` (never print the file); `test ! -e <RC12_KEY>`.

**W2. Commit C.**
- `DEVELOPER_DIR=/Library/Developer/CommandLineTools git -C <MAIN> rev-parse --verify '<C>^{commit}'` must print `<C>`.
- Every script that section B names must exist at `<C>`. Check each with `git -C <MAIN> cat-file -e <C>:<path>`. Among them: `rc12/sealed_kit.py`, `rc12/sealed_overlap.py`, `rc12/sealed_search.py`, `rc12/sealed_crypto.py`, `rc12/sealed_pack.py`, `rc12/sealed_public.py`, `oodh/gate.py`, `oodh/a2_counts.py` and the `rc12/audit_a2.py` it imports. If one is missing, track A has not finished: stop.
- Read section B of `git -C <MAIN> show <C>:rc12/LOCK_STEPS.md`. Note any difference from this prompt.

**W3. The PC.** B10 and B12 need it. S7 asks for the encryption at once in this session, and B13 waits for both verifies. So the build does not start unless the PC answers now.

First run `tailscale status`. If it says Tailscale is stopped, ask Max to turn the PC on and click Connect in the Mac's Tailscale menu, then wait for his word. Never start, stop or reconfigure Tailscale, a VPN or any network setting yourself. Then:

```
set -a && . <MAIN>/pc/local.env && set +a && ssh -i "$PLANCK_PC_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 "$PLANCK_PC_HOST" "wsl -d Ubuntu --exec bash -c \"test ! -e ~/.planck_keys/rc12_sealed.key && echo KEYFREE; openssl version; python3 --version\"" | tr -d '\r'
```

Run this as one command: the variables do not survive into a second one. The output must show `KEYFREE`, an OpenSSL 3 line and a Python 3 line.
- If ssh fails or times out, stop and report "blocked on the PC". Nothing has been created yet.
- If `KEYFREE` is missing, a key file already sits on the PC: stop and ask Max. Never delete it.

**W4. Your worktree.**

```
DEVELOPER_DIR=/Library/Developer/CommandLineTools git -C <MAIN> worktree add --detach <SEALED_WT> <C>
```

This writes only git's worktree record in `<MAIN>`'s repository metadata. `<MAIN>`'s files, index and branches are untouched. Then, in `<SEALED_WT>`:
- `git rev-parse HEAD` prints `<C>`;
- `git status --porcelain` prints nothing;
- `git check-ignore -q sealed/x private/x` exits 0.

**W5. env.sh.** Write `<SESSION_DIR>/env.sh` with mode 600 (paths only):

```
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
export P='<P>' C='<C>' RC12_KEY='<RC12_KEY>' PCWSL='<PCWSL>' RC12_MAX_WORKERS=4
# export PATH='<the OpenSSL 3 bin folder>':"$PATH"     (only if W1 needed it)
set -a; . '<MAIN>/pc/local.env'; set +a
cd '<SEALED_WT>'
```

Check: `. <SESSION_DIR>/env.sh && pwd && openssl version && git rev-parse HEAD`.

**W6. OOD-H Part 1, and private/.** B7 to B9 read `sealed/oodh`, which exists only in the main checkout. Copy that one folder, nothing else:

```
. <SESSION_DIR>/env.sh && mkdir -m 700 sealed private && cp -Rp '<MAIN>/sealed/oodh' sealed/oodh && diff -rq '<MAIN>/sealed/oodh' sealed/oodh && echo SAME
. <SESSION_DIR>/env.sh && want=$(awk '$1=="sealed/oodh/oodh_part1.jsonl" && $2=="sha256" {print $3}' oodh/HASHES.txt) && got=$(shasum -a 256 sealed/oodh/oodh_part1.jsonl | cut -d' ' -f1) && [ -n "$want" ] && [ "$want" = "$got" ] && echo PART1_OK
```

The output must show `SAME` and `PART1_OK`; the sha256 begins `40d2a76d1315d3cb`.

## The steps (LOCK_STEPS section B, run from `<SEALED_WT>`)

The variables are in env.sh. Paths are relative to the worktree root.

**B0. Pin the code.**

```
git diff --quiet "$C" HEAD -- rc12 oodh corpus/oodh.py experiments/E001_battery_and_probes/code experiments/E004_general_updating/code ':!*.txt' ':!*.md' && echo B0_OK
git status --porcelain -- rc12 oodh corpus/oodh.py experiments/E001_battery_and_probes/code experiments/E004_general_updating/code
```

The first must print `B0_OK`. The second must show no code file: text files such as `rc12/notes.txt` or `oodh/HASHES.txt` may move, code may not. Otherwise stop. Repeat B0 before B6 and before B9.

Then run **B7** (below) now, before B1. It reads only the code at `<C>` and `sealed/oodh`, and nothing B1 to B6 writes changes it, so a G1 failure (which needs a code change) stops you before any build or draw.

**B1.** `$P -B rc12/sealed_kit.py prepare sealed/rc12_work --commit $C`

**B2.** `$P -B rc12/sealed_overlap.py skeleton sealed/rc12_content` writes one `content_<module>.py` per content module (11) at the dev values. Rewrite every value as new content of your own under S1 and s3. A content file is data, never code: only its docstring and `NAME = <literal>` lines are accepted (`split_content.load_literals`).

- **New:**
  - the value pools;
  - the BIND intros and the neutral statement tails (`content_balance.py`);
  - HOLDER_F (female), HOLDER_M (male) and HOLDER_N (pronouns depend on the first two);
  - the PERSIST rule words;
  - the templates;
  - the fillers with their hand labels ("question", "request", "statement"). A filler holds no digit, name-like word, pool value, holder word or rule word, and shares 0 word 5-grams with the E001 and E004 fillers. Each has a one-sentence IDEAL reply;
  - the LOOKUP table format (`ROW`, `ROW_SEP`, OD5).

  User turns stay at 35 words or fewer (tables 70).
- **Shared with dev:** only WEEKDAY, MONTH (still without "May") and LANGUAGE. Every other string is new, short ones too: objects, aliases, topics, nouns, frames and questions such as "What's my name?" (S1 (d)).
- **May repeat dev's words** (S1 (d) exempts them; `sealed_overlap.D_SKIP`, `D_SKIP_NAMES`): the fillers' hand labels; the avoid lists `BIND_AVOID`, `OWN_AVOID` and `LOOKUP_AVOID`; BIND_NAME's `objw`.
- **The avoid lists.** Each one names the object words its own items use; no filler of the item may contain them.
  - `OWN_AVOID`: the object word that runs through each new OWN_PICK item's `q`, `before`, `after` and `probe` (dev: boat, puppy, band, quiz, team).
  - `LOOKUP_AVOID`: the object word through each LOOKUP item's `intro`, `q`, `ideal` and `abst` (dev: rota, tour, menu).
  - `BIND_AVOID`: the new BIND object words (`obj` or BIND_NAME `objw`; dev: bike, kitten, instrument, trip, name).

  Rewrite each list to match your new templates; keep a dev word only where your template keeps that object. OWN_AVOID has 5 slots for 4 items: the spare holds a second object word of one item (dev: quiz and team) or repeats one, never a common word.

  test_gens refuses:
  - an avoid word that no template uses (for BIND: no object word);
  - an OWN or LOOKUP item whose four texts share no word of its list;
  - an avoid word that runs through several items (an object word belongs to its own item).

  `{placeholders}` are not words. Case is ignored, as the build ignores it. An avoid word is letters only, as test_gens reads words.
- **Unchanged:** every name, type, dict key, list length and `{placeholder}` (`split_content.shape_diff` refuses anything else), and the identifier fields of `split_content.PINS` (RECALL_SLOTS `key`/`vtype`, TWOHOP `vtype`, LOOKUP `keys`/`vals`, the K kind). Keys are free only where CONTENT marks a name with `*` (OWN_LIST, PROJECTS; PROJECTS keys equal the new PROJECT pool).
- **Plural and singular forms.** No weekday, instrument, food or project value may have a plural or singular that is another common word. At least one instrument, food or project value ends in s, as dev's "drums" does. test_gens checks this: the grader's singular-form rule needs such a value.
- **No hedge or alternative cue in OWN pick values** (the BOAT, PUPPY, BAND and QUIZTEAM pools): no "guess", "seems", "maybe", "or", "/" or "?". The grader never counts a name it reads as hedged, an alternative or a question as asserted. A G6 source fixture would then get the wrong verdict on any seed whose chosen OWN pick records use it.
- Count your labels as q, r, s.

**B3.** `$P -B rc12/sealed_kit.py install sealed/rc12_work sealed/rc12_content`, then `$P -B rc12/sealed_kit.py check sealed/rc12_work`. The check must print that the copy equals its commit.

**B4. Seed draws (s13 S3 (iv)).** Run in the background, with output to `sealed/rc12_work_logs/b4.log`:

```
$P -B rc12/sealed_search.py sealed/rc12_work --seed-file sealed/rc12_seed.txt --log sealed/rc12_draws.txt --expect-labels q,r,s --max 200
```

The draw log lives outside the work copy and keeps every draw of every run. Each draw builds, then runs test_gens, G10, G1-G4, mutation_graders and mutation_audit, in that order (their detectors and fixtures depend on the items). It stops at the first failure and discards a failing seed; the first passing seed is kept. The cap is 200 draws per run. A new run on a log that already has draws (a restart, or a run after 200 failures) needs `--supersede`, and the log counts every draw. The dry run's searches took 1 to 20 draws, each 1 to 3 minutes. If the search exits 1 (no seed in 200), stop and write the report: what happens next is the main session's and Max's decision.

Watch the first draws. A content rule (test_gens `test_pools`, the fillers, the avoid lists) fails every seed the same way. So wait until the log holds this run's third `draw` line. If all three failed test_gens:
1. Ctrl-C right away (see "Running commands here"). The third draw's gate logs stay only until the next build ends; the interrupted draw gets no line.
2. Read `sealed/rc12_work/gate_logs/test_gens.txt`.
3. Fix the content and reinstall (B3).
4. Restart with `--supersede "test_gens content rule"`. The supersede line records why.

test_gens' G9 balance checks also fail by chance, so read the log before deciding the content is the cause.

**B5. S1 separation**, right after the items exist: `$P -B rc12/sealed_kit.py overlap sealed/rc12_work`. It must print `S1 overall: PASS`. It writes the counts, never text, to `sealed/rc12_work/OVERLAP.txt`, which the pack archives. As controls: on dev content it prints FAIL on (a) to (d), and on renamed values with dev templates FAIL on (b) to (d).

**Restart after a fix.** If B5 or B6 fails on content:
1. Fix `sealed/rc12_content`.
2. `mv sealed/rc12_seed.txt sealed/rc12_seed.rejected.<n>.txt`, then `mv sealed/rc12_work sealed/rc12_work.old<n>`.
3. B1 again (a fresh `sealed/rc12_work`), then B3.
4. B4 with `--supersede "<step and module, e.g. S1 (b) pools_fill_a>"`. The reason is published, so it must match `sealed_search.REASON`: a step and a module, never item text. The same `sealed/rc12_draws.txt` continues, and the earlier acceptance is void.

If B6 fails on a seed-sensitive gate and the content is not the cause, the same restart without a content change redraws the seed.

If a failure needs a code change: stop, write the gate table to `private/rc12_sealed_report.txt` and hand back to the main session. It fixes, commits and reruns A3 for a new `C`, and gives you the new `C` and a new worktree path. Then, before anything else (`<OLD_WT>` is the worktree you used so far, `<NEW_WT>` the new path):
1. W2 again for the new `C`.
2. W4 with `<NEW_WT>` and the new `C`, and its three checks. The fresh worktree has no `sealed/` or `private/`.
3. Record the draw count first: `grep -c '^draw ' <OLD_WT>/sealed/rc12_draws.txt`. Then move the plaintext and the draw log whole, never copy them: `mv <OLD_WT>/sealed <NEW_WT>/sealed && mv <OLD_WT>/private <NEW_WT>/private`. Then `ls -A <OLD_WT>/sealed <OLD_WT>/private` must say both are gone, and `grep -c '^draw ' <NEW_WT>/sealed/rc12_draws.txt` must print the same count. The draw log must continue in the new worktree: `sealed_search.py` starts a fresh log without `--supersede` when it finds none, and a fresh log would drop every earlier draw from SEED_ATTEMPTS.txt (s13 S3 publishes every draw).
4. W5 again: rewrite env.sh with the new `C` and `cd '<NEW_WT>'`, and run its check.
5. W6's checks on the moved copy, without copying again: `diff -rq -x gate_report.json '<MAIN>/sealed/oodh' sealed/oodh && echo SAME` (B7's `gate.py` rewrote `gate_report.json` in your copy, so it is left out), then the `PART1_OK` command as written.
6. `mv sealed/rc12_seed.txt sealed/rc12_seed.rejected.<n>.txt` (if it exists), then `mv sealed/rc12_work sealed/rc12_work.old<n>`.
7. B0 at the new `C`, B7 again (the new code), B1 (a fresh `sealed/rc12_work`), B2 only if the content needs a change, B3, then B4 with `--supersede "code change, new C <first 8 hex of C>"`.

Leave `<OLD_WT>` in place (never `git worktree remove`), now with no `sealed/` or `private/`, and name both worktrees in the report.

**B6. Every S4 gate.** Repeat B0 first. Then run in the background, with output to `sealed/rc12_work_logs/b6.log`:

```
$P -B rc12/sealed_kit.py gates sealed/rc12_work --seed-file sealed/rc12_seed.txt --expect sealed/rc12_work/expect.json
```

`sealed/rc12_work/GATES.txt` must end with `failing steps: 0`. The run takes about 20 to 30 minutes. In the dry run one B6 run stalled in `mutation_e2e` part 2 for over two hours with its parent idle. If a step uses no CPU for 10 minutes and has no worker processes (check with `ps -o pid,ppid,pcpu,etime,command` under the gates process), kill the gates run: first the stalled step, then the `sealed_kit.py gates` process. Then run B6 again from the start: the table must come from one run.

**B7. OOD-H gate (s14 (d) and (e)).** Run it right after the first B0, before B1 (see B0).
1. `$P -B oodh/gate.py` must exit 0: its G1 must pass with the lock's graders. If it does not, stop: the fix is a code change. Keep the GATE section it writes in `oodh/HASHES.txt`, and re-report its G2 rates. The `oodh_part1.jsonl` sha256 line must stay `40d2a76d1315d3cb...`.
2. Then `$P -B oodh/a2_counts.py` (lock step A2, s14 (e); counts only, no item text). Report its table in B14; it gates nothing.

**B8. S6 list.** `$P -B rc12/sealed_crypto.py ngrams sealed/rc12_work --oodh sealed/oodh/oodh_part1.jsonl --out private/rc12_decon13_sha256.txt.gz`

**B9. S5 pack.** Repeat B0 first. Then:

```
$P -B rc12/sealed_pack.py sealed/rc12_work --seed-file sealed/rc12_seed.txt --log sealed/rc12_draws.txt --oodh sealed/oodh/oodh_part1.jsonl --ngrams private/rc12_decon13_sha256.txt.gz --out sealed/rc12_pack
```

This writes the deterministic `RC-12-sealed.tar` (mode 600) and `SEALED_HASHES.txt` (with the S6 count and the S1 verdict). It refuses unless all of these hold:
- the copy equals `C`;
- GATES.txt is a full clean table whose recorded inputs are the files packed;
- the seed rebuilds the items byte for byte;
- the draw log's last run accepted this seed;
- OVERLAP.txt passed for these items.

Before B11, read `sealed/rc12_work/GATES.txt`, `sealed/rc12_draws.txt` and `sealed/rc12_pack/SEALED_HASHES.txt` and confirm they hold no item text. The kit withholds quoted lines and the content's own words, but read them anyway.

**B10. Key (S7, S8).** As written in LOCK_STEPS:

> `$P -B rc12/sealed_crypto.py keygen "$RC12_KEY"` prints only a fingerprint. Copy it straight into WSL, never through the Windows file system (source `pc/local.env` first):
>
> ```
> ssh -i "$PLANCK_PC_KEY" -o IdentitiesOnly=yes -o BatchMode=yes "$PLANCK_PC_HOST" "wsl -d Ubuntu --exec bash -c \"umask 077; mkdir -p ~/.planck_keys; cat > ~/.planck_keys/rc12_sealed.key; sha256sum ~/.planck_keys/rc12_sealed.key; stat -c %a ~/.planck_keys/rc12_sealed.key\"" < "$RC12_KEY" | tr -d '\r' | cut -c1-16
> ```
>
> The first printed line (the fingerprint) must equal the Mac's and the second must be `600` (this exact command passed the dry run with a dummy key, STEP 15). Max may add a third copy to his password manager. The key never goes into a repo, a notes file, MEMORY.md or a transcript.

Here, `pc/local.env` is gitignored and not in your worktree: env.sh sources the main checkout's copy, so the command runs exactly as written. If the PC does not answer now, stop before B11. Keep the plaintext where it is (gitignored, in your worktree), report, and resume at B10 when Max says the PC is back. If the fingerprints differ or the mode is not 600, stop and report; do not encrypt. Once both lines are right, tell Max the key file's path (never its content), so he can add the optional third copy himself (M4).

**B11. Encrypt and publish the public outputs.**

```
$P -B rc12/sealed_crypto.py encrypt sealed/rc12_pack/RC-12-sealed.tar --key "$RC12_KEY" --out rc12/lock/RC-12-sealed.tar.enc --hashes sealed/rc12_pack/SEALED_HASHES.txt
cp sealed/rc12_pack/SEALED_HASHES.txt rc12/lock/
cp sealed/rc12_work/GATES.txt rc12/lock/SEALED_GATES.txt
cp sealed/rc12_draws.txt rc12/lock/SEED_ATTEMPTS.txt
```

**B12. Verify on both machines** (s19 step 9's key check). As written in LOCK_STEPS:

> Mac: `$P -B rc12/sealed_crypto.py verify rc12/lock/RC-12-sealed.tar.enc --key "$RC12_KEY" --hashes rc12/lock/SEALED_HASHES.txt` prints PASS. PC: scp `rc12/sealed_crypto.py`, `rc12/sealed_kit.py`, `rc12/sealed_public.py` (sealed_kit imports it), the `.enc` and `SEALED_HASHES.txt` to `planck-kit`, then through `pcwsl.sh` run `cd "$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' | tr -d '\r')")/planck-kit" && python3 -B sealed_crypto.py verify RC-12-sealed.tar.enc --key ~/.planck_keys/rc12_sealed.key --hashes SEALED_HASHES.txt`, which prints PASS. The dry run passed both (STEP 15).

The PC part, concretely:
1. Copy the files:
   ```
   scp -q -i "$PLANCK_PC_KEY" -o IdentitiesOnly=yes -o BatchMode=yes rc12/sealed_crypto.py rc12/sealed_kit.py rc12/sealed_public.py rc12/lock/RC-12-sealed.tar.enc rc12/lock/SEALED_HASHES.txt "$PLANCK_PC_HOST":planck-kit/
   ```
2. Write `<SESSION_DIR>/b12_pc_verify.sh`: the line `#!/bin/bash`, then the one `cd ... && python3 -B sealed_crypto.py verify ...` line quoted above, exactly as written.
3. Run `"$PCWSL" <SESSION_DIR>/b12_pc_verify.sh`.

Both verifies must print PASS. If either fails, or the PC does not answer, do not run B13: report and wait.

**B13. Remove the plaintext (S7)**, only after both verifies pass. Remove `sealed/rc12_work*` (this covers `sealed/rc12_work_logs` and every `sealed/rc12_work.old<n>`), `sealed/rc12_pack`, `sealed/rc12_content`, `sealed/rc12_seed*.txt`, any `sealed/rc12_seed.txt.try*` and `sealed/rc12_draws.txt` (its public copy is in `rc12/lock`). The archive holds every file that counts. Then remove `sealed/oodh` too: it is your copy from W6 (the archive holds Part 1, and the original stays in the main checkout's `sealed/oodh`, the one copy OOD-H scoring reads after the lock). Then `ls -A sealed` must print nothing. If you restarted for a code change, also run `ls -A <OLD_WT>/sealed <OLD_WT>/private` for every earlier worktree: each must say both are gone (they moved to this worktree), so no plaintext outlives B13 anywhere.

**B14. Report.** First copy the S6 list out of `private/` for the pipeline, which never reads under `private/`, and check the copy:

```
cp -p private/rc12_decon13_sha256.txt.gz '<SESSION_DIR>/rc12_decon13_sha256.txt.gz' && shasum -a 256 '<SESSION_DIR>/rc12_decon13_sha256.txt.gz'
```

Its sha256 must equal the S6 list's line in `rc12/lock/SEALED_HASHES.txt`. Then report, counts only, in `private/rc12_sealed_report.txt` and in your reply:
- the `rc12/lock/` file list;
- the B6 gate table;
- the draw count and restarts;
- the sealed shortcut ceiling;
- the S1 counts (0, 0, 0);
- the OOD-H gate verdict and its G2 rates;
- the A2 counts (B7);
- the S6 count;
- both key fingerprints (matching);
- the date (the sealed date for D).

Add:
- the preflight results, including the OpenSSL versions on the Mac and the PC;
- `<SEALED_WT>`, and every earlier worktree of a code-change restart;
- each hand-off file's sha256, and the path and sha256 of the S6 list's copy in `<SESSION_DIR>`;
- any difference you found between this prompt and LOCK_STEPS at `<C>`.

## Stop and report (and do not go on) when

- any preflight check fails (W1 to W6);
- B0 shows a code change;
- B4 exits 1;
- a failure needs a code change;
- B6 cannot finish clean after the restarts above;
- B7's gate exits non-zero or the Part 1 sha256 moved;
- B9 refuses;
- the PC does not answer at B10 or B12;
- a fingerprint or mode check fails;
- either verify fails.

A stop leaves everything under `sealed/` and `private/` as it is. The report says which step stopped, with its output (counts and gate lines only).

## Hand-off

Leave these uncommitted in `<SEALED_WT>`; the main session copies them for lock day (D) and checks them against the sha256 values in your report:
- `rc12/lock/RC-12-sealed.tar.enc`
- `rc12/lock/SEALED_HASHES.txt`
- `rc12/lock/SEALED_GATES.txt`
- `rc12/lock/SEED_ATTEMPTS.txt`
- `oodh/HASHES.txt` (its new GATE section)

Do not remove the worktree. `git worktree remove` deletes gitignored files without asking, and in this worktree `private/` holds the S6 list and your report (before B13, `sealed/` also holds plaintext and your copy of OOD-H Part 1).
