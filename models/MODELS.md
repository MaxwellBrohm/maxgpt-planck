# Planck models: what we keep, where, and how

Every model this project trains, fine-tunes or tweaks is kept, with one line in
[`registry.jsonl`](registry.jsonl): successes, failures, diagnostics and runs that were killed. A failure
with its weights is evidence; a failure without them is an anecdote. Weights never go into git
(`.gitignore` has `*.pt`, `*.safetensors`, `experiments/**/weights/`); the registry is what git carries.

Written 2026-09-26. Tools: [`registry.py`](registry.py) (validate, list, add, verify, hash) and
[`regtools.py`](regtools.py) (run readers, hashing, safetensors headers). Both are stdlib only, and
neither imports torch or loads weights. Tests: `python3 models/tests/test_registry.py` and
`python3 models/tests/test_registry_cross.py`.

## 1. What is kept

- **Final weights of every run that took a real optimizer step**, whatever its outcome: scored seeds,
  reruns, LR-search points, ablation arms, failures. A queue passes `--save 1` (or the harness default)
  for every such job, including LR searches. Saving never changes a result; it is still logged when an
  experiment turns it on after pre-registration.
- **Diverged or crashed runs**: the last finite checkpoint. The harness raises on a non-finite loss
  without saving it, so its rolling checkpoints before the divergence are the record; they are not pruned.
- **Killed and dry runs** get a registry line with `status: not_saved` and `files: []`, so the count of
  trained models is complete. Dry runs (a handful of steps on padded batches, a pipeline check) are not
  saved by default.
- **Intermediate checkpoints (thinning)**, for from-scratch runs:
  1. always: `final_<step>.pt`, every `stable_<step>.pt` (branch points; the harness never prunes them),
     and every checkpoint a result was read from (E2's scorer reads final plus the two rolling before it);
  2. when a config keeps more rolling checkpoints (`keep_last 0`), the archive keeps those nearest 1%,
     3%, 10% and 30% of the run and drops the rest, and only after the run is registered and its results
     are read;
  3. pre-registered configs are not changed to keep more (E2/E3 and the S00x screens keep 2 rolling).
- **Public models** that are only evaluated (RC-12 baselines, E001) are not copied: their HF id and
  snapshot revision identify them.

## 2. Where

| place | location string | what lives there |
|---|---|---|
| Mac, in the repo | `mac:experiments/<E>/weights/<slug>__<tag>` | where Mac runs write; copies of small models stay |
| PC archive (D: drive) | `pc:/mnt/d/planck-archive/<id>` | every model, one directory per registry id |
| PC run areas | `pc:~/planck/runs/...`, `pc:~/planck/e006_run/weights/...` | where PC runs write; not the archive |
| off-site (optional) | `hf:<owner>/<repo>@<revision>` | not set up |

- **Small models keep a Mac copy**: up to 600 MB of files per model (every 135M fp32 fine-tune fits),
  within a 50 GB budget for all Mac copies. Mac copies on 2026-09-26: 33 models, 9.95 GB. Larger models
  live on the PC archive only.
- **The archive is the D: drive** (about 466 GB free), as `/mnt/d/planck-archive` from WSL. Layout: the
  registry id is the path, e.g. `/mnt/d/planck-archive/E005/135m/s1/model.safetensors`.
- `~/planck/dev/...` is a working area that has been cleaned by another process once; a copy there
  (for example E006's reference copies of E004/E005) is listed but is never the only copy.
- **Off-site backup (option, needs Max)**: a private Hugging Face model repo per experiment, pushed with
  `huggingface-cli upload`. This needs Max's HF token with write scope on the PC; nothing is uploaded
  until he provides it and says private or public. The location is then `hf:<owner>/<repo>@<commit>`.

## 3. Naming

- **id** = `<experiment>/<arm>/<run tag>[/<checkpoint>]`, stable forever, and the archive path.
  Examples: `E005/135m/s3`, `E003/p14m/lr1e-02/s0`, `E002/135m/s0r`, `E2/smoke/trunk/stable_400`,
  and later `E2/5m_e3_r1/trunk/final`. A killed attempt adds `/killed-<YYYYMMDDTHHMMSS>` (its end time).
- The arm is the model or arm short name the experiment's queue uses (`135m`, `ts1m`, `p70m`, `C`, `P`).
- A model is never renamed; a rerun is a new id (`s0r`, `<tag>_r2`).

## 4. Registry fields (schema 1)

One JSON object per line. `registry.py validate` enforces every rule below, and across entries: no file at a
location is claimed by two entries, and a checkpoint's `parent` is a registered id.

| field | meaning |
|---|---|
| `id`, `experiment`, `arm`, `role`, `seed` | identity; `role` is scored, rerun, lr_search, dry_run, smoke, branch_point, ... |
| `kind` | `finetune` (needs `base_model`), `from_scratch` (`base_model` null), `checkpoint` (needs `parent` id and `step`) |
| `base_model` | `{id, revision, revision_source}`: the HF id and the snapshot commit trained from |
| `params_total`, `params_body` | body = total minus token and position embeddings and an untied output head; `params_source` says where they came from |
| `tokenizer` | `{id, sha256}`: sha256 of the `tokenizer.json` shipped beside the weights (a not_saved run: the base snapshot's) |
| `data` | `{identity, sha256}`: the manifest or seeded stream trained on, and its sha256 where one exists |
| `code` | `{git_commit, relation, digest_sha256, digest_source}`: see below |
| `config_sha256` | harness runs: `config_sha256` from runs.jsonl. HF fine-tunes: sha256 of the canonical JSON `{"script", "args"}` (sorted keys, no spaces) of the training job's guard record command after the interpreter (the `ft_test.py` job, never an eval job such as `gen_probe.py` that reuses the tag); `config_source` names the record |
| `hparams` | steps, LR, batch and precision as recorded |
| `result` | the reading the experiment's results.json or AUDIT.md gives this model, or `not scored` |
| `status` | `success` (met its pre-registered rule), `failure` (scored and failed, or the run failed), `partial` (the experiment's reading is partial, e.g. E005 PARTIAL-X), `diagnostic` (a rerun, smoke or branch point, not in a pass-rule tally), `not_saved` (trained, weights not kept) |
| `run` | `{state, started, ended, log}`: completed, killed, dry, in_progress or unknown |
| `files` | `[{path, bytes, sha256}]`, paths relative to each location; `[]` exactly when `status` is not_saved |
| `locations` | `mac:`, `pc:~/planck...`, `pc:/mnt/d/planck-archive...` or `hf:` strings; never a machine address |
| `created` | a saved model: the weights file's mtime; otherwise the run's end |
| `release` | `with_writeup`, `never` or `undecided` (section 7) |
| `notes` | free text; every null among params, tokenizer, data, config, created, base revision and file hashes must be explained here by naming the field |

- **code.relation**: `pre_run` = the commit was made before the run started and the experiment's code
  is unchanged from it to HEAD apart from the process guard and queue scripts; `post_run` = the first
  commit holding the code came after the run, so the run-time code may differ. `digest_sha256` is the
  sha256 of a code-hash list written at run time (E005's `code_sha256_at_start.txt`, the harness's
  `code_sha256.txt`).
- **Never guess.** An unknown field is null, with the reason in `notes`.
- No private data: no absolute home paths, machine addresses or emails (`validate` rejects them).

## 5. How to register

```bash
# a saved E002-E006 style run (weights/<slug>__<tag>, out/<slug>__<tag>__run.json, logs/<job>.guard.json)
python3 models/registry.py add --from-run experiments/E003_correction_floor/weights/roneneldan__TinyStories-33M__s1 \
  --id E003/ts33m/s1 --set arm='"ts33m"' --set status='"failure"' --set result='"..."' \
  --set data='{"identity": "...", "sha256": null}' --set code='{...}' --set notes='"data.sha256: ..."'
# anything else (harness runs, not_saved runs): a JSON object or JSON lines with every field
python3 models/registry.py add --json entry.json
python3 models/registry.py validate
python3 models/registry.py verify --where mac        # re-hash every Mac copy
```

`add --from-run` fills hashes, parameter counts (run.json checks or the safetensors header), the base
revision (only when the local HF cache holds exactly one snapshot, older than the run), the config
sha256, run times and the Mac location from the training job's guard record (`ft_test.py` or `e00N_ft_test.py`); it
refuses a run without a finished one. The
experiment-specific fields are passed with `--set`. The E2 smoke lines are the template for harness
runs (fields from runs.jsonl's start and end events). An entry registered where its files are not
reachable carries null hashes; `verify --where pc --fill` on the PC fills them and never changes a
recorded hash.

**Every new queue** (checklist): `--save 1` on every trained job; the guard record keeps the command;
after the queue, register every run (saved or not) the same day; then copy to the archive.

## 6. How to archive and restore

Archive (on the PC, from WSL):

```bash
id=E005/135m/s1; mkdir -p /mnt/d/planck-archive/$id
cp -a <source dir>/. /mnt/d/planck-archive/$id/
# add "pc:/mnt/d/planck-archive/$id" to the entry's locations, then:
python3 models/registry.py validate && python3 models/registry.py verify --where pc --id "$id"
```

A source is deleted only after `verify` reports OK at the new location and the entry lists it. From the
Mac, `--root pc:/mnt/d/planck-archive=<mounted dir>` maps a PC location to a local directory.

Restore: `registry.py list --id '<glob>'` gives the locations; copy the directory, run `verify` on it,
then load it.
- HF fine-tunes (E002-E006): the directory is a `save_pretrained` folder; load it with transformers
  `from_pretrained(dir)` and the tokenizer from the same folder. `base_model.revision` pins the base.
- Harness runs: `harness/runio.load_checkpoint(path)` returns model and optimizer state, `model_cfg`,
  `config`, `schedule`, `data_state` and RNG state, enough to resume or branch; build the model from
  `model_cfg` at the code in `code`.

## 7. Public release

Weights are published with the writeup that reports them, failures included, under the project license
once it is chosen (README: "to be chosen before the first model release"). Exceptions and conditions:
- **never**: runs on the starter corpus (E2, E3, the S00x screens and their smoke): E2 notes FIXED 7
  and SCREENS.txt make them calibration only, never released, never an init for a recipe run.
- **fine-tunes of third-party models** (E002-E006) are released only under terms the base model's
  license allows; check its model card at the recorded revision. Until then `release` is `undecided`.
- a release carries the registry line and the result with it, and passes `validate` (no private paths).

## 8. Save behavior per experiment (2026-09-26)

| exp | trains | final weights saved by its queue | where | not saved | state | hook |
|---|---|---|---|---|---|---|
| E001 | no (scores public models and MaxGPT-3 read-only) | n/a | n/a | n/a | done | none needed |
| E002 | SmolLM2-135M-I, 400 steps | yes from seed 1 (`--save` default 1; dry and steps-0 runs skipped) | mac (5, 2.71 GB) | s0 (ran before the save code), 2 dry; the 360M arm never ran | done | none: no future runs |
| E003 | 8 public bases, 1M-160M | scored seeds yes (`--save 1`); LR search no (`--save 0`); dry no | mac (18, 1.82 GB) | 26 LR, 8 dry, 12 killed attempts | RUNNING on the Mac: ts33m s1-s3 and the p160m block remain | not edited; register the new dirs when the queue ends (p160m's 3 LR runs will be not_saved) |
| E004 | SmolLM2-135M-I + tiny ladder | 135M seeds yes; LR searches no; tiny-ladder seeds never ran (their queue lines carry `--save 0`) | mac (5, 2.71 GB) + pc dev copy | 21 LR, 7 dry, 1 killed | done | none: no future runs; a rerun of the ladder passes `--save 1` |
| E005 | SmolLM2-135M-I | yes (`--save 1`) | mac (5, 2.71 GB) + pc dev copy | 1 dry | done (AL diagnostic is eval only) | none |
| E006 | SmolLM2-135M-I arms C/P/G x 5 seeds + 2 TF32 twins | yes (`--save 1`), saved and hashed before scoring, `weights_sha256` in each guard record | `pc:~/planck/e006_run/weights` | dry runs (outputs deleted by design) | RUNNING on the PC (not checked: offline) | not edited; afterwards archive and register from the guard records |
| E2, E3 | harness, from scratch, 5M and 20M | yes: `final_*.pt` and `stable_*.pt` never pruned; rolling `keep_last 2` (pre-registered) | `pc:~/planck/runs/E2`, `.../E3` | rolling checkpoints beyond 2 | queue ready; whether it started is unknown from the Mac; smoke ran 2026-09-26 (5 lines) | none: runs need committed code and checked config sha256, and saving already meets section 1 |
| S001-S007 | harness, 5M | as E2 (SCREENS.txt: keep_last 2 plus final; C8 keeps finals until post-lock scoring) | PC | as E2 | pre-registered, no config yet | none: C8's "kept until then" becomes "kept" |
| RC-12 baselines | no (13 public models, inference) | nothing to save | n/a | n/a | PC queue | proposal: record each model's HF snapshot revision in the run records; not implemented (PC-side code, outside experiments/) |

No hook was implemented on 2026-09-26: every experiment either already saves final weights (E002
from seed 1, E003 scored, E004 135M, E005, E006, the harness), is running (E003, E006), is finished
with no future runs (editing its queue would make the committed script disagree with what ran), or is
pre-registered with committed configs whose sha256 the analyzer checks (E2, E3, screens).

Superseded: E002 notes, POST-HOC ADDITION 2 ("deleted when E002 is done"). Those weights are kept.

## 9. Backfill of 2026-09-26

117 lines, all from records on the Mac (no model loaded):
- **33 saved models on the Mac, 9,949,117,385 bytes** (9.83 GB of it `model.safetensors`), every file
  hashed and re-verified with `verify --where mac` (180 files OK). The 10 E004/E005 weights equal the
  `weights_sha256` E005's AL diagnostic recorded when it scored them.
- **5 E2 smoke checkpoints on the PC** (3 finals, 2 branch points), registered from runs.jsonl with
  null hashes: the PC was offline. Fill them there with `verify --where pc --fill`.
- **79 trained but not saved**: 47 LR-search runs (`--save 0`: E003 26, E004 21), 18 dry runs, E002's
  seed 0 (before the save code), and 13 killed attempts from the queue logs (E003 12, E004 1).
