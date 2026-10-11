"""Mutants for models/registry.jsonl's stage2_replay lines (registered 2026-10-11, models/MODELS.md section 12). Read by
mutation_screens.py (name, file, old, new, test file or file::-k expr); each must make the named tests fail.
"""
RG, RT = "models/registry.jsonl", "test_registry_replay.py"
RES = ('"result": "not read: registered 2026-10-11 before the stage 2 verdict step reads it (SCREENS.txt AMENDMENT '
       'BASE-REPLAY, AMENDMENT CAP-31); the queue scored it (SCORED, bpb.jsonl on the PC); no value of it was read for '
       'this entry"')
MUTANTS_REGREPLAY = [
    ("rr_config_sha_typo", RG, '"config_sha256": "b0f20a5a62417f37', '"config_sha256": "b0f20a5a62417f38', RT),
    ("rr_role_swapped", RG, '"id": "SCREENS/BASE/s101_hookoff/final", "experiment": "SCREENS", "arm": "BASE", "role": '
     '"base_hookoff"', '"id": "SCREENS/BASE/s101_hookoff/final", "experiment": "SCREENS", "arm": "BASE", "role": '
     '"base_replay"', RT),
    ("rr_result_read", RG, RES + ', "status": "success", "run": {"state": "completed", "started": "2026-10-10T22:28:12',
     RES.replace('"not read: ', '"read: ') + ', "status": "success", "run": {"state": "completed", "started": '
     '"2026-10-10T22:28:12', RT),
    ("rr_ended_after_the_hold", RG, '"ended": "2026-10-10T23:49:31-04:00"', '"ended": "2026-10-10T23:52:31-04:00"', RT),
    ("rr_wrong_commit", RG, '"config_source": "pc:~/planck/runs/runs.jsonl start event (experiments/screens/configs/'
     'base_s101_rep2.yaml at 2aa3787)"', '"config_source": "pc:~/planck/runs/runs.jsonl start event (experiments/'
     'screens/configs/base_s101_rep2.yaml at 27d8cb7)"', RT),
]
