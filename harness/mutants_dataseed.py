"""data_seed mutants (E3 Part 3; train.data_seed_of, train.check_data_seed and their uses in train.main), split from
mutants.py for the file size. Imported at the end of mutants.py, which it appends to (mutation_check.py --group
dataseed). Each is killed by test_data_seed.py; ds_numpy_global_seed is marked equivalent (nothing in the harness draws
from numpy's global RNG: the loader and packing use their own default_rng generators), so it is expected to survive."""
from mutants import mut

DS = ["test_data_seed.py", "test_data_seed_more.py"]
mut("ds_loader_gets_seed", "dataseed", "train.py", "loader = build_loader(cfg[\"data\"], mcfg.seq_len, micro, base, data_seed)",
    "loader = build_loader(cfg[\"data\"], mcfg.seq_len, micro, base, seed)",
    "the loader keeps the run seed: data_seed changes nothing", tests=DS)
mut("ds_seeds_the_init", "dataseed", "train.py", "    torch.manual_seed(seed)\n", "    torch.manual_seed(data_seed)\n",
    "data_seed also seeds torch's RNG: a data family's runs start from different weights", tests=DS)
mut("ds_absent_is_zero", "dataseed", "train.py", '    if "data_seed" not in cfg:\n        return seed\n',
    '    if "data_seed" not in cfg:\n        return 0\n',
    "a config without data_seed reads seed 0's stream: every existing config changes", tests=DS)
mut("ds_meta_always", "dataseed", "train.py", 'if "data_seed" in cfg:                 # E3 Part 3: in the checkpoints only',
    "if True:                 # E3 Part 3: in the checkpoints only",
    "flag-off checkpoints gain a data_seed key (no longer the pre-patch checkpoint)", tests=DS)
mut("ds_meta_never", "dataseed", "train.py", '        meta["data_seed"] = data_seed\n', "        pass\n",
    "the checkpoints never record data_seed", tests=DS)
mut("ds_start_always", "dataseed", "train.py", 'if "data_seed" in cfg:                  # E3 Part 3: in the start record',
    "if True:                  # E3 Part 3: in the start record",
    "flag-off start records gain a data_seed key", tests=DS)
mut("ds_start_never", "dataseed", "train.py", '        start["data_seed"] = data_seed\n', "        pass\n",
    "the start record never says which data seed a run used", tests=DS)
mut("ds_resume_unchecked", "dataseed", "train.py", "        check_data_seed(ck, data_seed, cfg, last)\n", "",
    "a resume under another data seed goes on (or dies in the loader's replay assertion)", tests=DS)
mut("ds_branch_unchecked", "dataseed", "train.py", '        check_data_seed(init_ck, data_seed, cfg, "the init_from checkpoint")\n',
    "", "a branch under another data seed than its trunk's goes on", tests=DS)
mut("ds_check_reads_seed", "dataseed", "train.py", 'was = ck.get("data_seed", ck.get("seed"))', 'was = ck.get("seed")',
    "the check compares the checkpoint's seed, refusing a valid resume when data_seed != seed", tests=DS)
mut("ds_check_always", "dataseed", "train.py",
    '    if "data_seed" not in cfg and "data_seed" not in ck:\n        return\n', "",
    "the check runs on flag-off runs too: a new refusal on configs without data_seed", tests=DS)
mut("ds_bool_and_negative_accepted", "dataseed", "train.py",
    "if isinstance(ds, bool) or not isinstance(ds, int) or ds < 0:", "if not isinstance(ds, int):",
    "data_seed true or -1 accepted", tests=DS)
mut("ds_any_value_accepted", "dataseed", "train.py",
    "if isinstance(ds, bool) or not isinstance(ds, int) or ds < 0:", "if False:",
    "data_seed 3.0, '3' or null reaches the loader", tests=DS)
mut("ds_numpy_global_seed", "dataseed", "train.py", "    np.random.seed(seed)\n", "    np.random.seed(data_seed)\n",
    "numpy's global RNG seeded by data_seed: nothing draws from it", equivalent=True, tests=DS)
# The 2026-10-11 verify's survivors (F1 to F3), killed by the assertions added after it (test_data_seed.py's split test,
# test_data_seed_more.py).
mut("ds_start_records_seed", "dataseed", "train.py", '        start["data_seed"] = data_seed\n',
    '        start["data_seed"] = seed\n', "the start record names the run seed, not the loader's", tests=DS)
mut("ds_fallback_accepts", "dataseed", "train.py", 'was = ck.get("data_seed", ck.get("seed"))',
    'was = ck.get("data_seed", data_seed)',
    "a checkpoint without data_seed is taken to match any data_seed the config adds", tests=DS)
mut("ds_resume_check_cfgless", "dataseed", "train.py", "        check_data_seed(ck, data_seed, cfg, last)\n",
    "        check_data_seed(ck, data_seed, {}, last)\n",
    "the resume check reads only the checkpoint: a config adding data_seed to a flag-off run is not checked", tests=DS)
mut("ds_branch_check_cfgless", "dataseed", "train.py",
    '        check_data_seed(init_ck, data_seed, cfg, "the init_from checkpoint")\n',
    '        check_data_seed(init_ck, data_seed, {}, "the init_from checkpoint")\n',
    "the branch check reads only the trunk: a branch adding data_seed to a flag-off trunk is not checked", tests=DS)
mut("ds_meta_truthy", "dataseed", "train.py", 'if "data_seed" in cfg:                 # E3 Part 3: in the checkpoints only',
    'if cfg.get("data_seed"):                 # E3 Part 3: in the checkpoints only',
    "data_seed 0 is left out of the checkpoints, and the run's own resume is then refused", tests=DS)
mut("ds_start_truthy", "dataseed", "train.py", 'if "data_seed" in cfg:                  # E3 Part 3: in the start record',
    'if cfg.get("data_seed"):                  # E3 Part 3: in the start record',
    "data_seed 0 is left out of the start record", tests=DS)
