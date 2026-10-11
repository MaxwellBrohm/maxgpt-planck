"""AMENDMENT BASE-REPLAY's record of what it read on the PC (2026-10-10 16:53 EDT, read only): the SCREENS queue log's
lifecycle lines for the 15 runs since the last regenerated measured_hours.tsv ("gpu.lock held" before each run's train
line, and its status.jsonl line; these files hold no score), their sha256 (first 16 hex), the GPU driver and torch.
"""
READ = "16:53"
PCSHA = ("93d1847aec949d30", "c8ced214b15e48d4")       # queue_screens.txt, status.jsonl
DRIVER, TORCH = "596.49", "2.13.0+cu130"
LOCK = [   # run, "gpu.lock held", status.jsonl time
    ("s005_base_s101", "2026-10-06 20:38:03", "2026-10-06 21:01:37"),
    ("s005_forget_g1_s101", "2026-10-06 21:01:38", "2026-10-06 21:53:14"),
    ("s004_canonac_g1_s101", "2026-10-06 21:53:15", "2026-10-06 22:17:43"),
    ("s007_smear_g1_s101", "2026-10-06 22:17:44", "2026-10-06 22:36:35"),
    ("s005_base_s102", "2026-10-06 22:36:36", "2026-10-06 23:00:10"),
    ("s005_forget_g1_s102", "2026-10-06 23:00:11", "2026-10-06 23:51:47"),
    ("s004_canonac_g1_s102", "2026-10-06 23:51:48", "2026-10-07 00:16:17"),
    ("s007_smear_g1_s102", "2026-10-07 00:16:18", "2026-10-07 00:35:10"),
    ("s006_mtp_g2_s101", "2026-10-07 00:39:01", "2026-10-07 01:00:13"),
    ("s006_mtp_g1_s101", "2026-10-07 01:00:14", "2026-10-07 01:21:27"),
    ("s006_mtp_g2_s102", "2026-10-07 01:21:28", "2026-10-07 01:42:40"),
    ("s006_mtp_g1_s102", "2026-10-07 01:42:41", "2026-10-07 02:03:54"),
    ("base_s103", "2026-10-07 02:08:58", "2026-10-07 02:27:17"),
    ("base_s104", "2026-10-07 02:27:18", "2026-10-07 02:45:37"),
    ("base_s105", "2026-10-07 02:45:39", "2026-10-07 03:03:58"),
]
LAST_START = ("stage2_base_diag", "27d8cb7505248ff4391a1dee9a51bad617938195", "2026-10-07 03:03:58")  # plan, commit, end
