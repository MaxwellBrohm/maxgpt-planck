"""Recorded values for SCREENS.txt S006 EXTENSION RESULT / STAGE 2 SEEDS LAUNCH CHECK (2026-10-06), read by
test_screens_stage2_seeds.py: S006's g 4 extension run's final bpb, its train seconds and tok/s, the queue times and the
PC files' sha256. Data only.
"""
# final checkpoint, split all, CHAT PROSE cccc gutenberg wikimedia (s2select_record.SETS), from
# ~/planck/runs/SCREENS/s006_mtp_g4_s1/bpb.jsonl (copied 2026-10-06 19:52, sha256 equal on both machines)
EXT_VALS = (1.1791872803448515, 1.417255281974049, 1.3478469789506382, 1.5361731343139235, 1.3660591680476344)
TRAIN = 1166                     # "train" 19:30:50 to "done" 19:50:16, seconds (PC queue log)
TOKS = 215044.5                  # median log.jsonl tok/s from the third record
LOCK, WAIT_S = "19:30:49", 0      # "waiting for gpu.lock" and "gpu.lock held" times (equal: no wait), wait seconds
QSTART, QEND, ENTRY_TIME = "19:30:47", "19:52:00", "20:07"   # queue start, mark and plan finished, entry written
PCSHA = ("30ffd248ecbf43c5", "4cb2eac7a104fbea", "7e10048e62c8478e")   # PC queue log, status.jsonl, stage2_s006x.txt
