# RC-12: Level R and the last lock decisions (Oct 11-14)

Written 2026-10-04 from two CPU-only analyses: an options pass, then a critic who recomputed every number with separate code (session scratchpad, rdesign/ and rcritic/). I re-derived the S7 column from runs/dev_panel_11b and the shortcut sums from logs/e2e_fakes.txt; both match. Nothing here is applied. No model, GPU or sealed/ was used.

## The problem
- Level R says Planck is non-inferior to Qwen2.5-0.5B (CI lower bound at least -3 points). Qwen2.5 scores 2.30 of 100 on the seven state families, less than the margin, so the test cannot tell it apart from a floor model.
- On dev, Veyra2-Blueberry-5M meets the bar, mostly by not looping. Under the widened s12 clause you took, no size from 5M up is left where Planck can claim Level R.
- Any bar tied to a public model is also met by a "model" that learned only RC-12's answer format plus a fixed-rule shortcut, because every public model scores below those fakes on the state families.

Dev numbers (template, mean of 3 seeds). S7 is the state score, 0 to 100: RECALL, CORR, BIND, TWOHOP, OWN (gated), TOPIC, ROLE. The shortcut row is my sum of the best fake per family in logs/e2e_fakes.txt (dev items) plus a never-repeat rule for LOOP.

| model | params | R now (9 families) | R from LOOP | S7 | PERSIST |
|---|---|---|---|---|---|
| Veyra2-Blueberry-5M | 5.0M | 5.01 | 4.46 | 0.25 | 0.03 |
| Loom-Spark-3.2 | 22.8M | 9.55 | 8.66 | 0.25 | 0.06 |
| Qwen2.5-0.5B (comparator) | 494M | 5.68 | 2.64 | 2.30 | 0.11 |
| LFM2.5-350M | 355M | 17.65 | 2.02 | 12.10 | 0.56 |
| LFM2-2.6B (best public) | 2.6B | 24.22 | 1.08 | 18.15 | 0.81 |
| shortcut mix (fixed rules, no model) | none | 42.3 | 11.1 | 36.6 (39.9 if OWN slipped its gate) | 0.25 with no skill |

## 1. Level R redesign (the big one)
- **A. Keep it as written.** No work. No Planck size can claim Level R on dev, so it becomes a report and PLAN's fallback claim (1) is gone. Even scored on S7 alone against Qwen2.5, 13 public models meet it, Vertex-15M among them.
- **B. A comparative fix:** non-inferior to a stronger model (LFM2-2.6B, LFM2.5-350M or Qwen3-0.6B), beat Qwen2.5 by +m, or beat the best public model at or below k x N. The shortcut mix passes every version with power about 1.00. Non-inferiority to a strong model has only 0.16 to 0.42 power at n 600 (no allowed n reaches 80% for LFM2-2.6B or LFM2.5-350M), and k x N depends on which models happen to be on the panel. Not honest.
- **C. F: S7 of at least 40, point estimate.** 40 is the existing G2 cheater bar (at most 0.40 per family), committed in SPEC v0.1 on 2026-09-25, before any baseline was scored. No public model is near it (best 18.15), so every size stays open. Planck needs a true S7 of about 41 to 43 for 80% power, depending on how much its training seeds differ. Weakness: the shortcut mix sits at 36.6 and passes 0.4% to 6% of the time (39.9 and 46% if OWN credit got past its gate).
- **D. F′ (recommended):** the 95% CI lower bound of S7 is at least 40; PERSIST is at least 0.40 (point); T0 is at least 0.90 with at least 3 training seeds; the OOD-H wording rule is unchanged (still paired against Qwen2.5, worse than -5 means qualified wording). The shortcut mix passes at most 1.7% of the time. Planck needs a true S7 of about 45 to 46, an average state family of 0.45 (LFM2-2.6B's is 0.18). No-skill PERSIST fakes reach at most 0.25; a true PERSIST of 0.50 passes 91% to 99% of the time.
- **E. Current test plus the four Level A keys at 0.60 or more.** Honest (no fake reaches 0.60), but it is just Level A without the loop criterion and the human test.

**Recommend D.** It keeps PLAN's own "lower bound of a 95% CI" form and changes only the reference, from "Qwen2.5 minus 3" to a shortcut ceiling fixed before dev. It stops Blueberry and Loom from passing, and it makes the claim harder, not easier.
- **s12 clause:** a public model passes if its S7 point estimate is at least 40 and its PERSIST at least 0.40. On dev, none does at any size.
- **Reported beside, never claimed:** R and PLAN's original non-inferiority CI against Qwen2.5; S7 per family; paired S7 against LFM2-2.6B; the sealed shortcut ceiling; loop rate; S7 of the generic-recipe 30M control.
- **Costs:** Level R gets much harder. PLAN's 40-60% odds no longer hold (both analysts' judgment), and fallback (1) gets harder with them. The Level R wording in PLAN s1 and prereg s1 changes, and the change is disclosed as a pre-lock deviation made after seeing dev, with the table above as the evidence. Code and text change before Oct 11, CPU only: score.py, score_stats.level_r and their mutants, s9's sizing text, s12's clause.
- **Draft wording:** "On sealed RC-12's seven state families, the lower bound of the 95% CI on Planck-{N}M's state score is {lo} of 100, above the pre-registered bar of 40 set by the shortcut-rule ceiling; the best public panel model scored {y} ({k} training seeds, {c} conversations). It is the smallest model on our curve that does this."
- **Sealed size under D:** stays 600, because n barely matters for this bar (true S7 needed: 45.7, 45.1 and 44.6 at n 600, 700 and 900). If you pick A, the old trio question returns: the dev-R trio gives 600, the size trio 700. I would take 700.

## 2. A claim for Planck querying a lookup store itself?
This means Planck asks a store for facts instead of reading tables pasted into user turns.
- **Key-level store** (ask for a table and key, get the value or "not found"): gameable. "Not found" is the X gold and the value is the P gold, so a model that only parses, queries and relays scores about 1.0 without reading or abstaining (the critic checked a dev item).
- **Table-level store** (returns the named table): keeps both probes on the model, but the store must answer in Planck's trained tool format, which breaks OD5 condition 2 and s13 S2. Every public tool template puts tools in a system block, which s4 forbids, so public rows would be diagnostics only. An addendum written after Planck's format is known is post hoc.
- **Options:** (a) no claim, only a post-lock diagnostic labelled "Planck-{N}M + lookup store"; (b) lock a table-level condition in full at the push (about a week of CPU work, plus the OD5 conflict); (c) write a deferred shape into s10b and lock it later.
- **Recommend (a).** The bare LOOKUP claim already tests reading supplied tables, and every public model scores 0.000 to 0.014 on it.

## 3. The sampled rerun check (s4, R1)
- **Evidence (GPU rerun, 2026-10-04):** greedy R2 passes on both engine families, every first divergence within 0.125 nats. Sampled R1 fails on both, with margins of 0.25 to 2.625 nats. Both samplers draw from seeded noise, so a tiny batch drift can flip a sampled pick even between tokens far apart in log p. Seed 101 against seed 1 was also 0 of 5 identical, the same as the right seed, so R1's identical share cannot catch a wrong seed.
- **Options:** (a) R2 gates, and R1's identical share, first divergence, shared prefix and margins are reported (static check C3 already verifies seeds are carried); (b) a noise-aware margin bar (new, version-fragile code and GPU time); (c) keep R1 as written, which fails every engine forever.
- **Recommend (a).** Optional: one wrong-seed replay per engine, to see whether shared-prefix length separates right seeds from wrong ones. Not measured; it costs two short GPU holds.

## 4. Should the LOOKUP claim also need T0 of at least 0.90?
- **Evidence:** Level R and Level A both require it; s10b does not. T0 is not a shortcut guard: 15 gated value rules score 1.00 on it (POS2, PENULT, ANTEPENULT and PENULT_MARKER score 0.00). The chance of passing T0 ≥ 0.90 is 0.80 at a true T0 of 0.91 to 0.93 and at least 0.97 at 0.96; the birthday-item weakness costs at most 0.07.
- **Options:** (a) add it (one line of score.py, a mutant, s10b text); (b) leave LOOKUP independent of T0.
- **Recommend (a).** All headlines then share one sanity gate, so a model that fails basic two-fact recall cannot carry one. The s12 clause does not change.

## 5. Other open lock items
- **Panel coverage at Planck's sizes.** Not on the panel: MiniMind2-Small 25.8M (no licence); spark-13m, flame-27m and Pebble-25M (Apache, remote code); Supra2-Medium-Instruct; newer Loom releases (Loom-Spark-3 12.2M, Weave-3 31.5M, Tapestry-3 69.2M). Options: (a) add the licensed ones now (item 14 took about two days with engine patches); (b) keep every claim worded "on our panel", as s1 and s10b already are, and list these in s12 as known and unscored. **Recommend (b).** Every sub-30M model scored so far has an S7 of at most 2.09, so one reaching 40 is unlikely (my judgment).
- **OOD-H Part 1 review (item 13)** is still yours. Its 150 threads are fixed only after you review them, which has to happen before the lock.

## Mechanical fixes (no decision; apply after the picks above)
- **Level A threshold (prereg s1 and PLAN s1):** "at least 0.80" becomes 0.60 for the four content bars, and 0.80 becomes 0.60 in the "Rules for both" list. s10, score.py BARS and the re-anchor result are already at 0.60.
- **README (item 15 follow-up):** Status is stale ("Next: dev baselines on RC-12", "No model is scored on it yet"); the intro's "13 released chat models ... from 90M to 2.6B" predates the 5M to 31M extras; neither README nor PLAN.md mentions the LOOKUP claim.
- **harness/test_rc12_eval.py:232** still expects "LOOKUP" among the summary families. Remove it and run the test (it needs torch and has not run since item 1). harness/rc12_eval.py's docstring still says "all 10 composite families".
- **Item 14 runs still finishing on the PC:** BananaMind-2-Mini, cRia-75M and Swen-28M's plain render. When ~/planck/logs/rc12_s14b_chain.DONE exists: verify, pull with an md5 manifest, rerun rc12/dev_extras14.py, then add the rows to s12 or mark them "not run" with the reason.
- **After decision 1:** score.py, score_stats.level_r and their mutants; prereg s9 (composite and sizing text), the s12 clause and the s1 Level R wording; PLAN's Level R wording and odds line; then the full check order.
