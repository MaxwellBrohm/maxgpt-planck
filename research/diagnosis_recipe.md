# Diagnosis: what limits a ~150M chat model (recipe and training-signal lens)

Date: 2026-09-23 (resumed after the crash). One of three independent diagnoses. Lens: how much of the "400-500M wall" is an artifact of how labs build, train and measure small models, which of those artifacts one person can remove, and which limits follow from the parameter count itself.

Evidence rules. Lane numbers (`lanes/*.md`) are used in their fact-checked form (`lanes/*.verify.md`); refuted claims are not used. Follow-up reports (`followup/*.md`) are used only in the form their own `.verify.md` files left them (the interrupted draft wrongly called them unchecked). Every number has a source or says "unsourced". Scale (parameters, tokens) is given for each piece of evidence, and evidence that exists only at 1B or above is flagged **[>=1B only]**. No model was loaded or run for this resumed step; everything below is reading, web research and arithmetic on files already on disk.

---

## 0. What survived the crash (read this first)

| work item | status | how it is used here |
|---|---|---|
| **Capacity-lens likelihood probe** (`capacity_probe/`, plain `User:/Assistant:` transcripts, every in-context foil is a same-type value that appears in the context) | **Complete** for 8 models on the main battery (1,116 items each: SmolLM2-135M base and Instruct, LFM2-350M, LFM2.5-350M, SmolLM2-360M-Instruct, Qwen2.5-0.5B-Instruct, Qwen3-0.6B, LFM2-2.6B); long-tail knowledge twin (`khard`, 80 items) complete for 9 models; state-update controls (`uprobe`, 576 items) complete for SmolLM2-135M-I and 360M-I; copy split complete for SmolLM2-135M-I. | Re-analyzed by me today from the saved JSONL with a copy of their `analyze.py` that writes to the scratchpad (pure JSON arithmetic, no model). Used as firsthand evidence. |
| Capacity probe, Gemma-3-270M-it main run | **Incomplete**: 420 of 1,116 items (all of distance 0, about 6 of 32 scenarios at distance 4, no distance 10, no knowledge items). The queue log says "exit=0" but the model log stops at item 400. | Distance-0 cells used; distance-4 cells shown with their tiny n. |
| Capacity probe, Falcon-H1-Tiny-90M | **Missing**: main run skipped (empty log); `khard` crashed with `RuntimeError: invalid low watermark ratio 1.4` (an MPS memory-cap environment setting, not a model fault). | Not available. This is the gap that matters most for the recipe lens (section 6, step 2). |
| Capacity probe, fine-tuning test (`ft_test.py`) | **Incomplete**: log stops after loading weights; no `ft__*` or `ftu__*` output exists. | No result. It remains the cheapest decisive test (limit 2). |
| **P1**, this lens's base-model token-curve probe (16 checkpoints: Pythia-160M/410M across training, SmolLM2-135M intermediate checkpoints, 0.5-1.2B bases) | **Raw outputs lost**: the session scratchpad (`.../scratchpad/recipe/`) was wiped by the reboot; only a summary table in the interrupted draft survives. | Not load-bearing. Its two headline claims are listed in section 2.4 as leads to re-run. |
| **P2**, audit of Max's post-training pipeline | **Re-done today**: code re-read (read-only); packing simulation re-run on 1,100 UltraChat conversations and saved with its script at `research/recipe_work/pack_sim.py`. Reproduces the draft's numbers. | Firsthand evidence. |
| **P3**, this lens's fine-tune pilot | **Never finished**; no results on disk. | No result. |

The crash itself is a finding for the plan: the earlier attempt ran three model processes at once on the 24 GB Mac. Every Mac experiment in section 6 must run alone, one model per process.

---

## 1. Bottom line

1. **There is no wall at 400-500M; there is a slope that recipes have moved models along by 2-4x in parameters.** Under a fixed recipe, multi-turn scores fall with size (LFM2 Multi-IF 32.85 / 40.92 / 45.28 at 350M / 700M / 1.2B, [arXiv 2511.23404](https://arxiv.org/html/2511.23404v1); MobileLLM MT-Bench 2.33 at 125M vs 3.28 at 350M after identical chat tuning, [arXiv 2402.14905](https://arxiv.org/abs/2402.14905)). Across recipes the order inverts: LFM2.5-230M (163M non-embedding) scores Multi-IF 37.70, above LFM2-350M ([card](https://huggingface.co/LiquidAI/LFM2.5-230M), vendor harness), and on the firsthand 8-model battery the 350M class is level with Qwen2.5-0.5B (`lanes/probe.verify.md`).
2. **The most binding limit that one person can remove is the shape of the chat training signal.** Public small-model SFT sets are shallow and rarely refer back, preference sets are single-turn, deflection personas are trained in, and packing cuts conversations apart. Max's own pipeline has all four (section 2.2): his DPO data is 100% single-turn, and his SFT packer gives **42.5% of supervised assistant tokens a history that is cut off and 57.4% an unrelated conversation in view; about 0.1% see a clean, complete history** (seq_len 2048).
3. **Firsthand, new since the draft: small models share one failure pattern, "the first-mentioned value wins".** On the capacity battery, once 4 or more unrelated turns separate a correction from the question, **every checkpoint from 106M to 440M of transformer body prefers the stale value** (0 to 7 of 32 correct for a single correction), including LFM2.5-350M (28T tokens plus RL) and Qwen3-0.6B (36T plus distillation). LFM2-2.6B gets 28 and 32 of 32. The same primacy pattern drives owner-binding errors at 135M (SmolLM2-135M base: 0.98 when the right name was mentioned first, 0.31 when second). Binding is recipe-sensitive at fixed size (post-training lifts the 135M's second-mention case to 0.65; SmolLM2-360M solves it; LFM2-350M does not). Updates at distance are not solved by any sub-0.6B recipe, and no one has published a small model trained on data that forces them. This is the likeliest place where a real parameter limit could sit near 150M, and the test that would settle it (`ft_test.py`) is written but never ran.
4. **Tokens are the soft limit that is expensive to remove.** Every working sub-600M chat model was trained at about 2,900 to 83,000 tokens per parameter (census and training lanes, corrected); Max's 150M on 100B tokens is about 670. Tokens do not fix the primacy failure (the 28-36T models still fail it), but they carry general competence, and 300B token-passes cost about 51 days of the four usable Titans after Ultra frees them (compute lane).
5. **Knowledge is the one limit that is clearly hard, and it should be scoped out rather than fought.** About 2 bits per parameter at best ([arXiv 2404.05405](https://arxiv.org/abs/2404.05405), synthetic, up to 0.5B), linear in size. Firsthand: on 40 long-tail facts, SmolLM2-135M picks the right answer closed-book 0.78 of the time and 1.00 when the fact was stated two turns earlier. Reading from context works at 135M; storing is what is scarce.
6. **The strongest counter-evidence is at Max's exact size.** ufakzeka-1 (151M non-embedding, 13.5B tokens, 154,506 SFT conversations with about 12% templated corrections, abstention, identity and memory families) reports that identity tracking over a long story "did not move across any data change we tried", preference optimization in three variants lowered conversation quality, and seed variance equaled the spread across every recipe ([arXiv 2609.25081](https://arxiv.org/abs/2609.25081), read in full text today). It was trained at 74-89 tokens per parameter, so it cannot separate "151M cannot" from "151M at 13.5B tokens cannot". "The wall is only data" is not established.
7. **Measurement is a limit in its own right.** Harness choice moves a tiny model's IFEval by up to 24 points, one seed moves a 151M recipe as much as any recipe change, and plain-transcript probing of heavily post-trained models is off-template (the same SmolLM2-135M-Instruct weights recall 0.15 through the chat template and 0.70 as a plain transcript, `lanes/probe.verify.md`). Until evals are fixed-set, multi-seed and held-out, most recipe effects at 150M are invisible or illusory.

---

## Premise verdict

Partly real, mostly misplaced. There is no cliff at 400-500M: LFM2.5-230M (163M non-embedding, 19T tokens) scores Multi-IF 37.70 against 41.68 for Qwen3.5-0.8B (vendor harness), Falcon-H1-Tiny-90M scores 4.33 on TII's 2-turn MT-Bench against 3.80 for SmolLM2-360M in the same harness, and on the firsthand battery the 350M class matches Qwen2.5-0.5B. What is real is a slope under any fixed recipe, and three artifacts that made it look like a wall at 0.5B: the big labs' smallest models are afterthoughts with inherited vocabularies (Gemma 3 270M is 62.6% embedding table) and generic single-turn post-training; "chat quality" benchmarks are knowledge-heavy and harness-dependent; and in 2024, when the premise formed, Qwen2-0.5B and Danube3-500M really were the smallest usable chat models. Two things keep the premise from being a myth. First, open-domain, knowledge-bearing, many-turn chat has not been shown below about 300-400M by anyone; past 3 turns the only sub-1B numbers are agent task completion (tau2-Bench, LFM2.5-230M 5.26 Telecom / 13.68 Retail, vendor) and 2020-era chit-chat ratings (BlenderBot 90M, [arXiv 2004.13637](https://arxiv.org/abs/2004.13637)). Second, one coherence skill fails in every sub-0.6B model probed, whatever its budget: holding a corrected value across several unrelated turns (0-7 of 32 correct, section 2.1), while the one larger model probed (LFM2-2.6B) passes. For Max the best reading is: a 150M model with a generic recipe on 100-300B tokens will sit well below the curve LFM2.5-230M and Falcon-H1-Tiny-90M define; most of that gap is training signal (cheap) and tokens (expensive); knowledge is a hard ceiling to design around; and state updates under interference are the open question that decides whether "real back-and-forth" is fully reachable at 150M.

---

## 2. Firsthand evidence

### 2.1 Capacity-lens likelihood battery, re-analyzed through the recipe lens

Design (from `capacity_probe/items.py`, `uprobe.py`, `khard.py`). Plain `User:/Assistant:` transcripts for every model (no chat template), scored by the summed log-probability of each candidate after a forced answer prefix. All information needed is in context, and the foil is a same-type value that is also in context, so "copy any name from the context" does not pass. Distances d = 0, 4, 10 unrelated user/assistant exchanges; 32 scenarios per cell; about +/-0.17 is the 95% interval around 0.5 at n=32.

Accuracy (fraction of items where gold beats every foil). Body = non-embedding parameters.

| model | body | tokens | owner binding d10 | two-hop d4 / d10 (pair both right d10) | 1 correction, d0 / d4 / d10 (of 32) | 3 corrections d10 (of 32) | "what's my name" d10 | "what's your name" d10 | long-tail closed / open book |
|---|---|---|---|---|---|---|---|---|---|
| SmolLM2-135M base | 106M | 2T | 0.59 | 0.55 / 0.53 (0.09) | 17 / 0 / 1 | 0 | 0.78 | 0.28 | 0.78 / 1.00 |
| SmolLM2-135M-Instruct | 106M | 2T + SFT/DPO | 0.81 | 0.50 / 0.53 (0.09) | 5 / 0 / 0 | 0 | 0.59 | 0.47 | 0.78 / 0.95 |
| Gemma-3-270M-it (partial) | 100M | 6T | n/a | 0.58 (n=12) / n/a | 24 / 0 (n=6) / n/a | n/a | n/a | n/a | 0.95 / 0.97 |
| LFM2-350M | 287M | 10T | 0.67 | 0.72 / 0.58 (0.19) | 18 / 7 / 5 | 0 | 0.88 | 0.19 | 0.85 / 1.00 |
| LFM2.5-350M | 287M | 28T + RL | 0.73 | 0.84 / 0.89 (0.78) | 1 / 0 / 0 | 0 | 0.97 | 0.03 | 0.85 / 1.00 |
| SmolLM2-360M-Instruct | 315M | 4T | 0.97 | 0.72 / 0.58 (0.31) | 23 / 0 / 0 | 0 | 0.91 | 1.00 | 0.88 / 0.97 |
| Qwen2.5-0.5B-Instruct | 358M | 18T | 0.88 | 0.77 / 0.67 (0.41) | 32 / 0 / 0 | 0 | 0.12 | 1.00 | 0.85 / 0.97 |
| Qwen3-0.6B | 440M | 36T | 1.00 | 1.00 / 0.97 (0.94) | 11 / 0 / 0 | 0 | 0.97 | 1.00 | 0.80 / 1.00 |
| LFM2-2.6B | 2.44B | 10-12T | 1.00 | 0.92 / 0.91 (0.81) | 32 / 28 / 32 | 13 | 0.97 | 1.00 | 1.00 / 1.00 |

Source: `research/capacity_probe/out/*.jsonl`, re-summarized today; token counts from the model cards via the census lane. Extra splits computed today from the same files:
- **Owner binding by mention order** (pooled over d; about 48 items per cell): SmolLM2-135M base 0.98 when the asked-about cat was named first vs 0.31 when named second ("my cat" question); SmolLM2-135M-Instruct 1.00 vs 0.65; LFM2-350M 1.00 vs 0.31; LFM2.5-350M 1.00 vs 0.57; SmolLM2-360M-Instruct, Qwen2.5-0.5B and Qwen3-0.6B 0.80-1.00 in every cell.
- **What wins after corrections at distance**: for three corrections at d=10, gold vs the original value is 0.00 for every model below 2.6B, while gold vs the previous correction is 0.38-0.56 (chance). The models are not confused between recent values; they return to the first one.
- **Surface form is not the cause** (`uprobe`, SmolLM2-135M-I and 360M-I only): when every statement and reply uses the exact wording of the answer prefix ("my dentist appointment is on D"), the 360M goes from 1.00 at d=0 to 0.00 at d=4 and d=10; with a neutral answer prefix it is 0.66 at d=0 and 0.00-0.06 at d>=4. On the main battery, the 360M's mean log-prob margin (gold minus original) for one correction is +0.58 at d=0, -3.69 at d=4 and -3.82 at d=10.
- **Copying from history works at 135M** (copy split, SmolLM2-135M-I, 1,002 scored assistant tokens from OASST1 validation threads): mean loss 0.61 nats on tokens whose bigram appeared earlier in the conversation vs 2.61 on novel tokens.

What it says through the recipe lens:
- **Perspective binding is a recipe artifact at fixed size.** At the same ~300M body, "what's your name?" after 10 turns is 1.00 for SmolLM2-360M-Instruct and 0.03 for LFM2.5-350M (which names the user instead); "what's my name?" is 0.12 for Qwen2.5-0.5B (which names the assistant) and 0.78 for the 135M base. Nothing about size predicts these; post-training does.
- **Second-mention binding is shortcut-prone at 100M and post-training reduces it.** The 135M base uses "first-mentioned wins" almost perfectly; its own SFT/DPO moved the hard case from 0.31 to 0.65. At 287M LFM2-350M still shows the shortcut; at 315M SmolLM2-360M does not. Mixed: recipe matters, and the 106M body may make the shortcut harder to leave.
- **Two-hop reference ("what does my sister do?") is at chance at ~100M body in both recipes tested** (SmolLM2-135M, Gemma-3-270M partial), and recipe-dependent at ~300M (LFM2.5-350M 0.84-0.89, pair accuracy 0.69-0.78; SmolLM2-360M 0.58-0.72). The recipe that would test a 100M body properly (Falcon-H1-Tiny-90M) is the missing run.
- **State updates across 4+ turns fail in every sub-0.6B model regardless of budget** (2T to 36T tokens, with or without distillation and RL). Within one recipe family (LFM2, 10-12T tokens), 350M gets 7 and 5 of 32 at d=4/10 and 2.6B gets 28 and 32. This matches the probe lane's generation test (21 of 24 multi-turn corrections fail from 90M to 600M, Qwen3-0.6B the partial exception, `lanes/probe.verify.md`) and the interference literature: stale values intrude and prompting barely helps ([PI-LLM, arXiv 2506.08184](https://arxiv.org/abs/2506.08184), smallest model Qwen3-0.6B); proactive interference shows "active primacy intrusion" and, across 39 models from 1B to 2.5T, model size predicts resistance to retroactive but not proactive interference (PI R^2 = 0.06, p = 0.14; [arXiv 2603.00270](https://arxiv.org/abs/2603.00270), as checked in `followup/floor.verify.md`) **[>=1B only]**. So above 1B the size effect is contested; below 0.6B nothing passes; no one has trained for it at small scale.
- **Knowledge vs reading**: open-book twins are 0.95-1.00 at every size; closed-book long-tail is 0.78 at 135M, 0.80-0.88 at 287-440M, 1.00 at 2.6B (two-way choice with a plausible foil, n=40, summed log-prob, so easy and slightly biased toward short candidates). Gemma-3-270M's 0.95 closed-book with only 100M body and 6T tokens is the one surprise.

Limits of this battery: likelihood read-out, not generation; plain transcripts are off-template for heavily post-trained models (LFM2.5-350M gets 1 of 32 even at d=0, so it is probably understated); one author's English templates; 32 scenarios per cell; Falcon-H1-Tiny missing.

### 2.2 Max's own post-training pipeline carries the generic signal problems (re-verified today)

Read-only from `maxgpt-ultra/posttrain/sft_data.py`, `posttrain/dpo.py`, `configs/ultra.yaml`, `configs/shakedown.yaml`:
- **SFT data**: UltraChat-200k (`build_sft_jsonl`, default n=100,000) plus English OpenAssistant threads (`sft_extra_oasst: true`). On 1,100 UltraChat rows fetched from 11 offsets of the HF datasets-server today: 6.31 messages per conversation on average (about 3 user turns), 9% with 10 or more messages, assistant replies 225 words on average (median 206). UltraChat's simulated users rarely refer back: 1.45 context-dependent queries per session vs 4.62 for real ShareGPT users ([Parrot, arXiv 2310.07301](https://arxiv.org/abs/2310.07301), dataset statistics, confirmed in `lanes/data.verify.md`).
- **DPO data**: UltraFeedback-binarized, n=60,000, each written as `{"prompt": [{"role": "user", "content": prompt}]}` (`dpo.py`, `build_pref_jsonl`). The preference stage carries zero multi-turn signal.
- **Packing**: `SFTDataset` concatenates all conversations with an EOS between them and slices fixed `seq_len` windows at fixed offsets, with no boundary alignment and no cross-conversation attention mask (`next_batch`). Simulation (`research/recipe_work/pack_sim.py`; token counts approximated as bytes / 4.36, the SmolLM2 49k BPE rate on OASST1 from `lanes/arch.verify.md`; mean 1,323 tokens per conversation):

| seq_len | supervised tokens whose conversation start is cut off | supervised tokens with an unrelated conversation visible before them | supervised tokens with a clean, complete history |
|---|---|---|---|
| 1,024 (shakedown config) | 69.1% | 30.8% | 0.1% |
| 2,048 (Ultra config) | 42.5% | 57.4% | 0.1% |
| 4,096 | 21.5% | 78.4% | 0.1% |

The "clean" share is near zero by construction: a conversation has a clean history only if it happens to start exactly at a window boundary. So every assistant token is trained either without its conversation's opening or with another conversation in view. The first teaches "earlier turns may be missing"; the second teaches "much of the context is irrelevant to you". Both push against using the history, which is the skill Max wants. Effect size unmeasured; the fix (pack whole conversations, mask across them) costs nothing.

### 2.3 ufakzeka-1, read at the source

[arXiv 2609.25081](https://arxiv.org/abs/2609.25081) (2026-09-18; full text read today): 151M non-embedding (182M with a 40,960 vocab), 13.5B tokens in three stages, then SFT on 154,506 conversations (3 epochs, 15% pretraining replay, prompt tokens weighted 0.2). By assistant words: generated stories and multi-turn dialogues 22%, public Turkish instruction sets 27%, Wikipedia-rendered facts 13%, long stitched sessions 8%, templated families for arithmetic, corrections, percentages, safety, abstention, identity and memory about 12%.
- "Data rounds fixed absences": whole missing classes moved to full marks and stayed (agent refusal on unseen phrasings 34 to 64 of 64).
- Two behaviors "did not move with data on any checkpoint of the last eight rounds": identity tracking over a long story (2-36% failures "with no relation to the data change") and arithmetic re-computation with a wrong number in context. The authors treat these as limits of 151M and plan a larger model as the test.
- Preference optimization in three variants lowered conversation quality; the release is SFT-only. (Contrast: TII's one DPO epoch raised 2-turn MT-Bench from about 3.1 to about 4.4 in both arms at 90M.)
- Seed variance equaled recipe spread: three seeds of one recipe scored 75.2 / 80.2 / 81.2 on helpfulness against 75-82 across fourteen checkpoints; a gate repaired with its own questions read 64/64 while held-out paraphrases gave 34/64.

Through the recipe lens ufakzeka-1 is a warning and a confound: its targeted data went only into SFT, on a base trained at 74-89 tokens per parameter.

### 2.4 Leads from the lost P1 probe (unverified; re-run before relying on them)

The interrupted draft reported two findings from 16 base-model checkpoints whose raw outputs no longer exist: (a) second-entity binding ("what's my friend's name?") at ~150M kept improving with tokens (SmolLM2-135M intermediate checkpoints roughly 0.04 at 252B, 0.23 at 1T, 0.56 at 2T; Pythia-410M no better than Pythia-160M at matched tokens); (b) "latest value wins" after 4-8 turns failed in every base model up to Gemma-3-1B, while LFM2.5-1.2B-Base (28T tokens) passed. Finding (b) agrees with section 2.1 on the models both cover. Finding (a) is the only token-curve evidence at Max's size and is worth re-running (section 6, step 3).

---

## 3. Ranked limits

Ranked by how much each one stands between a Max-built ~150M model and the goal (remember what was said, handle follow-ups and references, keep instructions, stay on topic). "Hard" follows from the parameter count; "soft" is an artifact of recipe, data or evaluation.

### Rank 1. The chat training signal is single-turn, shallow and mis-shaped (soft; cheap to remove)

**Mechanism.** A small model learns only the conversational behaviors its data asks for, with less spare capacity than a large model to generalize from single-turn examples to multi-turn use. The standard small-model pipeline supplies SFT sets of 2-3 exchanges with few back-references, long assistant replies written by large teachers for large students, identity and privacy disclaimers that become a reflex ("I don't have access to personal information"), single-turn preference data, and packing that cuts conversations apart. Chat format is learned in a short final stage after capacity has been spent on web text.

**Evidence.**
- Same weights, different framing: SmolLM2-135M-Instruct recalls a user fact 0.15 through its chat template and 0.70 as a plain transcript; the base gets 0.60 (hand-strict regrade: about 0.05-0.10 vs 0.45 vs 0.55; `lanes/probe.verify.md`, 20 trials per cell, 4 facts). At 135M the post-training policy is a first-order problem.
- Deflection tracks post-training, not size: on multi-turn user-fact turns Qwen2.5-0.5B 67%, LFM2.5-350M 40%, Gemma-3-270M 40%, SmolLM2-360M 27%, SmolLM2-135M 20%, Qwen3-0.6B 20%, LFM2-350M 7%, Falcon-H1-Tiny-90M 0% (`lanes/probe.verify.md` recount).
- About half of failed multi-turn recall checks have the fact retrievable by forced-prefix decoding (20 of 42 greedy-gold; 38 of 42 gold beats an out-of-context foil; `lanes/probe.verify.md`).
- Section 2.1: perspective errors at fixed size swing from 0.03 to 1.00 across recipes; SmolLM2-135M's own SFT/DPO lifted second-mention binding from 0.31 to 0.65.
- SmolLM-Instruct at 135M/360M/1.7B could not answer "Hi" or "Who are you" until 2.2k simple dialogues were added ([everyday-conversations](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k)).
- Falcon-H1-Tiny-90M, built with 25% SFT data mixed into all 800B pretraining tokens, never deflects on the probe battery and has the best fixed-history recall per parameter. TII's controlled comparison shows the anti-curriculum gain is on IFEval (66.08 vs 53.47 after DPO), while 2-turn MT-Bench (4.33 vs 4.40) and AlpacaEval slightly favor the classic pipeline; DPO drove MT-Bench from about 3.1 to about 4.3-4.4 in both arms ([TII blog](https://tiiuae-tiny-h1-blogpost.hf.space/), 90M, vendor).
- Max's pipeline (section 2.2): 0% multi-turn preference data; about 0.1% of supervised SFT tokens see a clean, complete history.
- Against: the multi-turn-data effects measured so far are at 7B-13B (Parrot data alone +0.18 to +0.24 MT-Bench at 13B, TurnWise +12.8 at 7B) **[>=1B only]**; a controlled Pythia 70M-1B midtraining study found generic FLAN instruction data had "minimal effect" ([arXiv 2510.14865](https://arxiv.org/abs/2510.14865)); ufakzeka-1's targeted SFT did not move long-story identity tracking at 151M.

**Attacks.** (a) Pack whole conversations with cross-document attention masking (zero compute). (b) Build a skeleton-render-verify multi-turn set dense in back-references, corrections with stale traps, two-person binding, instruction persistence and topic return, with short assistant turns (`followup/chatdata.md` design; its generation estimate is corrected in `chatdata.verify.md` to roughly 20-50 Mac-days per 100M accepted tokens on the base M5, so start with a 5-20M-token pilot or generate on spare GPU time). (c) Remove or rewrite deflection and identity boilerplate; vary or drop the default system prompt. (d) Multi-turn DPO pairs whose rejected side is a programmatic failure (stale value, deflection, answering as the user) sampled from the model's own rollouts, one epoch at most, keeping the SFT checkpoint as the fallback (DPO helped at 90M and hurt at 151M). (e) Chat template and role tokens in pretraining from step 0, with chat data mixed into the whole run or the decay phase (TII chose 25% in a sweep run at 100 GT total, the same budget as Max's build; 25% and 50% tied). (f) A plausible new technique: **distance-balanced, interference-balanced dialogue data**: sample fact-to-question gaps uniformly, and pair every correction dialogue with a no-correction twin and a two-slot variant, so that neither "first value wins" nor "last value wins" is a usable shortcut.

**Cheapest decisive test.** SmolLM2-135M base and Instruct (same weights, so recipe is the only variable), two equal-token SFT arms: generic (UltraChat or smol-smoltalk slice) vs generic plus about 5k verified multi-turn dialogues, both packed per conversation, 3 seeds each, one model process at a time on the Mac. Score the probe battery, the capacity battery with held-out templates, and held-out paraphrases. Soft if the targeted arm beats the generic arm by more than 15 points on recall, deflection and perspective on every seed.

### Rank 2. Same-type interference: the first-mentioned value wins (mixed; the likeliest real size-sensitive limit near 150M)

**Mechanism.** When two values of the same type compete in context (an original and a corrected appointment day; my cat and my sister's cat; my sister and my aunt), small models default to the first-mentioned one once the question is a few turns away. It shows up as stale values after corrections, wrong-owner answers, and failed two-hop references. The literature calls it proactive interference or "active primacy intrusion"; a 37.8M transformer trained on synthetic variable chains passes through an "early assignment" heuristic phase before it learns real binding when the data forces it ([Wu et al., arXiv 2505.20896](https://arxiv.org/abs/2505.20896)).

**Evidence.**
- Section 2.1: one correction, 4+ intervening turns: 0-7 of 32 for all eight sub-0.6B checkpoints (2T-36T tokens, with and without distillation and RL); LFM2-2.6B 28 and 32 of 32. The failure is primacy, not surface match (the `uprobe` same-wording control fails too) and not confusion among recent values (gold vs previous correction is at chance, gold vs original is 0.00).
- Probe lane (generation, chat templates): 21 of 24 multi-turn correction trials fail from 90M to 600M, with Qwen3-0.6B passing 2 of 3 (`lanes/probe.verify.md`).
- Owner binding at 135M: 0.98 first-mentioned vs 0.31 second-mentioned (base); post-training lifts the second case to 0.65; SmolLM2-360M and Qwen solve it. Two-hop reference at ~100M body is at chance in both recipes tested.
- Size vs recipe: within LFM2 (same recipe) 350M fails and 2.6B passes; across recipes at ~300M, binding and two-hop swing widely; above 1B, size predicts retroactive but not proactive interference resistance across 39 models ([arXiv 2603.00270](https://arxiv.org/abs/2603.00270)) **[>=1B only]**. Learnability evidence at small size is synthetic (Wu et al., 37.8M) or cumulative slot-filling rather than corrections (SimpleTOD, 82M DistilGPT2, MultiWOZ joint goal accuracy 54.5, [arXiv 2005.00796](https://arxiv.org/abs/2005.00796); `followup/floor.verify.md` notes it is not evidence about corrections).
- Against "just data": ufakzeka-1 had templated correction and memory families in SFT, and its long-story identity tracking did not move with data at 151M.

**Hard or soft.** Mixed. Binding and perspective are clearly recipe-sensitive at fixed size. Updates across several turns are unsolved below 0.6B, and no one has published a sub-1B model trained on data that forces them, so the parameter and signal explanations are both open.

**Attacks.** Interference-balanced correction and binding data from the start of training (not only SFT), including rollouts where the assistant itself repeated the old value; a harness-maintained "current facts" state card placed near the end of the context, so a correction becomes a copy from the most recent block (report it as a scaffold, not as the model's ability); on-policy training on the model's own multi-turn rollouts (rank 5). Architecture interplay worth one probe: primacy intrusion may lean on attention sinks, and Max's per-head gated attention cut first-token attention from 46.7% to 4.8% at 1.7B ([arXiv 2505.06708](https://arxiv.org/abs/2505.06708)) **[>=1B only]**; his 124M A/B checkpoints with and without the gate are a free comparison.

**Cheapest decisive test.** Run the already-written `capacity_probe/ft_test.py` on SmolLM2-135M-Instruct, alone on the Mac (400 steps on templated update, binding, two-hop and perspective dialogues whose names, events and distractors are disjoint from the eval battery), then re-score `items.py` and `uprobe.py`. If one-correction accuracy at d=10 rises from 0 of 32 to 26+ of 32 on the held-out templates while d=0 and knowledge items do not regress, the limit is signal; if it stays below 8 of 32 while trained templates pass, 106M of body cannot hold it with this architecture. Repeat on SmolLM2-360M-Instruct. The stronger test is a 30-60M from-scratch pair on the 5070 (1-2B tokens, 3 seeds) with and without 3% interference-balanced dialogues in pretraining.

### Rank 3. Too few pretraining tokens per parameter for general competence (mixed; soft in principle, expensive in practice)

**Mechanism.** At a fixed small size, loss and downstream skills keep improving to at least 10,000 tokens per parameter ([Sardana et al., 150M to 1.5T tokens, arXiv 2401.00448](https://arxiv.org/abs/2401.00448)). Chat quality rides on general competence: answering the single-turn version of each question, parsing ellipsis, not looping.

**Evidence.**
- Every working sub-600M chat model sits at about 2,900 (MobileLLM-350M, 1T) to 83,000 (LFM2.5-230M, 19T) tokens per parameter; Falcon-H1-0.5B-Instruct, the census's best chat scorer, is at about 4,800 (census and training verify files). Max's plan is about 670; his past models were at about 18 (MaxGPT-2) and 85 (MaxGPT-3) (`WRITEUP_NOTES.md`).
- At about 100M of body, Gemma-3-270M (6T) passes 0.75 of the probe lane's single-turn controls and SmolLM2-135M (2T) 0.46 (lenient controls; direction only).
- Tokens do not fix everything: the 28-36T models still fail updates at distance (rank 2).
- The slope flattens: under Chinchilla-style fits a 150M at 300B tokens is within about 0.07-0.09 nats of the same model at 2T, and the remaining gap to 0.5B-class models is mostly the parameter term (compute lane, low confidence, 2022 fits). Very long training also makes small models more fragile under fine-tuning (OLMo-1B beyond about 2.5T; [arXiv 2503.19206](https://arxiv.org/abs/2503.19206)).

**Cost.** 100B tokens is about 17 days on the four usable Titans, 300B about 51 days, 1T about 169 days, SmolLM2-135M's 2T about 11 months; the Titans are busy with Ultra until about Dec 7-14 (compute lane).

**Attacks.** Train 300B token-passes by reusing the 100B build for about 3 epochs (up to about 4 epochs is nearly as good as unique data, [arXiv 2305.16264](https://arxiv.org/abs/2305.16264)). Raise information per token rather than token count: chat-shaped and synthetic data, source tags on knowledge text, MiniPLM-style difference sampling with Ultra as teacher (+1.4 average at 200M compute-matched, [arXiv 2410.17215](https://arxiv.org/abs/2410.17215)). Spend fewer parameters on what tokens cannot fix (rank 4).

**Cheapest decisive test.** No pretraining needed: apply one identical targeted SFT to SmolLM2-135M's intermediate checkpoints at 252B, 1T and 2T ([intermediate checkpoints](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints), stable phase) and to Pythia-160M at 100B and 300B, then score the same battery with 3 seeds. If the 252B checkpoint lands within noise of 2T after targeted SFT, tokens are not the binding limit for mechanics at Max's budget.

### Rank 4. Parametric knowledge capacity (hard)

**Mechanism.** Facts are stored at about 2 bits per parameter at best (about 1,000 exposures, clean data), about 1 bit at 100 exposures, and up to 20x less when useful text is diluted by junk; source tags recover much of that ([arXiv 2404.05405](https://arxiv.org/abs/2404.05405), synthetic biographies, up to 0.5B). Looping does not raise it ([arXiv 2510.25741](https://arxiv.org/abs/2510.25741)). Across 93 open models, obscure-fact accuracy is log-linear in parameters and flat over model generations at fixed size ([arXiv 2604.24827](https://arxiv.org/abs/2604.24827), per `diagnosis_capacity.md`). A 150M model holds at most tens of MB of facts whatever the recipe.

**Evidence.** Firsthand long-tail twin (section 2.1): closed-book 0.78 at 135M vs open-book 1.00. SmolLM2-135M base TriviaQA 4.1 after 2T tokens, but Qwen2.5-0.5B scores 4.3, so this limit is linear and applies to every sub-1B model, including the ones that chat (training verify). Max's MaxGPT-2/3 "confident factual nonsense" (`WRITEUP_NOTES.md`).

**Why it matters for the wall.** Judged chat benchmarks and users score missing knowledge as bad conversation (MT-Bench puts 5 of 8 categories on math, coding, reasoning and knowledge, [arXiv 2306.05685](https://arxiv.org/abs/2306.05685)). A model can be coherent across turns and still look like it "cannot chat".

**Attacks.** Define the target as conversational mechanics plus grounding: answers from the user's words, the history, or retrieved passages, plus a trained "I don't know". Retrieval-trained readers hallucinate far less (BART-large 400M, 68.2% to 7.9% on Wizard of Wikipedia test-unseen, [arXiv 2104.07567](https://arxiv.org/abs/2104.07567), confirmed in `followup/retrieval.verify.md`). Knowledge-light pretraining with anonymized entities improved context-grounded tasks at 135M/360M (2.5B tokens, per-benchmark fine-tuning; [arXiv 2607.12831](https://arxiv.org/abs/2607.12831), confirmed in `followup/retrieval.verify.md`). Sparse memory if "150M" may mean active parameters (Memory+ on a 134M base matched a dense 373M on TriviaQA at 1T tokens, [arXiv 2412.09764](https://arxiv.org/abs/2412.09764)).

**Cheapest decisive test.** Split every eval item into a knowledge-free version (all needed facts in context) and a knowledge-bearing version and report them separately (the `khard` open/closed twin is a first instance). A pass on the knowledge-free version and a fail on the other belongs here, and no recipe change will fix it.

### Rank 5. Exposure bias and self-conditioning: loops, self-copying, drift (soft; needs on-policy signal and a teacher)

**Mechanism.** SFT trains on gold histories, but in a real chat the context fills with the model's own replies. Small models make more early errors, repetition probability rises with each repeat already in context ([arXiv 2206.02369](https://arxiv.org/abs/2206.02369)), and they learn nothing about recovering from their own mistakes.

**Evidence.** SmolLM2-135M repeats a 4-gram 3+ times in 22% of replies and reuses half of an earlier reply in 17% (probe lane); Falcon-90M does not loop. MiniLLM at 125M: off-policy students' excess error grows with length while on-policy training stops it ([arXiv 2306.08543](https://arxiv.org/abs/2306.08543)); its +6.1 judge points at 120M shrank to +0.3 at 340M (posttrain verify). Unlikelihood training cut repetition from 0.617 to 0.055 in a ~90M-class dialogue model ([arXiv 1911.03860](https://arxiv.org/abs/1911.03860), seq2seq). Distilled students loop more than their teachers ([arXiv 2512.12895](https://arxiv.org/abs/2512.12895)) **[>=1B only]**. Multi-turn on-policy distillation has a sub-1B result only on agent tasks (Guided-OPD, 0.6B student; posttrain verify); for conversation it exists only at 1.7B and up **[>=1B only]**.

**Attacks.** Dedup and repetition filtering of chat data; unlikelihood on self-repeated n-grams; SFT on corrected self-generated histories; on-policy distillation from post-trained MaxGPT-Ultra (same tokenizer, compute-trivial at 150M, available after Ultra's post-training around mid-December), mixing in teacher-written turns early to avoid the dirty-history trap Guided-OPD reports. Before Ultra is ready, SmolLM2-1.7B-Instruct shares SmolLM2-135M's exact vocabulary, so on-policy KD can be piloted on SmolLM2-135M.

**Cheapest decisive test.** Run the probe battery twice per model, once on canned golden history and once on the model's own replies. A large own-history gap says on-policy training is worth its cost; a small gap says spend the effort on rank 1.

### Rank 6. Measurement that cannot see the limit (soft; cheap; gates every other test)

**Mechanism.** If the eval mixes knowledge with coherence, changes with the harness, reads a post-trained model off-template, or moves as much with the seed as with the recipe, no recipe change at 150M can be seen, and illusory gains get kept.

**Evidence.** Harness gaps: Gemma 3 270M IFEval 51.2 (Google) vs 27.44 (TII); Qwen3-0.6B Multi-IF 33.3 (Qwen) vs 45.13 (Liquid) (eval verify). A constant "null model" scores 9.55 on MT-Bench ([arXiv 2410.07137](https://arxiv.org/abs/2410.07137)). ufakzeka-1: seed variance equal to recipe spread; 64/64 on repaired questions vs 34/64 on paraphrases. Max's A/B: consecutive held-out evals swing 0.17-0.21 nats, and paired training loss puts the recipe gain at 4.5% (0.134 nats), not the headline 6.9% (compute and training verify). Template vs plain framing moves the same weights from 0.15 to 0.70 recall. The probe battery's graders pass negated answers and guess lists (`lanes/probe.verify.md`). No sub-1B instruction model has a published coherence measurement past 3 turns except agent task completion.

**Attacks.** Fixed eval sets with held-out templates and paraphrases enforced in code; 3 seeds per data ablation; paired metrics (same data order, same eval windows); likelihood probes that run on every pretraining checkpoint in minutes; each model scored in its own chat format and in plain format; separate knowledge-free and knowledge-bearing scores; mutation-tested graders with adversarial wrong answers.

**Cheapest decisive test.** Train the same recipe with 3 seeds (three SFT seeds at 135M, or 30-60M from scratch) and measure the spread on the battery. Any recipe effect smaller than that spread is not evidence.

### Rank 7. Weak teacher signal (mixed; the labs' advantage is real, but its pretraining form barely works at this size)

**Mechanism.** The strongest sub-400M models were teacher-fed: LFM2 used top-32 logit distillation from an internal LFM1-7B in pretraining; LFM2.5-230M's SFT was distilled from LFM2.5-350M; Qwen3-0.6B got off- and on-policy distillation from much larger models. Max's only same-tokenizer teacher is MaxGPT-Ultra (1.1B, about 91 tokens per parameter).

**Evidence.** Controlled pretraining-KD results at Max's scale are null: MobileLLM 125M 43.9 (labels) vs 43.8 (labels plus KD from a 7B) at 2.6-3.2x the cost; MiniPLM 200M 39.9 vs 39.9 at equal compute ([arXiv 2402.14905](https://arxiv.org/abs/2402.14905), [arXiv 2410.17215](https://arxiv.org/abs/2410.17215)). Busbridge's law gives an Ultra-like teacher about -0.04 nats at 100B student tokens token-matched and about zero compute-matched ([arXiv 2502.08606](https://arxiv.org/abs/2502.08606), extrapolated below its fit range; reproduced in training verify). Post-training distillation is where small-scale evidence is positive (MiniLLM at 120M; MobileLLM-R1.5 360M GSM8K 24.5 to 52.8, but 140M only 4.1 to 8.3, math only).

**Attacks.** Put teacher compute into post-training (on-policy, rank 5) and data selection (difference sampling), not full-run online logit KD. Use cross-tokenizer distillation only if a same-tokenizer pilot shows a gain first (the 2026 cross-tokenizer results are at 1B+ students).

**Cheapest decisive test.** After Ultra finishes: a 10B-token pair at 150M, cross-entropy vs cross-entropy plus offline random-sampling KD from Ultra (about 10 Titan-days and 1.8 TB for 50B tokens of logits per the compute lane; a 10B slice is a fifth of that), compared on paired training loss and the likelihood battery. Before then, the on-policy SmolLM2-1.7B to 135M pilot tests the post-training form.

### Rank 8. Inherited vocabulary and embedding tax (soft; a free design choice)

**Mechanism.** Tiny models inherited from large families carry large vocabularies, so much of their parameter budget is an embedding table: Gemma 3 270M is 62.6% embeddings (about 100M of body); Qwen3.5-0.8B 34% (census, from configs).

**Evidence.** On the probe battery Gemma-3-270M (100M body) and SmolLM2-135M (106M body) both fall below Falcon-90M (74M body, 32k vocab). The vocab law puts the optimum near 16-24k at about 300M non-vocab parameters ([arXiv 2407.13623](https://arxiv.org/abs/2407.13623), in-scale point). Max's 49,152 vocab is about 19% of a 150M model at d=576, 25% at d=768. Counterpoint: Gemma's big table did not stop it from the best closed-book knowledge score among the small models in section 2.1.

**Why it ranks last.** It is a one-time design choice, not a wall, and keeping the 49k tokenizer is what makes Ultra a zero-alignment teacher. Worth an A/B (16k vs 49k at equal total parameters, 10B tokens) only if the teacher plan changes.

---

## 4. What one person can remove, and what is expensive

| limit | cheap to remove (days, Mac or one GPU) | expensive or out of reach |
|---|---|---|
| 1. training signal | conversation-level packing and masking; deflection filtering; multi-turn DPO pairs from own rollouts; chat template and chat data in pretraining; a 5-20M-token verified multi-turn pilot set | 100M+ tokens of verified multi-turn data on the Mac (roughly 20-50 Mac-days per 100M, corrected estimate); human-written long conversations |
| 2. interference | interference-balanced correction and binding data; state card; the `ft_test.py` run | nothing known, if the tests show a real size limit at ~100-150M body |
| 3. tokens | reusing the 100B build to ~300B passes (~51 Titan-days after December) | 2T+ tokens (11+ months of the usable Titans) |
| 4. knowledge | scoping the goal; grounded and "I don't know" training; retrieval harness | storing open-domain knowledge in 150M parameters (not possible) |
| 5. exposure bias | repetition filtering, unlikelihood, on-policy KD from SmolLM2-1.7B now or Ultra later | RL with a strong reward model |
| 6. measurement | fixed held-out battery, 3 seeds, paired metrics, likelihood probes | large human evaluations |
| 7. teacher | post-training KD; difference sampling | trillion-token logit KD from a 7B-class teacher |
| 8. vocabulary | a design choice | none |

---

## 5. Key uncertainties

- Whether updates across several turns (rank 2) are learnable at ~100-150M body when the data forces them. Every sub-0.6B model fails, one 2.6B model passes, nobody has trained a small model for it, and the test is written but has not run.
- Whether ufakzeka-1's unmoved identity tracking at 151M is a parameter limit or an undertraining limit (74-89 tokens per parameter). Nothing published separates them.
- Whether likelihood read-outs predict generation after SFT (entity tracking in Pythia is human-level by likelihood at 410M and near chance by generation, [arXiv 2608.18083](https://arxiv.org/html/2608.18083)); and how much plain-transcript probing understates heavily post-trained models such as LFM2.5-350M.
- Whether chat data mixed into pretraining improves multi-turn coherence at all (the only controlled evidence, Falcon-H1-Tiny, is an IFEval gain with 2-turn MT-Bench flat; a Pythia 70M-1B study found generic instruction midtraining had minimal effect).
- Whether DPO helps or hurts at 150M (helped at 90M for one epoch, hurt at 151M in three variants).
- How much of the gap to 0.5B remains at 100-300B tokens once ranks 1 and 2 are addressed; the token-curve evidence at Max's size (lost P1) must be re-measured.

---

## 6. Diagnosis path

Every Mac step runs one model process at a time (the earlier attempt crashed the Mac by running three).
1. **Fix measurement (days, Mac, no training).** Freeze the probe battery plus the capacity battery with held-out templates; add knowledge-free and knowledge-bearing splits; harden graders against negation and guess lists; score each model in its own chat format and in plain format; 3 seeds for anything trained.
2. **Finish the two Mac experiments that the crash cut off (hours).** (a) `capacity_probe/ft_test.py` on SmolLM2-135M-Instruct, then on SmolLM2-360M-Instruct: settles rank 2 at ~100M vs ~300M body. (b) The missing Falcon-H1-Tiny runs, on all five released 90M checkpoints (Instruct, Instruct-Curriculum, both pre-DPO versions, Base), with the MPS watermark environment variable unset or on CPU: separates SFT-in-pretraining, DPO and architecture on exactly the interference battery, and tests a 74M body with the best small recipe.
3. **Measure the token curve without pretraining (days, Mac).** Re-run the lost P1 base-model probe on Pythia-160M/410M checkpoints and SmolLM2-135M intermediate checkpoints, then apply one targeted SFT to the 252B/1T/2T SmolLM2 checkpoints and Pythia-160M at 100B/300B (rank 3 test).
4. **Fix Max's post-training pipeline (code only).** Whole-conversation packing with masking; multi-turn DPO pairs; deflection filter; test at 124M once hardware is free.
5. **Test pretraining-time signal at small scale (5070 when it returns).** 30-60M from-scratch pairs, with and without chat data and interference-balanced dialogues from step 0, 1-2B tokens, 3 seeds.
6. **Only then spend Titan time (after about mid-December).** 150M with the winning mix at 100B, extended to 300B if step 3 shows token-limited skills; fixed SFT packer; multi-turn DPO for at most one epoch; on-policy distillation from post-trained Ultra.
7. **Keep knowledge out of the coherence claim.** Report knowledge-free and knowledge-bearing scores separately, and add grounding and retrieval as a separate, labelled capability.

---

## 7. Housekeeping

- Nothing in Max's repos or on the school server was touched; `maxgpt-ultra/posttrain/*.py` and configs were only read.
- No model, tokenizer, torch, transformers or MLX process was started in this resumed step. The capacity battery was re-summarized from saved JSONL with a copy of `capacity_probe/analyze.py` that writes to the session scratchpad; the capacity folder was not modified.
- New file: `research/recipe_work/pack_sim.py` (self-contained packing simulation; its 1,100 UltraChat rows are cached in `research/recipe_work/uc_cache/`).
- Lost in the crash: this lens's P1 raw outputs and scripts (`probe_tokens.py`, `ft_signal.py`, `out/*.jsonl`) and the P3 fine-tune pilot, which never produced a result.
