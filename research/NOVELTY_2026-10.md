# Has anyone already done Planck? (novelty check, 2026-10-02)

For Max. Question asked 2026-10-02: make sure no one has done what Planck is doing. Scope: anything published, released or announced through 2026-10-02, with extra weight on 2026-09-01 onward, plus anything earlier that `research/REPORT.md` (2026-09-23) missed.

How it was checked: five search agents (arXiv full OAI harvest of cs for 2026-06-01 to 2026-10-02 plus keyword sweeps, Hugging Face models/papers/datasets APIs including every text-generation repo of 0.3M-420M parameters created 2026-06-01 to 2026-10-02, OpenReview, GitHub, Hacker News, an r/LocalLLaMA archive, about 100 web searches), then one verifier per top candidate reading the primary source. This final pass re-checked the four "major/partial" leads the verifiers had not reached (SmallTalkLLM, Loom, Vertex, BananaMind cards) at the source. Every item was grepped against this repo to see what was already cited. No model was run.

## 1. Has anyone done Planck's thing?

**No.** Nobody has shown a model under 150M (let alone under 30M) passing a strict multi-turn test with corrections, user-rule persistence and binding, and nobody has published a measured minimum-size curve for those skills. But the neighbourhood is crowded: the question, tiny from-scratch chat models, synthetic multi-turn training and facts-outside-the-weights are all already taken by someone, so Planck's novelty is the strict bar and the measured floor, not the idea.

Evidence:
- **Every dedicated multi-turn benchmark found scores 1B and up.** MultiChallenge (incl. its 2026-03-23 update), Multi-IF, MT-Bench-101, TurnWise, Lost in Conversation, LongMemEval/LoCoMo, DToM-Track, SPINE, and the new September state-tracking papers (CICM, IntentFlux, StateMemBench, MemOps, 2609.08435) all start at 1.5B or larger, most at 3B-7B or frontier models. The smallest model scored on a multi-turn correction benchmark is Qwen2.5-0.5B-Instruct (arXiv 2609.26035, as the base of a wrapper).
- **The only public multi-turn numbers below 150M are weak and loose.** BananaMind's self-reported 24% to 35% of 75 multi-turn items at 10M-139M; RxT's 3.1 to 3.8 out of 10 on a BLEU/cosine reward at 12M-160M; Loom Spark 3.2 recalling a turn-1-3 fact at turn 9-10 in 2 of 4 conversations; cRia-75M's second-turn MT-Bench 1.33/10. None is a pass/fail state test, and every one shows failure.
- **The one pre-registered curve on the same question has no results.** SmallTalkLLM (3.87M-25.70M, protocol dated 2026-08-19) says its scaling study has not been run, and nothing has been pushed since 2026-08-27.
- **The closest same-size explicit attempt failed.** ufakzeka-1 (151M body, 2026-09-18) published multi-turn gates and reported identity-tracking failures it calls the limits of 151M.
- REPORT.md's statement still holds: nothing published shows strict multi-turn chat below 1B.

## 2. The closest works, ranked by threat to Planck's core claim

**1. SmallTalkLLM (Anson Washeck).** https://github.com/AnsonWasheck/SmallTalkLLM . Repo created 2026-08-19, last push 2026-08-27, MIT, 2 stars. Not in this repo before today (RC-12's "SMALLTALK" is an unrelated fake responder).
- Shows: Planck's question almost word for word (how small can a model be and still hold casual conversation, with knowledge deliberately removed). A pre-registered protocol (`docs/REPORT.md`) with thresholds declared before training, a capability credited to the smallest size not undercut by a larger one, a ladder of 3.87M / 5.28M / 6.69M / 8.10M / 14.95M / 25.70M, a 4,096 vocab, a Qwen-teacher data pipeline, and a separate StateBench. Only the 6.69M model is trained (StateBench directional 0.583). A second line moves state, memory and repetition control into a deterministic harness around the frozen 6.7M model.
- Does not show: any curve result; any strict state test (its bar is 10 turns with no obviously broken reply, plus probes such as recalling a dog's name; no corrections, rule persistence, binding, two-hop or lookup); a sealed split; a comparator panel; seeds.
- Why first: same question, same priority sizes, same pre-registration habit. If it restarts, it could post a "smallest conversational model" curve first, but against a much looser bar. Planck's answer is the strict test and a dated lock.

**2. Loom family (Textile Labs).** https://huggingface.co/textilelabs/Loom-Spark-3.2 (created 2026-10-01). Siblings: Loom-Spark-3 12.2M (2026-09-11), Loom-Weave-3 31.5M (2026-09-20), Loom-Tapestry-3 69.2M (2026-09-21), Loom-Crucible-Preview 155.0M (2026-09-23), Tapestry/Spark 3 Flash 7.18M. MIT. Not in this repo.
- Shows: from-scratch tiny chat models with facts outside the weights in practice. The model writes a `<lookup>` query, a harness fetches one Wikipedia sentence, and the model answers from it. Own 4,096 BPE, OASST1 for multi-turn structure plus hand-written memory and identity data, no LM-written training text. A 133-item acceptance battery per release (Spark 3.2: 122/133; 42 of 44 turns on target in 10- and 12-turn chats; turn 9-10 recall 2 of 4).
- Does not show: corrections, user-rule persistence, binding, own-answer consistency or loops as scored families; any test with more than a handful of multi-turn items; seeds; a controlled curve (each release changes recipe and data, so the battery column is not a size curve).
- Why second: it ships every week at Planck's sizes with Planck's facts-outside idea, and is the group most likely to add multi-turn rows next.

**3. BananaMind-2 chat ladder and BananaMind Instruct Bench 1.1.** https://huggingface.co/BananaMind/BananaMind-2-Nano-Chat , https://huggingface.co/BananaMind/BananaMind-2-Pro-Preview-Chat , https://huggingface.co/datasets/BananaMind/BananaMind-Instruct-Bench-1.1 . Models 2026-07-21 to 2026-08-03, bench 2026-07-22. Not in this repo.
- Shows: from-scratch chat models at 9,968,128 / 25,178,752 / 49,559,552 / 138,971,520 parameters (own tokenizers, smol-smoltalk SFT) scored by deterministic graders with a multi-turn sub-score: 18/75, 19/75 and 26/75 multi-turn passes at 10M, 25M and 139M; system prompts 2/60 to 9/60. This is the closest existing "size vs multi-turn" table.
- Does not show: a pass at any size; free-running chat (the model writes one reply to a scripted history, so its own answers are never fed back and loops or own-answer consistency cannot be tested); item text (gated); a family breakdown; seeds; a controlled curve (tokenizers and token budgets differ by size). Self-reported, with repetition penalty 1.1. Correction to the sweep: the Pro chat model comes from a 51.9B-token checkpoint, not the 100B run.

**4. Micro Language Models (Cheng, ... Zettlemoyer, Gollakota).** https://arxiv.org/abs/2604.19642 , v1 2026-04-21; checkpoint https://huggingface.co/Sensente/Swen-28M (28,844,544 params, MIT). Already in REPORT.md.
- Shows: five sizes from 8.8M to 29.5M trained from scratch on multi-turn chat-format data (UltraChat, MOSS) with their own tokenizer, compared against 70M-256M baselines. Its GitHub README claims EMNLP 2026 acceptance (not confirmed elsewhere).
- Does not show: multi-turn competence. The model writes only the first 4-8 words of a reply, and the paper hands multi-turn context to a cloud model by design.

**5. Reactive Transformer (RxT), Adam Filipek.** https://arxiv.org/abs/2510.03561 , v1 2025-10-03; weights https://huggingface.co/ReactiveAI/RxT-Alpha-Nano and RxT-Alpha-Micro-Supervised (2025-10-06, 13.3M and 28.8M total, non-commercial RAML licence, click-through gate). Not in this repo.
- Shows: 12M / 26M / 100M / 160M stateful models trained from scratch on synthetic TinyStories multi-turn interactions; on 9-step dialogues the mean reward rises 3.1, 3.4, 3.7, 3.8 (of 10) against 2.4 for a 22M stateless baseline. A reviewer will cite "12M beats 22M".
- Does not show: a pass/fail test (the reward is BLEU plus cosine similarity), corrections, rules or binding, usable chat (its own cards call the memory weak), retrieval. Custom architecture. Nothing new from the org since 2026-03-05.

**6. ufakzeka-1.** https://arxiv.org/abs/2609.25081 , 2026-09-18, 182M total / 151M body. Already in REPORT.md. Explicit multi-turn gates (name recall 162/200, user correction 65/90, resisting stated wrong numbers 16/36), identity tracking never fixed, seed spread as large as recipe spread. One size, no curve, commercial-API teachers.

**7. LMLM and Co-LMLM (Cornell).** https://arxiv.org/abs/2505.15962 (v1 2025-05-21, ICLR 2026) and https://arxiv.org/abs/2607.07707 (2026-07-08). From-scratch pretraining that sends facts to an external knowledge base, 124M-382M (Co-LMLM at 135M and 360M). No chat, no multi-turn, nothing below 124M. Used in `experiments/K` (arch.md F16, K5) and CORPUS.md but not cited in REPORT.md. Together with KLLM (2607.12831, already cited) this means "train a small model from scratch to look facts up instead of memorizing" is published prior art.

**8. CICM, stale binding (UC Berkeley).** https://arxiv.org/abs/2609.38866 , 2026-09-30. A multi-turn update benchmark (1,200 preference dialogues) scored on 3B and up; Pythia-160M is used only for mechanism work, where removing the responsible heads fixes 38.4% of old-value errors (8.7% for random heads). Explains failure as several old values outweighing the current one in attention. Not tiny-scale behaviour and only one RC-12 family. Companions: https://arxiv.org/abs/2609.33883 (2026-09-27, 7B-70B, old and new values both stay readable after an update) and https://arxiv.org/abs/2610.00910 (2026-10-01, 1.5B-32B, the first-mentioned fact is easiest to reach).

**9. Vertex-0.6-15M-Instruct.** https://huggingface.co/VertexResearch/Vertex-0.6-15M-Instruct , 2026-09-06, 14,957,568 params, Apache-2.0. The only sub-30M card that claims multi-turn in-context memory (recalling a name from 1,000+ tokens back). No numbers at all; the card says greedy decoding loops badly.

**10. Further out (cite, no threat).**
- Veyra2 Instruct ladder, https://huggingface.co/veyra-ai/Veyra2-Mango-30M-Instruct , 2026-09-28: 4.9M / 9.9M / 15.7M / 30.7M, smoltalk2 SFT, single-turn benchmarks only.
- cRia-LM-75M-Instruct, https://huggingface.co/sz14/cRia-LM-75M-Instruct , 2026-09-11, 75.7M: two-turn MT-Bench 1.76 (second turn 1.33), multi-turn loss admitted.
- SYNTH / Monad paper, https://arxiv.org/abs/2609.37891 , 2026-09-29: Monad-56M, single-turn only, facts in the weights by design.
- FLM position paper, https://arxiv.org/abs/2509.02225 , 2025-09-02: argues for skills in weights and facts in tools, with single-turn probes on 135M and up. The closest published statement of Planck's thesis.
- Executable state machines for multi-turn data, https://arxiv.org/abs/2609.08435 , 2026-09-08: prior art for "skeleton, teacher renders, program verifies" (conversation_ideas.md idea 6).
- StateMemBench https://arxiv.org/abs/2608.19652 (2026-08-20), IntentFlux https://arxiv.org/abs/2609.32520 (2026-09-26), MemOps https://arxiv.org/abs/2607.12893 (2026-07-14), Cochinescu https://arxiv.org/abs/2609.26035 (2026-09-22), REA https://arxiv.org/abs/2610.00958 (2026-10-01): correction and rule-persistence ideas, all 0.5B to frontier.
- Pythia dialogue size sweep https://arxiv.org/abs/2509.16487 (2025-09-20): 160M still scores about 2.0-2.6/10 on GPT-4-judged turn-taking after SFT.
- Daedalus-150M https://arxiv.org/abs/2608.20210 (2026-08-20): pre-registered from-scratch base model, no chat.
- jkminder SFT ladder https://huggingface.co/jkminder/d12_135m_seed1_sft (2026-09-02 to 09-04, 135M-973M, cc-by-nc): ARC/MMLU only, but 8 identical-config replicates at 135M are outside data on SFT seed noise.
- Nawah-50M-RAG-Chat-8K https://huggingface.co/oddadmix/Nawah-50M-RAG-Chat-8K (2026-08-13): Arabic support bot, gold-forced multi-turn eval.
- ARK-65M https://huggingface.co/ThingAI/ARK-65M (2026-08-23): hobby chat model that sends facts to a web_search tool, no rigorous multi-turn eval.

## 3. Models to add to the RC-12 baseline panel

The panel's smallest model today is Falcon-H1-Tiny-90M, so nothing sits at Planck's priority sizes. Add these as reported extras (not headroom-panel members; they will mostly sit at the floor, which is the point). Section 12 is fixed at the lock, so this has to happen before Oct 11-14.

| model | total params | created | licence | why |
|---|---|---|---|---|
| BananaMind-2-Nano-Chat | 9,968,128 | 2026-07-22 | Apache-2.0 | from-scratch 10M chat model; checks its self-reported multi-turn failure independently |
| Vertex-0.6-15M-Instruct | 14,957,568 | 2026-09-06 | Apache-2.0 | the one sub-30M card claiming multi-turn memory; test the claim |
| Loom-Spark-3.2 | 22,827,840 | 2026-10-01 | MIT | facts-outside tiny chat; run with tools off, in its own `<user>/<loom>` format |
| BananaMind-2-Mini-Chat | 25,178,752 | 2026-07-22 | Apache-2.0 | the 25M point |
| Swen-28M (Micro LMs) | 28,844,544 | paper 2026-04-21 | MIT | the peer-reviewed (claimed) sub-30M chat-format model; caveat: trained for short openers |
| Veyra2 Blueberry-5M, Blueberry-10M, Mango-15M, Mango-30M Instruct | 4.9M / 9.9M / 15.7M / 30.7M | 2026-09-28 | Apache-2.0 | the only public ladder down to 5M with one family recipe |
| cRia-LM-75M-Instruct | 75,719,395 | 2026-09-11 | Apache-2.0 | sub-100M with a published two-turn score; needs trust_remote_code |
| RxT-Alpha-Nano, RxT-Alpha-Micro-Supervised | 13.3M, 28.8M | 2025-10-06 | RAML (non-commercial), gated | the prior art a reviewer will raise; needs a custom responder and Max's OK to accept the gate |

Optional: BananaMind-2-Pro-Preview-Chat (138,971,520; licence not checked) as the top of that ladder. Engine support (vLLM vs HF, custom code) is unchecked for all of these. Pin each one's repo revision like the rest of the panel.

Also widen the s12 rule. Today it says the Level R claim moves below LFM2.5-230M or Falcon-90M if either passes. Make it general: a Level R or Level A claim at size N stands only if no public panel model at or below N passes the same bar.

## 4. What should change in the plan, framing or priorities

1. **Rewrite the novelty sentence before anything goes public.** Planck should not claim to be the first to ask how small a chat model can be (SmallTalkLLM), the first from-scratch chat models under 30M (BananaMind-2, Loom, Veyra2, Vertex, Micro LMs, MiniMind), the first to train tiny models on synthetic multi-turn dialogue (RxT, Micro LMs), or the first to keep facts outside the weights (LMLM, Co-LMLM, KLLM, Loom). Suggested wording: *Planck measures the first floor for strict multi-turn state skills: the smallest from-scratch model that passes a sealed, mutation-tested 12-turn test on its own conversation history (corrections, binding, rule persistence, own-answer consistency, role, loops), with seeds, against a public panel from 5M to 2.6B in one harness.*
2. **Keep the Oct 11-14 lock and the timestamped push.** The lock's GitHub release and Software Heritage snapshot are Planck's dated priority claim. SmallTalkLLM timestamped its protocol on 2026-08-19, and Loom ships weekly. Do not let the push slip.
3. **Make the first public result the baseline table.** Scoring the sub-30M public chat models above on RC-12 (minutes each on the 5070) is new on its own: nobody has run a strict multi-turn test on any of them. It shows the floor is open, it uses only work already planned, and it can go out with the lock.
4. **Priorities: no change to the 30M-and-under focus.** That is where all the competing hobby work sits, but none of it has a strict result, so the floor headline is still open. The 60M and 150M anchors are less contested (ufakzeka-1 failed at 151M body).
5. **Use CICM's account as a free diagnostic for "first value wins" (E001, E004).** CORR already varies the number of corrections k = 1/2/3. Report the CORR pass rate by k for every model; CICM's mechanism predicts failure rising with k.
6. **Document masking in packing (PLAN E1, lines 433 and 807).** Loom measured 107/133 without per-conversation masking and 120/133 with it at 22.8M, on the same data. That is outside support for choosing masked packing.
7. **Cite in REPORT.md related work:** SmallTalkLLM, Loom, BananaMind-2 and its bench, RxT (2510.03561), LMLM (2505.15962), Co-LMLM (2607.07707), CICM (2609.38866), FLM (2509.02225), 2609.08435, SYNTH (2609.37891), Vertex, Veyra2, cRia. Fix REPORT.md line 102: the SYNTH paper gives Monad about 180B tokens, not 200B.
8. **Make "naked model" explicit in RC-12's claim text.** SmallTalkLLM and Loom both get part of their behaviour from a harness (state store, repetition control, search). RC-12 scores the model on its own history with no external state, apart from the LOOKUP family's supplied tables; say so in one line so the comparison cannot be blurred.

## 5. Gaps in this search

- **ICLR 2027 submissions were not searched.** The OpenReview group reports them as not public, and listing endpoints sit behind a bot challenge that was not bypassed. Papers also posted to arXiv are covered through 2026-10-02. Recheck around Oct 15-20.
- **BabyLM 2026** (workshop 2026-10-28, no interaction track this year): papers may appear late October. Recheck after Oct 28.
- **EMNLP 2026** accepted list not checked directly; the Micro LMs acceptance rests on its own README.
- **Gated or unverified:** BananaMind Instruct Bench items (reading them means accepting its terms, which needs Max's OK); BananaMind/bananamind-bunny-alpha (created 2026-10-02, gated, empty card); "Fact-Conditional Pretraining" at https://openreview.net/forum?id=bMIYS9fEqU (403 challenge, never confirmed to exist; facts-outside method with no chat claimed either way).
- **Rate limits:** some arXiv API and web queries failed with HTTP 429 (multi-turn with SmolLM2 or 135M; "knowledge-free conversation benchmark"; skills/facts separation with retrieval for small chat models). Semantic Scholar was rate-limited throughout. GitHub code search returned nothing.
- **Not searched directly:** X/Twitter, Discord, Chinese-language sources (Zhihu, Bilibili, the MiniMind community). Reddit came from the Arctic Shift archive, which may be incomplete; r/MachineLearning title queries timed out.
- **Not individually verified:** about 60 lower-ranked items from the sweeps (hobby models such as G1/G2-nano, Supra2-Medium, flame-27m, KeyLM-75M, ShallowSeek-mini; leaderboards; memory-architecture papers); only the sweep agents read them. Spot checks of Pebble-10M-Chat (11.1M, commonsense benchmarks only) and ARK-65M found no rigorous multi-turn evaluation. RxT's promised memory-benchmark paper was not found (one query only).
- Unannounced lab work is invisible by definition. From their own pages: Liquid's last sub-400M release is LFM2.5-230M (2026-06-25), and TII and Google posted nothing after June 2026. The HF API shows no new sub-1B chat model from HuggingFaceTB, LiquidAI, tiiuae, google, PleIAs, facebook or allenai since 2026-08-15.
