# Follow-up track: where is the skill floor for multi-turn chat?

Date: 2026-09-23. Scope: the smallest TOTAL parameter count (embeddings included) at which each multi-turn sub-skill has been shown, what that evidence was measured at, and whether a failure below that size is missing skill or missing knowledge. Builds on the verified first-run lanes in `../lanes/` (census, context, arch, data, eval, posttrain, training, frontier) and uses the corrected versions wherever a `.verify.md` changed a claim. The probe lane (`../lanes/probe.md`) is preliminary and unaudited; its numbers are marked **[probe hint]** and nothing below rests on them alone.

Conventions. "Total" always means embeddings and unembeddings included. Where a source reports non-embedding counts, I convert and say so. "Computed" means my arithmetic from a `config.json` fetched from Hugging Face. "Vendor" means a model card or blog number nobody has reproduced. Evidence measured only at 0.5B or above is flagged **[>=0.5B only]**.

---

## Bottom line

1. The mechanisms under multi-turn chat are cheap: two attention layers at width 64 solve multi-query associative recall perfectly (Zoology, synthetic), grammar is mastered by ~1-10M-parameter models in a narrow world (TinyStories, SimpleStories), and a 37.8M transformer learns exact variable-binding with distractors to above 99.9% (Wu et al., synthetic). So no sub-skill has a known parameter floor above ~40M when the training data teaches it directly.
2. In models trained on ordinary text plus generic chat SFT, the observed floors are much higher and they differ by sub-skill: fluent replies and turn-taking are shown at ~13-30M total, recall of a user fact across a 12-turn, 1.7K-token history at 91M (Falcon-H1-Tiny, a probe hint that fits TII's vendor numbers), multi-turn state tracking at 82M after task fine-tuning (DistilGPT2 on MultiWOZ, JGA 54.5), 3-turn instruction persistence at 230M (LFM2.5-230M, Multi-IF 37.7), and accepting a correction has not been shown in any general chat model up to 0.6B.
3. The gap between "mechanism floor" and "observed floor" is a data and post-training gap, not a capacity gap. The cleanest evidence: the same SmolLM2-135M weights recall a user fact 0.70 of the time from a plain transcript and 0.15 through their own chat template [probe hint]; a random-init 6-layer DST model gets JGA 16.5 while the same shape pretrained gets 54.5; role and end-of-turn tokens alone move DST from 16.8 to 55.8 at 124M.
4. Two things do scale with parameters in the evidence: resistance to proactive interference (stale values winning after updates, size is a significant predictor, context length is not, measured 0.6B and up) and implicit entity tracking in Pythia (53.5% at 70M total, human level by 410M, likelihood read-out only). Both are the "corrections" and "state" sub-skills, so those are where Planck's small end is most at risk.
5. Knowledge failures and skill failures separate cleanly in the small-model evidence: TinyStories finds facts track embedding width while context tracking tracks depth; in the probe, 40 of 63 failed recall or binding checks had the correct fact recoverable by forced-prefix decoding [probe hint]; follow-ups that fail at 135M-270M ("And what about Italy?") often fail their single-turn control too, which makes them knowledge failures.
6. Estimated floors for Planck with purpose-built data (my synthesis, not measured): fluent chat register and turn-taking ~10M; recall of user facts within 2K tokens ~20-40M with recall-dense data (the MTP paper says next-token training barely forms induction below ~30M non-embedding on stories, so this is the riskiest part of the small end); coreference and follow-ups ~30-60M; instruction persistence ~60-100M; corrections and state updates ~60-150M; not looping is a data property with no clear size floor.
7. The headline curve should therefore be 10M / 30M / 60M / 100M / 150M, with a likelihood-probe battery run on every checkpoint so each sub-skill's crossing point is measured rather than inferred.

---

## Detailed findings

### 0. Counting convention: several "small model" names are not totals

- **TinyStories.** Model names are roughly totals only if you count the ~10K token rows actually used ("We use GPT-Neo tokenizer but only keep the top 10K most common tokens", [arXiv 2305.07759](https://arxiv.org/abs/2305.07759) footnote 2). The released checkpoints store all 50,257 rows. Computed from the configs ([TinyStories-33M config](https://huggingface.co/roneneldan/TinyStories-33M/resolve/main/config.json) and siblings): TinyStories-1M (8 layers, d=64) is 0.40M in blocks, ~1.2M total at 10K vocab, 3.75M as stored; 8M (8 x 256) is ~9.4M / 19.7M; 28M (8 x 512) is ~31.4M / 52.0M; 33M (4 x 768) is ~37.6M / 68.5M; 1Layer-21M (1 x 1024) is ~24.9M / 66.2M.
- **Pythia.** Names include both embedding and unembedding ([Pythia repo](https://github.com/EleutherAI/pythia)). Computed non-embedding (vocab 50,304, untied): 70M is ~18.9M non-embedding (6 x 512), 160M ~85M (12 x 768), 410M ~302M (24 x 1024).
- **Gloeckle et al. MTP paper** reports **non-embedding** sizes ([arXiv 2404.19737](https://arxiv.org/abs/2404.19737)), so its "100M" is well above 100M total.
- **Falcon-H1-Tiny-90M-Instruct** is 91.1M total, 32,768 x 512 tied embedding = 16.8M (18.4%), ~74.3M non-embedding (census lane, verified).

### 1. Fluent, grammatical replies

- **TinyStories (1M to ~80M, narrow world, GPT-4-graded).** Grammar is learned first and plateaus earliest: "while grammar can be mastered by relatively small models, consistency and creativity only emerge at a larger size"; consistency with the story beginning "emerges when the hidden size of the model increases from 64 to 128"; "the largest model that we have trained on TinyStories (with roughly 80M parameters) reaches almost perfect scores in terms of grammar and consistency". The headline claim is fluent, consistent multi-paragraph stories "below 10 million parameters with an embedding dimension of 256" or with one transformer block. In the single worked example, the 1M model (d=64, 8 layers) scored grammar 6/10 and consistency 2/10, 8.3M scored 7/10 and 5/10, and 28M scored 9/10 and 9/10 ([arXiv 2305.07759](https://arxiv.org/abs/2305.07759) sec 3.1, 4). One example per size, so treat the per-size scores as illustrative.
- **SimpleStories (Apr 2025).** A suite of 1.25M (4 layers, d=128), 5M, 11M, 30M and 35M models, all with a 4,096-token vocabulary; the 1.25M is total including embeddings. The authors say they "move the frontier regarding the fewest-parameter language model that outputs grammatical natural language", and that their models "consistently outperform TinyStories-33M in all metrics despite having considerably fewer parameters" ([arXiv 2504.09184](https://arxiv.org/html/2504.09184)). The per-metric values are read off a figure (35M grammar roughly 88-90 on a 0-100 LLM-judge scale, approximate, not verified).
- **Open-domain dialogue register at 9-29M.** "Micro language models" (Apr 2026): five decoder-only models from 8.8M to 29.5M pretrained on 1.485B tokens of instruction-dialogue data (UltraChat, MOSS, Instruction_merge_set) plus 323.4M SFT tokens, built to write the first 4-8 words of a reply. GPT-4o-judged overall score on single-turn dialogue QA (judge validated against 10 humans, r=0.803): 8.8M 2.378, 14.43M 2.685, 28.85M 3.044, versus SmolLM2-135M-Instruct 3.373 and LaMini-Neo-125M 1.874. The paper is explicit that it covers "single-turn response initiation rather than long-context multi-turn dialogue management"; repetition appears at 16-word prefixes ([arXiv 2604.19642](https://arxiv.org/html/2604.19642)). Whether the stated sizes include embeddings was not checked.
- **Generic-recipe sub-100M instruct models.** Doge-20M-Instruct is 13.1M total (8 x 256, 32,768 tied vocab, so ~8.4M or 64% embedding, computed) with SFT on SmolTalk and DPO on UltraFeedback; IFEval 9.2 on its card, 7.3 in the repo table; Doge-60M-Instruct (54.6M total) 7.4; Doge-160M-Instruct 16.8 ([card](https://huggingface.co/SmallDoge/Doge-20M-Instruct), [repo](https://github.com/SmallDoges/small-doge)). Supra2-100M-Instruct (created 2026-08-03) is 100.7M total, Qwen3 architecture, 12 x 768, 32,768 tied vocab (25.2M, 25% embedding, computed), 30B pretraining tokens, ~300M SFT tokens of which 77.5% is smol-smoltalk; no benchmarks; its card says "Factual recall, reasoning, arithmetic, and long-range coherence are weak. Expect frequent hallucination and topic drift" ([card](https://huggingface.co/SupraLabs/Supra2-100M-Instruct)). MiniMind (Chinese-first hobby project): minimind2-small is 26M (8 x 512); minimind-3 (Apr 2026) is 63.9M total, 8 x 768, **6,400-token vocabulary** (4.9M, 7.7% embedding, computed), trained with multi-turn SFT, no quantitative multi-turn evaluation ([README](https://github.com/jingyaogong/minimind/blob/master/README_en.md), [config](https://huggingface.co/jingyaogong/minimind-3/resolve/main/config.json)).
- **Skill vs knowledge.** Grammar in a restricted vocabulary is a skill that sits at 1-10M. Open-domain fluency costs more because each extra word needs an embedding row and a meaning (knowledge). The μLM result says a chat register (short, relevant openers) is reachable at ~15-30M when the whole pretraining set is dialogue.

**Floor estimate:** narrow-register fluency ~1-5M total (TinyStories, SimpleStories); open-domain single-turn chat register ~15-30M with dialogue-heavy data (μLM); with a generic web + SmolTalk recipe, instruction following is near the floor below ~60M (Doge IFEval 7-9).

### 2. Turn-taking and role integrity

- **Format tokens matter more than size.** SimpleTOD (GPT-2 124M on MultiWOZ 2.1): joint goal accuracy 16.79 with neither end-of-segment nor user/system tokens, 21.5 with end tokens only, 22.22 with role tokens only, 55.76 with both ([arXiv 2005.00796](https://arxiv.org/abs/2005.00796) Table 4). A 3x effect from delimiters at fixed size.
- **Stopping is learnable at 1M-33M.** TinyStories: GPT-Neo and GPT-2 completions were truncated at the first repeated 4-gram "(after the point the models will just repeat the same sentences over and over). On the other hand, our models learn when to stop generating correctly" ([arXiv 2305.07759](https://arxiv.org/abs/2305.07759) Figure 4 caption).
- **Pre-LLM multi-turn chit-chat at ~90-120M.** TransferTransfo (GPT-1, 12 x 768, about 117M) won ConvAI2 PersonaChat metrics: perplexity 16.28, Hits@1 80.7, F1 19.5 ([arXiv 1901.08149](https://arxiv.org/html/1901.08149)); Hits@1 is candidate ranking, not generation. BlenderBot 90M beat 2.6B Meena on human-rated engagingness in multi-turn chats, and its self-chat gap to 2.7B was not significant (census and posttrain lanes, verified; [arXiv 2004.13637](https://arxiv.org/abs/2004.13637)).
- **Format vs content.** [probe hint] No template leaks or invented user turns in 149 greedy assistant turns per model across 8 models from 91M to 596M. Role failures showed up in content: "role capture" (the model adopts the user's self-description, "As a chef in a busy restaurant...") from 135M to 600M, and answering as the user in 6 failed checks.

**Floor estimate:** format-level turn-taking (stop at end of turn, do not write the user's turn) is a trained habit with no visible size floor down to at least ~90M, and probably far lower (TinyStories stopping at 1M-33M). First/second-person perspective binding is not reliable even at 0.6B in generic chat models [probe hint]; no source measures it lower.

### 3. Recalling a user-stated fact N turns back

**Mechanism floor (synthetic).**
- Induction needs two layers. Olsson et al.: only the one-layer model lacks the induction phase change ([transformer-circuits](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)). Sanford, Hsu, Telgarsky prove no one-layer transformer solves the induction heads task "unless its size is exponentially larger than the size sufficient for a two-layer transformer" ([arXiv 2408.14332](https://arxiv.org/abs/2408.14332)).
- Width is not the constraint for attention. Zoology, 2-layer models at d in {64, 128, 256, 512}, sequence lengths 64 to 512: "attention solves MQAR perfectly at all sequence lengths using a constant model dimension of 64", while gated convolutions need d to grow with sequence length ([arXiv 2312.04927](https://arxiv.org/html/2312.04927v1)). A 2 x 64 attention stack is well under 1M block parameters.
- Supervision shape matters even for attention. A September 2026 toy study (~29K-parameter cells, state 1,024 per head) finds recurrent cells sit at chance (~0.02) retrieving 4 key-value pairs from a distractor haystack under sparse supervision even though they can store 32 pairs, and a **distance curriculum** (sampling fact-to-query gaps uniformly) lifts them to 1.000; the attention baseline scores 1.000 on the fixed haystack, but rotary attention locks in on only 6 of 10 seeds under the collision-key curriculum ([arXiv 2609.16183](https://arxiv.org/html/2609.16183)). Toy scale only, but it is direct evidence that the distribution of recall distances in training data decides whether recall is learned at all.

**In models trained on natural text.**
- MTP paper, children's stories with two-token random names, 1M to 1B **non-embedding**, up to 90 epochs: "For small model sizes, next-token prediction models learn practically no or significantly worse induction capability than 2-token prediction models, with their disadvantage disappearing at the size of 100M nonembedding parameters" ([arXiv 2404.19737](https://arxiv.org/html/2404.19737) Figure 7 caption; arch.verify confirmed). This is the strongest evidence of a real floor near Planck's small end: plain next-token training on natural stories does not reliably form induction below ~30M non-embedding.
- Pythia (Pile, 300B tokens): induction heads appear after about 2B tokens across the tracked sizes 70M to 2.8B total; name-mover heads at 2-8B tokens ([arXiv 2407.10827](https://arxiv.org/html/2407.10827); context.verify corrected details).
- Recall in dialogue after task training: SimpleTOD Table 5, MultiWOZ 2.1 dialogue state tracking (the model must regenerate every slot value the user has set so far, every turn): 6-layer DistilGPT2 (82M total) JGA **54.54**, 12-layer GPT-2 (124M) **55.76**; the same 6- and 12-layer shapes from random init: **16.45** and **20.17** ([arXiv 2005.00796](https://arxiv.org/abs/2005.00796)). Pretraining on general text, not size, is the difference.
- [probe hint] With an identical 12-turn, 1,726-token history, Falcon-H1-Tiny-90M (91M total) emits every planted fact under forced-prefix decoding at every distance and answers 0.80 (short replies) and 0.75 (long replies) correctly with zero deflections. SmolLM2-135M loses retrieval past ~500-1,600 tokens and deflects; its recall is 0.70 from a plain transcript and 0.15 through its chat template.

**Floor estimate:** mechanism ~2 layers at d=64 (synthetic); reliable dialogue recall in a pretrained model shown at 82M (task-tuned) and 91M (general chat, probe hint). Below ~30M non-embedding the MTP result says ordinary next-token pretraining may not form the circuit at all, so at 10-30M total, recall has to be induced by the data (recall-dense, distance-varied) or an auxiliary objective.

### 4. Coreference and follow-ups

- **Pronoun agreement and name binding in Pythia.** Pythia models above 70M learn the gendered-pronoun task at similar rates; IOI (name-mover binding) is learned by 160M but "Pythia-70m ... does not learn the task" ([arXiv 2407.10827](https://arxiv.org/html/2407.10827), context.verify). In total parameters: pronouns by ~160M (probably lower), IOI-style binding at 162M total / ~85M non-embedding, not at 70M total / ~19M non-embedding.
- **Conversational QA with coreference, fine-tuned encoders.** CoQA (questions like "what did he do?" that depend on earlier turns), extractive, history window 64 tokens: DistilBERT (66M) F1 66.6, BERT-base (110M) 76.9, RoBERTa-base (125M) 81.2 ([arXiv 2009.08257](https://arxiv.org/html/2009.08257) Table 2); human F1 on CoQA is 88.8 per the CoQA paper ([arXiv 1808.07042](https://arxiv.org/abs/1808.07042)). The answer text is in the passage, so this isolates reference resolution from knowledge.
- **Entity tracking in narratives (June 2026).** Pythia implicit (likelihood forced-choice) accuracy rises from 53.5% at 70M to 89.6% at 12B, human level (79.3% implicit) reached "at 410 million parameters already"; base Pythia is "at or below chance" on the explicit generation version; OLMo 2 1B base scores 88.5% implicit ([arXiv 2608.18083](https://arxiv.org/html/2608.18083); context.verify: in generation, OLMo 2 1B-Instruct is 55.3% vs human 75.3%). Kim and Schuster: a fine-tuned T5 tracks box contents through state-changing operations, degrading on harder splits; text-only pretraining does not make the ability surface ([ACL 2023](https://aclanthology.org/2023.acl-long.213/); T5 size not re-checked).
- **TinyStories context-tracking prompts** (names, actions, setting of characters): the 1-layer model gets none right, the d=64 model "manages to maintain consistency several times", and the largest 8-layer model "answers most prompts, from all three categories, correctly" ([arXiv 2305.07759](https://arxiv.org/abs/2305.07759) sec 4.2).
- [probe hint] Elliptical follow-ups ("And what about Italy?") pass from 350M up, fail at 135M and 270M, pass at 91M (Falcon). Falcon fails the French follow-up but also its single-turn control, which makes that a knowledge failure.

**Floor estimate:** pronoun/name reference in narrow data ~10-30M (TinyStories); in open text, name binding ~160M total with Pile-era data, 66-125M with task supervision (CoQA, extractive). Generated (not likelihood) entity tracking stays below human even at 1B-13B in the naturalistic benchmark **[>=1B only for the generation numbers]**.

### 5. Instruction persistence

- **Depth, not width.** TinyStories-Instruct: "models that have only 1 layer seem to struggle quite substantially with following instructions (which likely heavily relies on global attention), and 2 layers seem to be sufficient for a certain extent", and instruction quality "depends more heavily on the number of layers" than plot coherence does ([arXiv 2305.07759](https://arxiv.org/abs/2305.07759) sec 3.1). Instructions there precede the story, so this is persistence within one document.
- **Single-turn instruction following, sub-100M.** Falcon-H1-Tiny-90M-Instruct IFEval 66.08 (vendor; SFT data mixed into pretraining plus DPO) vs SmolLM2-135M-Instruct 29.9 (card) or 30.69 (TII harness), Doge-20M 9.2/7.3, Doge-60M 7.4 (census and context lanes, verified; Doge above). Recipe dominates size by a wide margin.
- **Across turns.** The smallest published multi-turn instruction-following number is LFM2.5-230M, Multi-IF (3 turns, instructions accumulate) 37.70; LFM2-350M 32.85; LFM2.5-350M 44.92; Qwen3-0.6B 45.13 in Liquid's harness ([LFM2.5-230M card](https://huggingface.co/LiquidAI/LFM2.5-230M), [LFM2 report](https://arxiv.org/html/2511.23404v1) Table 6; census.verify and context.verify). Nothing below 230M.
- [probe hint] Persistence did not decay over three turns in six of eight models; the two ~100M models are the exceptions (SmolLM2-135M 0.67 at turn 1 to 0.33 at turn 3; Falcon-90M drops one-sentence and all-caps instructions after the first turn) and both ignore a system-prompt persona. At 350M+ a one-line system instruction changes behavior; at ~100M it does nothing.
- At large scale, drift within eight rounds is documented for 70B chat models and is attributed to attention decay toward the system prompt **[>=0.5B only]** ([arXiv 2402.10962](https://arxiv.org/abs/2402.10962)).

**Floor estimate:** within-document instruction following from 2 layers (TinyStories-Instruct, ~10-30M); single-turn IFEval-level following at 91M with the right recipe; persistence across 3 turns shown at 230M; ~100M generic chat models lose format instructions after one turn [probe hint].

### 6. Accepting corrections (state updates)

- **Proactive interference scales with size, not context length [>=0.5B only].** PI-LLM streams semantically related key-value updates and queries only the final values; accuracy "declines log-linearly toward zero as interference accumulates", and errors are retrievals of overwritten values. Across 30 models from Qwen3-0.6B to DeepSeek-V3, "parameter size class is a significant predictor" of interference resistance (t=3.03, p=0.005), "context length has no significant effect" (p=0.886); prompting the model to ignore earlier values helps little ([arXiv 2506.08184](https://arxiv.org/html/2506.08184v3)).
- **The stale-first shortcut is a learning stage, not a wall.** Wu et al. (ICML 2025) trained a 37.8M transformer (12 layers, 8 heads, d=512, RoPE) from scratch on 450K synthetic programs with assignment chains up to 4 hops and distractor chains; test accuracy **>99.9%** at every depth. On the way it passes through a phase (steps ~1,200 to 14,000, accuracy 12% to 56%) where it uses a "line-1 heuristic", picking the constant from the **first** program line, before a systematic dereferencing mechanism forms at steps ~34,000 to 105,400 ([arXiv 2505.20896](https://arxiv.org/html/2505.20896)). A small model can learn "follow the binding, ignore the earlier distractor" when the data forces it; a model that prefers the earliest value is under-trained on that pattern.
- **Dialogue state with updates at 82M.** MultiWOZ DST requires the belief state to hold the user's latest value per slot; SimpleTOD's 82M DistilGPT2 reaches JGA 54.5 (above). This is narrow-domain and supervised with gold belief states.
- [probe hint] 20 of 24 multi-turn correction trials fail across all eight models from 91M to 596M. For "meeting moved from 3 pm to 4 pm", every model puts more probability on the stale value after one distractor turn, while seven of eight prefer the corrected value when the same sentences arrive as one message.

**Floor estimate:** learnable at ~38M on synthetic binding data and at 82M in task-tuned DST; not observed in any general chat model up to 0.6B; in ≥0.6B models, interference resistance grows with size. This is the sub-skill with the largest gap between "possible" and "observed", which makes it the best target for targeted data and the most likely thing to pin Planck's floor if the data is not built for it.

### 7. Staying on topic and returning to topic

- Direct evidence is thin. MT-Bench-101's topic-shift task has no model below 6B **[>=0.5B only]** (context.verify). Supra2-100M reports "frequent hallucination and topic drift" (vendor). BlenderBot 90M-9.4B still "contradict or repeat themselves on occasion" and show "forgetfulness" in 14-turn chats (posttrain lane; [arXiv 2004.13637](https://arxiv.org/abs/2004.13637)).
- [probe hint] Topic return after a digression passes at 91M (Falcon), fails at 135M and 270M, passes from 350M up (one arithmetic item confounds it).

**Floor estimate:** unmeasured below 350M except one probe item at 91M; low confidence. Topic return is recall of an earlier topic plus a policy of going back to it, so its floor should sit at or slightly above the recall floor.

### 8. Not looping

- **Data-driven.** "Repetitions embedded in training data are a primary driver of repetitive text generation"; dropping attention to repeated words in training cut rep-2 from 47.05% to 9.78% on Wikitext-103 ([arXiv 2310.10226](https://arxiv.org/abs/2310.10226); model scale GPT-2 class, exact size not re-checked). Unlikelihood training in a BlenderBot-90M-class model: repetition 0.617 to 0.055 with F1 0.130 to 0.183 (posttrain lane; [arXiv 1911.03860](https://arxiv.org/abs/1911.03860)).
- **Not a size threshold.** TinyStories 1M-33M models stop correctly where GPT-2 and GPT-Neo loop (above). Falcon-90M shows no loops on "tell me more" (cross-turn 4-gram overlap 0.03) while SmolLM2-135M repeats a 4-gram 3+ times in 22% of replies and stays looped at its card temperature [probe hint]. TII reports smaller models "loop significantly more" and that chain-of-thought traces in the 90M tool-calling mix caused infinite loops (vendor; census.verify notes this finding comes from the tool-calling track).
- **Why it is worse in multi-turn:** the context fills with the model's own text, which is what the self-reinforcement effect feeds on (GPT-2/BART, [arXiv 2206.02369](https://arxiv.org/abs/2206.02369)). The induction-dominance paper once cited as the mechanism was withdrawn (context.verify).

**Floor estimate:** none in parameters; clean behavior shown at 1M-33M (narrow data) and 91M (chat). Loops are a data, objective and decoding property.

### 9. Knowledge vs skill: what fails for which reason

| Failure seen in small chat models | Skill or knowledge | Evidence |
|---|---|---|
| Wrong fact in a follow-up ("Italy is the capital of Italy") | knowledge (plus some binding) | fails the single-turn control too [probe hint]; facts track embedding width in TinyStories |
| Deflecting ("I don't have access to personal information") | post-training policy | 36 of 95 failed user-fact checks; not size-ordered (Qwen2.5-0.5B 45%, Falcon-90M 0%) [probe hint] |
| Forgetting a user fact | mostly policy, sometimes capacity | 40 of 63 failed recall/binding checks had the fact recoverable [probe hint]; SmolLM2-135M plain-transcript 0.70 vs template 0.15 |
| Stale value after a correction | skill (interference), data-limited | PI-LLM size effect at ≥0.6B; learnable at 37.8M synthetic |
| Speaking as the user, adopting the user's job | skill (perspective binding) | role capture 135M-600M [probe hint]; no lower measurement |
| Loops and self-copy | data and objective | Repetition-in/repetition-out; TinyStories stop cleanly |
| Grammar errors on rare words | knowledge (vocabulary) | TinyStories fluent at 1-10M inside a child vocabulary |

Knowledge ceilings for scale: at the Physics 3.3 ceiling of ~2 bits per parameter (reached only with ~1,000 exposures per fact; ~1 bit at 100; gated MLPs 1.3x lower at 100 exposures; context.verify), a 10M model holds at most ~2.5MB of facts and a 150M model ~37MB ([arXiv 2404.05405](https://arxiv.org/abs/2404.05405); arithmetic mine). SmolLM2-135M scores 4.1% on TriviaQA after 2T tokens (data lane).

### 10. Sub-100M chat and instruct models, with what they show in multi-turn

| Model | Total params (embedding share) | Training | Multi-turn evidence | Source |
|---|---|---|---|---|
| μLM-8.8M to 28.85M | 8.8M to 29.5M (not checked) | 1.49B dialogue tokens + 323M SFT | single-turn openers only; 28.85M judged 3.04 vs SmolLM2-135M 3.37 | [arXiv 2604.19642](https://arxiv.org/html/2604.19642) |
| Doge-20M-Instruct | 13.1M (64%) | SmolTalk SFT + DPO | none; IFEval 9.2 / 7.3 | [card](https://huggingface.co/SmallDoge/Doge-20M-Instruct) |
| MiniMind2-small / minimind-3 | 26M / 63.9M (minimind-3: 7.7%, vocab 6,400) | multi-turn SFT, tool data from Qwen3-4B | claimed, not measured | [README](https://github.com/jingyaogong/minimind/blob/master/README_en.md) |
| TinyStories-Instruct-33M | ~37.6M at 10K effective vocab, 68.5M stored | story + instruction data | single-document instructions only | [card](https://huggingface.co/roneneldan/TinyStories-Instruct-33M) |
| Doge-60M-Instruct | 54.6M (~31%, computed) | SmolTalk SFT + DPO | none; IFEval 7.4 | [repo](https://github.com/SmallDoges/small-doge) |
| PleIAs Monad | 56.7M (3.7%, vocab 8,192) | 200B SYNTH tokens | card: "no support yet for multi-turn" | [card](https://huggingface.co/PleIAs/Monad) |
| DistilGPT2 in SimpleTOD | 82M | general pretraining + MultiWOZ | task DST across turns, JGA 54.5 | [arXiv 2005.00796](https://arxiv.org/abs/2005.00796) |
| BlenderBot 90M | 87.5M (seq2seq) | Reddit + BST | 14-turn human evals, engaging, forgets and repeats | [arXiv 2004.13637](https://arxiv.org/abs/2004.13637) |
| Falcon-H1-Tiny-90M-Instruct | 91.1M (18.4%) | 800B tokens, 25% SFT in the mix, DPO | 2-turn MT-Bench 4.33 (vendor, likely GPT-4o-mini judge); 12-turn recall 0.75-0.80 [probe hint] | [TII blog](https://tiiuae-tiny-h1-blogpost.hf.space/) |
| Falcon-H1-Tiny-Multilingual-100M-Instruct | ~100M (not checked) | multilingual; anti-curriculum gave "no clear boost" | none published | [card](https://huggingface.co/tiiuae/Falcon-H1-Tiny-Multilingual-100M-Instruct) |
| Supra2-100M-Instruct | 100.7M (25%) | 30B web tokens, 300M SFT tokens | card: long-range coherence weak, topic drift | [card](https://huggingface.co/SupraLabs/Supra2-100M-Instruct) |

---

## What this means for Planck at 10M to 150M

1. **Nothing in the evidence forbids a 30M multi-turn model; the generic recipe forbids it.** Every sub-skill's mechanism has been shown at or below ~40M on data that teaches it. The observed floors in chat models (90M to 600M) come from recipes built for single-turn benchmarks. Planck's contribution can be exactly this: data and objectives that teach each sub-skill directly, measured per sub-skill.
2. **The 10M point is a register-and-format model.** Expect fluent short replies, clean turn-taking and stopping, and near-zero world knowledge (~2.5MB ceiling). Recall is the open risk: the MTP result says plain next-token training on natural stories barely forms induction below ~30M non-embedding. At 10M, recall must be forced by the data (dense, distance-varied fact-question pairs) or an auxiliary objective, and even then it is a bet.
3. **30M is the most informative point on the curve.** It is below every observed chat floor but above the synthetic mechanism floors (Zoology, Wu et al.). If Planck-30M passes user-fact recall within 2K tokens and simple follow-ups, the "skill is cheap" thesis is confirmed; if it fails with the fact still recoverable by forced prefix, the failure is policy and fixable; if the fact is gone, it is capacity.
4. **Corrections and perspective are the likely binding constraints at 60-150M.** They fail in every general chat model up to 0.6B [probe hint], PI-LLM shows the stale-value error scales with size at ≥0.6B, and no public SFT set targets them. They need dedicated data from day one, and they should be measured with likelihood probes (stale vs corrected, user vs sister) on every checkpoint, not just after SFT.
5. **Spend parameters on depth, not vocabulary.** TinyStories ties context tracking to layers and facts to width; instruction following needs at least 2 layers and improves with more. Embedding arithmetic (computed): at 10M and d=256, an 8,192 vocab costs 2.1M (21%) and a 32,768 vocab 8.4M (84%); at 30M and d=384, 8,192 costs 3.1M (10.5%); at 150M and d=768, 8,192 costs 6.3M (4.2%) while 49,152 costs 37.7M (25%). The small end of the curve needs a 4K-8K vocabulary (SimpleStories 4,096, MiniMind 6,400, Monad 8,192) or it is mostly embedding table.
6. **Use role and end-of-turn tokens from step 0, in pretraining.** SimpleTOD's 3x from delimiters and the probe's plain-transcript vs template result both say format is first-order at small size.
7. **Pretraining on general text matters for every sub-skill, even the "cheap" ones.** Random-init vs pretrained DST at the same shape was 16.5 vs 54.5; the BabyLM dialogue-only 135M model got dialogue minimal pairs up (63-64% vs 57-58%) but BLiMP down (56.05% vs 72.16%) ([arXiv 2510.20358](https://arxiv.org/html/2510.20358v1)). Dialogue-only pretraining buys form and costs general competence, which agrees with the data lane.

---

## Ledger ideas

Every "cheapest test" below is a design for when the Mac is free; nothing was run for this report. "Probe battery" means a likelihood-plus-generation battery per sub-skill (recall at distances 1/3/6/9 turns, stale vs corrected value, user vs third-party name, format persistence at turns 1-3, loop rate), scored on base and chat checkpoints.

1. **Recall-distance curriculum in dialogue data.**
   - Hypothesis: sampling the gap between a planted user fact and its question uniformly from 0 to the full context (instead of the natural short-gap skew) makes recall learnable at 10-30M.
   - Why: at toy scale, a distance curriculum took fixed-state cells from chance to 1.000 and improved lock-in rates; attention also failed to lock in on some seeds ([arXiv 2609.16183](https://arxiv.org/html/2609.16183)).
   - Cheapest test: 30M model, two arms differing only in the gap distribution of 20% synthetic recall dialogues, 2 seeds each, ~1B tokens; probe battery recall at 1/3/6/9 turns.
   - Win: recall accuracy at 9 turns at least 15 points higher in the curriculum arm on both seeds, with no loss on the other probes.

2. **Two-token prediction auxiliary loss below 30M only.**
   - Hypothesis: a 2-token-prediction head during pretraining forms induction in 10-30M models that otherwise barely form it.
   - Why: MTP "vastly improved" induction at ≤30M non-embedding on children's stories, with the gain gone at 100M ([arXiv 2404.19737](https://arxiv.org/abs/2404.19737)); at 340M+ MTP hurt, so this is size-specific.
   - Cheapest test: 10M and 30M, NTP vs NTP+2-token head, 2 seeds; measure repeated-name induction accuracy and probe recall.
   - Win: at 10M, induction accuracy at least 2x the NTP arm and user-fact recall up; at 30M, no regression in validation loss.

3. **"Latest value wins" correction data with stale traps.**
   - Hypothesis: a few percent of synthetic dialogues where a user value is corrected, then queried after 1-5 distractor turns, removes the stale-value preference at 30-150M.
   - Why: a 37.8M transformer learns to override a first-line heuristic when the data forces it ([arXiv 2505.20896](https://arxiv.org/abs/2505.20896)); DST at 82M holds latest slot values ([arXiv 2005.00796](https://arxiv.org/abs/2005.00796)); every general chat model to 0.6B fails this [probe hint].
   - Cheapest test: 30M, with and without 3% correction dialogues in the pretraining mix; probe log-prob margin corrected vs stale at 1/3/5 distractor turns.
   - Win: margin positive (corrected preferred) on at least 80% of items at 3 distractor turns, vs below 50% for the control.

4. **Perspective-binding data (I/you, user vs third party).**
   - Hypothesis: dialogues with two named people and "my/your/her" questions, plus role-swap traps ("Hi, I'm a chef" then "what do you do?"), eliminate role capture at 30-100M.
   - Why: role capture appears from 135M to 600M in generic chat models [probe hint]; format delimiters alone gave SimpleTOD a 3x gain, so role structure is learnable at small size.
   - Cheapest test: 60M, SFT with vs without 5% perspective data; probe forced choice user name vs sister name, and generation "what do you do?" after a user self-description.
   - Win: role-capture rate under 5% and forced-choice accuracy above 90%, vs the control's rate.

5. **Tiny vocabulary at the small end (4K-8K).**
   - Hypothesis: at 10-30M total, a 4,096-8,192 vocab gives better multi-turn skill per parameter than 16K-32K because blocks get the parameters.
   - Why: SimpleStories gets grammatical output from 1.25M with a 4,096 vocab and beats TinyStories-33M at smaller size ([arXiv 2504.09184](https://arxiv.org/html/2504.09184)); Doge-20M spends 64% on embeddings and has IFEval 7-9.
   - Cheapest test: 10M total, vocab 4K vs 8K vs 16K at equal total parameters (depth adjusted), same data by bytes; bits-per-byte plus the probe battery.
   - Win: the smaller-vocab arm matches or beats bits-per-byte and wins on at least 3 of 5 probe sub-skills.

6. **"TinyChat": a restricted-world conversation corpus to find the pure skill floor.**
   - Hypothesis: with a child-level vocabulary and no world knowledge needed, 1-10M models do multi-turn recall, follow-ups and corrections, giving a clean skill-only curve.
   - Why: TinyStories moved coherent story generation from hundreds of millions of parameters to under 10M by shrinking the world ([arXiv 2305.07759](https://arxiv.org/abs/2305.07759)); no one has done the same for dialogue.
   - Cheapest test: generate ~300M tokens of restricted-vocabulary multi-turn chats with planted facts and corrections (teacher off-Mac or later), train 1M/3M/10M, run the probe battery.
   - Win: any sub-skill above 80% at 10M or below, which would be the lowest published multi-turn floor.

7. **Repetition-filtered data plus unlikelihood on self-copies.**
   - Hypothesis: filtering repetitive spans and adding token-level unlikelihood on repeated n-grams in the model's own samples removes loops and self-copy at 10-30M.
   - Why: repetition in data is the main driver of degeneration ([arXiv 2310.10226](https://arxiv.org/abs/2310.10226)); unlikelihood cut repetition 0.617 to 0.055 at BlenderBot-90M class; SmolLM2-135M loops at 22% while Falcon-90M does not [probe hint].
   - Cheapest test: 30M, filtered vs unfiltered data, then +/- unlikelihood during SFT; measure cross-turn 4-gram overlap on "tell me more" chats at T=0 and T=0.7.
   - Win: loop rate under 5% at greedy with no drop in recall probes.

8. **Depth sweep at fixed 30M.**
   - Hypothesis: at 30M total, 16-24 thin layers beat 4-8 wide layers on recall, binding and persistence, while losing on facts.
   - Why: TinyStories ties context tracking and instruction following to layers and facts to width; MobileLLM deep-thin wins at 125M (arch lane).
   - Cheapest test: 30M total, 6x448 vs 12x320 vs 24x224 (vocab 8K), same tokens, 2 seeds; probe battery plus a small single-turn fact quiz.
   - Win: the deepest arm is best on at least 3 of the 4 context sub-skills on both seeds.

9. **Checkpoint-level likelihood probes as the floor measurement itself.**
   - Hypothesis: likelihood probes (stale vs corrected, gold vs in-context foil, user vs third party) cross chance before generation passes, so they locate each floor earlier and cheaper than chat evals.
   - Why: entity tracking is human-level by likelihood at 410M while generation stays at chance for base Pythia ([arXiv 2608.18083](https://arxiv.org/html/2608.18083)); the probe's forced-prefix read-out separated "fact gone" from "fact ignored".
   - Cheapest test: score public checkpoints first (Pythia-70M/160M/410M, SmolLM2-135M base, TinyStories-33M) on the probe set; then add it to every Planck checkpoint.
   - Win: a monotone crossing point per sub-skill that predicts the post-SFT generation pass/fail at each size.

10. **Plain transcript format instead of a special-token template at the small end.**
    - Hypothesis: at 10-60M, "User:/Assistant:" plain-text turns carry more of the base model's in-context skill into chat than a new special-token template learned only in SFT.
    - Why: the same SmolLM2-135M weights recall 0.70 from a plain transcript vs 0.15 through the template [probe hint]; SimpleTOD shows delimiters matter a lot.
    - Cheapest test: one 30M base model, two short SFT runs differing only in template (plain text vs special tokens, both present in pretraining vs SFT-only); probe battery.
    - Win: at least 20 points better recall in the plain or pretrained-template arm.

11. **Harness-side state card for corrections.**
    - Hypothesis: a one-line "current facts" block regenerated by the harness each turn lowers the corrections floor to the recall floor, because the model only has to copy.
    - Why: the stale-value error is interference, not access; externalizing state is standard for large models (context lane) and a 2-layer copy mechanism is cheap.
    - Cheapest test: at inference only, on any Planck checkpoint, run the correction probes with and without the card.
    - Win: correction accuracy within 10 points of plain recall accuracy with the card, at 30M.

---

## Open questions

1. Does ordinary next-token pretraining on chat-shaped text form induction below 30M non-embedding, or is the MTP paper's "practically no induction" result specific to children's stories with two-token random names? This decides whether 10M is a chat model or a register model.
2. Where does the stale-value (correction) preference flip in a model trained with correction data: 30M, 60M or never below 150M? PI-LLM's size effect starts at 0.6B and says nothing about trained-in behavior.
3. Is Falcon-H1-Tiny-90M's multi-turn recall (probe hint) a property of its anti-curriculum SFT-in-pretraining mix, its hybrid attention+Mamba layers, or its 800B tokens? A small ablation on Planck would separate the first from the others.
4. How much of the 135M-270M follow-up failure is knowledge? The probe controls suggest most of it; a knowledge-free follow-up set (answers present in context) would settle it.
5. Do topic return and self-consistency have floors distinct from recall, or are they recall plus policy? Nothing below 350M measures them outside the probe.
6. Is perspective binding (I vs you) a data gap or a capacity gap? It fails from 135M to 600M in generic models and nobody has trained against it at small size.
7. The TinyStories and SimpleStories per-size scores come from figures and single examples; a re-scoring of the released checkpoints on a fixed rubric would make the grammar floor numeric.
8. Doge IFEval numbers differ between card (9.2) and repo (7.3) for the same model; neither is independently reproduced.

---

## Summary table: sub-skill floors

| Sub-skill | Smallest size with evidence (total params) | Best evidence and scale | Confidence | What would move it lower |
|---|---|---|---|---|
| Fluent grammatical replies | ~1-5M in a narrow world; ~15-30M for open-domain chat register | TinyStories (<10M, d=256), SimpleStories 1.25M with 4K vocab; μLM 28.85M judged 3.04 vs SmolLM2-135M 3.37 (single-turn) | high (narrow), medium (open-domain) | restricted-vocabulary chat data; 4K-8K tokenizer |
| Turn-taking, stopping, not writing the user's turn | no floor seen; clean at 91M in chat, stopping learned at 1M-33M | SimpleTOD delimiters 16.8 to 55.8 at 124M; TinyStories stopping; 0 template leaks 91M-596M [probe hint] | medium | role and end tokens in pretraining from step 0 |
| Role integrity (I vs you, not adopting the user's persona) | not shown in any general chat model up to 0.6B | role capture 135M-600M [probe hint]; no published measurement | low | perspective-binding data (idea 4) |
| Recall of a user-stated fact N turns back | mechanism: 2 layers, d=64 (synthetic); in dialogue: 82M task-tuned, 91M general chat | Zoology MQAR; SimpleTOD DistilGPT2 JGA 54.5; Falcon-90M 0.75-0.80 at 1.7K tokens [probe hint]; MTP: weak induction ≤30M non-emb on stories | medium | distance curriculum (idea 1), 2-token auxiliary loss (idea 2), plain-transcript format (idea 10) |
| Coreference and follow-ups | ~66-125M with task supervision; name binding ~162M total (Pythia, not 70M) | CoQA DistilBERT 66.6 / BERT 76.9 F1; Tigges IOI; TinyStories context-tracking prompts at 8 layers | medium | knowledge-free follow-up data; depth (idea 8) |
| Entity/state tracking through updates | human level by likelihood at 410M (Pythia, 300B Pile tokens); generation below human through 13B | Drozdz and Heilbron 2026 | medium (likelihood), low for generation | code and state-update data; likelihood probes (idea 9) |
| Instruction persistence | within one document: 2 layers (TinyStories-Instruct); across 3 turns: 230M | LFM2.5-230M Multi-IF 37.7; Falcon-90M IFEval 66.1 single-turn; ~100M models drop format after 1 turn [probe hint] | medium | multi-turn instruction data; re-injected system prompt |
| Accepting corrections | learnable at 37.8M (synthetic binding), 82M (task DST); not observed in general chat ≤0.6B | Wu et al. >99.9%; SimpleTOD; PI-LLM size effect ≥0.6B; 20/24 fail 91M-596M [probe hint] | medium that it is learnable, high that generic recipes miss it | correction data (idea 3); state card (idea 11) |
| Staying on and returning to topic | one item at 91M [probe hint]; otherwise ≥350M | probe topic-return items; Supra2-100M "topic drift" | low | topic-return dialogues; recall fixes above |
| Not looping | no parameter floor; clean at 1M-33M (narrow) and 91M (chat) | TinyStories; Repetition In Repetition Out; unlikelihood at 90M class; SmolLM2-135M 22% loops [probe hint] | medium | repetition filtering plus unlikelihood (idea 7) |
