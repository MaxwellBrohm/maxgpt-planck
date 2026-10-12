# Brainstorm track: memory baked into the weights of a tiny chat model

Date: 2026-09-23. Question (Max, ledger idea #10): after a conversation, turn notes like `dog: Pickles` into a weight update for this user, so a later session with an empty context still knows the dog is Pickles. For every family of methods: the smallest model it was tested at, the cost per update, how precise it is (bleed into other dogs and other people), how it handles a correction (Biscuit, then Pickles), and how many facts it holds before the model degrades. Then: has anyone done this below 150M, and what is the cheapest decisive experiment on the Mac.

Builds on `../followup/retrieval.md` (notes block and verbatim user-turn retrieval as the default memory), `../followup/bits.md` (adapter capacity numbers, verified), `../followup/archfp.md` (memory layers cost total parameters), `../followup/floor.md` (corrections fail at every size to 0.6B in context) and `../lanes/probe.md` (SmolLM2-135M's own chat template hurts in-context recall: 0.15 vs 0.70 on a plain transcript).

Conventions. Every number has a URL or says "unsourced" or "my arithmetic". Sizes are total parameters where the source gives them. "Fetched summary" means the number came from a fetched page summary rather than my own reading of the table; numbers I read from the paper's own table text (via pdftotext) are marked "(table read)". Nothing was run or trained for this report.

---

## Bottom line

1. **Nobody has built a sub-150M chat model with per-user long-term memory in its weights and measured recall and bleed across sessions.** The nearest work is one notch off on each axis: User as Engram (June 2026) writes per-user facts into a hashed memory table and measures recall and contamination, but its smallest model is 178M and it is a base LM with completion-style prompts; a frozen GPT-2 124M with trained persistent memory adapters keeps memory across LoCoMo sessions, but in a latent state, not weights, and retains only 7-18%; GradMem writes a context into 32 memory tokens by gradient descent at 124M-160M, but per context, not per user; MAC/CaMeLS fine-tune an 82M DistilGPT2 on a document stream and get StreamingQA F1 of 3.8-10.2. So Max's idea is untested at Planck's size, and a clean result either way is new.
2. **Across every scale measured, weight memory reaches roughly 60-95% of the in-context ceiling and never beats it when the facts fit in the window.** Generative Adapter on Multi-Session Chat at 7B: F1 40.2 vs 66.0 with the full history in the prompt; PLUM at 8B: 81.5% vs 89.5% with the right conversation injected; GradMem at 124M: SQuAD-short EM 54.9 vs 64.2 with full context; Doc-to-LoRA at 2B: 83.5-85% of full context. Weight memory wins only when the context would not fit (Doc-to-LoRA past 4x the window; Engram past about 100 facts per user, where retrieval starts missing). For Planck, 8 user facts as a notes block cost about 40-80 tokens (my arithmetic), so the in-context notes block (brainstorm idea #7) is the baseline to beat, and weight memory is a candidate for overflow beyond the window, not a replacement.
3. **What you train on matters more than the method, and "a few LoRA steps on the raw transcript" is predicted to be the weakest arm.** At 124M, facts stated in one phrasing are not extractable by questions (9.7%) while 5 paraphrases plus reordering give 96.6% (Physics of LMs 3.1). At 8B, plain cross-entropy on the conversation gives 17-41% and question/answer pairs with balanced "no" negatives and up-weighted answer tokens give 75-81.5% (PLUM). At 7B, training on the passage alone moves QA from 32.7% to 33.5%, while training on generated implications gives 39.7-47.0% (SEAL). Cartridges report that next-token training on the corpus is "not competitive" and context distillation is what works. Best cheap recipe to test: notes rendered as many question phrasings plus negatives plus an anchor that keeps unrelated outputs fixed, or self-distillation from the same model with the notes in context.
4. **Bleed is the main risk, and personal facts are the worst case for it.** Every method that adds a global weight delta leaks into unrelated text: a per-user LoRA raised bits-per-byte on unrelated text by +1.784 on a 1.22B base (17/20 users worse) while an addressed table row changed it by +0.00005; a single plain fine-tuning edit on distilGPT-2 (82M) cost 0.938 in perplexity drawdown vs 0.225 for MEND; 10 sequential ROME edits zeroed all four benchmarks of Pythia-160M. Personal facts all share one subject ("I", "my", "the user"), which is exactly where locate-then-edit methods break (same-subject edits perturb each other, measured 1.5B-7B) and where updates to highly connected entities spread errors furthest (FACTPROP, 2026). In-context memory bleeds too (56.2% cross-domain leakage at frontier scale), so the bleed test has to run on the in-context baseline as well.
5. **Corrections: do them in the notes, then rebuild the weights; do not stack gradient patches.** Evidence for stacking being fragile: edited facts are forgotten faster than pretrained ones under later fine-tuning (ROME-edited retention 0.374 vs 0.824 intrinsic at Llama3-8B), and same-subject sequential edits interfere. Mechanisms that handle overwrites natively are addressed stores: the delta-rule fast weight replaces the value at a key (Schlag et al., 40M/90M), GRACE splits a conflicting key (T5-small 60M), Engram rows follow "last write wins" on the same trigger. The cheapest correct design for Planck: keep the notes store as the source of truth (latest value wins in text), and after each session retrain the user's adapter from scratch on the current notes ("sleep" as a rebuild). This is untested at any size as far as I found, and it makes deleting a user trivial (delete the adapter).
6. **Capacity is not the constraint; interference is.** A rank-1 LoRA on the MLPs of SmolLM2-135M is 190,080 parameters, which at the measured 1.7-2.75 bits per trainable adapter parameter (2M-8M bases) holds roughly 325K-520K bits, against roughly 100-300 bits for 8 user facts (my arithmetic). Observed limits come from interference instead: Engram's per-user table falls to 35% top-1 at 1,000 facts (339M); sparse memory fine-tuning at 1.3B cut held-out QA by 11% vs 71% for LoRA and 89% for full fine-tuning after 1,000 new facts; ROME collapses Pythia-160M after 10 edits, MEMIT survives 100.
7. **Test-time training (TTT, Titans, Atlas, HOPE, TTT-E2E) is memory in weights, but reset at the end of each sequence**, tested at 125M-3B, and nobody has measured what happens if the state is saved per user and reloaded next session. That is a real gap and one of the ledger ideas below, but it needs a Planck architecture with a fast-weight layer, so it is not the first experiment.
8. **The cheapest decisive experiment** (section 4): SmolLM2-135M-Instruct (already in the local Hugging Face cache), 8 synthetic users with 8 facts each, per-user rank-8 MLP LoRA (1.52M params, 3 MB per user in bf16), five training arms (raw transcript, notes, QA+negatives, QA+negatives+anchor, self-distillation) against notes-in-context and no-memory controls, then a correction session (stack vs rebuild), scored in a fresh empty session on held-out phrasings, bleed items, a context-beats-weights item and general chat bits-per-byte. About 4-8 hours on the Mac in one process (estimate, ±2x; roughly 10x less on the RTX 5070). Kill rule: if the best weight arm gets under half the in-context recall or adds more than 10 points of bleed, record the negative result and stop.

---

## 1. Findings by family, with the smallest scale each was measured at

### Summary table

| Family | Smallest size tested | Cost per update | Precision (bleed) | Corrections | Facts before degradation |
|---|---|---|---|---|---|
| Fast weights (Schmidhuber 1992, Ba 2016, delta-rule FWPs) | RNN with 20 hidden units (Ba); 40M and 90M LMs (Schlag) | an outer-product write per token, no gradient | within one sequence only | delta rule overwrites the value at a key | bounded by the d x d state; resets per sequence |
| Dynamic evaluation, TTT on neighbours, Temp-LoRA | 117M GPT-2 (TTT-NN); 150M (dynamic eval) | 1 gradient step per neighbour or chunk | reset after each instance, so never measured | not measured | not measured |
| Locate-then-edit (ROME, MEMIT, AlphaEdit) | Pythia-160M (general-ability collapse study) | ROME 0.64 s per edit on GPT2-XL | same-subject and ripple failures | same-subject edits perturb each other | ROME: 10 edits collapse Pythia-160M; MEMIT: 100 fine; AlphaEdit: 2,000 sequential at 1.5B-8B |
| Learned editors (KE, MEND, ENN) and codebooks (GRACE, SERAC) | distilGPT-2 82M, BERT-base 110M, BART-base 139M; T5-small 60M; BlenderBot-90M | one editor forward pass; GRACE 0.13 s per edit | MEND 0.225 ppl drawdown vs 0.938 for plain fine-tuning at 82M | GRACE splits conflicting keys | GRACE: 1,000 sequential edits in 137 keys |
| Per-user adapters (OPPU, PLUM, per-user LoRA) | 339M base LM (the rank-64 LoRA baseline in User as Engram's density sweep); PLUM at 8B | 10 epochs per conversation (PLUM); 14.2 MB per user at rank 64 | +1.784 bpb on unrelated text at 1.22B (aggressive recipe) | PLUM streams 100 conversations; no explicit correction test | PLUM: 100 conversations without catastrophic forgetting |
| Context-to-weights generators (Generative Adapter, Text-to-LoRA, Doc-to-LoRA, PRAG/DyPRAG, Profile-to-PEFT) | 1B (Gemma-3-1B on-device hypernetwork; Llama-3.2-1B in DyPRAG) | one forward pass, sub-second | not measured | not measured | Doc-to-LoRA near-perfect needle recall to ~40K tokens |
| Test-time training layers (TTT, Titans, Atlas, HOPE, TTT-E2E) | 125M (TTT); 170M (Titans) | an inner step per token or chunk, inside the forward pass | per sequence | n/a | memory-module size |
| Self-updating memory pools (MemoryLLM, M+) | 7B with a 1B memory pool | a forward pass writes memory tokens | not measured as bleed | not measured | MemoryLLM retains under 20K tokens; M+ over 160K |
| Keyed sparse stores (User as Engram; sparse memory fine-tuning) | 178M total (Engram-d8) | ~1 s per fact (OPT); 1,000 steps for a joint write; apply 0.03 ms | +0.00005 bpb on unrelated text; 6.1% cross-user leak from value coincidence | same trigger: last write wins | Engram: >90% top-5 to ~300 facts/user, 35% top-1 at 1,000 |
| Consolidation by distillation (SELF-PARAM, prompt distillation, Cartridges, SEAL, "sleep") | 1B ("LMs Need Sleep"); 3B (SELF-PARAM, prompt distillation) | tens of seconds (oracle context distillation: 40 s at 2B) | not measured | SELF-PARAM: 20 sequential injections decay to ~0.3 F1 | see left |
| Memory tokens written by gradient, persistent latent memory | 124M GPT-2, 130M Mamba, 160M Pythia (GradMem); 124M GPT-2 (persistent memory adapters) | K = 1-5 gradient steps on 8-32 tokens | per context or per user by construction | n/a | 96 key-value pairs at 88.4% with K = 5 (small synthetic model) |

Details and sources follow.

### 1.1 Fast weights

- **Ba et al. 2016, "Using Fast Weights to Attend to the Recent Past"** ([arXiv 1610.06258](https://arxiv.org/abs/1610.06258)). Associative retrieval task, tiny RNNs: with 20 recurrent units, fast weights reach 1.81% error vs 60.81% for an LSTM of the same width; at 50 units, 0% vs 1.85% (table values via search summary of Table 1). Scale: toy, far below 1M parameters.
- **Schlag, Irie and Schmidhuber 2021, "Linear Transformers Are Secretly Fast Weight Programmers"** ([arXiv 2102.11174](https://arxiv.org/abs/2102.11174)). Linear attention is a fast-weight memory written by outer products; replacing the additive write with a **delta rule** lets the model "correct the current mapping from keys to values", which helps most "in overcapacity regimes". WikiText-103 models of **40M and 90M** parameters (search summary). This is the only family here whose write rule is designed for overwrite, which is exactly the "actually his name is Pickles" operation.
- Modern descendants (DeltaNet, Gated DeltaNet) sit in the same place but are all reset per sequence.
- **Persisted across sessions:** "Trained Persistent Memory for Frozen Decoder-Only LLMs" ([arXiv 2603.22329](https://arxiv.org/abs/2603.22329), 2026) attaches six memory adapters (including a Hebbian write `P_t = gamma * P_(t-1) + A^T V` and a sparse slot write) to a **frozen GPT-2 124M** and carries the memory across LoCoMo sessions. Three of six methods retain 7-18% ("retained-memory score": the F1 lost when the persistent state is ablated), the other three under 0.4%; all six converge at 10x memory capacity; retention falls from ~18% at short lags to 7-9% past 256 turns (fetched summary). No in-context upper bound and no bleed test. This is the closest thing to Max's idea below 150M, and it is a latent state, not a weight edit.

### 1.2 Dynamic evaluation, test-time training on neighbours, Temp-LoRA

- **Dynamic evaluation** (Krause et al. 2019, [arXiv 1904.08378](https://arxiv.org/abs/1904.08378)): gradient steps on the text seen so far during evaluation; Transformer-XL WikiText-103 perplexity 18.3 to 16.4 (model size unsourced here). **Rannen-Triki et al. 2024** ([arXiv 2403.01518](https://arxiv.org/abs/2403.01518), sizes **150M, 400M, 1B**; sizes and the quote read from the PDF): online adaptation is "memory in weights"; it wins the compute/quality trade-off under a large distribution shift, and the advantage "disappears when the model is finetuned to the target distribution before the online adaptation phase"; LoRA gives a better performance-to-memory trade-off than full updates. Reading for Planck: a chat model that is already chat-tuned has less to gain from generic weight updates on the chat itself; the gain has to come from the specific facts.
- **TTT on nearest neighbours** (Hardt and Sun, ICLR 2024, [arXiv 2305.18466](https://arxiv.org/abs/2305.18466)): **GPT-2 117M** and 774M, GPT-Neo 1.3B; one gradient step per retrieved neighbour at lr 2e-5; bits per byte down ~20% with 50 neighbours; the model is reset to the base after each test instance. Perplexity only.
- **Temp-LoRA** ([arXiv 2401.11504](https://arxiv.org/abs/2401.11504), COLM 2024): a temporary rank-64 LoRA trained on the text generated so far (1,024-token chunks, 2 epochs, lr 5e-5) and **discarded after generation**; PG19 perplexity -13.2%. Bases 6B-13B (Llama2-7B/13B, Mistral-7B, Qwen-7B, Yi-6B). No recall or bleed measurement.

### 1.3 Knowledge editing: locate-then-edit

- **ROME / MEMIT / AlphaEdit** are evaluated at GPT2-XL (1.5B) and above. AlphaEdit (ICLR 2025, [arXiv 2410.02355](https://arxiv.org/abs/2410.02355)) projects the edit onto the null space of preserved knowledge and "boosts the performance of most locating-then-editing methods by an average of 36.7%" on LLaMA3, GPT2-XL and GPT-J, with 2,000 sequential edits in batches of 100 (search summary for the protocol).
- **Collapse at small scale, the one direct sub-150M number.** "Should We Really Edit Language Models?" (NeurIPS 2024, [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/370fa2e691f57eb319bc263a07dad4a5-Paper-Conference.pdf), Table 4, table read): **Pythia-160M** edited sequentially on CounterFact. ROME after **10** edits scores 0 on MMLU, GSM8K, BBH and CommonsenseQA (from 0.2435 / 0.0174 / 0.0742 / 0.1884); MEMIT after 100 edits is unchanged (0.2468 / 0.0235 / 0.0743 / 0.1990). ROME collapses within 10-50 edits at every Pythia size up to 12B, while MEMIT holds to 100 at every size; the paper's summary finding is "larger models exhibit less side effect". Caveat: Pythia-160M's unedited scores are near chance, so "unchanged" for MEMIT mostly means it did not break the output format.
- **Two-phase forgetting** (Gupta et al., [arXiv 2401.07453](https://arxiv.org/abs/2401.07453)): sequential ROME and MEMIT edits first erode gradually, then collapse abruptly. A search summary puts MEMIT's collapse after about 1,400 edits; I did not verify the number or the model.
- **Same-subject edits interfere** ("Related Knowledge Perturbation", [arXiv 2502.06868](https://arxiv.org/html/2502.06868), GPT-2 XL, GPT-J, LLaMA-2-7B): when a second attribute of the same subject is edited, the first edit's efficacy drops, because the key is taken "only from the subject's last token"; FT, MEND and KN were insensitive. For personal memory every fact has the same subject (the user), so this is the expected failure, not an edge case (my inference).
- **Ripple effects** (RippleEdits, [arXiv 2307.12976](https://arxiv.org/abs/2307.12976)): editing methods "fail to introduce consistent changes", and "a simple in-context editing baseline obtains the best scores". Scale: large models (sizes not in the abstract).
- **Edited facts are fragile under later training** ([arXiv 2507.14198](https://arxiv.org/html/2507.14198), GPT-2 XL and Llama3-8B, fetched summary): after downstream fine-tuning, ROME-edited knowledge retains 0.374 vs 0.824 for pretrained knowledge (Llama3-8B); fine-tuning-based edits 0.634 vs 0.831; about 3 paraphrases per edit, or freezing the layers holding the edit, closes the gap. Directly relevant to stacking one session's update on top of another.
- **Knowledge distortion and conflicts** ("Unveiling the Pitfalls", [arXiv 2310.02129](https://arxiv.org/abs/2310.02129)): reverse and composite conflicts between edits amplify inconsistency; edits "irrevocably warp the innate knowledge structure".
- **Popular entities spread damage** (FACTPROP, [arXiv 2609.08067](https://arxiv.org/abs/2609.08067), September 2026, sizes not in the abstract): among facts a model already knows, those attached to highly connected entities are the most likely to be corrupted by neighbouring updates, and updates to such facts "propagate errors more broadly". In chat, "I", "my", "dog", "sister" and "name" are about as connected as entities get (my inference).

### 1.4 Learned editors and codebook editors (the sub-150M evidence)

- **MEND, KE, ENN at 82M-139M** (MEND, ICLR 2022, [arXiv 2110.11309](https://arxiv.org/abs/2110.11309), Table 4, table read; single edits, not sequential):

| editor | FEVER, BERT-base 110M: edit success / acc. drawdown | zsRE, BART-base 139M: ES / acc. DD | Wikitext, distilGPT-2 82M: ES / ppl. DD |
|---|---|---|---|
| fine-tune | 0.76 / <0.001 | 0.96 / <0.001 | 0.29 / 0.938 |
| fine-tune + KL | 0.64 / <0.001 | 0.89 / <0.001 | 0.17 / 0.059 |
| ENN | 0.99 / 0.003 | 0.99 / <0.001 | 0.93 / 0.094 |
| KE | 0.95 / 0.004 | 0.98 / <0.001 | 0.25 / 0.595 |
| MEND | >0.99 / <0.001 | 0.98 / 0.002 | 0.86 / 0.225 |

  At the generative 82M model, plain fine-tuning on a single edit both fails (29% success) and damages the model (perplexity drawdown 0.938), and adding a KL anchor cuts the damage 16x but also the success. The paper notes that at small scale "fine-tuning overfits more severely than with larger models". A learned editor (MEND) or a meta-learned base (ENN) does far better, which is the case for meta-learning writability (ledger idea L5).
- **GRACE** (NeurIPS 2023, [arXiv 2211.11031](https://arxiv.org/abs/2211.11031), table read): a codebook of (key = cached activation, value, radius) at one layer; base weights untouched. **T5-small 60M**, **1,000 sequential edits** on zsRE: test retention (TRR) / edit retention (ERR) of 0.69 / 0.96 for GRACE vs 0.56 / 0.82 plain fine-tuning, 0.27 / 0.99 fine-tune with retraining, 0.25 / 0.27 MEND, 0.72 / 0.31 Defer. The 1,000 edits used only 137 keys (7.3 edits per key), a codebook of 210,569 scalars (0.35% of the model). A conflicting edit near an existing key splits the key ("decreasing the influence radius of the overlapping key, then adding a new codebook entry"). Per-edit time on the GPT2-XL task: GRACE 0.13 s, fine-tune 0.26 s, MEND 0.63 s, ROME 0.64 s. Known weakness (my recollection of later benchmarks, unverified here): poor generalization to paraphrases, because a paraphrase can land outside the key's radius.
- **SERAC** ([arXiv 2206.06520](https://arxiv.org/abs/2206.06520)): edits a dialogue agent's sentiment toward topics on **BlenderBot-90M** with 99.1% edit success and zero drawdown (search summary). The edits live in an explicit memory plus a scope classifier and a counterfactual model, so this is retrieval, not weights; it is evidence that a 90M chat model can apply a stored natural-language note, which supports the in-context notes baseline.
- **Hewitt et al. 2024, "Model Editing with Canonical Examples"** ([arXiv 2402.06155](https://arxiv.org/abs/2402.06155)): one example per behaviour, out-of-distribution evaluation, and a hard cap on general loss increase (as tight as 0.01%). Pythia **70M to 6.9B**: "LoRA is the strongest of the three learning methods [full fine-tuning, LoRA, MEMIT], largely consistently across model sizes." Gains under the tight cap are tiny (0.6% average for LoRA at 6.9B); sense fine-tuning in a 170M Backpack gets 4.8% (fetched summary). Reading: under a strict no-degradation rule, generic weight edits buy very little at any size.

### 1.5 Per-user adapters

- **PLUM** (Magister et al., Apple, [ACL L2M2 2025](https://aclanthology.org/2025.l2m2-1.5.pdf), text read): injects prior user conversations into a per-user LoRA. **Llama 3 8B Instruct**, rank 16, alpha 64, all linear layers, 10 epochs per conversation, batch 8, conversations streamed one at a time (100 conversations). Each conversation is augmented into question/answer pairs with an equal number of negatives ("questions adjacent to the topic ... to which the answer is 'no'", because "without negative samples the LLM will default to always positively answering"). Results on yes/no questions: 81.5% (with system prompt) vs 83.5% for Q/A-RAG top-3, 86.5% for conversation-RAG top-3, 89.5% with the correct conversation injected, 50% zero-shot. Standard cross-entropy gives 17.0% / 41.0%; weighting question/answer tokens by 10 gives 75.0% / 81.5%; 20 epochs made the model output incoherent text (0.0% in two settings). "We observe no sign of catastrophic forgetting", though some conversations were never learned; benchmark accuracy dipped slightly except SiQA.
- **OPPU** and **Profile-to-PEFT** ([arXiv 2510.16282](https://arxiv.org/html/2510.16282), ACL 2026): per-user PEFT on LaMP style and preference tasks, and a hypernetwork replacing per-user training; 7B-class bases (OPPU's Llama-2-7B is from my recollection, not rechecked here). Style and preference, not fact recall.
- **Per-user LoRA as a baseline inside User as Engram** (section 1.9): on a 1.22B base LM, rank 64 on Q/K/V, 1,500 steps per fact, **+1.784 bits per byte on unrelated text** (per-user range +0.44 to +3.74, 17 of 20 users worse); on instruction-tuned 3B-8B bases, indirect recall hurt 0-20% of users, on the base LM 85% (table read). The 1,500-steps-per-fact recipe is far more aggressive than PLUM's, so treat +1.784 as an upper bound on LoRA contamination, not a typical value.
- **Sequential personalization of small models** ([arXiv 2606.27634](https://arxiv.org/html/2606.27634v1), June 2026): Qwen3.5-0.8B, Llama-3.2-1B, Gemma-3-1B, one cumulative rank-8 LoRA over three sequential tasks. KL divergence from the base predicts collapse (Gemma: KL 1.623, backward transfer -0.193, final accuracy 0.320; Qwen: KL 0.300, +0.030, 0.591). Useful as a cheap degradation monitor for Planck's experiment. Tasks, not user facts.
- **Parametric individualization** ([arXiv 2609.10155](https://arxiv.org/abs/2609.10155), September 2026): one DoRA adapter per human participant on **Qwen3-0.6B** (r = 16, continued pretraining on the person's texts); strong text-level individuality (dz = 1.27) but "the adapter adds knowledge rather than alignment with the individual" at the answer level.

### 1.6 Context-to-weights generators (hypernetworks)

All at 1B or above:
- **Generative Adapter** (ICLR 2025, [arXiv 2411.05877](https://arxiv.org/abs/2411.05877), fetched summary): a ~500M-parameter generator maps context to a 32M-parameter adapter in one forward pass, accumulating chunks as a sum of outer products. **Personalization on Multi-Session Chat, Mistral-7B-Instruct: F1 40.2 vs 66.0 with the full conversation in the prompt and 8.1 closed-book**, at about 4x less compute and memory than prompting. The only measurement I found of "conversation into weights, then answer in a new session" on MSC, and it sits 26 F1 below the in-context ceiling at 7B.
- **Doc-to-LoRA / Text-to-LoRA** (Sakana, [project page](https://pub.sakana.ai/doc-to-lora/), [arXiv 2602.15902](https://www.alphaxiv.org/abs/2602.15902), [code](https://github.com/SakanaAI/doc-to-lora)): a ~309M Perceiver hypernetwork writes rank-8 MLP LoRAs for **Gemma-2-2b-it**, trained with the context-distillation objective. SQuAD: 83.5% of the full-context upper bound; long-context QA: 85% relative in under a second vs 90% for per-document context distillation taking 40 s; needle recall "near-perfect up to ~40K tokens" against an 8K native window; under 50 MB of memory. Text-to-LoRA targets Mistral-7B-Instruct (rank 8, ~3.4M adapter parameters).
- **Parametric RAG / DyPRAG** ([arXiv 2501.15915](https://arxiv.org/pdf/2501.15915), [arXiv 2503.23895](https://arxiv.org/html/2503.23895v4)): documents are augmented into paraphrases and QA pairs and encoded as LoRAs; DyPRAG at **Llama-3.2-1B** averages 27.57%, 1.06 points over PRAG and 0.58 over standard RAG (search summary).
- **On-device personalization hypernetwork** ([arXiv 2609.24979](https://arxiv.org/html/2609.24979), September 2026): **Gemma-3-1B-IT** with a 14.1M-parameter hypernetwork producing rank-4 LoRAs from user context; beats both per-user PEFT and in-context learning on LongLaMP generation.
- Hypernetwork editors at small scale exist only as KE and MEND (section 1.4). For Planck, a generator bigger than the model it serves (309M for a 20M Planck) makes no sense under a total-parameter headline; a tiny generator is untested.

### 1.7 Test-time training layers

- **TTT-Linear / TTT-MLP** (ICML 2025, [arXiv 2407.04620](https://arxiv.org/abs/2407.04620)): hidden state is a model updated by a self-supervised gradient step per token; **125M to 1.3B**.
- **Titans** ([arXiv 2501.00663](https://arxiv.org/html/2501.00663)): neural long-term memory learned at test time; 170M-760M (context lane). Its gains are beyond 8K tokens, irrelevant inside a 2-4K chat window (`../lanes/context.md`, and the verify file corrected which rows are TTT).
- **Atlas** ([arXiv 2505.23735](https://arxiv.org/abs/2505.23735)): 340M, 400M, 790M, 1.3B on 15B-100B tokens. **HOPE / Nested Learning** ([arXiv 2512.24695](https://arxiv.org/abs/2512.24695)): 340M, 760M, 1.3B; a "continuum memory system" of MLP blocks updated at different frequencies. **TTT-E2E** ([arXiv 2512.23675](https://arxiv.org/abs/2512.23675)): 3B main results, 760M ablations; compresses the context into weights by next-token prediction at test time, 2.7x faster than full attention at 128K.
- **"Do Language Models Need Sleep? Offline Recurrence"** ([arXiv 2605.26099](https://arxiv.org/html/2605.26099v2), May 2026): before clearing the KV cache, run N offline recurrent passes and write the context into SSM fast weights; toy 4-layer d = 256 hybrids plus Ouro 1.4B and Jet-Nemotron 2B. Within one inference instance, not across sessions.
- **Gap:** every one of these resets its fast state at the end of a sequence. Saving it per user and reloading it next session has not been measured by anyone I found (the persistent-memory-adapter paper in 1.1 is the nearest).

### 1.8 Self-updating memory pools

- **MemoryLLM** ([arXiv 2402.04624](https://arxiv.org/pdf/2402.04624)): Llama2-7B plus a 1B-parameter pool of memory tokens in every layer, updated by forward passes; "no degradation ... after nearly a million updates", but knowledge is retained only up to about 16-20K tokens. **M+** ([arXiv 2502.00592](https://arxiv.org/abs/2502.00592), ICML 2025): adds a long-term store and a co-trained retriever, extending retention "from under 20k to over 160k tokens". Both 7B-8B; a 1B pool is 50x Planck's whole budget.

### 1.9 Keyed sparse stores: the closest prior art

- **User as Engram** (Bojie Li, [arXiv 2606.19172](https://arxiv.org/abs/2606.19172), June 17 2026, single-author preprint, text read): Engram layers (hash-keyed n-gram memory tables, 50K x 256 = 51.2M parameters) grafted into Karpathy's nanochat, pretrained at **178M (d8), 339M, 625M and 1.22B total**. A user's facts become overrides of a few table rows at the addresses of the trigger n-gram; one shared LoRA carries the reasoning skill.
  - Contamination: per-user rows change bits per byte on unrelated text by +0.00005 (0 of 20 users worse) vs +1.784 for a per-user LoRA; "~33,000x" smaller over 3 seeds.
  - Recall at the smallest size (d8, 178M, Table 5): 15-step gradient write gives top-1 on 6 of 16 facts, top-5 on 7 of 16; the closed-form write gives 1 of 16.
  - Density: on d12 (339M), joint optimization keeps top-5 above 90% to about 300 facts per user and falls to 35% top-1 at 1,000 facts; the ceiling is "gradient interference among the co-active rows", not capacity or hash collisions.
  - Serving: 100 users x 100 facts on d12@1280: 62% top-1 / 96% top-5, 0.03 ms to apply a user's overrides, 88 KB per user vs 14.2 MB for a rank-64 LoRA; cross-user leak 6.1%, attributed to gold-value coincidence.
  - Corrections: when two facts share the same trigger words, "the last write wins". No explicit Biscuit-to-Pickles test.
  - Limits: base LMs with completion-format prompts, not a chat model; "recall without reasoning" (indirect questions over stored facts fail for both LoRA and Engram); requires an Engram-pretrained base; the 51.2M table alone is 2.5x a 20M Planck.
- **Sparse memory fine-tuning** (Lin et al., [arXiv 2510.15103](https://arxiv.org/html/2510.15103), October 2025): a **1.3B** model with one memory layer (1M slots, top-32 per head); new facts update only the top-t slots ranked by TF-IDF against slots used on background data. After learning 1,000 TriviaQA facts, NaturalQuestions F1 drops 89% with full fine-tuning, 71% with LoRA, 11% with sparse memory fine-tuning. The best "forget less" number in this survey, at 1.3B.

### 1.10 Consolidation: context distillation, self-study, "sleep"

- **What to train on, measured:**
  - Physics of LMs 3.1 ([arXiv 2309.14316](https://arxiv.org/html/2309.14316), **GPT-2 12 layers x 768, 124M**, fetched summary): biographies in one phrasing are memorized but not extractable by questions for held-out people (9.7%, and "essentially 0%" in the strict setting); 5 paraphrases plus sentence permutation give 96.6%.
  - SEAL ([arXiv 2506.10943](https://arxiv.org/html/2506.10943), Qwen2.5-7B, LoRA, one passage): no update 32.7%, train on passage 33.5%, passage plus the model's own implications 39.7%, plus GPT-4.1 implications 46.3%, SEAL 47.0%.
  - Data Quality over Capacity ([arXiv 2607.21861](https://arxiv.org/abs/2607.21861), July 2026, 4-bit Gemma-4-e4b): curating QA answers to 1-6-word canonical spans took closed-book accuracy from 57.7% to 85.7% on 15 documents, beating BM25-RAG (58.9%) and a gold-chunk oracle (65.6%); "capacity itself is a hard gate below which no data intervention helps".
  - Cartridges ([arXiv 2506.06266](https://arxiv.org/abs/2506.06266)): a trained KV prefix per corpus; next-token training on the corpus "is not competitive with ICL", self-study (synthetic conversations about the corpus plus context distillation) matches it at 38.6x less memory and 26.4x more throughput; cartridges compose at inference. Model sizes not verified here.
- **Self-distillation of context into weights:** SELF-PARAM ([arXiv 2410.00487](https://arxiv.org/html/2410.00487v2), ICLR 2025; OpenLLaMA-3B, Mistral-7B, Llama-3-8B) minimizes KL between the model with the context and the model without it on generated questions; after 20 sequential injections QA-F1 falls to about 0.3 vs about 0.14 without injection (fetched summary). Prompt distillation (Kujanpaa et al., [arXiv 2412.14964](https://arxiv.org/abs/2412.14964); Qwen2.5-3B, Llama-3-8B, Qwen2.5-14B) nears RAG.
- **"Sleep":** "Language Models Need Sleep" ([arXiv 2606.03979](https://arxiv.org/abs/2606.03979), June 2026, smallest Llama-3.2-1B): consolidation by distillation into a newly added low-rank expert plus RL "dreaming"; SQuAD knowledge incorporation 48.9% vs SEAL 46.7%. SCM ([arXiv 2604.20943](https://arxiv.org/pdf/2604.20943)) replays episodes with Hebbian strengthening and downscaling (not read in detail).
- **Forgetting dynamics of injected facts** (Chang et al., [arXiv 2406.11813](https://arxiv.org/abs/2406.11813), OLMo 1B and 7B): each exposure gives a small log-probability gain that then decays as a power law with further training; larger batches forget less. Relevant to how long a session's update survives later sessions' updates.

### 1.11 Memory tokens written by gradient (in-scale)

- **GradMem** ([arXiv 2603.13875](https://arxiv.org/html/2603.13875), March 2026, fetched summary): write a context into m memory tokens (m = 8 for key-value tasks, 32 for SQuAD) by K = 1-5 gradient steps on a reconstruction loss, weights frozen, then answer with the context removed. Pretrained **GPT-2 124M, Pythia-160M, Mamba-130M** (plus Llama-3.2-1B/3B in an appendix). SQuAD-short EM: 54.9 (GradMem) vs 42.6 (forward-only RMT) vs 64.2 (full context, GPT-2). 96 key-value pairs on a small 4-layer d = 128 model: 32.6% at K = 1, 88.4% at K = 5, 12.9% forward-only. The base must be meta-trained through the write (second-order MAML); the ablation without it collapses. This is the strongest in-scale evidence that a sub-200M model can hold a context in a small per-context parameter set, and cross-user bleed is zero by construction, because the tokens are only loaded for their owner.
- **Persona embeddings** (Li et al. 2016, [arXiv 1603.06155](https://arxiv.org/abs/1603.06155)): a learned per-speaker embedding in an LSTM seq2seq chat model improved perplexity, BLEU and human-judged speaker consistency. A per-user vector in a small chat model has existed for ten years; it encoded the model's persona, not memory of the user.

### 1.12 Bleed in in-context memory too

- **PersistBench / over-personalization** ([arXiv 2608.08300](https://arxiv.org/html/2608.08300v1), August 2026): with memories in context, frontier and 70B+ open models fail the cross-domain leakage subset 56.2% of the time on average and memory-induced sycophancy 96.5%; domain-partitioned memory cuts leakage by 8.8%. So "does the Pickles fact leak into unrelated answers" is not a weights-only problem, and the experiment must measure it for the notes-in-context baseline too.
- **Supersede** ([arXiv 2606.27472](https://arxiv.org/html/2606.27472v1), June 2026): with a bounded self-maintained notes field instead of full history, LongMemEval knowledge-update accuracy drops 92% to 77% (gpt-5.4) and 82% to 63% (gpt-4.1-mini); GRPO on Qwen2.5-3B lifts held-out accuracy 9.0% to 16.7%. Corrections are hard for notes-based memory as well, even at frontier scale.

---

## 2. Has it been tested at this size?

Direct answer: **no.** I found no sub-150M chat model with per-user long-term memory in its weights whose recall and bleed were measured across sessions. What exists at or near the size:

| Work | Size | What was stored | Across sessions? | Chat model? | Bleed measured? | Correction tested? |
|---|---|---|---|---|---|---|
| User as Engram ([2606.19172](https://arxiv.org/abs/2606.19172)) | 178M smallest (339M for most results) | per-user facts as hashed table rows | yes (stored per user) | no (base LM, completion prompts) | yes (unrelated-text bpb, cross-user leak) | only "last write wins" on collisions |
| Trained persistent memory adapters ([2603.22329](https://arxiv.org/abs/2603.22329)) | GPT-2 124M, frozen | latent memory state (Hebbian, slots) | yes (LoCoMo) | no | no | no |
| GradMem ([2603.13875](https://arxiv.org/abs/2603.13875)) | GPT-2 124M, Pythia-160M, Mamba-130M | a context, in 8-32 memory tokens | no (per context) | no | not needed (per context) | no |
| MAC / CaMeLS ([MAC](https://proceedings.neurips.cc/paper_files/paper/2024/file/eaf956b52bae51fbf387b8be4cc3ce18-Paper-Conference.pdf), [CaMeLS](https://arxiv.org/abs/2305.15076)) | DistilGPT2 82M | news documents, by online fine-tuning | stream of documents | no | unrelated QA only | no |
| MEND / KE / ENN ([2110.11309](https://arxiv.org/abs/2110.11309)) | 82M-139M | single edits | no | no | drawdown | no |
| GRACE ([2211.11031](https://arxiv.org/abs/2211.11031)) | T5-small 60M | 1,000 sequential QA edits | yes (streaming) | no | test retention | conflicting keys split |
| SERAC ([2206.06520](https://arxiv.org/abs/2206.06520)) | BlenderBot-90M | sentiment edits, outside the weights | n/a | yes | drawdown | no |
| Should We Really Edit ([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/370fa2e691f57eb319bc263a07dad4a5-Paper-Conference.pdf)) | Pythia-160M | CounterFact edits | sequential | no | general benchmarks | no |
| Hewitt et al. ([2402.06155](https://arxiv.org/abs/2402.06155)) | Pythia-70M up | canonical-example edits | no | no | loss-increase cap | no |
| Physics of LMs 3.1 ([2309.14316](https://arxiv.org/abs/2309.14316)) | 124M | synthetic biographies, in pretraining | n/a | no | n/a | no |

MAC's own numbers at 82M show why this is a real question: online fine-tuning on a document stream gives StreamingQA F1 3.76 (uniform), 5.79 (CaMeLS), 10.18 (MAC), vs 13.54-21.79 for the same methods at LLaMA-2-7B (MAC Table 1, table read). Those are documents full of knowledge the small model lacks; user facts in a chat are much simpler, which is why the Planck test is not already answered by them.

---

## 3. What this implies for the design (my synthesis)

1. **Baseline first.** The notes block in context (idea #7) is the bar. For weight memory to earn a place in Planck it must either match notes-in-context at equal bleed, or add recall when notes overflow the window.
2. **Train on questions, not on the transcript.** Render each note into many question phrasings with short answers, add "no" negatives and same-type-other-person items, and anchor unrelated outputs. Or distill from the same model with the notes in context. Raw-transcript next-token loss is the arm most likely to fail (Physics 3.1, PLUM, SEAL, Cartridges all say so at their scales).
3. **Put the per-user parameters where they cannot leak.** Per-user adapters loaded only for their user give zero cross-user bleed by construction; within-user bleed (the sister's dog, "suggest a puppy name") still has to be measured. A shared model that absorbs everyone's facts is the worst case (FACTPROP, same-subject perturbation), and Planck should not do that.
4. **Resolve corrections in text, then rebuild.** The notes store applies latest-value-wins; the adapter is rebuilt from the current notes after each session, instead of patching the previous adapter. Measure both.
5. **Prefer MLP placement.** At 2M-8M bases, a fixed LoRA budget stores 2.43 bits per parameter in MLP directions vs 1.30 in attention ([arXiv 2607.21351](https://arxiv.org/abs/2607.21351), verified in `../followup/bits.verify.md`); Engram edits fail "if written into the wrong layer"; Doc-to-LoRA targets MLPs.
6. **Watch the in-context skill.** The most Planck-specific risk is that a weight update teaches the model to answer from weights when the context says otherwise (the stale value winning), which is the correction failure the probe already found in context. The experiment includes a "context beats weights" item for that reason.

---

## 4. Experiment design: the cheapest decisive test on the Mac

**Question it decides:** at 135M, can a per-user weight update built from a session's notes give empty-context recall close to the notes-in-context baseline, without bleed or general damage, and can it take a correction?

### 4.1 Model and format

- **SmolLM2-135M-Instruct** (134.5M total, d = 576, 30 layers, MLP width 1,536, 9 heads / 3 KV heads, 49,152 tied vocabulary; [config](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/config.json)). Already in `~/.cache/huggingface/hub`, so no download. It is the closest open model to Planck's upper end and the probe already characterized it.
- **Format pre-check (10 minutes):** run the notes-in-context control through both the chat template and the plain `User:/Assistant:` transcript; the probe found 0.15 vs 0.70 recall for the two. Use whichever gives the higher in-context recall for every arm, because a weight arm cannot be judged against a broken ceiling.
- **Later:** rerun the same harness on Planck checkpoints (20M, 60M) when they exist; the likelihood tier works on any model. Falcon-H1-Tiny-90M has better in-context recall (1.00 retrieval in the probe) but its Mamba fallback is slow and memory-hungry on MPS, so it belongs on the 5070.

### 4.2 Data (template-generated, no teacher needed)

- **8 synthetic users x 8 slots:** dog's name, sister's name, home city, job, favourite food, car, allergy, birthday month. Values drawn from pools the base does not favour (check each value's base-rate log-probability for its slot question and reject the top 5% most likely, so a hit cannot be a guess).
- **Session 1 transcript:** 10-14 turns of small talk with the 8 facts stated once each in natural first-person lines, plus the notes extracted by rule (`user.dog_name: Biscuit`).
- **Session 2 transcript:** corrections for 2 slots ("actually his name is Pickles, not Biscuit"), 2 new facts (so 10 facts total), and small talk. Notes updated with latest-value-wins.
- **Question phrasings per slot:** 10 train phrasings ("What's my dog's name?", "Remind me what I called my dog"), and 3 held-out phrasings used only in evaluation: one direct but unseen wording, one indirect ("Start a birthday card for my dog, using his name"), one reverse ("Who is Pickles?", gold: your dog).
- **Negatives (PLUM-style), per user:** 8 questions about unstated slots ("What's my cat's name?", gold: "You haven't told me") and 8 same-slot-other-person items ("My friend Sam's dog is Rex. What's Sam's dog called?", gold: Rex).
- **Anchor set (shared):** 64 unrelated chat prompts from the cached OASST data (`research/capacity_probe/oasst_threads.json`) with the base model's top-50 next-token logits precomputed once.

### 4.3 Arms

| arm | per-user update trained on | notes |
|---|---|---|
| C0 no memory | nothing | floor: base-rate guessing |
| C1 notes in context | nothing; notes block in the system prompt | the bar (idea #7) |
| C2 transcript in context | nothing; session transcripts in context | second bar |
| A1 raw transcript | next-token loss on the session transcript (user and assistant tokens) | Max's literal idea |
| A2 notes only | next-token loss on the notes lines, paraphrased 5 ways | Physics 3.1 augmentation |
| A3 QA + negatives | chat-format QA on the 10 train phrasings per slot plus the 16 negatives, loss on answer tokens | PLUM recipe, shrunk |
| A4 = A3 + anchor | A3 plus KL to the base's precomputed logits on the anchor set (weight 1.0) | locality term |
| A5 self-distillation | teacher = the same frozen model with the notes in context, greedy answers to the train phrasings and negatives; student = LoRA without notes, full-vocabulary KL on answer tokens | SELF-PARAM / Doc-to-LoRA objective |

**After session 2, two variants of the best two arms:** *stack* (continue training the session-1 adapter on session-2 items only) and *rebuild* (fresh adapter trained on the full current notes). This is the correction test.

### 4.4 Hyperparameters

- LoRA rank 8 on gate, up and down projections of all 30 layers: 30 x 3 x 8 x (576 + 1,536) = **1,520,640 parameters, 3.0 MB per user in bf16** (my arithmetic). Ablation later: rank 1 (190,080 parameters, 380 KB) and attention-only placement (115,200 x r parameters).
- alpha 16, AdamW, lr picked in stage 0 from {1e-4, 3e-4, 1e-3} on 2 users (PLUM-style runs used 10 epochs; 20 broke PLUM's 8B model), batch 16, 10 epochs over the session's items, answer-token loss up-weighted as in PLUM, fp32 master weights (bf16 autocast only if it is faster on MPS), gradient clip 1.0.
- Seeds: 3 (LoRA init, data order, which value pool draw).

### 4.5 Evaluation (fresh session, empty context unless stated)

- **Recall** (primary), held-out phrasings only: (a) likelihood margin, log p(gold) minus the best foil, with the answer pre-started ("Your dog's name is"), foils = the stale pre-correction value, another user's value for the same slot, a value from this user's other slots, and the base model's most likely value; (b) greedy generation graded with the probe's user-fact grader logic (gold outside a greeting, no deflection, not claimed as the assistant's own).
- **Bleed:**
  - B1 other person, same slot ("What do you think Sam's dog is called?" with no Sam fact given): rate of emitting this user's value, vs the C0 base rate.
  - B2 unstated slot ("What's my cat's name?"): rate of emitting any stored value instead of saying it was not given.
  - B3 open generation ("Suggest five names for a puppy", 10 samples at temperature 0.8): frequency of the stored dog name vs base.
  - B4 the assistant itself ("What's your dog's name?"): claims the user's dog.
  - B5 cross-user: user A's adapter answering user B's recall questions emits A's values.
  - B6 unrelated knowledge: likelihood margin on 20 general chat facts ("capital of France") unchanged.
- **Context beats weights:** in the fresh session the user says "I just adopted a new dog, Rex" and then asks the dog's name: gold Rex, failure the stored name.
- **Correction:** after session 2, margin Pickles over Biscuit, greedy pass rate, stale-value emission; plus retention of the 6 untouched session-1 facts.
- **General ability:** bits per byte on 100 held-out OASST conversations (relative change); KL from the base on the anchor prompts (the early-warning signal from arXiv 2606.27634); and a 10-conversation subset of the probe battery's in-context recall and correction items (does the adapter damage in-context memory?).
- **Measurement mutation tests** (Max's rule, run in stage 0): an adapter trained on a different user must score near C0 on recall; a deliberately leaky adapter (trained to answer every "dog name" question with the stored name, including other people's) must turn B1 and B3 red; a session-1-only adapter must fail the correction item; an over-trained adapter (lr 1e-2, 100 epochs) must trip the bits-per-byte check.

### 4.6 Pre-registered pass and kill rules

- Pass: best weight arm's held-out recall at least 0.8x the C1 notes-in-context arm and at least 60% absolute; B1, B3, B4 within 5 points of C0; correction pass at least 80% with stale emission at most 10%; context-beats-weights at least 80%; bits per byte within 1% relative; in-context battery drop at most 5 points.
- Kill: best weight arm under 0.5x C1, or any bleed item more than 10 points over C0 in every arm. Record the negative result in the ledger and stop (the "stop building when the test undercuts the premise" rule).

### 4.7 Wall-clock (estimates, not measured)

- LoRA training costs about 4N FLOPs per token (forward plus activation gradients, no frozen-weight gradients) = 5.4e8 FLOPs per token for 134.5M parameters (my arithmetic). The first run measured 1.5-2.1 effective TFLOPS for full training on this Mac at T = 1,024 (`../followup/loop.md`), but short sequences and batch 16 will be less efficient, so assume **1,000-4,000 tokens/s**.
- One A3-style update: about 128 items x ~48 tokens x 10 epochs = ~61K tokens, so **15-60 s per update** (my arithmetic).
- Stage 0 (format check, lr pick, grader mutation tests, pipeline on 2 users): ~1 h.
- Stage 1 (session 1 only, 5 training arms, 8 users, 3 seeds = 120 updates): 0.5-2 h training plus ~1 h of likelihood evaluation.
- Stage 2 (session 2, stack vs rebuild for the 2 best arms, 8 users, 3 seeds = 96 updates, plus generation-graded evaluation on a subset): 1.5-3 h.
- **Total about 4-8 h, one overnight.** The RTX 5070 would be about 10x faster by the loop lane's estimate, under an hour.
- Stage 3 if stage 1 passes: capacity and stacking sweep on the best arm (facts per user 1 / 4 / 16 / 64; sessions 1 / 2 / 5 / 10), rank and placement ablations: another 3-5 h.

### 4.8 Mac safety (from the crash history)

One Python process, one model loaded; the runner refuses to start if `pgrep` finds another torch or MLX process, and it never runs beside the Gemma teacher. 135M in fp32 is 0.54 GB of weights; LoRA optimizer state is ~18 MB (1.52M x 12 bytes); activations at batch 16 x 128 tokens are small, so the process should stay under about 3 GB (estimate). Cap it with `torch.mps.set_per_process_memory_fraction(0.25)`. Adapters are saved to disk and loaded one at a time for evaluation; never hold several model copies.

---

## 5. Ledger ideas (untested at 10M-150M unless stated)

| # | Idea | Why it could work | Cheapest test | Novelty |
|---|---|---|---|---|
| L1 | Per-user LoRA from notes rendered as QA + negatives + KL anchor (the section 4 experiment) | PLUM at 8B reached 81.5% vs 89.5% ceiling; Physics 3.1 says augmentation makes facts extractable at 124M | Section 4 stage 1 | tweak-of-known |
| L2 | Rebuild-from-notes after every session ("sleep" as rebuild) vs stacking patches | edited facts are fragile under later updates; same-subject edits interfere; notes already resolve corrections | Section 4 stage 2 | untested |
| L3 | Per-user self-distillation (teacher = same tiny model with notes in context) | Doc-to-LoRA's oracle context distillation reached 90% of full context at 2B; SELF-PARAM, Cartridges | Arm A5 | tweak-of-known |
| L4 | Per-user memory tokens (32 x 576 = 18,432 parameters, 37 KB in bf16) written by a few gradient steps; the "cartridge" of a user | GradMem at 124M: EM 54.9 vs 64.2 full context; zero cross-user bleed by construction; 80x smaller than a rank-8 LoRA | add as arm A6 in section 4, same data as A3 | known-apply (GradMem) in a new setting (chat, users) |
| L5 | Meta-train Planck so a few gradient steps on notes write facts cleanly (first-order Reptile or MAML on note-to-question episodes during SFT) | ENN and MEND beat plain fine-tuning 3x at 82M; GradMem's write fails without meta-learning | after L1: add 5% meta-episodes to a 20M Planck SFT run, compare L1 recall with and without | untested for chat |
| L6 | Delta-rule fast-weight layer whose state is saved per user at session end and reloaded next session | the delta rule overwrites (built for corrections); Hebbian persistent memory at 124M retained 7-18% on LoCoMo | needs a Planck variant with one DeltaNet layer; test on synthetic multi-session recall at 5M-20M | untested |
| L7 | Small addressed memory table as the only per-user write target (Engram-style hashed rows or product keys), sized to the budget (e.g. 4K x 128 = 0.5M) | +0.00005 bpb contamination vs +1.784 for LoRA at 1.22B; "last write wins" gives corrections | at the 60M-150M points only; a 51.2M table does not fit a 20M budget | known at 178M+, untested below |
| L8 | Same-slot-other-person negatives and "unstated slot" negatives in the update data | PLUM: without negatives the model says yes to everything; FACTPROP: connected entities spread errors | ablate the negatives inside A3 | tweak-of-known |
| L9 | MLP-only vs attention-only vs both for the per-user adapter, and rank 1 vs 8 vs 32 | MLP stores 2.43 vs 1.30 bits per LoRA parameter at 2M-8M; capacity is 1,000x+ what 8 facts need | stage 3 ablation | tweak-of-known |
| L10 | Hybrid: notes block in context and a per-user adapter together; measure whether the adapter adds recall when the notes are truncated to fit | Engram overtakes retrieval past ~100 facts per user; Planck's window fills at roughly that size | evaluate C1 with notes cut to 4 of 16 facts, with and without the adapter | untested |
| L11 | KL-from-base as a live guard: stop or roll back a user's update when KL on the anchor prompts passes a threshold | KL predicted collapse at 0.8B-1B (Gemma KL 1.623) | log it in section 4 and check it against the bits-per-byte result | known-apply |
| L12 | Sparse update: only the top-t MLP rows (or LoRA rows) most specific to the note tokens, ranked by TF-IDF against background usage | sparse memory fine-tuning at 1.3B: 11% vs 71% (LoRA) forgetting | mask the LoRA update to top-t rows by gradient-activation specificity, compare with A4 | untested at small scale |
| L13 | Reverse and indirect recall as their own scores ("Who is Pickles?", a birthday card for the dog) | reversal curse; Engram and per-user adapters show "recall without reasoning" | part of the held-out phrasings in section 4 | known-apply |

---

## 6. Open questions

1. Does a per-user weight update damage the in-context memory skill that Planck exists to have? Engram found a per-user LoRA disrupted a base LM's completions (85% of users worse on indirect recall) while instruction-tuned 3B-8B bases absorbed it; at 135M the "reasoning skill" is thin, so the damage may be visible. The context-beats-weights item and the in-context battery subset test this.
2. Will SmolLM2-135M's deflection habit ("I don't have access to personal information", 24% of its final user-fact turns in the probe) block recall from weights the same way it blocks recall from context, and does the QA arm fix it or just move it?
3. Is the first-person subject too generic to edit locally at all? If B1 and B3 bleed in every arm, the answer is that personal facts need an addressed store (L4, L6, L7), not a weight delta.
4. Where is the crossover at which weight memory beats notes in context for Planck's 2K window? Engram's crossover against retrieval is about 100 facts per user; a notes block of 100 facts is on the order of 1-2K tokens (my arithmetic), which is Planck's whole window.
5. How long does a session's update survive later sessions' updates under stacking (power-law forgetting at OLMo 1B-7B, untested at 135M), and does rebuilding from notes make the question moot?
6. Privacy and deletion: a per-user adapter or memory-token file can be deleted outright; anything merged into shared weights needs unlearning. Rebuild-from-notes keeps deletion trivial. Worth stating in the writeup.
7. Numbers to verify before relying on them: the MEMIT collapse point (~1,400 edits, search summary only), the GradMem ablation, the Physics 3.1 single-phrasing number (9.7% vs "essentially 0%"), Cartridges' model sizes, and the Generative Adapter MSC figures (fetched summaries).
