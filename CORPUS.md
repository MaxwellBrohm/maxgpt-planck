# Planck corpus design v1

Date: 2026-09-24. Status: this is a design only. Nothing gets built until Max makes the D8 rulings (Fri 9/25, section 1.4) and the teacher pilot runs (Oct 5-8).

Inputs:
- the four research tracks verified today (sources, selection, synthetic scale, token budget), with every fact-check correction applied;
- `PLAN.md`, `LEDGER.md`, `research/REPORT.md`, and the research lanes and follow-ups with their verify files.

No model was loaded, run or trained, and no dataset was downloaded.

**Markers**
- **(est.)**: my own arithmetic from the cited inputs, or my judgment. Throughput, yield and timing figures for Max's machines have not been measured and could be off by 2x.
- **unsourced**: I found no source.
- **[>=1B only]**: the evidence exists only at 1B parameters or more.

**Token units**
- Common Pile counts are in the Comma tokenizer ([Comma card](https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset)). FineWeb counts are GPT-2 tokens.
- Planck's in-domain 8k BPE measured 3.83 bytes/token on OASST, against 4.36 for SmolLM2's 49k tokenizer (`research/followup/tokenizer.md`). So the same chat text is about 14% more Planck tokens. On general prose the gap is probably larger (unmeasured).
- Read every quoted count as at least 1.14x that many Planck tokens.

---

## 0. The short version

1. **Token budgets.**

   | size | base budget (tok/param) |
   |---|---|
   | 5M | 2,000 |
   | 10M | 2,000, rising to 5,000 as the first extension |
   | 20M | 1,000 |
   | 30M | 1,000 |
   | 60M | 500 |
   | 150M | 500; 670 only if the Titans reach about 67k tok/s |

   - At the central rate this uses about 1,620 of the roughly 1,700 5070-hours between Dec 4 and Feb 21 (est.).
   - Any spare time goes to an extension queue ranked by measured gain per GPU-hour.

2. **Small gains are worth buying, but only after better uses of the same hour.**
   - One doubling of tokens removes 15-22% of the remaining data-limited loss. At these budgets that is about 0.05-0.10 nats, roughly a fifth to a quarter of the step to the next curve size (est., from three published fits that overstate gains at extreme ratios).
   - Tokens cannot stand in for size. By the same fits, a 30M needs 22-630x its 500 tok/param budget to reach a 60M's loss.
   - Only 3M-5M models reach 20,000 tok/param cheaply. A 10M at 20,000 with 3 seeds would take the 5070's entire curve window.

3. **The corpus has five parts:**
   - verified skill chat, the backbone for the target skills;
   - grounded rephrasings, which turn fact-dense human text into practice at reading from context;
   - human dialogue, which is small and serves as a realism and out-of-distribution anchor;
   - simple narrative;
   - plain prose chosen by Planck's own classifiers.

   Nothing from Ultra's build is reused. FineWeb-Edu, code, math, science, law and bulk Wikipedia are out.

4. **Clean human chat is scarce and clean human prose is not.**
   - Chat-shaped human dialogue that passes D8 comes to about 45-70M tokens. Grey-license sets would add about 10-20M.
   - The openly licensed general-language part of Common Pile v0.1 is about 100B Comma tokens, counting Library of Congress books.
   - After the date gate and Planck's selection, the strictly open prose core is about 15-25B (est.). That covers every base budget at 4 repeats or fewer.
   - Only the 150M extension and the 20,000 tok/param probe need more: either about 6-8 repeats, or a pre-2023 web bucket if Max accepts Reading A (ruling Q1).

5. **Selection:**
   - provenance and date gates;
   - dedup (global exact, plus a global near-dedup pass over the buckets that will be upsampled);
   - DSIR weights toward human conversation;
   - four fastText classifiers trained on about 60k Gemma 4 12B labels: conversational usefulness, fact density, plainness, cleanliness;
   - graded upsampling curves set per model size.

   Mix weights come from a swarm of about 55 5M models regressed on bits per byte, confirmed at 20M against the hand mix.

6. **Synthetic data:**
   - Pick the renderer by accepted Planck tokens per 5070-hour at realistic request lengths. Ministral 3 8B is likely 2.5-3.5x Gemma 4 12B on this card (est.).
   - Target 150-300M teacher renders (R0) plus 2 re-verified paraphrases of each (R1).
   - Re-draw slot values on every pass (R2).
   - Add 100-150M grounded rephrasings (RH) by the freeze.
   - That gives 0.5-0.9B surface-unique chat tokens at Dec 4, and 1-2B later only under a pre-registered growth rule.

7. **PLAN fixes this implies:**
   - P-186 as written costs about 1,150-2,380 5070-hours, not 40-80 (redesign in 5.5).
   - Remove "readability-filtered FineWeb-Edu" from the mix and from tokenizer training.
   - The teacher gate should measure at realistic request lengths.
   - The PC needs about 250-300 GB free, not 150.

---

## 1. Principles

### 1.1 What the corpus is for

The corpus has two jobs:
- **Teach the language.** Fluent, plain, everyday English, including spoken and conversational registers, and coherent narrative.
- **Teach conversational mechanics over the model's own context:**
  - recall across turns and reference resolution;
  - updating after corrections;
  - whose-is-whose binding;
  - holding instructions;
  - no role capture and no loops;
  - reading answers from a supplied passage or lookup.

The thesis is that skill lives in the weights and knowledge lives outside them.

Why spend little capacity on facts:
- Models store about 2 bits of knowledge per parameter after about 1,000 exposures, and about 1 bit at 100 ([Physics of LMs 3.3](https://arxiv.org/abs/2404.05405), verified in `research/lanes/data.verify.md`).
- A 110M model trained with pruning that flattens fact frequency memorized 1.3x more facts and matched a 1.3B trained on everything ([Cram Less to Fit More](https://arxiv.org/abs/2604.08519)). Fact-dense data overflows small models.
- So the corpus converts fact-dense text into grounded dialogue, where the fact sits in the context (section 4.6), instead of storing it.

Why density comes before duration:
- Below a critical mixing ratio, a model memorizes almost nothing from a data type, however long it trains, and that ratio follows a power law in model size ([Gu et al.](https://arxiv.org/abs/2505.18091); shown for synthetic biographies on Pythia-architecture models from 14M to 6.9B).
- Pythia-70M is still near chance (53.5%) on implicit entity tracking after 300B generic tokens, about 4,300 tok/param ([Drozdz and Heilbron](https://arxiv.org/abs/2608.18083)).
- Applying this to conversational skills is an analogy, not a measurement. Still, it makes the share of skill-dense chat the first lever and the token count the third, after size.

Why small sizes should be selective, with repetition priced in:
- In a data-scarce setting where every filtered set had to repeat, 15M models never preferred the unfiltered pool, even at 100B training tokens ([A Bitter Lesson for Data Filtering](https://arxiv.org/html/2605.19407v1)).
- Crossing points came earlier as models grew: 80M already crossed on the smallest pool, and 330M+ eventually preferred unfiltered data.
- Planck draws from a far larger pool, so filtering does not force repetition at base budgets. What the paper supports is letting small models repeat their best buckets longer, which is what section 3.5 does.

### 1.2 What it deliberately leaves out

**Out:**
- code, math, science, law and patents (about 373B Comma tokens of the Common Pile);
- reasoning traces and think-mode text;
- most encyclopedia text;
- knowledge-dense educational web (FineWeb-Edu and its derivatives);
- all closed-model text;
- anything NC, ND, of unknown license, or under restrictive terms;
- Reddit and film subtitles;
- OCR text that a restricted model rewrote.

**No reuse of Ultra.** None of Ultra's shards, filters or mix weights are reused.
- Two of Ultra's sources reappear only as slices re-selected from scratch: Wikimedia (Simple Wikipedia, Wikibooks, talk pages, and a fictional-KB lookup format) and Mixtral-written Cosmopedia (story formats only, as an A/B arm).
- FineWeb-Edu, which was 55% of Ultra, is out.

**Kept on purpose:**
- Everyday common-sense knowledge carried by plain prose. Conversation needs it, and there is no clean way to remove it.
- Fact-dense text as seeds for grounded dialogues (section 4.6).

### 1.3 D8 and provenance

**D8 as locked:**
- Human-written text under an open license is allowed.
- Every model-written token must come from an open-weight model under Apache-2.0 or an equally permissive license.
- Closed-model text is for baselines only.
- Claude writes code, prompts, rubrics, the gold eval sets and the human-test scripts. Nothing Claude writes becomes training text.

**Every training document carries a provenance record** with these fields:
- source id and version;
- license, per document where the source records one (Common Pile subsets keep it in `metadata.license`; the Hub card is not enough);
- author type: human, or which model under which license;
- who wrote the user turns, for dialogue;
- creation or dump date;
- the generator's prompt hash, for renders;
- a ShareAlike flag;
- decontamination status.

**The traps the fact-checkers found sit exactly in these fields:**
- **The Common Pile DPI subset.** Its include list marks HH-RLHF, HelpSteer and ProsocialDialog as human-written ([include.csv](https://github.com/r-three/common-pile/blob/main/sources/data_provenance/include.csv)). It also marks AgentInstruct-alfworld as crowdsourced, although its trajectories are GPT-4 ([card](https://huggingface.co/datasets/THUDM/AgentInstruct)). It carries DialogSum, which reuses DailyDialog (NC). Several of its license labels are code-repo licenses. It needs an entry-by-entry audit, not a three-name filter.
- **SmolTalk.**
  - v1's constraints, rewrite and summarize subsets were generated by Qwen2.5-72B, which is under the Qwen license ([pipeline](https://github.com/huggingface/smollm/tree/main/text/data/smoltalk)).
  - SmolTalk2's dialogue subsets have Apache answers, but their user turns come from non-Apache sources. Multi-Turn IF, for example, takes its first prompt from Tulu 3 Personas IF ([card](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)).
- **SYNTH.** It includes DeepSeek-Prover generations under the DeepSeek license ([stats](https://datasets-server.huggingface.co/statistics?dataset=PleIAs/SYNTH&config=default&split=train)).
- **Common Corpus.** Parts were OCR-corrected, and sometimes rewritten, by OCRonos, a Llama-3-8B fine-tune ([arXiv 2506.01732](https://arxiv.org/pdf/2506.01732)).
- **AirDialogue.** Its card says CC-BY-NC-4.0 and tags the language as machine-generated ([card](https://huggingface.co/datasets/google/air_dialogue)).
- **No stated data license** for Wizard of Wikipedia, Wizard of Internet, PersonaChat, MSC or BST.
- **CoQA.** Its news passages are CNN text.
- **QuAC** is CC BY-SA 4.0 ([quac.ai](https://quac.ai/)).
- **Post-2022 text in "human" sources** (see the date gate in 3.1).

### 1.4 Rulings Max needs (Fri 9/25, one line each)

| # | question | my recommended default | what it changes |
|---|---|---|---|
| Q1 | Is a dataset-level open license on a web crawl enough (Reading A), or must each document be openly licensed (Reading B)? FineWeb (ODC-BY plus Common Crawl ToU) licenses the compilation; the page text keeps its copyright ([card](https://huggingface.co/datasets/HuggingFaceFW/fineweb)) | The headline core follows Reading B (Common Pile). Pre-Dec-2022 FineWeb is allowed as one extra bucket (W) only if Max says yes, and the swarm sets its weight | Without W, the 150M extension and 20,000 tok/param probes repeat the core 6-8x instead of 4x |
| Q2 | Is text from Apache or MIT open weights released by closed labs allowed (Whisper large-v3, gpt-oss)? | Yes, labeled in the provenance column. It passes by the letter of D8 ([Whisper API](https://huggingface.co/api/models/openai/whisper-large-v3), [gpt-oss API](https://huggingface.co/api/models/openai/gpt-oss-120b)) | Common Pile CC YouTube transcripts (4.7B) and Nemotron-Personas-USA as prompt seeds |
| Q3 | Human first turns from WildChat (ODC-BY), re-answered by an Apache teacher? | No for v1. Later turns react to ChatGPT, users paste its text, and much of it is code or knowledge tasks | Nothing in v1 |
| Q4 | Grey dialogue sets with no stated data license (PersonaChat/ConvAI2, MSC, BST, WoW, WoI)? | Eval and baselines only | Loses about 10-20M tokens that fit the skills closely |
| Q5 | Selectors trained on restricted-model labels (the FineWeb-Edu classifier on Llama-3-70B labels; DCLM positives from GPT-4-written OpenHermes)? | Not used in the headline recipe. This is a D8-spirit call: the FineWeb-Edu classifier is tagged apache-2.0 on the Hub | Planck trains its own classifiers (3.3) |
| Q6 | Date gate for text that might be model-written | Keep documents created before 2022-12-01 wherever the source records a date | Shrinks CCCC, StackExchange, IRC and wiki sources by an unmeasured fraction |
| Q7 | Mac night-shift paraphraser (a 4B-class model under the guard runner with an 8 GB cap, after the current experiment ends) | Yes, but only after the freeze | Adds 20-71M R1 or RH tokens per 100 Mac-hours (est.) |
| Q8 | Released rephrasings of CC BY-SA sources (Wikipedia, StackExchange, QuAC, SGD) | Release them under CC BY-SA, as the license requires | Affects the release terms in section 8 of PLAN |

---

## 2. Components

### 2.1 The mix

Shares are the hand prior. The swarm (E-sel-1) and P-052 set the final shares, but each share is always capped by its pool size and repeat limit.

| component | what it is | role | D8 | pool at the Dec 4 freeze (est.) | 3-10M | 20-30M | 60M | 150M | max repeats |
|---|---|---|---|---|---|---|---|---|---|
| **S. Skill chat** | R0: teacher renders of program-written skeletons (events S1-S9, social basics, grounded lookup on a fictional knowledge base). R1: paraphrases of each render, re-verified. R2: slot values re-drawn on every pass | skill events; dialogue | pass (Apache-2.0 generators) | R0 150-300M accepted; R0+R1 0.5-0.9B surface-unique | 25% | 25% | 20% | 12% | 25 surface passes; 50 with R2 plus spacing if P-187 allows |
| **RH. Grounded rephrasings** | Open human passages, mostly from the fact-dense bucket, placed in context and turned into dialogues that answer only from the passage; plus inpainting over human dialogue text | reading from context; dialogue; register | pass (Apache generator; the source's license carries through) | 100-150M (about 300M by January) | 8% | 8% | 8% | 5% | 16 |
| **HD. Human dialogue** | Assistant-style, task, grounded, multi-party and fiction dialogue (2.2) | realism; out-of-distribution anchor; multi-speaker binding | pass per source; grey sets excluded by default | 0.3-0.5B, of which about 50-80M is clean chat-shaped | 4% | 4% | 4% | 3% | 16 per chat-shaped set; 4 for IRC and fiction |
| **N. Simple narrative** | Cosmopedia v1 stories (children's first) and plain-register public-domain fiction | language; coherence; simple register | pass | 2-3B | 13% | 8% | 6% | 5% | 4 (8 at 10M and under if E-sel-5 allows) |
| **P. Plain prose, strictly open** | Common Pile CCCC, Gutenberg, LoC books, non-encyclopedic Wikimedia, non-technical StackExchange, CC YouTube (Q2), small open sets; all Planck-selected | language; breadth of register | pass under both readings, after the date gate | 15-25B | 50% | 55% | 62% | 75% | 4 (top buckets up to 7 at 30M and under if E-sel-5 allows) |
| **W. Web before Dec 2022** (a bucket inside P's share) | FineWeb (not -Edu) dumps before Dec 2022, globally deduplicated and Planck-selected | breadth | Reading A only (Q1) | 10-40B | weight set by the swarm; 0 if Q1 is no | | | | 4 |
| *Model-written share* | S + RH + about half of N | | | | ~39% | ~37% | ~31% | ~19% | cap 50% at every size (6.2 of the synthscale track) |

### 2.2 Sources inside each component

Raw sizes are sourced. "After selection" figures are all (est.).

**Skill chat and seed pools**

| source | license and author | D8 | role | raw available | after selection |
|---|---|---|---|---|---|
| R0 renders | Planck's own. Generator: Gemma 4 12B, Ministral 3 8B or Qwen3.5-9B, all apache-2.0 ([Gemma](https://huggingface.co/api/models/google/gemma-4-12B-it), [Ministral](https://huggingface.co/api/models/mistralai/Ministral-3-8B-Instruct-2512), [Qwen](https://huggingface.co/api/models/Qwen/Qwen3.5-9B)) | pass | skill events | generated | 150-300M accepted |
| R1 paraphrases | Apache 3-8B model (Qwen3.5-4B, SmolLM3-3B, Ministral 3 8B) | pass | surface variety | generated | 2x R0 |
| Slot pools and seeds (not training text) | SSA first names and Census surnames (PD); GeoNames (CC-BY); [Tatoeba](https://tatoeba.org/en/stats/sentences_by_language) English, 2,045,770 sentences, CC-BY-2.0; [Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA), 1M records, CC-BY-4.0, written by gpt-oss-120b (Q2); persona text kept out of training strings | pass | diversity | as listed | n/a |

**Human dialogue**

| source | license and author | D8 | role | raw available | after selection |
|---|---|---|---|---|---|
| [OASST2](https://datasets-server.huggingface.co/statistics?dataset=OpenAssistant/oasst2&config=default&split=train) English | Apache-2.0, human. The authors say their filters "cannot remove all ChatGPT-generated content" ([paper, App. D](https://arxiv.org/pdf/2304.07327)) | pass, after an AI-ism filter; OOD-H threads removed | assistant turns | 61,278 messages, about 8M tokens | 6-7M |
| [Dolly-15k](https://datasets-server.huggingface.co/statistics?dataset=databricks/databricks-dolly-15k&config=default&split=train) | CC-BY-SA-3.0, human | pass | single-turn | 15,011 rows, about 2.9M | 2.5M |
| [Aya](https://datasets-server.huggingface.co/statistics?dataset=CohereLabs/aya_dataset&config=default&split=train) English | Apache-2.0, human | pass | single-turn | 3,944 rows, about 0.66M | all |
| [MultiWOZ 2.2](https://github.com/budzianowski/multiwoz) | MIT (dataset README) | pass | task state; a program can check slot recall | 10,438 dialogues | all, minus a held-out slice |
| [Taskmaster-1/2/3](https://github.com/google-research-datasets/Taskmaster) | CC-BY-4.0 | pass. All of TM-3 is self-dialog (one person wrote both sides) and part of it is auto-templated, so it is tagged | changes of mind | 13,215 / 17,289 / 23,789 dialogues | TM-1/2 all; TM-3 with the templated part removed |
| [SGD](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | CC-BY-SA-4.0 | pass | slot tracking | 20k+ dialogues | all |
| ABCD, CaSiNo, CraigslistBargains, STAR | MIT / Apache / CC-BY / CDLA-Permissive per the DPI list; check each at its source | pass after the check | negotiation, customer service | small each | all |
| MultiDoGO ([ACL](https://aclanthology.org/D19-1460/)) | CDLA-Permissive | pass after the check | Wizard-of-Oz task dialogue | 81K+ dialogues | size unmeasured |
| Task dialogue, total | | | | about 25-45M (est.) | 20-40M |
| [Topical-Chat](https://github.com/alexa/Topical-Chat) | CDLA-Sharing-1.0 | pass | grounded chit-chat, 21.8 turns on average | 10,784 conversations | all |
| QuAC | CC BY-SA 4.0 ([quac.ai](https://quac.ai/)) | pass | conversational QA over a passage | about 14k dialogues | all |
| [CoQA](https://stanfordnlp.github.io/coqa/), Literature and Wikipedia domains only | CC BY-SA 4.0 (the other domains fail) | pass | reading from context | part of 8,000+ conversations | those domains |
| Grounded human chat, total | | | | about 10-20M (est.) | all |
| [Ubuntu IRC](https://huggingface.co/datasets/common-pile/ubuntu_irc) | public domain, per-document license; logs run to Mar 2025 | pass, date-gated | multi-party addressing | 1.9B | 50-200M social, short-turn slice |
| Wikipedia talk pages (Common Pile `wikimedia`; WikiConv/ConvoKit threaded versions) | CC BY-SA; WikiConv terms not verified | pass after the check | threaded discussion | not reported | unmeasured |
| Congressional hearings (Common Pile [`usgpo`](https://huggingface.co/datasets/common-pile/usgpo)) | public domain | pass | Q&A with named speakers | part of 8.8B | unmeasured |
| [UK Hansard](https://huggingface.co/datasets/common-pile/uk_hansard) | Open Parliament Licence per document; includes Welsh and other devolved bodies | pass per document, English only | formal turn-taking | about 2.3-2.5B | up to 100M, if E-sel-4 admits it |
| [Gutenberg Dialogue](https://github.com/ricsinaruto/gutenberg-dialog), English | MIT, from PD books | pass | fiction dialogue with no speaker names | 2,526,877 dialogues, about 0.43B (the utterance-length unit is assumed to be words) | about 0.2B |
| Gutenberg plays | PD | pass | speaker-labeled dialogue for binding | unmeasured | unmeasured |

**Narrative**

| source | license and author | D8 | role | raw available | after selection |
|---|---|---|---|---|---|
| [Cosmopedia v1](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) stories | Apache-2.0; Mixtral-8x7B (Apache); topics seeded from GPT-written prompts | pass | simple narrative | about 2.95B, of which children's is about 0.76B (row-share estimate) | 0.5-1B |
| Cosmopedia v2 story and dialogue formats | ODC-BY; Mixtral | pass | narrative | about 4.6B + 0.6B (row shares) | A/B arm only |
| Public-domain fiction in plain register | PD | pass | narrative | subset of Gutenberg and LoC | 1-2B |

**Prose**

| source | license and author | D8 | role | raw available | after selection |
|---|---|---|---|---|---|
| [CCCC](https://huggingface.co/datasets/common-pile/cccc) | CC BY / BY-SA / CC0 per document; snapshots up to 2024 | pass, date-gated | plain web prose | 15.2B (260 GB raw) | 5-8B |
| [Project Gutenberg](https://huggingface.co/datasets/common-pile/project_gutenberg) | PD | pass | older-register prose | 5.7B | 1.5-2.5B, with archaic text capped |
| [LoC Selected Digitized Books](https://huggingface.co/datasets/common-pile/library_of_congress) | PD | pass | books | 9.5B (47.8 GB) | 2-4B, after an OCR-quality filter |
| [Wikimedia](https://huggingface.co/datasets/common-pile/wikimedia), non-encyclopedic parts (Simple Wikipedia about 67M; Wikibooks, Wikivoyage, Wikinews) | CC BY-SA; March 2025 dumps | pass, date-gated where page dates allow | plain explanation | part of 15.8B | 1-3B |
| [StackExchange](https://huggingface.co/datasets/common-pile/stackexchange), non-technical sites | CC BY-SA; to Dec 2024 | pass, date-gated | Q&A register | part of 23.9B | 1-3B |
| [CC YouTube](https://huggingface.co/datasets/common-pile/youtube) | CC BY; transcribed by Whisper | pass if Q2 is yes | spoken register | 4.7B | 1-2B from vlog and interview channels |
| [wikiteam](https://huggingface.co/datasets/common-pile/wikiteam) | open per document; laundering risk | pass | fan wikis | 4.3B (437.5 GB raw) | optional, 0-1.5B |
| news, pressbooks, OER, foodista, PD Review | CC BY and similar | pass | small | about 0.33B | 0.2B |
| Common Pile [DPI](https://huggingface.co/datasets/common-pile/data_provenance_initiative) | per entry | pass only after the entry audit | supervised sets | 0.92B | small |
| FineWeb before Dec 2022 (W) | ODC-BY compilation; page text copyrighted | Reading A only | web prose | part of 18.5T GPT-2 tokens | 10-40B |

**Knowledge base**

| source | license and author | D8 | role | raw available | after selection |
|---|---|---|---|---|---|
| Lookup knowledge base | fictional, written by program plus teacher; a small real Wikipedia slice for eval only | pass | grounded lookup families | generated | inside S |

### 2.3 Excluded, and why

| source | reason |
|---|---|
| FineWeb-Edu, DCLM-Edu, Ultra-FineWeb | Ultra's main source. It selects knowledge-dense text. Its selector was trained on Llama-3-70B labels ([FineWeb paper](https://arxiv.org/abs/2406.17557)). It passes only under Reading A anyway |
| DCLM-baseline | Its classifier positives are OpenHermes-2.5 (GPT-4) and r/ELI5. It is about 80% duplicates ([Zyphra card](https://huggingface.co/datasets/Zyphra/dclm-dedup)). Baseline arm only |
| Nemotron-CC-v2 | Under an NVIDIA agreement, and includes rewrites by Qwen2.5-72B, DeepSeek-V3 and Nemotron-4-340B ([card](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2)) |
| SYNTH | DeepSeek-Prover rows; reasoning-trace format; knowledge from Wikipedia |
| Common Corpus, YouTube-Commons | OCRonos rewrites; captions of unknown provenance plus machine translations |
| Institutional Books 1.0 | Noncommercial terms, no redistribution ([page](https://huggingface.co/datasets/institutional/institutional-books-1.0)) |
| Full English Wikipedia (about 5.0B tokens, est.) | Densest knowledge source. Used only in the lookup knowledge base and eval |
| Common Pile code, science, law, patents, government (about 373B), `pre_1929_books` (OCR, overlaps LoC), `doab` | Out of scope, or low value |
| Reddit in every form | No license; Reddit sued Anthropic over scraping ([NatLawReview](https://natlawreview.com/article/beyond-copyright-reddits-lawsuit-against-anthropic)) |
| OpenSubtitles, TED (NC-ND), Santa Barbara (ND), CHILDES/TalkBank (NC-SA; its rules bar commercial LLM use), Switchboard, Fisher, BNC, the BabyLM bundle | License |
| DailyDialog, EmpatheticDialogues, No Robots, PersonaHub (NC); LIMA (gated) | License |
| HH-RLHF, HelpSteer, ProsocialDialog, AgentInstruct, DialogSum, ConvoSumm, TweetSumm | Model-written, or built from NC or Reddit sources |
| SmolTalk v1 Qwen2.5-72B subsets, SmolTalk2 dialogue subsets, smol-magpie-ultra, Everyday-Conversations | Non-Apache generators or user turns |
| TinyStories, SimpleStories, TinyDialogues, TinyChat, SODA, UltraChat, WildChat assistant turns | Closed-model text (baselines only) |
| AirDialogue | NC tag and a machine-generated tag on the official card |
| PersonaChat/ConvAI2, MSC, BST, WoW, WoI, CRD3, CANDOR | No stated license, laundering risk, or unknown terms (Q4) |

### 2.4 Supply against demand

| component | available (est.) | needed for base budgets (5.3) | needed for extensions and probes |
|---|---|---|---|
| S surface | 0.5-0.9B at the freeze | 0.5B or less at 25 passes (the largest reader is 10M at 5,000: 12.5B chat token-passes) | 150M extended to about 165B: 0.8B. 5M at 20,000 on a Titan: 1.0B, or 0.5B at 50 passes with R2 |
| RH | 100-150M at the freeze, about 300M by January | 10M at 5,000: 4B passes, so 250M at 16 repeats | 150M at 100B: 5B passes, so 310M |
| HD | 0.3-0.5B | 2B passes at 10M/5,000, about 4-7 average repeats | fine |
| N | 2-3B | 6.5B passes at 10M/5,000, 2-3 repeats | fine |
| P (+W) | 15-25B (+10-40B) | largest base run is 150M at 75-100B: 15-20B unique at 4 repeats or fewer | 150M extension: about 33B unique, so W, or 6-8 average repeats of P |

If R0 lands at 150M and P-162 fails (paraphrases do not help), the 10M at 25% chat would read R0 about 83 times. The chat share would then have to fall to about 15% (at 50 passes with R2), or R0 must grow.

---

## 3. The selection pipeline

The order is: gate, clean, score, label, grade, transform, admit, monitor.

Every stage gets a planted-positive mutation test before it touches real data (Max's rule):
1. Plant what the stage must remove or rank.
2. Watch the output fail without the stage.
3. Watch it pass with the stage.

All of this runs on CPU, except the labeling in 3.3.

### 3.1 Step 0: provenance, date and language gates

- **Per-document license** from source metadata. Anything not on the allowed list is dropped.
- **DPI entry-by-entry audit** against each entry's own card.
- **Date gate.** Keep documents created before 2022-12-01 wherever the source records a date: CCCC snapshot date, IRC log date, StackExchange post date, wiki page creation.
  - **Evidence:**
    - Among Common Crawl sites grouped by when they were created, the share that is LLM-dominant rose from 2.1% (late 2022) to 29.4% (early 2025). Across all sampled sites it is 6.0% ([DeGenTWeb](https://arxiv.org/abs/2605.00087)).
    - Over 5% of newly created English Wikipedia articles were flagged as AI-written at a 1% false-positive threshold ([arXiv 2410.08044](https://arxiv.org/abs/2410.08044)).
  - Pre-2023 text is safer, not clean: 2.1% of the pre-ChatGPT cohort is already flagged.
  - How much the gate removes is not measured. Record it.
- **English language ID.** This drops Welsh Senedd text from Hansard, for example.

### 3.2 Step 1: hygiene

1. **Normalize, then exact-dedup globally** across all sources.
   - OLMo 3's exact pass removed 67% of documents ([arXiv 2512.13961](https://arxiv.org/abs/2512.13961)).
   - Global exact dedup matters because upsampling later multiplies any duplicate that survives.
2. **MinHash near-dedup.**
   - First within each source and snapshot. FineWeb found per-snapshot beat global dedup, in a setting with no upsampling ([arXiv 2406.17557](https://arxiv.org/abs/2406.17557)).
   - Then a **global cross-source pass over the buckets that will be upsampled.** OLMo 3 and Essential-Web dedup globally because they deliberately reintroduce repetition.
   - Settings: Jaccard 0.7, 14 bands x 9 rows ([Essential-Web](https://arxiv.org/abs/2506.14111)).
3. **Boilerplate line removal.** Lines that recur across many documents in a source are dropped. This is a cheap stand-in for OLMo 3's suffix-array pass, which removed 14% of bytes.
4. **Quality heuristics** (Gopher/FineWeb rules) on the Common Pile sources.
5. **PII scrub.** IRC nicknames are mapped to stable speaker labels, not deleted.
6. **13-gram decontamination** against:
   - RC-12 dev and sealed, using hashed n-gram sets produced in the sealed session so the pipeline never sees sealed plaintext;
   - OOD-H;
   - the human-test scripts.

**Evidence that this matters for Planck specifically:**
- Deduplicated models emit memorized text 10x less often, and dedup removes train-test overlap affecting over 4% of validation sets ([Lee et al.](https://arxiv.org/abs/2107.06499)).
- 3% repeated data cut effective model size by up to 3x on a copying task. The damage peaks when loss on the repeated subset nears zero ([Hernandez et al.](https://arxiv.org/abs/2205.10487), 1.5M-800M).
- Copying and induction are exactly Planck's target skills.

### 3.3 Step 2: cheap scores on everything

- **DSIR importance weights**: hashed unigrams and bigrams, 10k buckets.
  - Target set: OASST2 (minus OOD-H), Topical-Chat, task dialogue, the IRC slice, and pilot renders capped at 25% of the target so it does not learn teacher style.
  - **Evidence:** at about 110M, from scratch, DSIR-selected Pile data gave +2% average GLUE over random selection. KL reduction on the hashed features correlated with downstream accuracy at r = 0.82 ([DSIR](https://arxiv.org/abs/2302.03169)).
  - **Caveat:** that was a masked LM scored by GLUE fine-tuning. Transfer to a causal chat decoder is untested, so DSIR is an arm and a feature, not a default.
- **Program features:**
  - fact-density proxies: digits, mid-sentence capitals, rare-word rate against our own frequency list;
  - dialogic markers: I/you rate, questions, quotes;
  - statistical simplicity: gzip ratio, distinct-2/3;
  - cleanliness: lines ending in punctuation, symbol ratio.

  None of these is used until it is checked against the teacher's fact-density labels.

### 3.4 Step 3: teacher labels, then Planck's own classifiers

**Why build our own.**
- Off-the-shelf selectors optimize for benchmark knowledge.
  - FineWeb-Edu's threshold was chosen as a trade-off between MMLU/ARC and HellaSwag-type benchmarks.
  - On benchmarks, the conversational registers Interactive Discussion (0.431) and Spoken (0.422) score below Description (0.452) and Opinion (0.447) ([register study](https://arxiv.org/html/2504.01542)).
- They also carry restricted or closed labels (Q5).
- The choice of positives matters: DCLM's positive set lifted CORE by 3.5 points over conventional positives at 7B ([DCLM](https://arxiv.org/html/2406.11794v3)) [>=1B only].
- Several axes beat one: Meta-rater gained +3.23 at 1.3B ([arXiv 2504.14194](https://arxiv.org/abs/2504.14194)) [>=1B only].

**Sample about 60k documents:**
- 40k stratified over DSIR deciles and sources;
- 10k human dialogue;
- 5k fact-dense anchors (Wikipedia, StackExchange);
- 5k teacher renders.

**Four rubrics, each an additive 0-5 scale.** Claude writes the rubrics; the teacher applies them.
- **CU (conversational usefulness):** person-to-person, everyday exchange, back-references and corrections, a register a small assistant should copy.
- **FD (fact density):** checkable specific facts per 100 words.
- **RS (plainness):** common adult vocabulary. Not child-level ([Readability is not Learnability](https://arxiv.org/abs/2510.13915): statistical simplicity, not readability, predicts learnability).
- **CL (cleanliness):** complete, coherent, no extraction junk or spam.

**Labeling.**
- Gemma 4 12B, thinking off.
- The document goes first and the rubric second, so prefix caching shares the document across the four prompts.
- Cost: 5-15 5070-hours (est.), scheduled while the teacher is loaded for the pilot.

**Validity gates (E-sel-0).** An axis that fails is dropped.
- Qwen3.5-9B relabels 3k documents; Spearman must be at least 0.6 per axis (my threshold).
- The classifier must match held-out teacher labels at Spearman 0.7 or more (my threshold).
- Claude reads about 30 documents per score level per axis.
- Mutation tests, each of which must move the score in the stated direction:
  - wrapping an encyclopedia article in User/Assistant markers raises CU by at most 1;
  - replacing named entities lowers FD;
  - shuffling sentences lowers CL;
  - second-person spam keeps CU low;
  - fact lists in chat format keep FD high.

**Classifiers.**
- One fastText model per axis, using the Ultra-FineWeb recipe: dim 256, word 3-grams, 3 epochs, 3 classes ([arXiv 2505.05427](https://arxiv.org/html/2505.05427v1)).
- Scoring rate is about 52k tokens/s per CPU core. That figure is derived from "80 CPUs, 1,000 h, 15T tokens", and it is unclear whether "CPUs" means cores or sockets.
- The measured tokenization rate on Lambda's 64 cores is 1.8M tok/s (`research/lanes/compute.md`). That caps a combined tokenize-and-score pipeline.

**Arm, not default: PreSelect** ([arXiv 2503.00808](https://arxiv.org/abs/2503.00808)).
- It trains a fastText scorer on documents whose loss predicts ability across a family of models. 30B selected tokens beat 300B unselected at 1B [>=1B only].
- Planck's own 55-model swarm can serve as the family, with chat bits per byte as the target.

**Essential-Web labels** are used only as a coarse first-stage filter on W:
- Its distilled labeler agrees poorly on extraction artifacts (kappa 0.27 vs 0.74) and missing content (0.48 vs 0.66).
- Its "Comment Section" type means Reddit threads and its "Creative Writing" type includes song lyrics ([card](https://huggingface.co/datasets/EssentialAI/essential-web-v1.0)).

### 3.5 Step 4: buckets and per-size upsampling curves

**Discard the CL bottom 20-40%** (swept).
- Classifier filtering works mainly by removing noise.
- The best keep fraction varies with compute ([Apple, arXiv 2510.00866](https://arxiv.org/html/2510.00866v4)), so each size gets its own threshold.

**Buckets.**
- A bucket is source x CU tertile x FD (low, high).
- Each bucket is stored as separate compressed text shards.
- The mix is a sampling manifest over buckets, so reweighting never forces a rebuild.

**Within each bucket, a monotone OLMo 3-style upsampling curve over CU vigintiles.**
- Capped at 4x by default, or 7x at 30M and under if E-sel-5 allows.
- OLMo 3's quality-aware upsampling beat flat filtering in data-constrained 1B/100B simulations. Their example curve drops the bottom 40% and repeats the top 5% 7x, with 7x set empirically ([arXiv 2512.13961](https://arxiv.org/html/2512.13961v2)) [>=1B only].

**Alternative arm: temperature sampling**, with p proportional to exp(score / 2).
- Hard top-k cuts raised perplexity by about 1.6-2.6 over uniform sampling at 1.3B ([QuRating](https://arxiv.org/html/2402.09739v3)) [>=1B only].

**Semantic diversification** (D4-style), applied only to the top buckets that repeat.
- At 125M, repeating D4-selected data beat fresh random data ([D4](https://arxiv.org/html/2308.12284)).
- On 10-20B tokens it costs a small fraction of D4's 888 A100-hours for 400B.

**Source caps** (my judgment): archaic public-domain prose at most 25% of P; spoken YouTube at most 10% of P.

### 3.6 Step 5: fact-dense text, converted rather than stored

**Default.**
- The FD-high bucket seeds RH grounded dialogues (4.6).
- Its raw-prose weight is set by the swarm.

**Arms (E-sel-3):**
- keep as is;
- weight x0.2;
- entity anonymization of FD-high;
- x0.2 plus source tags.

**Evidence:**
- Selecting documents low in facts and trivia helped all three commonsense tasks at 1.3B ([QuRating](https://arxiv.org/abs/2402.09739)) [>=1B only].
- Entity-anonymized pretraining at 135M-360M lowered closed-book recall and improved contextual QA and fact verification ([KLLM](https://arxiv.org/html/2607.12831v1)). But KLLM used Flair NER at about 87% recall, and it anonymized inputs at inference as well. Anonymizing only the training text is untested.
- LMLM at 382M reached FactScore 31.9 vs 14.0, but only with database lookup; without the database it scored 12.8 ([LMLM](https://arxiv.org/html/2505.15962v3)).

No study measures multi-turn skill as a function of fact density, so this is an experiment, not a default.

### 3.7 Steps 6-8: tags, admission, monitoring

- **Source tags** (arm E-sel-6).
  - Tag non-chat sources, then drop the tags in the final decay.
  - MeCo matched standard pretraining with 33% less data at 1.6B ([arXiv 2501.01956](https://arxiv.org/abs/2501.01956)) [>=1B only].
  - Source tags cut the junk penalty from 20x to 2x in Physics 3.3 (knowledge storage).
- **Admitting human dialogue by anneal-as-eval (E-sel-4).**
  - Take decay branches with 30% candidate data from a trunk.
  - Ultra-FineWeb's version cost about 110 H100-hours against about 1,200 from scratch, not counting the trunk.
- **Memorization monitor for every repeated pool.**
  - Track loss on a seen sample against an unseen held-out sample.
  - If the gap exceeds a pre-registered bound (0.1 nats, my number), that pool's repeats are lowered.
  - The TII memorization window was measured at constant LR ([arXiv 2607.04969](https://arxiv.org/abs/2607.04969)), so P-187 checks the bound under WSD.

### 3.8 Thresholds to sweep

| knob | values | decided in |
|---|---|---|
| CL discard | 20%, 40% | E-sel-2 |
| CU curve cap | 4x, 7x | E-sel-5 |
| CU sampling | curve, temperature 2 | E-sel-2 |
| DSIR weighting | on, off | E-sel-2 |
| FD-high weight | set by swarm; forced to 0.2 | E-sel-1, E-sel-3 |
| Anonymization of FD-high | p = 0, 1 | E-sel-3 |
| W bucket | 0, or swarm weight (only if Q1 is yes) | E-src |
| Archaic cap within P | 10%, 25% | E-sel-1 |
| Each human-dialogue source | in, out | E-sel-4 |
| CL and CU thresholds per size | proxy-set, re-checked at 20M | E-sel-2 |

---

## 4. The synthetic pipeline at scale

### 4.1 Pool architecture

| layer | what | adds | cost |
|---|---|---|---|
| R0 | Teacher renders program-written skeletons in script mode; a program verifies them | structure, naturalness, skill events | the expensive part |
| R1 | 2-4 paraphrases per render. Slot spans are locked, and the full checker and decontamination re-run | surface variety | about 5-10x cheaper per token (est.) |
| R2 | At load time, typed slot spans get fresh same-type values from the D8 pools, and the cheap checks re-run | new value tokens on every pass | CPU only |
| RH | Grounded rephrasings and inpainting of open human text (4.6) | reading from context; human register | cheap model; prefill-heavy |

**Pool growth after the freeze.**
- The growth rule is pre-registered at the freeze, not a fixed size.
- After Dec 4 the pipeline is frozen: skeleton generator, teacher, checker, paraphraser, RH prompts and filters.
- New shards from the frozen pipeline become pool v2, v3 and so on, each hashed.
- A run uses the newest version hashed before its launch, or before its extension.

### 4.2 Teacher and throughput per machine

**Why Gemma 4 12B is tight on memory on the 12 GB RTX 5070.**
- Google's 4-bit checkpoint is 10.26 GB, of which the bf16 embedding and a separate lm_head copy are 2.01 GB each ([file tree](https://huggingface.co/api/models/google/gemma-4-12B-it-qat-w4a16-ct?blobs=true)).
- Loaded once, the tied weights would be about 8.2 GB resident.
- A 12 GiB card at 0.9 utilization, headless, then leaves about 2.7 GB for the KV cache (est.).
- Gemma's 1,024-token sliding window caps KV growth per conversation. Ministral 3 8B and the Qwen3.5 attention layers grow linearly with length.

**Published anchors** ([arXiv 2601.09527](https://arxiv.org/html/2601.09527v1)). "TPS" there means total output tokens per second.
- Qwen3-8B NVFP4 at concurrency 32: 1,987 TPS on a 16 GB RTX 5070 Ti and 1,249 on a 5060 Ti.
- Gemma3-12B: 486 TPS (NVFP4, c32, 5070 Ti, 9.3 s time to first token) and 305.7 TPS (W4A16, c64, 5060 Ti, 38 s time to first token).
- All of these are 256-in/256-out workloads. At 8k context, Qwen3-8B NVFP4 fell to 411 TPS at c8, even on a 5090.
- Planck renders run 2-3k tokens per request, so every rate below is probably biased high.

| machine | model and role | raw tok/s (est.) | accepted Planck tokens per hour (est.) | per week, full time (est.) | notes |
|---|---|---|---|---|---|
| 5070 | Gemma 4 12B render (vLLM W4A16 QAT, fp8 KV; also llama.cpp with the 6.98 GB QAT GGUF) | 200-450 | 0.3-1.2M | 41-165M (143 h) | about 10-16 conversations in flight |
| 5070 | Ministral 3 8B render (NVFP4, fp8 KV) | 700-1,100 | 1.0-2.8M | 143-400M | 16-21 in flight at 2.5k tokens (fact-check re-derivation) |
| 5070 | Qwen3.5-9B render (NVFP4) | 500-1,100 | 0.7-2.8M | 100-400M | open vLLM prefix-cache bugs [#45238](https://github.com/vllm-project/vllm/issues/45238), [#55766](https://github.com/vllm-project/vllm/issues/55766), [#40696](https://github.com/vllm-project/vllm/issues/40696) |
| 5070 | Qwen3.5-4B or SmolLM3-3B paraphrase | 2,000-3,500 | 5-12.5M | 0.7-1.8B | optimistic |
| 5070 | Ministral 3 8B paraphrase (8B arm) | 700-1,100 | 1.5-3.5M | 0.2-0.5B | Kang et al. found 8B generators beat 3B |
| 5070 | Gemma 4 E4B via llama.cpp, official q4_0 GGUF 5.15 GB ([tree](https://huggingface.co/api/models/google/gemma-4-E4B-it-qat-q4_0-gguf?blobs=true)) | unmeasured | unmeasured | | a cheap arm; the vLLM checkpoint (11.5 GB) does not fit |
| Mac M5 | 4B-class paraphrase or RH (MLX or vllm-metal) | 80-200 | 0.2-0.7M | 20-71M per 100 h | Q7; after the current experiment; 8 GB cap |
| Mac M5 | Gemma 4 12B | 35-90 | 0.05-0.23M | 5-23M per 100 h | too slow for R0 |
| One Titan card, after a driver upgrade to 580+ | Ministral 3 8B, W4A16 built from the BF16 repo (the official release is FP8, which Turing cannot run natively) | 600-1,400 | 0.9-3.6M | 124-515M | speculative. fp16 numerics unverified. Gemma runs only via llama.cpp, because of vLLM [#38918](https://github.com/vllm-project/vllm/issues/38918) |

Conversions behind the table (est.):
- Accepted tokens = raw x yield. Yield is 40-65% for renders and 70-90% for paraphrases (unsourced).
- Planck-token ratio is about 1.0-1.1x after stripping template tokens. Measured only for Gemma's tokenizer (4.47 bytes/token on OASST).

**5070-hours for 300M R0 tokens:**
- Gemma 4 12B: 259-1,042 h.
- Qwen3.5-9B: 106-417 h.
- Ministral 3 8B: 107-300 h.

PLAN budgets 170-330 h. With Gemma the budget almost certainly does not cover 300M.

**The teacher gate (PLAN Phase 1) should measure:**
- accepted Planck tokens and bytes per 5070-hour;
- at maximum concurrency with fp8 KV;
- on realistic 2-3k-token requests.

The winner is the fastest teacher whose naturalness is within 10% of the best (chatdata L12).

**Levers to A/B in the pilot:**
- fp8 KV;
- `max_model_len` of about 3k;
- a headless card;
- prefix caching;
- 2-4 samples per skeleton, so one prefill buys retries on hard cells;
- structured output for the turn format;
- MTP or speculative decoding. vllm-metal reports +20% at c1 and +9% at c16 for Gemma 4 E4B ([blog](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)).
- Check vLLM's log for the NVFP4-to-Marlin fallback warning ([#47749](https://github.com/vllm-project/vllm/issues/47749), ModelOpt mixed checkpoints).

### 4.3 Verification and yield

**The checker:**
- the 13 checks of `chatdata.md` 3.6;
- the assistant self-claim rule (P-061);
- the deflection filter (L10);
- single-turn repair: re-render only the failed turn, with the rest of the conversation as context. This is tested in the pilot.

**The checker's false-accept rate matters more than its yield.** It is mutation-tested with planted errors: a stale value, a wrong owner, a restated answer, a leaked role. Imperfect verifiers still prevent collapse ([arXiv 2406.07515](https://arxiv.org/abs/2406.07515)).

**Expected yield is 40-65% overall, and 20-40% on correction and long-distance cells. Treat 40-65% as an upper band.**
- APIGen's program checks passed 34-84% depending on generator strength, and its generators were 33B-236B ([arXiv 2406.18518](https://arxiv.org/abs/2406.18518)).
- DeepDialogue accepted 0.54-0.65 of dialogues, and acceptance fell with length: at 10 turns rejections outnumbered acceptances ([arXiv 2505.19978](https://arxiv.org/html/2505.19978v1)).

### 4.4 Diversity controls

**Prevention:**
1. **Skeleton randomization.** Events, distances, distractors, turn count, openings and closings, user style (terse, chatty, typos), and required words. This carries most of the structural diversity.
2. **Personas as prompt seeds** from Nemotron-Personas-USA (Q2), not as training strings.
3. **Topics** from a list of several thousand everyday topics. Never repeat a (topic, persona, skill set) triple.
4. **50-200 teacher-written paraphrases of the render instructions**, rotated per request.
5. **Sampling.**
   - Gemma's defaults are T 1.0, top-p 0.95, top-k 64, plus min-p 0.05-0.1 ([min-p](https://arxiv.org/abs/2407.01082)).
   - Ministral's card recommends T below 0.1, so test it at T 0.7-1.0.
   - Temperature alone does not fix repetition: TinyStories needed random required words ([arXiv 2305.07759](https://arxiv.org/abs/2305.07759)).
6. **A dynamic "do not start with" list** built from the last shard's 50 most common openings (unsourced).
7. **Verbalized sampling for openings and user styles.** Ask for 5 options with probabilities and sample from the tail. This gave 1.6-2.1x diversity on creative tasks ([arXiv 2510.01171](https://arxiv.org/abs/2510.01171)).
8. **Cross-model user turns (L14) and a second teacher on about 30% of shards.**
   - Different models produce "strikingly similar" outputs ([Artificial Hivemind](https://arxiv.org/abs/2510.22954)), so expect less from this than it sounds.
   - 76% of the syntactic templates in model text also appear in pretraining data, against 35% for human text ([arXiv 2407.00211](https://arxiv.org/abs/2407.00211)).
9. **Reject rather than cap.** A render that pushes a sentence over the corpus cap (0.1% of conversations) is rejected at generation time.

**Measurement.** Run on CPU for every 10M-token shard, against the pilot band and against OASST2 human turns:
- gzip ratio;
- distinct-2/3;
- long-n-gram frequency;
- syntactic template rate ([diversity package](https://arxiv.org/abs/2403.00553));
- Self-BLEU;
- embedding clusters of openings;
- the coverage matrix.

**Alarms are two-sided.** Lower n-gram diversity speeds learning at 262K-33M parameters but makes models brittle out of distribution ([Readability is not Learnability](https://arxiv.org/html/2510.13915v1)). So:
- A drift out of the pilot band in either direction is flagged, and L3 measures the trade-off.
- The hard alarms are one-sided:
  - any non-template 8-gram in more than 0.1% of conversations;
  - any opening cluster above 2% of openings;
  - any coverage cell below its minimum.

### 4.5 Repetition limits

**Default caps**, pre-registered at the freeze:
- 25 surface passes by default.
- Up to 50 with R2 plus spacing.
- Up to 100 only if P-187 shows no loss.

**Evidence:**
- Up to 4 epochs of a whole corpus is nearly free; value falls to 1/e by about 16 ([Muennighoff](https://arxiv.org/abs/2305.16264)).
- At 90M, a run that was 100% SFT data read a 2 GT SFT epoch 50 times over 100 GT with "no clear memorization artifacts" ([Falcon-H1-Tiny blog](https://tiiuae-tiny-h1-blogpost.hf.space/)).
- At 100M, spaced high-quality math peaked at about 125-175 observed epochs, at constant LR, in one domain ([TII](https://arxiv.org/html/2607.04969v1)).

**The Planck-specific risk is memorized planted facts.**
- In-context learning gives way to in-weights learning once items can be memorized. The same paper notes L2 regularization may keep in-context learning ([Singh et al.](https://arxiv.org/abs/2311.08360)).

**Mitigations:**
- R2 slot refresh.
- Spacing wider than the memorization window. Falcon's conservative estimate, scaled, is about 0.5 / 1.5 / 3 / 7.5 GT at 10 / 30 / 60 / 150M; the optimistic one is 0.1 / 0.3 / 0.6 / 1.5 GT.
- Weight decay retuned for many passes. With 200M tokens, the best weight decay was 30x standard ([arXiv 2509.14786](https://arxiv.org/abs/2509.14786)).
- BPE-dropout on repeated chat (P-160).

**P-187, a pool-pass curve (proposed):**
- At 5M, hold total tokens and chat share fixed and vary the unique pool so chat is read 10 / 25 / 50 / 100 / 200 times, with and without R2. 2 seeds.
- Then a 20M check at 25 and 100 passes, because the window scales with size.
- Metrics: RC-12 dev margins, plus a probe that asks for a pool conversation's planted fact without its context.
- Win rule: the largest pass count within seed noise of the 10-pass arm.

### 4.6 Rephrasing open human text into dialogue

1. **RH-G, grounded dialogues (the main use).**
   - A human passage goes into the context. The teacher writes a short dialogue in which the user asks and the assistant answers only from the passage.
   - Near-miss and unanswerable questions are included.
   - Checks: answer spans overlap the passage, unsupported claims are rejected, and the chat checks apply.
   - Seeds come mostly from the FD-high bucket, so fact-dense text becomes practice at reading, not stored facts.
   - MIND found a knowledge gap between the two participants essential ([arXiv 2410.12881](https://arxiv.org/abs/2410.12881)) [>=1B only].
2. **RH-I, inpainting.**
   - Human sentences stay verbatim as one speaker, and the teacher writes only the other speaker. Half the tokens stay human.
   - Dialog Inpainting's 19M dialogues matched real ConvQA data on human-rated adequacy ([arXiv 2205.09073](https://arxiv.org/abs/2205.09073)).
   - Seeds: OASST2 threads (not OOD-H) and Gutenberg Dialogue.
3. **R1 paraphrases (P-162).**
   - Reformulation beat repetition at 134M-13B ([MGA](https://arxiv.org/abs/2502.04235)).
   - The auxiliary-views paper found reformulations help even factual recall, and that the rephraser's quality does not decide the gain. It also found that "paraphrasing helps only at smaller batch sizes" ([arXiv 2609.04180](https://arxiv.org/abs/2609.04180)). So P-162 runs at the curve's batch size (64-128 x 2048).
   - 8B generators consistently beat 3B in [Kang et al.](https://arxiv.org/html/2510.01631v1), and [BeyondWeb](https://arxiv.org/abs/2508.10975) gives 47.3 / 48.8 / 49.2 for 1B / 3B / 8B rephrasers. So the pilot includes an 8B paraphraser arm and judges it by a 5M proxy, not by fidelity alone.
4. **How much of it.**
   - About 30% rephrased plus 70% natural was best at 100M-3B. For QA-style data, about 50% was best at smaller configurations, measured on loss (Kang).
   - Real plus rephrased data trained about 3x faster at 128M-1.3B ([WRAP](https://arxiv.org/abs/2401.16380)).
   - Keeping real data alongside synthetic bounds collapse ([arXiv 2404.01413](https://arxiv.org/abs/2404.01413)).
   - Hence the 50% cap on model-written text.
5. **Cost.** 2-6M accepted RH tokens per 5070-hour with a 4B model (est., prefill-heavy), or 20-70M per Mac-week at 100 hours.
6. **P-165 arms at 20-30M:** grounded dialogue, high-quality rephrase and QA rephrase, each at 20% of prose.
7. **ShareAlike.** Released RH text derived from Wikipedia, StackExchange or QuAC is released under CC BY-SA (Q8).

### 4.7 Pool size by phase

| phase | dates | machine | work | accepted at end (est.) |
|---|---|---|---|---|
| Build | Sep 25 - Oct 4 | CPU | skeleton generator; mutation-tested checker; diversity battery; slot-span format; R2 loader | 0 |
| Pilot | Oct 5-8 | 5070 | 200 skeletons through each renderer: Gemma 4 12B (vLLM and llama.cpp), Ministral 3 8B, Qwen3.5-9B, Qwen3.5-4B, Gemma 4 E4B GGUF. 3-4 paraphrasers, one of them 8B, on 200 accepted conversations. 50 RH seeds | about 0.3M |
| Pilot set | Oct 9-18 | 5070 | the winner renders every family | R0 20-30M |
| Bulk 1 | Oct 19 - Nov 8 | 5070 | R0 to 100M, after the P-012/P-053 gate (Oct 20-28); R1 x1; RH 30-50M | R0 100M, R1 50-100M, RH 30-50M |
| Rate check | Nov 8 | | If the measured rate x remaining hours is under 300M: cap R0 at 150-200M and add R1 and RH | |
| Bulk 2 | Nov 9-22 | 5070, in gaps | R0 to 150-300M; R1 to x2; RH to 100-150M | surface 0.5-0.9B; RH 100-150M |
| Freeze | Dec 4 | | pool v1 hashed; growth rule pre-registered | |
| Growth | Dec 4 - Feb 21 | Mac nights (Q7); inner Titans only with IT approval and a driver upgrade | R1 to x3-4; RH to about 300M | surface 1-2B, only if P-186 calls for long runs |

**Generation cap:** about 350 5070-hours through Nov 22. After the freeze the 5070 only trains.

### 4.8 Stop rule and the trade against training

What one 5070-hour buys (est.):
- 0.3-1.2M Gemma R0 tokens;
- or 1.0-2.8M Ministral R0 tokens;
- or 5-12.5M 4B R1 tokens;
- or 0.25-0.5B tokens of 10M training;
- or 0.12-0.19B tokens of 60M training.

**Stop R0 on the 5070 when both hold:**
- the largest planned run would read R0 about 100 times or fewer, and
- it would read the R0+R1 surface 25 times or fewer (50 with R2, if P-187 supports it).

Past that point, the hour is worth more as training.

---

## 5. Token budget

### 5.1 The answer to Max

**Yes, more tokens keep helping at these sizes.**
- A 150M model was still improving at 10,000 tok/param on single-epoch data ([Sardana et al.](https://arxiv.org/abs/2401.00448)).
- MobileLLM-125M gained 1.3 points of average zero-shot accuracy going from 0.25T to 1T tokens, about +0.65 per doubling ([arXiv 2402.14905](https://arxiv.org/abs/2402.14905), Table 10).

**Yes, a slight gain is worth buying when the GPU-hour has no better use.** It usually does have one first:
1. seeds at the size that carries the claim;
2. growing the chat pool when passes exceed the caps;
3. the next curve size.

So the rule is: set base budgets that fit with 3 seeds, then spend every spare hour in an extension queue ranked by measured gain per GPU-hour.

**Lab budgets are out of reach.**
- SmolLM2-135M used 2T tokens, Qwen2.5-0.5B 18T, LFM2.5-230M 19T.
- For Planck-150M on the 4 Titans, those would take roughly 14 months to a decade (est.).
- "Tens of thousands of tokens per parameter" is cheap only at 3-5M. At 10M, 20,000 tok/param with 3 seeds is about 1,670 5070-hours, the whole curve window.

### 5.2 What the evidence says about more tokens

- **Diminishing returns.**
  - Each doubling removes 15-22% of the remaining data-limited loss (β = 0.24-0.37 across the [Hoffmann](https://arxiv.org/abs/2404.10102), [Besiroglu](https://arxiv.org/abs/2404.10102) and Sardana fits).
  - These are upper-bound-ish. Sardana's fits over typical ratios had β = 0.13-0.16, and fits from ordinary ratios overestimate the value of tokens at extreme ratios.
  - Extending an already long run has a lower exponent ([Liew and Kato](https://arxiv.org/abs/2510.06548)).
- **Skills are not loss.**
  - Averaged benchmarks track loss ([Gadre et al.](https://arxiv.org/abs/2403.08540)).
  - Single abilities sit at chance until loss crosses a threshold ([Du et al.](https://arxiv.org/abs/2403.15796)).
  - Below the critical density they may not appear at all (1.1).
- **Late saturation risk.**
  - Pythia up to 410M drops and then plateaus late in training, and widths under 1,000 degenerate ([Godey et al.](https://arxiv.org/abs/2404.07647)). Every Planck width, 192-640, is under 1,000.
  - Counter-evidence: MobileLLM-125M and Sardana's 150M kept improving.
  - Decay branches detect saturation directly: a later branch that scores worse than an earlier one.
- **Harder fine-tuning after heavy pretraining.**
  - 15M-90M models improve as base models with more pretraining, but after fine-tuning they degrade beyond a token budget. The inflection comes earlier at higher fine-tuning LR ([Springer et al.](https://arxiv.org/abs/2503.19206)).
  - So every branch is scored after the same short SFT, and SFT learning rates stay low.
- **Narrow corpora scale differently.** TinyStories has a data exponent of 0.185 against 0.141 for WikiText ([Cagnetta et al.](https://arxiv.org/html/2602.07488)). The web-fit numbers are a prior, not a prediction for Planck.

### 5.3 Budget per model size

**Rates (est., ±2x):**
- 5070: 5M 100-190k tok/s, 10M 70-140k, 20M 50-80k, 30M 42-74k, 60M 34-54k (`research/followup/loop.md`, `research/lanes/compute.md`).
- These come from AdamW MFU anchors and **leave out NorMuon's overhead**, which measured up to 1.5x on a Titan A/B. `bench_micro` must run with NorMuon.
- Titans: 4-card DDP for the 150M at about 42k tok/s after the NorMuon correction (35-80k).
- The Mac is about a tenth of the 5070 and is not used for curve runs.

**Calendar:** Dec 4 - Feb 21 on the 5070 is about 1,706 hours at 90% uptime.

**Unique tokens needed** use the section 2.1 shares:
- general text (N + P) at 4 repeats or fewer;
- skill chat at 25 surface passes or fewer.

| size (total params) | base budget | tokens per seed | seeds | 5070 h per seed, central (range) | 3 seeds | 4-Titan days per seed | unique general needed | chat surface needed |
|---|---|---|---|---|---|---|---|---|
| 3M (only if 5M passes a Level A family on dev) | 10,000 | 30B | 1-3 | 35-60 (rate unmeasured) | | | 4.7B | 0.3B |
| 5M (5.11M) | 2,000 | 10B | 3 | 20 (15-28) | 60 (44-83) | about 0.5-1 | 1.6B | 0.1B |
| 10M (9.96M) | 2,000; 5,000 after extension 1 | 20B (50B) | 3 | 56 (40-79) | 167 (119-238) | 1.2 (2.9) | 3.2B (7.9B) | 0.2B (0.5B) |
| 20M (19.8M) | 1,000 | 20B | 3 | 85 (69-111) | 256 (208-333) | about 1.5-2.5 | 3.2B | 0.2B |
| 30M (29.7M) | 1,000 | 30B | 3 | 152 (113-198) | 455 (338-595) | about 3.2 | 4.7B | 0.3B |
| 60M (60.8M) | 500 | 30B | 3 | 194 (154-245) | 581 (463-735) | 4.0 (2.5-7.3) | 5.1B | 0.24B |
| 150M (147.8M) | 500; 670 only if the smoke test gives 67k tok/s or more | 75B (100B) | 3 | not on the 5070 | | 20.7 at 42k (15.8 at 55k; 24.8 at 35k) | 15B (20B) | 0.36B (0.48B) |
| generic control, 30M | 500 | 15B | 1 | 76 (56-99) | | | | |
| post-training and dev eval | | | | | 30 (20-40) | | | |
| **5070 total** | | | | | **1,624 (1,258-2,115) of about 1,706** | | | |

**Titan dates for 150M** (serial seeds from Dec 17, 75B each):
- At 42k tok/s: Jan 6, Jan 27, Feb 17.
- At 55k: Jan 1, Jan 17, Feb 2.
- 100B per seed fits only at about 67k tok/s or more.

**Re-plan rule** (PLAN's rule, extended), if `bench_micro` comes in slow:
1. The control drops to 5B.
2. 60M seed 3 moves to the Titans, labeled as the fp16 platform.
3. 20M drops to 2 seeds unless it is the claimed size.
4. 5M stays at 2,000.

**Tokens per body parameter.** At 10M the body is 7.86M, so tok/param per body parameter is 27% higher than per total parameter. The curve therefore mixes size with training ratio. Every size also takes a decay branch at 500 tok/param, which gives a clean equal-ratio curve to report beside it.

### 5.4 The extension queue

**Mechanics.**
- Keep every run's pre-decay trunk (the stable-phase checkpoint at 80% of its budget).
- To extend, continue the stable phase from that trunk and take a new 20% decay.
- Never re-warm a decayed checkpoint ([Hägele et al.](https://arxiv.org/abs/2405.18392)).
- Two open-ended-trunk caveats:
  - The weight-decay timescale follows D/N ([Power Lines](https://arxiv.org/abs/2505.13738)), so an extended trunk runs slightly mistuned.
  - The batch schedule assumes a known horizon.

**Default order,** re-ranked by measured gain per hour. The ranking multiplies by 1 if the size's dev composite is within the minimum detectable effect of a Level A threshold, and by 0.5 otherwise.
1. 10M, 2,000 to 5,000: +34B per seed, 283 h (202-405) for 3 seeds. This doubles as P-186's 5,000-tok/param point at a curve size.
2. 5M, 2,000 to 5,000: +17B per seed, 101 h (75-142).
3. 20M, 1,000 to 2,000: +24B per seed, 308 h (250-400).
4. 30M, 1,000 to 2,000: +36B per seed, 545 h (405-714).
5. The best 150M seed's trunk on the Titans after its seeds finish. This is realistic only at 55k tok/s or more. It is reported as "Planck-150M-ext (1 seed)" and never carries Level A.
6. If E3 or P-026 shows seed lock-in, spend spare hours on seeds 4-5 at the claimed size before tokens.

**Every extension raises chat token-passes.** The pool must grow under the growth rule (4.1), or the share falls. The share never goes below the critical share P-051 measures.

### 5.5 How P-186 sets and adjusts this

**P-186 as written does not fit.**
- Taking 5M and 10M to 20,000 tok/param with 2 seeds is about 1,150-2,380 5070-hours.
- The ledger priced it at 40-80 hours.
- The 5070 has only about 280-690 spare hours in Phase 3.

**Redesign:**
1. **Before the freeze (Oct 19 - Dec 3): 5M x 2 seeds.**
   - One trunk to 2,000 tok/param (10.2B tokens), with decay branches at 250 / 500 / 1,000 / 2,000.
   - Each branch is scored as a base model and after an identical short SFT, on bits per byte (a)-(c) (section 6) and RC-12 dev margins (P-004).
   - Cost 35-75 h (est.). Optionally continue one seed to 5,000 (+21-47 h).
   - This trunk also carries E-sel-5.
2. **Decision rules at the freeze:**
   - **Flat by 1,000.** The pooled margin slope per doubling has a 95% CI that includes zero from 1,000 to 2,000, and the bits-per-byte gain is under 1%. Then 5M and 10M base budgets drop to 1,000, extension 1 is cancelled, and the freed 200-300 h go to 5-seed lock-in runs at 10M and 30M.
   - **Still rising at 2,000.** Keep the base budgets and the queue order.
   - **Rising faster than the web fits predict** (a 2,000-to-4,000 margin gain above a third of the 5M-to-10M gap at equal tokens). Promote extension 1 into the base, and pay for it with the re-plan rule.
3. **During the curve.**
   - The 10M runs take branches at 500 / 1,000 / 2,000 (and 5,000 on extension). That gives a 3-seed token curve at a curve size for about 14% overhead.
   - Every size takes a 500 branch.
4. **After the 150M seeds,** one Titan card runs a separate 5M trunk to 20,000 tok/param, labeled fp16, with branches at 2,000 and 5,000 that overlap the bf16 ones. This is for the writeup, not for setting budgets.
5. **P-050 costs no training.** Score SmolLM2-135M's public checkpoints, which run from 251.7B to 2.013T tokens and are pre-decay until 1.6T ([HF](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints)). Read the slope, not the level.

**Detectability.** RC-12's minimum detectable effect with 3 seeds is about 3-4 points (`research/followup/loop.verify.md`). An effect the size of MobileLLM's per-doubling gain is invisible one doubling at a time. Token decisions therefore use the slope of continuous margins fitted across 3 or more branches, not pairwise pass rates.

### 5.6 What each extra doubling is expected to buy

Loss gains are from the three web-text fits with total parameters (est.). They are extrapolated below 70M and overstate gains at extreme ratios.

| doubling | 3-seed cost (est.) | loss gain (nats) | as a share of the step to the next size at equal tok/param | benchmark analogue |
|---|---|---|---|---|
| 5M, 2,000 to 4,000 | about 80 h | 0.085-0.104 | 0.18-0.26 | about +0.5-0.7 points (MobileLLM 125M-350M) |
| 10M, 2,000 to 4,000 | about 220 h | 0.072-0.086 | 0.17-0.25 | same |
| 10M, 5,000 to 10,000 | 500 h (357-714) | 0.057-0.066 | 0.13-0.21 | same |
| 20M, 1,000 to 2,000 | 308 h | 0.072-0.086 | 0.34-0.45 (20M to 30M is a small step) | same |
| 30M, 1,000 to 2,000 | 545 h | 0.065-0.076 | 0.20-0.26 | same |
| 60M, 500 to 1,000 | 698 h on the 5070, or about 14 days of 4 Titans | 0.065-0.076 | 0.20-0.23 | same |
| 150M, 500 to 1,000 (1 seed) | 19-30 days of 4 Titans | 0.049-0.059 | n/a | same |

**Tokens do not replace size.** By the same fits, a 30M needs 22-630x its 500 tok/param budget to match a 60M at 500.

**On RC-12 nothing is known yet.**
- A threshold skill may jump, or may not move.
- A skill below its critical density will not move however many doublings are added. The lever for that is the share of skill-dense chat, not tokens.

---

## 6. Proxy experiments that decide the mix

### 6.1 Metrics

All metrics are fixed-window bits per byte, which does not depend on the tokenizer:
- **(a)** held-out assistant turns of dev renders from the frozen pipeline (not RC-12);
- **(b)** held-out human conversational turns: an OASST2 dev slice disjoint from OOD-H, plus Topical-Chat, task dialogue and IRC. **OOD-H is never used for mix decisions**;
- **(c)** planted-fact answer spans (copying from context);
- **(d)** guards: held-out strict-open prose, and held-out pre-2023 web text.

At 20M and above, after the Oct 11-14 lock, RC-12 dev Tier 0 margins are added.

### 6.2 Why these metrics and decision rules

- Switching a benchmark to bits per byte raised decision accuracy from 68.3% to 95.3% ([arXiv 2508.13144](https://arxiv.org/html/2508.13144v1)).
- A single-scale ranking at 150M picks the better 1B recipe about 80% of the time, and continuous metrics predict better ([DataDecide](https://arxiv.org/abs/2504.11393)).
- Mixing methods often fail to beat stratified sampling, by up to 6.9 perplexity points ([Aioli](https://arxiv.org/abs/2411.05735)). So a simple baseline arm is always kept.
- Filtering by classifier does not necessarily lower loss on the high-quality set itself (Apple). So bits per byte and battery margins are reported as separate decisions.

### 6.3 The experiments, in order

Costs are 5070-hours (est.).

| order | id | question | size and tokens | runs | decision rule | window | 5070 h |
|---|---|---|---|---|---|---|---|
| 1 | E-sel-0 | Are the classifiers valid? | no training | 60k labels; 3k double-labeled | the gates in 3.4 | Oct 5-12 | 5-15 |
| 2 | P-187 | How many passes can chat take? | 5M at about 1B tokens, plus a 20M check | 20 + 2 | the largest pass count within seed noise of 10 passes; the 20M check must agree | Oct 19 - Nov 1 | 35-65 |
| 3 | E-sel-1 | Weights for the general domains (the P buckets, W, N, HD sources), with chat fixed at 25% | 5M at 0.25B (past the induction transition) | about 55, Dirichlet around the hand prior | LightGBM per metric. Minimize (a)+(b)+(c) subject to (d) no more than 3% worse than the natural mix. Output the top 3 mixes | Oct 26 - Nov 8 | 20-40 |
| 4 | E-sel-4 | Which human-dialogue sources, RH variants and narrative sources pay off? | 30M trunk; decay branches over the last 10% with 30% candidate data | about 8 candidates x 1-2 seeds | ranked by change in (a), (b) and margins; feeds E-sel-1's prior and P-058 | Oct 26 - Nov 8 | 10-20 |
| 5 | E-sel-2 + E-src | Is selection worth it, which kind, and does W help? | 20M at 1B, 2 seeds | 6 arms: hygiene only, DSIR, CU curve, swarm best, hand prior, swarm best + W (only if Q1 is yes). CL discard swept at 20% and 40% | the swarm mix replaces the hand prior only if it wins by more than 2 seed SDs on both seeds | Nov 9-22 | 40-65 |
| 6 | E-sel-3 | Knowledge density (extends P-083) | 20M at 1B, 2 seeds | 4 arms (3.6) | adopt if (a)-(c) and margins are no worse and lookup-reading items (P-084/085) improve; closed-book is reported, never optimized | Nov 9-22 | 25-45 |
| 7 | P-162 + P-165 | Paraphrase vs repeat; grounded vs high-quality vs QA rephrase | 5M, then 30M at the curve batch size | 2 + 3 arms | the paraphrased arm must beat exact repeats at equal tokens; the best rephrase type gets the RH share | Nov 9-22 | 15-30 |
| 8 | E-sel-6 | Source tags (P-161) | 5M, then a 30M branch | 2-3 seeds | keep if (a)-(c) improve with tags dropped in the final decay | Nov 9-22 | 5-10 |
| 9 | E-sel-5 | Does the ranking survive overtraining? | 5M to 2,000 tok/param on the P-186 trunk, plus a hygiene-only trunk | 2 trunks | if the gap closes by 2,000, flatten curves for long runs; if it widens, allow 7x at 30M and under | Oct 19 - Dec 3 | 15-28 |
| 10 | P-186 | The token slope | 5.5 | 2 seeds | 5.5 | Oct 19 - Dec 3 | 35-75 |
| 11 | 3f | Combined recipe | 30M, 3 seeds (PLAN) | | PLAN | Nov 26 - Dec 3 | (in PLAN) |
| | **total** | | | | | | **about 205-395** |

- This program replaces or absorbs ledger items P-083, P-161, P-164, P-165 and P-162, and part of P-058. P-186 was already listed at 40-80 h.
- **Net new load is about 100-230 5070-hours (est.).**
- **Proxy trust:**
  - RegMix's 1M proxies ranked mixtures for 1B at Spearman 97.12 on loss, and proxy tokens saturated around 0.25B ([RegMix](https://arxiv.org/html/2407.01492v2)).
  - OLMo 3 used 30M proxies at 3B tokens.
  - CLIMB's small-proxy result does not apply here, because its proxies were first pretrained on 10T tokens.
  - A generative pass rate is trusted only at 20M and above, and only for effects of 5 points or more.
- **Honest limit.** Gains of a few percent sit below the minimum detectable effect of a 2-3-seed comparison (`research/followup/loop.md` section 5). Small steps are adopted only where bits per byte supports them, and their sum is checked in 3f, not assumed.

---

## 7. Build schedule, disk and GPU budget

### 7.1 Schedule, fitted to the PLAN phases

| dates | PLAN phase | corpus work | machine |
|---|---|---|---|
| Fri Sep 25 | 0 | Max's rulings Q1-Q8 | none |
| Sep 25 - Oct 4 | 0 | Code: provenance and date gate, hygiene, hashed 13-gram decontamination, DSIR, the feature scorer, the bucket writer, the mix sampler with the R2 loader, the diversity battery. Rubrics for CU, FD, RS and CL. A mutation test for every stage | Mac CPU (code only; the current experiment keeps the GPU) |
| Oct 1-8 | 1 | Convert the human-dialogue sets (small); run the DPI entry audit; stream a 1-2B-token strict-open sample through the gates | PC CPU |
| Oct 5-8 | 1 | Expanded teacher pilot (4.7) | 5070 |
| Oct 5-12 | 1 | E-sel-0 labels (Gemma, while it is loaded); Qwen3.5-9B relabels 3k | 5070 |
| Oct 9-14 | 1-2 | fastText classifiers and gates; score the sample | CPU |
| Oct 11-14 | 0 | Pre-registration lock (PLAN); RC-12 n-gram hashes handed to the pipeline | none |
| by Oct 15 | 1 | Tokenizer training sample: chat pilot + OASST2 (minus OOD-H) + strict-open sample after hygiene and the CL cut. **This replaces PLAN's FineWeb-Edu slice** | CPU |
| Oct 18 | 1 | Tokenizer v1 frozen. Its nested truncations serve E7 | none |
| Oct 9-18 | 2 | Pilot chat set, R0 20-30M | 5070 |
| Oct 19 - Nov 1 | 2-3 | Strict-open core v0: stream the selected Common Pile subsets, gate, clean, score, bucket; store compressed text shards; HD v1; P-187 | PC CPU; 5070 |
| by Nov 1 | 2 | General text for 10-60M ready (PLAN date), built from the core, not FineWeb-Edu | PC |
| Oct 19 - Nov 8 | 2 | R0 to 100M after the P-012/P-053 gate; R1 x1; RH 30-50M | 5070 |
| Oct 26 - Nov 8 | 3b | E-sel-1 swarm; E-sel-4 anneal branches | 5070 |
| Nov 8 | 2 | Rate check: sets the R0 : R1 : RH split and drafts the growth rule | none |
| Nov 9-22 | 2-3 | R0 to 150-300M; R1 x2; RH to 100-150M. E-sel-2/E-src, E-sel-3, P-162/165, E-sel-6 | 5070 |
| Nov 21-22 | 3 | Lambda kickoff (Max, remote): rerun the frozen, deterministic pipeline on the same sources, with hashes checked against the PC's manifest, plus the W bucket if Q1 is yes. 150M text ready by Dec 14 | Lambda CPUs at nice 19, Ultra thermals watched |
| Nov 26 - Dec 3 | 3f | The combined-recipe test uses mix v1 | 5070 |
| Dec 4 | freeze | Pool v1 and per-size mix manifests hashed; growth rule and pass caps pre-registered | none |
| Dec 4-14 | 4 | 150M-only settings: chat share, and the data for the T=4096 phase | none |
| Dec 4 - Feb 21 | 4 | The curve (5.3). R1/RH growth on Mac nights if Q7 is yes | 5070, Titans, Mac |

### 7.2 GPU budget

5070-hours, all (est.):

| item | window | 5070 h |
|---|---|---|
| Teacher pilot, expanded (+E4B GGUF, llama.cpp Gemma, 8B paraphraser, MTP arms) | Oct 5-8 | 15-30 |
| E-sel-0 labels | Oct 5-12 | 5-15 |
| R0 renders | Oct 9 - Nov 22 | Ministral 107-300 for 300M; Gemma 259-1,042 |
| R1 x2 of R0 | Oct 19 - Nov 22 | 50-120 with a 4B model; more with 8B |
| RH, 100-150M | Oct 19 - Nov 22 | 17-75 |
| **Generation cap** | through Nov 22 | **about 350.** If R0 + R1 + RH would exceed it, R0 shrinks first, but never below 150M |
| Selection and pool proxies (section 6, without 3f) | Oct 19 - Dec 3 | 205-395, of which 100-230 is net new |

**Capacity check, Oct 12 - Dec 3.**
- PLAN puts the 5070 at 580-990 of about 1,270 hours.
- This design adds about 130-320 net hours: proxies, RH, the extra pilot arms.
- That is 710-1,310 hours, so the high end does not fit.

**Cut order if short:**
1. E-sel-6 and PreSelect.
2. E-sel-3 to 2 arms.
3. P-187's 20M check moves onto the first 20M curve seed's branches.
4. R0 drops to 150-200M, with R1 x3.

**Titans:** 150M seeds as in 5.3. Generation on the Titans only after IT approval and a driver of 580 or later.

**Mac:** eval (PLAN), plus Q7 night-shift paraphrasing after the freeze. It never runs alongside another model process.

### 7.3 Disk budget

| machine | free now | holds | needs (est.) |
|---|---|---|---|
| Mac | about 126 GB (task brief) | code, rubrics, metadata, classifier files (<1 GB), sample shards of 5 GB or less | 10 GB or less. No raw corpora. The screening subset (30-50B tokens, about 120-200 GB raw) is streamed on the PC instead |
| PC | unknown (check on the first SSH) | teachers and paraphrasers (about 45 GB); HD (under 5 GB); the strict-open core as zstd text (25B tokens x about 4.2 bytes, about 105 GB raw, about 30-35 GB compressed); tokenized shards for up to 2 vocabularies (about 50 GB each); the chat pool with slot spans and provenance (10-20 GB); checkpoints (30-60 GB); streaming scratch (about 50 GB) | **about 250-300 GB free, up from PLAN's 150** |
| Lambda | about 3.1 TB (`research/lanes/compute.md`) | streamed raw Common Pile subsets (about 570 GB raw without wikiteam's 437.5 GB); the W bucket's TB-scale stream if Q1 is yes; the 150M tokenized mix (20-40B unique x 2 bytes, about 40-80 GB) | fits. Download time is unmeasured, and so is CPU contention with Ultra |

Home bandwidth to stream about 570 GB to the PC is unmeasured. At 50-100 Mbit/s that is about 13-25 hours (est.).

### 7.4 PLAN edits this implies

1. Replace "readability-filtered FineWeb-Edu" in the Phase 2 mix v0, the baseline recipe, the tokenizer training sample and the Lambda 150M prep with the strict-open core plus Planck selection.
2. Move FineWeb-Edu from "allowed" to "excluded (Q5)" in the D8 audit table. W stays conditional on Q1.
3. Teacher gate: accepted Planck tokens per 5070-hour at realistic request lengths with fp8 KV. Add the E4B GGUF, llama.cpp Gemma, 8B paraphraser and MTP arms.
4. Chat pool: the R0/R1/R2/RH architecture, the growth rule, P-187, and a 350-hour generation cap.
5. P-186: the cost fix and the redesign (5.5).
6. New ledger rows: E-sel-0 to E-sel-6, E-src and P-187. Mark P-083, P-161, P-164 and P-165 as absorbed.
7. Curve: 5M and 20M points with the budgets in 5.3; 3M conditional.
8. PC disk: about 250-300 GB free.

---

## 8. Risks and open questions

| risk | early sign | response |
|---|---|---|
| The strict-open core yields less than 15B after the date gate and selection | Measured keep rates on the Oct 1-8 sample | Loosen the CL cut to 20%. Add wikiteam. Ask Max about W (Q1). Accept 6-8 repeats at 150M |
| The CU classifier learns "chat-shaped", not "useful" | E-sel-0 mutation test (a) fails; poor agreement with Qwen | Drop CU. Fall back to DSIR plus CL |
| Selection helps bits per byte but not the battery | E-sel-2: (a)/(b) improve, margins flat | Adopt only the cheapest arm within noise. Report it as a form gain, not a skill gain |
| Gemma wins on naturalness but is 3x slower | Pilot | Gemma renders the hardest cells; Ministral renders the rest; more R1 |
| Yield under 30% on correction or long-distance cells | Pilot yield by cell | Teacher-written templates for those cells (P-065); single-turn repair |
| Planted facts get memorized from repeated chat | P-187's no-context fact probe rises | Turn R2 on everywhere. Cut the pass cap. Raise weight decay |
| Narrow register makes the model brittle on human input | (b) and OOD-H style dev bits per byte worsen as narrative and chat shares rise | Raise the HD and P shares. Diversity alarms are two-sided |
| Post-2022 model text in "human" sources | Date-gate losses, or detector spot checks | Stricter gate; record the residual in the provenance table |
| The 5M swarm ranking does not hold at 60-150M | E-sel-2 at 20M disagrees with the swarm | Hand prior plus the 20M winner. There is no 150M-scale check before Dec 14 (open) |
| NorMuon overhead makes every 5070 figure 1.5x worse | `bench_micro` | Re-plan rule (5.3); the extension queue shrinks first |
| Late saturation at widths under 1,000 | A later decay branch scores worse | Stop extending that size; spend on seeds |
| Overtrained models fine-tune worse | Post-SFT branch scores flatten while base scores rise | Budget by post-SFT scores; lower the SFT learning rate |
| PC disk or bandwidth short | First SSH check | Build the core on Lambda earlier (needs Max) |
| Grey or conflicting licenses surface later | Source audits | The provenance record allows a surgical removal and a rebuild of the affected buckets only |

**Open questions:**
1. Can a 12B teacher label conversational usefulness reliably? WebOrganizer refined its 8B labels with 405B labels. Planck's only checks are Qwen agreement and Claude's rubric reads.
2. What are the real sizes of MultiDoGO, WikiConv, the congressional hearings and Gutenberg plays, and are their licenses as expected?
3. Does training-only entity anonymization help when the model never sees anonymized input at inference?
4. Does R1 still pay at the curve's batch sizes, given the auxiliary-views batch-size caveat?
5. What is the true Planck-token multiplier on general prose? It is at least 1.14x on chat.
6. What fraction of CCCC and StackExchange survives the Dec 2022 date gate?
7. Does the tokenizer, frozen on Oct 18 from a provisional sample, lose measurable compression on the final mix? Answer by measuring bytes per token on the frozen mix at Dec 4.
8. Which sizes does Max want at 3 seeds if the re-plan rule triggers: the ≤30M points or the 60M anchor?

---

## 9. Sources

**Corpora and cards**
- Common Pile: https://arxiv.org/abs/2506.05209
- Comma card: https://huggingface.co/datasets/common-pile/comma_v0.1_training_dataset
- Common Pile subsets: https://huggingface.co/datasets/common-pile/cccc, /project_gutenberg, /library_of_congress, /wikimedia, /stackexchange, /youtube, /wikiteam, /ubuntu_irc, /uk_hansard, /usgpo, /data_provenance_initiative
- DPI list: https://github.com/r-three/common-pile/blob/main/sources/data_provenance/include.csv
- FineWeb: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- FineWeb paper: https://arxiv.org/abs/2406.17557
- DCLM: https://arxiv.org/abs/2406.11794
- DCLM dedup: https://huggingface.co/datasets/Zyphra/dclm-dedup
- Essential-Web: https://arxiv.org/abs/2506.14111 and https://huggingface.co/datasets/EssentialAI/essential-web-v1.0
- Nemotron-CC-v2: https://huggingface.co/datasets/nvidia/Nemotron-CC-v2
- SYNTH: https://huggingface.co/datasets/PleIAs/SYNTH
- Common Corpus: https://huggingface.co/datasets/PleIAs/common_corpus and https://arxiv.org/pdf/2506.01732
- Institutional Books: https://huggingface.co/datasets/institutional/institutional-books-1.0
- DeGenTWeb: https://arxiv.org/abs/2605.00087
- AI text in Wikipedia: https://arxiv.org/abs/2410.08044

**Dialogue data**
- OASST paper: https://arxiv.org/abs/2304.07327
- OASST2 stats: https://datasets-server.huggingface.co/statistics?dataset=OpenAssistant/oasst2&config=default&split=train
- Dolly: https://datasets-server.huggingface.co/statistics?dataset=databricks/databricks-dolly-15k&config=default&split=train
- Aya: https://datasets-server.huggingface.co/statistics?dataset=CohereLabs/aya_dataset&config=default&split=train
- MultiWOZ: https://github.com/budzianowski/multiwoz
- Taskmaster: https://github.com/google-research-datasets/Taskmaster
- SGD: https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
- MultiDoGO: https://aclanthology.org/D19-1460/
- AirDialogue: https://huggingface.co/datasets/google/air_dialogue
- Topical-Chat: https://github.com/alexa/Topical-Chat
- QuAC: https://quac.ai/
- CoQA: https://stanfordnlp.github.io/coqa/
- Gutenberg Dialogue: https://github.com/ricsinaruto/gutenberg-dialog
- MSC: https://arxiv.org/html/2107.07567
- ConvAI2: https://arxiv.org/html/1902.00098v1
- WoW: https://arxiv.org/abs/1811.01241
- HH-RLHF: https://huggingface.co/datasets/Anthropic/hh-rlhf
- HelpSteer: https://huggingface.co/datasets/nvidia/HelpSteer
- ProsocialDialog: https://huggingface.co/datasets/allenai/prosocial-dialog
- AgentInstruct: https://huggingface.co/datasets/THUDM/AgentInstruct
- SmolTalk: https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- SmolTalk2: https://huggingface.co/datasets/HuggingFaceTB/smoltalk2
- SmolTalk pipelines: https://github.com/huggingface/smollm/tree/main/text/data/smoltalk
- TalkBank rules: https://talkbank.org/0share/rules.html
- TED policy: https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy
- Santa Barbara Corpus: https://www.linguistics.ucsb.edu/research/santa-barbara-corpus-spoken-american-english
- Reddit lawsuit: https://natlawreview.com/article/beyond-copyright-reddits-lawsuit-against-anthropic

**Synthetic data and seeds**
- Cosmopedia: https://huggingface.co/datasets/HuggingFaceTB/cosmopedia
- SmolLM corpus: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- Nemotron-Personas-USA: https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA
- Tatoeba stats: https://tatoeba.org/en/stats/sentences_by_language
- flan-ul2-tinystories: https://huggingface.co/datasets/crumb/flan-ul2-tinystories
- Model licenses: https://huggingface.co/api/models/<id> for Gemma 4 12B and E4B, Qwen3.5-2B/4B/9B, Ministral-3-8B-Instruct-2512, SmolLM3-3B, Qwen2.5-72B-Instruct, gpt-oss-120b, whisper-large-v3

**Selection**
- Bitter Lesson for Data Filtering: https://arxiv.org/html/2605.19407v1
- Removing Noise, not Finding Gold: https://arxiv.org/abs/2510.00866
- OLMo 3: https://arxiv.org/abs/2512.13961
- QuRating: https://arxiv.org/abs/2402.09739
- Meta-rater: https://arxiv.org/abs/2504.14194
- WebOrganizer: https://arxiv.org/abs/2502.10341
- Register study: https://arxiv.org/abs/2504.01542
- Ultra-FineWeb: https://arxiv.org/abs/2505.05427
- DSIR: https://arxiv.org/abs/2302.03169
- PreSelect: https://arxiv.org/abs/2503.00808
- Perplexity pruning: https://arxiv.org/abs/2405.20541
- Deduplication: https://arxiv.org/abs/2107.06499
- SemDeDup: https://arxiv.org/abs/2303.09540
- D4: https://arxiv.org/abs/2308.12284
- Repeated data: https://arxiv.org/abs/2205.10487
- Data filtering scaling laws: https://arxiv.org/abs/2404.07177
- KLLM: https://arxiv.org/abs/2607.12831
- LMLM: https://arxiv.org/abs/2505.15962
- Cram Less to Fit More: https://arxiv.org/abs/2604.08519
- Physics of LMs 3.3: https://arxiv.org/abs/2404.05405
- MeCo: https://arxiv.org/abs/2501.01956
- Readability is not Learnability: https://arxiv.org/abs/2510.13915
- Llama 3 license: https://raw.githubusercontent.com/meta-llama/llama3/main/LICENSE

**Mixing and proxies**
- RegMix: https://arxiv.org/abs/2407.01492
- Aioli: https://arxiv.org/abs/2411.05735
- CLIMB: https://arxiv.org/abs/2504.13161
- DataDecide: https://arxiv.org/abs/2504.11393
- Signal and noise: https://arxiv.org/abs/2508.13144
- Tiny-LR proxies: https://arxiv.org/abs/2512.24503

**Synthetic scale**
- Consumer Blackwell vLLM benchmarks: https://arxiv.org/html/2601.09527v1
- vLLM issues: https://github.com/vllm-project/vllm/issues/38918, /45238, /55766, /40696, /47749
- vllm-metal: https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0
- Checkpoints: https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct, https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf, https://huggingface.co/AxionML/Qwen3.5-9B-NVFP4, https://huggingface.co/Firworks/Ministral-3-8B-Instruct-2512-nvfp4
- APIGen: https://arxiv.org/abs/2406.18518
- DeepDialogue: https://arxiv.org/abs/2505.19978
- Beyond Model Collapse: https://arxiv.org/abs/2406.07515
- TinyStories: https://arxiv.org/abs/2305.07759
- Syntactic templates: https://arxiv.org/abs/2407.00211
- Diversity scores: https://arxiv.org/abs/2403.00553
- Artificial Hivemind: https://arxiv.org/abs/2510.22954
- Verbalized Sampling: https://arxiv.org/abs/2510.01171
- min-p: https://arxiv.org/abs/2407.01082
- Synthetic data diversity: https://arxiv.org/abs/2410.15226

**Repetition and rephrasing**
- Data-constrained LMs: https://arxiv.org/abs/2305.16264
- TII memorization window: https://arxiv.org/abs/2607.04969
- Falcon-H1-Tiny: https://tiiuae-tiny-h1-blogpost.hf.space/
- Transient in-context learning: https://arxiv.org/abs/2311.08360
- Pre-training under infinite compute: https://arxiv.org/abs/2509.14786
- MGA: https://arxiv.org/abs/2502.04235
- Auxiliary views: https://arxiv.org/abs/2609.04180
- WRAP: https://arxiv.org/abs/2401.16380
- Kang et al.: https://arxiv.org/abs/2510.01631
- BeyondWeb: https://arxiv.org/abs/2508.10975
- MIND: https://arxiv.org/abs/2410.12881
- Dialog Inpainting: https://arxiv.org/abs/2205.09073
- Model collapse and accumulation: https://arxiv.org/abs/2404.01413

**Token budget**
- Beyond Chinchilla-Optimal: https://arxiv.org/abs/2401.00448
- Chinchilla replication: https://arxiv.org/abs/2404.10102
- Scaling with over-training: https://arxiv.org/abs/2403.08540
- MobileLLM: https://arxiv.org/abs/2402.14905
- Softmax-bottleneck saturation: https://arxiv.org/abs/2404.07647
- Overtrained models fine-tune worse: https://arxiv.org/abs/2503.19206
- Emergence and loss: https://arxiv.org/abs/2403.15796
- Critical mixing ratio: https://arxiv.org/abs/2505.18091
- Entity tracking in Pythia: https://arxiv.org/abs/2608.18083
- Scaling laws from language statistics: https://arxiv.org/abs/2602.07488
- WSD and cooldown: https://arxiv.org/abs/2405.18392
- Power Lines: https://arxiv.org/abs/2505.13738
- Reusing overtrained models: https://arxiv.org/abs/2510.06548
- SmolLM2: https://arxiv.org/abs/2502.02737
- SmolLM2-135M checkpoints: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints
- LFM2.5-230M: https://www.liquid.ai/blog/lfm2-5-230m
- Llama 3: https://ai.meta.com/blog/meta-llama-3/

**Internal (read-only)**
- `PLAN.md`
- `LEDGER.md`
- `research/REPORT.md`
- `research/lanes/data.md`
- `research/lanes/compute.md`
- `research/followup/chatdata.md`
- `research/followup/tokenizer.md`
- `research/followup/loop.md`
- the `.verify.md` file for each of these

Budget and fit arithmetic scripts: `(scratch file)` and `fits.py`.


---

## Addendum (Sep 24, after the priority change)

This design was drafted before Max made models of 30M and under the main target. Two updates follow.

**1. Rulings adopted.** Max delegated these calls ("do everything you recommended"), so the recommended defaults in 1.4 are adopted: Q1 Reading B for the headline core (no web-crawl bucket W); Q2 yes, labeled; Q3 no for v1; Q4 eval and baselines only; Q5 not used, Planck trains its own classifiers; Q6 date gate at 2022-12-01; Q7 yes, after the freeze; Q8 BY-SA rephrasings released under BY-SA. Q1 costs nothing for the small models: the strict open prose core (about 15-25B tokens after selection, est.) covers every budget below at 4 repeats or fewer. Only a later 150M extension would need the W bucket or more repeats, and that decision can wait.

**2. Token budgets rebalanced toward the small models (est., to be replaced by P-186 and bench_micro).** The 5070's curve window (about 1,700 hours) now goes mostly to 30M and under; 60M and 150M become upper anchors on the Titans after mid-December (1 seed each unless time allows more).

| size | budget (tok/param) | tokens per seed | seeds | 5070 hours, 3 seeds (central est.) |
|---|---|---|---|---|
| 3M (if 5M passes a Level A family) | 10,000-20,000 | 30-60B | 3 | 100-200 |
| 5M | 5,000 | 25B | 3 | ~150 |
| 10M | 5,000 | 50B | 3 | ~400 |
| 20M | 2,000 | 40B | 3 | ~500 |
| 30M | 1,000-2,000 | 30-60B | 3 | ~430-860 |
| 60M, 150M | as the Titans allow | | 1+ | Titans |

The totals land at roughly 1,600-2,100 hours depending on the 3M and 30M choices, so the final split follows the measured rates and P-186's measured gain per doubling: whichever size gains most per GPU-hour gets the extension time. Rates are the section 5.3 estimates (plus or minus 2x), so every number here moves once bench_micro runs on the PC.

## Addendum, 2026-09-25 night: tokenizer v0 and the starter corpus

**Built.** `corpus/` (extractor, hygiene, IRC speaker mapping, boilerplate removal, OOD-H reserve, tokenizer
sample) and `tokenizer/` (nested byte-level BPE, 2k to 32k, health checks, P-097/P-098/P-101). Tokenizer v0 was
trained on a 719 MB sample of the 3.9 GB starter download; files and results are in `tokenizer/v0/`. It is
provisional: v1 freezes on a larger sample.

**Results.** P-098 holds exactly: the 8k cut from the 32k model is byte-identical to a separately trained 8k,
so one nested tokenizer serves the whole curve. P-101 fails its 5% bar (best 4.4% with real added tokens), so no
superword tokens. At 8k, OASST2 held-out text is 3.71 bytes per token.

**Ruling on CCCC (Claude, 2026-09-25, under Max's standing permission; reversible).** Section 3.1 assumed a
per-document license. The release has none: 0 of 15,000 sampled documents across all 10 snapshots carry a
license field. Common Pile selected CCCC pages by detecting a CC license on each page and reviewing the top
1,000 domains by hand (537 kept). CCCC is admitted with the provenance basis "dataset-level: Common Pile
page-level CC detection and domain review; per-document license not recorded", and `make_tok_sample.py` refuses
it unless `--allow-unrecorded-license cccc` is passed, so every use is explicit and recorded in the manifest.
The model card will say this. If Max rules otherwise, CCCC leaves the mix and the P bucket's web share moves to
Gutenberg, LoC books and Wikimedia.

**Open before any model trains on this corpus (the tokenizer does not depend on them):**
- StackExchange is dated by the question only; later answers can be newer than the date gate. The fix is the
  official Stack Exchange dump with per-post dates (the AI-ism and year filter drops the obvious cases).
- Near-dedup is not built. Boilerplate removal (lines on 10+ distinct pages) also removes some real page content
  repeated across URLs; near-dedup should run first, as section 3.2 orders.
- OOD-H Part 1 must skip the 11 reserved OASST2 trees whose user turns appear verbatim elsewhere
  (`corpus/oodh.leaked_reserve_trees()`), and the reserve holds only 376 English ready threads with 2+ user
  turns (14 with 3+). If Part 1 needs 150 longer threads, the reserve rule changes before tokenizer v1.
- Dolly and OASST2 are dated 2023 and exempt from the date gate (section 2.2); both now pass the AI-ism filter.
