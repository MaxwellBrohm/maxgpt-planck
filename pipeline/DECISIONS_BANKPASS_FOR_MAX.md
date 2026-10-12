# Bank pass: what needs your call (2026-10-04)

The bank pass replaces the FAKE lists skeletons are built from (key lines, openings, names, cities, topics, word
lists) with text from the three Apache-2.0 teachers or open human sources, source recorded. W0 (code) and W1 (a word
list from our corpus) are done: no training data, no download, no model output. W2 on needs the four calls below.

## What W0 and W1 produced
- **Code** in `pipeline/bankpass/` (21 files). A bank folder is used only after every bank in it passes its checks
  (source, license, both judges, held-out gates, hashes). Anything FAKE or of an unknown type keeps the run marked
  FAKE. 57 tests pass, and they caught 100 of 100 deliberate code breaks.
- **Word list** on the PC (`~/planck/runs/bankpass/wordlist_v0/full/`), counted over all 9.3M core v0 documents:
  19,135 ranked word families, as RS (top 2,000), RM (top 5,000) and RL (all). Two 5% samples overlapped only 0.93
  at the top (the bar was 0.95), so the full count ran (65 min of PC CPU).
- **Mined examples**: 1,276 sentences real people wrote (YouTube CC-BY-4.0, OASST2 Apache-2.0), for prompts only,
  with any sentence holding a name removed.
- **Fixed today**: the loader silently skipped bank types it did not know; two word-family bugs ("water" was a form
  of "wat"); a part-of-speech rule; real names in the mined sentences.

## 1. Downloads: SSA first names, Census surnames, GeoNames cities, Nemotron-Personas-USA, LDNOOBW
- **Why**: names and cities fill most chats (919 name and 1,056 city slots per 1,000 chats, BANKPASS s1), but today
  they come from Claude-written FAKE lists of 113 names and 60 cities. Personas give each bank call its own voice.
  LDNOOBW replaces a 14-term FAKE safety list.
- **Options**: (a) all five; (b) names and cities only; (c) none: the teachers write names and cities too.
- **Cost**: PC CPU only. Sizes not checked (nothing downloaded). Licenses per BANKPASS: SSA and Census public
  domain; GeoNames, Nemotron-Personas-USA and LDNOOBW CC-BY-4.0 (LDNOOBW's to confirm at download).
- **Recommendation**: (a): the real spread of names and places, clean licenses; personas stay prompt-side only.

## 2. The real bank generation (stage P), and where its files live
- **What**: the three teachers write all line banks, pools and labels plus 1,000 topics with word sets, and judge
  each other's lines. Kept on the PC outside data/, trainable false until RC-12 decontamination clears.
- **Cost** (BANKPASS s6 estimate, not measured): about 2.8M output tokens; GPU time Qwen 35 min, Ministral 22,
  Gemma 96, in 7 lock holds, plus waits for the lock.
- **Options**: (a) go, with item files private on the PC and only manifests (hashes, counts, author shares) in the
  repo; (b) go, and commit the item files to the public repo; (c) wait.
- **Recommendation**: (a). Publishing items can wait until the pilot shows they work. The same goes for the word
  list file (derived counts, like corpus/data/common_words_en.txt): into the repo after the teachers label it.

## 3. The D8 reading: who may write bank text
- **As designed**: each bank is a third Qwen, a third Ministral, a third Gemma; a line is kept only when both other
  teachers judge it right; prompts carry no line Claude wrote, only mined human sentences cited by document id, and
  a bank line that copies a 6-word run from one is dropped.
- **Options**: (a) as designed; (b) no example sentences in prompts at all; (c) one teacher writes everything.
- **Evidence**: with (c), one teacher's own wording would fill the skeletons, which tilts the teacher pick toward
  it. The name gate removed 458,392 of 1.9M candidate example sentences (any capital after the first word).
- **Recommendation**: (a).

## 4. VOCAB_OOL gating, and p_exact for short banks
- **VOCAB_OOL** is the share of words outside the word list. Measured today on dry pilot 3's accepted chats: 3 to
  7% of words fall outside RM, and the most common ones are everyday chat words (goodbye, hobby, chore, aunt,
  coworker, puppy). The list ranks a word by how common it is across six kinds of text (speech, chat, prose,
  books, web, how-to), so words common only in chat rank low.
- **Options**: (a) report only, in the pilot; (b) gate on RM now; (c) gate once the topic word sets join the list.
- **Recommendation**: (a), then (c) once the pilot shows the rate per teacher. (b) counts goodbye against a chat.
- **p_exact**: a bank short of its target uses its exact lines less often (0.7 x kept / target) instead of
  repeating them. Options: on, or off (repeat up to the cap of 50 uses per 100k chats). **Recommendation**: on.

## One engineering change, made at W3 unless you object
- Part of speech: three agreeing teachers decide; the corpus guess only breaks a 2 to 1 split. The guess calls
  "people" and "information" adjectives, and under today's rule each such error bars a good word from required words.
