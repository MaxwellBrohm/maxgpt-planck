"""Bank pass (pipeline/BANKPASS.txt): the code that replaces every FAKE bank, pool and word list with teacher-written,
human-source or computed items stored with provenance. Built 2026-10-03 and 10-04 (W0 + W1). Nothing here has
generated a bank line: generation and the downloads wait for Max (BANKPASS s8), and every bank stays FAKE until a
frozen bank directory passes load.py's admit checks.

  specs       every bank's class, hole sets, word bounds and side, read off the FAKE banks
  store       item and call records (s2h), JSONL append with fsync, restart by done ids, manifest and bank refs
  plan        the seeded, deterministic stage P call list (s1 targets, s2i, author thirds)
  prompts     zero-shot bank prompts (no Claude-written example line) and the per-call decode spec (s2b)
  gen         runs a plan against a client; dry by default, restartable, 33-minute hold cap
  templatize  value-filled lines -> templates with holes (s2a)
  gates       item checks (s2c), held-out gates (s2d), BANK_PROMPT_ECHO, query/statement pair compatibility
  judge       cross-judging by the two non-author teachers on a fresh fill (s2e)
  itemize     call records -> item records (templatize, item checks, held-out gates)
  trim        dedup, frame caps, author thirds, p_exact (s2f, s2g)
  admit       per-bank admit checks (fail closed)
  load        loaders: a frozen bank directory -> the shapes banks.py, banks_keys.py and pools.py use; FAKE fallback
  wordlist    the controlled word list counter over core v0 (s4), PC CPU
  wordtable   the count table between wordlist and wordfam
  wordfam     families, POS proxy, drops, the RS / RM / RL lists, and value_features (article, plural, mass)
  wordstats   the word list CLI: TSVs, stats, sample stability, the full read's candidate vocabulary
  wordload    the word list loader: TSV with provenance -> families, lists, features, seeds; FAKE fallback
  mine        mined human example sentences from core v0 (s0), cited by doc id
  sources     W2: the five open sources, pinned, each licence checked at the source (evidence file), SOURCE.json
  human       W2: human pools (SSA names by era and sex, Census surnames, GeoNames cities with countries)
  humanseed   W2: persona seeds (Nemotron-Personas-USA, prompt side) and the LDNOOBW safety rubric list
  humanbuild  W2: sources -> one bank directory (banks + aux), its manifest and the admit result
  fixtures    test-only fixture banks and a fake bank teacher; mutate: the scratch mutation run
"""
