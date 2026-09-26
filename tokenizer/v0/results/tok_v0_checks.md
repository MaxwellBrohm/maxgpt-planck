# Tokenizer v0 CPU checks

Sizes: 2k, 4k, 8k, 16k, 32k. Held-out sources: cccc, dolly, gutenberg, irc, oasst2, stackexchange, wikimedia.

## Bytes per token on held-out, per source (higher is better)

| source | docs | MB | 2k | 4k | 8k | 16k | 32k |
|---|---:|---:|---:|---:|---:|---:|---:|
| cccc | 878 | 5.01 | 2.570 | 2.905 | 3.240 | 3.542 | 3.786 |
| dolly | 2854 | 2.77 | 2.785 | 3.196 | 3.593 | 3.960 | 4.240 |
| gutenberg | 31 | 5.06 | 2.745 | 3.064 | 3.362 | 3.616 | 3.824 |
| irc | 242 | 4.98 | 2.317 | 2.526 | 2.698 | 2.826 | 2.924 |
| oasst2 | 1094 | 2.17 | 2.881 | 3.305 | 3.710 | 4.035 | 4.279 |
| stackexchange | 1308 | 5.00 | 3.023 | 3.423 | 3.755 | 4.013 | 4.191 |
| wikimedia | 1317 | 4.99 | 2.722 | 3.079 | 3.418 | 3.721 | 3.951 |
| ALL | 7724 | 29.99 | 2.682 | 3.012 | 3.315 | 3.569 | 3.764 |

## P-098: nested-truncated vs separately trained, same sample

Size 8k. Vocab identical: True. Merges identical: True (7921 vs 7921).

| source | truncated bpt | separate bpt | truncated vs separate (%) |
|---|---:|---:|---:|
| cccc | 3.240 | 3.240 | 0.0000 |
| dolly | 3.593 | 3.593 | 0.0000 |
| gutenberg | 3.362 | 3.362 | 0.0000 |
| irc | 2.698 | 2.698 | 0.0000 |
| oasst2 | 3.710 | 3.710 | 0.0000 |
| stackexchange | 3.755 | 3.755 | 0.0000 |
| wikimedia | 3.418 | 3.418 | 0.0000 |
| ALL | 3.315 | 3.315 | 0.0000 |

Ledger default ('truncated 8k compresses about as well'): holds (identical files).

## P-101: chat-phrase superword tokens at a fixed vocabulary size

Base 8k; phrases mined from the training chat (dolly 8208 docs, irc 599 docs, oasst2 3338 docs); 4416392 distinct 2-4-word phrases. Each arm truncates the base by N merge-made tokens and adds the top N phrases, so the total stays the same. Gain = rise in bytes/token on held-out chat.

| N | ranking | pooled gain (%) | dolly gain (%) | irc gain (%) | oasst2 gain (%) | real added-token gain, pooled (%) | cost of dropping N merges (%) | phrase tokens used |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | freq | 2.72 | 3.42 | 2.27 | 3.24 | 2.58 | -0.13 | 87771 |
| 100 | saved | 2.69 | 3.40 | 2.24 | 3.15 | 2.54 | -0.13 | 83955 |
| 300 | freq | 4.05 | 4.23 | 3.83 | 4.53 | 3.87 | -0.37 | 132325 |
| 300 | saved | 3.96 | 4.12 | 3.76 | 4.42 | 3.78 | -0.37 | 123591 |
| 500 | freq | 4.63 | 4.54 | 4.59 | 4.88 | 4.41 | -0.70 | 156254 |
| 500 | saved | 4.51 | 4.43 | 4.48 | 4.71 | 4.28 | -0.70 | 144557 |

Most frequent phrases: `' of the'` (19855), `' in the'` (15138), `' to the'` (8406), `' is a'` (8170), `' on the'` (7329), `' and the'` (5666), `' is the'` (5642), `' for the'` (4768), `' to be'` (4689), `' in a'` (4223)

Bar (>= 5.0% on the specified frequency ranking): fails. Best frequency-ranked gain 4.63%; best of any ranking 4.63%. With real added tokens (which also fire inside words) the best frequency-ranked gain is 4.41%: fails.

## P-097: name tokenization across surface forms (200 common US names)

| size | 1 token ' Name' | 1 token 'Name' | 1 token ' name' | 1 token 'NAME' | mean tokens ' Name' | count differs | first id differs | first piece differs | ' Name' and 'Name' split alike |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2k | 0.005 | 0.000 | 0.010 | 0.000 | 3.34 | 0.995 | 1.000 | 0.495 | 0.905 |
| 4k | 0.020 | 0.000 | 0.045 | 0.000 | 2.89 | 0.945 | 1.000 | 0.605 | 0.730 |
| 8k | 0.160 | 0.005 | 0.055 | 0.000 | 2.42 | 0.915 | 1.000 | 0.725 | 0.540 |
| 16k | 0.375 | 0.065 | 0.070 | 0.000 | 1.88 | 0.945 | 1.000 | 0.820 | 0.430 |
| 32k | 0.775 | 0.215 | 0.100 | 0.000 | 1.29 | 0.990 | 1.000 | 0.935 | 0.355 |

Fractions of names. Definitions in tokenizer/checks_names.py.

## P-100

Out of scope for v0: P-100 compares the free 8k BPE with an 8k built from the teacher's vocabulary; the teacher is picked by the Phase 1 pilot, which has not run, so v0 cannot measure it.
