from q9_probe import load, RS, RL, has_tpl, SEEDS
import re
from collections import Counter
models = [("E005", "base")] + [("E005", t) for t in SEEDS] + [("E004", t) for t in SEEDS]
print("\nfirst sentence of each turn; same-as-previous-turn reply; 'latest plan' phrase")
print(f"{'model':10} {'1st=tplS':>8} {'1st=tplL':>8} {'sameprev':>9} {'latestplan':>10} {'distinct_first/turns':>21} {'mean_words_1st':>14}")
for exp, t in models:
    C = load(exp, t)
    n = s1 = l1 = same = lp = 0
    firsts = []
    wsum = 0
    for c in C:
        prev = None
        for tt in c["turns"]:
            a = tt["assistant"].strip()
            first = re.split(r"\n|(?<=[.!?])\s", a, maxsplit=1)[0].strip()
            firsts.append(first)
            wsum += len(first.split())
            n += 1
            s1 += bool(re.fullmatch(r".*", first)) and has_tpl(first, RS)
            l1 += has_tpl(first, RL)
            lp += "latest plan" in a.lower()
            if prev is not None:
                same += first == prev
            prev = first
    nprev = n - len(C)
    print(f"{exp + ' ' + t:10} {s1 / n:8.2f} {l1 / n:8.2f} {same:4d}/{nprev:3d} {lp / n:10.2f} {len(set(firsts)):>10}/{n:<10} {wsum / n:14.1f}")
