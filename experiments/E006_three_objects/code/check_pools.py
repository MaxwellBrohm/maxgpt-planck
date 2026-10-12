"""E004: which candidate values are ONE token (with a leading space) in all three tokenizer families.
Tokenizers only, no model. usage: check_pools.py > ../logs/check_pools.txt"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer
toks = {m: AutoTokenizer.from_pretrained(m) for m in
        ["HuggingFaceTB/SmolLM2-135M-Instruct", "roneneldan/TinyStories-1M", "EleutherAI/pythia-14m"]}
POOLS = {
 "day": "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(),
 "color": "red blue green yellow purple orange black white silver gray pink brown".split(),
 "month": "January February March April May June July August September October November December".split(),
 "city": "Paris London Rome Berlin Madrid Boston Dublin Chicago Vienna Denver Seattle Austin Toronto Lisbon Prague Tokyo Sydney Miami Atlanta Houston Phoenix Portland Dallas Oslo".split(),
 "dish": "pizza pasta tacos curry soup salad sushi chili stew burgers noodles lasagna pancakes risotto".split(),
 "sport": "tennis golf hockey rugby cricket baseball volleyball bowling swimming boxing skiing basketball football".split(),
 "number": "2 3 4 5 6 7 8 9 10 11 12".split(),
 "animal": "cat dog horse rabbit parrot turtle hamster goat pony".split(),
 "instrument": "piano guitar violin drums flute cello trumpet harp banjo".split(),
}
for k, vals in POOLS.items():
    ok, bad = [], []
    for v in vals:
        n = [len(t(" " + v, add_special_tokens=False)["input_ids"]) for t in toks.values()]
        (ok if max(n) == 1 else bad).append(v if max(n) == 1 else f"{v}{n}")
    print(k, "OK:", " ".join(ok)); print("   multi:", " ".join(bad))
