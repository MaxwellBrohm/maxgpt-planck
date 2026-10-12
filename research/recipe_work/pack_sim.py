"""Packing simulation of Max's SFTDataset (maxgpt-ultra/posttrain/sft_data.py), no model or tokenizer loaded.

SFTDataset concatenates ChatML-encoded conversations (EOS between them) and cuts fixed
seq_len windows at fixed offsets, with no boundary alignment and no cross-conversation
attention mask. For every supervised (assistant) token this script asks: did its own
conversation start in an earlier window (history cut off), or in the same window after
other text (an unrelated conversation is visible), or exactly at the window start (clean)?

Token counts are approximated as UTF-8 bytes / 4.36 (SmolLM2 49k BPE on OASST1,
lanes/arch.verify.md) plus ~3 header tokens and ~2 end tokens per message and 1 EOS.
Data: 1,100 UltraChat-200k train_sft rows from 11 offsets via the HF datasets-server.
Run: python3 pack_sim.py   (downloads ~7.5 MB of JSON into ./uc_cache/ on first run)
"""
import glob, json, os, random, statistics, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "uc_cache")
BPT = 4.36
URL = ("https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4/ultrachat_200k"
       "&config=default&split=train_sft&offset={off}&length=100")


def fetch():
    os.makedirs(CACHE, exist_ok=True)
    for off in range(0, 200001, 20000):
        p = os.path.join(CACHE, f"uc_{off}.json")
        if not os.path.exists(p):
            urllib.request.urlretrieve(URL.format(off=off), p)


def main():
    fetch()
    convs, asst_words = [], []
    for f in sorted(glob.glob(os.path.join(CACHE, "uc_*.json"))):
        for r in json.load(open(f))["rows"]:
            msgs = r["row"]["messages"]
            segs = []
            for m in msgs:
                segs.append((3, False))
                segs.append((max(1, round(len(m["content"].encode()) / BPT)) + 2, m["role"] == "assistant"))
                if m["role"] == "assistant":
                    asst_words.append(len(m["content"].split()))
            segs.append((1, False))
            convs.append((segs, len(msgs)))
    lens = [sum(n for n, _ in s) for s, _ in convs]
    print(f"conversations {len(convs)}; tokens/conv mean {statistics.mean(lens):.0f} median {statistics.median(lens):.0f}")
    print(f"messages/conv mean {statistics.mean(k for _, k in convs):.2f}; share with >=10 messages {sum(k >= 10 for _, k in convs) / len(convs):.2f}")
    print(f"assistant words/reply mean {statistics.mean(asst_words):.0f} median {statistics.median(asst_words):.0f}")
    for seq_len in (1024, 2048, 4096):
        res = {"history_cut": 0, "other_conversation_visible": 0, "clean": 0}
        for rep in range(5):
            order = convs[:]
            random.Random(rep).shuffle(order)
            pos = 0
            for segs, _ in order:
                start = pos
                for n, sup in segs:
                    if sup:
                        for t in range(pos, pos + n):
                            if t // seq_len > start // seq_len:
                                res["history_cut"] += 1
                            elif start % seq_len == 0:
                                res["clean"] += 1
                            else:
                                res["other_conversation_visible"] += 1
                    pos += n
        tot = sum(res.values())
        print(seq_len, {k: round(v / tot, 3) for k, v in res.items()})


if __name__ == "__main__":
    main()
