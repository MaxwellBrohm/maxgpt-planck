"""U_same / U_neutral control items, copied verbatim from research/capacity_probe/uprobe.py
(build() only, no model code), so the fine-tuned models are scored on the same items as REPORT 3.3."""
import random

import items as I


def build(n_scen=32, distances=(0, 4, 10), seed=99):
    rng = random.Random(seed)
    out = []
    for d in distances:
        for s in range(n_scen):
            for k in (1, 2, 3):
                days = rng.sample(I.DAYS, k + 1)
                fills = [I.distractors(rng, 1) for _ in range(k)] + [I.distractors(rng, d)]
                cands = {"gold": " " + days[k], "orig": " " + days[0]}
                if k >= 2:
                    cands["prev"] = " " + days[k - 1]
                # U_same
                turns = [(f"My dentist appointment is on {days[0]}.", f"Okay, your dentist appointment is on {days[0]}.")]
                for i in range(1, k + 1):
                    turns += fills[i - 1]
                    turns.append((f"Actually, my dentist appointment is on {days[i]} now.",
                                  f"Got it, your dentist appointment is on {days[i]}."))
                turns += fills[k]
                out.append(dict(task=f"Usame_k{k}", d=d, cands=cands,
                                prompt=I.transcript(turns, "What day is my dentist appointment?", "Your dentist appointment is on")))
                # U_neutral
                turns = [(f"I have a dentist appointment on {days[0]}.", f"Okay, noted: {days[0]}.")]
                for i in range(1, k + 1):
                    turns += fills[i - 1]
                    turns.append((f"Actually, the appointment got moved to {days[i]}.", f"Got it, moved to {days[i]}."))
                turns += fills[k]
                out.append(dict(task=f"Uneutral_k{k}", d=d, cands=cands,
                                prompt=I.transcript(turns, "Which day do I need to go to the dentist?", "That would be")))
    return out
