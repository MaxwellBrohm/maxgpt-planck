"""Curated FAILURES.md entries (rendered by make_failures.py, text pulled verbatim from transcripts)."""

HEADER = """# FAILURES: a gallery of multi-turn breakdowns in 90M-600M chat models

Every model line below is quoted verbatim from `transcripts/*__greedy.jsonl` (greedy decoding,
repetition_penalty 1.0, each model's own chat template, probed 2026-09-23 on an M5 Mac).
Replies longer than 420 characters are cut and the cut is marked; newlines are shown as ⏎.
Distractor turns are collapsed. "Forced-prefix probe" = the log-probability the model assigns
to the gold answer when its reply is started for it ("Your name is" -> " Priya"), computed on
exactly the same context; token rank 1 means greedy decoding from that prefix would say it.
Grades are the programmatic checks in `battery.py` (strict: gold present, no deflection, no
role capture).

The single most important pattern: **in most failures the fact is still retrievable** (forced
prefix ranks it first), but the model chooses a different kind of reply: a privacy/identity
deflection, an answer about itself instead of the user, or a copy of its own earlier text.
"""

ENTRIES = [
    {"section": "1. The fact is in context and retrievable, but the model deflects",
     "intro": "Deflection is the largest single failure class across the models (see the taxonomy in "
              "lanes/probe.md). The forced-prefix probe shows the information survived; the reply policy did not use it.",
     "items": [
         {"title": "Gemma 3 270M: gold is the top token, reply is an identity disclaimer", "model": "Gemma3-270M",
          "tid": "R_d4_city", "show": [0, 5],
          "why": "The city is recoverable at rank 1 (log-prob -0.19) four turns later, yet the model answers a question about "
                 "the user as if it were about itself. This is a post-training behavior, not a memory limit."},
         {"title": "LFM2.5-350M: 'I don't have access to personal information' about a fact the user gave two turns ago",
          "model": "LFM2.5-350M", "tid": "R_d2_number", "show": [0, 3],
          "why": "The newest, most heavily post-trained sub-400M model in the probe (28T tokens plus RL, per its card) "
                 "still deflects on a fact the user stated two turns earlier, while the forced-prefix probe still puts "
                 "'417' at or near the top."},
         {"title": "Qwen2.5-0.5B: the 'wall' model deflects too", "model": "Qwen2.5-0.5B", "tid": "R_d4_pet", "show": [0, 5],
          "why": "The 0.5B model Max treats as the smallest real chat model misreads 'my cat' as a question about the "
                 "assistant and deflects. Deflection is not a small-model artifact that disappears at 500M."},
         {"title": "SmolLM2-135M: deflection at distance 2 with the name ranked first", "model": "SmolLM2-135M",
          "tid": "R_d2_name", "show": [0, 3],
          "why": "At 135M the name is still the top continuation after 'Your name is' (log-prob -0.60). The failure is "
                 "the reply template the model reaches for, learned from privacy-style refusals."},
     ]},
    {"section": "2. Perspective errors: answering as the user, or about itself",
     "intro": "The model swaps 'you' and 'I': it adopts the user's name, job or pet as its own, or answers a question "
              "about the user with a statement about the assistant.",
     "items": [
         {"title": "Qwen3-0.6B: 'I'm Marcus.'", "model": "Qwen3-0.6B", "tid": "R_d6_name", "show": [0, 7],
          "why": "Retrieval worked (the name is right) but the speaker binding did not. A lenient grader that only "
                 "checks for the name would score this as a pass; ours does not."},
         {"title": "SmolLM2-135M: takes the sister's name as its own", "model": "SmolLM2-135M", "tid": "RI_binding",
          "why": "Two entities (user Oscar, sister Lena) and two speakers. At 135M the binding fails even in the "
                 "single-turn control (forced-choice margin Oscar vs Lena is negative in both)."},
         {"title": "LFM2.5-350M: asked what it does for work, it becomes the chef", "model": "LFM2.5-350M",
          "tid": "RI_identity",
          "why": "Role capture at 350M after one turn. The user's self-description becomes the assistant's persona."},
         {"title": "LFM2-350M: 'At the start, I was complaining about...'", "model": "LFM2-350M", "tid": "L_chitchat",
          "show": [1, 5],
          "why": "Asked what the USER complained about, the model invents a complaint of its own. The real answer "
                 "(being bored) is four turns back."},
     ]},
    {"section": "3. Corrections do not stick: the first-stated value wins",
     "intro": "Seven of eight models pass more single-turn controls ('Monday. Actually Tuesday. Which day?') than "
              "multi-turn versions with one distractor in between (SmolLM2-135M fails both). The forced-choice margin "
              "(corrected minus stale log-prob) is negative for K_time in all eight models.",
     "items": [
         {"title": "LFM2-350M: acknowledges green, answers blue", "model": "LFM2-350M", "tid": "K_color",
          "why": "The correction was acknowledged in the model's own reply, and it still reverts to the first value "
                 "two turns later."},
         {"title": "LFM2.5-350M: meeting moved to 4 pm, answer says 3 pm", "model": "LFM2.5-350M", "tid": "K_time",
          "why": "State update across turns is the ability that fails most uniformly in this probe, from 90M to 600M."},
         {"title": "SmolLM2-360M: 'I think your favorite color is blue'", "model": "SmolLM2-360M", "tid": "K_color",
          "why": "Same failure one size family up from 135M."},
     ]},
    {"section": "4. Copying its own earlier text (attractor states)",
     "intro": "Small models re-emit their own earlier reply, verbatim or nearly. This turns a single mistake into a "
              "persistent one, and makes a stale value or an invented persona sticky.",
     "items": [
         {"title": "SmolLM2-135M: final answer is a copy of its turn-0 reply, so the stale value comes back",
          "model": "SmolLM2-135M", "tid": "K_color",
          "why": "The 'correction failure' here is really a copy failure: the reply to the last question is a nearly "
                 "verbatim re-emission of the reply to the first message."},
         {"title": "SmolLM2-135M: an invented persona ('a dental AI') persists across turns", "model": "SmolLM2-135M",
          "tid": "K_day",
          "why": "The model's own first reply defines who it is for the rest of the chat. Self-conditioning, not the "
                 "user, drives the conversation."},
         {"title": "Qwen3-0.6B: same reply four times under a system persona", "model": "Qwen3-0.6B", "tid": "I_persona",
          "why": "Even the strongest model in the probe can lock into a one-line attractor under greedy decoding. The "
                 "format marker ('Arr') is present every time, but the content is dead."},
         {"title": "SmolLM2-135M: 'Tell me more' produces the same paragraph", "model": "SmolLM2-135M",
          "tid": "L_tell_more", "show": [3, 4, 5],
          "why": "Cross-turn 4-gram overlap 0.95: open-ended continuation prompts collapse into repetition at 135M."},
     ]},
    {"section": "5. Referring to its own earlier answer",
     "intro": "",
     "items": [
         {"title": "LFM2-350M: names the wrong list position", "model": "LFM2-350M", "tid": "O_second_item",
          "why": "The list is the model's own immediately preceding reply; it says 'after strawberries, is kiwi' when kiwi was third."},
         {"title": "Gemma 3 270M: 'I put Finny last' (Finny was first)", "model": "Gemma3-270M", "tid": "O_last_item",
          "why": "Ordinal reference into its own output fails at 270M."},
         {"title": "SmolLM2-135M: invents an item that was never on its list", "model": "SmolLM2-135M",
          "tid": "O_second_item",
          "why": "At 135M the answer is not even one of the listed items."},
     ]},
    {"section": "6. Instruction persistence and follow-ups",
     "intro": "",
     "items": [
         {"title": "Qwen2.5-0.5B: capitals for one turn, then drops the instruction", "model": "Qwen2.5-0.5B",
          "tid": "I_caps",
          "why": "Persistence decays after the acknowledgement turn. The single-turn control ('Reply only in capital "
                 "letters: what do bees make?') is the capability baseline."},
         {"title": "Gemma 3 270M: elliptical follow-up misparsed", "model": "Gemma3-270M", "tid": "F_ellipsis_capital",
          "why": "'And what about Italy?' needs the previous question's frame. The single-turn control ('What is the "
                 "capital of Italy?') passes, so this is a context failure."},
     ]},
    {"section": "7. The 90M model fails differently: no deflection, but a how-to-guide template for everything",
     "intro": "Falcon-H1-Tiny-90M-Instruct (800B tokens, SFT data mixed into pretraining) never deflected on a user-fact "
              "question in this probe, and it scored above both SmolLM2-135M and Gemma-3-270M overall. Its failures are a "
              "different kind: it turns personal questions into generic advice listicles, or fills in a template placeholder.",
     "items": [
         {"title": "Falcon-H1-Tiny-90M: 'Which city do I live in?' becomes a guide to choosing a city", "model": "Falcon-H1-Tiny-90M",
          "tid": "R_d4_city", "show": [0, 5],
          "why": "The model answers a generic version of the question in its default listicle format. Compare the same "
                 "model on R_d4_pet, which it passes ('Your cat, Biscuit, is known as **Biscuit**.')."},
         {"title": "Falcon-H1-Tiny-90M: 'Your name is [Your Name].'", "model": "Falcon-H1-Tiny-90M", "tid": "R_d6_name",
          "show": [0, 7],
          "why": "A template placeholder instead of the fact. The SFT data taught the answer shape without the binding to "
                 "the conversation."},
         {"title": "Falcon-H1-Tiny-90M: the only model that remembered what the user complained about", "model": "Falcon-H1-Tiny-90M",
          "tid": "L_chitchat", "show": [1, 5],
          "why": "Four turns later it recalls 'bored'. No other model did, in any of the 21 greedy and sampled runs of the other seven. "
                 "Then it drifts into advice about a 'reading journey' that the user never started."},
     ]},
    {"section": "8. For contrast: what working multi-turn behavior looks like at 350M-600M",
     "intro": "",
     "items": [
         {"title": "LFM2.5-350M recalls a pet name after 10 distractor turns (1,510 prompt tokens)", "model": "LFM2.5-350M",
          "tid": "R_d10_pet", "show": [0, 11],
          "why": "Long-range recall itself is not beyond a 350M model."},
         {"title": "Qwen3-0.6B applies a correction across a distractor", "model": "Qwen3-0.6B", "tid": "K_time",
          "why": "One of only four multi-turn correction passes out of 24 greedy attempts across eight models."},
     ]},
]
