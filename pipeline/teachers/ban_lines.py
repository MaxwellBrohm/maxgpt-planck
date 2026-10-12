"""Sentences for the phrase ban checks on the real tokenizers (teachers/test_decode_pc.RealPhraseBan). FAKE test text
written by Claude, modeled on dry pilot 4's service lines; nothing here is training data."""

# dp4 service lines (2026-10-04, aiism-v4) the ban must cover on every tokenizer; the "glad" forms are not coverable on
# Qwen and Ministral (a bare "glad" splits, so vLLM keeps no space form) and Ministral's bare "anything" splits too:
# AI_ISM still rejects those (teachers/test_round2.PhraseBan)
DP4_SERVICE = ["I can certainly help you with that.", "I can help you with that.", "Is there anything else?",
               "Would you like to know anything else?", "Would you like to add anything else?",
               "Do you need anything else?", "I am here to help you.", "I'm an assistant here to help.",
               "I’m just here to assist.", "I am functioning well and ready to help you now.",
               "I'd love to help with that.", "How else can I help?", "I would be happy to help you.",
               "Tariq, I can certainly do that for you.", "May I help you with anything else?", "Happy to help!",
               "How may I help you today?", "I can assist with that.", "Would you like me to remember anything else?",
               "I can definitely help you with that."]
NOT_COVERABLE = {"qwen3.5-9b": {"Glad I could assist.", "I'm glad I could help."},
                 "ministral-3-8b": {"Glad I could assist.", "I'm glad I could help.", "Anything else to adjust?",
                                    "Would you like me to remember anything else?"},
                 "gemma-4-12b": set()}
BENIGN = ["I would be glad to see you then.", "I can see why you like it.", "I am here now.", "That helps a lot.",
          "Nothing else is on the list.", "You can help your sister move.", "I can call you Sam from now on."]
