"""Round 4 rubric patterns (2026-10-04, SPEC 16; the dry pilot 4 v1 quality review, notes.txt). Claude wrote these;
under D8 that is allowed because they are rubric, not training text: nothing here is ever inserted into a
conversation. Matched on straight-quoted text, case-insensitive unless a pattern says otherwise (check_r4.py)."""
import re

from lexicons import _alt

# ---- FALSE_MEMORY, never said (HIGH 1): the assistant says the USER said or called it something no user line holds
# ("You called me Tavi earlier", "You mentioned that my name is Tavi at the start": the card name, said by nobody but
# the system line). "you asked" is not here: it points at a question, not at a value ("You asked about the museum; it
# opens at nine" is a fine lookup answer)
USER_SAID_RE = _alt([r"you(?:'ve| have)? (?:called|named) me", r"you (?:mentioned|said|told me|shared|brought up|gave me)",
                     r"as you (?:said|mentioned|told me|put it)", r"you(?:'ve| have) (?:mentioned|said|told me|shared)",
                     r"remember(?:ed)? you (?:saying|mentioning|telling me|calling me|naming|sharing)",
                     r"remember(?:ed)? you (?:said|mentioned|told me|called me|shared)"])
# ---- FALSE_MEMORY, remember (HIGH 5): "I remember" for a value the user said one line before ("I remember you saying
# Voulgrel", Ministral, on the line after the plant; 11 Ministral and 5 Qwen chats); "I'll remember" is a promise
REMEMBER_RE = _alt([r"i (?:do |still |definitely |clearly |certainly )?(?:remember|recall)(?! to)"])

# ---- AI_ISM, assistant turns (MEDIUM 6 and the dp4 service family the sampling ban targets): the substitutes that
# passed round 3 ("I am functioning well and ready to help you now", "I'm an assistant here to help", "Glad I could
# assist", "how else can help?", "I can certainly do that for you", "I have that information stored right here for
# you") and the "anything else" closers round 3 missed ("Would you like to remember anything else?")
SERVICE_RE = _alt([r"here to (?:help|assist|support)", r"ready to (?:help|assist)", r"glad (?:i could|to) (?:help|assist)",
                   r"(?:happy|glad|pleased) to assist", r"how else can(?: i)? (?:help|assist)",
                   r"i can (?:certainly |definitely |surely |absolutely |also )?assist",
                   r"i can (?:certainly|definitely|surely|absolutely) do that", r"i(?:'d| would) love to help",
                   r"stored (?:right )?here for you", r"at your service", r"anything else\s*\?",
                   r"(?:here|ready) for anything else",   # "more than anything else I can think of" is no closer
                   # the Qwen ban arms' substitutes (round 4 GPU hold, aiism-v3 on): "I am here to answer any questions",
                   # "I am here to listen", "What else can I help with?" ("I hope I can help" is lexicons' "i can help")
                   r"(?:i'm|i am) (?:just |always |only |also )?here to (?:listen|answer)",
                   r"what else can i (?:help|do|assist)",
                   # the Gemma and Ministral ban arms': "I would be happy to helps" (broken), "Would you like to know
                   # anything more?", "Do you need anything extra?", "Did you have anything else in mind?", "How may I be
                   # of service today?", "I can try to help you with that", "Is there anything you need?"
                   r"(?:happy|glad) to help\w+", r"(?:know|add|need|discuss|share) anything (?:else|more|extra)",
                   r"anything else in mind", r"be of service", r"i can try to help",
                   r"is there anything (?:new|more|you need|you'd like|you would like)"])

# ---- ASSIST_CASE, a name in lowercase inside an assistant turn (D1; dry pilot 4: 126 of Qwen's REQ_SPAN hits in
# lowercase chats are a capitalized value written lowercase, "The broga ferry departs on wednesday", and 121 of those
# lines have no capital but the first letter). Values that are also everyday words are left out ("you may want",
# "add pepper", "a pebble"): REQ_SPAN still holds them where they are required.
AMBIG_LOWER = set("""may march august will mark bill rose grace hope joy faith summer nice reading bath mobile sunny
pepper olive mango pebble socks maple pumpkin peanut poppy clover toffee marble biscuit pickles noodle waffles muffin
scout sprout bubbles nutmeg rusty domino bingo twiggy tinker gizmo luna kit sol nova ember lark wren coda aster
fenn remy suki bram dawn eve ray sky ivy jade amber pearl ruby violet iris lily daisy holly heather autumn""".split())

# ---- ASSIST_VOICE, the user named in the third person (HIGH 4: "Hana says your class takes place in Utrecht", "Tistu
# says it starts at noon", "Una sees that", "Dev will have that name now"): the user's name or nickname followed by a
# verb of the user's own doing; a vocative takes a comma ("Hana, your class ...")
THIRD_VERB = (r"(?:says|said|sees|saw|hopes|confirms|thinks|knows|wants|needs|asks|asked|tells|told|mentions|mentioned|"
              r"will|would|has|had|likes|loves|enjoys|seems|feels|plans|wishes|believes|prefers|gets|goes|is going)")

# ---- PERSPECTIVE, the user's things claimed (HIGH 5): "Thank you for telling me that my name is really Arlo" (the
# user's name), "My club begins in January" (the user's book club)
NAME_ADV = r"(?:really|actually|still|now|just|indeed|truly) "   # the adverb-free form is check_behav's

# ---- PLANT_MISSING, a rule ask that never states its rule (HIGH 3: "help me remember my rules.", then the assistant
# keeps a rule nobody asked for; "The rule about months is done. What should I use from here?"): rules with no
# required item must name what the rule is about
RULE_STATES = {"one_sentence": _alt([r"(?:one|single|1) sentences?", r"sentence each", r"one line"]),
               "end_question": _alt([r"questions?", r"ask(?:ing)? me (?:something|back|a)", r"question mark", r"\?\s*at"
                                     r" the end"])}

# ---- ANSWER_WRONG, a person bound to the wrong relation or a job taken for a person (HIGH 5: "your friend is
# actually Greta", Greta being the coworker; "your gardener", the uncle being the gardener)
REL_GAP = 3                    # words allowed between "your <relation>" and the person's name
_ASK_ME = re.compile(r"^\W*(?:(?:can|could|would|will) you |please |just )?ask me\b", re.I)


def ask_me_lead(s):
    """a user line that opens by telling the assistant to ask the user: the guidance's "ask the assistant to ..."
    turned around ("Ask me to remind what my cousin does for work.", Qwen, dry pilot 4)."""
    return _ASK_ME.search(s)
