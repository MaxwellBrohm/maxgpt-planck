"""Checker rubric patterns (SPEC section 6). Claude wrote these; under D8 that is allowed because they are rubric,
not training text: nothing here is ever inserted into a conversation. SAFETY_TERMS is a FAKE stand-in until the
real blocklist exists. All patterns are matched on lowercased text unless a function says otherwise."""
import re


def _alt(words):
    return re.compile(r"(?<![a-z])(?:" + "|".join(words) + r")(?![a-z])", re.I)


STOPWORDS = set("""a an the and or but so if then than that this these those there here it its it's i i'm i've i'd
i'll me my mine you you're you've you'd you'll your yours we us our they them their he him his she her be am is are
was were been being do does did done have has had having to of in on at by for with about from into over under up
down out off again just only also too very not no yes can could would should will shall may might must what which
who whom whose when where why how all any some each every both few more most other such own same as well now get got
go going gone make made let lets like one ok okay oh hi hey hello sure thanks thank please really much many way
thing things something anything nothing lot bit maybe yeah yep""".split())

# topic-neutral conversational words that OFFTOPIC ignores on both sides (2026-09-27): a reply swapped in from a
# chat on another topic passed on "help", "sound", "start" or "new" alone
GENERIC = set("""help sound think feel try keep start new good great nice morning day time idea love enjoy fun happy
glad hope want need use look know see tell say ask talk chat question tip little big small easy hard sure lot best
way""".split())
# forms that are topic-neutral as written though their stem is not (2026-09-28): "Noted" is an acknowledgement (Gemma
# wrote "I have noted" 233 times in dry pilot 2), and matched the noun "note" of a noisy-neighbour topic set
GENERIC_FORMS = {"noted", "noting"}

FUNCTION_WORDS = set("""the a an and or but to of in on at for with is are was were be it i you my your we they
that this what how do does did not can so if have has had me our their there here just""".split())

# ---- AI-isms and assistant boilerplate (AI_ISM) ------------------------------------------------------------------
AI_ISM_RE = _alt([r"as an ai", r"language model", r"great question", r"feel free to", r"i'm here to help",
                  r"i am here to help", r"certainly!", r"happy to help", r"i hope this helps", r"as a virtual",
                  r"i'm just an ai", r"i am just an ai", r"artificial intelligence", r"large language",
                  r"how can i assist", r"is there anything else i can"])
# assistant turns only (a user may ask "how can I help my son read?"): the service-greeting family (v1 review: "Hello,
# how can I help you today?" 33 times in Gemma's dry-pilot accepts) and the substitutes the sampling ban drew out
ASSIST_ISM_RE = _alt([r"how (?:can|may) i ?(?:help|assist)", r"what can i (?:help you with|do for you)",
                      r"(?:i'm|i am) (?:just |always |only )?here to (?:help|assist|chat|support)",
                      r"happy to (?:assist|and help)", r"anything else i can (?:help|do|assist)",
                      r"help with (?:something|anything) else", r"helpful assistant",
                      # 10-03 (dry pilot 3 v1 review, MEDIUM 6): the substitutes that replaced the banned ones, "I can
                      # certainly help you ..." (Gemma, 95 accepted chats) and the "anything else" closers (50 lines)
                      r"i can (?:certainly |definitely |surely |absolutely |also )?help",
                      r"(?:is there|would you like(?: me)?(?: to (?:know|discuss|talk about|add|share|ask|find))?|"
                      r"do you (?:want|need|have)(?: to (?:know|add|ask))?|need|(?:can|may) i (?:help|assist)(?: you)?"
                      r"(?: with)?) anything else",
                      r"anything else (?:to (?:add|adjust|know|discuss|ask)|you(?:'d| would)? (?:like|want|need))"])

# ---- deflection on a given item (DEFLECT) -------------------------------------------------------------------------
DEFLECT_RE = _alt([r"i don't have access", r"i do not have access", r"as an ai,? i can't know", r"i can't remember",
                   r"i cannot remember", r"i don't remember", r"i do not remember", r"i have no memory",
                   r"i can't recall", r"i cannot recall", r"i don't retain", r"i'm not able to remember",
                   r"i am not able to remember", r"i don't keep track"])

# ---- abstention (ABSTAIN_MISSING): a hedge, and an offer to take the fact -----------------------------------------
HEDGE_RE = _alt([r"you haven't (?:told|mentioned|said|shared)", r"you have not (?:told|mentioned|said)",
                 r"you didn't (?:tell|mention|say|share)", r"you did not (?:tell|mention|say)",
                 r"you never (?:told|mentioned|said)", r"i don't know", r"i do not know", r"not sure",
                 r"hasn't come up", r"has not come up", r"didn't come up", r"no idea", r"haven't heard",
                 r"i don't think you've", r"don't think you (?:told|mentioned|said)", r"not something you've",
                 # audit 2026-09-27: the guidance's own wording and its close forms were read as no hedge; 09-28:
                 # negative forms only (the positive "was mentioned" let an invented answer pass as a hedge)
                 r"(?:wasn't|was not|was never|(?:has|have)(?: not| never|n't) been|not been) mentioned",
                 r"(?:we|you) (?:did not|didn't|have not|haven't) (?:mention|talk about|discuss)(?:ed)?",
                 r"no record",
                 # 09-28 (dp2 recheck): "No friend was mentioned so far", "Nothing has been mentioned about it"
                 r"(?:no|nothing)(?: \w+){0,3} (?:was|were|has been|had been) (?:ever )?mentioned"])
OFFER_RE = _alt([r"tell me", r"let me know", r"what is it", r"what's (?:it|its|their|his|her|the)",
                 r"want to share", r"like to share", r"i can (?:note|remember|keep)", r"i'll (?:note|remember|keep)",
                 r"share it", r"fill me in", r"(?:let me|i will|i'll|i can) (?:make a )?note", r"note (?:it|that) down"])

# ---- lookup (LOOKUP_EMPTY, LOOKUP_UNNEEDED) -----------------------------------------------------------------------
NOT_FOUND_RE = _alt([r"couldn't find", r"could not find", r"can't find", r"cannot find", r"not found",
                     r"no results?", r"nothing came up", r"didn't find", r"did not find", r"no information",
                     r"no record", r"came up empty", r"no luck", r"found nothing", r"turned up nothing"])
LOOKUP_TALK_RE = re.compile(r"<\s*/?\s*(?:lookup|result)\s*>|(?<![a-z])(?:look (?:it|that|this) up|looking (?:it|that)"
                            r" up|let me (?:check|search|look)|i'll (?:check|search|look it up)|search for it)"
                            r"(?![a-z])", re.I)
TAG_RE = re.compile(r"<\s*/?\s*(?:lookup|result)\s*>", re.I)

# ---- format leaks inside a turn (END_IN_TURN, 2026-09-28): the END marker (Ministral: "Bye. END. END."), a turn that
# ends on "End." or "The end.", and script talk seen after the no-END regex ("just the script and end")
END_TOKEN_RE = re.compile(r"(?<![A-Za-z])END(?![A-Za-z])")
END_TAIL_RE = re.compile(r"(?:^|[.!?]\s+)(?:the\s+)?end[.!]*\s*$", re.I)
SCRIPT_TALK_RE = _alt([r"(?:this|that|it) ends the (?:script|chat)", r"end of (?:the|this) script",
                       r"just the script"])

# ---- copied guidance (PROMPT_ECHO, 2026-09-28): instruction clauses of render_intents' guidance, old and new, that
# are never chat content (v1 review: "without saying the answer" in 13 accepted user lines, the answer then given;
# the round 2 recheck read a paraphrase of it, "without telling me the answer", in an accepted line too)
GUIDE_META_RE = _alt([r"without (?:saying|telling|giving|revealing|stating)(?: me| you)? the answer",
                      r"without saying it", r"as if you(?: had|'d) said it", r"request a rule", r"drop the earlier rule",
                      r"steer the chat", r"sum up what you have told", r"a near choice",
                      r"the looked up value", r"(?:name )?no other value", r"not repeating the value",
                      r"has not named a topic", r"a few words (?:before|around) the answer", r"the answer itself",
                      r"leading in with what", r"in one short sentence", r"give no guess", r"offer to note it",
                      r"without inventing one", r"no new topic", r"said before the detour",
                      r"keep the rule in this reply",
                      # 10-03 (round 3 wording, render_intents): the lead-in and the abstain guidance
                      r"pointing back to (?:when|what)", r"ask(?:s|ing)? for it in a question"])
# on an event's query line any withholding clause is the old guidance paraphrased (round 2 recheck: accepted recall
# questions asked to be reminded "without telling me directly", "without telling it", "without telling me the city")
WITHHOLD_RE = _alt([r"without (?:saying|telling|giving|revealing|stating|mentioning)"])

# ---- a plant said only inside a question (PLANT_MISSING, 2026-09-28): clauses split at , ; : and conjunctions, and
# before "that" (a that-clause states: "a hat that is purple", "remember that my dog is Rex"); a clause asks when it
# opens with an auxiliary, a request frame or a wh-word and is its sentence's first clause or ends a "?" sentence
# ("my dog is old and is called Rex" states: "is" after "and" continues the subject)
PLANT_CLAUSE_SPLIT = re.compile(r"[;:,]+\s*|\s+(?:since|because|but|and|so|though|although)\s+|\s+(?=that\s)", re.I)
AUX_START_RE = re.compile(r"^\s*(?:can|could|would|will|shall|should|may|might|do|does|did|is|isn't|are|aren't|was|"
                          r"were|am|(?:please\s+)?(?:tell|remind|show) me)(?![a-z'])", re.I)
WH_START_RE = re.compile(r"^\s*(?:what|what's|which|who|who's|whose|where|where's|when|how|how's|why)(?![a-z'])", re.I)
# a question frame around a statement still states (dp2 recheck, 2026-09-28): an embedded clause with its own verb
# after know / remember / tell you ("did you know our team lunch is this Friday?"), and a pseudo-cleft that does not
# end in "?" ("Where I live is Porto.", "What time my interview starts is noon."); "tell me the name of my friend X?"
# and "do you know when my lunch starts at noon?" still ask
EMBED_DECL_RE = re.compile(r"\b(?:know|knew|remember|recall|guess|heard|believe|think|mention(?:ed)?|told you|tell you)"
                           r"\s+(?:that\s+)?(?:(?:my|our|his|her|their|the|this|i|we)\b[^?]*?\b(?:is|are|was|were|am|"
                           r"lives?|works?|starts?|called|named)\b|(?:i'm|i am|it's|it is)\b)[^?]*VALUEMARK", re.I)
PSEUDO_CLEFT_RE = re.compile(r"^\s*(?:what|where|when|which|who)(?:\s+(?:time|day|month|city|colou?r|job|food))?\s+"
                             r"(?:i|my|we|our|the|his|her|their)\b.*\b(?:is|was|are)\b", re.I)

# ---- role swap denial (S6 role_swap) ------------------------------------------------------------------------------
DENY_RE = _alt([r"don't have", r"do not have", r"haven't got", r"of my own", r"i'm an assistant",
                r"i am an assistant", r"not something i have", r"i don't really have", r"no .{1,20} for me",
                # audit 2026-09-27: denials the list missed (after quote normalization)
                r"(?:i'm|i am) just an assistant", r"an assistant (?:doesn't|does not)",
                r"i (?:don't|do not) (?:eat|own|collect)",
                r"i have no (?:favou?rite \w+|hobb(?:y|ies)|job|pets?|family|home|name)"])

# ---- first-person self claims outside the card (SELF_CLAIM), by must_not category ----------------------------------
SELF_CLAIM = {
    "plans": [r"my plans?", r"i'm planning", r"i am planning", r"i plan to", r"i'm heading", r"i'm going on",
              r"my trip", r"my weekend", r"this weekend i", r"tomorrow i", r"i'll be (?:going|visiting|travelling)"],
    "family": [r"my (?:mom|mum|mother|dad|father|sister|brother|wife|husband|kids?|children|son|daughter|family|"
               r"grandma|grandpa|grandmother|grandfather|aunt|uncle|cousin|partner|parents?|niece|nephew)"],
    "body": [r"my (?:back|legs?|hands?|head|stomach|feet|arms?|body)", r"i ate", r"i'm hungry", r"i'm tired",
             r"i slept", r"i'm sleepy", r"i was sick", r"i exercised"],
    "past": [r"i went", r"i used to", r"when i was", r"i grew up", r"i remember when", r"last (?:week|year|weekend|"
             r"night|summer|winter) i", r"i once", r"i've been to", r"i visited", r"back when i",
             r"i (?:booked|bought|cooked|baked|watched|met|drove|played|painted|planted)",
             r"i called(?! you(?![a-z]))"],   # "I called you Dev earlier" is the call-me rule, not a past act
    "preferences": [r"i love", r"i like", r"i enjoy", r"my favou?rite", r"i prefer", r"i hate", r"i'm a fan",
                    r"i adore", r"my hobby", r"i'm into",
                    # 09-28 (v1 review: a denial that goes on to claim, "..., but I do love falconry")
                    r"i (?:do|really|also|still|truly|just|absolutely) (?:love|like|enjoy|adore|prefer)",
                    r"i spend (?:my|most|a lot|time)", r"my (?:evenings|mornings|weekends|free time|spare time|days off)",
                    r"like my own",    # a garbled denial; "a place to call my own", "I don't have my own" deny
                    # 10-03 (dry pilot 3 v1 review, MEDIUM 5): "I'd love to catch up with old friends" passed "i love";
                    # wanting to hear, know, help or see the user's thing is not a claim. "myself" after an act or a
                    # liking, not negated ("I'd try a stack of them myself"; "I do not eat food myself" denies)
                    r"i(?:'d| would) (?:really |also |just |absolutely )?(?:love|enjoy) to(?! (?:help|hear|know|learn|"
                    r"chat|talk|see (?:how|what|it|that|them|your|you)|read (?:about|it|that|your))(?![a-z]))",
                    r"(?<!not )(?<!n't )(?<!never )(?:try|tried|eat|ate|cook|bake|make|watch|read|play|do|did|like|"
                    r"love|enjoy|prefer|partial to)(?: (?!(?:not|never|no)(?![\w'])|\w*n't(?![\w']))[\w']+){0,5} myself"],
    "places": [r"i live", r"i'm from", r"i am from", r"my (?:home|house|city|town|flat|apartment|garden|kitchen)",
               r"where i live", r"near me"],
}
SELF_CLAIM_RE = {k: _alt(v) for k, v in SELF_CLAIM.items()}

# ---- third-person narration inside a turn (USER_VOICE, ASSIST_VOICE) ------------------------------------------------
USER_VOICE_RE = re.compile(r"(?<![a-z])(?:the user|user's|the assistant|(?:he|she|they) (?:says|asks|tells|mentions|"
                           r"replies))(?![a-z])|^\s*(?:user|they|he|she)\s+(?:says|asks|tells|mentions)", re.I)
ASSIST_VOICE_RE = re.compile(r"(?<![a-z])(?:the user|user's|the assistant|assistant's)(?![a-z])", re.I)

# ---- formatting (ROLE_LABEL, MARKDOWN, EMOJI, STAGE_DIR, THOUGHT_TAG) ------------------------------------------------
ROLE_WORDS = ["user", "assistant", "model", "ai", "bot", "human", "system", "me", "you"]
ROLE_LABEL_RE = re.compile(r"(?:^|\s)(?:user|assistant|model|ai|bot|human|system)\s*:|(?<![A-Za-z])[UAT]\d+\s*:",
                           re.I)
MARKDOWN_RE = re.compile(r"\*\*|__|`|^\s*#|^\s*[-*+\u2022]\s|^\s*\d+[.)]\s|(?<![A-Za-z0-9])\*[^*\s][^*]*\*"
                         r"|\[[^\]]+\]\([^)]+\)|\u2022")
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27bf\u2b00-\u2bff\ufe0f\u200d\u2190-\u21ff\u2300-\u23ff"
                      "\u3030\u303d\u3297\u3299\U000E0020-\U000E007F]|(?:^|\\s)(?:[:;]-?[)(DPp]|<3)(?=\\s|$)")
STAGE_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
THOUGHT_TAG_RE = re.compile(r"<\|?(?:turn|channel|start_of_turn|end_of_turn|think)\|?>|</?think>|<\|?\w+\|>|"
                            r"<\|\w+>", re.I)

# ---- SAFETY: FAKE blocklist (real list pending) --------------------------------------------------------------------
SAFETY_PROVENANCE = "FAKE"
SAFETY_TERMS = [r"kill (?:yourself|myself|him|her|them)", r"suicide", r"self[- ]harm", r"bomb", r"cocaine", r"heroin",
                r"porn", r"nude", r"sexy", r"gun", r"weapon", r"murder", r"racist", r"slur"]
SAFETY_RE = _alt(SAFETY_TERMS)


def self_claims(text):
    """[(category, matched text)] for first-person self claims in an assistant turn."""
    out = []
    for cat, rx in SELF_CLAIM_RE.items():
        out += [(cat, m.group(0)) for m in rx.finditer(text)]
    return out


# ---- round 3 (2026-10-03, dry pilot 3 v1 review; SPEC 15; check_r3.py) -------------------------------------------
# PERSPECTIVE: the assistant says it told the user what the user told it ("As you were told before, your shift is in
# December": 22 Qwen accepts after the round 2 lead-in wording; "As told before my name is Ember", never said; "I told
# you, he's based in Perth"). check_r3 lets the frame stand when an earlier assistant turn named a value it names.
TOLD_FRAME_RE = _alt([r"you(?: were|'ve been| have been| had been) told", r"as told(?! by you)",
                      r"(?:as|like) i (?:said|mentioned|told you)", r"i(?:'ve| have)? (?:already )?told you",
                      r"i (?:said|mentioned)(?: (?:it|that|this))? (?:earlier|before)", r"the one i mentioned"])
# FALSE_MEMORY: a claim about the chat's own history that the chat contradicts (v1 review HIGH 4): forgetting on a
# turn that is not an abstain answer ("Sounds like fun, but I've forgotten", an acknowledgement); the assistant's own
# error when the user corrected themselves ("Thank you for correcting me", "Of course, I meant Osaka"); "earlier" for
# a value said in the line just before ("You mentioned a beekeeper earlier", one turn after the user said it)
FORGOT_RE = _alt([r"i(?:'ve| have)? forgot(?:ten)?", r"(?:it )?slipped my mind", r"i(?:'ve| have) lost track",
                  r"i (?:can't|cannot|can not|don't|do not) (?:remember|recall)"])
SELF_ERR_RE = _alt([r"correct(?:ing|ed) me", r"my (?:mistake|error|bad)", r"i stand corrected", r"i was wrong",
                    r"i (?:got|had) (?:it|that|this) wrong", r"i meant(?! to)", r"i misspoke", r"i misheard",
                    r"(?:sorry|apologi[sz]e) (?:for|about) (?:the|my|that) (?:mistake|error|mix ?up|confusion)"])
EARLIER_RE = _alt([r"(?:mentioned|said|told me|shared|brought up|talked about)(?: [\w']+){0,6} (?:earlier|before|"
                   r"previously)", r"(?:earlier|before|previously),? you (?:mentioned|said|told|shared)"])
# LIVE_DATA: the assistant reports current weather it cannot observe ("It is quite cold outside with snow falling",
# "the weather is so sunny today"); a hedge in or just before the match ("I hope", "if", "sounds") lets it pass
_WX = r"(?:cold|warm|hot|sunny|rainy|raining|snowy|snowing|windy|cloudy|chilly|freezing|stormy|humid|wet|foggy)"
_NOW = r"(?:outside|out there|today|tonight|right now|now|this (?:morning|afternoon|evening|week)|at the moment)"
LIVE_RE = re.compile(r"(?<![a-z])(?:(?:it is|it's|it has been|it's been)\b[^.!?]{0,25}?\b" + _WX + r"\b[^.!?]{0,40}?\b"
                     + _NOW + r"|the weather (?:is|has been|looks)\b[^.!?]{0,40}?\b" + _NOW + r"|(?:the|this) " + _WX +
                     r" weather " + _NOW + r"|(?:snow|rain) (?:is )?(?:falling|coming down|pouring)|(?:the|that|this) "
                     r"(?:snow|rain|sun|sky|fog) (?:looks|is)\b[^.!?]{0,25}?\b(?:outside|out there)|the forecast "
                     r"(?:says|shows|calls for|is|predicts))(?![a-z])", re.I)
LIVE_HEDGE_RE = _alt([r"hope", r"if", r"might", r"may", r"could", r"maybe", r"perhaps", r"sound(?:s|ing)?",
                      r"must be", r"i bet", r"probably", r"wish", r"when(?:ever)?"])   # "when it is wet outside"
# RUN_ON: a request glued to the line before it with no stop or joining word ("You have not told me the start time
# yet please share it", Qwen abstains, LOW 10); "..., please", "so please" and "would you please" are joined
RUN_ON_RE = re.compile(r"(?<![A-Za-z'])please\s+(?:share|tell|let|give|send)(?![a-z])", re.I)
RUN_ON_JOIN = {"so", "and", "but", "then", "or", "you", "would", "could", "can", "will", "kindly", "just", "yes", "oh",
               "okay", "ok", "sure"}   # dp3 re-check: "Hello Lola, yes please tell me about the offer"
