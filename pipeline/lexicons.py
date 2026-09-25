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

FUNCTION_WORDS = set("""the a an and or but to of in on at for with is are was were be it i you my your we they
that this what how do does did not can so if have has had me our their there here just""".split())

# ---- AI-isms and assistant boilerplate (AI_ISM) ------------------------------------------------------------------
AI_ISM_RE = _alt([r"as an ai", r"language model", r"great question", r"feel free to", r"i'm here to help",
                  r"i am here to help", r"certainly!", r"happy to help", r"i hope this helps", r"as a virtual",
                  r"i'm just an ai", r"i am just an ai", r"artificial intelligence", r"large language",
                  r"how can i assist", r"is there anything else i can"])

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
                 r"i don't think you've", r"don't think you (?:told|mentioned|said)", r"not something you've"])
OFFER_RE = _alt([r"tell me", r"let me know", r"what is it", r"what's (?:it|its|their|his|her|the)",
                 r"want to share", r"like to share", r"i can (?:note|remember|keep)", r"i'll (?:note|remember|keep)",
                 r"share it", r"fill me in"])

# ---- lookup (LOOKUP_EMPTY, LOOKUP_UNNEEDED) -----------------------------------------------------------------------
NOT_FOUND_RE = _alt([r"couldn't find", r"could not find", r"can't find", r"cannot find", r"not found",
                     r"no results?", r"nothing came up", r"didn't find", r"did not find", r"no information",
                     r"no record", r"came up empty", r"no luck", r"found nothing", r"turned up nothing"])
LOOKUP_TALK_RE = re.compile(r"<\s*/?\s*(?:lookup|result)\s*>|(?<![a-z])(?:look (?:it|that|this) up|looking (?:it|that)"
                            r" up|let me (?:check|search|look)|i'll (?:check|search|look it up)|search for it)"
                            r"(?![a-z])", re.I)
TAG_RE = re.compile(r"<\s*/?\s*(?:lookup|result)\s*>", re.I)

# ---- role swap denial (S6 role_swap) ------------------------------------------------------------------------------
DENY_RE = _alt([r"don't have", r"do not have", r"haven't got", r"of my own", r"i'm an assistant",
                r"i am an assistant", r"not something i have", r"i don't really have", r"no .{1,20} for me"])

# ---- first-person self claims outside the card (SELF_CLAIM), by must_not category ----------------------------------
SELF_CLAIM = {
    "plans": [r"my plans?", r"i'm planning", r"i am planning", r"i plan to", r"i'm heading", r"i'm going on",
              r"my trip", r"my weekend", r"this weekend i", r"tomorrow i", r"i'll be (?:going|visiting|travelling)"],
    "family": [r"my (?:mom|mum|mother|dad|father|sister|brother|wife|husband|kids?|children|son|daughter|family|"
               r"grandma|grandpa|grandmother|grandfather|aunt|uncle|cousin|partner|parents?|niece|nephew)"],
    "body": [r"my (?:back|legs?|hands?|head|stomach|feet|arms?|body)", r"i ate", r"i'm hungry", r"i'm tired",
             r"i slept", r"i'm sleepy", r"i was sick", r"i exercised"],
    "past": [r"i went", r"i used to", r"when i was", r"i grew up", r"i remember when", r"last (?:week|year|weekend|"
             r"night|summer|winter) i", r"i once", r"i've been to", r"i visited", r"back when i"],
    "preferences": [r"i love", r"i like", r"i enjoy", r"my favou?rite", r"i prefer", r"i hate", r"i'm a fan",
                    r"i adore", r"my hobby", r"i'm into"],
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
