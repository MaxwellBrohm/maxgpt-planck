"""W3 prompts, part 2 (BANKPASS s3 W, I, O(2), Q, s4 word labels): topic word sets, filler intents, topic-specific
openings, instruction paraphrases and the word-list labels (part of speech, verb forms, -ly meaning). Claude wrote
this wording (prompt, never training text); zero-shot, no example line. The paraphrase checklist is rubric: Claude
wrote it from render_prompt's base block, and a second teacher answers it."""
from teachers import decode as D

from bankpass import prompts as PR

PROMPT_VERSION = "bp-prompts4-v1"
END = "\nAfter the last line write END on its own line."


def wordset_prompt(topic, seed=None):
    return (f"A person is chatting with an assistant about this topic: {topic}.\nList 20 different nouns, 10 different "
            "verbs and 10 different adjectives a person would use when talking about it: everyday single words, "
            "lowercase, separated by commas. Write three lines: the nouns after 'nouns:', the verbs after 'verbs:' and "
            "the adjectives after 'adjectives:'." + END)


WORDSET_HEADS = ("nouns", "verbs", "adjectives")


def wordset_res():
    return [h + ": [a-z][a-z' ,]*" for h in WORDSET_HEADS]


def intent_prompt(topic, k, seed=None):
    return (f"A person is chatting with an assistant about this topic: {topic}.\nWrite {k} different things the person "
            "could do in one of their messages in that chat, each as a short instruction to the person, three to "
            "twelve words, starting with a verb such as ask, say, tell or mention. Do not write the message itself. "
            "Keep each one specific to the topic and different from the others. One per line, no numbering."
            f"\n{PR._RULES}" + END)


def topen_prompt(topics, k):
    if len(topics) == 1:      # amendment 3: one topic per call
        return (f"A person wants to start a chat with an assistant about this topic: {topics[0]}.\nWrite {k} different "
                "first messages they might send: each greets the assistant and brings up the topic in the person's "
                "own words, and says nothing about the person such as a name, city, job, pet, plan or relative. "
                f"One per line, no numbering.\n{PR._RULES}" + END)
    body = "\n".join(f"Topic {i}: {t}" for i, t in enumerate(topics, 1))
    return (f"For each topic below, write {k} different first messages a person might send to start a chat with an "
            "assistant about it: each greets the assistant and brings up the topic in the person's own words, and "
            "says nothing about the person such as a name, city, job, pet, plan or relative. Write the lines for "
            f"topic one first, then topic two, and so on: {k * len(topics)} lines in all, no numbering, no topic "
            f"labels.\n{body}\n{PR._RULES}" + END)


# ---- word list labels (s4, the W3 engineering change: three agreeing teachers decide POS) ------------------------
POS = ("noun", "verb", "adjective", "adverb", "other")


def pos_prompt(words):
    return ("For each word below, write the word, a colon and its most common part of speech in everyday English: "
            "noun, verb, adjective, adverb or other. One word per line, in the same order.\n" + "\n".join(words) + END)


def verbs_prompt(words):
    return ("For each verb below, write the verb, a colon, its past tense, a comma and its past participle, in "
            "lowercase. One verb per line, in the same order.\n" + "\n".join(words) + END)


def ly_prompt(pairs):
    return ("For each pair of words below, write the pair as given, a colon, and same if the second word means the "
            "same as the first in the manner of an action, or different if its meaning has changed. One pair per "
            "line, in the same order.\n" + "\n".join(f"{a}, {b}" for a, b in pairs) + END)


def word_res(kind, rows):
    if kind == "pos":
        return [D.esc(w) + ": (?:" + "|".join(POS) + ")" for w in rows]
    if kind == "pos2":
        return [D.esc(w) + ": (?:" + "|".join(POS2) + ")" for w in rows]
    if kind == "verbs":
        return [D.esc(w) + ": [a-z]+, [a-z]+" for w in rows]
    if kind == "ly":
        return [D.esc(f"{a}, {b}") + ": (?:same|different)" for a, b in rows]
    raise KeyError(kind)


# ---- instruction paraphrases (s3 Q) ---------------------------------------------------------------------------
def base_block():
    """the base block a paraphrase rewrites: render_prompt's first instruction variant (head, tail)."""
    import render_prompt as RP
    _, head, tail = RP.VARIANTS[0]
    return head.replace("\n", " "), tail


def para_prompt():
    head, tail = base_block()
    return ("Below are instructions given to a writer, in two parts. Rewrite each part in your own words. Keep every "
            "instruction and its meaning, add no new instruction, and do not shorten it into a list. Write part one "
            "on one line and part two on the next line, with no labels.\nPart one: " + head + "\nPart two: " + tail
            + END)


# rubric (Claude, from VARIANTS[0]); a paraphrase is kept when the checking teacher answers yes to all and no to
# "added"
CHECKLIST = ("It asks for one chat between a user and an assistant.", "It says to follow a script line by line.",
             "It asks for plain everyday sentences.", "It forbids emojis.", "It forbids bold or other markdown.",
             "It forbids lists.", "It forbids stage directions or actions.",
             "It forbids speaker names inside a line.", "It allows typos only when the user description asks for them.",
             "It asks for one output line per script line.", "Each line starts with its label and a colon.",
             "The lines keep the script order.", "The last line is END.", "Nothing else is written.",
             "Lines marked copy exactly are copied character for character.", "Each line stays inside its word range.",
             "Every must include item is used exactly as written.")


def check_prompt(head, tail):
    pts = "\n".join(f"{i}. {c}" for i, c in enumerate(CHECKLIST, 1))
    return ("Read these instructions:\n" + head + "\n" + tail + "\n\nFor each numbered point, answer yes if the "
            "instructions above say it and no if they do not. Then answer added: yes if the instructions give any "
            "instruction that is not among the points, else no. Write one answer per line as the number, a colon and "
            "yes or no, then the added line.\n" + pts + END)


def check_res():
    return [f"{i}: (?:yes|no)" for i in range(1, len(CHECKLIST) + 1)] + ["added: (?:yes|no)"]


TOPIC_GROUPS = ("home and chores", "food and cooking", "health and fitness", "work and study", "money and shopping",
                "travel and transport", "hobbies and crafts", "pets and animals", "family and friends",
                "events and celebrations", "nature and garden", "technology at home", "books films and music",
                "other")


def group_prompt(topic):
    return (f"Which group does this chat topic belong to: {topic}?\nAnswer with one of: " + ", ".join(TOPIC_GROUPS)
            + "." + END)


def group_res():
    return ["(?:" + "|".join(TOPIC_GROUPS) + ")"]


# ---- version 2 (2026-10-04, after Qwen's first hold): pools without seed words, POS with the labels models use ----
# Qwen hold 1 copied the three seed nouns into values ("molecular pie", "standpoint dog", "macro lens cake"), wrote
# activities for entity kinds and sentences for pet kinds; greedy POS with five labels gave "noun" to all 8,000 words
# ("the: noun"). A live probe in the same hold: the labels below gave the: determiner, quickly: adverb, wow:
# interjection; the asks below gave names, singular kinds and bare nouns.
POS2 = ("noun", "verb", "adjective", "adverb", "pronoun", "preposition", "conjunction", "determiner", "interjection",
        "number", "other")
POOL_ASK2 = {
    "job": "jobs people do for a living, each the usual name of the job for one person, in one to three words",
    "hobby": "hobbies people do in their free time, each the usual name of the hobby in one to three words",
    "food": "foods a person could name as their favourite, each the usual name of the food in one to three words",
    "grocery": "things people buy at a grocery shop, each the common name of the thing in one to three words, with no "
               "amount",
    "chore": "household chores, each the usual name of the chore in one to four words",
    "plan": "kinds of personal plans or events a person puts on their calendar, each the common name of that kind of "
            "plan or event in one to three words, with no day, time or place",
    "object": "everyday things a person owns and could describe by their colour, each the common name of the thing in "
              "one to three words, without its colour",
    "pet_kind": "kinds of animals people keep as pets, each the animal's common name in the singular, one or two words",
    "pet_name": "names people give their pets, each one or two words starting with a capital letter",
    "assistant_name": "short first names a person might give a chat assistant, one word each, starting with a capital "
                      "letter; names, not ordinary words",
    "relation": "words for one person in someone's family or social life, such as relatives, friends, or people at "
                "work or next door, each one or two words, in the singular",
    "entity_kind": "kinds of places or events found in towns and cities that a person might look up, each the common "
                   "noun for that kind of place or event (a type of shop, venue, building, service or yearly event), "
                   "one or two words, with no names, days or times",
}


def pool_prompt2(vtype, n, seed=None):
    case = "" if vtype in ("pet_name", "assistant_name") else " Write them in lowercase."
    return (f"List {n} different {POOL_ASK2[vtype]}. Do not put a, an, the or my in front.{case} Include ordinary "
            f"ones and less common ones. One per line, no numbering.\n{PR._RULES}{PR.voice(seed)}" + END)


def pos_prompt2(words):
    return ("For each word below, write the word, a colon and its most common part of speech in everyday English: "
            + ", ".join(POS2[:-1]) + " or other. One word per line, in the same order.\n" + "\n".join(words) + END)


def topic_prompt2(n, seed_words, seed=None):
    """amendment 5: Ministral wrote questions and requests of 9 to 13 words for version 1 (421 of 500 too long)."""
    return (f"List {n} different everyday subjects a person might chat with an assistant about. Write each as a short "
            "noun phrase of three to seven words in lowercase that names the subject: not a question, not a request, "
            "nothing about yourself, no names. Let these words nudge you toward variety: " + ", ".join(seed_words)
            + ". After each phrase write ' | ' and how common a chat subject it is: common, uncommon or rare."
            f"\n{PR._RULES}{PR.voice(seed)}" + END)
