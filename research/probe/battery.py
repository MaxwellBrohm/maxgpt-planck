"""Multi-turn coherence battery for small chat models.

Every conversation is a list of scripted user turns. The model's OWN replies are
fed back as history (true multi-turn). Checks are deterministic functions of the
reply text (and, for "refer to your own answer" tests, of the model's earlier
replies). Run `python battery.py` to execute the grader self-test (mutation test:
every check must pass its good fixture, fail an empty answer, fail a plausible
wrong answer; and no gold answer may appear in the question that asks for it).
"""
import re

# ---------------------------------------------------------------- distractors
# No names, numbers, cities or pets in these, so they cannot leak a gold answer.
D = [
    "What's a good way to keep houseplants healthy?",          # 0
    "Can you give me a tip for sleeping better?",              # 1
    "Why is the sky blue?",                                    # 2
    "What are some healthy breakfast ideas?",                  # 3
    "How do I make my coffee taste less bitter?",              # 4
    "What's the difference between weather and climate?",      # 5
    "Can you explain what a noun is?",                         # 6
    "How can I stay focused while studying?",                  # 7
    "What's a fun indoor activity for a rainy day?",           # 8
    "Why do leaves change color in the fall?",                 # 9
    "How do I make a paper airplane fly farther?",             # 10
    "What does a librarian do?",                               # 11
]

# ---------------------------------------------------------------- helpers
def norm(s):
    return (s or "").lower()


def rx(pattern):
    return re.compile(pattern, re.I)


NEG_CTX = rx(r"(not|n't|instead of|rather than|from|changed|no longer|originally|previously|earlier|before|used to|was|were)\W+(\w+\W+){0,3}$")


def mentions(pattern, text):
    return bool(rx(pattern).search((text or "").replace("\u2019", "'")))


def asserted(pattern, text):
    """True if pattern occurs somewhere NOT preceded by a negation/"changed from" context."""
    for m in rx(pattern).finditer(text or ""):
        before = text[max(0, m.start() - 40):m.start()]
        if not NEG_CTX.search(before):
            return True
    return False


def words(t):
    return re.findall(r"[A-Za-z0-9']+", t or "")


def n_sentences(t):
    t = (t or "").strip()
    t = re.sub(r"\b(e\.g|i\.e|etc|dr|mr|mrs|ms|st|vs|approx|u\.s)\.", lambda m: m.group(0).replace(".", ""), t, flags=re.I)
    t = re.sub(r"(\d)\.(\d)", r"\1\2", t)
    parts = [p for p in re.split(r"(?<=[.!?])\s+|\n+", t) if len(re.findall(r"[A-Za-z]+", p)) >= 2]
    return len(parts)


def ends_with_question(t):
    t = (t or "").rstrip()
    t = re.sub(r"[\s\"'*_)\]’”]+$", "", t)
    t = re.sub(r"[^\x00-\x7F]+$", "", t).rstrip()  # trailing emoji
    return t.endswith("?")


def caps_ratio(t):
    letters = [c for c in (t or "") if c.isalpha()]
    if len(letters) < 5:
        return 0.0
    return sum(c.isupper() for c in letters) / len(letters)


def parse_list(t):
    """Items of a numbered / bulleted list, cleaned to their head words."""
    items = []
    for line in (t or "").splitlines():
        m = re.match(r"^\s*(?:\d+[.)]|[-*•])\s+(.*\S)", line)
        if m:
            s = m.group(1)
            s = re.sub(r"[*_`\"]", "", s)
            s = re.split(r"\s[-–—:(]|[:(,]", s)[0].strip()
            if s:
                items.append(s)
    if len(items) < 3:  # fallback: "1. Apple 2. Kiwi 3. Orange" on one line
        found = re.findall(r"(?:^|\s)\d+[.)]\s*([A-Za-z][A-Za-z' ]{1,30}?)(?=\s+\d+[.)]|[,.;\n]|$)", t or "")
        if len(found) >= 3:
            items = [f.strip() for f in found]
    return items


def stems(t):
    return [w.lower()[:5] for w in re.findall(r"[A-Za-z]+", t or "") if len(w) >= 3]


def item_key(item):
    """A matching key for a list item: the sequence of 5-char stems of its words."""
    k = tuple(stems(item))
    return k or None


def find_seq(seq, key):
    for i in range(len(seq) - len(key) + 1):
        if tuple(seq[i:i + len(key)]) == key:
            return i
    return -1


def last_int(t):
    nums = re.findall(r"(?<![\d.])\d+(?!\d|\.\d)", (t or "").replace(",", ""))
    return int(nums[-1]) if nums else None


def num_pat(n):
    return r"(?<![\d.,])" + str(n) + r"(?![\d])"


# Deflection: claims it cannot remember / has no access / cites privacy instead of answering.
NO_MEMORY = rx(r"(don't|do not|cannot|can't|unable to|not able to)\s+(\w+\s+){0,3}(access|remember|recall|know|store|retain|provide)|as an ai|as a helpful ai|i don't have (any |the )?(personal|memory|information|ability)|do not have (any |the )?(personal|memory|information|ability)|personal (information|data|details)|privacy|no (memory|access)|haven't (told|mentioned|shared)|you (haven't|didn't) (tell|mention|share|say)")

# ---------------------------------------------------------------- check factories
# Every check is a dict: name, turn (index of the assistant reply it grades),
# fn(ans, replies) -> bool, good (fixture that must pass), bad (plausible wrong
# answer that must fail), ctx (fake earlier replies for dynamic checks).

IDENTITY_DEFLECT = rx(r"\bi(?:'m| am) (?:just |only )?(?:an ai\b|a (?:large )?language model|an artificial intelligence|a virtual assistant|a chatbot|a text-based ai|qwen\b|gemma\b|smollm\b)"
                      r"|\bas an? (?:ai|(?:large )?language model|chatbot|text-based ai)\b"
                      r"|\b(?:don't|do not|can't|cannot) have (?:a|any) (?:physical|name|cat|dog|pet|favorite|personal)")


def unquote(t):
    return (t or "").replace("\u2019", "'").replace("\u2018", "'")


def deflects(ans):
    """Memory/privacy deflection or an identity disclaimer instead of an answer about the user."""
    a = unquote(ans)
    return bool(NO_MEMORY.search(a) or IDENTITY_DEFLECT.search(a))


FIRST_P = rx(r"\b(i|i'm|i've|my|me|mine)\b")
SECOND_P = rx(r"\b(you|your|you're|yours)\b")


def captures(pattern, ans):
    """Perspective error: the first sentence that contains the gold speaks in the first
    person and never addresses the user, e.g. "Hello! I'm Marcus." or "My dog is named Waffles!"."""
    for sent in re.split(r"(?<=[.!?])\s+|\n+", ans or ""):
        if mentions(pattern, sent):
            return bool(FIRST_P.search(sent)) and not SECOND_P.search(sent)
    return False


VOCATIVE = r"\b(hi|hello|hey|okay|ok|dear|thanks|thank you|welcome|nice to meet you|good to meet you)\W+(?:there\W+)?"


def strip_vocative(pattern, ans):
    """Remove greeting uses of the gold ("Hello Marcus!", "Okay, Priya!") so an echo in a
    greeting is not counted as an answer."""
    return re.sub(VOCATIVE + r"(?:" + pattern + r")\b\W?", " ", ans or "", flags=re.I)


def has(name, turn, pattern, good, bad, guard=True, capture=True):
    """Gold pattern present (outside a greeting). With guard=True the reply must also not
    deflect ("Hi Priya! Sorry, I don't know what you told me your name was" fails) and, if
    capture=True, must not state the user's fact as its own ("I'm Marcus.")."""
    def fn(ans, replies):
        if guard:
            ans = strip_vocative(pattern, ans)
        if not mentions(pattern, ans):
            return False
        if guard and deflects(ans):
            return False
        if guard and capture and captures(pattern, ans):
            return False
        return True
    return dict(name=name, turn=turn, fn=fn, good=good, bad=bad, gold=pattern, guard=guard)


def has_not_stale(name, turn, new, old, good, bad):
    """Corrected value present, stale value never asserted (a negated mention is fine)."""
    def fn(ans, replies):
        if not mentions(new, ans) or deflects(ans) or captures(new, ans):
            return False
        if not asserted(old, ans):
            return True
        # both values asserted: accept only if the corrected value is the one the reply ends on
        last_new = max(m.start() for m in rx(new).finditer(ans))
        last_old = max(m.start() for m in rx(old).finditer(ans))
        return last_new > last_old
    return dict(name=name, turn=turn, fn=fn, good=good, bad=bad, gold=new, guard=True)


def binding(name, turn, right, wrong_assert, good, bad):
    def fn(ans, replies):
        return mentions(right, ans) and not mentions(wrong_assert, ans) and not deflects(ans) and not captures(right, ans)
    return dict(name=name, turn=turn, fn=fn, good=good, bad=bad, gold=right, guard=True)


def degenerate(ans):
    """A loop: some word 3-gram occurs 4+ times in the reply."""
    ws = [w.lower() for w in words(ans)]
    g = {}
    for i in range(len(ws) - 2):
        k = tuple(ws[i:i + 3]); g[k] = g.get(k, 0) + 1
    return max(g.values(), default=0) >= 4


def fresh(ans, replies, turn):
    """Format checks only count a real reply: not a loop, not a verbatim copy of an earlier reply."""
    a = norm(ans).strip()
    return not degenerate(ans) and all(a != norm(x).strip() for x in replies[:turn])


def one_sentence(name, turn, good, bad):
    return dict(name=name, turn=turn, fn=lambda a, r: n_sentences(a) == 1 and fresh(a, r, turn), good=good, bad=bad)


def end_q(name, turn, good, bad):
    return dict(name=name, turn=turn, fn=lambda a, r: ends_with_question(a) and fresh(a, r, turn), good=good, bad=bad)


def all_caps(name, turn, good, bad):
    return dict(name=name, turn=turn, fn=lambda a, r: caps_ratio(a) >= 0.9 and fresh(a, r, turn), good=good, bad=bad)


def has_fmt(name, turn, pattern, good, bad):
    """Format marker present in a fresh (non-looping, non-copied) reply."""
    return dict(name=name, turn=turn, fn=lambda a, r: mentions(pattern, a) and fresh(a, r, turn), good=good, bad=bad)


def shorter_than(name, turn, src_turn, topic, good, bad, ctx):
    def fn(ans, replies):
        prev = replies[src_turn]
        wa, wp = len(words(ans)), len(words(prev))
        return 3 <= wa < wp and mentions(topic, ans) and norm(ans).strip() != norm(prev).strip()
    return dict(name=name, turn=turn, fn=fn, good=good, bad=bad, ctx=ctx)


def list_ref(name, turn, src_turn, idx, good, bad, ctx):
    """Answer names item idx of the list the model produced at src_turn, and it is
    the first list item mentioned (so reciting the whole list does not pass).
    None (ungradable) if the model produced no 3-item list or the target item is
    not distinct from the others."""
    def fn(ans, replies):
        items = parse_list(replies[src_turn])
        if len(items) < 3:
            return None
        keys = [item_key(i) for i in items[:3]]
        k = keys[idx]
        if not k or sum(1 for kk in keys if kk == k) > 1:
            return None
        a = stems(ans)
        pos_k = find_seq(a, k)
        if pos_k < 0:
            return False
        others = [find_seq(a, kk) for kk in keys if kk and kk != k]
        return all(p < 0 or pos_k <= p for p in others)
    return dict(name=name, turn=turn, fn=fn, good=good, bad=bad, ctx=ctx, dynamic=True)


def add_ref(name, turn, src_turn, delta, good, bad, ctx):
    """Answer contains (model's own earlier number) + delta."""
    def fn(ans, replies):
        n = last_int(replies[src_turn])
        if n is None:
            return None
        return mentions(num_pat(n + delta), ans)
    return dict(name=name, turn=turn, fn=fn, good=good, bad=bad, ctx=ctx, dynamic=True)


def not_claim(name, turn, claims, good, bad):
    """Fails if any sentence speaks in the first person about the USER's job/identity
    (e.g. "As a chef in a busy restaurant, my day is dynamic") without addressing the user."""
    job = rx(r"\b(" + claims + r")\b")
    first = rx(r"\b(i|i'm|i've|my|me)\b")
    second = rx(r"\b(you|your|you're)\b")
    def fn(ans, replies):
        if len(words(ans)) < 2:
            return False
        for sent in re.split(r"(?<=[.!?])\s+|\n+", ans):
            if job.search(sent) and first.search(sent) and not second.search(sent):
                return False
        return True
    return dict(name=name, turn=turn, fn=fn, good=good, bad=bad)


# ---------------------------------------------------------------- the battery
TESTS = []


def T(**kw):
    kw.setdefault("system", None)
    kw.setdefault("distance", None)
    kw.setdefault("forced", [])
    kw.setdefault("control_for", None)
    kw.setdefault("max_new", {})
    TESTS.append(kw)


# (1) recall of a user-stated fact at increasing distance (distance = number of
# intervening user turns between the fact and the question).
RECALL = [
    ("R_d2_name", 2, "Hi! My name is Priya, and I'm trying to get more organized this year.", [0, 1],
     "Quick check: what did I tell you my name was?", r"\bpriya\b", "Your name is Priya.", "Your name is Maria.",
     ("Your name is", " Priya", " Maria")),
    ("R_d2_number", 2, "I just moved into apartment 417 in a new building downtown.", [2, 3],
     "What's my apartment number again?", num_pat(417), "Your apartment number is 417.", "Your apartment number is 714.",
     ("Your apartment number is", " 417", " 714")),
    ("R_d4_pet", 4, "I have a cat named Biscuit who loves sitting in cardboard boxes.", [4, 5, 6, 7],
     "What's my cat called?", r"\bbiscuit", "Your cat is called Biscuit.", "Your cat is called Whiskers.",
     ("Your cat is called", " Biscuit", " Whiskers")),
    ("R_d4_city", 4, "I live in Tucson and I work at a small bakery.", [8, 9, 10, 11],
     "Which city do I live in?", r"\btucson\b", "You live in Tucson.", "You live in Phoenix.",
     ("You live in", " Tucson", " Phoenix")),
    ("R_d6_name", 6, "Hello, I'm Marcus. I'm learning to play the guitar.", [0, 2, 4, 6, 8, 10],
     "What's my name?", r"\bmarcus\b", "Your name is Marcus.", "Your name is Michael.",
     ("Your name is", " Marcus", " Michael")),
    ("R_d6_number", 6, "My grandmother turns 94 next week and I want to plan something nice for her.", [1, 3, 5, 7, 9, 11],
     "How old is my grandmother about to be?", num_pat(94) + r"|ninety[- ]four", "She is turning 94.", "She is turning 90.",
     ("Your grandmother is turning", " 94", " 90")),
    ("R_d10_pet", 10, "By the way, my dog is named Waffles.", list(range(10)),
     "What's my dog's name?", r"\bwaffles?\b", "Your dog's name is Waffles.", "Your dog's name is Buddy.",
     ("Your dog's name is", " Waffles", " Buddy")),
]
for tid, dist, fact, dix, q, gold, good, bad, forced in RECALL:
    turns = [fact] + [D[i] for i in dix] + [q]
    T(id=tid, cat="recall", distance=dist, turns=turns,
      checks=[has("recall", len(turns) - 1, gold, good, bad)], forced=[forced])
    T(id=tid + "__ctrl", cat="control", control_for=tid, distance=0, turns=[fact + " " + q],
      checks=[has("recall", 0, gold, good, bad)], forced=[forced])

multi_fact = "Quick intro: I'm Elena, I live in Oslo, and I have a parrot named Kiwi."
multi_q = "Can you summarize the three things I told you about myself at the very start?"
# "summarize what I told you" invites quoting the user in the first person, so the
# role-capture guard is off here (deflection guard stays on).
multi_checks = lambda t: [
    has("name", t, r"\belena\b", "You're Elena, you live in Oslo, and your parrot is Kiwi.", "You're Anna, you live in Bergen, and your parrot is Polly.", capture=False),
    has("city", t, r"\boslo\b", "You're Elena, you live in Oslo, and your parrot is Kiwi.", "You're Anna, you live in Bergen, and your parrot is Polly.", capture=False),
    has("pet", t, r"\bkiwi\b", "You're Elena, you live in Oslo, and your parrot is Kiwi.", "You're Anna, you live in Bergen, and your parrot is Polly.", capture=False),
]
T(id="R_d5_multi", cat="recall", distance=5, turns=[multi_fact] + [D[i] for i in [2, 4, 6, 8, 10]] + [multi_q],
  checks=multi_checks(6))
T(id="R_d5_multi__ctrl", cat="control", control_for="R_d5_multi", distance=0, turns=[multi_fact + " " + multi_q],
  checks=multi_checks(0))

# (2) coreference / elliptical follow-ups
T(id="F_ellipsis_capital", cat="followup", turns=["What is the capital of France?", "And what about Italy?"],
  checks=[has("knows_paris", 0, r"\bparis\b", "The capital of France is Paris.", "The capital of France is Lyon.", guard=False),
          has("ellipsis", 1, r"\brome\b|\broma\b", "The capital of Italy is Rome.", "The capital of Italy is Milan.", guard=False)])
T(id="F_ellipsis_capital__ctrl", cat="control", control_for="F_ellipsis_capital", turns=["What is the capital of Italy?"],
  checks=[has("ellipsis", 0, r"\brome\b|\broma\b", "The capital of Italy is Rome.", "The capital of Italy is Milan.", guard=False)])

T(id="F_translate", cat="followup", turns=["How do you say 'thank you' in Spanish?", "And in French?"],
  checks=[has("knows_gracias", 0, r"gracias", "It's 'gracias'.", "It's 'danke'.", guard=False),
          has("ellipsis", 1, r"\bmerci\b", "In French it's 'merci'.", "In French it's 'gracias'.", guard=False)])
T(id="F_translate__ctrl", cat="control", control_for="F_translate", turns=["How do you say 'thank you' in French?"],
  checks=[has("ellipsis", 0, r"\bmerci\b", "In French it's 'merci'.", "In French it's 'gracias'.", guard=False)])

T(id="F_coref_she", cat="followup", turns=["My brother Tom lives in Denver, and my sister Ana lives in Miami.", "Where does she live?"],
  checks=[binding("coref", 1, r"\bmiami\b", r"\b(she|ana|your sister)\s+(\w+\s+)?lives in denver",
                  "She lives in Miami.", "She lives in Denver.")])
T(id="F_coref_she__ctrl", cat="control", control_for="F_coref_she",
  turns=["My brother Tom lives in Denver, and my sister Ana lives in Miami. Where does she live?"],
  checks=[binding("coref", 0, r"\bmiami\b", r"\b(she|ana|your sister)\s+(\w+\s+)?lives in denver",
                  "She lives in Miami.", "She lives in Denver.")])

T(id="F_whynot", cat="followup", turns=["Is it a good idea to touch a hot stove?", "Why not?"],
  checks=[has("why", 1, r"\bburn|\bpain|\binjur|\bhurt|\bblister|damage (to )?(your )?skin", "Because you could burn yourself.",
              "Because it keeps the kitchen tidy.", guard=False)])
T(id="F_whynot__ctrl", cat="control", control_for="F_whynot", turns=["Why is it a bad idea to touch a hot stove?"],
  checks=[has("why", 0, r"\bburn|\bpain|\binjur|\bhurt|\bblister|damage (to )?(your )?skin", "Because you could burn yourself.",
              "Because it keeps the kitchen tidy.", guard=False)])

EX_TOPIC = r"exercis|health|heart|body|fit|strong|mood|energy|active|muscle|weight|stress|mind|sleep|feel"
T(id="F_shorter", cat="followup", turns=["Write two sentences about why exercise is good for you.", "Make it shorter."],
  checks=[shorter_than("shorter", 1, 0, EX_TOPIC, "Exercise keeps you healthy.",
                       "Exercise strengthens your heart and muscles, and it also improves your mood, sleep, and energy every single day of the week.",
                       ctx=["Exercise strengthens your heart and muscles. It also improves your mood and helps you sleep better."])])
T(id="F_shorter__ctrl", cat="control", control_for="F_shorter", turns=["Write one short sentence about why exercise is good for you."],
  checks=[dict(name="shorter", turn=0, fn=lambda a, r: 3 <= len(words(a)) <= 20 and mentions(EX_TOPIC, a),
               good="Exercise keeps your body and mind healthy.",
               bad="Exercise is good for you because it strengthens your heart, builds muscle, improves your mood, helps you sleep, and lowers stress over time.")])

# (3) instruction persistence
q1 = ["How does a rainbow form?", "What causes earthquakes?", "How do vaccines work?"]
T(id="I_one_sentence", cat="instruction", turns=["From now on, answer every question in exactly one sentence."] + q1,
  checks=[one_sentence(f"one_sentence_t{t}", t, "Vaccines train your immune system to recognize a germ.",
                       "Vaccines train your immune system. They contain weakened germs.") for t in (1, 2, 3)])
T(id="I_one_sentence__ctrl", cat="control", control_for="I_one_sentence", turns=["In exactly one sentence: how do vaccines work?"],
  checks=[one_sentence("one_sentence_t3", 0, "Vaccines train your immune system to recognize a germ.",
                       "Vaccines train your immune system. They contain weakened germs.")])

q2 = ["I like hiking on weekends.", "My favorite food is pasta.", "I'm reading a mystery novel right now."]
T(id="I_end_question", cat="instruction", turns=["For the rest of our chat, always end each of your replies with a question for me."] + q2,
  checks=[end_q(f"end_q_t{t}", t, "Mysteries are fun! Who is your favorite detective?",
                "Mysteries are fun. Enjoy the book!") for t in (1, 2, 3)])
T(id="I_end_question__ctrl", cat="control", control_for="I_end_question",
  turns=["I'm reading a mystery novel right now. Reply briefly and end your reply with a question for me."],
  checks=[end_q("end_q_t3", 0, "Mysteries are fun! Who is your favorite detective?", "Mysteries are fun. Enjoy the book!")])

q3 = ["What is your favorite season and why?", "Name two animals that live in the desert.", "What do bees make?"]
T(id="I_caps", cat="instruction", turns=["Please reply ONLY in capital letters for the rest of this conversation."] + q3,
  checks=[all_caps(f"caps_t{t}", t, "BEES MAKE HONEY AND WAX.", "Bees make honey and wax.") for t in (1, 2, 3)])
T(id="I_caps__ctrl", cat="control", control_for="I_caps", turns=["Reply only in capital letters: what do bees make?"],
  checks=[all_caps("caps_t3", 0, "BEES MAKE HONEY AND WAX.", "Bees make honey and wax.")])

PIRATE_SYS = "You are Captain Pip, a cheerful pirate. Always say 'Arr' somewhere in every reply."
ARR = r"\ba+r{2,}h*\b"
T(id="I_persona", cat="instruction", system=PIRATE_SYS,
  turns=["Hi there!", "What's a good snack for a long trip?", "Tell me about the weather at sea.", "What's your name?"],
  checks=[has_fmt(f"arr_t{t}", t, ARR, "Arr, matey! Hardtack be the best snack.", "Ahoy, matey! Hardtack is the best snack.") for t in (0, 1, 2, 3)]
  + [has("name_pip", 3, r"\bpip\b", "Arr, I be Captain Pip!", "Arr, I be Captain Jack!", guard=False)])
T(id="I_persona__ctrl", cat="control", control_for="I_persona", system=PIRATE_SYS, turns=["What's your name?"],
  checks=[has_fmt("arr_t3", 0, ARR, "Arr, I be Captain Pip!", "Ahoy, I be Captain Pip!"),
          has("name_pip", 0, r"\bpip\b", "Arr, I be Captain Pip!", "Arr, I be Captain Jack!", guard=False)])

# (4) user corrections
CORR = [
    ("K_day", "Please remember that I have a dentist appointment on Monday.", "Actually, sorry, I got that wrong. It's on Tuesday.",
     2, "Which day is my dentist appointment?", r"\btuesday\b", r"\bmonday\b",
     "It's on Tuesday, not Monday.", "Your appointment is on Monday.", ("Your dentist appointment is on", " Tuesday", " Monday")),
    ("K_time", "My meeting with Sam is at 3 pm.", "Correction: the meeting got moved to 4 pm.",
     9, "What time should I show up for my meeting with Sam?", r"(?<![\d:])4(?![\d])|\bfour\b", r"(?<![\d:])3(?![\d])\s*(pm|p\.m|o'clock)|\bthree\b",
     "Show up at 4 pm.", "Show up at 3 pm.", ("You should show up at", " 4", " 3")),
    ("K_color", "My favorite color is blue.", "Hmm, actually I changed my mind. My favorite color is green now.",
     5, "What's my favorite color at this point?", r"\bgreen\b", r"\bblue\b",
     "Your favorite color is green.", "Your favorite color is blue.", ("Your favorite color is", " green", " blue")),
]
for tid, a, b, di, q, new, old, good, bad, forced in CORR:
    T(id=tid, cat="correction", turns=[a, b, D[di], q],
      checks=[has_not_stale("corrected", 3, new, old, good, bad)], forced=[forced])
    T(id=tid + "__ctrl", cat="control", control_for=tid, turns=[a + " " + b + " " + q],
      checks=[has_not_stale("corrected", 0, new, old, good, bad)], forced=[forced])

# (5) referring to the model's own earlier answer
FRUIT_CTX = ["1. Oranges\n2. Kiwi\n3. Strawberries"]
T(id="O_second_item", cat="own_answer",
  turns=["List three fruits that are high in vitamin C. Use a numbered list with just the fruit names.", "What was the second fruit on your list?"],
  checks=[list_ref("second_item", 1, 0, 1, "The second fruit was kiwi.", "The second fruit was oranges.", FRUIT_CTX)],
  max_new={0: 120})
FISH_CTX = ["1. Bubbles\n2. Finn\n3. Goldie"]
T(id="O_last_item", cat="own_answer",
  turns=["Suggest three names for a pet goldfish. Just give a numbered list of names.", "Which name did you put last?"],
  checks=[list_ref("last_item", 1, 0, 2, "The last name was Goldie.", "The last name was Bubbles.", FISH_CTX)],
  max_new={0: 120})
T(id="O_add10", cat="own_answer", turns=["What is 15 times 4?", "Add 10 to your answer."],
  checks=[has("knows_60", 0, num_pat(60), "15 times 4 is 60.", "15 times 4 is 45.", guard=False),
          add_ref("add_to_own", 1, 0, 10, "60 plus 10 is 70.", "60 plus 10 is 60.", ["15 times 4 is 60."])])
T(id="O_add10__ctrl", cat="control", control_for="O_add10", turns=["What is 15 times 4, plus 10?"],
  checks=[has("add_to_own", 0, num_pat(70), "It's 70.", "It's 60.")])

# (6) role integrity (plus global per-turn flags computed in the runner)
T(id="RI_identity", cat="role", turns=["Hi, I'm Jordan and I work as a chef in a busy restaurant.", "What do you do for work?", "And what do I do for work?"],
  checks=[not_claim("assistant_not_user", 1, r"chef|cook|jordan|restaurant|kitchen",
                    "I'm an AI assistant, so I don't have a job like your work as a chef.",
                    "As a chef in a busy restaurant, my day is always dynamic."),
          has("user_job", 2, r"\byou(?:'re|\s+are|\s+work|\s+said|\s+mentioned|\s+told)[^.?!]{0,40}\b(chef|cook)|\byour\s+(?:job|work|role|profession)[^.?!]{0,30}\b(chef|cook)|\bas a (chef|cook), you\b",
              "You work as a chef.", "As a culinary assistant, you would typically assist the chef.")])
T(id="RI_binding", cat="role", turns=["My sister is named Lena and I'm named Oscar.", D[1], "What's my name?"],
  checks=[binding("binding", 2, r"\boscar\b", r"(your name is|you're|you are)\s+lena|\bsister(?:'s name)?,?\s+(?:is\s+)?(?:named\s+|called\s+)?oscar\b", "Your name is Oscar.", "Your name is Lena.")],
  forced=[("Your name is", " Oscar", " Lena")])
T(id="RI_binding__ctrl", cat="control", control_for="RI_binding", turns=["My sister is named Lena and I'm named Oscar. What's my name?"],
  checks=[binding("binding", 0, r"\boscar\b", r"(your name is|you're|you are)\s+lena|\bsister(?:'s name)?,?\s+(?:is\s+)?(?:named\s+|called\s+)?oscar\b", "Your name is Oscar.", "Your name is Lena.")],
  forced=[("Your name is", " Oscar", " Lena")])

# (7) topic return after a digression
T(id="T_party", cat="topic_return",
  turns=["I'm hosting a dinner party for 6 guests on Saturday.", "Unrelated question: how far away is the Moon?",
         "And why does the Moon have phases?", "Okay, back to my party. If each guest eats 2 cupcakes, how many cupcakes should I bake?"],
  checks=[has("party_math", 3, num_pat(12) + r"|\btwelve\b", "You should bake 12 cupcakes.", "You should bake 10 cupcakes.", guard=False)])
T(id="T_party__ctrl", cat="control", control_for="T_party",
  turns=["I'm hosting a dinner party for 6 guests on Saturday. If each guest eats 2 cupcakes, how many cupcakes should I bake?"],
  checks=[has("party_math", 0, num_pat(12) + r"|\btwelve\b", "You should bake 12 cupcakes.", "You should bake 10 cupcakes.", guard=False)])

DRAGON = "I'm writing a story about a young dragon named Ember who is secretly afraid of heights."
DQ = "What is my dragon afraid of, and what is its name?"
dragon_checks = lambda t: [has("fear", t, r"height|high places|being high|high up", "Ember is afraid of heights.", "Ember is afraid of water."),
                           has("name", t, r"\bember\b", "Ember is afraid of heights.", "Blaze is afraid of heights.")]
T(id="T_dragon", cat="topic_return",
  turns=[DRAGON, "Can you suggest a title for it?", "Totally different topic: what's the tallest mountain in the world?",
         "Anyway, back to my story. " + DQ],
  checks=dragon_checks(3))
T(id="T_dragon__ctrl", cat="control", control_for="T_dragon", turns=[DRAGON + " " + DQ], checks=dragon_checks(0))

# (9) free-form chats for loop / repetition metrics
T(id="L_tell_more", cat="loop", turns=["Tell me about the ocean.", "Tell me more.", "Go on.", "What else?", "Keep going.", "Tell me more."],
  checks=[], max_new={i: 200 for i in range(6)})
T(id="L_chitchat", cat="loop",
  turns=["hey, how's it going?", "not bad, just kind of bored today. any ideas?", "hmm, i don't really like reading",
         "maybe. what would you do if you were me?", "haha fair. do you like music?", "cool. wait, what was i complaining about at the start?"],
  checks=[has("topic_recall", 5, r"\bbor(ed|edom|ing)\b|nothing to do", "You said you were bored.", "You said you didn't like reading.")],
  max_new={i: 200 for i in range(6)})


# ---------------------------------------------------------------- self-test
def selftest():
    n = 0
    for t in TESTS:
        for c in t["checks"]:
            replies = list(c.get("ctx", []))
            pad = lambda: replies + [""] * (c["turn"] - len(replies))
            ctx = pad()
            g = c["fn"](c["good"], ctx)
            e = c["fn"]("", ctx)
            b = c["fn"](c["bad"], ctx)
            assert g is True, (t["id"], c["name"], "good fixture failed", c["good"])
            assert not e, (t["id"], c["name"], "empty answer passed")
            assert not b, (t["id"], c["name"], "plausible wrong answer passed", c["bad"])
            if c.get("guard"):
                dfl = c["good"] + " Sorry, I don't remember what you told me earlier."
                assert not c["fn"](dfl, ctx), (t["id"], c["name"], "deflecting echo passed", dfl)
                n += 1
            # vacuous-pass guard: the gold answer must not occur in the question that asks for it
            if "gold" in c:
                q = t["turns"][c["turn"]]
                if t["cat"] != "control":
                    assert not mentions(c["gold"], q), (t["id"], c["name"], "gold answer appears in its own question", q)
                # and must not occur in any distractor
                for d in D:
                    assert not mentions(c["gold"], d), (t["id"], c["name"], "gold in distractor", d)
            n += 3
    # extra unit checks for the helpers the dynamic graders rely on
    assert parse_list("1. **Oranges**: rich\n2. Kiwi - also\n3. Strawberries") == ["Oranges", "Kiwi", "Strawberries"]
    assert item_key("Strawberries") == ("straw",) and item_key("Kiwi") == ("kiwi",)
    assert last_int("15 x 4 = 60.") == 60
    assert n_sentences("Hi there. How are you?") == 2 and n_sentences("A rainbow forms when light refracts in droplets.") == 1
    assert n_sentences("It is approx. 3.5 km, e.g. far.") == 1
    assert ends_with_question("What about you? \U0001F60A") and not ends_with_question("Enjoy!")
    assert asserted(r"\bmonday\b", "It's Monday.") and not asserted(r"\bmonday\b", "It's Tuesday, not Monday.")
    assert not asserted(r"\bblue\b", "You changed it from blue to green.")
    # correction grader: narrating the change is fine, ending on the stale value is not
    kt = [c for t in TESTS if t["id"] == "K_time" for c in t["checks"]][0]
    assert kt["fn"]("It was scheduled for 3 pm and moved to 4 pm, so arrive at 4 pm.", []) is True
    assert kt["fn"]("It moved to 4 pm, but you should show up at 3 pm.", []) is False
    kc = [c for t in TESTS if t["id"] == "K_color" for c in t["checks"]][0]
    assert kc["fn"]("You switched to green, but your favorite color is blue.", []) is False
    # format checks reject loops and verbatim copies of an earlier reply
    cap = [c for t in TESTS if t["id"] == "I_caps" for c in t["checks"]][1]
    assert cap["fn"]("SOLVE:\n\nSOLVE:\n\nSOLVE:\n\nSOLVE:\n\nSOLVE:\n\nSOLVE:", ["", "", ""]) is False
    arr = [c for t in TESTS if t["id"] == "I_persona" for c in t["checks"]][2]
    assert arr["fn"]("Arr! Let's get to the fight!", ["Arr! Let's get to the fight!", "x", ""]) is False
    assert arr["fn"]("Arr! The sea be rough.", ["Arr! Let's get to the fight!", "x", ""]) is True
    # deflection detector: curly apostrophes and identity disclaimers
    assert deflects("I don\u2019t have access to your personal schedule.")
    assert deflects("I'm sorry, but I'm Qwen, a language model created by Alibaba Cloud. I don't have a cat.")
    assert deflects("I am a large language model, and I do not have a name.")
    assert not deflects("Your cat is called Biscuit.") and not deflects("You live in Tucson, Arizona.")
    # perspective (role-capture) guard on user-fact checks
    g = lambda tid: [c for t in TESTS if t["id"] == tid for c in t["checks"]][0]
    assert g("R_d6_name")["fn"]("Hello! I'm Marcus. What's your name?", []) is False
    assert g("R_d6_name")["fn"]("I remember! Your name is Marcus.", []) is True
    assert g("R_d6_name")["fn"]("Hello Marcus! Welcome to the world of music!", []) is False
    assert g("R_d2_name")["fn"]("Okay, Priya! It's good to know. What's on your mind?", []) is False
    assert g("R_d2_name")["fn"]("Hi Priya! You told me your name is Priya.", []) is True
    assert g("R_d10_pet")["fn"]("By the way, my dog is named **Waffles**! \U0001F60A", []) is False
    assert g("RI_binding")["fn"]("I'm Oscar.", []) is False
    assert g("RI_binding")["fn"]("Since you're naming your sister Oscar, it's a great choice.", []) is False
    assert g("RI_binding")["fn"]("Lena is your sister, and Oscar is yours.", []) is True
    assert g("RI_binding")["fn"]("Your sister's name is Oscar.", []) is False
    assert g("RI_binding")["fn"]("Your name is Oscar, and your sister's name is Lena.", []) is True
    assert g("K_color")["fn"]("My favorite color is green.", []) is False
    ri = [c for t in TESTS if t["id"] == "RI_identity" for c in t["checks"]][0]
    assert ri["fn"]("I enjoy tasks that help the restaurant succeed.", []) is False
    assert ri["fn"]("I'm Qwen, an AI assistant. How can I help?", []) is True
    ll = list_ref("x", 1, 0, 2, "", "", [])
    gctx = ["1. Rainbow Goldfish\n2. Goldfish Gold\n3. Goldy Goldfish"]
    assert ll["fn"]("The last name I put was Goldy Goldfish.", gctx) is True
    assert ll["fn"]("The last name was Rainbow Goldfish.", gctx) is False
    assert ll["fn"]("Bubbles", ["1. Bubbles\n2. Bubbles\n3. Bubbles"]) is None
    # dynamic list grader: reciting the whole list must fail
    lr = list_ref("x", 1, 0, 1, "", "", [])
    assert lr["fn"]("Oranges, Kiwi and Strawberries.", FRUIT_CTX) is False
    assert lr["fn"]("The second one was kiwi.", FRUIT_CTX) is True
    assert lr["fn"]("anything", ["no list here"]) is None
    ids = [t["id"] for t in TESTS]
    assert len(ids) == len(set(ids))
    return n, len(TESTS)


if __name__ == "__main__":
    n, k = selftest()
    multi = [t for t in TESTS if t["cat"] != "control"]
    print(f"selftest OK: {n} grader assertions over {k} conversations "
          f"({len(multi)} multi-turn, {k - len(multi)} single-turn controls)")
    lens = [len(t["turns"]) for t in multi]
    print("multi-turn lengths:", sorted(lens))
