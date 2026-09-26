"""Item generator for the capacity-lens likelihood probe.

Every in-context item puts ALL the information needed to answer inside the
conversation, so a failure is a failure to route/compose/update, not missing
world knowledge. The knowledge items (K) are the opposite: closed book, nothing
in context. Every in-context foil is a value that DOES appear in the context
(a same-type distractor), unlike the probe lane's never-seen foil.

Format: plain "User:/Assistant:" transcript for every model (no chat template),
so the probe reads the representation, not a post-training reply policy.
"""
import random

USER_NAMES = ["Priya", "Marcus", "Elena", "Oscar", "Jordan", "Nadia", "Tobias", "Hannah",
              "Rafael", "Ingrid", "Kofi", "Mei", "Dmitri", "Lucia", "Samir", "Greta"]
# female names only for sister/aunt, so gender cannot shortcut the 2-hop items
FEMALE_NAMES = ["Lena", "Chloe", "Beatrice", "Zara", "Ruth", "Yuki", "Amara", "Iris",
                "Nadia", "Hannah", "Ingrid", "Mei", "Lucia", "Greta", "Freya", "Talia"]
ASSIST_NAMES = ["Sam", "Robin", "Alex", "Casey", "Morgan", "Quinn", "Riley", "Jamie"]
PET_NAMES = ["Biscuit", "Pepper", "Waffles", "Mango", "Pickle", "Noodle", "Ziggy", "Clover",
             "Juniper", "Toffee", "Pumpkin", "Marble", "Nugget", "Sprout", "Olive", "Pretzel"]
JOBS = ["dentist", "pilot", "teacher", "chef", "nurse", "lawyer", "farmer", "plumber",
        "librarian", "baker", "painter", "banker", "tailor", "barber", "surgeon", "carpenter"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

DISTRACTORS = [
    ("Can you give me a tip for sleeping better?",
     "Try to keep a regular schedule and avoid screens for an hour before bed."),
    ("What's a good way to stay focused while studying?",
     "Work in short blocks with small breaks, and keep your phone in another room."),
    ("Why do leaves change color in autumn?",
     "As days get shorter, trees stop making chlorophyll, so the green fades and other pigments show."),
    ("How long should I boil an egg for a soft yolk?",
     "About six minutes in boiling water, then move it to cold water to stop the cooking."),
    ("Any advice for a first job interview?",
     "Research the company, prepare a few short stories about your work, and arrive a little early."),
    ("What's an easy houseplant for beginners?",
     "A snake plant is hard to kill: it likes indirect light and only needs water every couple of weeks."),
    ("How do I make my phone battery last longer?",
     "Lower the screen brightness, close apps you are not using, and turn off location when you do not need it."),
    ("Why is exercise good for mood?",
     "It releases chemicals that reduce stress and it usually improves sleep, which also helps mood."),
    ("How can I learn a new language faster?",
     "Practice a little every day, speak out loud early, and learn the most common words first."),
    ("What's a quick healthy lunch idea?",
     "A wrap with beans, rice, salsa and some vegetables is fast, filling and cheap."),
    ("How do I keep bread from going stale?",
     "Store it in a paper bag at room temperature, or freeze slices you will not eat within two days."),
    ("Why does the ocean taste salty?",
     "Rivers carry dissolved minerals from rocks into the sea, and when water evaporates the salt stays behind."),
    ("How do I start saving money?",
     "Set up an automatic transfer on payday, even a small one, and track where your spending goes for a month."),
    ("What's a fun thing to do on a rainy afternoon?",
     "You could try a new recipe, start a puzzle, or pick a film series and watch it in order."),
    ("How often should I water a cactus?",
     "Only when the soil is completely dry, which is often every two to four weeks."),
    ("Why do we get hiccups?",
     "A sudden spasm of the diaphragm makes you breathe in sharply, and the vocal cords snap shut."),
]

# Closed-book knowledge control (two-way forced choice; nothing in context).
# (question, answer prefix, gold, foil, tier) tier 1 = very common, 2 = common, 3 = less common
KNOWLEDGE = [
    ("What is the capital of France?", "The capital of France is", "Paris", "Lyon", 1),
    ("What is the capital of Japan?", "The capital of Japan is", "Tokyo", "Osaka", 1),
    ("What is the capital of Italy?", "The capital of Italy is", "Rome", "Milan", 1),
    ("What planet is known as the Red Planet?", "The Red Planet is", "Mars", "Venus", 1),
    ("How many legs does a spider have?", "A spider has", "eight", "six", 1),
    ("What gas do plants absorb from the air?", "Plants absorb", "carbon", "oxygen", 1),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by", "Shakespeare", "Dickens", 1),
    ("What is the largest ocean?", "The largest ocean is the", "Pacific", "Atlantic", 1),
    ("What is frozen water called?", "Frozen water is called", "ice", "steam", 1),
    ("Which animal is known as the king of the jungle?", "The king of the jungle is the", "lion", "tiger", 1),
    ("What is the capital of Australia?", "The capital of Australia is", "Canberra", "Sydney", 2),
    ("What is the capital of Canada?", "The capital of Canada is", "Ottawa", "Toronto", 2),
    ("What is the capital of Brazil?", "The capital of Brazil is", "Brasilia", "Rio", 2),
    ("What is the capital of Turkey?", "The capital of Turkey is", "Ankara", "Istanbul", 2),
    ("What is the chemical symbol for gold?", "The chemical symbol for gold is", "Au", "Ag", 2),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by", "Leonardo", "Michelangelo", 2),
    ("What is the hardest natural substance?", "The hardest natural substance is", "diamond", "quartz", 2),
    ("Which planet is closest to the Sun?", "The planet closest to the Sun is", "Mercury", "Venus", 2),
    ("Who developed the theory of relativity?", "The theory of relativity was developed by", "Einstein", "Newton", 2),
    ("What is the largest planet in our solar system?", "The largest planet is", "Jupiter", "Saturn", 2),
    ("In which country are the pyramids of Giza?", "The pyramids of Giza are in", "Egypt", "Mexico", 2),
    ("What is the longest river in South America?", "The longest river in South America is the", "Amazon", "Orinoco", 2),
    ("Which organ pumps blood through the body?", "Blood is pumped by the", "heart", "liver", 2),
    ("What language is mainly spoken in Brazil?", "The main language of Brazil is", "Portuguese", "Spanish", 2),
    ("Who was the first person to walk on the Moon?", "The first person to walk on the Moon was", "Neil", "Buzz", 2),
    ("What is the capital of Kazakhstan?", "The capital of Kazakhstan is", "Astana", "Almaty", 3),
    ("What is the capital of Nigeria?", "The capital of Nigeria is", "Abuja", "Lagos", 3),
    ("What is the capital of New Zealand?", "The capital of New Zealand is", "Wellington", "Auckland", 3),
    ("What is the capital of Switzerland?", "The capital of Switzerland is", "Bern", "Zurich", 3),
    ("What is the capital of Morocco?", "The capital of Morocco is", "Rabat", "Casablanca", 3),
    ("What is the capital of Vietnam?", "The capital of Vietnam is", "Hanoi", "Saigon", 3),
    ("Which element has atomic number 1?", "The element with atomic number 1 is", "hydrogen", "helium", 3),
    ("Who wrote One Hundred Years of Solitude?", "One Hundred Years of Solitude was written by", "Gabriel", "Jorge", 3),
    ("Who composed The Four Seasons?", "The Four Seasons was composed by", "Vivaldi", "Bach", 3),
    ("What is the currency of Japan?", "The currency of Japan is the", "yen", "won", 3),
    ("Which planet has the most prominent rings?", "The planet with the most prominent rings is", "Saturn", "Neptune", 3),
    ("Who discovered penicillin?", "Penicillin was discovered by", "Alexander", "Louis", 3),
    ("What is the smallest prime number?", "The smallest prime number is", "two", "one", 3),
    ("In which year did World War II end?", "World War II ended in", "1945", "1939", 3),
    ("What is the largest desert in the world by area?", "The largest desert by area is", "Antarctica", "Sahara", 3),
    ("Who was the first President of the United States?", "The first President of the United States was", "George", "Thomas", 2),
    ("Which blood cells carry oxygen?", "Oxygen is carried by", "red", "white", 2),
    ("What is the tallest mountain on Earth?", "The tallest mountain on Earth is Mount", "Everest", "Kilimanjaro", 1),
    ("What is the main ingredient of guacamole?", "The main ingredient of guacamole is", "avocado", "tomato", 2),
    ("Which country gifted the Statue of Liberty to the USA?", "The Statue of Liberty was a gift from", "France", "England", 2),
    ("What is the capital of Egypt?", "The capital of Egypt is", "Cairo", "Alexandria", 2),
    ("What is the capital of Spain?", "The capital of Spain is", "Madrid", "Barcelona", 1),
    ("What is the capital of Germany?", "The capital of Germany is", "Berlin", "Munich", 1),
    ("Who wrote Pride and Prejudice?", "Pride and Prejudice was written by", "Jane", "Emily", 3),
    ("What is the capital of Peru?", "The capital of Peru is", "Lima", "Cusco", 3),
    ("What is the capital of Kenya?", "The capital of Kenya is", "Nairobi", "Mombasa", 3),
    ("Which scientist proposed the three laws of motion?", "The three laws of motion were proposed by", "Newton", "Galileo", 2),
    ("How many continents are there?", "There are", "seven", "five", 1),
    ("What is the boiling point of water in Celsius?", "Water boils at", "100", "90", 1),
    ("Which is the largest mammal?", "The largest mammal is the", "blue", "African", 2),
    ("Which instrument has 88 keys?", "The instrument with 88 keys is the", "piano", "organ", 2),
    ("What is the capital of Argentina?", "The capital of Argentina is", "Buenos", "Cordoba", 2),
    ("What is the capital of South Korea?", "The capital of South Korea is", "Seoul", "Busan", 2),
    ("Who painted The Starry Night?", "The Starry Night was painted by", "Vincent", "Claude", 3),
    ("What is the chemical formula for table salt?", "The chemical formula for table salt is", "NaCl", "KCl", 3),
]


def transcript(turns, question, prefix):
    lines = []
    for u, a in turns:
        lines.append(f"User: {u}")
        lines.append(f"Assistant: {a}")
    lines.append(f"User: {question}")
    lines.append(f"Assistant: {prefix}")
    return "\n".join(lines)


def distractors(rng, n):
    return rng.sample(DISTRACTORS, n) if n <= len(DISTRACTORS) else [rng.choice(DISTRACTORS) for _ in range(n)]


def build(n_scen=32, distances=(0, 4, 10), seed=1234):
    """Return a list of item dicts: {task, cond, d, prompt, cands: {label: text}, gold, foils: [...]}"""
    rng = random.Random(seed)
    items = []
    for d in distances:
        for s in range(n_scen):
            # ---- R1: owner binding (my cat vs my sister's cat), both in context
            a, b = rng.sample(PET_NAMES, 2)
            first_mine = rng.random() < 0.5
            intro = (f"Hi! Quick background: my cat is named {a}, and my sister's cat is named {b}."
                     if first_mine else
                     f"Hi! Quick background: my sister's cat is named {b}, and my cat is named {a}.")
            turns = [(intro, "Nice, two cats in the family! How can I help today?")] + distractors(rng, d)
            items.append(dict(task="R1_owner", cond="mine", d=d,
                              prompt=transcript(turns, "What's my cat's name again?", "Your cat's name is"),
                              cands={"gold": " " + a, "foil": " " + b}))
            items.append(dict(task="R1_owner", cond="sister", d=d,
                              prompt=transcript(turns, "What's my sister's cat's name again?", "Your sister's cat's name is"),
                              cands={"gold": " " + b, "foil": " " + a}))

            # ---- R2: two-hop composition (sister -> Lena -> dentist) vs one-hop (Lena -> dentist)
            sis, cou = rng.sample(FEMALE_NAMES, 2)  # cou = aunt
            js, jc = rng.sample(JOBS, 2)
            names_msg = (f"My sister's name is {sis} and my aunt's name is {cou}."
                         if rng.random() < 0.5 else
                         f"My aunt's name is {cou} and my sister's name is {sis}.")
            jobs_msg = (f"By the way, {sis} works as a {js}, and {cou} works as a {jc}."
                        if rng.random() < 0.5 else
                        f"By the way, {cou} works as a {jc}, and {sis} works as a {js}.")
            dd = distractors(rng, d)
            turns = ([(names_msg, "Thanks for telling me about your family.")] + dd[: d // 2] +
                     [(jobs_msg, "Those sound like interesting jobs.")] + dd[d // 2:])
            for who, name, gold, foil in (("sister", sis, js, jc), ("aunt", cou, jc, js)):
                items.append(dict(task="R2_twohop", cond=who, d=d,
                                  prompt=transcript(turns, f"What does my {who} do for work?", f"Your {who} works as a"),
                                  cands={"gold": " " + gold, "foil": " " + foil}))
                items.append(dict(task="R2_onehop", cond=who, d=d,
                                  prompt=transcript(turns, f"What does {name} do for work?", f"{name} works as a"),
                                  cands={"gold": " " + gold, "foil": " " + foil}))

            # ---- U: state updates, k = 1, 2, 3 corrections (latest vs original, latest vs previous)
            for k in (1, 2, 3):
                days = rng.sample(DAYS, k + 1)
                turns = [(f"I have a dentist appointment on {days[0]}.", f"Okay, noted: {days[0]}.")]
                for i in range(1, k + 1):
                    turns += distractors(rng, 1)
                    turns.append((f"Actually, the appointment got moved to {days[i]}.", f"Got it, moved to {days[i]}."))
                turns += distractors(rng, d)
                cands = {"gold": " " + days[k], "orig": " " + days[0]}
                if k >= 2:
                    cands["prev"] = " " + days[k - 1]
                items.append(dict(task=f"U_k{k}", cond="update", d=d,
                                  prompt=transcript(turns, "What day is my dentist appointment?", "Your dentist appointment is on"),
                                  cands=cands))

            # ---- P: perspective / speaker binding (user vs sister vs assistant names)
            sn = rng.choice(FEMALE_NAMES)
            un = rng.choice([n for n in USER_NAMES if n != sn])
            an = rng.choice(ASSIST_NAMES)
            turns = [(f"Hi, I'm {un}. My sister {sn} told me about you.",
                      f"Hi {un}, nice to meet you! I'm {an}, your assistant for today.")] + distractors(rng, d)
            items.append(dict(task="P_myname", cond="user", d=d,
                              prompt=transcript(turns, "Sorry, what's my name again?", "Your name is"),
                              cands={"gold": " " + un, "sister": " " + sn, "assistant": " " + an}))
            items.append(dict(task="P_yourname", cond="assistant", d=d,
                              prompt=transcript(turns, "And what's your name?", "My name is"),
                              cands={"gold": " " + an, "user": " " + un}))
    for q, pre, gold, foil, tier in KNOWLEDGE:
        items.append(dict(task="K_closedbook", cond=f"tier{tier}", d=0,
                          prompt=transcript([], q, pre), cands={"gold": " " + gold, "foil": " " + foil}))
    return items


if __name__ == "__main__":
    it = build()
    from collections import Counter
    print(len(it), Counter((x["task"], x["d"]) for x in it))
    for x in it[:3] + [x for x in it if x["task"] == "U_k3"][-1:] + [x for x in it if x["task"] == "R2_twohop"][-1:]:
        print("-" * 60); print(x["task"], x["cond"], x["d"], x["cands"]); print(x["prompt"])
