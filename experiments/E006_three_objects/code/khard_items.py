"""Long-tail knowledge items (tier 4) with open-book twins, copied verbatim from
research/capacity_probe/khard.py (items and build() only, no model code)."""
import items as I

# (question, answer prefix, gold, foil, fact sentence)
KHARD = [
    ("What is the capital of Bhutan?", "The capital of Bhutan is", "Thimphu", "Paro", "the capital of Bhutan is Thimphu"),
    ("What is the capital of Burkina Faso?", "The capital of Burkina Faso is", "Ouagadougou", "Bamako", "the capital of Burkina Faso is Ouagadougou"),
    ("What is the capital of Kyrgyzstan?", "The capital of Kyrgyzstan is", "Bishkek", "Osh", "the capital of Kyrgyzstan is Bishkek"),
    ("What is the capital of Laos?", "The capital of Laos is", "Vientiane", "Luang Prabang", "the capital of Laos is Vientiane"),
    ("What is the capital of Tanzania?", "The capital of Tanzania is", "Dodoma", "Dar es Salaam", "the capital of Tanzania is Dodoma"),
    ("What is the capital of Myanmar?", "The capital of Myanmar is", "Naypyidaw", "Yangon", "the capital of Myanmar is Naypyidaw"),
    ("What is the capital of Belize?", "The capital of Belize is", "Belmopan", "Belize City", "the capital of Belize is Belmopan"),
    ("What is the capital of Malawi?", "The capital of Malawi is", "Lilongwe", "Blantyre", "the capital of Malawi is Lilongwe"),
    ("What is the capital of Ivory Coast?", "The capital of Ivory Coast is", "Yamoussoukro", "Abidjan", "the capital of Ivory Coast is Yamoussoukro"),
    ("What is the capital of Estonia?", "The capital of Estonia is", "Tallinn", "Tartu", "the capital of Estonia is Tallinn"),
    ("What is the capital of Slovenia?", "The capital of Slovenia is", "Ljubljana", "Maribor", "the capital of Slovenia is Ljubljana"),
    ("What is the capital of Uruguay?", "The capital of Uruguay is", "Montevideo", "Salto", "the capital of Uruguay is Montevideo"),
    ("What is the capital of Paraguay?", "The capital of Paraguay is", "Asuncion", "Encarnacion", "the capital of Paraguay is Asuncion"),
    ("What is the capital of Mongolia?", "The capital of Mongolia is", "Ulaanbaatar", "Erdenet", "the capital of Mongolia is Ulaanbaatar"),
    ("Who wrote The Master and Margarita?", "The Master and Margarita was written by", "Mikhail Bulgakov", "Boris Pasternak", "The Master and Margarita was written by Mikhail Bulgakov"),
    ("Who composed The Rite of Spring?", "The Rite of Spring was composed by", "Igor Stravinsky", "Sergei Rachmaninoff", "The Rite of Spring was composed by Igor Stravinsky"),
    ("Who painted The Garden of Earthly Delights?", "The Garden of Earthly Delights was painted by", "Hieronymus Bosch", "Pieter Bruegel", "The Garden of Earthly Delights was painted by Hieronymus Bosch"),
    ("Who wrote The Brothers Karamazov?", "The Brothers Karamazov was written by", "Fyodor Dostoevsky", "Leo Tolstoy", "The Brothers Karamazov was written by Fyodor Dostoevsky"),
    ("Who wrote Things Fall Apart?", "Things Fall Apart was written by", "Chinua Achebe", "Wole Soyinka", "Things Fall Apart was written by Chinua Achebe"),
    ("Who painted The Persistence of Memory?", "The Persistence of Memory was painted by", "Salvador Dali", "Rene Magritte", "The Persistence of Memory was painted by Salvador Dali"),
    ("Who wrote Don Quixote?", "Don Quixote was written by", "Miguel de Cervantes", "Lope de Vega", "Don Quixote was written by Miguel de Cervantes"),
    ("Who composed The Magic Flute?", "The Magic Flute was composed by", "Wolfgang Amadeus Mozart", "Joseph Haydn", "The Magic Flute was composed by Wolfgang Amadeus Mozart"),
    ("Who proposed a Sun-centered model of the solar system in 1543?", "It was proposed by", "Nicolaus Copernicus", "Tycho Brahe", "a Sun-centered model of the solar system was proposed in 1543 by Nicolaus Copernicus"),
    ("What is the chemical symbol for tungsten?", "The chemical symbol for tungsten is", "W", "Tu", "the chemical symbol for tungsten is W"),
    ("What is the chemical symbol for potassium?", "The chemical symbol for potassium is", "K", "P", "the chemical symbol for potassium is K"),
    ("Which element has atomic number 26?", "The element with atomic number 26 is", "iron", "nickel", "the element with atomic number 26 is iron"),
    ("In which year did the Berlin Wall fall?", "The Berlin Wall fell in", "1989", "1991", "the Berlin Wall fell in 1989"),
    ("In which year did the French Revolution begin?", "The French Revolution began in", "1789", "1776", "the French Revolution began in 1789"),
    ("What is the largest moon of Saturn?", "The largest moon of Saturn is", "Titan", "Rhea", "the largest moon of Saturn is Titan"),
    ("What is the largest moon of Neptune?", "The largest moon of Neptune is", "Triton", "Nereid", "the largest moon of Neptune is Triton"),
    ("What is the longest river in Europe?", "The longest river in Europe is the", "Volga", "Danube", "the longest river in Europe is the Volga"),
    ("What is the official language of Suriname?", "The official language of Suriname is", "Dutch", "Portuguese", "the official language of Suriname is Dutch"),
    ("What is the currency of Switzerland?", "The currency of Switzerland is the", "franc", "euro", "the currency of Switzerland is the franc"),
    ("What is the currency of India?", "The currency of India is the", "rupee", "taka", "the currency of India is the rupee"),
    ("What is the deepest ocean trench?", "The deepest ocean trench is the", "Mariana", "Tonga", "the deepest ocean trench is the Mariana Trench"),
    ("Who discovered the electron?", "The electron was discovered by", "J.J. Thomson", "Ernest Rutherford", "the electron was discovered by J.J. Thomson"),
    ("Who wrote One Hundred Years of Solitude?", "It was written by", "Gabriel Garcia Marquez", "Mario Vargas Llosa", "One Hundred Years of Solitude was written by Gabriel Garcia Marquez"),
    ("Which planet has the moon Io?", "Io is a moon of", "Jupiter", "Saturn", "Io is a moon of Jupiter"),
    ("What is the capital of Kazakhstan?", "The capital of Kazakhstan is", "Astana", "Almaty", "the capital of Kazakhstan is Astana"),
    ("What is the capital of Nigeria?", "The capital of Nigeria is", "Abuja", "Lagos", "the capital of Nigeria is Abuja"),
]


def build():
    rng = __import__("random").Random(7)
    out = []
    for q, pre, gold, foil, fact in KHARD:
        c = {"gold": " " + gold, "foil": " " + foil}
        out.append(dict(task="K4_closed", prompt=I.transcript([], q, pre), cands=c))
        d = rng.sample(I.DISTRACTORS, 2)
        turns = [(f"Fun fact I read today: {fact}.", "Interesting, thanks for sharing!")] + d
        out.append(dict(task="K4_open", prompt=I.transcript(turns, q, pre), cands=c))
    return out
