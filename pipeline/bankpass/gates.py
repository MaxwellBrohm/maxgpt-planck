"""Item checks and held-out gates for bank items (BANKPASS s2c, s2d). Every check reuses the checker's own pattern
where one exists (gate.DASH_RE, heldout.DIGIT_RE, lexicons.*), so a bank item and a rendered turn are judged by the
same rule. Held-out: E004 vocab_hits and echo_hits on the template with holes as type tags AND on random fills (5-grams
form across hole edges), pool_ok on pool values, and the E004 frame test on every (query, statement) pair of a key.
RC-12 lists join automatically through heldout.load_ext when they land (the gate hash changes, admit refuses stale)."""
import random
import re

import banks as B
import banks_keys as BK
import gate as G
import heldout
import lexicons as L
import pools as P
from bankpass import specs, store

PROMPT_N, MINED_N, N_FILLS = 5, 6, 20
TAG = {"v": "{v}", "av": "{v}", "old": "{v}", "X": "{v}", "n": "{v}", "A": "{v}", "N": "{v}", "W": "{v}",
       "items": "{v}", "e": "{o}", "ord": "{v}", "o": "{o}", "L": "{o}", "lab": "{o}", "t": "{o}", "pred": "is {v}",
       "p": "it", "p_obj": "it", "M": ""}
_HOLE = re.compile(r"\{(\w+)\}")
_W = re.compile(r"[a-z0-9']+")


def words(s):
    return _W.findall(s.lower())


def tagged(template):
    """holes -> E004 type tags (<v>, <o>), so the frame test reads a template as text_e004.normalize would."""
    return _HOLE.sub(lambda m: TAG.get(m.group(1), "{v}"), template).strip()


def bare(template):
    return re.sub(r"\s+", " ", _HOLE.sub(" ", template)).strip()


def item_checks(text, bank, spec=None, prompt=None, mined=()):
    """[(code, detail)] for one template (s2c). prompt: the call's prompt text; mined: example sentences shown."""
    spec = spec or specs.line_specs()[bank]
    s, out = store.straight(text), []
    plain = bare(s)
    for code, rx in (("DASH", G.DASH_RE), ("DIGIT", heldout.DIGIT_RE), ("MARKDOWN", L.MARKDOWN_RE),
                     ("EMOJI", L.EMOJI_RE), ("THOUGHT_TAG", L.THOUGHT_TAG_RE), ("ROLE_LABEL", L.ROLE_LABEL_RE),
                     ("STAGE_DIR", L.STAGE_RE), ("END_IN_TURN", L.END_TOKEN_RE), ("AI_ISM", L.AI_ISM_RE),
                     ("SAFETY", L.SAFETY_RE)):
        m = rx.search(plain if code != "STAGE_DIR" else s)
        if m:
            out.append((code, m.group(0).strip()))
    if spec["side"] == "system" and L.ASSIST_ISM_RE.search(plain):
        out.append(("AI_ISM", L.ASSIST_ISM_RE.search(plain).group(0)))
    if any(c.isalpha() and not ("A" <= c <= "Z" or "a" <= c <= "z") for c in plain):
        out.append(("NON_ENGLISH", "non-latin or accented letter"))
    elif len(words(plain)) >= 8 and not set(words(plain)) & (L.FUNCTION_WORDS | L.STOPWORDS):
        out.append(("NON_ENGLISH", "no English function word"))
    if spec["side"] == "user" and L.USER_VOICE_RE.search(plain):
        out.append(("USER_VOICE", L.USER_VOICE_RE.search(plain).group(0)))
    if spec["side"] == "user" and bank != "swap.q" and re.search(r"(?<![a-z])your\s+\{(?:o|L|lab)\}", s, re.I):
        out.append(("PERSPECTIVE", "your <own thing>"))
    n_w = len(words(_HOLE.sub("x", s)))
    if not spec["min_w"] <= n_w <= spec["max_w"]:
        out.append(("LEN_ITEM", f"{n_w} words"))
    out += [("HOLE", p) for p in specs.hole_problems(bank, s, spec)]
    role = spec.get("role")
    if role in ("plant", "corr", "twin") and "?" in s:
        out.append(("SPEECH_FORM", "statement holds a question mark"))
    if role in ("query", "bait") and not s.rstrip().endswith("?"):
        out.append(("SPEECH_FORM", "question without a question mark"))
    if spec["class"] == "M" and not re.fullmatch(r"[A-Za-z][A-Za-z' ,]*,", s):
        out.append(("MARKER_FORM", "a marker is words ending in a comma"))
    out += prompt_echo(s, prompt, mined)
    return out


def prompt_echo(text, prompt=None, mined=()):
    """BANK_PROMPT_ECHO: a word 5-gram shared with the prompt or a 6-gram shared with a mined example (s0)."""
    w, out = words(bare(text)), []
    if prompt:
        g = {tuple(words(prompt)[i:i + PROMPT_N]) for i in range(len(words(prompt)) - PROMPT_N + 1)}
        hit = [x for x in (tuple(w[i:i + PROMPT_N]) for i in range(len(w) - PROMPT_N + 1)) if x in g]
        if hit:
            out.append(("BANK_PROMPT_ECHO", "prompt: " + " ".join(hit[0])))
    for ex in mined:
        ew = words(ex)
        g = {tuple(ew[i:i + MINED_N]) for i in range(len(ew) - MINED_N + 1)}
        hit = [x for x in (tuple(w[i:i + MINED_N]) for i in range(len(w) - MINED_N + 1)) if x in g]
        if hit:
            out.append(("BANK_PROMPT_ECHO", "example: " + " ".join(hit[0])))
            break
    return out


# ---- fills ----------------------------------------------------------------------------------------------------
def sample_fill(bank, spec, rng):
    """one random value per hole of the bank, drawn from the loaded pools (FAKE until real pools are installed)."""
    key = spec.get("key")
    d = BK.KEYS.get(key) if key else None
    nouns = P.pool(d["noun"]).values if d and d["noun"] else P.pool("plan").values
    if d and d["noun"] == "relation":      # W3 fix: a {p} template needs a relation with pronouns (KeyError 'p')
        nouns = [x for x in nouns if BK.pronouns(key, x)[0]] or nouns
    noun = rng.choice(nouns)
    vt = d["vtype"] if d else "name"
    v = rng.choice(P.pool(vt).values)
    old = rng.choice([x for x in P.pool(vt).values if x != v])
    p, p_obj = BK.pronouns(key, noun) if d else ("it", "it")
    lt = rng.choice(sorted(B.LIST_NAMES))
    items = rng.sample(P.pool({"grocery": "grocery", "city": "city", "name": "name", "chore": "chore"}[lt]).values, 3)
    vt_l = rng.choice(sorted(B.LOOKUP_PRED))
    kind = rng.choice(P.pool("entity_kind").values)
    return dict(v=v, av=B.av(v, P.features(vt, v)), old=old, o=noun, p=p, p_obj=p_obj, M=rng.choice(["", B.MARKERS[0]]),
                L=B.LIST_NAMES[lt], items=B.join_items(items), t=rng.choice(P.pool("topic").values),
                A=rng.choice(P.pool("assistant_name").values), N=rng.choice(P.pool("number_word").values),
                X=rng.choice(P.pool("name").values), W=rng.choice(P.pool("avoid_word").values),
                n=rng.choice(P.pool("name").values), lab=BK.KEYS["job"]["label"], ord=rng.choice(P.pool("ordinal").values),
                e=f"{P.nonce(rng)} {kind.capitalize()}", pred=B.LOOKUP_PRED[vt_l].format(v=rng.choice(P.pool(vt_l).values)))


def heldout_checks(template, bank, spec=None, n_fills=N_FILLS, seed=0):
    """[(code, detail)]: HELDOUT_VOCAB / HELDOUT_ECHO on the tagged template and on n_fills random fills."""
    spec = spec or specs.line_specs()[bank]
    out = []
    v = [h for h in heldout.vocab_hits(bare(template)) if h != "digit"]
    if v:
        out.append(("HELDOUT_VOCAB", v[0]))
    e = heldout.echo_hits(tagged(template))
    if e:
        out.append(("HELDOUT_ECHO", "template " + e[0]))
    rng = random.Random(f"bankpass-fill:{seed}:{bank}:{template}")
    for _ in range(n_fills):
        f = sample_fill(bank, spec, rng)
        txt = B.fill(template, **f)
        v = [h for h in heldout.vocab_hits(txt) if h != "digit"]
        e = heldout.echo_hits(txt, [f["o"]], [f["v"], f["old"]])
        if v or e:
            out.append(("HELDOUT_VOCAB", "fill " + v[0]) if v else ("HELDOUT_ECHO", "fill " + e[0]))
            break
    return out


def pool_checks(value, vtype):
    """a pool value: pool_ok (held-out phrases, head nouns, digits, E004 5-grams) and the form checks."""
    out = [] if heldout.pool_ok(value) else [("HELDOUT_POOL", value)]
    for code, rx in (("DASH", G.DASH_RE), ("MARKDOWN", L.MARKDOWN_RE), ("EMOJI", L.EMOJI_RE), ("SAFETY", L.SAFETY_RE)):
        if rx.search(value):
            out.append((code, value))
    if any(c.isalpha() and not ("A" <= c <= "Z" or "a" <= c <= "z") for c in value):
        out.append(("NON_ENGLISH", value))
    if not 1 <= len(value.split()) <= 4:
        out.append(("LEN_ITEM", value))
    return out


SEED_W = (4, 80)


def seed_checks(text):
    """a persona seed (prompt side only, s3 Q): no held-out term, honorific or digit (vocab_hits), no E004 5-gram,
    no FAKE-list or role-label hit, plain letters only, 4 to 80 words."""
    out = [("HELDOUT_VOCAB", h) for h in heldout.vocab_hits(text)[:1]]
    out += [("HELDOUT_ECHO", h) for h in heldout.echo5(text)[:1]]
    for code, rx in (("SAFETY", L.SAFETY_RE), ("ROLE_LABEL", L.ROLE_LABEL_RE), ("THOUGHT_TAG", L.THOUGHT_TAG_RE)):
        if rx.search(text):
            out.append((code, rx.search(text).group(0)))
    if any(c.isalpha() and not ("A" <= c <= "Z" or "a" <= c <= "z") for c in text):
        out.append(("NON_ENGLISH", "non-latin or accented letter"))
    if not SEED_W[0] <= len(text.split()) <= SEED_W[1]:
        out.append(("LEN_ITEM", f"{len(text.split())} words"))
    return out


def pair_echoes(queries, statements, obj_terms=(), values=()):
    """the s2d per-key frame test: {(query, statement)} pairs whose E004 frames echo (stored incompatible), and the
    lines incompatible with over half of their partners (dropped)."""
    bad = {(q, s) for q in queries for s in statements
           if heldout.frame_echo(tagged(q), tagged(s), list(obj_terms), list(values))}
    drop = {q for q in queries if statements and sum((q, s) in bad for s in statements) * 2 > len(statements)}
    drop |= {s for s in statements if queries and sum((q, s) in bad for q in queries) * 2 > len(queries)}
    return bad, drop
