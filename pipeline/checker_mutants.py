"""Checker mutants for mutation_checker.py. A mutant is (name, apply) where apply() patches the live checker and
returns an undo function. Four families:
  off:<check>      one registered check function returns nothing
  drop:<CODE>      every hit with one code is filtered out (each code needs its own planted fixture)
  const:<name>     a threshold constant moved one step either way
  src:<name>       a one-line source edit (off-by-one, flipped comparison, weakened regex), compiled into a fresh
                   copy of the module and swapped into every module that imported from it."""
import ast
import inspect
import re
import sys
import types

import check_base
import check_behav
import check_events
import check_text
import checker
import parse

MODULES = [checker, check_text, check_events, check_behav, check_base, parse]

# lower values are the nearest shares a turn of legal length can reach below the threshold (fixtures_bound pins them)
CONSTS = [("REPEAT_MIN", 2, 4), ("CONSEC_MAX", 0.24, 0.26), ("SELF_COPY_MAX", 0.48, 0.51), ("ECHO_MAX", 0.59, 0.61),
          ("PERSONA_N", 7, 9), ("PROMPT_N", 5, 7), ("SOCIAL_MAX_W", 19, 21)]

SRC = [  # (name, module, old, new)
    ("len_max_off_by_one", check_text, 'if not (t["min_w"] <= n <= hi):', 'if not (t["min_w"] <= n <= hi + 1):'),
    ("len_min_off_by_one", check_text, 'if not (t["min_w"] <= n <= hi):', 'if not (t["min_w"] - 1 <= n <= hi):'),
    ("plural_forms_dropped", check_text, 'stems = {w, w + "s", w + "es",', 'stems = {w, w + "es",'),
    ("shotgun_needs_3", check_events, "if present + len(named) >= 2:", "if present + len(named) >= 3:"),
    ("stale_ignored", check_events, 'stale = [v for v in g.get("stale", []) if ctx.has(i, v)]', "stale = []"),
    ("early_off_by_one", check_events, 'if t["i"] < first and ctx.has(t["i"], v):',
     'if t["i"] < first - 1 and ctx.has(t["i"], v):'),
    ("early_inclusive", check_events, 'if t["i"] < first and ctx.has(t["i"], v):',
     'if t["i"] <= first and ctx.has(t["i"], v):'),
    ("dist_window_short", check_events, "for i in range(last + 1, q):", "for i in range(last + 3, q):"),
    ("dist_counts_own_turns", check_events, 'if i in ctx.text and e["id"] not in ctx.by_i[i]["events"] and ctx.has(i, a):',
     'if i in ctx.text and ctx.has(i, a):'),
    ("return_digression_ok", check_events, "if not o or d:", "if not o:"),
    ("contains_flipped", check_events, 'if g["answer"] == "yes" \\', 'if g["answer"] != "yes" \\'),
    ("count_others_ok", check_events, "if not ctx.has(i, g[\"answer\"]) or others:", "if not ctx.has(i, g[\"answer\"]):"),
    ("restates_any", check_events, 'if a and leakable and a != ctx.card and ctx.has(q, a):',
     'if a and a != ctx.card and ctx.has(q, a):'),
    ("offer_not_needed", check_behav, 'elif not (L.OFFER_RE.search(s) or s.rstrip().endswith("?")):', "elif False:"),
    ("start_name_anywhere", check_behav, 'return re.match(r"\\s*" + re.escape(a["name"]) + r"(?![A-Za-z])", s) is not None',
     'return re.search(re.escape(a["name"]), s) is not None'),
    ("one_sentence_two_ok", check_behav, "return len(sentences(s)) == 1", "return len(sentences(s)) <= 2"),
    ("identity_any_name", check_behav, "if not ctx.has_sys or m.group(1) != ctx.card:", "if False:"),
    ("identity_card_wrong", check_behav, "if not ctx.has_sys or m.group(1) != ctx.card:", "if True:"),
    ("swap_no_deny_ok", check_behav, "if not L.DENY_RE.search(ctx.text[i]):", "if False:"),
    ("case_insensitive_values", check_base, "rx = golds.value_re(v)\n        if self.lower_user",
     "rx = re.compile(golds.value_re(v).pattern, re.I)\n        if self.lower_user"),
    ("lowercase_user_strict", check_base, 'if self.lower_user and self.by_i[i]["role"] == "user":', "if False:"),
    ("parse_no_end_ok", parse, "if not ended:", "if False:"),
    ("parse_order_ok", parse, "if seen != order:", "if False:"),
    ("parse_no_unquote", parse, 'drift.append("quoted")\n        return t[1:-1].strip()', 'return t'),
    ("parse_wrap_ok", parse, 'codes.append(("FORMAT_WRAP" if started else "FORMAT_EXTRA", line.strip()[:40]))',
     'codes.append(("FORMAT_EXTRA", line.strip()[:40])) if not started else None'),
    ("parse_thought_ok", parse, 'codes.append(("THOUGHT_TAG", "non-empty thought block"))', "pass"),
    ("parse_no_merge", parse, "if not m or nxt is None or m.group(1) + m.group(2) != nxt:", "if True:"),
]


def _read(path):
    with open(path) as f:
        return f.read()


def _swap_everywhere(old_mod, new_mod):
    """point every name that other checker modules bound to old_mod (or its members) at new_mod's version."""
    saved = []
    for m in MODULES + [sys.modules[__name__]]:
        for k, v in list(vars(m).items()):
            rep = None
            if v is old_mod:
                rep = new_mod
            elif getattr(v, "__module__", None) == old_mod.__name__ and hasattr(new_mod, getattr(v, "__name__", "")):
                rep = getattr(new_mod, v.__name__)
            if rep is not None and rep is not v:
                saved.append((m, k, v))
                setattr(m, k, rep)
    saved.append((checker, "REGISTRY", checker.REGISTRY))
    checker.REGISTRY = [getattr(new_mod, f.__name__) if f.__module__ == old_mod.__name__ else f
                        for f in checker.REGISTRY]

    def undo():
        for m, k, v in reversed(saved):
            setattr(m, k, v)
    return undo


def src_mutant(mod, old, new):
    def apply():
        src = _read(mod.__file__)
        if src.count(old) != 1:
            raise AssertionError(f"mutant anchor not unique in {mod.__name__}: {old!r}")
        fresh = types.ModuleType(mod.__name__)
        fresh.__file__ = mod.__file__
        exec(compile(src.replace(old, new), mod.__file__, "exec"), fresh.__dict__)
        return _swap_everywhere(mod, fresh)
    return apply


def off_mutant(fn):
    def apply():
        old = checker.REGISTRY
        checker.REGISTRY = [(lambda ctx: []) if f is fn else f for f in old]

        def undo():
            checker.REGISTRY = old
        return undo
    return apply


def drop_mutant(code):
    def apply():
        old_reg, old_ct = checker.REGISTRY, checker.check_turns
        checker.REGISTRY = [(lambda f: (lambda ctx: [h for h in f(ctx) if h[0] != code]))(f) for f in old_reg]

        def check_turns(skel, turn_texts, built=None, wordlist=None, parse_codes=(), drift=()):
            return old_ct(skel, turn_texts, built, wordlist, [c for c in parse_codes if c[0] != code], drift)
        checker.check_turns = check_turns

        def undo():
            checker.REGISTRY, checker.check_turns = old_reg, old_ct
        return undo
    return apply


def const_mutant(name, value):
    def apply():
        old = getattr(check_text, name)
        setattr(check_text, name, value)
        return lambda: setattr(check_text, name, old)
    return apply


CONST_CODES = {"REPEAT_MIN": "REPEAT_4GRAM", "CONSEC_MAX": "CONSEC_REP", "SELF_COPY_MAX": "SELF_COPY",
               "ECHO_MAX": "ECHO_USER", "PERSONA_N": "PERSONA_LEAK", "PROMPT_N": "PROMPT_ECHO",
               "SOCIAL_MAX_W": "LEN_ASSIST"}


def codes_in(src):
    """reason codes named as string literals in a piece of source (to aim a mutant at its fixtures)."""
    return {c for c in re.findall(r'"([A-Z][A-Z_0-9]{3,})"', src) if c in checker.RANK}


def enclosing_codes(mod, anchor):
    src = _read(mod.__file__)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            seg = ast.get_source_segment(src, node) or ""
            if anchor in seg:
                return codes_in(seg)
    return set()


def all_mutants():
    """[(name, apply, target codes)]; targets only order the cases, every case still counts."""
    out = [(f"off:{f.__name__}", off_mutant(f), codes_in(inspect.getsource(f))) for f in checker.REGISTRY]
    out += [(f"drop:{c}", drop_mutant(c), {c}) for c in checker.ORDER]
    out += [(f"const:{n}={v}", const_mutant(n, v), {CONST_CODES[n]}) for n, lo, hi in CONSTS for v in (lo, hi)]
    out += [(f"src:{n}", src_mutant(m, o, w), enclosing_codes(m, o)) for n, m, o, w in SRC]
    return out
