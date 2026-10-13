"""RC-12 split configuration and the content hook for the sealed split (prereg s13 S1-S3). Pure stdlib, no model.

SPLIT and SEED: the split this process builds. Dev is the default and builds byte-identically. Any other split comes
only from the environment (RC12_SPLIT, RC12_GEN_SEED), which sealed_kit.py sets inside a work copy, and needs a long
random hex seed so its published SHA-256 (s13 S5) cannot be inverted by guessing.

apply(g, names): the generator code is the same for dev and sealed (s13 S3); only its content differs. Every module
holding dev wording or values (sealed_kit.CONTENT_MODULES) names its content constants in CONTENT and calls
apply(globals(), CONTENT) right after them, before anything is derived from them. On dev it reads nothing, and it
refuses to run where a work copy or a content file sits (sealed items are never graded with dev pools). On any
other split it reads content_<module>.py (written by the sealed session, copied into the work copy by sealed_kit.py)
as data, never as code: only a docstring and NAME = <literal> lines (load_literals). The file must define exactly
those names, each with the dev value's shape: the same type, the same dict keys (a name ending in "*" in CONTENT
frees the keys, as for OWN_LIST, whose keys are topics), the same list and tuple lengths, the same {placeholder}
fields in every string, and the identifier fields of PINS at their dev values. So the sealed session changes wording
and values only, never a layout, a count, a balance or code. A missing content file on a split other than dev is
refused, so dev wording cannot be reused silently."""
import ast
import os
import re
import string
import sys

SPLIT = os.environ.get("RC12_SPLIT", "dev")
SEED = 1212
if SPLIT != "dev":
    if not re.fullmatch(r"[a-z]+", SPLIT):
        raise SystemExit(f"RC12_SPLIT {SPLIT!r}: lowercase letters only")
    SEED = os.environ.get("RC12_GEN_SEED", "")
    if not re.fullmatch(r"[0-9a-f]{32,}", SEED):
        raise SystemExit("RC12_SPLIT is not dev: RC12_GEN_SEED must be 32+ lowercase hex characters")


def fields(s):
    """the sorted {placeholder} field names of a format string (unparsable braces: the raw brace runs)."""
    try:
        return sorted({f for _, f, _, _ in string.Formatter().parse(s) if f is not None})
    except ValueError:
        return ["<raw>"] + re.findall(r"[{}]+", s)


def shape_diff(dev, new, where, free_keys=False):
    """first difference in shape between a dev value and its replacement, or None."""
    if type(dev) is not type(new):
        return f"{where}: type {type(new).__name__}, dev has {type(dev).__name__}"
    if isinstance(dev, str):
        return None if fields(dev) == fields(new) else f"{where}: fields {fields(new)}, dev has {fields(dev)}"
    if isinstance(dev, dict):
        if len(dev) != len(new):
            return f"{where}: {len(new)} keys, dev has {len(dev)}"
        if free_keys:
            pairs = zip(dev.values(), new.values())
            names = [f"{where}[{k!r}]" for k in new]
        else:
            if set(dev) != set(new):
                return f"{where}: keys {sorted(map(str, new))}, dev has {sorted(map(str, dev))}"
            pairs = ((dev[k], new[k]) for k in dev)
            names = [f"{where}[{k!r}]" for k in dev]
        for (a, b), name in zip(pairs, names):
            d = shape_diff(a, b, name)
            if d:
                return d
        return None
    if isinstance(dev, (list, tuple)):
        if len(dev) != len(new):
            return f"{where}: length {len(new)}, dev has {len(dev)}"
        for i, (a, b) in enumerate(zip(dev, new)):
            d = shape_diff(a, b, f"{where}[{i}]")
            if d:
                return d
        return None
    if isinstance(dev, (int, float, bool)) or dev is None:
        return None if dev == new else f"{where}: {new!r}, dev has {dev!r} (numbers are layout, not content)"
    return f"{where}: unsupported type {type(dev).__name__}"


# identifier fields inside content structures that code or cells key on: they must keep their dev values
PINS = {"pools_slots.RECALL_SLOTS": ("key", "vtype"), "fam_twohop.TWOHOP_SCHEMAS": ("vtype",),
        "pools_misc.LOOKUP_SHAPES": ("keys", "vals"), "pools_misc.K_ITEMS": (0,)}


def content_path(mod):
    """the content_<mod>.py the import path would find, or None."""
    for d in sys.path:
        p = os.path.join(d or ".", f"content_{mod}.py")
        if os.path.isfile(p):
            return p
    return None


def load_literals(path):
    """a content file's names and values. It is DATA, never run: only a docstring and top-level NAME = <literal>
    assignments are accepted (so a content file cannot import, patch code or compute anything)."""
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    out = {}
    for i, node in enumerate(tree.body):
        if i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            raise SystemExit(f"{path}:{node.lineno}: only NAME = <literal> lines are allowed in a content file")
        try:
            out[node.targets[0].id] = ast.literal_eval(node.value)
        except ValueError:
            raise SystemExit(f"{path}:{node.lineno}: {node.targets[0].id} is not a plain literal")
    return out


def pinned(mod, name, dev, new):
    """first pinned identifier field that differs from dev, or None."""
    fields = PINS.get(f"{mod}.{name}")
    if not fields:
        return None
    items = [(k, dev[k], new[k]) for k in dev] if isinstance(dev, dict) else list(zip(range(len(dev)), dev, new))
    for k, a, b in items:
        for f in fields:
            if a[f] != b[f]:
                return f"{mod}.{name}[{k}][{f!r}] is {b[f]!r}, pinned to dev's {a[f]!r} (an identifier, not content)"
    return None


def apply(g, names):
    """replace module g's content constants `names` on a split other than dev (module docstring)."""
    mod = g["__name__"]
    if SPLIT == "dev":
        if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(g["__file__"]))),
                                       ".rc12_workcopy")) or content_path(mod):
            raise SystemExit(f"{mod}: a work copy or a content file is present but RC12_SPLIT is dev; sealed content "
                             "is never graded with dev pools")
        return
    path = content_path(mod)
    if path is None:
        raise SystemExit(f"split {SPLIT}: content_{mod}.py is missing (prereg s13 S3)")
    over = load_literals(path)
    want = {n.rstrip("*") for n in names}
    if set(over) != want:
        raise SystemExit(f"content_{mod}.py defines {sorted(set(over) - want)} extra and lacks "
                         f"{sorted(want - set(over))}")
    for n in names:
        key = n.rstrip("*")
        d = shape_diff(g[key], over[key], f"{mod}.{key}", free_keys=n.endswith("*")) or \
            pinned(mod, key, g[key], over[key])
        if d:
            raise SystemExit(f"content_{mod}.py: {d}")
    for n in names:
        g[n.rstrip("*")] = over[n.rstrip("*")]
