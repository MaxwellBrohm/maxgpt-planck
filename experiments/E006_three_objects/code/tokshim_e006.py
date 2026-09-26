"""E006 tokenizer loader for the no-model checks. Loads a TOKENIZER only, never a model.
  load("hf")    transformers.AutoTokenizer for SmolLM2-135M-Instruct (the PC queue: what training uses)
  load("shim")  the same tokenizer.json through the `tokenizers` library, and the chat template from
                tokenizer_config.json rendered with jinja2 as transformers does (trim_blocks, lstrip_blocks), with
                no torch and no transformers import. For Mac-side tests only; the PC's validate_e006 re-runs every
                count with "hf", and the stream stats must equal E005's run.json either way.
  load()        E006_TOK=hf|shim, default hf.
The shim implements only what E006's encoders call: tok(text, add_special_tokens=...).input_ids,
apply_chat_template(msgs, tokenize=False, add_generation_prompt=...), convert_tokens_to_ids, decode, eos_token(_id),
pad_token_id, unk_token_id, chat_template."""
import glob
import json
import os
from types import SimpleNamespace

SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


def snapshot_dir(model_id=SMOL, revision=REVISION):
    roots = [os.environ.get("HF_HOME") and os.path.join(os.environ["HF_HOME"], "hub"),
             os.path.expanduser("~/.cache/huggingface/hub")]
    for r in [x for x in roots if x]:
        d = os.path.join(r, "models--" + model_id.replace("/", "--"), "snapshots", revision)
        if os.path.isfile(os.path.join(d, "tokenizer.json")):
            return d
    raise FileNotFoundError(f"no {model_id} snapshot {revision[:8]} with tokenizer.json under {roots}")


class ShimTok:
    def __init__(self, d):
        from tokenizers import Tokenizer
        self._t = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        cfg = json.load(open(os.path.join(d, "tokenizer_config.json")))
        self.chat_template = cfg["chat_template"]
        self.eos_token, self.bos_token = cfg["eos_token"], cfg.get("bos_token")
        self.pad_token, self.unk_token = cfg.get("pad_token"), cfg.get("unk_token")
        self.eos_token_id = self._t.token_to_id(self.eos_token)
        self.pad_token_id = self._t.token_to_id(self.pad_token) if self.pad_token else None
        self.unk_token_id = self._t.token_to_id(self.unk_token) if self.unk_token else None
        self._tpl = None

    def __call__(self, text, add_special_tokens=True):
        return SimpleNamespace(input_ids=self._t.encode(text, add_special_tokens=add_special_tokens).ids)

    def convert_tokens_to_ids(self, t):
        return self._t.token_to_id(t)

    def decode(self, ids, skip_special_tokens=False):
        return self._t.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        if tokenize:
            raise NotImplementedError("the shim renders text only")
        if self._tpl is None:
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            self._tpl = env.from_string(self.chat_template)
        return self._tpl.render(messages=msgs, add_generation_prompt=add_generation_prompt,
                                bos_token=self.bos_token, eos_token=self.eos_token)


def load(kind=None):
    kind = kind or os.environ.get("E006_TOK", "hf")
    if kind == "shim":
        return ShimTok(snapshot_dir())
    if kind == "hf":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(SMOL)
    raise ValueError(f"unknown tokenizer kind {kind!r}")


def snapshot_files():
    """sorted file names of the pinned snapshot (identity 5 lists them)."""
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(snapshot_dir(), "*")))
