"""T001 toy models and the arm registry (PREREG.txt MODELS AND ARMS)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Attention, CausalConv, GatedDelta
from task import VOCAB

D_MODEL, N_HEADS, D_MLP = 128, 2, 512
FG_LEN = 865                       # longest training row (n = 96): the forget gate's and gdn's alpha half-life

# arm -> (mixer per layer, canon on?) ; a mixer is an Attention kind or "gdn" / "gdn_nc"
ARMS = {
    "base": (["base", "base"], False),
    "fg": (["fg", "fg"], False),
    "sb": (["sb", "sb"], False),
    "sa": (["sa", "sa"], False),
    "alibi": (["alibi", "alibi"], False),
    "gdn": (["base", "gdn"], False),
    "canon": (["base", "base"], True),
    "smear": (["smear", "smear"], False),
    "gdn_nc": (["base", "gdn_nc"], False),
    "base1": (["base"], False),
    "smear1": (["smear"], False),
    "canon1": (["base"], True),
}
PICK_ARMS = ("base", "gdn", "canon", "fg", "sb", "sa", "alibi")
DIAG_ARMS = ("smear", "gdn_nc", "base1", "smear1", "canon1")


class Block(nn.Module):
    def __init__(self, mixer: str, canon: bool, n_layers: int):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(D_MODEL), nn.RMSNorm(D_MODEL)
        if mixer in ("gdn", "gdn_nc"):
            self.mix = GatedDelta(D_MODEL, N_HEADS, conv=(mixer == "gdn"))
        else:
            self.mix = Attention(D_MODEL, N_HEADS, mixer, fg_len=FG_LEN)
        self.canon_a = CausalConv(D_MODEL, 4) if canon else None
        self.canon_c = CausalConv(D_MODEL, 4) if canon else None
        self.up = nn.Linear(D_MODEL, D_MLP, bias=False)
        self.down = nn.Linear(D_MLP, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        if self.canon_a is not None:
            h = h + self.canon_a(h)
        x = x + self.mix(h)
        h = self.n2(x)
        if self.canon_c is not None:
            h = h + self.canon_c(h)
        return x + self.down(F.gelu(self.up(h)))


class ToyLM(nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        mixers, canon = ARMS[arm]
        self.arm = arm
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList(Block(m, canon, len(mixers)) for m in mixers)
        self.nf = nn.RMSNorm(D_MODEL)
        self.init_weights(len(mixers))

    @torch.no_grad()
    def init_weights(self, n_layers: int) -> None:
        """Draws in module order; constants (canon, fg, alibi, smear, norms) draw nothing, so those arms share every
        base draw at a seed. gdn's gate bias: alpha at the forget gate's half-life (FG_LEN), beta at 0.5."""
        out_std = 0.02 / math.sqrt(2 * n_layers)
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Embedding):
                mod.weight.normal_(0.0, 0.02)
            elif isinstance(mod, nn.Linear):
                last = name.endswith(".o") or name.endswith(".down")
                mod.weight.normal_(0.0, out_std if last else 0.02)
                if mod.bias is not None:
                    mod.bias.zero_()
            elif isinstance(mod, CausalConv) and not mod.zero:
                mod.weight.uniform_(-1.0 / math.sqrt(mod.k), 1.0 / math.sqrt(mod.k))
        f =2.0 ** (-1.0 / FG_LEN)
        for mod in self.modules():
            if isinstance(mod, GatedDelta):
                mod.gates.bias[: mod.h] = math.log(f / (1.0 - f))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.nf(x) @ self.emb.weight.t()


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
