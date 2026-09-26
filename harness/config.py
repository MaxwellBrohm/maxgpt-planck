"""MaxGPT-Planck model configuration.

One dataclass describes every architecture arm the harness can build. Field names
follow MaxGPT-Ultra's model/config.py (Max's AI Model/maxgpt-ultra) so configs read the
same; the Planck-only fields are the sharing and looping switches at the bottom.

Layer vocabulary used everywhere in the harness:
  unique layer     = a Block with its own parameters (counted once)
  effective layer  = one application of a Block in the forward pass
  prelude + core + coda: the unique layers. The core is applied n_loops times.
  effective depth  = n_prelude + n_layers * n_loops + n_coda
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass


@dataclass
class PlanckConfig:
    # --- core dimensions ---
    vocab_size: int = 8192
    d_model: int = 192
    n_layers: int = 8            # unique CORE layers (prelude and coda are extra)
    n_heads: int = 3             # query heads
    n_kv_heads: int = 3          # key/value heads (GQA when fewer than n_heads)
    head_dim: int = 64           # per-head width; n_heads * head_dim need not equal d_model
    mlp_hidden: int = 512        # SwiGLU inner width; 0 = attention-only blocks
    seq_len: int = 2048

    # --- block knobs (Max's validated block, all on in the Planck baseline) ---
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    qk_norm: bool = True         # RMSNorm on per-head Q and K
    attn_gate: bool = True       # per-head sigmoid gate on the attention output
    value_residual: bool = True  # normalized value residual from effective layer 0
    norm_scaling: bool = True    # 1/sqrt(effective depth index) on pre-norm outputs
    rms_eps: float = 1e-5
    init_std: float = 0.02

    # --- sharing and looping (P-110, P-111, P-150) ---
    n_loops: int = 1             # the core is applied this many times
    loop_order: str = "cyclic"   # "cyclic": ABCABC, "immediate": AABBCC
    n_prelude: int = 0           # unique layers before the looped core
    n_coda: int = 0              # unique layers after the looped core
    qk_share: int = 1            # consecutive unique layers sharing one W_q and W_k
    kv_tie: bool = False         # K = V: no separate value projection

    # --- S004 Canon layers (P-148; experiments/S004_canon/notes.txt). Default off. ---
    canon: str = ""              # "" off; sites: "A" before attention, "C" before the MLP ("AC")
    canon_kernel: int = 4        # causal depthwise taps per Canon conv (used only when canon is on)

    # --- S005 forget gate on attention logits (P-020; experiments/S005_forget_gate/notes.txt). Default off.
    forget_gate: bool = False    # per head f = sigmoid(w.a + b); logit (i, j) += sum over (j, i] of log f

    # --- S007 smeared keys (P-024; experiments/S007_smeared_key/notes.txt). Default off. ---
    smear_key: bool = False      # raw key k_t + alpha_h * k_(t-1) in the same document; alpha per KV head, 0 at init

    def __post_init__(self) -> None:
        assert self.vocab_size > 0 and self.d_model > 0
        assert self.n_layers >= 1, "need at least one core layer"
        assert self.n_heads >= 1 and self.n_kv_heads >= 1
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be a multiple of n_kv_heads ({self.n_kv_heads})"
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        assert self.mlp_hidden >= 0
        assert self.n_loops >= 1 and self.n_prelude >= 0 and self.n_coda >= 0
        assert self.loop_order in ("cyclic", "immediate"), self.loop_order
        assert self.qk_share >= 1
        assert self.canon in ("", "A", "C", "AC"), f"canon must be '', 'A', 'C' or 'AC', got {self.canon!r}"
        assert self.canon_kernel >= 2, "canon_kernel must be >= 2 (tap 0 is the token itself)"
        assert isinstance(self.forget_gate, bool), f"forget_gate must be true or false, got {self.forget_gate!r}"
        assert isinstance(self.smear_key, bool), f"smear_key must be true or false, got {self.smear_key!r}"

    # ------------------------------------------------------------------ #
    # Derived layer bookkeeping (used by both the model and the counter)
    # ------------------------------------------------------------------ #
    @property
    def n_unique(self) -> int:
        return self.n_prelude + self.n_layers + self.n_coda

    @property
    def depth(self) -> int:
        """Effective depth: how many Block applications one forward pass makes."""
        return self.n_prelude + self.n_layers * self.n_loops + self.n_coda

    def schedule(self) -> list[int]:
        """Unique block index for each effective layer, in forward order."""
        pre = list(range(self.n_prelude))
        core_ids = [self.n_prelude + i for i in range(self.n_layers)]
        if self.loop_order == "cyclic":
            core = core_ids * self.n_loops
        else:
            core = [c for c in core_ids for _ in range(self.n_loops)]
        coda = [self.n_prelude + self.n_layers + i for i in range(self.n_coda)]
        return pre + core + coda

    def vr_blocks(self) -> set[int]:
        """Unique blocks that ever run at an effective position > 0. Only these get
        value-residual mixing parameters (Ultra gives them to every layer but the first)."""
        if not self.value_residual:
            return set()
        return {u for pos, u in enumerate(self.schedule()) if pos > 0}

    def qk_owner(self, u: int) -> int:
        """Unique block whose W_q/W_k block u uses (the first of its share group)."""
        return (u // self.qk_share) * self.qk_share

    def canon_sites(self) -> str:
        """Canon sites every block has: "A" (before attention), "C" (before the MLP, so only
        when mlp_hidden > 0)."""
        return "".join(s for s in self.canon if s == "A" or self.mlp_hidden > 0)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if not self.canon:   # S004 off: exactly the pre-flag dict (checkpoint model_cfg, resume
            del d["canon"], d["canon_kernel"]   # and init_from checks against older checkpoints)
        if not self.forget_gate:   # S005 off: the same rule
            del d["forget_gate"]
        if not self.smear_key:     # S007 off: the same rule
            del d["smear_key"]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PlanckConfig":
        valid = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - valid
        assert not unknown, f"unknown config keys: {sorted(unknown)}"
        return cls(**d)

    def replace(self, **kw) -> "PlanckConfig":
        return dataclasses.replace(self, **kw)
