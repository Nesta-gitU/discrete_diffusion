# ----------------------------------------------
#  adaln_transformer.py
#  (PyTorch ≥ 2.2, Python ≥ 3.10)
# ----------------------------------------------
import math
import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from src.their_utils.nn import timestep_embedding           # ← your helper

import torch.nn.functional as F

class TwoLayerFiLMHead(nn.Module):
    """
    hidden → ReLU → hidden/2 → ReLU → out
    FiLM is applied *twice*: after each linear.
    """
    def __init__(self, hidden_dim: int, out_dim: int,
                 cond_dim: int,    # t-embedding width (e.g. 4*D)
                 ratio: float = 0.5,
                 use_scale: bool = True,
                 positive: bool = False):
        super().__init__()
        mid_dim = int(out_dim / ratio)          # e.g. 256 → 512 → 128
        self.lin1 = nn.Linear(hidden_dim, mid_dim)
        self.lin2 = nn.Linear(mid_dim, out_dim)

        # FiLM parameters for both layers, zero-init (“-Zero” style)
        n_aff = 2 if use_scale else 1
        self.film1 = nn.Linear(cond_dim, n_aff * mid_dim)
        self.film2 = nn.Linear(cond_dim, n_aff * out_dim)
        for m in (self.film1, self.film2):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

        self.use_scale, self.positive = use_scale, positive

    def _apply_film(self, h, coeffs):
        if self.use_scale:
            beta, gamma_hat = coeffs.chunk(2, -1)
            return (1 + gamma_hat).unsqueeze(1) * h + beta.unsqueeze(1)
        return h + coeffs.unsqueeze(1)

    def forward(self, h, t_embed):
        # layer 1
        h = self.lin1(h)
        h = self._apply_film(h, self.film1(t_embed))
        h = F.relu(h, inplace=True)

        # layer 2
        h = self.lin2(h)
        h = self._apply_film(h, self.film2(t_embed))

        if self.positive:
            h = torch.log1p(torch.exp(h))  # softplus, but log σ doesn’t need positivity
        return h

# ---------- 1. Time-conditioning modules -----------------
class TimestepEmbed(nn.Module):
    """
    Maps a sinusoidal / Fourier timestep embedding ➜ (shift, scale).
    The final linear layer is *zero-initialised* so the network starts
    identical to a vanilla Transformer (AdaLN-Zero trick).
    """
    def __init__(self, hidden_dim: int, embed_dim: int | None = None,
                 use_scale: bool = True):
        super().__init__()
        embed_dim = embed_dim or hidden_dim * 4       # DiT default
        self.use_scale = use_scale

        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, hidden_dim * (2 if use_scale else 1))
        )
        # --- AdaLN-Zero: start with zero influence -------------
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t_embed: torch.Tensor):
        """
        t_embed : (B, embed_dim)  – pre-computed Fourier features of t.
        Returns  : (B, D) shift , (B, D) scale
        """
        out = self.net(t_embed)
        if self.use_scale:
            shift, scale = out.chunk(2, dim=-1)
            scale = 1.0 + scale        # γ = 1 + Δγ    (stable at init)
            return shift, scale
        else:                           # shift-only variant
            return out, torch.ones_like(out)


class AdaLayerNorm(nn.Module):
    """LayerNorm(x) → γ·x̂ + β where (β, γ) come from timestep."""
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor,
                shift: torch.Tensor, scale: torch.Tensor):
        # x          : (B, L, D)
        # shift/scale: (B, D)
        mean = x.mean(-1, keepdim=True)
        var  = (x - mean).pow(2).mean(-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * scale.unsqueeze(1) + shift.unsqueeze(1)


# ---------- 2. One encoder block with AdaLN ----------------
class EncoderBlockAdaLN(nn.Module):
    """
    Minimal re-implementation of nn.TransformerEncoderLayer
    with AdaLayerNorm injected before attention & MLP.
    """
    def __init__(self, hidden_dim: int, num_heads: int,
                 mlp_dim: int, dropout: float,
                 use_scale: bool = True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

        # AdaLN instances
        self.ada_ln1 = AdaLayerNorm(hidden_dim)
        self.ada_ln2 = AdaLayerNorm(hidden_dim)

        # Small projection from (timestep → shift/scale)
        self.t_proj = TimestepEmbed(hidden_dim, use_scale=use_scale)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor,
                attn_mask=None, key_padding_mask=None):
        """x: (B, L, D)  t_embed: (B, embed_dim)"""
        shift1, scale1 = self.t_proj(t_embed)
        x = x + self.dropout(self.attn(
                query=self.ada_ln1(x, shift1, scale1),
                key=x, value=x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False)[0])

        shift2, scale2 = self.t_proj(t_embed)
        y = self.ffn(self.ada_ln2(x, shift2, scale2))
        return x + self.dropout(y)


# ---------- 3. The full encoder stack ---------------------
class TransformerEncoderAdaLN8M(nn.Module):
    """
    A near-drop-in replacement for your original TransformerEncoder8M
    with DiT-style AdaLN-Zero conditioning.
    """
    def __init__(self, vocab_size: int,
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 2,
                 num_heads: int = 8,
                 mlp_dim: int = 1024,
                 num_layers: int = 5,
                 dropout: float = 0.01):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ---------- Encoder stack --------------------------
        self.layers = nn.ModuleList([
            EncoderBlockAdaLN(hidden_dim, num_heads,
                              mlp_dim, dropout,
                              use_scale=True)           # AdaLN-Zero (γ,β)
            for _ in range(num_layers)
        ])

        # Time-embedding → shared across layers
        self.time_embed_dim = hidden_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(input_dim, self.time_embed_dim),   # same dims as before
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)  # no zero-init here
        )

        self.position_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output head
        #self.output_proj = nn.Sequential(
        #    nn.Linear(hidden_dim, hidden_dim),
        #    nn.GELU(),
        #    nn.Linear(hidden_dim, output_dim)
        #)

        self.mean_head = TwoLayerFiLMHead(hidden_dim, int(hidden_dim/2),
                                  cond_dim=self.time_embed_dim,
                                  positive=False)

        self.logvar_head = TwoLayerFiLMHead(hidden_dim, int(hidden_dim/2),
                                            cond_dim=self.time_embed_dim,
                                            positive=False)  # log σ already un-clamped

        # Logging helper
        self.register_buffer("emb_norm", torch.tensor(0.0), persistent=False)

        # Convenience
        self.hidden_dim = hidden_dim
        self.input_dim  = input_dim

    # -------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor,
                mask: torch.Tensor | None = None):
        """
        x   : (B, L, 128)  – token features
        t   : (B,)         – diffusion timestep (int32 or float32)
        mask: (B, L) bool  – optional padding mask (1=pad)
        """
        # 1) token -> hidden
        h = self.input_proj(x)                       # (B, L, D)

        # 2) time embedding
        t_fourier = timestep_embedding(t, self.input_dim, True)  # (B, 128)
        t_embed   = self.time_embed(t_fourier)                   # (B, 4D)
        if not torch.all(t==0): #only log if its not the reconloss
            self.emb_norm = t_embed.norm(dim=-1).mean()  # for logging

        # 3) add position + dropout
        pos_ids = torch.arange(x.size(1), device=x.device)
        h = h + self.position_embeddings(pos_ids)               # (B, L, D)
        h = self.dropout(self.layer_norm(h))

        # 4) encoder stack
        with sdpa_kernel([SDPBackend.MATH]):     # keeps JVP checkpoints happy
            for layer in self.layers:
                h = layer(h, t_embed,           # ← passes same t to every block
                           key_padding_mask=mask)

        # 5) projector head
        #h = self.output_proj(h) #old verion also worked reasonalby well
        mu = self.mean_head(h, t_embed)
        logvar = self.logvar_head(h, t_embed)

        h = torch.cat([mu, logvar], dim=-1)
        return h
        
