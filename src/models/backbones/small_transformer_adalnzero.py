# -----------------------------------------------------------
# adaln_zero_transformer.py
# -----------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from src.their_utils.nn import timestep_embedding      # your helper
# -----------------------------------------------------------
# 0.  AdaLayerNorm (unchanged)
# -----------------------------------------------------------
class AdaLayerNorm(nn.Module):
    """LayerNorm(x) → γ·x̂ + β, where β, γ come from timestep."""
    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.hidden_dim = hidden_dim

    def forward(self, x, shift, scale):
        mean = x.mean(-1, keepdim=True)
        var  = (x - mean).pow(2).mean(-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * scale.unsqueeze(1) + shift.unsqueeze(1)

# -----------------------------------------------------------
# 1.  Conditioning MLP per block: (β1,γ1,β2,γ2,α1,α2)
# -----------------------------------------------------------
class BlockTimestepParams(nn.Module):
    def __init__(self, hidden_dim: int, embed_dim: int, use_scale: bool = True):
        super().__init__()
        self.use_scale = use_scale
        n_vec = 6 if use_scale else 4               # β1 γ1 β2 γ2 α1 α2
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, n_vec * hidden_dim)
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, t_embed):
        vec = self.proj(t_embed)                    # (B, n_vec·D)
        if self.use_scale:
            
            beta1, gamma1, beta2, gamma2, alpha1, alpha2 = vec.chunk(6, dim=-1)
            gamma1, gamma2 = 1.5 * torch.tanh(gamma1 / 1.5), 1.5 * torch.tanh(gamma2 / 1.5)

            gamma1, gamma2 = 1.0 + gamma1, 1.0 + gamma2            # γ = 1 + γ̂
        else:

            beta1, beta2, alpha1, alpha2 = vec.chunk(4, dim=-1)
            gamma1 = gamma2 = torch.ones_like(beta1)
        return (beta1, gamma1), (beta2, gamma2), (alpha1, alpha2)

# -----------------------------------------------------------
# 2.  One encoder block with AdaLN-Zero (α-gates)
# -----------------------------------------------------------
class EncoderBlockAdaLN(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int,
                 mlp_dim: int, dropout: float,
                 embed_dim: int, use_scale: bool = True):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

        self.ada_ln1 = AdaLayerNorm(hidden_dim)
        self.ada_ln2 = AdaLayerNorm(hidden_dim)
        self.t_cond  = BlockTimestepParams(hidden_dim, embed_dim, use_scale)

    def forward(self, x, t_embed, attn_mask=None, key_padding_mask=None):
        (β1, γ1), (β2, γ2), (α1, α2) = self.t_cond(t_embed)

        # --- attention branch ---------------------------------
        h = self.ada_ln1(x, β1, γ1)
        attn_out = self.attn(h, x, x,
                             attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask,
                             need_weights=False)[0]
        x = x + α1.unsqueeze(1) * attn_out

        # --- MLP branch ---------------------------------------
        h = self.ada_ln2(x, β2, γ2)
        mlp_out = self.ffn(h)
        x = x + α2.unsqueeze(1) * mlp_out
        return x

# -----------------------------------------------------------
# 3.  FiLM head (optional two-layer version)
# -----------------------------------------------------------
class TwoLayerFiLMHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int,
                 cond_dim: int, mid_factor: float = 2,
                 use_scale: bool = True, positive: bool = False):
        super().__init__()
        mid_dim = int(hidden_dim * mid_factor)
        self.lin1 = nn.Linear(hidden_dim, mid_dim)
        self.lin2 = nn.Linear(mid_dim, out_dim)

        n_aff = 2 if use_scale else 1
        self.film1 = nn.Linear(cond_dim, n_aff * mid_dim)
        self.film2 = nn.Linear(cond_dim, n_aff * out_dim)
        for m in (self.film1, self.film2):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

        self.use_scale = use_scale
        self.positive  = positive

    def _apply_film(self, h, coeffs, c=1.5):
        if self.use_scale:
            beta, gamma_hat= coeffs.chunk(2, dim=-1)
            gamma_hat = c * torch.tanh(gamma_hat / c)
            return (1+gamma_hat).unsqueeze(1) * h + beta.unsqueeze(1)
        return h + coeffs.unsqueeze(1)

    def forward(self, h, t_embed):
        h = self.lin1(h)
        h = self._apply_film(h, self.film1(t_embed))
        h = F.relu(h)

        h = self.lin2(h)
        h = self._apply_film(h, self.film2(t_embed))

        if self.positive:
            h = F.softplus(h)
        return h

# -----------------------------------------------------------
# 4.  Full Transformer encoder with AdaLN-Zero backbone
# -----------------------------------------------------------
class TransformerEncoderAdaLN8M(nn.Module):
    def __init__(self, max_seq_len: int,
                 input_dim: int = 128, hidden_dim: int = 256,
                 output_dim: int = 256,           # 128 μ + 128 log σ
                 num_heads: int = 8, mlp_dim: int = 1024,
                 num_layers: int = 5, dropout: float = 0.01):
        super().__init__()
        # token projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # shared time embedding (sinusoid → MLP)
        self.time_embed_dim = hidden_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(input_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )

        # encoder stack
        self.layers = nn.ModuleList([
            EncoderBlockAdaLN(hidden_dim, num_heads, mlp_dim, dropout,
                              embed_dim=self.time_embed_dim, use_scale=True)
            for _ in range(num_layers)
        ])

        # FiLM-conditioned mean / log-var heads
        self.mu_head  = TwoLayerFiLMHead(hidden_dim, 128,
                                         self.time_embed_dim,
                                         positive=False)
        self.lv_head  = TwoLayerFiLMHead(hidden_dim, 128,
                                         self.time_embed_dim,
                                         positive=False)

        # position embeddings
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # misc
        self.register_buffer("emb_norm", torch.tensor(0.0), persistent=False)
        self.hidden_dim = hidden_dim
        self.input_dim  = input_dim

    # -------------------------------------------------------
    def forward(self, x, t, mask=None):
        """
        x : (B, L, 128)    token features
        t : (B,)           diffusion timestep
        mask : (B, L) bool optional padding mask
        """
        h = self.input_proj(x)                      # (B,L,D)

        # time embedding
        t_fourier = timestep_embedding(t, self.input_dim, True)  # (B,128)
        t_embed   = self.time_embed(t_fourier)                   # (B,4D)
        self.emb_norm = t_embed.norm(dim=-1).mean()              # logging

        # add position
        pos_ids = torch.arange(x.size(1), device=x.device)
        h = h + self.position_embeddings(pos_ids)
        h = self.dropout(h)

        # encoder blocks
        with sdpa_kernel([SDPBackend.MATH]):
            for block in self.layers:
                h = block(h, t_embed, key_padding_mask=mask)

        # FiLM heads → μ, log σ
        mu      = self.mu_head(h, t_embed)
        log_var = self.lv_head(h, t_embed)

        return torch.cat([mu, log_var], dim=-1)     # (B,L,256)
