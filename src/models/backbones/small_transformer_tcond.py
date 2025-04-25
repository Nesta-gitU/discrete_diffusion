import torch
import torch.nn as nn

from src.their_utils.nn import timestep_embedding

class TransformerEncoder8M(nn.Module):
    def __init__(self, vocab_size, input_dim=128, hidden_dim=256, output_dim=2, num_heads=8, mlp_dim=1024, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) #the big model has two of these layers and a non-linearity in between

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim) #the big model has two of these layers and a non-linearity in between

        #set up t_conditioning
        time_embed_dim = hidden_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(input_dim, time_embed_dim), #bit confused if the first arg should be input dim here
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_dim),
        )

        self.position_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim



    def forward(self, x, t, mask=None):
        """
        x: [bs, seqlen, 128]
        t: [bs]
        Returns: [bs, seqlen, 256]
        """
        emb_x = self.input_proj(x)  # [bs, seqlen, 256]

        # concatenate time and positional embeddings
        emb = self.time_embed(timestep_embedding(t, self.input_dim, True))
        seq_length = x.size(1)

        position_ids = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        if mask is not None:
            x = self.encoder(emb_inputs, src_key_padding_mask=mask)
            raise NotImplementedError("mask not implemented")
        else:
            out = self.encoder(emb_inputs)
        
        out = self.output_proj(out)

        return out