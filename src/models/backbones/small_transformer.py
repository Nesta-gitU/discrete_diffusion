import torch
import torch.nn as nn

class TransformerEncoder8M(nn.Module):           #hidden_dim=256
    def __init__(self, vocab_size, input_dim=128, hidden_dim=256, output_dim=256,
                 num_heads=8, mlp_dim=1024, num_layers=5, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        #bigger output_proj
        #self.output_proj = nn.Sequential(
        #    nn.Linear(hidden_dim, hidden_dim),
        #    nn.GELU(),
        #    nn.Linear(hidden_dim, output_dim)
        #)

        self.output_dim = int(output_dim/2)

        # Positional embeddings (+1 for CLS token)
        self.position_embeddings = nn.Embedding(vocab_size + 1, hidden_dim)
        self.LayerNorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, input_dim]
        mask: [batch_size, seq_len] (optional)
        Returns: [batch_size, output_dim] (CLS token output)
        """
        batch_size, seq_len, _ = x.size()
        emb_x = self.input_proj(x)  # [batch_size, seq_len, hidden_dim]

        # Expand and prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_dim]
        x = torch.cat((cls_tokens, emb_x), dim=1)  # [batch_size, seq_len + 1, hidden_dim]

        # Position embeddings
        position_ids = torch.arange(seq_len + 1, dtype=torch.long, device=x.device)
        position_embeddings = self.position_embeddings(position_ids).unsqueeze(0)  # [1, seq_len + 1, hidden_dim]
        x = x + position_embeddings
        x = self.dropout(self.LayerNorm(x))

        # Adjust mask for CLS token if provided
        if mask is not None:
            cls_mask = torch.zeros((batch_size, 1), dtype=mask.dtype, device=mask.device)
            mask = torch.cat((cls_mask, mask), dim=1)  # [batch_size, seq_len + 1]
            out = self.encoder(x, src_key_padding_mask=mask)
        else:
            out = self.encoder(x)

        # Extract CLS token output
        cls_output = out[:, 0, :]  # [batch_size, hidden_dim]
        cls_output = self.output_proj(cls_output)  # [batch_size, output_dim]

        return cls_output
