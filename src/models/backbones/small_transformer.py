import torch
import torch.nn as nn

class TransformerEncoder8M(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2, num_heads=8, mlp_dim=1024, num_layers=4, dropout=0.1):
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

    def forward(self, x, mask=None):
        """
        x: [bs, seqlen, 128]
        Returns: [bs, seqlen, 256]
        """
        x = self.input_proj(x)  # [bs, seqlen, 256]
        if mask is None:
            x = self.encoder(x, src_key_padding_mask=mask)
        else:
            x = self.encoder(x)
        
        x = self.output_proj(x)
        return x