import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionEncoder(nn.Module):
    def __init__(self, nhead, d_model=64, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.dim = d_model
        self.head = nhead

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True, # [B, N, D] order
            norm_first=True, # apply layernorm before attention layer
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x): # [B*nhead, N, self.dim]
        return self.encoder(x)