import torch
import torch.nn as nn
import torch.nn.functional as F


class OctaveAttentionBlock(nn.Module):

    def __init__(self, d_model=128, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.1, num_octaves=8):
        super().__init__()

        self.num_octaves = num_octaves

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x):
        BO, T, D = x.shape # x: [B*8, 14, 128]
        B = BO // self.num_octaves

        x = x.reshape(B, self.num_octaves, T, D) # [B, 8, 14, 128]
        x = x.permute(0, 2, 1, 3) # [B, 14, 8, 128]
        x = x.reshape(B * T, self.num_octaves, D) # [B*14, 8, 128]

        x = self.encoder(x) # [B*14, 8, 128]

        return x