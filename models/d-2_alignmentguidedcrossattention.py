import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentGuidedCrossAttention(nn.Module):

    def __init__(
        self,
        d_model=128,
        nhead=8,
        dropout=0.1,
        alignment_weight=1.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.alignment_weight = alignment_weight

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        query_tokens,
        context_tokens,
        alignment_bias=None,
    ):
        # query_tokens:   [BO, Tq, D]
        # context_tokens: [BO, Tk, D]

        BO, Tq, D = query_tokens.shape
        Tk = context_tokens.shape[1]

        q_in = self.norm1(query_tokens)
        k_in = self.norm1(context_tokens)

        Q = self.q_proj(q_in)
        K = self.k_proj(k_in)
        V = self.v_proj(context_tokens)

        # [BO, T, D] -> [BO, H, T, Dh]
        Q = Q.view(BO, Tq, self.nhead, self.head_dim).transpose(1, 2)
        K = K.view(BO, Tk, self.nhead, self.head_dim).transpose(1, 2)
        V = V.view(BO, Tk, self.nhead, self.head_dim).transpose(1, 2)

        # [BO, H, Tq, Tk]
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1),
        ) / (self.head_dim ** 0.5)

        if alignment_bias is not None:
            # alignment_bias: [BO, Tq, Tk]

            scores = scores + (
                self.alignment_weight
                * alignment_bias.unsqueeze(1)
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        x = torch.matmul(attn, V)
        # [BO, H, Tq, Dh]

        x = x.transpose(1, 2).contiguous()
        x = x.view(BO, Tq, D)

        x = self.out_proj(x)

        x = query_tokens + x

        x = x + self.ffn(
            self.norm2(x)
        )

        return x, attn