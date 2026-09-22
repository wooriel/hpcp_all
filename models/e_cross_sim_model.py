import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenAttentionPool(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()

        self.score = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, 8, 14, 128]

        B, O, T, D = x.shape

        x = x.reshape(B, O * T, D) # [B, 112, 128]

        weights = self.score(x) # [B, 112, 1]
        weights = torch.softmax(weights, dim=1) # [B, 112, 1]

        x = torch.sum(weights * x, dim=1) # [B, 128]

        return x
    

class CrossBinModel(nn.Module):

    def __init__(self, tokenizer, self_attention, cross_attention, classification_head, binary_head, d_model=128, num_octaves=8):
        super().__init__()

        self.num_octaves = num_octaves
        self.d_model = d_model

        self.tokenizer = tokenizer
        self.self_attention = self_attention
        self.cross_attn = cross_attention

        self.classification_head = classification_head
        self.binary_head = binary_head

        # Individual-song aggregation
        self.song_pool = TokenAttentionPool(
            d_model=d_model
        )

        # Cross-attention aggregation
        self.pair_pool = TokenAttentionPool(
            d_model=d_model
        )

    def encode(self, x):
        B = x.shape[0]

        x = self.tokenizer(x)
        # [B*8, 14, 128]

        x = self.self_attention(x)
        # [B*8, 14, 128]

        return x

    def forward(self, src, tgt, use_classification=True):
        B = src.shape[0]

        src_tokens = self.encode(src)
        tgt_tokens = self.encode(tgt)

        outputs = {}

        if use_classification:
            src_song_tokens = src_tokens.reshape(B, 8, 14, 128)
            tgt_song_tokens = tgt_tokens.reshape(B, 8, 14, 128)

            src_feature = self.song_pool(src_song_tokens)
            tgt_feature = self.song_pool(tgt_song_tokens)

            outputs["src_class_logits"] = self.classification_head(src_feature)
            outputs["tgt_class_logits"] = self.classification_head(tgt_feature)

        ab_tokens, attn_ab = self.cross_attn(src_tokens, tgt_tokens)
        ba_tokens, attn_ba = self.cross_attn(tgt_tokens, src_tokens)

        ab_tokens = ab_tokens.reshape(B, 8, 14, 128)
        ba_tokens = ba_tokens.reshape(B, 8, 14, 128)

        ab_feature = self.pair_pool(ab_tokens)
        ba_feature = self.pair_pool(ba_tokens)

        outputs["pair_logits"] = self.binary_head(
            ab_feature,
            ba_feature,
        )
        outputs["attn_ab"] = attn_ab
        outputs["attn_ba"] = attn_ba

        return outputs
    
    # def forward(self, src, tgt):

    #     B = src.shape[0]

    #     # ==========================================
    #     # Individual-song encoding
    #     # ==========================================

    #     src_tokens = self.encode(src)
    #     # [B*8,14,128]

    #     tgt_tokens = self.encode(tgt)
    #     # [B*8,14,128]


    #     # ==========================================
    #     # Classification branch
    #     # ==========================================

    #     src_song_tokens = src_tokens.reshape(
    #         B,
    #         self.num_octaves,
    #         14,
    #         self.d_model,
    #     )
    #     # [B,8,14,128]

    #     tgt_song_tokens = tgt_tokens.reshape(
    #         B,
    #         self.num_octaves,
    #         14,
    #         self.d_model,
    #     )
    #     # [B,8,14,128]

    #     src_feature = self.song_pool(
    #         src_song_tokens
    #     )
    #     # [B,128]

    #     tgt_feature = self.song_pool(
    #         tgt_song_tokens
    #     )
    #     # [B,128]

    #     src_class_logits = self.classification_head(
    #         src_feature
    #     )
    #     # [B,1703]

    #     tgt_class_logits = self.classification_head(
    #         tgt_feature
    #     )
    #     # [B,1703]


    #     # ==========================================
    #     # Bidirectional cross-attention
    #     # ==========================================

    #     ab_tokens, attn_ab = self.cross_attn(
    #         src_tokens,
    #         tgt_tokens,
    #     )
    #     # [B*8,14,128]

    #     ba_tokens, attn_ba = self.cross_attn(
    #         tgt_tokens,
    #         src_tokens,
    #     )
    #     # [B*8,14,128]


    #     # ==========================================
    #     # Pair aggregation
    #     # ==========================================

    #     ab_tokens = ab_tokens.reshape(
    #         B,
    #         self.num_octaves,
    #         14,
    #         self.d_model,
    #     )
    #     # [B,8,14,128]

    #     ba_tokens = ba_tokens.reshape(
    #         B,
    #         self.num_octaves,
    #         14,
    #         self.d_model,
    #     )
    #     # [B,8,14,128]

    #     ab_feature = self.pair_pool(
    #         ab_tokens
    #     )
    #     # [B,128]

    #     ba_feature = self.pair_pool(
    #         ba_tokens
    #     )
    #     # [B,128]


    #     # ==========================================
    #     # Binary decision
    #     # ==========================================

    #     pair_logits = self.binary_head(
    #         ab_feature,
    #         ba_feature,
    #     )
    #     # [B,1]

    #     return {
    #         "src_feature": src_feature,
    #         "tgt_feature": tgt_feature,

    #         "src_class_logits": src_class_logits,
    #         "tgt_class_logits": tgt_class_logits,

    #         "pair_logits": pair_logits,

    #         "attn_ab": attn_ab,
    #         "attn_ba": attn_ba,
    #     }
    

# class CrossBinModel(nn.Module):
#     def __init__(self, encoder, classification_head, binary_head):
#         super().__init__()

#         self.encoder = encoder
#         self.classification_head = classification_head
#         self.binary_head = binary_head

#     def forward(self, src, tgt):
#         src_tokens = self.encoder(src)
#         tgt_tokens = self.encoder(tgt)

#         ab_tokens, attn_ab = self.cross_attn(
#             src_tokens,
#             tgt_tokens,
#         )

#         ba_tokens, attn_ba = self.cross_attn(
#             tgt_tokens,
#             src_tokens,
#         )

#         B = src.shape[0]

#         ab_tokens = ab_tokens.reshape(B, 8, 14, 128)
#         ba_tokens = ba_tokens.reshape(B, 8, 14, 128)

#         ab_feature = ab_tokens.mean(dim=(1, 2)) # [B, 128]

#         ba_feature = ba_tokens.mean(dim=(1, 2)) # [B, 128]

#         # [B, 128]
#         src_class_logits = self.classification_head(src_feature)

#         tgt_class_logits = self.classification_head(tgt_feature)

#         pair_logits = self.binary_head(ab_feature, ba_feature)

#         return {
#             "src_feature": src_tokens,
#             "tgt_feature": tgt_tokens,
#             "src_class_logits": src_class_logits,
#             "tgt_class_logits": tgt_class_logits,
#             "pair_logits": pair_logits,
#         }