import torch
import torch.nn as nn
import torch.nn.functional as F


class CQTViTTokenizer(nn.Module):
    def __init__(self, embedding_dim=64, patch_time=24):
        super().__init__()
        self.dim = embedding_dim
        self.patch = patch_time

        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=self.dim,
            kernel_size=(self.patch, self.patch),
            stride=(self.patch, self.patch),
        )

    def forward(self, x):# [B, 1, 96, 4375]
        B = x.shape[0]
        x = self.patch_embed(x) # [B, 64, 4, 182]
        x = x.permute(0, 2, 3, 1) # [B, 4, 182, 64]
        x = x.reshape(B * 4, -1, self.dim) # [B*4, 182, 64]

        return x
    

class CQTLinearTokenizer(nn.Module): # used for CQT
    def __init__(self, d_model=64, patch_freq=24, patch_time=24):
        super().__init__()
        self.dim = d_model
        self.patch_freq = patch_freq
        self.patch_time = patch_time
        self.proj = nn.Linear(patch_freq * patch_time, d_model)

    def forward(self, x): # x: [B, 1, 96, 4375]
        B, _, F, T = x.shape
        x = x.squeeze(1) # [B, 96, 4375]

        usable_F = (F // self.patch_freq) * self.patch_freq
        usable_T = (T // self.patch_time) * self.patch_time
        x = x[:, :usable_F, :usable_T]

        n_freq = usable_F // self.patch_freq
        n_time = usable_T // self.patch_time

        x = x.reshape(B, n_freq, self.patch_freq, n_time, self.patch_time) # [B, 4, 24, 182, 24]
        x = x.permute(0, 1, 3, 2, 4) # [B, 4, 182, 24, 24]
        x = x.reshape(B, n_freq, n_time, self.patch_freq * self.patch_time) # [B, 4, 182, 576]
        x = self.proj(x) # [B, 4, 182, 64]
        x = x.reshape(B * n_freq, n_time, self.dim) # [B*4, 182, 64]

        return x
    

class HPCPViTTokenizer(nn.Module):
    def __init__(self, dim=64, patch_time=12):
        super().__init__()
        self.dim = dim
        self.patch = patch_time
        
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=self.dim,
            kernel_size=(12, self.patch),
            stride=(12, self.patch),
        )

    def forward(self, x): # [B, 1, 12, 4375]
        B = x.shape[0]
        x = self.patch_embed(x)   # [B, 64, 1, 364]
        x = x.permute(0, 2, 3, 1) # [B, 1, 364, 64]
        x = x.reshape(B, -1, self.dim) # [B, 364, 64]

        return x
    

# class RawHPCPTokenizer(nn.Module):
#     def __init__(self, dim=64, patch_time=12):
#         super().__init__()
#         self.dim = dim
#         self.patch = patch_time
#         self.proj = nn.Linear(self.patch, self.dim)

#     def forward(self, x): # x: [B, 1, 12, T]
#         x = x.squeeze(1) # [B, 12, T]
#         x = x.transpose(1, 2) # [B, T, 12]
#         x = self.proj(x) # [B, T, 64]

#         return x
    
    
# class RawTokenizer(nn.Module):
#     def __init__(self, d_model=64, patch_time=12):
#         super().__init__()

#         self.patch_time = patch_time
#         self.proj = nn.Linear(12 * patch_time, d_model)

#     def forward(self, x): # x: [B, 1, 96, 4375]
#         x = x.squeeze(1) # [B, 96, 4375]
#         x = x.transpose(1, 2) # [B, 4375, 96]

#         B, T, C = x.shape
#         usable_T = (T // self.patch_time) * self.patch_time
#         x = x[:, :usable_T, :] # [B, 4368, 96]
#         x = x.reshape(B, usable_T // self.patch_time, C * self.patch_time) # [B, 364, 144]
#         x = self.proj(x) # [B, 364, 64]

#         return x
    

class RawTokenizer(nn.Module):
    def __init__(self, d_model=64, patch_h=23, patch_w=25):
        super().__init__()

        self.patch_h = patch_h
        self.patch_w = patch_w

        patch_dim = 1 * patch_h * patch_w
        self.proj = nn.Linear(patch_dim, d_model)

    def forward(self, x):
        B, C, H, W = x.shape # x: [B, 1, 23, 4375]

        usable_h = (H // self.patch_h) * self.patch_h # patch_h로 나눠지는 길이
        usable_w = (W // self.patch_w) * self.patch_w # patch_w로 나눠지는 길이

        x = x[:, :, :usable_h, :usable_w] # x: [B, 1, 23, 4375]

        x = x.unfold(2, self.patch_h, self.patch_h)
        x = x.unfold(3, self.patch_w, self.patch_w) # [B, 1, 1, 23, 175, 25]

        x = x.permute(0, 2, 3, 1, 4, 5) # [B, 1, 175, 1, 23, 25] = [B, H_patch, W_patch, C, patch_h, patch_w]
        x = x.reshape(B, -1, C * self.patch_h * self.patch_w) # [B, 1*175, 575]
        x = self.proj(x) # [B, N, d_model]

        return x
