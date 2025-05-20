# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn.init import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0., inner_dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
        self.inner_drop = nn.Dropout(inner_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.inner_drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """
        kv_token_length: if None, the attention is calculated between all tokens. If a number is set, only these tokens are used for keys and values and all tokens for the query
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % self.num_heads == 0, f"dim {self.dim} must be divisible by num_heads {self.num_heads}"
        self.depth = self.dim // self.num_heads

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape  # batch size, number of tokens (visual and keypoints) and number of channels
        # qkv = self.qkv(x)  # linear transformation of current tokens to queries, keys and values (channel dimension tripled)
        # qkv = qkv.reshape(B, N, 3, self.num_heads, torch.div(C, self.num_heads, rounding_mode='floor'))  # channels are split into 3 dims (qkv), number of heads and remaining channels
        # qkv = qkv.permute(2, 0, 3, 1, 4)  # now we have the order 3, B, heads, N, C
        # q, k, v = qkv[0], qkv[1], qkv[2]  # q, k, v have the dimensions B, heads, N, C
        q = self.wq(x) # (batch_size, seq_len, dim)
        k = self.wk(x) # (batch_size, seq_len, dim)
        v = self.wv(x) # (batch_size, seq_len, dim)

        q = self.split_heads(q, B)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, B)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, B)  # (batch_size, num_heads, seq_len_v, depth)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        k_transposed = k.transpose(-2, -1)  # k has dimensions B, heads, C, N
        attn = q @ k_transposed  # matrix multiplication in the last two dimensions, attn has dimensions B, heads, N, N_visual
        # scaling
        dk = k.shape[-1]
        attn = attn / np.sqrt(dk)
        if attn_mask is not None: # attn has shape B, heads, seq_len, seq_len
            attn = attn + (attn_mask * -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # matrix multiplication of attention and values, dimensions are B, heads, N, C (all N, not only visual)
        x = x.transpose(1, 2)  # dimensions are B, N, heads, C
        x = x.reshape(B, N, C)  # removing heads dimension
        x = self.proj(x)  # linear projection from channels to channels
        x = self.proj_drop(x)
        return x, attn

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return x.permute(0, 2, 1, 3)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 dropout=0., attn_dropout=0., inner_dropout=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, return_attention=True):
        super().__init__()
        self.return_attention = return_attention
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_dropout, proj_drop=dropout)
        # self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, add_bias_kv=qkv_bias, dropout=attn_dropout, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, dropout=dropout,
                       inner_dropout=inner_dropout)

    def forward(self, x, pos_encoding=None, mask=None):
        """
        @param x: The input to the transformer block with visual tokens and keypoint (and thickness) tokens
        @param pos_encoding: if available, it is added before the block
        @param return_attention: return the attention result without additional mlp and norm execution
        @return:
        """
        if pos_encoding is not None:
            assert pos_encoding.size(1) == x.size(1)
            x = x + pos_encoding
        y = self.norm1(x)
        # mask = mask.repeat(1, self.attn.num_heads, x.shape[1], 1) if mask is not None else None
        attn = self.attn(y, attn_mask=mask)
        # y, attn = self.attn(y, y, y, attn_mask=mask.reshape(-1, mask.shape[2], mask.shape[3]) if mask is not None else None)
        y = self.drop_path(y)
        x = x + y
        z = self.mlp(self.norm2(x))
        x = x + self.drop_path(z)
        if self.return_attention:
            return x, attn
        return x