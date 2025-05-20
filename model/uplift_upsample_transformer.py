# -*- coding: utf-8 -*-
"""
Created on 16.10.23

@author: Katja

"""
import logging
import math
from typing import Callable

import einops
import numpy as np

import torch
from torch import nn

import model.vision_transformer as vit
from model.uplift_upsample_transformer_config import UpliftUpsampleConfig


class LearnablePELayer(nn.Module):

    def __init__(self, shape):
        super().__init__()
        pe = torch.empty(shape)
        nn.init.trunc_normal_(pe, std=0.02)
        self.pe = nn.Parameter(pe)
        # prepare einops repeat string to repeat the positional encoding batch wise
        channel_str = " ".join([f"d{i}" for i in range(len(shape))])
        self.repeat_str = f"{channel_str} -> b {channel_str}"

    def forward(self, x):
        b = x.shape[0]
        batched_pe = einops.repeat(self.pe, self.repeat_str, b=b)
        return batched_pe


class LearnableMaskedTokenLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        learnable_token = torch.empty((dim,))
        nn.init.trunc_normal_(learnable_token, std=0.02)
        self.learnable_token = nn.Parameter(learnable_token)

    def forward(self, x):
        b, n = x.shape[:2]
        return einops.repeat(self.learnable_token, "c -> b n c", b=b, n=n)


class StridedMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, dropout=0., inner_dropout=0.,
                 stride=1, kernel_size=3, padding=None):
        super().__init__()
        self.out_features = out_features
        self.hidden_features = hidden_features or out_features
        self.stride = stride

        self.conv1 = nn.Conv1d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        if padding is None:
            padding = kernel_size // 2
        self.strided_conv = nn.Conv1d(in_channels=hidden_features, out_channels=out_features, kernel_size=kernel_size,
                                      stride=stride, padding=padding)
        self.drop = nn.Dropout(dropout)
        self.inner_drop = nn.Dropout(inner_dropout)

    def forward(self, x):
        x = x. transpose(1, 2)
        x = self.conv1(x)
        x = self.act(x)
        x = self.inner_drop(x)
        x = self.strided_conv(x)
        x = self.drop(x)
        return x


class StridedTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 dropout=0., attn_dropout=0., inner_dropout=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 kernel_size=3, stride=3, padding=None,
                 return_attention=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        self.return_attention = return_attention
        self.norm1 = norm_layer(dim)
        self.attn = vit.Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_dropout, proj_drop=dropout)
        # self.attn = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, add_bias_kv=qkv_bias, dropout=attn_dropout,
        #                                         batch_first=True)
        self.drop_path = vit.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = StridedMlp(in_features=dim, out_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, dropout=dropout,
                       inner_dropout=inner_dropout, kernel_size=kernel_size, stride=stride, padding=padding)

        self.max_pool = None
        if stride > 1:
            # NOTE: The strange pool size of 1 is taken directly from the authors original code!
            # They confirmed this in a github issue.
            self.max_pool = nn.MaxPool1d(kernel_size=1, stride=stride)

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
        x = x + self.drop_path(y)
        z = self.mlp(self.norm2(x))
        z = self.drop_path(z)

        if self.max_pool is not None:
            identity = x
            if self.padding == 0:
                identity = identity[:, 1:-1]
            identity = self.max_pool(identity.transpose(1, 2))
        else:
            identity = x

        x = identity + z
        x = x.transpose(1, 2)
        if self.return_attention:
            return x, attn
        return x


class UpliftUpsampleTransformer(nn.Module):

    def __init__(
            self,
            in_feature_size: int = 2,
            full_output: bool = True,
            num_frames: int = 9,
            num_keypoints: int = 17,
            spatial_d_model: int = 16,
            temporal_d_model: int = 256,
            spatial_depth: int = 3,
            temporal_depth: int = 3,
            strides: list = [3, 3, 3],
            paddings: list = None,
            num_heads: int = 8,
            mlp_ratio: float = 2.0,
            qkv_bias: bool = True,
            attn_drop_rate: float = 0.1,
            drop_rate: float = 0.1,
            drop_path_rate: tuple = (0, 0, 0),
            norm_layer: Callable = nn.LayerNorm,
            output_bn: bool = False,
            has_strided_input: bool = False,
            first_strided_token_attention_layer: int = 0,
            token_mask_rate: float = 0.0,
            learnable_masked_token: bool = False,
            return_attention: bool = False,
            ):
        """Model for the Uplift Upsample Transformer.
        """
        super(UpliftUpsampleTransformer, self).__init__()

        out_dim = num_keypoints * 3

        self.full_output = full_output
        self.num_frames = num_frames
        self.spatial_d_model = spatial_d_model
        self.temporal_d_model = temporal_d_model
        self.spatial_depth = spatial_depth
        self.temporal_depth = temporal_depth
        self.strides = strides
        self.has_strided_input = has_strided_input
        self.first_strided_token_attention_layer = first_strided_token_attention_layer
        self.token_mask_rate = token_mask_rate
        self.learnable_masked_token = learnable_masked_token
        self.return_attention = return_attention

        # Keypoint embedding and PE
        if self.spatial_depth > 0:
            self.keypoint_embedding = nn.Linear(in_features=in_feature_size, out_features=spatial_d_model)
        # Note that "token_dropout" might be misleading
        # It does not drop complete tokens, but performs standard dropout (independent across all axes)
        self.token_dropout = nn.Dropout(drop_rate)

        if spatial_depth > 0:
            self.spatial_pos_encoding = LearnablePELayer(shape=(num_keypoints, self.spatial_d_model))
        self.temporal_pos_encoding = LearnablePELayer(shape=(self.num_frames, self.temporal_d_model))

        self.strided_temporal_pos_encodings = nn.ModuleList()
        if len(self.strides) > 0:
            seq_len = self.num_frames
            for i, s in enumerate(self.strides):
                p = 1 if paddings is None else paddings[i][0]
                pe_shape = (seq_len, self.temporal_d_model)
                self.strided_temporal_pos_encodings.append(
                    LearnablePELayer(shape=pe_shape))
                seq_len = math.ceil((seq_len + p * 2 - 2) / s)

        # Token masking. This is the actual dropout on token level
        if token_mask_rate > 0 and learnable_masked_token is True:
            self.learnable_masked_token_layer = LearnableMaskedTokenLayer(dim=self.temporal_d_model)

        # Masked input token
        if self.has_strided_input is True:
            self.learnable_strided_input_token_layer = LearnableMaskedTokenLayer(dim=self.temporal_d_model)

        # Spatial blocks
        if self.spatial_depth > 0:
            dpr = drop_path_rate[0] if type(drop_path_rate) is list else drop_path_rate
            path_drop_rates = np.linspace(0, dpr, self.spatial_depth)
            self.spatial_blocks = nn.ModuleList([
                vit.TransformerBlock(dim=self.spatial_d_model, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, attn_dropout=attn_drop_rate,
                                     dropout=drop_rate, drop_path=path_drop_rates[i], norm_layer=norm_layer,
                                     return_attention=True)
                for i in range(self.spatial_depth)])

            self.spatial_norm = norm_layer(self.spatial_d_model, eps=1e-6)
        self.spatial_to_temporal_mapping = nn.Linear(self.spatial_d_model * num_keypoints, self.temporal_d_model)

        # Full sequence temporal blocks
        if self.temporal_depth > 0:
            dpr = drop_path_rate[1] if type(drop_path_rate) is list else drop_path_rate
            path_drop_rates = np.linspace(0, dpr, self.temporal_depth)
            self.temporal_blocks = nn.ModuleList([
                vit.TransformerBlock(dim=self.temporal_d_model, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, attn_dropout=attn_drop_rate,
                                     dropout=drop_rate, inner_dropout=drop_rate,
                                     drop_path=path_drop_rates[i],
                                     act_layer=nn.ReLU,
                                     norm_layer=norm_layer,
                                     return_attention=True)
                for i in range(self.temporal_depth)])

        # Strided temporal blocks
        if len(self.strides) > 0:
            pad_values = paddings
            if paddings is None:
                pad_values = [None] * len(strides)

            dpr = drop_path_rate[2] if type(drop_path_rate) is list else drop_path_rate
            path_drop_rates = np.linspace(0, dpr, len(strides))
            self.strided_temporal_blocks = nn.ModuleList([
                StridedTransformerBlock(dim=temporal_d_model, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias, attn_dropout=attn_drop_rate,
                                        dropout=drop_rate, inner_dropout=drop_rate,
                                        stride=s, kernel_size=3,
                                        padding=pad_values[i],
                                        drop_path=path_drop_rates[i],
                                        act_layer=nn.ReLU,
                                        norm_layer=norm_layer,
                                        return_attention=True)
                for i, s in enumerate(strides)])

        self.head1 = None
        if self.full_output is True and self.temporal_depth > 0:
            self.head1 = nn.ModuleList()
            if output_bn:
                self.head1.append(nn.BatchNorm1d(self.temporal_d_model, momentum=0.1, eps=1e-5))
            self.head1.append(nn.Linear(self.temporal_d_model, out_dim))

        self.head2 = nn.ModuleList()
        if output_bn:
            self.head2.append(nn.BatchNorm1d(temporal_d_model, momentum=0.1, eps=1e-5))
        self.head2.append(nn.Linear(self.temporal_d_model, out_dim))

    def random_token_masking(self, x, masked_token_value):
        # Input has shape B, N, C
        b, n, c = x.shape
        # Ensure that central frame is never masked
        mid_index = self.num_frames // 2
        center_mask = torch.range(start=0, end=self.num_frames, dtype=torch.int32)
        center_mask = torch.not_equal(center_mask, mid_index)
        center_mask = einops.repeat(center_mask, "n -> b n", b=b)

        # Draw random token masking
        # If mask is 1, token will be masked
        # Mask has shape B,N
        token_mask = torch.rand(size=(b, n), dtype=torch.float32)
        token_mask = token_mask < self.token_mask_rate

        # Merge token mask and center mask
        token_mask = torch.logical_and(center_mask, token_mask)

        # Mask: B,N,C
        token_mask = (torch.unsqueeze(token_mask, dim=-1)).float()
        inv_token_mask = 1. - token_mask

        # Mask token
        output = (x * inv_token_mask) + (masked_token_value * token_mask)
        return output

    def spatial_transformation(self, x):
        b, n, p, c = x.shape
        if self.spatial_depth == 0:
            x = einops.rearrange(x, "b n p c -> b n (p c)")
        else:
            # Fuse batch size and frames for frame-independent processing
            x = einops.rearrange(x, "b n p c -> (b n) p c")
            # ToDo: Handle stride mask with gather/scatter for efficiency during training
            x = self.keypoint_embedding(x)
            batched_sp_pe = self.spatial_pos_encoding(x)
            x = x + batched_sp_pe
            x = self.token_dropout(x)

            pe = None
            for block in self.spatial_blocks:
                x, att = block(x, pos_encoding=pe)
            x = self.spatial_norm(x)
            x = einops.rearrange(x, "(b n) p c -> b n (p c)", n=n)

        x = self.spatial_to_temporal_mapping(x)
        return x

    def temporal_transformation(self, x, training=None, stride_mask=None):
        if training is True and self.token_mask_rate > 0:
            masked_token_value = 0. if self.learnable_masked_token is False else self.learnable_masked_token_layer(x)
            x = self.random_token_masking(x, masked_token_value=masked_token_value)

        batched_temp_pe = self.temporal_pos_encoding(x)

        if self.has_strided_input is True:
            # (B, N, C)
            strided_input_token = self.learnable_strided_input_token_layer(batched_temp_pe)
            # (B, N)
            # Stride mask is 1 on valid (i.e. non-masked) indices !!!
            stride_mask = stride_mask.float()
            inv_stride_mask = 1. - stride_mask
            # Masked input token (B, N, C)
            x = (stride_mask[..., None] * x) + (inv_stride_mask[..., None] * strided_input_token)

        x = x + batched_temp_pe

        pe = None
        att_list = []
        if self.temporal_depth > 0:
            for i, block in enumerate(self.temporal_blocks):
                if self.has_strided_input and i < self.first_strided_token_attention_layer:
                    # Use inverted stride_mask to disable attention on the strided input tokens
                    # Must be broadcastable to (B, HEADS, QUERIES, KEYS)
                    attn_mask = inv_stride_mask[:, None, None, :]
                else:
                    attn_mask = None
                x, att = block(x, pos_encoding=pe, mask=attn_mask)
                att_list.append(att)
        # x is (B, N, C)
        return x, att_list

    def strided_temporal_transformation(self, x, stride_mask=None):
        b, c, n = x.shape

        for i, block in enumerate(self.strided_temporal_blocks):
            if self.temporal_depth == 0 and self.has_strided_input and i < self.first_strided_token_attention_layer:
                # Use inverted stride_mask to disable attention on the strided input tokens
                # Must be broadcastable to (B, HEADS, QUERIES, KEYS)
                stride_mask = stride_mask.float()
                inv_stride_mask = 1. - stride_mask
                attn_mask = inv_stride_mask[:, None, None, :]
                logging.info(
                    "NOTE: Without temporal transformer blocks, \
                        deferred upsampling token attention will be used in strided transformer.")
            else:
                attn_mask = None
            pe = self.strided_temporal_pos_encodings[i](x)
            x, att = block(x, pos_encoding=pe, mask=attn_mask)
        # x is (B, N, C)
        return x

    def forward(self, inputs):
        if self.has_strided_input:
            x, stride_mask = inputs[0], inputs[1]
        else:
            x = inputs
            stride_mask = None
        b, n, p, _ = x.shape
        x = self.spatial_transformation(x)
        # Full sequence temporal transformer
        x, att_list = self.temporal_transformation(x, stride_mask=stride_mask)
        # Prediction for full sequence
        full_output = None
        if self.full_output is True and self.temporal_depth > 0:
            full_output = x
            for layer in self.head1:
                full_output = layer(full_output)
            full_output = full_output.reshape(b, n, p, 3)

        # Strided transformer
        if len(self.strides) > 0:
            x = self.strided_temporal_transformation(x, stride_mask=stride_mask)
            # Prediction for central frame
            central_output = x
        else:
            central_output = x[:, self.num_frames // 2, :]
            central_output = central_output[:, None, :]
        for layer in self.head2:
            central_output = layer(central_output)
        central_output = einops.rearrange(central_output, "b n (p c) -> (b n) p c", n=1, p=p, c=3)

        if self.return_attention:
            return full_output, central_output, att_list
        else:
            return full_output, central_output


def build_uplift_upsample_transformer(config: UpliftUpsampleConfig):
    input_shape = (config.SEQUENCE_LENGTH, config.NUM_KEYPOINTS, 2)
    has_strided_input = config.MASK_STRIDE is not None
    if has_strided_input:
        if type(config.MASK_STRIDE) is int and config.MASK_STRIDE == 1:
            has_strided_input = False
        if type(config.MASK_STRIDE) is list and config.MASK_STRIDE[0] == 1:
            has_strided_input = False

    model = UpliftUpsampleTransformer(
        in_feature_size=config.IN_FEATURE_SIZE,
        full_output=not config.USE_REFINE,
        num_frames=config.SEQUENCE_LENGTH,
        num_keypoints=config.NUM_KEYPOINTS,
        spatial_d_model=config.SPATIAL_EMBED_DIM,
        temporal_d_model=config.TEMPORAL_EMBED_DIM,
        spatial_depth=config.SPATIAL_TRANSFORMER_BLOCKS,
        temporal_depth=config.TEMPORAL_TRANSFORMER_BLOCKS,
        strides=config.STRIDES,
        paddings=config.PADDINGS,
        num_heads=config.NUM_HEADS,
        mlp_ratio=config.MLP_RATIO,
        qkv_bias=config.QKV_BIAS,
        attn_drop_rate=config.ATTENTION_DROP_RATE,
        drop_rate=config.DROP_RATE,
        drop_path_rate=config.DROP_PATH_RATE,
        output_bn=config.OUTPUT_BN,
        has_strided_input=has_strided_input,
        first_strided_token_attention_layer=config.FIRST_STRIDED_TOKEN_ATTENTION_LAYER,
        token_mask_rate=config.TOKEN_MASK_RATE,
        learnable_masked_token=config.LEARNABLE_MASKED_TOKEN,
    )

    return model