# -*- coding: utf-8 -*-
"""
Created on 17.10.23

@author: Katja

"""
from copy import deepcopy
import logging
import pickle

import torch

from model import build_uplift_upsample_transformer
from model import UpliftUpsampleConfig


def create_block_sequence(weights_tf, weights_pt, block_name_tf, block_name_pt, num_blocks,
                          tf_ln_count=0, tf_fc_count=0, tf_block_count=0,
                          mlp_name="mlp", mlp_weight_name="dense", tf_special_mlp_count=0, tf_special_block_count=0):

    for i in range(num_blocks):
        tf_num = f"_{tf_block_count}" if tf_block_count > 0 else ""
        tf_block_count += 1
        for j in range(1, 3):
            tf_norm_num = f"_{tf_ln_count}" if tf_ln_count > 0 else ""
            tf_ln_count += 1
            weights_pt[f"{block_name_pt}.{i}.norm{j}.weight"] = torch.from_numpy(
                weights_tf.pop(f"{block_name_tf}_{i + 1}/layer_normalization{tf_norm_num}/gamma:0"))
            weights_pt[f"{block_name_pt}.{i}.norm{j}.bias"] = torch.from_numpy(
                weights_tf.pop(f"{block_name_tf}_{i + 1}/layer_normalization{tf_norm_num}/beta:0"))
        for layer in ["wq", "wk", "wv", "proj"]:
            tf_dense_num = f"_{tf_fc_count}" if tf_fc_count > 0 else ""
            tf_fc_count += 1
            weights_pt[f"{block_name_pt}.{i}.attn.{layer}.weight"] = torch.from_numpy(
                weights_tf.pop(f"{block_name_tf}_{i + 1}/mha{tf_num}/dense{tf_dense_num}/kernel:0")).transpose(0, 1)
            weights_pt[f"{block_name_pt}.{i}.attn.{layer}.bias"] = torch.from_numpy(
                weights_tf.pop(f"{block_name_tf}_{i + 1}/mha{tf_num}/dense{tf_dense_num}/bias:0"))
        if mlp_name != "mlp":
            tf_num = f"_{tf_special_block_count}" if tf_special_block_count > 0 else ""
            tf_special_block_count += 1
        for j in range(1, 3):
            if mlp_weight_name == "dense":
                tf_dense_num = f"_{tf_fc_count}" if tf_fc_count > 0 else ""
                tf_fc_count += 1
                pt_name = f"fc{j}"
            else:
                tf_dense_num = f"_{tf_special_mlp_count}" if tf_special_mlp_count > 0 else ""
                tf_special_mlp_count += 1
                if mlp_name == "strided_mlp":
                    if j == 1:
                        pt_name = "conv1"
                    else:
                        pt_name = "strided_conv"
                else:
                    raise RuntimeError("Unknown MLP type")

            weight_tf = torch.from_numpy(
                    weights_tf.pop(f"{block_name_tf}_{i + 1}/{mlp_name}{tf_num}/{mlp_weight_name}{tf_dense_num}/kernel:0"))
            if len(weight_tf.shape) == 2:
                weight_tf = weight_tf.transpose(1, 0)
            elif len(weight_tf.shape) == 3:
                weight_tf = weight_tf.permute(2, 1, 0)
            weights_pt[f"{block_name_pt}.{i}.mlp.{pt_name}.weight"] = weight_tf
            weights_pt[f"{block_name_pt}.{i}.mlp.{pt_name}.bias"] = torch.from_numpy(
                    weights_tf.pop(f"{block_name_tf}_{i + 1}/{mlp_name}{tf_num}/{mlp_weight_name}{tf_dense_num}/bias:0"))
    return tf_ln_count, tf_fc_count, tf_block_count

def convert_tf_to_torch_weights(tf_weights):
    torch_weights = {}
    torch_weights["keypoint_embedding.weight"] = torch.from_numpy(tf_weights.pop("keypoint_embedding/kernel:0")).transpose(1, 0)
    torch_weights["keypoint_embedding.bias"] = torch.from_numpy(tf_weights.pop("keypoint_embedding/bias:0"))
    torch_weights["spatial_pos_encoding.pe"] = torch.from_numpy(tf_weights.pop("spatial_pe/positional_encoding_weights:0"))
    torch_weights["temporal_pos_encoding.pe"] = torch.from_numpy(tf_weights.pop("temporal_pe/positional_encoding_weights:0"))
    for i in range(1, 4):
        torch_weights[f"strided_temporal_pos_encodings.{i-1}.pe"] = torch.from_numpy(tf_weights.pop(f"strided_temporal_pe_{i}/positional_encoding_weights:0"))
    torch_weights["learnable_strided_input_token_layer.learnable_token"] = torch.from_numpy(tf_weights.pop("strided_input_token_layer/learnable_masked_token:0"))

    tf_layer_norm_count, tf_dense_count, tf_blocks_count = create_block_sequence(tf_weights, torch_weights, "spatial_block", "spatial_blocks", 4)
    torch_weights["spatial_norm.weight"] = torch.from_numpy(tf_weights.pop("spatial_norm/gamma:0"))
    torch_weights["spatial_norm.bias"] = torch.from_numpy(tf_weights.pop("spatial_norm/beta:0"))
    torch_weights["spatial_to_temporal_mapping.weight"] = torch.from_numpy(tf_weights.pop("spatial_to_temporal_fc/kernel:0")).transpose(1, 0)
    torch_weights["spatial_to_temporal_mapping.bias"] = torch.from_numpy(tf_weights.pop("spatial_to_temporal_fc/bias:0"))
    tf_layer_norm_count, tf_dense_count, tf_blocks_count = create_block_sequence(tf_weights, torch_weights, "temporal_block", "temporal_blocks", 4, tf_layer_norm_count, tf_dense_count, tf_blocks_count)
    create_block_sequence(tf_weights, torch_weights, "strided_temporal_block", "strided_temporal_blocks", 3, tf_layer_norm_count,
                          tf_dense_count, tf_blocks_count, "strided_mlp", "conv1d")
    torch_weights["head1.0.weight"] = torch.from_numpy(tf_weights.pop("temporal_fc/kernel:0")).transpose(1, 0)
    torch_weights["head1.0.bias"] = torch.from_numpy(tf_weights.pop("temporal_fc/bias:0"))
    torch_weights["head2.0.weight"] = torch.from_numpy(tf_weights.pop("strided_temporal_fc/kernel:0")).transpose(1, 0)
    torch_weights["head2.0.bias"] = torch.from_numpy(tf_weights.pop("strided_temporal_fc/bias:0"))
    return torch_weights


def filter_weights(weights: dict, model: torch.nn.Module) -> dict:
    out_weights = deepcopy(weights)
    logging.info("Filtering given weights.")
    mstate = model.state_dict()
    for key, val in weights.items():
        if key not in mstate or val.size() != mstate[key].size():
            out_weights.pop(key)
            logging.info(f"Removed `{key}` from weights.")
    return out_weights

