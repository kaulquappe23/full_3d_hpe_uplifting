# -*- coding: utf-8 -*-
"""
Created on 02.07.25

@author: Katja

"""
import torch
from tqdm import tqdm
import numpy as np

from run.eval import transform_rotations


def execute_inference(loader, model, cfg, betas=None, smoothing=False):
    model.to(cfg.DEVICE)
    model.eval()

    results = []
    for b_i, kpts2d in enumerate(tqdm(loader, colour="YELLOW")):
        model_input = kpts2d.to(cfg.DEVICE).float()

        assert not model.has_strided_input

        with torch.no_grad():
            if hasattr(cfg, "SMPLX_LAYER") and any(cfg.SMPLX_LAYER):
                betas = betas.to(cfg.DEVICE).float()
                output = model(model_input, betas)
            else:
                output = model(model_input)
            full_output, central_output = output
            assert isinstance(full_output, tuple) and isinstance(central_output, tuple)
            # pred_kpts3d, pred_c_kpts3d = full_output[0], central_output[0]
            pred_rots, pred_c_rots = full_output[1], central_output[1]
            results.append(pred_c_rots)
    results = torch.concat(results)
    results = transform_rotations(cfg, results)

    if smoothing:
        filter = [1, 3, 5, 3, 1]
        filter = np.asarray(filter)
        filter = filter / np.sum(filter)

        smoothed_data = np.zeros_like(results)

        pad_width = ((filter.shape[0] // 2, filter.shape[0] // 2), (0, 0), (0, 0))
        results = np.pad(results, pad_width=pad_width, mode="edge")
        _, num_joints, num_dims = results.shape
        for joint_idx in range(num_joints):
            for dim_idx in range(num_dims):
                smoothed_signal = np.convolve(results[:, joint_idx, dim_idx], filter, mode='valid')
                smoothed_data[:, joint_idx, dim_idx] = smoothed_signal
        results = smoothed_data
    results = torch.from_numpy(results).to(cfg.DEVICE).float()

    return results