# -*- coding: utf-8 -*-
"""
Created on 02.07.25

@author: Katja

"""
import sys

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

from dataset.camera import normalize_screen_coordinates


class InferenceDataset(torch.utils.data.Dataset):

    def __init__(self, kpts_2d, cfg):
        self.kpts_2d = kpts_2d
        self.seq_len = cfg.SEQUENCE_LENGTH
        assert cfg.SEQUENCE_STRIDE == 1

        padding_type = cfg.PADDING_TYPE
        if padding_type == "zeros":
            self.pad_type = "constant"
        elif padding_type == "copy":
            self.pad_type = "edge"
        else:
            raise ValueError(f"Padding type not supported: {padding_type}")

    def __getitem__(self, item):
        start_idx = item - self.seq_len // 2
        end_idx = item + self.seq_len // 2 + 1

        pad_left = 0 if start_idx >= 0 else np.abs(start_idx)
        pad_right = 0 if end_idx < self.__len__() else end_idx - self.__len__() + 1

        start_idx = max(0, start_idx)
        end_idx = min(self.__len__() - 1, end_idx)

        sequence = self.kpts_2d[start_idx:end_idx]
        sequence = np.pad(
                array=sequence,
                pad_width=((pad_left, pad_right), (0, 0), (0, 0)),
                mode=self.pad_type,
                )
        return sequence

    def __len__(self):
        return self.kpts_2d.shape[0]



def get_inference_dataloader(kpts_2d, cfg):
    kpts_2d[..., :2] = normalize_screen_coordinates(
            kpts_2d[..., :2], w=cfg.RES_W, h=cfg.RES_H
            )
    dataset = InferenceDataset(kpts_2d, cfg)
    if sys.gettrace() is None:
        WORKERS = 16
    else:
        WORKERS = 0

    data_loader = DataLoader(
            dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=WORKERS,
            pin_memory=False,
            drop_last=False,
            )
    return data_loader