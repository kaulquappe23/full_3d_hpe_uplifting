import logging
import sys
import numpy as np
from torch.utils.data import DataLoader

import paths
from dataset.fit3d.seq_generator import Fit3DSequenceGenerator
from utils.config import Config


def create_fit3d_datasets(
    cfg: Config,
    train_subset: str,
    val_subset: str,
):
    train_ds, val_ds, val_batches, train_loader, val_loader = [None] * 5

    for split, selection in zip(["train", "val"], [train_subset, val_subset]):
        if selection is not None:
            # The dataset is subsampled to every Nth frame (i.e. a sequence is extracted at every Nth frame)
            # The frame rate is not changed, however!
            dataset, data_loader = create_subset_data(cfg, split, selection)

            logging.info(f"{selection} - Sequences: {len(dataset)}\n")

            if split == "train":
                train_ds = dataset
                train_loader = data_loader
            else:
                num_examples = len(dataset)
                # update cfg for logging, not set initially through cfg
                if cfg.VALIDATION_EXAMPLES < 0:
                    cfg.VALIDATION_EXAMPLES = num_examples
                    assert cfg.VALIDATION_EXAMPLES <= num_examples

                val_batches = int(
                    np.ceil(num_examples / cfg.BATCH_SIZE))
                val_ds = dataset
                val_loader = data_loader

    return train_ds, train_loader, val_ds, val_loader, val_batches


def create_subset_data(
    cfg: Config,
    split: str,
    selection: str,
):
    if sys.gettrace() is None:
        WORKERS = 16
    else:
        WORKERS = 0

    kwargs = paths.FIT3D_PROCESSED
    subsample = (
        cfg.TRAIN_SUBSAMPLE
        if selection == "train"
        else (
            cfg.VALIDATION_SUBSAMPLE
            if (selection == "val")
            else cfg.TEST_SUBSAMPLE
        )
    )
    shuffle = selection == "train"
    stride_mask_rand_shift = cfg.STRIDE_MASK_RAND_SHIFT and selection == "train"

    do_flip = False # there is either WBA, or no flipping at all

    load_smplx = True \
        if hasattr(cfg, "LOAD_SMPLX") and cfg.LOAD_SMPLX \
        else False
    rot_type = cfg.ROT_REP if hasattr(cfg, "ROT_REP") else None

    dataset = Fit3DSequenceGenerator(
        subset=selection,
        seq_len=cfg.SEQUENCE_LENGTH,
        subsample=subsample,
        stride=cfg.SEQUENCE_STRIDE,
        padding_type=cfg.PADDING_TYPE,
        rotation_type=rot_type,
        flip_lr_indices=cfg.AUGM_FLIP_KEYPOINT_ORDER,
        mask_stride=cfg.MASK_STRIDE,
        stride_mask_align_global=selection == "test",
        rand_shift_stride_mask=stride_mask_rand_shift,
        flip_augment=do_flip,
        seed=cfg.SEED,
        load_smplx_params=load_smplx,
        **kwargs,
    )

    # half of batch size if WBA
    batch_size = (
        cfg.BATCH_SIZE // 2
        if cfg.IN_BATCH_AUGMENT and split == "train"
        else cfg.BATCH_SIZE
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=WORKERS,
        pin_memory=False,
        drop_last=shuffle,
    )
    return dataset, data_loader
