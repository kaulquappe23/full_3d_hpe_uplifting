# -*- coding: utf-8 -*-
"""
Created on 01 Jun 2021, 16:35

@author: waterplant365
"""


def format_time(seconds):
    """
    Utiliy to format time (copied from keras code).
    :param seconds: time duration in seconds.
    :return: formatted time string
    """
    if seconds > 3600:
        time_string = '%d:%02d:%02d' % (seconds // 3600,
                                       (seconds % 3600) // 60, seconds % 60)
    elif seconds > 60:
        time_string = '%d:%02d' % (seconds // 60, seconds % 60)
    else:
        time_string = '%ds' % seconds
    return time_string


def generate_run_name(cfg, cur_time: str) -> str:
    angles_w, bones_w, flip, weights, comment, rot_t = [None] * 6
    smplx, mesh_loss, vposer = [None] * 3
    if cfg.LOSS_ANGLES_ENABLED:
        angles_w = f"{cfg.LOSS_ANGLES}-{cfg.ANGLES_LOSS_WEIGHT}"
    if cfg.SMPLX_LAYER[0]:
        smplx = "full"
    if cfg.SMPLX_LAYER[1]:
        smplx = smplx + "-" + "central" if smplx else "central"
    smplx = "smplx-" + smplx if smplx else None
    rot_t = cfg.ROT_REP.name.lower().replace("_", "-")

    if cfg.LOSS_MESH_ENABLED:
        mesh_loss = "mesh-loss-joints"

    if cfg.VPOSER:
        vposer = "vposer"

    if cfg.BONES_LOSS_ENABLED:
        bones_w = f"bones-{cfg.BONES_LOSS_WEIGHT}"

    if cfg.IN_BATCH_AUGMENT:
        flip = "flip-train"

    if cfg.PRETRAINED_WEIGHTS:
        weights = cfg.PRETRAINED_WEIGHTS.split("/")[-1].split(".")[0]
        weights = weights.replace("_", "-")

    ds = cfg.DATASET_NAME
    epochs = cfg.EPOCHS

    params = ["run", cur_time, ds, weights, flip, bones_w,
            rot_t, angles_w, smplx, mesh_loss, vposer, epochs]
    params = [str(x) for x in params if x]
    run_name = f"_".join(params)
    return run_name