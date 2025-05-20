import logging
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter

from run.loss.loss_angles import angle_loss
from run.loss.loss_joints import joint_loss_smplx, joint_loss, bone_loss_2
from run.metric_calculation import frame_wise_rotations, frame_wise_eval
from utils import time_formatting
from utils.config import Config
from utils.metric_history import MetricHistory


def train_step(model, optimizer, data, train_ds, cfg):
    device = cfg.DEVICE
    batch_augment = cfg.IN_BATCH_AUGMENT
    root_kpt = cfg.ROOT_KEYTPOINT
    mid_index = cfg.SEQUENCE_LENGTH // 2
    bs = cfg.BATCH_SIZE

    rots, betas = None, None
    if len(data) > 8:
        # if trained with rotations
        kpts3d, kpts2d, rots, _, cams, _, _, _, stride_msk, betas = data
        rots = rots.to(device)
        c_rots = rots[:, mid_index]
    else:
        # if UU is trained
        kpts3d, kpts2d, _, cams, _, _, _, stride_msk = data
    if isinstance(cams, list):
        cams = cams[0]

    kpts3d = kpts3d.to(device)

    if batch_augment:
        if len(kpts3d.shape) == 5:  # in batch augmentation already performed
            # disabled, legacy from UU
            flip_3d = kpts3d[:, 1]
            kpts3d = kpts3d[:, 0]
            flip_2d = kpts2d[:, 1]
            kpts2d = kpts2d[:, 0]
            flip_cam = cams.clone()
        else:
            # flips during WBA
            flip_3d, flip_2d, flip_cam = train_ds.flip_batch(kpts3d, kpts2d, cams)

        # compose mini-batches
        kpts3d = torch.cat([kpts3d, flip_3d], dim=0)
        kpts2d = torch.cat([kpts2d, flip_2d], dim=0)
        stride_msk = torch.cat([stride_msk, stride_msk], dim=0)
        if isinstance(cams, dict):
            cams["flipped_intrinsic"] = flip_cam["intrinsic"]
            cams["flipped_extrinsic"] = flip_cam["extrinsic"]
        else:
            cams = torch.cat([cams, flip_cam], dim=0)

        if rots is not None:
            flip_rots = train_ds.flip_rotations(rots)
            rots = torch.cat([rots, flip_rots], dim=0)
            c_rots = rots[:, mid_index]

        if betas is not None:
            # betas are not changing
            betas = torch.cat([betas, betas], dim=0)

    # root relative
    kpts3d = kpts3d - kpts3d[:, :, root_kpt:root_kpt+1, :]
    c_kpts3d = kpts3d[:, mid_index]

    model_input = kpts2d.to(device).float()
    if model.has_strided_input:
        masked_kpts2d = kpts2d * stride_msk[:, :, None, None]
        model_input = [masked_kpts2d.to(device).float(), stride_msk.to(device)]

    optimizer.zero_grad()

    if hasattr(cfg, "SMPLX_LAYER") and any(cfg.SMPLX_LAYER):
        # joints are regressed through SMPL_X
        betas = betas.to(device).float()
        output = model(model_input, betas)
    else:
        output = model(model_input)
    full_output, central_output = output

    # unpack predictions
    pred_rots, pred_c_rots = None, None
    if isinstance(full_output, tuple) and isinstance(central_output, tuple):
        pred_kpts3d, pred_c_kpts3d = full_output[0], central_output[0]
        pred_rots, pred_c_rots = full_output[1], central_output[1]
    else:
        pred_kpts3d, pred_c_kpts3d = full_output, central_output

    loss_dict = {}
    if hasattr(cfg, "LOSS_MESH_ENABLED") and cfg.LOSS_MESH_ENABLED:
        # didnt work out
        assert betas is not None and not any(cfg.SMPLX_LAYER)
        assert pred_rots is not None and pred_c_rots is not None
        betas = betas.to(device).float()
        loss = joint_loss_smplx(pred_rots, kpts3d, pred_c_rots, c_kpts3d, cfg, bs, betas)
        loss_dict["mesh"] = loss
    else:
        # joint loss from UU
        loss = joint_loss(pred_kpts3d, kpts3d, pred_c_kpts3d, c_kpts3d, cfg, bs)
        loss_dict["joints"] = loss

    if hasattr(cfg, "LOSS_ANGLES_ENABLED") and cfg.LOSS_ANGLES_ENABLED:
        loss_angle = angle_loss(pred_rots, rots, pred_c_rots, c_rots, cfg)
        if hasattr(cfg, "ANGLES_LOSS_WEIGHT") and not cfg.ANGLES_LOSS_WEIGHT == 1:
            loss = loss * (loss_angle / loss)
        loss += loss_angle
        loss_dict["angles"] = loss_angle

    if hasattr(cfg, "BONES_LOSS_ENABLED") and cfg.BONES_LOSS_ENABLED:
        loss_bones = bone_loss_2(pred_kpts3d, kpts3d, pred_c_kpts3d, c_kpts3d, cfg, bs)
        loss += loss_bones * cfg.BONES_LOSS_WEIGHT
        loss_dict["bones"] = loss_bones

    loss_dict["main"] = loss
    loss.backward()
    optimizer.step()

    return model, optimizer, loss_dict


def update_ema(model, ema_model, decay) -> torch.nn.Module:
    with torch.no_grad():
        for param_s, param_t in zip(model.parameters(), ema_model.parameters()):
            param_t.data = param_t.data * decay + param_s.data * (
                1.0 - decay
            )

        for name, buffer_s in model.named_buffers():
            if "bm_x.parents" in name:
                continue
            buffer_t = ema_model.state_dict()[name]
            buffer_t.data = buffer_t.data * decay + buffer_s.data * (
                1.0 - decay
            )
    return ema_model


def validation(model, loader, ep, cfg, prefix = ""):
    model.eval()
    start = time.time()

    root_kpt = cfg.ROOT_KEYTPOINT
    mid_index = cfg.SEQUENCE_LENGTH // 2
    val_examples = cfg.VALIDATION_EXAMPLES
    device = cfg.DEVICE

    logging.info(f"Running validation on {val_examples} examples")

    gt_kpts3d = []
    gt_rots = []
    all_pred_kpts3d = []
    all_pred_rots = []
    gt_subjects = []
    gt_actions = []
    examples = 0
    val_loss = 0
    for b_i, data in enumerate(loader):
        rots, betas = None, None
        if len(data) > 8:
            # estimation with rotations
            kpts3d, kpts2d, rots, _, _, \
                subjects, actions, _, stride_msk, betas = data
        else:
            # estimation UU
            kpts3d, kpts2d, _, _, subjects, actions, _, stride_msk = data

        preds, v_loss = val_step(
            model,
            kpts2d=kpts2d,
            kpts3d=kpts3d,
            rots=rots,
            stride_msk=stride_msk,
            cfg=cfg,
            betas=betas,
        )
        val_loss += v_loss.item()
        pred_c_kpts3d, pred_c_rots = preds, None
        if isinstance(preds, tuple):
            pred_c_kpts3d, pred_c_rots = preds

        if cfg.EVAL_FLIP:
            flip_pred_kpts_3d = eval_flip_batch(
                model, kpts2d, kpts3d, cfg, cfg, stride_msk)
            pred_c_kpts3d += flip_pred_kpts_3d
            pred_c_kpts3d /= 2.0

        # Only collect as many examples as needed
        exs_num = min(cfg.BATCH_SIZE, val_examples - examples)

        # Perform root-shift right before metric calculation
        kpts3d = kpts3d - kpts3d[:, :, root_kpt:root_kpt + 1, :]
        c_kpts3d = kpts3d[:, mid_index]

        gt_kpts3d.extend(c_kpts3d[:exs_num].numpy())
        gt_subjects.extend(subjects[:exs_num].numpy())
        gt_actions.extend(actions[:exs_num].numpy())

        all_pred_kpts3d.extend(pred_c_kpts3d[:exs_num].cpu().numpy())

        if rots is not None and pred_c_rots is not None:
            c_rots = rots[:, mid_index]
            gt_rots.extend(c_rots[:exs_num])
            all_pred_rots.extend(pred_c_rots[:exs_num])

        examples += exs_num

    gt_kpts3d = np.stack(gt_kpts3d, axis=0).astype(np.float64)
    # Add dummy valid flag
    gt_kpts3d = np.concatenate(
        [gt_kpts3d, np.ones(gt_kpts3d.shape[:-1] + (1,))],
        axis=-1,
    )
    all_pred_kpts3d = np.stack(all_pred_kpts3d, axis=0).astype(np.float64)
    gt_subjects = np.stack(gt_subjects, axis=0)
    gt_actions = np.stack(gt_actions, axis=0)
    assert b_i == (cfg.VALIDATION_BATCHES - 1)

    action_wise_results = None
    frame_results = frame_wise_eval(
        pred_3d=all_pred_kpts3d,
        gt_3d=gt_kpts3d,
        root_index=root_kpt,
    )

    rotation_results = {}
    if len(all_pred_rots) > 0 and len(gt_rots) > 0:
        gt_rots = torch.stack(gt_rots, dim=0).double().to(device)
        all_pred_rots = torch.stack(all_pred_rots, dim=0).double().to(device)
        rotation_results = frame_wise_rotations(
            all_pred_rots, gt_rots, root_kpt
        )

    duration = time.time() - start
    duration_str = time_formatting.format_time(duration)

    logging.info(f"{prefix} - Finished validation in {duration_str}, loss: {val_loss / b_i:.6f}, ")

    results = {
        "ep": ep,
        "loss": val_loss / b_i,
    }
    results = {**results, **frame_results, **rotation_results}
    if action_wise_results is not None:
        for metr in ["mpjpe", "nmpjpe", "pampjpe"]:
            results[f"aw-{metr}"] = action_wise_results[metr]

    return results


def val_step(model, kpts2d, kpts3d, stride_msk, cfg, rots = None, betas = None):
    device = cfg.DEVICE
    bs = cfg.BATCH_SIZE
    root_kpt = cfg.ROOT_KEYTPOINT
    seq_len = cfg.SEQUENCE_LENGTH
    mid_index =  seq_len // 2

    kpts3d = kpts3d.to(device)
    kpts3d = kpts3d - kpts3d[:, :, root_kpt:root_kpt + 1, :]
    c_kpts3d = kpts3d[:, mid_index]
    if rots is not None:
        rots = rots.to(device)
        c_rots = rots[:, mid_index]

    model_input = kpts2d.to(device).float()
    if model.has_strided_input:
        masked_kpts2d = kpts2d * stride_msk[:, :, None, None]
        model_input = [masked_kpts2d.to(device).float(), stride_msk.to(device)]

    with torch.no_grad():
        if hasattr(cfg, "SMPLX_LAYER") and any(cfg.SMPLX_LAYER):
            betas = betas.to(device).float()
            output = model(model_input, betas)
        else:
            output = model(model_input)
        full_output, central_output = output

        pred_rots, pred_c_rots = None, None
        if isinstance(full_output, tuple) and isinstance(central_output, tuple):
            pred_kpts3d, pred_c_kpts3d = full_output[0], central_output[0]
            pred_rots, pred_c_rots = full_output[1], central_output[1]
        else:
            pred_kpts3d, pred_c_kpts3d = full_output, central_output

        if hasattr(cfg, "LOSS_MESH_ENABLED") and cfg.LOSS_MESH_ENABLED:
            assert betas is not None and not any(cfg.SMPLX_LAYER)
            assert pred_rots is not None and pred_c_rots is not None
            betas = betas.to(device).float()
            loss = joint_loss_smplx(pred_rots, kpts3d, pred_c_rots, c_kpts3d, cfg, bs, betas)
        else:
            loss = joint_loss(pred_kpts3d, kpts3d, pred_c_kpts3d, c_kpts3d, cfg, bs)

        if hasattr(cfg, "LOSS_ANGLES_ENABLED") and cfg.LOSS_ANGLES_ENABLED:
            loss_angle = angle_loss(pred_rots, rots, pred_c_rots, c_rots, cfg)
            loss = loss * (loss_angle / loss)
            loss += loss_angle

        if hasattr(cfg, "BONES_LOSS_ENABLED") and cfg.BONES_LOSS_ENABLED:
            loss_bones = bone_loss_2(pred_kpts3d, kpts3d, pred_c_kpts3d, c_kpts3d, cfg, bs)
            loss += loss_bones * cfg.BONES_LOSS_WEIGHT

    return (pred_c_kpts3d, pred_c_rots), loss


def eval_flip_batch(model, kpts2d, kpts3d, cfg, stride_msk) -> torch.Tensor:
    ag_flip_order = cfg.AUGM_FLIP_KEYPOINT_ORDER

    flip_2d = kpts2d
    flip_2d = torch.concat(
        [
            flip_2d[:, :, :, :1] * -1.0,
            flip_2d[:, :, :, 1:],
        ],
        dim=-1,
    )
    flip_2d = flip_2d[:, :, ag_flip_order]

    flip_3d = kpts3d
    flip_3d = torch.concat(
        [
            flip_3d[:, :, :, :1] * -1.0,
            flip_3d[:, :, :, 1:],
        ],
        dim=-1,
    )
    flip_3d = flip_3d[:, :, ag_flip_order]

    (flip_pred_kpts_3d, _), _ = val_step(
        model,
        kpts2d=flip_2d,
        kpts3d=flip_3d,
        stride_msk=stride_msk,
        cfg=cfg,
    )
    flip_pred_kpts_3d = torch.concat(
        [
            flip_pred_kpts_3d[:, :, :1] * -1.0,
            flip_pred_kpts_3d[:, :, 1:],
        ],
        dim=-1,
    )
    flip_pred_kpts_3d = flip_pred_kpts_3d[:, ag_flip_order]

    return flip_pred_kpts_3d


def update_loggers(
    results: dict,
    tb_writer: SummaryWriter,
    metric_hist: Optional[MetricHistory] = None,
    subset: str = "val",
    prefix: str = "",
) -> tuple:
    """Update existing loggers e.g. tensorboard etc
    """
    tb_pref = f"{prefix}/" if len(prefix) > 0 else ""
    mh_pref = f"/{prefix}" if len(prefix) > 0 else ""
    ep = results["ep"]
    for metr, value in results.items():
        if metr == "ep" or not value:
            continue
        metr = metr.upper() if metr != "loss" else metr
        tb_writer.add_scalar(f"{tb_pref}{subset}/{metr}", value, ep)

        if subset == "val":
            logging.info(f"{tb_pref}{metr}: {value:.2f},")
            metric_hist.add_data(f"{metr}{mh_pref}", value, ep)

    return tb_writer, metric_hist


def save_best_weight(
    cfg: Config,
    metric_hist: MetricHistory,
    ep: int,
    ckp_dict: dict,
    prefix: str="",
) -> str:
    mh_pref = f"/{prefix}" if len(prefix) > 0 else ""
    best_metric = cfg.BEST_CHECKPOINT_METRIC

    pre_best_weights_path = ckp_dict["weights"]["pre_best_path"]
    if cfg.EMA_ENABLED and prefix == "ema":
        pre_best_weights_path = ckp_dict["weights"]["pre_best_ema_path"]
    if best_metric is not None and cfg.VALIDATION_NAME is not None:
        # Save best checkpoint as .h5
        best_val, best_ep = metric_hist.best_value(
            f"{best_metric}{mh_pref}"
        )
        if best_ep == ep:
            logging.info(
                f"Saving currently best checkpoint @ epoch={best_ep} "
                f"({best_metric}: {best_val}) as .pth:"
            )
            weightsp = os.path.join(
                cfg.LOG_OUT_DIR,
                f"best_weights_{prefix + '_' if len(prefix) > 0 else ''}{best_ep:04d}.pth",
            )
            torch.save(ckp_dict, weightsp)

            if pre_best_weights_path is not None \
                and os.path.exists(pre_best_weights_path):
                os.remove(pre_best_weights_path)

            pre_best_weights_path = weightsp
    return pre_best_weights_path