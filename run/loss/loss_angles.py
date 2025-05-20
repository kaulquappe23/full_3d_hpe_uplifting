import torch
from torch.nn.functional import mse_loss

from model import UpliftUpsampleConfig
from utils.rotation import RotType, symmetric_orthogonalization, transform_to_matrix
from utils.rotation import angle_from_axis_angle, angle_from_quarterion, angle_from_rot_matrix


def geodesic_loss(
    preds: torch.Tensor,
    gt: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    eps = 1e-7
    # Compute the relative rotation matrices
    R_rel = torch.matmul(preds, gt.transpose(-1, -2))

    # Compute the trace of the relative rotation matrices
    trace_R_rel = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)

    # Compute the angles theta in radians
    dists = torch.acos(torch.clamp((trace_R_rel - 1) / 2, -1 + eps, 1 - eps))

    if reduction == "mean":
        return dists.mean()
    elif reduction == "sum":
        return dists.sum()

    return dists


def angle_loss(
    preds: torch.Tensor,
    gt: torch.Tensor,
    pred_central: torch.Tensor,
    gt_central: torch.Tensor,
    model_cfg: UpliftUpsampleConfig,
) -> torch.Tensor:
    LOSSES = {
        "mse": mse_loss,
        "geodesic": geodesic_loss,
    }
    loss_func = LOSSES[model_cfg.LOSS_ANGLES]
    loss_w_centr = model_cfg.LOSS_WEIGHT_CENTER
    loss_w_seq = model_cfg.LOSS_WEIGHT_SEQUENCE
    rot_type = model_cfg.ROT_REP

    if model_cfg.LOSS_ANGLES in ["geodesic"] and rot_type != RotType.ROT_MAT:
        pred_central = transform_to_matrix(pred_central, rot_type)
        preds = transform_to_matrix(preds, rot_type)
        gt_central = transform_to_matrix(gt_central, rot_type)
        gt = transform_to_matrix(gt, rot_type)

    loss_central = loss_func(pred_central, gt_central)
    loss_all = loss_func(preds, gt)
    loss = (loss_w_centr * loss_central) + (loss_w_seq * loss_all)
    return loss