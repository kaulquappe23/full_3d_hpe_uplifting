import torch

from model import UpliftUpsampleConfig, UpliftPoseConfig, SMPLX_Layer

smplx = SMPLX_Layer()
smplx_full = SMPLX_Layer(full_seq=True)


def joint_loss(
    preds: torch.Tensor,
    gt: torch.Tensor,
    pred_central: torch.Tensor,
    gt_central: torch.Tensor,
    model_cfg: UpliftUpsampleConfig,
    bs: int,
) -> torch.Tensor:
    seq_len = model_cfg.SEQUENCE_LENGTH
    loss_w_centr = model_cfg.LOSS_WEIGHT_CENTER
    loss_w_seq = model_cfg.LOSS_WEIGHT_SEQUENCE
    num_kpts = model_cfg.NUM_KEYPOINTS

    # central_loss is: (B, K)
    central_loss = torch.linalg.norm(pred_central - gt_central, dim=-1)
    # Aggregate loss over keypoints and batch
    central_loss = torch.sum(central_loss) / (bs * num_kpts)

    if model_cfg.TEMPORAL_TRANSFORMER_BLOCKS > 0:
        # sequence_loss is: (B, N, K)
        seq_loss = torch.linalg.norm(preds - gt, dim=-1)
        # Aggregate loss over keypoints, sequence and batch
        seq_loss = torch.sum(seq_loss) / (bs * seq_len * num_kpts)
        loss = (loss_w_centr * central_loss) + (loss_w_seq * seq_loss)
    else:
        # Fallback without temporal transformer blocks: disable sequence loss
        loss = (loss_w_centr + loss_w_seq) * central_loss
    return loss


def bone_loss(
    preds: torch.Tensor,
    gt: torch.Tensor,
    pred_central: torch.Tensor,
    gt_central: torch.Tensor,
    model_cfg: UpliftUpsampleConfig,
    bs: int,
) -> torch.Tensor:
    seq_len = model_cfg.SEQUENCE_LENGTH
    loss_w_centr = model_cfg.LOSS_WEIGHT_CENTER
    loss_w_seq = model_cfg.LOSS_WEIGHT_SEQUENCE
    num_kpts = model_cfg.NUM_KEYPOINTS
    bones = model_cfg.BONES
    bones_w = model_cfg.BONES_LOSS_WEIGHT
    n_bones = len(bones)

    device = pred_central.device
    def restruct_bones(bones, idx):
        b = torch.Tensor([b[idx] for b in bones]).int()
        b1, b2 = b[:, 0], b[:, 1]
        # b1, b2 = torch.zeros((1, num_kpts, 1)), torch.zeros((1, num_kpts, 1))
        # b1[:, b[0]], b2[:, b[1]] = 1, 1
        return b1.to(device), b2.to(device)

    l_b1, l_b2 = restruct_bones(bones, 0)
    r_b1, r_b2 = restruct_bones(bones, 1)

    # central_loss is: (B, K)
    centr_l = pred_central[:, l_b1] - pred_central[:, l_b2]
    centr_l = torch.linalg.norm(centr_l, dim=-1)

    centr_r = pred_central[:, r_b1] - pred_central[:, r_b2]
    centr_r = torch.linalg.norm(centr_r, dim=-1)

    central_loss = torch.sum(torch.pow(centr_l - centr_r, 2))
    central_loss /= (bs)

    if model_cfg.TEMPORAL_TRANSFORMER_BLOCKS > 0:
        seq_l = preds[:, :, l_b1] - preds[:, :, l_b2]
        seq_l = torch.linalg.norm(seq_l, dim=-1)

        seq_r = preds[:, :, r_b1] - preds[:, :, r_b2]
        seq_r = torch.linalg.norm(seq_r, dim=-1)

        seq_loss = torch.sum(torch.pow(centr_l - centr_r, 2))
        seq_loss /= (bs * seq_len)

        loss = (loss_w_centr * central_loss) + (loss_w_seq * seq_loss)
    else:
        # Fallback without temporal transformer blocks: disable sequence loss
        loss = (loss_w_centr + loss_w_seq) * central_loss
    return bones_w * loss


def bone_loss_2(
    preds: torch.Tensor,
    gt: torch.Tensor,
    pred_central: torch.Tensor,
    gt_central: torch.Tensor,
    model_cfg: UpliftUpsampleConfig,
    bs: int,
) -> torch.Tensor:
    seq_len = model_cfg.SEQUENCE_LENGTH
    loss_w_centr = model_cfg.LOSS_WEIGHT_CENTER
    loss_w_seq = model_cfg.LOSS_WEIGHT_SEQUENCE
    num_kpts = model_cfg.NUM_KEYPOINTS
    bones = model_cfg.BONES
    bones_w = model_cfg.BONES_LOSS_WEIGHT
    n_bones = len(bones)

    device = pred_central.device
    def restruct_bones(bones, idx):
        b = torch.Tensor([b[idx] for b in bones]).int()
        b1, b2 = b[:, 0], b[:, 1]
        return b1.to(device), b2.to(device)

    l_b1, l_b2 = restruct_bones(bones, 0)
    r_b1, r_b2 = restruct_bones(bones, 1)

    def manh_dist(t1, t2, dim):
        distances = torch.sum(torch.abs(t1 - t2), dim=dim)
        return distances

    l_dist_p = manh_dist(pred_central[:, l_b1],  pred_central[:, l_b2], 2)
    r_dist_p = manh_dist(pred_central[:, r_b1],  pred_central[:, r_b2], 2)
    dist_p = l_dist_p + r_dist_p

    l_dist_gt = manh_dist(gt_central[:, l_b1],  gt_central[:, l_b2], 2)
    r_dist_gt = manh_dist(gt_central[:, r_b1],  gt_central[:, r_b2], 2)
    dist_gt = l_dist_gt + r_dist_gt

    loss = manh_dist(dist_gt, dist_p, 1)

    loss = torch.mean(loss)
    return loss


def joint_loss_smplx(
    pred_rots: torch.Tensor,
    gt_kpts3d: torch.Tensor,
    pred_c_rots: torch.Tensor,
    gt_c_kpts3d: torch.Tensor,
    model_cfg: UpliftPoseConfig,
    bs: int,
    betas: torch.Tensor,
) -> torch.Tensor:
    assert pred_rots.device == betas.device
    mid_frame = model_cfg.SEQUENCE_LENGTH // 2

    c_betas = betas[:, mid_frame]
    pred_c_kpts3d = smplx(pred_c_rots, c_betas, None)

    if model_cfg.TEMPORAL_TRANSFORMER_BLOCKS > 0:
        pred_kpts3d = smplx_full(pred_rots, betas, None)

    loss = joint_loss(pred_kpts3d, gt_kpts3d, pred_c_kpts3d, gt_c_kpts3d, model_cfg, bs)
    return loss
