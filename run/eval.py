import os

from dataset.fit3d.dataset_generator import create_fit3d_datasets
from dataset.fit3d.keypoint_order import Fit3DOrder26P, SMPLX37Order
from dataset.fit3d.preparation.utils import read_meta
from paths import LOG_DIR, FIT3D_DIR
from run.metric_calculation import frame_wise_eval, frame_wise_rotations
from run.train_utils import val_step
from utils.rotation_conversions import quaternion_to_axis_angle, matrix_to_axis_angle

import random
import logging
import argparse
from glob import glob

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from omegaconf import OmegaConf

from model import UpliftPoseConfig, UpliftUpsampleConfig
from utils.rotation import RotType

# import eval
from model import build_model, SMPLX_Layer
from dataset.fit3d import splits
import utils.metrics as m

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def collect_predictions(model, loader, cfg) :
    model.eval()

    root_kpt = cfg.ROOT_KEYTPOINT
    mid_index = cfg.SEQUENCE_LENGTH // 2
    val_batches = len(loader)

    gt_kpts3d, all_pred_kpts3d = [], []
    gt_rots, all_pred_rots = [], []
    gt_subjects, gt_actions, gt_cameras = [], [], []
    gt_betas, gt_transl = [], []

    examples = 0
    val_loss = 0
    for b_i, data in enumerate(loader):
        rots, betas = None, None
        if len(data) > 8:
            kpts3d, kpts2d, rots, _, cam_ids, \
                subjects, actions, _, stride_msk, betas = data
        else:
            kpts3d, kpts2d, _, cam_ids, subjects, actions, _, stride_msk = data

        (pred_kpts3d, pred_rots), v_loss = val_step(
            model,
            kpts2d=kpts2d,
            kpts3d=kpts3d,
            rots=rots,
            stride_msk=stride_msk,
            cfg=cfg,
            betas=betas,
        )
        val_loss += v_loss.item()

        # # Only collect as many examples as needed
        # exs_num = min(cfg.training.batch_size, val_examples - examples)
        exs_num = cfg.BATCH_SIZE

        # Perform root-shift right before metric calculation
        kpts3d = kpts3d - kpts3d[:, :, root_kpt:root_kpt + 1, :]
        c_kpts3d = kpts3d[:, mid_index]
        gt_kpts3d.extend(c_kpts3d[:exs_num].numpy())
        all_pred_kpts3d.extend(pred_kpts3d[:exs_num].cpu().numpy())

        gt_subjects.extend(subjects[:exs_num].numpy())
        gt_actions.extend(actions[:exs_num].numpy())
        gt_cameras.extend(np.array(cam_ids[1])[:exs_num])

        if rots is not None:
            c_rots = rots[:, mid_index]
            gt_rots.extend(c_rots[:exs_num].numpy())

        if pred_rots is not None:
            all_pred_rots.extend(pred_rots[:exs_num].cpu().numpy())

        if betas is not None:
            gt_betas.extend(betas[:, mid_index])

        examples += exs_num
        if b_i % 100 == 0:
            logging.info(f"{b_i+1}/{val_batches} evaluated..")

    gt_kpts3d = np.stack(gt_kpts3d, axis=0).astype(np.float64)
    all_pred_kpts3d = np.stack(all_pred_kpts3d, axis=0).astype(np.float64)
    assert gt_kpts3d.shape == all_pred_kpts3d.shape

    # add validity flag for validation
    gt_kpts3d = np.concatenate(
        [gt_kpts3d, np.ones(gt_kpts3d.shape[:-1] + (1,))],
        axis=-1,
    )

    gt_subjects = np.stack(gt_subjects, axis=0)
    gt_actions = np.stack(gt_actions, axis=0)
    gt_cameras = np.stack(gt_cameras, axis=0)
    assert gt_subjects.shape == gt_kpts3d.shape[:1]
    assert gt_actions.shape == gt_kpts3d.shape[:1]

    if len(gt_rots) > 0 and len(all_pred_rots) > 0:
        gt_rots = np.stack(gt_rots, axis=0).astype(np.float32)
        all_pred_rots = np.stack(all_pred_rots, axis=0).astype(np.float32)

        all_pred_rots = transform_rotations(cfg, all_pred_rots, True)
        gt_rots = transform_rotations(cfg, gt_rots)

        assert gt_rots.shape == all_pred_rots.shape
        if not cfg.ESTIMATE_HANDS:
            all_pred_rots[:, cfg.KEYPOINT_ORDER.hands()] = 0

    gt_betas = np.stack(gt_betas, axis=0) if len(gt_betas) > 0 else None

    out = {
        "pred_rots": all_pred_rots,
        "gt_rots": gt_rots,
        "pred_kpts3d": all_pred_kpts3d,
        "gt_kpts3d": gt_kpts3d,
        "betas": gt_betas,
        "subjects": gt_subjects,
        "actions": gt_actions,
        "cam_ids": gt_cameras,
    }
    return out


def regress_joints(rots, betas, bs, smplx_layer):
    out = []
    num_dp = rots.shape[0]
    for i in range(0, num_dp, bs):
        rots_b = rots[i:i + bs]
        betas_b = betas[i:i + bs]

        kpts3d = smplx_layer(rots_b, betas_b, None)
        out.extend(kpts3d)

        if i//bs % 100 == 0:
            logging.info(f"{i//bs}/{num_dp//bs} regressed...")
    out = torch.stack(out, axis=0).detach().cpu().numpy()
    return out


def load_model(cfg, weights):
    model = build_model(cfg)
    logging.info(f"Loading weights from {weights}")
    weights = torch.load(weights, map_location=cfg.DEVICE, weights_only=False)["model"].state_dict()
    model.load_state_dict(weights)
    model = model.to(cfg.DEVICE)
    return model


def transform_rotations(cfg, rots, ortho=False):
    rots = torch.from_numpy(rots).clone()
    if cfg.ROT_REP == RotType.ROT_MAT:
        rots = matrix_to_axis_angle(rots)
    elif cfg.ROT_REP == RotType.QUATERNION:
        rots = quaternion_to_axis_angle(rots)

    return rots.cpu().numpy()


def save_predictions(data, dest_dir):
    out_gt_rots, out_pred_rots, out_betas, out_gt_kpts, out_pred_kpts = data

    os.makedirs(dest_dir, exist_ok=True)

    sbjs = "_".join(sorted(list(out_gt_kpts.keys())))

    out_gt_fp = os.path.join(dest_dir, f"fit3d_kpts3d_{sbjs}_gt.npz")
    np.savez_compressed(out_gt_fp, kpts3d=out_gt_kpts)

    out_pred_fp = os.path.join(dest_dir, f"fit3d_kpts3d_{sbjs}_pred.npz")
    np.savez_compressed(out_pred_fp, kpts3d=out_pred_kpts)

    if len(out_gt_rots) > 0 and len(out_pred_rots) > 0:
        out_gt_fp = os.path.join(dest_dir,  f"fit3d_rot3d_{sbjs}_gt.npz")
        np.savez_compressed(out_gt_fp, rotations_3d=out_gt_rots)

        out_pred_fp = os.path.join(dest_dir, f"fit3d_rot3d_{sbjs}_pred.npz")
        np.savez_compressed(out_pred_fp, rotations_3d=out_pred_rots)

        out_betas_fp = os.path.join(dest_dir, f"fit3d_betas_{sbjs}.npz")
        np.savez_compressed(out_betas_fp, betas=out_betas)


def format_results(results):
    cam_names, _, _ = read_meta(os.path.join(FIT3D_DIR, "fit3d_info.json"))
    subject_dict = {i: name for i, name in enumerate(splits.all_subjects)}
    action_dict = {i: name for i, name in enumerate(splits.actions)}

    out_gt_rots = {}
    out_pred_rots = {}
    out_betas = {}
    out_gt_kpts = {}
    out_pred_kpts = {}

    subjects, actions = results["subjects"], results["actions"]
    cameras = results["cam_ids"]
    betas = results["betas"]
    pred_rots, gt_rots = results["pred_rots"], results["gt_rots"]
    pred_kpts, gt_kpts = results["pred_kpts3d"], results["gt_kpts3d"]

    rots_available = len(pred_rots) > 0 and len(gt_rots) > 0

    for s in np.unique(subjects):
        sbj = subject_dict[s]

        if rots_available:
            out_pred_rots[sbj] = {}
            out_gt_rots[sbj] = {}
            out_betas[sbj] = {}

        out_gt_kpts[sbj] = {}
        out_pred_kpts[sbj] = {}

        for a in np.unique(actions):
            act = action_dict[a]

            if rots_available:
                out_pred_rots[sbj][act] = []
                out_gt_rots[sbj][act] = []
                out_betas[sbj][act] = []

            out_gt_kpts[sbj][act] = []
            out_pred_kpts[sbj][act] = []
            for cam in cam_names:
                msk = (subjects == s) & (actions == a) & (cam == cameras)

                if rots_available:
                    out_pred_rots[sbj][act].append(pred_rots[msk])
                    out_gt_rots[sbj][act].append(gt_rots[msk])
                    out_betas[sbj][act].append(betas[msk])

                out_gt_kpts[sbj][act].append(gt_kpts[msk])
                out_pred_kpts[sbj][act].append(pred_kpts[msk])

    return out_gt_rots, out_pred_rots, out_betas, out_gt_kpts, out_pred_kpts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", type=str)
    parser.add_argument("-w", "--weights", type=str)
    parser.add_argument("-", "--log", type=str, required=False)
    parser.add_argument("--per_joint", action="store_true", help="Output metric scores per joint")
    parser.add_argument("--kpts_37", action="store_true", help="Evaluate on 37 keypoints from A2B paper")
    parser.add_argument("--val", action="store_true", help="Evaluate on test and val set, otherwise test set only")
    args = parser.parse_args()

    if not args.log:
        args.log = os.path.dirname(args.weights)


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                mode="w",
                filename=os.path.join(args.log, "eval_after_training.log"))
    ])
    logging.info(f"Evaluation joints {args.weights}...")

    cfg = UpliftPoseConfig(config_file=args.cfg)

    # make enum from string
    if hasattr(cfg, "ROT_REP") and cfg.ROT_REP:
        cfg.OUT_DIM = RotType[cfg.ROT_REP].value
        cfg.ROT_REP = RotType[cfg.ROT_REP]

    cfg.KEYPOINT_ORDER = eval(cfg.KEYPOINT_ORDER)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.DEVICE = device

    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    model = load_model(cfg, args.weights)
    model.to(cfg.DEVICE)
    model.eval()

    smplx_layer = SMPLX_Layer(kpts_all=True, keypoint_order=cfg.KEYPOINT_ORDER)
    smplx_layer = smplx_layer.to(device)

    subsets = ["test"]
    if args.val:
        subsets.append("val")
    for subset in subsets:
        _, _, ds, loader, batches = (
            create_fit3d_datasets(
                cfg=cfg,
                train_subset=None,
                val_subset=subset,
            )
        )
        logging.info(f"Collecting predictions on {len(ds)} examples - {len(loader)} batches")
        results = collect_predictions(model, loader, cfg)

        logging.info(f"Running evaluation on {subset} ..")
        gt_kpts3d = results["gt_kpts3d"]
        gt_kpts3d_all, pred_kpts3d_all = None, None
        gt_rots, pred_rots = None, None

        betas = torch.from_numpy(results["betas"]).to(device)
        pred_rots = torch.from_numpy(results["pred_rots"]).to(device)
        gt_rots = torch.from_numpy(results["gt_rots"]).to(device)
        bs = cfg.BATCH_SIZE

        logging.info(f"Regressing GT-kpts through SMPL-X")
        gt_kpts3d_all = regress_joints(gt_rots, betas, bs, smplx_layer)

        logging.info(f"Regressing predicted kpts through SMPL-X")
        pred_kpts3d_all = regress_joints(pred_rots, betas, bs, smplx_layer)

        pred_kpts3d = pred_kpts3d_all[:, Fit3DOrder26P.smplx_joints()].copy()

        # Overall MPJPE, PA-MPJPE
        frame_results = frame_wise_eval(
            pred_3d=pred_kpts3d,
            gt_3d=gt_kpts3d,
            root_index=cfg.ROOT_KEYTPOINT,
        )
        logging.info(f"Results Fit3DOrder26P...")
        for met, err in frame_results.items():
            logging.info(f"{met.upper()} - {err:.2f}mm")

        if args.per_joint:
            # MPJPE per frame and per joint -> average per joint
            frame_mpjpe = m.mpjpe(
                pred=pred_kpts3d,
                gt=gt_kpts3d,
                root_index=cfg.ROOT_KEYTPOINT,
                normalize=False,
            ) * 1000.0
            frame_mpjpe = np.mean(frame_mpjpe, axis=0)
            logging.info("MPJPE per Joint - Fit3DOrder26P...")
            for err, kp_name in zip(frame_mpjpe, Fit3DOrder26P._point_names):
                logging.info(f"{kp_name} - {err:.2f}mm")
        logging.info("\n")

        # eval SMPL-X pose vectors
        if pred_rots is not None:
            # overall MPJAE
            rotation_results = frame_wise_rotations(
                pred_rot=pred_rots,
                gt_rot=gt_rots,
                root_index=cfg.ROOT_KEYTPOINT,
            )
            logging.info(f"MPJAE - {rotation_results['mpjae']:.3f}\xB0")
            # MPJAE without fingers
            mpjae = m.joint_angle_error(pred_rots, gt_rots, average=False)
            mpjae_without_fingers = np.mean(mpjae[:, :-4])
            logging.info(f"MPJAE without Fingers - {mpjae_without_fingers:.2f}\xB0")

            if args.per_joint:
                logging.info(f"MPJAE per Joint:")
                # MPJAE per joint
                mpjae = np.mean(mpjae, axis=0)
                assert mpjae.shape[0] == Fit3DOrder26P._num_joints
                for err, kp_name in zip(mpjae, Fit3DOrder26P._point_names):
                    logging.info(f"{kp_name} - {err:.2f}\xB0")
            logging.info("\n")

        # eval for SMPLX37Order
        if pred_kpts3d_all is not None and args.kpts_37:
            rel_joints = SMPLX37Order.from_SMPLX_order()
            gt_kpts3d = gt_kpts3d_all[:, rel_joints].copy()
            pred_kpts3d = pred_kpts3d_all[:, rel_joints].copy()

            gt_kpts3d = np.concatenate(
                [gt_kpts3d, np.ones(gt_kpts3d.shape[:-1] + (1,))],
                axis=-1,
            )

            frame_results = frame_wise_eval(
                pred_3d=pred_kpts3d,
                gt_3d=gt_kpts3d,
                root_index=cfg.ROOT_KEYTPOINT,
            )
            logging.info(f"Results SMPLX37Order...")
            for met, err in frame_results.items():
                logging.info(f"{met.upper()} - {err:.2f}")

            frame_mpjpe = m.mpjpe(
                pred=pred_kpts3d,
                gt=gt_kpts3d,
                root_index=cfg.ROOT_KEYTPOINT,
                normalize=False,
            ) * 1000.0
            frame_mpjpe = np.mean(frame_mpjpe, axis=0)
            logging.info("MPJPE per Joint - SMPLX37Order...")
            for err, kp_name in zip(frame_mpjpe, SMPLX37Order.names):
                logging.info(f"{kp_name} - {err:.2f}")

        formated_res = format_results(results)
        save_predictions(formated_res, os.path.join(args.log, f"{subset}_res"))
        logging.info(f"saved results to {os.path.join(args.log, f"{subset}_res")}")
