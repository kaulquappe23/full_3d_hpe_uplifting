import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from dataset.fit3d.keypoint_order import Fit3DOrder26P
from dataset.fit3d.preparation.smplx_helper import SMPLXHelper
from dataset.fit3d.preparation.utils import read_meta, read_data, get_pathes, project_3d_to_2d
from paths import FIT3D_PROCESSED_ROOT, FIT3D_DIR


def get_rel_joints(joints: np.ndarray) -> np.ndarray:
    body = joints[Fit3DOrder26P._body_smplx, ...]
    fingers = joints[
        Fit3DOrder26P._fingers_smplx, ...
    ]  # left & right (thumb + pinky) fingers
    sjoints = np.concatenate((body, fingers), axis=0)
    return sjoints


def collect_j_camera(smplx_model, cam_params: dict) -> tuple[np.ndarray, np.ndarray]:
    j3d_in_camera = smplx_model.joints
    num_fr, num_j, dims = j3d_in_camera.shape
    j2d_out = np.empty(shape=(num_fr, Fit3DOrder26P._num_joints, 2))
    j3d_out = np.empty(shape=(num_fr, Fit3DOrder26P._num_joints, dims))
    for fr_id, joints in enumerate(j3d_in_camera):
        joints = np.array(joints)
        j3d_out[fr_id] = get_rel_joints(joints)

        j2d = project_3d_to_2d(
            joints, cam_params["intrinsics_w_distortion"], "w_distortion"
        )
        j2d_out[fr_id] = get_rel_joints(j2d)
    return j3d_out, j2d_out


def collect_rot_camera(smplx_model) -> np.ndarray:
    global_orient = smplx_model.global_orient
    body_pose = smplx_model.body_pose
    left_hand_pose = smplx_model.left_hand_pose[:, Fit3DOrder26P._fingers_pose]
    right_hand_pose = smplx_model.right_hand_pose[:, Fit3DOrder26P._fingers_pose]

    hand_pose = np.concatenate(
        [
            left_hand_pose[:, 0, np.newaxis],
            right_hand_pose[:, 0, np.newaxis],
            left_hand_pose[:, 1, np.newaxis],
            right_hand_pose[:, 1, np.newaxis],
        ],
        axis=1,
    )
    rot3d_out = np.concatenate([global_orient, body_pose, hand_pose], axis=1)

    num_fr, _, dim1, dim2 = global_orient.shape
    assert rot3d_out.shape == (num_fr, Fit3DOrder26P._num_joints, dim1, dim2), "Pose shape mismatch"

    return rot3d_out


def collect_j_world(smplx_model) -> np.ndarray:
    j3d_world = smplx_model.joints
    num_fr, num_j, dims = j3d_world.shape
    j3d_out = np.empty(shape=(num_fr, Fit3DOrder26P._num_joints, dims))
    for fr_id, joints in enumerate(j3d_world):
        joints = np.array(joints)
        j3d_out[fr_id] = get_rel_joints(joints)
    return j3d_out


def collect_betas(smplx_model) -> np.ndarray:
    betas = smplx_model.betas
    return betas

def format_camera(cam_params: dict, cam_id: str) -> dict:
    intrinsics = cam_params["intrinsics_w_distortion"]
    extrinsics = cam_params["extrinsics"]
    camera_dict = dict(
        id=cam_id,
        R=extrinsics["R"],
        T=extrinsics["T"],
        fx=intrinsics["f"][0][0],
        fy=intrinsics["f"][0][1],
        cx=intrinsics["c"][0][0],
        cy=intrinsics["c"][0][1],
        k=intrinsics["k"],
        p=intrinsics["p"],
    )
    return camera_dict


def create_vp3d(out_dir: str) -> None:
    cam_names, subjects, actions = read_meta(os.path.join(FIT3D_DIR, "fit3d_info.json"))
    out_p_2d_npz = os.path.join(out_dir, "fit3d_2d_gt.npz")
    out_p_3d_cam = os.path.join(out_dir, "fit3d_3d_camera.npz")
    out_p_3d_world = os.path.join(out_dir, "fit3d_3d_world.npz")
    out_p_rot3d_cam = os.path.join(out_dir, "fit3d_rot3d_camera.npz")
    out_cam_meta = os.path.join(out_dir, "fit3d_cameras_meta.npz")
    out_betas = os.path.join(out_dir, "fit3d_betas.npz")

    kps_2d = {}
    rots_3d_cam = {}
    kps_3d_cam = {}
    kps_3d_world = {}
    cam_meta = {}
    betas = {}

    print(datetime.now())
    for ids, sbj in enumerate(subjects):
        kps_2d[sbj] = {}
        kps_3d_cam[sbj] = {}
        kps_3d_world[sbj] = {}
        rots_3d_cam[sbj] = {}
        cam_meta[sbj] = {}
        betas[sbj] = {}

        for act in actions[sbj]:
            kps_2d[sbj][act] = []
            kps_3d_cam[sbj][act] = []
            rots_3d_cam[sbj][act] = []
            cam_meta[sbj][act] = []
            betas[sbj][act] = []

            tick = time.time()
            for idc, cam in enumerate(cam_names):
                pathes = get_pathes(FIT3D_DIR, "train", sbj, act, cam)
                frames, cam_params, _, smplx_params = read_data(*pathes)

                smplx_helper = SMPLXHelper()
                camera_smplx_params = smplx_helper.get_camera_smplx_params(smplx_params, cam_params)
                # camera_smplx_params["return_full_pose"] = True
                world_smplx_params = smplx_helper.get_world_smplx_params(smplx_params)
                # world_smplx_params["return_full_pose"] = True
                cam_model = smplx_helper.model(**camera_smplx_params)
                world_model = smplx_helper.model(**world_smplx_params)

                j3d_cam, j2d_cam = collect_j_camera(cam_model, cam_params)
                j3d_world = collect_j_world(world_model)
                rot_cam = collect_rot_camera(cam_model)
                camera_dict = format_camera(cam_params, cam)
                wbetas = collect_betas(world_model)
                cbetas = collect_betas(cam_model)
                assert (wbetas == cbetas).all(), "Betas mismatch world & camera"

                kps_2d[sbj][act].append(j2d_cam.astype(np.float16))
                kps_3d_cam[sbj][act].append(j3d_cam.astype(np.float32))
                rots_3d_cam[sbj][act].append(rot_cam.astype(np.float32))
                cam_meta[sbj][act].append(camera_dict)
                betas[sbj][act].append(wbetas.numpy().astype(np.float32))

                if act not in kps_3d_world[sbj].keys():
                    kps_3d_world[sbj][act] = j3d_world
                print(f"Done {sbj} {act} {idc}")
            tock = time.time()
            step_duration = tock - tick
            print(f"Action duration {step_duration}")

        np.savez_compressed(out_p_2d_npz, positions_2d=kps_2d)
        np.savez_compressed(out_p_3d_cam, positions_3d=kps_3d_cam)
        np.savez_compressed(out_p_rot3d_cam, rotations_3d=rots_3d_cam)
        np.savez_compressed(out_p_3d_world, positions_3d=kps_3d_world)
        np.savez_compressed(out_cam_meta, cameras=cam_meta)
        np.savez_compressed(out_betas, betas=betas)
    print(datetime.now())



if __name__ == "__main__":
    processed_dir = FIT3D_PROCESSED_ROOT
    os.makedirs(processed_dir, exist_ok=True)
    create_vp3d(processed_dir)
