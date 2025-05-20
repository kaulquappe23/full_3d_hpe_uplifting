"""
Utils for fit3d dataset.
"""

import os
import json
import numpy as np
import cv2


def j3d_to_camview(points3d: np.ndarray, cam_params: dict) -> np.ndarray:
    ext_t = cam_params["extrinsics"]["T"]
    ext_R_t = cam_params["extrinsics"]["R"]
    j3d_in_c = np.matmul(np.array(points3d) - ext_t, np.transpose(ext_R_t))
    return j3d_in_c


def project_3d_to_2d(
    points3d: np.ndarray, intrinsics: dict, intrinsics_type: str
) -> np.ndarray:
    if intrinsics_type == "w_distortion":
        p = intrinsics["p"][:, [1, 0]]
        x = points3d[:, :2] / points3d[:, 2:3]
        r2 = np.sum(x**2, axis=1)
        radial = 1 + np.transpose(
            np.matmul(intrinsics["k"], np.array([r2, r2**2, r2**3]))
        )
        tan = np.matmul(x, np.transpose(p))
        xx = x * (tan + radial) + r2[:, np.newaxis] * p
        proj = intrinsics["f"] * xx + intrinsics["c"]
    elif intrinsics_type == "wo_distortion":
        xx = points3d[:, :2] / points3d[:, 2:3]
        proj = intrinsics["f"] * xx + intrinsics["c"]
    return proj


def read_meta(meta_path: str) -> tuple:
    with open(meta_path, "r") as f:
        meta = json.load(f)
    cam_names = meta["all_camera_names"]
    subjects = meta["train_subj_names"]
    subj_to_act = meta["subj_to_act"]
    return cam_names, subjects, subj_to_act


def read_video(vid_path: str) -> np.ndarray:
    frames = []
    cap = cv2.VideoCapture(vid_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.array(frames)
    return frames


def read_cam_params(cam_path: str) -> dict:
    with open(cam_path, "r") as f:
        cam_params = json.load(f)
        for key1 in cam_params:
            for key2 in cam_params[key1]:
                cam_params[key1][key2] = np.array(cam_params[key1][key2])
    return cam_params


def read_j3d(j3d_path: str) -> np.ndarray:
    with open(j3d_path, "r") as f:
        j3ds = np.array(json.load(f)["joints3d_25"])
    return j3ds


def read_smplx(smplx_path: str) -> dict:
    with open(smplx_path, "r") as f:
        smplx_params = json.load(f)
    return smplx_params


def read_annotations(ann_path: str, action_name: str) -> list:
    with open(ann_path, "r") as f:
        annotations = json.load(f)
    return annotations[action_name]


def read_data(
    vid_path: str,
    cam_path: str,
    j3d_path: str,
    smplx_path: str,
) -> tuple:
    cam_params = read_cam_params(cam_path)
    j3ds = read_j3d(j3d_path)
    smplx_params = read_smplx(smplx_path)

    seq_len = j3ds.shape[-3]
    frames = read_video(vid_path)[:seq_len]

    return frames, cam_params, j3ds, smplx_params


def get_pathes(root: str, subset: str, subject: str, action: str, camera: str) -> tuple:
    sbj_root = os.path.join(root, subset, subject)

    vid_path = os.path.join(sbj_root, "videos", camera, f"{action}.mp4")
    cam_path = os.path.join(sbj_root, "camera_parameters", camera, f"{action}.json")
    j3d_path = os.path.join(sbj_root, "joints3d_25", f"{action}.json")
    smplx_path = os.path.join(sbj_root, "smplx", f"{action}.json")
    return vid_path, cam_path, j3d_path, smplx_path


def save_images(frames: np.ndarray, dir_fp: str) -> list:
    os.makedirs(dir_fp, exist_ok=True)
    fr_pathes = []
    for idx, fr in enumerate(frames):
        out_fp = os.path.join(dir_fp, f"{idx:05d}.png")
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        cv2.imwrite(out_fp, fr)
        # save pathes for dbs
        fr_pathes.append(out_fp)
    return fr_pathes