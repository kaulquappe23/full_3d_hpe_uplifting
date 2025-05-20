import os

import torch
import numpy as np

from utils.rotation import RotType

from dataset.fit3d.keypoint_order import Fit3DOrder26P

from dataset.skeleton import Skeleton
from dataset.mocap_dataset import MocapDataset
from utils.rotation_conversions import matrix_to_quaternion, matrix_to_axis_angle

RES_W, RES_H = 900, 900
FPS = 50


# legacy from UU, never used
fit3d_skeleton = Skeleton(
    parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13,
             14, 16, 17, 18, 19, 20, 21, 20, 21],
    joints_left=[1, 4, 7, 10, 13, 16, 18, 20, 22, 24,],
    joints_right=[2, 5, 8, 11, 14, 17, 19, 21, 23, 25],
)


class Fit3dDataset(MocapDataset):
    def __init__(
        self,
        kpts3d_path,
        kpts3d_cam_path,
        kpts2d_path,
        camdata_path,
        rots_cam_path,
        betas_path = None,
        rot_type = RotType.ROT_MAT,
    ):
        super().__init__(fps=FPS, skeleton=fit3d_skeleton)
        self._kpts3d_path = kpts3d_path # world coords
        self._kpts3d_cam_path = kpts3d_cam_path # cam coords
        self._kpts2d_path = kpts2d_path
        self._rots_cam_path = rots_cam_path # cam coords
        self._camdata_path = camdata_path
        self._betas_path = betas_path
        self._rot_type = rot_type

        self._cameras = self._load_cam_data()
        self._data = self._load_3d_keypoints()

        if self._rots_cam_path:
            self._load_rotations()

        if self._betas_path:
            self._load_betas()

        # transform from rot matrix
        if self._rot_type is not None:
            self._transform_rotations()

    def load_2d_keypoints(self) -> np.ndarray:
        keypoints_2d = np.load(
            self._kpts2d_path, allow_pickle=True
        )["positions_2d"].item()
        return keypoints_2d

    def _load_3d_keypoints(self) -> dict:
        positions_data = np.load(
            self._kpts3d_path, allow_pickle=True,
        )["positions_3d"].item()
        positions_camera = np.load(
            self._kpts3d_cam_path, allow_pickle=True,
        )["positions_3d"].item()

        data = {}
        for subject, actions in positions_data.items():
            data[subject] = {}
            for action, positions in actions.items():
                data[subject][action] = {
                    "positions": positions,
                    "positions_3d": positions_camera[subject][action],
                    "cameras": self._cameras[subject][action],
                    "frame_rate": FPS,
                }
        return data

    def _load_rotations(self) -> None:
        rotations_camera = np.load(
            self._rots_cam_path, allow_pickle=True
        )["rotations_3d"].item()

        for subject, actions in self._data.items():
            for action, _ in actions.items():
                self._data[subject][action]["rotations"] = \
                    rotations_camera[subject][action]

    def _load_betas(self) -> None:
        betas_camera = np.load(
            self._betas_path, allow_pickle=True,
        )["betas"].item()

        for subject, actions in self._data.items():
            for action, _ in actions.items():
                self._data[subject][action]["betas"] = \
                    betas_camera[subject][action]

    def _load_transl(self) -> None:
        transl_smplx = np.load(
            self._transl_path, allow_pickle=True,
        )["transl"].item()

        for subject, actions in self._data.items():
            for action, _ in actions.items():
                self._data[subject][action]["transl"] = \
                    transl_smplx[subject][action]

    def _load_cam_data(self) -> dict:
        npz_data = np.load(
            self._camdata_path, allow_pickle=True,
        )["cameras"].item()
        cam_data = {}
        for subject, actions in npz_data.items():
            cam_data[subject] = {}
            for action, cameras in actions.items():
                cam_data[subject][action] = []
                for camera in cameras:
                    center = Fit3dDataset._normalize_cam_center(
                        camera["cx"], camera["cy"])
                    focal_len = Fit3dDataset._normalize_focal_len(
                        camera["fx"], camera["fy"])
                    radial_distortion = np.zeros((3,)) # never used
                    tan_distortion = np.empty((2,)) # never used
                    intristics = np.concatenate((
                        [RES_W, RES_H],
                        focal_len, center,
                        radial_distortion,
                        tan_distortion,
                    ))
                    cam_data[subject][action].append({
                        "id": camera["id"],
                        "center": center,
                        "focal_length": focal_len,
                        "translation": camera["T"],
                        "intrinsic": intristics,
                        "tangential_distortion": tan_distortion,
                        "radial_distortion": radial_distortion,
                    })
        return cam_data

    def _transform_rotations(self) -> None:
        for subject, actions in self._data.items():
            for action, values in actions.items():
                rots = values["rotations"]
                rot_trans = []
                for rot_mats in rots:
                    rt = Fit3dDataset._transform_rot_matrix(
                        self._rot_type, rot_mats)
                    rot_trans.append(rt)
                self._data[subject][action]["rotations"] = rot_trans

    @staticmethod
    def _normalize_cam_center(cx: float, cy: float) -> np.ndarray:
        X = np.array([cx, cy])
        norm_X = X / RES_W * 2 - [1, RES_H / RES_W]
        return norm_X

    @staticmethod
    def _normalize_focal_len(fx: float, fy: float) -> np.ndarray:
        flen = np.array([fx, fy])
        flen = flen / RES_W * 2
        return flen

    @staticmethod
    def _transform_rot_matrix(type_rot, rotations):
        # Fit3d gt is in rot mat format
        if type_rot == RotType.ROT_MAT \
            and rotations.shape[-2:] == (3,3)\
                and len(rotations.shape) == 4:
            return rotations

        device = "cuda" if torch.cuda.is_available() else "cpu"
        rotations = torch.from_numpy(rotations).clone().to(device)
        if type_rot == RotType.AXIS_ANGLE:
            if rotations.shape[-1] == 3 and len(rotations.shape) == 3:
                # dont do anything if data already axis-angle
                pass
            else:
                rotations = matrix_to_axis_angle(rotations)
        elif type_rot == RotType.QUATERNION:
            if rotations.shape[-1] == 4 and len(rotations.shape) == 3:
                # dont do anything if data already quaternion
                pass
            else:
                rotations = matrix_to_quaternion(rotations)

        return rotations.cpu().numpy()
