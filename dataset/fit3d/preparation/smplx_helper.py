import os
import copy

import cv2
import torch
import numpy as np

from smplx import build_layer

from paths import SMPLX_DIR
from utils.smplx_config import SMPLX_CFG


class SMPLXHelper:
    def __init__(
        self,
        models_path: str = SMPLX_DIR,
    ):
        self.models_path = models_path
        self.cfg = SMPLX_CFG
        self.model = build_layer(self.models_path, **self.cfg, model="smplx")
        self.image_shape = (900, 900)

    def get_tpose_joints(self, betas):
        bs, _ = betas.shape
        zero_smplx_params = {
            "body_pose": torch.zeros((bs, 21, 3)),
            "global_orient": torch.zeros((bs, 3)),
            "betas": betas,
        }
        zero_model = self.model(**zero_smplx_params)
        zero_joints = zero_model.joints.detach().clone()
        return zero_joints

    def get_world_smplx_params(self, smplx_params):
        world_smplx_params = {
            key: torch.from_numpy(
                np.array(smplx_params[key]).astype(np.float32))
            for key in smplx_params
        }
        return world_smplx_params

    def get_camera_smplx_params(self, smplx_params, cam_params):
        pelvis = (
            self.model(
                betas=torch.from_numpy(
                    np.array(smplx_params["betas"]).astype(np.float32)
                )
            )
            .joints[:, 0, :]
            .numpy()
        )
        camera_smplx_params = copy.deepcopy(smplx_params)
        camera_smplx_params["global_orient"] = np.matmul(
            np.array(smplx_params["global_orient"]).transpose(0, 1, 3, 2),
            np.transpose(cam_params["extrinsics"]["R"]),
        ).transpose(0, 1, 3, 2)
        camera_smplx_params["transl"] = (
            np.matmul(
                smplx_params["transl"] + pelvis -
                cam_params["extrinsics"]["T"],
                np.transpose(cam_params["extrinsics"]["R"]),
            )
            - pelvis
        )
        camera_smplx_params = {
            key: torch.from_numpy(
                np.array(camera_smplx_params[key]).astype(np.float32))
            for key in camera_smplx_params
        }
        return camera_smplx_params


    def get_template_params(self, batch_size=1):
        smplx_params = {}
        smplx_params_all = self.model()
        for key1 in [
            "transl",
            "global_orient",
            "body_pose",
            "betas",
            "left_hand_pose",
            "right_hand_pose",
            "jaw_pose",
            "expression",
            "leye_pose",
            "reye_pose",
        ]:
            key2 = key1 if key1 in smplx_params_all else "jaw_pose"
            smplx_params[key1] = np.repeat(
                smplx_params_all[key2].cpu().detach().numpy(), batch_size, axis=0
            )
        smplx_params["transl"][:, 2] = 3
        smplx_params["global_orient"][:, :, 1, 1] = -1
        smplx_params["global_orient"][:, :, 2, 2] = -1
        return smplx_params

    def get_template(self):
        smplx_posed_data = self.model()
        smplx_template = {
            "vertices": smplx_posed_data.vertices[0].cpu().detach().numpy(),
            "triangles": self.model.faces,
        }
        return smplx_template

