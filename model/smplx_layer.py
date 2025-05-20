import einops
import torch
import smplx
from torch import nn

from paths import SMPLX_DIR
from utils.smplx_config import SMPLX_CFG



class SMPLX_Layer(nn.Module):
    """
    """
    def __init__(self, full_seq=False, kpts_all=False, aa=True, keypoint_order=None, *args, **kwargs):
        super().__init__()

        assert SMPLX_CFG["model_type"] == "smplx"
        self.model_type = SMPLX_CFG["model_type"]
        self.num_betas = SMPLX_CFG["smplx"]["betas"]["num"]
        self.gender = SMPLX_CFG["gender"]
        self.full_seq = full_seq
        self.return_all = kpts_all
        self.kp_ord = keypoint_order

        # possible only 2 variants
        if aa:
            # axis-angle
            self.bm_x = smplx.create(
                SMPLX_DIR,
                model_type=self.model_type,
                gender=self.gender,
                use_pca=False,
                flat_hand_mean=True,
                num_betas=self.num_betas,
            )
        else:
            # rotation matrix
            self.bm_rot_mat = smplx.build_layer(
                SMPLX_DIR,
                model_type=self.model_type,
                gender=self.gender,
                use_pca=False,
                flat_hand_mean=True,
                num_betas=self.num_betas,
            )

    def forward(self, pose, shape, transl=None):
        dtype = pose.dtype
        device = pose.device
        # if device == "cuda":
#        self.cuda()

        if not self.full_seq:
            pose = pose.unsqueeze(1)
            shape = shape.unsqueeze(1)

        num_dp = pose.shape[0]
        batch_size = 20 if pose.shape[1] > 1 else pose.shape[0]
        processed_batches = []
        for i in range(0, num_dp, batch_size):
            # process in batches due to memory capacity
            pose_b = pose[i:i + batch_size]
            shape_b = shape[i:i + batch_size]

            if len(pose_b.shape) == 5:
                b, n, _, c1, c2 = pose_b.shape
                pose_b = einops.rearrange(pose_b, "b n p c1 c2 -> (b n) p c1 c2")
                return_smplx = self.form_rot_mat
            else:
                b, n, _, c = pose_b.shape
                pose_b = einops.rearrange(pose_b, "b n p c -> (b n) p c")
                return_smplx = self.form_axis_angle

            shape_b = einops.rearrange(shape_b, "b n p -> (b n) p")

            output = return_smplx(pose_b, shape_b, dtype, device)
            j3d_b = output.joints

            if not self.return_all:
                j3d_b = j3d_b[:, self.kp_ord.smplx_joints()]
            j3d_b = einops.rearrange(j3d_b, "(b n) p c -> b n p c", b=b, n=n, c=3)

            processed_batches.append(j3d_b)

        j3d = torch.cat(processed_batches, dim=0)
        j3d = j3d.squeeze(1) if not self.full_seq else j3d
        return j3d

    def form_axis_angle(self, pose_b, shape_b, dtype, device):
        bs_b = pose_b.shape[0]
        kwargs_pose = {
            "betas": shape_b,
            # "transl": transl,
            "jaw_pose": torch.zeros((bs_b, 3), dtype=dtype, device=device),
            "expression": torch.zeros((bs_b, 10), dtype=dtype, device=device),
            "leye_pose": torch.zeros((bs_b, 3), dtype=dtype, device=device),
            "reye_pose": torch.zeros((bs_b, 3), dtype=dtype, device=device),
        }
        kwargs_pose["global_orient"] = pose_b[:, self.kp_ord.global_orient]
        kwargs_pose["body_pose"] = pose_b[:, self.kp_ord.body_pose_smplx()]

        # only important hand joints, other in rest pose
        lh_pose = torch.zeros((bs_b, 15, 3), dtype=dtype, device=device)
        lh_pose[:, self.kp_ord._fingers_pose] = pose_b[:, self.kp_ord._left_fingers]
        kwargs_pose["left_hand_pose"] = lh_pose

        rh_pose = torch.zeros((bs_b, 15, 3), dtype=dtype, device=device)
        rh_pose[:, self.kp_ord._fingers_pose] = pose_b[:, self.kp_ord._right_fingers]
        kwargs_pose["right_hand_pose"] = rh_pose

        # Forward using the parametric 3d model SMPL-X layer
        return self.bm_x(**kwargs_pose)

    def form_rot_mat(self, pose_b, shape_b, dtype, device):
        bs_b = pose_b.shape[0]
        kwargs_pose = {
            "betas": shape_b,
            # "transl": transl,
            "jaw_pose": torch.zeros((bs_b, 3, 3), dtype=dtype, device=device),
            "expression": torch.zeros((bs_b, 10), dtype=dtype, device=device),
            "leye_pose": torch.zeros((bs_b, 3, 3), dtype=dtype, device=device),
            "reye_pose": torch.zeros((bs_b, 3, 3), dtype=dtype, device=device),
        }
        kwargs_pose["global_orient"] = pose_b[:, self.kp_ord.global_orient]
        kwargs_pose["body_pose"] = pose_b[:, self.kp_ord.body_pose_smplx()]

        # only important hand joints, other in rest pose
        lh_pose = torch.zeros((bs_b, 15, 3, 3), dtype=dtype, device=device)
        lh_pose[:, self.kp_ord._fingers_pose] = pose_b[:, self.kp_ord._left_fingers]
        kwargs_pose["left_hand_pose"] = lh_pose

        rh_pose = torch.zeros((bs_b, 15, 3, 3), dtype=dtype, device=device)
        rh_pose[:, self.kp_ord._fingers_pose] = pose_b[:, self.kp_ord._right_fingers]
        kwargs_pose["right_hand_pose"] = rh_pose

        # Forward using the parametric 3d model SMPL-X layer
        return self.bm_rot_mat(**kwargs_pose)


    def cuda(self):
        if hasattr(self, "bm_x"):
            self.bm_x.cuda()
        if hasattr(self, "bm_rot_mat"):
            self.bm_rot_mat.cuda()
