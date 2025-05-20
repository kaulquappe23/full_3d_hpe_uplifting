from enum import Enum
import math

import torch
from utils import rotation_conversions


class RotType(Enum):
    AXIS_ANGLE = 3
    QUATERNION = 4
    SIX_D = 6
    ROT_MAT = 9


def transform_to_matrix(input_t, rotation_type):
    if rotation_type == RotType.AXIS_ANGLE:
        return rotation_conversions.axis_angle_to_matrix(input_t)
    elif rotation_type == RotType.QUATERNION:
        return rotation_conversions.quaternion_to_matrix(input_t)
    elif rotation_type == RotType.SIX_D:
        return rotation_conversions.rotation_6d_to_matrix(input_t)
    else:
        raise NotImplementedError(f"{rotation_type} is not supported")


def symmetric_orthogonalization(x, **kwargs):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

    x: should have size [batchsize, seq, num_kpts, 9] or
        [batchsize, num_kpts, 9]

    Output has size [batchsize, seq, num_kpts, 3, 3] or
        [batchsize, num_kpts, 3, 3],
    where each inner 3x3 matrix is in SO(3).
    """
    # Check the input dimensions
    if len(x.shape) == 4:  # [batchsize, seq, num_kpts, 9]
        batchsize, seq, num_kpts, _ = x.shape
        x = x.view(-1, 9)  # Reshape to [batchsize * seq * num_kpts, 9]
        reshape_back = True
    elif len(x.shape) == 3:  # [batchsize, num_kpts, 9]
        batchsize, num_kpts, _ = x.shape
        seq = 1
        x = x.view(-1, 9)  # Reshape to [batchsize * num_kpts, 9]
        reshape_back = False
    else:
        raise ValueError("Input must be of shape [batchsize, seq, num_kpts, 9] \
            or [batchsize, num_kpts, 9]")

    m = x.view(-1, 3, 3)  # Reshape to [N, 3, 3] to all previous dimensions
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)

    if reshape_back:
        r = r.view(batchsize, seq, num_kpts, 3, 3)
    else:
        r = r.view(batchsize, num_kpts, 3, 3)

    identity = torch.eye(3, device=r.device, dtype=r.dtype).expand_as(r)
    assert torch.allclose(torch.matmul(r, r.transpose(-2, -1)), identity, atol=1e-5), \
        "r must be orthogonal."

    ones = torch.ones(r.shape[:-2], device=r.device, dtype=r.dtype)
    assert torch.allclose(torch.det(r), ones, atol=1e-5), \
        "r must have a determinant of 1."

    return r


def joints_to_axis_angle(
    angles: torch.Tensor,
    joints: torch.Tensor,
    betas: torch.Tensor,
    zero_joints: torch.Tensor,
    kinematic_tree: list,
) -> torch.Tensor:
    angles = angles.clone()
    joints = joints.clone()
    assert angles.shape[0] == joints.shape[0] and joints.shape[0] == betas.shape[0]

    bs, num_kpts, _ = joints.shape
    zero_joints -= zero_joints[:, 0]
    joints -= joints[:, 0].clone()

    P_old = collect_positions(zero_joints, kinematic_tree, num_kpts)
    P_new = collect_positions(joints, kinematic_tree, num_kpts)

    rot_axis = []
    for _, (po, pn) in enumerate(zip(P_old, P_new)):
        if po is None:
            axis = torch.zeros((bs, 3))
        else:
            if len(po.shape) == 3:
                po = torch.mean(po, dim=-1)
                pn = torch.mean(pn, dim=-1)
            axis = torch.cross(po, pn, dim=-1)
            axis = axis / torch.linalg.vector_norm(axis, ord=2, dim=-1)
        rot_axis.append(axis)

    rot_axis = torch.stack(rot_axis, dim=1)
    rot_axis = torch.nan_to_num(rot_axis)

    rot_axis[:, 1:22] *= angles
    return rot_axis


def find_rotation_matrix_svd_batch(p_old, p_new):
    H = p_new @ p_old.transpose(-2, -1)
    U, _, V = torch.svd(H)
    Vt = torch.transpose(V, -2, -1)
    R = torch.matmul(U, Vt)

    det_batch = torch.det(R)
    neg_det_mask = det_batch < 0
    U[neg_det_mask, :, -1] *= -1
    R = torch.matmul(U, Vt.transpose(-2, -1))
    return R


def form_axis_from_rot_mat(R: torch.Tensor) -> torch.Tensor:
    bs, _, _ = R.shape
    trace_R = torch.einsum("...ii", R)

    angles = torch.arccos((trace_R - 1) / 2).clamp(-1.0, 1.0)  # Ensure values are within valid range
    axes = torch.zeros(bs, 3, dtype=torch.float32)

    small_angle_mask = angles < 1e-8
    large_angle_mask = (angles - math.pi).abs() < 1e-8
    general_mask = ~(small_angle_mask | large_angle_mask)

    # Handle small angles (close to 0)
    axes[small_angle_mask] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    # Handle large angles (close to 180 degrees)
    if large_angle_mask.any():
        R_plus_I = R[large_angle_mask] + torch.eye(3, dtype=torch.float32)
        axes[large_angle_mask] = torch.nn.functional.normalize(R_plus_I[:, :, 0], dim=1)

    # Handle general case
    if general_mask.any():
        R_gen = R[general_mask]
        rx = R_gen[:, 2, 1] - R_gen[:, 1, 2]
        ry = R_gen[:, 0, 2] - R_gen[:, 2, 0]
        rz = R_gen[:, 1, 0] - R_gen[:, 0, 1]
        axis_gen = torch.stack((rx, ry, rz), dim=1)
        axes[general_mask] = torch.nn.functional.normalize(axis_gen, dim=1)

    return axes


def collect_positions(joints: torch.Tensor, kinematic_tree: list, num_joints: int) -> list:
    children = [[] for _ in range(num_joints)]

    for parent, child in kinematic_tree:
        children[parent].append(joints[:, child].clone())

    out = []
    for vs in children:
        if len(vs) > 1:
            out.append(torch.stack(vs, -1))
        elif len(vs) == 1:
            out.append(vs[0])
        else:
            out.append(None)
    return out


def angle_from_rot_matrix(rot_matrix):
    trace = torch.einsum('...ii->...', rot_matrix)
    angles = torch.acos((trace - 1) / 2)
    return angles

def angle_from_axis_angle(axis_angle):
    return torch.norm(axis_angle, dim=-1)

def angle_from_quarterion(quat):
    quat = quat.clone()
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return 2 * torch.atan2(torch.sqrt(x**2 + y**2 + z**2), w)


def flip_axis_angle(pose, flip_lr_indices, flipped_idx):
    pose = pose.clone()
    pose[..., :, 1] *= -1
    pose[..., :, 2] *= -1

    pose_fl = pose[:, :, flip_lr_indices].clone()
    return pose_fl

def flip_quaternion(pose, flip_lr_indices, flipped_idx):
    pose = pose.clone()
    pose[..., :, 2] *= -1
    pose[..., :, 3] *= -1
    pose_fl = pose[:, :, flip_lr_indices].clone()
    return pose_fl

