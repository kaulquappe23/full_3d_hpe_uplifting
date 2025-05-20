# Code adapted from: https://github.com/facebookresearch/VideoPose3D
# Original Code: Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch

# import torch

from dataset.quaternion import np_qrot, np_qinverse


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2  # x and y coordinates

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]


def image_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Reverse camera frame normalization
    return (X + [1, h / w]) * w / 2


def world_to_camera(X, R, t):
    Rt = np_qinverse(R)  # Invert rotation
    return np_qrot(np.tile(Rt, (*X.shape[:-1], 1)), X - t)  # Rotate and translate


def camera_to_world(X, R, t):
    return np_qrot(np.tile(R, (*X.shape[:-1], 1)), X) + t


def project_to_2d_linear(X, f, c):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    """
    assert X.shape[-1] == 3

    XX = X[..., :2] / X[..., 2:]

    return f * XX + c


def project_to_2d_linear_torch(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)

    return f * XX + c


def project_to_2d_batched(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(
        k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape) - 1),
        dim=len(r2.shape) - 1,
        keepdim=True,
    )
    tan = torch.sum(p * XX, dim=len(XX.shape) - 1, keepdim=True)

    XXX = XX * (radial + tan) + p * r2

    return f * XXX + c


def project_to_2d(X, focal_length, center, radial_dist, tangential_dist):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3

    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X).float()
    if not isinstance(radial_dist, torch.Tensor):
        radial_dist = torch.from_numpy(radial_dist).float()
    if not isinstance(tangential_dist, torch.Tensor):
        tangential_dist = torch.from_numpy(tangential_dist).float()
    if not isinstance(focal_length, torch.Tensor):
        focal_length = torch.from_numpy(focal_length).float()
    if not isinstance(center, torch.Tensor):
        center = torch.from_numpy(center).float()

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2] ** 2, dim=len(XX.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(
        radial_dist * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape) - 1),
        dim=len(r2.shape) - 1,
        keepdim=True,
    )
    tan = torch.sum(tangential_dist * XX, dim=len(XX.shape) - 1, keepdim=True)

    XXX = XX * (radial + tan) + tangential_dist * r2

    return focal_length * XXX + center
