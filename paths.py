# -*- coding: utf-8 -*-
"""
Created on 14.05.25

@author: Katja

"""
SMPLX_DIR = "/.../smpl-models"
VPOSER_DIR = '/.../V02_05'
FIT3D_DIR = "/...fit3d"
LOG_DIR = "/.../3d_hpe_rot"
FIT3D_PROCESSED_ROOT = f"{FIT3D_DIR}/processed"
FIT3D_PROCESSED = {
    "world_kpts3d_path": f"{FIT3D_PROCESSED_ROOT}/fit3d_3d_world.npz",
    "cam_kpts3d_path": f"{FIT3D_PROCESSED_ROOT}/fit3d_3d_camera.npz",
    "cam_rot_path": f"{FIT3D_PROCESSED_ROOT}/fit3d_rot3d_camera.npz", # pose vectors theta
    "kpts2d_path": f"{FIT3D_PROCESSED_ROOT}/fit3d_2d_vitpose.npz",
    "cam_path": f"{FIT3D_PROCESSED_ROOT}/fit3d_cameras_meta.npz",
    "betas_path": f"{FIT3D_PROCESSED_ROOT}/fit3d_betas.npz"
}