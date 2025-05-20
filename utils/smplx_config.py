# -*- coding: utf-8 -*-
"""
Created on 14.05.25

@author: Katja

"""

SMPLX_CFG = {
    "ext": "npz",
    "extra_joint_path": "",
    "folder": "transfer_data/body_models",
    "gender": "neutral",
    "joint_regressor_path": "",
    "model_type": "smplx",
    "num_expression_coeffs": 10,
    "smplx": {
        "betas": {"create": True, "num": 10, "requires_grad": True},
        "body_pose": {"create": True, "requires_grad": True, "type": "aa"},
        "expression": {"create": True, "num": 10, "requires_grad": True},
        "global_rot": {"create": True, "requires_grad": True, "type": "aa"},
        "jaw_pose": {"create": False, "requires_grad": True, "type": "aa"},
        "left_hand_pose": {
            "create": False,
            "pca": {"flat_hand_mean": False, "num_comps": 12},
            "requires_grad": True,
            "type": "aa",
        },
        "leye_pose": {"create": False, "requires_grad": True, "type": "aa"},
        "reye_pose": {"create": False, "requires_grad": True, "type": "aa"},
        "right_hand_pose": {
            "create": False,
            "pca": {"flat_hand_mean": False, "num_comps": 12},
            "requires_grad": True,
            "type": "aa",
        },
        "translation": {"create": True, "requires_grad": True},
    },
    "use_compressed": False,
    "use_face_contour": True,
}