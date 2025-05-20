from utils.rotation import RotType

from model.uplift_upsample_transformer_config import UpliftUpsampleConfig

class UpliftPoseConfig(UpliftUpsampleConfig):
    ROT_REP = "ROT_MAT"

    BONES = None
    BONES_LOSS_WEIGHT = 0.05
    BONES_LOSS_ENABLED = False

    SMPLX_LAYER = [False, False]
    LOAD_SMPLX = any(SMPLX_LAYER)
    EVAL_BETAS = any(SMPLX_LAYER)

    LOSS_ANGLES_ENABLED = False
    LOSS_ANGLES = "mse"
    ANGLES_LOSS_WEIGHT = 1.0

    LOSS_MESH_ENABLED = False

    LOSS_ANGLE_LIMIT = False

    VPOSER = False
    ESTIMATE_HANDS = True