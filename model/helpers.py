from model.uplift_pose_transformer import UpliftPoseTransformer
from model.uplift_upsample_transformer import UpliftUpsampleTransformer
from model.uplift_upsample_transformer_config import UpliftUpsampleConfig
from model.uplift_pose_config import UpliftPoseConfig



def build_model(config):
    arch = config.ARCH
    if arch == "UpliftUpsampleTransformer":
        model = build_uplift_upsample_transformer(config)
    elif arch == "UpliftPoseTransformer":
        model = build_uplift_pose_transformer(config)
    else:
        NotImplementedError(f"`{arch}` architecture is not supported.")
    return model


def build_uplift_upsample_transformer(config: UpliftUpsampleConfig) -> UpliftUpsampleTransformer:
    input_shape = (config.SEQUENCE_LENGTH, config.NUM_KEYPOINTS, 2)
    has_strided_input = config.MASK_STRIDE is not None
    if has_strided_input:
        if type(config.MASK_STRIDE) is int and config.MASK_STRIDE == 1:
            has_strided_input = False
        if type(config.MASK_STRIDE) is list and config.MASK_STRIDE[0] == 1:
            has_strided_input = False

    model = UpliftUpsampleTransformer(
        in_feature_size=config.IN_FEATURE_SIZE,
        full_output=not config.USE_REFINE,
        num_frames=config.SEQUENCE_LENGTH,
        num_keypoints=config.NUM_KEYPOINTS,
        spatial_d_model=config.SPATIAL_EMBED_DIM,
        temporal_d_model=config.TEMPORAL_EMBED_DIM,
        spatial_depth=config.SPATIAL_TRANSFORMER_BLOCKS,
        temporal_depth=config.TEMPORAL_TRANSFORMER_BLOCKS,
        strides=config.STRIDES,
        paddings=config.PADDINGS,
        num_heads=config.NUM_HEADS,
        mlp_ratio=config.MLP_RATIO,
        qkv_bias=config.QKV_BIAS,
        attn_drop_rate=config.ATTENTION_DROP_RATE,
        drop_rate=config.DROP_RATE,
        drop_path_rate=config.DROP_PATH_RATE,
        output_bn=config.OUTPUT_BN,
        has_strided_input=has_strided_input,
        first_strided_token_attention_layer=config.FIRST_STRIDED_TOKEN_ATTENTION_LAYER,
        token_mask_rate=config.TOKEN_MASK_RATE,
        learnable_masked_token=config.LEARNABLE_MASKED_TOKEN,
    )

    return model

def build_uplift_pose_transformer(config: UpliftPoseConfig) -> UpliftPoseTransformer:
    input_shape = (config.SEQUENCE_LENGTH, config.NUM_KEYPOINTS, 2)
    has_strided_input = config.MASK_STRIDE is not None
    if has_strided_input:
        if type(config.MASK_STRIDE) is int and config.MASK_STRIDE == 1:
            has_strided_input = False
        if type(config.MASK_STRIDE) is list and config.MASK_STRIDE[0] == 1:
            has_strided_input = False

    model = UpliftPoseTransformer(
        in_feature_size=config.IN_FEATURE_SIZE,
        out_dim=config.OUT_DIM,
        full_output=not config.USE_REFINE,
        num_frames=config.SEQUENCE_LENGTH,
        num_keypoints=config.NUM_KEYPOINTS,
        spatial_d_model=config.SPATIAL_EMBED_DIM,
        temporal_d_model=config.TEMPORAL_EMBED_DIM,
        spatial_depth=config.SPATIAL_TRANSFORMER_BLOCKS,
        temporal_depth=config.TEMPORAL_TRANSFORMER_BLOCKS,
        strides=config.STRIDES,
        paddings=config.PADDINGS,
        num_heads=config.NUM_HEADS,
        mlp_ratio=config.MLP_RATIO,
        qkv_bias=config.QKV_BIAS,
        attn_drop_rate=config.ATTENTION_DROP_RATE,
        drop_rate=config.DROP_RATE,
        drop_path_rate=config.DROP_PATH_RATE,
        output_bn=config.OUTPUT_BN,
        has_strided_input=has_strided_input,
        first_strided_token_attention_layer=config.FIRST_STRIDED_TOKEN_ATTENTION_LAYER,
        token_mask_rate=config.TOKEN_MASK_RATE,
        learnable_masked_token=config.LEARNABLE_MASKED_TOKEN,
        smplx_layer=config.SMPLX_LAYER,
        vposer_layer=config.VPOSER,
        keypoint_order=config.KEYPOINT_ORDER
    )

    return model
