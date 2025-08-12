from typing import Any, Callable

import einops
import torch
import torch.nn as nn

from model.uplift_upsample_transformer import UpliftUpsampleTransformer
from model.smplx_layer import SMPLX_Layer
from model.vposer_layer import VPoserLayer

from utils.rotation import symmetric_orthogonalization

class UpliftPoseTransformer(UpliftUpsampleTransformer):
    def __init__(
        self,
        in_feature_size: int = 2,
        out_dim: int = 3,
        full_output: bool = True,
        num_frames: int = 9,
        num_keypoints: int = 17,
        spatial_d_model: int = 16,
        temporal_d_model: int = 256,
        spatial_depth: int = 3,
        temporal_depth: int = 3,
        strides: list = ...,
        paddings: list = None,
        num_heads: int = 8,
        mlp_ratio: float = 2,
        qkv_bias: bool = True,
        attn_drop_rate: float = 0.1,
        drop_rate: float = 0.1,
        drop_path_rate: tuple = ...,
        norm_layer: Callable[..., Any] = nn.LayerNorm,
        output_bn: bool = False,
        has_strided_input: bool = False,
        first_strided_token_attention_layer: int = 0,
        token_mask_rate: float = 0,
        learnable_masked_token: bool = False,
        return_attention: bool = False,
        smplx_layer: tuple = (False, False),
        vposer_layer: bool = False,
        keypoint_order = None,
    ):
        """Model for Uplift Pose and Rotations Transformer
        """
        super().__init__(
            in_feature_size,
            full_output,
            num_frames,
            num_keypoints,
            spatial_d_model,
            temporal_d_model,
            spatial_depth,
            temporal_depth,
            strides,
            paddings,
            num_heads,
            mlp_ratio,
            qkv_bias,
            attn_drop_rate,
            drop_rate,
            drop_path_rate,
            norm_layer,
            output_bn,
            has_strided_input,
            first_strided_token_attention_layer,
            token_mask_rate,
            learnable_masked_token,
            return_attention,
        )
        self.out_dim = out_dim # dimension of rot representation 3, 4 or 9
        self.num_keypoints = num_keypoints
        self.output_bn = output_bn

        # select initialisation and forward function
        if vposer_layer:
            assert self.out_dim == 3, "Only Axis-Angle is supported.."
            self._init_1()
            self.forward_func = self.forward_1
        else:
            self._init_2()
            self.forward_func = self.forward_2

        # central frame smplx layer
        self.smplx = SMPLX_Layer(aa=self.out_dim == 3, keypoint_order=keypoint_order) if smplx_layer[1] else None
        # full sequence smplx layer
        self.smplx_full = SMPLX_Layer(full_seq=True, aa=self.out_dim == 3, keypoint_order=keypoint_order) if smplx_layer[0] else None

    def _init_1(self):
        """Initialisation of model with VPoser.
        """
        # head for VPoser latent space vector - full sequence
        if self.full_output is True and self.temporal_depth > 0:
            self.head_poZ = nn.ModuleList()
            if self.output_bn:
                self.head_poZ.append(nn.BatchNorm1d(self.temporal_d_model, momentum=0.1, eps=1e-5))
            self.head_poZ.append(nn.Linear(self.temporal_d_model, 32))
            self.head_poZ.append(VPoserLayer())

        # head for VPoser latent space vector - central frame
        self.head_poZ_c = nn.ModuleList()
        if self.output_bn:
            self.head_poZ_c.append(nn.BatchNorm1d(self.temporal_d_model, momentum=0.1, eps=1e-5))
        self.head_poZ_c.append(nn.Linear(self.temporal_d_model, 32))
        self.head_poZ_c.append(VPoserLayer())

        # for integrity -> kpts outside of VPoser latent space
        if self.full_output is True and self.temporal_depth > 0:
            self.head_rest = nn.ModuleList()
            if self.output_bn:
                self.head_rest.append(nn.BatchNorm1d(self.temporal_d_model, momentum=0.1, eps=1e-5))
            self.head_rest.append(nn.Linear(self.temporal_d_model, self.out_dim * 5)) # global orient plus 4 hands

        self.head_rest_c = nn.ModuleList()
        if self.output_bn:
            self.head_rest_c.append(nn.BatchNorm1d(self.temporal_d_model, momentum=0.1, eps=1e-5))
        self.head_rest_c.append(nn.Linear(self.temporal_d_model, self.out_dim * 5))

    def _init_2(self):
        """Default initialisation of model.
        """
        self.head3 = None
        if self.full_output is True and self.temporal_depth > 0:
            # full sequence rotations
            self.head3 = nn.ModuleList()
            if self.output_bn:
                self.head3.append(nn.BatchNorm1d(self.temporal_d_model, momentum=0.1, eps=1e-5))
            self.head3.append(nn.Linear(self.temporal_d_model, self.out_dim * self.num_keypoints))

        # central rotations
        self.head4 = nn.ModuleList()
        if self.output_bn:
            self.head4.append(nn.BatchNorm1d(self.temporal_d_model, momentum=0.1, eps=1e-5))
        self.head4.append(nn.Linear(self.temporal_d_model, self.out_dim * self.num_keypoints))

    def forward(self, inputs, betas=None):
        return self.forward_func(inputs, betas)

    def forward_1(self, inputs, betas=None):
        """Forward pass of model with VPoser.
        """
        if self.has_strided_input:
            x, stride_mask = inputs[0], inputs[1]
        else:
            x = inputs
            stride_mask = None

        b, n, p, _ = x.shape
        mid_frame = self.num_frames // 2

        if betas is not None and len(betas.shape) == 1:
            betas = torch.tile(betas, (b, n, 1))

        x = self.spatial_transformation(x)
        # Full sequence temporal transformer
        x, att_list = self.temporal_transformation(x, stride_mask=stride_mask)

        kpts, rots = None, None
        if self.full_output is True and self.temporal_depth > 0:
            body_pose = x
            for layer in self.head_poZ:
                body_pose = layer(body_pose)
            body_pose = body_pose.reshape(b, n, 21, 3)

            go_n_hands = x
            for layer in self.head_rest:
                go_n_hands = layer(go_n_hands)
            go_n_hands = go_n_hands.reshape(b, n, 5, 3)
            go = go_n_hands[..., 0, :].unsqueeze(-2)
            hands = go_n_hands[..., 1:, :]
            rots = torch.cat((go, body_pose, hands), dim=-2)

            # path through smplx if enabled
            if self.smplx_full is not None:
                kpts = self.smplx_full(rots, betas, None)
            else:
                kpts = x
                for layer in self.head1:
                    kpts = layer(kpts)
                kpts = kpts.reshape(b, n, p, 3)

        if len(self.strides) > 0:
            x = self.strided_temporal_transformation(x, stride_mask=stride_mask)
            c_kpts = x
            c_body_pose = x
            c_go_n_hands = x
        else:
            def get_midframe(x, mid_frame):
                x_mid = x[:, mid_frame, :]
                return x_mid[:, None, :]

            c_body_pose = get_midframe(x, mid_frame)
            c_kpts = get_midframe(x, mid_frame)
            c_go_n_hands = get_midframe(x, mid_frame)

        for layer in self.head_poZ_c:
            c_body_pose = layer(c_body_pose)

        for layer in self.head_rest_c:
            c_go_n_hands = layer(c_go_n_hands)
        c_go_n_hands = einops.rearrange(c_go_n_hands, "b n (p c) -> (b n) p c", b=b, n=1, p=5, c=3)

        c_go = c_go_n_hands[..., 0, :].unsqueeze(-2)
        c_hands = c_go_n_hands[..., 1:, :]
        c_rots = torch.cat((c_go, c_body_pose, c_hands), dim=-2)

        if self.smplx is not None:
            c_betas = betas[:, mid_frame]
            c_kpts = self.smplx(c_rots, c_betas, None)
        else:
            for layer in self.head2:
                c_kpts = layer(c_kpts)
            c_kpts = einops.rearrange(c_kpts, "b n (p c) -> (b n) p c", n=1, p=p, c=3)

        full_out = (kpts, rots)
        central_out = (c_kpts, c_rots)

        return full_out, central_out


    def forward_2(self, inputs, betas=None):
        """Default forward pass of model with VPoser.
        """
        if self.has_strided_input:
            x, stride_mask = inputs[0], inputs[1]
        else:
            x = inputs
            stride_mask = None

        b, n, p, _ = x.shape
        mid_frame = self.num_frames // 2
        x = self.spatial_transformation(x)
        # Full sequence temporal transformer
        x, att_list = self.temporal_transformation(x, stride_mask=stride_mask)

        kpts, rots = None, None
        if self.full_output is True and self.temporal_depth > 0:
            rots = x
            for layer in self.head3:
                rots = layer(rots)
            rots = rots.reshape(b, n, p, self.out_dim)
            if self.out_dim == 9:
                rots = symmetric_orthogonalization(rots)

            if self.smplx_full is not None:
                kpts = self.smplx_full(rots, betas, None)
            else:
                kpts = x
                for layer in self.head1:
                    kpts = layer(kpts)
                kpts = kpts.reshape(b, n, p, 3)

        # Strided transformer
        if len(self.strides) > 0:
            x = self.strided_temporal_transformation(x, stride_mask=stride_mask)
            # Prediction for central frame
            c_kpts = x
            c_rots = x
        else:
            def get_midframe(x, mid_frame):
                x_mid = x[:, mid_frame, :]
                return x_mid[:, None, :]
            c_kpts = get_midframe(x, mid_frame)
            c_rots = get_midframe(x, mid_frame)

        for layer in self.head4:
            c_rots = layer(c_rots)
        c_rots = einops.rearrange(c_rots, "b n (p c) -> (b n) p c", n=1, p=p, c=self.out_dim)

        # SVD to ensure rot matrix fulfills the properties
        if self.out_dim == 9:
            c_rots = symmetric_orthogonalization(c_rots)

        if self.smplx is not None:
            assert betas is not None
            c_betas = betas[:, mid_frame]
            c_kpts = self.smplx(c_rots, c_betas, None)
        else:
            for layer in self.head2:
                c_kpts = layer(c_kpts)
            c_kpts = einops.rearrange(
                c_kpts,
                "b n (p c) -> (b n) p c", n=1, p=p, c=3)

        full_out = (kpts, rots)
        central_out = (c_kpts, c_rots)

        return full_out, central_out

