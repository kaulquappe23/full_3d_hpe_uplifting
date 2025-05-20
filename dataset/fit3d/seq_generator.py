import math
import logging
from typing import Optional

import torch
import numpy as np

from torch.utils.data import Dataset

from dataset.fit3d import splits
from dataset.fit3d.dataset import Fit3dDataset, RES_H, RES_W
from dataset.fit3d.keypoint_order import Fit3DOrder26P
from dataset.camera import normalize_screen_coordinates
from dataset.mocap_dataset import MocapDataset
from utils.rotation import flip_axis_angle, flip_quaternion
from utils.rotation_conversions import matrix_to_axis_angle, axis_angle_to_matrix


class Fit3DSequenceGenerator(Dataset):
    def __init__(
        self,
        subset: str,
        seq_len: int,
        subsample: int,
        rotation_type: str,
        stride: int = 1,
        padding_type: str = "zeros",
        flip_augment: bool = True,
        flip_lr_indices: Optional[list] = None,
        mask_stride: Optional[list] = None,
        stride_mask_align_global: bool = False,
        rand_shift_stride_mask: bool = False,
        load_smplx_params: bool = False,
        seed: int = 0,
        **kwargs: dict,
    ):
        self.seq_len = seq_len
        self.subsample = subsample
        self.stride = stride  # this is the used output stride s_out
        self.target_frame_rate = 50
        if padding_type == "zeros":
            self.pad_type = "constant"
        elif padding_type == "copy":
            self.pad_type = "edge"
        else:
            raise ValueError(f"Padding type not supported: {padding_type}")
        self.flip_augment = flip_augment
        self.flip_lr_indices = flip_lr_indices
        self.abs_mask_stride = mask_stride
        if self.abs_mask_stride is not None:  # these are the absolute input strides s_in that are used during training
            if type(self.abs_mask_stride) is not list:
                self.abs_mask_stride = [self.abs_mask_stride]
            for ams in self.abs_mask_stride:
                assert ams >= self.stride
                assert ams % self.stride == 0
        self.stride_mask_align_global = stride_mask_align_global
        self.rand_shift_stride_mask = rand_shift_stride_mask
        self.stride_shift_rng = np.random.default_rng(seed=seed)
        self.mask_stride_rng = np.random.default_rng(seed=seed)
        if self.rand_shift_stride_mask is True:
            assert self.stride_mask_align_global is False
        self.subset = subset

        if self.flip_augment is True:
            assert flip_lr_indices is not None

        self._world_kpts3d_fp = kwargs.get("world_kpts3d_path")
        self._kpts2d_fp = kwargs.get("kpts2d_path")
        self._cam_kpts3d_fp = kwargs.get("cam_kpts3d_path")
        self._cam_rot_fp = kwargs.get("cam_rot_path")
        self._cam_fp = kwargs.get("cam_path")
        self.rotation_type = rotation_type

        self._load_smplx_params = load_smplx_params
        self._betas_fp = kwargs.get("betas_path")
        if self._load_smplx_params:
            assert self._betas_fp

        # Load 2d, 3d poses and other info
        self._load_data()

        # Generate all central locations of sequences.
        self.seq_locations = self._generate_seq_locations()

    def __len__(self):
        return len(self.seq_locations)

    def __getitem__(self, item):
        s_i, i, do_flip, frame_rate = self.seq_locations[item]

        frame_rate = int(frame_rate)
        stride = self.stride # s_out
        mult = 1
        assert frame_rate % self.target_frame_rate == 0
        if frame_rate != self.target_frame_rate: # adjust stride to match target frame rate
            mult = frame_rate // self.target_frame_rate
            stride *= mult

        if self.abs_mask_stride is None:
            abs_mask_stride = stride # train with s_in = s_out
        else:
            if len(self.abs_mask_stride) == 1:
                abs_mask_stride = self.abs_mask_stride[0]
            else: # randomly choose one of the possible input strides
                abs_mask_stride = self.abs_mask_stride[
                self.mask_stride_rng.integers(
                    low=0, high=len(self.abs_mask_stride), endpoint=False)]
            abs_mask_stride *= mult # adjust to target frame rate

        # relative stride between output and input sequences
        mask_stride = abs_mask_stride // stride
        # the real sequence length is (self.seq_len - 1) * stride + 1,
        # self.seq_len is the real number of frames that is used
        # number of frames to consider to the left
        left = (self.seq_len - 1) * stride // 2
        # number of frames to consider to the right
        right = (self.seq_len - 1) * stride - left

        do_flip = do_flip == 1.
        video_3d, video_2d = self.poses_3d[s_i], self.poses_2d[s_i]
        camera = self.camera_params[s_i]
        subject, action = self.subjects[s_i], self.actions[s_i]
        if self.rotations:
            video_rot = self.rotations[s_i]
        if self.betas:
            beta = self.betas[s_i]
        if self.transl:
            transl = self.transl[s_i]

        video_len = video_3d.shape[0]
        begin, end = i - left, i + right + 1 # begin and end frame index for selected central frame
        pad_left, pad_right = 0, 0
        if begin < 0: # we would need more poses at the beginning, therefore we need to pad the sequence on the left side
            pad_left = math.ceil(-begin / stride)
            last_pad = begin + ((pad_left - 1) * stride)
            begin = last_pad + stride

        if end > video_len: # we would need more poses at the end, therefore we need to pad the sequence on the right side
            pad_right = math.ceil((end - video_len) / stride)
            first_pad = end - ((pad_right - 1) * stride)
            end = first_pad - stride

        # Base case:
        seq_3d = video_3d[begin: end: stride]
        seq_2d = video_2d[begin: end: stride]
        mask = np.ones(seq_3d.shape[0], dtype=np.float32)

        seq_rot = video_rot[begin: end: stride] if self.rotations else None
        seq_beta = seq_beta = beta[begin: end: stride] if self.betas else None
        seq_transl = transl[begin: end: stride] if self.transl else None

        # Pad if necessary
        if pad_left > 0 or pad_right > 0:
            seq_3d = self._padding(seq_3d, pad_left, pad_right)
            seq_2d = self._padding(seq_2d, pad_left, pad_right)
            mask = self._padding(mask, pad_left, pad_right)

            if seq_rot is not None:
                seq_rot = self._padding(seq_rot, pad_left, pad_right)

            if seq_beta is not None:
                seq_beta = self._padding(seq_beta, pad_left, pad_right)

        # Generate stride mask that is centered on the central frame
        stride_mask = self._generate_stride_mask(stride, mask_stride, abs_mask_stride, i)

        assert seq_3d.shape[0] == self.seq_len
        assert seq_2d.shape[0] == self.seq_len
        assert mask.shape[0] == self.seq_len
        assert stride_mask.shape[0] == self.seq_len

        if seq_rot is not None:
            assert seq_rot.shape[0] == self.seq_len

        # # disabled augmentation except for WBA
        # if do_flip:
        #     seq_3d, seq_2d, seq_rot, camera = self._flipping(seq_3d, seq_2d, seq_rot, camera)

        if self._load_smplx_params:
            return seq_3d, seq_2d, seq_rot, mask, camera, subject, action, \
                i, stride_mask, seq_beta
        else:
            return seq_3d, seq_2d, mask, camera, subject, action, i, stride_mask

    def get_betas(self, idx: int) -> np.ndarray:
        s_i, i, do_flip, frame_rate = self.seq_locations[idx]

        frame_rate = int(frame_rate)
        stride = self.stride # s_out
        mult = 1
        assert frame_rate % self.target_frame_rate == 0
        if frame_rate != self.target_frame_rate: # adjust stride to match target frame rate
            mult = frame_rate // self.target_frame_rate
            stride *= mult

        if self.abs_mask_stride is None:
            abs_mask_stride = stride # train with s_in = s_out
        else:
            if len(self.abs_mask_stride) == 1:
                abs_mask_stride = self.abs_mask_stride[0]
            else: # randomly choose one of the possible input strides
                abs_mask_stride = self.abs_mask_stride[
                self.mask_stride_rng.integers(
                    low=0, high=len(self.abs_mask_stride), endpoint=False)]
            abs_mask_stride *= mult # adjust to target frame rate

        mask_stride = abs_mask_stride // stride
        left = (self.seq_len - 1) * stride // 2
        right = (self.seq_len - 1) * stride - left

        video_3d = self.poses_3d[s_i]
        beta = self.betas[s_i]

        video_len = video_3d.shape[0]
        begin, end = i - left, i + right + 1
        pad_left, pad_right = 0, 0
        if begin < 0:
            pad_left = math.ceil(-begin / stride)
            last_pad = begin + ((pad_left - 1) * stride)
            begin = last_pad + stride

        if end > video_len:
            pad_right = math.ceil((end - video_len) / stride)
            first_pad = end - ((pad_right - 1) * stride)
            end = first_pad - stride

        seq_beta = beta[begin: end: stride]
        seq_beta = self._padding(seq_beta, pad_left, pad_right)
        stride_mask = self._generate_stride_mask(stride, mask_stride, abs_mask_stride, i)

        return seq_beta, stride_mask

    def _flipping(self, seq_3d: np.ndarray, seq_2d: np.ndarray, seq_rot: np.ndarray, camera: np.ndarray) -> tuple:
        # Width (or x coord) is 0 centered, so flipping is simply sign inversion
        seq_3d = seq_3d[:, self.flip_lr_indices].copy()
        seq_3d[..., 0] *= -1
        seq_2d = seq_2d[:, self.flip_lr_indices].copy()
        seq_2d[..., 0] *= -1
        camera = camera.copy()
        # Flip cx (principal point)
        camera[0][4] *= -1
        # Flip t2 (tangential distortion)
        camera[0][9] *= -1
        return seq_3d, seq_2d, seq_rot, camera

    def _padding(self, seq: np.ndarray, pad_left: int, pad_right: int) -> np.ndarray:
        if len(seq.shape) == 3:
            seq = np.pad(
                array=seq,
                pad_width=((pad_left, pad_right), (0, 0), (0, 0)),
                mode=self.pad_type,
            ) # type: ignore
        elif len(seq.shape) == 4:
            seq = np.pad(
                array=seq,
                pad_width=((pad_left, pad_right), (0, 0), (0, 0), (0, 0)),
                mode=self.pad_type,
            ) # type: ignore
        elif len(seq.shape) == 2:
            seq = np.pad(
                array=seq,
                pad_width=((pad_left, pad_right), (0, 0)),
                mode=self.pad_type,
            ) # type: ignore
        else:
            # numpy constant padding defaults to 0 values
            seq = np.pad(seq, (pad_left, pad_right), mode="constant")
        return seq

    def _generate_stride_mask(self, stride: int, mask_stride, abs_mask_stride, i) -> np.ndarray:
        mid_index = self.seq_len // 2 # index of the central frame in sequence_2d and 3d
        seq_indices = np.arange(0, self.seq_len) - mid_index
        seq_indices *= stride # video frame indices relative to the central frame
        if self.stride_mask_align_global is True: # real (absolute) video frame indices!
            # Shift mask such that it is aligned on the global frame indices
            # This is required for inference mode
            seq_indices += i

        elif self.rand_shift_stride_mask is True:
            # Shift stride mask randomly by [ceil(-mask_stride/2), floor(mask_stride/2)]
            max_shift = int(np.ceil((mask_stride - 1) / 2))
            endpoint = mask_stride % 2 != 0 # include max_shift in the range
            rand_shift = self.stride_shift_rng.integers(low=-max_shift, high=max_shift, endpoint=endpoint)
            rand_shift *= stride # sequence indices are in (relative) video frame indices, so we need to shift relative to the stride
            seq_indices += rand_shift # shift the used frames randomly such that the central frame can also be masked

        stride_mask = np.equal(seq_indices % abs_mask_stride, 0) # create mask for used and unused frames in the 2d sequence
        # stride mask contains False for every unused 2d pose and True for every used pose
        return stride_mask

    def _load_data(self) -> None:
        selected_subjects = splits.subjects_by_split[self.subset]
        dataset_3d, poses_2d_dataset = self.load_npz_data()
        (
            self.camera_params,
            self.poses_3d,
            self.poses_2d,
            self.rotations,
            self.frame_names,
            self.subjects,
            self.actions,
            self.frame_rates,
            self.conversion_dicts,
            self.betas,
            self.transl,
        ) = Fit3DSequenceGenerator.filter_and_subsample_dataset(
            dataset=dataset_3d,
            poses_2d=poses_2d_dataset,
            subjects=selected_subjects,
            action_filter="*",
            downsample=1,
        )

    def _generate_seq_locations(self) -> np.ndarray:
        seq_locations = []
        for s_i, video_3d in enumerate(self.poses_3d):
            assert len(video_3d) == len(self.poses_2d[s_i])
            # positions are all possible frames of the video. Will be used as the target central position later
            positions = np.arange(start=0, stop=len(video_3d), step=self.subsample)
            num_frames = positions.shape[0]
            seq_num = np.tile([s_i], reps=(num_frames))
            frame_rates_tiled = np.tile([self.frame_rates[s_i]], reps=num_frames)
            do_flip = np.zeros(shape=(num_frames), dtype=positions.dtype)
            # add flipped version already to the sequence locations
            if self.flip_augment:
                seq_num = np.concatenate([seq_num, seq_num], axis=0)
                frame_rates_tiled = np.concatenate([frame_rates_tiled, frame_rates_tiled], axis=0)
                positions = np.concatenate([positions, positions], axis=0)
                do_flip = np.concatenate([do_flip, 1 - do_flip], axis=0)

            seq_locations.append(np.stack([seq_num, positions, do_flip, frame_rates_tiled], axis=-1))

        seq_locations = np.concatenate(seq_locations, axis=0)
        return seq_locations

    # TODO: tangential distortion
    def flip_batch(self, seq_3d, seq_2d, camera):
        # Width (or x coord) is 0 centered, so flipping is simply sign inversion
        seq_3d = seq_3d[:, :, self.flip_lr_indices].clone()
        seq_3d[..., 0] *= -1
        seq_2d = seq_2d[:, :, self.flip_lr_indices].clone()
        seq_2d[..., 0] *= -1
        camera = camera.clone()
        # Flip cx (principal point)
        camera[4] *= -1
        # Flip t2 (tangential distortion)
        camera[9] *= -1
        return seq_3d, seq_2d, camera

    def flip_rotations(self, seq_rot):
        flipped_idx = Fit3DOrder26P._flipped_indices
        if len(seq_rot.shape) == 4:
            if seq_rot.shape[-1] == 3:
                seq_rot_fl = flip_axis_angle(seq_rot, self.flip_lr_indices, flipped_idx)
            else:
                seq_rot_fl = flip_quaternion(seq_rot, self.flip_lr_indices, flipped_idx)
        else:
            # case for rotation matrices
            # didnt found a plausible way to flip directly as rot mat, so fliped through axis-angle
            seq_rot_fl = matrix_to_axis_angle(seq_rot)
            seq_rot_fl = flip_axis_angle(seq_rot_fl, self.flip_lr_indices, flipped_idx)
            seq_rot_fl = axis_angle_to_matrix(seq_rot_fl)
        return seq_rot_fl

    def load_npz_data(self) -> tuple:
        """
        Load VP3d-style 3D pose dataset, along with fitting 2D poses
        :return: dataset: MocapDataset, keypoints
        """

        logging.info(f"Loading 3D dataset from\n\t{self._world_kpts3d_fp}"
              f"&\n\t{self._cam_kpts3d_fp}"
              f"&\n\t{self._cam_rot_fp}")
        dataset = Fit3dDataset(
            kpts3d_path=self._world_kpts3d_fp,
            kpts3d_cam_path=self._cam_kpts3d_fp,
            kpts2d_path=self._kpts2d_fp,
            rots_cam_path=self._cam_rot_fp,
            camdata_path=self._cam_fp,
            betas_path=self._betas_fp,
            rot_type=self.rotation_type,
        )

        logging.info(f"Loading 2D poses from {self._kpts2d_fp}")
        keypoints = dataset.load_2d_keypoints()

        for subject in dataset.subjects():
            assert subject in keypoints, \
                f"{subject=} is missing from the 2D detections dataset"
            for action in dataset[subject].keys():
                assert action in keypoints[subject], \
                    f"{action=} of {subject=} is missing from the 2D dataset"

                if "positions_3d" not in dataset[subject][action]:
                    continue

                # for each action, there is one global 3D pose and 2D pose per camera
                for cam_idx in range(len(keypoints[subject][action])):

                    mocap_len = dataset[subject][action]["positions_3d"][cam_idx].shape[0]
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_len

                    if keypoints[subject][action][cam_idx].shape[0] > mocap_len:
                        # Shorten sequence
                        keypoints[subject][action][cam_idx] = \
                            keypoints[subject][action][cam_idx][:mocap_len]

                assert len(keypoints[subject][action]) == \
                    len(dataset[subject][action]["positions_3d"])

        logging.info("Normalizing 2D poses to [-1, 1].")
        for subject in keypoints.keys():
            for action in keypoints[subject]:
                for cam_idx, kps in enumerate(keypoints[subject][action]):
                    # Normalize camera frame
                    kps[..., :2] = normalize_screen_coordinates(
                        kps[..., :2], w=RES_W, h=RES_H
                    )
                    keypoints[subject][action][cam_idx] = kps

        return dataset, keypoints

    @staticmethod
    def filter_and_subsample_dataset(
        dataset: MocapDataset,
        poses_2d: np.ndarray,
        subjects: list,
        action_filter: str,
        downsample: int = 1,
    ):
        """_summary_

        Args:
            dataset (MocapDataset): Dataset with data groupped per action and subject
            poses_2d (np.ndarray): 2D normalised keypoints.
            subjects (list): subjects to include.
            action_filter (str): actions to include.
            downsample (int, optional): downsample the framerate by this int-factor.
                Defaults to 1 -> no downsample.

        Returns:
            tuple: separate lists for 3d, 2d keypoints, camera parameters, frame
                names, frame rates, subjects, actions, SMPLX keypoint rotations
                and SMPLX beta parameters.
        """

        logging.info(f"Filtering subjects: {subjects}")

        action_filter = None if action_filter == "*" else action_filter
        if action_filter is not None:
            logging.info(f"Filtering actions: {action_filter}")

        out_poses_3d = []
        out_poses_2d = []
        out_rot_cam = []
        out_cam_params = []
        out_frame_names = []
        out_subjects = []
        out_actions = []
        out_frame_rates = []
        out_betas = []
        out_transl = []

        # Mapping of names to indices
        subject_dict = {name: i for i, name in enumerate(splits.all_subjects)}
        action_dict = {name: i for i, name in enumerate(splits.actions)}


        for subject in subjects:
            for action in poses_2d[subject].keys():
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action == a:
                            found = True
                            break
                    if not found:
                        continue

                poses_2d_seqs = poses_2d[subject][action]
                for i in range(len(poses_2d_seqs)):  # Iterate over cameras
                    out_poses_2d.append(poses_2d_seqs[i].copy())
                    out_subjects.append(subject_dict[subject])
                    out_actions.append(action_dict[action])

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject][action]
                    assert len(cams) == len(poses_2d_seqs), \
                        "Camera count mismatch 'poses_2d'"
                    for cam in cams:
                        # out_cam_params.append(cam["id"])
                        if "intrinsic" in cam:
                            # return both intristics and cam_id for easier processing later
                            out_cam_params.append([cam["intrinsic"].copy(), cam["id"]])

                if "positions_3d" in dataset[subject][action]:
                    poses_3d_seqs = dataset[subject][action]["positions_3d"]
                    assert len(poses_3d_seqs) == len(poses_2d_seqs), \
                        "Camera count mismatch 'poses_3d'"
                    for i in range(len(poses_3d_seqs)):  # Iterate over cameras
                        out_poses_3d.append(poses_3d_seqs[i].copy())
                        if "frame_rate" in dataset[subject][action].keys():
                            frame_rate = dataset[subject][action]["frame_rate"]
                        else:
                            frame_rate = 50
                        out_frame_rates.append(frame_rate)

                if "rotations" in dataset[subject][action]:
                    rot_seqs = dataset[subject][action]["rotations"]
                    assert len(rot_seqs) == len(poses_2d_seqs), \
                        "Camera count mismatch 'rotations'"
                    for i in range(len(rot_seqs)):  # Iterate over cameras
                        out_rot_cam.append(rot_seqs[i].copy())

                if "betas" in dataset[subject][action]:
                    beta_seq = dataset[subject][action]["betas"]
                    assert len(beta_seq) == len(poses_2d_seqs), \
                        "Camera count mismatch 'betas'"
                    for i in range(len(beta_seq)):  # Iterate over cameras
                        if isinstance(beta_seq[i], torch.Tensor):
                            out_betas.append(beta_seq[i].numpy().copy())
                        else:
                            out_betas.append(beta_seq[i].copy())

                if "transl" in dataset[subject][action]:
                    transl_seq = dataset[subject][action]["transl"]

                    assert len(transl_seq) == len(poses_2d_seqs), \
                        "Camera count mismatch 'transl'"
                    for i in range(len(transl_seq)):  # Iterate over cameras
                        if isinstance(transl_seq[i], torch.Tensor):
                            out_transl.append(transl_seq[i].numpy().copy())
                        else:
                            out_transl.append(transl_seq[i].copy())

        # return None if len is 0
        out = []
        result_lists = [
            out_cam_params,
            out_poses_3d,
            out_frame_names,
            out_frame_rates,
            out_rot_cam,
            out_betas,
            out_transl
        ]
        for x in result_lists:
            out.append(None if len(x) == 0 else x)
        (
            out_cam_params,
            out_poses_3d,
            out_frame_names,
            out_frame_rates,
            out_rot_cam,
            out_betas,
            out_transl
        ) = out

        if downsample > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::downsample]
                if out_poses_3d:
                    out_poses_3d[i] = out_poses_3d[i][::downsample]
                if out_frame_names:
                    out_frame_names[i] = out_frame_names[i][::downsample]
                if out_rot_cam:
                    out_rot_cam[i] = out_rot_cam[i][::downsample]
                if out_betas:
                    out_betas[i] = out_betas[i][::downsample]
        return (
            out_cam_params,
            out_poses_3d,
            out_poses_2d,
            out_rot_cam,
            out_frame_names,
            out_subjects,
            out_actions,
            out_frame_rates,
            (subject_dict, action_dict),
            out_betas,
            out_transl
        )

