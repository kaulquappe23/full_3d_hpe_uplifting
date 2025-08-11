# -*- coding: utf-8 -*-
"""
Created on 08.07.25

@author: Katja

"""
import json
import os
import shutil

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import paths
import argparse
import logging
import random
import cv2

from dataset.fit3d.keypoint_order import Fit3DOrder26P
from run.eval import load_model, regress_joints
from visualization.a2b.inference import get_betas
from visualization.extract_frames import extract_frames_ffmpeg
from visualization.inference_dataloader import get_inference_dataloader
from visualization.inference_model import execute_inference

from model import UpliftPoseConfig
from utils.rotation import RotType

from model import SMPLX_Layer
from visualization.visualize import plot_smplx_and_keypoints_matplotlib, create_video, render_pose
from vitpose.vitpose_onnx import execute_vitpose_inference

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def parse_float_list(input_string):
    float_values = []
    parts = input_string.strip().split(",")

    for part in parts:
        cleaned_part = part.strip()
        if not cleaned_part:
            continue  # Skip empty strings that might result from multiple delimiters
        try:
            # Attempt to convert the cleaned part to a float
            float_value = float(cleaned_part)
            float_values.append(float_value)
        except ValueError:
            print(f"Warning: Could not convert '{cleaned_part}' to a float. Skipping this value.")

    return float_values

def vis(rot, beta, cam, kpts, smplx_layer, device, output_dir, pelvis_pos=None, matplotlib=False, images=None):
    smplx_res = smplx_layer.from_axis_angle(rot, beta, torch.float32, device)
    vertices = smplx_res.vertices.detach().cpu().numpy()
    faces = smplx_layer.bm_x.faces

    min_coord = np.min(vertices)
    max_coord = np.max(vertices)
    coord = max_coord if max_coord > np.abs(min_coord) else np.abs(min_coord)
    coord += 0.25
    axes_lims = [-coord, coord, -coord, coord, -coord, coord]
    if pelvis_pos is not None:
        cur_roots = kpts[:, 0]
        real_offset = - cur_roots + pelvis_pos
        vertices += real_offset[:, None]

    for i in tqdm(range(0, rot.shape[0], 1), colour="GREEN", desc="Rendering plots"):
        if matplotlib:
            plot_smplx_and_keypoints_matplotlib(vertices[i], faces, kpts[i], axes_lims=axes_lims,
                                                save_path=f"{output_dir}{os.sep}plot_{i+1:05d}.png",
                                                view_azim=cam)
        else:
            if images is None:
                render_pose(np.zeros((900,900,3)), vertices[i], faces, cam, save_path=f"{output_dir}{os.sep}rendered_{i+1:05d}.png")
            else:
                render_pose(images[i], vertices[i], faces, cam, save_path=f"{output_dir}{os.sep}rendered_{i+1:05d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", type=str, required=True)
    parser.add_argument("-w", "--weights", type=str, required=True)
    parser.add_argument("-v", "--video", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=False)
    parser.add_argument("-b", "--betas", type=str, required=False)
    parser.add_argument("--matplotlib", action="store_true", help="If set, the visualization is done with matplotlib instead of pyrender, which takes a lot more time.")
    parser.add_argument("--smoothing", action="store_true", help="If set, a smoothing is performed across the single frame results.")
    parser.add_argument("-a", "--anthros", type=str, required=False, help="Path to a json file containing a dictionary mapping from anthropometric name to value. An example is provided in example_anthros.json")
    parser.add_argument("--cam", type=str, required=False, help="pyrender: [fx, fy, cx, cy] of the camera. If not set, a default camera is used. matplotlib: azimuth angle for the camera, if not set, a standard of 30 is used. ")
    parser.add_argument("--pelvis_pos", type=str, required=False, help="pyrender: to reproject to the image, the pelvis positions are needed. If not set, the rendering is done without the image. Provide the values as a npy file.")
    parser.add_argument("--fit3d", type=str, required=False, help="Currently, the onnx export does not work correctly. To load the vitpose results from a file, set this path. It should be the npz file that you can download as describedin the readme.")
    args = parser.parse_args()

    if not args.output:
        args.output = os.path.dirname(args.video)
    os.makedirs(args.output, exist_ok=True)
    if os.path.exists(os.path.join(args.output, "frames")):
        shutil.rmtree(os.path.join(args.output, "frames"))
    os.makedirs(os.path.join(args.output, "frames"), exist_ok=True)

    extract_frames_ffmpeg(args.video, os.path.join(args.output, "frames"), fps=50)
    first_im = Image.open(os.path.join(args.output, "frames", "frame_00001.jpg"))
    w, h = first_im.size

    video_path = args.video

    images = sorted([os.path.join(args.output, "frames", f) for f in os.listdir(os.path.join(args.output, "frames"))])

    # args.fit3d has to be paths.FIT3D_PROCESSED["kpts2d_path"] if the file should be loaded
    if args.fit3d:
        try:
            video_split = video_path.split(os.sep)
            video_name = video_split[-1][:-4]
            camera_name = video_split[-2]
            subject_name = video_split[-4]
            cam_nums = ["50591643", "58860488", "60457274", "65906101"]
            cam_id = cam_nums.index(camera_name)
        except Exception as e:
            print("For using the prepared fit3d vitpose files, the filename of the video has to contain camera, action and subject.")
            raise e

        kpts_2d_vitpose = np.load(args.fit3d, allow_pickle=True)["positions_2d"].item()[subject_name][video_name][cam_id]
    else:
        kpts_2d_vitpose = execute_vitpose_inference(images, visualize=False)

    frame_keypoints = []

    # For saving pelvis pos hip locations:
    # offset = np.load(paths.FIT3D_PROCESSED["cam_kpts3d_path"], allow_pickle=True
    #                           )["positions_3d"].item()[subject_name][video_name][cam_id][:, 0]
    # with open(os.path.join(args.output, "offset.npy"), "wb") as f:
    #     np.save(f, offset)


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                mode="w",
                filename=os.path.join(args.output, "visualization.log"))
    ])
    logging.info(f"Logging to {args.output}...")
    logging.info(f"Evaluation with model weights {args.weights}...")

    cfg = UpliftPoseConfig(config_file=args.cfg)
    cfg.RES_H = h
    cfg.RES_W = w

    # make enum from string
    if hasattr(cfg, "ROT_REP") and cfg.ROT_REP:
        cfg.OUT_DIM = RotType[cfg.ROT_REP].value
        cfg.ROT_REP = RotType[cfg.ROT_REP]

    cfg.KEYPOINT_ORDER = eval(cfg.KEYPOINT_ORDER)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.DEVICE = device

    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)
    np.random.seed(cfg.SEED)
    random.seed(cfg.SEED)

    model = load_model(cfg, args.weights)

    smplx_layer = SMPLX_Layer(kpts_all=True, keypoint_order=cfg.KEYPOINT_ORDER)
    smplx_layer = smplx_layer.to(device)

    loader = get_inference_dataloader(kpts_2d_vitpose, cfg)

    if not args.betas and not args.anthros:
        logging.info("No beta parameters and no anthropometric measurements given. Using template human mesh...")
        args.betas = torch.zeros(10)
    elif args.betas:
        assert args.betas[0] == "[" and args.betas[-1] == "]"
        betas = parse_float_list(args.betas[1:-1])
        args.betas = torch.from_numpy(np.asarray(betas))
    elif args.anthros:
        with open(args.anthros, "r") as f:
            subject_dict = json.load(f)
        subject_name = list(subject_dict.keys())[0]
        get_betas(args.anthros, os.path.join(args.output, "beta_params.json"))
        with open(os.path.join(args.output, "beta_params.json"), "r") as f:
            beta_res = json.load(f)
            betas = beta_res["visualize_A2B_nn_neutral"][subject_name][:10]
            args.betas = torch.from_numpy(np.asarray(betas))
    args.betas = args.betas.to(cfg.DEVICE)

    logging.info(f"Collecting predictions on {len(loader)} batches")
    results = execute_inference(loader, model, cfg, betas=args.betas, smoothing=args.smoothing)

    logging.info(f"Regressing predicted kpts through SMPL-X")
    betas = args.betas.repeat(results.shape[0], 1).float()
    pred_kpts3d_all = regress_joints(results, betas, cfg.BATCH_SIZE, smplx_layer)
    pred_kpts3d = pred_kpts3d_all[:, Fit3DOrder26P.smplx_joints()]

    if args.matplotlib:
        cam = 30 if not args.cam else int(args.cam)
        logging.info(f"Rendering plots... This will take a while")
        pelvis_pos = None
    else:
        cam = parse_float_list(args.cam[1:-1]) if args.cam else [1095.4581298828125, 1089.53662109375, 469.0216979980469, 462.4344787597656] # example camera stolen from fit3d
        logging.info(f"Rendering plots...")
        pelvis_pos = args.pelvis_pos if args.pelvis_pos else np.asarray([-0.20148872, -0.1755374, 4.482773])[None, :] # example pelvis positions that fit camera, stolen, too
        if args.pelvis_pos:
            with open(args.pelvis_pos, "rb") as f:
                pelvis_pos = np.load(f)

    images = None
    if args.pelvis_pos and not args.matplotlib and args.cam:
        im_vid = sorted([os.path.join(args.output, "frames", img) for img in os.listdir(os.path.join(args.output, "frames")) if
                         img.endswith(".jpg") and img.startswith("frame")])
        images = [cv2.cvtColor(cv2.imread(vid_path), cv2.COLOR_RGB2BGR) for vid_path in im_vid]
    shutil.rmtree(os.path.join(args.output, "plots"))
    os.makedirs(os.path.join(args.output, f"plots"), exist_ok=True)
    vis(results, betas, cam, pred_kpts3d, smplx_layer, cfg.DEVICE, os.path.join(args.output, f"plots"), pelvis_pos, args.matplotlib, images)

    logging.info(f"Creating combined video...")
    create_video(args.output, h, w)







