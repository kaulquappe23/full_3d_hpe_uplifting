import onnx
import torch
import onnxruntime as rt
import cv2
import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt

from dataset.fit3d.keypoint_order import Fit3DOrder26P
from vitpose import helpers
from tqdm import tqdm


def toTensor(img_orig):
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    mean = mean.reshape(1, 1, *mean.shape)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    std = std.reshape(1, 1, *std.shape)

    img = img_orig.astype(np.float32) / 255
    img = img - mean
    img = img / std
    img = torch.Tensor(img)
    img = img.permute(2,0,1)
    #img = img.reshape(-1,*img.shape)
    img = img.numpy()
    return img



def execute_vitpose_inference(img_files, visualize=False, batch_size=32):
    # Constants
    bbox_thr = 0.2
    det_cat_id = 0

    images = []

    for img_file in tqdm(img_files, colour="CYAN", desc="Loading frames in memory"):
        img = cv2.imread(img_file)
        max_size_length = max(img.shape)
        if max_size_length > 1024:
            new_w = int(img.shape[1] * 1024 / max_size_length)
            new_h = int(img.shape[0] * 1024 / max_size_length)
            img = cv2.resize(img, dsize=(new_w, new_h))
        #img_orig = cv2.resize(img, dsize=(640,416))
        img_orig = img[:img.shape[0] // 16 * 16, : img.shape[1] // 16 * 16,:]
        img = toTensor(img_orig)
        images.append(img)
    images = np.stack(images)

    print(f"ONNX is running on {rt.get_device()}")
    mmdet_onnx_file = "vitpose/models_vitpose/rtmdet_x_8xb32-300e_coco.onnx"
    sess_options = rt.SessionOptions()
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_mmdet = rt.InferenceSession(mmdet_onnx_file, sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    mmpose_onnx_file = "vitpose/models_vitpose/td-hm_ViTPose-large_8xb64-210e_coco-256x192_fit3D_s10.onnx"
    sess_mmpose = rt.InferenceSession(mmpose_onnx_file, sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    frames_boxes = []
    num_batches = int(np.ceil(images.shape[0] / batch_size))
    h, w = images.shape[2:]
    for i in tqdm(range(num_batches), colour="CYAN", desc="Person detection"):
        end_batch = (i + 1) * batch_size if (i + 1) * batch_size < len(images) else len(images)
        in_batch = images[i * batch_size: end_batch]
        out_mmdet = sess_mmdet.run(None, {"input": in_batch})
        object_boxes, object_ids = out_mmdet # First dimension is the number of images
        batch_size = object_boxes.shape[0]
        for b in range(batch_size):
            boxes, boxIds = out_mmdet[0][b], out_mmdet[1][b]
            frame_boxes = {}
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2, score = box
                if boxIds[idx] == det_cat_id and score > bbox_thr and 0 <= x1 < w and 0 <= x2 < w and 0 <= y1 < h and 0 <= y2 < h:
                    frame_boxes[box[4]] = box[:4]
            if len(frame_boxes) == 0:
                if len(frames_boxes) > 0:
                    frames_boxes.append(np.copy(frames_boxes[-1]))
                else:
                    frames_boxes.append([-1, -1, -1, -1])
            else:
                max_score = max(list(frame_boxes.keys()))
                frames_boxes.append(frame_boxes[max_score])

    all_keypoints = []
    for i, (person_box, img_orig) in enumerate(tqdm(zip(frames_boxes, images), colour="CYAN", desc="2D Keypoint detection")):
        result = dict()
        result['image_size'] = [192,256] #img_orig.shape[:2] # OpenCV image with (w, h, b)
        x1, y1, x2, y2 = person_box
        result['bbox'] = person_box
        result['bbox'][0] = x1 # in x, y, w, h
        result['bbox'][1] = y1 # in x, y, w, h
        result['bbox'][2] = x2-x1
        result['bbox'][3] = y2-y1
        result['img'] = img_orig.transpose(1, 2, 0)
        prep = helpers.TopDownGetBboxCenterScale()
        result = prep(result)
        tt= helpers.TopdownAffine((192,256), use_udp=True)

        tt.transform(result)

        # img_roi = toTensor(result['img'])
        img_roi = result['img'].transpose(2, 0, 1)
        out_mmpose = sess_mmpose.run(None, {"input": [img_roi]})
        heatmaps = out_mmpose[0][0]

        keypoints, scores = helpers.get_heatmap_maximum(heatmaps)
        # unsqueeze the instance dimension for single-instance results
        keypoints = keypoints[None]
        scores = scores[None]
        blur_kernel_size: int = 11
        keypoints = helpers.refine_keypoints_dark_udp(
            keypoints, heatmaps, blur_kernel_size=blur_kernel_size)
        W, H = heatmaps.shape[-1], heatmaps.shape[-2]
        keypoints = keypoints / [W - 1, H - 1] * [img_roi.shape[-1], img_roi.shape[-2]]

        # Show the output in the original image
        final_keypoints = []
        for keypoint in keypoints[0]:
            kp = keypoint - np.array(result['image_size'], dtype=np.float32) / 2
            scale_factor = np.array(result['bbox_scale'], dtype=np.float32) / np.array(result['image_size'], dtype=np.float32)
            kp_full_image = np.multiply(kp, scale_factor) + result['bbox_center']
            final_keypoints.append(kp_full_image)

        keypoints = np.stack(final_keypoints)
        final_keypoints = keypoints[Fit3DOrder26P.vitpose_rl_joints()]

        if visualize:
            img = np.copy(img_orig.transpose(1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406], np.float32)
            mean = mean.reshape(1, 1, *mean.shape)
            std = np.array([0.229, 0.224, 0.225], np.float32)
            std = std.reshape(1, 1, *std.shape)

            img = img * std
            img = img + mean
            img = img * 255
            img = np.array(img, np.uint8)
            if i < 128:
                cv2.circle(img, keypoints[i].astype(np.int32), 1, (0, 0, 255), 1)
                cv2.putText(img, f"{i}", keypoints[i].astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for keypoint in final_keypoints:
                cv2.circle(img, keypoint.astype(np.int32), 3, (0, 255, 0), 3)
            cv2.imwrite(img_files[i].replace("frames", "vitpose"), img)

        all_keypoints.append(final_keypoints)
    return np.stack(all_keypoints)

