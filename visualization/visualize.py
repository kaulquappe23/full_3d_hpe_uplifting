# -*- coding: utf-8 -*-
"""
Created on 01.07.25

@author: Katja

"""
import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_smplx_and_keypoints_matplotlib(
        vertices,
        faces,
        keypoints,
        save_path=None,
        view_elev=0,  # Elevation angle for camera (degrees)
        view_azim=0,  # Azimuth angle for camera (degrees)
        face_color=[0.8, 0.8, 0.8, 0.5],  # Light grey, highly transparent
        keypoint_color='r',  # Red for keypoints
        keypoint_size=50,  # Significantly larger keypoint markers
        axes_lims=(-1, 1, -1, 1, -1, 1),
        rerender=False
        ):
    """
    Renders an SMPL-X model and 3D keypoints using Matplotlib.
    Swaps Y and Z axes. Ensures equal axis scaling. Improves keypoint visibility.
    """
    if not rerender and os.path.exists(save_path):
        return
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- AXIS SWAPPING & Optional Inversion ---
    # Original: (X, Y, Z)
    # Desired in plot: (X, Z_orig, Y_orig) where Y_orig is Matplotlib's Z-axis (up)

    # Apply Y-axis inversion first if requested (to flip model upright)
    vertices_modified_y = vertices.copy()
    vertices_modified_y[:, 1] *= -1
    keypoints_modified_y = keypoints.copy()
    keypoints_modified_y[:, 1] *= -1

    # Now perform the Y and Z swap for Matplotlib's coordinate system
    transformed_vertices = vertices_modified_y[:, [0, 2, 1]]  # Swap Y and Z columns
    transformed_keypoints = keypoints_modified_y[:, [0, 2, 1]]  # Swap Y and Z columns

    # 1. Plot the SMPL-X Mesh
    mesh_faces = transformed_vertices[faces]
    poly = Poly3DCollection(
            mesh_faces,
            facecolors=face_color,
            linewidths=0.5,
            edgecolors=face_color,
            alpha=face_color[3]
            )
    ax.add_collection3d(poly)

    # 2. Plot the 3D Keypoints
    if transformed_keypoints is not None and len(transformed_keypoints) > 0:
        ax.scatter(
                transformed_keypoints[:, 0],
                transformed_keypoints[:, 1],  # Now this is original Z
                transformed_keypoints[:, 2],  # Now this is original Y (Matplotlib's Z-axis)
                c=keypoint_color,
                s=keypoint_size,
                label='Keypoints',
                depthshade=False
                )

    # Set limits for all axes based on the maximum range, centered around midpoints
    ax.set_xlim(axes_lims[0], axes_lims[1])
    ax.set_ylim(axes_lims[2], axes_lims[3])
    ax.set_zlim(axes_lims[4], axes_lims[5])

    # Matplotlib's `set_box_aspect` is the most reliable way for equal aspect in 3D
    # It sets the physical dimensions of the bounding box.
    ax.set_box_aspect((1, 1, 1))  # All sides of the box are equal length in display units

    # 4. Set Camera View
    ax.view_init(elev=view_elev, azim=view_azim)

    # 5. Set Labels and Title
    ax.set_xlabel('X')
    ax.set_ylabel('Z')  # Matplotlib's Y-axis is now original Z
    ax.set_zlabel('Y')  # Matplotlib's Z-axis is now original Y (up)

    # 6. Save or Show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)
        # print(f"Plot saved to {save_path}")
    else:
        plt.show()


def render_pose(img, vertices, faces, intrinsics, return_mask=False, alpha=1, color=(0.7, 0.7, 0.7, 1), save_path=None):
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import pyrender
    import trimesh

    cam_pose = np.eye(4)
    rot = trimesh.transformations.euler_matrix(0, np.pi, np.pi, 'rxyz')
    cam_pose[:3, :3] = rot[:3, :3]

    camera = pyrender.IntrinsicsCamera(
            fx=intrinsics[0],
            fy=intrinsics[1],
            cx=intrinsics[2],
            cy=intrinsics[3])

    # the inverse is same
    pyrender2opencv = np.array([[1.0, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])


    # render material
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.4,
            alphaMode='BLEND',
            emissiveFactor=(0.2, 0.2, 0.2),
            baseColorFactor=color)

    # get body mesh
    body_trimesh = trimesh.Trimesh(vertices, faces, process=False)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

    # prepare camera and light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    if cam_pose is None:
        cam_pose = pyrender2opencv @ np.eye(4)

    # build scene
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3))
    scene.add(light, pose=cam_pose)
    scene.add(camera, pose=cam_pose)
    scene.add(body_mesh, 'mesh')

    # render scene
    # os.environ["PYOPENGL_PLATFORM"] = "egl"  # include this line if use in vscode
    r = pyrender.OffscreenRenderer(viewport_width=img.shape[1],
                                   viewport_height=img.shape[0],
                                   point_size=1.0)

    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0
    # alpha = 1.0  # set transparency in [0.0, 1.0]
    # color[:, :, -1] = color[:, :, -1] * alpha
    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    valid_mask = valid_mask.astype(float) * alpha
    img = img / 255
    # output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * img)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    output_img = (color[:, :, :] * valid_mask + (1 - valid_mask) * img)

    # output_img = color

    img = (output_img * 255).astype(np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if return_mask:
        return img, valid_mask, (color * 255).astype(np.uint8)

    if save_path is not None:
        image_array_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image_array_bgr)

    return img


def create_video(output_dir, height, width):
    im_plot = sorted([os.path.join(output_dir, "plots", img) for img in os.listdir(os.path.join(output_dir, "plots")) if img.endswith(".png")])
    im_vid = sorted([os.path.join(output_dir, "frames", img) for img in os.listdir(os.path.join(output_dir, "frames")) if img.endswith(".jpg")  and img.startswith("frame")])

    min_len = min(len(im_vid), len(im_plot))
    im_vid = im_vid[:min_len]
    im_plot = im_plot[:min_len]

    plot = cv2.imread(im_plot[0])
    plot_h, plot_w = plot.shape[:2]
    dest_w = int(plot_w / plot_h * height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or 'MJPG' for .avi files
    video = cv2.VideoWriter(os.path.join(output_dir, "visualization.mp4"), fourcc, fps=50, frameSize=(width + dest_w, height))

    for plot_path, vid_path in tqdm(zip(im_plot, im_vid), colour="BLUE", desc="Creating video", total=min_len):
        frame = np.full((height, width + dest_w, 3), (255, 255, 255), dtype=np.uint8)
        vid_frame = cv2.imread(vid_path)
        plot = cv2.imread(plot_path)
        plot = cv2.resize(plot, (height, dest_w))
        frame[:, :width] = vid_frame
        frame[:, width:] = plot
        video.write(frame)

    video.release()
    print(f"Video saved as {os.path.join(output_dir, 'visualization.mp4')}")




