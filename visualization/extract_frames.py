# -*- coding: utf-8 -*-
"""
Created on 01.07.25

@author: Katja

"""
import os
import subprocess


def extract_frames_ffmpeg(video_path, output_folder, fps=50):
    """
    Converts a video to distinct JPEG frames using FFmpeg.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where frames will be saved.
        fps (int, optional): Number of frames per second to extract. Defaults to 50.
                             Set to -1 to extract all original frames.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Ensure output_folder ends with a separator for path joining
    output_folder = os.path.join(output_folder, "")

    # Output frame filename pattern (e.g., frame_00001.jpg, frame_00002.jpg)
    output_filename_pattern = os.path.join(output_folder, "frame_%05d.jpg")

    # FFmpeg command to extract frames
    # -i: Input file
    # -vf: Video filter graph (here, "fps=50" sets the frame rate)
    # -q:v: Video quality (1 means highest quality JPEG)
    # %05d: Placeholder for a 5-digit padded frame number
    if fps == -1:
        # Extract all original frames without re-encoding FPS
        ffmpeg_command = [
            'ffmpeg',
            '-i', video_path,
            '-q:v', '1',
            output_filename_pattern
        ]
        print(f"Extracting all original frames from '{video_path}' to '{output_folder}'...")
    else:
        ffmpeg_command = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps={fps}',
            '-q:v', '1',
            output_filename_pattern
        ]
        print(f"Extracting frames at {fps} FPS from '{video_path}' to '{output_folder}'...")

    try:
        # Execute the FFmpeg command
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully extracted frames to: {output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in your system's PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")