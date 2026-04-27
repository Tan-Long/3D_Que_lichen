#!/usr/bin/env python3
"""
Step 1: Extract RGB frames from rgb.mp4.

Stray Scanner stores RGB as a single mp4 video. This script extracts
individual frames aligned to the depth frame indices (000000.png, 000001.png, ...).

Usage:
    python 01_extract_rgb_frames.py c9ae76fff8
"""

import argparse
import os
import subprocess

def _resolve_ffmpeg() -> str:
    """Return ffmpeg executable path.

    Priority:
    1) ffmpeg on PATH
    2) imageio-ffmpeg bundled binary
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return "ffmpeg"
    except Exception:
        pass

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "ffmpeg is not available. Install ffmpeg or pip install imageio-ffmpeg."
        ) from e

def extract_frames(dataset_dir: str) -> None:
    video_path = os.path.join(dataset_dir, "rgb.mp4")
    rgb_dir = os.path.join(dataset_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    # Count how many depth frames we have — we extract that many RGB frames
    depth_dir = os.path.join(dataset_dir, "depth")
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(f"Missing depth directory: {depth_dir}")
    n_depth = len([f for f in os.listdir(depth_dir) if f.endswith(".png")])
    print(f"Found {n_depth} depth frames. Extracting matching RGB frames...")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Missing video: {video_path}")

    # Use ffmpeg for robust frame extraction on macOS.
    # -start_number 0 to match depth frame naming (000000, 000001, ...)
    # select=lt(n\,N) ensures we only extract the first N frames.
    ffmpeg_bin = _resolve_ffmpeg()
    out_pattern = os.path.join(rgb_dir, "%06d.jpg")
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"select=lt(n\\,{n_depth})",
        "-vsync",
        "0",
        "-start_number",
        "0",
        out_pattern,
    ]
    subprocess.run(cmd, check=True)

    extracted = len([f for f in os.listdir(rgb_dir) if f.lower().endswith(".jpg")])
    print(f"Extracted {extracted} RGB frames to {rgb_dir}/")
    if extracted < n_depth:
        # Video may have 1-2 fewer frames than depth (common with Stray Scanner
        # due to video encoding). Warn but only fail if the gap is large.
        gap = n_depth - extracted
        if gap <= 2:
            print(f"WARNING: {gap} fewer RGB frames than depth frames — acceptable.")
        else:
            raise RuntimeError(
                f"RGB frame count mismatch: extracted {extracted}, expected {n_depth}. "
                f"Check {video_path} and {depth_dir}."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract RGB frames from Stray Scanner mp4")
    parser.add_argument("dataset", help="Path to the dataset folder (e.g., c9ae76fff8)")
    args = parser.parse_args()
    extract_frames(args.dataset)
