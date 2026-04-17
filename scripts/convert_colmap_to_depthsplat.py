#!/usr/bin/env python3
"""Convert a COLMAP directory to a DepthSplat .torch chunk.

Expects the standard COLMAP layout:

    <input_dir>/
        images/           *.png or *.jpg (any sortable names)
        sparse/0/
            cameras.bin
            images.bin
            points3D.bin  (not read)

Writes a single chunk compatible with DatasetRE10k to:

    <output_dir>/test/000000.torch

Supports COLMAP camera models SIMPLE_PINHOLE (id 0) and PINHOLE (id 1).
Other models (OPENCV, RADIAL, ...) include distortion and are rejected; undistort
the images first with `colmap image_undistorter` if needed.

Usage:
    uv run python scripts/convert_colmap_to_depthsplat.py \
        --input_dir /path/to/colmap_scene \
        --output_dir datasets/custom_colmap \
        --scene_key my_scene
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# COLMAP camera model ids we support.
SIMPLE_PINHOLE = 0
PINHOLE = 1


def read_colmap_cameras(path: Path) -> dict:
    """Read COLMAP cameras.bin and return intrinsics keyed by camera_id."""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<i", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]

            if model_id == PINHOLE:
                fx, fy, cx, cy = struct.unpack("<4d", f.read(32))
            elif model_id == SIMPLE_PINHOLE:
                f_shared, cx, cy = struct.unpack("<3d", f.read(24))
                fx = fy = f_shared
            else:
                raise ValueError(
                    f"Camera {camera_id} uses model id {model_id} (not SIMPLE_PINHOLE "
                    "or PINHOLE). Undistort the images first with "
                    "`colmap image_undistorter` before converting."
                )

            cameras[camera_id] = {
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            }
    return cameras


def read_colmap_images(path: Path) -> dict:
    """Read COLMAP images.bin and return poses keyed by image name."""
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            f.read(4)  # image_id (unused)
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<i", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)  # skip 2D point observations
            images[name.decode()] = {
                "qvec": (qw, qx, qy, qz),
                "tvec": (tx, ty, tz),
                "camera_id": camera_id,
            }
    return images


def colmap_to_w2c_3x4(qvec, tvec) -> np.ndarray:
    """COLMAP stores world-to-camera quaternion + translation directly.

    COLMAP and OpenCV share axes (+X right, +Y down, +Z forward), so the 3x4 w2c
    can be assembled without any axis flip.
    """
    R = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
    t = np.array(tvec).reshape(3, 1)
    return np.hstack([R, t]).astype(np.float32)


def load_raw(path: Path) -> torch.Tensor:
    """Load an image file as a raw uint8 byte tensor (PNG/JPG undecoded)."""
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def main():
    parser = argparse.ArgumentParser(description="COLMAP -> DepthSplat .torch converter")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="COLMAP scene root containing images/ and sparse/0/.")
    parser.add_argument("--output_dir", type=str, default="datasets/custom_colmap")
    parser.add_argument("--scene_key", type=str, default="custom_colmap")
    parser.add_argument("--images_subdir", type=str, default="images")
    parser.add_argument("--sparse_subdir", type=str, default="sparse/0")
    parser.add_argument("--stride", type=int, default=1,
                        help="Keep every Nth frame (default 1 = keep all).")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Cap number of frames (0 = no cap).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    images_dir = input_dir / args.images_subdir
    sparse_dir = input_dir / args.sparse_subdir
    output_dir = Path(args.output_dir)

    cameras = read_colmap_cameras(sparse_dir / "cameras.bin")
    colmap_images = read_colmap_images(sparse_dir / "images.bin")
    print(f"COLMAP: {len(cameras)} cameras, {len(colmap_images)} registered images")

    sorted_names = sorted(colmap_images.keys())
    selected = sorted_names[:: args.stride]
    if args.max_frames:
        selected = selected[: args.max_frames]
    print(f"Converting {len(selected)} frames (stride={args.stride})")

    cameras_list = []
    images_raw = []
    timestamps = []

    for i, name in enumerate(selected):
        pose = colmap_images[name]
        cam = cameras[pose["camera_id"]]

        w2c_3x4 = colmap_to_w2c_3x4(pose["qvec"], pose["tvec"])

        W, H = cam["width"], cam["height"]
        fx_n = cam["fx"] / W
        fy_n = cam["fy"] / H
        cx_n = cam["cx"] / W
        cy_n = cam["cy"] / H

        camera_row = [fx_n, fy_n, cx_n, cy_n, 0.0, 0.0] + w2c_3x4.flatten().tolist()
        cameras_list.append(camera_row)

        img_path = images_dir / name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        images_raw.append(load_raw(img_path))
        timestamps.append(i)

    example = {
        "key": args.scene_key,
        "cameras": torch.tensor(np.array(cameras_list), dtype=torch.float32),
        "images": images_raw,
        "timestamps": torch.tensor(timestamps, dtype=torch.int64),
        "url": args.scene_key,
    }

    stage_dir = output_dir / "test"
    stage_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = stage_dir / "000000.torch"
    torch.save([example], chunk_path)

    index_path = stage_dir / "index.json"
    with index_path.open("w") as f:
        json.dump({args.scene_key: chunk_path.name}, f, indent=2)

    print(f"Saved chunk: {chunk_path}")
    print(f"Saved index: {index_path}")
    print(f"  scene_key  : {args.scene_key}")
    print(f"  cameras    : {example['cameras'].shape}  (N, 18)")
    print(f"  images     : {len(example['images'])} frames (raw bytes)")
    print("  intrinsics : normalized (fx/W, fy/H, cx/W, cy/H)")
    print("  extrinsics : world-to-camera, OpenCV axes")


if __name__ == "__main__":
    main()
