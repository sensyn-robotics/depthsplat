#!/usr/bin/env python3
"""Convert custom cubemap images + COLMAP poses to DepthSplat .torch format.

Reads front-facing cubemap images from data/images/ and COLMAP poses from
data/colmap/sparse/0/, then creates a .torch chunk compatible with DatasetRE10k.

Usage:
    python scripts/convert_custom_cubemap.py \
        --input_dir data/images \
        --colmap_dir data/colmap/sparse/0 \
        --output_dir datasets/custom_cubemap \
        --direction front
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def read_colmap_images(path: Path) -> dict:
    """Read COLMAP images.bin and return image poses keyed by name."""
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
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
            f.read(num_points2D * 24)  # skip 2D points
            images[name.decode()] = {
                "qvec": (qw, qx, qy, qz),
                "tvec": (tx, ty, tz),
                "camera_id": camera_id,
            }
    return images


def qvec_to_rotmat(qvec):
    """Convert COLMAP quaternion (w,x,y,z) to rotation matrix."""
    return Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()


def colmap_to_w2c_3x4(qvec, tvec):
    """Convert COLMAP quaternion + translation to w2c 3x4 matrix.

    COLMAP stores the world-to-camera rotation quaternion and translation
    directly, so we just need to assemble the 3x4 matrix.
    """
    R = qvec_to_rotmat(qvec)
    t = np.array(tvec).reshape(3, 1)
    return np.hstack([R, t]).astype(np.float32)


def load_raw(path: Path) -> torch.Tensor:
    """Load image file as raw bytes tensor."""
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def main():
    parser = argparse.ArgumentParser(description="Convert cubemap data to .torch format")
    parser.add_argument("--input_dir", type=str, default="data/images")
    parser.add_argument("--colmap_dir", type=str, default="data/colmap/sparse/0")
    parser.add_argument("--output_dir", type=str, default="datasets/custom_cubemap")
    parser.add_argument(
        "--direction",
        type=str,
        default="front",
        choices=["front", "back", "left", "right", "up"],
    )
    parser.add_argument("--scene_key", type=str, default="custom_cubemap")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    colmap_dir = Path(args.colmap_dir)
    output_dir = Path(args.output_dir)

    # Read COLMAP data
    colmap_images = read_colmap_images(colmap_dir / "images.bin")
    print(f"COLMAP: {len(colmap_images)} registered images")

    # Direction index mapping
    dir_idx_map = {"front": 0, "back": 1, "left": 2, "right": 3, "up": 4}
    dir_idx = dir_idx_map[args.direction]

    # Find data images for the chosen direction, sorted by scene number
    data_images = sorted(input_dir.glob(f"*_{dir_idx}_{args.direction}.png"))
    num_frames = len(data_images)
    print(f"Found {num_frames} {args.direction} images in {input_dir}")

    # Map data scene N → COLMAP image name: (N-1)*6 + 1 + dir_idx
    images_raw = []
    cameras_list = []
    timestamps = []

    for i, data_path in enumerate(data_images):
        scene_num = int(data_path.stem.split("_")[0])
        colmap_num = (scene_num - 1) * 6 + 1 + dir_idx
        colmap_name = f"{colmap_num:06d}_{dir_idx}_{args.direction}.png"

        if colmap_name not in colmap_images:
            print(f"  WARNING: {colmap_name} not in COLMAP, skipping scene {scene_num}")
            continue

        pose = colmap_images[colmap_name]
        w2c_3x4 = colmap_to_w2c_3x4(pose["qvec"], pose["tvec"])

        # Use theoretical cubemap intrinsics (90° FOV, centered)
        # instead of COLMAP's unreliable estimates
        fx, fy, cx, cy = 0.5, 0.5, 0.5, 0.5

        camera = [fx, fy, cx, cy, 0.0, 0.0] + w2c_3x4.flatten().tolist()
        cameras_list.append(camera)
        images_raw.append(load_raw(data_path))
        timestamps.append(i)

        print(f"  Scene {scene_num:3d} -> {colmap_name} -> OK")

    print(f"\nConverted {len(images_raw)} frames")

    # Build example dict matching the format expected by DatasetRE10k
    example = {
        "key": args.scene_key,
        "cameras": torch.tensor(np.array(cameras_list), dtype=torch.float32),
        "images": images_raw,
        "timestamps": torch.tensor(timestamps, dtype=torch.int64),
        "url": args.scene_key,
    }

    # Save as .torch chunk
    stage_dir = output_dir / "test"
    stage_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = stage_dir / "000000.torch"
    torch.save([example], chunk_path)
    print(f"Saved chunk to {chunk_path}")
    print(f"  Scene key: {args.scene_key}")
    print(f"  Cameras shape: {example['cameras'].shape}")
    print(f"  Images: {len(example['images'])} frames")


if __name__ == "__main__":
    main()
