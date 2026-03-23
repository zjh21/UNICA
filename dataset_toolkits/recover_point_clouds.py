"""
Recover point clouds from position maps.

Usage:
    python dataset_toolkits/recover_point_clouds.py <input> [--output_dir DIR] [--normalize]

    input: Path to a single position map file or a folder (processed recursively).
"""

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import glob
from pathlib import Path

import cv2 as cv
import numpy as np
import trimesh


def recover_points_from_position_map(position_map_path):
    """
    Recover 3D points from a position map (EXR or PNG).

    Non-zero pixels are treated as 3D coordinates in [0, 1] range and
    converted to [-0.5, 0.5].

    Returns:
        Nx3 numpy array of 3D points.
    """
    position_map = cv.imread(position_map_path, cv.IMREAD_UNCHANGED)
    if position_map is None:
        raise ValueError(f"Could not read: {position_map_path}")

    if np.max(position_map) > 1.0:
        position_map = position_map / 255.0

    # Vectorized extraction of non-background pixels
    mask = np.any(position_map != 0, axis=2)
    points = position_map[mask, :3].copy()
    points -= 0.5

    print(f"  Recovered {len(points)} points")
    return points


def normalize_point_cloud(points):
    """
    Normalize point cloud so the longest axis spans [-0.5, 0.5].

    Returns:
        (normalized_points, transform_info dict)
    """
    if len(points) == 0:
        return points, {}

    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    center = (min_coords + max_coords) / 2.0
    max_range = np.max(max_coords - min_coords)

    if max_range == 0:
        max_range = 1.0

    scale_factor = 1.0 / max_range
    normalized = (points - center) * scale_factor

    return normalized, {
        "center": center,
        "scale_factor": scale_factor,
        "max_range": max_range,
    }


def save_point_cloud(points, output_path):
    """Save point cloud as PLY file."""
    if len(points) == 0:
        print(f"  No points to save")
        return

    pc = trimesh.PointCloud(vertices=points)
    pc.export(output_path)
    print(f"  Saved {len(points)} points to {output_path}")


def process_single(position_map_path, output_dir=None, normalize=False):
    """Process a single position map and save a point cloud."""
    print(f"\nProcessing: {position_map_path}")

    if output_dir is None:
        output_dir = Path(position_map_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(position_map_path).stem
    points = recover_points_from_position_map(position_map_path)

    if len(points) > 0:
        if normalize:
            points, _ = normalize_point_cloud(points)

        suffix = "_normalized" if normalize else ""
        output_path = output_dir / f"{base_name}_pointcloud{suffix}.ply"
        save_point_cloud(points, str(output_path))


def process_folder(folder_path, output_dir=None, extensions=None, normalize=False):
    """Process all position maps in a folder recursively."""
    if extensions is None:
        extensions = ["exr", "png"]

    folder_path = Path(folder_path)

    all_files = []
    for ext in extensions:
        all_files.extend(
            glob.glob(str(folder_path / "**" / f"*.{ext}"), recursive=True)
        )
    all_files = sorted(set(all_files))

    if not all_files:
        print(f"No files found in {folder_path}")
        return

    print(f"Found {len(all_files)} files")

    for i, file_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}]", end=" ")
        try:
            if output_dir is None:
                file_out = Path(file_path).parent
            else:
                relative = Path(file_path).parent.relative_to(folder_path)
                file_out = Path(output_dir) / relative
                file_out.mkdir(parents=True, exist_ok=True)

            process_single(file_path, file_out, normalize)
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nDone. Processed {len(all_files)} files.")


def main():
    parser = argparse.ArgumentParser(
        description="Recover point clouds from position maps"
    )
    parser.add_argument("input", help="Position map file or folder")
    parser.add_argument(
        "--output_dir", default=None, help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["exr", "png"],
        help="File extensions to process (default: exr png)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize point clouds to [-0.5, 0.5]",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: not found: {args.input}")
        return

    if input_path.is_file():
        process_single(str(input_path), args.output_dir, args.normalize)
    elif input_path.is_dir():
        process_folder(str(input_path), args.output_dir, args.extensions, args.normalize)


if __name__ == "__main__":
    main()
