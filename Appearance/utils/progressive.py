"""
Progressive 4D inference utilities — Procrustes-based denormalization.

Given refined 3DGS PLY files (in normalized space) and position-map NPY files
(in unnormalized world space), this module computes an optimal similarity
transformation (rotation, uniform scale, translation) via Procrustes analysis
and applies it to the Gaussians, preserving their relative topology.
"""

import os
import time
import warnings

import numpy as np
import torch
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation

from utils.posmap_utils import upscale_foreground_aware


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_position_map_npy(
    posmap_path: str,
    target_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load an NPY position map and upscale with foreground-aware interpolation.

    Args:
        posmap_path: Path to the ``.npy`` position map (128, 128, 3).
        target_size: Target resolution for upscaling.

    Returns:
        upscaled_posmap: ``(3, target_size, target_size)`` tensor.
        upscaled_mask:   ``(1, target_size, target_size)`` boolean tensor.
    """
    posmap = np.load(posmap_path).astype(np.float32)  # (128, 128, 3)

    if posmap.shape != (128, 128, 3):
        raise ValueError(
            f"Expected position map shape (128, 128, 3), got {posmap.shape}"
        )

    posmap_tensor = torch.from_numpy(posmap).permute(2, 0, 1)  # (3, 128, 128)

    # Foreground: any channel non-zero
    mask = torch.any(posmap_tensor != 0.0, dim=0, keepdim=True)  # (1, 128, 128)

    return upscale_foreground_aware(posmap_tensor, mask, target_size=target_size)


def load_ply(ply_path: str) -> Dict[str, np.ndarray]:
    """Load a PLY file and return all vertex properties as a dict."""
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]
    return {prop: vertex[prop].copy() for prop in vertex.data.dtype.names}


def save_ply(data: Dict[str, np.ndarray], output_path: str) -> None:
    """Save a dict of vertex properties as a PLY file."""
    num_points = len(next(iter(data.values())))
    dtype_list = [(name, "f4") for name in data.keys()]
    elements = np.empty(num_points, dtype=dtype_list)
    for name, values in data.items():
        elements[name] = values.astype(np.float32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(output_path)


# ---------------------------------------------------------------------------
# Procrustes analysis
# ---------------------------------------------------------------------------

def procrustes_analysis(
    source: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """
    Procrustes analysis: find rotation *R*, scale *s*, translation *t* that
    minimise ``||s * (source @ R) + t - target||²_F``.

    Returns:
        R:    (3, 3) rotation matrix.
        s:    scalar scale factor.
        t:    (3,) translation vector.
        rmse: root-mean-square error after transformation.
    """
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    # Cross-covariance → SVD
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # ensure proper rotation
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    source_var = np.sum(source_centered ** 2)
    s = np.sum(S) / source_var if source_var > 1e-8 else 1.0

    t = target_mean - s * (source_mean @ R)

    transformed = s * (source @ R) + t
    rmse = np.sqrt(np.mean(np.sum((transformed - target) ** 2, axis=1)))

    return R, s, t, rmse


def transform_quaternions_batch(
    quaternions_wxyz: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    Transform Gaussian orientation quaternions for a scene rotation *R*.

    When positions transform as ``x_new = s * x @ R + t``, the Gaussian
    covariance transforms as ``Σ_new = R.T @ Σ @ R``, so the quaternion
    rotation becomes ``q_new = q_{R^{-1}} * q_old``.

    Args:
        quaternions_wxyz: (N, 4) in [w, x, y, z] order (3DGS convention).
        R: (3, 3) rotation matrix from Procrustes.

    Returns:
        (N, 4) transformed quaternions in [w, x, y, z] order.
    """
    rot_scene = Rotation.from_matrix(R).inv()  # R.T

    # 3DGS [w,x,y,z] → scipy [x,y,z,w]
    quat_xyzw = quaternions_wxyz[:, [1, 2, 3, 0]]
    norms = np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    quat_xyzw = quat_xyzw / np.clip(norms, 1e-8, None)

    rot_new = rot_scene * Rotation.from_quat(quat_xyzw)

    # scipy [x,y,z,w] → 3DGS [w,x,y,z]
    new_xyzw = rot_new.as_quat()
    return new_xyzw[:, [3, 0, 1, 2]].astype(np.float32)


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_single_pair(
    ply_path: str,
    posmap_path: str,
    attr_mask: np.ndarray,
    output_path: str,
    target_size: int = 1024,
    verbose: bool = False,
) -> bool:
    """
    Denormalize a single PLY using a position-map NPY via Procrustes analysis.

    Args:
        ply_path:    Path to the refined PLY (normalized space).
        posmap_path: Path to the NPY position map (world space).
        attr_mask:   (1024, 1024) boolean mask from the attribute map.
        output_path: Where to write the denormalized PLY.
        target_size: Resolution for position-map upscaling.
        verbose:     Print detailed diagnostics.

    Returns:
        ``True`` on success.
    """
    ply_data = load_ply(ply_path)
    num_points = len(ply_data["x"])

    source_positions = np.stack(
        [ply_data["x"], ply_data["y"], ply_data["z"]], axis=1
    )

    # Load & upscale position map
    upscaled_posmap, posmap_mask = load_position_map_npy(posmap_path, target_size)
    upscaled_posmap_np = upscaled_posmap.permute(1, 2, 0).numpy()
    posmap_mask_np = posmap_mask.squeeze(0).numpy()

    combined_mask = attr_mask & posmap_mask_np
    if combined_mask.sum() == 0:
        if verbose:
            print("  Warning: No overlapping foreground pixels")
        return False

    target_positions = upscaled_posmap_np[combined_mask]  # (N, 3)

    if verbose:
        print(f"  PLY points: {num_points}")
        print(f"  Position map foreground: {posmap_mask_np.sum()}")
        print(f"  Attribute map foreground: {attr_mask.sum()}")
        print(f"  Combined mask foreground: {combined_mask.sum()}")

    # Handle point-count mismatch
    if target_positions.shape[0] != num_points:
        if verbose:
            print(
                f"  Warning: Point count mismatch. "
                f"PLY: {num_points}, Mask: {target_positions.shape[0]}"
            )
        min_pts = min(num_points, target_positions.shape[0])
        source_for_proc = source_positions[:min_pts]
        target_for_proc = target_positions[:min_pts]
    else:
        source_for_proc = source_positions
        target_for_proc = target_positions

    # Procrustes
    R, s, t, rmse = procrustes_analysis(source_for_proc, target_for_proc)

    if verbose:
        rot_angle = np.degrees(
            np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        )
        print(f"  Procrustes — scale: {s:.6f}, "
              f"rotation: {rot_angle:.2f}°, RMSE: {rmse:.6f}")

    # Transform positions: x_new = s * x @ R + t
    new_positions = s * (source_positions @ R) + t
    ply_data["x"] = new_positions[:, 0].astype(np.float32)
    ply_data["y"] = new_positions[:, 1].astype(np.float32)
    ply_data["z"] = new_positions[:, 2].astype(np.float32)

    # Transform rotations
    if all(k in ply_data for k in ("rot_0", "rot_1", "rot_2", "rot_3")):
        old_q = np.stack(
            [ply_data["rot_0"], ply_data["rot_1"],
             ply_data["rot_2"], ply_data["rot_3"]],
            axis=1,
        )
        new_q = transform_quaternions_batch(old_q, R)
        ply_data["rot_0"] = new_q[:, 0]
        ply_data["rot_1"] = new_q[:, 1]
        ply_data["rot_2"] = new_q[:, 2]
        ply_data["rot_3"] = new_q[:, 3]

    # Transform scales (log-space)
    log_s = np.log(s) if s > 0 else 0.0
    for k in ("scale_0", "scale_1", "scale_2"):
        if k in ply_data:
            ply_data[k] = (ply_data[k] + log_s).astype(np.float32)

    save_ply(ply_data, output_path)
    return True


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def find_ply_npy_pairs(
    ply_dir: str,
    npy_dir: str,
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """
    Match PLY files in *ply_dir* with NPY files in *npy_dir* by relative path.

    Returns:
        pairs:   list of ``(ply_path, npy_path, relative_stem)`` tuples.
        missing: list of PLY paths with no matching NPY.
    """
    ply_files = sorted(glob(os.path.join(ply_dir, "**", "*.ply"), recursive=True))
    pairs: List[Tuple[str, str, str]] = []
    missing: List[str] = []

    for ply_path in ply_files:
        rel = os.path.relpath(ply_path, ply_dir)
        npy_rel = str(Path(rel).with_suffix(".npy"))
        npy_path = os.path.join(npy_dir, npy_rel)
        if os.path.exists(npy_path):
            pairs.append((ply_path, npy_path, rel))
        else:
            missing.append(ply_path)

    return pairs, missing


def run_progressive(
    ply_dir: str,
    npy_dir: str,
    attr_mask: np.ndarray,
    target_resolution: int = 1024,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Run progressive 4D inference (Procrustes denormalization) on all PLY files.

    Matches refined PLY files in *ply_dir* with position-map NPY files in
    *npy_dir*, then overwrites each PLY with the denormalized version.

    Args:
        ply_dir:           Directory containing refined PLY files.
        npy_dir:           Directory containing NPY position maps.
        attr_mask:         (1024, 1024) boolean foreground mask.
        target_resolution: Resolution for position-map upscaling.
        verbose:           Print progress information.

    Returns:
        Statistics dict with ``total``, ``success``, ``failed``, ``skipped``.
    """
    pairs, missing = find_ply_npy_pairs(ply_dir, npy_dir)

    stats = {
        "total": len(pairs) + len(missing),
        "success": 0,
        "failed": 0,
        "skipped": len(missing),
    }

    if not pairs:
        print("Progressive: no matching PLY-NPY pairs found.")
        return stats

    sep = "-" * 70
    print(sep)
    print(f"Progressive 4D — Procrustes denormalization")
    print(f"  PLY source: {ply_dir}")
    print(f"  NPY source: {npy_dir}")
    print(f"  Matched pairs: {len(pairs)}  |  Skipped (no NPY): {len(missing)}")
    print(sep)

    for ply_path, npy_path, rel in tqdm(pairs, desc="Progressive", disable=not verbose):
        try:
            ok = process_single_pair(
                ply_path,
                npy_path,
                attr_mask,
                output_path=ply_path,  # overwrite in-place
                target_size=target_resolution,
                verbose=verbose,
            )
            if ok:
                stats["success"] += 1
            else:
                stats["failed"] += 1
        except Exception as e:
            stats["failed"] += 1
            if verbose:
                print(f"\nError: {rel}: {e}")

    print(sep)
    print(f"Progressive complete — "
          f"OK: {stats['success']}  |  "
          f"Failed: {stats['failed']}  |  "
          f"Skipped: {stats['skipped']}")
    print(sep)
    return stats
