"""
Script to denormalize 3DGS targets using position maps with Procrustes analysis.

This script:
1. Loads paired 3DGS PLY files and position map NPY files
2. Upscales 128x128 position maps to 1024x1024 with foreground-aware interpolation
3. Performs Procrustes analysis to find optimal rotation, scale, and translation
4. Transforms 3DGS targets using the computed transformation (preserving topology)
5. Saves the modified 3DGS targets preserving folder structure

The key difference from direct xyz replacement:
- Procrustes analysis finds a global similarity transformation (R, s, t) that minimizes
  ||s * X @ R + t - Y||² where X is source positions and Y is target positions
- This transformation is applied to all Gaussians, preserving their relative structure
- Gaussian rotations are also transformed to maintain correct orientation
- Gaussian scales are adjusted by the uniform scale factor

Directory structure:
    Input:
        3DGS/{caseName}/1_Forward.ply
        3DGS/{caseName}/2_Forward.ply
        ...
        posmap/{caseName}/1_Forward.npy
        posmap/{caseName}/2_Forward.npy
        ...
    
    Output:
        final/{caseName}/1_Forward.ply
        final/{caseName}/2_Forward.ply
        ...

Usage:
    python denormalize_3dgs_targets.py \
        --gs_dir /path/to/3DGS \
        --posmap_dir /path/to/posmap \
        --attribute_map_path /path/to/base_attribute_map.npy \
        --output_dir /path/to/final
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List
from pathlib import Path
import argparse
from glob import glob
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
import time


def upscale_foreground_aware(
    position_map: torch.Tensor,
    mask: torch.Tensor,
    target_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Upscale position map with foreground-aware interpolation.
    Background pixels don't contribute to interpolation.

    Args:
        position_map: (3, H, W) tensor, background is 0
        mask: (3, H, W) or (1, H, W) boolean tensor indicating foreground
        target_size: Target size for upscaling (default 1024)

    Returns:
        upscaled_posmap: (3, target_size, target_size) tensor
        upscaled_mask: (1, target_size, target_size) boolean tensor
    """
    # Ensure mask is single channel
    if mask.dim() == 3 and mask.shape[0] == 3:
        mask_1ch = mask[0:1]  # Take first channel
    elif mask.dim() == 3 and mask.shape[0] == 1:
        mask_1ch = mask
    else:
        mask_1ch = mask.unsqueeze(0) if mask.dim() == 2 else mask

    # Create weight map (1 for foreground, 0 for background)
    weight_map = mask_1ch.float()

    # Create a version of position map with background set to 0
    # This is key: background pixels won't contribute to interpolation
    position_map_masked = position_map.clone()
    mask_3ch = mask if mask.shape[0] == 3 else mask_1ch.expand(3, -1, -1)
    position_map_masked = position_map_masked * mask_3ch.float()

    # Add batch dimension for interpolation
    position_map_masked = position_map_masked.unsqueeze(0)  # (1, 3, H, W)
    weight_map = weight_map.unsqueeze(0)  # (1, 1, H, W)

    # Interpolate both position map and weight map
    upscaled_posmap = F.interpolate(
        position_map_masked,
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    )
    upscaled_weights = F.interpolate(
        weight_map,
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    )

    # Normalize by weights to get proper interpolation
    # This ensures interpolation only uses foreground neighbors
    upscaled_weights_safe = torch.clamp(upscaled_weights, min=1e-8)
    upscaled_posmap = upscaled_posmap / upscaled_weights_safe

    # Create mask: only keep pixels with sufficient foreground contribution
    # Threshold: at least 25% foreground weight means this pixel is valid
    threshold = 0.25
    upscaled_mask = (upscaled_weights > threshold)

    # Set background pixels to 0
    upscaled_posmap = upscaled_posmap.squeeze(0)  # (3, target_size, target_size)
    upscaled_mask_3ch = upscaled_mask[0].expand_as(upscaled_posmap)
    upscaled_posmap = torch.where(
        upscaled_mask_3ch,
        upscaled_posmap,
        torch.tensor(0.0, dtype=upscaled_posmap.dtype, device=upscaled_posmap.device)
    )

    upscaled_mask = upscaled_mask[0]  # (1, target_size, target_size)
    
    return upscaled_posmap, upscaled_mask


def load_position_map_npy(
    posmap_path: str, 
    target_size: int = 1024
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load NPY position map and upscale with foreground-aware interpolation.
    
    Args:
        posmap_path: Path to the .npy position map (128, 128, 3)
        target_size: Target resolution for upscaling
        
    Returns:
        upscaled_posmap: (3, target_size, target_size) tensor
        upscaled_mask: (1, target_size, target_size) boolean tensor
    """
    # Load position map
    posmap = np.load(posmap_path).astype(np.float32)  # (128, 128, 3)
    
    if posmap.shape != (128, 128, 3):
        raise ValueError(f"Expected position map shape (128, 128, 3), got {posmap.shape}")
    
    # Convert to tensor (3, 128, 128)
    posmap_tensor = torch.from_numpy(posmap).permute(2, 0, 1)
    
    # Create mask: foreground has non-zero values (check all channels)
    # Use any channel being non-zero as foreground indicator
    mask = torch.any(posmap_tensor != 0.0, dim=0, keepdim=True)  # (1, 128, 128)
    
    # Upscale with foreground-aware interpolation
    upscaled_posmap, upscaled_mask = upscale_foreground_aware(
        posmap_tensor,
        mask,
        target_size=target_size
    )
    
    return upscaled_posmap, upscaled_mask


def load_attribute_map(attribute_map_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the base attribute map and extract the mask.
    
    Args:
        attribute_map_path: Path to the .npy attribute map file
        
    Returns:
        attribute_map: (1024, 1024, 14) array
        mask: (1024, 1024) boolean array
    """
    attribute_map = np.load(attribute_map_path)
    
    if attribute_map.shape != (1024, 1024, 14):
        raise ValueError(f"Expected attribute map shape (1024, 1024, 14), got {attribute_map.shape}")
    
    # Create mask: foreground pixels have non-zero values across any channel
    mask = np.any(attribute_map != 0, axis=2)
    
    return attribute_map, mask


def load_ply(ply_path: str) -> Dict[str, np.ndarray]:
    """
    Load PLY file and extract all attributes.
    
    Returns:
        Dictionary with all vertex properties
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    # Preserve property order
    data = {}
    property_names = list(vertex.data.dtype.names)
    for prop in property_names:
        data[prop] = vertex[prop].copy()
    
    return data


def save_ply(data: Dict[str, np.ndarray], output_path: str):
    """
    Save data as PLY file.
    
    Args:
        data: Dictionary with vertex properties
        output_path: Path to save the PLY file
    """
    num_points = len(data['x'])
    
    # Build dtype from data - maintain order
    dtype_list = [(name, 'f4') for name in data.keys()]
    
    elements = np.empty(num_points, dtype=dtype_list)
    
    for name, values in data.items():
        elements[name] = values.astype(np.float32)
    
    el = PlyElement.describe(elements, 'vertex')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PlyData([el]).write(output_path)


def procrustes_analysis(
    source: np.ndarray, 
    target: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """
    Perform Procrustes analysis to find optimal similarity transformation.
    
    Finds rotation R, scale s, translation t that minimizes:
    ||s * (source @ R) + t - target||²_F
    
    This preserves the topology (relative positions) of the source points
    while aligning them to the target points.
    
    Args:
        source: (N, 3) source points (original 3DGS positions)
        target: (N, 3) target points (from position map)
        
    Returns:
        R: (3, 3) rotation matrix
        s: scalar scale factor
        t: (3,) translation vector
        rmse: root mean squared error after transformation
    """
    n = source.shape[0]
    
    # Center the points
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    
    source_centered = source - source_mean
    target_centered = target - target_mean
    
    # Compute the cross-covariance matrix
    H = source_centered.T @ target_centered
    
    # SVD of cross-covariance
    U, S, Vt = np.linalg.svd(H)
    
    # Optimal rotation: R = V @ U.T
    R = Vt.T @ U.T
    
    # Handle reflection case (ensure proper rotation with det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute optimal scale
    # s = trace(S) / ||source_centered||²_F
    source_var = np.sum(source_centered ** 2)
    if source_var > 1e-8:
        s = np.sum(S) / source_var
    else:
        s = 1.0
    
    # Compute translation
    # t = target_mean - s * source_mean @ R
    t = target_mean - s * (source_mean @ R)
    
    # Compute RMSE
    transformed = s * (source @ R) + t
    rmse = np.sqrt(np.mean(np.sum((transformed - target) ** 2, axis=1)))
    
    return R, s, t, rmse


def transform_quaternions_batch(
    quaternions_wxyz: np.ndarray, 
    R: np.ndarray
) -> np.ndarray:
    """
    Transform Gaussian orientation quaternions when scene is transformed.
    
    When positions are transformed as: x_new = s * x @ R + t
    The equivalent column-vector transformation is: x_new = s * R.T @ x + t
    
    The Gaussian covariance transforms as:
    Σ_new = R.T @ Σ @ R = R.T @ (R_q @ S² @ R_q.T) @ R = (R.T @ R_q) @ S² @ (R.T @ R_q).T
    
    So the new rotation matrix is R_new = R.T @ R_old, which in quaternion form is:
    q_new = q_{R.T} * q_old = q_R^{-1} * q_old
    
    Args:
        quaternions_wxyz: (N, 4) array with [w, x, y, z] convention (3DGS format)
        R: (3, 3) rotation matrix from Procrustes analysis
    
    Returns:
        (N, 4) transformed quaternions in [w, x, y, z] convention
    """
    # Get the scene rotation (R.T = R^{-1} since R is orthogonal)
    rot_R = Rotation.from_matrix(R)
    rot_scene = rot_R.inv()  # This represents R.T
    
    # Convert from 3DGS convention [w, x, y, z] to scipy convention [x, y, z, w]
    quaternions_xyzw = quaternions_wxyz[:, [1, 2, 3, 0]]
    
    # Handle potential numerical issues with quaternion normalization
    norms = np.linalg.norm(quaternions_xyzw, axis=1, keepdims=True)
    quaternions_xyzw = quaternions_xyzw / np.clip(norms, 1e-8, None)
    
    # Create Rotation objects for all Gaussian quaternions
    rot_gaussians = Rotation.from_quat(quaternions_xyzw)
    
    # Apply scene rotation: new_orientation = rot_scene * old_orientation
    # This composes the rotations: first apply old Gaussian rotation, then scene rotation
    rot_new = rot_scene * rot_gaussians
    
    # Convert back to [w, x, y, z] convention
    new_quaternions_xyzw = rot_new.as_quat()
    new_quaternions_wxyz = new_quaternions_xyzw[:, [3, 0, 1, 2]]
    
    return new_quaternions_wxyz.astype(np.float32)


def process_single_pair(
    ply_path: str,
    posmap_path: str,
    attr_mask: np.ndarray,
    output_path: str,
    target_size: int = 1024,
    verbose: bool = False
) -> bool:
    """
    Process a single PLY and position map pair using Procrustes analysis.
    
    Instead of directly replacing xyz coordinates (which breaks topology),
    this function:
    1. Computes correspondence between Gaussians and position map points
    2. Performs Procrustes analysis to find optimal similarity transformation
    3. Applies the transformation to all Gaussians (positions, rotations, scales)
    
    This preserves the relative structure of the Gaussians while moving them
    to the target coordinate space.
    
    Args:
        ply_path: Path to input PLY file
        posmap_path: Path to input NPY position map
        attr_mask: (1024, 1024) boolean mask from attribute map
        output_path: Path to save output PLY file
        target_size: Target resolution for upscaling
        verbose: Print detailed info
        
    Returns:
        True if successful, False otherwise
    """
    # Load PLY data
    ply_data = load_ply(ply_path)
    num_points = len(ply_data['x'])
    
    # Get source positions (original 3DGS positions)
    source_positions = np.stack([ply_data['x'], ply_data['y'], ply_data['z']], axis=1)
    
    # Load and upscale position map
    upscaled_posmap, posmap_mask = load_position_map_npy(posmap_path, target_size)
    
    # Convert to numpy (H, W, 3)
    upscaled_posmap_np = upscaled_posmap.permute(1, 2, 0).numpy()
    posmap_mask_np = posmap_mask.squeeze(0).numpy()
    
    # Combine masks (intersection of attribute map and position map foregrounds)
    combined_mask = attr_mask & posmap_mask_np
    
    if combined_mask.sum() == 0:
        if verbose:
            print(f"  Warning: No overlapping foreground pixels")
        return False
    
    # Get target positions from position map (in row-major order matching PLY)
    target_positions = upscaled_posmap_np[combined_mask]  # (N, 3)
    
    if verbose:
        print(f"  PLY points: {num_points}")
        print(f"  Position map foreground: {posmap_mask_np.sum()}")
        print(f"  Attribute map foreground: {attr_mask.sum()}")
        print(f"  Combined mask foreground: {combined_mask.sum()}")
    
    # Handle point count mismatch
    if target_positions.shape[0] != num_points:
        if verbose:
            print(f"  Warning: Point count mismatch. PLY: {num_points}, Mask: {target_positions.shape[0]}")
        # Use minimum number of points for Procrustes
        min_points = min(num_points, target_positions.shape[0])
        source_for_procrustes = source_positions[:min_points]
        target_for_procrustes = target_positions[:min_points]
    else:
        source_for_procrustes = source_positions
        target_for_procrustes = target_positions
    
    # Perform Procrustes analysis to find optimal similarity transformation
    time1 = time.time()
    R, s, t, rmse = procrustes_analysis(source_for_procrustes, target_for_procrustes)
    procrustes_time = time.time() - time1
    print(procrustes_time)
    
    if verbose:
        print(f"  Procrustes results:")
        print(f"    Scale factor: {s:.6f}")
        print(f"    Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
        print(f"    Rotation matrix determinant: {np.linalg.det(R):.6f}")
        # Compute rotation angle
        rotation_angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        print(f"    Rotation angle: {np.degrees(rotation_angle):.2f} degrees")
        print(f"    Alignment RMSE: {rmse:.6f}")
        
        # Show position ranges before and after
        source_range = source_positions.max(axis=0) - source_positions.min(axis=0)
        print(f"    Source position range: [{source_positions.min():.4f}, {source_positions.max():.4f}]")
        print(f"    Source extent: [{source_range[0]:.4f}, {source_range[1]:.4f}, {source_range[2]:.4f}]")
    
    # Transform all positions using Procrustes transformation
    # x_new = s * x @ R + t
    new_positions = s * (source_positions @ R) + t
    
    if verbose:
        new_range = new_positions.max(axis=0) - new_positions.min(axis=0)
        print(f"    New position range: [{new_positions.min():.4f}, {new_positions.max():.4f}]")
        print(f"    New extent: [{new_range[0]:.4f}, {new_range[1]:.4f}, {new_range[2]:.4f}]")
    
    # Update positions in PLY data
    ply_data['x'] = new_positions[:, 0].astype(np.float32)
    ply_data['y'] = new_positions[:, 1].astype(np.float32)
    ply_data['z'] = new_positions[:, 2].astype(np.float32)
    
    # Transform Gaussian rotations if present
    has_rotation = all(key in ply_data for key in ['rot_0', 'rot_1', 'rot_2', 'rot_3'])
    if has_rotation:
        # Stack quaternions in [w, x, y, z] order (3DGS convention)
        old_quaternions = np.stack([
            ply_data['rot_0'],  # w
            ply_data['rot_1'],  # x
            ply_data['rot_2'],  # y
            ply_data['rot_3']   # z
        ], axis=1)
        
        # Transform quaternions
        new_quaternions = transform_quaternions_batch(old_quaternions, R)
        
        # Update PLY data
        ply_data['rot_0'] = new_quaternions[:, 0]
        ply_data['rot_1'] = new_quaternions[:, 1]
        ply_data['rot_2'] = new_quaternions[:, 2]
        ply_data['rot_3'] = new_quaternions[:, 3]
        
        if verbose:
            print(f"    Transformed {num_points} quaternions")
    
    # Transform scales (in log space)
    # new_log_scale = old_log_scale + log(s)
    log_scale = np.log(s) if s > 0 else 0.0
    
    if 'scale_0' in ply_data:
        ply_data['scale_0'] = (ply_data['scale_0'] + log_scale).astype(np.float32)
    if 'scale_1' in ply_data:
        ply_data['scale_1'] = (ply_data['scale_1'] + log_scale).astype(np.float32)
    if 'scale_2' in ply_data:
        ply_data['scale_2'] = (ply_data['scale_2'] + log_scale).astype(np.float32)
    
    if verbose and 'scale_0' in ply_data:
        print(f"    Log scale adjustment: {log_scale:.6f}")
    
    # Save modified PLY
    save_ply(ply_data, output_path)
    
    return True


def find_pairs(gs_dir: str, posmap_dir: str) -> List[Tuple[str, str, str, str]]:
    """
    Find all matching PLY and NPY pairs.
    
    Args:
        gs_dir: Directory containing 3DGS/{caseName}/*.ply files
        posmap_dir: Directory containing posmap/{caseName}/*.npy files
        
    Returns:
        List of (ply_path, posmap_path, case_name, file_name) tuples
    """
    pairs = []
    missing_posmaps = []
    
    # Find all PLY files
    ply_pattern = os.path.join(gs_dir, "**", "*.ply")
    ply_files = sorted(glob(ply_pattern, recursive=True))
    
    for ply_path in ply_files:
        # Get relative path from gs_dir
        rel_path = os.path.relpath(ply_path, gs_dir)
        
        # Construct corresponding NPY path
        npy_rel_path = str(Path(rel_path).with_suffix('.npy'))
        posmap_path = os.path.join(posmap_dir, npy_rel_path)
        
        if os.path.exists(posmap_path):
            case_name = str(Path(rel_path).parent)
            file_name = Path(rel_path).stem
            pairs.append((ply_path, posmap_path, case_name, file_name))
        else:
            missing_posmaps.append(ply_path)
    
    if missing_posmaps:
        print(f"Warning: {len(missing_posmaps)} PLY files have no matching position maps")
        if len(missing_posmaps) <= 5:
            for p in missing_posmaps:
                print(f"  - {p}")
        else:
            for p in missing_posmaps[:3]:
                print(f"  - {p}")
            print(f"  ... and {len(missing_posmaps) - 3} more")
    
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Denormalize 3DGS targets using position maps with Procrustes analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python denormalize_3dgs_targets.py \\
        --gs_dir /path/to/3DGS \\
        --posmap_dir /path/to/posmap \\
        --attribute_map_path /path/to/base_attribute_map.npy \\
        --output_dir /path/to/final

This script uses Procrustes analysis to find an optimal similarity transformation
(rotation, uniform scale, translation) that aligns the original 3DGS positions
to the target positions from position maps. This preserves the relative topology
of the Gaussians while moving them to the target coordinate space.

Directory structure:
    Input:
        3DGS/{caseName}/1_Forward.ply
        posmap/{caseName}/1_Forward.npy
    
    Output:
        final/{caseName}/1_Forward.ply
        """
    )
    
    parser.add_argument('--gs_dir', type=str, default='/mnt/d/zjh/test_results/berserker_s1/AR/FVD_3dgs',
                       help='Directory containing 3DGS PLY files (3DGS/{caseName}/*.ply)')
    parser.add_argument('--posmap_dir', type=str, default='/mnt/d/zjh/test_results/berserker_s1/AR/FVD/0-w_00001_Idle',
                       help='Directory containing position map NPY files (posmap/{caseName}/*.npy)')
    parser.add_argument('--attribute_map_path', type=str, default="/mnt/d/zjh/dataset/berserker_s1/base_attrmap.npy",
                       help='Path to the base attribute map (.npy file, shape 1024x1024x14)')
    parser.add_argument('--output_dir', type=str, default='/mnt/d/zjh/test_results/berserker_s1/AR/final_3dgs',
                       help='Output directory for denormalized PLY files')
    parser.add_argument('--target_resolution', type=int, default=1024,
                       help='Target resolution for upscaling (default: 1024)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed processing information')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.gs_dir):
        raise FileNotFoundError(f"3DGS directory not found: {args.gs_dir}")
    
    if not os.path.exists(args.posmap_dir):
        raise FileNotFoundError(f"Position maps directory not found: {args.posmap_dir}")
    
    if not os.path.exists(args.attribute_map_path):
        raise FileNotFoundError(f"Attribute map file not found: {args.attribute_map_path}")
    
    # Load base attribute map for mask
    print(f"Loading attribute map from: {args.attribute_map_path}")
    _, attr_mask = load_attribute_map(args.attribute_map_path)
    print(f"  Foreground pixels in attribute map: {attr_mask.sum()}")
    
    # Find all matching pairs
    print(f"\nSearching for PLY-NPY pairs...")
    print(f"  3DGS directory: {args.gs_dir}")
    print(f"  Position maps directory: {args.posmap_dir}")
    pairs = find_pairs(args.gs_dir, args.posmap_dir)
    
    if len(pairs) == 0:
        print("No matching PLY-NPY pairs found!")
        return
    
    print(f"\nFound {len(pairs)} matching pairs")
    print(f"Output directory: {args.output_dir}")
    print(f"Target resolution: {args.target_resolution}x{args.target_resolution}")
    print(f"\nUsing Procrustes analysis for topology-preserving transformation")
    print()
    
    # Statistics
    successful = 0
    failed = 0
    total_rmse = 0.0
    
    # Process each pair
    for ply_path, posmap_path, case_name, file_name in tqdm(pairs, desc="Processing pairs"):
        try:
            # Construct output path
            output_path = os.path.join(args.output_dir, case_name, f"{file_name}.ply")
            
            if args.verbose:
                print(f"\nProcessing: {ply_path}")
                print(f"  Position map: {posmap_path}")
                print(f"  Output: {output_path}")
            
            success = process_single_pair(
                ply_path,
                posmap_path,
                attr_mask,
                output_path,
                target_size=args.target_resolution,
                verbose=args.verbose
            )
            
            if success:
                successful += 1
            else:
                failed += 1
            
        except Exception as e:
            print(f"\nError processing {ply_path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            failed += 1
            continue
    
    # Summary
    print()
    print("=" * 60)
    print("Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(pairs)}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()