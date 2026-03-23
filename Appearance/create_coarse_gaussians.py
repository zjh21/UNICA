#!/usr/bin/env python3
"""
Script to create 3DGS targets from position maps.

This script:
1. Loads all EXR position maps from a folder and its subfolders
2. Downscales to 128x128, then upscales to 1024x1024 with foreground-aware interpolation
3. Normalizes positions to AABB [-0.5, 0.5]
4. Combines with base attribute map (SH, scaling, rotation, opacity)
5. Saves as PLY files, preserving the original folder structure

Usage:
    python create_3dgs_targets.py \
        --position_maps_dir /path/to/position_maps \
        --attribute_map_path /path/to/attribute_map.npy \
        --output_dir /path/to/output
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
from pathlib import Path
import argparse
from glob import glob
from tqdm import tqdm
from plyfile import PlyData, PlyElement
import pdb

# For position map preprocessing
from diffusers.image_processor import VaeImageProcessor


def upscale_foreground_aware(
    position_map: torch.Tensor,
    mask: torch.Tensor,
    target_size: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Upscale position map with foreground-aware interpolation.
    Background pixels don't contribute to interpolation.

    Args:
        position_map: (3, H, W) tensor with values in [-1, 1], background is -1
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

    # Set background pixels to -1
    upscaled_posmap = upscaled_posmap.squeeze(0)  # (3, target_size, target_size)
    upscaled_mask_3ch = upscaled_mask[0].expand_as(upscaled_posmap)  # Remove batch dim and expand
    upscaled_posmap = torch.where(
        upscaled_mask_3ch,
        upscaled_posmap,
        torch.tensor(-1.0, dtype=upscaled_posmap.dtype, device=upscaled_posmap.device)
    )

    upscaled_mask = upscaled_mask[0]  # (1, target_size, target_size) - use indexing instead of squeeze
    
    return upscaled_posmap, upscaled_mask


def load_and_process_position_map(
    position_map_path: str,
    preprocessor: VaeImageProcessor,
    target_size: int = 1024
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load position map and process it through downscale -> upscale pipeline.
    
    Args:
        position_map_path: Path to the .exr position map
        preprocessor: VaeImageProcessor for downscaling
        target_size: Target resolution for upscaling
        
    Returns:
        upscaled_posmap: (3, target_size, target_size) tensor in [-1, 1] range
        upscaled_mask: (1, target_size, target_size) boolean tensor
    """
    # Load position map with OpenCV
    posmap = cv2.imread(position_map_path, cv2.IMREAD_UNCHANGED)
    
    if posmap is None:
        raise ValueError(f"Could not load position map from {position_map_path}")
    
    # OpenCV loads BGR, but for position maps we likely have XYZ in channels
    # The VaeImageProcessor expects RGB order, so we may need to handle this
    # However, for position maps, channel order represents XYZ coordinates
    # Let's keep the original order (assuming EXR stores as RGB = XYZ)
    
    # Ensure float32
    posmap = posmap.astype(np.float32)
    
    # Normalize to [0, 1] if values are outside this range
    # EXR files may have values in various ranges
    if posmap.max() > 1.0 or posmap.min() < 0.0:
        # Check if it's in 0-255 range
        if posmap.max() > 1.0 and posmap.max() <= 255.0:
            posmap = posmap / 255.0
        else:
            # Normalize based on actual range
            pos_min = posmap.min()
            pos_max = posmap.max()
            if pos_max > pos_min:
                posmap = (posmap - pos_min) / (pos_max - pos_min)
    
    # Downscale to 128x128 using VaeImageProcessor
    # This returns tensor of shape (1, 3, 128, 128) in [-1, 1] range
    # posmap_128 = preprocessor.preprocess(posmap, 128, 128).squeeze(0)  # (3, 128, 128)
    posmap_128 = torch.from_numpy(posmap).permute(2,0,1)
    
    # Extract foreground mask at 128x128 (background is -1)
    mask_128 = (posmap_128 != 0.5)  # (3, 128, 128) boolean tensor
    
    # Upscale to target resolution with foreground-aware interpolation
    upscaled_posmap, upscaled_mask = upscale_foreground_aware(
        posmap_128,
        mask_128,
        target_size=target_size
    )
    
    return upscaled_posmap, upscaled_mask


def normalize_positions_to_aabb(
    positions: torch.Tensor,
    mask: torch.Tensor,
    aabb_min: float = -0.5,
    aabb_max: float = 0.5
) -> torch.Tensor:
    """
    Normalize positions to fit in AABB with bounding box centered at origin.
    
    Args:
        positions: (3, H, W) - positions to normalize (in any coordinate space)
        mask: (1, H, W) or (H, W) - foreground mask
        aabb_min: Minimum value of AABB
        aabb_max: Maximum value of AABB
        
    Returns:
        Normalized positions (3, H, W) in [aabb_min, aabb_max] range
    """
    # Ensure mask is 2D for indexing
    if mask.dim() == 3:
        mask_2d = mask.squeeze(0)  # (H, W)
    else:
        mask_2d = mask
    
    # Get foreground positions
    foreground_positions = positions[:, mask_2d]  # (3, N)
    
    if foreground_positions.shape[1] == 0:
        return positions
    
    # Find bounding box
    pos_min = foreground_positions.min(dim=1)[0]  # (3,)
    pos_max = foreground_positions.max(dim=1)[0]  # (3,)
    
    # Calculate center
    bbox_center = (pos_min + pos_max) / 2  # (3,)
    
    # Center the positions
    pos_centered = positions - bbox_center.view(3, 1, 1)  # (3, H, W)
    
    # Calculate longest axis
    pos_range = pos_max - pos_min  # (3,)
    longest_axis = pos_range.max()  # scalar
    
    # Scale to fit AABB
    if longest_axis > 0:
        scale = (aabb_max - aabb_min) / longest_axis
        pos_scaled = pos_centered * scale
    else:
        pos_scaled = pos_centered
    
    return pos_scaled


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


def construct_and_save_ply(
    positions: np.ndarray,
    attributes: np.ndarray,
    mask: np.ndarray,
    output_path: str
):
    """
    Construct Gaussian attributes and save as PLY file.
    
    The attribute map has 14 channels:
    - 0-2: xyz (discarded, use positions from position maps)
    - 3-5: SH DC (spherical harmonics, rank 0)
    - 6-8: scaling (log scale, stored as raw values)
    - 9-12: rotation (quaternion, stored as raw values)
    - 13: opacity (inverse sigmoid, stored as raw values)
    
    Args:
        positions: (H, W, 3) normalized positions in [-0.5, 0.5]
        attributes: (H, W, 14) attribute map
        mask: (H, W) boolean mask
        output_path: Path to save the PLY file
    """
    # Extract foreground points
    fg_positions = positions[mask]  # (N, 3)
    fg_attributes = attributes[mask]  # (N, 14)
    
    num_points = fg_positions.shape[0]
    
    if num_points == 0:
        print(f"Warning: No foreground points found, skipping {output_path}")
        return
    
    # Extract individual attributes
    # Channels: 0-2: xyz (discard), 3-5: SH DC, 6-8: scaling, 9-12: rotation, 13: opacity
    sh_dc = fg_attributes[:, 3:6]  # (N, 3)
    scaling = fg_attributes[:, 6:9]  # (N, 3)
    rotation = fg_attributes[:, 9:13]  # (N, 4)
    opacity = fg_attributes[:, 13:14]  # (N, 1)
    
    # Create PLY data structure following standard 3DGS format
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ]
    
    elements = np.empty(num_points, dtype=dtype_full)
    
    # Fill in the data
    elements['x'] = fg_positions[:, 0].astype(np.float32)
    elements['y'] = fg_positions[:, 1].astype(np.float32)
    elements['z'] = fg_positions[:, 2].astype(np.float32)
    
    # Normals (set to 0, not used in 3DGS but often included in PLY)
    elements['nx'] = np.zeros(num_points, dtype=np.float32)
    elements['ny'] = np.zeros(num_points, dtype=np.float32)
    elements['nz'] = np.zeros(num_points, dtype=np.float32)
    
    # SH DC coefficients
    elements['f_dc_0'] = sh_dc[:, 0].astype(np.float32)
    elements['f_dc_1'] = sh_dc[:, 1].astype(np.float32)
    elements['f_dc_2'] = sh_dc[:, 2].astype(np.float32)
    
    # Opacity (stored as inverse sigmoid value)
    elements['opacity'] = opacity[:, 0].astype(np.float32)
    
    # Scaling (stored as log scale value)
    elements['scale_0'] = scaling[:, 0].astype(np.float32)
    elements['scale_1'] = scaling[:, 1].astype(np.float32)
    elements['scale_2'] = scaling[:, 2].astype(np.float32)
    
    # Rotation quaternion
    elements['rot_0'] = rotation[:, 0].astype(np.float32)
    elements['rot_1'] = rotation[:, 1].astype(np.float32)
    elements['rot_2'] = rotation[:, 2].astype(np.float32)
    elements['rot_3'] = rotation[:, 3].astype(np.float32)
    
    # Create PLY element and save
    el = PlyElement.describe(elements, 'vertex')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    PlyData([el]).write(output_path)


def get_output_path(exr_path: str, input_root: str, output_root: str) -> str:
    """
    Get output PLY path preserving the folder structure.
    
    Args:
        exr_path: Full path to the EXR file
        input_root: Root input directory
        output_root: Root output directory
        
    Returns:
        Output PLY path with same relative structure
    """
    # Get relative path from input root
    rel_path = os.path.relpath(exr_path, input_root)
    
    # Change extension from .exr to .ply
    rel_path_ply = str(Path(rel_path).with_suffix('.ply'))
    
    # Combine with output root
    output_path = os.path.join(output_root, rel_path_ply)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create 3DGS targets from position maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python create_3dgs_targets.py \\
        --position_maps_dir /path/to/position_maps \\
        --attribute_map_path /path/to/base_attribute_map.npy \\
        --output_dir /path/to/output_ply_files

The script will:
1. Find all .exr files in the position_maps_dir and its subfolders
2. Process each through downscale (128x128) -> upscale (1024x1024) pipeline
3. Normalize positions to AABB [-0.5, 0.5]
4. Combine with base attribute map
5. Save as PLY files preserving the original folder structure
        """
    )
    
    parser.add_argument('--position_maps_dir', type=str, default="/media/muliyanpo/BlackDisk/CVPR/ADNAS/test_results/bandit/posmap",
                       help='Root directory containing position maps (will search recursively for .exr files)')
    parser.add_argument('--attribute_map_path', type=str, default="/media/muliyanpo/BlackDisk/CVPR/ADNAS/Datasets/bandit/base_attrmap.npy",
                       help='Path to the base attribute map (.npy file, shape 1024x1024x14)')
    parser.add_argument('--output_dir', type=str, default="/media/muliyanpo/BlackDisk/CVPR/ADNAS/test_results/bandit/coarse_3dgs",
                       help='Output directory for PLY files (will mirror input folder structure)')
    parser.add_argument('--target_resolution', type=int, default=1024,
                       help='Target resolution for upscaling (default: 1024)')
    parser.add_argument('--aabb_min', type=float, default=-0.5,
                       help='Minimum value of AABB for normalization (default: -0.5)')
    parser.add_argument('--aabb_max', type=float, default=0.5,
                       help='Maximum value of AABB for normalization (default: 0.5)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed processing information')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.position_maps_dir):
        raise FileNotFoundError(f"Position maps directory not found: {args.position_maps_dir}")
    
    if not os.path.exists(args.attribute_map_path):
        raise FileNotFoundError(f"Attribute map file not found: {args.attribute_map_path}")
    
    # Initialize preprocessor
    print("Initializing VaeImageProcessor...")
    preprocessor = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True)
    
    # Load base attribute map
    print(f"Loading attribute map from: {args.attribute_map_path}")
    attribute_map, attr_mask = load_attribute_map(args.attribute_map_path)
    print(f"  Attribute map shape: {attribute_map.shape}")
    print(f"  Foreground pixels in attribute map: {attr_mask.sum()}")
    
    # Find all EXR files recursively
    exr_pattern = os.path.join(args.position_maps_dir, "**", "*.exr")
    position_maps = sorted(glob(exr_pattern, recursive=True))
    
    if len(position_maps) == 0:
        print(f"No .exr files found in: {args.position_maps_dir}")
        return
    
    print(f"Found {len(position_maps)} EXR files")
    print(f"Output directory: {args.output_dir}")
    print(f"Target resolution: {args.target_resolution}x{args.target_resolution}")
    print(f"AABB range: [{args.aabb_min}, {args.aabb_max}]")
    print()
    
    # Statistics
    successful = 0
    failed = 0
    skipped = 0
    
    # Process each position map
    for posmap_path in tqdm(position_maps, desc="Processing position maps"):
        try:
            # Get output path preserving folder structure
            output_path = get_output_path(
                posmap_path, args.position_maps_dir, args.output_dir
            )
            
            if args.verbose:
                print(f"\nProcessing: {posmap_path}")
                print(f"  Output: {output_path}")
            
            # Load and process position map
            upscaled_posmap, posmap_mask = load_and_process_position_map(
                posmap_path,
                preprocessor,
                target_size=args.target_resolution
            )
            
            # Convert from [-1, 1] to [0, 1]
            positions = (upscaled_posmap + 1.0) / 2.0
            
            # Normalize to AABB
            normalized_positions = normalize_positions_to_aabb(
                positions,
                posmap_mask,
                aabb_min=args.aabb_min,
                aabb_max=args.aabb_max
            )
            
            # Convert to numpy and transpose to (H, W, 3)
            normalized_positions_np = normalized_positions.permute(1, 2, 0).numpy()
            posmap_mask_np = posmap_mask.squeeze(0).numpy()
            
            # Combine masks: use attribute map mask (more precise) AND position map mask
            # This ensures we only include pixels that are foreground in BOTH
            combined_mask = attr_mask & posmap_mask_np
            
            if combined_mask.sum() == 0:
                if args.verbose:
                    print(f"  Warning: No overlapping foreground pixels, skipping")
                skipped += 1
                continue
            
            if args.verbose:
                print(f"  Position map foreground: {posmap_mask_np.sum()}")
                print(f"  Attribute map foreground: {attr_mask.sum()}")
                print(f"  Combined mask foreground: {combined_mask.sum()}")
            
            # Save PLY
            construct_and_save_ply(
                normalized_positions_np,
                attribute_map,
                combined_mask,
                output_path
            )
            
            if args.verbose:
                print(f"  Saved to: {output_path}")
            
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {posmap_path}: {e}")
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
    print(f"  Skipped (no overlap): {skipped}")
    print(f"  Total: {len(position_maps)}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()