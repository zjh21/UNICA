"""
Shared utility functions for Gaussian Splatting operations.

Provides position-map loading, foreground-aware upscaling, attribute
extraction, PLY I/O, camera construction, and rendering helpers that are
used by both the training pipeline and the inference script.
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from plyfile import PlyData, PlyElement

from scene.cameras import Camera


# ---------------------------------------------------------------------------
# Attribute-map helpers
# ---------------------------------------------------------------------------

def load_attribute_map(attribute_map_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the base attribute map and derive a foreground mask.

    The attribute map stores 14 channels per pixel:
    0-2  xyz (base positions),  3-5  SH DC coefficients,
    6-8  log-space scaling,     9-12 rotation quaternions,  13 opacity.

    Args:
        attribute_map_path: Path to the ``.npy`` file (expected shape 1024×1024×14).

    Returns:
        attribute_map: ``(1024, 1024, 14)`` float array.
        mask: ``(1024, 1024)`` boolean array (``True`` = foreground).
    """
    if not os.path.exists(attribute_map_path):
        raise FileNotFoundError(f"Attribute map not found: {attribute_map_path}")

    attribute_map = np.load(attribute_map_path)
    if attribute_map.shape != (1024, 1024, 14):
        raise ValueError(
            f"Expected attribute map shape (1024, 1024, 14), got {attribute_map.shape}"
        )

    mask = np.any(attribute_map != 0, axis=2)
    return attribute_map, mask


# ---------------------------------------------------------------------------
# Position-map helpers
# ---------------------------------------------------------------------------

def load_exr_position_map(path: str) -> np.ndarray:
    """
    Load an EXR position-map file.

    Args:
        path: Path to the ``.exr`` file.

    Returns:
        ``(H, W, 3)`` float32 numpy array.
    """
    posmap = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if posmap is None:
        raise ValueError(f"Could not load position map from {path}")
    return posmap.astype(np.float32)


def upscale_foreground_aware(
    position_map: torch.Tensor,
    mask: torch.Tensor,
    target_size: int = 1024,
    bg_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Upscale a position map so that background pixels do not bleed into
    the foreground during bilinear interpolation.

    The trick is to interpolate both the (masked) position map and a
    binary weight map, then divide out the weights.

    Args:
        position_map: ``(3, H, W)`` tensor.
        mask: ``(3, H, W)`` or ``(1, H, W)`` boolean foreground mask.
        target_size: Desired spatial resolution.
        bg_value: Value written to background pixels in the output.

    Returns:
        upscaled_posmap: ``(3, target_size, target_size)`` tensor.
        upscaled_mask:   ``(1, target_size, target_size)`` boolean tensor.
    """
    # Collapse to single-channel mask.
    if mask.dim() == 3 and mask.shape[0] == 3:
        mask_1ch = mask[0:1]
    elif mask.dim() == 3 and mask.shape[0] == 1:
        mask_1ch = mask
    else:
        mask_1ch = mask.unsqueeze(0) if mask.dim() == 2 else mask

    mask_3ch = mask if mask.shape[0] == 3 else mask_1ch.expand(3, -1, -1)
    weight_map = mask_1ch.float()
    position_map_masked = position_map * mask_3ch.float()

    # Add batch dim for F.interpolate.
    upscaled_posmap = F.interpolate(
        position_map_masked.unsqueeze(0),
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    upscaled_weights = F.interpolate(
        weight_map.unsqueeze(0),
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )

    # Normalise by the interpolated weights.
    upscaled_posmap = upscaled_posmap / torch.clamp(upscaled_weights, min=1e-8)

    # Pixels with insufficient foreground contribution become background.
    upscaled_mask = upscaled_weights > 0.25
    upscaled_posmap = upscaled_posmap.squeeze(0)
    upscaled_posmap = torch.where(
        upscaled_mask[0].expand_as(upscaled_posmap),
        upscaled_posmap,
        torch.tensor(bg_value, dtype=upscaled_posmap.dtype, device=upscaled_posmap.device),
    )
    upscaled_mask = upscaled_mask[0]  # (1, H, W)

    return upscaled_posmap, upscaled_mask


def normalize_positions_to_aabb(
    positions: torch.Tensor,
    mask: torch.Tensor,
    aabb_min: float = -0.5,
    aabb_max: float = 0.5,
) -> Tuple[torch.Tensor, float]:
    """
    Centre and uniformly scale foreground positions so that the longest
    bounding-box axis spans ``[aabb_min, aabb_max]``.

    Args:
        positions: ``(3, H, W)`` position tensor.
        mask: ``(1, H, W)`` or ``(H, W)`` foreground mask.
        aabb_min: Lower bound of the target AABB.
        aabb_max: Upper bound of the target AABB.

    Returns:
        normalized: ``(3, H, W)`` normalised positions.
        norm_scale: The uniform scale factor that was applied.
    """
    mask_2d = mask.squeeze(0) if mask.dim() == 3 else mask
    fg = positions[:, mask_2d]

    if fg.shape[1] == 0:
        return positions, 1.0

    pos_min = fg.min(dim=1)[0]
    pos_max = fg.max(dim=1)[0]
    centre = (pos_min + pos_max) / 2
    centred = positions - centre.view(3, 1, 1)
    longest = (pos_max - pos_min).max()

    if longest > 0:
        scale = (aabb_max - aabb_min) / longest
        return centred * scale, float(scale)
    return centred, 1.0


# ---------------------------------------------------------------------------
# Gaussian extraction & PLY I/O
# ---------------------------------------------------------------------------

def extract_gaussians(
    positions_np: np.ndarray,
    posmap_mask_np: np.ndarray,
    attribute_map: np.ndarray,
    attr_mask: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """
    Build a Gaussian parameter dict from a position array and an attribute map.

    Foreground is the intersection of *posmap_mask_np* and *attr_mask*.
    XYZ comes from *positions_np*; all other attributes come from
    *attribute_map* (channels 3-13).

    Args:
        positions_np:  ``(H, W, 3)`` float array.
        posmap_mask_np: ``(H, W)`` boolean mask (position map).
        attribute_map:  ``(H, W, 14)`` attribute map.
        attr_mask:      ``(H, W)`` boolean mask (attribute map).

    Returns:
        Dictionary with keys ``means (N,3)``, ``features_dc (N,3)``,
        ``opacities (N,1)``, ``scales (N,3)``, ``quats (N,4)``.
    """
    combined_mask = attr_mask & posmap_mask_np
    if combined_mask.sum() == 0:
        raise ValueError(
            "No overlapping foreground pixels between position map and attribute map"
        )

    fg_pos = positions_np[combined_mask].astype(np.float32)
    fg_attr = attribute_map[combined_mask]

    return {
        "means": torch.from_numpy(fg_pos),
        "features_dc": torch.from_numpy(fg_attr[:, 3:6].astype(np.float32)),
        "opacities": torch.from_numpy(fg_attr[:, 13:14].astype(np.float32)),
        "scales": torch.from_numpy(fg_attr[:, 6:9].astype(np.float32)),
        "quats": torch.from_numpy(fg_attr[:, 9:13].astype(np.float32)),
    }


def save_ply(gs_dict: Dict[str, torch.Tensor], path: str) -> None:
    """
    Write Gaussian parameters to a PLY file.

    Args:
        gs_dict: Dictionary with keys ``means``, ``features_dc``,
                 ``opacities``, ``scales``, ``quats``.
        path: Output ``.ply`` file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    xyz = gs_dict["means"].detach().cpu().numpy()
    features_dc = gs_dict["features_dc"].detach().cpu().numpy()
    opacities = gs_dict["opacities"].detach().cpu().numpy().reshape(-1)
    scales = gs_dict["scales"].detach().cpu().numpy()
    quats = gs_dict["quats"].detach().cpu().numpy()

    n = xyz.shape[0]
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    elements = np.empty(n, dtype=dtype)
    elements["x"] = xyz[:, 0];        elements["y"] = xyz[:, 1];        elements["z"] = xyz[:, 2]
    elements["f_dc_0"] = features_dc[:, 0]
    elements["f_dc_1"] = features_dc[:, 1]
    elements["f_dc_2"] = features_dc[:, 2]
    elements["opacity"] = opacities
    elements["scale_0"] = scales[:, 0]; elements["scale_1"] = scales[:, 1]; elements["scale_2"] = scales[:, 2]
    elements["rot_0"] = quats[:, 0];   elements["rot_1"] = quats[:, 1]
    elements["rot_2"] = quats[:, 2];   elements["rot_3"] = quats[:, 3]

    PlyElement.describe(elements, "vertex")
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


# ---------------------------------------------------------------------------
# Renderable wrapper (used by the 3DGS diff-renderer)
# ---------------------------------------------------------------------------

class RenderableGaussians:
    """
    Thin wrapper that gives a ``gs_params`` dictionary the same attribute
    interface that the standard 3DGS renderer expects from *GaussianModel*.
    """

    def __init__(self, gs_dict: Dict[str, torch.Tensor], sh_degree: int = 0):
        self.active_sh_degree = sh_degree
        self.max_sh_degree = sh_degree

        self._xyz = gs_dict["means"]

        features_dc = gs_dict["features_dc"]
        if features_dc.dim() == 2:
            features_dc = features_dc.unsqueeze(1)
        self._features_dc = features_dc
        self._features_rest = torch.zeros(
            self._xyz.shape[0], 0, 3,
            device=self._xyz.device, dtype=self._xyz.dtype,
        )

        self._scaling = gs_dict["scales"]
        self._rotation = gs_dict["quats"]
        self._opacity = gs_dict["opacities"]

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return F.normalize(self._rotation, dim=-1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def get_covariance(self, scaling_modifier: float = 1.0):
        S = self.get_scaling * scaling_modifier
        R = self._build_rotation_matrix(self.get_rotation)
        L = torch.zeros((S.shape[0], 3, 3), dtype=S.dtype, device=S.device)
        L[:, 0, 0] = S[:, 0]
        L[:, 1, 1] = S[:, 1]
        L[:, 2, 2] = S[:, 2]
        RS = torch.bmm(R, L)
        return torch.bmm(RS, RS.transpose(1, 2))

    @staticmethod
    def _build_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.zeros((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R


# ---------------------------------------------------------------------------
# Camera / device / metric helpers
# ---------------------------------------------------------------------------

def create_camera_from_info(
    cam_info: dict,
    data_device: str = "cuda",
) -> Camera:
    """
    Construct a ``Camera`` object from a camera-info dictionary produced
    by the dataset.  Must be called in the main process (not in data
    workers) because it moves tensors to *data_device*.
    """
    return Camera(
        colmap_id=cam_info["uid"],
        R=cam_info["R"],
        T=cam_info["T"],
        FoVx=cam_info["FovX"],
        FoVy=cam_info["FovX"],  # square FOV assumed
        image=cam_info["image_tensor"],
        gt_alpha_mask=cam_info["alpha_tensor"],
        image_name=cam_info["image_name"],
        uid=cam_info["uid"],
        data_device=data_device,
    )


def move_to_device(data, device):
    """Recursively move nested tensors / dicts / lists to *device*."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    return data


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio for images in [0, 1]."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float("inf"))
    return 20 * torch.log10(1.0 / torch.sqrt(mse))