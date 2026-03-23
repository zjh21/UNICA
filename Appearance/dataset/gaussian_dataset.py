"""
Gaussian Dataset
================
Each sample is one animation frame.  The dataset loads an EXR position map,
combines it with a shared attribute map to build coarse 3DGS parameters,
and pairs them with ground-truth multi-view renderings for supervision.
"""

import os
import random

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import Tuple

from scene.dataset_readers_posmap import readCameraParametersFromTransforms
from utils.posmap_utils import (
    load_attribute_map,
    load_exr_position_map,
    upscale_foreground_aware,
    extract_gaussians,
)


class GaussianDataset(Dataset):
    """
    Dataset structure::

        posmap_3dgs/{case}/{frame}.exr
        renders/{case}/{frame}/transforms.json, *.png
    """

    def __init__(
        self,
        posmap_root: str,
        attribute_map_path: str,
        renders_root: str,
        num_views: int = 8,
        white_background: bool = False,
        random_background: bool = False,
        target_resolution: int = 1024,
    ):
        super().__init__()
        self.posmap_root = posmap_root
        self.renders_root = renders_root
        self.num_views = num_views
        self.white_background = white_background
        self.random_background = random_background
        self.target_resolution = target_resolution

        print(f"Loading attribute map from: {attribute_map_path}")
        self.attribute_map, self.attr_mask = load_attribute_map(attribute_map_path)
        print(f"  Shape: {self.attribute_map.shape}  |  "
              f"Foreground pixels: {self.attr_mask.sum():,}")

        self.samples = self._collect_samples()
        print(f"Found {len(self.samples)} samples")
        if random_background:
            print("Random background augmentation enabled")

    # -----------------------------------------------------------------
    # Sample collection
    # -----------------------------------------------------------------

    def _collect_samples(self):
        samples = []
        for case in sorted(os.listdir(self.posmap_root)):
            case_posmap = os.path.join(self.posmap_root, case)
            if not os.path.isdir(case_posmap):
                continue
            case_renders = os.path.join(self.renders_root, case)
            if not os.path.isdir(case_renders):
                print(f"Warning: no renders folder for case {case}")
                continue
            for exr_file in sorted(os.listdir(case_posmap)):
                if not exr_file.endswith(".exr"):
                    continue
                frame = exr_file.replace(".exr", "")
                renders_path = os.path.join(case_renders, frame)
                transforms_path = os.path.join(renders_path, "transforms.json")
                if os.path.exists(transforms_path):
                    samples.append({
                        "case": case,
                        "frame": frame,
                        "exr_path": os.path.join(case_posmap, exr_file),
                        "renders_path": renders_path,
                        "transforms_path": transforms_path,
                    })
                else:
                    print(f"Warning: no transforms.json for {case}/{frame}")
        return samples

    # -----------------------------------------------------------------
    # Position-map → coarse Gaussians
    # -----------------------------------------------------------------

    def _load_and_process_position_map(
        self, path: str, target_size: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        posmap = load_exr_position_map(path)
        posmap_tensor = torch.from_numpy(posmap).permute(2, 0, 1)  # (3, H, W)
        mask = torch.any(posmap_tensor != 0.0, dim=0, keepdim=True).expand(3, -1, -1)
        return upscale_foreground_aware(posmap_tensor, mask, target_size, bg_value=0.0)

    def _create_gaussians_from_posmap(self, exr_path: str) -> dict:
        upscaled, mask = self._load_and_process_position_map(
            exr_path, target_size=self.target_resolution,
        )
        positions_np = upscaled.permute(1, 2, 0).numpy() - 0.5
        mask_np = mask.squeeze(0).numpy()
        return extract_gaussians(positions_np, mask_np, self.attribute_map, self.attr_mask)

    # -----------------------------------------------------------------
    # Image helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _get_available_views(renders_path: str):
        return sorted(f for f in os.listdir(renders_path) if f.endswith(".png"))

    @staticmethod
    def _load_image(image_path: str, use_white_bg: bool):
        image = Image.open(image_path)
        if image.mode == "RGBA":
            data = np.array(image, dtype=np.float32) / 255.0
            rgb, alpha = data[:, :, :3], data[:, :, 3:4]
            rgb = rgb * alpha + (1 - alpha) if use_white_bg else rgb * alpha
            return torch.from_numpy(rgb), torch.from_numpy(alpha)
        data = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(data), None

    # -----------------------------------------------------------------
    # __len__ / __getitem__
    # -----------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        use_white_bg = random.random() > 0.5 if self.random_background else self.white_background

        gs_params = self._create_gaussians_from_posmap(sample["exr_path"])

        cam_params_list = readCameraParametersFromTransforms(
            sample["renders_path"], "transforms.json", extension=".png",
        )
        cam_dict = {c["image_name"]: c for c in cam_params_list}

        available = self._get_available_views(sample["renders_path"])
        sampled_names = random.sample(available, min(self.num_views, len(available)))

        images, camera_infos = [], []
        for name in sampled_names:
            img_path = os.path.join(sample["renders_path"], name)
            img_tensor, alpha_tensor = self._load_image(img_path, use_white_bg)
            images.append(img_tensor)

            base = name.replace(".png", "")
            cam = cam_dict.get(base)
            if cam is None:
                for cn, cv in cam_dict.items():
                    try:
                        if int(cn) == int(base):
                            cam = cv
                            break
                    except ValueError:
                        if cn.lstrip("0") == base.lstrip("0"):
                            cam = cv
                            break
            if cam is None:
                raise ValueError(f"Camera not found for view {name}")

            camera_infos.append({
                "R": cam["R"],
                "T": cam["T"],
                "FovX": cam["FovX"],
                "uid": cam["uid"],
                "image_name": cam["image_name"],
                "image_tensor": img_tensor.permute(2, 0, 1),
                "alpha_tensor": alpha_tensor.permute(2, 0, 1) if alpha_tensor is not None else None,
            })

        return {
            "gs_params": gs_params,
            "camera_infos": camera_infos,
            "images": images,
            "sample_info": sample,
            "white_background": use_white_bg,
        }


def gaussian_collate_fn(batch):
    """Identity collate — each sample may have a different number of Gaussians."""
    return batch