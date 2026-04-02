#!/usr/bin/env python3
"""
PTv3 Gaussian Refiner — Inference Script
=========================================
Processes EXR position maps through the trained PTv3 refiner and outputs
PLY files containing refined 3D Gaussians.

With ``--progressive``, the script additionally denormalizes the refined
Gaussians into world-space positions using Procrustes analysis on paired
NPY position maps (requires ``save_npy=true`` during geometry inference).

Usage::

    python inference.py --config configs/inference.yaml
    python inference.py --config configs/inference.yaml --progressive
    python inference.py --config configs/inference.yaml single_file.input=path/to/file.exr
"""

import os
import time
import warnings
from glob import glob

import torch
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Dict, List, Tuple
from omegaconf import OmegaConf

from models.ptv3_refiner import PTv3GaussianRefiner
from utils.posmap_utils import (
    load_attribute_map,
    load_exr_position_map,
    upscale_foreground_aware,
    normalize_positions_to_aabb,
    extract_gaussians,
    save_ply,
)
from utils.progressive import run_progressive


class PTv3Inference:
    """
    Inference pipeline for the PTv3 Gaussian Refiner.

    Loads an EXR position map, combines it with a shared attribute map to
    build coarse Gaussians, refines them through the trained network, and
    writes the result as a PLY file.
    """

    def __init__(
        self,
        attribute_map_path: str,
        checkpoint_path: str,
        target_resolution: int = 1024,
        aabb_min: float = -0.5,
        aabb_max: float = 0.5,
        device: str = "cuda",
    ):
        self.target_resolution = target_resolution
        self.aabb_min = aabb_min
        self.aabb_max = aabb_max
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        sep = "=" * 70
        print(sep)
        print("PTv3 Gaussian Refiner — Inference")
        print(sep)

        print(f"Loading attribute map from: {attribute_map_path}")
        self.attribute_map, self.attr_mask = load_attribute_map(attribute_map_path)
        print(f"  Shape: {self.attribute_map.shape}  |  "
              f"Foreground pixels: {self.attr_mask.sum():,}")

        print(f"Loading model from: {checkpoint_path}")
        self.model = self._load_model(checkpoint_path)
        print(f"  Device: {self.device}")
        print(sep)

    # -----------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------

    def _load_model(self, checkpoint_path: str) -> PTv3GaussianRefiner:
        model = PTv3GaussianRefiner()
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "epoch" in ckpt:
                print(f"  Checkpoint epoch: {ckpt['epoch'] + 1}")
            if "train_psnr" in ckpt:
                print(f"  Training PSNR: {ckpt['train_psnr']:.2f}")
        else:
            model.load_state_dict(ckpt)

        model = model.to(self.device).eval()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")
        return model

    # -----------------------------------------------------------------
    # Position-map processing
    # -----------------------------------------------------------------

    def _load_and_process_position_map(
        self, path: str, target_size: int = 1024,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        posmap = load_exr_position_map(path)

        # Normalise to [0, 1] if needed.
        if posmap.max() > 1.0 or posmap.min() < 0.0:
            if posmap.max() <= 255.0:
                posmap = posmap / 255.0
            else:
                lo, hi = posmap.min(), posmap.max()
                if hi > lo:
                    posmap = (posmap - lo) / (hi - lo)

        posmap_t = torch.from_numpy(posmap).permute(2, 0, 1)
        mask = posmap_t != 0.5
        return upscale_foreground_aware(posmap_t, mask, target_size, bg_value=-1.0)

    def _create_gaussians_from_posmap(
        self, exr_path: str,
    ) -> Dict[str, torch.Tensor]:
        upscaled, mask = self._load_and_process_position_map(
            exr_path, target_size=self.target_resolution,
        )
        positions = (upscaled + 1.0) / 2.0
        normalized, _ = normalize_positions_to_aabb(
            positions, mask, self.aabb_min, self.aabb_max,
        )
        positions_np = normalized.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze(0).numpy()
        return extract_gaussians(positions_np, mask_np, self.attribute_map, self.attr_mask)

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    def process_single(
        self, exr_path: str,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Process a single EXR file through the PTv3 model.

        Returns:
            Tuple of (output_gs, input_gs) parameter dictionaries.
        """
        gs_params = self._create_gaussians_from_posmap(exr_path)
        input_gs = {k: v.clone() for k, v in gs_params.items()}
        gs_device = {k: v.to(self.device) for k, v in gs_params.items()}

        with torch.no_grad():
            out_gs = self.model([gs_device])[0]

        out_gs = {k: v.cpu() for k, v in out_gs.items()}
        return out_gs, input_gs

    # -----------------------------------------------------------------
    # Directory helpers
    # -----------------------------------------------------------------

    @staticmethod
    def find_exr_files(input_dir: str) -> List[str]:
        exr_files = []
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(".exr"):
                    exr_files.append(os.path.join(root, f))
        return sorted(exr_files)

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        save_input: bool = False,
        verbose: bool = True,
    ) -> Dict[str, int]:
        """Process all EXR files under *input_dir* and write PLY results."""
        exr_files = self.find_exr_files(input_dir)
        print(f"Found {len(exr_files)} EXR files in {input_dir}")
        if not exr_files:
            print("Nothing to process.")
            return {"total": 0, "success": 0, "failed": 0}

        os.makedirs(output_dir, exist_ok=True)
        stats = {"total": len(exr_files), "success": 0, "failed": 0}
        failed = []
        t0 = time.time()

        for exr_path in tqdm(exr_files, desc="Processing", disable=not verbose):
            try:
                rel = os.path.relpath(exr_path, input_dir)
                out_sub = os.path.join(output_dir, os.path.dirname(rel))
                os.makedirs(out_sub, exist_ok=True)
                base = os.path.splitext(os.path.basename(exr_path))[0]

                out_gs, in_gs = self.process_single(exr_path)
                save_ply(out_gs, os.path.join(out_sub, f"{base}.ply"))
                if save_input:
                    save_ply(in_gs, os.path.join(out_sub, f"{base}_input.ply"))
                stats["success"] += 1
            except Exception as e:
                stats["failed"] += 1
                failed.append((exr_path, str(e)))
                if verbose:
                    print(f"\nError: {exr_path}: {e}")

        elapsed = time.time() - t0
        sep = "=" * 70
        print(f"\n{sep}")
        print("Processing Complete")
        print(sep)
        print(f"  Total: {stats['total']}  |  "
              f"OK: {stats['success']}  |  Failed: {stats['failed']}")
        print(f"  Elapsed: {elapsed:.2f}s", end="")
        if stats["success"]:
            print(f"  ({elapsed / stats['success']:.3f}s per file)")
        else:
            print()
        print(f"  Output: {output_dir}")
        if failed:
            print("\nFailed files:")
            for p, e in failed[:10]:
                print(f"  {p}: {e}")
            if len(failed) > 10:
                print(f"  … and {len(failed) - 10} more")
        print(sep)
        return stats

    def process_single_file(
        self,
        exr_path: str,
        output_path: str,
        save_input: bool = False,
    ):
        """Process one EXR file and save the output PLY."""
        print(f"Processing: {exr_path}")

        out_gs, in_gs = self.process_single(exr_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        save_ply(out_gs, output_path)
        print(f"Saved output: {output_path}")

        if save_input:
            base, ext = os.path.splitext(output_path)
            in_path = f"{base}_input{ext}"
            save_ply(in_gs, in_path)
            print(f"Saved input:  {in_path}")

        print(f"  Input Gaussians:  {in_gs['means'].shape[0]:,}")
        print(f"  Output Gaussians: {out_gs['means'].shape[0]:,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="PTv3 Inference — EXR → PLY")
    parser.add_argument("--config", type=str, default="configs/inference.yaml",
                        help="Path to the YAML config file")
    parser.add_argument("--progressive", action="store_true",
                        help="Enable progressive 4D inference "
                             "(put 3DGS to world coordinates for actual avatar movement)")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    # Resolve relative paths against the project root so the script works
    # regardless of the working directory (e.g. running from Appearance/).
    _project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
    _project_root = os.path.normpath(_project_root)
    for _key in ['input_dir', 'output_dir', 'attribute_map_path', 'checkpoint']:
        _val = OmegaConf.select(cfg, f"paths.{_key}")
        if _val is not None and not os.path.isabs(_val):
            OmegaConf.update(cfg, f"paths.{_key}", os.path.join(_project_root, _val))
    for _key in ['input', 'output']:
        _val = OmegaConf.select(cfg, f"single_file.{_key}")
        if _val is not None and not os.path.isabs(_val):
            OmegaConf.update(cfg, f"single_file.{_key}", os.path.join(_project_root, _val))

    if cfg.single_file.input is not None and cfg.single_file.output is None:
        base = os.path.splitext(os.path.basename(cfg.single_file.input))[0]
        cfg.single_file.output = os.path.join(cfg.paths.output_dir, f"{base}.ply")

    engine = PTv3Inference(
        attribute_map_path=cfg.paths.attribute_map_path,
        checkpoint_path=cfg.paths.checkpoint,
        target_resolution=cfg.processing.target_resolution,
        aabb_min=cfg.processing.aabb_min,
        aabb_max=cfg.processing.aabb_max,
        device=cfg.processing.device,
    )

    if cfg.single_file.input is not None:
        engine.process_single_file(
            exr_path=cfg.single_file.input,
            output_path=cfg.single_file.output,
            save_input=cfg.processing.save_input,
        )
    else:
        engine.process_directory(
            input_dir=cfg.paths.input_dir,
            output_dir=cfg.paths.output_dir,
            save_input=cfg.processing.save_input,
            verbose=not cfg.processing.quiet,
        )

    # ------------------------------------------------------------------
    # Progressive 4D: denormalize refined PLYs into world space
    # ------------------------------------------------------------------
    if args.progressive:
        input_dir = cfg.paths.input_dir
        output_dir = cfg.paths.output_dir
        verbose = not cfg.processing.quiet

        # Check whether NPY position maps exist in the input directory
        npy_files = glob(os.path.join(input_dir, "**", "*.npy"), recursive=True)
        if not npy_files:
            warnings.warn(
                "Progressive 4D inference is not available due to missing "
                "npy files. Please re-run Geometry/inference.py with "
                "save_npy=true."
            )
        else:
            run_progressive(
                ply_dir=output_dir,
                npy_dir=input_dir,
                attr_mask=engine.attr_mask,
                target_resolution=cfg.processing.target_resolution,
                verbose=verbose,
            )


if __name__ == "__main__":
    main()