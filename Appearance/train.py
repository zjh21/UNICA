"""
PTv3 Gaussian Refiner — Training Script
========================================
Multi-GPU training with Hugging Face Accelerate.

Usage::

    accelerate launch train.py --config configs/train.yaml
    accelerate launch train.py --config configs/train.yaml training.lr=5e-5
"""

import os

import torch
import numpy as np
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from PIL import Image
from pytorch_msssim import ssim
import lpips
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.gaussian_dataset import GaussianDataset, gaussian_collate_fn
from models.ptv3_refiner import PTv3GaussianRefiner
from utils.posmap_utils import (
    RenderableGaussians,
    save_ply,
    create_camera_from_info,
    move_to_device,
    compute_psnr,
)
from gaussian_renderer import render


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_gaussians(gs_dict, camera, pipe, background, sh_degree=0):
    """Render a Gaussian dict through the differentiable rasteriser."""
    gaussians = RenderableGaussians(gs_dict, sh_degree=sh_degree)
    rendered = render(camera, gaussians, pipe, background)["render"]
    return rendered


def compute_focal_weight(ssim_value, focal_cfg):
    """
    Focal-style re-weighting: harder samples (lower SSIM) receive a higher
    loss weight.  Returns 1.0 when focal loss is disabled.
    """
    if not focal_cfg.enabled:
        return 1.0

    ssim_val = ssim_value.detach().item() if isinstance(ssim_value, torch.Tensor) else float(ssim_value)
    alpha = focal_cfg.alpha_scale * (1.0 - ssim_val)
    alpha = max(focal_cfg.alpha_min, min(focal_cfg.alpha_max, alpha))
    return alpha ** focal_cfg.gamma


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, accelerator, pipe, epoch, cfg, lpips_fn):
    model.train()
    device = accelerator.device

    total_loss = total_l1 = total_ssim = total_lpips = total_psnr = total_fw = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}") if accelerator.is_main_process else dataloader

    for batch_idx, batch in enumerate(progress):
        optimizer.zero_grad()
        b_loss = b_l1 = b_ssim = b_lpips = b_psnr = b_fw = 0.0
        n_images = 0

        for sample in batch:
            use_white = sample.get("white_background", cfg.rendering.white_background)
            bg = torch.tensor([1, 1, 1] if use_white else [0, 0, 0],
                              dtype=torch.float32, device=device)

            gs_params = move_to_device(sample["gs_params"], device)
            out_gs = model([gs_params])[0]

            for cam_info, gt_img in zip(sample["camera_infos"], sample["images"]):
                camera = create_camera_from_info(cam_info, data_device=device)
                rendered = render_gaussians(out_gs, camera, pipe, bg)
                gt_tensor = gt_img.permute(2, 0, 1).to(device)

                l1_val = torch.abs(rendered - gt_tensor).mean()

                rendered_b = rendered.unsqueeze(0)
                gt_b = gt_tensor.unsqueeze(0)
                ssim_val = ssim(rendered_b, gt_b, data_range=1.0, size_average=True)
                ssim_loss = 1.0 - ssim_val

                lpips_val = lpips_fn(rendered_b * 2 - 1, gt_b * 2 - 1).mean()

                fw = compute_focal_weight(ssim_val, cfg.loss.focal)
                loss = fw * (cfg.loss.lambda_l1 * l1_val
                             + cfg.loss.lambda_ssim * ssim_loss
                             + cfg.loss.lambda_lpips * lpips_val)

                b_loss += loss
                b_l1 += l1_val.item()
                b_ssim += ssim_loss.item()
                b_lpips += lpips_val.item()
                b_fw += fw
                with torch.no_grad():
                    b_psnr += compute_psnr(rendered, gt_tensor)
                n_images += 1

                del camera, rendered, gt_tensor, rendered_b, gt_b

            del gs_params, out_gs

        b_loss /= n_images
        accelerator.backward(b_loss)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

        total_loss += b_loss.item()
        total_l1 += b_l1 / n_images
        total_ssim += b_ssim / n_images
        total_lpips += b_lpips / n_images
        total_psnr += (b_psnr / n_images).item()
        total_fw += b_fw / n_images
        num_batches += 1

        if accelerator.is_main_process:
            info = {
                "loss": f"{b_loss.item():.4f}",
                "ssim": f"{1 - b_ssim / n_images:.4f}",
                "psnr": f"{(b_psnr / n_images).item():.2f}",
                "avg_loss": f"{total_loss / num_batches:.4f}",
                "avg_psnr": f"{total_psnr / num_batches:.2f}",
            }
            if cfg.loss.focal.enabled:
                info["fw"] = f"{b_fw / n_images:.2f}"
            progress.set_postfix(info)

    # Gather across processes.
    metrics = torch.tensor(
        [total_loss, total_l1, total_ssim, total_lpips, total_psnr, total_fw, num_batches],
        device=device,
    )
    gathered = accelerator.gather(metrics)
    if accelerator.num_processes > 1:
        total_loss = gathered[::7].sum().item()
        total_l1 = gathered[1::7].sum().item()
        total_ssim = gathered[2::7].sum().item()
        total_lpips = gathered[3::7].sum().item()
        total_psnr = gathered[4::7].sum().item()
        total_fw = gathered[5::7].sum().item()
        num_batches = gathered[6::7].sum().item()

    avg_loss = total_loss / num_batches
    avg_psnr = total_psnr / num_batches

    if accelerator.is_main_process:
        msg = (f"  Epoch {epoch + 1} — Loss: {avg_loss:.4f}, "
               f"L1: {total_l1 / num_batches:.4f}, "
               f"SSIM: {total_ssim / num_batches:.4f}, "
               f"LPIPS: {total_lpips / num_batches:.4f}, "
               f"PSNR: {avg_psnr:.2f}")
        if cfg.loss.focal.enabled:
            msg += f", Focal wt: {total_fw / num_batches:.3f}"
        print(msg)

    return avg_loss, avg_psnr


# ---------------------------------------------------------------------------
# Sample visualisation
# ---------------------------------------------------------------------------

def save_sample_output(model, dataloader, accelerator, pipe, output_dir, epoch, cfg):
    if not accelerator.is_main_process:
        return
    device = accelerator.device
    model.eval()

    sample = next(iter(dataloader))[0]
    use_white = sample.get("white_background", cfg.rendering.white_background)
    bg = torch.tensor([1, 1, 1] if use_white else [0, 0, 0],
                      dtype=torch.float32, device=device)

    gs_params = move_to_device(sample["gs_params"], device)
    unwrapped = accelerator.unwrap_model(model)

    with torch.no_grad():
        out_gs = unwrapped([gs_params])[0]

    info = sample["sample_info"]
    ply_dir = os.path.join(output_dir, "sample_outputs")
    os.makedirs(ply_dir, exist_ok=True)
    ply_path = os.path.join(
        ply_dir, f"epoch_{epoch + 1:03d}_{info['case']}_{info['frame']}.ply",
    )
    save_ply(out_gs, ply_path)

    img_dir = os.path.join(ply_dir, f"epoch_{epoch + 1:03d}_renders")
    os.makedirs(img_dir, exist_ok=True)

    with torch.no_grad():
        for i, (cam_info, gt_img) in enumerate(
            zip(sample["camera_infos"][:3], sample["images"][:3])
        ):
            camera = create_camera_from_info(cam_info, data_device=device)
            rendered = render_gaussians(out_gs, camera, pipe, bg)
            rendered_np = (rendered.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            gt_np = (gt_img.numpy() * 255).clip(0, 255).astype(np.uint8)
            input_rendered = render_gaussians(gs_params, camera, pipe, bg)
            input_np = (input_rendered.cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)

            comparison = np.concatenate([gt_np, input_np, rendered_np], axis=1)
            Image.fromarray(comparison).save(os.path.join(img_dir, f"view_{i:02d}.png"))
            del camera, rendered, input_rendered

    del gs_params, out_gs
    torch.cuda.empty_cache()
    print(f"  Saved sample output to: {ply_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="Train PTv3 Gaussian Refiner")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Path to the YAML config file")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    # --- Accelerator ---
    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        gradient_accumulation_steps=1,
    )
    set_seed(cfg.training.seed)

    if accelerator.is_main_process:
        os.makedirs(cfg.data.output_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.data.output_dir, "checkpoints"), exist_ok=True)
    accelerator.wait_for_everyone()

    accelerator.print(f"Processes: {accelerator.num_processes}  |  "
                      f"Device: {accelerator.device}  |  "
                      f"Mixed precision: {cfg.training.mixed_precision}")
    if cfg.loss.focal.enabled:
        fc = cfg.loss.focal
        accelerator.print(f"Focal loss: scale={fc.alpha_scale}, gamma={fc.gamma}, "
                          f"alpha∈[{fc.alpha_min}, {fc.alpha_max}]")

    # --- Pipeline params for the 3DGS renderer ---
    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)

    # --- Data ---
    accelerator.print("Loading dataset …")
    train_dataset = GaussianDataset(
        posmap_root=cfg.data.posmap_root,
        attribute_map_path=cfg.data.attribute_map_path,
        renders_root=cfg.data.renders_root,
        num_views=cfg.training.num_views,
        white_background=cfg.rendering.white_background,
        random_background=cfg.rendering.random_background,
        target_resolution=cfg.rendering.target_resolution,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        collate_fn=gaussian_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        collate_fn=gaussian_collate_fn,
        pin_memory=True,
    )

    # --- Model ---
    accelerator.print("Creating model …")
    model = PTv3GaussianRefiner()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Trainable parameters: {num_params:,}")

    accelerator.print("Loading LPIPS model …")
    lpips_fn = lpips.LPIPS(net="vgg").to(accelerator.device)
    lpips_fn.eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    # --- Optimiser / scheduler ---
    optimizer = AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.lr * 0.01)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler,
    )

    # --- Resume ---
    start_epoch = 0
    if cfg.training.resume is not None:
        accelerator.print(f"Resuming from: {cfg.training.resume}")
        ckpt = torch.load(cfg.training.resume, map_location=accelerator.device)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        accelerator.print(f"Resumed at epoch {start_epoch}")

    # --- Training ---
    sep = "=" * 80
    accelerator.print(sep)
    accelerator.print("Starting training …")
    accelerator.print(f"  Posmap root      : {cfg.data.posmap_root}")
    accelerator.print(f"  Attribute map    : {cfg.data.attribute_map_path}")
    accelerator.print(f"  Renders root     : {cfg.data.renders_root}")
    accelerator.print(f"  Target resolution: {cfg.rendering.target_resolution}")
    accelerator.print(f"  Random background: {cfg.rendering.random_background}")
    accelerator.print(f"  Focal loss       : {cfg.loss.focal.enabled}")
    accelerator.print(f"  Eff. batch size  : {cfg.training.batch_size * accelerator.num_processes}")
    accelerator.print(sep)

    best_psnr = 0.0

    for epoch in range(start_epoch, cfg.training.epochs):
        train_loss, train_psnr = train_one_epoch(
            model, train_loader, optimizer, accelerator, pipe, epoch, cfg, lpips_fn,
        )
        scheduler.step()
        torch.cuda.empty_cache()

        if train_psnr > best_psnr:
            best_psnr = train_psnr

        if accelerator.is_main_process:
            ckpt = {
                "epoch": epoch,
                "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "train_psnr": train_psnr,
            }
            path = os.path.join(cfg.data.output_dir, "checkpoints",
                                f"checkpoint_epoch_{epoch + 1:03d}.pth")
            torch.save(ckpt, path)
            print(f"  Saved checkpoint: {path}")

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            save_sample_output(model, val_loader, accelerator, pipe,
                               cfg.data.output_dir, epoch, cfg)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print("-" * 80)

    accelerator.print(sep)
    accelerator.print(f"Training complete!  Best PSNR: {best_psnr:.2f}")
    accelerator.print(sep)


if __name__ == "__main__":
    main()