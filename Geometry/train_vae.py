"""
UNICA — VAE fine-tuning script (step-based).

Usage:
    python train_vae.py --config configs/vae.yaml
    accelerate launch train_vae.py --config configs/vae.yaml
"""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import json
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from diffusers import AutoencoderKL
from accelerate import Accelerator
from omegaconf import OmegaConf

from datasets.vae_dataset import GeoVAEDataset


class VAETrainer:
    """Trainer for fine-tuning a pre-trained VAE on geometry (position-map) data."""

    def __init__(self, cfg):
        self.cfg = cfg

        # ---- Accelerator --------------------------------------------------------
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
            mixed_precision=cfg.get("mixed_precision", "no"),
        )
        self.device = self.accelerator.device

        # ---- Model --------------------------------------------------------------
        self._print("Loading VAE model...")
        vae = AutoencoderKL.from_pretrained(cfg.pretrained_vae_path)

        # ---- Dataset / DataLoader -----------------------------------------------
        self._print("Creating dataset...")
        train_dataset = GeoVAEDataset(
            root_dir=cfg.data_dir,
            image_size=cfg.image_size,
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

        # Foreground mask (from the dataset, if available)
        self.mask = train_dataset.mask
        if self.mask is not None:
            self.mask = self.mask.to(self.device)

        # ---- Optimiser / Scheduler ----------------------------------------------
        self.optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.num_steps,
        )

        # ---- Prepare with Accelerate -------------------------------------------
        (
            self.vae,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            vae, self.optimizer, self.train_loader, self.scheduler,
        )

        # ---- Loss hyper-parameters ----------------------------------------------
        self.l1_weight = cfg.get("l1_weight", 1.0)
        self.kl_weight = cfg.get("kl_weight", 1e-6)

        # ---- Training state -----------------------------------------------------
        self.global_step = 0
        self.train_history = {"total_loss": [], "l1_loss": [], "kl_loss": []}

        # ---- Directories --------------------------------------------------------
        self.checkpoint_dir = Path(cfg.checkpoint_dir)
        self.log_dir = Path(cfg.log_dir)
        if self.accelerator.is_main_process:
            self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
            self.log_dir.mkdir(exist_ok=True, parents=True)
        self.accelerator.wait_for_everyone()

        # ---- Resume -------------------------------------------------------------
        if cfg.get("resume_from"):
            self._load_checkpoint(cfg.resume_from)

    # ---------------------------------------------------------------------- utils
    def _print(self, msg):
        """Print only on the main process."""
        if self.accelerator.is_main_process:
            print(msg)

    def _unwrap_vae(self):
        """Return the unwrapped (non-DDP) VAE model."""
        return self.accelerator.unwrap_model(self.vae)

    # ------------------------------------------------------------------- training
    def _compute_losses(self, images, reconstructed, posterior):
        """Compute L1 and KL losses (masked if a foreground mask is available)."""
        if self.mask is not None:
            mask = self.mask.unsqueeze(0).expand(images.shape[0], -1, -1, -1)
            l1 = F.l1_loss(reconstructed[mask], images[mask])
        else:
            l1 = F.l1_loss(reconstructed, images)

        kl = posterior.kl().mean()
        total = self.l1_weight * l1 + self.kl_weight * kl
        return total, {"l1": l1, "kl": kl}

    def _train_step(self, batch):
        """Execute a single optimisation step."""
        images = batch.to(self.device)
        vae = self._unwrap_vae()

        with self.accelerator.accumulate(self.vae):
            posterior = vae.encode(images).latent_dist
            z = posterior.sample()
            reconstructed = vae.decode(z).sample

            total_loss, losses = self._compute_losses(images, reconstructed, posterior)

            self.accelerator.backward(total_loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss.item(), {k: v.item() for k, v in losses.items()}

    # --------------------------------------------------------------- visualisation
    def _visualize(self, batch, num_samples=4):
        """Save a reconstruction visualisation (main process only)."""
        self.vae.eval()

        if self.accelerator.is_main_process:
            vae = self._unwrap_vae()
            images = batch[: min(num_samples, batch.shape[0])].to(self.device)

            with torch.no_grad():
                z = vae.encode(images).latent_dist.mode()
                z_scaled = z * vae.config.scaling_factor
                reconstructed = vae.decode(z_scaled / vae.config.scaling_factor).sample

            images = torch.clamp((images + 1) / 2, 0, 1)
            reconstructed = torch.clamp((reconstructed + 1) / 2, 0, 1)

            orig = images[0].cpu().permute(1, 2, 0).numpy() * 255.0
            recon = reconstructed[0]
            if self.mask is not None:
                recon = recon * self.mask.float()
            recon = recon.cpu().permute(1, 2, 0).numpy() * 255.0

            cv2.imwrite(
                str(self.log_dir / f"orig_step_{self.global_step}.png"), orig,
            )
            cv2.imwrite(
                str(self.log_dir / f"reconstruction_step_{self.global_step}.png"), recon,
            )
            print(f"Saved visualisation to {self.log_dir}")

        self.accelerator.wait_for_everyone()
        self.vae.train()

    # --------------------------------------------------------------- checkpointing
    def _save_checkpoint(self, is_best=False):
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            unwrapped = self._unwrap_vae()
            checkpoint = {
                "global_step": self.global_step,
                "vae_state_dict": unwrapped.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "train_history": self.train_history,
            }
            path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
            torch.save(checkpoint, path)
            torch.save(checkpoint, self.checkpoint_dir / "latest_checkpoint.pt")
            if is_best:
                torch.save(checkpoint, self.checkpoint_dir / "best_checkpoint.pt")
            print(f"Saved checkpoint to {path}")

        self.accelerator.wait_for_everyone()

    def _load_checkpoint(self, checkpoint_path):
        self._print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self._unwrap_vae().load_state_dict(checkpoint["vae_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.train_history = checkpoint.get("train_history", self.train_history)
        self._print(f"Resumed from step {self.global_step}")

    # ------------------------------------------------------------------ main loop
    def train(self):
        cfg = self.cfg
        num_steps = cfg.num_steps
        vis_every = cfg.get("visualize_every", 10000)
        ckpt_every = cfg.get("checkpoint_every", 10000)
        log_every = cfg.get("log_every", 100)

        self._print(
            f"Training on {self.accelerator.num_processes} GPU(s)  |  "
            f"steps={num_steps}  batch/device={cfg.batch_size}"
        )

        self.vae.train()
        running_losses = {"total_loss": [], "l1_loss": [], "kl_loss": []}

        pbar = (
            tqdm(total=num_steps, initial=self.global_step, desc="Training", ncols=100)
            if self.accelerator.is_main_process
            else None
        )

        while self.global_step < num_steps:
            for batch in self.train_loader:
                if self.global_step >= num_steps:
                    break

                total_loss, losses = self._train_step(batch)

                running_losses["total_loss"].append(total_loss)
                running_losses["l1_loss"].append(losses["l1"])
                running_losses["kl_loss"].append(losses["kl"])

                if self.accelerator.sync_gradients:
                    self.global_step += 1
                    self.scheduler.step()

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix(
                            loss=f"{total_loss:.4f}",
                            l1=f'{losses["l1"]:.4f}',
                            kl=f'{losses["kl"]:.6f}',
                        )

                    # Periodic logging
                    if self.global_step % log_every == 0:
                        if self.accelerator.is_main_process:
                            for k, v in running_losses.items():
                                if v:
                                    self.train_history[k].append(np.mean(v))
                            running_losses = {"total_loss": [], "l1_loss": [], "kl_loss": []}
                            with open(self.log_dir / "training_history.json", "w") as f:
                                json.dump(self.train_history, f, indent=2)

                    # Visualisation
                    if self.global_step > 0 and self.global_step % vis_every == 0:
                        self._visualize(batch, num_samples=cfg.batch_size)
                        self.vae.train()

                    # Checkpoint
                    if self.global_step > 0 and self.global_step % ckpt_every == 0:
                        self._save_checkpoint()

                    if self.global_step >= num_steps:
                        break

        if pbar is not None:
            pbar.close()

        self.accelerator.wait_for_everyone()
        self._print("Training completed!")
        self._save_checkpoint()

    # ---------------------------------------------------------- plotting (optional)
    def plot_training_curves(self):
        if not self.accelerator.is_main_process or not self.train_history["total_loss"]:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for idx, (key, values) in enumerate(self.train_history.items()):
            if idx >= 3:
                break
            axes[idx].plot(values)
            axes[idx].set_title(key.replace("_", " ").title())
            axes[idx].set_xlabel(f"Step (x{self.cfg.get('log_every', 100)})")
            axes[idx].set_ylabel("Loss")
            axes[idx].grid(True)

        plt.suptitle(f"Training Curves — Step {self.global_step}")
        plt.tight_layout()
        plt.savefig(self.log_dir / "training_curves.png", dpi=100, bbox_inches="tight")
        plt.close()


# -------------------------------------------------------------------------- entry
def main():
    parser = argparse.ArgumentParser(description="UNICA — VAE fine-tuning")
    parser.add_argument("--config", type=str, default="configs/vae.yaml",
                        help="Path to the YAML configuration file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    trainer = VAETrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()