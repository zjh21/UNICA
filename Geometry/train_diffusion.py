"""
UNICA — Diffusion model training script (two-stage).

Stage 1: Train the denoising UNet and action embeddings jointly.
Stage 2: Freeze action embeddings; continue training the UNet with
         oversampled turning frames.

Usage:
    python train_diffusion.py --config configs/diffusion.yaml
    accelerate launch train_diffusion.py --config configs/diffusion.yaml
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import logging
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torchvision.utils import save_image
from omegaconf import OmegaConf

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.unet_3d import UNet3DConditionModel
from models.action_embeddings import ActionEmbeddings
from datasets.action_dataset import GeoActionDataset


# ===================================================================== trainer ===
class DiffusionTrainer:
    """Two-stage trainer for the UNICA diffusion model."""

    # ------------------------------------------------------------------ init ----
    def __init__(self, cfg):
        self.cfg = cfg

        # Accelerator
        project_config = ProjectConfiguration(
            project_dir=cfg.output_dir,
            logging_dir=os.path.join(cfg.output_dir, "logs"),
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
            mixed_precision=cfg.get("mixed_precision", "no"),
            log_with="tensorboard",
            project_config=project_config,
        )

        # Logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger = get_logger(__name__, log_level="INFO")
        if self.accelerator.is_local_main_process:
            logging.getLogger("diffusers").setLevel(logging.INFO)
        else:
            logging.getLogger("diffusers").setLevel(logging.ERROR)

        # Seed
        if cfg.get("seed") is not None:
            set_seed(cfg.seed)
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)

        # Weight dtype
        self.weight_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(self.accelerator.mixed_precision, torch.float32)

        # Models
        self._init_models()

        # Output directories
        self.output_dir = Path(cfg.output_dir)
        if self.accelerator.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "checkpoints").mkdir(exist_ok=True)
            (self.output_dir / "samples").mkdir(exist_ok=True)
            self.accelerator.init_trackers("unica_diffusion", config=dict(cfg))

        self._models_prepared = False

    # ---------------------------------------------------------- model loading ----
    def _init_models(self):
        cfg = self.cfg
        self.logger.info("Initialising models...")

        # ---- VAE (frozen) ----
        # Skip pretrained weights since finetuned weights are always loaded
        vae_config_path = Path(cfg.vae_model_path) / "config.json"
        with open(vae_config_path) as f:
            vae_config = json.load(f)
        self.vae = AutoencoderKL.from_config(vae_config)
        ckpt = torch.load(cfg.vae_finetune_path, map_location="cpu", weights_only=False)
        self.vae.load_state_dict(ckpt["vae_state_dict"])
        self.logger.info("Loaded VAE from config + finetuned weights (skipped pretrained)")
        self.vae.requires_grad_(False)
        self.vae.eval()

        # ---- Denoising UNet ----
        # Skip pretrained weights when resuming (checkpoint will overwrite them)
        will_resume = cfg.get('resume_from_checkpoint') is not None
        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path=cfg['pretrained_model_path'],
            subfolder="unet",
            unet_additional_kwargs={
                "use_inflated_groupnorm": True,
                "unet_use_cross_frame_attention": False,
                "unet_use_temporal_attention": True,
                "use_motion_module": False,
            },
            load_pretrained_weights=not will_resume,
        )
        self.denoising_unet.requires_grad_(True)

        if cfg.get("enable_xformers_memory_efficient_attention", True):
            try:
                self.denoising_unet.enable_xformers_memory_efficient_attention()
                self.logger.info("xformers memory-efficient attention enabled")
            except Exception:
                self.logger.info("xformers not available — using default attention")

        if cfg.get("gradient_checkpointing", True):
            self.denoising_unet.enable_gradient_checkpointing()
            self.logger.info("Gradient checkpointing enabled")

        # ---- action Embeddings ----
        self.action_embeddings = ActionEmbeddings(
            num_actions=cfg.get("num_actions", 5),
            embedding_dim=768,
        )

        # ---- Noise Scheduler ----
        self.noise_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon",
        )

    # ------------------------------------------------------------- helpers ----
    def _unwrap(self, model):
        return self.accelerator.unwrap_model(model)

    def _prepare_training(self, optimizer, dataloader, lr_scheduler):
        """Prepare models (once) and stage-specific training components."""
        if not self._models_prepared:
            (
                self.denoising_unet,
                self.action_embeddings,
                self.vae,
                optimizer,
                dataloader,
                lr_scheduler,
            ) = self.accelerator.prepare(
                self.denoising_unet,
                self.action_embeddings,
                self.vae,
                optimizer,
                dataloader,
                lr_scheduler,
            )
            self._models_prepared = True
        else:
            optimizer, dataloader, lr_scheduler = self.accelerator.prepare(
                optimizer, dataloader, lr_scheduler,
            )
        return optimizer, dataloader, lr_scheduler

    def _create_optimizer(self, trainable_params, num_training_steps, stage_name=""):
        cfg = self.cfg

        lr = cfg.learning_rate
        if cfg.get("scale_lr", False):
            lr *= (
                cfg.get("gradient_accumulation_steps", 1)
                * cfg.batch_size
                * self.accelerator.num_processes
            )

        optimizer_cls = torch.optim.AdamW
        if cfg.get("use_8bit_adam", False):
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit
                self.logger.info("Using 8-bit AdamW")
            except ImportError:
                self.logger.warning("bitsandbytes not found — falling back to AdamW")

        n_params = sum(p.numel() for p in trainable_params)
        self.logger.info(f"[{stage_name}] Trainable parameters: {n_params:,}")

        optimizer = optimizer_cls(
            trainable_params,
            lr=lr,
            betas=(cfg.get("adam_beta1", 0.9), cfg.get("adam_beta2", 0.999)),
            weight_decay=cfg.get("adam_weight_decay", 1e-2),
            eps=cfg.get("adam_epsilon", 1e-8),
        )

        lr_scheduler = get_scheduler(
            cfg.get("lr_scheduler", "constant_with_warmup"),
            optimizer=optimizer,
            num_warmup_steps=cfg.get("lr_warmup_steps", 500)
            * cfg.get("gradient_accumulation_steps", 1),
            num_training_steps=num_training_steps,
        )
        return optimizer, lr_scheduler

    def _compute_snr(self, timesteps):
        """Signal-to-noise ratio for Min-SNR loss weighting."""
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        alpha = (alphas_cumprod ** 0.5).to(timesteps.device)[timesteps].float()
        sigma = ((1.0 - alphas_cumprod) ** 0.5).to(timesteps.device)[timesteps].float()

        while len(alpha.shape) < len(timesteps.shape):
            alpha = alpha[..., None]
            sigma = sigma[..., None]

        return (alpha.expand(timesteps.shape) / sigma.expand(timesteps.shape)) ** 2

    # --------------------------------------------------------- train step ----
    def _train_step(self, batch):
        """Single forward + loss computation (diffusion forcing)."""
        cfg = self.cfg
        images = batch["images"].to(dtype=self.weight_dtype)
        action_indices = batch["action_type"]
        B, F, C, H, W = images.shape
        vae = self._unwrap(self.vae)

        # Encode every frame into latent space
        with torch.no_grad():
            latents = []
            for f in range(F):
                lat = vae.encode(images[:, f]).latent_dist.sample()
                lat = lat * vae.config.scaling_factor
                latents.append(lat.unsqueeze(2))
            all_latents = torch.cat(latents, dim=2)  # (B, C_lat, F, h, w)

        # --- Diffusion forcing: independent noise levels per frame ---
        # Target frame (4th)
        target_t = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (B,), device=images.device,
        ).long()

        # Conditioning frames (1st–3rd)
        cond_t = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (B, 3), device=images.device,
        ).long()

        # With some probability keep conditioning frames clean
        clean_mask = torch.rand(B, device=images.device) < cfg.get("clean_cond_prob", 0.3)
        cond_t[clean_mask] = 0

        cond_latents = []
        for f in range(3):
            frame_lat = all_latents[:, :, f : f + 1, :, :]
            ft = cond_t[:, f]
            if (ft > 0).any():
                noisy = self.noise_scheduler.add_noise(
                    frame_lat, torch.randn_like(frame_lat), ft,
                )
            else:
                noisy = frame_lat
            cond_latents.append(noisy)
        cond_latents = torch.cat(cond_latents, dim=2)

        # Noise for the target frame
        target_lat = all_latents[:, :, 3:4, :, :]
        noise = torch.randn_like(target_lat)
        if cfg.get("noise_offset", 0) > 0:
            noise += cfg.noise_offset * torch.randn(
                (B, 1, 1, 1, 1), device=noise.device, dtype=noise.dtype,
            )
        noisy_target = self.noise_scheduler.add_noise(target_lat, noise, target_t)

        noisy_latents = torch.cat([cond_latents, noisy_target], dim=2).to(
            dtype=self.weight_dtype,
        )

        # Classifier-free guidance dropout
        uncond_mask = torch.rand(B, device=images.device) < cfg.get("uncond_ratio", 0.1)
        action_emb = self.action_embeddings(action_indices)       # (B, 1, D)
        uncond_emb = torch.zeros_like(action_emb)
        final_emb = torch.where(uncond_mask.view(B, 1, 1), uncond_emb, action_emb)

        # UNet forward
        pred = self.denoising_unet(
            noisy_latents, target_t, encoder_hidden_states=final_emb,
        ).sample

        # Loss only on the 4th-frame prediction
        pred_4th = pred[:, :, 3:4, :, :].to(torch.float32)
        noise = noise.to(torch.float32)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                target_lat.to(torch.float32), noise, target_t,
            )
        else:
            raise ValueError(
                f"Unknown prediction type: {self.noise_scheduler.config.prediction_type}"
            )

        snr_gamma = cfg.get("snr_gamma", 0)
        if snr_gamma == 0:
            loss = Fn.mse_loss(pred_4th, target, reduction="mean")
        else:
            snr = self._compute_snr(target_t)
            if self.noise_scheduler.config.prediction_type == "v_prediction":
                snr = snr + 1
            weights = (
                torch.stack(
                    [snr, snr_gamma * torch.ones_like(target_t)], dim=1,
                ).min(dim=1)[0]
                / snr
            )
            loss = Fn.mse_loss(pred_4th, target, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * weights
            loss = loss.mean()

        return loss

    # ---------------------------------------------------- visualisation ----
    @torch.no_grad()
    def _visualize(self, val_dataloader, global_step, stage_name="", cfg_scale=7.5):
        """Generate and save sample visualisations (main process only)."""
        self.vae.eval()
        self.denoising_unet.eval()
        self.action_embeddings.eval()
        vae = self._unwrap(self.vae)

        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= 2:
                break

            images = batch["images"].to(
                device=self.accelerator.device, dtype=self.weight_dtype,
            )
            action_idx = batch["action_type"].to(self.accelerator.device)
            mask = batch["mask"][0].to(self.accelerator.device)
            B, F, C, H, W = images.shape

            # Encode first 3 frames
            inp_lats = []
            for f in range(3):
                lat = vae.encode(images[:, f]).latent_dist.sample()
                lat = lat * vae.config.scaling_factor
                inp_lats.append(lat.unsqueeze(2))
            inp_lats = torch.cat(inp_lats, dim=2)

            cond_emb = self.action_embeddings(action_idx)
            uncond_emb = torch.zeros_like(cond_emb)

            # Start from pure noise
            lat_shape = (B, 4, 1, inp_lats.shape[-2], inp_lats.shape[-1])
            lat_4th = torch.randn(lat_shape, device=images.device, dtype=self.weight_dtype)

            self.noise_scheduler.set_timesteps(50, device=images.device)
            for t in self.noise_scheduler.timesteps:
                combined = torch.cat([inp_lats, lat_4th], dim=2)
                t_batch = t.unsqueeze(0).repeat(B).to(images.device)

                if cfg_scale > 1.0:
                    up = self.denoising_unet(
                        combined, t_batch, encoder_hidden_states=uncond_emb,
                    ).sample[:, :, 3:4]
                    cp = self.denoising_unet(
                        combined, t_batch, encoder_hidden_states=cond_emb,
                    ).sample[:, :, 3:4]
                    noise_pred = up + cfg_scale * (cp - up)
                else:
                    noise_pred = self.denoising_unet(
                        combined, t_batch, encoder_hidden_states=cond_emb,
                    ).sample[:, :, 3:4]

                lat_4th = self.noise_scheduler.step(noise_pred, t, lat_4th).prev_sample

            # Decode
            mask_f = mask.unsqueeze(0).float()
            bg = (-1.0) * (~mask).unsqueeze(0).float()

            decoded_inp = []
            for f in range(3):
                dec = vae.decode(inp_lats[:, :, f] / vae.config.scaling_factor).sample
                decoded_inp.append(dec * mask_f + bg)

            gen_4th = vae.decode(
                lat_4th.squeeze(2).to(self.weight_dtype) / vae.config.scaling_factor,
            ).sample
            gen_4th = gen_4th * mask_f + bg
            gt_4th = images[:, 3]

            # Save images
            samples_dir = self.output_dir / "samples"
            prefix = f"{stage_name}step_{global_step:06d}_batch_{batch_idx}"

            for f in range(3):
                save_image(
                    decoded_inp[f][0].cpu().float(),
                    samples_dir / f"{prefix}_input_{f}.png",
                    normalize=True,
                    value_range=(-1, 1),
                )
            save_image(
                gen_4th[0].cpu().float(),
                samples_dir / f"{prefix}_generated_3.png",
                normalize=True,
                value_range=(-1, 1),
            )
            save_image(
                gt_4th[0].cpu().float(),
                samples_dir / f"{prefix}_gt_3.png",
                normalize=True,
                value_range=(-1, 1),
            )

            comparison = torch.stack(
                [d[0].cpu().float() for d in decoded_inp]
                + [gt_4th[0].cpu().float(), gen_4th[0].cpu().float()],
                dim=0,
            )
            save_image(
                comparison,
                samples_dir / f"{prefix}_comparison.png",
                nrow=5,
                normalize=True,
                value_range=(-1, 1),
            )

        self.denoising_unet.train()
        self.action_embeddings.train()

    # ---------------------------------------------------- checkpointing ----
    def _save_checkpoint(self, global_step, stage_name=""):
        if self.accelerator.is_main_process:
            save_path = self.output_dir / "checkpoints" / f"checkpoint-{stage_name}{global_step}"
            self.accelerator.save_state(str(save_path))
            self.logger.info(f"Saved checkpoint → {save_path}")
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        max_keep = self.cfg.get("num_keep_checkpoints", 5)
        ckpt_dir = self.output_dir / "checkpoints"
        entries = []
        for d in ckpt_dir.iterdir():
            if d.is_dir() and d.name.startswith("checkpoint-"):
                name = d.name.replace("checkpoint-", "")
                for prefix in ("stage1_", "stage2_"):
                    name = name.replace(prefix, "")
                try:
                    entries.append((int(name), d))
                except ValueError:
                    continue
        entries.sort(key=lambda x: x[0])
        for _, path in entries[: max(0, len(entries) - max_keep)]:
            try:
                shutil.rmtree(path)
                self.logger.info(f"Removed old checkpoint: {path.name}")
            except Exception as e:
                self.logger.warning(f"Could not remove {path.name}: {e}")

    def _save_final_model(self, global_step, stage_name=""):
        if self.accelerator.is_main_process:
            checkpoint = {
                "global_step": global_step,
                "denoising_unet_state_dict": self._unwrap(self.denoising_unet).state_dict(),
                "action_embeddings_state_dict": self._unwrap(self.action_embeddings).state_dict(),
            }
            path = self.output_dir / "checkpoints" / f"{stage_name}final_model.pt"
            torch.save(checkpoint, path)
            self.logger.info(f"Saved final model → {path}")

    def _load_checkpoint(self):
        """Attempt to load a checkpoint.  Returns *(stage, step)*."""
        resume = self.cfg.get("resume_from_checkpoint")
        if not resume:
            return None, 0

        ckpt_dir = self.output_dir / "checkpoints"

        if resume == "latest":
            if not ckpt_dir.exists():
                self.logger.warning("No checkpoint directory found")
                return None, 0

            entries = []
            for d in ckpt_dir.iterdir():
                if d.is_dir() and d.name.startswith("checkpoint-"):
                    name = d.name.replace("checkpoint-", "")
                    stage = None
                    for s in ("stage1_", "stage2_"):
                        if name.startswith(s):
                            stage = s.rstrip("_")
                            name = name.replace(s, "")
                            break
                    try:
                        entries.append((int(name), d, stage))
                    except ValueError:
                        continue

            if not entries:
                self.logger.warning("No valid checkpoints found")
                return None, 0

            entries.sort(key=lambda x: x[0])
            step, path, stage = entries[-1]
        else:
            path = Path(resume)
            step, stage = 0, None
            name = path.name.replace("checkpoint-", "")
            for s in ("stage1_", "stage2_"):
                if name.startswith(s):
                    stage = s.rstrip("_")
                    name = name.replace(s, "")
                    break
            try:
                step = int(name)
            except ValueError:
                pass

        if path.exists():
            self.logger.info(f"Resuming from {path}  (stage={stage}, step={step})")
            self.accelerator.load_state(str(path))
            return stage, step

        self.logger.warning(f"Checkpoint not found: {path}")
        return None, 0

    # -------------------------------------------------- generic loop ----
    def _train_loop(
        self,
        dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        trainable_params,
        max_steps,
        stage_name,
        start_step=0,
        train_embeddings=True,
    ):
        """Run one training stage."""
        cfg = self.cfg
        grad_accum = cfg.get("gradient_accumulation_steps", 1)

        steps_per_epoch = max(len(dataloader) // grad_accum, 1)
        num_epochs = (max_steps // steps_per_epoch) + 1

        self.logger.info(f"***** {stage_name} *****")
        self.logger.info(f"  Batches / epoch  = {len(dataloader)}")
        self.logger.info(f"  Num epochs       = {num_epochs}")
        self.logger.info(f"  Total steps      = {max_steps}")

        global_step = start_step
        progress_bar = tqdm(
            range(start_step, max_steps),
            desc=f"{stage_name}",
            disable=not self.accelerator.is_local_main_process,
            initial=start_step,
            total=max_steps,
        )

        for epoch in range(num_epochs):
            self.denoising_unet.train()
            if train_embeddings:
                self.action_embeddings.train()
            else:
                self.action_embeddings.eval()

            train_loss_accum = 0.0

            for batch in dataloader:
                if global_step >= max_steps:
                    break

                with self.accelerator.accumulate(self.denoising_unet):
                    loss = self._train_step(batch)

                    avg_loss = self.accelerator.gather(
                        loss.detach().reshape(1),
                    ).mean()
                    train_loss_accum += avg_loss.item() / grad_accum

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        max_norm = cfg.get("max_grad_norm", 1.0)
                        if max_norm > 0:
                            self.accelerator.clip_grad_norm_(
                                trainable_params, max_norm,
                            )

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    self.accelerator.log(
                        {f"{stage_name}_loss": train_loss_accum}, step=global_step,
                    )
                    train_loss_accum = 0.0

                    # Checkpoint
                    ckpt_every = cfg.get("checkpointing_steps", 10000)
                    if global_step % ckpt_every == 0:
                        self._save_checkpoint(global_step, f"{stage_name}_")

                    # Visualisation
                    vis_every = cfg.get("validation_steps", 10000)
                    if global_step % vis_every == 0 or global_step == 1:
                        if self.accelerator.is_main_process:
                            self.logger.info(
                                f"Generating samples at step {global_step}",
                            )
                            self._visualize(
                                val_dataloader, global_step, f"{stage_name}_",
                            )

                    progress_bar.set_postfix(
                        loss=f"{loss.detach().item():.4f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    )

            if global_step >= max_steps:
                break

        progress_bar.close()
        return global_step

    # ---------------------------------------------------------- public API ----
    def train(self):
        cfg = self.cfg
        stage1_steps = cfg.get("stage1_steps", 100000)
        stage2_steps = cfg.get("stage2_steps", 600000)

        resume_stage, resume_step = self._load_checkpoint()

        run_stage1 = resume_stage != "stage2"
        if resume_stage == "stage2":
            self.logger.info("Resuming from Stage 2 — skipping Stage 1")
        elif resume_stage == "stage1":
            self.logger.info(f"Resuming Stage 1 from step {resume_step}")

        # ====================== STAGE 1 ======================
        if run_stage1:
            self.logger.info("=" * 60)
            self.logger.info("STAGE 1  —  Training UNet + action embeddings")
            self.logger.info("=" * 60)

            dataset = GeoActionDataset(
                root_folder=cfg.data_folder,
                image_size=cfg.get("image_size", 128),
                stage="stage1",
            )
            dataloader = DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=True,
                num_workers=cfg.get("num_workers", 4),
                pin_memory=True,
                drop_last=True,
            )
            val_dataloader = DataLoader(
                dataset,
                batch_size=min(4, cfg.batch_size),
                shuffle=False,
                num_workers=2,
            )

            trainable_params = list(self.denoising_unet.parameters()) + list(
                self.action_embeddings.parameters()
            )
            num_opt_steps = stage1_steps * cfg.get("gradient_accumulation_steps", 1)
            optimizer, lr_scheduler = self._create_optimizer(
                trainable_params, num_opt_steps, "Stage 1",
            )

            optimizer, dataloader, lr_scheduler = self._prepare_training(
                optimizer, dataloader, lr_scheduler,
            )

            start = resume_step if resume_stage == "stage1" else 0
            self._train_loop(
                dataloader,
                val_dataloader,
                optimizer,
                lr_scheduler,
                trainable_params,
                stage1_steps,
                "stage1",
                start_step=start,
                train_embeddings=True,
            )

            self.accelerator.wait_for_everyone()
            self._save_final_model(stage1_steps, "stage1_")
            self.logger.info("Stage 1 completed.")

        # ====================== STAGE 2 ======================
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2  —  Freeze embeddings, oversample in-between")
        self.logger.info("=" * 60)

        self.action_embeddings.requires_grad_(False)
        self.logger.info("action embeddings frozen")

        dataset = GeoActionDataset(
            root_folder=cfg.data_folder,
            image_size=cfg.get("image_size", 128),
            stage="stage2",
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            dataset,
            batch_size=min(4, cfg.batch_size),
            shuffle=False,
            num_workers=2,
        )

        trainable_params = [
            p for p in self.denoising_unet.parameters() if p.requires_grad
        ]
        num_opt_steps = stage2_steps * cfg.get("gradient_accumulation_steps", 1)
        optimizer, lr_scheduler = self._create_optimizer(
            trainable_params, num_opt_steps, "Stage 2",
        )

        optimizer, dataloader, lr_scheduler = self._prepare_training(
            optimizer, dataloader, lr_scheduler,
        )

        start = resume_step if resume_stage == "stage2" else 0
        final_step = self._train_loop(
            dataloader,
            val_dataloader,
            optimizer,
            lr_scheduler,
            trainable_params,
            stage2_steps,
            "stage2",
            start_step=start,
            train_embeddings=False,
        )

        self.accelerator.wait_for_everyone()
        self._save_final_model(stage1_steps + final_step, "stage2_")
        self.accelerator.end_training()
        self.logger.info("Training completed!")


# ------------------------------------------------------------------ entry ----
def main():
    parser = argparse.ArgumentParser(description="UNICA — Diffusion training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/diffusion.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    trainer = DiffusionTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()