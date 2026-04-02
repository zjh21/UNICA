"""Auto-regressive inference script for geometry position map generation."""

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import json
import logging
import random
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import trimesh
import yaml
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets.action_dataset import GeoActionDataset
from models.action_embeddings import ActionEmbeddings
from models.unet_3d import UNet3DConditionModel
from utils.renormalization import apply_keypoint_renormalization


# ==================== Config Utilities ====================

def load_config(config_path):
    """Load YAML configuration file via OmegaConf."""
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-regressive inference")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    return parser.parse_args()


def expand_motion_sequence(motion_seq_config):
    """Expand compact motion sequence config into a flat list.

    Supports two formats:
        Compact: [{"action": "Forward", "count": 60}, ...]
        Flat:    ["Forward", "Forward", ...]
    """
    if not motion_seq_config:
        return ['Forward'] * 100

    # Use Mapping (covers dict, DictConfig, OrderedDict, etc.)
    if isinstance(motion_seq_config[0], Mapping):
        sequence = []
        for item in motion_seq_config:
            sequence.extend([item['action']] * item['count'])
        return sequence

    return list(motion_seq_config)


# ==================== PCA Reconstruction ====================

def transform_pca(pca, pose_conds):
    """Transform foreground pixels through PCA and reconstruct.

    Args:
        pca: Fitted PCA model.
        pose_conds: Foreground pixels array of shape (M, 3).

    Returns:
        Reconstructed foreground pixels of shape (M, 3).
    """
    pose_conds = pose_conds.reshape(1, -1)
    lowdim = pca.transform(pose_conds)
    reconstructed = pca.inverse_transform(lowdim)
    return reconstructed.reshape(-1, 3)


def apply_pca_reconstruction_to_tensor(tensor, pca, pca_mask):
    """Apply PCA-based reconstruction to a position map tensor.

    Args:
        tensor: Position map tensor of shape (3, H, W) in range [-1, 1].
        pca: Fitted PCA model.
        pca_mask: Boolean mask of shape (H, W) indicating foreground pixels.

    Returns:
        Reconstructed position map tensor of shape (3, H, W) in range [-1, 1].
    """
    tensor_01 = torch.clamp((tensor + 1) / 2.0, 0, 1)
    pos_map = tensor_01.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)

    pose_conds = pos_map[pca_mask]
    new_pose_conds = transform_pca(pca, pose_conds)
    pos_map[pca_mask] = new_pose_conds.astype(pos_map.dtype)

    reconstructed = torch.from_numpy(pos_map.transpose(2, 0, 1)).to(tensor.device, dtype=tensor.dtype)
    return reconstructed * 2.0 - 1.0


# ==================== DDIM Sampling ====================

def denoise_with_ddim(
    noisy_latent_4th, input_latents, motion_embeds,
    denoising_unet, noise_scheduler, start_timestep, num_steps, device,
    cfg_scale=1.0,
):
    """Denoise the 4th frame using DDIM with specified number of steps.

    Args:
        noisy_latent_4th: Noisy latent of 4th frame (B, 4, 1, h, w).
        input_latents: Clean latents of frames 1-3 (B, 4, 3, h, w).
        motion_embeds: Motion condition embeddings (B, 1, 768).
        denoising_unet: The UNet model.
        noise_scheduler: DDIM scheduler.
        start_timestep: Starting timestep for denoising.
        num_steps: Number of DDIM denoising steps.
        device: Torch device.
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance).

    Returns:
        Denoised latent (B, 4, 1, h, w).
    """
    B = noisy_latent_4th.shape[0]
    current_latent = noisy_latent_4th.clone()

    if num_steps == 1:
        timesteps = [start_timestep]
    else:
        timesteps = np.linspace(start_timestep, 0, num_steps + 1)[:-1].astype(int).tolist()

    uncond_embeds = torch.zeros_like(motion_embeds) if cfg_scale > 1.0 else None

    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t], device=device).long()
        noisy_latents = torch.cat([input_latents, current_latent], dim=2)
        timestep_batch = t_tensor.repeat(B)

        with torch.no_grad():
            if cfg_scale > 1.0:
                uncond_pred = denoising_unet(noisy_latents, timestep_batch, encoder_hidden_states=uncond_embeds).sample
                cond_pred = denoising_unet(noisy_latents, timestep_batch, encoder_hidden_states=motion_embeds).sample
                noise_pred = uncond_pred[:, :, 3:4] + cfg_scale * (cond_pred[:, :, 3:4] - uncond_pred[:, :, 3:4])
            else:
                pred = denoising_unet(noisy_latents, timestep_batch, encoder_hidden_states=motion_embeds).sample
                noise_pred = pred[:, :, 3:4]

        prev_t = timesteps[i + 1] if i < len(timesteps) - 1 else 0

        alpha_prod_t = noise_scheduler.alphas_cumprod[t].to(device).float()
        alpha_prod_t_prev = (
            noise_scheduler.alphas_cumprod[prev_t].to(device).float()
            if prev_t > 0
            else torch.tensor(1.0, device=device, dtype=torch.float32)
        )

        current_latent_f = current_latent.float()
        noise_pred_f = noise_pred.float()

        # DDIM update (deterministic, eta=0)
        pred_x0 = (current_latent_f - (1.0 - alpha_prod_t).sqrt() * noise_pred_f) / alpha_prod_t.sqrt()
        current_latent = alpha_prod_t_prev.sqrt() * pred_x0 + (1.0 - alpha_prod_t_prev).sqrt() * noise_pred_f

    return current_latent


# ==================== Point Cloud / Mesh Utilities ====================

def tensor_to_position_map(tensor, mask=None):
    """Convert a position map tensor (3, H, W) in [-1, 1] to numpy (H, W, 3) in [0, 1]."""
    if mask is not None:
        if mask.shape[0] == 1:
            mask_expanded = mask.repeat(3, 1, 1)
        else:
            mask_expanded = mask
        tensor = tensor * mask_expanded.float()

    position_map = torch.clamp((tensor + 1) / 2.0, 0, 1)
    return position_map.cpu().numpy().transpose(1, 2, 0)


def save_position_map_as_exr(tensor, mask, output_path):
    """Save position map as EXR file."""
    if mask.shape[0] == 1:
        mask_expanded = mask.repeat(3, 1, 1)
    else:
        mask_expanded = mask

    masked = tensor * mask_expanded.float()
    position_map = torch.clamp((masked + 1) / 2.0, 0, 1)
    position_map = position_map.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    cv2.imwrite(str(output_path), position_map)


def save_position_map_with_shift_as_npy(tensor, mask, accumulated_shift, output_path):
    """Save position map with accumulated shift as NPY file.

    Output array is (H, W, 3) with foreground pixels in position space
    [-0.5, 0.5] + accumulated_shift, and background pixels set to 0.0.
    """
    if mask.dim() == 3:
        mask_np = mask[0].cpu().numpy()
    else:
        mask_np = mask.cpu().numpy()
    if mask_np.ndim == 3:
        mask_np = mask_np[0]

    position_map = torch.clamp((tensor + 1) / 2.0, 0, 1)
    position_map = position_map.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    position_map = position_map - 0.5  # Convert to position space

    if accumulated_shift is not None:
        position_map = position_map + accumulated_shift

    position_map = position_map * mask_np[:, :, np.newaxis]
    np.save(str(output_path), position_map.astype(np.float32))


def recover_points_from_position_map(position_map, translation=None, threshold_factor=0.05):
    """Recover 3D points and mesh faces from a position map.

    Args:
        position_map: numpy array (H, W, 3) in range [0, 1].
        translation: Optional translation to apply (3,) array.
        threshold_factor: Factor of bounding box diagonal for edge validity.

    Returns:
        points: (N, 3) array of 3D points.
        faces: (M, 3) array of triangle faces.
    """
    h, w = position_map.shape[:2]
    mask = np.any(position_map != 0, axis=2)

    vertex_index_map = np.full((h, w), -1, dtype=np.int32)
    position_3d_map = np.zeros((h, w, 3), dtype=np.float32)

    recovered_points = []
    vertex_count = 0

    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                point_3d = position_map[y, x].copy() - 0.5
                if translation is not None:
                    point_3d = point_3d + translation
                recovered_points.append(point_3d)
                vertex_index_map[y, x] = vertex_count
                position_3d_map[y, x] = point_3d
                vertex_count += 1

    recovered_points = np.array(recovered_points) if recovered_points else np.zeros((0, 3))

    if len(recovered_points) == 0:
        return recovered_points, np.zeros((0, 3), dtype=np.int32)

    bbox_diagonal = np.linalg.norm(recovered_points.max(axis=0) - recovered_points.min(axis=0))
    adaptive_threshold = bbox_diagonal * threshold_factor

    def is_edge_valid(pos1, pos2):
        return np.linalg.norm(pos1 - pos2) <= adaptive_threshold

    faces = []
    for y in range(h - 1):
        for x in range(w - 1):
            idx_tl = vertex_index_map[y, x]
            idx_tr = vertex_index_map[y, x + 1]
            idx_bl = vertex_index_map[y + 1, x]
            idx_br = vertex_index_map[y + 1, x + 1]

            valid = {}
            if idx_tl != -1:
                valid['tl'] = (idx_tl, position_3d_map[y, x])
            if idx_tr != -1:
                valid['tr'] = (idx_tr, position_3d_map[y, x + 1])
            if idx_bl != -1:
                valid['bl'] = (idx_bl, position_3d_map[y + 1, x])
            if idx_br != -1:
                valid['br'] = (idx_br, position_3d_map[y + 1, x + 1])

            if len(valid) == 4:
                tl_p, tr_p = valid['tl'][1], valid['tr'][1]
                bl_p, br_p = valid['bl'][1], valid['br'][1]
                if is_edge_valid(tl_p, br_p):
                    if is_edge_valid(tl_p, tr_p) and is_edge_valid(tr_p, br_p):
                        faces.append([valid['tl'][0], valid['br'][0], valid['tr'][0]])
                    if is_edge_valid(tl_p, bl_p) and is_edge_valid(bl_p, br_p):
                        faces.append([valid['tl'][0], valid['bl'][0], valid['br'][0]])

            elif len(valid) == 3:
                positions = list(valid.values())
                i0, p0 = positions[0]
                i1, p1 = positions[1]
                i2, p2 = positions[2]
                if is_edge_valid(p0, p1) and is_edge_valid(p1, p2) and is_edge_valid(p0, p2):
                    faces.append([i0, i1, i2])

    faces = np.array(faces) if faces else np.zeros((0, 3), dtype=np.int32)
    return recovered_points, faces


def save_mesh_with_color(points, faces, output_path, color):
    """Save mesh (or point cloud) as PLY with a given RGB color."""
    if len(points) == 0:
        return
    colors = np.tile(color, (len(points), 1)).astype(np.uint8)
    if len(faces) > 0:
        mesh = trimesh.Trimesh(vertices=points, faces=faces, vertex_colors=colors)
    else:
        mesh = trimesh.PointCloud(vertices=points, colors=colors)
    mesh.export(output_path)


def get_color_gradient(index, total):
    """Get a blue-to-red gradient color for the given index."""
    t = 0.5 if total == 1 else index / (total - 1)
    return (int(255 * t), 0, int(255 * (1 - t)))


# ==================== Main Inference Function ====================

def test_autoregressive(cfg):
    """Run auto-regressive inference."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    weight_dtype = torch.float32
    if cfg.get('mixed_precision', 'no') == "fp16":
        weight_dtype = torch.float16
    elif cfg.get('mixed_precision', 'no') == "bf16":
        weight_dtype = torch.bfloat16

    # ==================== Load Models ====================
    logger.info("Loading models...")

    has_vae_finetune = cfg.get('vae_finetune_path') and os.path.exists(cfg['vae_finetune_path'])
    has_unet_finetune = cfg.get('checkpoint_path') and (
        Path(cfg['checkpoint_path']).is_dir() or Path(cfg['checkpoint_path']).exists()
    )
    skip_pretrained = cfg.get('skip_pretrained_weights', False) or (has_vae_finetune and has_unet_finetune)

    if skip_pretrained:
        logger.info("Finetuned weights available — skipping pretrained weight loading")

    if skip_pretrained and has_vae_finetune:
        # Load only the architecture config, skip pretrained weights
        vae_config_path = Path(cfg['vae_model_path']) / "config.json"
        with open(vae_config_path) as f:
            vae_config = json.load(f)
        vae = AutoencoderKL.from_config(vae_config)
        logger.info("Initialized VAE from config")
    else:
        vae = AutoencoderKL.from_pretrained(cfg['vae_model_path'])
    vae.requires_grad_(False)
    vae.eval()
    vae = vae.to(device, dtype=weight_dtype)

    if has_vae_finetune:
        ckpt = torch.load(cfg['vae_finetune_path'], map_location=device, weights_only=False)
        vae.load_state_dict(ckpt['vae_state_dict'])
        logger.info(f"Loaded VAE finetuned weights from {cfg['vae_finetune_path']}")

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path=cfg['pretrained_model_path'],
        subfolder="unet",
        unet_additional_kwargs={
            "use_inflated_groupnorm": True,
            "unet_use_cross_frame_attention": False,
            "unet_use_temporal_attention": True,
            "use_motion_module": False,
        },
        load_pretrained_weights=not skip_pretrained,
    )
    denoising_unet.requires_grad_(False)
    denoising_unet.eval()
    denoising_unet = denoising_unet.to(device, dtype=weight_dtype)

    motion_embeddings = ActionEmbeddings(num_actions=5, embedding_dim=768)
    motion_embeddings.requires_grad_(False)
    motion_embeddings.eval()
    motion_embeddings = motion_embeddings.to(device, dtype=weight_dtype)

    # ==================== Load Checkpoint ====================
    checkpoint_path = Path(cfg['checkpoint_path'])
    if checkpoint_path.is_dir():
        # Accelerate checkpoint format
        for name, ext in [("model.safetensors", "safetensors"), ("pytorch_model.bin", "bin")]:
            path = checkpoint_path / name
            if path.exists():
                logger.info(f"Loading UNet from {path}")
                state = load_file(path) if ext == "safetensors" else torch.load(path, map_location=device)
                denoising_unet.load_state_dict(state, strict=False)
                break
        else:
            logger.warning(f"No UNet checkpoint found in {checkpoint_path}")

        for name, ext in [("model_1.safetensors", "safetensors"), ("pytorch_model_1.bin", "bin")]:
            path = checkpoint_path / name
            if path.exists():
                logger.info(f"Loading motion embeddings from {path}")
                state = load_file(path) if ext == "safetensors" else torch.load(path, map_location=device)
                motion_embeddings.load_state_dict(state)
                break
        else:
            logger.warning(f"No motion embeddings checkpoint found in {checkpoint_path}")
    else:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        denoising_unet.load_state_dict(ckpt['denoising_unet_state_dict'])
        motion_embeddings.load_state_dict(ckpt['motion_embeddings_state_dict'])

    logger.info("Successfully loaded checkpoint")

    if cfg.get('enable_xformers_memory_efficient_attention', True):
        try:
            denoising_unet.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory efficient attention enabled")
        except Exception:
            logger.info("xformers not available, using default attention")

    noise_scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
        prediction_type="epsilon",
    )

    # ==================== PCA (optional) ====================
    use_pca = cfg.get('use_pca_reconstruction', False)
    pca_model, pca_mask = None, None

    if use_pca:
        pca_path = cfg.get('pca_path')
        pca_mask_path = cfg.get('pca_mask_path')
        if not pca_path or not pca_mask_path:
            raise ValueError("pca_path and pca_mask_path must be provided when use_pca_reconstruction is True")
        if not os.path.exists(pca_path):
            raise FileNotFoundError(f"PCA model not found: {pca_path}")
        if not os.path.exists(pca_mask_path):
            raise FileNotFoundError(f"PCA mask not found: {pca_mask_path}")

        pca_model = joblib.load(pca_path)
        pca_mask = np.load(pca_mask_path)
        logger.info(f"PCA reconstruction enabled (mask pixels={pca_mask.sum()})")
    else:
        logger.info("PCA reconstruction disabled")

    # ==================== Motion Sequence ====================
    motion_sequence = expand_motion_sequence(cfg.get('motion_sequence'))
    total_frames = len(motion_sequence)
    num_frames_to_generate = total_frames - 3

    if num_frames_to_generate < 1:
        raise ValueError("motion_sequence must have at least 4 elements (3 initial + at least 1 to generate)")

    all_motion_types = ['Idle', 'Forward', 'Backward', 'Left', 'Right']
    for i, name in enumerate(motion_sequence):
        if name not in all_motion_types:
            raise ValueError(f"Invalid motion type at index {i}: '{name}'. Must be one of {all_motion_types}")

    logger.info(f"Motion sequence: {total_frames} total frames (3 initial + {num_frames_to_generate} generated)")
    for mt, count in Counter(motion_sequence).items():
        logger.info(f"  {mt}: {count} frames")

    # ==================== Dataset ====================
    logger.info("Loading dataset...")
    test_dataset = GeoActionDataset(
        root_folder=cfg['data_folder'],
        image_size=cfg.get('image_size', 128),
    )

    max_samples = cfg.get('max_test_samples', 20)
    if max_samples and max_samples < len(test_dataset):
        indices = random.sample(range(len(test_dataset)), max_samples)
        test_dataset = Subset(test_dataset, indices)
        logger.info(f"Using subset of {max_samples} samples")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True,
    )

    # ==================== Generation Settings ====================
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_scale = cfg.get('cfg_scale', 1.0)
    pure_noise_timestep = cfg.get('pure_noise_timestep', 999)
    pure_noise_steps = cfg.get('pure_noise_steps', 10)
    save_meshes = cfg.get('save_meshes', True)
    save_exr = cfg.get('save_exr', True)
    save_npy = cfg.get('save_npy', True)
    y_coordinate = cfg.get('keypoint_y_coordinate', 66)
    x_ranges = cfg.get('keypoint_x_ranges', [(0, 96)])
    print("x ranges: ", x_ranges)
    use_renormalization = cfg.get('use_renormalization', True)
    threshold_factor = cfg.get('threshold_factor', 0.05)

    logger.info(f"DDIM sampling: timestep={pure_noise_timestep}, steps={pure_noise_steps}, cfg_scale={cfg_scale}")

    generation_info = []

    # ==================== Auto-regressive Loop ====================
    with torch.no_grad():
        for sample_idx, batch in enumerate(tqdm(test_dataloader, desc="Processing samples")):
            images = batch['images'].to(device, dtype=torch.float32)
            case_name = batch['folder_name'][0]
            mask = batch['mask'][0].to(device)

            logger.info(f"Processing sample {sample_idx + 1}/{len(test_dataloader)}: {case_name}")

            # Initialize sliding window of 3 frames
            current_frames = [images[0, i].clone() for i in range(3)]

            # Store all frame data
            all_frames_data = []
            accumulated_shift = np.zeros(3, dtype=np.float32)

            for i in range(3):
                all_frames_data.append({
                    'frame': current_frames[i].clone(),
                    'motion_type': motion_sequence[i],
                    'index': i,
                    'shift': np.zeros(3),
                    'accumulated_shift': np.zeros(3).copy(),
                })

            sample_frames_info = []
            frame_pbar = tqdm(range(3, total_frames), desc="  Generating frames", leave=False)

            for frame_idx in frame_pbar:
                motion_name = motion_sequence[frame_idx]
                motion_idx = all_motion_types.index(motion_name)
                frame_pbar.set_description(f"  Frame {frame_idx + 1}/{total_frames} ({motion_name})")

                # Motion embedding
                motion_indices = torch.tensor([motion_idx], device=device, dtype=torch.long)
                cond_motion_embeds = motion_embeddings(motion_indices)

                # Encode 3 conditioning frames
                latents = []
                for cf in current_frames:
                    lat = vae.encode(cf.to(dtype=weight_dtype).unsqueeze(0)).latent_dist.mode()
                    lat = lat * vae.config.scaling_factor
                    latents.append(lat.unsqueeze(2))
                input_latents = torch.cat(latents, dim=2)

                # Generate 4th frame via DDIM
                latent_shape = (1, 4, 1, input_latents.shape[-2], input_latents.shape[-1])
                pure_noise = torch.randn(latent_shape, device=device, dtype=weight_dtype)

                denoised_latent = denoise_with_ddim(
                    pure_noise, input_latents, cond_motion_embeds,
                    denoising_unet, noise_scheduler,
                    pure_noise_timestep, pure_noise_steps, device,
                    cfg_scale=cfg_scale,
                )

                # Decode
                decoded = vae.decode(denoised_latent.squeeze(2).to(dtype=weight_dtype) / vae.config.scaling_factor).sample
                generated_4th = decoded[0].to(torch.float32)
                generated_4th = generated_4th * mask.float() + (-1.0) * (~mask).float()

                raw_generated_4th = generated_4th.clone()

                # Renormalization
                frame2_renorm, frame3_renorm, frame4_renorm, rgb_shift = apply_keypoint_renormalization(
                    current_frames[1], current_frames[2], raw_generated_4th, mask,
                    y_coordinate, x_ranges, apply_renorm=use_renormalization,
                )

                shift_position_space = rgb_shift.cpu().numpy() / 2.0
                accumulated_shift = accumulated_shift + shift_position_space

                all_frames_data.append({
                    'frame': raw_generated_4th.clone(),
                    'motion_type': motion_name,
                    'index': frame_idx,
                    'shift': shift_position_space.copy(),
                    'accumulated_shift': accumulated_shift.copy(),
                })

                sample_frames_info.append({
                    'frame_idx': frame_idx,
                    'motion_type': motion_name,
                    'rgb_shift': rgb_shift.cpu().numpy().tolist(),
                    'accumulated_shift': accumulated_shift.tolist(),
                })

                # Update sliding window
                if use_renormalization:
                    current_frames = [frame2_renorm.clone(), frame3_renorm.clone(), frame4_renorm.clone()]
                else:
                    current_frames = [current_frames[1].clone(), current_frames[2].clone(), raw_generated_4th.clone()]

            # ==================== Save Outputs ====================
            sample_output_dir = output_dir / case_name
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Saving {len(all_frames_data)} frames to {sample_output_dir}")

            # Apply PCA reconstruction as post-processing if enabled
            if use_pca and pca_model is not None and pca_mask is not None:
                logger.info("  Applying PCA reconstruction as post-processing...")
                for fd in all_frames_data:
                    pca_frame = apply_pca_reconstruction_to_tensor(fd['frame'], pca_model, pca_mask)
                    fd['frame'] = pca_frame * mask.float() + (-1.0) * (~mask).float()

            for fd in all_frames_data:
                frame = fd['frame']
                mt = fd['motion_type']
                idx = fd['index']
                prefix = f"{idx + 1}_{mt}"

                if save_meshes:
                    position_map = tensor_to_position_map(frame, mask)
                    translation = fd['accumulated_shift'] if use_renormalization else None
                    points, faces = recover_points_from_position_map(
                        position_map, translation=translation, threshold_factor=threshold_factor
                    )
                    if len(points) > 0:
                        color = get_color_gradient(idx, len(all_frames_data))
                        save_mesh_with_color(points, faces, str(sample_output_dir / f'{prefix}.ply'), color)

                if save_exr:
                    save_position_map_as_exr(frame, mask, sample_output_dir / f'{prefix}.exr')

                if save_npy:
                    acc_shift = fd['accumulated_shift'] if use_renormalization else None
                    save_position_map_with_shift_as_npy(frame, mask, acc_shift, sample_output_dir / f'{prefix}.npy')

            generation_info.append({
                'case_name': case_name,
                'total_frames': total_frames,
                'frames': sample_frames_info,
            })
            logger.info(f"  Sample {case_name} completed")

    # Save generation metadata
    info_dict = {
        'config': {
            'pure_noise_timestep': pure_noise_timestep,
            'pure_noise_steps': pure_noise_steps,
            'cfg_scale': cfg_scale,
            'use_renormalization': use_renormalization,
            'keypoint_y_coordinate': y_coordinate,
            'keypoint_x_ranges': x_ranges,
            'use_pca_reconstruction': use_pca,
        },
        'num_samples': len(test_dataloader),
        'total_frames': total_frames,
        'frames_generated': num_frames_to_generate,
        'motion_sequence': motion_sequence,
        'generation_details': generation_info,
    }

    with open(output_dir / 'generation_info.json', 'w') as f:
        json.dump(info_dict, f, indent=2)

    logger.info(f"Auto-regressive testing completed successfully!")
    logger.info(f"Generated {total_frames} total frames ({num_frames_to_generate} generated + 3 initial)")
    logger.info(f"Results saved to {output_dir}")

    return info_dict


# ==================== Entry Point ====================

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    # Resolve relative paths in config against the project root so the script
    # works regardless of the working directory (e.g. running from Geometry/).
    _project_root = Path(__file__).resolve().parent.parent
    for _key in [
        'data_folder', 'vae_model_path', 'vae_finetune_path',
        'pretrained_model_path', 'checkpoint_path',
        'pca_path', 'pca_mask_path', 'output_dir',
    ]:
        _val = config.get(_key)
        if _val is not None and not os.path.isabs(_val):
            config[_key] = str(_project_root / _val)

    test_autoregressive(config)