import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
from PIL import Image
import torchvision.transforms as transforms

from diffusers.image_processor import VaeImageProcessor
from torch.utils.data import Dataset
import torch
import cv2
import random

# ==================== Dataset Implementation ====================
class GeoActionDataset(Dataset):
    def __init__(self, root_folder, image_size=128, stage='stage1'):
        self.root_folder = root_folder
        self.image_size = image_size
        self.stage = stage
        self.action_types = ['Idle', 'Forward', 'Backward', 'Left', 'Right']
        self.action_to_idx = {action: idx for idx, action in enumerate(self.action_types)}
        
        # Collect all valid subfolders
        self.samples = []  # Normal samples (not near turning points)
        self.in_between_samples = []  # Samples near turning points
        
        # Get all subfolders and sort them by name
        subfolders = sorted([f for f in os.listdir(root_folder) 
                           if os.path.isdir(os.path.join(root_folder, f))])
        
        # Parse folder names to extract sequence and frame number
        # Format: {sequence_name}_{frame_number}_{action_type}
        sequence_data = {}  # {sequence_name: [(frame_num, action_type, folder_path, subfolder_name)]}
        
        for subfolder in subfolders:
            folder_path = os.path.join(root_folder, subfolder)
            
            # Extract action type
            action_type = None
            for mt in self.action_types:
                if mt in subfolder:
                    action_type = mt
                    break
            
            if action_type is not None:
                # Check if all 4 images exist
                images_exist = all(
                    os.path.exists(os.path.join(folder_path, f"{i}.exr"))
                    for i in range(1, 5)
                )
                if images_exist:
                    # Parse sequence name and frame number
                    # Format: sequencename_framenumber_actiontype
                    parts = subfolder.rsplit('_', 2)  # Split from right, max 2 splits
                    if len(parts) == 3:
                        sequence_name = parts[0]
                        try:
                            frame_num = int(parts[1])
                            
                            if sequence_name not in sequence_data:
                                sequence_data[sequence_name] = []
                            sequence_data[sequence_name].append((frame_num, action_type, folder_path, subfolder))
                        except ValueError:
                            print(f"Warning: Could not parse frame number from {subfolder}")
        
        # Identify turning points for each sequence
        turning_points = {}  # {sequence_name: [frame_numbers]}
        
        for sequence_name, frames in sequence_data.items():
            # Sort by frame number
            frames.sort(key=lambda x: x[0])
            
            turning_points[sequence_name] = []
            
            # Detect action changes
            for i in range(1, len(frames)):
                prev_action = frames[i-1][1]
                curr_action = frames[i][1]
                curr_frame_num = frames[i][0]
                
                if prev_action != curr_action:
                    turning_points[sequence_name].append(curr_frame_num)
                    print(f"Turning point detected: {sequence_name} at frame {curr_frame_num} ({prev_action} → {curr_action})")
        
        # Categorize samples as normal or in-between
        for sequence_name, frames in sequence_data.items():
            turning_frame_nums = turning_points.get(sequence_name, [])
            
            for frame_num, action_type, folder_path, subfolder in frames:
                is_in_between = False
                
                # Check if this frame is in-between (5 before or 40 after a turning point)
                for turning_point in turning_frame_nums:
                    if turning_point - 5 <= frame_num <= turning_point + 40:
                        is_in_between = True
                        break
                
                sample = (folder_path, action_type, subfolder)
                
                if is_in_between:
                    self.in_between_samples.append(sample)
                else:
                    self.samples.append(sample)
        
        print(f"\n{'='*60}")
        print(f"Dataset Statistics for {stage}:")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.samples) + len(self.in_between_samples)}")
        print(f"Normal samples (far from turning points): {len(self.samples)}")
        print(f"In-between samples (near turning points): {len(self.in_between_samples)}")
        print(f"Number of sequences: {len(sequence_data)}")
        print(f"Number of turning points: {sum(len(tp) for tp in turning_points.values())}")
        print(f"{'='*60}\n")
        
        # Image transforms
        self.preprocessor = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True)
        
        # Extract foreground mask from first sample
        all_samples = self.samples + self.in_between_samples
        if len(all_samples) > 0:
            self.mask = self._extract_foreground(all_samples[0])
        else:
            self.mask = None
    
    def _extract_foreground(self, sample):
        """Extract common foreground from first image."""
        folder_path, _, _ = sample
        img_path = os.path.join(folder_path, "1.exr")
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = self.preprocessor.preprocess(image, self.image_size, self.image_size).squeeze(0)
        
        # Create foreground mask (non-background pixels)
        mask = (image != -1.0)  # Pixels that are NOT background
        return mask
    
    def __len__(self):
        if self.stage == 'stage1':
            # Stage 1: Only use normal samples (exclude in-between)
            return len(self.samples)
        else:
            # Stage 2: Return total samples for epoch calculation
            # Actual sampling uses oversampling strategy
            return len(self.samples) + len(self.in_between_samples)
    
    def __getitem__(self, idx):
        if self.stage == 'stage1':
            # Stage 1: Only normal samples, exclude in-between frames
            if len(self.samples) == 0:
                raise ValueError("No normal samples available for stage 1")
            folder_path, action_type, folder_name = self.samples[idx % len(self.samples)]
        else:
            # Stage 2: Oversample in-between frames
            # 70% probability of getting in-between frame, 30% normal frame
            if len(self.in_between_samples) > 0 and random.random() < 0.7:
                # Sample from in-between frames
                sample_idx = random.randint(0, len(self.in_between_samples) - 1)
                folder_path, action_type, folder_name = self.in_between_samples[sample_idx]
            else:
                # Sample from normal frames
                if len(self.samples) > 0:
                    sample_idx = random.randint(0, len(self.samples) - 1)
                    folder_path, action_type, folder_name = self.samples[sample_idx]
                else:
                    # Fallback if no normal samples (shouldn't happen)
                    sample_idx = random.randint(0, len(self.in_between_samples) - 1)
                    folder_path, action_type, folder_name = self.in_between_samples[sample_idx]
        
        # Load 4 consecutive frames
        images = []
        for i in range(1, 5):
            img_path = os.path.join(folder_path, f"{i}.exr")
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            image = self.preprocessor.preprocess(image, self.image_size, self.image_size).squeeze(0)
            images.append(image)
        
        # Stack images along a new dimension
        images = torch.stack(images, dim=0)  # (4, 3, H, W)
        
        # Get action type index
        action_idx = self.action_to_idx[action_type]
        
        return {
            'images': images,
            'action_type': action_idx,
            'action_name': action_type,
            'folder_name': folder_name,
            'mask': self.mask  # Include mask in batch
        }