import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from diffusers.image_processor import VaeImageProcessor
import cv2
import pdb

class GeoVAEDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_size=512):
        """
        Dataset class for loading images from subfolders.
        
        Args:
            root_dir: Root directory containing subfolders
            transform: Optional transform to apply to images
            image_size: Size to resize images to
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        
        self.preprocessor = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True)
        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] for VAE
            ])
            
        # Collect all image paths from two levels of folders
        self.image_paths = []
        for first_level_dir in sorted(self.root_dir.iterdir()):
            if first_level_dir.is_dir():
                # Get all .exr files in this directory
                    for exr_file in sorted(first_level_dir.glob('*.exr')):
                        self.image_paths.append(exr_file)
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        
        # Extract foreground (assuming first image is representative)
        if len(self.image_paths) > 0:
            self.mask = self._extract_foreground()
        else:
            self.mask = None
    
    def _extract_foreground(self, num_samples=10):
        """Extract common foreground from a sample of images."""
        image = cv2.imread(str(self.image_paths[0]), cv2.IMREAD_UNCHANGED)
        image = self.preprocessor.preprocess(image, self.image_size, self.image_size).squeeze(0)
    
        # Create foreground mask (non-background pixels)
        mask = (image != -1.0)  # Pixels that are NOT background
        return mask
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # image = Image.open(img_path).convert('RGB')
        # image = self.preprocessor.preprocess(image, 512, 512).squeeze(0)
        # if self.transform:
            # image = self.transform(image)
            
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if np.max(image) > 1:
            image = (image / 255.0).astype(np.float32)
        image = self.preprocessor.preprocess(image, self.image_size, self.image_size).squeeze(0)
        assert torch.all(image[~self.mask] == -1.0), "Not all masked pixels are -1.0"
        # print(image[:, 125:135, 125:135])
        return image