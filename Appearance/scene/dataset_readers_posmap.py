import os
import sys
from PIL import Image
from typing import NamedTuple
import numpy as np
import json
from pathlib import Path
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import torch


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool


class SceneInfo(NamedTuple):
    point_cloud: object  # BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def getNerfppNorm(cam_info):
    """
    Calculate NeRF++ normalization parameters.
    """
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readCameraParametersFromTransforms(path, transformsfile, extension=".png"):
    """
    Read camera parameters (extrinsics and intrinsics) from transforms.json file.
    Returns list of camera parameters without loading actual images.
    """
    cam_params = []

    transforms_file_path = os.path.join(path, transformsfile)
    
    if not os.path.exists(transforms_file_path):
        raise FileNotFoundError(f"Transforms file not found: {transforms_file_path}")
    
    with open(transforms_file_path) as json_file:
        contents = json.load(json_file)
        
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # Handle file path to get the relative path and image name
            file_path = frame["file_path"]
            fovx = frame["camera_angle_x"]
            
            # Check if it's a relative or absolute path
            if not file_path.endswith(extension):
                file_path = file_path + extension
            
            # Remove leading './' if present
            if file_path.startswith('./'):
                file_path = file_path[2:]
            
            image_name = Path(file_path).stem

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            # Store camera parameters with relative file path
            cam_params.append({
                'uid': idx,
                'R': R,
                'T': T,
                'FovX': fovx,
                'file_path': file_path,  # relative path like "000.png"
                'image_name': image_name
            })

    return cam_params


def loadImagesForCameras(cam_params, images_folder, white_background=False):
    """
    Load images for cameras given camera parameters and image folder path.
    
    Args:
        cam_params: List of camera parameter dicts from readCameraParametersFromTransforms
        images_folder: Path to folder containing the images
        white_background: Whether to use white background
    
    Returns:
        List of CameraInfo objects with loaded image paths
    """
    cam_infos = []
    
    for cam_param in cam_params:
        # Construct full image path
        image_path = os.path.join(images_folder, cam_param['file_path'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            continue
        
        # Load image to get dimensions
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        # Calculate FovY from FovX
        from utils.graphics_utils import focal2fov, fov2focal
        fovy = focal2fov(fov2focal(cam_param['FovX'], image.size[0]), image.size[1])
        
        cam_infos.append(CameraInfo(
            uid=cam_param['uid'],
            R=cam_param['R'],
            T=cam_param['T'],
            FovY=fovy,
            FovX=cam_param['FovX'],
            image_path=image_path,
            image_name=cam_param['image_name'],
            width=image.size[0],
            height=image.size[1],
            depth_path="",
            depth_params=None,
            is_test=False
        ))
    
    return cam_infos


def load_position_map_to_pointcloud(position_map_path, default_color=0.5):
    """
    Load position map and convert to BasicPointCloud.
    
    Args:
        position_map_path: Path to the position map (.exr file)
        default_color: Default grayscale color value for points
    
    Returns:
        BasicPointCloud object
    """
    import cv2 as cv
    
    # Read position map
    position_map = cv.imread(position_map_path, cv.IMREAD_UNCHANGED | cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    
    if position_map is None:
        raise ValueError(f"Could not read position map from {position_map_path}")
    
    # Normalize if needed
    if position_map.dtype == np.uint8:
        position_map = position_map.astype(np.float32) / 255.0
    elif position_map.dtype == np.uint16:
        position_map = position_map.astype(np.float32) / 65535.0
    else:
        position_map = position_map.astype(np.float32)
    
    # Ensure 3 channels
    if len(position_map.shape) == 2:
        position_map = np.stack([position_map] * 3, axis=-1)
    elif position_map.shape[2] > 3:
        position_map = position_map[:, :, :3]
    
    print(f"Position map shape: {position_map.shape}")
    print(f"Position map dtype: {position_map.dtype}")
    print(f"Position map value range: [{position_map.min():.3f}, {position_map.max():.3f}]")
    
    h, w = position_map.shape[:2]
    
    # Create a mask for non-background pixels (pixels that are not all zeros)
    mask = np.any(position_map != 0, axis=2)
    
    # Extract 3D points from non-background pixels
    recovered_points = []
    
    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                # Get the 3D position from the pixel
                point_3d = position_map[y, x].copy()
                
                # Convert from [0, 1] to [-0.5, 0.5]
                point_3d = point_3d - 0.5
                
                recovered_points.append(point_3d)
    
    recovered_points = np.array(recovered_points) if recovered_points else np.array([]).reshape(0, 3)
    
    print(f"Recovered {len(recovered_points)} points from position map")
    
    # Create default colors (gray)
    colors = np.ones_like(recovered_points) * default_color
    
    # Create BasicPointCloud
    pcd = BasicPointCloud(points=recovered_points, colors=colors, normals=None)
    
    return pcd