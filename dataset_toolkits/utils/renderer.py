import numpy as np
import torch

from pytorch3d.renderer import (
    OrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    BlendParams,
)
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.blending import hard_rgb_blend
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes


class VertexAttributeShader(ShaderBase):
    """Custom shader that renders vertex attributes directly without lighting."""

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get("blend_params", self.blend_params)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images


class PositionMapRenderer:
    """Renderer for position maps using PyTorch3D orthographic cameras."""

    def __init__(self, img_w: int, img_h: int, bg_color=(0, 0, 0), device="cuda"):
        self.img_w = img_w
        self.img_h = img_h
        self.device = device if torch.cuda.is_available() else "cpu"

        raster_settings = RasterizationSettings(
            image_size=(img_h, img_w),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            max_faces_per_bin=50000,
        )

        blend_params = BlendParams(background_color=bg_color)
        shader = VertexAttributeShader(device=self.device, blend_params=blend_params)

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=None, raster_settings=raster_settings),
            shader=shader,
        )
        self.mesh = None

    def set_camera(self, extr, xmag=0.65, ymag=0.65):
        """Set orthographic camera from extrinsic matrix with adjustable magnification."""
        affine_mat = np.identity(4, np.float32)
        affine_mat[0, 0] = -1
        affine_mat[1, 1] = -1
        extr = affine_mat @ extr
        extr[:3, :3] = np.linalg.inv(extr[:3, :3])
        extr = torch.from_numpy(extr).to(torch.float32).to(self.device)

        focal_x = self.img_w / (2.0 * xmag)
        focal_y = self.img_h / (2.0 * ymag)

        cameras = OrthographicCameras(
            focal_length=((focal_x, focal_y),),
            principal_point=((self.img_w / 2.0, self.img_h / 2.0),),
            R=extr[:3, :3].unsqueeze(0),
            T=extr[:3, 3].unsqueeze(0),
            in_ndc=False,
            device=self.device,
            image_size=((self.img_h, self.img_w),),
        )
        self.renderer.rasterizer.cameras = cameras

    def set_model(self, vertices, vertex_attributes):
        """Set mesh model with vertex attributes as texture."""
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices)
        if isinstance(vertex_attributes, np.ndarray):
            vertex_attributes = torch.from_numpy(vertex_attributes)

        vertices = vertices.to(torch.float32).to(self.device)
        vertex_attributes = vertex_attributes.to(torch.float32).to(self.device)

        faces = (
            torch.arange(0, vertices.shape[0], dtype=torch.int64)
            .to(self.device)
            .reshape(-1, 3)
        )
        textures = TexturesVertex([vertex_attributes])
        self.mesh = Meshes([vertices], [faces], textures=textures)

    def render(self):
        """Render the mesh and return as numpy array (H, W, 4)."""
        img = self.renderer(self.mesh, cameras=self.renderer.rasterizer.cameras)
        return img[0].cpu().numpy()
