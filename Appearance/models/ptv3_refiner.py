"""
PTv3 Gaussian Refiner
=====================
A Point Transformer V3 backbone with per-attribute residual heads that
refines coarse 3D Gaussian Splatting parameters.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List

from .pointtransformer_v3 import PointTransformerV3Model


# ---------------------------------------------------------------------------
# Feature dimensions for each Gaussian attribute
# ---------------------------------------------------------------------------

FEATURE2CHANNEL = {
    "means": 3,
    "features_dc": 3,
    "opacities": 1,
    "scales": 3,
    "quats": 4,
}

ALL_FEATURES = ["means", "features_dc", "opacities", "scales", "quats"]


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

class MinMaxScaler:
    """Scale point positions to [0, 1] and undo the transform."""

    def __init__(self):
        self.min_ = None
        self.max_ = None
        self.scale_ = None

    def fit(self, data: torch.Tensor):
        self.min_ = data.min(dim=0, keepdim=True)[0]
        self.max_ = data.max(dim=0, keepdim=True)[0]
        self.scale_ = torch.clamp(self.max_ - self.min_, min=1e-6)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.min_) / self.scale_

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.scale_ + self.min_


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PTv3GaussianRefiner(nn.Module):
    """
    Point Transformer V3 based refiner for 3D Gaussian Splatting.

    Given a set of coarse Gaussians (means, SH-DC, opacity, scales,
    quaternions), the model predicts *residual* corrections through a PTv3
    backbone followed by per-attribute MLP heads.
    """

    def __init__(self):
        super().__init__()

        # ---- Configuration (hard-coded for reproducibility) ----
        self.sh_degree = 0
        self.output_head_nlayer = 4
        self.output_head_width = 128
        self.grid_resolution = 192
        self.input_feat_to_mlp = True
        self.zeroinit = True

        self.input_features = ["means", "scales", "opacities", "quats", "features_dc"]
        self.output_features = ["means", "scales", "opacities", "quats", "features_dc"]

        self.res_feature_activation = nn.ModuleDict({
            "means": nn.Tanh(),
            "features_dc": nn.Identity(),
            "scales": nn.Identity(),
            "opacities": nn.Identity(),
            "quats": nn.Identity(),
        })

        # ---- Backbone ----
        in_channels = sum(FEATURE2CHANNEL[f] for f in self.input_features)
        self.gs_features_dim = in_channels

        self.backbone = PointTransformerV3Model(
            in_channels=in_channels,
            enable_flash=True,
            output_dim=64,
            enc_dim=32,
            turn_off_bn=False,
            stride=(1, 2, 2, 2),
            embedding_type="MLP",
        )

        # ---- Per-attribute MLP heads ----
        head_in = self.backbone.output_dim
        if self.input_feat_to_mlp:
            head_in += in_channels

        self.features_outputhead = nn.ModuleDict()
        for feat_name in self.output_features:
            layers = []
            for i in range(self.output_head_nlayer - 1):
                dim_in = head_in if i == 0 else self.output_head_width
                layers.extend([nn.Linear(dim_in, self.output_head_width), nn.ReLU()])
            final_in = self.output_head_width if self.output_head_nlayer > 1 else head_in
            layers.append(nn.Linear(final_in, FEATURE2CHANNEL[feat_name]))
            self.features_outputhead[feat_name] = nn.Sequential(*layers)

        # Zero-init last layers so that initial output ≈ input.
        if self.zeroinit:
            for module in self.features_outputhead.values():
                nn.init.zeros_(module[-1].weight)
                nn.init.zeros_(module[-1].bias)

    # -----------------------------------------------------------------
    # Normalisation / un-normalisation
    # -----------------------------------------------------------------

    @staticmethod
    def normalize_gs(batch_gs):
        scalers, normed = [], []
        for gs in batch_gs:
            scaler = MinMaxScaler()
            scaler.fit(gs["means"])
            normed.append({
                "means": scaler.transform(gs["means"]),
                "scales": gs["scales"] + torch.log(scaler.scale_),
                "features_dc": gs["features_dc"],
                "opacities": gs["opacities"],
                "quats": gs["quats"],
            })
            scalers.append(scaler)
        return normed, scalers

    @staticmethod
    def unnormalize_gs(batch_gs, scalers):
        out = []
        for gs, scaler in zip(batch_gs, scalers):
            d = {}
            for k in gs:
                if k == "means":
                    d[k] = scaler.inverse_transform(gs[k])
                elif k == "scales":
                    d[k] = gs[k] - torch.log(scaler.scale_)
                else:
                    d[k] = gs[k]
            out.append(d)
        return out

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(self, batch_gs: List[dict]) -> List[dict]:
        """
        Args:
            batch_gs: List of dicts, each with keys ``means (N,3)``,
                ``features_dc (N,3)``, ``opacities (N,1)``, ``scales (N,3)``,
                ``quats (N,4)``.

        Returns:
            List of dicts with refined Gaussian parameters (same structure).
        """
        device = batch_gs[0]["means"].device

        # 1. Normalise.
        batch_norm, scalers = self.normalize_gs(batch_gs)

        # 2. Concatenate features across the batch.
        offset = torch.tensor(
            [gs["means"].shape[0] for gs in batch_norm]
        ).cumsum(0).to(device)

        feat_list = []
        for gs in batch_norm:
            feat_list.append(torch.cat([gs[k] for k in self.input_features], dim=1))
        feat = torch.cat(feat_list, dim=0)

        coord = torch.cat([gs["means"] for gs in batch_norm], dim=0)

        # 3. Backbone.
        backbone_out = self.backbone({
            "coord": coord,
            "grid_size": torch.ones(3, device=device) / self.grid_resolution,
            "offset": offset,
            "feat": feat,
            "grid_coord": torch.floor(coord * self.grid_resolution).int(),
        })
        hidden = backbone_out["feat"]

        if self.input_feat_to_mlp:
            hidden = torch.cat([hidden, feat], dim=1)

        # 4. Per-attribute residual predictions.
        output = OrderedDict()
        for name in self.output_features:
            pred = self.features_outputhead[name](hidden)
            pred = self.res_feature_activation[name](pred)
            output[name] = pred

        # 5. Un-batchify and add residuals.
        out_norm = []
        left = 0
        for right, in_gs in zip(offset.tolist(), batch_norm):
            out_gs = {k: in_gs[k] + output[k][left:right] for k in self.output_features}
            out_norm.append(out_gs)
            left = right

        # Copy any features that are not predicted.
        for key in ALL_FEATURES:
            if key not in self.output_features:
                for o, i in zip(out_norm, batch_norm):
                    o[key] = i[key]

        # 6. Un-normalise.
        return self.unnormalize_gs(out_norm, scalers)