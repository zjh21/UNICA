# UNICA: A Unified Neural Framework for Controllable 3D Avatars

<!-- TODO: Add project page, arxiv, and other badge links below -->
[Paper](<!-- TODO: arxiv URL -->) | [Models](<https://huggingface.co/zjh21/UNICA>)

![Teaser image](<assets/teaser.png>)


## 🎬 Video Demo
<div align="center">
  <video src="https://github.com/user-attachments/assets/172ad9d0-c59b-4dea-8768-529875d2bc9b" width="100%" poster=""> </video>
</div>

## Installation

We tested on Ubuntu 22.04 and CUDA 11.8. Other similar configurations should also work.

```bash
git clone --recursive https://github.com/zjh21/UNICA.git
cd UNICA
conda create -n unica python=3.8 -y
conda activate unica

# Install the PyTorch and other dependencies
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# [For Appearance Only] Install Pointcept and flash-attention for Point Transformer v3
cd Appearance
pip install Pointcept/
pip install Pointcept/libs/pointops
pip install flash-attn --no-build-isolation

# [For Appearance Only] Install 3DGS-related libraries
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

# [Optional] Install PyTorch3D — only required for dataset preparation (position map rendering)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## Inference & Training

UNICA inference is a **two-stage** pipeline. You should run **Geometry first**, then **Appearance**. Please refer to [Geometry/README.md](Geometry/README.md) and [Appearance/README.md](Appearance/README.md) for detailed instructions.

<div align="center">
  <video src="https://github.com/user-attachments/assets/0bbc71d8-a27c-4c01-ac1e-b72a4353d8fb" width="100%" poster=""> </video>
  <video src="https://github.com/user-attachments/assets/4604e1ef-8669-4b43-8139-a86dd8cf9b86" width="100%" poster=""> </video>
  <video src="https://github.com/user-attachments/assets/9f095cde-9cbe-410b-8e8f-ef419348653b" width="100%" poster=""> </video>
  <video src="https://github.com/user-attachments/assets/697ad646-a47f-43f1-be3f-da4ace1cbf78" width="100%" poster=""> </video>
</div>


## Acknowledgements

This project builds upon [SplatFormer](https://github.com/ChenYutongTHU/SplatFormer) and [Champ](https://github.com/fudan-generative-vision/champ). We thank the authors for their excellent work.
