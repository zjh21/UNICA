# UNICA: A Unified Neural Framework for Controllable 3D Avatars

<!-- TODO: Add project page, arxiv, and other badge links below -->
[Project Page](<!-- TODO: project page URL -->) | [Paper](<!-- TODO: arxiv URL -->) | [Models](<https://huggingface.co/zjh21/UNICA>)

![Teaser image](<assets/teaser.png>)

<!-- TODO: Add any additional model weight entries as needed -->

## Installation

We tested on a server configured with Ubuntu 22.04 and CUDA 11.8. Other similar configurations should also work.

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


## Acknowledgements

This project builds upon [SplatFormer](https://github.com/ChenYutongTHU/SplatFormer) and [Champ](https://github.com/fudan-generative-vision/champ). We thank the authors for their excellent work.
