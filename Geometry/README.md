# UNICA: A Unified Neural Framework for Controllable 3D Avatars [Geometry Part]

![Pipeline](assets/pipeline.png)
The Geometry stage generates sequences of **position maps** for animated avatars using a **latent diffusion model**. Given three initial position-map frames and a sequence of discrete motion commands (*Idle*, *Forward*, *Backward*, *Left*, *Right*), the model auto-regressively produces new position maps at each time step.
The generated position maps can be converted to 3D meshes for quick visualization, or passed to the [Appearance](../Appearance/README.md) stage for texture generation.

<br><br>


# Inference

## 1. Weight Preparation

Download pretrained weights from <https://huggingface.co/zjh21/UNICA> and place them in the `weights` path. The expected structure is:

```
weights/                              # at the root path
├── berserker/   
│   ├── diffusion                     # model(_1).safetensors or pytorch_model(_1).bin
│   ├── pca                           # pca_200.ckpt and pca_mask.npy
│   ├── vae.pt
├── cowgirl/   
│   ├── diffusion                     # model(_1).safetensors or pytorch_model(_1).bin
│   ├── pca                           # pca_200.ckpt and pca_mask.npy
│   ├── vae.pt
├── ...

Geometry/
├── configs
├── inference.py
```

## 2. Run Inference

From the repository root:

```bash
cd Geometry
python inference.py --config configs/inference/cowgirl.yaml
```

The script should be runnable if weights are correctly prepared. Or edit `configs/inference/${caseName}.yaml` to specify your data paths, model checkpoint paths, and the desired motion sequence. 

## 3. Input Data Structure

The inference script expects a `data_folder` containing one or more subfolders, each holding **4 consecutive position map frames** as `.exr` files:

```
<data_folder>/
├── <sequence_name>/
│   ├── 1.exr
│   ├── 2.exr
│   ├── 3.exr
│   └── 4.exr
├── ...
```

> **Note:** For the provided example data in `assets/`, the 4 `.exr` files are simply **duplications of the 1st frame**. Only the first 3 frames are used as the initial conditioning window; the 4th frame is not consumed by the auto-regressive pipeline.

## 4. Output Format

Outputs are saved to the directory specified by `output_dir` in the config, organized by input subfolder name:

```
<output_dir>/
├── <sequence_name>/
│   ├── 1_Forward.exr
│   ├── 1_Forward.npy
│   ├── 1_Forward.ply
│   ├── 2_Forward.exr
│   ├── 2_Forward.npy
│   ├── 2_Forward.ply
│   ├── ...
```

Each generated frame produces up to three files (configurable via `save_exr`, `save_npy`, `save_meshes` in the config):

| File | Description |
|------|-------------|
| `*.exr` | Raw model output — 6-view position map in [0, 1] range |
| `*.npy` | Position map with **accumulated positional shifts** applied, for progressive 4D inference |
| `*.ply` | Triangle mesh created by connecting neighboring foreground pixels — useful for a **quick visual check** of geometry quality without the Appearance stage |

<br><br>


# Training

🚧 We are still in the process of cleaning up the training code. In the meantime, we are preparing a data acquisition guide to help you prepare the training data. Below is an example of how to run the training once the data is ready.

## 1. Pretrained Weight Preparation

Download the Stable Diffusion v1.5 UNet and VAE weights from [HuggingFace](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/unet) and organize them as follows:

```
pretrained_models/
├── stable-diffusion-v1-5/
│   ├── unet/
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.bin
│   └── vae/
│       ├── config.json
│       └── diffusion_pytorch_model.bin
```

## 2. Training Data Structure

The training data folder should contain subfolders of **4-frame position map groups**, each labeled with an action type:

```
<data_folder>/
├── {transitionSequence}_{frameNumber:05d}_{action}/
│   ├── 1.exr
│   ├── 2.exr
│   ├── 3.exr
│   └── 4.exr
├── ...
```

For example:

```
posmap/
├── 0-a_00001_Idle/
│   ├── 1.exr
│   ├── 2.exr
│   ├── 3.exr
│   └── 4.exr
├── 0-a_00002_Idle/
│   ├── 1.exr
│   ├── 2.exr
│   ├── 3.exr
│   └── 4.exr
├── ...
├── w-d-rfoot_00157_Backward/
│   ├── 1.exr
│   ├── 2.exr
│   ├── 3.exr
│   └── 4.exr
├── ...
```

## 3. Launch Training

**Step 1: VAE Fine-tuning**

From `Geometry/`:

```bash
accelerate launch train_vae.py --config configs/train_vae.yaml
```

Edit `configs/train_vae.yaml` to set your `data_dir`, `pretrained_vae_path`, and output directories.

**Step 2: Diffusion Model Training**

```bash
accelerate launch train_diffusion.py --config configs/train_diffusion.yaml
```

Edit `configs/train_diffusion.yaml` to set your data folder, pretrained model paths, and the path to the fine-tuned VAE checkpoint from Step 1 (`vae_finetune_path`).

> **Note:** Two-stage diffusion training is already packed in the script. The stage boundary is controlled by `stage1_steps` and `stage2_steps` in the config.
