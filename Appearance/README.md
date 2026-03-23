# [ICLR' 25] SplatFormer: Point Transformer for Robust 3D Gaussian Splatting
[Project page](https://sergeyprokudin.github.io/splatformer/) | [Paper](https://openreview.net/pdf?id=9NfHbWKqMF) <br>

![Teaser image](assets/teaser.png)

This repo contains the official implementation for the paper "SplatFormer: Point Transformer for Robust 3D Gaussian Splatting". Our approach uses a point transformer to refine 3DGS for out-of-distribution novel view synthesis in a single feed-forward.

## Installation
We tested on a server configured with Ubuntu 22.04, cuda 11.8, and gcc 8.5.0. Other similar configurations should also work.
```
git clone --recursive git@github.com:ChenYutongTHU/SplatFormer.git
cd SplatFormer
conda create -n splatformer python=3.8 -y
conda activate splatformer

# Install the pytorch version for your cuda version.
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install pointcept and flash-attention for point transformer v3
pip install Pointcept/
pip install flash-attn --no-build-isolation

# Install Other dependencies
pip install -r requirements.txt

# Install gsplat
pip install git+https://github.com/nerfstudio-project/gsplat.git@v0.1.11
```

## Out-of-distribution (OOD) Novel View Synthesis Test Sets
Our OOD-NVS test sets can be downloaded [here](https://drive.google.com/file/d/1-mUCl-yxe1aE0rrQDHKlXk1J2n8d1-60/view?usp=sharing). There are three object-centric OOD NVS test sets rendered from ShapeNet-core, Objaverse-v1, and GSO, and one real-world iPhone image set captured by us. All scene directories are in colmap-like structure. 

For object-centric, you can follow this [instruction](DataGenerator/) to re-render the 3D scenes by yourself. We use the ground-truth camera pose provided by Blender and save the bounding box coordinates of the object, which will be used in initialization for 3DGS training.

For real-world iPhone captures, we use [hloc](https://github.com/cvg/Hierarchical-Localization) to estimate the camera poses. The provided point cloud is estimated from the training cameras and the provided images are already undistorted.

## Training SplatFormer
### Training set generation 
We provide rendering and 3DGS scripts to generate our training datasets. Please see [DatasetGenerator](DataGenerator/) for more details.
After generating the rendered images and initial 3DGS, please put them under train-set as 

    .
    └── train-set                    
        ├── objaverseOOD 
            ├── colmap
               ├── 
            ├── nerfstudio   
               ├──  
We also provide a small subset of our training set [here](https://drive.google.com/file/d/15QfgGKtYvGRwZaRst0bbUAdlIw5k41TU/view?usp=sharing).

### Train
```
sh scripts/train-on-objaverse_gpux8-accum4.sh
sh scripts/train-on-shapenet_gpux8-accum4.sh
```
By default, we use 8x4090 gpus or 8x3090 gpus, and a accumulation step of 4. You can change the configurations in the training script. You can download our trained checkpoints [here](https://drive.google.com/drive/folders/1WkrOexVd8S0lqbnr8Jx0wSqAiQQYVKdm?usp=sharing).

## Evaluating SplatFormer
To evaluate the trained SplatFormer, please download the [OOD-NVS test sets and the initial 3DGS](https://drive.google.com/file/d/1-mUCl-yxe1aE0rrQDHKlXk1J2n8d1-60/view?usp=drive_link) trained using the input views, unextract them and put them under test-set/ as 

    .
    └── test-set                    
        ├── GSOOOD  
            ├── colmap
            └── nerfstudio     
        ├── objaverseOOD         
        └── RealOOD                
You can download SplatFormers trained on Objaverse-v1 [here](https://drive.google.com/file/d/1l5RAGrkdRFRhR6KDgxZk5JdkDBM-7jh_/view?usp=sharing) and run the evaluations.
```
sh scripts/train-on-objaverse_inference.sh # Evaluate the model trained on Objaverse-v1 on Objaverse-OOD, GSO-OOD, and Real-OOD
sh scripts/train-on-objaverse_inference.sh # Evaluate the model trained on ShapeNet on ShapeNet-OOD
```
Then under the output directory (e.g. outputs/objaverse_splatformer/test), you can see the evaluation metrics in eval.log, OOD renderings in objaverse/pred, and compare with 3DGS in objaverse/compare.

* Note: Our SplatFormer takes 3DGS trained for 10k steps as input. In our paper, we report 3DGS trained for 30k steps (default setting) as baselines. The two 3DGS training configurations lead to only small difference in the evaluation performance.

## Real-time 3DGS Viewer
1. First, follow the [instrutions](https://github.com/graphdeco-inria/gaussian-splatting/tree/main?tab=readme-ov-file#installation-from-source) to install the SIBR viewers.
2. Run the evaluation as described above and pass the flag `--save_viewer`, and you will see plys saved in outputs/objaverse_splatformer/test/objaverse/viewer (iteration)
3. Run the following command to launch the real-time viewer
```
VIEW_DIR=outputs/objaverse_splatformer/test/objaverse/viewer/0a6e1a80d2e34d5981d6b2b440bbc8cd-10 # Take one scene for example
cd SIBR_viewers/install/shaders/core
../../bin/SIBR_gaussianViewer_app \
    -m $VIEW_DIR --load_iteration iteration_1 
    # iteration_1 is the SplatFormer's output; iteration_0 is the original 3DGS input
```

## Citation
If you find our work helpful, please consider citing:
```bibtex
@inproceedings{chen2024splatformer,
title = {SplatFormer: Point Transformer for Robust 3D Gaussian Splatting},
author = {Chen, Yutong and Mihajlovic, Marko and Chen, Xiyi and Wang, Yiming and Prokudin, Sergey and Tang, Siyu},
booktitle = {International Conference on Learning Representations (ICLR)},
year = {2025}
}
```

## LICENSE
The objects from [objaverse-v1](https://huggingface.co/datasets/allenai/objaverse) we use for training and test are all licensed as creative commons distributable objects. The [Google Scanned Objects (GSO)](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research) dataset is under the CC-BY 4.0 License. Please refer to their websites for more details.
