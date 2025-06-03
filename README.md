# RIEBench: Representation-based image editing benchmark

A simple benchmark for turning on/off different types of features in SDXL Turbo while generating images.

![image](https://github.com/user-attachments/assets/7cc9d9df-15a4-44da-a325-0b072429ffdf)

![image](https://github.com/user-attachments/assets/eb36211c-68cd-41e0-b14a-4e158c65ee57)



# Installation

1. Create a new environment (recommended)
```
conda create -n riebench python=3.12
conda activate riebench
```

2. Install sdxl-unbox 
```
cd sdxl-unbox
pip install -r requirements.txt
```

3. Install grounded SAM2 (`/path/to/cuda-12.1/` normally is `/usr/local/cuda-12.1`)
```
cd Grounded-SAM-2
export CUDA_HOME=/path/to/cuda-12.1/ 
pip install -e .
pip install --no-build-isolation -e grounding_dino
cd checkpoints
bash download_ckpts.sh
cd ../gdino_checkpoints
bash download_ckpts.sh
```

4. Install remaining requirements (in `RIEBench/`)
```
pip install -r requirements.txt
```

5. Download the $k=160$, $n_f=5120$ SAEs
```
GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:wendlerc/sdxl-turbo-saes
cd sdxl-turbo-saes
git lfs pull --include="\
unet.down_blocks.2.attentions.1_k160_hidden5120_auxk256_bs4096_lr0.0001/*,\
unet.mid_block.attentions.0_k160_hidden5120_auxk256_bs4096_lr0.0001/*,\
unet.up_blocks.0.attentions.0_k160_hidden5120_auxk256_bs4096_lr0.0001/*,\
unet.up_blocks.0.attentions.1_k160_hidden5120_auxk256_bs4096_lr0.0001/*"
```

# Getting started

## Running the interventions

Transporting 80 SAE features with strength 2
```
papermill main.ipynb out/main.ipynb -p k_trans 80 -p m1 2
```

Transporting 10000 neurons with strength 2
```
papermill main.ipynb out/main.ipynb -p k_trans 10000 -p m1 2 -p mode neurons
```

Steering
```
papermill main.ipynb out/main.ipynb -p m1 1 -p mode steering
```

This will result in a bunch of subfolders containing the resulting images in `results`.

## Computing the metrics

Using ... you can compute the LPIPS and CLIP scores for a method/result folder.

# Citation

```
@misc{surkov2025onestepenoughsparseautoencoders,
      title={One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion Models}, 
      author={Viacheslav Surkov and Chris Wendler and Antonio Mari and Mikhail Terekhov and Justin Deschenaux and Robert West and Caglar Gulcehre and David Bau},
      year={2025},
      eprint={2410.22366},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.22366}, 
}
```
