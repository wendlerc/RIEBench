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