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
