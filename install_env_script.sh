#!/bin/bash

set -e  # Exit on error

# echo "Creating conda environment..."
# conda env create -f cu124.yaml || conda env update -f cu124.yaml


echo ">>> Installing system dependencies into current conda environment..."

conda install -y -c nvidia/label/cuda-12.4.1 cuda-toolkit cuda
conda install -y -c conda-forge gxx=13 gcc=13 libxcrypt poetry

# To avoid torchinductor and modelopt errors looking for crypt.h
conda env config vars set CPATH=$CONDA_PREFIX/include 

echo "Activating environment..."

# other dependencies
pip install torch==2.5.1 # +cu121
pip install torchaudio==2.5.1 # +cu121
pip install torchvision==0.20.1 # +cu121
pip install huggingface-hub

echo "Installing external repos..."

pip install git+https://github.com/dgcnz/detrex.git
pip install git+https://github.com/facebookresearch/detectron2.git


echo "Setup complete!"
