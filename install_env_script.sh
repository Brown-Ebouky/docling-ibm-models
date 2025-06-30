#!/bin/bash

set -e  # Exit on error

# echo "Creating conda environment..."
# conda env create -f cu124.yaml || conda env update -f cu124.yaml

# Get name of current active conda environment
CURRENT_ENV=$(basename "$CONDA_PREFIX")
echo ">>> Installing system dependencies into current conda environment: $CURRENT_ENV"


conda install -y -c nvidia/label/cuda-12.4.1 cuda-toolkit cuda
conda install -y -c conda-forge gxx=13 gcc=13 libxcrypt poetry

# To avoid torchinductor and modelopt errors looking for crypt.h
conda env config vars set CPATH=$CONDA_PREFIX/include 

echo ">>> Reactivating environment $CURRENT_ENV to apply env var changes..."
echo "Activating environment..."

# Load conda shell functions
eval "$(conda shell.bash hook)"
conda deactivate
conda activate "$CURRENT_ENV"

# other dependencies
pip install torch==2.5.1 # +cu121
pip install torchaudio==2.5.1 # +cu121
pip install torchvision==0.20.1 # +cu121
pip install huggingface-hub

echo "Installing external repos..."

pip install git+https://github.com/dgcnz/detrex.git
pip install git+https://github.com/facebookresearch/detectron2.git

echo "Installing current docling-ibm-models repo"
pip install -e .

echo "Setup complete!"
