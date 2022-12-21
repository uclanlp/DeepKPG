#!/usr/bin/env bash

CURRENT_DIR=$PWD

conda create --name deepkpg-gen python==3.8
conda activate deepkpg-gen
#conda config --add channels conda-forge
#conda config --add channels pytorch

#conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
#conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt

git clone https://github.com/NVIDIA/apex
cd apex
export CXX=g++
export CUDA_HOME=/usr/local/cuda-11.4
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd $CURRENT_DIR
