#!/usr/bin/env bash

SRC_DIR=/local/wasiahmad/workspace/projects/NeuralKpGen/data/oagk/oagkx
TGT_DIR=/local/wasiahmad/workspace/projects/NeuralKpGen/data/oagk/oagkx_processed
mkdir -p $TGT_DIR

python preprocess.py -data_dir $SRC_DIR -out_dir $TGT_DIR;
