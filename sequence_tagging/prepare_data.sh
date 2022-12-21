#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;


# scikp
DATA_DIR=${HOME_DIR}/data/scikp
if [[ ! -d $DATA_DIR/kp20k/bioformat ]]; then
    python -W ignore $HOME_DIR/data/bioConverter.py \
        -dataset KP20k \
        -data_dir $DATA_DIR/kp20k/processed \
        -out_dir $DATA_DIR/kp20k/bioformat \
        -max_src_len 510 \
        -workers 60;
fi
for dataset in inspec nus krapivin semeval; do
    if [[ ! -d $DATA_DIR/$dataset/bioformat ]]; then
        python -W ignore $HOME_DIR/data/bioConverter.py \
            -dataset $dataset \
            -data_dir $DATA_DIR/$dataset/processed \
            -out_dir $DATA_DIR/$dataset/bioformat \
            -max_src_len 510 \
            -workers 60;
    fi
done


# kpbiomed
DATA_DIR=${HOME_DIR}/data/kpbiomed
for dataset in kpbiomed-small kpbiomed-medium kpbiomed-large; do
    if [[ ! -d $DATA_DIR/$dataset/bioformat ]]; then
        python -W ignore $HOME_DIR/data/bioConverter.py \
            -dataset $dataset \
            -data_dir $DATA_DIR/$dataset/processed \
            -out_dir $DATA_DIR/$dataset/bioformat \
            -max_src_len 510 \
            -workers 60;
    fi
done


# KPTimes
DATA_DIR=${HOME_DIR}/data/
if [[ ! -d $DATA_DIR/kptimes/bioformat ]]; then
    python -W ignore $HOME_DIR/data/bioConverter.py \
        -dataset KPTimes \
        -data_dir $DATA_DIR/kptimes/processed \
        -out_dir $DATA_DIR/kptimes/bioformat \
        -max_src_len 510 \
        -workers 60;
fi


# OpenKP
DATA_DIR=${HOME_DIR}/data/
if [[ ! -d $DATA_DIR/openkp/bioformat ]]; then
    python -W ignore $HOME_DIR/data/bioConverter.py \
        -dataset OpenKP \
        -data_dir $DATA_DIR/openkp/processed \
        -out_dir $DATA_DIR/openkp/bioformat \
        -max_src_len 510 \
        -workers 60;
fi


# StackEx
DATA_DIR=${HOME_DIR}/data/
if [[ ! -d $DATA_DIR/stackex/bioformat ]]; then
    python -W ignore $HOME_DIR/data/bioConverter.py \
        -dataset StackEx \
        -data_dir $DATA_DIR/stackex/processed \
        -out_dir $DATA_DIR/stackex/bioformat \
        -max_src_len 510 \
        -workers 60;
fi
