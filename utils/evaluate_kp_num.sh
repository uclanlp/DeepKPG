#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;

SCRIPT_DIR="$(dirname "$0")"
HOME_DIR=`realpath "${SCRIPT_DIR}/.."`


DATASET=$1
OUT_DIR=$2


function eval-kp-num () {
    EVAL_DS=$1
    RAW_FILE=${DATA_DIR_PREFIX}/${EVAL_DS}/processed/test.json

    python ${SCRIPT_DIR}/evaluate_kp_num.py \
        --dataset ${EVAL_DS} \
	--pred_file ${OUT_DIR}/${EVAL_DS}_predictions.txt \
	--raw_file ${DATA_DIR_PREFIX}/${EVAL_DS}/processed/test.json \
        --out_dir ${OUT_DIR} \
	--metric mse
}


if [[ $DATASET == 'scikp' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/scikp
    for dataset in kp20k inspec krapivin nus semeval; do
	    eval-kp-num $dataset
    done
else
    DATA_DIR_PREFIX=${HOME_DIR}/data
    eval-kp-num $DATASET
fi

