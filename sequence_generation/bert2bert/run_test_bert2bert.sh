#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
HOME_DIR=`realpath ../..`


# GPU, dataset, and model dir for decoding
export CUDA_VISIBLE_DEVICES=$1
DATASET=$2  
MODEL_DIR=$3 


function decode () {
    # use ${MODEL_DIR} to ensure using the possibly changed tokenizer (e.g., expanded with [digit])
    TOKENIZER=${MODEL_DIR}    # "facebook/bart-base"
    EVAL_DATASET=$1
    BATCH_SIZE_PER_GPU=32
    MAX_PRED_LENGTH=100

    python run_decode_kpgen_bert2bert_hf4.2.1.py \
        --model_name_or_path $MODEL_DIR \
        --test_file "${DATA_DIR_PREFIX}/${EVAL_DATASET}/json/test.json" \
        --src_column "src" \
        --tgt_column "tgt" \
        --eval_batch_size $BATCH_SIZE_PER_GPU \
        --num_beams 1 \
        --max_pred_length ${MAX_PRED_LENGTH} \
        --output_dir $MODEL_DIR \
        --output_file_name "$MODEL_DIR/${EVAL_DATASET}_hypotheses.txt" 

}


function evaluate () {
    EVAL_DATASET=$1

    python -W ignore ${HOME_DIR}/utils/evaluate.py \
        --src_dir ${DATA_DIR_PREFIX}/${EVAL_DATASET}/fairseq \
        --file_prefix $MODEL_DIR/${EVAL_DATASET} \
        --tgt_dir $MODEL_DIR \
        --log_file $EVAL_DATASET \
        --k_list 5 M;

    python ${HOME_DIR}/utils/parse_result_to_csv_5M.py ${MODEL_DIR}/results_log_${EVAL_DATASET}.txt
}


if [[ $DATASET == 'kp20k' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/scikp
    for dataset in kp20k inspec krapivin nus semeval; do
	#for dataset in semeval; do
        decode $dataset
        evaluate $dataset
    done
elif [[ $DATASET =~ ^(kpbiomed-small|kpbiomed-medium|kpbiomed-large)$ ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/kpbiomed
    decode $DATASET
    evaluate $DATASET
else
    DATA_DIR_PREFIX=${HOME_DIR}/data
    decode $DATASET
    evaluate $DATASET
fi
