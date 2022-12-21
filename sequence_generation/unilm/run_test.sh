#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../..`;
export PYTHONPATH=$HOME_DIR
# folder used to cache package dependencies
CACHE_DIR=~/.cache/torch/transformers
MODEL_DIR=${HOME_DIR}/models


# change here for inference 
export CUDA_VISIBLE_DEVICES=$1
MODEL=$2  # bert-base scibert roberta
DATASET=$3  # kp20k kptimes stackex openkp
SAVE_DIR=$4
CKPT_NAME=$5
BSZ=${6:-256}


function decode () {

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

EVAL_DATASET=$1
SPLIT=test
INPUT_FILE=${DATA_DIR_PREFIX}/${EVAL_DATASET}/json/${SPLIT}.json

python decode.py \
    --model_type $MODEL_TYPE \
    --tokenizer_name $MODEL_NAME_OR_PATH \
     --fp16 \
    --input_file $INPUT_FILE \
    --split $SPLIT \
    --do_lower_case \
    --model_path $SAVE_DIR/$CKPT_NAME/ \
    --max_seq_length 512 \
    --max_tgt_length 48 \
    --batch_size $BSZ \
    --beam_size 1 \
    --mode s2s \
    --output_file $SAVE_DIR/$CKPT_NAME/$EVAL_DATASET.$SPLIT \
    --workers 60 \
    2>&1 | tee $SAVE_DIR/decoding.log;

}


function evaluate () {

EVAL_DATASET=$1
SPLIT=test
INPUT_FILE=${DATA_DIR_PREFIX}/${EVAL_DATASET}/json/${SPLIT}.json

python -W ignore ${HOME_DIR}/utils/evaluate.py \
    --src_file $INPUT_FILE \
    --pred_file $SAVE_DIR/$CKPT_NAME/$EVAL_DATASET.$SPLIT \
    --file_prefix $SAVE_DIR/$CKPT_NAME/${EVAL_DATASET} \
    --tgt_dir $SAVE_DIR/$CKPT_NAME/ \
    --log_file $EVAL_DATASET \
    --k_list 5 M;

python ${HOME_DIR}/utils/parse_result_to_csv_5M.py $SAVE_DIR/$CKPT_NAME/results_log_${EVAL_DATASET}.txt

}

AVAILABLE_MODEL_CHOICES=(
    unilm1
    unilm2
    minilm
    minilm2-bert-base
    minilm2-bert-large
    minilm2-roberta
    bert-tiny
    bert-mini
    bert-small
    bert-medium
    bert-base
    bert-large
    scibert
    roberta
)


if [[ $MODEL == 'unilm1' ]]; then
    MODEL_TYPE=unilm
    MODEL_NAME_OR_PATH=unilm1-base-cased
elif [[ $MODEL == 'unilm2' ]]; then
    MODEL_TYPE=unilm
    MODEL_NAME_OR_PATH=unilm1.2-base-uncased
elif  [[ $MODEL == 'minilm' ]]; then
    MODEL_TYPE=minilm
    MODEL_NAME_OR_PATH=minilm-l12-h384-uncased
elif  [[ $MODEL == 'bert-tiny' ]]; then
    MODEL_TYPE=xbert
    MODEL_NAME_OR_PATH=bert-tiny-uncased
    PER_GPU_TRAIN_BATCH_SIZE=32
elif  [[ $MODEL == 'bert-base' ]]; then
    MODEL_TYPE=bert
    MODEL_NAME_OR_PATH=bert-base-uncased
elif  [[ $MODEL == 'bert-large' ]]; then
    MODEL_TYPE=bert
    MODEL_NAME_OR_PATH=bert-large-uncased
    PER_GPU_TRAIN_BATCH_SIZE=4
    LR=3e-5
    NUM_WARM_STEPS=1000
    NUM_TRAIN_STEPS=20000
elif  [[ $MODEL == 'scibert' ]]; then
    MODEL_TYPE=xbert
    MODEL_NAME_OR_PATH=scibert_scivocab_uncased
elif  [[ $MODEL == 'roberta' ]]; then
    MODEL_TYPE=roberta
    MODEL_NAME_OR_PATH=roberta-base
elif  [[ $MODEL == 'news_roberta' ]]; then
    MODEL_TYPE=xbert
    MODEL_NAME_OR_PATH=news_roberta_base
elif  [[ $MODEL == 'minilm2-bert-base' ]]; then
    MODEL_TYPE=$MODEL
    MODEL_NAME_OR_PATH=${MODEL_DIR}/MiniLM-L6-H768-distilled-from-BERT-Base
elif  [[ $MODEL == 'minilm2-bert-large' ]]; then
    MODEL_TYPE=$MODEL
    MODEL_NAME_OR_PATH=${MODEL_DIR}/MiniLM-L6-H768-distilled-from-BERT-Large
elif  [[ $MODEL == 'minilm2-roberta' ]]; then
    MODEL_TYPE=$MODEL
    MODEL_NAME_OR_PATH=${MODEL_DIR}/MiniLM-L6-H768-distilled-from-RoBERTa-Large
else
    echo -n "... Wrong model choice!! available choices: [$(IFS=\| ; echo "${AVAILABLE_MODEL_CHOICES[*]}")]";
    exit 1
fi


if [[ $DATASET == 'kp20k' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/scikp
    for dataset in kp20k inspec krapivin nus semeval; do
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
