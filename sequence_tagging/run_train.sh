#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;

# for tiny, mini, small, medium sized BERT - check
# https://huggingface.co/models?search=google%2Fbert_uncased_L-
AVAILABLE_MODEL_CHOICES=(
    unilm
    minilm
    bert-tiny
    bert-mini
    bert-small
    bert-medium
    bert-base
    scibert
    roberta
    bart
)

export CUDA_VISIBLE_DEVICES=0

# data parameters
DATASET=kp20k

# model parameters
MODEL_NAME=bert-base  # bert-base scibert roberta newsbert ...
CRF_FLAG=false  # true false

# training parameters
LR=1e-5   # 1e-5, 3e-5, 6e-5, 1e-4, 3e-4
LR_SCHEDULE=linear   # linear, polynomial, constant
WEIGHT_DECAY=0.0
WARMUP=1000
BSZ=8
GRADACC=4
SEED=2022


if [[ $CRF_FLAG == true ]]; then
    USE_CRF="--use_crf"; SUFFIX="_crf";
else
    USE_CRF=""; SUFFIX="";
fi

export MODEL_TYPE=bert


function train () {

python $CURRENT_DIR/source/run_tag.py $USE_CRF \
    --data_dir $DATA_DIR_PREFIX/${DATASET}/bioformat \
    --dataset_name ${DATASET} \
    --do_lower_case \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 512 \
    --num_train_epochs 5 \
    --warmup_steps ${WARMUP} \
    --per_gpu_train_batch_size ${BSZ} \
    --per_gpu_eval_batch_size ${BSZ} \
    --gradient_accumulation_steps ${GRADACC} \
    --save_steps ${SAVESTEP} \
    --seed ${SEED} \
    --learning_rate ${LR} \
    --learning_rate_schedule ${LR_SCHEDULE} \
    --weight_decay ${WEIGHT_DECAY} \
    --do_train \
    --do_eval \
    --log_file $OUTPUT_DIR/train.log \
    --overwrite_output_dir \
    --save_only_best_checkpoint \
    --workers 60;

}


function evaluate () {

EVAL_DATASET=$1

python $CURRENT_DIR/source/run_tag.py $USE_CRF \
    --data_dir $DATA_DIR_PREFIX/$EVAL_DATASET/bioformat \
    --dataset_name $EVAL_DATASET \
    --do_lower_case \
    --model_type $MODEL_TYPE \
    --model_name_or_path $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size 64 \
    --seed ${SEED} \
    --do_predict \
    --log_file $OUTPUT_DIR/$1-eval.log \
    --workers 60;

}


function compute_score () {

EVAL_DATASET=$1

python -W ignore ${HOME_DIR}/utils/evaluate.py \
    --src_dir ${DATA_DIR_PREFIX}/$EVAL_DATASET/bioformat \
    --file_prefix $OUTPUT_DIR/$EVAL_DATASET \
    --tgt_dir $OUTPUT_DIR \
    --log_file $EVAL_DATASET \
    --k_list 5 10 M;

}


if [[ $MODEL_NAME == 'unilm' ]]; then
    export BERT_MODEL=unilm-base-cased
elif  [[ $MODEL_NAME == 'minilm' ]]; then
    export BERT_MODEL=microsoft/MiniLM-L12-H384-uncased
elif  [[ $MODEL_NAME == 'bert-tiny' ]]; then
    export BERT_MODEL=google/bert_uncased_L-2_H-128_A-2
elif  [[ $MODEL_NAME == 'bert-mini' ]]; then
    export BERT_MODEL=google/bert_uncased_L-4_H-256_A-4
elif  [[ $MODEL_NAME == 'bert-small' ]]; then
    export BERT_MODEL=google/bert_uncased_L-4_H-512_A-8
elif  [[ $MODEL_NAME == 'bert-medium' ]]; then
    export BERT_MODEL=google/bert_uncased_L-8_H-512_A-8
elif  [[ $MODEL_NAME == 'bert-base' ]]; then
    export BERT_MODEL=bert-base-uncased
elif  [[ $MODEL_NAME == 'newsbert' ]]; then
    export BERT_MODEL=${HOME_DIR}/newsbert/checkpoints/news-bert-base
elif  [[ $MODEL_NAME == 'scibert' ]]; then
    export BERT_MODEL=allenai/scibert_scivocab_uncased
elif  [[ $MODEL_NAME == 'roberta' ]]; then
    export MODEL_TYPE=roberta
    export BERT_MODEL=roberta-base
elif  [[ $MODEL_NAME == 'bart' ]]; then
    export MODEL_TYPE=bart
    export BERT_MODEL=facebook/bart-base
else
    echo -n "... Wrong model choice!! available choices: \
                [$(IFS=\| ; echo "${AVAILABLE_MODEL_CHOICES[*]}")]" ;
    exit 1
fi

if [[ $BERT_MODEL == 'unilm-base-cased' ]]; then
    MODEL_NAME_OR_PATH=${CURRENT_DIR}/models/${BERT_MODEL}
fi


export OUTPUT_DIR=${HOME_DIR}/lm_kpgen_experiments/seq_tagging/$(date +'%Y%m%d-%H%M')-${DATASET}-${MODEL_NAME}_CRF${CRF_FLAG}_seed${SEED}_LR${LR}_lrschedule${LR_SCHEDULE}_weightdecay${WEIGHT_DECAY}_WARMUP${WARMUP}_BSZ${BSZ}x${GRADACC}
export MODEL_NAME_OR_PATH=$BERT_MODEL
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/code_backup
cp -r source *.sh ${OUTPUT_DIR}/code_backup


if [[ $DATASET == 'kp20k' ]] ; then
    # SAVESTEP is per DSSIZE / BSZ / GRADACC (i.e., actual # of weight updates)
    SAVESTEP=2000
    DATA_DIR_PREFIX=${HOME_DIR}/data/scikp
    train
elif [[ $DATASET =~ ^(kpbiomed-small|kpbiomed-medium|kpbiomed-large)$ ]]; then
    SAVESTEP=2000
    DATA_DIR_PREFIX=${HOME_DIR}/data/kpbiomed
    train
else
    SAVESTEP=2000
    DATA_DIR_PREFIX="${HOME_DIR}/data"
    train
fi
