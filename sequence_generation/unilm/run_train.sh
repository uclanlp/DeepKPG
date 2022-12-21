#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;

CURRENT_DIR=`pwd`
HOME_DIR=`realpath ../..`;
# folder used to cache package dependencies
CACHE_DIR=~/.cache/torch/transformers
MODEL_DIR=${HOME_DIR}/models

export PYTHONPATH=$HOME_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL=bert-base   # scibert biobert roberta bert-base newsbert

DATASET=kp20k

PER_GPU_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
LR=1e-4
NUM_WARM_STEPS=2000
NUM_TRAIN_STEPS=20000
SEED=1234


function train () {

python train.py \
    --seed ${SEED} \
    --data_dir ${DATA_DIR_PREFIX}/${DATASET}/json/ \
    --train_file ${DATA_DIR_PREFIX}/${DATASET}/json/train.json \
    --validation_file ${DATA_DIR_PREFIX}/${DATASET}/json/valid.json \
    --output_dir $SAVE_DIR \
    --log_dir $SAVE_DIR \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --logging_steps 250 \
    --save_steps 1000 \
    --save_limit 3 \
    --do_lower_case \
    --max_source_seq_length 464 \
    --max_target_seq_length 48 \
    --random_prob 0.1  --keep_prob 0.1  \
    --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
    --valid_batch_size 32 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LR \
    --num_warmup_steps $NUM_WARM_STEPS \
    --num_training_steps $NUM_TRAIN_STEPS \
    --fp16 --fp16_opt_level O1 \
    --cache_dir $CACHE_DIR \
    --workers 60 \
    2>&1 | tee ${SAVE_DIR}/finetune.log;

}

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
    --model_path $SAVE_DIR/$CKPT_NAME \
    --max_seq_length 512 \
    --max_tgt_length 48 \
    --batch_size 256 \
    --beam_size 1 \
    --mode s2s \
    --output_file $SAVE_DIR/$CKPT_NAME.$EVAL_DATASET.$SPLIT \
    --workers 60 \
    2>&1 | tee $SAVE_DIR/decoding.log;

}


function evaluate () {

EVAL_DATASET=$1
SPLIT=test
INPUT_FILE=${DATA_DIR_PREFIX}/${EVAL_DATASET}/json/${SPLIT}.json

python -W ignore ${HOME_DIR}/utils/evaluate.py \
    --src_file $INPUT_FILE \
    --pred_file $SAVE_DIR/$CKPT_NAME.$EVAL_DATASET.$SPLIT \
    --file_prefix $SAVE_DIR/${EVAL_DATASET} \
    --tgt_dir $SAVE_DIR \
    --log_file $EVAL_DATASET \
    --k_list 5 M;

}

AVAILABLE_MODEL_CHOICES=(
    unilm1
    unilm2
    minilm
    minilm2-bert-base
    minilm2-bert-large
    minilm2-roberta
    bert-base
    bert-large
    scibert
    roberta
)

SAVE_DIR=${HOME_DIR}/lm_kpgen_experiments/unilm/$(date +'%Y%m%d-%H%M')_${DATASET}_checkpoints_${MODEL}_gpu${CUDA_VISIBLE_DEVICES}_pergpubsz${PER_GPU_TRAIN_BATCH_SIZE}x${GRADIENT_ACCUMULATION_STEPS}_LR${LR}_WARMUP${NUM_WARM_STEPS}_seed${SEED}
mkdir -p $SAVE_DIR/code_backup
cp *.py *.sh ${SAVE_DIR}/code_backup   


if [[ $DATASET == 'kp20k' ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/scikp
elif [[ $DATASET =~ ^(kpbiomed-small|kpbiomed-medium|kpbiomed-large)$ ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/kpbiomed/
else
    DATA_DIR_PREFIX=${HOME_DIR}/data
fi


if [[ $MODEL == 'unilm1' ]]; then
    rm ${DATA_DIR_PREFIX}/${DATASET}/json/cached_features_for_*_unilm.pt
    MODEL_TYPE=unilm
    MODEL_NAME_OR_PATH=unilm1-base-cased
    CKPT_NAME=ckpt-20000
elif [[ $MODEL == 'unilm2' ]]; then
    rm ${DATA_DIR_PREFIX}/${DATASET}/json/cached_features_for_*_unilm.pt
    MODEL_TYPE=unilm
    MODEL_NAME_OR_PATH=unilm1.2-base-uncased
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'minilm' ]]; then
    MODEL_TYPE=minilm
    MODEL_NAME_OR_PATH=minilm-l12-h384-uncased
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'bert-base' ]]; then
    MODEL_TYPE=bert
    MODEL_NAME_OR_PATH=bert-base-uncased
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'newsbert' ]]; then
    MODEL_TYPE=bert
    MODEL_NAME_OR_PATH=${HOME_DIR}/newsbert/checkpoints/news-bert-base
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'biobert' ]]; then
    rm ${DATA_DIR_PREFIX}/${DATASET}/json/cached_features_for_*_xbert.pt
    MODEL_TYPE=xbert
    MODEL_NAME_OR_PATH=biobert
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'bert-large' ]]; then
    MODEL_TYPE=bert
    MODEL_NAME_OR_PATH=bert-large-uncased
    CKPT_NAME=ckpt-20000
    PER_GPU_TRAIN_BATCH_SIZE=4
    LR=3e-5
    NUM_WARM_STEPS=1000
    NUM_TRAIN_STEPS=20000
elif  [[ $MODEL == 'scibert' ]]; then
    rm ${DATA_DIR_PREFIX}/${DATASET}/json/cached_features_for_*_xbert.pt
    MODEL_TYPE=xbert
    MODEL_NAME_OR_PATH=scibert_scivocab_uncased
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'roberta' ]]; then
    MODEL_TYPE=roberta
    MODEL_NAME_OR_PATH=roberta-base
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'minilm2-bert-base' ]]; then
    MODEL_TYPE=$MODEL
    MODEL_NAME_OR_PATH=${MODEL_DIR}/MiniLM-L6-H768-distilled-from-BERT-Base
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'minilm2-bert-large' ]]; then
    MODEL_TYPE=$MODEL
    MODEL_NAME_OR_PATH=${MODEL_DIR}/MiniLM-L6-H768-distilled-from-BERT-Large
    CKPT_NAME=ckpt-20000
elif  [[ $MODEL == 'minilm2-roberta' ]]; then
    MODEL_TYPE=$MODEL
    MODEL_NAME_OR_PATH=${MODEL_DIR}/MiniLM-L6-H768-distilled-from-RoBERTa-Large
    CKPT_NAME=ckpt-20000
else
    echo -n "... Wrong model choice!! available choices: [$(IFS=\| ; echo "${AVAILABLE_MODEL_CHOICES[*]}")]";
    exit 1
fi


train
