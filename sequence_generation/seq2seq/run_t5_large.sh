#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
HOME_DIR=`realpath ../..`


# Confirm GPUs, model, and dataset before training
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_NAME_OR_PATH="t5-large"
MODEL_NAME_SHORT=$MODEL_NAME_OR_PATH      # for creating the output dir

DATASET=kp20k


function train () {
    BATCH_SIZE_PER_GPU=2
    GRAD_ACCUMULATION_STEPS=8
    N_EPOCHS=5
    N_WARMUP_STEPS=100
    N_EVAL_STEPS=5000
    LR=6e-5
    LR_schedule='polynomial'    # 'linear'
    SEED=1234
    
    OUT_DIR=${HOME_DIR}/lm_kpgen_experiments/${MODEL_NAME_SHORT}/$(date +'%Y%m%d-%H%M')_${DATASET}_checkpoints_${MODEL_NAME_SHORT}_lr${LR}_${LR_schedule}_seed${SEED}
    mkdir -p ${OUT_DIR}/code_backup
    cp *.py *.sh ${OUT_DIR}/code_backup


    # give the argument --num_gpus=X to deepspeed if we don't use CUDA_VISIBLE_DEVICES   
    deepspeed run_finetune_kpgen_seq2seq_hf.py \
        --output_dir $OUT_DIR \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --do_train \
        --do_eval \
        --train_file "${DATA_DIR_PREFIX}/${DATASET}/json/train.json" \
        --validation_file "${DATA_DIR_PREFIX}/${DATASET}/json/valid.json" \
        --src_column "src" \
        --tgt_column "tgt" \
	--source_prefix "generate keyphrases: " \
        --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
        --per_device_eval_batch_size ${BATCH_SIZE_PER_GPU} \
        --gradient_accumulation_steps ${GRAD_ACCUMULATION_STEPS} \
        --num_train_epochs ${N_EPOCHS} \
        --learning_rate ${LR} \
        --lr_scheduler_type ${LR_schedule} \
        --warmup_steps ${N_WARMUP_STEPS} \
        --logging_strategy 'steps' \
        --logging_steps 500 \
        --evaluation_strategy 'steps' \
        --eval_steps ${N_EVAL_STEPS} \
        --save_strategy 'steps' \
        --save_steps ${N_EVAL_STEPS} \
        --save_total_limit 1 \
        --load_best_model_at_end true \
        --overwrite_output_dir \
        --predict_with_generate \
        --seed ${SEED} \
        --deepspeed "../deepspeed_configs/normal_fp16_gpuonly.json" \
        --fp16 true \
        --half_precision_backend "auto"

}


function decode () {
    EVAL_DATASET=$1
    BATCH_SIZE_PER_GPU=32

    python run_decode_kpgen_seq2seq_hf.py \
        --model_name_or_path $OUT_DIR \
        --tokenizer_name $OUT_DIR \
        --test_file "${DATA_DIR_PREFIX}/${EVAL_DATASET}/json/test.json" \
        --src_column "src" \
        --tgt_column "tgt" \
	--source_prefix "generate keyphrases: " \
        --num_beams 1 \
        --per_device_eval_batch_size $BATCH_SIZE_PER_GPU \
        --output_dir $OUT_DIR \
        --output_file_name "$OUT_DIR/${EVAL_DATASET}_hypotheses.txt" 

}


function evaluate () {
    EVAL_DATASET=$1

    python -W ignore ${HOME_DIR}/utils/evaluate.py \
        --src_dir ${DATA_DIR_PREFIX}/${EVAL_DATASET}/fairseq \
        --file_prefix $OUT_DIR/${EVAL_DATASET} \
        --tgt_dir $OUT_DIR \
        --log_file $EVAL_DATASET \
        --k_list 5 M;
}


if [[ $DATASET == 'kp20k' ]] ; then
    DATA_DIR_PREFIX="${HOME_DIR}/data/scikp"
    train
    for dataset in kp20k inspec krapivin nus semeval; do
        decode $dataset
        evaluate $dataset
    done	
elif [[ $DATASET =~ ^(kpbiomed-small|kpbiomed-medium|kpbiomed-large)$ ]]; then
    DATA_DIR_PREFIX="${HOME_DIR}/data/kpbiomed"
    train
    decode $DATASET
    evaluate $DATASET
else
    DATA_DIR_PREFIX="${HOME_DIR}/data"
    train
    decode $DATASET
    evaluate $DATASET
fi
