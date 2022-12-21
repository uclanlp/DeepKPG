#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
HOME_DIR=`realpath ../..`


# Change GPUs, model, and dataset before training
export CUDA_VISIBLE_DEVICES=0,1,2,3
NGPUS=4

# note: use --random_init_encoder or --random_init_decoder for RND2BERTor BERT2RND
ENCODER_NAME="google/bert_uncased_L-8_H-768_A-12"
ENC_NAME_SHORT="BERT-8-768-12"
DECODER_NAME="google/bert_uncased_L-4_H-768_A-12"
DEC_NAME_SHORT="BERT-4-768-12"


DATASET=kp20k


function train () {
    SEED=1234    # 1234 42 2022
    BATCH_SIZE_PER_GPU=4
    GRAD_ACCUMULATION_STEPS=2
    N_EPOCHS=20
    MAX_STEPS=120000
    N_WARMUP_STEPS=2000
    N_EVAL_STEPS=2000
    LR=5e-5
    LR_SCHEDULE='linear'
    DROPOUT=0.0 
    ATTN_DROPOUT=0.0
    # N_WORKERS=16    # moderate as needed

    OUT_DIR="${HOME_DIR}/lm_kpgen_experiments/bert2bert/$(date +'%Y%m%d-%H%M')_${DATASET}_checkpoints_${ENC_NAME_SHORT}+${DEC_NAME_SHORT}_device${CUDA_VISIBLE_DEVICES}_pergpubsz${BATCH_SIZE_PER_GPU}x${GRAD_ACCUMULATION_STEPS}_lr${LR}_${LR_SCHEDULE}_seed${SEED}"
    mkdir -p ${OUT_DIR}/code_backup
    cp *.py *.sh ${OUT_DIR}/code_backup

    #python run_finetune_kpgen_bert2bert_hf4.2.1.py \
    accelerate launch --num_processes ${NGPUS} 	--mixed_precision fp16 \
	run_finetune_kpgen_bert2bert_hf4.2.1.py \
        --output_dir $OUT_DIR \
        --encoder ${ENCODER_NAME} \
	--decoder ${DECODER_NAME} \
        --max_pred_length 80 \
        --num_beams 1 \
        --train_file "${DATA_DIR_PREFIX}/${DATASET}/json/train.json" \
        --validation_file "${DATA_DIR_PREFIX}/${DATASET}/json/valid.json" \
        --src_column "src" \
        --tgt_column "tgt" \
        --num_train_epochs ${N_EPOCHS} \
        --learning_rate ${LR} \
        --lr_scheduler_type ${LR_SCHEDULE} \
        --warmup_steps ${N_WARMUP_STEPS} \
	    --max_steps ${MAX_STEPS} \
        --per_device_train_batch_size ${BATCH_SIZE_PER_GPU} \
        --per_device_eval_batch_size ${BATCH_SIZE_PER_GPU} \
        --gradient_accumulation_steps ${GRAD_ACCUMULATION_STEPS} \
        --dropout ${DROPOUT} \
        --attention_dropout ${ATTN_DROPOUT} \
        --logging_steps 500 \
        --evaluation_strategy 'steps' \
        --eval_steps ${N_EVAL_STEPS} \
        --save_steps ${N_EVAL_STEPS} \
        --save_total_limit 10 \
        --predict_with_generate \
        --overwrite_output_dir \
        --seed ${SEED} \
        --fp16 true \

}


if [[ $DATASET == 'kp20k' ]] ; then
    DATA_DIR_PREFIX="${HOME_DIR}/data/scikp"
    train
elif [[ $DATASET =~ ^(kpbiomed-small|kpbiomed-medium|kpbiomed-large)$ ]]; then
    DATA_DIR_PREFIX="${HOME_DIR}/data/kpbiomed"
    train
else
    DATA_DIR_PREFIX="${HOME_DIR}/data"
    train
fi
