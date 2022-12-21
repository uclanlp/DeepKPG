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


export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
OUTPUT_DIR=$3
MODEL_NAME=${4:-bert-base}
CRF_FLAG=${5:-false}

if [[ $CRF_FLAG == true ]]; then
    USE_CRF="--use_crf"; SUFFIX="_crf";
else
    USE_CRF=""; SUFFIX="";
fi

SEED=1234

export MODEL_TYPE=bert


function evaluate () {

EVAL_DATASET=$1

python $CURRENT_DIR/source/run_tag.py $USE_CRF \
    --data_dir $DATA_DIR_PREFIX/$EVAL_DATASET/bioformat \
    --dataset_name $EVAL_DATASET \
    --do_lower_case \
    --model_type $MODEL_TYPE \
    --model_name_or_path $OUTPUT_DIR \
    --tokenizer_name ${BERT_MODEL} \
    --output_dir $OUTPUT_DIR \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size 32 \
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

export MODEL_NAME_OR_PATH=$BERT_MODEL


if [[ $DATASET == 'kp20k' ]] ; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/scikp
    for dataset in kp20k krapivin nus inspec semeval; do
        # remove files cached by other runs
        rm ${DATA_DIR_PREFIX}/${dataset}/bioformat/cached_test_${dataset}*        
        evaluate $dataset
        compute_score $dataset
	python ${HOME_DIR}/utils/parse_result_to_csv_510M.py ${OUTPUT_DIR}/results_log_${dataset}.txt
    done
elif [[ $DATASET =~ ^(kpbiomed-small|kpbiomed-medium|kpbiomed-large)$ ]]; then
    DATA_DIR_PREFIX=${HOME_DIR}/data/kpbiomed
    rm ${DATA_DIR_PREFIX}/${DATASET}/bioformat/cached_test_${dataset}*
    evaluate $DATASET
    compute_score $DATASET
    python ${HOME_DIR}/utils/parse_result_to_csv_510M.py ${OUTPUT_DIR}/results_log_${DATASET}.txt
else
    DATA_DIR_PREFIX="${HOME_DIR}/data"
    rm ${DATA_DIR_PREFIX}/${DATASET}/bioformat/cached_test_${dataset}*
    evaluate $DATASET
    compute_score $DATASET
    python ${HOME_DIR}/utils/parse_result_to_csv_510M.py ${OUTPUT_DIR}/results_log_${DATASET}.txt
fi
