#!/usr/bin/env bash

SRC_DIR=`realpath ../..`;


function download_data () {

FILE=kp_datasets.zip
if [[ ! -f $FILE ]]; then
    # https://drive.google.com/open?id=1DbXV1mZXm_o9bgfwPV9PV0ZPcNo1cnLp
    fileid="1DbXV1mZXm_o9bgfwPV9PV0ZPcNo1cnLp"
    baseurl="https://drive.google.com/uc?export=download"
    curl -c ./cookie -s -L "${baseurl}&id=${fileid}" > /dev/null
    curl -Lb ./cookie "${baseurl}&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    unzip ${FILE} && rm ./cookie
    rm -rf kp20k_sorted && rm -rf cross_domain_sorted
fi

}


function prepare () {

OUTDIR=$2

if [[ ! -d $OUTDIR ]]; then
    echo "============Processing KP20k dataset============"
    PYTHONPATH=$SRC_DIR python -W ignore ../prepare.py \
        -dataset KP20k \
        -data_dir kp20k_separated \
        -out_dir kp20k/${OUTDIR} \
        -tokenizer $1 \
        -workers 60
    echo "Aggregating statistics of the processed KP20k dataset"
    python ../data_stat.py \
        -choice 'processed' \
        -train_file kp20k/${OUTDIR}/train.json \
        -valid_file kp20k/${OUTDIR}/valid.json \
        -test_file kp20k/${OUTDIR}/test.json
    echo "============Processing Cross-Domain datasets============"
    for dataset in inspec nus krapivin semeval; do
        PYTHONPATH=$SRC_DIR python -W ignore ../prepare.py \
            -dataset $dataset \
            -data_dir cross_domain_separated \
            -out_dir ${dataset}/${OUTDIR} \
            -tokenizer $1 \
            -workers 60
    done
fi

}


function fairseq_prepare () {

outdir=kp20k/fairseq
mkdir -p $outdir
for split in train valid test; do
    PYTHONPATH=$SRC_DIR python -W ignore ../format.py \
        -input_json kp20k/processed/${split}.json \
        -output_source $outdir/${split}.source \
        -output_target $outdir/${split}.target;
done
for dataset in inspec nus krapivin semeval; do
   outdir=${dataset}/fairseq
   mkdir -p $outdir
   PYTHONPATH=$SRC_DIR python -W ignore ../format.py \
       -input_json ${dataset}/processed/test.json \
       -output_source $outdir/test.source \
       -output_target $outdir/test.target;
done

}


function json_prepare () {

outdir=kp20k/json
mkdir -p $outdir
for split in train valid test; do
    PYTHONPATH=$SRC_DIR python -W ignore ../format.py \
        -format json \
        -input_json kp20k/processed/${split}.json \
        -output_source $outdir/${split}.json;
done
for dataset in inspec nus krapivin semeval; do
    outdir=${dataset}/json
    mkdir -p $outdir
    PYTHONPATH=$SRC_DIR python -W ignore ../format.py \
        -format json \
        -input_json ${dataset}/processed/test.json \
        -output_source $outdir/test.json;
done

}


download_data
prepare WhiteSpace processed
fairseq_prepare
json_prepare
