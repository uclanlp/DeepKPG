#!/usr/bin/env bash

function download_data () {

FILE=StackEx.zip
if [[ ! -f $FILE ]]; then
    fileid="15UG9MieOTwXEHLhBNwgCrQUPVpyVxi2H"
    baseurl="https://drive.google.com/uc?export=download"
    curl -c ./cookie -s -L "${baseurl}&id=${fileid}" > /dev/null
    curl -Lb ./cookie "${baseurl}&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
    unzip ${FILE} && rm ./cookie && rm -rf __MACOSX
    echo "Aggregating statistics of the raw StackEx dataset"
    python ../data_stat.py \
        -choice 'raw' \
        -train_file stackexchange_train.json \
        -valid_file stackexchange_valid.json \
        -test_file stackexchange_test.json
fi

}

function prepare () {

SRC_DIR=../..
OUTDIR=$2

if [[ ! -d $OUTDIR ]]; then
    echo "============Processing StackEx dataset============"
    PYTHONPATH=$SRC_DIR python -W ignore ../prepare.py \
        -dataset StackEx \
        -data_dir . \
        -out_dir $OUTDIR \
        -tokenizer $1 \
        -workers 60
    echo "Aggregating statistics of the processed StackEx dataset"
    python ../data_stat.py \
        -choice 'processed' \
        -train_file $OUTDIR/train.json \
        -valid_file $OUTDIR/valid.json \
        -test_file $OUTDIR/test.json
fi

}

function fairseq_prepare () {

outdir=fairseq
mkdir -p $outdir
for split in train valid test; do
    PYTHONPATH=$SRC_DIR python -W ignore ../format.py \
        -input_json processed/${split}.json \
        -output_source $outdir/${split}.source \
        -output_target $outdir/${split}.target;
done

}

function json_prepare () {

outdir=json
mkdir -p $outdir
for split in train valid test; do
    PYTHONPATH=$SRC_DIR python -W ignore ../format.py \
        -format json \
        -input_json processed/${split}.json \
        -output_source $outdir/${split}.json;
done

}


download_data
prepare WhiteSpace processed
fairseq_prepare
json_prepare
