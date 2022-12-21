#!/usr/bin/env bash

function download_oagk () {
    FILE=OAGK.zip
    if [[ ! -f "$FILE" ]]; then
        baseurl=https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2943
        curl --remote-name-all ${baseurl}/{${FILE},README.txt}
        unzip ${FILE}
        echo "Aggregating statistics of the raw OAGK dataset"
        python ../data_stat.py \
            -choice 'raw' \
            -train_file OAGK/oagk_train.txt \
            -valid_file OAGK/oagk_val.txt \
            -test_file OAGK/oagk_test.txt;
    fi
}

function download_oagkx () {
    FILE=oagkx.zip
    if [[ ! -f "$FILE" ]]; then
        baseurl=https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3062
        curl --remote-name-all ${baseurl}/{${FILE},README.txt}
        unzip ${FILE}
    fi
}

function prepare () {

SRC_DIR=../..
OUTDIR=$2

if [[ ! -d $OUTDIR ]]; then
    echo "============Processing OAGK dataset============"
    PYTHONPATH=$SRC_DIR python -W ignore ../prepare.py \
        -dataset OAGK \
        -data_dir OAGK \
        -out_dir $OUTDIR \
        -tokenizer $1 \
        -workers 60;
    echo "Aggregating statistics of the processed OAGK dataset"
    python ../data_stat.py \
        -choice 'processed' \
        -train_file $OUTDIR/train.json \
        -valid_file $OUTDIR/valid.json \
        -test_file $OUTDIR/test.json;
fi

}

function prepare-oagkx () {

SRC_DIR=../..
OUTDIR=$2

if [[ ! -d $OUTDIR ]]; then
    echo "============Processing OAGK dataset============"
    PYTHONPATH=$SRC_DIR python -W ignore ../prepare.py \
        -dataset OAGK \
        -data_dir OAGK \
        -out_dir $OUTDIR \
        -tokenizer $1 \
        -workers 60;
    echo "Aggregating statistics of the processed OAGK dataset"
    python ../data_stat.py \
        -choice 'processed' \
        -train_file $OUTDIR/train.json \
        -valid_file $OUTDIR/valid.json \
        -test_file $OUTDIR/test.json;
fi

}

#download_oagk
download_oagkx
#prepare-oagkx WhiteSpace processed
#prepare BertTokenizer processed
