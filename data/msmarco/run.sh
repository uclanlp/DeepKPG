#!/usr/bin/env bash

URL_PREFIX=https://msmarco.blob.core.windows.net/msmarcoranking

function download_file () {

if [[ ! -f "$1" ]]; then
    wget ${URL_PREFIX}/${1}
fi

}

function preprocess () {

download_file msmarco-docs-lookup.tsv.gz
download_file orcas.tsv.gz
if [[ ! -f "msmarco-docs.tsv" ]]; then
    download_file msmarco-docs.tsv.gz
    gzip -d msmarco-docs.tsv.gz
fi
for split in train dev; do
    download_file msmarco-doc${split}-queries.tsv.gz
    download_file msmarco-doc${split}-qrels.tsv.gz
done

python -W ignore preprocess.py;

}


function prepare () {

SRC_DIR=../..
OUTDIR=$2

if [[ ! -d $OUTDIR ]]; then
    echo "============Processing MSMARCO dataset============"
    PYTHONPATH=$SRC_DIR python -W ignore ../prepare.py \
        -dataset MSMARCO \
        -data_dir . \
        -out_dir $OUTDIR \
        -tokenizer $1 \
        -workers 60;
    echo "Aggregating statistics of the processed MSMARCO dataset"
    python ../data_stat.py \
        -choice 'processed' \
        -train_file $OUTDIR/train.json \
        -valid_file $OUTDIR/valid.json;
fi

}

preprocess
prepare BertTokenizer processed

