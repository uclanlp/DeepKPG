#!/usr/bin/env bash

SRC_DIR=`realpath ../../../`;


function download_data () {

    zip_file=pubmed-dataset.zip 
    if [[ ! -f "${zip_file}" ]]; then
        wget https://archive.org/download/armancohan-long-summarization-paper-code/${zip_file}
    fi

    unzip ${zip_file}
    out_folder=pubmed-dataset
    mv ${out_folder}/train.txt train.json
    mv ${out_folder}/val.txt valid.json
    mv ${out_folder}/test.txt test.json
    rm -r ${out_folder} '__MACOSX'

}


function prepare () {

    OUTDIR=$2
    if [[ ! -d $OUTDIR ]]; then
        echo "============Processing pubmed dataset============"
        PYTHONPATH=$SRC_DIR python -W ignore ../../prepare_summarization.py \
            -dataset pubmed \
            -data_dir . \
            -out_dir $OUTDIR \
            -tokenizer $1 \
            -workers 60;
    fi

}


function fairseq_prepare () {

    outdir=fairseq
    mkdir -p $outdir
    for split in train valid test; do
        PYTHONPATH=$SRC_DIR python -W ignore ../../format_summarization.py \
            -input_json processed/${split}.json \
            -output_source $outdir/${split}.source \
            -output_target $outdir/${split}.target;
    done

}


function json_prepare () {

    outdir=json
    mkdir -p $outdir
    for split in train valid test; do
        PYTHONPATH=$SRC_DIR python -W ignore ../../format_summarization.py \
            -format json \
            -input_json processed/${split}.json \
            -output_source $outdir/${split}.json;
    done

}


download_data
prepare WhiteSpace processed
fairseq_prepare
json_prepare