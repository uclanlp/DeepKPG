#!/usr/bin/env bash

SRC_DIR=`realpath ../..`;


function download_data () {

    python download_data.py

    for resource in small medium large; do
	mkdir -p kpbiomed-$resource
	mv train_${resource}.jsonl kpbiomed-$resource/train.jsonl
	cp validation.jsonl kpbiomed-$resource/valid.jsonl
	cp test.jsonl kpbiomed-$resource/test.jsonl
    done

    rm validation.jsonl test.jsonl
    
}


function prepare () {

    SIZE=$2
    OUTDIR=$3

    if [[ ! -d $OUTDIR ]]; then
	echo "============Processing kpbiomed-${SIZE} dataset============"
	PYTHONPATH=$SRC_DIR python -W ignore ../prepare.py \
		  -dataset kpbiomed-${SIZE} \
		  -data_dir kpbiomed-${SIZE}/ \
		  -out_dir $OUTDIR \
		  -tokenizer $1 \
		  -workers 60;
	echo "Aggregating statistics of the processed kpbiomed-${SIZE} dataset"
	python ../data_stat.py \
               -choice 'processed' \
               -train_file $OUTDIR/train.json \
               -valid_file $OUTDIR/valid.json \
               -test_file $OUTDIR/test.json;
    fi
    
}


function fairseq_prepare () {

outdir=$1/fairseq
mkdir -p $outdir
for split in train valid test; do
    PYTHONPATH=$SRC_DIR python -W ignore ../format.py \
        -input_json $1/processed/${split}.json \
        -output_source $outdir/${split}.source \
        -output_target $outdir/${split}.target;
done

}


function json_prepare () {

outdir=$1/json
mkdir -p $outdir
for split in train valid test; do
    PYTHONPATH=$SRC_DIR python -W ignore ../format.py \
        -format json \
        -input_json $1/processed/${split}.json \
        -output_source $outdir/${split}.json;
done

}


download_data
for size in small medium large; do
    prepare WhiteSpace ${size} kpbiomed-${size}/processed
    fairseq_prepare kpbiomed-${size}
    json_prepare kpbiomed-${size}
done
