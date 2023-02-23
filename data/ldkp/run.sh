#!/usr/bin/env bash

SRC_DIR=`realpath ../..`;


function download_data () {

    for ds in ldkp3k ldkp10k; do
	python download_ldkp_from_hf.py ${ds}

	for resource in small medium large; do
	    mkdir -p ${ds}-${resource}
	    mv ${ds}_${resource}_train.jsonl ${ds}-${resource}/train.jsonl
	    mv ${ds}_${resource}_validation.jsonl ${ds}-${resource}/valid.jsonl
	    mv ${ds}_${resource}_test.jsonl ${ds}-${resource}/test.jsonl
	done
    done
    
}


function prepare () {

    DS=$2
    SIZE=$3
    OUTDIR=$4

    if [[ ! -d $OUTDIR ]]; then
	echo "============Processing ${DS}-${SIZE} dataset============"
	PYTHONPATH=$SRC_DIR python -W ignore ../prepare.py \
		  -dataset ${DS}-${SIZE} \
		  -data_dir ${DS}-${SIZE}/ \
		  -out_dir $OUTDIR \
		  -tokenizer $1 \
		  -workers 60;
	echo "Aggregating statistics of the processed ${DS}-${SIZE} dataset"
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
for ds in ldkp3k ldkp10k; do
    for size in small medium large; do
	prepare WhiteSpace ${ds} ${size} ${ds}-${size}/processed
	fairseq_prepare ${ds}-${size}
	json_prepare ${ds}-${size}
    done
done
