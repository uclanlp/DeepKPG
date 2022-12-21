#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;
MODEL_DIR=${HOME_DIR}/models
mkdir -p $MODEL_DIR

####################################################################
URL_PREFIX=https://1ubnpq.bn.files.1drv.com
FILENAME=MiniLMv2-L6-H768-distilled-from-RoBERTa-Large
FILE_ID="y4mIX6ParIAPno8mrrumh3CSQIi7cu5LzTBRWVS1jOO-2ddbEItW4EhjD_qg7R_KMjbekZcpfUHTLpwbOlv86gidJFbwMEkq4s8CDtNMDse\
Dn1ebWmv5LDSUjXbEtg-a4DXlNKimn3hefuz6rewH199n8nGIxqtmPNHVzLwL052oq49bKW1rZv_yf2AWV6TgTP9CI2JWK9NwCyjIKQ__6AMow"
DIRNAME=MiniLM-L6-H768-distilled-from-RoBERTa-Large

if [[ ! -d $MODEL_DIR/$DIRNAME ]]; then
    curl ${URL_PREFIX}/$FILE_ID -L -o $MODEL_DIR/${FILENAME}.zip
    unzip $MODEL_DIR/${FILENAME}.zip -d $MODEL_DIR
    rm $MODEL_DIR/${FILENAME}.zip
fi

####################################################################
URL_PREFIX=https://yub8mw.bn.files.1drv.com
FILENAME=MiniLMv2-L6-H768-distilled-from-BERT-Large
FILE_ID="y4mUt_vLuGg959r15dOE1FtgkxtCzYFDe_qoSG5xOj6TuxU1R4PAsg-jZJsM_F1VZO8ITXW79UqEtezjsOsQmRJvsrmugeGDpyqQs_\
A7EDs4jWYa5QdImdzW0VBVPq3JWXFch3aLG9kFj-8FuvLHsoU1lUIyGKEqgyeV5JIJVm0olhBVCKQqKimVd79H-LfYTVU8eAWyfR222b1Trt3yeN_tA"
DIRNAME=MiniLM-L6-H768-distilled-from-BERT-Large

if [[ ! -d $MODEL_DIR/$DIRNAME ]]; then
    curl ${URL_PREFIX}/$FILE_ID -L -o $MODEL_DIR/${FILENAME}.zip
    unzip $MODEL_DIR/${FILENAME}.zip -d $MODEL_DIR
    rm $MODEL_DIR/${FILENAME}.zip
fi

####################################################################
URL_PREFIX=https://0kbnpq.bn.files.1drv.com
FILENAME=MiniLMv2-L6-H768-distilled-from-BERT-Base
FILE_ID="y4mqalvr7sPsJ11AcolvfJ0srhytBfOBBd85zY6bqHDLgbrZKAjIgzDdPl7mHqgdWvXaDRfcEPrc2jpTE3VLMSRH3Kg9Cy87BnQUd2Crvub\
Ztb8j2U7uH6UO68i8X6zSCUTlFwcRnhfOkwmZp0I-oYsrz_3k4UrSvAn586gYB_KNCS6-IhRHWYWvTXKjV4Qp8DkkWYahAB5NjOjxmDjBngZeA"
DIRNAME=MiniLM-L6-H768-distilled-from-BERT-Base

if [[ ! -d $MODEL_DIR/$DIRNAME ]]; then
    curl ${URL_PREFIX}/$FILE_ID -L -o $MODEL_DIR/${FILENAME}.zip
    unzip $MODEL_DIR/${FILENAME}.zip -d $MODEL_DIR
    rm $MODEL_DIR/${FILENAME}.zip
fi