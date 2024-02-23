# DeepKPG

## News
- [**2024/2**] [On Leveraging Encoder-only Pre-trained Language Models for Effective Keyphrase Generation](https://arxiv.org/abs/2402.14052) is accepted to LREC-COLING 2024.
- [**2023/10**] [Rethinking Model Selection and Decoding for Keyphrase Generation with Pre-trained Sequence-to-Sequence Models](https://arxiv.org/abs/2310.06374) is accepted to EMNLP 2023.
- [**2023/5**] We have released the weights of [SciBART](#scibart).

## Introduction
We provide support for a range of Deep Keyphrase Generation and Extraction methods with Pre-trained Language Models (PLMs). This repository contains the code for two papers:
- [Rethinking Model Selection and Decoding for Keyphrase Generation with Pre-trained Sequence-to-Sequence Models](https://arxiv.org/abs/2310.06374)
- [Pre-trained Language Models for Keyphrase Generation: A Thorough Empirical Study](https://arxiv.org/abs/2212.10233) 

The methods and models we cover as follows
- [Keyphrase Extraction with BERT-like encoder-only PLMs with or without CRF](#keyphrase-extraction-via-sequence-tagging).
- [Keyphrase Generation with encoder-only PLMs following the BERT2BERT paradigm](#bert2bert).
- [Keyphrase Generation with encoder-only PLMs following the UniLM (prefix-LM fine-tuning) paradigm](#unilm). 
- [Keyphrase Generation with sequence-to-sequence PLMs such as BART and T5](#keyphrase-generation-with-sequence-to-sequence-plms).
- Code and checkpoints for in-domain PLMs: [SciBART](#scibart), [NewsBART](#newsbart), and [NewsBERT](#newsbert).

For semantic-based evaluation, please refer to [KPEval](https://github.com/uclanlp/KPEval/).

If you find this work helpful, please consider citing
```
@article{https://doi.org/10.48550/arxiv.2212.10233,
  doi = {10.48550/ARXIV.2212.10233},
  url = {https://arxiv.org/abs/2212.10233},
  author = {Wu, Di and Ahmad, Wasi Uddin and Chang, Kai-Wei},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Pre-trained Language Models for Keyphrase Generation: A Thorough Empirical Study},
  publisher = {arXiv},
  year = {2022}, 
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Getting Started
This project requires a GPU environment with CUDA. We recommend following the steps below to use the project.

### Set up a conda environment
```
conda create --name deepkpg python==3.8.13
conda activate deepkpg
```

### Install the packages
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install --upgrade -r requirements.txt
```

### Install apex (optional):
```
git clone https://github.com/NVIDIA/apex
cd apex
export CXX=g++
export CUDA_HOME=/usr/local/cuda-11.3
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
cd ..
```

### Data Preparation
We support a range of keyphrase datasets. They can be prepared by simply running the corresponding `run.sh` in corresponding folders. Detailed instructions and introduction of the datasets can be found [here](https://github.com/xiaowu0162/DeepKPG/tree/main/data#readme).

## Keyphrase Extraction via Sequence Tagging
### Dependencies
After setting up the PyTorch environment as described in the steps above, you can set up the environment for sequence tagging by running the command below. The main difference is `transformers==3.0.2`.
```
pip install -r sequence_tagging/requirements.txt
```
### Training a PLM-based keyphrase extraction model
- Prepare the data by running `cd sequence_tagging ; bash prepare_data.sh`. The script will preprocess several datasets. If you only want to process a single dataset, please comment out the other blocks in the script. 
- Modify the parameters in `run_train.sh`. By default, the script fine-tunes bert-base-uncased without CRF on KP20k.
- `bash run_train.sh`
### Inference and Evaluation
```
bash run_test.sh GPU DATASET OUTPUT_DIR [PLM_NAME] [USE_CRF]
```

## BERT2BERT
### Dependencies
After setting up the PyTorch environment as described in the steps above, you can set up the environment for sequence tagging by running the command below. The main difference is `transformers==4.2.1 datasets==1.1.1 accelerate==0.10.0`.
```
pip install -r sequence_generation/bert2bert/requirements.txt
```
### Training a BERT2BERT style keyphrase generation model
- Prepare the data by running the corresponding `run.sh` in the dataset folders.
- `cd sequence_generation/bert2bert`
- Modify the parameters in `run_train_bert2bert.sh`. By default, the script fine-tunes a BERT2BERT model with an 8-layer BERT encoder and a 4-layer BERT decoder on KP20k. For RND2BERTor BERT2RND, simply pass the flag `--random_init_encoder` or `--random_init_decoder`.
- `bash run_train_bert2bert.sh`
### Inference and Evaluation
```
bash run_test_bert2bert.sh GPU DATASET OUTPUT_DIR
```

## UniLM
### Dependencies
After setting up the PyTorch environment as described in the steps above, you can set up the environment for sequence tagging by running the command below. The main difference is `transformers==3.0.2`.
```
pip install -r sequence_generation/unilm/requirements.txt
```
### Training a UniLM style keyphrase generation model with encoder-only PLMs
- Prepare the data by running the corresponding `run.sh` in the dataset folders.
- `cd sequence_generation/unilm`
- Modify the parameters in `run_train.sh`. By default, the script fine-tunes a bert-base model on KP20k. If apex is not installed, you may remove the flags `--fp16 --fp16_opt_level O1` to run the script.
- `bash run_train.sh`
### Inference and Evaluation
```
bash run_test_bert2bert.sh GPU BASE_PLM DATASET OUTPUT_DIR CKPT_NAME [BATCH_SIZE]
```

## Keyphrase Generation with sequence-to-sequence PLMs
### Dependencies
After setting up the PyTorch environment as described in the steps above, you can directly run sequence generation experiments in the `sequence_generation/seq2seq` folder.
### Training, Inference, and Evaluation
- Prepare the data by running the corresponding `run.sh` in the dataset folders.
- `cd sequence_generation/seq2seq`
- Modify the parameters in `run_t5_large.sh` or `run_bart_large.sh`. 
- You can directly replace the "-large" with "-base" to run the corresponding base-sized models. We empirically find that for base models 15 epochs and warmup 2000 steps with batch size 64 work well. You can also use `uclanlp/newsbart-base` or `bloomberg/KeyBART` in `run_bart_large.sh`. 
- `bash run_t5_large.sh` or `bash run_bart_large.sh`.

## SciBART
We pre-train BART-base and BART-large from scratch using paper titles and abstracts from a scientific corpus [S2ORC](https://github.com/allenai/s2orc). The pre-training was done with fairseq and the model is converted to huggingface and released here - [uclanlp/scibart-base](https://huggingface.co/uclanlp/scibart-base) and [uclanlp/scibart-large](https://huggingface.co/uclanlp/scibart-large). 

As we train a new vocabulary from scratch on the S2ORC corpus using sentencepiece, SciBart is incompatible with the original `BartTokenizer`. We are submitting a pull request to huggingface to include our new tokenizer. For now, to use SciBart, you can clone and install transformers from our own branch: 
```
git clone https://github.com/xiaowu0162/transformers.git -b scibart-integration
cd transformers
pip install -e .
```
Then, you may use the model as usual:
```
from transformers import BartForConditionalGeneration, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('uclanlp/scibart-large')
model = BartForConditionalGeneration.from_pretrained('uclanlp/scibart-large')
print(tokenizer.batch_decode(model.generate(**tokenizer('This is an example of a <mask> computer.', return_tensors='pt'))))
```

## NewsBART
We continue pre-train facebook's BART-base on the [realnews](https://github.com/rowanz/grover/tree/master/realnews) dataset without changing the vocabulary. More details regarding the pre-training can be found in our paper. The model is released on [huggingface model hub](https://huggingface.co/uclanlp/newsbart-base). Fine-tuning it for keyphrase generation is supported in `sequence_generation/seq2seq`.

## NewsBERT
We continue pre-train bert-base-uncased on the [realnews](https://github.com/rowanz/grover/tree/master/realnews) dataset without changing the vocabulary. More details regarding the pre-training can be found in our paper. The model is released on [huggingface model hub](https://huggingface.co/uclanlp/newsbert). Fine-tuning it for keyphrase extraction or generation is fully supported in `sequence_tagging`, `sequence_generation/unilm`, and `sequence_generation/bert2bert`.
