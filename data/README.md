## Downloading and Preprocessing Datasets

Make sure you install the environment (instructions in the parent folder). Then, go to the directory after the dataset name you want to use and execute the `run.sh` script.

### Example: download and preprocess the KPTimes dataset

```bash
$ cd kptimes
$ bash run.sh
```

### Data Format
After the preprocessing script finishes, the will be three folders.

- `processed`: One json file containing a entry with key `id`, `title`, `abstract`, `present_kps`, and `absent_kps` for each document in the training, validation, and test dataset. 
- `fairseq`: `x.source` and and `x.target` files for x in `train`, `valid`, and `test`. Mainly used for validation and finetuning with fairseq. `x.source` contains the inputs consisting of title and abstract, concatenated with a `[sep]` token. `x.target` contains the target keyphrases concatenated with ` ; `.
- `json`: same content as `fairseq` in json format. Used for fine-tuning the sequence generation models.
- If you wish to run sequence labeling for keyphrase extraction, please follow the data preprocessing procedure in the `sequence_tagging` folder. After preprocessing, there will be a `bioformat` folder containing the data required for sequence tagging.

## Keyphrase Generation/Extraction Datasets

### KP20k, Inspec, NUS, SemEval, Krapivin

- Paper: https://www.aclweb.org/anthology/P17-1054/
- Download data from: https://drive.google.com/open?id=1DbXV1mZXm_o9bgfwPV9PV0ZPcNo1cnLp

### KPBiomed

- Paper: https://arxiv.org/abs/2211.12124
- Download data from: https://huggingface.co/datasets/taln-ls2n/kpbiomed

### KPTimes

- Paper: https://www.aclweb.org/anthology/W19-8617/
- Download data from: https://github.com/ygorg/KPTimes

### StackEx

- Paper: https://www.aclweb.org/anthology/2020.acl-main.710/
- Download data from: https://github.com/memray/OpenNMT-kpg-release

### OpenKP

- Paper: https://www.aclweb.org/anthology/D19-1521/
- Download data from: https://github.com/microsoft/OpenKP#download-the-dataset

### OAGK and OAGKX

- Paper: https://www.aclweb.org/anthology/N19-1070/
- Download data from: http://hdl.handle.net/11234/1-2943

### LDKP

- Paper: https://arxiv.org/pdf/2203.15349.pdf
- Download data from: https://huggingface.co/datasets/midas/ldkp3k and https://huggingface.co/datasets/midas/ldkp10k

### MSMARCO (Query Prediction from Clicked Documents)

- Paper: https://arxiv.org/abs/2006.05324
- Download data from: https://microsoft.github.io/TREC-2020-Deep-Learning/

## Summarization Datasets

### arxiv and pubmed

- Paper: https://aclanthology.org/N18-2097/
- Download data from: https://github.com/armancohan/long-summarization
