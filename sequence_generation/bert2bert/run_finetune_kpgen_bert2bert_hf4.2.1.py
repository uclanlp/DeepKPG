# based on https://github.com/patrickvonplaten/notebooks/blob/217eefd5fa7db93b075c84c50ebb5964d86fa03c/BERT2BERT_for_CNN_Dailymail.ipynb

import argparse
from transformers import AutoTokenizer, EncoderDecoderModel, BertModel, RobertaModel, AutoConfig, BertConfig, RobertaConfig, TrainingArguments, Seq2SeqTrainer
from transformers.integrations import TensorBoardCallback
from dataclasses import dataclass, field
from typing import Optional
import datasets
import random
import numpy as np
import shutil
from torchinfo import summary


def parse_args():
    parser = argparse.ArgumentParser(description='Finetune a BERT2BERT model on keyphrase generation.')
    # model
    # parser.add_argument('--model_name_or_path', type=str, help='Model name or path, or special configuration name')
    parser.add_argument('--encoder', type=str, help='Model name or path for the encoder')
    parser.add_argument('--decoder', type=str, help='Model name or path for the decoder')
    parser.add_argument('--random_init_encoder', type=bool, default=False, help='Randomly initialize the encoder')
    parser.add_argument('--random_init_decoder', type=bool, default=False, help='Randomly initialize the decoder')
    parser.add_argument('--max_pred_length', type=int)
    parser.add_argument('--num_beams', type=int, default=1)

    # data args
    parser.add_argument('--train_file', type=str, help='A json/csv file containing the training data.')
    parser.add_argument('--validation_file', type=str, help='A json/csv file containing the validation data.')
    parser.add_argument('--src_column', type=str, help='Name of the column containing the source text.')
    parser.add_argument('--tgt_column', type=str, help='Name of the column containing the target text.')

    # training args
    parser.add_argument('--num_train_epochs', type=int, help='Number of epochs to train.')
    parser.add_argument('--per_device_train_batch_size', type=int, help='Batch size for training.')
    parser.add_argument('--per_device_eval_batch_size', type=int, help='Batch size for validation.')
    parser.add_argument('--learning_rate', type=float, help='Learning rate.')
    parser.add_argument('--lr_scheduler_type', type=str)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--warmup_steps', type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--attention_dropout', type=float)
    parser.add_argument('--fp16', type=bool, default=False)
    parser.add_argument('--seed', type=int)

    # logging and saving args
    parser.add_argument('--output_dir', type=str, help='Output directory.')
    parser.add_argument('--overwrite_output_dir', default=False, action="store_true")
    parser.add_argument('--logging_steps', type=int)
    parser.add_argument('--evaluation_strategy', type=str, help='Evaluation strategy: steps, epoch, no.')
    parser.add_argument('--eval_steps', type=int)
    parser.add_argument('--predict_with_generate', default=False, action="store_true")
    parser.add_argument('--save_steps', type=int)
    parser.add_argument('--save_total_limit', type=int)

    return parser.parse_args()


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear", metadata={"help": f"Which lr scheduler to use."}
    )
    generation_max_length: Optional[int] = field(default=256)
    generation_num_beams: Optional[int] = field(default=1)


def get_custom_bert2bert(args):
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    
    bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(args.encoder, args.decoder)

    # if random decoder specified, random init a model and replace the decoder weights
    if args.random_init_decoder:
        random_config = AutoConfig.from_pretrained(decoder_name)
        random_decoder = BertModel(config=random_config, add_pooling_layer=False)
        state_dict = random_decoder.state_dict()
        state_dict.update({k: v for k, v in bert2bert.decoder.bert.state_dict().items() if 'crossattention' in k})
        bert2bert.decoder.bert.load_state_dict(state_dict)

    # if random encoder specified, random init a model and replace the encoder weights
    if args.random_init_encoder:
        random_config = AutoConfig.from_pretrained(encoder_name)
        random_encoder = BertModel(config=random_config, add_pooling_layer=True)
        state_dict = random_encoder.state_dict()
        bert2bert.encoder.load_state_dict(state_dict)

    # set special tokens
    bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
    bert2bert.config.eos_token_id = tokenizer.eos_token_id
    bert2bert.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    bert2bert.config.vocab_size = bert2bert.config.decoder.vocab_size
    bert2bert.config.max_length = args.max_pred_length
    bert2bert.config.min_length = 1
    #bert2bert.config.no_repeat_ngram_size = 3
    #bert2bert.config.early_stopping = True
    #bert2bert.config.length_penalty = 2.0
    bert2bert.config.num_beams = args.num_beams

    return tokenizer, bert2bert 


def main(args):
    tokenizer, bert2bert = get_custom_bert2bert(args)
    print(bert2bert)
    print(summary(bert2bert))
    
    data_files = {}
    data_files["train"] = args.train_file
    data_files["validation"] = args.validation_file
    extension = args.train_file.split(".")[-1]
    raw_datasets = datasets.load_dataset(extension, data_files=data_files)
    train_data = raw_datasets["train"]
    val_data = raw_datasets["validation"]

    encoder_max_length=512
    decoder_max_length=args.max_pred_length


    def process_data_to_model_inputs(batch):
        inputs = tokenizer(batch[args.src_column], padding="max_length", truncation=True, max_length=encoder_max_length)
        outputs = tokenizer(batch[args.tgt_column], padding="max_length", truncation=True, max_length=decoder_max_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

        return batch

    train_data = train_data.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=args.per_device_train_batch_size, 
        remove_columns=[args.src_column, args.tgt_column] #, "id"]
    )
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    val_data = val_data.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=args.per_device_eval_batch_size, 
        remove_columns=[args.src_column, args.tgt_column] #, "id"]
    )
    val_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )


    def compute_metrics_kpgen(eval_preds):
        def postprocess_text_kpgen(preds, labels):
            preds = [[x.strip() for x in pred.strip().split(';')] for pred in preds]
            labels = [[x.strip() for x in label.strip().split(';')] for label in labels]
            return preds, labels
            
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # print some examples
        random_samples = random.sample(list(zip(decoded_preds, decoded_labels)), 10)
        for cur_preds, cur_labels in random_samples:
            print({'pred': cur_preds, 'label': cur_labels})
        
        decoded_preds, decoded_labels = postprocess_text_kpgen(decoded_preds, decoded_labels)     

        # batch-wise f1
        kp2scores = {}
        for cur_label in set([l for case in decoded_labels for l in case]):
            kp2scores[cur_label] = {"tp": 0, "fp": 0, "fn": 0, "label_count": 0, "pred_count": 0, "precision": 0, "recall": 0, "f1": 0}
        for cur_labels, cur_preds in zip(decoded_labels, decoded_preds):
            cur_preds = set(cur_preds)
            cur_labels = set(cur_labels)
            for cur_label in cur_labels:
                kp2scores[cur_label]["label_count"] += 1
                if cur_label in cur_preds:
                    kp2scores[cur_label]["tp"] += 1
                else:
                    kp2scores[cur_label]["fn"] += 1
            for cur_pred in cur_preds:
                if cur_pred in kp2scores:
                    kp2scores[cur_pred]["pred_count"] += 1
                    if cur_pred not in cur_labels:
                        kp2scores[cur_pred]["fp"] += 1
        for cur_label in kp2scores.keys():
            kp2scores[cur_label]['precision'] = (kp2scores[cur_label]["tp"] / kp2scores[cur_label]["pred_count"]) if kp2scores[cur_label]["pred_count"] != 0 else 0
            kp2scores[cur_label]['recall'] = (kp2scores[cur_label]["tp"] / kp2scores[cur_label]["label_count"]) if kp2scores[cur_label]["pred_count"] != 0 else 0 
            kp2scores[cur_label]['f1'] = ((kp2scores[cur_label]['precision'] + kp2scores[cur_label]['recall']) / 2 * kp2scores[cur_label]['recall'] * kp2scores[cur_label]['precision']) if kp2scores[cur_label]['recall'] * kp2scores[cur_label]['precision'] != 0 else 0

        result = {'f1': np.mean([x['f1'] for x in kp2scores.values()])}
        return result

    # set training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        do_train=True,
        do_eval=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler_type,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,       
        warmup_steps=args.warmup_steps,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        overwrite_output_dir=args.overwrite_output_dir,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        fp16=args.fp16, 
        seed=args.seed
    )

    # instantiate trainer
    trainer = Seq2SeqTrainer(
        # config=bert2bert.config,
        model=bert2bert,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics_kpgen,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    # some warnings may interfere with tqdm logging
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
