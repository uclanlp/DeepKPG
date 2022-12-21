# https://github.com/patrickvonplaten/notebooks/blob/217eefd5fa7db93b075c84c50ebb5964d86fa03c/BERT2BERT_for_CNN_Dailymail.ipynb

import torch

import time
import argparse
import datasets
from transformers import BertTokenizer, EncoderDecoderModel


def parse_args():
    parser = argparse.ArgumentParser(description='Finetune a BERT2BERT model on keyphrase generation.')
    # model
    parser.add_argument('--model_name_or_path', type=str, help='Model name or path, or special configuration name')
    parser.add_argument('--eval_batch_size', type=int, help='Batch size for testing.')
    parser.add_argument('--max_pred_length', type=int)
    parser.add_argument('--num_beams', type=int, default=1)

    # data args
    parser.add_argument('--test_file', type=str, help='A json/csv file containing the training data.')
    parser.add_argument('--src_column', type=str, help='Name of the column containing the source text.')
    parser.add_argument('--tgt_column', type=str, help='Name of the column containing the target text.')

    # output
    parser.add_argument('--output_dir', type=str, help='Directory to store the output file.')
    parser.add_argument('--output_file_name', type=str, help='Name of the output file.')

    # benchmark flags
    parser.add_argument('--benchmarking', type=bool, default=False, help='Benchmarking or normal prediction.')
    parser.add_argument('--max_examples', type=int, default=None, help='Maximum number of samples for benchmarking.')

    return parser.parse_args()


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = EncoderDecoderModel.from_pretrained(args.model_name_or_path).to(device) #.cuda()
    model.config.max_length = args.max_pred_length
    model.config.min_length = 1
    model.config.num_beams = args.num_beams

    model.eval()

    data_files = {}
    data_files["test"] = args.test_file
    extension = args.test_file.split(".")[-1]
    raw_datasets = datasets.load_dataset(extension, data_files=data_files)
    test_data = raw_datasets["test"]

    batch_size = args.eval_batch_size

    # map data correctly
    def generate_keyphrases(batch):
        # Tokenizer will automatically set [BOS] <text> [EOS]
        # cut off at BERT max length 512
        inputs = tokenizer(batch[args.src_column], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)  #.cuda()
        attention_mask = inputs.attention_mask.to(device)   #.cuda()

        outputs = model.generate(input_ids, attention_mask=attention_mask)

        # all special tokens including will be removed
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        batch["pred"] = output_str

        return batch

    if args.benchmarking:
        if args.max_examples is not None:
            indices = torch.arange(args.max_examples)
            test_data = test_data.select(indices)
            print(f'Benchmarking with {args.max_examples} examples')
        else:
            print(f'Benchmarking with the entire dataset')

    start_time = time.time()
    results = test_data.map(generate_keyphrases, batched=True, batch_size=batch_size, remove_columns=[args.src_column])
    elapsed_time = time.time() - start_time
    
    if not args.benchmarking:
        pred_str = results["pred"]
        label_str = results[args.tgt_column]
        with open(args.output_file_name, 'w') as f:
            for pred in pred_str:
                f.write(pred)
                f.write('\n')
    else:
        print("------ Inference done with %s seconds ------" % (elapsed_time))
        print(f"Per Item Speed: {elapsed_time/test_data.__len__()} s/example")
        print(f"Throughput: {test_data.__len__()/elapsed_time} examples/s")
           

if __name__ == '__main__':
    args = parse_args()
    main(args)
