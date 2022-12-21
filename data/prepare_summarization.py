import os
import argparse
import json

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from data.prep_util import *


def load_data(filename, dataset_name):
    data = []
    if not os.path.exists(filename):
        return []
    with open(filename) as f:
        for line in tqdm(f, total=count_file_lines(filename)):
            ex = json.loads(line)
            ex = {
                'id': ex['article_id'],
                'abstract': ' '.join([x.strip() for x in (' '.join(ex['abstract_text']).replace('<S>', '').replace('</S>', '')).split()]),
                'article': ' '.join(ex['article_text'])
            }
            data.append(ex)
        print('Dataset loaded from %s.' % filename)
    return data


def main(config, TOK):
    pool = Pool(config.workers, initializer=TOK.initializer)

    train_dataset = []
    dataset = load_data(config.train, config.dataset)
    if dataset:
        with tqdm(total=len(dataset), desc='Processing') as pbar:
            for i, ex in enumerate(pool.imap(TOK.process_summarization, dataset, 100)):
                pbar.update()
                train_dataset.append(ex)
        with open(os.path.join(config.out_dir, 'train.json'), 'w', encoding='utf-8') as fw:
            fw.write('\n'.join([json.dumps(ex) for ex in train_dataset]))

    valid_dataset = []
    dataset = load_data(config.valid, config.dataset)
    if dataset:
        with tqdm(total=len(dataset), desc='Processing') as pbar:
            for i, ex in enumerate(pool.imap(TOK.process_summarization, dataset, 100)):
                pbar.update()
                valid_dataset.append(ex)
        with open(os.path.join(config.out_dir, 'valid.json'), 'w', encoding='utf-8') as fw:
            fw.write('\n'.join([json.dumps(ex) for ex in valid_dataset]))

    test_dataset = []
    dataset = load_data(config.test, config.dataset)
    if dataset:
        with tqdm(total=len(dataset), desc='Processing') as pbar:
            for i, ex in enumerate(pool.imap(TOK.process_summarization, dataset, 100)):
                pbar.update()
                test_dataset.append(ex)
        with open(os.path.join(config.out_dir, 'test.json'), 'w', encoding='utf-8') as fw:
            fw.write('\n'.join([json.dumps(ex) for ex in test_dataset]))

    if config.form_vocab:
        if config.tokenizer == 'BertTokenizer':
            with open(os.path.join(config.out_dir, 'vocab.txt'), 'w') as fw:
                for token, index in TOK.vocab.items():
                    if token in UNUSED_TOKEN_MAP:
                        if UNUSED_TOKEN_MAP[token] not in TOK.vocab:
                            token = UNUSED_TOKEN_MAP[token]
                    fw.write('{} {}'.format(token.lower(), index) + '\n')
        else:
            vocab = create_vocab_summarization(train_dataset + valid_dataset + test_dataset)
            with open(os.path.join(config.out_dir, 'vocab.txt'), 'w', encoding='utf-8') as fw:
                fw.write('\n'.join(['{} {}'.format(v, i) for i, v in enumerate(vocab)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-data_dir', required=True,
                        help='Directory where the source files are located')
    parser.add_argument('-out_dir', required=True,
                        help='Directory where the output files will be saved')
    parser.add_argument('-tokenizer', default='BertTokenizer',
                        choices=['BertTokenizer', 'SpacyTokenizer', 'WhiteSpace'])
    parser.add_argument('-dataset', required=True)
    parser.add_argument('-workers', type=int, default=20)

    opt = parser.parse_args()
    opt.form_vocab = True

    if not os.path.exists(opt.data_dir):
        raise FileNotFoundError

    Path(opt.out_dir).mkdir(parents=True, exist_ok=True)

    options = dict()
    options['tokenizer'] = opt.tokenizer
    options['replace_digit_tokenizer'] = 'wordpunct'

    if opt.dataset in ['arxiv', 'pubmed']:
        opt.train = os.path.join(opt.data_dir, 'train.json')
        opt.valid = os.path.join(opt.data_dir, 'valid.json')
        opt.test = os.path.join(opt.data_dir, 'test.json')
    else:
        raise NotImplementedError

    TOK = MultiprocessingTokenizer(options)
    main(opt, TOK)
