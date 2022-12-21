import os
import argparse
import json

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from nltk.stem.porter import *

stemmer = PorterStemmer()


class Preparer(object):

    def __init__(self, max_src_len):
        self.max_src_len = max_src_len

    def initializer(self):
        pass

    def process(self, example):
        title_tokenized = example['title']['tokenized']
        abstract_tokenized = example['abstract']['tokenized']
        paragraph = (title_tokenized + ' ' + abstract_tokenized).split()
        paragraph = paragraph[:self.max_src_len]
        pkp_tokenized = example['present_kps']['tokenized']

        return {
            'id': example['id'],
            'paragraph': paragraph,
            'pkp_tokenized': pkp_tokenized,
            'labels': compute_bio_tags(paragraph, pkp_tokenized)
        }


def stem_word_list(word_list):
    return [stemmer.stem(w.strip().lower()) for w in word_list]


def stem_text(text):
    return ' '.join(stem_word_list(text.split()))


def compute_bio_tags(para_tokens, present_kps):
    present_kps = [stem_text(pkp) for pkp in present_kps]
    para_tokens = stem_word_list(para_tokens)
    paragraph = ' '.join(para_tokens)
    bio_tags = ['O'] * len(para_tokens)
    for pkp in present_kps:
        if pkp in paragraph:
            pkp_tokens = pkp.split()
            for j in range(0, len(para_tokens) - len(pkp_tokens) + 1):
                if pkp_tokens == para_tokens[j:j + len(pkp_tokens)]:
                    bio_tags[j] = 'B'
                    if len(pkp_tokens) > 1:
                        bio_tags[j + 1:j + len(pkp_tokens)] = ['I'] * (len(pkp_tokens) - 1)

    return bio_tags


def load_data(filename):
    data = []
    if filename:
        with open(filename) as f:
            for line in f:
                data.append(json.loads(line.strip()))
    return data


def main(config, TOK):
    pool = Pool(config.workers, initializer=TOK.initializer)

    train_dataset = []
    dataset = load_data(config.train)
    if dataset:
        with tqdm(total=len(dataset), desc='Processing') as pbar:
            for i, ex in enumerate(pool.imap(TOK.process, dataset, 100)):
                pbar.update()
                train_dataset.append(ex)
        with open(os.path.join(config.out_dir, 'train.txt'), 'w', encoding='utf-8') as fw:
            for ex in train_dataset:
                fw.write(
                    json.dumps({'source': ex['paragraph'], 'target': ex['labels']})
                    + '\n'
                )

    valid_dataset = []
    dataset = load_data(config.valid)
    if dataset:
        with tqdm(total=len(dataset), desc='Processing') as pbar:
            for i, ex in enumerate(pool.imap(TOK.process, dataset, 100)):
                pbar.update()
                valid_dataset.append(ex)
        with open(os.path.join(config.out_dir, 'valid.txt'), 'w', encoding='utf-8') as fw:
            for ex in valid_dataset:
                fw.write(
                    json.dumps({'source': ex['paragraph'], 'target': ex['labels']})
                    + '\n'
                )

    test_dataset = []
    dataset = load_data(config.test)
    if dataset:
        with tqdm(total=len(dataset), desc='Processing') as pbar:
            for i, ex in enumerate(pool.imap(TOK.process, dataset, 100)):
                pbar.update()
                test_dataset.append(ex)
        with open(os.path.join(config.out_dir, 'test.txt'), 'w', encoding='utf-8') as fw:
            for ex in test_dataset:
                fw.write(
                    json.dumps({'source': ex['paragraph'], 'target': ex['labels']})
                    + '\n'
                )
        with open(os.path.join(config.out_dir, 'test.source'), 'w', encoding='utf-8') as fw1, \
                open(os.path.join(config.out_dir, 'test.target'), 'w', encoding='utf-8') as fw2:
            for ex in test_dataset:
                fw1.write(' '.join(ex['paragraph']) + '\n')
                fw2.write(';'.join(ex['pkp_tokenized']) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-data_dir', required=True,
                        help='Directory where the source files are located')
    parser.add_argument('-out_dir', required=True,
                        help='Directory where the output files will be saved')
    parser.add_argument('-dataset', required=True)
    parser.add_argument('-max_src_len', type=int, default=510)
    parser.add_argument('-workers', type=int, default=20)

    opt = parser.parse_args()

    if not os.path.exists(opt.data_dir):
        raise FileNotFoundError

    Path(opt.out_dir).mkdir(parents=True, exist_ok=True)

    if opt.dataset == 'KP20k':
        opt.train = os.path.join(opt.data_dir, 'train.json')
        opt.valid = os.path.join(opt.data_dir, 'valid.json')
        opt.test = os.path.join(opt.data_dir, 'test.json')

    elif opt.dataset in ['inspec', 'krapivin', 'semeval', 'nus']:
        opt.form_vocab = False
        opt.train = ''
        opt.valid = ''
        opt.test = os.path.join(opt.data_dir, 'test.json')

    elif opt.dataset in ['KPTimes', 'OpenKP', 'StackEx', 'kpbiomed-small', 'kpbiomed-medium', 'kpbiomed-large']:
        opt.train = os.path.join(opt.data_dir, 'train.json')
        opt.valid = os.path.join(opt.data_dir, 'valid.json')
        opt.test = os.path.join(opt.data_dir, 'test.json')

    else:
        raise Exception("dataset not recognized")

    TOK = Preparer(opt.max_src_len)
    main(opt, TOK)
