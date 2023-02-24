import os
import argparse
import json

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from data.prep_util import *


def load_data(filename, dataset_name):
    data = []
    # for KP20k and cross-domain datasets
    if dataset_name in ['KP20k', 'inspec', 'krapivin', 'semeval', 'nus']:
        if not os.path.exists(filename[0]):
            return []
        with open(filename[0]) as f1, open(filename[1]) as f2:
            for source, target in tqdm(zip(f1, f2), total=count_file_lines(filename[0])):
                source = source.strip()
                target = target.strip()
                if not source:
                    continue
                if not target:
                    continue

                source = source.replace('<digit>', DIGIT)
                target = target.replace('<digit>', DIGIT)

                src_parts = source.split('<eos>', 1)
                assert len(src_parts) == 2
                title = src_parts[0].strip()
                abstract = src_parts[1].strip()

                keywords = target.split('<peos>')
                assert len(keywords) == 2
                present_keywords = [kp.strip() for kp in keywords[0].split(';') if kp]
                absent_keywords = [kp.strip() for kp in keywords[1].split(';') if kp]
                ex = {
                    'id': len(data),
                    'title': title,
                    'abstract': abstract,
                    'present_keywords': present_keywords,
                    'absent_keywords': absent_keywords
                }
                data.append(ex)
        print('Dataset loaded from %s and %s.' % (filename[0], filename[1]))
    else:
        if not os.path.exists(filename):
            return []
        with open(filename) as f:
            for line in tqdm(f, total=count_file_lines(filename)):
                ex = json.loads(line)
                if dataset_name == 'StackEx':
                    ex = {
                        'id': ex['id'],
                        'title': ex['title'],
                        'abstract': ex['question'],
                        'keyword': ex['tags']
                    }
                elif dataset_name == 'OpenKP':
                    present_keywords = []
                    if 'KeyPhrases' in ex:
                        present_keywords = [' '.join(pkp) for pkp in ex['KeyPhrases']]
                    ex = {
                        'id': ex['url'],
                        'title': '',
                        'abstract': ex['text'],
                        'present_keywords': present_keywords,
                        'absent_keywords': []
                    }
                elif dataset_name in ['kpbiomed-small', 'kpbiomed-medium', 'kpbiomed-large']:
                    ex = {
                        'id': ex['id'],
                        'title': ex['title'],
                        'abstract': ex['abstract'],
                        'keyword': ';'.join(ex['keyphrases'])
                    }
                elif 'ldkp' in dataset_name:
                    title_txt = ""
                    if "title" in ex["sections"]:
                        title_idx = ex["sections"].index("title")
                        ex["sections"].pop(title_idx)
                        title_txt = ' '.join(ex["sec_text"].pop(title_idx))

                    abstract_txt_list = [""]
                    if "abstract" in ex["sections"]:
                        abstract_idx = ex["sections"].index("abstract")
                        ex["sections"].pop(abstract_idx)
                        abstract_txt_list.append(' '.join(ex["sec_text"].pop(abstract_idx)))

                    abstract_txt_list += [' '.join(x) for x in ex["sec_text"]]
                    abstract_txt = ' '.join(abstract_txt_list).replace('Abstract-', '')
                    
                    ex = {
                        'id': ex['id'],
                        'title': title_txt,
                        'abstract': abstract_txt,
                        'keyword': ';'.join(ex['extractive_keyphrases']
                                            + ex['abstractive_keyphrases'])
                    }
                    
                else:
                    if 'id' not in ex:
                        ex['id'] = len(data)
                    if 'keywords' in ex:
                        ex['keyword'] = ex['keywords']
                        ex.pop('keywords')

                data.append(ex)
        print('Dataset loaded from %s.' % filename)
    return data


def main(config, TOK):
    pool = Pool(config.workers, initializer=TOK.initializer)

    train_dataset = []
    dataset = load_data(config.train, config.dataset)
    if dataset:
        with tqdm(total=len(dataset), desc='Processing') as pbar:
            for i, ex in enumerate(pool.imap(TOK.process, dataset, 100)):
                pbar.update()
                train_dataset.append(ex)
        with open(os.path.join(config.out_dir, 'train.json'), 'w', encoding='utf-8') as fw:
            fw.write('\n'.join([json.dumps(ex) for ex in train_dataset]))

    valid_dataset = []
    dataset = load_data(config.valid, config.dataset)
    if dataset:
        with tqdm(total=len(dataset), desc='Processing') as pbar:
            for i, ex in enumerate(pool.imap(TOK.process, dataset, 100)):
                pbar.update()
                valid_dataset.append(ex)
        with open(os.path.join(config.out_dir, 'valid.json'), 'w', encoding='utf-8') as fw:
            fw.write('\n'.join([json.dumps(ex) for ex in valid_dataset]))

    test_dataset = []
    dataset = load_data(config.test, config.dataset)
    if dataset:
        with tqdm(total=len(dataset), desc='Processing') as pbar:
            for i, ex in enumerate(pool.imap(TOK.process, dataset, 100)):
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
            vocab = create_vocab(train_dataset + valid_dataset + test_dataset)
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
    options['kp_separator'] = ';'

    if opt.dataset == 'KPTimes':
        opt.train = os.path.join(opt.data_dir, 'KPTimes.train.jsonl')
        opt.valid = os.path.join(opt.data_dir, 'KPTimes.valid.jsonl')
        opt.test = os.path.join(opt.data_dir, 'KPTimes.test.jsonl')

    if opt.dataset in ['kpbiomed-small', 'kpbiomed-medium', 'kpbiomed-large'] \
       or 'ldkp' in opt.dataset:
        opt.train = os.path.join(opt.data_dir, 'train.jsonl')
        opt.valid = os.path.join(opt.data_dir, 'valid.jsonl')
        opt.test = os.path.join(opt.data_dir, 'test.jsonl')

    if opt.dataset == 'OAGK':
        options['kp_separator'] = ','
        opt.train = os.path.join(opt.data_dir, 'oagk_train.txt')
        opt.valid = os.path.join(opt.data_dir, 'oagk_val.txt')
        opt.test = os.path.join(opt.data_dir, 'oagk_test.txt')

    if opt.dataset == 'KP20k':
        options['replace_digit_tokenizer'] = None
        options['kp_separator'] = None
        opt.train = (os.path.join(opt.data_dir, 'train_src.txt'),
                     os.path.join(opt.data_dir, 'train_trg.txt'))
        opt.valid = (os.path.join(opt.data_dir, 'valid_src.txt'),
                     os.path.join(opt.data_dir, 'valid_trg.txt'))
        opt.test = (os.path.join(opt.data_dir, 'test_src.txt'),
                    os.path.join(opt.data_dir, 'test_trg.txt'))

    if opt.dataset in ['inspec', 'krapivin', 'semeval', 'nus']:
        options['replace_digit_tokenizer'] = None
        options['kp_separator'] = None
        opt.form_vocab = False
        opt.train = ('', '')
        opt.valid = ('', '')
        opt.test = (
            os.path.join(opt.data_dir, 'word_{}_testing_context.txt'.format(opt.dataset)),
            os.path.join(opt.data_dir, 'word_{}_testing_allkeywords.txt'.format(opt.dataset))
        )

    if opt.dataset == 'StackEx':
        opt.train = os.path.join(opt.data_dir, 'stackexchange_train.json')
        opt.valid = os.path.join(opt.data_dir, 'stackexchange_valid.json')
        opt.test = os.path.join(opt.data_dir, 'stackexchange_test.json')

    if opt.dataset == 'OpenKP':
        options['kp_separator'] = None
        opt.train = os.path.join(opt.data_dir, 'OpenKPTrain.jsonl')
        opt.valid = os.path.join(opt.data_dir, 'OpenKPDev.jsonl')
        opt.test = os.path.join(opt.data_dir, 'OpenKPEvalPublic.jsonl')

    if opt.dataset == 'MSMARCO':
        opt.train = os.path.join(opt.data_dir, 'train.json')
        opt.valid = os.path.join(opt.data_dir, 'valid.json')
        opt.test = ''

    TOK = MultiprocessingTokenizer(options)
    main(opt, TOK)
