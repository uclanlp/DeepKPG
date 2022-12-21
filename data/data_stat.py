import os
import json
import argparse
import subprocess
from tqdm import tqdm
from prettytable import PrettyTable


def count_file_lines(file_path):
    """
    Counts the number of lines in a file using wc utility.
    :param file_path: path to file
    :return: int, no of lines
    """
    num = subprocess.check_output(['wc', '-l', file_path])
    num = num.decode('utf-8').strip().split(' ')
    return int(num[0])


def raw_statistics(args):
    records = {'train': 0, 'valid': 0, 'test': 0}
    document_tokens = {'train': 0, 'valid': 0, 'test': 0}
    average_number_keyphrase = {'train': 0, 'valid': 0, 'test': 0}
    keyphrase_tokens = {'train': 0, 'valid': 0, 'test': 0}
    document_lengths = {'train': set(), 'valid': set(), 'test': set()}
    unique_document_tokens = {'train': set(), 'valid': set(), 'test': set()}
    unique_keyphrase_tokens = {'train': set(), 'valid': set(), 'test': set()}

    attribute_list = ["Records", "Document Tokens", "Keyphrase Tokens",
                      "Unique Document Tokens", "Unique Keyphrase Tokens"]

    def read_data(split, filename):
        if os.path.isfile(filename):
            with open(filename) as f1:
                for line in tqdm(f1, total=count_file_lines(filename)):
                    example = json.loads(line)
                    if 'text' in example:
                        document = example['text'].split()
                    else:
                        title = example['title']
                        abstract = example.get('abstract')
                        if not abstract:
                            abstract = example.get('question')  # StackEx data
                        document = title.split() + abstract.split()

                    keywords = example.get('keyword')
                    if not keywords:
                        keywords = example.get('tags')  # StackEx data
                    if not keywords:
                        keywords = example.get('KeyPhrases')  # OpenKP data
                    if keywords:
                        keyphrases = [kp.split() for kp in keywords.split(';')] \
                            if isinstance(keywords, str) else keywords
                        average_number_keyphrase[split] += len(keyphrases)
                        keyphrase_tokens[split] += sum([len(kp) for kp in keyphrases])  # total length of all kps
                        unique_keyphrase_tokens[split].update([tok for kp in keyphrases for tok in kp])

                    records[split] += 1
                    document_tokens[split] += len(document)
                    unique_document_tokens[split].update(document)
                    document_lengths[split].add(len(document))

    read_data('train', args.train_file)
    read_data('valid', args.valid_file)
    read_data('test', args.test_file)

    table = PrettyTable()
    table.field_names = ["Attribute", "Train", "Dev", "Test"]
    table.align["Attribute"] = "l"
    table.align["Train"] = "r"
    table.align["Valid"] = "r"
    table.align["Test"] = "r"
    for attr in attribute_list:
        var = eval('_'.join(attr.lower().split()))
        val1 = len(var['train']) if isinstance(var['train'], set) else var['train']
        val2 = len(var['valid']) if isinstance(var['valid'], set) else var['valid']
        val3 = len(var['test']) if isinstance(var['test'], set) else var['test']
        table.add_row([attr, val1, val2, val3])

    table.add_row([
        'Max Document Length',
        '%.2f' % max(document_lengths['train']) if len(document_lengths['train']) > 0 else 0,
        '%.2f' % max(document_lengths['valid']) if len(document_lengths['valid']) > 0 else 0,
        '%.2f' % max(document_lengths['test']) if len(document_lengths['test']) > 0 else 0
    ])
    table.add_row([
        'Avg. Document Length',
        '%.2f' % (document_tokens['train'] / records['train']) if records['train'] > 0 else 0,
        '%.2f' % (document_tokens['valid'] / records['valid']) if records['valid'] > 0 else 0,
        '%.2f' % (document_tokens['test'] / records['test']) if records['test'] > 0 else 0
    ])
    table.add_row([
        'Avg. Keyphrase Length',
        '%.2f' % (keyphrase_tokens['train'] / average_number_keyphrase['train']),
        '%.2f' % (keyphrase_tokens['valid'] / average_number_keyphrase['valid']),
        '%.2f' % (keyphrase_tokens['test'] / average_number_keyphrase['test'])
    ])
    table.add_row([
        '#Keyphrase / Document',
        '%.2f' % (average_number_keyphrase['train'] / records['train']) if records['train'] > 0 else 0,
        '%.2f' % (average_number_keyphrase['valid'] / records['valid']) if records['valid'] > 0 else 0,
        '%.2f' % (average_number_keyphrase['test'] / records['test']) if records['test'] > 0 else 0
    ])
    print(table)


def proc_statistics(args):
    records = {'train': 0, 'valid': 0, 'test': 0}
    document_tokens = {'train': 0, 'valid': 0, 'test': 0}
    average_number_keyphrase = {'train': 0, 'valid': 0, 'test': 0}
    keyphrase_tokens = {'train': 0, 'valid': 0, 'test': 0}
    document_lengths = {'train': list(), 'valid': list(), 'test': list()}
    unique_document_tokens = {'train': set(), 'valid': set(), 'test': set()}
    unique_keyphrase_tokens = {'train': set(), 'valid': set(), 'test': set()}
    total_present_keyphrase = {'train': 0, 'valid': 0, 'test': 0}
    total_absent_keyphrase = {'train': 0, 'valid': 0, 'test': 0}

    attribute_list = ["Records", "Unique Document Tokens", "Unique Keyphrase Tokens"]

    def read_data(split, filename):
        if os.path.exists(filename):
            with open(filename) as f:
                for line in tqdm(f, total=count_file_lines(filename)):
                    ex = json.loads(line.strip())
                    document = ex['title']['tokenized'].split() + ex['abstract']['tokenized'].split()
                    present_kps = [pkp.split() for pkp in ex['present_kps']['tokenized']]  # 1d list
                    absent_kps = [akp.split() for akp in ex['absent_kps']['tokenized']]  # 1d list

                    records[split] += 1
                    document_tokens[split] += len(document)
                    unique_document_tokens[split].update(document)
                    document_lengths[split].append(len(document))
                    total_present_keyphrase[split] += len(present_kps)
                    total_absent_keyphrase[split] += len(absent_kps)

                    average_number_keyphrase[split] += len(present_kps) + len(absent_kps)
                    keyphrase_tokens[split] += sum([len(kp) for kp in present_kps])
                    keyphrase_tokens[split] += sum([len(kp) for kp in absent_kps])
                    unique_keyphrase_tokens[split].update([tok for kp in present_kps for tok in kp])
                    unique_keyphrase_tokens[split].update([tok for kp in absent_kps for tok in kp])

    read_data('train', args.train_file)
    read_data('valid', args.valid_file)
    read_data('test', args.test_file)

    table = PrettyTable()
    table.field_names = ["Attribute", "Train", "Valid", "Test"]
    table.align["Attribute"] = "l"
    table.align["Train"] = "r"
    table.align["Valid"] = "r"
    table.align["Test"] = "r"
    for attr in attribute_list:
        var = eval('_'.join(attr.lower().split()))
        val1 = len(var['train']) if isinstance(var['train'], set) else var['train']
        val2 = len(var['valid']) if isinstance(var['valid'], set) else var['valid']
        val3 = len(var['test']) if isinstance(var['test'], set) else var['test']
        table.add_row([attr, val1, val2, val3])

    table.add_row([
        'Max Doc. Length',
        '%d' % max(document_lengths['train']) if len(document_lengths['train']) > 0 else 0,
        '%d' % max(document_lengths['valid']) if len(document_lengths['valid']) > 0 else 0,
        '%d' % max(document_lengths['test']) if len(document_lengths['test']) > 0 else 0
    ])
    table.add_row([
        'Avg. Doc. Length',
        '%.2f' % (document_tokens['train'] / records['train']) if records['train'] > 0 else 0,
        '%.2f' % (document_tokens['valid'] / records['valid']) if records['valid'] > 0 else 0,
        '%.2f' % (document_tokens['test'] / records['test']) if records['test'] > 0 else 0
    ])
    table.add_row([
        'Avg. Keyphrase Length',
        '%.2f' % (keyphrase_tokens['train'] / average_number_keyphrase['train'])
        if average_number_keyphrase['train'] > 0 else 0,
        '%.2f' % (keyphrase_tokens['valid'] / average_number_keyphrase['valid'])
        if average_number_keyphrase['valid'] > 0 else 0,
        '%.2f' % (keyphrase_tokens['test'] / average_number_keyphrase['test'])
        if average_number_keyphrase['test'] > 0 else 0
    ])
    table.add_row([
        'Avg. #PKp / Document',
        '%.2f' % (total_present_keyphrase['train'] / records['train']) if records['train'] > 0 else 0,
        '%.2f' % (total_present_keyphrase['valid'] / records['valid']) if records['valid'] > 0 else 0,
        '%.2f' % (total_present_keyphrase['test'] / records['test']) if records['test'] > 0 else 0
    ])
    table.add_row([
        'Avg. #AKp / Document',
        '%.2f' % (total_absent_keyphrase['train'] / records['train']) if records['train'] > 0 else 0,
        '%.2f' % (total_absent_keyphrase['valid'] / records['valid']) if records['valid'] > 0 else 0,
        '%.2f' % (total_absent_keyphrase['test'] / records['test']) if records['test'] > 0 else 0
    ])
    print(table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data_stat.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-train_file', default='', help='Train filepath')
    parser.add_argument('-valid_file', default='', help='Validation filepath')
    parser.add_argument('-test_file', default='', help='Test filepath')
    parser.add_argument('-choice', default='raw', choices=['raw', 'processed'])
    opt = parser.parse_args()

    if opt.choice == 'raw':
        raw_statistics(opt)
    elif opt.choice == 'processed':
        proc_statistics(opt)
