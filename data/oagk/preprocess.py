import argparse
import glob
import re
import json

TITLE_SEP = '[sep]'
KP_SEP = ';'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-data_dir', required=True,
                        help='Directory where the source files are located')
    parser.add_argument('-out_dir', required=True,
                        help='Directory where the source files are located')
    opt = parser.parse_args()

    files = glob.glob("{}/*.txt".format(opt.data_dir))
    total = 0
    file_id = 0
    src_writer = open('{}/split.{}.source'.format(opt.out_dir, file_id), 'w', encoding='utf8')
    tgt_writer = open('{}/split.{}.target'.format(opt.out_dir, file_id), 'w', encoding='utf8')
    for file in files:
        with open(file, 'r') as fd:
            for line in fd:
                ex = json.loads(line.strip())
                source = ex['title'] + ' {} '.format(TITLE_SEP) + ex['abstract']
                target = [re.sub("[\n\r\t ]+", " ", kw).strip() for kw in ex['keywords'].split(',')]
                target = [t for t in target if t]
                if len(target) == 0:
                    continue
                target = ' {} '.format(KP_SEP).join(target)
                src_writer.write(source + '\n')
                tgt_writer.write(target + '\n')
                total += 1
                if total % 2000000 == 0:
                    src_writer.close()
                    tgt_writer.close()
                    file_id += 1
                    src_writer = open('{}/split.{}.source'.format(opt.out_dir, file_id), 'w', encoding='utf8')
                    tgt_writer = open('{}/split.{}.target'.format(opt.out_dir, file_id), 'w', encoding='utf8')

    src_writer.close()
    tgt_writer.close()
