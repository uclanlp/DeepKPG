import json
import argparse
from tqdm import tqdm

KP_SEP = ';'
TITLE_SEP = '[sep]'


def process(infile, outsrc, outtgt, format):
    out_examples = []
    with open(infile, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin):
            ex = json.loads(line)
            source = ex['title']['text'] + ' {} '.format(TITLE_SEP) + ex['abstract']['text']
            kps = ex['present_kps']['text'] + ex['absent_kps']['text']
            target = ' {} '.format(KP_SEP).join([t for t in kps if t])
            #if len(source) == 0 or len(target) == 0:
            #    continue
            if format == 'txt':
                out_examples.append((source, target))
            elif format == 'json':
                out_examples.append({'src': source, 'tgt': target})

    if format == 'txt':
        with open(outsrc, 'w', encoding='utf-8') as fsrc, open(outtgt, 'w', encoding='utf-8') as ftgt:
            fsrc.write('\n'.join([ex[0] for ex in out_examples]))
            ftgt.write('\n'.join([ex[1] for ex in out_examples]))
    elif format == 'json':
        with open(outsrc, 'w', encoding='utf-8') as fout:
            fout.write('\n'.join([json.dumps(ex) for ex in out_examples]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='format.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-input_json', default='', help='input json file')
    parser.add_argument('-output_source', default='', help='output source file')
    parser.add_argument('-output_target', default='', help='output target file')
    parser.add_argument('-format', default='txt', help='output format type')
    args = parser.parse_args()
    process(args.input_json, args.output_source, args.output_target, args.format)
