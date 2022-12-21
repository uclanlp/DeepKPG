import numpy as np
from tqdm import tqdm
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--pred_file', type=str, help='Path to kptimes_predictions.txt')
    parser.add_argument('--raw_file', type=str, help='Path to processed/test.json')
    parser.add_argument('--metric', type=str, default='mse', choices=['mae', 'mse'])
    parser.add_argument('--out_dir', type=str, default=None, help='Path to write the output report')
    parser.add_argument('--stdout', type=bool, default=True)

    return parser.parse_args()


def main(args):
    pkp_ref, akp_ref, allkp_ref = [], [], []
    pkp_error, akp_error, allkp_error = [], [], []

    with open(args.pred_file) as pred_f, open(args.raw_file) as raw_f:
        for pred_line in tqdm(pred_f.readlines(), desc=args.dataset):
            pred_entry = json.loads(pred_line)
            n_pkp_pred, n_akp_pred = len(pred_entry['present']), len(pred_entry['absent'])
            n_allkp_pred = n_pkp_pred + n_akp_pred

            raw_entry = json.loads(raw_f.readline())
            n_pkp_raw, n_akp_raw = len(raw_entry['present_kps']['text']), len(raw_entry['present_kps']['text'])
            n_allkp_raw = n_pkp_raw + n_akp_raw
            pkp_ref.append(n_pkp_raw)
            akp_ref.append(n_akp_raw)
            allkp_ref.append(n_allkp_raw)

            if args.metric == 'mae':
                pkp_error.append(np.abs(n_pkp_raw - n_pkp_pred))
                akp_error.append(np.abs(n_akp_raw - n_akp_pred))
                allkp_error.append(np.abs(n_allkp_raw - n_allkp_pred))               
            elif args.metric == 'mse':
                pkp_error.append((n_pkp_raw - n_pkp_pred)**2)
                akp_error.append((n_akp_raw - n_akp_pred)**2)
                allkp_error.append((n_allkp_raw - n_allkp_pred)**2)
                
    if args.out_dir is not None:
        with open(args.out_dir + f'/{args.dataset}_kp_num_report.txt', 'w') as out_f:
            out_f.write(f'metric: {args.metric}\n')
            out_f.write(f'pkp: {np.mean(pkp_error)}\t(ref #kp {np.mean(pkp_ref)})\n')
            out_f.write(f'akp: {np.mean(akp_error)}\t(ref #kp {np.mean(pkp_ref)})\n')
            out_f.write(f'allkp: {np.mean(allkp_error)}\t(ref #kp {np.mean(pkp_ref)})\n')

    if args.stdout:
        print(args.pred_file)
        print(f'metric: {args.metric}')
        print(f'pkp: {np.mean(pkp_error)}\t(ref #kp {np.mean(pkp_ref)})')
        print(f'akp: {np.mean(akp_error)}\t(ref #kp {np.mean(pkp_ref)})')
        print(f'allkp: {np.mean(allkp_error)}\t(ref #kp {np.mean(pkp_ref)})')


if __name__ == '__main__':
    args = parse_args()
    main(args)
