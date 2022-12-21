# adapted from https://github.com/kenchan0226/keyphrase-generation-rl/blob/master/evaluate_prediction.py
import os
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from nltk.stem.porter import *

KP_SEP = ';'
TITLE_SEP = '[sep]'
UNK_WORD = '[unk]'

stemmer = PorterStemmer()

INVALIDATE_UNK = True
DISABLE_EXTRA_ONE_WORD_FILTER = True

logger = logging.getLogger()


def update_score_dict(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed, k_list, score_dict, tag):
    num_targets = len(trg_token_2dlist_stemmed)
    num_predictions = len(pred_token_2dlist_stemmed)

    is_match = compute_match_result(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed,
                                    type='exact', dimension=1)
    is_match_substring_2d = compute_match_result(trg_token_2dlist_stemmed,
                                                 pred_token_2dlist_stemmed, type='sub', dimension=2)
    
    # Classification metrics
    precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = \
        compute_classification_metrics_at_ks(is_match, num_predictions, num_targets, k_list=k_list,
                                             meng_rui_precision=False)

    # Ranking metrics
    ndcg_ks, dcg_ks = ndcg_at_ks(is_match, k_list=k_list, num_trgs=num_targets, method=1, include_dcg=True)
    alpha_ndcg_ks, alpha_dcg_ks = alpha_ndcg_at_ks(is_match_substring_2d, k_list=k_list, method=1,
                                                   alpha=0.5, include_dcg=True)
    ap_ks = average_precision_at_ks(is_match, k_list=k_list,
                                    num_predictions=num_predictions, num_trgs=num_targets)

    #for topk, precision_k, recall_k, f1_k, num_matches_k, num_predictions_k in \
    #        zip(k_list, precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks):
    for topk, precision_k, recall_k, f1_k, num_matches_k, num_predictions_k, ndcg_k, dcg_k, alpha_ndcg_k, alpha_dcg_k, ap_k in \
        zip(k_list, precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks, ndcg_ks, dcg_ks,
            alpha_ndcg_ks, alpha_dcg_ks, ap_ks):
        score_dict['precision@{}_{}'.format(topk, tag)].append(precision_k)
        score_dict['recall@{}_{}'.format(topk, tag)].append(recall_k)
        score_dict['f1_score@{}_{}'.format(topk, tag)].append(f1_k)
        score_dict['num_matches@{}_{}'.format(topk, tag)].append(num_matches_k)
        score_dict['num_predictions@{}_{}'.format(topk, tag)].append(num_predictions_k)
        score_dict['num_targets@{}_{}'.format(topk, tag)].append(num_targets)
        score_dict['AP@{}_{}'.format(topk, tag)].append(ap_k)
        score_dict['NDCG@{}_{}'.format(topk, tag)].append(ndcg_k)
        score_dict['AlphaNDCG@{}_{}'.format(topk, tag)].append(alpha_ndcg_k)

    score_dict['num_targets_{}'.format(tag)].append(num_targets)
    score_dict['num_predictions_{}'.format(tag)].append(num_predictions)
    return score_dict


def stem_str_2d_list(str_2dlist):
    # stem every word in a list of word list
    # str_list is a list of word list
    stemmed_str_2dlist = []
    for str_list in str_2dlist:
        stemmed_str_list = [stem_word_list(word_list) for word_list in str_list]
        stemmed_str_2dlist.append(stemmed_str_list)
    return stemmed_str_2dlist


def stem_str_list(str_list):
    # stem every word in a list of word list
    # str_list is a list of word list
    stemmed_str_list = []
    for word_list in str_list:
        stemmed_word_list = stem_word_list(word_list)
        stemmed_str_list.append(stemmed_word_list)
    return stemmed_str_list


def stem_word_list(word_list):
    return [stemmer.stem(w.strip().lower()) for w in word_list]


def check_valid_keyphrases(str_list, invalidate_unk=True):
    num_pred_seq = len(str_list)
    is_valid = np.zeros(num_pred_seq, dtype=bool)
    for i, word_list in enumerate(str_list):
        keep_flag = True

        if len(word_list) == 0:
            keep_flag = False

        for w in word_list:
            if invalidate_unk:
                if w == UNK_WORD or w == ',' or w == '.':
                    keep_flag = False
            else:
                if w == ',' or w == '.':
                    keep_flag = False
        is_valid[i] = keep_flag

    return is_valid


def compute_extra_one_word_seqs_mask(str_list):
    num_pred_seq = len(str_list)
    mask = np.zeros(num_pred_seq, dtype=bool)
    num_one_word_seqs = 0
    for i, word_list in enumerate(str_list):
        if len(word_list) == 1:
            num_one_word_seqs += 1
            if num_one_word_seqs > 1:
                mask[i] = False
                continue
        mask[i] = True
    return mask, num_one_word_seqs


def check_duplicate_keyphrases(keyphrase_str_list):
    """
    :param keyphrase_str_list: a 2d list of tokens
    :return: a boolean np array indicate, 1 = unique, 0 = duplicate
    """
    num_keyphrases = len(keyphrase_str_list)
    not_duplicate = np.ones(num_keyphrases, dtype=bool)
    keyphrase_set = set()
    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        if '_'.join(keyphrase_word_list) in keyphrase_set:
            not_duplicate[i] = False
        else:
            not_duplicate[i] = True
        keyphrase_set.add('_'.join(keyphrase_word_list))
    return not_duplicate


def check_present_keyphrases(src_str, keyphrase_str_list, match_by_str=False):
    """
    :param src_str: stemmed word list of source text
    :param keyphrase_str_list: stemmed list of word list
    :return:
    """
    num_keyphrases = len(keyphrase_str_list)
    is_present = np.zeros(num_keyphrases, dtype=bool)

    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        joined_keyphrase_str = ' '.join(keyphrase_word_list)

        if joined_keyphrase_str.strip() == "":  # if the keyphrase is an empty string
            is_present[i] = False
        else:
            if not match_by_str:  # match by word
                # check if it appears in source text
                match = False
                for src_start_idx in range(len(src_str) - len(keyphrase_word_list) + 1):
                    match = True
                    for keyphrase_i, keyphrase_w in enumerate(keyphrase_word_list):
                        src_w = src_str[src_start_idx + keyphrase_i]
                        if src_w != keyphrase_w:
                            match = False
                            break
                    if match:
                        break
                if match:
                    is_present[i] = True
                else:
                    is_present[i] = False
            else:  # match by str
                if joined_keyphrase_str in ' '.join(src_str):
                    is_present[i] = True
                else:
                    is_present[i] = False
    return is_present


def compute_match_result(trg_str_list, pred_str_list, type='exact', dimension=1):
    assert type in ['exact', 'sub'], "Right now only support exact matching and substring matching"
    assert dimension in [1, 2], "only support 1 or 2"
    num_pred_str = len(pred_str_list)
    num_trg_str = len(trg_str_list)
    if dimension == 1:
        is_match = np.zeros(num_pred_str, dtype=bool)
        for pred_idx, pred_word_list in enumerate(pred_str_list):
            joined_pred_word_list = ' '.join(pred_word_list)
            for trg_idx, trg_word_list in enumerate(trg_str_list):
                joined_trg_word_list = ' '.join(trg_word_list)
                if type == 'exact':
                    if joined_pred_word_list == joined_trg_word_list:
                        is_match[pred_idx] = True
                        break
                elif type == 'sub':
                    if joined_pred_word_list in joined_trg_word_list:
                        is_match[pred_idx] = True
                        break
    else:
        is_match = np.zeros((num_trg_str, num_pred_str), dtype=bool)
        for trg_idx, trg_word_list in enumerate(trg_str_list):
            joined_trg_word_list = ' '.join(trg_word_list)
            for pred_idx, pred_word_list in enumerate(pred_str_list):
                joined_pred_word_list = ' '.join(pred_word_list)
                if type == 'exact':
                    if joined_pred_word_list == joined_trg_word_list:
                        is_match[trg_idx][pred_idx] = True
                elif type == 'sub':
                    if joined_pred_word_list in joined_trg_word_list:
                        is_match[trg_idx][pred_idx] = True
    return is_match


def compute_classification_metrics_at_ks(
        is_match, num_predictions, num_trgs, k_list=[5, 10], meng_rui_precision=False
):
    """
    :param is_match: a boolean np array with size [num_predictions]
    :param predicted_list:
    :param true_list:
    :param topk:
    :return: {'precision@%d' % topk: precision_k, 'recall@%d' % topk: recall_k, 'f1_score@%d' % topk: f1, 'num_matches@%d': num_matches}
    """
    assert is_match.shape[0] == num_predictions
    # topk.sort()
    if num_predictions == 0:
        precision_ks = [0] * len(k_list)
        recall_ks = [0] * len(k_list)
        f1_ks = [0] * len(k_list)
        num_matches_ks = [0] * len(k_list)
        num_predictions_ks = [0] * len(k_list)
    else:
        num_matches = np.cumsum(is_match)
        num_predictions_ks = []
        num_matches_ks = []
        precision_ks = []
        recall_ks = []
        f1_ks = []
        for topk in k_list:
            if topk == 'M':
                topk = num_predictions
            elif topk == 'G':
                # topk = num_trgs
                if num_predictions < num_trgs:
                    topk = num_trgs
                else:
                    topk = num_predictions
            elif topk == 'O':
                topk = num_trgs

            if meng_rui_precision:
                if num_predictions > topk:
                    num_matches_at_k = num_matches[topk - 1]
                    num_predictions_at_k = topk
                else:
                    num_matches_at_k = num_matches[-1]
                    num_predictions_at_k = num_predictions
            else:
                if num_predictions > topk:
                    num_matches_at_k = num_matches[topk - 1]
                else:
                    num_matches_at_k = num_matches[-1]
                num_predictions_at_k = topk

            precision_k, recall_k, f1_k = compute_classification_metrics(
                num_matches_at_k, num_predictions_at_k, num_trgs
            )
            precision_ks.append(precision_k)
            recall_ks.append(recall_k)
            f1_ks.append(f1_k)
            num_matches_ks.append(num_matches_at_k)
            num_predictions_ks.append(num_predictions_at_k)
    return precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks


def compute_classification_metrics(num_matches, num_predictions, num_trgs):
    precision = compute_precision(num_matches, num_predictions)
    recall = compute_recall(num_matches, num_trgs)
    f1 = compute_f1(precision, recall)
    return precision, recall, f1


def compute_precision(num_matches, num_predictions):
    return num_matches / num_predictions if num_predictions > 0 else 0.0


def compute_recall(num_matches, num_trgs):
    return num_matches / num_trgs if num_trgs > 0 else 0.0


def compute_f1(precision, recall):
    return float(2 * (precision * recall)) / (precision + recall) if precision + recall > 0 else 0.0


def update_f1_dict(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed, k_list, f1_dict, tag):
    num_targets = len(trg_token_2dlist_stemmed)
    num_predictions = len(pred_token_2dlist_stemmed)
    is_match = compute_match_result(trg_token_2dlist_stemmed, pred_token_2dlist_stemmed,
                                    type='exact', dimension=1)
    # Classification metrics
    precision_ks, recall_ks, f1_ks, num_matches_ks, num_predictions_ks = \
        compute_classification_metrics_at_ks(is_match, num_predictions, num_targets, k_list=k_list,
                                             meng_rui_precision=False)
    for topk, precision_k, recall_k in zip(k_list, precision_ks, recall_ks):
        f1_dict['precision_sum@{}_{}'.format(topk, tag)] += precision_k
        f1_dict['recall_sum@{}_{}'.format(topk, tag)] += recall_k
    return f1_dict


def dcg_at_k(r, k, num_trgs, method=1):
    """
    Reference from https://www.kaggle.com/wendykan/ndcg-example and https://gist.github.com/bwhite/3726239
    Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    num_predictions = r.shape[0]
    if k == 'M':
        k = num_predictions
    elif k == 'G':
        #k = num_trgs
        if num_predictions < num_trgs:
            k = num_trgs
        else:
            k = num_predictions
    elif k == 'O':
        k = num_trgs

    if num_predictions == 0:
        dcg = 0.
    else:
        if num_predictions > k:
            r = r[:k]
            num_predictions = k
        if method == 0:
            dcg = r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            discounted_gain = r / np.log2(np.arange(2, r.size + 2))
            dcg = np.sum(discounted_gain)
        else:
            raise ValueError('method must be 0 or 1.')
    return dcg


def dcg_at_ks(r, k_list, num_trgs, method=1):
    num_predictions = r.shape[0]
    if num_predictions == 0:
        dcg_array = np.array([0] * len(k_list))
    else:
        k_max = -1
        for k in k_list:
            if k == 'M':
                k = num_predictions
            elif k == 'G':
                #k = num_trgs
                if num_predictions < num_trgs:
                    k = num_trgs
                else:
                    k = num_predictions
            elif k == 'O':
                k = num_trgs

            if k > k_max:
                k_max = k
        if num_predictions > k_max:
            r = r[:k_max]
            num_predictions = k_max
        if method == 1:
            discounted_gain = r / np.log2(np.arange(2, r.size + 2))
            dcg = np.cumsum(discounted_gain)
            return_indices = []
            for k in k_list:
                if k == 'M':
                    k = num_predictions
                elif k == 'G':
                    #k = num_trgs
                    if num_predictions < num_trgs:
                        k = num_trgs
                    else:
                        k = num_predictions
                elif k == 'O':
                    k = num_trgs

                return_indices.append((k - 1) if k <= num_predictions else (num_predictions - 1))
            return_indices = np.array(return_indices, dtype=int)
            dcg_array = dcg[return_indices]
        else:
            raise ValueError('method must 1.')
    return dcg_array


def ndcg_at_k(r, k, num_trgs, method=1, include_dcg=False):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    if r.shape[0] == 0:
        ndcg = 0.0
        dcg = 0.0
    else:
        dcg_max = dcg_at_k(np.array(sorted(r, reverse=True)), k, num_trgs, method)
        if dcg_max <= 0.0:
            ndcg = 0.0
        else:
            dcg = dcg_at_k(r, k, num_trgs, method)
            ndcg = dcg / dcg_max
    if include_dcg:
        return ndcg, dcg
    else:
        return ndcg


def ndcg_at_ks(r, k_list, num_trgs, method=1, include_dcg=False):
    if r.shape[0] == 0:
        ndcg_array = [0.0] * len(k_list)
        dcg_array = [0.0] * len(k_list)
    else:
        dcg_array = dcg_at_ks(r, k_list, num_trgs, method)
        ideal_r = np.array(sorted(r, reverse=True))
        dcg_max_array = dcg_at_ks(ideal_r, k_list, num_trgs, method)
        ndcg_array = dcg_array / dcg_max_array
        ndcg_array = np.nan_to_num(ndcg_array)
    if include_dcg:
        return ndcg_array, dcg_array
    else:
        return ndcg_array


def alpha_dcg_at_k(r_2d, k, method=1, alpha=0.5):
    """
    :param r_2d: 2d relevance np array, shape: [num_trg_str, num_pred_str]
    :param k:
    :param method:
    :param alpha:
    :return:
    """
    if r_2d.shape[-1] == 0:
        alpha_dcg = 0.0
    else:
        # convert r_2d to gain vector
        num_trg_str, num_pred_str = r_2d.shape
        if k == 'M':
            k = num_pred_str
        elif k == 'G':
            #k = num_trg_str
            if num_pred_str < num_trg_str:
                k = num_trg_str
            else:
                k = num_pred_str
        elif k == 'O':
            k = num_trg_str
                
        if num_pred_str > k:
            num_pred_str = k
        gain_vector = np.zeros(num_pred_str)
        one_minus_alpha_vec = np.ones(num_trg_str) * (1 - alpha)  # [num_trg_str]
        cum_r = np.concatenate((np.zeros((num_trg_str, 1)), np.cumsum(r_2d, axis=1)), axis=1)
        for j in range(num_pred_str):
            gain_vector[j] = np.dot(r_2d[:, j], np.power(one_minus_alpha_vec, cum_r[:, j]))
        alpha_dcg = dcg_at_k(gain_vector, k, num_trg_str, method)
    return alpha_dcg


def alpha_dcg_at_ks(r_2d, k_list, method=1, alpha=0.5):
    """
    :param r_2d: 2d relevance np array, shape: [num_trg_str, num_pred_str]
    :param ks:
    :param method:
    :param alpha:
    :return:
    """
    if r_2d.shape[-1] == 0:
        return [0.0] * len(k_list)
    # convert r_2d to gain vector
    num_trg_str, num_pred_str = r_2d.shape
    # k_max = max(k_list)
    k_max = -1
    for k in k_list:
        if k == 'M':
            k = num_pred_str
        elif k == 'G':
            #k = num_trg_str
            if num_pred_str < num_trg_str:
                k = num_trg_str
            else:
                k = num_pred_str
        elif k == 'O':
            k = num_trg_str

        if k > k_max:
            k_max = k
    if num_pred_str > k_max:
        num_pred_str = k_max
    gain_vector = np.zeros(num_pred_str)
    one_minus_alpha_vec = np.ones(num_trg_str) * (1 - alpha)  # [num_trg_str]
    cum_r = np.concatenate((np.zeros((num_trg_str, 1)), np.cumsum(r_2d, axis=1)), axis=1)
    for j in range(num_pred_str):
        gain_vector[j] = np.dot(r_2d[:, j], np.power(one_minus_alpha_vec, cum_r[:, j]))
    return dcg_at_ks(gain_vector, k_list, num_trg_str, method)


def alpha_ndcg_at_k(r_2d, k, method=1, alpha=0.5, include_dcg=False):
    """
    :param r_2d: 2d relevance np array, shape: [num_trg_str, num_pred_str]
    :param k:
    :param method:
    :param alpha:
    :return:
    """
    if r_2d.shape[-1] == 0:
        alpha_ndcg = 0.0
        alpha_dcg = 0.0
    else:
        num_trg_str, num_pred_str = r_2d.shape
        if k == 'M':
            k = num_pred_str
        elif k == 'G':
            #k = num_trg_str
            if num_pred_str < num_trg_str:
                k = num_trg_str
            else:
                k = num_pred_str
        elif k == 'O':
            k = num_trg_str
                
        # convert r to gain vector
        alpha_dcg = alpha_dcg_at_k(r_2d, k, method, alpha)
        # compute alpha_dcg_max
        r_2d_ideal = compute_ideal_r_2d(r_2d, k, alpha)
        alpha_dcg_max = alpha_dcg_at_k(r_2d_ideal, k, method, alpha)
        if alpha_dcg_max <= 0.0:
            alpha_ndcg = 0.0
        else:
            alpha_ndcg = alpha_dcg / alpha_dcg_max
            alpha_ndcg = np.nan_to_num(alpha_ndcg)
    if include_dcg:
        return alpha_ndcg, alpha_dcg
    else:
        return alpha_ndcg


def alpha_ndcg_at_ks(r_2d, k_list, method=1, alpha=0.5, include_dcg=False):
    """
    :param r_2d: 2d relevance np array, shape: [num_trg_str, num_pred_str]
    :param k:
    :param method:
    :param alpha:
    :return:
    """
    if r_2d.shape[-1] == 0:
        alpha_ndcg_array = [0] * len(k_list)
        alpha_dcg_array = [0] * len(k_list)
    else:
        # k_max = max(k_list)
        num_trg_str, num_pred_str = r_2d.shape
        k_max = -1
        for k in k_list:
            if k == 'M':
                k = num_pred_str
            elif k == 'G':
                #k = num_trg_str
                if num_pred_str < num_trg_str:
                    k = num_trg_str
                else:
                    k = num_pred_str
            elif k == 'O':
                k = num_trg_str
                    
            if k > k_max:
                k_max = k
        # convert r to gain vector
        alpha_dcg_array = alpha_dcg_at_ks(r_2d, k_list, method, alpha)
        # compute alpha_dcg_max
        r_2d_ideal = compute_ideal_r_2d(r_2d, k_max, alpha)
        alpha_dcg_max_array = alpha_dcg_at_ks(r_2d_ideal, k_list, method, alpha)
        alpha_ndcg_array = alpha_dcg_array / alpha_dcg_max_array
        alpha_ndcg_array = np.nan_to_num(alpha_ndcg_array)
    if include_dcg:
        return alpha_ndcg_array, alpha_dcg_array
    else:
        return alpha_ndcg_array


def compute_ideal_r_2d(r_2d, k, alpha=0.5):
    num_trg_str, num_pred_str = r_2d.shape
    one_minus_alpha_vec = np.ones(num_trg_str) * (1 - alpha)  # [num_trg_str]
    cum_r_vector = np.zeros((num_trg_str))
    ideal_ranking = []
    greedy_depth = min(num_pred_str, k)
    for rank in range(greedy_depth):
        gain_vector = np.zeros(num_pred_str)
        for j in range(num_pred_str):
            if j in ideal_ranking:
                gain_vector[j] = -1000.0
            else:
                gain_vector[j] = np.dot(r_2d[:, j], np.power(one_minus_alpha_vec, cum_r_vector))
        max_idx = np.argmax(gain_vector)
        ideal_ranking.append(max_idx)
        current_relevance_vector = r_2d[:, max_idx]
        cum_r_vector = cum_r_vector + current_relevance_vector
    return r_2d[:, np.array(ideal_ranking, dtype=int)]


def average_precision(r, num_predictions, num_trgs):
    if num_predictions == 0 or num_trgs == 0:
        return 0
    r_cum_sum = np.cumsum(r, axis=0)
    precision_sum = sum([compute_precision(r_cum_sum[k], k + 1) for k in range(num_predictions) if r[k]])
    '''
    precision_sum = 0
    for k in range(num_predictions):
        if r[k] is False:
            continue
        else:
            precision_k = precision(r_cum_sum[k], k+1)
            precision_sum += precision_k
    '''
    return precision_sum / num_trgs


def average_precision_at_k(r, k, num_predictions, num_trgs):
    if k == 'M':
        k = num_predictions
    elif k == 'G':
        #k = num_trgs
        if num_predictions < num_trgs:
            k = num_trgs
        else:
            k = num_predictions
    elif k == 'O':
        k = num_trgs

    if k < num_predictions:
        num_predictions = k
        r = r[:k]
    return average_precision(r, num_predictions, num_trgs)


def average_precision_at_ks(r, k_list, num_predictions, num_trgs):
    if num_predictions == 0 or num_trgs == 0:
        return [0] * len(k_list)
    # k_max = max(k_list)
    k_max = -1
    for k in k_list:
        if k == 'M':
            k = num_predictions
        elif k == 'G':
            #k = num_trgs
            if num_predictions < num_trgs:
                k = num_trgs
            else:
                k = num_predictions
        elif k == 'O':
            k = num_trgs
        if k > k_max:
            k_max = k
    if num_predictions > k_max:
        num_predictions = k_max
        r = r[:num_predictions]
    r_cum_sum = np.cumsum(r, axis=0)
    precision_array = [compute_precision(r_cum_sum[k], k + 1) * r[k] for k in range(num_predictions)]
    precision_cum_sum = np.cumsum(precision_array, axis=0)
    average_precision_array = precision_cum_sum / num_trgs
    return_indices = []
    for k in k_list:
        if k == 'M':
            k = num_predictions
        elif k == 'G':
            #k = num_trgs
            if num_predictions < num_trgs:
                k = num_trgs
            else:
                k = num_predictions
        elif k == 'O':
            k = num_trgs
            
        return_indices.append( (k-1) if k <= num_predictions else (num_predictions-1) )
    return_indices = np.array(return_indices, dtype=int)
    return average_precision_array[return_indices]



def filter_prediction(disable_valid_filter, disable_extra_one_word_filter, pred_token_2dlist_stemmed):
    """
    Remove the duplicate predictions, can optionally remove invalid predictions and extra one word predictions
    :param disable_valid_filter:
    :param disable_extra_one_word_filter:
    :param pred_token_2dlist_stemmed:
    :param pred_token_2d_list:
    :return:
    """
    num_predictions = len(pred_token_2dlist_stemmed)
    is_unique_mask = check_duplicate_keyphrases(pred_token_2dlist_stemmed)  # boolean array, 1=unqiue, 0=duplicate
    pred_filter = is_unique_mask
    if not disable_valid_filter:
        is_valid_mask = check_valid_keyphrases(pred_token_2dlist_stemmed)
        pred_filter = pred_filter * is_valid_mask
    if not disable_extra_one_word_filter:
        extra_one_word_seqs_mask, num_one_word_seqs = compute_extra_one_word_seqs_mask(pred_token_2dlist_stemmed)
        pred_filter = pred_filter * extra_one_word_seqs_mask
    filtered_stemmed_pred_str_list = [word_list for word_list, is_keep in
                                      zip(pred_token_2dlist_stemmed, pred_filter) if
                                      is_keep]
    num_duplicated_predictions = num_predictions - np.sum(is_unique_mask)
    return filtered_stemmed_pred_str_list, num_duplicated_predictions, is_unique_mask


def find_unique_target(trg_token_2dlist_stemmed):
    """
    Remove the duplicate targets
    :param trg_token_2dlist_stemmed:
    :return:
    """
    num_trg = len(trg_token_2dlist_stemmed)
    is_unique_mask = check_duplicate_keyphrases(trg_token_2dlist_stemmed)  # boolean array, 1=unqiue, 0=duplicate
    trg_filter = is_unique_mask
    filtered_stemmed_trg_str_list = [word_list for word_list, is_keep in
                                     zip(trg_token_2dlist_stemmed, trg_filter) if
                                     is_keep]
    num_duplicated_trg = num_trg - np.sum(is_unique_mask)
    return filtered_stemmed_trg_str_list, num_duplicated_trg


def separate_present_absent_by_source(src_token_list_stemmed, keyphrase_token_2dlist_stemmed, match_by_str):
    is_present_mask = check_present_keyphrases(src_token_list_stemmed, keyphrase_token_2dlist_stemmed, match_by_str)
    present_keyphrase_token2dlist = []
    absent_keyphrase_token2dlist = []
    for keyphrase_token_list, is_present in zip(keyphrase_token_2dlist_stemmed, is_present_mask):
        if is_present:
            present_keyphrase_token2dlist.append(keyphrase_token_list)
        else:
            absent_keyphrase_token2dlist.append(keyphrase_token_list)
    return present_keyphrase_token2dlist, absent_keyphrase_token2dlist, is_present_mask


def process_input_ks(ks):
    ks_list = []
    for k in ks:
        if k != 'M' and k != 'G' and k != 'O':
            k = int(k)
        ks_list.append(k)
    return ks_list


def report_ranking_scores(score_dict, topk_list, present_tag):
    output_str = ""
    result_list = []
    field_list = []
    for topk in topk_list:
        map_k = sum(score_dict['AP@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['AP@{}_{}'.format(topk, present_tag)])
        # avg_dcg_k = sum(score_dict['DCG@{}_{}'.format(topk, present_tag)]) / len(
        #    score_dict['DCG@{}_{}'.format(topk, present_tag)])
        avg_ndcg_k = sum(score_dict['NDCG@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['NDCG@{}_{}'.format(topk, present_tag)])
        # avg_alpha_dcg_k = sum(score_dict['AlphaDCG@{}_{}'.format(topk, present_tag)]) / len(
        #    score_dict['AlphaDCG@{}_{}'.format(topk, present_tag)])
        avg_alpha_ndcg_k = sum(score_dict['AlphaNDCG@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['AlphaNDCG@{}_{}'.format(topk, present_tag)])
        output_str += (
            "Begin==================Ranking metrics {}@{}==================Begin\n".format(present_tag, topk))
        output_str += "\tMAP@{}={:.5}\tNDCG@{}={:.5}\tAlphaNDCG@{}={:.5}\n".format(topk, map_k, topk, avg_ndcg_k, topk,
                                                                                   avg_alpha_ndcg_k)
        field_list += ['MAP@{}_{}'.format(topk, present_tag), 'avg_NDCG@{}_{}'.format(topk, present_tag),
                       'AlphaNDCG@{}_{}'.format(topk, present_tag)]
        result_list += [map_k, avg_ndcg_k, avg_alpha_ndcg_k]

    return output_str, field_list, result_list


def main(predictions, exp_path, result_file_suffix, k_list=[5, 'M']):
    source, target, preds = [], [], []
    for hyp, ref, src in zip(predictions[0], predictions[1],
                             predictions[2]):
        source.append(src)
        target.append(';'.join(ref))
        preds.append(';'.join(hyp))

    score_dict = defaultdict(list)
    k_list = process_input_ks(k_list)
    topk_dict = {'present': k_list, 'absent': k_list, 'all': k_list}

    total_num_src = 0
    total_num_src_with_present_keyphrases = 0
    total_num_src_with_absent_keyphrases = 0
    total_num_unique_predictions = 0
    total_num_present_filtered_predictions = 0
    total_num_present_unique_targets = 0
    total_num_absent_filtered_predictions = 0
    total_num_absent_unique_targets = 0
    max_unique_targets = 0

    sum_incorrect_fraction_for_identifying_present = 0
    sum_incorrect_fraction_for_identifying_absent = 0

    predicted_keyphrases = []
    for data_idx, (src_l, trg_l, pred_l) in \
            enumerate(tqdm(zip(source, target, preds),
                           total=len(source), desc='Evaluating...')):
        total_num_src += 1
        # convert the str to token list
        pred_str_list = pred_l.strip().split(';')
        pred_str_list = pred_str_list[:200]
        pred_token_2dlist = [pred_str.strip().split(' ') for pred_str in pred_str_list]
        trg_str_list = trg_l.strip().split(';')
        trg_token_2dlist = [trg_str.strip().split(' ') for trg_str in trg_str_list]

        src_l = src_l.strip()
        if TITLE_SEP in src_l:
            [title, context] = src_l.strip().split(TITLE_SEP)
        else:
            title = ""
            context = src_l
        src_token_list = title.strip().split(' ') + context.strip().split(' ')

        num_predictions = len(pred_str_list)

        # perform stemming
        stemmed_src_token_list = stem_word_list(src_token_list)

        stemmed_trg_token_2dlist = stem_str_list(trg_token_2dlist)
        # TODO: test stemmed_trg_variation_token_3dlist

        stemmed_pred_token_2dlist = stem_str_list(pred_token_2dlist)

        # Filter out duplicate, invalid, and extra one word predictions
        filtered_stemmed_pred_token_2dlist, num_duplicated_predictions, is_unique_mask = filter_prediction(
            INVALIDATE_UNK, DISABLE_EXTRA_ONE_WORD_FILTER, stemmed_pred_token_2dlist
        )
        total_num_unique_predictions += (num_predictions - num_duplicated_predictions)
        num_filtered_predictions = len(filtered_stemmed_pred_token_2dlist)

        # Remove duplicated targets
        unique_stemmed_trg_token_2dlist, num_duplicated_trg = find_unique_target(stemmed_trg_token_2dlist)
        # unique_stemmed_trg_token_2dlist = stemmed_trg_token_2dlist
        num_unique_targets = len(unique_stemmed_trg_token_2dlist)
        # max_unique_targets += (num_trg - num_duplicated_trg)

        if num_unique_targets > max_unique_targets:
            max_unique_targets = num_unique_targets

        # separate present and absent keyphrases
        present_filtered_stemmed_pred_token_2dlist, absent_filtered_stemmed_pred_token_2dlist, is_present_mask = \
            separate_present_absent_by_source(stemmed_src_token_list, filtered_stemmed_pred_token_2dlist, False)
        present_unique_stemmed_trg_token_2dlist, absent_unique_stemmed_trg_token_2dlist, _ = \
            separate_present_absent_by_source(stemmed_src_token_list, unique_stemmed_trg_token_2dlist, False)

        # save the predicted keyphrases
        filtered_pred_token_2dlist = [kp_tokens for kp_tokens, is_unique
                                      in zip(pred_token_2dlist, is_unique_mask) if is_unique]
        result = {'id': data_idx, 'present': [], 'absent': []}
        for kp_tokens, is_present in zip(filtered_pred_token_2dlist, is_present_mask):
            if is_present:
                result['present'] += [' '.join(kp_tokens)]
            else:
                result['absent'] += [' '.join(kp_tokens)]
        predicted_keyphrases.append(result)

        total_num_present_filtered_predictions += len(present_filtered_stemmed_pred_token_2dlist)
        total_num_present_unique_targets += len(present_unique_stemmed_trg_token_2dlist)
        total_num_absent_filtered_predictions += len(absent_filtered_stemmed_pred_token_2dlist)
        total_num_absent_unique_targets += len(absent_unique_stemmed_trg_token_2dlist)
        if len(present_unique_stemmed_trg_token_2dlist) > 0:
            total_num_src_with_present_keyphrases += 1
        if len(absent_unique_stemmed_trg_token_2dlist) > 0:
            total_num_src_with_absent_keyphrases += 1

        # compute all the metrics and update the score_dict
        score_dict = update_score_dict(unique_stemmed_trg_token_2dlist, filtered_stemmed_pred_token_2dlist,
                                       topk_dict['all'], score_dict, 'all')
        # compute all the metrics and update the score_dict for present keyphrase
        score_dict = update_score_dict(present_unique_stemmed_trg_token_2dlist,
                                       present_filtered_stemmed_pred_token_2dlist,
                                       topk_dict['present'], score_dict, 'present')
        # compute all the metrics and update the score_dict for absent keyphrase
        score_dict = update_score_dict(absent_unique_stemmed_trg_token_2dlist,
                                       absent_filtered_stemmed_pred_token_2dlist,
                                       topk_dict['absent'], score_dict, 'absent')

    if len(predicted_keyphrases) > 0:
        with open('{}_predictions.txt'.format(args.file_prefix), 'w') as fw:
            for item in predicted_keyphrases:
                fw.write(json.dumps(item) + '\n')

    total_num_unique_targets = total_num_present_unique_targets + total_num_absent_unique_targets
    total_num_filtered_predictions = total_num_present_filtered_predictions + total_num_absent_filtered_predictions

    result_txt_str = ""

    # report global statistics
    result_txt_str += (
            'Total #samples: %d\t # samples with present keyphrases: %d\t # samples with absent keyphrases: %d\n' % (
        total_num_src, total_num_src_with_present_keyphrases, total_num_src_with_absent_keyphrases))
    result_txt_str += ('Max. unique targets per src: %d\n' % (max_unique_targets))
    result_txt_str += ('Total #unique predictions: %d\n' % total_num_unique_predictions)

    # report statistics and scores for all predictions and targets
    result_txt_str_all, field_list_all, result_list_all = report_stat_and_scores(
        total_num_filtered_predictions, total_num_unique_targets, total_num_src, score_dict,
        topk_dict['all'], 'all'
    )
    result_txt_str_present, field_list_present, result_list_present = report_stat_and_scores(
        total_num_present_filtered_predictions, total_num_present_unique_targets, total_num_src, score_dict,
        topk_dict['present'], 'present'
    )
    result_txt_str_absent, field_list_absent, result_list_absent = report_stat_and_scores(
        total_num_absent_filtered_predictions, total_num_absent_unique_targets, total_num_src, score_dict,
        topk_dict['absent'], 'absent'
    )
    result_txt_str += (result_txt_str_all + result_txt_str_present + result_txt_str_absent)
    field_list = field_list_all + field_list_present + field_list_absent
    result_list = result_list_all + result_list_present + result_list_absent

    # Write to files
    # topk_dict = {'present': [5, 10, 'M'], 'absent': [5, 10, 50, 'M'], 'all': [5, 10, 'M']}
    results_txt_file = open(os.path.join(exp_path, "results_log_{}.txt".format(result_file_suffix)), "w")

    result_txt_str += "===================================Separation====================================\n"
    result_txt_str += "Avg error fraction for identifying present keyphrases: {:.5}\n".format(
        sum_incorrect_fraction_for_identifying_present / total_num_src)
    result_txt_str += "Avg error fraction for identifying absent keyphrases: {:.5}\n".format(
        sum_incorrect_fraction_for_identifying_absent / total_num_src)

    # Report MAE on lengths
    result_txt_str += "===================================MAE stat====================================\n"

    num_targets_present_array = np.array(score_dict['num_targets_present'])
    num_predictions_present_array = np.array(score_dict['num_predictions_present'])
    num_targets_absent_array = np.array(score_dict['num_targets_absent'])
    num_predictions_absent_array = np.array(score_dict['num_predictions_absent'])

    all_mae = mae(num_targets_present_array + num_targets_absent_array,
                  num_predictions_present_array + num_predictions_absent_array)
    present_mae = mae(num_targets_present_array, num_predictions_present_array)
    absent_mae = mae(num_targets_absent_array, num_predictions_absent_array)

    result_txt_str += "MAE on keyphrase numbers (all): {:.5}\n".format(all_mae)
    result_txt_str += "MAE on keyphrase numbers (present): {:.5}\n".format(present_mae)
    result_txt_str += "MAE on keyphrase numbers (absent): {:.5}\n".format(absent_mae)

    results_txt_file.write(result_txt_str)
    results_txt_file.close()

    return


def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())


def mae(a, b):
    return (np.abs(a - b)).mean()


def report_stat_and_scores(total_num_filtered_predictions, num_unique_trgs, total_num_src, score_dict, topk_list,
                           present_tag):
    result_txt_str = "===================================%s====================================\n" % (present_tag)
    result_txt_str += "#predictions after filtering: %d\t #predictions after filtering per src:%.3f\n" % \
                      (total_num_filtered_predictions, total_num_filtered_predictions / total_num_src)
    result_txt_str += "#unique targets: %d\t #unique targets per src:%.3f\n" % \
                      (num_unique_trgs, num_unique_trgs / total_num_src)

    classification_output_str, classification_field_list, classification_result_list = report_classification_scores(
        score_dict, topk_list, present_tag
    )
    result_txt_str += classification_output_str
    field_list = classification_field_list
    result_list = classification_result_list

    ranking_output_str, ranking_field_list, ranking_result_list = report_ranking_scores(
        score_dict, topk_list, present_tag
    )
    result_txt_str += ranking_output_str
    field_list += ranking_field_list
    result_list += ranking_result_list
    
    return result_txt_str, field_list, result_list


def report_classification_scores(score_dict, topk_list, present_tag):
    output_str = ""
    result_list = []
    field_list = []

    #print(score_dict)
    #exit()
    for topk in topk_list:
        total_predictions_k = sum(score_dict['num_predictions@{}_{}'.format(topk, present_tag)])
        total_targets_k = sum(score_dict['num_targets@{}_{}'.format(topk, present_tag)])
        total_num_matches_k = sum(score_dict['num_matches@{}_{}'.format(topk, present_tag)])
        # Compute the micro averaged recall, precision and F-1 score
        micro_avg_precision_k, micro_avg_recall_k, micro_avg_f1_score_k = compute_classification_metrics(
            total_num_matches_k, total_predictions_k, total_targets_k)
        # Compute the macro averaged recall, precision and F-1 score
        macro_avg_precision_k = sum(score_dict['precision@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['precision@{}_{}'.format(topk, present_tag)])
        macro_avg_recall_k = sum(score_dict['recall@{}_{}'.format(topk, present_tag)]) / len(
            score_dict['recall@{}_{}'.format(topk, present_tag)])
        macro_avg_f1_score_k = (2 * macro_avg_precision_k * macro_avg_recall_k) / \
                               (macro_avg_precision_k + macro_avg_recall_k) if \
            (macro_avg_precision_k + macro_avg_recall_k) > 0 else 0.0
        output_str += (
            "Begin===============classification metrics {}@{}===============Begin\n".format(present_tag, topk))
        output_str += ("#target: {}, #predictions: {}, #corrects: {}\n".format(total_predictions_k, total_targets_k,
                                                                               total_num_matches_k))
        output_str += "Micro:\tP@{}={:.5}\tR@{}={:.5}\tF1@{}={:.5}\n".format(topk, micro_avg_precision_k, topk,
                                                                             micro_avg_recall_k, topk,
                                                                             micro_avg_f1_score_k)
        output_str += "Macro:\tP@{}={:.5}\tR@{}={:.5}\tF1@{}={:.5}\n".format(topk, macro_avg_precision_k, topk,
                                                                             macro_avg_recall_k, topk,
                                                                             macro_avg_f1_score_k)
        field_list += ['macro_avg_p@{}_{}'.format(topk, present_tag), 'macro_avg_r@{}_{}'.format(topk, present_tag),
                       'macro_avg_f1@{}_{}'.format(topk, present_tag)]
        result_list += [macro_avg_precision_k, macro_avg_recall_k, macro_avg_f1_score_k]
    return output_str, field_list, result_list


def run_eval(predictions, dir_name, file_suffix, k_list):
    main(predictions, dir_name, file_suffix, k_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default=None, help="Prefix of the filenames")
    parser.add_argument('--is_validation', type=bool, default=False, 
                        help="Eval on the validation set")
    parser.add_argument('--src_file', type=str, default=None, help="Path of the source file")
    parser.add_argument('--pred_file', type=str, default=None, help="Path of the prediction file")

    parser.add_argument('--file_prefix', type=str, required=True, help="Prefix of the prediction file")
    parser.add_argument('--tgt_dir', type=str, required=True, help="Path of target directory")
    parser.add_argument('--log_file', type=str, required=True, help="Path of the log file")
    parser.add_argument('--k_list', nargs='+', default=[5, 'M'], help='K values for evaluation')
    args = parser.parse_args()

    hypotheses = []
    references = []
    sources = []

    if args.src_file and args.pred_file:
        with open(args.src_file) as f1, open(args.pred_file) as f2:
            for source, pred in zip(f1, f2):
                ex = json.loads(source.strip())
                sources.append(ex['src'].lower())
                references.append(ex['tgt'].lower().split(' {} '.format(KP_SEP)))
                preds = pred.split(' {} '.format(KP_SEP))
                preds = [p.replace('[ digit ]', '[digit]') for p in preds]
                hypotheses.append(preds)

    elif args.src_dir:
        if args.is_validation:
            source_file_name = '{}/valid.source'.format(args.src_dir)
            target_file_name = '{}/valid.target'.format(args.src_dir)
            hyp_file_name = '{}_hypotheses.txt'.format(args.file_prefix)
        else:
            source_file_name = '{}/test.source'.format(args.src_dir)
            target_file_name = '{}/test.target'.format(args.src_dir)
            hyp_file_name = '{}_hypotheses.txt'.format(args.file_prefix)
        with open(source_file_name) as f1, \
             open(hyp_file_name) as f2, \
             open(target_file_name) as f3:
            for source, candidate, gold in zip(f1, f2, f3):
                sources.append(source.strip().lower())

                refs = gold.lower().split(KP_SEP)
                mod_refs = []
                for r in refs:
                    r = r.strip()
                    r = r.replace(' ##', '')
                    r = r.replace('[ digit ]', '[digit]')
                    mod_refs.append(r)
                references.append(mod_refs)

                preds = candidate.lower().split(KP_SEP)
                mod_preds = []
                for p in preds:
                    p = p.strip()
                    p = p.replace(' ##', '')
                    p = p.replace('[ digit ]', '[digit]')
                    mod_preds.append(p)
                hypotheses.append(mod_preds)

    else:
        raise ValueError('Unknown output format')

    run_eval((hypotheses, references, sources), args.tgt_dir, args.log_file, args.k_list)
