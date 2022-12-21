# coding=utf-8
# The MIT License (MIT)

# Copyright (c) Microsoft Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Tokenization classes for MiniLM."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from transformers.tokenization_bert import BertTokenizer, whitespace_tokenize
from transformers.tokenization_roberta import RobertaTokenizer

logger = logging.getLogger(__name__)

BERT_DOWNLOAD_URL_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}
VOCAB_FILES_NAMES_ROBERTA = {'vocab_file': 'vocab.json'}

def join_path(model_name):    
    return BERT_DOWNLOAD_URL_PREFIX + "/" + model_name + "/" + VOCAB_FILES_NAMES['vocab_file']


def join_path_roberta(model_name):    
    return BERT_DOWNLOAD_URL_PREFIX + "/" + model_name + "/" + VOCAB_FILES_NAMES_ROBERTA['vocab_file']


PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
        {
            'bert-tiny-uncased': join_path("google/bert_uncased_L-2_H-128_A-2"),
            'bert-mini-uncased': join_path("google/bert_uncased_L-4_H-256_A-4"),
            'bert-small-uncased': join_path("google/bert_uncased_L-4_H-512_A-8"),
            'bert-medium-uncased': join_path("google/bert_uncased_L-8_H-512_A-8"),
            'scibert_scivocab_uncased': join_path("allenai/scibert_scivocab_uncased")
        }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'bert-tiny-uncased': 512,
    'bert-mini-uncased': 512,
    'bert-small-uncased': 512,
    'bert-medium-uncased': 512,
    'scibert_scivocab_uncased': 512
}

# based on models provided at https://huggingface.co/google
for l in [2, 4, 6, 8, 10, 12]:
    for h in [128, 256, 512, 768]:
        a = h // 64
        model_name_suffix = 'bert_uncased_L-{}_H-{}_A-{}'.format(l, h, a)
        model_name = 'google/{}'.format(model_name_suffix)
        PRETRAINED_VOCAB_FILES_MAP['vocab_file'][model_name_suffix] = join_path(model_name)
        PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES[model_name_suffix] = 512


class xBertTokenizer(BertTokenizer):
    r"""
    Constructs a MinilmTokenizer.
    :class:`~transformers.MinilmTokenizer` is identical to BertTokenizer and runs end-to-end tokenization: punctuation splitting + wordpiece
    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
        do_lower_case: Whether to lower case the input. Only has an effect when do_wordpiece_only=False
        do_basic_tokenize: Whether to do basic tokenization before wordpiece.
        max_len: An artificial maximum length to truncate tokenized sequences to; Effective maximum length is always the
            minimum of this value (if specified) and the underlying BERT model's sequence length.
        never_split: List of tokens which will never be split during tokenization. Only has an effect when
            do_wordpiece_only=False
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES


class xBertTokenizerRoberta(RobertaTokenizer):
    r"""
    Constructs a MinilmTokenizer.
    :class:`~transformers.MinilmTokenizer` is identical to BertTokenizer and runs end-to-end tokenization: punctuation splitting + wordpiece
    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
        do_lower_case: Whether to lower case the input. Only has an effect when do_wordpiece_only=False
        do_basic_tokenize: Whether to do basic tokenization before wordpiece.
        max_len: An artificial maximum length to truncate tokenized sequences to; Effective maximum length is always the
            minimum of this value (if specified) and the underlying BERT model's sequence length.
        never_split: List of tokens which will never be split during tokenization. Only has an effect when
            do_wordpiece_only=False
    """

    vocab_files_names = VOCAB_FILES_NAMES_ROBERTA
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES


class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)
