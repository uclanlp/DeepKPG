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
""" MiniLM model configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

BERT_DOWNLOAD_URL_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
CONFIG_FILE = "config.json"


def join_path(model_name):
    return BERT_DOWNLOAD_URL_PREFIX + "/" + model_name + "/" + CONFIG_FILE


xBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-tiny-uncased': join_path("google/bert_uncased_L-2_H-128_A-2"),
    'bert-mini-uncased': join_path("google/bert_uncased_L-4_H-256_A-4"),
    'bert-small-uncased': join_path("google/bert_uncased_L-4_H-512_A-8"),
    'bert-medium-uncased': join_path("google/bert_uncased_L-8_H-512_A-8"),
    'scibert_scivocab_uncased': join_path("allenai/scibert_scivocab_uncased")
}

# based on models provided at https://huggingface.co/google
for l in [2, 4, 6, 8, 10, 12]:
    for h in [128, 256, 512, 768]:
        a = h // 64
        model_name_suffix = 'bert_uncased_L-{}_H-{}_A-{}'.format(l, h, a)
        model_name = 'google/{}'.format(model_name_suffix)
        xBERT_PRETRAINED_CONFIG_ARCHIVE_MAP[model_name_suffix] = join_path(model_name)


class xBertConfig(PretrainedConfig):
    r"""
        :class:`~transformers.MinilmConfig` is the configuration class to store the configuration of a
        `MinilmModel`.
        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `MiniLMModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `MiniLMModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = xBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 pad_token_id=0,
                 **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
