# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

# from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function
from torch.nn.parameter import Parameter
import argparse
import collections
import logging
import json
import math
import os
import gensim
import random
import pickle
from tqdm import tqdm, trange
import pandas as pd
from visdom import Visdom
import re
# import jieba
import time
#from zhon.hanzi import punctuation
# from focalloss import *
#from losses import *
# from radam import *
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel, BertConfig, BertForMultipleChoice
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logger = logging.getLogger()
logger.setLevel('DEBUG')
BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler() # 输出到控制台的handler
chlr.setFormatter(formatter)
chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
fhlr = logging.FileHandler('newsa+.log') # 输出到文件的handler
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)
 

idiom_vocab = eval(open('idiomList.txt').readline())
idiom_vocab = {each: i for i, each in enumerate(idiom_vocab)}
word_to_idx = {word: i for i, word in enumerate(idiom_vocab)}
idx_to_word = {i: word for i, word in enumerate(idiom_vocab)}

# wvmodel = gensim.models.KeyedVectors.load_word2vec_format('file2.txt', binary=False, encoding='utf-8')

# vocab_size = 3848
# embed_size = 768
# weight = torch.zeros(vocab_size, embed_size)



class GlobalAttention(nn.Module):
    """
    Global Attention as described in 'Effective Approaches to Attention-based Neural Machine Translation'
    """
    def __init__(self, enc_hidden, dec_hidden):
        super(GlobalAttention, self).__init__()
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden

        # a = h_t^T W h_s
        self.linear_in = nn.Linear(enc_hidden, dec_hidden, bias=False)
        # W [c, h_t]
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.ln = nn.LayerNorm(768)
        self.linear_out = nn.Linear(dec_hidden + enc_hidden, dec_hidden)
        self.softmax = nn.Softmax()
        self.weight = Parameter(torch.ones(1, 1, 1))
        self.bias = Parameter(torch.ones(1, 1, 1))
        # self.weight1 = Parameter(torch.zeros(1, 1, 1))
        # self.bias1 = Parameter(torch.ones(1, 1, 1))
        self.tanh = nn.Tanh()
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()
        self.sig  = nn.Sigmoid()

    def sequence_mask(self, lengths, max_len=None):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return (torch.arange(0, max_len)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))

    def forward(self, inputs, context):
        """
        input (FloatTensor): batch x tgt_len x dim: decoder's rnn's output. (h_t)
        context (FloatTensor): batch x src_len x dim: src hidden states
        context_lengths (LongTensor): the source context lengths.
        """
        # (batch, tgt_len, src_len)
        align = self.score(inputs, context) #64*10*768,64*256*768->64*10*256
        batch, tgt_len, src_len = align.size()


#        mask = self.sequence_mask(context_lengths)
#        # (batch, 1, src_len)
#        mask = mask.unsqueeze(1)  # Make it broadcastable.
#        if next(self.parameters()).is_cuda:
#            mask = mask.cuda()
#        align.data.masked_fill_(1 - mask, -float('inf')) # fill <pad> with -inf

        align_vectors = self.softmax(align.view(batch*tgt_len, src_len))
        align_vectors = align_vectors.view(batch, tgt_len, src_len) #64*10*256

        align_vectors_g = align_vectors  #64,10,256 xi
        # align_vectors_gn = align_vectors_g * self.avg_pool(align_vectors)  #xi*g->ci 64,10,256 * 64,10,1
        align_vectors_gn = align_vectors_g
        align_vectors_gn = align_vectors_gn.sum(dim=1, keepdim=True) #ci 64*1*256

        # t = align_vectors_gn.view(batch, -1)   #ci 64*256
        # t = t - t.mean(dim=1, keepdim=True) #µc 64*256
        # std = t.std(dim=1, keepdim=True) + 1e-5 #σc 64*1
        # t = t / std #cˆi 
        # t = t.view(batch, -1, src_len) #64,1,256
        # t = t * self.weight + self.bias #ai
        # t = t.view(batch, -1, src_len)     #64,1,256
        
        t = align_vectors_gn 
        align_vectors_g = align_vectors_g * self.sig(t) #(64,10,256)*(64,1,256) #ˆxi
        # x = x.view(b, c, h, w)
        align_vectors_g = align_vectors_g.view(batch, tgt_len, src_len)
        align_vectors_g = align_vectors_g * self.avg_pool(align_vectors)
        align_vectors_g = align_vectors_g.view(batch, tgt_len, src_len)

        # (batch, tgt_len, src_len) * (batch, src_len, enc_hidden) -> (batch, tgt_len, enc_hidden)
        c = torch.bmm(align_vectors_g, context) #(64*10*256,64*256*768)->(64*10*768) t
        
        c = c * self.weight + self.bias
        c = c.view(batch, tgt_len, self.enc_hidden)

        # c = self.ln(c)     #64,1,256

        # \hat{h_t} = tanh(W [c_t, h_t])
        concat_c = torch.cat([c, inputs], 2).view(batch*tgt_len, self.enc_hidden + self.dec_hidden) #64*10,768+768
        attn_h1 = self.linear_out(concat_c).view(batch, tgt_len, self.dec_hidden)
        
        attn_h = self.prelu(attn_h1)
        


        # transpose will make it non-contiguous
        attn_h = attn_h.transpose(0, 1).contiguous()
        align_vectors_g = align_vectors_g.transpose(0, 1).contiguous()
        # (tgt_len, batch, dim)
        return attn_h, align_vectors_g

    def score(self, h_t, h_s):  #64*10*768,64*256*768
        """
        h_t (FloatTensor): batch x tgt_len x dim, inputs
        h_s (FloatTensor): batch x src_len x dim, context
        """
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        src_batch, src_len, src_dim = h_s.size()

        h_t = h_t.view(tgt_batch*tgt_len, tgt_dim) #64*10,768
        h_t_ = self.linear_in(h_t)
        h_t = h_t.view(tgt_batch, tgt_len, tgt_dim)
        # (batch, d, s_len)
        h_s_ = h_s.transpose(1, 2) #64*768*256
        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_) #64*10*256



class BertForCloze(PreTrainedBertModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `option_ids`: a torch.LongTensor of shape [batch_size, num_choices]
        `positions`: a torch.LongTensor of shape [batch_size]
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    """
    def __init__(self, config, num_choices):
        super(BertForCloze, self).__init__(config)
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.attention = GlobalAttention(config.hidden_size, config.hidden_size)
        # self.attention1 = Global1Attention(config.hidden_size, config.hidden_size)
        self.idiom_embedding = nn.Embedding(len(idiom_vocab), config.hidden_size)
        # self.idiom_embedding = nn.Embedding.from_pretrained(weight)
        # self.idiom_embedding.weight.requires_grad = True
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, option_ids, token_type_ids, attention_mask, positions, labels=None):

        encoded_layer, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False) #64*256*768
        blank_states = encoded_layer[[i for i in range(len(positions))], positions] # [batch, hidden_state]64*768
        encoded_options = self.idiom_embedding(option_ids) #64*10*768
        multiply_result = torch.einsum('abc,ac->abc', encoded_options , blank_states) #64*10*768 ht
        att, align = self.attention(multiply_result, encoded_layer)# (tgt_len, batch, dim) #10*64*768
        att = att.transpose(0, 1).contiguous()
        align = align.transpose(0, 1).contiguous()

        pooled_output = self.dropout(att) #64*10*768
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if labels is not None:
            # loss_fct = FocalLoss(gamma =0.25)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits, align


class ChidExample(object):
    def __init__(self,
                 tag,
                 context,
                 options,
                 label=None):
        self.tag = tag
        self.context = context
        self.options = options
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "tag: %s" % (self.tag)
        s += ", context: %s" % (self.context)
        s += ", options: [%s]" % (", ".join(self.options))
        if self.label is not None:
            s += ", answer: %s" % self.options[self.label]
        return s


def read_chid_examples(input_data, is_training, ans_dict=None):
    if is_training:
        assert ans_dict is not None

    examples = []
    for data in input_data:
        data = eval(data)
        options = data['candidates']
        for context in data['content']:
            tags = re.findall("#idiom\d+#", context)
            for tag in tags:
                tmp_context = context
                for other_tag in tags:
                    if other_tag != tag:
                        tmp_context = tmp_context.replace(other_tag, "[UNK]")
                label = None
                if is_training:
                    label = ans_dict[tag]
                examples.append(ChidExample(
                    tag=tag,
                    context=tmp_context,
                    options=options,
                    label=label
                ))
    return examples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 tag,
                 input_ids,
                 input_mask,
                 segment_ids,
                 option_ids,
                 position,
                 label):
        self.tag = tag # use int number
        self.input_ids = input_ids # [max_seq_length]
        self.input_mask = input_mask # [max_seq_length]
        self.segment_ids = segment_ids # [max_seq_length]
        self.option_ids = option_ids # [num_choices]
        self.position = position
        self.label = label


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
#tag=编号，label=答案编号，options=成语选项
    features = []
    for example in examples:

        label = None
        if example.label is not None:
            label = example.label
        tag = example.tag
        context = example.context
        parts = re.split(tag, context)
        assert len(parts) == 2 , tag
        before_part = tokenizer.tokenize(parts[0]) if len(parts[0]) > 0 else []
        after_part = tokenizer.tokenize(parts[1]) if len(parts[1]) > 0 else []

        half_length = int(max_seq_length / 2)
        if len(before_part) < half_length: # cut at tail
            st = 0
            ed = min(len(before_part) + 1 + len(after_part), max_seq_length - 2)
        elif len(after_part) < half_length: # cut at head
            ed = len(before_part) + 1 + len(after_part)
            st = max(0, ed - (max_seq_length - 2))
        else: # cut at both sides
            st = len(before_part) + 3 - half_length
            ed = len(before_part) + 1 + half_length

        option_ids = [idiom_vocab[each] for each in example.options]
        tokens = before_part + ["[MASK]"] + after_part
        tokens = ["[CLS]"] + tokens[st:ed] + ["[SEP]"]
        position = tokens.index("[MASK]")
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        features.append(InputFeatures(
            tag=int(tag[6: -1]),
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            option_ids=option_ids,
            position=position,
            label=label
        ))
        
    return features


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return max((x-1.)/(warmup-1.), 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_config_file", default=None, type=str, required=True,
                        help="The config json file corresponding to the pre-trained BERT model. "
                             "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default=None, type=str, required=True,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).")
    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--train_ans_file", default=None, type=str, help="SQuAD answer for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=32, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--restore',
                        default=False)
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
#        torch.backends.cudnn.benchmark = True
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir)==False:
        # raise ValueError("Output directory () already exists and is not empty.")
        os.makedirs(args.output_dir, exist_ok=True)


    import pickle as cPickle
    train_examples = None
    num_train_steps = None
    if args.do_train:
        raw_test_data = open(args.predict_file, mode='r')
        raw_train_data = open(args.train_file,mode='r')
        if os.path.exists("train_file_baseline.pkl") and False:
            train_examples=cPickle.load(open("train_file_baseline.pkl",mode='rb'))
        else:
            ans_dict = {}
            with open(args.train_ans_file) as f:
                for line in f:
                    line = line.split(',')
                    ans_dict[line[0]] = int(line[1])
            train_examples = read_chid_examples(raw_train_data, is_training=True, ans_dict=ans_dict)
            cPickle.dump(train_examples,open("newtrain_file_baseline.pkl",mode='wb'))

        #tt = len(train_examples) // 2
        #train_examples = train_examples[:tt]

        logger.info("train examples {}".format(len(train_examples)))
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    bert_config = BertConfig.from_json_file(args.bert_config_file)
    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    model = BertForCloze(bert_config, num_choices=10)
    if args.init_checkpoint is not None:
        logger.info('load bert weight')
        state_dict=torch.load(args.init_checkpoint, map_location='cpu')
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        # new_state_dict=state_dict.copy()
        # for kye ,value in state_dict.items():
        #     new_state_dict[kye.replace("bert","c_bert")]=value
        # state_dict=new_state_dict
        if metadata is not None:
            state_dict._metadata = metadata
        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                # logger.info("name {} chile {}".format(name,child))
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        logger.info("missing keys:{}".format(missing_keys))
        logger.info('unexpected keys:{}'.format(unexpected_keys))
        logger.info('error msgs:{}'.format(error_msgs))
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex import amp
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate)
        # optimizer = BertAdam(optimizer_grouped_parameters,
        #                      lr=args.learning_rate,
        #                      warmup=args.warmup_proportion,
        #                      t_total=t_total)
        # optimizer = RAdam(optimizer_grouped_parameters,
        #                      lr=args.learning_rate)                     
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.restore:
        checkpoint = torch.load('amp_checkpoint.pt')

        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        amp.load_state_dict(checkpoint['amp'])



    global_step = 0
    if args.do_train:
        cached_train_features_file = args.train_file + '_{0}_v{1}'.format(str(args.max_seq_length), str(4))
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length
            )

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)

        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_labels= torch.tensor([f.label for f in train_features],dtype=torch.long)
        all_option_ids = torch.tensor([f.option_ids for f in train_features], dtype=torch.long)
        all_positions = torch.tensor([f.position for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_labels, all_option_ids, all_positions)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True,pin_memory=True)
        loss_ini = 50
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            vizname = 'epoch'+ str(_)
            viz = Visdom(env=str(vizname))
            vis = Visdom(env='loss')
            via = Visdom(env='ac')
            model.train()
            model.zero_grad()
            epoch_itorator=tqdm(train_dataloader,disable=None)
            for step, batch in enumerate(epoch_itorator):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, labels, option_ids, positions = batch
                loss = model(input_ids, option_ids, segment_ids, input_mask, positions, labels)
#                print('att', loss.size())
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
#                    model, optimizer = amp.initialize(model, optimizer, opt_level= "O1")
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    # if args.fp16:
                    #      torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (step + 1) % 1000 == 0:
                    logger.info("loss@{}:{}".format(step, loss.cpu().item()))
                steptotal =  step + _ * int(len(train_examples) / args.train_batch_size)
                if (steptotal + 1) % 50 == 0:
                    vis.line([loss.cpu().item()], [steptotal], win='train_loss', update='append')
                if (step + 1) % 50 == 0:
                    viz.line([loss.cpu().item()], [step], win='train_loss', update='append')
            loss_total = str(loss.cpu().item())
            print(loss_total)
            loss_ini = loss_total
            logger.info("loss:%f", loss.cpu().item())
            logger.info("loss+:{}".format(loss.cpu().item()))
            raw_test_data_pre = open(args.predict_file, mode='r')
            eval_examples = read_chid_examples(
                raw_test_data_pre, is_training=False)
            # eval_examples=eval_examples[:100]
            eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length
            )

            logger.info("***** Running predictions *****")
            logger.info("  Num orig examples = %d", len(eval_examples))
            logger.info("  Num split examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.predict_batch_size)

            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_option_ids = torch.tensor([f.option_ids for f in eval_features], dtype=torch.long)
            all_positions = torch.tensor([f.position for f in eval_features], dtype=torch.long)
            all_tags= torch.tensor([f.tag for f in  eval_features],dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_option_ids, all_positions, all_tags)
        # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

            model.eval()
            reader1 = pd.read_csv('dev_answer1.csv', usecols=[1], header=None)
            total_dev_loss = 0
            all_results = {}
            logger.info("Start evaluating")
            for input_ids, input_mask, segment_ids, option_ids, positions, tags in \
                    tqdm(eval_dataloader, desc="Evaluating",disable=None):
                if len(all_results) % 1000 == 0:
                    logger.info("Processing example: %d" % (len(all_results)))
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                option_ids = option_ids.to(device)
                positions = positions.to(device)
                with torch.no_grad():
                    batch_logits,align = model(input_ids, option_ids, segment_ids, input_mask, positions)
                for i, tag in enumerate(tags):
                    logits = batch_logits[i].detach().cpu().numpy()
                    logit = [logits]
                    logit = torch.tensor(logit)

                    inum = int(tag) - 577157
                    dlabel = [reader1[1][inum]]
                    dlabel = torch.tensor(dlabel) 
                    # loss_dev =FocalLoss(gamma=0.25)
                    loss_dev = CrossEntropyLoss()
                    dev_loss = loss_dev(logit, dlabel)
                    total_dev_loss += dev_loss
                  #  for index1, dlabel in zip(reader1[0], reader1[1]):
                  #      if index1[6:11] == str(tag):
                  #          loss_dev =CrossEntropyLoss()
                  #          dev_loss = loss_dev(logits, dlabel)
                  #          total_dev_loss += dev_loss
                  #          continue
                    ans = np.argmax(logits)
                    all_results["#idiom%06d#" % tag] = ans
                    
            predict_name = "ln11saprediction"+str(_)+".csv"
            output_prediction_file = os.path.join(args.output_dir, predict_name)
            with open(output_prediction_file, "w") as f:
                for each in all_results:
                    f.write(each + ',' + str(all_results[each]) + "\n")
            raw_test_data.close()
            pre_ac = 0
            outputpre = 'output_model/'+predict_name
            reader2 = pd.read_csv(outputpre, usecols=[0,1], header=None)
            
            for index2, ans2 in zip(reader2[0], reader2[1]):
                     num = index2[6:12]
                     num = int(num) - 577157
                     ans1 = reader1[1][num]
                     if ans1 == ans2:
                         pre_ac += 1
            print(pre_ac)
            per = (pre_ac)/23011
            pernum = per*100
            logger.info("accuracy:%f",pernum)
            devlossmean = total_dev_loss/(23011/128)
            logger.info("devloss:%f",devlossmean)
            via.line([pernum], [_], win='accuracy', update='append')
            via.line([devlossmean], [_], win='loss', update='append')
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict()
                }
            torch.save(checkpoint, 'checkpoint/amp_checkpoint.pt')

            outmodel = 'ln11samodel'+str(pernum)+'.bin'
            output_model_file = os.path.join(args.output_dir, outmodel)
            if args.do_train:
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                torch.save(model_to_save.state_dict(), output_model_file)
        raw_test_data.close()
        raw_train_data.close()
# Save a trained model
#    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
#    if args.do_train:
#        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
#        torch.save(model_to_save.state_dict(), output_model_file)

    # Load a trained model that you have fine-tuned
 
 
 
 


    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        list1 = os.listdir('./output_model/')
        list1 = sorted(list1, key=lambda x: os.path.getmtime(os.path.join('./output_model/', x)))
        output_model_file = os.path.join(args.output_dir, list1[-1])
        # output_model_file = os.path.join(args.output_dir, 'n11samodel77.33258007040111.bin')
        model_state_dict = torch.load(output_model_file)
        model = BertForCloze(bert_config, num_choices=10)
        model.load_state_dict(model_state_dict)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # raw_test_data_pre = open('./data/dev.txt', mode='r')    
        raw_test_data_pre = open('./data/out.txt', mode='r')
        # raw_test_data_pre = open('new_test_data.txt', mode='r')
        eval_examples = read_chid_examples(
            raw_test_data_pre, is_training=False)
        # eval_examples=eval_examples[:100]
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length
        )

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_option_ids = torch.tensor([f.option_ids for f in eval_features], dtype=torch.long)
        all_positions = torch.tensor([f.position for f in eval_features], dtype=torch.long)
        all_tags= torch.tensor([f.tag for f in  eval_features],dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_option_ids, all_positions, all_tags)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        all_results = {}
        all_results1 = {}
        all_results2 = {}
        # reader1 = pd.read_csv('test_ans.csv', usecols=[1], header=None)
        reader1 = pd.read_csv('./data/out_answer.csv', usecols=[1], header=None) #dev_answer1.csv
        # reader1 = pd.read_csv('dev_answer1.csv', usecols=[1], header=None)
        total_dev_loss = 0
        logger.info("Start evaluating")
        for input_ids, input_mask, segment_ids, option_ids, positions, tags in \
                tqdm(eval_dataloader, desc="Evaluating",disable=None):
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            option_ids = option_ids.to(device)
            positions = positions.to(device)
            with torch.no_grad():
                batch_logits,align = model(input_ids, option_ids, segment_ids, input_mask, positions)
            for i, tag in enumerate(tags):
                logits = batch_logits[i].detach().cpu().numpy()
                
                ans = np.argmax(logits)
                all_results["#idiom%06d#" % tag] = ans

                # matric = align[i].detach().cpu().numpy()
                # all_results1["#idiom%06d#" % tag] = matric[ans]
                # gr_logic = logits[:]
                # gr_logic = sorted(gr_logic, reverse=True)
                # all_results2["#idiom%06d#" % tag] = gr_logic



        output_prediction_file = os.path.join(args.output_dir, "testprediction.csv")
        # output_m_file = os.path.join(args.output_dir, "ealign.csv")
        # output_ma_file = os.path.join(args.output_dir, "sdmv.csv")

        with open(output_prediction_file, "w") as f:
            for each in all_results:
                f.write(each + ',' + str(all_results[each]) + "\n")
        # with open(output_m_file, "w") as f:
        #     for each in all_results1:
        #         f.write(each + ',' + str(all_results1[each]) + "\n")
        # with open(output_ma_file, "w") as f:
        #     for each in all_results1:
        #         f.write(each + ',' + str(all_results2[each]) + "\n")
        raw_test_data_pre.close()
        reader2 = pd.read_csv(output_prediction_file, usecols=[0,1], header=None)
        pre_ac = 0   
        for index2, ans2 in zip(reader2[0], reader2[1]):
            num = index2[6: -1]
            # num = int(num)-1
            # num = re.findall(r"\d+\.?\d*",index2)
            num = int(num) - 623377
            # num = int(num) - 577157
            ans1 = reader1[1][num]
            if ans1 == ans2:
                pre_ac += 1
        print(pre_ac)
        # per = (pre_ac)/23011
        # per = (pre_ac)/24948
        per = (pre_ac)/27704
        pernum = per*100
        logger.info("accuracy:%f",pernum)
        # devlossmean = total_dev_loss/(23011/128)
        # logger.info("devloss:%f",devlossmean)

if __name__ == "__main__":
    main()
