from triplet.hypergraph.TensorGlobalNetworkParam import TensorGlobalNetworkParam
from triplet.hypergraph.NetworkModel import NetworkModel
import torch.nn as nn
import torch
from triplet.hypergraph.Utils import *
from triplet.common import LinearInstance
from triplet.common.eval import nereval
import re
import os
from termcolor import colored
import random
import numpy as np
from triplet.hypergraph.BatchTensorNetwork import BatchTensorNetwork
from jet_t_utils.tag_reader import TagReader
from jet_t_utils.neural_model import TriextractNeuralBuilder
from jet_t_utils.network_compiler import TriextractNetworkCompiler
from collections import OrderedDict


# the following parameters are required to adjust for different datasets
train_file = 'data/triplet_data/15res/train.txt'
dev_file = 'data/triplet_data/15res/dev.txt'
test_file = 'data/triplet_data/15res/test.txt'
trial_file = 'data/triplet_data/15res/trial.txt'
opinion_offset = 3 # This equals to M+1 in the paper
dropout = 0.5  # 0.7 for 14lap only 0.5 for the rest datasets
use_bert = False # default bert base


# keep the folowing parameters for reproducing our results
require_training = True
TRIAL = False
num_train = -1
visual = False
num_dev = -1
num_test = -1
num_iter = 20
batch_size = 1
polarity = 3
opinion_direction = 2
device = 'cpu'
optimizer_str = 'adam'
NetworkConfig.NEURAL_LEARNING_RATE = 0.01
num_thread = 1
token_emb_size = 300
lstm_hidden_size = 300
pos_emb_size = 100
char_emb_size = 0
charlstm_hidden_dim = 0
emb_file = 'data/glove.840B.300d.txt'
# emb_file = None

# default bert base is used
bert_emb = 0
if use_bert:
	bert_emb = 768

# A trial run for debugging
if TRIAL == True:
	data_size = -1
	train_file = trial_file
	dev_file = trial_file
	test_file = trial_file
	num_iter = 200


# setting network configurations
NetworkConfig.DEVICE = torch.device(device)
NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH = True
NetworkConfig.IGNORE_TRANSITION = False
NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING = False
seed = 42
torch.manual_seed(seed)
torch.set_num_threads(40)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
UNK = '<UNK>'
PAD = '<PAD>'
if num_thread > 1:
	NetworkConfig.NUM_THREADS = num_thread
	print ('Set NUM_THREADS = ', num_thread)


# read data
train_insts = TagReader.read_inst(train_file, True, num_train, opinion_offset)
dev_insts = TagReader.read_inst(dev_file, False, num_dev, opinion_offset)
test_insts = TagReader.read_inst(test_file, False, num_test,opinion_offset)
TagReader.label2id_map['<STOP>'] = len(TagReader.label2id_map)
print('Map: ', TagReader.label2id_map)


# create vocab and char dict
max_size = -1
vocab2id = {}
char2id = {PAD:0, UNK:1}
labels = ['x'] * len(TagReader.label2id_map)
for key in TagReader.label2id_map:
	labels[TagReader.label2id_map[key]] = key
for inst in train_insts + dev_insts + test_insts:
	max_size = max(len(inst.input), max_size)
	for word in inst.input:
		if word not in vocab2id:
			vocab2id[word] = len(vocab2id)
for inst in train_insts:
	max_size = max(len(inst.input), max_size)
	for word in inst.input:
		for ch in word:
			if ch not in char2id:
				char2id[ch] = len(char2id)
print (colored('vocab2id: ', 'red'), len(vocab2id))


# convert char and word to id and add to instance
chars = [None] * len(char2id)
for key in char2id:
	chars[char2id[key]] = key
for inst in train_insts + dev_insts + test_insts:
	max_word_length = max([len(word) for word in inst.input])
	inst.word_seq = torch.LongTensor([vocab2id[word] if word in vocab2id else vocab2id[UNK] for word in inst.input]).to(NetworkConfig.DEVICE)
	char_seq_list = [[char2id[ch] if ch in char2id else char2id[UNK] for ch in word]+[char2id[PAD]]* (max_word_length - len(word)) for word in inst.input]
	inst.char_seq_tensor = torch.LongTensor(char_seq_list).to(NetworkConfig.DEVICE)
	inst.char_seq_len = torch.LongTensor([len(word) for word in inst.input]).to(NetworkConfig.DEVICE)


# prepare model
gnp = TensorGlobalNetworkParam()
fm = TriextractNeuralBuilder(gnp, len(vocab2id), len(TagReader.label2id_map), char2id, chars, char_emb_size, charlstm_hidden_dim, lstm_hidden_size , dropout , pos_emb_size, token_emb_size, bert_emb)
fm.labels = labels
fm.load_pretrain(emb_file, vocab2id)
compiler = TriextractNetworkCompiler(TagReader.label2id_map, max_size, polarity, opinion_offset, opinion_direction)
evaluator = nereval()
model = NetworkModel(fm, compiler, evaluator)


if require_training:
    if batch_size == 1:
        model.learn(train_insts, num_iter, dev_insts, test_insts, optimizer_str, batch_size)
    else:
        model.learn_batch(train_insts, num_iter, dev_insts, test_insts, optimizer_str, batch_size)
else:
    state_dict = torch.load('best_model.pt')
    model.load_state_dict(state_dict)
























