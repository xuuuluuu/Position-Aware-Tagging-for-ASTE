from triplet.hypergraph.NetworkCompiler import NetworkCompiler
from triplet.hypergraph.NetworkIDMapper import NetworkIDMapper
from triplet.hypergraph.TensorBaseNetwork import TensorBaseNetwork
from triplet.hypergraph.TensorGlobalNetworkParam import TensorGlobalNetworkParam
from triplet.hypergraph.NeuralBuilder import NeuralBuilder
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
# class TriextracInstance(LinearInstance):
# 	# target sentiment opinion pairs from output 


class TriextractNetworkCompiler(NetworkCompiler):
	def __init__(self, label_map, max_size, polarity, opinion_offset, opinion_direction):
		#label map is a dict
		#max_size is the max length of the sentence in train valid and test dataset
		super().__init__()
		self.labels = ['x'] * len(label_map)
		self.label2id = label_map
		for key in self.label2id:
			self.labels[self.label2id[key]] = key
		self.MAX_OPINION_OFFSET = opinion_offset
		self.OPINION_DIRECTION = opinion_direction
		self.POLARITY = polarity
		NetworkIDMapper.set_capacity(np.asarray([max_size, 100, self.POLARITY, self.OPINION_DIRECTION, self.MAX_OPINION_OFFSET, self.MAX_OPINION_OFFSET, 3], dtype = np.int64))
		#pos, label_id, opinion_direction, opinion_start, opinion_end, node_type
		self._all_nodes = None
		self._all_children = None
		self._max_size = max_size
		print ('The max size: ', self._max_size)
		print ("Building generic network..")
		self.build_generic_network()

	def to_root(self, size):
		return self.to_node(size-1, len(self.labels)-1, 0, self.OPINION_DIRECTION - 1, self.MAX_OPINION_OFFSET - 1, self.MAX_OPINION_OFFSET - 1, 2)

	def to_leaf(self, ):
		#remove comma?
		#ans: does not matter
		return self.to_node(0, 0, 0, 0, 0, 0, 0)

	def to_BStag(self, pos, label_id, opinion):
		return self.to_node(pos, label_id, opinion[0], opinion[1], opinion[2], opinion[3], 1) #TODO: get label id

	def to_tag(self, pos, label_id, ):
		return self.to_node(pos, label_id, 0, 0, 0, 0, 1)

	def to_node(self, pos, label_id, opinion_sentiment, opinion_dir, opinion_s, opinion_e, node_type): #TODO RESHAPE ARRAY
		return NetworkIDMapper.to_hybrid_node_ID(np.asarray([pos, label_id, opinion_sentiment, opinion_dir, opinion_s, opinion_e, node_type]))

	def compile_labeled(self, network_id, inst, param):
		#build a labeled network
		#what is network id and param? for what purpose?
		#ans: no longer in use, param==fm
		builder = TensorBaseNetwork.NetworkBuilder.builder()
		leaf = self.to_leaf()
		builder.add_node(leaf)
		output_label, opinion_pair  = inst.get_output()
		#abstract method? where to find the content
		children = [leaf]
		# print(opinion_pair)
		m=0
		for i in range(inst.size()):
			label = output_label[i]
			if 'S' in label or 'B' in label:
				bstag_node = self.to_BStag(i, self.label2id[label], opinion_pair[m][1:])
				builder.add_node(bstag_node)
				builder.add_edge(bstag_node, children)
				children = [bstag_node]
				m+=1
			else:
				tag_node = self.to_tag(i, self.label2id[label])
				builder.add_node(tag_node)
				builder.add_edge(tag_node, children)
				children = [tag_node]

		root = self.to_root(inst.size())
		builder.add_node(root)
		builder.add_edge(root, children)
		network = builder.build(network_id, inst, param, self)
		#why write self at the end?
		#ans: call this class as a instance
		return network

	def compile_unlabeled_old(self, network_id, inst, param):
		builder = TensorBaseNetwork.NetworkBuilder.builder()
		leaf = self.to_leaf()
		builder.add_node(leaf)

		children = [leaf]
		for i in range(inst.size()):
			current = [None for k in range(len(self.labels))]
			for l in range(len(self.labels)):
				tag_node = self.to_tag(i,l)
				builder.add_node(tag_node)
				for child in children:
					builder.add_edge(tag_node, child)
				current[l] = tag_node
			children = current
		root = self.to_root(inst.size())
		builder.add_node(root)
		for child in children:
			builder.add_edge(root, [child])
		network = builder.build(network_id, inst, param, self)
		return network

	def compile_unlabeled(self, network_id, inst, param):
		# return self.compile_labeled(network_id, inst, param)
		builder = TensorBaseNetwork.NetworkBuilder.builder()
		leaf = self.to_leaf()
		builder.add_node(leaf)

		children = [leaf]
		for i in range(inst.size()):
			current = [] # changed from NOne list to empty list, might slow the process
			for l in range(len(self.labels)):
				# so there is no more start and stop nodes?
				#ans: check outside loop
				if l == self.label2id[START] or l==self.label2id[STOP]:
					continue
				if 'S' in self.labels[l] or 'B'in self.labels[l]:
					for p in range(self.POLARITY):
						for s in range(1, self.MAX_OPINION_OFFSET):
							for e in range(s, self.MAX_OPINION_OFFSET):
								for d in range(self.OPINION_DIRECTION):
									if d == 0:
										if i-s > -1 and i-e>-1:
											bstag_node = self.to_BStag(i, l, (p, d, s, e))
											builder.add_node(bstag_node)
											for child in children:
												if child is not None:
													builder.add_edge(bstag_node, [child])
											current.append(bstag_node)
									else:
										if i+s < inst.size() and i+e < inst.size():
											bstag_node = self.to_BStag(i, l, (p, d, s, e))
											builder.add_node(bstag_node)
											for child in children:
												if child is not None:
													builder.add_edge(bstag_node, [child])
											current.append(bstag_node)
									
				else:
					tag_node = self.to_tag(i, l)
					builder.add_node(tag_node)
					for child in children:
						if child is not None:
							builder.add_edge(tag_node, [child])
					current.append(tag_node)
			children = current
			root = self.to_root(i+1)
			builder.add_node(root)
			for child in children:
				if child is not None:
					builder.add_edge(root, [child])

		network = builder.build(network_id, inst, param, self)
		return network


	def build_generic_network(self, ):
		builder = TensorBaseNetwork.NetworkBuilder.builder()
		leaf = self.to_leaf()
		builder.add_node(leaf)

		children = [leaf]
		for i in range(self._max_size):
			current = [] # changed from NOne list to empty list, might slow the process
			for l in range(len(self.labels)):
				# so there is no more start and stop nodes?
				#ans: check outside loop

				if l == self.label2id[START] or l==self.label2id[STOP]:
					continue
				if 'S' in self.labels[l] or 'B'in self.labels[l]:
					for p in range(self.POLARITY):
						for s in range(1, self.MAX_OPINION_OFFSET):
							for e in range(s, self.MAX_OPINION_OFFSET):
								for d in range(self.OPINION_DIRECTION):
									if d == 0:
										if p-s > 0 and p-e>0:
											bstag_node = self.to_BStag(i, l, (p, d, s, e))
											builder.add_node(bstag_node)
											for child in children:
												if child is not None:
													builder.add_edge(bstag_node, [child])
											current.append(bstag_node)
									else:
										if p+s < self._max_size and p+e < self._max_size:
											bstag_node = self.to_BStag(i, l, (p, d, s, e))
											builder.add_node(bstag_node)
											for child in children:
												if child is not None:
													builder.add_edge(bstag_node, [child])
											current.append(bstag_node)
									
				else:
					tag_node = self.to_tag(i, l)
					builder.add_node(tag_node)
					for child in children:
						if child is not None:
							builder.add_edge(tag_node, [child])
					current.append(tag_node)
			children = current
			root = self.to_root(i+1)
			builder.add_node(root)
			for child in children:
				if child is not None:
					builder.add_edge(root, [child])
		self._all_nodes, self._all_children, self.num_hyperedge = builder.pre_build()
		#what is the return? , _all_nodes?, _all_children? and num_hyperedge?
		#Ans: initialise at the begining

	def decompile(self, network):
		inst = network.get_instance()
		size = inst.size()
		root_node = self.to_root(size)
		all_nodes = network.get_all_nodes()
		curr_idx = np.argwhere(all_nodes == root_node)[0][0]
		prediction = [None for i in range(size)]
		pairs = []
		for i in range(size):
			children = network.get_max_path(curr_idx)
			#what children looks like? a list of?
			#Ans: for hypergraph children, might be multiple children
			child = children[0]
			child_arr = network.get_node_array(child)
			prediction[size - i - 1] = self.labels[child_arr[1]]
			#pos, label_id, opinion_direction, opinion_start, opinion_end, node_type
			if prediction[size - i - 1] =='B' or prediction[size - i - 1] == 'S':
				pairs.append((size - i - 1, child_arr[-5], child_arr[-4], child_arr[-3], child_arr[-2]))
			curr_idx = child
		inst.set_prediction((prediction, pairs))
		return inst

class TriextractNeuralBuilder(NeuralBuilder):
	def __init__(self, gnp, voc_size, label_size, char2id, chars, char_emb_size, charlstm_hidden_dim, lstm_hidden_size = 100, dropout = 0.5):
		#what is gnp? never used?
		#Ans: transition
		super().__init__(gnp)
		self.token_embed = 100
		self.label_size = label_size
		print('vocab size: ', voc_size)
		self.char_emb_size = char_emb_size
		lstm_input_size = self.token_embed
		if char_emb_size > 0:
			from triplet.features.char_lstm import CharBiLSTM
			self.char_bilstm = CharBiLSTM(char2id, chars, char_emb_size, charlstm_hidden_dim).to(NetworkConfig.DEVICE)
			lstm_input_size += charlstm_hidden_dim
		self.word_drop = nn.Dropout(dropout).to(NetworkConfig.DEVICE)
		self.word_embed = nn.Embedding(voc_size, self.token_embed).to(NetworkConfig.DEVICE)
		self.lstm_drop = nn.Dropout(dropout).to(NetworkConfig.DEVICE)
		self.rnn = nn.LSTM(lstm_input_size, lstm_hidden_size, batch_first = True, bidirectional = True).to(NetworkConfig.DEVICE)
		self.segmentlstm = nn.LSTM(lstm_hidden_size*2, lstm_hidden_size, batch_first = True, bidirectional = True).to(NetworkConfig.DEVICE)
		self.linear = nn.Linear(lstm_hidden_size *2, label_size).to(NetworkConfig.DEVICE)
		self.linear_span = nn.Linear(lstm_hidden_size * 2 , 2).to(NetworkConfig.DEVICE)
		self.linear_polar = nn.Linear(lstm_hidden_size * 2 , 3).to(NetworkConfig.DEVICE)
	def load_pretrain(self, path, word2idx):
		emb = load_emb_glove(path, word2idx, self.token_embed)
		#where is the load_emb_glove function from?
		#Ans: utils
		self.word_embed.weight.data.copy_(torch.from_numpy(emb))
		#weight?
		#Ans: assign pretrain embedding
		self.word_embed = self.word_embed.to(NetworkConfig.DEVICE)

	def build_nn_graph(self, instance):
		word_vec = self.word_embed(instance.word_seq.unsqueeze(0))
		word_rep = [word_vec]
		if self.char_emb_size > 0:
			char_seq_tensor = instance.char_seq_tensor.unsqueeze(0)
			char_seq_len = instance.char_seq_len.unsqueeze(0)
			char_features = self.char_bilstm.get_last_hiddens(char_seq_tensor, char_seq_len)			
			word_rep.append(char_features)
		word_rep = torch.cat(word_rep, 2)
		word_rep = self.word_drop(word_rep)
		#where to add dropout?
		#ans after each layer
		lstm_out, (hn, cn) = self.rnn(word_rep, None)
		lstm_out = self.lstm_drop(lstm_out)
		segment_out, _ = self.segmentlstm(lstm_out)
		segment_out = segment_out.squeeze(0)
		# print('segement', segment_out.size())
		linear_output = self.linear(lstm_out).squeeze(0)
		#score of each node
		instance_len = instance.size()
		span_score = {}
		polar_score = {}

		for i in range(instance_len):
			for j in range(i, instance_len):
				if i==0 and j+1 == instance_len:
					segment_emb = torch.cat([segment_out[i][0:100], segment_out[j][100:]], 0)
				elif i==0 and j+1 < instance_len:
					segment_emb = torch.cat([segment_out[i][0:100], segment_out[i][100:] - segment_out[j+1][100:]], 0)
				elif  i>0 and j+1 == instance_len:
					segment_emb = torch.cat([segment_out[j][0:100] - segment_out[i-1][0:100], segment_out[j][100:]], 0)
				else:
					segment_emb = torch.cat([segment_out[j][0:100] - segment_out[i-1][0:100] , segment_out[i][100:] - segment_out[j+1][100:]], 0)
				span_score[i, j] = self.linear_span(segment_emb)
				polar_score[i, j] = self.linear_polar(segment_emb)


		zero_col = torch.zeros(1, self.label_size).to(NetworkConfig.DEVICE)
		return torch.cat([linear_output, zero_col], 0), span_score, polar_score

	# def generate_batches(self, train_insts, batch_size):
	def get_nn_score(self, network, parent_k):
		parent_arr = network.get_node_array(parent_k)
		# print(parent_arr)
		pos = parent_arr[0]
		label_id = parent_arr[1]
		polarity = parent_arr[2]
		direction = parent_arr[3]
		start = parent_arr[4]
		end = parent_arr[5]
		node_type = parent_arr[-1]
		if node_type == 0 or node_type == 2:
			return torch.tensor(0.0).to(NetworkConfig.DEVICE)
		else:
			nn_output, span_score, polar_score = network.nn_output
			label_str = self.labels[label_id]
			# print(nn_output)
			# print(span_score)
			# print('polar score', polar_score)
			base_score = nn_output[pos][label_id]
			if label_str == 'B' or label_str == 'S':

				if direction == 1:
					opinion_boundary = (pos + start, pos + end)
				else:
					opinion_boundary = (pos - end, pos - start)
				# if opinion_boundary[1] < or opinion_boundary[0] >0: 
				base_score = base_score + span_score[opinion_boundary][0] + polar_score[opinion_boundary][polarity]

			return base_score
	def get_label_id(self, network, parent_k):
		parent_arr = network.get_node_array(parent_k)
		return parent_arr[1]

	# def build_node2nn_output(self, network):
	# 	size = network.count_nodes()
	# 	print('node size: ', size)
	# 	sent_len = network.get_instance().size()
	# 	nodeid2nn = [0]* size
	# 	for k in range(size):
	# 		parent_arr = network.get_node_array(k)
	# 		pos = parent_arr[0]
	# 		label_id = parent_arr[1]
	# 		polarity = parent_arr[2]
	# 		direction = parent_arr[3]
	# 		start = parent_arr[4]
	# 		end = parent_arr[5]

	# 		node_type = parent_arr[-1]
	# 		if node_type ==0 or node_type ==2:
	# 			idx = sent_len * self.label_size
	# 		else:
	# 			idx = pos * self.label_size + label_id 
	# 		nodeid2nn[k] = idx
	# 	return nodeid2nn

class TagReader():
	label2id_map = {'<START>': 0}
	@classmethod
	def read_inst(cls, file, is_labeled, number, opinion_offset):
		insts = []
		inputs = []
		outputs = []
		f = open(file, 'r', encoding = 'utf-8')
		if not is_labeled:
			opinion_offset = 10000
		###read AAAI2020 data
		for line in f:
			line = line.strip()
			line = line.split('####')
			# print (line)
			input = line[0].split()
			t_output = line[1].split()
			o_output = line[2].split()
			output = ['O' for x in range(len(input))]
			polarity = [0 for x in range(len(input))]
			for i,t in enumerate(t_output):
				t = t.split('=')[1]
				if t != 'O':
					output[i]= t
					if t.split('-')[1] == 'POS':
						polarity[i] = 1
					elif t.split('-')[1] == 'NEG':
						polarity[i] = 2
					elif t.split('-')[1] == 'NEU':
						polarity[i] = 0

			output_t_idx = [ 0 for x in range(len(input)) ]
			for i,tag in enumerate(output):
				if tag != 'O':
					tag = tag.split('-')[0]
					output_t_idx[i] = len(tag)
			output_t = cls.ot2bieos_ts(output)
			# print ('t', output_t)

			output = ['O' for x in range(len(input))]
			for i,t in enumerate(o_output):
				t = t.split('=')[1]
				if t != 'O':
					output[i]= t
			output_o_idx = [ 0 for x in range(len(input)) ]
			for i,tag in enumerate(output):
				if tag != 'O':
					tag = tag.split('-')[0]
					output_o_idx[i] = len(tag)
			output_o = cls.ot2bieos_op(output)
			# print ('o', output_o)

			output = ['O' for x in range(len(input))]
			for i in range(len(output)):
				if output_t[i]!='O':
					output[i] = output_t[i]
				# elif output_o[i]!='0':
				# 	output[i] = output_o[i]

			pairs = {}
			target = [None]
			for i, t in enumerate(output_t_idx):
				if t !=0 and t not in target:
					
					opinion_idx = [j for j,x in enumerate(output_o_idx) if x == t]
					if len(opinion_idx) >0:
						target.append(t)
						dire = 0
						if opinion_idx[0] - i >0:
							dire = 1
						if len(opinion_idx) == 1:
							if t not in pairs.keys():
								pairs[t] = [(i, polarity[i], dire, abs(opinion_idx[0] - i), abs(opinion_idx[0]-i))]
							else:
								pairs[t].append((i, polarity[i], dire, abs(opinion_idx[0] - i), abs(opinion_idx[0]-i)))
						elif len(opinion_idx) > 1:
							split = []
							for idx in range(0, len(opinion_idx)-1):
								if opinion_idx[idx+1] - opinion_idx[idx] != 1:
									split.append(idx)
							span = []
							start = 0
							if len(split) > 0:
								for s in split:
									if dire == 0:
										if t not in pairs.keys():
											pairs[t] = [(i, polarity[i], dire, abs(opinion_idx[0] - i), abs(opinion_idx[0]-i))]
										else:
											pairs[t].append((i, polarity[i], dire, abs(opinion_idx[0] - i), abs(opinion_idx[0]-i)))
									else:
										if t not in pairs.keys():
											pairs[t] = [(i, polarity[i], dire, abs(opinion_idx[0] - i), abs(opinion_idx[0]-i))]
										else:
											pairs[t].append((i, polarity[i], dire, abs(opinion_idx[0] - i), abs(opinion_idx[0]-i)))
									start = s+1								
							else:
								if dire == 0:
									if t not in pairs.keys():
										pairs[t] = [(i, polarity[i], dire, abs(opinion_idx[0] - i), abs(opinion_idx[0]-i))]
									else:
										pairs[t].append((i, polarity[i], dire, abs(opinion_idx[0] - i), abs(opinion_idx[0]-i)))
								else:
									if t not in pairs.keys():
										pairs[t] = [(i, polarity[i], dire, abs(opinion_idx[0] - i), abs(opinion_idx[0]-i))]
									else:
										pairs[t].append((i,polarity[i],  dire, abs(opinion_idx[0] - i), abs(opinion_idx[0]-i)))
					else:
						for i, idx in enumerate(output_t_idx):
							if idx == t:
								output[i] = 'O'
			new_output = []
			for o in output:
				if o != 'O':
					label = o.split('-')
					new_output.append(label[0])
				else:
					new_output.append(o)

			new_pairs = []
			for key in pairs.keys():
				if pairs[key][0][-1] < opinion_offset:
					new_pairs.append(pairs[key][0])
				else:
					remove_target_id = pairs[key][0][0]
					if new_output[remove_target_id] =='S':
						new_output[remove_target_id] = 'O'
					elif new_output[remove_target_id] =='B':
						new_output[remove_target_id] = 'O'
						for i in range(remove_target_id, len(input)):
							if new_output[i]!='E':
								new_output[i] = 'O'
							elif new_output[i]=='E':
								new_output[i] = 'O'
								break
				#this part remove multiple opinion phrases
			# print(pairs)
			# print (new_output)

			output = (new_output, new_pairs)
			inst = LinearInstance(len(insts) + 1, 1, input, output)
			for label in output[0]:
				if not label in TagReader.label2id_map and is_labeled:
					output_id = len(TagReader.label2id_map)
					TagReader.label2id_map[label] = output_id
			if is_labeled:
				inst.set_labeled()
			else:
				inst.set_unlabeled()
			insts.append(inst)
			if len(insts) >= number and number >0:
						break
		return insts

	@staticmethod
	def ot2bieos_ts( ts_tag_sequence):
		"""
		ot2bieos function for ts task
		:param ts_tag_sequence: tag sequence for targeted sentiment
		:return:
		"""
		n_tags = len(ts_tag_sequence)
		new_ts_sequence = []
		prev_pos = '$$$'
		for i in range(n_tags):
			cur_ts_tag = ts_tag_sequence[i]
			if cur_ts_tag == 'O':
				new_ts_sequence.append('O')
				cur_pos = 'O'
			else:
				cur_pos, cur_sentiment = cur_ts_tag.split('-')
				# cur_pos is T
				if cur_pos != prev_pos:
					# prev_pos is O and new_cur_pos can only be B or S
					if i == n_tags - 1:
						new_ts_sequence.append('S-%s' % cur_sentiment)
					else:
						next_ts_tag = ts_tag_sequence[i + 1]
						if next_ts_tag == 'O':
							new_ts_sequence.append('S-%s' % cur_sentiment)
						else:
							new_ts_sequence.append('B-%s' % cur_sentiment)
				else:
					# prev_pos is T and new_cur_pos can only be I or E
					if i == n_tags - 1:
						new_ts_sequence.append('E-%s' % cur_sentiment)
					else:
						next_ts_tag = ts_tag_sequence[i + 1]
						if next_ts_tag == 'O':
							new_ts_sequence.append('E-%s' % cur_sentiment)
						else:
							new_ts_sequence.append('I-%s' % cur_sentiment)
			prev_pos = cur_pos
		return new_ts_sequence

	@staticmethod
	def ot2bieos_op(ts_tag_sequence):
		"""
		ot2bieos function for ts task
		:param ts_tag_sequence: tag sequence for targeted sentiment
		:return:
		"""
		n_tags = len(ts_tag_sequence)
		new_ts_sequence = []
		prev_pos = '$$$'
		for i in range(n_tags):
			cur_ts_tag = ts_tag_sequence[i]
			if cur_ts_tag == 'O':
				new_ts_sequence.append('O')
				cur_pos = 'O'
			else:
				cur_pos = cur_ts_tag
				# cur_pos is T
				if cur_pos != prev_pos:
					# prev_pos is O and new_cur_pos can only be B or S
					if i == n_tags - 1:
						new_ts_sequence.append('s-o')
					else:
						next_ts_tag = ts_tag_sequence[i + 1]
						if next_ts_tag == 'O':
							new_ts_sequence.append('s-o')
						else:
							new_ts_sequence.append('b-o')
				else:
					# prev_pos is T and new_cur_pos can only be I or E
					if i == n_tags - 1:
						new_ts_sequence.append('e-o')
					else:
						next_ts_tag = ts_tag_sequence[i + 1]
						if next_ts_tag == 'O':
							new_ts_sequence.append('e-o')
						else:
							new_ts_sequence.append('i-o')
			prev_pos = cur_pos
		return new_ts_sequence


START = '<START>'
STOP = '<STOP>'
UNK = '<UNK>'
PAD = '<PAD>'

if __name__ == '__main__':
	NetworkConfig.BUILD_GRAPH_WITH_FULL_BATCH = True
	NetworkConfig.IGNORE_TRANSITION = False
	NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING = False
	seed = 42
	torch.manual_seed(seed)
	torch.set_num_threads(40)
	np.random.seed(seed)
	random.seed(seed)

	train_file = 'data/conll/tsa_train'
	dev_file = 'data/conll/tsa_dev'
	test_file = 'data/conll/tsa_test.txt'
	trial_file = 'data/conll/trial.txt.bieos'
	gold_pair_file = 'data/conll/14rest_test_pairs.npy'

	TRIAL = True
	num_train = 1
	visual = False
	num_dev = 1
	num_test = 1
	num_iter = 20
	batch_size = 1
	polarity = 3
	opinion_offset = 5
	opinion_direction = 2
	device = 'cuda'
	optimizer_str = 'adam'
	NetworkConfig.NEURAL_LEARNING_RATE = 0.01
	num_thread = 1
	# emb_file = 'data/glove.6B.100d.txt'
	emb_file = None


	char_emb_size = 30
	charlstm_hidden_dim = 50

	if TRIAL == True:
		data_size = -1
		train_file = trial_file
		dev_file = trial_file
		test_file = trial_file

	NetworkConfig.DEVICE = torch.device(device)
	torch.cuda.manual_seed(seed)
	if num_thread > 1:
		NetworkConfig.NUM_THREADS = num_thread
		print ('Set NUM_THREADS = ', num_thread)

	train_insts = TagReader.read_inst(train_file, True, num_train, opinion_offset)
	dev_insts = TagReader.read_inst(dev_file, False, num_dev, opinion_offset)
	test_insts = TagReader.read_inst(test_file, False, num_test,opinion_offset)

	TagReader.label2id_map['<STOP>'] = len(TagReader.label2id_map)
	print('Map: ', TagReader.label2id_map)
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

	chars = [None] * len(char2id)
	for key in char2id:
		chars[char2id[key]] = key
	for inst in train_insts + dev_insts + test_insts:
		max_word_length = max([len(word) for word in inst.input])
		inst.word_seq = torch.LongTensor([vocab2id[word] if word in vocab2id else vocab2id[UNK] for word in inst.input]).to(NetworkConfig.DEVICE)
		char_seq_list = [[char2id[ch] if ch in char2id else char2id[UNK] for ch in word]+[char2id[PAD]]* (max_word_length - len(word)) for word in inst.input]
        # char_seq_list = [[char2id[ch] if ch in char2id else char2id[UNK] for ch in word] + [char2id[PAD]] * (max_word_length - len(word)) for word in inst.input]

		#why padding word length?
		#ans: similar with sentence

		inst.char_seq_tensor = torch.LongTensor(char_seq_list).to(NetworkConfig.DEVICE)
		inst.char_seq_len = torch.LongTensor([len(word) for word in inst.input]).to(NetworkConfig.DEVICE)
	gnp = TensorGlobalNetworkParam()
	#what is gnp?
	fm = TriextractNeuralBuilder(gnp, len(vocab2id), len(TagReader.label2id_map), char2id, chars, char_emb_size, charlstm_hidden_dim,)
	#what is fm?
	#ans: feature manager
	fm.labels = labels
	#what is this? label is a function?
	#no use
	fm.load_pretrain(emb_file, vocab2id)
	print(list(TagReader.label2id_map.keys()))
	compiler = TriextractNetworkCompiler(TagReader.label2id_map, max_size, polarity, opinion_offset, opinion_direction)

	evaluator = nereval()
	model = NetworkModel(fm, compiler, evaluator)

	if visual:
		from triplet.hypergraph import Visualizer
		class LinearVisualizer(Visualizer):
			def __init__(self, compiler, fm, labels, span = 50):
				super().__init__(compiler, fm)
				self.labels = labels
				self.span = span
			def nodearr2label(self, node_arr):
				print (node_arr)
				pos, label_id, polarity, direction, opinion_s, opinion_e, node_type = node_arr
				label = self.labels[label_id]
				label_str = str(label) if label else'()'
				direc = '<-'
				new_opinion_s = pos - opinion_e
				new_opinion_e = pos - opinion_s
				if direction == 1:
					direc = '->'
					new_opinion_s = opinion_s +  pos
					new_opinion_e = opinion_e +  pos
				polar = '='
				if polarity ==1:
					polar = '+'
				elif polarity == 2:
					polar = '-'

				return str(pos) + ' ' + label_str + ' '+ str(polar) + ' ' + str(direc) + '(' + str(new_opinion_s) + ','  + str(new_opinion_e) + ')'
			
			def nodearr2color(self, node_arr):
				if node_arr[-1] == 0 or node_arr[-1] == 2:
					return 'blue'
				if 'S' in self.labels[node_arr[1]] or 'B' in self.labels[node_arr[1]]:
					return 'red'
				if node_arr[-1] == 1:
					return 'green'

			def nodearr2coord(self, node_arr):
				span = self.span
				pos, label_id, polarity, direction, opinion_s, opinion_e, node_type = node_arr
				if node_type == 0:
					x = -1
					y = 0
				elif node_type == 2:
					x = pos + 1
					y = 0
				elif node_type == 1:
					x = pos 
					y = 0
					y -= (label_id*0.6+ 0.9* polarity + direction*3.3+opinion_s+opinion_e*4.7) * 2
				return (x,y)

		visualizer = LinearVisualizer(compiler, fm, labels)
		inst = train_insts[0]
		inst.is_labeled = True
		visualizer.visualize_inst(inst)
		exit()


	if batch_size == 1:
		model.learn(train_insts, num_iter, dev_insts, test_insts, optimizer_str, batch_size)
	else:
		model.learn_batch(train_insts, num_iter, dev_insts, test_insts, optimizer_str, batch_size)

	model.load_state_dict(torch.load('best_model.pt'))

	if batch_size == 1:
		results = model.test(test_insts)
	else:
		results = model.test_batch(test_insts, batch_size)

	with open('result.txt', 'w') as f:
		for inst in results:
			f.write(str(inst.get_input()) +'\n')
			f.write(str(inst.get_output()) +'\n')
			f.write(str(inst.get_prediction()) +'\n')
			f.write('\n')
			print(inst.get_input())
			print(inst.get_output())
			print(inst.get_prediction())
			print()

	ret = model.evaluator.eval(test_insts)
	print (ret)






















