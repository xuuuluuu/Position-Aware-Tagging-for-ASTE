from triplet.hypergraph.NeuralBuilder import NeuralBuilder
import torch
import torch.nn as nn
from triplet.hypergraph.Utils import *
from transformers import *


class TriextractNeuralBuilder(NeuralBuilder):
	def __init__(self, gnp, voc_size, label_size, char2id, chars, char_emb_size, charlstm_hidden_dim, lstm_hidden_size, token_emb_size, pos_emb_size, dropout, bert_emb ):
		#what is gnp? never used?
		#Ans: transition
		super().__init__(gnp)
		self.token_embed = token_emb_size
		self.label_size = label_size
		self.bert_emb = bert_emb
		print('vocab size: ', voc_size)
		self.char_emb_size = char_emb_size
		lstm_input_size = self.token_embed + bert_emb
		if char_emb_size > 0:
			from triplet.features.char_lstm import CharBiLSTM
			self.char_bilstm = CharBiLSTM(char2id, chars, char_emb_size, charlstm_hidden_dim).to(NetworkConfig.DEVICE)
			lstm_input_size += charlstm_hidden_dim
		self.word_drop = nn.Dropout(dropout).to(NetworkConfig.DEVICE)
		self.word_embed = nn.Embedding(voc_size, self.token_embed).to(NetworkConfig.DEVICE)
		self.pos_embed_range_max = 200
		self.pos_embed = nn.Embedding(self.pos_embed_range_max, pos_emb_size) #100 is the max_len
		self.pos_embed_linear = nn.Linear(pos_emb_size, 1)
		self.lstm_drop = nn.Dropout(dropout).to(NetworkConfig.DEVICE)
		self.rnn = nn.LSTM(lstm_input_size, lstm_hidden_size, batch_first = True, bidirectional = True).to(NetworkConfig.DEVICE)
		self.linear = nn.Linear(lstm_hidden_size *2, label_size).to(NetworkConfig.DEVICE)
		self.linear_span = nn.Linear(int(lstm_hidden_size * 2 ), 1).to(NetworkConfig.DEVICE)
		self.linear_polar = nn.Linear(int(lstm_hidden_size * 3 ), 3).to(NetworkConfig.DEVICE)
		self.lstm_hidden_size = lstm_hidden_size

	def load_pretrain(self, path, word2idx):
		emb = load_emb_glove(path, word2idx, self.token_embed)
		self.word_embed.weight.data.copy_(torch.from_numpy(emb))
		self.word_embed = self.word_embed.to(NetworkConfig.DEVICE)

	def build_nn_graph(self, instance):
		word_vec = self.word_embed(instance.word_seq.unsqueeze(0))
		# generate bert word embedding, not finetune
		if self.bert_emb > 0:
			from bert_serving.client import BertClient
			bc = BertClient(port=8880)
			tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

			tokens = []
			orig_to_tok_index = []# 0 - >0, 1-> len(all word_piece)
			for i, word in enumerate(instance.input):
				orig_to_tok_index.append(len(tokens))
				word_tokens = tokenizer.tokenize(word)
				for sub_token in word_tokens:
					tokens.append(sub_token)
			vec = bc.encode([tokens], show_tokens=True, is_tokenized=True)
			vec = vec[0][:, 1:, :][:, orig_to_tok_index, :]
			bert_vec = torch.tensor(vec).to(word_vec.device)

		word_rep = [word_vec]
		if self.char_emb_size > 0:
			char_seq_tensor = instance.char_seq_tensor.unsqueeze(0)
			char_seq_len = instance.char_seq_len.unsqueeze(0)
			char_features = self.char_bilstm.get_last_hiddens(char_seq_tensor, char_seq_len)			
			word_rep.append(char_features)
		word_rep = torch.cat(word_rep, 2)
		
		#concate bert word embedding
		if self.bert_emb >0:
			word_rep = torch.cat((word_rep, bert_vec),2)

		word_rep = self.word_drop(word_rep)
		lstm_out, (hn, cn) = self.rnn(word_rep, None)
		lstm_out = self.lstm_drop(lstm_out)
		lstm_out = lstm_out.squeeze(0)
		linear_output = self.linear(lstm_out).squeeze(0)
		#score of each node
		instance_len = instance.size()
		lstm_hidden_size = self.lstm_hidden_size

		seg_embs = {}
		for i in range(instance_len):
			for j in range(i, instance_len):
				if i==0 and j+1 == instance_len:
					segment_emb = torch.cat([lstm_out[j][:lstm_hidden_size], lstm_out[i][lstm_hidden_size:]], 0)
				elif i==0 and j+1 < instance_len:
					segment_emb = torch.cat([lstm_out[j][:lstm_hidden_size], lstm_out[i][lstm_hidden_size:] - lstm_out[j+1][lstm_hidden_size:]], 0)
				elif  i>0 and j+1 == instance_len:
					segment_emb = torch.cat([lstm_out[j][:lstm_hidden_size] - lstm_out[i-1][:lstm_hidden_size], lstm_out[i][lstm_hidden_size:]], 0)
				else:
					segment_emb = torch.cat([lstm_out[j][:lstm_hidden_size] - lstm_out[i-1][:lstm_hidden_size] , lstm_out[i][lstm_hidden_size:] - lstm_out[j+1][lstm_hidden_size:]], 0)
				seg_embs[i,j] = segment_emb

		span_score = {}
		polar_score = {}

		for i in range(instance_len):
			for j in range(i, instance_len):
				span_score[i, j] = self.linear_span(seg_embs[i,j])

		offset = [i for i in range(self.pos_embed_range_max)]
		offset = torch.LongTensor(offset)
		offset_score = self.pos_embed(offset)
		offset_score = self.pos_embed_linear(offset_score)

		zero_col = torch.zeros(1, self.label_size).to(NetworkConfig.DEVICE)
		return torch.cat([linear_output, zero_col], 0), span_score, polar_score, offset_score, lstm_out, seg_embs


	def get_nn_score(self, network, parent_k):
		parent_arr = network.get_node_array(parent_k)

		pos = parent_arr[0]
		label_id = parent_arr[1]
		polarity = parent_arr[2]
		direction = parent_arr[3]
		start = parent_arr[4]
		end = parent_arr[5]
		node_type = parent_arr[-1]
		lstm_hidden_size = self.lstm_hidden_size
		if node_type == 0 or node_type == 2:
			return torch.tensor(0.0).to(NetworkConfig.DEVICE)
		else:
			nn_output, span_score, polar_score, offset_score, lstm_out, seg_embs = network.nn_output

			label_str = self.labels[label_id]
			base_score = nn_output[pos][label_id]
			if label_str == 'b' or label_str == 's':

				if direction == 1:
					target_boundary = (pos + start, pos + end)
					pos_target = start+1
				else:
					target_boundary = (pos - end, pos - start)
					pos_target = 100 + end

				pos_score = offset_score[pos_target][0]
				base_score = base_score + span_score[target_boundary][0]  + pos_score + self.linear_polar(torch.cat((seg_embs[target_boundary], lstm_out[pos][lstm_hidden_size:]), 0))[polarity]
				# Ablation
				# base_score = base_score  + pos_score + self.linear_polar(torch.cat((seg_embs[target_boundary], lstm_out[pos][lstm_hidden_size:]), 0))[polarity]

			return base_score


	def get_label_id(self, network, parent_k):
		parent_arr = network.get_node_array(parent_k)
		return parent_arr[1]

		