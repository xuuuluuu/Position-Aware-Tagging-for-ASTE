from triplet.common import LinearInstance


class TagReader():
	### 0 neu, 1 pos, 2 neg
	label2id_map = {'<START>': 0}
	@classmethod
	def read_inst(cls, file, is_labeled, number, opinion_offset):
		insts = []
		inputs = []
		outputs = []
		total_p = 0
		original_p = 0
		f = open(file, 'r', encoding = 'utf-8')

		### read AAAI2020 data
		for line in f:
			line = line.strip()
			line = line.split('####')
			input = line[0].split()  # sentence
			t_output = line[1].split()  # target
			o_output = line[2].split()	# opinion
			raw_pairs = eval(line[3])	# triplets

			# prepare tagging sequence
			output = ['O' for x in range(len(input))]
			for i,t in enumerate(o_output):
				t = t.split('=')[1]
				if t != 'O':
					output[i]= t
			output_o = cls.ot2bieos_o(output)
			output = ['O' for x in range(len(input))]
			for i in range(len(input)):
				if output_o[i]!='O':
					output[i] = output_o[i].split('-')[0]

			# re-format original triplets to jet_o tagging format
			new_raw_pairs = []
			for new_pair in raw_pairs:
				opinion_s = new_pair[1][0]
				opinion_e = new_pair[1][-1]
				target_s = new_pair[0][0]
				target_e = new_pair[0][-1]
				# change sentiment to value --> 0 neu, 1 pos, 2 neg
				if new_pair[2] == 'NEG':
					polarity = 2
				elif new_pair[2] == 'POS':
					polarity = 1
				else:
					polarity = 0
				# check direction and append
				if target_s < opinion_s:
					dire = 0
					new_raw_pairs.append(([opinion_s, opinion_e], polarity, dire, opinion_s-target_e, opinion_s-target_s))
				else:
					dire = 1
					new_raw_pairs.append(([opinion_s, opinion_e], polarity, dire, target_s-opinion_s, target_e-opinion_s))

			new_raw_pairs.sort(key = lambda x: x[0][0]) 
			original_p += len(raw_pairs)

			# remove train data that offset (M) larger than setting and nosiy data during training
			if is_labeled:
				new_pairs = []
				opinion_idxs = []
				remove_idxs = []
				for pair in new_raw_pairs:
					if pair[-1] < opinion_offset and 0<pair[-2]<=pair[-1]:
						new_pairs.append(pair)
						opinion_idxs.extend(list(range(pair[0][0], pair[0][1]+1)))
					else:
						remove_idxs.extend(list(range(pair[0][0], pair[0][-1]+1)))
				for idx in remove_idxs:
					if idx not in opinion_idxs:
						output[idx] = 'O'
			else:
				# keep all original triplets during eval and test for calculating F1 score
				new_pairs = new_raw_pairs
				output = output		

			total_p += len(new_pairs)
			output = (output, new_pairs)

			if len(new_pairs) > 0:
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
		print ('# of original triplets: ', original_p)
		print ('# of triplets for current setup: ', total_p)
		return insts

	@staticmethod
	def ot2bieos_o(ts_tag_sequence):
		"""
		ot2bieos function for opinion
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
