
from triplet.hypergraph.NetworkCompiler import NetworkCompiler
from triplet.hypergraph.NetworkIDMapper import NetworkIDMapper
from triplet.hypergraph.TensorBaseNetwork import TensorBaseNetwork
import numpy as np
START = '<START>'
STOP = '<STOP>'
class TriextractNetworkCompiler(NetworkCompiler):
	def __init__(self, label_map, max_size, polarity, target_offset, target_direction):
		# label map is a dict
		# max_size is the max length of the sentence in train valid and test dataset
		super().__init__()
		self.labels = ['x'] * len(label_map)
		self.label2id = label_map
		for key in self.label2id:
			self.labels[self.label2id[key]] = key
		self.MAX_OFFSET = target_offset
		self.TARGET_DIRECTION = target_direction
		self.POLARITY = polarity
		NetworkIDMapper.set_capacity(np.asarray([max_size, 10, self.POLARITY, self.TARGET_DIRECTION, self.MAX_OFFSET, self.MAX_OFFSET, 3], dtype = np.int64))
		# pos, label_id, polar, target_direction, target_start, target_end, node_type
		self._all_nodes = None
		self._all_children = None
		self._max_size = max_size
		print ('The max size: ', self._max_size)


	def to_root(self, size):
		return self.to_node(size-1, len(self.labels)-1, 2, self.TARGET_DIRECTION - 1, self.MAX_OFFSET - 1, self.MAX_OFFSET - 1, 2)

	def to_leaf(self, ):
		return self.to_node(0, 0, 0, 0, 0, 0, 0)

	def to_BStag(self, pos, label_id, target):
		# target: (polarity, direction, start, end)
		return self.to_node(pos, label_id, target[0], target[1], target[2], target[3], 1)

	def to_tag(self, pos, label_id, ):
		return self.to_node(pos, label_id, 2, 1, self.MAX_OFFSET - 1, self.MAX_OFFSET - 1, 1)

	def to_node(self, pos, label_id, target_sentiment, target_dir, target_s, target_e, node_type):
		return NetworkIDMapper.to_hybrid_node_ID(np.asarray([pos, label_id, target_sentiment, target_dir, target_s, target_e, node_type]))

	def compile_labeled(self, network_id, inst, param):
		# build a labeled network
		builder = TensorBaseNetwork.NetworkBuilder.builder()
		leaf = self.to_leaf()
		builder.add_node(leaf)
		output_label, target_pair  = inst.get_output()
		children = [leaf]
		labled_idx = []
		for i in range(inst.size()):
			label = output_label[i]
			if 's' in label or 'b' in label:
				for p in target_pair:
					if p[0][0] == i:
						bstag_node = self.to_BStag(i, self.label2id[label], p[1:])
						builder.add_node(bstag_node)
						builder.add_edge(bstag_node, children)
						children = [bstag_node]
						break # only keep one opinion tag when overlapped during training
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


	def compile_unlabeled(self, network_id, inst, param):
		# return self.compile_labeled(network_id, inst, param)
		builder = TensorBaseNetwork.NetworkBuilder.builder()
		leaf = self.to_leaf()
		builder.add_node(leaf)

		children = [leaf]
		children_label = ['<START>']
		for i in range(inst.size()):
			current = [] # changed from None list to empty list, might slow the process
			current_label = []
			for l in range(len(self.labels)):
				# so there is no more start and stop nodes?
				#ans: check outside loop
				if l == self.label2id['<START>'] or l==self.label2id['<STOP>']:
					continue

				if 's' == self.labels[l]:
					for p in range(self.POLARITY):
						for s in range(1, self.MAX_OFFSET):
							for e in range(s, self.MAX_OFFSET):
								for d in range(self.TARGET_DIRECTION):
									if d == 0:
										if i-e>-1:
											bstag_node = self.to_BStag(i, l, (p, d, s, e))
											builder.add_node(bstag_node)
											for c, child in enumerate(children):
												if child is not None:
													if children_label[c]!='i' and children_label[c]!='b': 
											# 			print('++++', children_label[c])
														builder.add_edge(bstag_node, [child])
											current.append(bstag_node)
											current_label.append(self.labels[l])
									else:
										if i+e < inst.size():
											bstag_node = self.to_BStag(i, l, (p, d, s, e))
											builder.add_node(bstag_node)
											for c, child in enumerate(children):
												if child is not None:
													if children_label[c]!='i' and children_label[c]!='b':
													 	builder.add_edge(bstag_node, [child])
											current.append(bstag_node)
											current_label.append(self.labels[l])

				if 'b' == self.labels[l]:
					for p in range(self.POLARITY):
						for s in range(1, self.MAX_OFFSET):
							for e in range(s, self.MAX_OFFSET):
								for d in range(self.TARGET_DIRECTION):
									if d == 0:
										if i-e>-1:
											bstag_node = self.to_BStag(i, l, (p, d, s, e))
											builder.add_node(bstag_node)
											for c, child in enumerate(children):
												if child is not None:
													if children_label[c]!='i' and children_label[c]!='b': 
														builder.add_edge(bstag_node, [child])
											current.append(bstag_node)
											current_label.append(self.labels[l])
									else:
										if i+e < inst.size():
											bstag_node = self.to_BStag(i, l, (p, d, s, e))
											builder.add_node(bstag_node)
											for c, child in enumerate(children):
												if child is not None:
													if children_label[c]!='i' and children_label[c]!='b': 
														builder.add_edge(bstag_node, [child])
											current.append(bstag_node)
											current_label.append(self.labels[l])									
				elif 'i' == self.labels[l]:
					tag_node = self.to_tag(i, l)
					builder.add_node(tag_node)
					for c, child in enumerate(children):
						if child is not None:
							if children_label[c]!='e' and children_label[c]!='s' and children_label[c]!='O':  
								builder.add_edge(tag_node, [child])
					current.append(tag_node)
					current_label.append(self.labels[l])
				elif 'e' == self.labels[l]:
					tag_node = self.to_tag(i, l)
					builder.add_node(tag_node)
					for c, child in enumerate(children):
						if child is not None:
							if children_label[c]=='b' or children_label[c]=='i':
								builder.add_edge(tag_node, [child])
					current.append(tag_node)
					current_label.append(self.labels[l])
				elif 'O' == self.labels[l]:
					tag_node = self.to_tag(i, l)
					builder.add_node(tag_node)
					for c, child in enumerate(children):
						if child is not None:
							if children_label[c]!='i' and children_label[c]!='i':
								builder.add_edge(tag_node, [child])
					current.append(tag_node)
					current_label.append(self.labels[l])

			children = current
			children_label = current_label
		root = self.to_root(i+1)
		builder.add_node(root)
		for c, child in enumerate(children):
			if child is not None:
				if children_label[c]!='i' and children_label[c]!='b':
					builder.add_edge(root, [child])

		network = builder.build(network_id, inst, param, self)
		return network


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
			#pos, label_id, target_direction, target_start, target_end, node_type
			if prediction[size - i - 1] == 's':
				pairs.append(([size - i - 1, size - i - 1], child_arr[-5], child_arr[-4], child_arr[-3], child_arr[-2]))
			elif prediction[size - i - 1] == 'b':
				for r in range(size - i - 1, size):
					if prediction[r] == 'e':
						pairs.append(([size - i - 1, r], child_arr[-5], child_arr[-4], child_arr[-3], child_arr[-2]))
						break
			# print('check id ', child_arr[-5])
			# print('----')
			curr_idx = child

		inst.set_prediction((prediction, pairs))
		return inst