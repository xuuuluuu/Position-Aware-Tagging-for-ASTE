from triplet.hypergraph.NetworkCompiler import NetworkCompiler
from triplet.hypergraph.NetworkIDMapper import NetworkIDMapper
from triplet.hypergraph.TensorBaseNetwork import TensorBaseNetwork
import numpy as np
START = '<START>'
STOP = '<STOP>'

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
		NetworkIDMapper.set_capacity(np.asarray([max_size, 10, self.POLARITY, self.OPINION_DIRECTION, self.MAX_OPINION_OFFSET, self.MAX_OPINION_OFFSET, 3], dtype = np.int64))
		#pos, label_id, opinion_direction, opinion_start, opinion_end, node_type
		self._all_nodes = None
		self._all_children = None
		self._max_size = max_size
		print ('The max size: ', self._max_size)
		print ("Building generic network..")
		self.build_generic_network()

	def to_root(self, size):
		return self.to_node(size-1, len(self.labels)-1, 2, self.OPINION_DIRECTION - 1, self.MAX_OPINION_OFFSET - 1, self.MAX_OPINION_OFFSET - 1, 2)

	def to_leaf(self, ):
		return self.to_node(0, 0, 0, 0, 0, 0, 0)

	def to_BStag(self, pos, label_id, opinion):
		return self.to_node(pos, label_id, opinion[0], opinion[1], opinion[2], opinion[3], 1) #TODO: get label id

	def to_tag(self, pos, label_id, ):
		return self.to_node(pos, label_id, 2, 1, self.MAX_OPINION_OFFSET - 1, self.MAX_OPINION_OFFSET - 1, 1)

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
		children = [leaf]
		m=0
		for i in range(inst.size()):
			label = output_label[i]
			if 'S' in label or 'B' in label:
				for p in opinion_pair:
					if p[0][0] == i:
						bstag_node = self.to_BStag(i, self.label2id[label], p[1:])
						builder.add_node(bstag_node)
						builder.add_edge(bstag_node, children)
						children = [bstag_node]
						break # only keep one target tag when overlapped during training
			else:
				tag_node = self.to_tag(i, self.label2id[label])
				builder.add_node(tag_node)
				builder.add_edge(tag_node, children)
				children = [tag_node]

		root = self.to_root(inst.size())
		builder.add_node(root)
		builder.add_edge(root, children)
		network = builder.build(network_id, inst, param, self)
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
		children_label = ['<START>']
		for i in range(inst.size()):
			current = [] # changed from NOne list to empty list, might slow the process
			current_label = []
			for l in range(len(self.labels)):
				# so there is no more start and stop nodes?
				#ans: check outside loop
				if l == self.label2id['<START>'] or l==self.label2id['<STOP>']:
					continue

				if 'S' == self.labels[l]:
					for p in range(self.POLARITY):
						for s in range(1, self.MAX_OPINION_OFFSET):
							for e in range(s, self.MAX_OPINION_OFFSET):
								for d in range(self.OPINION_DIRECTION):
									if d == 0:
										if i-e>-1:
											bstag_node = self.to_BStag(i, l, (p, d, s, e))
											builder.add_node(bstag_node)
											for c, child in enumerate(children):
												if child is not None:
													if children_label[c]!='I' and children_label[c]!='B': 
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
													if children_label[c]!='I' and children_label[c]!='B':
													 	builder.add_edge(bstag_node, [child])
											current.append(bstag_node)
											current_label.append(self.labels[l])

				if 'B' == self.labels[l]:
					for p in range(self.POLARITY):
						for s in range(1, self.MAX_OPINION_OFFSET):
							for e in range(s, self.MAX_OPINION_OFFSET):
								for d in range(self.OPINION_DIRECTION):
									if d == 0:
										if i-e>-1:
											bstag_node = self.to_BStag(i, l, (p, d, s, e))
											builder.add_node(bstag_node)
											for c, child in enumerate(children):
												if child is not None:
													if children_label[c]!='I' and children_label[c]!='B': 
														builder.add_edge(bstag_node, [child])
											current.append(bstag_node)
											current_label.append(self.labels[l])
									else:
										if i+e < inst.size():
											bstag_node = self.to_BStag(i, l, (p, d, s, e))
											builder.add_node(bstag_node)
											for c, child in enumerate(children):
												if child is not None:
													if children_label[c]!='I' and children_label[c]!='B': 
														builder.add_edge(bstag_node, [child])
											current.append(bstag_node)
											current_label.append(self.labels[l])									
				elif 'I' == self.labels[l]:
					tag_node = self.to_tag(i, l)
					builder.add_node(tag_node)
					for c, child in enumerate(children):
						if child is not None:
							if children_label[c]!='E' and children_label[c]!='S' and children_label[c]!='O':  
								builder.add_edge(tag_node, [child])
					current.append(tag_node)
					current_label.append(self.labels[l])
				elif 'E' == self.labels[l]:
					tag_node = self.to_tag(i, l)
					builder.add_node(tag_node)
					for c, child in enumerate(children):
						if child is not None:
							if children_label[c]=='B' or children_label[c]=='I':
								builder.add_edge(tag_node, [child])
					current.append(tag_node)
					current_label.append(self.labels[l])
				elif 'O' == self.labels[l]:
					tag_node = self.to_tag(i, l)
					builder.add_node(tag_node)
					for c, child in enumerate(children):
						if child is not None:
							if children_label[c]!='I' and children_label[c]!='I':
								builder.add_edge(tag_node, [child])
					current.append(tag_node)
					current_label.append(self.labels[l])

			children = current
			children_label = current_label
		root = self.to_root(i+1)
		builder.add_node(root)
		for c, child in enumerate(children):
			if child is not None:
				if children_label[c]!='I' and children_label[c]!='B':
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
			if prediction[size - i - 1] == 'S':
				pairs.append(([size - i - 1, size - i - 1], child_arr[-5], child_arr[-4], child_arr[-3], child_arr[-2]))
			elif prediction[size - i - 1] == 'B':
				for r in range(size - i - 1, size):
					if prediction[r] == 'E':
						pairs.append(([size - i - 1, r], child_arr[-5], child_arr[-4], child_arr[-3], child_arr[-2]))
						break
			# print('check id ', child_arr[-5])
			# print('----')
			curr_idx = child

		inst.set_prediction((prediction, pairs))
		return inst

