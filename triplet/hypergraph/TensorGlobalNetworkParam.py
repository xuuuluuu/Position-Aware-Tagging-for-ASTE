import threading
import torch
import torch.nn as nn
from triplet.hypergraph.NetworkConfig import NetworkConfig


class TensorGlobalNetworkParam(nn.Module):

    def __init__(self):
        super(TensorGlobalNetworkParam, self).__init__()
        self.locked = False
        self._size = 0

        self.tuple2id = {}
        self.tuple2id[()] = 0
        self.transition_mat = None

        self.lock = threading.Lock()

        self.network2nodeid2nn = None
        self.network2stagenodes2nodeid2nn = None

    def set_network2nodeid2nn_size(self, size):
        self.network2nodeid2nn = [None] * size
        self.network2stagenodes2nodeid2nn = [None] * size

    def set_network2nodeid2nn_batch_size(self, num_batch):
        self.network2nodeid2nn = [None] * num_batch * 2
        self.network2stagenodes2nodeid2nn = [None] * num_batch * 2

    def is_locked(self):
        return self.locked

    def size(self):
        return self._size

    def finalize_transition(self):
        self.tuple_size = len(self.tuple2id)
        if NetworkConfig.IGNORE_TRANSITION:
            pass
            #self.transition_mat = nn.Parameter(torch.zeros(self.tuple_size)).to(NetworkConfig.DEVICE)
            #self.transition_mat.requires_grad = False
        else:
            self.transition_mat = nn.Parameter(torch.randn(self.tuple_size)).to(NetworkConfig.DEVICE)
            self.transition_mat.data[0] = -float('inf') # padding

        self.locked = True

    def get_transition_id(self, parent_label_id, children_label_ids) -> int:
        # print(self.tuple2id)
        return self.tuple2id[tuple([parent_label_id] + children_label_ids)]
        
    def add_transition(self, transition):
        with self.lock:
            parent_label_id, children_label_ids = transition
            t = tuple([parent_label_id] + children_label_ids)
            if not self.locked and t not in self.tuple2id:
                tuple2id_size = len(self.tuple2id)
                self.tuple2id[t] = tuple2id_size

            return self.tuple2id[t]

    def print_transition(self, id2labels):
        try:
            for tuple in self.tuple2id:
                labels = [id2labels[l] for l in tuple]
                print(labels, self.transition_mat[self.tuple2id[tuple]].data)
        except:
            pass
