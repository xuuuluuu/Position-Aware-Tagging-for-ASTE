from abc import ABC, abstractmethod
import torch.nn as nn

class NeuralBuilder(nn.Module):

    def __init__(self, gnp):
        super(NeuralBuilder, self).__init__()
        self.gnp = gnp

    ##initilaize neural network

    def get_param_g(self):
        return self.gnp

    @abstractmethod
    def generate_batches(self, train_insts, batch_size):
        pass

    @abstractmethod
    def build_nn_graph(self, instance):
        pass

    @abstractmethod
    def build_nn_graph_batch(self, batch_input_seqs):
        pass

    @abstractmethod
    def get_nn_score(self, network, parent_k):
        ## given a node parent_k, return score.
        # parent_k -> network.nn_output  score
        pass

    @abstractmethod
    def get_nn_score_batch(self, network, parent_k):
        ## given a node parent_k, return score.
        # parent_k -> network.nn_output  score
        pass

    @abstractmethod
    def build_node2nn_output(self, network):
        pass

    @abstractmethod
    def build_node2nn_output_batch(self, network):
        pass
