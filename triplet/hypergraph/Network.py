# import NetworIDMapper
from triplet.hypergraph.NetworkIDMapper import NetworkConfig, NetworkIDMapper
import math

from abc import abstractmethod
import torch
import numpy as np
from triplet.hypergraph.Utils import log_sum_exp

class Network:

    def __init__(self, network_id, instance, fm):
        self.network_id = network_id
        self.inst = instance
        self.fm = fm
        self.gnp = fm.gnp
        self.nodeid2labelid = {}
        self.node2hyperedge = []

        self.nodeid2childrenids = []


    def get_network_id(self):
        return self.network_id

    ##TODO: multithread
    def get_thread_id(self):
        pass

    def get_instance(self):
        return self.inst

    def inside(self):
        # self.inside_scores = [torch.tensor([-math.inf])] * (self.count_nodes() + 1)  #[torch.tensor([0.0])] * self.count_nodes()
        self.inside_scores = torch.Tensor(self.count_nodes() + 1).fill_(-math.inf)
        self.inside_scores[-1] = 0
        self.inside_scores = self.inside_scores.to(NetworkConfig.DEVICE)
        for k in range(self.count_nodes()):
            self.get_inside(k)
        if math.isinf(self.get_insides()) and self.get_insides() > 0:
            raise Exception("Error: network (ID=", self.network_id, ") has zero inside score")

        weight = self.get_instance().weight
        return self.get_insides() * weight

    def get_insides(self):
        return self.inside_scores[self.count_nodes() - 1]

    def get_inside(self, k):

        current_label_id = self.nodeid2labelid[k]

        children_list_k = self.get_children(k)

        size = len(children_list_k)

        emission_expr = self.fm.get_nn_score(self, k)

        if len(children_list_k[0]) > 0:
            children_list_index_tensor = self.nodeid2childrenids[k]
            inside_view = self.inside_scores.view(1, self.count_nodes() + 1).expand(size, self.count_nodes() + 1)

            for_expr = torch.sum(torch.gather(inside_view, 1, children_list_index_tensor), 1)

            tuple_list_tensor = self.node2hyperedge[k]
            trans_expr = torch.gather(self.gnp.transition_mat[current_label_id], 0, tuple_list_tensor)

            score = for_expr + trans_expr + emission_expr
        else:
            score = emission_expr
        self.inside_scores[k] = log_sum_exp(score)

    def get_label_id(self, node_k):
        if node_k not in self.nodeid2labelid:
            self.nodeid2labelid[node_k] = self.fm.get_label_id(self, node_k)

        return self.nodeid2labelid[node_k]

    def touch(self):
        self.empty_index = self.count_nodes()
        for k in range(self.count_nodes()):
            self.touch_node(k)

    def touch_node(self, k):
        '''
        :param k:
        :return:
        '''
        if self.is_removed(k):
            return

        children_list_k = self.get_children(k)
        parent_label_id = self.get_label_id(k)

        children_list_tensor = []
        tuple_id_list_tensor = []

        for children_k_index in range(len(children_list_k)):
            children_k = children_list_k[children_k_index]
            rhs = tuple([self.get_label_id(child_k) for child_k in children_k])
            # self.gnp.add_transition((parent_label_id, rhs))
            tuple_id = self.gnp.add_transition((parent_label_id, rhs))
            tuple_id_list_tensor.append(tuple_id)


            children_k_list = list(children_k) #[children_k[0], children_k[1]] if len(children_k) > 1 else [children_k[0], self.empty_index]
            while(len(children_k_list) < 2):
                children_k_list.append(self.empty_index)

            children_k_list = torch.tensor(children_k_list)
            children_list_tensor.append(children_k_list)

        children_list_tensor = torch.stack(children_list_tensor, 0).to(NetworkConfig.DEVICE)
        self.nodeid2childrenids.append(children_list_tensor)

        tuple_id_list_tensor = torch.tensor(tuple_id_list_tensor).to(NetworkConfig.DEVICE)
        self.node2hyperedge.append(tuple_id_list_tensor)


    def get_node_array(self, k):
        node = self.get_node(k)
        return NetworkIDMapper.to_hybrid_node_array(node)


    @abstractmethod
    def get_children(self, k) -> np.ndarray:
        pass


    @abstractmethod
    def get_node(self, k):
        pass


    @abstractmethod
    def count_nodes(self) -> int:
        pass


    @abstractmethod
    def is_removed(self, k):
        pass

    def max(self):
        self._max = torch.Tensor(self.count_nodes() + 1).fill_(-math.inf)  # self.getMaxSharedArray()
        self._max[-1] = 0.0
        self._max = self._max.to(NetworkConfig.DEVICE)
        self._max_paths = [-1] * self.count_nodes()  # self.getMaxPathSharedArray()
        for k in range(self.count_nodes()):
            self.maxk(k)

    def get_max_path(self, k):

        return self._max_paths[k]

    def maxk(self, k):

        children_list_k = self.get_children(k)
        size = len(children_list_k)
        current_label_id = self.nodeid2labelid[k]
        emission_expr = self.fm.get_nn_score(self, k)

        if len(children_list_k[0]) > 0:
            children_list_index_tensor = self.nodeid2childrenids[k]
            max_view = self._max.view(1, self.count_nodes() + 1).expand(size, self.count_nodes() + 1)
            for_expr = torch.sum(torch.gather(max_view, 1, children_list_index_tensor), 1)

            tuple_list_tensor = self.node2hyperedge[k]
            trans_expr = torch.gather(self.gnp.transition_mat[current_label_id], 0, tuple_list_tensor)

            score = for_expr + trans_expr + emission_expr

        else:
            score = emission_expr


        self._max[k], max_id = torch.max(score, 0)
        self._max_paths[k] = children_list_k[max_id]

    # def max(self):
    #     self._max = [torch.tensor(0.0)] * self.count_nodes() # self.getMaxSharedArray()
    #     self._max_paths = [torch.tensor(0)] * self.count_nodes() # self.getMaxPathSharedArray()
    #     for k in range(self.count_nodes()):
    #         self.maxk(k)
    #
    #
    #
    # def get_max_path(self, k):
    #
    #     return self._max_paths[k]
    #
    #
    # def maxk(self, k):
    #     if self.is_removed(k):
    #         self._max[k] = float("-inf")
    #         return
    #
    #     children_list_k = self.get_children(k)
    #     self._max[k] = torch.tensor(-math.inf)
    #
    #     current_label_id = self.nodeid2labelid[k]
    #     emission = self.fm.extract_helper(self, k)
    #
    #     if len(children_list_k[0]) > 0:
    #         children_k_index = 0
    #         for children_k in children_list_k:
    #             #children_k = children_list_k[children_k_index]
    #
    #             #fa = self.param.extract(self, k, children_k, children_k_index)
    #             #score = fa.get_score(self.param)
    #
    #             transition = self.gnp.transition_mat[current_label_id][self.node2hyperedge[k][children_k_index]]
    #             score = transition + emission
    #
    #             #print('children_k:', children_k, '\tself._max:', len(self._max), '\tcount_nodes():', self.count_nodes())
    #             score += sum([self._max[child_k] for child_k in children_k])
    #
    #             # for child_k in children_k:
    #             #     score += self._max[child_k]
    #
    #             # print('maxk:',type(score), '\t', type(self._max[k]))
    #             if score >= self._max[k]:
    #                 self._max[k] = score
    #                 self._max_paths[k] = torch.tensor(children_k)
    #
    #             children_k_index += 1
    #     else:
    #         self._max[k] = emission
    #         self._max_paths[k] = torch.tensor(-1)
