# import NetworIDMapper
import math
from triplet.hypergraph.Utils import *
from triplet.hypergraph.NetworkConfig import LossType


class TensorNetwork:

    def __init__(self, network_id, instance, fm, node_count, num_stage = -1, num_row = -1, num_hyperedge = -1, staged_nodes = None):
        self.network_id = network_id
        self.inst = instance
        self.fm = fm
        self.gnp = fm.gnp
        self.nodeid2labelid = {}
        #self.node2hyperedge = []

        self.num_stage = num_stage
        self.num_row = num_row
        self.num_hyperedge = num_hyperedge
        self.staged_nodes = staged_nodes

        self.size = node_count
        self.neg_inf_idx = -1
        self.zero_idx = self.neg_inf_idx - 1


    def get_network_id(self):
        return self.network_id

    ##TODO: multithread
    def get_thread_id(self):
        pass

    def get_instance(self):
        return self.inst

    def inside(self):

        self.inside_scores = torch.Tensor(self.size + 2).fill_(-math.inf)
        self.inside_scores[self.zero_idx] = 0
        self.inside_scores = self.inside_scores.to(NetworkConfig.DEVICE)

        if not NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING:
            emissions = [self.fm.get_nn_score(self, k) for k in range(self.size)]
            emissions = torch.stack(emissions, 0)
            self.inside_scores[self.staged_nodes[0]] = emissions[self.staged_nodes[0]]  ## REIMINDER: stageIdx = 0.
        else:
            #emissions = torch.take(self.nn_output, self.nodeid2nn[self.staged_nodes[0]])
            emissions = torch.take(self.nn_output,  self.stagenodes2nodeid2nn[0])
            self.inside_scores[self.staged_nodes[0]] = emissions  ## REIMINDER: stageIdx = 0.




        for stage_idx in range(1, self.num_stage):  ## starting from stage Idx = 1

            childrens_stage = self.get_children(stage_idx)

            for_expr = torch.sum(torch.take(self.inside_scores, childrens_stage), 2)  # this line is same as the above two lines

            if not NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING:
                emission_expr = emissions[self.staged_nodes[stage_idx]].view(self.num_row[stage_idx], 1).expand(self.num_row[stage_idx], self.num_hyperedge[stage_idx])
            else:
                #emission_expr = torch.take(self.nn_output, self.nodeid2nn[self.staged_nodes[stage_idx]]).view(self.num_row[stage_idx], 1).expand(self.num_row[stage_idx], self.num_hyperedge)
                emission_expr = torch.take(self.nn_output, self.stagenodes2nodeid2nn[stage_idx]).view(self.num_row[stage_idx], 1).expand(self.num_row[stage_idx], self.num_hyperedge[stage_idx])

            if not NetworkConfig.IGNORE_TRANSITION:
                trans_expr = torch.take(self.gnp.transition_mat, self.trans_id[stage_idx])  # this line is same as the above two lines
                score = for_expr + trans_expr + emission_expr
            else:
                score = for_expr + emission_expr

            if NetworkConfig.LOSS_TYPE == LossType.CRF:
                self.inside_scores[self.staged_nodes[stage_idx]] = logSumExp(score) #torch.max(score, 1) #
            else: # LossType.SSVM
                self.inside_scores[self.staged_nodes[stage_idx]], _ = torch.max(score, 1)  # torch.max(score, 1) #


        # final_inside = self.get_insides()
        final_inside = self.inside_scores[-3]

        if math.isinf(final_inside) and final_inside > 0:
            raise Exception("Error: network (ID=", self.network_id, ") has zero inside score")

        weight = self.get_instance().weight
        return final_inside * weight

    # def get_insides(self):
    #     #print('self.inside_scores[-2]:',self.inside_scores[-2])
    #     return self.inside_scores[-3]


    def get_label_id(self, node_k):
        if node_k not in self.nodeid2labelid:
            self.nodeid2labelid[node_k] = self.fm.get_label_id(self, node_k)

        return self.nodeid2labelid[node_k]

    def touch(self, is_train = True):

        self.trans_id = [None] * self.num_stage

        for stage_idx in range(1, self.num_stage):
            self.touch_stage(stage_idx)

        self.children = [torch.from_numpy(self.children[stage_idx]).to(NetworkConfig.DEVICE) for stage_idx in range(self.num_stage)]

        if NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING:
            # if is_train:
            #     if self.fm.gnp.network2nodeid2nn[self.network_id] == None:
            #         self.fm.gnp.network2nodeid2nn[self.network_id] = self.fm.build_node2nn_output(self)
            #
            #     self.nodeid2nn = torch.LongTensor(self.fm.gnp.network2nodeid2nn[self.network_id])
            # else:
            #     self.nodeid2nn = torch.LongTensor(self.fm.build_node2nn_output(self))


            if is_train:
                if self.fm.gnp.network2stagenodes2nodeid2nn[self.network_id] == None:
                    nodeid2nn = torch.LongTensor(self.fm.build_node2nn_output(self)).to(NetworkConfig.DEVICE)
                    self.stagenodes2nodeid2nn = [None] * self.num_stage
                    for stage_idx in range(self.num_stage):
                        self.stagenodes2nodeid2nn[stage_idx] = nodeid2nn[self.staged_nodes[stage_idx]]
                    self.fm.gnp.network2stagenodes2nodeid2nn[self.network_id] = self.stagenodes2nodeid2nn
                else:
                    self.stagenodes2nodeid2nn = self.fm.gnp.network2stagenodes2nodeid2nn[self.network_id]
            else:
                nodeid2nn = torch.LongTensor(self.fm.build_node2nn_output(self)).to(NetworkConfig.DEVICE)
                self.stagenodes2nodeid2nn = [None] * self.num_stage
                for stage_idx in range(self.num_stage):
                    self.stagenodes2nodeid2nn[stage_idx] = nodeid2nn[self.staged_nodes[stage_idx]]


        #self.num_row = torch.LongTensor(self.num_row)


    def touch_stage(self, stage_idx):

        if not NetworkConfig.IGNORE_TRANSITION:
            children_list_k = self.get_children(stage_idx)  ## numpy type,  num_row[stage_idx] x num_hyper_edge x 2
            trans_stage_np = np.full((self.num_row[stage_idx], self.num_hyperedge[stage_idx]), 0)

            for idx in range(len(children_list_k)):
                node_id = self.staged_nodes[stage_idx][idx]
                parent_label_id = self.get_label_id(node_id)

                for children_k_index in range(len(children_list_k[idx])):
                    children_k = children_list_k[idx][children_k_index]
                    rhs = [self.get_label_id(child_k) for child_k in children_k if child_k < self.size]
                    if len(rhs) > 0:
                        transition_id = self.gnp.add_transition((parent_label_id, rhs))
                        trans_stage_np[idx][children_k_index] = transition_id
            self.trans_id[stage_idx] = torch.from_numpy(trans_stage_np).to(NetworkConfig.DEVICE)

        self.staged_nodes[stage_idx] = torch.from_numpy(self.staged_nodes[stage_idx]).to(NetworkConfig.DEVICE)


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
        self._max = torch.Tensor(self.size + 2).fill_(-math.inf)  # self.getMaxSharedArray()
        self._max[self.zero_idx] = 0
        self._max = self._max.to(NetworkConfig.DEVICE)

        self.max_paths = torch.LongTensor(self.size, 2).fill_(-1)  # self.getMaxPathSharedArray()
        self.max_paths = self.max_paths.to(NetworkConfig.DEVICE)

        if not NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING:
            emissions = [self.fm.get_nn_score(self, k) for k in range(self.size)]
            emissions = torch.stack(emissions, 0)
            self._max[self.staged_nodes[0]] = emissions[self.staged_nodes[0]]
        else:
            #emissions = torch.take(self.nn_output, self.nodeid2nn[self.staged_nodes[0]])
            emissions = torch.take(self.nn_output, self.stagenodes2nodeid2nn[0])
            self._max[self.staged_nodes[0]] = emissions  ## REIMINDER: stageIdx = 0.

        for stage_idx in range(1, self.num_stage):  ## starting from stage Idx = 1

            childrens_stage = self.get_children(stage_idx)

            for_expr = torch.sum(torch.take(self._max, childrens_stage), 2)  # this line is same as the above two lines

            if not NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING:
                emission_expr = emissions[self.staged_nodes[stage_idx]].view(self.num_row[stage_idx], 1).expand(self.num_row[stage_idx], self.num_hyperedge[stage_idx])
            else:
                #emission_expr = torch.take(self.nn_output, self.nodeid2nn[self.staged_nodes[stage_idx]]).view(self.num_row[stage_idx], 1).expand(self.num_row[stage_idx], self.num_hyperedge)
                emission_expr = torch.take(self.nn_output, self.stagenodes2nodeid2nn[stage_idx]).view(self.num_row[stage_idx], 1).expand(self.num_row[stage_idx], self.num_hyperedge[stage_idx])

            if not NetworkConfig.IGNORE_TRANSITION:
                trans_expr = torch.take(self.gnp.transition_mat, self.trans_id[stage_idx])  # this line is same as the above two lines
                score = for_expr + trans_expr + emission_expr
            else:
                score = for_expr + emission_expr

            self._max[self.staged_nodes[stage_idx]], max_id_list = torch.max(score, 1)  # max_id_list: max_number

            max_id_list = max_id_list.view(self.num_row[stage_idx], 1, 1).expand(self.num_row[stage_idx], 1, NetworkConfig.HYPEREDGE_ORDER)
            self.max_paths[self.staged_nodes[stage_idx]] = torch.gather(childrens_stage, 1, max_id_list).squeeze(1) ## max_number, 2


        self.max_paths = self.max_paths.cpu().numpy()

        self.non_exist_node_id = self.size


    def get_max_path(self, k):
        ## TODO: children might contains non exist node ids careful
        return self.max_paths[k]

