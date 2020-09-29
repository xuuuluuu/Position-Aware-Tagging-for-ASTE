# import NetworIDMapper
import math
from triplet.hypergraph.Utils import *
from triplet.hypergraph.NetworkConfig import LossType


class BatchTensorNetwork:

    def __init__(self, fm, batch_network_id, batch_networks, batch_network_id_range): #network_ids, instances, fm, num_stage = -1, num_row = -1, num_hyperedge = -1, staged_nodes = None, batch_size
        # self.network_id = network_id
        # self.inst = instance
        # self.fm = fm
        # self.gnp = fm.gnp

        #self.node2hyperedge = []

        # self.num_stage = num_stage
        # self.num_row = num_row
        # self.num_hyperedge = num_hyperedge
        # self.staged_nodes = staged_nodes
        self.fm = fm
        self.gnp = fm.gnp

        self.batch_network_id = batch_network_id
        self.batch_networks = batch_networks
        self.batch_network_id_range = batch_network_id_range

        self.batch_size = len(batch_networks)
        self.nodeid2labelid = [None] * self.batch_size
        for i in range(self.batch_size):
            self.nodeid2labelid[i] = {}

        self.sizes = torch.LongTensor([network.size for network in batch_networks]).to(NetworkConfig.DEVICE)
        self.max_size = max(self.sizes)
        self.neg_inf_idx = self.sizes + 1
        self.zero_idx = self.neg_inf_idx - 1


        self.max_num_stage = max([network.num_stage for network in batch_networks])



        self.max_num_row = np.amax(np.concatenate([np.expand_dims(np.concatenate((network.num_row, np.full((self.max_num_stage-network.num_stage), 0, dtype=np.int)),axis=0 ), axis = 0) for network in batch_networks], axis=1), axis = 0) #num_stage      the max number of nodes at each stage in the batch
        self.max_num_hyperedge =np.amax(np.concatenate([np.expand_dims(np.concatenate((network.num_hyperedge, np.full((self.max_num_stage-network.num_stage), 0, dtype=np.int)),axis=0 ), axis = 0) for network in batch_networks], axis=1), axis = 0) #num_stage      the max number of hyperedges at each stage in the batch

        max_max_num_row = max(self.max_num_row)
        max_max_num_hyperedge = max(self.max_num_hyperedge)

        self.max_max_num_row = max_max_num_row
        self.max_max_num_hyperedge = max_max_num_hyperedge

        self.children = [None] * self.max_num_stage
        for i in range(self.max_num_stage):
            self.children[i] = np.full((self.batch_size, max_max_num_row, max_max_num_hyperedge, NetworkConfig.HYPEREDGE_ORDER), -1, dtype=np.int64)
            for b in range(self.batch_size):
                self.children[i][b, :, :, 0] = self.neg_inf_idx[b]
                self.children[i][b, :, :, 1] = self.zero_idx[b]


        offset =  self.max_size + 2 #max_max_num_row * max_max_num_hyperedge * NetworkConfig.HYPEREDGE_ORDER
        for batch_id, network in enumerate(batch_networks):
            for stage_idx in range(network.num_stage):
                all_children_list = network.get_children(stage_idx) #num_row * num_hypedge * ORDER

                for row in range(network.num_row[stage_idx]):
                    for hyperedge in range(network.num_hyperedge[stage_idx]):
                        for ORDER in range(NetworkConfig.HYPEREDGE_ORDER):
                            self.children[stage_idx][batch_id, row, hyperedge, ORDER] = all_children_list[row, hyperedge, ORDER] + batch_id * offset



        self.staged_nodes = [None] * self.max_num_stage
        for i in range(self.max_num_stage):
            self.staged_nodes[i] = np.full((self.batch_size, max_max_num_row), 0, dtype=np.int64)
        #offset = self.max_num_row
        for batch_id, network in enumerate(batch_networks):
            for stage_idx in range(network.num_stage):
                staged_nodes = network.staged_nodes[stage_idx]  # num_row * num_hypedge * ORDER

                for row in range(network.num_row[stage_idx]):
                    self.staged_nodes[stage_idx][batch_id][row] = staged_nodes[row]



    def get_network_id(self):
        return self.network_id

    def get_instance(self):
        return self.inst

    def inside(self):

        self.inside_scores = torch.Tensor(self.batch_size, self.max_size + 2).fill_(-math.inf)
        self.inside_scores[:,self.zero_idx] = 0
        self.inside_scores = self.inside_scores.to(NetworkConfig.DEVICE)

        # if not NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING:
        #     emissions = [self.fm.get_nn_score(self, k) for k in range(self.size)]
        #     emissions = torch.stack(emissions, 0)
        #     self.inside_scores[self.staged_nodes[0]] = emissions[self.staged_nodes[0]]  ## REIMINDER: stageIdx = 0.
        # else:
        #emissions = torch.take(self.nn_output, self.nodeid2nn[self.staged_nodes[0]])
        emissions = torch.take(self.nn_batch_output,  self.stagenodes2nodeid2nn[0])

        # self.staged_nodes[stage_idx] : batch_size * max_max_num_row
        #self.inside_scores[self.staged_nodes[0]] = emissions  ## REIMINDER: stageIdx = 0.
        self.inside_scores.scatter_(1, self.staged_nodes[0], emissions)



        for stage_idx in range(1, self.max_num_stage):  ## starting from stage Idx = 1

            childrens_stage = self.get_children(stage_idx)

            for_expr = torch.sum(torch.take(self.inside_scores, childrens_stage), 3)  # batch_size * num_node * num_hyperedge


            # if not NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING:
            #     emission_expr = emissions[self.staged_nodes[stage_idx]].view(self.num_row[stage_idx], 1).expand(self.num_row[stage_idx], self.num_hyperedge[stage_idx])
            # else:
                #emission_expr = torch.take(self.nn_output, self.nodeid2nn[self.staged_nodes[stage_idx]]).view(self.num_row[stage_idx], 1).expand(self.num_row[stage_idx], self.num_hyperedge)

            emission_expr = torch.take(self.nn_batch_output, self.stagenodes2nodeid2nn[stage_idx]).view(self.batch_size, self.max_max_num_row, 1).expand(self.batch_size, self.max_max_num_row, self.max_max_num_hyperedge )
            #batch_size * num_nodes  in this stage * num_hyperedge


            # emission_expr = torch.take(self.nn_output, self.stagenodes2nodeid2nn[:, stage_idx]).view(self.batch_size,
            #                                                                                          self.num_row[:,
            #                                                                                          stage_idx],
            #                                                                                          1).expand(
            #     self.num_row[:, stage_idx], self.num_hyperedge[:, stage_idx])

            if not NetworkConfig.IGNORE_TRANSITION:
                trans_expr = torch.take(self.gnp.transition_mat, self.trans_id[stage_idx])   #batch_size * num_nodes  in this stage * num_hyperedge
                score = for_expr + trans_expr + emission_expr
            else:
                score = for_expr + emission_expr


            if NetworkConfig.LOSS_TYPE == LossType.CRF:
                ret = logSumExp_batch(score)
            else: # LossType.SSVM
                ret, _ = torch.max(score, 2)

            #self.inside_scores[self.staged_nodes[stage_idx]] = ret
            self.inside_scores.scatter_(1, self.staged_nodes[stage_idx], ret)


        # final_inside = self.get_insides()
        #final_inside = self.inside_scores[-3]

        #final_insides = self.inside_scores[:, self.sizes - 1]
        roots = (self.sizes - 1).view(self.batch_size, 1)
        final_insides = torch.gather(self.inside_scores, 1, roots)

        # if math.isinf(final_inside) and final_inside > 0:
        #     raise Exception("Error: network (ID=", self.network_id, ") has zero inside score")

        #weight = self.get_instance().weight
        return final_insides # * weight

    # def get_insides(self):
    #     #print('self.inside_scores[-2]:',self.inside_scores[-2])
    #     return self.inside_scores[-3]


    def get_label_id(self, nid, node_k):
        if node_k not in self.nodeid2labelid[nid]:
            self.nodeid2labelid[nid][node_k] = self.fm.get_label_id(self.batch_networks[nid], node_k)

        return self.nodeid2labelid[nid][node_k]

    def touch(self, is_train = True):

        self.trans_id = [None] * self.max_num_stage #[None] * self.num_stage

        for stage_idx in range(0, self.max_num_stage):
            self.touch_stage(stage_idx)

        self.children = [torch.from_numpy(self.children[stage_idx]).to(NetworkConfig.DEVICE) for stage_idx in range(self.max_num_stage)]

        if NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING:
            # if is_train:
            #     if self.fm.gnp.network2nodeid2nn[self.network_id] == None:
            #         self.fm.gnp.network2nodeid2nn[self.network_id] = self.fm.build_node2nn_output(self)
            #
            #     self.nodeid2nn = torch.LongTensor(self.fm.gnp.network2nodeid2nn[self.network_id])
            # else:
            #     self.nodeid2nn = torch.LongTensor(self.fm.build_node2nn_output(self))


            if is_train:

                if self.fm.gnp.network2stagenodes2nodeid2nn[self.batch_network_id] == None:
                    nodeid2nn = torch.LongTensor(self.fm.build_node2nn_output_batch(self)).to(NetworkConfig.DEVICE) #batch_size * num_nodes
                    self.stagenodes2nodeid2nn = [None] * self.max_num_stage
                    for stage_idx in range(self.max_num_stage):
                            #self.staged_nodes[stage_idx] : batch_size * num_rows

                        self.stagenodes2nodeid2nn[stage_idx] = torch.gather(nodeid2nn, 1, self.staged_nodes[stage_idx])  #nodeid2nn[self.staged_nodes[stage_idx]]


                    self.fm.gnp.network2stagenodes2nodeid2nn[self.batch_network_id] = self.stagenodes2nodeid2nn
                else:
                    self.stagenodes2nodeid2nn = self.fm.gnp.network2stagenodes2nodeid2nn[self.batch_network_id]

            else:
                nodeid2nn = torch.LongTensor(self.fm.build_node2nn_output_batch(self)).to(NetworkConfig.DEVICE)
                self.stagenodes2nodeid2nn = [None] * self.max_num_stage
                for stage_idx in range(self.max_num_stage):
                    self.stagenodes2nodeid2nn[stage_idx] = torch.gather(nodeid2nn, 1, self.staged_nodes[stage_idx])  #nodeid2nn[self.staged_nodes[stage_idx]]


        #self.num_row = torch.LongTensor(self.num_row)


    def touch_stage(self, stage_idx):

        if stage_idx > 0:
            if not NetworkConfig.IGNORE_TRANSITION:
                children_list_k_batch = self.get_children(stage_idx)  # self.batch_size, max_max_num_row, max_max_num_hyperedge, NetworkConfig.HYPEREDGE_ORDER
                trans_stage_np = np.full((self.batch_size, self.max_max_num_row, self.max_max_num_hyperedge ), 0)

                for nid in range(self.batch_size):
                    network = self.batch_networks[nid]

                    if stage_idx >= network.num_stage:
                        continue

                    for idx in range(len(children_list_k_batch[nid])):
                        if idx >= len(network.staged_nodes[stage_idx]):
                            continue

                        node_id = self.staged_nodes[stage_idx][nid, idx]
                        parent_label_id = self.get_label_id(nid, node_id)

                        for children_k_index in range(len(children_list_k_batch[nid][idx])):

                            if children_k_index >= len(network.get_children(stage_idx)[idx]):
                                continue
                            # children_k = children_list_k_batch[nid][idx][children_k_index]
                            children_k = network.get_children(stage_idx)[idx][children_k_index]
                            rhs = [self.get_label_id(nid, child_k) for child_k in children_k if child_k < self.sizes[nid]]
                            if len(rhs) > 0:
                                transition_id = self.gnp.add_transition((parent_label_id, rhs))
                                trans_stage_np[nid][idx][children_k_index] = transition_id


                self.trans_id[stage_idx] = torch.from_numpy(trans_stage_np).to(NetworkConfig.DEVICE)


        self.staged_nodes[stage_idx] = torch.from_numpy(self.staged_nodes[stage_idx]).to(NetworkConfig.DEVICE)


    #@abstractmethod
    def get_children(self, stage_idx) -> np.ndarray:
        return self.children[stage_idx]


    #@abstractmethod
    def get_node(self, stage_idx):
        return self.staged_nodes[stage_idx]


    @abstractmethod
    def count_nodes(self) -> int:
        pass


    def max(self):
        self._max = torch.Tensor(self.batch_size, self.max_size + 2).fill_(-math.inf)  # self.getMaxSharedArray()
        self._max[:,self.zero_idx] = 0
        self._max = self._max.to(NetworkConfig.DEVICE)

        self.max_paths = torch.LongTensor(self.batch_size, self.max_size, 2).fill_(-1)  # self.getMaxPathSharedArray()
        self.max_paths = self.max_paths.to(NetworkConfig.DEVICE)

        # if not NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING:
        #     emissions = [self.fm.get_nn_score(self, k) for k in range(self.size)]
        #     emissions = torch.stack(emissions, 0)
        #     self._max[self.staged_nodes[0]] = emissions[self.staged_nodes[0]]
        # else:
        #     #emissions = torch.take(self.nn_output, self.nodeid2nn[self.staged_nodes[0]])

        emissions = torch.take(self.nn_batch_output, self.stagenodes2nodeid2nn[0])

        #self._max[self.staged_nodes[0]] = emissions  ## REIMINDER: stageIdx = 0.
        self._max.scatter_(1, self.staged_nodes[0], emissions)

        for stage_idx in range(1, self.max_num_stage):  ## starting from stage Idx = 1

            childrens_stage = self.get_children(stage_idx)

            for_expr = torch.sum(torch.take(self._max, childrens_stage), 3)  # this line is same as the above two lines

            # if not NetworkConfig.NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING:
            #     emission_expr = emissions[self.staged_nodes[stage_idx]].view(self.num_row[stage_idx], 1).expand(self.num_row[stage_idx], self.num_hyperedge[stage_idx])
            # else:
            #     #emission_expr = torch.take(self.nn_output, self.nodeid2nn[self.staged_nodes[stage_idx]]).view(self.num_row[stage_idx], 1).expand(self.num_row[stage_idx], self.num_hyperedge)
            emission_expr = torch.take(self.nn_batch_output, self.stagenodes2nodeid2nn[stage_idx]).view(self.batch_size, self.max_max_num_row, 1).expand(self.batch_size, self.max_max_num_row, self.max_max_num_hyperedge)

            if not NetworkConfig.IGNORE_TRANSITION:
                trans_expr = torch.take(self.gnp.transition_mat, self.trans_id[stage_idx])  # this line is same as the above two lines
                score = for_expr + trans_expr + emission_expr
            else:
                score = for_expr + emission_expr

            ret, max_id_list = torch.max(score, 2)  # max_id_list: max_number
            #self._max[self.staged_nodes[stage_idx]], max_id_list
            self._max.scatter_(1, self.staged_nodes[stage_idx], ret)



            max_id_list = max_id_list.view(self.batch_size, self.max_max_num_row, 1, 1).expand(self.batch_size, self.max_max_num_row, 1, NetworkConfig.HYPEREDGE_ORDER)
            ret_id = torch.gather(childrens_stage, 2, max_id_list).squeeze(2) ##  batch_size * num_row * 2
            #self.max_paths[self.staged_nodes[stage_idx]]

            self.max_paths[:, :, 0].scatter_(1, self.staged_nodes[stage_idx], ret_id[:, :, 0])
            self.max_paths[:, :, 1].scatter_(1, self.staged_nodes[stage_idx], ret_id[:, :, 1])

            # childrens_stage: self.batch_size, self.max_num_row, self.max_num_hyperedge, NetworkConfig.HYPEREDGE_ORDER

        self.max_paths = self.max_paths.cpu().numpy()

        offset = self.max_size + 2
        for nid in range(self.batch_size):

            self.max_paths[nid] -= offset * nid
        #self.non_exist_node_id = self.size


    def get_max_path(self, nid, k):
        ## TODO: children might contains non exist node ids careful
        return self.max_paths[nid][k]

