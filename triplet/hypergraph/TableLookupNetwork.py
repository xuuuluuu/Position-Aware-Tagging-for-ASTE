from triplet.hypergraph.Network import Network


class TableLookupNetwork(Network):

    def __init__(self, network_id, inst, nodes, children, param, compiler, num_stage = -1, num_row = -1):
        super().__init__(network_id, inst, param)
        self.nodes = nodes
        self.children = children
        self.num_stage = num_stage
        self.num_row = num_row

    def get_node(self, k):
        return self.nodes[k]

    def get_children(self, k):
        '''
        :param k: node_k if BaseNetwork;  stage_idx if TensorBaseNetwork
        :return:
        '''
        return self.children[k]

    def get_all_nodes(self):
        return self.nodes

    def get_all_children(self):
        return self.children

    def count_nodes(self):
        return len(self.nodes)

    def is_removed(self, k):
        return False

    def is_root(self, k):
        if self.num_stage == -1:
            return self.count_nodes() - 1 == k
        else:
            return (self.num_stage - 1) * self.num_row == k

