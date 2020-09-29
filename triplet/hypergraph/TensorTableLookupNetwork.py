from triplet.hypergraph.TensorNetwork import TensorNetwork
from triplet.hypergraph.NetworkIDMapper import NetworkIDMapper

class TensorTableLookupNetwork(TensorNetwork):

    def __init__(self, network_id, inst, nodes, children, node_count, param, compiler, num_stage = -1, num_row = -1, num_hyperedge = -1, staged_nodes = None):
        super().__init__(network_id, inst, param, node_count, num_stage, num_row, num_hyperedge, staged_nodes)
        self.nodes = nodes
        self.children = children
        # self.num_stage = num_stage
        # self.num_row = num_row
        self.nodeid2arr = [None] * self.size
        for k in range(len(self.nodeid2arr)):
            node_long = self.get_node(k)
            self.nodeid2arr[k] = NetworkIDMapper.to_hybrid_node_array(node_long)

    def get_node_array(self, k):
        return self.nodeid2arr[k]
        # node = self.get_node(k)
        # return NetworkIDMapper.to_hybrid_node_array(node)

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

