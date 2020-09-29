import networkx as nx
import matplotlib.pyplot as plt
from abc import abstractmethod
from triplet.hypergraph.NetworkCompiler import NetworkCompiler
from triplet.hypergraph.NeuralBuilder import NeuralBuilder
from triplet.hypergraph.TensorTableLookupNetwork import TensorTableLookupNetwork
from triplet.hypergraph.NetworkIDMapper import NetworkIDMapper

class Visualizer():

    def __init__(self, compiler : NetworkCompiler, fm : NeuralBuilder):
        self.compiler = compiler
        self.fm = fm
        self.G = nx.Graph()
        self.font_size = 8
        self.input = None

    @abstractmethod
    def nodearr2label(self, node_arr):
        pass

    @abstractmethod
    def nodearr2color(self, node_arr):
        pass

    @abstractmethod
    def nodearr2coord(self, node_arr):
        pass

    def visualize_inst(self, inst):
        network = self.compiler.compile(0, inst, self.fm)
        input = inst.get_input()
        self.visualize(network, input)


    def visualize(self, network : TensorTableLookupNetwork, input : list):

        self.input = input
        G = self.G

        label_dict = {}
        for node in network.nodes:
            node_arr = NetworkIDMapper.to_hybrid_node_array(node)
            node_arr = tuple(node_arr)
            G.add_node(node_arr, pos=self.nodearr2coord(node_arr))
            label_dict[node_arr] = self.nodearr2label(node_arr)

        color_values = [self.nodearr2color(node_arr) for node_arr in G.nodes()]


        for stage_idx in range(network.num_stage):

            children_list_k = network.get_children(stage_idx)  ## numpy type,  num_row[stage_idx] x num_hyper_edge x 2

            for idx in range(len(children_list_k)):
                node_id = network.staged_nodes[stage_idx][idx]
                node_arr = network.get_node_array(node_id)
                node_arr = tuple(node_arr)

                for children_k_index in range(len(children_list_k[idx])):
                    children_k = children_list_k[idx][children_k_index]
                    for child_k in children_k:
                        if child_k < network.size:
                            child_k_arr = network.get_node_array(child_k)
                            child_k_arr = tuple(child_k_arr)
                            G.add_edge(node_arr, child_k_arr)

        nx.draw(G, nx.get_node_attributes(G, 'pos'), labels=label_dict, with_labels=True, node_color=color_values, font_size=self.font_size) #, node_size = 60
        plt.show()


# def test():
#     G = nx.Graph()
#     # G.add_edges_from(
#     #     [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
#     #      ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])
#     #
#     # val_map = {'A': 1.0,
#     #            'D': 0.5714285714285714,
#     #            'H': 0.0}
#     #
#     # values = [val_map.get(node, 0.25) for node in G.nodes()]
#     #
#     # nx.draw(G, cmap = plt.get_cmap('jet'), node_color = values)
#     # plt.show()
#
#     def map_arr2color(node_arr):
#         return node_arr[0] / 4.0
#
#     def map_arr2label(node_arr):
#         return str(node_arr[1]) + '----' + str(node_arr[0])
#
#     def map_arr2coord(node_arr):
#         return (node_arr[0], node_arr[1])
#
#     nodes = [(0,0), (1,5), (1,3), (3,1)]
#
#     labeldict = {}
#
#     for node in nodes:
#         G.add_node(node, pos = map_arr2coord(node))
#         labeldict[node] = map_arr2label(node)
#
#
#     G.add_edge(nodes[0], nodes[1])
#
#
#     color_values = [ map_arr2color(node) for node in G.nodes()]
#
#
#     nx.draw(G, nx.get_node_attributes(G, 'pos'), labels=labeldict, with_labels=True, node_color = color_values)
#     plt.show()


#test()