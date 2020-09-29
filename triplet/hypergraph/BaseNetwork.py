from triplet.hypergraph.TableLookupNetwork import TableLookupNetwork
from triplet.hypergraph.NetworkIDMapper import NetworkIDMapper
from triplet.hypergraph.Utils import eprint


class BaseNetwork(TableLookupNetwork):

    def __init__(self, network_id, instance, nodes, children, node_count, param, compiler):
        super().__init__(network_id, instance, nodes, children, param, compiler)
        self.node_count = node_count

        self.is_visible = [False for i in range(node_count)]


    def count_nodes(self):
        return self.node_count




    class NetworkBuilder:

        def __init__(self):
            self._children_tmp = {}

        @staticmethod
        def builder():
            return BaseNetwork.NetworkBuilder()

        @staticmethod
        def quick_build(network_id, instance, nodes, children, node_count, param, compiler):
            return BaseNetwork(network_id, instance, nodes, children, node_count, param, compiler)

        def add_node(self, node):
            if node in self._children_tmp:
                return False
            else:
                self._children_tmp[node] = None
                return True

        def add_edge(self, parent, child):
            self.check_link_validity(parent, child)
            if parent not in self._children_tmp or self._children_tmp[parent] == None:
                self._children_tmp[parent] = []
            existing_children = self._children_tmp[parent]
            for k in range(len(existing_children)):
                if existing_children[k] == child:
                    return False
            existing_children.append(child)
            return True

        def num_nodes_tmp(self):
            return len(self._children_tmp)

        def get_children_tmp(self, node):
            return self._children_tmp[node]

        def get_nodes_tmp(self):
            nodes = []
            for key in self._children_tmp:
                nodes.append(key)
            return nodes

        def remove_tmp(self, node):
            if node not in self._children_tmp:
                return False
            self._children_tmp.pop(node)
            return True

        def contains_node(self, node):
            return node in self._children_tmp

        def contains_edge(self, parent, child):
            if parent not in self._children_tmp:
                return False
            children = self._children_tmp[parent]
            for presentChild in children:
                if presentChild == child:
                    return True
            return False

        def build(self, network_id, instance, param, compiler):

            values = []

            for node in self._children_tmp:
                values.append(node)

            node_list = [0 for i in range(len(self._children_tmp))]
            is_visible = [False for i in range(len(node_list))]

            nodes_value2id_map = {}


            values.sort()
            for k in range(len(values)):
                node_list[k] = values[k]
                is_visible[k] = True
                nodes_value2id_map[node_list[k]] = k
            # node_list.sort()
            children_list = [None for i in range(len(node_list))]

            for parent in self._children_tmp:
                #print("builder parent: ", parent, " chidren_tmp: " , self._children_tmp[parent])
                parent_index = nodes_value2id_map[parent]
                childrens = self._children_tmp[parent]
                if childrens == None:
                    children_list[parent_index] = [[]]  # new int[1][0]

                else:
                    children_list[parent_index] = [None for i in range(len(childrens))]

                    for k in range(len(children_list[parent_index])):
                        children = childrens[k]
                        children_index = []

                        for m in range(len(children)):
                            if children[m] < 0:
                                children_index.append(children[m])
                            else:
                                children_index.append(nodes_value2id_map[children[m]])

                        children_list[parent_index][k] = children_index
                    #print("parent is :", parent, " children_list, ", children_list[parent_index])
            for k in range(len(children_list)):
                if children_list[k] == None:
                    children_list[k] = [[]]

            result = None

            # if network_id != None or instance != None or param != None or compiler != None:
            result = BaseNetwork.NetworkBuilder.quick_build(network_id, instance, node_list, children_list, len(node_list), param, compiler)
            # TODO: handle the case when network_id != None or instance != None or param != None or compiler != None
            # this is for rudimentary network builder

            result.is_visible = is_visible
            return result


        def check_link_validity(self, parent, children):
            for child in children:
                if child < 0:
                    continue

                if child >= parent:
                    eprint(NetworkIDMapper.to_hybrid_node_array(parent))
                    eprint(NetworkIDMapper.to_hybrid_node_array(child))
                    eprint()
                    raise Exception(
                        "In an edge, the parent needs to have larger node ID in order to have a proper schedule for inference. Violation: ",
                        parent, "\t", children)

            self.check_node_validity(parent)

            for child in children:
                if child < 0:
                    continue

                self.check_node_validity(child)


        def check_node_validity(self, node):
            if node not in self._children_tmp:
                raise Exception("Node not found:", NetworkIDMapper.to_hybrid_node_array(node))






