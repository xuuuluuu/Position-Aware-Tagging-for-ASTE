from __future__ import print_function
from triplet.hypergraph.NetworkConfig import NetworkConfig
import numpy as np



class NetworkIDMapper:
    CAPACITY = NetworkConfig.DEFAULT_CAPACITY_NETWORK

    @staticmethod
    def get_capacity():
        return NetworkIDMapper.CAPACITY

    @staticmethod
    def set_capacity(new_capacity):
        NetworkIDMapper.CAPACITY = new_capacity
        v = np.zeros(len(NetworkIDMapper.CAPACITY), dtype=np.int64)
        for k in range(len(v)):
            v[k] = new_capacity[k] - 1
            u = NetworkIDMapper.to_hybrid_node_array(NetworkIDMapper.to_hybrid_node_ID(v))
            if not np.array_equal(u, v):
                raise Exception("The capacity appears to be too large: ", new_capacity)
        print("Capacity successfully set to: {}".format(new_capacity))

    @staticmethod
    def to_hybrid_node_array(value):
        result = np.zeros(len(NetworkIDMapper.CAPACITY), dtype=np.int64)

        for k in range(len(result) - 1, 0, -1):
            v = value // NetworkIDMapper.CAPACITY[k]
            result[k] = value % NetworkIDMapper.CAPACITY[k]
            value = v


        result[0] = value;
        return result


    @staticmethod
    def to_hybrid_node_ID(array):

        for item in array:
            if isinstance(item, float):
                raise Exception("find float")

        # print('array:', array)
        # print('NetworkIDMapper.CAPACITY:', NetworkIDMapper.CAPACITY)
        if len(array) != len(NetworkIDMapper.CAPACITY):
            raise Exception("array size is ", len(array))


        v = array[0]

        for k in range(1, len(array)):
            #print(array[k], NetworkIDMapper.CAPACITY[k])
            if array[k] >= NetworkIDMapper.CAPACITY[k]:
                raise Exception("Invalid: capacity for ", k, " is ", NetworkIDMapper.CAPACITY[k], " but the value is ", array[k])
            v = v * NetworkIDMapper.CAPACITY[k] + array[k]


        return v


if __name__ == "__main__":
    print('to_hybrid_node_ID:')
    NetworkIDMapper.set_capacity(np.asarray([1000, 1000, 1000]))
    # a = NetworkIDMapper.to_hybrid_node_ID(np.asarray([100, 1]))
    # print(a)
    # b = NetworkIDMapper.to_hybrid_node_array(a)
    # print(b)