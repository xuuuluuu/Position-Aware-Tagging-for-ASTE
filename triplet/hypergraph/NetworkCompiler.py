from abc import ABC, abstractmethod



class NetworkCompiler:

    def __init__(self):
        pass

    def compile(self, network_id, instance, fm):

        if instance.is_labeled:
            return self.compile_labeled(network_id, instance, fm)
        else:
            return self.compile_unlabeled(network_id, instance, fm)


    @abstractmethod
    def compile_labeled(self, network_id, inst, param):
        pass

    @abstractmethod
    def compile_unlabeled(self, network_id, inst, param):
        pass

    @abstractmethod
    def decompile(self, network):
        pass
