from abc import ABC, abstractmethod

class Instance:

    def __init__(self, instance_id, weight, input = None, output = None):
        self.instance_id = instance_id
        self.weight = weight
        self.input = input
        self.output = output
        self.labeled_instance = None
        self.unlabeled_instance = None
        self.prediction = None
        self.is_labeled = True

    def set_instance_id(self, inst_id):
        self.instance_id = inst_id

    def get_instance_id(self):
        return self.instance_id

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight

    def set_labeled(self):
        self.is_labeled = True

    def set_unlabeled(self):
        self.is_labeled = False

    def remove_output(self):
        self.output = None

    # def is_labeled(self):
    #     return self.is_labeled

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def duplicate(self):
        pass

    @abstractmethod
    def removeOutput(self):
        pass

    @abstractmethod
    def removePrediction(self):
        pass

    @abstractmethod
    def get_input(self):
        pass

    @abstractmethod
    def get_output(self):
        pass

    @abstractmethod
    def get_prediction(self):
        pass

    @abstractmethod
    def set_prediction(self):
        pass

    @abstractmethod
    def has_output(self):
        pass

    @abstractmethod
    def has_prediction(self):
        pass

    def get_islabeled(self):
        return self.is_labeled

    def get_labeled_instance(self):
        if self.is_labeled:
            return self
        else:
            self.labeled_instance

    def set_label_instance(self, inst):
        self.labeled_instance = inst

    def get_unlabeled_instance(self):
        self.unlabeled_instance

    def set_unlabel_instance(self, inst):
        self.unlabeled_instance = inst



if __name__ == "__main__":
    inst = Instance(1, 1.0)
    print(inst.input)
    print(inst.output)
    inst.size()