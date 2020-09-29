import torch
import numpy as np
from enum import Enum

class LossType(Enum):
    CRF = 0,
    SSVM = 1,

class NetworkConfig:
    DEFAULT_CAPACITY_NETWORK = np.asarray([1000,1000, 1000, 1000, 1000], dtype=int)
    NUM_THREADS = 1
    GPU_ID = -1
    DEVICE = torch.device("cpu")  #device = torch.device("cuda:" + args.gpuid)
    NEURAL_LEARNING_RATE = 0.015
    lr_decay = 0.05
    l2=1e-08
    BUILD_GRAPH_WITH_FULL_BATCH = True
    IGNORE_TRANSITION = False
    ECHO_TRAINING_PROGRESS = -1
    ECHO_TEST_RESULT_DURING_EVAL_ON_DEV = False
    LOSS_TYPE = LossType.CRF
    NEUTRAL_BUILDER_ENABLE_NODE_TO_NN_OUTPUT_MAPPING = False
    HYPEREDGE_ORDER = 2


