import torch.nn as nn
from .util import Flatten

def get_network(n_in, n_classes):
    fn = nn.Sequential(
        Flatten(),
        nn.Linear(n_in, n_classes)
    )
    return fn
