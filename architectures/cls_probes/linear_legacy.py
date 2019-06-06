import torch.nn as nn

def get_network(n_in, n_classes):
    fn = nn.Linear(n_in, n_classes)
    fn.legacy = True ## HACK to maintain bw compat with old models
    fn.n_in = n_in
    return fn
