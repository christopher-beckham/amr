import torch.nn as nn

def get_network(n_in, n_classes):
    fn = nn.Sequential(
        Flatten(),
        nn.Linear(n_in, n_classes)
    )
    return fn
