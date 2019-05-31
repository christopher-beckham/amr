import torch.nn as nn
from torchvision.models.resnet import (BasicBlock,
                                       conv1x1)
import torch


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def get_network(nf, nf2, n_classes):
    """
    """
    ds = nn.Sequential(
        conv1x1(nf, nf2),
        nn.AvgPool2d(2),
    )
    resblock1 = BasicBlock(inplanes=nf, planes=nf2, stride=2,
                           downsample=ds)
    resblock2 = BasicBlock(inplanes=nf2, planes=nf2, stride=2,
                           downsample=nn.AvgPool2d(2))
    fn = nn.Sequential(
        resblock1,
        resblock2,
        nn.AdaptiveAvgPool2d(1),
        Flatten(),
        nn.Linear(nf2, n_classes),
    )
    return fn
