import torch
from torch import nn
import torch.nn.functional as F
#from .shared import networks
from .konny.model import BetaVAE_B
from . import discriminators
from functools import partial

# TODO: add logging to this
def get_network(n_channels, ndf, **kwargs):
    gen = BetaVAE_B(nc=n_channels, vae=True)
    disc_x = discriminators.Discriminator(nf=ndf,
                                          input_nc=n_channels,
                                          n_classes=1,
                                          sigmoid=True,
                                          spec_norm=True)
    return {
        'gen': gen,
        'disc_x': disc_x,
        'class_mixer': None
    }
