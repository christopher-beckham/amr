import torch
from torch import nn
import torch.nn.functional as F
from .konny.model import BetaVAE_B, Discriminator
from functools import partial

import logging
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)

# TODO: add logging to this
def get_network(n_channels, ndf=32, **kwargs):

    logging.warning("kwargs `ndf` and `ngf` are ignored for this module!")

    gen = BetaVAE_B(nc=n_channels, vae=False)
    disc_x = Discriminator(nc=n_channels, ndf=ndf)
    return {
        'gen': gen,
        'disc_x': disc_x,
        'class_mixer': None
    }
