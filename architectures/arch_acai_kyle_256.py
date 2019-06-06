import torch
from torch import nn
import numpy as np
import math
from .shared.spectral_normalization import SpectralNorm
from .arch_kyle import (EncoderDecoder, Discriminator)
from functools import partial

def get_network(n_channels, **kwargs):
    args = {
        'width': 32,
        'latent_width': 4,
        'depth': 16, # nfg?
        'advdepth': 16, #nfd?
        'latent': 16, #4x4x8 = 256
    }
    scales = int(round(math.log(args['width'] // args['latent_width'], 2)))
    gen = EncoderDecoder(scales, n_channels, args['depth'], args['latent'])
    disc_x = Discriminator(scales, args['advdepth'], args['latent'], n_channels)
    return {
        'gen': gen,
        'disc_x': disc_x,
        'class_mixer': None
    }

if __name__ == '__main__':
    tmp = get_network()
    gen, disc = tmp['gen'], tmp['disc_x']
    print(gen)
    xfake = torch.ones((5,1,32,32))
    print(gen.encode(xfake).shape)
    print(disc)
