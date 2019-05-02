import torch
from torch import nn
import numpy as np
import math
from .shared.spectral_normalization import SpectralNorm
from functools import partial

activation = nn.LeakyReLU
norm_layer = partial(nn.InstanceNorm2d, affine=True)

# authors use this initializer, but it doesn't seem essential
def Initializer(layers, slope=0.2):
    for layer in layers:
        if hasattr(layer, 'weight'):
            w = layer.weight.data
            std = 1/np.sqrt((1 + slope**2) * np.prod(w.shape[:-1]))
            w.normal_(std=std)  
        if hasattr(layer, 'bias'):
            layer.bias.data.zero_()

def Encoder(scales, depth, latent, colors, instance_norm=False, spec_norm=False):
    if spec_norm is False:
        sn = lambda x: x
    else:
        sn = SpectralNorm
    layers = []
    layers.append(nn.Conv2d(colors, depth, 1, padding=1))
    kp = depth
    for scale in range(scales):
        k = depth << scale
        if instance_norm:
            layers.extend([sn(nn.Conv2d(kp, k, 3, padding=1)), norm_layer(k), activation()])
            layers.extend([sn(nn.Conv2d(k, k, 3, padding=1)), norm_layer(k), activation()])
        else:
            layers.extend([sn(nn.Conv2d(kp, k, 3, padding=1)), activation()])
            layers.extend([sn(nn.Conv2d(k, k, 3, padding=1)), activation()])
        layers.append(nn.AvgPool2d(2))
        kp = k
    k = depth << scales
    if instance_norm:
        layers.extend([sn(nn.Conv2d(kp, k, 3, padding=1)), norm_layer(k), activation()])
    else:
        layers.extend([sn(nn.Conv2d(kp, k, 3, padding=1)), activation()])
    layers.append(sn(nn.Conv2d(k, latent, 3, padding=1)))
    Initializer(layers)
    return nn.Sequential(*layers)

def Decoder(scales, depth, latent, colors, instance_norm=False):
    layers = []
    kp = latent
    for scale in range(scales - 1, -1, -1):
        k = depth << scale
        if instance_norm:
            layers.extend([nn.Conv2d(kp, k, 3, padding=1), norm_layer(k), activation()])
            layers.extend([nn.Conv2d(k, k, 3, padding=1), norm_layer(k), activation()])
        else:
            layers.extend([nn.Conv2d(kp, k, 3, padding=1), activation()])
            layers.extend([nn.Conv2d(k, k, 3, padding=1), activation()])           
        layers.append(nn.Upsample(scale_factor=2))
        kp = k
    if instance_norm:
        layers.extend([nn.Conv2d(kp, depth, 3, padding=1), norm_layer(depth), activation()])
    else:
        layers.extend([nn.Conv2d(kp, depth, 3, padding=1), activation()])
    layers.append(nn.Conv2d(depth, colors, 3, padding=1))
    Initializer(layers)
    return nn.Sequential(*layers)

class EncoderDecoder(nn.Module):
    def __init__(self, scales, n_channels, depth, latent):
        super(EncoderDecoder, self).__init__()
        encoder = Encoder(scales,
                          depth,
                          latent,
                          n_channels,
                          instance_norm=True)
        decoder = Decoder(scales,
                          depth,
                          latent,
                          n_channels,
                          instance_norm=True)
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec
    def encode(self, x):
        return self.encoder(x)
    def decode(self, x):
        return self.decoder(x)

class Discriminator(nn.Module):
    def __init__(self, scales, depth, latent, colors):
        super().__init__()
        self.encoder = Encoder(scales, depth, latent, colors,
                               spec_norm=True)
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.mean(x, -1)
        return torch.sigmoid(x), None
