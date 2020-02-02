"""
MIT License

Copyright (c) 2017 Christian Cosgrove
Copyright (c) 2019 Christopher Beckham

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
import torch.nn.functional as F
#from .shared.spectral_normalization import SpectralNorm
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
import numpy as np

channels = 3

class CBN2d(nn.Module):
    def __init__(self, y_dim, bn_f):
        super(CBN2d, self).__init__()
        self.bn = nn.BatchNorm2d(bn_f, affine=False)
        self.scale = nn.Linear(y_dim, bn_f)
        nn.init.xavier_uniform(self.scale.weight.data, 1.0)
        self.shift = nn.Linear(y_dim, bn_f)
        nn.init.xavier_uniform(self.shift.weight.data, 0.)
        # https://github.com/pfnet-research/sngan_projection/blob/13c212a7f751c8f0cfd24bc5f35410a61ecb9a45/source/links/categorical_conditional_batch_normalization.py
        # Basically, they initialise with all ones for scale and all zeros for shift.
        # Though that is basically for a one-hot encoding, and we dont have that.
    def forward(self, x, y):
        scale = self.scale(y)
        scale = scale.view(scale.size(0), scale.size(1), 1, 1)
        shift = self.shift(y)
        shift = shift.view(shift.size(0), shift.size(1), 1, 1)
        return self.bn(x)*scale + shift

class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.spec_norm(self.conv1),
                nn.ReLU(),
                self.spec_norm(self.conv2)
            )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                self.spec_norm(self.conv1),
                nn.ReLU(),
                self.spec_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        self.bypass = nn.Sequential()
        if in_channels != out_channels:
            self.bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass.weight.data, np.sqrt(2))
            self.bypass = self.spec_norm(self.bypass)
        if stride != 1:
            self.bypass = nn.Sequential(
                self.bypass,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )


    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, spec_norm=False):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x: x

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.spec_norm(self.conv1),
            nn.ReLU(),
            self.spec_norm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.spec_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class Discriminator(nn.Module):
    def __init__(self,
                 nf,
                 input_nc=3,
                 n_out=1,
                 n_classes=0,
                 sigmoid=False,
                 spec_norm=False):
        """
        


        """
        super(Discriminator, self).__init__()

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x : x
            
        self.model = nn.Sequential(
            FirstResBlockDiscriminator(input_nc, nf,
                                       stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf, nf*2,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf*2, nf*4,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf*4, nf*8,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(nf*8, nf*8,
                                  spec_norm=spec_norm),
            nn.ReLU(),
            nn.AvgPool2d(4),
        )
        self.fc = nn.Linear(nf*8, n_out)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = self.spec_norm(self.fc)

        if n_classes > 0:
            self.cls = nn.Linear(nf*8, n_classes)
            nn.init.xavier_uniform(self.cls.weight.data, 1.)
            self.cls = self.spec_norm(self.cls)
        else:
            self.cls = None
        
        self.sigmoid = sigmoid

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, x.size(1))
        result = self.fc(x)
        if self.sigmoid:
            result = F.sigmoid(result)
        if self.cls is not None:
            cls = F.sigmoid(self.cls(x))
        else:
            cls = None
        return result, cls

#############

class DiscriminatorNew(nn.Module):
    def __init__(self,
                 nf,
                 n_downsampling=4,
                 input_nc=3,
                 n_out=1,
                 n_classes=0,
                 sigmoid=False,
                 spec_norm=False):
        
        super(DiscriminatorNew, self).__init__()

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x : x

        self.resblock1 = FirstResBlockDiscriminator(input_nc, nf,
                                                    stride=2,
                                                    spec_norm=spec_norm)
        
        nfs = [min(2**i,8) for i in range(n_downsampling-1)]
        resblocks_ds = []
        for i in range(len(nfs)-1):
            resblocks_ds.append(ResBlockDiscriminator(nf*nfs[i],
                                                      nf*nfs[i+1],
                                                      stride=2,
                                                      spec_norm=spec_norm))
        self.resblocks_ds = nn.Sequential(*resblocks_ds)
        self.resblock_final = ResBlockDiscriminator(nf*nfs[-1],
                                                    nf*nfs[-1],
                                                    spec_norm=spec_norm)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(4)
        
        self.fc = nn.Linear(nf*8, n_out)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = self.spec_norm(self.fc)

        if n_classes > 0:
            self.cls = nn.Linear(nf*8, n_classes)
            nn.init.xavier_uniform(self.cls.weight.data, 1.)
            self.cls = self.spec_norm(self.cls)
        else:
            self.cls = None
        
        self.sigmoid = sigmoid

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblocks_ds(x)
        x = self.resblock_final(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(-1, x.size(1))
        result = self.fc(x)
        if self.sigmoid:
            result = F.sigmoid(result)
        if self.cls is not None:
            cls = F.sigmoid(self.cls(x))
        else:
            cls = None
        return result, cls






    
##############

class DiscriminatorCode(nn.Module):
    def __init__(self,
                 input_nc,
                 n_out,
                 sigmoid=False,
                 spec_norm=False):
        super(DiscriminatorCode, self).__init__()

        if spec_norm:
            self.spec_norm = SpectralNorm
        else:
            self.spec_norm = lambda x : x
            
        self.model = nn.Sequential(
            ResBlockDiscriminator(input_nc, input_nc*2,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(input_nc*2, input_nc*4,
                                  stride=2, spec_norm=spec_norm),
            ResBlockDiscriminator(input_nc*4, input_nc*8,
                                  spec_norm=spec_norm),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(input_nc*8, n_out)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        self.fc = self.spec_norm(self.fc)

        self.sigmoid = sigmoid

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, x.size(1))
        result = self.fc(x)
        if self.sigmoid:
            result = F.sigmoid(result)
        return result

#################

class CelebACodeClassifier(nn.Module):
    def __init__(self,
                 input_nc,
                 nf,
                 n_classes):
        super(CelebACodeClassifier, self).__init__()
        
        self.model = nn.Sequential(
            ResBlockDiscriminator(input_nc, nf,
                                  stride=2, spec_norm=False),
            ResBlockDiscriminator(nf, nf*2,
                                  stride=2, spec_norm=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(nf*2, n_classes)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, x.size(1))
        result = self.fc(x)
        return F.sigmoid(result)

class CelebACodeClassifierLight(nn.Module):
    def __init__(self,
                 input_nc,
                 n_classes):
        super(CelebACodeClassifierLight, self).__init__()

        self.pool = nn.AvgPool2d(4)


        '''
        self.conv1 = nn.Conv2d(input_nc, input_nc // 4,
                               kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(input_nc // 4, input_nc // 8,
                               kernel_size=3, stride=2)
        '''

        '''
        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU()
        )
        '''
        
        self.fc = nn.Linear(input_nc*2*2, n_classes)
        nn.init.xavier_uniform(self.fc.weight.data, 1.)
        
    def forward(self, x):
        #x = self.model(x)
        x = self.pool(x)
        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        result = self.fc(x)
        return F.sigmoid(result)

class ClassMixer(nn.Module):
    def __init__(self, input_nc, y_dim):
        super(ClassMixer, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(y_dim, input_nc),
            nn.ReLU(),
            nn.Linear(input_nc, input_nc),
            nn.ReLU(),
            nn.Linear(input_nc, input_nc),
            nn.ReLU()
        ) # That ReLU at the end is a bug
        
    def forward(self, h1, h2, y):
        mask = F.sigmoid(self.embed(y))
        mask_4d = mask.view(-1, mask.size(1), 1, 1)
        return mask_4d*h1 + (1.-mask_4d)*h2, mask
