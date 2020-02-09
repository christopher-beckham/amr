import torch
import numpy as np
from collections import OrderedDict
from torch import optim
from itertools import chain
from .swapgan import SwapGAN
from torch import nn

from torch.distributions import Dirichlet

class KGAN(SwapGAN):

    def __init__(self, k=10, *args, **kwargs):
        super(KGAN, self).__init__(*args, **kwargs)
        if self.cls > 0:
            raise NotImplementedError("FourGAN not implemented for cls > 0")
        self.dirichlet = Dirichlet(torch.FloatTensor([1.0]*k))
        self.k = k

    def sampler(self, bs, f, is_2d, **kwargs):
        """Sampler function, which outputs an alpha which
        you can use to produce a convex combination between
        two examples.

        :param bs: batch size
        :param f: number of units / feature maps at encoding
        :param is_2d: is the bottleneck a 2d tensor?
        :returns: an alpha of shape `(bs, f)` is `is_2d` is set,
          otherwise `(bs, f, 1, 1)`.
        :rtype: 

        """
        if self.mixer == 'mixup':
            with torch.no_grad():
                alpha = self.dirichlet.sample_n(bs)
                if not is_2d:
                    alpha = alpha.reshape(-1, alpha.size(1), 1, 1)
        elif self.mixer == 'fm':
            #alpha = torch.randint(0, 3, size=(bs, f, 1, 1)).long()
            if is_2d:
                alpha = np.zeros((bs, self.k, f)).astype(np.float32)
            else:
                alpha = np.zeros((bs, self.k, f, 1, 1)).astype(np.float32)
            for b in range(bs):
                for j in range(f):
                    alpha[b, np.random.randint(0,self.k), j] = 1.
            alpha = torch.from_numpy(alpha).float()
        else:
            raise Exception("Not implemented for mixup scheme: %s" % self.mixer)
        if self.use_cuda:
            alpha = alpha.cuda()
        return alpha

    def mix(self, enc):
        """Perform mixing operation on the encoding `enc`.
        :param enc: encoding of shape (bs, f) (if 2d) or
          (bs, f, h, w) if 4d.
        """
        perms = [torch.arange(0, enc.size(0))] + \
                [torch.randperm(enc.size(0)) for _ in range(self.k-1)]
        is_2d = True if len(enc.size()) == 2 else False
        alpha = self.sampler(enc.size(0), enc.size(1), is_2d)
        enc_mix = 0.
        #import pdb
        #pdb.set_trace()
        if self.mixer == 'mixup':
            for i in range(len(perms)):
                enc_mix += alpha[:, i:i+1]*enc[perms[i]]
        else:
            for i in range(len(perms)):
                enc_mix = alpha[:, i]*enc[perms[i]]
        return enc_mix, perms[0]
