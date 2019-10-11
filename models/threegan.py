import torch
from torch import nn
from torch.distributions import Dirichlet
from torch import optim
import numpy as np
from collections import OrderedDict
from itertools import chain
from .swapgan import SwapGAN

class ThreeGAN(SwapGAN):
    """This class is old. You should instead use KGAN,
    which allows you to choose the k value for mixing.
    """

    def __init__(self, *args, **kwargs):
        super(ThreeGAN, self).__init__(*args, **kwargs)
        if self.cls > 0:
            raise NotImplementedError("ThreeGAN not implemented for cls > 0")
        self.dirichlet = Dirichlet(torch.FloatTensor([1.0, 1.0, 1.0]))

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
                alpha = np.zeros((bs, 3, f)).astype(np.float32)
            else:
                alpha = np.zeros((bs, 3, f, 1, 1)).astype(np.float32)
            for b in range(bs):
                for j in range(f):
                    alpha[b, np.random.randint(0,3), j] = 1.
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
        perm = torch.randperm(enc.size(0))
        perm2 = torch.randperm(enc.size(0))
        is_2d = True if len(enc.size()) == 2 else False
        alpha = self.sampler(enc.size(0), enc.size(1), is_2d)
        if self.mixer == 'mixup':
            enc_mix = alpha[:, 0:1]*enc + \
                      alpha[:, 1:2]*enc[perm] + \
                      alpha[:, 2:3]*enc[perm2]
        else:
            enc_mix = alpha[:, 0]*enc + \
                      alpha[:, 1]*enc[perm] + \
                      alpha[:, 2]*enc[perm2]
        return enc_mix, perm
