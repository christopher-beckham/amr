import torch
import numpy as np
from collections import OrderedDict
from torch import optim
from itertools import chain
from .swapgan import SwapGAN
from torch import nn

class TwoGAN(SwapGAN):

    def __init__(self, *args, **kwargs):
        super(TwoGAN, self).__init__(*args, **kwargs)

    def sampler_mixup(self, bs, f, is_2d, p=None):
        """Mixup sampling function

        :param bs: batch size
        :param f: number of features / channels
        :param is_2d: should sampled alpha be 2D, instead of 4D?
        :param p: Bernoulli parameter `p`. If this is None, then
          we simply sample p ~ U(0,1).
        :returns: an alpha of shape (bs, 1) if `is_2d`, otherwise
          (bs, 1, 1, 1).
        :rtype: 

        """
        shp = (bs, 1) if is_2d else (bs, 1, 1, 1)
        if p is None:
            alphas = []
            for i in range(bs):
                alpha = np.random.uniform(0, 1)
                alphas.append(alpha)
        else:
            alphas = [p]*bs
        alphas = np.asarray(alphas).reshape(shp)
        alphas = torch.from_numpy(alphas).float()
        if self.use_cuda:
            alphas = alphas.cuda()
        return alphas

    def sampler_mixup2(self, bs, f, is_2d, p=None):
        """Mixup2 sampling function

        :param bs: batch size
        :param f: number of features / channels
        :param is_2d: should sampled alpha be 2D, instead of 4D?
        :param p: Bernoulli parameter `p`. If this is None, then
          we simply sample p ~ U(0,1).
        :returns: an alpha of shape (bs, f) if `is_2d`, otherwise
          (bs, f, 1, 1).
        :rtype: 

        """
        shp = (bs, f) if is_2d else (bs, f, 1, 1)
        if p is None:
            alphas = np.random.uniform(0, 1, size=shp)
        else:
            alphas = np.zeros(shp)+p
        alphas = torch.from_numpy(alphas).float()
        if self.use_cuda:
            alphas = alphas.cuda()
        return alphas

    def sampler_fm(self, bs, f, is_2d, p=None):
        """Bernoulli mixup sampling function

        :param bs: batch size
        :param f: number of features / channels
        :param is_2d: should sampled alpha be 2D, instead of 4D?
        :param p: Bernoulli parameter `p`. If this is `None`, then
          we simply sample m ~ Bern(p), where p ~ U(0,1).
        :returns: an alpha of shape (bs, f) if `is_2d`, otherwise
          (bs, f, 1, 1).
        :rtype: 

        """
        shp = (bs, f) if is_2d else (bs, f, 1, 1)
        if p is None:
            alphas = torch.bernoulli(torch.rand(shp)).float()
        else:
            rnd_state = np.random.RandomState(0)
            rnd_idxs = np.arange(0, f)
            rnd_state.shuffle(rnd_idxs)
            rnd_idxs = torch.from_numpy(rnd_idxs)
            how_many = int(p*f)
            alphas = torch.zeros(shp).float()
            if how_many > 0:
                rnd_idxs = rnd_idxs[0:how_many]
                alphas[:, rnd_idxs] += 1.
        if self.use_cuda:
            alphas = alphas.cuda()
        return alphas

    def sampler_fm2(self, bs, f, is_2d, p=None):
        """Bernoulli mixup sampling function. Has
          same expectation as fm but higher variance.

        :param bs: batch size
        :param f: number of features / channels
        :param is_2d: should sampled alpha be 2D, instead of 4D?
        :param p: Bernoulli parameter `p`. If this is `None`, then
          we simply sample m ~ Bern(p), where p ~ U(0,1).
        :returns: an alpha of shape (bs, f) if `is_2d`, otherwise
          (bs, f, 1, 1).
        :rtype: 

        """
        shp = (bs, f) if is_2d else (bs, f, 1, 1)
        if p is None:
            this_p = torch.rand(1).item()
            alphas = torch.bernoulli(torch.zeros(shp)+this_p).float()
        else:
            rnd_state = np.random.RandomState(0)
            rnd_idxs = np.arange(0, f)
            rnd_state.shuffle(rnd_idxs)
            rnd_idxs = torch.from_numpy(rnd_idxs)
            how_many = int(p*f)
            alphas = torch.zeros(shp).float()
            if how_many > 0:
                rnd_idxs = rnd_idxs[0:how_many]
                alphas[:, rnd_idxs] += 1.
        if self.use_cuda:
            alphas = alphas.cuda()
        return alphas

    def sample(self, x_batch):
        """Output a random mixup sample.

        :param x_batch: the batch to mix. This will be mixed
          with an index-permuted version of itself, i.e.,
          mix between `x_batch` and `x_batch[perm_indices]`.
        :returns: 
        :rtype: 

        """
        self._eval()
        if self.use_cuda:
            x_batch = x_batch.cuda()
        with torch.no_grad():
            enc = self.generator.encode(x_batch)
            is_2d = True if len(enc.size()) == 2 else False
            perm = torch.randperm(x_batch.size(0))
            alpha = self.sampler(enc.size(0), enc.size(1), is_2d)
            enc_mix = alpha*enc + (1.-alpha)*enc[perm]
            dec_enc_mix = self.generator.decode(enc_mix)
            return dec_enc_mix

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
            return self.sampler_mixup(bs, f, is_2d, **kwargs)
        elif self.mixer == 'mixup2':
            return self.sampler_mixup2(bs, f, is_2d, **kwargs)
        elif self.mixer == 'fm':
            return self.sampler_fm(bs, f, is_2d, **kwargs)
        elif self.mixer == 'fm2':
            return self.sampler_fm2(bs, f, is_2d, **kwargs)

    def mix(self, enc, perm=None, **kwargs):
        """Perform mixing operation on the encoding `enc`.
        :param enc: encoding of shape (bs, f) (if 2d) or
          (bs, f, h, w) if 4d.
        :param perm: the permuted indices of `enc` to mix
          with, such that the mix is between `enc` and `enc[perm]`.
          If this is `None`, the method will generate its own
          internally.`
        """
        if perm is None:
            perm = torch.randperm(enc.size(0))
        is_2d = True if len(enc.size()) == 2 else False
        alpha = self.sampler(enc.size(0), enc.size(1), is_2d, **kwargs)
        enc_mix = alpha*enc + (1.-alpha)*enc[perm]
        return enc_mix, perm
