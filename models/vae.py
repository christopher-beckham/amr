import torch
import numpy as np
from collections import OrderedDict
from torch import optim
from itertools import chain
#from .swapgan import SwapGAN
from .base_ae import BaseAE
from torch import nn
from torch import distributions as distns
from torch.nn import functional as F

class VAE(BaseAE):
    # TODO: Don't subclass SwapGAN for this...

    def __init__(self, *args, **kwargs):
        super(VAE, self).__init__(*args, **kwargs)

        if kwargs['recon_loss'] == 'bce':
            # Assume both input and output are in range
            # [-1, +1].
            self.recon_loss = lambda x,y: F.binary_cross_entropy(x*0.5 + 0.5, y*0.5 + 0.5, size_average=False).\
                div(x.size(0))
        else:
            raise Exception("This loss is not implemented for VAE")

        #if self.beta > 0:
        #    raise Exception("Consistency loss (beta) makes no sense for this class. Make sure beta == 0")

    def mix(self, enc, **kwargs):
        """
        """
        perm = torch.randperm(enc.size(0))
        #enc1, _, _, _ = self._sample_from_normal(enc)
        #enc2, _, _, _ = self._sample_from_normal(enc[perm])

        # Take the means only. Don't sample anything
        # here.
        enc1 = enc[:, 0:(enc.size(1)//2)]
        enc2 = enc[:, 0:(enc.size(1)//2)][perm]
        is_2d = True if len(enc1.size()) == 2 else False
        alpha = self.sampler(enc1.size(0), enc1.size(1), is_2d, **kwargs)
        enc_mix = alpha*enc1 + (1.-alpha)*enc2
        return enc_mix, perm

    def sample(self, x_batch):
        enc = self.generator.encode(x_batch)
        enc_sampled, _, _, _ = self._sample_from_normal(enc)
        return self.generatod.decode(enc_sampled)

    def _sample_from_normal(self, enc):
        enc_mu, enc_var = enc[:, 0:(enc.size(1)//2)], enc[:, (enc.size(1)//2)::]
        distn = distns.normal.Normal(enc_mu, enc_var)
        enc_sampled = distn.rsample()
        return enc_sampled, enc_mu, enc_var, distn

    def train_on_instance(self,
                          x_batch,
                          y_batch,
                          **kwargs):
        self._train()
        for key in self.optim:
            self.optim[key].zero_grad()

        # Reconstruction loss.
        enc = self.generator.encode(x_batch)
        enc_sampled, enc_mu, enc_var, distn = self._sample_from_normal(enc)
        dec_enc = self.generator.decode(enc_sampled)
        # todo: move 0.5 into recon_loss
        recon_loss = self.recon_loss(dec_enc, x_batch)


        # KL loss.
        prior = distns.normal.Normal(torch.zeros_like(enc_mu),
                                     torch.ones_like(enc_var))
        kl_loss = distns.kl.kl_divergence(distn, prior).mean()

        dec_enc_rand = self.generator.decode(prior.rsample())

        gen_loss = self.lamb*recon_loss + self.beta*kl_loss

        gen_loss.backward()
        self.optim['g'].step()

        losses = {
            'gen_loss': gen_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
        }

        outputs = {
            'recon': dec_enc,
            'sample': dec_enc_rand,
            'input': x_batch,
        }
        return losses, outputs

    def eval_on_instance(self,
                         x_batch,
                         y_batch,
                         **kwargs):
        self._eval()
        with torch.no_grad():
            enc = self.generator.encode(x_batch)
            enc_mu, enc_var = enc[:, 0:(enc.size(1)//2)], enc[:, (enc.size(1)//2)::]
            distn = distns.normal.Normal(enc_mu, enc_var)
            enc_sampled = distn.rsample()
            dec_enc = self.generator.decode(enc_sampled)

            prior = distns.normal.Normal(torch.zeros_like(enc_mu),
                                         torch.ones_like(enc_var))
            dec_enc_rand = self.generator.decode(prior.rsample())

            losses = {}
            outputs = {
                'recon': dec_enc,
                'sample': dec_enc_rand,
                'input': x_batch,
            }
        return losses, outputs
