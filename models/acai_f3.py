import torch
import numpy as np
from collections import OrderedDict
from torch import optim
from itertools import chain
#from .swapgan import SwapGAN
from .twogan import TwoGAN
from torch import nn

class ACAIF3(TwoGAN):
    """
    Fixed version of ACAI with min() formulation and
    the discriminator also on reconstruction d( g(f(x)) ).
    """

    def __init__(self, *args, **kwargs):
        super(ACAIF3, self).__init__(*args, **kwargs)

    def train_on_instance(self,
                          x_batch,
                          y_batch,
                          **kwargs):
        self._train()
        for key in self.optim:
            self.optim[key].zero_grad()

        ## ------------------
        ## Generator training
        ## ------------------
        enc = self.generator.encode(x_batch)
        dec_enc = self.generator.decode(enc)
        recon_loss = torch.mean(torch.abs(dec_enc-x_batch))
        #disc_g_recon_loss = self.gan_loss_fn(self.disc_x(dec_enc)[0], 0)
        perm = torch.randperm(x_batch.size(0))
        is_2d = True if len(enc.size()) == 2 else False
        alpha = self.sampler(x_batch.size(0), enc.size(1), is_2d)
        enc_mix = alpha*enc + (1.-alpha)*enc[perm]
        dec_enc_mix = self.generator.decode(enc_mix)

        disc_out_g = self.disc_x(dec_enc_mix)[0]
        disc_g_mix_loss = self.gan_loss_fn(disc_out_g, 0) # min(0,1) == 0
        consist_loss = torch.mean(torch.abs(self.generator.encode(dec_enc_mix)-enc_mix))

        gen_loss = self.lamb*recon_loss
        if self.disable_mix is False:
            gen_loss = gen_loss + disc_g_mix_loss + self.beta*consist_loss

        if (kwargs['iter']-1) % self.update_g_every == 0:
            gen_loss.backward()
            self.optim['g'].step()

        ## ----------------------
        ## Discriminator on image
        ## ----------------------
        self.optim['disc_x'].zero_grad()
        d_losses = []
        # Do real images.
        dx_out, _ = self.disc_x(x_batch)
        d_x_real = self.gan_loss_fn(dx_out, 0) # min(0,1) == 0
        # Do reconstruction.
        if self.disable_g_recon is False:
            # Note that this corresponds to the critic regularisation term
            # in the paper, only that right now we don't use a gamma term.
            # The difference between this and our method is that we also have
            # an explicit term in the generator that tries to make reconstructions
            # look realistic as well.
            d_x_fake = self.gan_loss_fn(self.disc_x(dec_enc.detach())[0], 0) # min(0,1) == 0
            d_losses.append(d_x_fake)
        # Do fake imagees.
        if self.mixer == 'mixup':
            #alpha_reshp = alpha
            pass
        elif self.mixer == 'mixup2':
            # 'mixup2' does both batch + channel axis so we
            # just need to do a view() op.
            #alpha_reshp = alpha.view(-1, enc.size(1))
            raise NotImplementedError()
        elif self.mixer == 'fm':
            #alpha_reshp = alpha.view(-1, enc.size(1))
            raise NotImplementedError()
        # Classify x_mix as min(alpha, 1-alpha)
        d_x_mix = self.gan_loss_fn(self.disc_x(dec_enc_mix.detach())[0],
                                    2*torch.min(alpha.view(-1), (1-alpha).view(-1)))

        d_losses.append(d_x_real)
        d_losses.append(d_x_mix)
        d_x = sum(d_losses)
        d_x.backward()
        self.optim['disc_x'].step()

        ## ----------------------------------------------
        ## Classifier on bottleneck (NOTE: for debugging)
        ## ----------------------------------------------
        if self.cls_enc is not None:
            self.optim['cls_enc'].zero_grad()
            if hasattr(self.cls_enc, 'legacy'):
                enc_flat = enc.detach().view(-1, self.cls_enc.n_in)
            else:
                enc_flat = enc.detach()
            cls_enc_out = self.cls_enc(enc_flat)
            cls_enc_preds_log = torch.log_softmax(cls_enc_out, dim=1)
            cls_enc_loss = nn.NLLLoss()(cls_enc_preds_log,
                                        y_batch.argmax(dim=1).long())
            with torch.no_grad():
                cls_enc_preds = torch.softmax(cls_enc_out, dim=1)
                cls_enc_acc = (cls_enc_preds.argmax(dim=1) == y_batch.argmax(dim=1).long()).float().mean()

            cls_enc_loss.backward()
            self.optim['cls_enc'].step()

        losses = {
            'gen_loss': gen_loss.item(),
            'disc_g_mix': disc_g_mix_loss.item(),
            'recon': recon_loss.item(),
            'consist': consist_loss.item(),
            'd_x': d_x.item() / len(d_losses)
        }
        if self.cls_enc is not None:
            losses['cls_enc_loss'] = cls_enc_loss.item()
            losses['cls_enc_acc'] = cls_enc_acc.item()

        outputs = {
            'recon': dec_enc,
            'mix': dec_enc_mix,
            'perm': perm,
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
            perm = torch.randperm(x_batch.size(0))
            is_2d = True if len(enc.size()) == 2 else False
            dec_enc = self.generator.decode(enc)
            alpha = self.sampler(x_batch.size(0), enc.size(1), is_2d)
            enc_mix = alpha*enc + (1.-alpha)*enc[perm]
            dec_enc_mix = self.generator.decode(enc_mix)

            losses = {}
            if self.cls_enc is not None:
                if hasattr(self.cls_enc, 'legacy'):
                    enc_flat = enc.detach().view(-1, self.cls_enc.n_in)
                else:
                    enc_flat = enc.detach()
                cls_enc_out = self.cls_enc(enc_flat)
                cls_enc_preds = torch.softmax(cls_enc_out, dim=1)
                cls_enc_acc = (cls_enc_preds.argmax(dim=1) == y_batch.argmax(dim=1).long()).float().mean()
                losses['cls_enc_Acc'] = cls_enc_acc.item()
            outputs = {
                'recon': dec_enc,
                'mix': dec_enc_mix,
                'perm': perm,
                'input': x_batch,
            }
        return losses, outputs
