import torch
import numpy as np
from collections import OrderedDict
from torch import optim
from itertools import chain
from swapgan import SwapGAN
from torch import nn

from torch.distributions import Dirichlet

class ThreeGAN(SwapGAN):

    def __init__(self, *args, **kwargs):
        super(ThreeGAN, self).__init__(*args, **kwargs)
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
            alpha = self.dirichlet.sample_n(bs)
        elif self.mixer == 'fm':
            #alpha = torch.randint(0, 3, size=(bs, f, 1, 1)).long()
            alpha = torch.zeros((bs, 3, f)).float()
            for b in range(bs):
                for j in range(alpha.shape[2]):
                    alpha[b, np.random.randint(0, alpha.shape[1]), j] = 1.            
        else:
            raise Exception("Not implemented for mixup scheme: %s" % self.mixer)
        if self.use_cuda:
            alpha = alpha.cuda()
        return alpha
        
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
        disc_g_recon_loss = self.gan_loss_fn(self.disc_x(dec_enc)[0], 1)
        perm = torch.randperm(x_batch.size(0))
        perm2 = torch.randperm(x_batch.size(0))
        is_2d = True if len(enc.size()) == 2 else False
        alpha = self.sampler(x_batch.size(0), enc.size(1), is_2d)

        if self.mixer == 'mixup':
            enc_mix = alpha[:, 0].view(x_batch.size(0), 1, 1, 1)*enc + \
                      alpha[:, 1].view(x_batch.size(0), 1, 1, 1)*enc[perm] + \
                      alpha[:, 2].view(x_batch.size(0), 1, 1, 1)*enc[perm2]
        else:
            enc_mix = alpha[:, 0].view(x_batch.size(0), enc.size(1), 1, 1)*enc + \
                      alpha[:, 1].view(x_batch.size(0), enc.size(1), 1, 1)*enc[perm] + \
                      alpha[:, 2].view(x_batch.size(0), enc.size(1), 1, 1)*enc[perm2]
        
        dec_enc_mix = self.generator.decode(enc_mix)
        disc_g_mix_loss = self.gan_loss_fn(self.disc_x(dec_enc_mix)[0], 1)
        if self.beta > 0:
            consist_loss = torch.mean(torch.abs(self.generator.encode(dec_enc_mix)-enc_mix))
        else:
            consist_loss = torch.FloatTensor([0.])
            if self.use_cuda:
                consist_loss = consist_loss.cuda()
        
        gen_loss = self.lamb*recon_loss
        if self.disable_g_recon is False:
            gen_loss = gen_loss + disc_g_recon_loss
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
        dx_out, cx_out = self.disc_x(x_batch)
        d_x_real = self.gan_loss_fn(dx_out, 1)
        d_losses.append(d_x_real)
        if self.disable_g_recon is False:
            # Do reconstruction.
            d_x_fake = self.gan_loss_fn(self.disc_x(dec_enc.detach())[0], 0)
            d_losses.append(d_x_fake)
        if self.disable_mix is False:
            # Do mixes.
            d_out_mix = self.gan_loss_fn(self.disc_x(dec_enc_mix.detach())[0], 0)
            d_losses.append(d_out_mix)
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
            'disc_g_recon': disc_g_recon_loss.item(),
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
            dec_enc = self.generator.decode(enc)
            recon_loss = torch.mean(torch.abs(dec_enc-x_batch))
            #disc_g_recon_loss = self.gan_loss_fn(self.disc_x(dec_enc)[0], 0)
            perm = torch.randperm(x_batch.size(0))
            perm2 = torch.randperm(x_batch.size(0))
            is_2d = True if len(enc.size()) == 2 else False
            alpha = self.sampler(x_batch.size(0), enc.size(1), is_2d)
            if self.mixer == 'mixup':
                enc_mix = alpha[:, 0].view(x_batch.size(0), 1, 1, 1)*enc + \
                          alpha[:, 1].view(x_batch.size(0), 1, 1, 1)*enc[perm] + \
                          alpha[:, 2].view(x_batch.size(0), 1, 1, 1)*enc[perm2]
            else:
                enc_mix = alpha[:, 0].view(x_batch.size(0), enc.size(1), 1, 1)*enc + \
                          alpha[:, 1].view(x_batch.size(0), enc.size(1), 1, 1)*enc[perm] + \
                          alpha[:, 2].view(x_batch.size(0), enc.size(1), 1, 1)*enc[perm2]
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
