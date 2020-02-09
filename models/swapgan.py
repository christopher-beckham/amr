import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from itertools import chain
import numpy as np
from collections import OrderedDict
from .base import Base

class SwapGAN(Base):
    def __init__(self,
                 generator,
                 disc_x,
                 class_mixer=None,
                 lamb=1.0,
                 beta=1.0,
                 cls=1.0,
                 sigma=None,
                 dropout=None,
                 disable_g_recon=False,
                 disable_mix=False,
                 recon_loss='l1',
                 cls_loss='bce',
                 gan_loss='bce',
                 mixer='mixup',
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 update_g_every=1,
                 cls_enc=None,
                 cls_enc_weight_decay=0.,
                 cls_enc_backprop_grads=False,
                 cls_enc_weight=1.,
                 handlers=[]):
        """The SwapGAN class is the base class for any classes which
        utilise mixup. This class is 'abstract' and must have certain
        methods implemented (for instance, see `TwoGAN` for the
        implementation of this in the k=2 mixing case).

        Args:
            generator: the autoencoder network, a `torch.nn` Module. This
              network should have an `encode` and a `decode` method.
            disc_x: the discriminator, a `torch.nn` Module.
            class_mixer: class mixer network (if doing supervised mixing)
            lamb: the weight of the reconstruction loss
            beta: the weight of the consistency loss (generally not used)
            cls: if `cls` > 0, then the supervised mixing losses will be
              enabled. This is also the weight of the loss term that tries to
              fool the auxiliary classifier component of the discriminator.
            disable_g_recon: if `True`, disable the loss which tries to
              fool the discriminator with reconstructions. (This loss probably
              isn't necessary, but all experiments in the paper have this false
              by default.)
            disable_mix: if `True`, all mixing losses are disabled, meaning the
              GAN losses only try to make the reconstructions look realistic
              (this makes the class ignore the `mixup` argument)
            recon_loss: the loss function to use for the reconstruction loss.
              The choices are 'bce' (binary cross-entropy) or 'mse'
              (mean-squared error).
            cls_loss: the loss function to use for the auxiliary classifier
              component (if cls > 0). The choices are 'bce' (binary cross-entropy)
              or 'mse' (mean-squared error).
            gan_loss: the loss function to use for the GAN component. The choices
              are 'bce' (binary cross-entropy) or 'mse' (mean-squared error).
            mixup: the mixing function to use.
            opt: optimiser from `torch.optim` class
            opt_args: a dictionary of kwargs to pass to the optimiser `opt`.
            update_g_every: update G how many every iterations? (This term balances
              the number of updates in D versus G.)
            cls_enc: the classifier probe, a `torch.nn` Module.
            cls_enc_weight_decay: weight decay for classifier probe.
            cls_enc_backprop_grads: if set to `True`, the classifier becomes part of
              the autoencoder. This means that it will contribute gradients back into
              the encoder part. (This is not used in the paper -- the classifier probe
              stays separate.)
            cls_enc_weight: only use if `cls_enc_backprop_grads` is `True`.
            handlers: a list of handlers
        """

        use_cuda = True if torch.cuda.is_available() else False
        mix_types = ['mixup', 'mixup2', 'fm', 'fm2']
        if mixer not in mix_types:
            raise Exception("mixer must be either: " % (",".join(mix_types)))
        if cls > 0. and class_mixer is None:
            raise Exception("If cls > 0 then a class mixer must be specified")
        self.generator = generator
        self.disc_x = disc_x
        self.class_mixer = class_mixer
        self.lamb = lamb
        self.beta = beta
        self.cls = cls
        self.disable_g_recon = disable_g_recon
        self.disable_mix = disable_mix
        self.mixer = mixer
        self.cls_enc = cls_enc

        self.optim = {}
        # If we're not backpropping grads from the classifier back into
        # the autoencoder, then we need to set up a separate optimiser
        # just for it. We do that here.
        if self.cls_enc is not None and cls_enc_backprop_grads is False:
            cls_opt_args = dict(opt_args) # copy
            if cls_enc_weight_decay != 0:
                cls_opt_args['weight_decay'] = cls_enc_weight_decay
            self.optim['cls_enc'] = opt(filter(lambda p: p.requires_grad,
                                               self.cls_enc.parameters()), **cls_opt_args)
        self.cls_enc_weight = cls_enc_weight

        g_params = self.generator.parameters()
        if self.class_mixer is not None:
            g_params = chain(g_params, self.class_mixer.parameters())
        if cls_enc_backprop_grads:
            g_params = chain(g_params, self.cls_enc.parameters())

        optim_g = opt(filter(lambda p: p.requires_grad, g_params), **opt_args)
        optim_disc_x = opt(filter(lambda p: p.requires_grad,
                                  self.disc_x.parameters()), **opt_args)

        self.optim['g'] = optim_g
        self.optim['disc_x'] = optim_disc_x

        self.update_g_every = update_g_every
        self.schedulers = []
        #if scheduler_fn is not None:
        #    for key in self.optim:
        #        self.scheduler[key] = scheduler_fn(
        #            self.optim[key], **scheduler_args)
        self.handlers = handlers
        self.use_cuda = use_cuda
        ##################
        # Loss functions #
        ##################
        if recon_loss == 'l1':
            self.recon_loss = lambda x,y: torch.mean(torch.abs(x-y))
        elif recon_loss == 'l2':
            self.recon_loss = lambda x,y: torch.mean((x-y)**2)
        elif recon_loss == 'bce':
            self.recon_loss = lambda x,y: nn.BCELoss()( (x*0.5 + 0.5), (y*0.5 + 0.5))
        elif recon_loss == 'bce_sum':
            self.recon_loss = lambda x,y: F.binary_cross_entropy(x*0.5 + 0.5, y*0.5 + 0.5, size_average=False).\
                div(x.size(0))
        else:
            raise Exception("recon_loss must be either l1 or bce!")
        if gan_loss == 'bce':
            self.gan_loss_fn = self.bce
        elif gan_loss == 'mse':
            self.gan_loss_fn = self.mse
        else:
            raise Exception("Only bce or mse is currently supported for gan_loss")
        if cls_loss == 'bce':
            self.cls_loss_fn = self.bce
        elif gan_loss == 'mse':
            self.cls_loss_fn = self.mse
        else:
            raise Exception("Only bce or mse is currently supported for cls_loss")
        ########
        # cuda #
        ########
        if self.use_cuda:
            self.generator.cuda()
            self.disc_x.cuda()
            if self.class_mixer is not None:
                self.class_mixer.cuda()
            if self.cls_enc is not None:
                self.cls_enc.cuda()
        self.last_epoch = 0
        self.load_strict = True

    def _train(self):
        self.generator.train()
        self.disc_x.train()
        if self.class_mixer is not None:
            self.class_mixer.train()
        if self.cls_enc is not None:
            self.cls_enc.train()

    def _eval(self):
        self.generator.eval()
        self.disc_x.eval()
        if self.class_mixer is not None:
            self.class_mixer.eval()
        if self.cls_enc is not None:
            self.cls_enc.train()

    def bce(self, prediction, target):
        if not hasattr(target, '__len__'):
            target = torch.ones_like(prediction)*target
            if prediction.is_cuda:
                target = target.cuda()
        loss = torch.nn.BCELoss()
        if prediction.is_cuda:
            loss = loss.cuda()
        target = target.view(-1, 1)
        return loss(prediction, target)

    def mse(self, prediction, target):
        return torch.mean((prediction-target)**2)

    def reconstruct(self, x_batch):
        """Get reconstruction.

        :param x_batch: 
        :returns: 
        :rtype: 

        """
        self._eval()
        if self.use_cuda:
            x_batch = x_batch.cuda()
        with torch.no_grad():
            enc = self.generator.encode(x_batch)
            dec = self.generator.decode(enc)
            return dec

    def sampler(self, bs, f, is_2d, **kwargs):
        raise NotImplementedError("This method must be subclassed!")

    def sample(self, x_batch):
        raise NotImplementedError("This method must be subclassed!")

    def mix(self, enc):
        raise NotImplementedError("This method must be subclassed!")

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
        recon_loss = self.recon_loss(dec_enc, x_batch)
        disc_g_recon_loss = self.gan_loss_fn(self.disc_x(dec_enc)[0], 1)
        enc_mix, perm = self.mix(enc)

        dec_enc_mix = self.generator.decode(enc_mix)
        disc_g_mix_loss = self.gan_loss_fn(self.disc_x(dec_enc_mix)[0], 1)
        if self.beta > 0:
            consist_loss = self.recon_loss(self.generator.encode(dec_enc_mix),
                                           enc_mix)
        else:
            consist_loss = torch.FloatTensor([0.])
            if self.use_cuda:
                consist_loss = consist_loss.cuda()

        gen_loss = self.lamb*recon_loss
        if self.disable_g_recon is False:
            gen_loss = gen_loss + disc_g_recon_loss
        if self.disable_mix is False:
            gen_loss = gen_loss + disc_g_mix_loss + self.beta*consist_loss

        # If the classifier branch does not have its own optimiser,
        # that means its updates become part of G's updates.
        if self.cls_enc is not None and 'cls_enc' not in self.optim:
            cls_enc_loss, cls_enc_acc = \
                self._classifier_on_instance(enc, y_batch)
            gen_loss = gen_loss + self.cls_enc_weight*cls_enc_loss

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
        disc_loss = d_x
        disc_loss.backward()
        self.optim['disc_x'].step()

        # If 'cls_enc' is in optim class, it means it must be optimised
        # separately from everything else.
        if self.cls_enc is not None:
            if 'cls_enc' in self.optim:
                self.optim['cls_enc'].zero_grad()
                cls_enc_loss, cls_enc_acc = self._classifier_on_instance(enc.detach(),
                                                                         y_batch)
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

    def _classifier_on_instance(self,
                                enc,
                                y_batch):

        if hasattr(self.cls_enc, 'legacy'):
            enc_flat = enc.view(-1, self.cls_enc.n_in)
        else:
            enc_flat = enc
        cls_enc_out = self.cls_enc(enc_flat)
        cls_enc_preds_log = torch.log_softmax(cls_enc_out, dim=1)
        cls_enc_loss = nn.NLLLoss()(cls_enc_preds_log,
                                    y_batch.argmax(dim=1).long())
        with torch.no_grad():
            cls_enc_preds = torch.softmax(cls_enc_out, dim=1)
            cls_enc_acc = (cls_enc_preds.argmax(dim=1) == y_batch.argmax(dim=1).long()).float().mean()

        return cls_enc_loss, cls_enc_acc

    def eval_on_instance(self,
                         x_batch,
                         y_batch,
                         **kwargs):
        self._eval()
        with torch.no_grad():
            enc = self.generator.encode(x_batch)
            dec_enc = self.generator.decode(enc)
            enc_mix, perm = self.mix(enc)
            dec_enc_mix = self.generator.decode(enc_mix)

            losses = {}

            if self.cls_enc is not None:
                cls_enc_loss, cls_enc_acc = self._classifier_on_instance(enc,
                                                                         y_batch)
                losses['cls_enc_acc'] = cls_enc_acc.item()

            outputs = {
                'recon': dec_enc,
                'mix': dec_enc_mix,
                'perm': perm,
                'input': x_batch,
            }

        return losses, outputs

    def prepare_batch(self, batch):
        if len(batch) != 2:
            raise Exception("Expected batch to only contain two elements: " +
                            "X_batch and y_batch")
        X_batch = batch[0].float()
        y_batch = batch[1].float() # assuming one-hot encoding
        if self.use_cuda:
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()
        return [X_batch, y_batch]

    def save(self, filename, epoch):
        dd = {}
        # Save the models.
        dd['g'] = self.generator.state_dict()
        dd['disc_x'] = self.disc_x.state_dict()
        if self.class_mixer is not None:
            dd['class_mixer'] = self.class_mixer.state_dict()
        if self.cls_enc is not None:
            dd['cls_enc'] = self.cls_enc.state_dict()
        # Save the models' optim state.
        for key in self.optim:
            dd['optim_%s' % key] = self.optim[key].state_dict()
        dd['epoch'] = epoch
        torch.save(dd, filename)

    def load(self, filename):
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        dd = torch.load(filename,
                        map_location=map_location)
        # Load the models.
        self.generator.load_state_dict(dd['g'], strict=self.load_strict)
        self.disc_x.load_state_dict(dd['disc_x'], strict=self.load_strict)
        if self.class_mixer is not None:
            self.class_mixer.load_state_dict(dd['class_mixer'],
                                             strict=self.load_strict)
        if self.cls_enc is not None:
            self.cls_enc.load_state_dict(dd['cls_enc'],
                                         strict=self.load_strict)
            # Load the last epoch for the LR scheduler
            #self.cls_sched.last_epoch = dd['epoch']
        # Load the models' optim state.
        for key in self.optim:
            self.optim[key].load_state_dict(dd['optim_%s' % key])
        self.last_epoch = dd['epoch']
