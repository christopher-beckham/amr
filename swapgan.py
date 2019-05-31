import torch
import numpy as np
from collections import OrderedDict
from torch import optim
from itertools import chain
from base import Base
from torch import nn

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
                 cls_loss='bce',
                 gan_loss='bce',
                 mixer='mixup',
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 update_g_every=1,
                 cls_enc=None,
                 cls_enc_weight_decay=0.,
                 cls_enc_backprop_grads=False,
                 handlers=[]):
        """
        :param generator: the autoencoding network. (Note that this network
          should have an `encode` and a `decode` method.)
        :param disc_x: the discriminator on x.
        :param class_mixer: class mixer network (you only need to pass
          something here if `cls` > 0).
        :param lamb: the weight of the reconstruction loss.
        :param beta: the weight of the consistency loss.
        :param cls: if `cls` > 0, then the supervised mixing losses will be
          enabled. This is also the weight of the loss term that tries to
          fool the auxiliary classifier component of the discriminator.
        :param sigma: TODO
        :param dropout: TODO
        :param disable_g_recon: if `True`, disable the loss which tries to
          fool the discriminator with reconstructions. Note that it should
          be ok to disable this loss, since the discriminator-fooling mixing
          loss naturally encompasses this loss too.
        :param disable_mix: if `True`, all mixing losses are disabled, leaving
          us with just an adversarial reconstruction autoencoder (ARAE).
        :param cls_loss: the loss function to use for the auxiliary classifier
          component (if cls > 0). The choices are 'bce' (binary cross-entropy)
          or 'mse' (mean-squared error). (TODO also add support for cat x-entropy)
        :param gan_loss: the loss function to use for the GAN component. The choices
          are 'bce' (binary cross-entropy) or 'mse' (mean-squared error).
        :param mixup: the mixing function to use. Choices are 'mixup', 'mixup2',
          and 'fm'. 'mixup' will sample an alpha ~ U(0,1) in the shape (bs,1,1,1),
          'mixup2' will sample an alpha ~ U(0,1) in the shape (bs,f,1,1) and 'fm'
          will sample an m ~ Bern(p) (p ~ U(0,1)) in the shape (bs,f,1,1).
        :param opt: optimiser from `torch.optim` class
        :param opt_args: a dictionary of kwargs to pass to the `opt` argument
        :param update_g_every: update G how many every iterations?
        :param cls_enc: TODO
        :returns: 
        :rtype: 
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
        self.eps = 1e-4
        self.disable_g_recon = disable_g_recon
        self.disable_mix = disable_mix
        self.mixer = mixer
        print("lamb = %f" % lamb)
        print("beta = %f" % beta)

        if self.class_mixer is not None:
            g_params = chain(self.generator.parameters(),
                             self.class_mixer.parameters())
        else:
            g_params = self.generator.parameters()
        optim_g = opt(filter(lambda p: p.requires_grad, g_params), **opt_args)
        optim_disc_x = opt(filter(lambda p: p.requires_grad,
                                  self.disc_x.parameters()), **opt_args)

        self.dropout = None
        if sigma is not None:
            self.dropout = nn.Dropout2d(sigma)

        self.optim = {
            'g': optim_g,
            'disc_x': optim_disc_x,
        }
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
        self.last_epoch = 0
        self.load_strict = True
        ###########################################
        # Classifier on the bottleneck (optional) #
        ###########################################
        self.cls_enc = None
        if cls_enc is not None:
            # If cls_enc_backprop_grads is enabled, then
            # we need the classifier optim class to update
            # the weights of both the autoencoder and the
            # classifier, so set that here.
            if cls_enc_backprop_grads:
                print("cls_enc_backprop_grads = True")
                cls_enc_params = chain(cls_enc.parameters(),
                                       self.generator.parameters())
            else:
                cls_enc_params = cls_enc.parameters()
            # If necessary, add classifier-specific weight
            # decay.
            cls_opt_args = dict(opt_args) # copy
            if cls_enc_weight_decay != 0:
                cls_opt_args['weight_decay'] = cls_enc_weight_decay
            self.optim['cls_enc'] = opt(filter(lambda p: p.requires_grad,
                                               cls_enc_params), **cls_opt_args)
            if self.use_cuda:
                cls_enc.cuda()
            self.cls_enc = cls_enc
        self.cls_enc_backprop_grads = cls_enc_backprop_grads

    def _get_stats(self, dict_, mode):
        stats = OrderedDict({})
        for key in dict_.keys():
            stats[key] = np.mean(dict_[key])
        return stats

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
        if self.dropout is not None:
            enc = self.dropout(enc)
        perm = torch.randperm(x_batch.size(0))
        dec_enc = self.generator.decode(enc)
        recon_loss = torch.mean(torch.abs(dec_enc-x_batch))
        disc_g_recon_loss = self.gan_loss_fn(self.disc_x(dec_enc)[0], 1)
        is_2d = True if len(enc.size()) == 2 else False
        alpha = self.sampler(x_batch.size(0), enc.size(1), is_2d)
        enc_mix = alpha*enc + (1.-alpha)*enc[perm]
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

        ## ----------------------------------
        ## Classifier (if supervised enabled)
        ## ----------------------------------
        if self.cls > 0.:
            alpha_sup = torch.bernoulli(torch.rand((y_batch.size(0), y_batch.size(1)))).float()
            if self.use_cuda:
                alpha_sup = alpha_sup.cuda()
            y_mix = alpha_sup*y_batch + (1.-alpha_sup)*y_batch[perm]
            enc_sup_mix, h_mask = self.class_mixer(enc, enc[perm], y_mix)
            dec_enc_sup_mix = self.generator.decode(enc_sup_mix)
            disc_g_mix_sup_out, disc_g_mix_sup_cls_out = self.disc_x(dec_enc_sup_mix)
            disc_g_mix_sup_cls_loss = self.cls_loss_fn(disc_g_mix_sup_cls_out, y_mix) # cls loss
            disc_g_mix_sup_loss = self.gan_loss_fn(disc_g_mix_sup_out, 1) # GAN loss
            consist_sup_loss = torch.mean(torch.abs(self.generator.encode(dec_enc_sup_mix)-enc_sup_mix))
            h_mask_prior = torch.mean((h_mask.sum(dim=1) - enc.size(1)/2.)**2)
            gen_loss = gen_loss + disc_g_mix_sup_loss + \
                       self.beta * consist_sup_loss + \
                       self.cls * disc_g_mix_sup_cls_loss + \
                       self.eps * h_mask_prior
            with torch.no_grad():
                h_mask_loss = torch.mean((h_mask.sum(dim=1).mean()))

        if (kwargs['iter']-1) % self.update_g_every == 0:
            gen_loss.backward(retain_graph=True if self.cls_enc_backprop_grads else False)
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
            if self.cls > 0.:
                # Do supervised mixes.
                d_out_sup_mix = self.gan_loss_fn(self.disc_x(dec_enc_sup_mix.detach())[0], 0)
                d_losses.append(d_out_sup_mix)

        d_x = sum(d_losses)
        disc_loss = d_x
        if self.cls > 0.:
            c_x_real = self.cls_loss_fn(cx_out, y_batch)
            disc_loss = disc_loss + c_x_real
        disc_loss.backward()
        self.optim['disc_x'].step()

        ## ----------------------------------------------
        ## Classifier on bottleneck (NOTE: for debugging)
        ## ----------------------------------------------
        if self.cls_enc is not None:
            if self.cls_enc_backprop_grads is False:
                enc = enc.detach()
            cls_enc_losses = self.train_classifier_on_instance(enc, y_batch)

        losses = {
            'gen_loss': gen_loss.item(),
            'disc_g_recon': disc_g_recon_loss.item(),
            'disc_g_mix': disc_g_mix_loss.item(),
            'recon': recon_loss.item(),
            'consist': consist_loss.item(),
            'd_x': d_x.item() / len(d_losses)
        }
        if self.cls > 0.:
            losses['mask_mean'] = h_mask_loss.item(),
            losses['cls_g_mix'] = disc_g_mix_sup_cls_loss.item()
            if self.cls > 0.:
                losses['consist_sup'] = consist_sup_loss.item()
            with torch.no_grad():
                z1_acc = ((cx_out > 0.5).float() == y_batch).float().mean()
                z1_base_acc = ((cx_out*0.) == y_batch).float().mean()
            losses['z1_acc'] = z1_acc.item()
            losses['z1_base_acc'] = z1_base_acc.item()
        if self.cls_enc is not None:
            losses['cls_enc_loss'] = cls_enc_losses['cls_enc_loss']
            losses['cls_enc_acc'] = cls_enc_losses['cls_enc_acc']
        outputs = {
            'recon': dec_enc,
            'mix': dec_enc_mix,
            'perm': perm,
            'input': x_batch,
        }
        return losses, outputs

    def train_classifier_on_instance(self,
                                     enc,
                                     y_batch):
        self.optim['cls_enc'].zero_grad()

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
        cls_enc_loss.backward()

        self.optim['cls_enc'].step()

        return {
            'cls_enc_loss': cls_enc_loss.item(),
            'cls_enc_acc': cls_enc_acc.item()
        }

    def eval_on_instance(self,
                         x_batch,
                         y_batch,
                         **kwargs):
        self._eval()
        with torch.no_grad():
            enc = self.generator.encode(x_batch)
            if self.dropout is not None:
                enc = self.dropout(enc)
            perm = torch.randperm(x_batch.size(0))
            dec_enc = self.generator.decode(enc)
            is_2d = True if len(enc.size()) == 2 else False
            alpha = self.sampler(x_batch.size(0), enc.size(1), is_2d)
            enc_mix = alpha*enc + (1.-alpha)*enc[perm]
            dec_enc_mix = self.generator.decode(enc_mix)

            losses = {}
            if self.cls > 0.:
                cx_out = self.disc_x(x_batch)[1]
                z1_acc = ((cx_out > 0.5).float() == y_batch).float().mean()
                z1_base_acc = ((cx_out*0.) == y_batch).float().mean()
                losses['z1_acc'] = z1_acc.item()
                losses['z1_base_acc'] = z1_base_acc.item()

            if self.cls_enc is not None:
                cls_enc_losses = self.eval_classifier_on_instance(enc,
                                                                  y_batch)
                losses['cls_enc_acc'] = cls_enc_losses['cls_enc_acc']

            outputs = {
                'recon': dec_enc,
                'mix': dec_enc_mix,
                'perm': perm,
                'input': x_batch,
            }

        return losses, outputs

    def eval_classifier_on_instance(self,
                                    enc,
                                    y_batch):
        if hasattr(self.cls_enc, 'legacy'):
            enc_flat = enc.view(-1, self.cls_enc.n_in)
        else:
            enc_flat = enc
        cls_enc_out = self.cls_enc(enc_flat)
        cls_enc_preds = torch.softmax(cls_enc_out, dim=1)
        cls_enc_acc = (cls_enc_preds.argmax(dim=1) == y_batch.argmax(dim=1).long()).float().mean()
        return {
            'cls_enc_acc': cls_enc_acc.item()
        }

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
