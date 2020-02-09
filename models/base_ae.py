import torch
import numpy as np
from collections import OrderedDict
from torch import optim
from itertools import chain
from .base import Base
from torch import nn

class BaseAE(Base):
    def __init__(self,
                 generator,
                 lamb=1.0,
                 beta=1.0,
                 recon_loss='l1',
                 gan_loss='bce',
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 handlers=[]):
        """
        """

        super(BaseAE, self).__init__()

        use_cuda = True if torch.cuda.is_available() else False
        self.generator = generator
        self.lamb = lamb
        self.beta = beta
        print("lamb = %f" % lamb)
        print("beta = %f" % beta)

        self.optim = {}
        optim_g = opt(filter(lambda p: p.requires_grad, self.generator.parameters()), **opt_args)
        self.optim['g'] = optim_g

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
        else:
            raise Exception("recon_loss must be either l1 or bce!")
        ########
        # cuda #
        ########
        if self.use_cuda:
            self.generator.cuda()
        self.last_epoch = 0
        self.load_strict = True

    def _get_stats(self, dict_, mode):
        stats = OrderedDict({})
        for key in dict_.keys():
            stats[key] = np.mean(dict_[key])
        return stats

    def _train(self):
        self.generator.train()

    def _eval(self):
        self.generator.eval()

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

    def sampler(self, bs, f, is_2d, p=None):
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

    def sample(self, x_batch):
        raise NotImplementedError("This method must be subclassed!")

    def mix(self, enc):
        raise NotImplementedError("This method must be subclassed!")

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
        for key in self.optim:
            self.optim[key].load_state_dict(dd['optim_%s' % key])
        self.last_epoch = dd['epoch']
