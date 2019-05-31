import argparse
import torch
import sys
import glob
import os
import numpy as np
from torch import nn
from torch.utils.data import (DataLoader,
                              Subset)
from iterators.datasets import (CelebADataset,
                                CelebAHqDataset,
                                ZapposDataset,
                                ZapposPairDataset,
                                FashionGenDataset,
                                MnistDatasetOneHot,
                                MnistDataset012,
                                KMnistDatasetOneHot,
                                SvhnDatasetOneHot)
from torchvision import datasets
from torchvision.transforms import transforms
from PIL import Image
from torchvision.utils import save_image
from swapgan import SwapGAN
#from swapgan_pair import PairSwapGAN
#from amr_classifier import ClassifierOnly
#from acai import ACAI
from acai_f import ACAIF
#from acai_softmax import ACAI_Softmax
from threegan import ThreeGAN
#from pixel_arae import PixelARAE
from functools import partial
from importlib import import_module
from tools import (generate_tsne,
                   compute_inception,
                   compute_fid,
                   train_logreg,
                   save_embedding,
                   save_class_embedding,
                   save_interp,
                   save_interp_supervised,
                   save_frames_continuous,
                   save_consistency_plot,
                   line2dict,
                   count_params)

# Paired SwapGAN requires specialised dataset classes
# to work.
PAIR_SWAPGAN_SUPPORTED_DATASETS = ['zappos_hq']

def get_model(ngf, norm_layer, cpt=None, use_cuda=True):
    from architectures.shared import networks
    # TODO: we need to unify this with the
    # __main__ part below.
    if norm_layer == 'batch':
        norm_layer = nn.BatchNorm2d
    else:
        norm_layer = partial(nn.InstanceNorm2d, affine=True)
    # Load the models.
    gen = networks.ResnetEncoderDecoder(input_nc=3,
                                        output_nc=3,
                                        ngf=ngf,
                                        n_blocks=4,
                                        n_downsampling=3,
                                        norm_layer=norm_layer)
    if use_cuda:
        gen = gen.cuda()
    if cpt is not None:
        print("Loading checkpoint for encoder: %s" % cpt)
        dd = torch.load(cpt)
        gen.load_state_dict(dd['g'])
    gen.eval()
    return gen

"""
from classifier import Classifier
from architectures.classifier import ResNetCelebA

def get_classifier(mode='fid'):
    if mode not in ['fid', 'incep']:
        raise Exception("Mode must be either fid or incep")
    # HACK
    class ClassifierForFID(nn.Module):
        def __init__(self, base):
            super(ClassifierForFID, self).__init__()
            self.base = base
        def forward(self, x):
            return self.base(x), None
    cls = Classifier(ResNetCelebA())
    cls.load("results_cls/test/25.pkl")
    if mode == 'fid':
        base = cls.classifier.base
        base.eval()
        cls = ClassifierForFID(base)
    else:
        cls = cls.classifier
        cls.eval()
    return cls
"""

if __name__ == '__main__':

    def parse_args():
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--name', type=str, default="my_experiment")
        parser.add_argument('--model', type=str, default='swapgan',
                            choices=['swapgan', 'acai', 'acaif',
                                     'acai_softmax',
                                     'threegan',
                                     'pixel_arae',
                                     'classifier'])
        parser.add_argument('--dataset', type=str, default='celeba',
                            choices=['celeba',
                                     'celeba_hq',
                                     'zappos',
                                     'zappos_hq',
                                     'flowers',
                                     'mnist',
                                     'mnist012_small',
                                     'mnist_small',
                                     'kmnist',
                                     'svhn',
                                     'fashiongen'],
                            help="""
                            celeba = CelebA (64px) (set env var DATASET_CELEBA)
                            celeba_hq = CelebA-HQ (128px) (set env DATASET_CELEBAHQ)
                            zappos = Zappos shoe dataset (64px) (set env var DATASET_ZAPPOS)
                            zappos_hq = Zappos shoe dataset (128px) (set env var DATASET_ZAPPOSHQ)
                            flowers = TODO (64px)
                            mnist = MNIST (32px)
                            mnist012_small = MNIST for only digits 0,1,2 (16px)
                            mnist_small = MNIST (16px)
                            kmnist = KNIST (32px)
                            svhn = SVHN (32px)
                            fashiongen = TODO
                            """)
        parser.add_argument('--subset_train', type=int, default=None,
                            help="""If set, artficially decrease the size of the training
                            data. Use this to easily perform data ablation experiments.""")
        parser.add_argument('--data_dir', type=str,
                            default=None,
                            help="** NO LONGER USED **")
        parser.add_argument('--attr_file', type=str,
                            default="attr_cfgs/all.txt")
        parser.add_argument('--arch', type=str,
                            default='architectures/arch_celeba.py')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--epochs', type=int, default=200)
        parser.add_argument('--n_channels', type=int, default=3,
                            help="""Number of input channels in image. This is
                            "passed to the `get_network` function defined by
                            "`--arch`.""")
        parser.add_argument('--ngf', type=int, default=64,
                            help="""# channel multiplier for the autoencoder. This
                            "is passed to the `get_network` function defined by
                            `--arch`.""")
        parser.add_argument('--ndf', type=int, default=64,
                            help="""# channel multiplier for the discriminator. This
                            is passed to the `get_network` function defined by
                            `--arch`.""")
        parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
        parser.add_argument('--lamb', type=float, default=1.0,
                            help="""Weight for reconstruction loss between x
                            "and its reconstruction.""")
        parser.add_argument('--beta', type=float, default=1.0,
                            help="""Weight for consistency loss between enc(x) and
                            enc(dec(enc(x)))""")
        parser.add_argument('--cls', type=float, default=0.,
                            help="""If > 0, then the supervised mixing losses
                            will be enabled. This is also the weight of the loss
                            term which tries to fool the auxiliary classifier with
                            mixes generated by the class mixing function.""")
        # TENPORARY
        parser.add_argument('--sigma', type=float, default=0., help="dropout p")
        parser.add_argument('--eps', type=float, default=1e-8, help="For supervised formulation")
        # ---------
        parser.add_argument('--mixer', type=str, choices=['mixup', 'mixup2', 'fm', 'fm2'],
                            help="""The mixing function to use. Choices are 'mixup',
                            "'mixup2', and 'fm'. 'mixup' will sample an alpha ~ U(0,1)
                            "in the shape (bs,1,1,1), 'mixup2' will sample an alpha ~ U(0,1)
                            "in the shape (bs,f,1,1) and 'fm' will sample an m ~ Bern(p)
                            "(p ~ U(0,1)) in the shape (bs,f,1,1).""")
        parser.add_argument('--disable_g_recon', action='store_true')
        parser.add_argument('--disable_mix', action='store_true')
        parser.add_argument('--cls_loss', type=str, default='bce',
                            choices=['bce', 'mse'],
                            help="""The loss function to use for the auxiliary classifier
                            component (if cls > 0). The choices are 'bce' (binary 
                            cross-entropy) or 'mse' (mean-squared error).""")
        parser.add_argument('--gan_loss', type=str, default='bce',
                            choices=['bce', 'mse'],
                            help="""The loss function to use for the GAN component. The 
                            choices are 'bce' (binary cross-entropy) or 'mse' (mean-squared 
                            error).""")
        parser.add_argument('--beta1', type=float, default=0., help="beta1 term of ADAM")
        parser.add_argument('--beta2', type=float, default=0.999, help="beta2 term of ADAM")
        parser.add_argument('--weight_decay', type=float, default=0.0,
                            help="""L2 weight decay on params (note: applies to optimisers for both
                            the generator, discriminator, and classifier probe (if set)""")
        parser.add_argument('--update_g_every', type=int, default=5)
        parser.add_argument('--cls_probe', type=str, default=None,
                            help="""Architecture for classifier to branch off the bottleneck
                            (if you are performing downstream classification)""")
        parser.add_argument('--cls_probe_args', type=str, default=None, help="python dict")
        parser.add_argument('--cls_probe_weight_decay', type=float, default=0,
                            help="Weight decay term specifically for cls_probe optimiser")
        parser.add_argument('--cls_probe_backprop_grads', action='store_true',
                            help="""If set to true, the classifier probe can backprop gradients
                            back into the autoencoder""")
        parser.add_argument('--save_path', type=str, default=None)
        parser.add_argument('--val_batch_size', type=int, default=64)
        parser.add_argument('--save_every', type=int, default=5)
        parser.add_argument('--save_images_every', type=int, default=1)
        parser.add_argument('--resume', type=str, default='auto')
        parser.add_argument('--load_nonstrict', action='store_true')
        parser.add_argument('--no_verbose', action='store_true')
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--mode', type=str,
                            choices=['train',
                                     'interp_train',
                                     'interp_valid',
                                     'interp_pixel_train',
                                     'interp_pixel_valid',
                                     'interp_sup_train',
                                     'interp_sup_valid',
                                     'consistency',
                                     'save_frames',
                                     'tsne',
                                     'fid_train',
                                     'fid_valid',
                                     'incep_train',
                                     'incep_valid',
                                     'embedding',
                                     'logreg',
                                     'class_embeddings',
                                     'dump_g',
                                     'pdb'],
                            default='train',
                            help="""
                            train = training
                            interp_train = generate interpolations on samples
                              from the training set.
                            interp_valid = generate interpolations on samples
                              from the validation set.
                            interp_pixel_train = generate pixel space interps
                              from the training set.
                            interp_pixel_valid = generate pixel space interps
                              from the valid set.
                            interp_sup_train = generate interpolations on samples
                              using the class mixer function, on the training set.
                            interp_sup_valid = generate interpolations on samples
                              using the class mixer function, on the valid set.
                            """)
        parser.add_argument('--mode_override', type=str, default=None)
        #parser.add_argument('--cpu', action='store_true')
        args = parser.parse_args()
        return args

    args = parse_args()
    print(args)

    if args.mode == 'train':
        torch.manual_seed(args.seed)
    #else:
    #    torch.manual_seed(0)

    use_cuda = True if torch.cuda.is_available() else False

    if args.dataset == 'celeba':
        # TODO: clean this up a bit
        with open(args.attr_file) as f:
            attrs = f.readline().rstrip().split()
        print("attrs = ")
        for s in attrs:
            print(" %s" % s)
        train_transforms = [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        ids = "celeba_frontal.txt"
        ds_train = CelebADataset(root=os.environ['DATASET_CELEBA'],
                                 ids=ids,
                                 attrs=attrs,
                                 transforms_=train_transforms,
                                 missing_ind=False,
                                 mode='train')
        ds_valid = CelebADataset(root=os.environ['DATASET_CELEBA'],
                                 ids=ids,
                                 attrs=attrs,
                                 transforms_=train_transforms,
                                 missing_ind=False,
                                 mode='valid')
        n_classes = len(attrs)
    elif args.dataset == 'celeba_hq':
        train_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        ds_train = CelebAHqDataset(root=os.environ['DATASET_CELEBAHQ'],
                                   ids="iterators/celebahq_frontal.txt",
                                   transforms_=train_transforms,
                                   mode='train')
        ds_valid = CelebAHqDataset(root=os.environ['DATASET_CELEBAHQ'],
                                   ids="iterators/celebahq_frontal.txt",
                                   transforms_=train_transforms,
                                   mode='valid')
        n_classes = 0
    elif args.dataset == 'flowers':
        train_transforms = [
            transforms.Resize(80),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        ds_train = OxfordFlowers102Dataset(root=os.environ['DATASET_FLOWERS'],
                                           transforms_=train_transforms,
                                           mode='train')
        ds_valid = OxfordFlowers102Dataset(root=os.environ['DATASET_FLOWERS'],
                                           transforms_=train_transforms,
                                           mode='valid')
        n_classes = 102
    elif args.dataset in ['zappos', 'zappos_hq']:
        sz = 64 if args.dataset == 'zappos' else 128
        train_transforms = [
            transforms.Resize(sz),
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        if args.model == 'swapgan_pair':
            zappos_dir = os.environ['DATASET_ZAPPOS']
            ds_train = ZapposPairDataset(root=zappos_dir,
                                         pairs="iterators/zappo_pairs_iou0.7.txt",
                                         transforms_=train_transforms,
                                         mode='train')
            ds_valid = ZapposPairDataset(root=zappos_dir,
                                         pairs="iterators/zappo_pairs_iou0.7.txt",
                                         transforms_=train_transforms,
                                         mode='valid')
        else:
            ds_train = ZapposDataset(root=zappos_dir,
                                     ids="iterators/zappo_bg_stats.csv",
                                     transforms_=train_transforms,
                                     mode='train')
            ds_valid = ZapposDataset(root=zappos_dir,
                                     ids="iterators/zappo_bg_stats.csv",
                                     transforms_=train_transforms,
                                     mode='valid')
        n_classes = 0
    elif args.dataset in ['mnist', 'mnist012_small', 'mnist_small']:
        if args.dataset == 'mnist':
            mnist_class = MnistDatasetOneHot
            img_sz = 32
        elif args.dataset == 'mnist012_small':
            mnist_class = MnistDataset012
            img_sz = 16
        elif args.dataset == 'mnist_small':
            mnist_class = MnistDatasetOneHot
            img_sz = 16
        ds_train = mnist_class('iterators', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(img_sz),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,),
                                                        (0.5,))
                               ]))
        ds_valid = mnist_class('iterators', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(img_sz),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,),
                                                        (0.5,))
                               ]))
        n_classes = 10
    elif args.dataset == 'kmnist':
        ds_train = KMnistDatasetOneHot('iterators', train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(32),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,),
                                                                (0.5,))
                                       ]))
        ds_valid = KMnistDatasetOneHot('iterators', train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.Resize(32),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,),
                                                                (0.5,))
                                       ]))
        n_classes = 10
    elif args.dataset == 'svhn':
        ds_train = SvhnDatasetOneHot('iterators', split='train', download=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(32),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,),
                                                              (0.5,))
                                     ]))
        ds_valid = SvhnDatasetOneHot('iterators', split='test', download=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(32),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,),
                                                              (0.5,))
                                     ]))
        n_classes = 10
    elif args.dataset == 'fashiongen':
        train_transforms = [
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        ds_train = FashionGenDataset(root=os.environ['DATASET_FASHIONGEN'],
                                     transforms_=train_transforms,
                                     mode='train')
        ds_valid = FashionGenDataset(root=os.environ['DATASET_FASHIONGEN'],
                                     transforms_=train_transforms,
                                     mode='valid')
        n_classes = 0

    if args.subset_train is not None:
        # The subset is randomly sampled from the
        # training data, and changes depending on
        # the seed.
        indices = np.arange(0, args.subset_train)
        rs = np.random.RandomState(args.seed)
        rs.shuffle(indices)
        indices = indices[0:args.subset_train]
        ds_train = Subset(ds_train, indices=indices)

    if args.mode == 'train':
        bs = args.batch_size
    else:
        bs = args.val_batch_size
    loader_train = DataLoader(ds_train,
                              batch_size=bs,
                              shuffle=True,
                              num_workers=args.num_workers)
    loader_valid = DataLoader(ds_valid,
                              batch_size=bs,
                              shuffle=True,
                              num_workers=args.num_workers)

    if args.save_path is None:
        args.save_path = os.environ['RESULTS_DIR']
    if args.seed == 0:
        save_path = args.save_path
    else:
        save_path = "%s/s%i" % (args.save_path, args.seed)
    
    mod = import_module(args.arch.replace("/", ".").\
                        replace(".py", ""))
    dd = mod.get_network(n_channels=args.n_channels,
                         ngf=args.ngf,
                         ndf=args.ndf,
                         n_classes=0 if args.cls == 0 else n_classes)
    gen = dd['gen']
    disc_x = dd['disc_x']
    class_mixer = dd['class_mixer']
    print("Generator:")
    print(gen)
    print("# params: %i" % count_params(gen))
    print("Discriminator:")
    print(disc_x)
    print("# params: %i" % count_params(disc_x))
    if class_mixer is not None:
        print("Class mixer:")
        print(class_mixer)
        print("# params: %i" % count_params(class_mixer))

    def image_handler_default(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1:
            if kwargs['epoch'] % args.save_images_every == 0:
                mode = kwargs['mode']
                epoch = kwargs['epoch']
                recon = outputs['recon']*0.5 + 0.5
                inputs = outputs['input']*0.5 + 0.5
                inputs_permed = inputs[outputs['perm']]
                recon_permed = recon[outputs['perm']]
                mix = outputs['mix']*0.5 + 0.5
                imgs = torch.cat((inputs, recon, inputs_permed, recon_permed, mix))
                save_image( imgs,
                            nrow=inputs.size(0),
                            filename="%s/%s/%i_%s.png" % (save_path, args.name, epoch, mode))

    def image_handler_swapgan(losses, batch, outputs, kwargs):
        if kwargs['iter'] == 1:
            if kwargs['epoch'] % args.save_images_every == 0:
                mode = kwargs['mode']
                epoch = kwargs['epoch']
                input1 = outputs['input1']*0.5 + 0.5
                recon1 = outputs['recon1']*0.5 + 0.5
                input2 = outputs['input2']*0.5 + 0.5
                recon2 = outputs['recon2']*0.5 + 0.5
                mix = outputs['mix']*0.5 + 0.5
                imgs = torch.cat((input1, recon1, input2, recon2, mix))
                save_image( imgs,
                            nrow=input1.size(0),
                            filename="%s/%s/%i_%s.png" % (save_path, args.name, epoch, mode))

    def image_handler_blank(losses, batch, outputs, kwargs):
        return
            

    if args.model == 'swapgan':
        gan_class = SwapGAN
        image_handler = image_handler_default
    elif args.model == 'swapgan_pair':
        """
        if args.dataset not in PAIR_SWAPGAN_SUPPORTED_DATASETS:
            raise Exception("PairSwapGAN currently only works with these datasets: %s" % \
                            PAIR_SWAPGAN_SUPPORTED_DATASETS)
        gan_class = PairSwapGAN
        image_handler = image_handler_swapgan
        """
        raise NotImplementedError()
    elif args.model == 'acai':
        gan_class = ACAI
        image_handler = image_handler_default
    elif args.model == 'acaif':
        gan_class = ACAIF
        image_handler = image_handler_default
    elif args.model == 'acai_softmax':
        gan_class = ACAI_Softmax
        image_handler = image_handler_default
    elif args.model == 'threegan':
        gan_class = ThreeGAN
        image_handler = image_handler_default
    elif args.model == 'pixel_arae':
        gan_class = PixelARAE
        image_handler = image_handler_default
    elif args.model == 'classifier':
        gan_class = ClassifierOnly
        image_handler = image_handler_blank

    cls_enc = None
    if args.cls_probe is not None:
        mod_cls = import_module(args.cls_probe.replace("/", ".").\
                            replace(".py", ""))
        cls_enc = mod_cls.get_network(**eval(args.cls_probe_args))
        print("Classifier probe:")
        print(cls_enc)
        print("# params: %i" % count_params(cls_enc))

    gan = gan_class(
        generator=gen,
        disc_x=disc_x,
        class_mixer=class_mixer,
        lamb=args.lamb,
        beta=args.beta,
        cls=args.cls,
        mixer=args.mixer,
        sigma=args.sigma,
        cls_loss=args.cls_loss,
        gan_loss=args.gan_loss,
        disable_mix=args.disable_mix,
        disable_g_recon=args.disable_g_recon,
        cls_enc=cls_enc,
        opt_args={'lr': args.lr,
                  'betas': (args.beta1, args.beta2),
                  'weight_decay': args.weight_decay},
        update_g_every=args.update_g_every,
        cls_enc_weight_decay=args.cls_probe_weight_decay,
        cls_enc_backprop_grads=args.cls_probe_backprop_grads,
        handlers=[image_handler]
    )
    gan.eps = args.eps
    if args.load_nonstrict:
        gan.load_strict = False

    latest_model = None
    if args.resume is not None:
        if args.resume == 'auto':
            # autoresume
            model_dir = "%s/%s" % (save_path, args.name)
            # List all the pkl files.
            files = glob.glob("%s/*.pkl" % model_dir)
            # Make them absolute paths.
            files = [os.path.abspath(key) for key in files]
            if len(files) > 0:
                # Get creation time and use that.
                latest_model = max(files, key=os.path.getctime)
                print("Auto-resume mode found latest model: %s" %
                      latest_model)
                gan.load(latest_model)
        else:
            gan.load(args.resume)

    if latest_model is not None:
        cpt_name = os.path.basename(latest_model)
    else:
        cpt_name = os.path.basename(args.resume)

    expt_dir = "%s/%s" % (save_path, args.name)
    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
        
    from subprocess import check_output
    args_file = "%s/args.txt" % expt_dir
    f_mode = 'w' if not os.path.exists(args_file) else 'a'
    with open(args_file, f_mode) as f_args:
        f_args.write("argparse: %s\n" % str(args))
        if latest_model is not None:
            f_args.write("latest_model = %s" % latest_model)
        git_branch = check_output("git rev-parse --symbolic-full-name --abbrev-ref HEAD", shell=True)
        git_branch = git_branch.decode('utf-8').rstrip()
        f_args.write("git_branch: %s\n" % git_branch)

 
    if args.mode == 'train':

        gan.train(itr_train=loader_train,
                  itr_valid=loader_valid,
                  epochs=args.epochs,
                  model_dir=expt_dir,
                  result_dir=expt_dir,
                  save_every=args.save_every,
                  verbose=False if args.no_verbose else True)

    elif 'consistency' in args.mode:

        batch = iter(loader_train).next()
        if len(batch) == 3:
            x_batch, y_batch = batch[0:2], batch[2::]
        else:
            x_batch, y_batch = batch[0], batch[1]
        if use_cuda:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

        out_dir = "%s/%s/%s/%s/model-%s/" % (save_path, args.name, args.mode, args.mixer, cpt_name)
        save_consistency_plot(gan, x_batch, out_dir)

    elif 'interp' in args.mode:

        torch.manual_seed(0)

        if args.mode in ['interp_train', 'interp_sup_train', 'interp_pixel_train']:
            batch = iter(loader_train).next()
        elif args.mode in ['interp_valid', 'interp_sup_valid', 'interp_pixel_valid']:
            batch = iter(loader_valid).next()
        if len(batch) == 3:
            x_batch, y_batch = batch[0:2], batch[2::]
        else:
            x_batch, y_batch = batch[0], batch[1]
        if use_cuda:
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()

        kwargs = {'num_interps': 10, 'padding': 2, 'show_real':False,
                  'enumerate_all':True}
        if args.mode_override is not None:
            override_kwargs = line2dict(args.mode_override)
            for key in override_kwargs:
                kwargs[key] = override_kwargs[key]
        print("kwargs:", kwargs)
        
        if '_sup_' in args.mode:
            out_dir = "%s/%s/%s/%s/model-%s/" % (save_path, args.name, args.mode, args.mixer, cpt_name)
            save_interp_supervised(gan,
                                   x_batch,
                                   y_batch,
                                   out_dir,
                                   num=kwargs['num_interps'],
                                   padding=kwargs['padding'],
                                   enumerate_all=kwargs['enumerate_all'])
        elif '_pixel_' in args.mode:
            out_dir = "%s/%s/%s/model-%s/" % (save_path, args.name, args.mode, cpt_name)
            save_interp(gan,
                        x_batch,
                        out_dir,
                        mix_input=True,
                        num=kwargs['num_interps'],
                        padding=kwargs['padding'],
                        show_real=kwargs['show_real'])
        else:
            out_dir = "%s/%s/%s/%s/model-%s/" % (save_path, args.name, args.mode, args.mixer, cpt_name)
            save_interp(gan,
                        x_batch,
                        out_dir,
                        num=kwargs['num_interps'],
                        padding=kwargs['padding'],
                        show_real=kwargs['show_real'])

        print("Saved interpolations to: %s" % out_dir)

    elif args.mode == 'save_frames':

        torch.manual_seed(0)

        # TODO: only does training iterator

        kwargs = {'num_interps': 10,
                  'framerate': 30,
                  'resize_to': -1,
                  'crf': 23}
        if args.mode_override is not None:
            override_kwargs = line2dict(args.mode_override)
            for key in override_kwargs:
                kwargs[key] = override_kwargs[key]
        print("kwargs:", kwargs)
        
        x_batch, _ = iter(loader_train).next()
        if use_cuda:
            x_batch = x_batch.cuda()
        out_dir = "%s/%s/%s/%s/model-%s/" % (save_path, args.name, args.mode, args.mixer, cpt_name)
        save_frames_continuous(gan,
                               x_batch,
                               out_dir,
                               num_interps=kwargs['num_interps'],
                               framerate=kwargs['framerate'],
                               resize_to=kwargs['resize_to'],
                               crf=kwargs['crf'])

    elif args.mode == 'tsne':

        kwargs = {'use_labels': 1, 'n_cores': 4, 'n_repeats': 5}
        if args.mode_override is not None:
            override_kwargs = line2dict(args.mode_override)
            for key in override_kwargs:
                kwargs[key] = override_kwargs[key]
        print("kwargs:", kwargs)
        
        embed_dst = "%s/%s/tsne/model-%s/" % \
                    (save_path, args.name, cpt_name)
        generate_tsne(loader_train,
                      gan,
                      save_path=embed_dst,
                      use_labels=kwargs['use_labels'])

    elif args.mode == 'embedding':

        embed_train = "%s/%s/embedding/model-%s/embedding.npz" % \
                    (save_path, args.name, cpt_name)
        save_embedding(loader_train,
                       gan,
                       save_file=embed_train)
        embed_valid = "%s/%s/embedding/model-%s/embeddings_valid.npz" % \
                      (save_path, args.name, cpt_name)
        save_embedding(loader_valid,
                       gan,
                       save_file=embed_valid)
        
    elif args.mode == 'logreg':

        kwargs = {'max_iters': 10000}
        if args.mode_override is not None:
            override_kwargs = line2dict(args.mode_override)
            for key in override_kwargs:
                kwargs[key] = override_kwargs[key]
        logreg_dst = "%s/%s/logreg/model-%s/" % \
                     (save_path, args.name, cpt_name)
        train_logreg(loader_train,
                     gan,
                     save_path=logreg_dst,
                     max_iters=kwargs['max_iters'])
        
    elif args.mode in ['fid_train', 'fid_valid']:

        if '_train' in args.mode:
            loader = loader_train
        else:
            loader = loader_valid

        cls = get_classifier('fid')

        compute_fid(loader=loader,
                    gan=gan,
                    cls=cls,
                    save_path="%s/%s/fid/%s/model-%s/" % (save_path, args.name, args.mixer, cpt_name))

    elif args.mode in ['incep_train', 'incep_valid']:

        if '_train' in args.mode:
            loader = loader_train
        else:
            loader = loader_valid

        cls = get_classifier('incep')

        compute_inception(loader=loader,
                          gan=gan,
                          cls=cls,
                          save_path="%s/%s/incep/%s/model-%s/" % (save_path, args.name, args.mixer, cpt_name),
                          batch_size=args.val_batch_size,
                          n_classes=n_classes)

    elif args.mode == 'class_embeddings':

        save_path = "%s/%s/class_embeddings/%s/model-%s/" % (save_path, args.name, args.mixer, cpt_name)
        for _, y_batch in loader_train:
            break
        n_classes = y_batch.size(1)
        save_class_embedding(gan, n_classes, save_path)

    elif args.mode == 'dump_g':
        save_path = "tmp/%s/dump_g" % args.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("Dumping G weights to %s..." % save_path)
        torch.save(gan.generator.state_dict(), "%s/%s" % (save_path, cpt_name))

        
    elif args.mode == 'pdb':
        import pdb
        pdb.set_trace()
