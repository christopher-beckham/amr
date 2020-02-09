import argparse
import torch
import sys
import glob
import os
import numpy as np
import pickle
import yaml
import random
import string

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
                                SvhnDatasetOneHot,
                                CifarDatasetOneHot,
                                TinyImagenetDataset,
                                DSpriteDataset,
                                OxfordFlowers102Dataset)
from torchvision import datasets
from torchvision.transforms import transforms
from PIL import Image
from torchvision.utils import save_image

# TODO: cleanup
from models.swapgan import SwapGAN
from models.amr_classifier import ClassifierOnly
from models.acai_f2 import ACAIF2
from models.acai_f3 import ACAIF3
from models.threegan import ThreeGAN
from models.twogan import TwoGAN
from models.twogan_supervised import TwoGANSupervised
from models.kgan import KGAN
from models.vae import VAE
from models.ae import AE

from functools import partial
from importlib import import_module
from tools import (
                   find_latest_pkl_in_folder,
                   train_logreg,
                   save_embedding,
                   save_class_embedding,
                   save_interp,
                   save_interp_supervised,
                   save_frames_continuous,
                   save_consistency_plot,
                   line2dict,
                   count_params,
                   dsprite_disentanglement,
                   dsprite_disentanglement_fv)


from models.classifier import Classifier

def get_shuffled_indices(length):
    rnd_state = np.random.RandomState(0)
    idxs = np.arange(length)
    rnd_state.shuffle(idxs)
    return idxs

DO_NOT_EXPORT = ['trial_id',
                 'config',
                 'which',
                 'name',
                 'epochs',
                 'save_path',
                 'val_batch_size',
                 'save_every',
                 'save_images_every',
                 'resume',
                 'load_nonstrict',
                 'no_verbose',
                 'num_workers',
                 'vis_seed',
                 'mode',
                 'mode_override']

if __name__ == '__main__':

    def parse_args():
        parser = argparse.ArgumentParser(description="")

        subparsers = parser.add_subparsers(
            help='Either load args from a config or specify them on the command line',
            dest='which')

        parser_load = subparsers.add_parser('load', help='Run from a YAML file')
        parser_load.add_argument('--config', type=str, default=None)

        # load = load from a yaml file
        # run
        parser_run = subparsers.add_parser('run', help='Run from the command line')

        parser_run.add_argument('--model', type=str, default='swapgan',
                                choices=['swapgan',
                                         'acai', 'acaif', 'acaif2', 'acaif3', 'tester',
                                         'threegan',
                                         'supervised',
                                         'kgan',
                                         'pixel_arae',
                                         'vae',
                                         'ae',
                                         'classifier'])
        parser_run.add_argument('--dataset', type=str, default='celeba',
                                choices=['celeba',
                                         'celeba32',
                                         'celeba_hq',
                                         'cifar10',
                                         'tiny-imagenet',
                                         'zappos',
                                         'zappos_cls',
                                         'zappos_hq',
                                         'flowers',
                                         'mnist',
                                         'mnist012_small',
                                         'mnist_small',
                                         'kmnist',
                                         'svhn',
                                         'dsprite',
                                         'dsprite64',
                                         'fashiongen'],
                                help="""
                                celeba = CelebA (64px) (set env var DATASET_CELEBA)
                                celeba32 = CelebA (32px) (set env var DATASET_CELEBA)
                                celeba_hq = CelebA-HQ (128px) (set env DATASET_CELEBAHQ)
                                zappos = Zappos shoe dataset (64px) (set env var DATASET_ZAPPOS)
                                zappos_cls = Zappos for classification (64px, fix shuffle bug)
                                zappos_hq = Zappos shoe dataset (128px) (set env var DATASET_ZAPPOSHQ)
                                flowers = TODO (64px)
                                mnist = MNIST (32px)
                                mnist012_small = MNIST for only digits 0,1,2 (16px)
                                mnist_small = MNIST (16px)
                                kmnist = KNIST (32px)
                                svhn = SVHN (32px)
                                fashiongen = TODO
                                """)
        parser_run.add_argument('--dataset_args', type=str,
                                default=None,
                                help="""Extra args to pass to the dataset class.""")
        parser_run.add_argument('--subset_train', type=int, default=None,
                                help="""If set, artficially decrease the size of the training
                                data. Use this to easily perform data ablation experiments.""")
        parser_run.add_argument('--arch', type=str,
                                default='architectures/arch_celeba.py')
        parser_run.add_argument('--batch_size', type=int, default=32)
        parser_run.add_argument('--n_channels', type=int, default=3,
                                help="""Number of input channels in image. This is
                                "passed to the `get_network` function defined by
                                "`--arch`.""")
        parser_run.add_argument('--ngf', type=int, default=64,
                                help="""# channel multiplier for the autoencoder. This
                                "is passed to the `get_network` function defined by
                                `--arch`.""")
        parser_run.add_argument('--ndf', type=int, default=64,
                                help="""# channel multiplier for the discriminator. This
                                is passed to the `get_network` function defined by
                                `--arch`.""")
        parser_run.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
        parser_run.add_argument('--lamb', type=float, default=1.0,
                                help="""Weight for reconstruction loss between x
                                "and its reconstruction.""")
        parser_run.add_argument('--beta', type=float, default=1.0,
                                help="""Weight for consistency loss between enc(x) and
                                enc(dec(enc(x)))""")
        parser_run.add_argument('--cls', type=float, default=0.,
                                help="""If > 0, then the supervised mixing losses
                                will be enabled. This is also the weight of the loss
                                term which tries to fool the auxiliary classifier with
                                mixes generated by the class mixing function.""")
        parser_run.add_argument('--k', type=int, default=None, help="For kgan only")
        # TENPORARY
        parser_run.add_argument('--sigma', type=float, default=0., help="dropout p")
        parser_run.add_argument('--eps', type=float, default=1e-8, help="For supervised formulation")
        # ---------
        parser_run.add_argument('--mixer', type=str, choices=['mixup', 'mixup2', 'fm', 'fm2'],
                                help="""The mixing function to use. Choices are 'mixup',
                                "'mixup2', and 'fm'. 'mixup' will sample an alpha ~ U(0,1)
                                "in the shape (bs,1,1,1), 'mixup2' will sample an alpha ~ U(0,1)
                                "in the shape (bs,f,1,1) and 'fm' will sample an m ~ Bern(p)
                                "(p ~ U(0,1)) in the shape (bs,f,1,1).""")
        parser_run.add_argument('--disable_g_recon', action='store_true')
        parser_run.add_argument('--disable_mix', action='store_true')
        parser_run.add_argument('--cls_loss', type=str, default='bce',
                                choices=['bce', 'mse'],
                                help="""The loss function to use for the auxiliary classifier
                                component (if cls > 0). The choices are 'bce' (binary 
                                cross-entropy) or 'mse' (mean-squared error).""")
        parser_run.add_argument('--gan_loss', type=str, default='bce',
                                choices=['bce', 'mse'],
                                help="""The loss function to use for the GAN component. The 
                                choices are 'bce' (binary cross-entropy) or 'mse' (mean-squared 
                                error).""")
        parser_run.add_argument('--recon_loss', type=str, default='l1',
                                choices=['l1', 'l2', 'bce', 'bce_sum'],
                                help="""The loss function to use for the reconstruction component. The
                                choices are 'bce' (binary cross-entropy) or 'l1' (L1 loss).""")
        parser_run.add_argument('--beta1', type=float, default=0., help="beta1 term of ADAM")
        parser_run.add_argument('--beta2', type=float, default=0.999, help="beta2 term of ADAM")
        parser_run.add_argument('--weight_decay', type=float, default=0.0,
                                help="""L2 weight decay on params (note: applies to optimisers for both
                                the generator, discriminator, and classifier probe (if set)""")
        parser_run.add_argument('--update_g_every', type=int, default=5)
        parser_run.add_argument('--cls_probe', type=str, default=None,
                                help="""Architecture for classifier to branch off the bottleneck
                                (if you are performing downstream classification)""")
        parser_run.add_argument('--cls_probe_args', type=str, default=None, help="python dict")
        parser_run.add_argument('--cls_probe_weight_decay', type=float, default=0,
                                help="Weight decay term specifically for cls_probe optimiser")
        parser_run.add_argument('--cls_probe_weight', type=float, default=1,
                                help="")
        parser_run.add_argument('--bpc', action='store_true',
                                help="""If set to true, the classifier probe can backprop gradients
                                back into the autoencoder""")
        parser_run.add_argument('--seed', type=int, default=0)

        for parser_obj in [parser_run, parser_load]:

            parser_obj.add_argument('--name', type=str, default=None,
                                    help="""The name of the experiment.
                                    """)
            parser_obj.add_argument('--epochs', type=int, default=200)
            parser_obj.add_argument('--trial_id', type=str, default=None)
            parser_obj.add_argument('--save_path', type=str, default=None)
            parser_obj.add_argument('--val_batch_size', type=int, default=64)
            parser_obj.add_argument('--save_every', type=int, default=5)
            parser_obj.add_argument('--save_images_every', type=int, default=1)
            parser_obj.add_argument('--resume', type=str, default=None)
            parser_obj.add_argument('--load_nonstrict', action='store_true')
            parser_obj.add_argument('--no_verbose', action='store_true')
            parser_obj.add_argument('--vis_seed', type=int, default=0,
                                help="Seed used for visualisation modes")
            parser_obj.add_argument('--num_workers', type=int, default=4)
            parser_obj.add_argument('--mode', type=str,
                                    choices=['train',
                                             'interp_train',
                                             'interp_valid',
                                             'interp_pixel_train',
                                             'interp_pixel_valid',
                                             'interp_sup_train',
                                             'interp_sup_valid',
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
                                             'dsprite_disentangle',
                                             'dsprite_disentangle_fv',
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
            parser_obj.add_argument('--mode_override', type=str, default=None)

        args = parser.parse_args()
        return args

    args = parse_args()

    args_dict = vars(args) # NOTE: this is a view, not a copy of `args`

    # If the config subparser was chosen, then `args` must be deserialised
    # from the file, otherwise continue.
    if args.which == 'load':

        print("args.which == load, so loading from config...")
        loaded_dict = yaml.load(open(args.config))
        #print("Loaded dict: ", loaded_dict)

        for key in loaded_dict:
            # We assume '' or 'true' ==> a boolean.
            if type(loaded_dict[key]) == str:
                if loaded_dict[key].lower() == 'true':
                    loaded_dict[key] = True
                elif loaded_dict[key].lower() == 'false':
                    loaded_dict[key] = False
                elif loaded_dict[key].lower() == 'null':
                    loaded_dict[key] = None

        for key in loaded_dict:
            if key not in DO_NOT_EXPORT:
                args_dict[key] = loaded_dict[key]
            else:
                print("WARNING: unknown key `%s` is specified in yaml, ignoring..." % key)

    # Make a copy of the args dict, remove
    # the stuff that isn't needed for the
    # exported config, then export
    args_dict_copy = dict(args_dict)
    for key in DO_NOT_EXPORT:
        if key in args_dict:
            del args_dict_copy[key]
    args_dict_copy_yaml = yaml.dump(args_dict_copy)
    print("Arguments:")
    print("  " + args_dict_copy_yaml.replace("\n", "\n  "))

    if args.mode == 'train':
        torch.manual_seed(args.seed)

    use_cuda = True if torch.cuda.is_available() else False

    ds_test = None
    ds_kwargs = eval(args.dataset_args) if args.dataset_args is not None else {}
    if args.dataset == 'celeba' or args.dataset == 'celeba32':
        img_sz = 64 if args.dataset == 'celeba' else 32
        train_transforms = [
            transforms.Resize(img_sz),
            transforms.CenterCrop(img_sz),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        ids = "celeba_frontal.txt"
        ds_train = CelebADataset(root=os.environ['DATASET_CELEBA'],
                                 ids=ids,
                                 transforms_=train_transforms,
                                 mode='train',
                                 **ds_kwargs)
        ds_valid = CelebADataset(root=os.environ['DATASET_CELEBA'],
                                 ids=ids,
                                 transforms_=train_transforms,
                                 mode='valid',
                                 **ds_kwargs)
        n_classes = len(ds_train.attrs)
    elif args.dataset == 'celeba_hq':
        train_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        ds_train = CelebAHqDataset(root=os.environ['DATASET_CELEBAHQ'],
                                   ids="iterators/celebahq_frontal.txt",
                                   transforms_=train_transforms,
                                   mode='train',
                                   **ds_kwargs)
        ds_valid = CelebAHqDataset(root=os.environ['DATASET_CELEBAHQ'],
                                   ids="iterators/celebahq_frontal.txt",
                                   transforms_=train_transforms,
                                   mode='valid',
                                   **ds_kwargs)
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
                                           mode='train',
                                           **ds_kwargs)
        ds_valid = OxfordFlowers102Dataset(root=os.environ['DATASET_FLOWERS'],
                                           transforms_=train_transforms,
                                           mode='valid',
                                           **ds_kwargs)
        n_classes = 102
    elif args.dataset in ['zappos', 'zappos_hq']:
        sz = 64 if args.dataset == 'zappos' else 128
        train_transforms = [
            transforms.Resize(sz),
            transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        ds_train = ZapposDataset(root=os.environ['DATASET_ZAPPOS'],
                                 ids="iterators/zappo_bg_stats.csv",
                                 transforms_=train_transforms,
                                 mode='train',
                                 **ds_kwargs)
        ds_valid = ZapposDataset(root=os.environ['DATASET_ZAPPOS'],
                                 ids="iterators/zappo_bg_stats.csv",
                                 transforms_=train_transforms,
                                 mode='valid',
                                 **ds_kwargs)
        n_classes = 0
    elif args.dataset == 'zappos_cls':
        train_transforms = [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        ds_train = ZapposDataset(root=os.environ['DATASET_ZAPPOS'],
                                 ids=None,
                                 shuffle=True,
                                 transforms_=train_transforms,
                                 mode='train',
                                 **ds_kwargs)
        ds_valid = ZapposDataset(root=os.environ['DATASET_ZAPPOS'],
                                 ids=None,
                                 shuffle=True,
                                 transforms_=train_transforms,
                                 mode='valid',
                                 **ds_kwargs)
        n_classes = 21
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
        ds_train_valid = mnist_class('iterators', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(img_sz),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,),
                                                              (0.5,))
                                     ]))
        idxs = get_shuffled_indices(60000)
        ds_train = Subset(ds_train_valid, idxs[0:50000]) # train is first 50k
        ds_valid = Subset(ds_train_valid, idxs[50000::]) # valid is last 10k
        ds_test = mnist_class('iterators', train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.Resize(img_sz),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,),
                                                       (0.5,))
                              ]))
        n_classes = 10
    elif args.dataset == 'kmnist':
        ds_train_and_valid = KMnistDatasetOneHot('iterators', train=True, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.Resize(32),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5,),
                                                                          (0.5,))
                                                 ]))
        idxs = get_shuffled_indices(60000)
        ds_train = Subset(ds_train_and_valid, indices=idxs[0:50000])
        ds_valid = Subset(ds_train_and_valid, indices=idxs[50000:])
        ds_test = KMnistDatasetOneHot('iterators', train=False, download=True,
                                      transform=transforms.Compose([
                                          transforms.Resize(32),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,),
                                                               (0.5,))
                                      ]))
        n_classes = 10
    elif args.dataset == 'svhn':
        ds_train_and_valid = SvhnDatasetOneHot('iterators', split='train', download=True,
                                               transform=transforms.Compose([
                                                   transforms.Resize(32),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5,),
                                                                        (0.5,))
                                               ]))
        idxs = get_shuffled_indices(ds_train_and_valid.data.shape[0])
        ds_train = Subset(ds_train_and_valid, indices=idxs[0:int(len(idxs)*0.9)])
        ds_valid = Subset(ds_train_and_valid, indices=idxs[int(len(idxs)*0.9)::])
        ds_test = SvhnDatasetOneHot('iterators', split='test', download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,),
                                                             (0.5,))
                                    ]))
        n_classes = 10
    elif args.dataset == 'tiny-imagenet':
        this_transform = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(56),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,),
                                 (0.5,))
        ]

        ds_train_and_valid = TinyImagenetDataset(root=os.environ['DATASET_TINY_IMAGENET'],
                                                 transforms_=this_transform,
                                                 mode='train',
                                                 **ds_kwargs)
        idxs = get_shuffled_indices(len(ds_train_and_valid))
        ds_train = Subset(ds_train_and_valid, indices=idxs[0:int(len(idxs)*0.9)])
        ds_valid = Subset(ds_train_and_valid, indices=idxs[int(len(idxs)*0.9)::])

        # This is actually the valid set -- but the actual test set labels
        # don't exist publicly.
        ds_test = TinyImagenetDataset(root=os.environ['DATASET_TINY_IMAGENET'],
                                      transforms_=this_transform,
                                      mode='valid',
                                      **ds_kwargs)
        n_classes = 200
    elif args.dataset == 'cifar10':
        ds_train_valid = CifarDatasetOneHot('iterators', train=True, download=True,
                                            transform=transforms.Compose([
                                                transforms.Resize(32),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,),
                                                                     (0.5,))
                                            ]))
        idxs = get_shuffled_indices(50000)
        ds_train = Subset(ds_train_valid, idxs[0:40000]) # train is first 40k
        ds_valid = Subset(ds_train_valid, idxs[40000::]) # valid is last 10k

        ds_test = CifarDatasetOneHot('iterators', train=False, download=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(32),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,),
                                                              (0.5,))
                                     ]))
        n_classes = 10
    elif args.dataset == 'dsprite':
        ds_train = DSpriteDataset(root='iterators',
                                  seed=args.seed,
                                  **ds_kwargs)
        """
        ds_valid = DSpriteDataset('iterators', split='test', download=True,
                                     transform=transforms.Compose([
                                         transforms.Resize(32),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,),
                                                              (0.5,))
                                     ]))
        """
        ds_valid = ds_train # for now they are the same
        n_classes = 0 # not applicable here
    elif args.dataset == 'fashiongen':
        train_transforms = [
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        ds_train = FashionGenDataset(root=os.environ['DATASET_FASHIONGEN'],
                                     transforms_=train_transforms,
                                     mode='train',
                                     **ds_kwargs)
        ds_valid = FashionGenDataset(root=os.environ['DATASET_FASHIONGEN'],
                                     transforms_=train_transforms,
                                     mode='valid',
                                     **ds_kwargs)
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
    loader_test = None
    if ds_test is not None:
        loader_test = DataLoader(ds_test,
                                 batch_size=bs,
                                 shuffle=False,
                                 num_workers=1)

    if args.save_path is None:
        print("`save_path` not specified, so retrieving from $SAVE_PATH...")
        args.save_path = os.environ['SAVE_PATH']
    if args.seed == 0:
        save_path = args.save_path
    else:
        save_path = "%s/s%i" % (args.save_path, args.seed)
    print("Save path: %s" % save_path)

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
    print("  " + str(gen).replace("\n","\n  "))
    print("  # learnable params: %i" % count_params(gen))
    print("Discriminator:")
    print("  " + str(disc_x).replace("\n"," \n  "))
    print("  # learnable params: %i" % count_params(disc_x))
    if class_mixer is not None:
        print("Class mixer:")
        print("  " + str(class_mixer).replace("\n", "\n  "))
        print("  # learnable params: %i" % count_params(class_mixer))

    from handlers import (image_handler_default,
                          image_handler_blank,
                          image_handler_vae,
                          image_handler_ae,
                          dsprite_handler,
                          test_set_handler)

    if args.trial_id is None:
        trial_id = os.environ['SLURM_JOB_ID']
        print("args.trial_id is `None`, so generating name from SLURM instead:\n  " +
              trial_id)
        args.trial_id = trial_id

    if args.name is None:
        name = ("".join([ random.choice(string.ascii_letters[0:26]) for j in range(5) ]))
        print("args.name is `None`, so generating a random 5 digit code:\n  " +
              name)
        args.name = name

    # e.g. expt_dir = /results/<experiment name>/<trial id>
    expt_dir = "%s/%s/%s" % (save_path, args.name, trial_id)
    if not os.path.exists(expt_dir):
        os.makedirs(expt_dir)
    print("Experiment dir: %s" % expt_dir)

    # Save the yaml args to the experiment dir
    with open("%s/cfg.yaml" % expt_dir, "w") as f_yaml:
        f_yaml.write(args_dict_copy_yaml)

    if args.model == 'swapgan':
        gan_class = TwoGAN
        image_handler = image_handler_default
    elif args.model == 'supervised':
        gan_class = TwoGANSupervised
        image_handler = image_handler_default
    elif args.model == 'acaif2':
        gan_class = ACAIF2
        image_handler = image_handler_default
    elif args.model == 'acaif3':
        gan_class = ACAIF3
        image_handler = image_handler_default
    elif args.model == 'threegan':
        gan_class = ThreeGAN
        image_handler = image_handler_default
    elif args.model == 'kgan':
        if args.k is None:
            raise Exception("`k` must be > 0 for KGAN")
        gan_class = partial(KGAN, k=args.k)
        image_handler = image_handler_default
    elif args.model == 'classifier':
        gan_class = ClassifierOnly
        image_handler = image_handler_blank
    elif args.model == 'vae':
        gan_class = VAE
        image_handler = image_handler_vae
    elif args.model == 'ae':
        gan_class = AE
        image_handler = image_handler_ae

    handlers = [
        image_handler(save_path=expt_dir,
                      save_images_every=args.save_images_every)
    ]
    if args.dataset == 'dsprite':
        is_vae = True if args.model == 'vae' else False
        handlers.append(
            dsprite_handler(gen=gen,
                            dataset=ds_train,
                            is_vae=is_vae)
        )

    cls_enc = None
    if args.cls_probe is not None:
        mod_cls = import_module(args.cls_probe.replace("/", ".").\
                                replace(".py", ""))
        cls_enc = mod_cls.get_network(**eval(args.cls_probe_args))
        print("Classifier probe:")
        print("  " + str(cls_enc).replace("\n", "\n  "))
        print("  # params: %i" % count_params(cls_enc))

    # TODO: check for BaseAE, not VAE
    if gan_class not in [VAE, AE]:
        gan = gan_class(
            generator=gen,
            disc_x=disc_x,
            class_mixer=class_mixer,
            lamb=args.lamb,
            beta=args.beta,
            cls=args.cls,
            mixer=args.mixer,
            sigma=args.sigma,
            recon_loss=args.recon_loss,
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
            cls_enc_weight=args.cls_probe_weight,
            cls_enc_backprop_grads=args.bpc,
            handlers=handlers
        )
        gan.eps = args.eps
        if args.load_nonstrict:
            gan.load_strict = False
    else:
        gan = gan_class(
            generator=gen,
            lamb=args.lamb,
            beta=args.beta,
            recon_loss=args.recon_loss,
            opt_args={'lr': args.lr,
                      'betas': (args.beta1, args.beta2),
                      'weight_decay': args.weight_decay},
            handlers=handlers
        )
        if args.load_nonstrict:
            gan.load_strict = False

    if args.cls_probe is not None and loader_test is not None:
        # If we have a linear classifier probe turned on and
        # a test set exists, assume that we want to also make
        # test set classifications per epoch (these will be
        # periodically saved to disk, and not seen during training).
        handlers.append(
            test_set_handler(
                gan=gan,
                dataset=loader_test,
                save_path="%s/%s/test_preds" % (save_path, args.name)
            )
        )


    latest_model = None
    cpt_name = 'none'
    if args.resume is not None:
        model_dir = "%s/%s" % (save_path, args.name)
        if args.resume == 'auto':
            # autoresume
            latest_model = find_latest_pkl_in_folder(model_dir)
            if latest_model is not None:
                gan.load(latest_model)
        elif args.resume.isdigit():
            gan.load("%s/%s.pkl" % (model_dir, args.resume))
        else:
            gan.load(args.resume)
        if latest_model is not None:
            cpt_name = os.path.basename(latest_model)
        else:
            cpt_name = os.path.basename(args.resume)

    if args.mode == 'train':

        gan.train(itr_train=loader_train,
                  itr_valid=loader_valid,
                  epochs=args.epochs,
                  model_dir=expt_dir,
                  result_dir=expt_dir,
                  save_every=args.save_every)

    elif 'interp' in args.mode:

        torch.manual_seed(args.vis_seed)

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

        torch.manual_seed(args.vis_seed)

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

    elif args.mode == 'dsprite_disentangle':
        if args.dataset != 'dsprite':
            raise Exception("The mode `dsprite_disentangle` is only compatible " +
                            "with the `dsprite` dataset!")
        kwargs = {'num_examples': 50000}
        if args.mode_override is not None:
            override_kwargs = line2dict(args.mode_override)
            for key in override_kwargs:
                kwargs[key] = override_kwargs[key]
        print("kwargs:", kwargs)
        dsprite_disentanglement(gan,
                                ds_train,
                                save_path="%s/%s/disentangle/%s/model-%s/" % (save_path, args.name, args.mixer, cpt_name),
                                num_examples=kwargs['num_examples'])

    elif args.mode == 'dsprite_disentangle_fv':
        if args.dataset != 'dsprite':
            raise Exception("The mode `dsprite_disentangle` is only compatible " +
                            "with the `dsprite` dataset!")
        override_kwargs = {}
        if args.mode_override is not None:
            override_kwargs = line2dict(args.mode_override)
        print("kwargs:", override_kwargs)
        dval = dsprite_disentanglement_fv(gan.generator,
                                          ds_train,
                                          is_vae=True if args.model == 'vae' else False,
                                          verbose=True,
                                          save_path="%s/%s/disentangle_fv/%s/model-%s/" % (save_path, args.name, args.mixer, cpt_name),
                                          **override_kwargs)
        print("Disentanglement metric: %f" % dval['dfv'])

    elif args.mode == 'pdb':
        import pdb
        pdb.set_trace()
