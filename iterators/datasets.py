# Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/stargan/datasets.py

import glob
import random
import os
import numpy as np
import torch
from scipy import io
from collections import Counter

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets

from skimage.transform import rescale

class CelebADataset(Dataset):
    def __init__(self,
                 root,
                 ids=None,
                 transforms_=None,
                 mode='train',
                 attrs=[],
                 missing_ind=False):
        self.transform = transforms.Compose(transforms_)
        if ids is None:
            self.files = sorted(glob.glob('%s/*.jpg' % root))
        else:
            ids_file = open(ids).read().split("\n")
            self.files = ["%s/%s.jpg" % (root, id_) for id_ in ids_file]
        self.files = self.files[:-2000] if mode == 'train' else self.files[-2000:]
        self.label_path = "%s/list_attr_celeba.txt" % root
        self.missing_ind = missing_ind
        self.annotations = self.get_annotations(attrs)
        self.keys1 = list(self.annotations.keys())

    def get_annotations(self, attrs):
        """Extracts annotations for CelebA"""
        annotations = {}
        lines = [line.rstrip() for line in open(self.label_path, 'r')]
        self.label_names = lines[1].split()
        for _, line in enumerate(lines[2:]):
            filename, *values = line.split()
            labels = []
            for attr in attrs:
                idx = self.label_names.index(attr)
                labels.append(1 * (values[idx] == '1'))
            if self.missing_ind:
                # Basically add a label saying this is the
                # 'everything else' class.
                if 1 not in labels:
                    labels.append(1)
                else:
                    labels.append(0)
            annotations[filename] = labels
        return annotations

    def __getitem__(self, index):
        filepath = self.files[index % len(self.files)]
        filename = filepath.split('/')[-1]
        img = self.transform(Image.open(filepath))
        label = self.annotations[filename]
        label = torch.FloatTensor(np.array(label))
        return img, label

    def __len__(self):
        return len(self.files)


class ZapposDataset(Dataset):
    def __init__(self,
                 root,
                 ids=None,
                 shuffle=False,
                 thresh=0.65,
                 transforms_=None,
                 mode='train'):
        self.transform = transforms.Compose(transforms_)
        files = []
        if ids is not None:
            with open(ids) as f:
                for line in f:
                    line = line.rstrip().split(",")
                    if float(line[1]) <= thresh:
                        files.append("%s/%s" % (root, line[0]))
        else:
            files = glob.glob("%s/**" % root, recursive=True)
        self.files = np.asarray(
            [os.path.abspath(file_) for file_ in files if file_.endswith('.jpg')]
        )

        # Now figure out all the class names and make a dictionary
        # mapping them to indices.
        self.classes = []
        marker = "ut-zap50k-images-square"
        for filename in self.files:
            self.classes.append( "-".join(filename[ filename.index(marker)+len(marker)+1 :: ].split("/")[0:2]) )
        counter = Counter(self.classes)
        class_names = sorted(counter.keys())
        self.name2idx = {name:i for i,name in enumerate(class_names)}
        self.classes = np.asarray(self.classes)

        # Shuffle files and classes if necessary.
        rnd_state = np.random.RandomState(0)
        idxs = np.arange(0, len(self.files))
        rnd_state.shuffle(idxs)
        self.files = self.files[idxs]
        self.classes = self.classes[idxs]

        self.files = self.files[:-2000] if mode == 'train' else self.files[-2000:]

    def __getitem__(self, index):
        filepath = self.files[index]
        label = self.name2idx[self.classes[index]]
        img = Image.open(filepath).convert('RGB')
        img = self.transform(img)
        return img, torch.from_numpy([label]).long()
    #torch.zeros((1, 1)).float()

    def __len__(self):
        return len(self.files)






class ZapposPairDataset(Dataset):
    def __init__(self,
                 root,
                 pairs,
                 transforms_=None,
                 mode='train'):
        self.transform = transforms.Compose(transforms_)
        files = []
        with open(pairs) as f:
            for line in f:
                line = line.rstrip().split(",")
                p1, p2 = line
                p1 = p1.replace("./ut-zap50k-images-square", "")
                p2 = p2.replace("./ut-zap50k-images-square", "")
                files.append( ("%s/%s" % (root, p1), "%s/%s" % (root, p2)) )
        self.files = files[:-2000] if mode == 'train' else files[-2000:]

    def __getitem__(self, index):
        filepath1, filepath2 = self.files[index]
        img1 = Image.open(filepath1).convert('RGB')
        img1 = self.transform(img1)
        img2 = Image.open(filepath2).convert('RGB')
        img2 = self.transform(img2)
        return img1, img2, torch.zeros((1, 1)).float()

    def __len__(self):
        return len(self.files)


















# TODO: should refactor all of these datasets...
class FashionGenDataset(Dataset):
    def __init__(self,
                 root,
                 transforms_=None,
                 mode='train'):
        self.transform = transforms.Compose(transforms_)
        files = glob.glob("%s/*.png" % root)
        self.files = [os.path.abspath(file_) for file_ in files if file_.endswith('.png')]
        self.files = self.files[:-150] if mode == 'train' else self.files[-150:]

    def __getitem__(self, index):
        filepath = self.files[index]
        img = Image.open(filepath).convert('RGB')
        img = self.transform(img)
        return img, torch.zeros((1, 1)).float()

    def __len__(self):
        return len(self.files)

# TODO: should refactor all of these datasets...
class CelebAHqDataset(Dataset):
    def __init__(self,
                 root,
                 ids,
                 transforms_=None,
                 mode='train'):
        self.transform = transforms.Compose(transforms_)
        if ids is None:
            self.files = sorted(glob.glob('%s/*.png' % root))
        else:
            ids_file = open(ids).read().split("\n")
            if ids_file[-1] == '':
                ids_file = ids_file[:-1]
            # TODO: remove .png from the frontal ids list
            self.files = ["%s/%s" % (root, id_) for id_ in ids_file]
        self.files = self.files[:-2000] if mode == 'train' else self.files[-2000:]

    def __getitem__(self, index):
        filepath = self.files[index]
        img = Image.open(filepath).convert('RGB')
        img = self.transform(img)
        return img, torch.zeros((1, 1)).float()

    def __len__(self):
        return len(self.files)


class MnistDatasetOneHot(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super(MnistDatasetOneHot, self).__init__(*args, **kwargs)
    def __getitem__(self, index):
        get_item = super(MnistDatasetOneHot, self).__getitem__
        x_batch, y_batch = get_item(index)
        y_batch_onehot = torch.eye(10)[y_batch]
        y_batch_onehot = y_batch_onehot.float()
        return x_batch, y_batch_onehot

class SvhnDatasetOneHot(datasets.SVHN):
    def __init__(self, *args, **kwargs):
        super(SvhnDatasetOneHot, self).__init__(*args, **kwargs)
    def __getitem__(self, index):
        get_item = super(SvhnDatasetOneHot, self).__getitem__
        x_batch, y_batch = get_item(index)
        y_batch_onehot = torch.eye(10)[y_batch]
        y_batch_onehot = y_batch_onehot.float()
        return x_batch, y_batch_onehot

class CifarDatasetOneHot(datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CifarDatasetOneHot, self).__init__(*args, **kwargs)
    def __getitem__(self, index):
        get_item = super(CifarDatasetOneHot, self).__getitem__
        x_batch, y_batch = get_item(index)
        y_batch_onehot = torch.eye(10)[y_batch]
        y_batch_onehot = y_batch_onehot.float()
        return x_batch, y_batch_onehot


class MnistDataset012(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super(MnistDataset012, self).__init__(*args, **kwargs)
        if hasattr(self, 'train_data'):
            idcs = np.where(np.isin( self.train_labels.numpy(), [0,1,2] ))[0]
            self.train_data = self.train_data[idcs]
            self.train_labels = self.train_labels[idcs]
        else:
            idcs = np.where(np.isin( self.test_labels.numpy(), [0,1,2] ))[0]
            self.test_data = self.test_data[idcs]
            self.test_labels = self.test_labels[idcs]

    def __getitem__(self, index):
        get_item = super(MnistDataset012, self).__getitem__
        x_batch, y_batch = get_item(index)
        y_batch_onehot = torch.eye(10)[y_batch]
        y_batch_onehot = y_batch_onehot.float()
        return x_batch, y_batch_onehot

    def __len__(self):
        if hasattr(self, 'train_data'):
            return len(self.train_data)
        else:
            return len(self.test_data)


class KMnistDatasetOneHot(datasets.KMNIST):
    def __init__(self, *args, **kwargs):
        super(KMnistDatasetOneHot, self).__init__(*args, **kwargs)
    def __getitem__(self, index):
        get_item = super(KMnistDatasetOneHot, self).__getitem__
        x_batch, y_batch = get_item(index)
        y_batch_onehot = torch.eye(10)[y_batch]
        y_batch_onehot = y_batch_onehot.float()
        return x_batch, y_batch_onehot


class TinyImagenetDataset(Dataset):
    def __init__(self,
                 root,
                 transforms_=None,
                 mode='train'):
        if mode not in ['train', 'valid']:
            raise Exception()
        self.transform = transforms.Compose(transforms_)
        class_names = open('%s/wnids.txt' % root).read().split("\n")[0:-1]
        class2idx = {}
        for i in range(len(class_names)):
            class2idx[class_names[i]] = i
        if mode == 'train':
            self.files = glob.glob('%s/train/*/images/*.JPEG' % root)
            self.labels = [class2idx[os.path.basename(fname).split("_")[0]] for fname in self.files]
        else:
            self.files = []
            self.labels = []
            with open('%s/val/val_annotations.txt' % root) as f:
                for line in f:
                    this_fname, this_class = line.split("\t")[0:2]
                    self.files.append('%s/val/images/%s' % (root, this_fname))
                    self.labels.append(class2idx[this_class])
        self.root = root
        self.class2idx = class2idx

    def __getitem__(self, index):
        filepath = self.files[index]
        img = self.transform(Image.open(filepath).convert('RGB'))
        label = torch.eye(len(self.class2idx.keys()))[self.labels[index]].float()
        return img, label

    def __len__(self):
        return len(self.files)


class OxfordFlowers102Dataset(Dataset):
    def __init__(self,
                 root,
                 transforms_=None,
                 mode='train',
                 attrs=[],
                 missing_ind=False):
        self.transform = transforms.Compose(transforms_)
        ids = np.arange(1, 8189+1)
        indices = np.arange(0, len(ids))
        rnd_state = np.random.RandomState(0)
        rnd_state.shuffle(indices)
        labels = io.loadmat('%s/imagelabels.mat' % root)['labels'].flatten()-1
        # Shuffle both ids and labels with the same indices.
        labels = labels[indices]
        ids = ids[indices]
        if mode == 'train':
            # Training set is first 90%.
            self.ids = ids[0:int(len(ids)*0.9)]
            self.labels = labels[0:int(len(ids)*0.9)]
        else:
            # Valid set is last 10%.
            self.ids = ids[int(len(ids)*0.9)::]
            self.labels = labels[int(len(ids)*0.9)::]
        self.root = root

    def __getitem__(self, index):
        jpg_name = "image_" + str(self.ids[index]).zfill(5) + ".jpg"
        filepath = "%s/jpg/%s" % (self.root, jpg_name)
        img = self.transform(Image.open(filepath))
        label = torch.eye(102)[self.labels[index]].float()
        return img, label

    def __len__(self):
        return len(self.ids)

class DSpriteDataset(Dataset):
    def __init__(self, root, seed=0):
        dataset_zip = np.load('%s/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz' % root,
                              encoding='latin1' )
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]
        # Define number of values per latents and functions to convert to indices
        self.latents_sizes = self.metadata['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:],
                                            np.array([1,])))
        self.set_rnd_state(seed)

    def set_rnd_state(self, seed):
        self.rnd_state = np.random.RandomState(seed)

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = self.rnd_state.randint(lat_size, size=size)

        return samples

    def sample_conditional(self, which_idx, which_val, size):
        latents_sampled = self.sample_latent(size=size)
        latents_sampled[:, which_idx] = which_val
        indices_sampled = self.latent_to_index(latents_sampled)
        imgs_sampled = self.imgs[indices_sampled]
        return torch.FloatTensor((imgs_sampled - 0.5) / 0.5)

    def __getitem__(self, idx):
        # We won't be using torch transforms here
        # cos converting to PIL and back to np is
        # ugly.
        this_img = self.imgs[idx]
        #this_img = rescale(this_img, 0.5,
        #                   anti_aliasing=False,
        #                   preserve_range=True)
        this_img = this_img[np.newaxis, :, :]
        this_img = (this_img - 0.5) / 0.5
        this_img = torch.FloatTensor(this_img)
        return this_img, torch.zeros((1,1)).float()

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    ds = ZapposDataset(root="/tmp/beckhamc/ut-zap50k-images-square",
                       ids=None)
    print(ds)
