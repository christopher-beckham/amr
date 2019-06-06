# Source: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/stargan/datasets.py

import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets

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
                 ids,
                 thresh=0.65,
                 transforms_=None,
                 mode='train'):
        self.transform = transforms.Compose(transforms_)
        #files = glob.glob("%s/**" % root, recursive=True)
        files = []
        with open(ids) as f:
            for line in f:
                line = line.rstrip().split(",")
                if float(line[1]) <= thresh:
                    files.append("%s/%s" % (root, line[0]))
        self.files = [os.path.abspath(file_) for file_ in files if file_.endswith('.jpg')]
        self.files = self.files[:-2000] if mode == 'train' else self.files[-2000:]

    def __getitem__(self, index):
        filepath = self.files[index]
        img = Image.open(filepath).convert('RGB')
        img = self.transform(img)
        return img, torch.zeros((1, 1)).float()

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
