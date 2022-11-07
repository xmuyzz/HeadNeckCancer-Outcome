import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import nibabel as nib
import torch
import torchtuples as tt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pycox.datasets import metabric
from pycox.models import PCHazard, CoxPH, LogisticHazard, DeepHitSingle


def collate_fn(batch):
    """
    Stacks the entries of a nested tuple
    """
    return tt.tuplefy(batch).stack()


class DatasetPCHazard():
    """
    Dataset class for PCHazard model
    Includes image and lables for training and tuning dataloader
    """
    def __init__(self, data, idx_duration, event, t_frac):
        self.data = data
        self.idx_duration, self.event, self.t_frac = tt.tuplefy(
            idx_duration, event, t_frac).to_tensor()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        img = self.data[index]
        return img, (self.idx_duration[index], self.event[index], self.t_frac[index])


class Dataset0(Dataset):
    """Dataset class for Logistic Hazard model
    """
    def __init__(self, df, transform):
        self.df = df
        self.img_dir = df['img_dir'].to_list()
        self.time = torch.from_numpy(df['time'].to_numpy())
        self.event = torch.from_numpy(df['event'].to_numpy())
        self.transform = transform

    def __len__(self):
        #print('event size:', self.event.size(dim=0))
        return self.event.shape[0]

    def __getitem__(self, index):
        assert type(index) is int
        img = nib.load(self.img_dir[index]).get_data()
        # choose image channel
        #img = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        #if self.channel == 1:
        #    img = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        #elif self.channel == 3:
        #    img = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        # convert numpy array to torch tensor
        #img = torch.from_numpy(img).float()
        # data and label transform
        #print(arr.shape)
        if self.transform:
            img = self.transform(img)
        return img, (self.time[index], self.event[index])


class DatasetPred(Dataset):
    """Dataset class for CoxPH model
    """
    def __init__(self, df, transform):
        self.img_dir = df['img_dir'].to_list()
        self.transform = transform

    def __len__(self):
        print('data size:', len(self.img_dir))
        return len(self.img_dir)

    def __getitem__(self, index):
        assert type(index) is int
        img = nib.load(self.img_dir[index])
        arr = img.get_data()
        img = arr.reshape(1, arr.shape[0], arr.shape[1], arr.shape[2])
        # choose image channel
        #if self.channel == 1:
        #    img = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        #elif self.channel == 3:
        #    img = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        #img = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        #img = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        # convert numpy array to torch tensor
        #img = torch.from_numpy(img).float()
        # data and label transform
        if self.transform:
            img = self.transform(img)
        return img


class DatasetDeepHit(Dataset):
    """Dataset class for DeepHit model
    """
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        event = self.df['event'].to_numpy()
        return event.shape[0]

    def __getitem__(self, index):
        assert type(index) is int
        # image
        img_dir = self.df['img_dir'].to_list()
        img = nib.load(img_dir[index])
        arr = img.get_data()
        img = arr.reshape(1, arr.shape[0], arr.shape[1], arr.shape[2])
        # target
        time = self.df['time'].to_numpy()
        event = self.df['event'].to_numpy()
        rank_mat = pair_rank_mat(time[index], event[index])
        time = torch.from_numpy(time)
        event = torch.from_numpy(event)
        rank_mat = torch.from_numpy(rank_mat)
        if self.transform:
            img = self.transform(img)

        return img, (time[index], event[index], rank_mat)


class DatasetCoxPH(Dataset):
    """Dataset class for CoxPH method
    """
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        event = self.df['event'].to_numpy()
        return event.shape[0]

    def __getitem__(self, index):
        assert type(index) is int
        img_dir = self.df['img_dir'].to_list()
        img = nib.load(img_dir[index]).get_data()
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        time = torch.from_numpy(self.df['time'].to_numpy())
        event = torch.from_numpy(self.df['event'].to_numpy())
        if self.transform:
            img = self.transform(img)
        return img, (time[index], event[index])







