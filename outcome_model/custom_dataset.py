import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import nibabel as nib
import SimpleITK as sitk
import torch
import torchtuples as tt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pycox.datasets import metabric
from pycox.models import PCHazard, CoxPH, LogisticHazard, DeepHitSingle
import ast

def collate_fn(batch):
    """
    Stacks the entries of a nested tuple
    """
    #print(tt.tuplefy(batch))
    #print(tt.tuplefy(batch).shapes())
    #print(tt.tuplefy(batch).stack())
    #print('batch:', batch)
    # print('batch:', type(batch))
    # for i in batch:
    #     print('data:')
    #     img = i[0]
    #     #print(img)
    #     print(img.shape)   
    #     print(i[1])
    data = tt.tuplefy(batch).stack()
    #print(data.shapes())
    #print('tt stack works')

    return data


class Dataset0(Dataset):
    """Dataset class for Logistic Hazard model
    """
    def __init__(self, df, transform, in_channels):
        self.df = df
        self.img_dir = df['img_dir'].to_list()
        self.time = torch.from_numpy(df['time'].to_numpy())
        self.event = torch.from_numpy(df['event'].to_numpy())
        #self.time, self.event = tt.tuplefy(df['time'].to_numpy(), df['event'].to_numpy()).to_tensor()
        #print(self.time)
        #print(self.event)
        self.transform = transform
        self.in_channels = in_channels
        #self.transform = False

    def __len__(self):
        #print('event size:', self.event.size(dim=0))
        return self.event.shape[0]

    def __getitem__(self, index):
        #if not hasattr(index, '__iter__'):
        #   index = [index]
        #print('index:', index)
        assert type(index) is int
        #img = nib.load(self.img_dir[index]).get_fdata()
        img = sitk.ReadImage(self.img_dir[index])
        img = sitk.GetArrayFromImage(img)
        #print(img.shape)
        # choose image channel
        #img = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        #img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img = img.reshape(self.in_channels, img.shape[0], img.shape[1], img.shape[2])
        #print(img.shape)
        
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
        # need to convert np array to torch tensor before output to tuples
        img = torch.from_numpy(img).float()
        return img, (self.time[index], self.event[index])


class Dataset_Concat_Tr(Dataset):
    """Dataset class for Logistic Hazard model
    """
    def __init__(self, df, transform, in_channels):
        self.df = df
        self.img_dir = df['img_dir'].to_numpy()
        # self.age = torch.from_numpy(df['Age>65'].to_numpy())
        # self.sex = torch.from_numpy(df['Female'].to_numpy())
        # self.n_stage = torch.from_numpy(df['N-Stage-0123'].to_numpy())
        # self.t_stage = torch.from_numpy(df['T-Stage-1234'].to_numpy())
        #print(df['Age>65_oh'])
        df['Age>65_oh'] = df['Age>65_oh'].apply(ast.literal_eval)
        df['Female_oh'] = df['Female_oh'].apply(ast.literal_eval)
        df['N-Stage-0123_oh'] = df['N-Stage-0123_oh'].apply(ast.literal_eval)
        df['T-Stage-1234_oh'] = df['T-Stage-1234_oh'].apply(ast.literal_eval)
        df['Smoking>10py_oh'] = df['Smoking>10py_oh'].apply(ast.literal_eval)
        df['HPV_oh'] = df['HPV_oh'].apply(ast.literal_eval)
        
        # self.age = torch.from_numpy(np.array(df['Age>65_oh'].tolist()))
        # self.sex = torch.from_numpy(np.array(df['Female_oh'].tolist()))
        # self.n_stage = torch.from_numpy(np.array(df['N-Stage-0123_oh'].tolist()))
        # self.t_stage = torch.from_numpy(np.array(df['T-Stage-1234_oh'].tolist()))
        # self.smoking = torch.from_numpy(np.array(df['Smoking>10py_oh'].tolist()))
        # self.hpv = torch.from_numpy(np.array(df['HPV_oh'].tolist()))

        self.age = np.array(df['Age>65_oh'].tolist())
        self.sex = np.array(df['Female_oh'].tolist())
        self.n_stage = np.array(df['N-Stage-0123_oh'].tolist())
        self.t_stage = np.array(df['T-Stage-1234_oh'].tolist())
        self.smoking = np.array(df['Smoking>10py_oh'].tolist())
        self.hpv = np.array(df['HPV_oh'].tolist())

        self.time = torch.from_numpy(df['time'].to_numpy())
        self.event = torch.from_numpy(df['event'].to_numpy())
        #self.time, self.event = tt.tuplefy(df['time'].to_numpy(), df['event'].to_numpy()).to_tensor()
        self.in_channels = in_channels
        self.transform = transform

    def __len__(self):
        #print('event size:', self.event.size(dim=0))
        return self.event.shape[0]

    def __getitem__(self, index):
        assert type(index) is int
        # get img and convert to torch tensor
        img = sitk.ReadImage(self.img_dir[index])
        img = sitk.GetArrayFromImage(img)
        img = img.reshape(self.in_channels, img.shape[0], img.shape[1], img.shape[2])
        img = self.transform(img)
        img = torch.from_numpy(img).float()
        # make clinical list
        #clinical_list = [self.age[index], self.sex[index], self.n_stage[index], self.t_stage[index]]
        clinical_list = [self.age[index], 
                         self.sex[index], 
                         self.n_stage[index], 
                         self.t_stage[index], 
                         self.smoking[index],
                         self.hpv[index]]
        #print('clinical list:', clinical_list)
        clinical = torch.from_numpy(np.array(clinical_list))

        return (img, clinical), (self.time[index], self.event[index])


class Dataset_Concat_Ts(Dataset):
    """
    Dataset class for cumstomized DenseNet model
    """
    def __init__(self, df, transform, in_channels):
        self.df = df
        self.img_dir = df['img_dir'].to_numpy()
        # self.age = torch.from_numpy(df['Age>65_oh'].to_numpy())
        # self.sex = torch.from_numpy(df['Female_oh'].to_numpy())
        # self.n_stage = torch.from_numpy(df['N-Stage-0123_oh'].to_numpy())
        # self.t_stage = torch.from_numpy(df['T-Stage-1234_oh'].to_numpy())
        # self.smoking = torch.from_numpy(df['Smoking>10py_oh'].to_numpy())
        # self.hpv = torch.from_numpy(df['HPV_oh'].to_numpy())
        #print(df['Age>65_oh'])
        # df['Age>65_oh'] = df['Age>65_oh'].apply(lambda x: eval(x))
        # #df['Age>65_oh'] = df['Age>65_oh'].apply(ast.literal_eval)
        # df['Female_oh'] = df['Female_oh'].apply(ast.literal_eval)
        # df['N-Stage-0123_oh'] = df['N-Stage-0123_oh'].apply(ast.literal_eval)
        # df['T-Stage-1234_oh'] = df['T-Stage-1234_oh'].apply(ast.literal_eval)
        # df['Smoking>10py_oh'] = df['Smoking>10py_oh'].apply(ast.literal_eval)
        # df['HPV_oh'] = df['HPV_oh'].apply(ast.literal_eval)       
        # df['Age>65_oh'] = df['Age>65_oh'].apply(ast.literal_eval)
        # df['Female_oh'] = df['Female_oh'].apply(ast.literal_eval)
        # df['N-Stage-0123_oh'] = df['N-Stage-0123_oh'].apply(ast.literal_eval)
        # df['T-Stage-1234_oh'] = df['T-Stage-1234_oh'].apply(ast.literal_eval)
        # df['Smoking>10py_oh'] = df['Smoking>10py_oh'].apply(ast.literal_eval)
        # df['HPV_oh'] = df['HPV_oh'].apply(ast.literal_eval)

        self.age = np.array(df['Age>65_oh'].tolist())
        self.sex = np.array(df['Female_oh'].tolist())
        self.n_stage = np.array(df['N-Stage-0123_oh'].tolist())
        self.t_stage = np.array(df['T-Stage-1234_oh'].tolist())
        self.smoking = np.array(df['Smoking>10py_oh'].tolist())
        self.hpv = np.array(df['HPV_oh'].tolist())

        # self.age = torch.from_numpy(np.array(df['Age>65_oh'].tolist()))
        # self.sex = torch.from_numpy(np.array(df['Female_oh'].tolist()))
        # self.n_stage = torch.from_numpy(np.array(df['N-Stage-0123_oh'].tolist()))
        # self.t_stage = torch.from_numpy(np.array(df['T-Stage-1234_oh'].tolist()))
        # self.smoking = torch.from_numpy(np.array(df['Smoking>10py_oh'].tolist()))
        # self.hpv = torch.from_numpy(np.array(df['HPV_oh'].tolist()))

        self.in_channels = in_channels
        self.transform = transform

    def __len__(self):
        print('data size:', len(self.img_dir))
        return len(self.img_dir)

    def __getitem__(self, index):
        assert type(index) is int
        # get img and convert to torch tensor
        img = sitk.ReadImage(self.img_dir[index])
        img = sitk.GetArrayFromImage(img)
        img = img.reshape(self.in_channels, img.shape[0], img.shape[1], img.shape[2])
        img = torch.from_numpy(img).float()
        # make clinical list
        clinical_list = [self.age[index], 
                         self.sex[index], 
                         self.n_stage[index], 
                         self.t_stage[index], 
                         self.smoking[index],
                         self.hpv[index]]
        clinical = torch.from_numpy(np.array(clinical_list))

        return (img, clinical)


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
        img = sitk.ReadImage(img_dir[index])
        img = sitk.GetArrayFromImage(img)
        #img = nib.load(img_dir[index]).get_fdata()
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        time = torch.from_numpy(self.df['time'].to_numpy())
        event = torch.from_numpy(self.df['event'].to_numpy())
        if self.transform:
            img = self.transform(img)
        img = torch.from_numpy(img).float()
        return img, (time[index], event[index])


class DatasetPred(Dataset):
    """Dataset class for CoxPH model
    """
    def __init__(self, df, transform, in_channels):
        self.img_dir = df['img_dir'].to_list()
        #self.transform = transform
        self.transform = False
        self.in_channels = in_channels

    def __len__(self):
        print('data size:', len(self.img_dir))
        return len(self.img_dir)

    def __getitem__(self, index):
        assert type(index) is int
        #img = nib.load(self.img_dir[index])
        #arr = img.get_fdata()
        img = sitk.ReadImage(self.img_dir[index])
        arr = sitk.GetArrayFromImage(img)
        #img = arr.reshape(1, arr.shape[0], arr.shape[1], arr.shape[2])
        img = arr.reshape(self.in_channels, arr.shape[0], arr.shape[1], arr.shape[2])
        # choose image channel
        #if self.channel == 1:
        #    img = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        #elif self.channel == 3:
        #    img = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        #img = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        #img = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        # convert numpy array to torch tensor
        # data and label transform
        if self.transform:
            img = self.transform(img)
        img = torch.from_numpy(img).float()
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
        arr = img.get_fdata()
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
        img = torch.from_numpy(img).float()
        return img, (time[index], event[index], rank_mat)


# class DatasetPCHazard():
#     """
#     Dataset class for PCHazard model
#     Includes image and lables for training and tuning dataloader
#     """
#     def __init__(self, data, idx_duration, event, t_frac):
#         self.data = data
#         self.idx_duration, self.event, self.t_frac = tt.tuplefy(idx_duration, event, t_frac).to_tensor()

#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self, index):
#         img = self.data[index]
#         img = torch.from_numpy(img).float()
#         return img, (self.idx_duration[index], self.event[index], self.t_frac[index])


class DatasetPCHazard(Dataset):
    def __init__(self, df, transform, in_channels):
        self.df = df
        self.img_dir = df['img_dir'].to_list()
        self.time = torch.from_numpy(df['time'].to_numpy())
        self.event = torch.from_numpy(df['event'].to_numpy())
        self.t_frac = torch.from_numpy(df['t_frac'].to_numpy())
        self.transform = transform
        self.in_channels = in_channels
        #self.transform = False

    def __len__(self):
        return self.event.shape[0]

    def __getitem__(self, index):
        assert type(index) is int
        img = sitk.ReadImage(self.img_dir[index])
        img = sitk.GetArrayFromImage(img)
        img = img.reshape(self.in_channels, img.shape[0], img.shape[1], img.shape[2])
        if self.transform:
            img = self.transform(img)
        # need to convert np array to torch tensor before output to tuples
        img = torch.from_numpy(img).float()
        return img, (self.time[index], self.event[index], self.t_frac[index])
