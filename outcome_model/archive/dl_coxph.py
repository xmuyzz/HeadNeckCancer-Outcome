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
from torchvision.utils import save_image
import torchvision.models as models
from pycox.datasets import metabric
from pycox.evaluation import EvalSurv
from pycox.models import PCHazard, CoxPH, LogisticHazard, DeepHitSingle
from pycox.utils import kaplan_meier
from pycox.models.data import pair_rank_mat, _pair_rank_mat
import monai
from monai.utils import first
from monai.transforms import (AddChannel, AsChannelFirst, EnsureChannelFirst, RepeatChannel,
    ToTensor, RemoveRepeatedChannel, EnsureType, Compose, CropForeground, LoadImage,
    Orientation, RandSpatialCrop, Spacing, Resize, ScaleIntensity, RandRotate, RandZoom,
    RandGaussianNoise, RandGaussianSharpen, RandGaussianSmooth, RandFlip, Rotate90, RandRotate90, 
    EnsureType, RandAffine)
#from get_data.get_dataset import get_dataset
from custom_dataset import collate_fn, DatasetPCHazard, Dataset0, DatasetPred, DatasetDeepHit, DatasetCoxPH


def get_df(csv_dir, task, tumor_type):
    """
    Prerpocess image and lable for DataLoader
    Args:
        batch_size {int} -- batch size for data loading;
        _cox_model {str} -- cox model name;
        number_durations {int} -- number to discretize survival time;
    Returns:
        Dataloaders for train, tune and val datasets;
    """    
    ## load train and val dataset
    if tumor_type == 'pn':
        tr_fn = 'tr_img_label_pn.csv'
        va_fn = 'va_img_label_pn.csv'
        ts_fn = 'ts_img_label_pn.csv'
    if tumor_type == 'p':
        tr_fn = 'tr_img_label_p.csv'
        va_fn = 'va_img_label_p.csv'
        ts_fn = 'ts_img_label_p.csv'
    if tumor_type == 'n':
        tr_fn = 'tr_img_label_n.csv'
        va_fn = 'va_img_label_n.csv'
        ts_fn = 'ts_img_label_n.csv'
    df_tr = pd.read_csv(csv_dir + '/' + tr_fn)
    df_va = pd.read_csv(csv_dir + '/' + va_fn)
    df_ts = pd.read_csv(csv_dir + '/' + ts_fn)
    print('df_tr shape:', df_tr.shape)
    print('df_va shape:', df_va.shape)
    print('df_ts shape:', df_ts.shape)
    
    dfs = []
    for df in [df_tr, df_va, df_ts]:
        if task == 'rfs':
            df = df[df['rfs_time'].notna()]
            times = df['rfs_time'].values
            events = df['rfs_event'].values
        elif task == 'os':
            df = df[df['death_time'].notna()]
            times = df['death_time'].values
            events = df['death_event'].values
        elif task == 'lc':
            df = df[df['lr_time'].notna()]
            times = df['lr_time'].values
            events = df['lr_event'].values
        elif task == 'dc':
            df = df[df['ds_time'].notna()]
            times = df['ds_time'].values
            events = df['ds_event'].values
        print('df_ts:', df_ts['lr_event'].value_counts())
        print('df_va:', df_va['lr_event'].value_counts())
        print('df_tr:', df_tr['lr_event'].value_counts())

        #out_features = 1
        df['time'], df['event'] = [times, events]
        #print('time:', df['time'])
        #print('event:', df['event'])
        dfs.append(df)
    df_tr = dfs[0]
    df_va = dfs[1]
    df_ts = dfs[2]
    #print('out_features:', out_features)
     
    return df_tr, df_va, df_ts


def dl_coxph(csv_dir, batch_size, task, tumor_type):
    """
    DataLoader with image augmentation using Pycox/TorchTuple and Monai packages.
    Args:
        _cox_model {str} -- cox model name;
        _outcome_model {str} -- outcome model name ('os|lrc|dc');
    Retunrs:
        data loader with real time augmentation;
    """
    # image transforma with MONAI
    tr_transforms = Compose([
        #EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        #Resized(spatial_size=(96, 96, 96)),
        RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
        RandGaussianSharpen(),
        RandGaussianSmooth(),
        RandAffine(prob=0.5, translate_range=10),
        RandFlip(prob=0.5, spatial_axis=None),
        RandRotate(prob=0.5, range_x=5, range_y=5, range_z=5),
        EnsureType(data_type='tensor')])
    va_transforms = Compose([
        #AddChannel,
        #EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        EnsureType(data_type='tensor')])
    ts_transforms = Compose([
        #AddChannel,
        #EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        EnsureType(data_type='tensor')])
    baseline_transforms = Compose([
        #AddChannel,
        #EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        EnsureType(data_type='tensor')])
    
    # get dataset for train, tune and val
    df_tr, df_va, df_ts = get_df(csv_dir, task, tumor_type)

    # train, val, test, callback (cb), cox baseline (bl) dataset
    ds_tr = DatasetCoxPH(df=df_tr, transform=tr_transforms)
    ds_va = DatasetCoxPH(df=df_va, transform=va_transforms)
    ds_ts = DatasetPred(df_ts, transform=ts_transforms)
    ds_bl = DatasetCoxPH(df=df_tr[0:50], transform=va_transforms)
    ds_cb = DatasetPred(df_va, transform=va_transforms)
    
    # check data
    check_data = False
    if check_data:
        check_loader = DataLoader(ds_tr, batch_size=1, collate_fn=collate_fn)
        check_data = first(check_loader)
        print('\ncheck image and lable shape:', check_data.size)

    # batch data loader using Pycox and TorchTuple
    # train, val, test, callback (cb), cox baseline (bl) dataset
    dl_tr = DataLoader(dataset=ds_tr, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    dl_va = DataLoader(dataset=ds_va, shuffle=False, batch_size=batch_size)
    dl_ts = DataLoader(dataset=ds_ts, shuffle=False, batch_size=batch_size)
    dl_cb = DataLoader(dataset=ds_cb, shuffle=False, batch_size=batch_size)
    dl_bl = DataLoader(dataset=ds_bl, shuffle=False, batch_size=batch_size)
    print('/nsuccessfully created data loaders!')

    return dl_tr, dl_va, dl_ts, dl_cb, dl_bl, df_va







