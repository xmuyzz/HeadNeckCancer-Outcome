import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
#import nibabel as nib
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
from pycox.models import PCHazard, CoxPH, LogisticHazard, DeepHitSingle, MTLR
from pycox.utils import kaplan_meier
from pycox.models.data import pair_rank_mat, _pair_rank_mat
import monai
from monai.utils import first
from monai.transforms import (AddChannel, AsChannelFirst, EnsureChannelFirst, RepeatChannel,
    ToTensor, RemoveRepeatedChannel, EnsureType, Compose, CropForeground, LoadImage,
    Orientation, RandSpatialCrop, Spacing, Resize, ScaleIntensity, RandRotate, RandZoom,
    RandGaussianNoise, RandGaussianSharpen, RandGaussianSmooth, RandFlip, Rotate90, RandRotate90, 
    EnsureType, RandAffine, AdjustContrast)
#from get_data.get_dataset import get_dataset
from custom_dataset import (collate_fn, DatasetPCHazard, Dataset0, DatasetPred, DatasetDeepHit, 
    DatasetCoxPH, Dataset_Concat_Tr, Dataset_Concat_Ts)



def get_df(data_dir, metric_dir, surv_type, img_size, img_type, tumor_type, cox, num_durations):
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
    #csv_dir = data_dir + '/data/' + img_size + '_' + img_type 
    csv_dir = data_dir + '/' + img_size + '_' + img_type + '/' + surv_type
    tr_fn = 'tr_img_label_' + tumor_type + '.csv'
    va_fn = 'va_img_label_' + tumor_type + '.csv'
    df_tr = pd.read_csv(csv_dir + '/' + tr_fn)
    df_va = pd.read_csv(csv_dir + '/' + va_fn)
    print('df_tr shape:', df_tr.shape)
    print('df_va shape:', df_va.shape)
    
    # label transformation for parametric model
    if cox == 'PCHazard':
        labtrans = PCHazard.label_transform(num_durations)
    elif cox == 'LogisticHazard':
        labtrans = LogisticHazard.label_transform(num_durations)
    elif cox == 'MTLR':
        labtrans = MTLR.label_transform(num_durations)  
    elif cox == 'DeepHit':
        labtrans = DeepHitSingle.label_transform(num_durations)
    elif cox == 'CoxPH':
        print('no need for label transformation!')
    else:
        print('choose other cox models!')
        
    get_target = lambda df: (df[surv_type + '_time'].values, df[surv_type + '_event'].values)
    
    if cox in ['LogisticHazard', 'DeepHit', 'MTLR']:
        #print(df)
        y_tr = labtrans.fit_transform(*get_target(df_tr))
        y_va = labtrans.transform(*get_target(df_va))
        print('y_va:', y_va)
        out_features = labtrans.out_features
        duration_index = labtrans.cuts
        np.save(metric_dir + '/duration_index.npy', duration_index)
    elif cox == 'PCHazard':
        #print(df)
        y_tr = labtrans.fit_transform(*get_target(df_tr))
        y_va = labtrans.transform(*get_target(df_va))
        print('y_va:', y_va)
        out_features = labtrans.out_features
        duration_index = labtrans.cuts
        np.save(metric_dir + '/duration_index.npy', duration_index)   
        df_tr['time'], df_tr['event'], df_tr['t_frac'] = [y_tr[0], y_tr[1], y_tr[2]]
        df_va['time'], df_va['event'], df_va['t_frac'] = [y_va[0], y_va[1], y_va[2]]    
        # df_tr['time'], df_tr['event'] = [y_tr[0], y_tr[1]]
        # df_va['time'], df_va['event'] = [y_va[0], y_va[1]]  
        # df_tr['t_frac'] = y_tr[2]
        # df_va['t_frac'] = y_va[2]
    elif cox == 'CoxPH':
        y_tr = get_target(df_tr)
        y_va = get_target(df_va)
        out_features = 1
    df_tr['time'], df_tr['event'] = [y_tr[0], y_tr[1]]
    df_va['time'], df_va['event'] = [y_va[0], y_va[1]]
    #print('out_features:', out_features)
    #print('duration_index:', duration_index)
    #print(labtrans.cuts[y_train[0]])
    
    return df_tr, df_va


def dl_train(data_dir, metric_dir, batch_size, cnn_name, cox, num_durations, surv_type, img_size, img_type, tumor_type, 
             rot_prob, gauss_prob, flip_prob, in_channels):
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
        RandGaussianNoise(prob=gauss_prob, mean=0.0, std=0.1),
        #RandGaussianSharpen(),
        #RandGaussianSmooth(),
        #RandAffine(prob=0.5, translate_range=10),
        #RandFlip(prob=flip_prob, spatial_axis=1),
        #Orientation(axcodes='RPI'),
        RandFlip(prob=flip_prob, spatial_axis=1),
        #AdjustContrast(gamma=1),
        RandRotate(prob=rot_prob, range_x=10, range_y=10, range_z=10),
        EnsureType(data_type='numpy')])
    va_transforms = Compose([
        #AddChannel,
        #EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        EnsureType(data_type='numpy')])

    df_tr, df_va = get_df(data_dir, metric_dir, surv_type, img_size, img_type, tumor_type, cox, num_durations)
    #print('df_va_time:', df_va['time'])
    #print('df_va_rfs_time:', df_va['rfs_time'])
    #print('df_va_event:', df_va['event'].to_list())
    
    # train, tune, val dataset
    if cox == 'CoxPH':
        #ds_tr = DatasetCoxPH(df=df_tr, transform=tr_transforms)
        #ds_va = DatasetCoxPH(df=df_va, transform=va_transforms)
        ds_tr = Dataset0(df=df_tr, transform=None)
        ds_va = Dataset0(df=df_va, transform=None)
    elif cox in ['LogisticHazard', 'MTLR']:
        if cnn_name == 'DenseNet_Concat':
            ds_tr = Dataset_Concat_Tr(df=df_tr, transform=tr_transforms, in_channels=in_channels)
            ds_va = Dataset_Concat_Tr(df=df_va, transform=va_transforms, in_channels=in_channels)
        else:
            ds_tr = Dataset0(df=df_tr, transform=tr_transforms, in_channels=in_channels)
            ds_va = Dataset0(df=df_va, transform=va_transforms, in_channels=in_channels)
        print('ds_tr:', ds_tr)
    elif cox == 'DeepHit':
        ds_tr = DatasetDeepHit(df=df_tr, transform=tr_transforms)
        ds_va = DatasetDeepHit(df=df_va, transform=va_transforms)
    elif cox == 'PCHazard':
        ds_tr = DatasetPCHazard(df=df_tr, transform=tr_transforms, in_channels=in_channels)
        ds_va = DatasetPCHazard(df=df_va, transform=va_transforms, in_channels=in_channels)
    else:
        print('choose another cox model!')
    #ds_bl = DatasetCoxPH(df=df_tr[0:50], transform=va_transforms)
    ds_bl = Dataset0(df=df_tr[0:50], transform=va_transforms, in_channels=in_channels)

    if cnn_name == 'DenseNet_Concat':
        ds_cb = Dataset_Concat_Ts(df_va, transform=va_transforms, in_channels=in_channels)
    else:
        ds_cb = DatasetPred(df_va, transform=va_transforms, in_channels=in_channels)
    #ds_va = DatasetPred(df_va, transform=va_transforms)
    
    # check data
    check_data = True
    if check_data:
        #check_loader = DataLoader(ds_tr, batch_size=10, collate_fn=collate_fn)
        #check_data = first(check_loader)
        dl_tr = DataLoader(ds_tr, batch_size=10, collate_fn=collate_fn)
        batch = next(iter(dl_tr))
        print('\ncheck image and lable shape:', batch.shapes())
        #print('data type:', batch.dtypes())
        #dl_cb = DataLoader(ds_cb, batch_size=10)
        #batch = next(iter(dl_cb))
        #print('\ncheck image and lable shape:', batch.size())
        #print('data type:', batch.dtypes())

    # batch data loader using Pycox and TorchTuple
    dl_tr = DataLoader(dataset=ds_tr, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
    dl_cb = DataLoader(dataset=ds_cb, shuffle=False, batch_size=batch_size)
    dl_va = DataLoader(dataset=ds_va, shuffle=False, batch_size=batch_size)
    dl_bl = DataLoader(dataset=ds_bl, shuffle=False, batch_size=batch_size)

    #batch = next(iter(dl_tr))
    #print(batch.shapes())
    
    print('\nsuccessfully created data loaders!')

    return dl_tr, dl_va, dl_cb, dl_bl, df_va



