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
from get_data.get_dataset import get_dataset
from custom_dataset import collate_fn, DatasetPCHazard, Dataset0, DatasetPred, DatasetDeepHit, DatasetCoxPH


def data_prep(pro_data_dir, batch_size, cox, num_durations, task,
              tumor_type, input_data_type, i_kfold):
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
    fns_tr, fns_val, fn_ts = get_dataset(tumor_type, input_data_type)
    fn_tr = fns_tr[i_kfold]
    fn_val = fns_val[i_kfold]
    df_tr = pd.read_csv(pro_data_dir + '/' + fn_train)
    #df_train = df_train_.sample(frac=0.9, random_state=200)
    #df_tune = df_train_.drop(df_train.index)
    df_val = pd.read_csv(pro_data_dir + '/' + fn_val)
    df_ts = pd.read_csv(pro_data_dir + '/' + fn_ts)
    print('df_train shape:', df_tr.shape)
    #print('df_tune shape:', df_tune.shape)
    print('df_val shape:', df_val.shape)
    print('df_test shape:', df_ts.shape)

    if cox == 'PCHazard':
        labtrans = PCHazard.label_transform(num_durations)
    elif cox == 'LogisticHazard':
        labtrans = LogisticHazard.label_transform(num_durations)
    elif cox == 'DeepHit':
        labtrans = DeepHitSingle.label_transform(num_durations)
    else:
        print('choose other cox models!')
    
    """
    Outcome prediction model: 
        1) overall survival: death_time, death_event; 
        2) local/regional control: lr_time, lr_event;
        3) distant control: ds_time, ds_event;
    """
    if task == 'overall_survival':
        get_target = lambda df: (df['death_time'].values, df['death_event'].values)
    elif task == 'local_control':
        get_target = lambda df: (df['lr_time'].values, df['lr_event'].values)
    elif task == 'distant_control':
        get_target = lambda df: (df['ds_time'].values, df['ds_event'].values)
    # label transform 
    print('df_test:', df_test['lr_event'].value_counts())
    print('df_val:', df_val['lr_event'].value_counts())
    print('df_tune:', df_tune['lr_event'].value_counts())
    print('df_train:', df_train['lr_event'].value_counts())
    dfs = []
    for df in [df_tr, df_val, df_ts]:
        if cox in ['PCHazard', 'LogisticHazard', 'DeepHit']:
            y = labtrans.fit_transform(*get_target(df))
            out_features = labtrans.out_features
            duration_index = labtrans.cuts
            #print('y[0]:', y[0])
            #print('y[1]:', y[1])
        elif cox == 'CoxPH':
            y = get_target(df)
            out_features = 1
        # create new df
        df['time'], df['event'] = [y[0], y[1]]
        #print('time:', df['time'])
        #print('event:', df['event'])
        dfs.append(df)
    df_tr = dfs[0]
    df_val = dfs[1]
    df_ts = dfs[2]
    #print('out_features:', out_features)
    #print('duration_index:', duration_index)
    #print(labtrans.cuts[y_train[0]])
    # save duration index for train and evaluate steps
    if not os.path.exists(pro_data_dir + '/duration_index.npy'):
        np.save(pro_data_dir + '/duration_index.npy', duration_index)
     
    return df_train, df_val, df_test


def data_loader_transform(pro_data_dir, batch_size, cox, num_durations, 
                          task, tumor_type, input_data_type, i_kfold):
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
    val_transforms = Compose([
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
    df_tr, df_val, df_ts = data_prep(
        pro_data_dir,
        batch_size,
        cox,
        num_durations,
        task,
        tumor_type,
        input_data_type, 
        i_kfold)

    # train, tune, val dataset
    if cox in ['CoxPH']:
        ds_tr = DatasetCoxPH(df=df_tr, transform=tr_transforms)
        ds_val = DatasetCoxPH(df=df_val, transform=val_transforms)
    elif cox in ['PCHazard', 'LogisticHazard']:
        ds_tr = Dataset0(df=df_tr, transform=tr_transforms)
        ds_val = Dataset0(df=df_val, transform=val_transforms)
    elif cox in ['DeepHit']:
        ds_tr = DatasetDeepHit(df=df_tr, transform=tr_transforms)
        ds_val = DatasetDeepHit(df=df_val, transform=val_transforms)
    else:
        print('choose another cox model!')
    ds_baseline = DatasetCoxPH(df=df_tr[0:50], transform=val_transforms)
    ds_val_cb = DatasetPred(df_val, transform=val_transforms)
    ds_val = DatasetPred(df_val, transform=val_transforms)
    ds_ts = DatasetPred(df_ts, transform=ts_transforms)
    
    # check data
    check_data = False
    if check_data:
        check_loader = DataLoader(ds_train, batch_size=1, collate_fn=collate_fn)
        check_data = first(check_loader)
        print('\ncheck image and lable shape:', check_data.size)

    # batch data loader using Pycox and TorchTuple
    dl_tr = DataLoader(dataset=ds_train, shuffle=True, batch_size, collate_fn)
    #dl_tune = DataLoader(dataset=ds_tune, shuffle=True, batch_size, collate_fn)
    # tuning set dataloader for c-index callback
    dl_val_cb = DataLoader(dataset=ds_val_cb, shuffle=False, batch_size)
    dl_val = DataLoader(dataset=ds_val, shuffle=False, batch_size)
    dl_ts = DataLoader(dataset=ds_test, shuffle=False, batch_size)
    dl_baseline = DataLoader(dataset=ds_baseline, shuffle=False, batch_size)
    #print('successfully created data loaders!')

    return dl_train, dl_val, dl_test, dl_val_cb, df_val, dl_baseline







