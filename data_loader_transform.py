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
import monai
from monai.utils import first
from monai.transforms import (AddChannel, AsChannelFirst, EnsureChannelFirst, RepeatChannel,
    ToTensor, RemoveRepeatedChannel, EnsureType, Compose, CropForeground, LoadImage,
    Orientation, RandSpatialCrop, Spacing, Resize, ScaleIntensity, RandRotate, RandZoom,
    RandGaussianNoise, RandFlip, Rotate90, RandRotate90, EnsureType, RandAffine)


def get_dataset(tumor_type, input_data_type):
    if tumor_type == 'primary_node':
        if input_data_type == 'masked_img':
            fns_train = [
                'df_pn_masked_train0.csv',
                'df_pn_masked_train1.csv',
                'df_pn_masked_train2.csv',
                'df_pn_masked_train3.csv',
                'df_pn_masked_train4.csv']
            fns_val = [
                'df_pn_masked_val0.csv',
                'df_pn_masked_val1.csv',
                'df_pn_masked_val2.csv',
                'df_pn_masked_val3.csv',
                'df_pn_masked_val4.csv']
            fn_test = 'df_pn_masked_test.csv'
        elif input_data_type == 'raw_img':
            fns_train = [
                'df_pn_raw_train0.csv',
                'df_pn_raw_train1.csv',
                'df_pn_raw_train2.csv',
                'df_pn_raw_train3.csv',
                'df_pn_raw_train4.csv']
            fns_val = [
                'df_pn_raw_val0.csv',
                'df_pn_raw_val1.csv',
                'df_pn_raw_val2.csv',
                'df_pn_raw_val3.csv',
                'df_pn_raw_val4.csv']
            fn_test = 'df_pn_raw_test.csv'
    if tumor_type == 'primary':
        if input_data_type == 'masked_img':
            fns_train = [
                'df_p_masked_train0.csv',
                'df_p_masked_train1.csv',
                'df_p_masked_train2.csv',
                'df_p_masked_train3.csv',
                'df_p_masked_train4.csv']
            fns_val = [
                'df_p_masked_val0.csv',
                'df_p_masked_val1.csv',
                'df_p_masked_val2.csv',
                'df_p_masked_val3.csv',
                'df_p_masked_val4.csv']
            fn_test = 'df_p_maksed_test.csv'
        elif input_data_type == 'raw_img':
            fns_train = [
                'df_p_raw_train0.csv',
                'df_p_raw_train1.csv',
                'df_p_raw_train2.csv',
                'df_p_raw_train3.csv',
                'df_p_raw_train4.csv']
            fns_val = [
                'df_p_raw_val0.csv',
                'df_p_raw_val1.csv',
                'df_p_raw_val2.csv',
                'df_p_raw_val3.csv',
                'df_p_raw_val4.csv']
            fn_test = 'df_p_raw_test.csv'
    if tumor_type == 'node':
        if input_data_type == 'masked_img':
            fns_train = [
                'df_n_masked_train0.csv',
                'df_n_masked_train1.csv',
                'df_n_masked_train2.csv',
                'df_n_masked_train3.csv',
                'df_n_masked_train4.csv']
            fns_val = [
                'df_n_masked_val0.csv',
                'df_n_masked_val1.csv',
                'df_n_masked_val2.csv',
                'df_n_masked_val3.csv',
                'df_n_masked_val4.csv']
            fn_test = 'df_n_masked_test.csv'
        elif input_data_type == 'raw_img':
            fns_train = [
                'df_n_raw_train0.csv',
                'df_n_raw_train1.csv',
                'df_n_raw_train2.csv',
                'df_n_raw_train3.csv',
                'df_n_raw_train4.csv']
            fns_val = [
                'df_n_raw_val0.csv',
                'df_n_raw_val1.csv',
                'df_n_raw_val2.csv',
                'df_n_raw_val3.csv',
                'df_n_raw_val4.csv']
            fn_test = 'df_n_raw_test.csv'

    return fns_train, fns_val, fn_test


def data_prep(pro_data_dir, batch_size, _cox_model, num_durations, _outcome_model,
              tumor_type, input_data_type, i_kfold):
    
    """
    Prerpocess image and lable for DataLoader
    
    Args:
        batch_size {int} -- batch size for data loading;
        _cox_model {str} -- cox model name;
    
    Keyword args:
        number_durations {int} -- number to discretize survival time;

    Returns:
        Dataloaders for train, tune and val datasets;
    """    
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    ## load train and val dataset
    fns_train, fns_val, fn_test = get_dataset(
            tumor_type=tumor_type, 
            input_data_type=input_data_type)
    fn_train = fns_train[i_kfold]
    fn_val = fns_val[i_kfold]
    df_train_ = pd.read_csv(os.path.join(pro_data_dir, fn_train))
    df_train = df_train_.sample(frac=0.9, random_state=200)
    df_tune = df_train_.drop(df_train.index)
    df_val = pd.read_csv(os.path.join(pro_data_dir, fn_val))
    df_test = pd.read_csv(os.path.join(pro_data_dir, fn_test))
    print('df_train shape:', df_train.shape)
    print('df_tune shape:', df_tune.shape)
    print('df_val shape:', df_val.shape)
    print('df_test shape:', df_test.shape)

    """
    The LogisticHazard is a discrete-time method, meaning it requires discretization 
    of the event times to be applied to continuous-time data. 
    We let num_durations define the size of this (equidistant) discretization grid, meaning our 
    network will have num_durations output nodes.
    """
    if _cox_model == 'PCHazard':
        labtrans = PCHazard.label_transform(num_durations)
    elif _cox_model == 'LogisticHazard':
        labtrans = LogisticHazard.label_transform(num_durations)
    elif _cox_model == 'DeepHit':
        labtrans = DeepHitSingle.label_transform(num_durations)
    else:
        print('choose other cox models!')
    
    """
    Outcome prediction model: 
        1) overall survival: death_time, death_event; 
        2) local/regional control: lr_time, lr_event;
        3) distant control: ds_time, ds_event;
    """
    if _outcome_model == 'overall_survival':
        get_target = lambda df: (df['death_time'].values, df['death_event'].values)
    elif _outcome_model == 'local_control':
        get_target = lambda df: (df['lr_time'].values, df['lr_event'].values)
    elif _outcome_model == 'distant_control':
        get_target = lambda df: (df['ds_time'].values, df['ds_event'].values)
    # label transform 
    dfs = []
    for df in [df_train, df_tune, df_val, df_test]:
        if _cox_model in ['PCHazard', 'LogisticHazard', 'DeepHit']:
            #labtrans = PCHazard.label_transform(num_durations)
            y = labtrans.fit_transform(*get_target(df))
            #y = labtrans.transform(*get_target(df))
            out_features = labtrans.out_features
            duration_index = labtrans.cuts
            #print('y[0]:', y[0])
            #print('y[1]:', y[1])
        elif _cox_model == 'CoxPH':
            y = get_target(df)
            out_features = 1
            duration_index = labtrans.cuts
        # create new df
        df['time'], df['event'] = [y[0], y[1]]
        dfs.append(df)
    df_train = dfs[0]
    df_tune = dfs[1]
    df_val = dfs[2]
    df_test = dfs[3]
    #print('out_features:', out_features)
    #print('duration_index:', duration_index)
    #print(labtrans.cuts[y_train[0]])
    # save duration index for train and evaluate steps
    if not os.path.exists(os.path.join(pro_data_dir, 'duration_index.npy')):
        np.save(os.path.join(pro_data_dir, 'duration_index.npy'), duration_index)
     
    return df_train, df_tune, df_val, df_test


def data_loader_transform(pro_data_dir, batch_size, _cox_model, num_durations, 
                          _outcome_model, tumor_type, input_data_type, i_kfold):
    
    """
    DataLoader with image augmentation using Pycox/TorchTuple and Monai packages.

    Args:
        _cox_model {str} -- cox model name;
        _outcome_model {str} -- outcome model name ('os|lrc|dc');

    Retunrs:
        data loader with real time augmentation;
    """

    # image transforma with MONAI
    train_transforms = Compose([
        #EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        #Resized(spatial_size=(96, 96, 96)),
        RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
        RandAffine(prob=0.5, translate_range=10),
        RandFlip(prob=0.1, spatial_axis=2),
        #RandRotate90(prob=0.1, max_k=3, spatial_axes=(0, 1)),
        EnsureType(data_type='tensor')
        ])
    tune_transforms = Compose([
        #AddChannel,
        #EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        #RandGaussianNoise(prob=0.1, mean=0.0, std=0.1),
        #RandAffine(prob=0.5, translate_range=10),
        EnsureType(data_type='tensor')
        ])
    val_transforms = Compose([
        #AddChannel,
        #EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        EnsureType(data_type='tensor')
        ])
    test_transforms = Compose([
        #AddChannel,
        #EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        EnsureType(data_type='tensor')
        ])

    # get dataset for train, tune and val
    df_train, df_tune, df_val, df_test = data_prep(
        pro_data_dir=pro_data_dir,
        batch_size=batch_size,
        _cox_model=_cox_model,
        num_durations=num_durations,
        _outcome_model=_outcome_model,
        tumor_type=tumor_type,
        input_data_type=input_data_type, 
        i_kfold=i_kfold)

    # train, tune, val dataset
    if _cox_model == 'CoxPH':
        # no need to discrete labels
        dataset_train = Dataset1(x_train, *y_train)
        dataset_tune = Dataset1(x_tune, *y_tune)
        dataset_val = DatasetPred(x_val)
        dataset_val = DatasetPred(x_test)
    elif _cox_model in ['PCHazard', 'LogisticHazard', 'DeepHit']:
        # need to dicrete labels
        dataset_train = Dataset0(df_train, transform=train_transforms)
        dataset_tune = Dataset0(df_tune, transform=tune_transforms)
        dataset_tune_cb = DatasetPred(df_tune, transform=val_transforms)
        dataset_val = DatasetPred(df_val, transform=val_transforms)
        dataset_test = DatasetPred(df_test, transform=val_transforms)
    else:
        print('choose another cox model!')
 
    # check data
    #check_loader = DataLoader(dataset_train, batch_size=1)
    #check_data = first(check_loader)
    #print('\ncheck image and lable shape:', check_data)

    # batch data loader using Pycox and TorchTuple
    dl_train = DataLoader(
        dataset=dataset_train, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn)
    dl_tune = DataLoader(
        dataset=dataset_tune, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn)
    # tuning set dataloader for c-index callback
    dl_tune_cb = DataLoader(
        dataset=dataset_tune_cb,
        batch_size=batch_size,
        shuffle=False)
    dl_val = DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=False)
    dl_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=False)
    #print('successfully created data loaders!')

    return dl_train, dl_tune, dl_val, dl_test, dl_tune_cb, df_tune



def collate_fn(batch):
    """
    Stacks the entries of a nested tuple
    """
    return tt.tuplefy(batch).stack()


class Dataset1():
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
    """
    Dataset class for CoxPH model
    Includes image and lables for training and tuning dataloader
    """
    def __init__(self, df, channel=3, transform=None, target_transform=None):
        self.img_dir = df['img_dir'].to_list()
        #self.time, self.event = tt.tuplefy(
        #    df['sur_duration'].to_numpy(), 
        #    df['survival'].to_numpy()).to_tensor()
        self.time = torch.from_numpy(df['time'].to_numpy())
        self.event = torch.from_numpy(df['event'].to_numpy())
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        #print('event size:', self.event.size(dim=0))
        return self.event.shape[0]

    def __getitem__(self, index):
        if type(index) is not int:
            raise ValueError(f'Need `index` to be `int`. Got {type(index)}.')
        img = nib.load(self.img_dir[index])
        arr = img.get_data()
        # choose image channel
        #img = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        img = arr.reshape(1, arr.shape[0], arr.shape[1], arr.shape[2])
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
        if self.target_transform:
            label = self.target_transform(label)
        return img, (self.time[index], self.event[index])


class DatasetPred(Dataset):

    """
    Dataset class for CoxPH model
    Only include image for validation and test dataloader
    """

    def __init__(self, df, channel=3, transform=None):
        self.img_dir = df['img_dir'].to_list()
        self.transform = transform

    def __len__(self):
        print('data size:', len(self.img_dir))
        return len(self.img_dir)

    def __getitem__(self, index):
        if type(index) is not int:
            raise ValueError(f'Need `index` to be `int`. Got {type(index)}.')
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


