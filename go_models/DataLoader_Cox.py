import matplotlib.pyplot as plt
import nibabel as nib
#from tqdm import tqd
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn_pandas import DataFrameMapper
from PIL import Image
import gc
import torchtuples as tt
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader
#from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.models as models
from pycox.datasets import metabric
from pycox.evaluation import EvalSurv
from pycox.models import (
    PCHazard,
    LogisticHazard,
    CoxPH,
    DeepHitSingle
    )
from pycox.utils import kaplan_meier
import monai
from monai.utils import first
from monai.data import DataLoader, Dataset
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    Resized,
    ToTensord,
    ScaleIntensityd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    Flipd,
    RandRotate90d,
    EnsureTyped,
    EnsureType,
    RandAffined,
    )
#from models import ResNet
from models.cnn import cnn3d
#from go_models.data_loader import DataLoader



def DataLoader_Cox(proj_dir, batch_size=8, _cox_model='LogisticHazard', num_durations=10):

    """
    Create data augmentation and dataloder for image and lable inputs.
    
    Arguments:
        proj_dir {path} -- project path for preprocessed data.

    Keyword arguments:
        batch_size {int} -- batch size for data loading.

    Return:
        train_DataLoader, tune_DataLoader, val_DataLoader
    """


    torch.cuda.empty_cache()

    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(pro_data_dir): os.mkdir(pro_data_dir)

    ## load train and val dataset
    df_train_ = pd.read_csv(os.path.join(pro_data_dir, 'df_train0.csv'))
    df_train = df_train_.sample(frac=0.8, random_state=200)
    df_tune = df_train_.drop(df_train.index)
    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    print('df_train shape:', df_train.shape)
    print('df_tune shape:', df_tune.shape)
    print('df_val shape:', df_val.shape)


    # get train and val label (events, duration time)
    #-------------------------------------------------
    """
    The LogisticHazard is a discrete-time method, meaning it requires discretization of the 
    event times to be applied to continuous-time data.
    We let num_durations define the size of this (equidistant) discretization grid, meaning our
    network will have num_durations output nodes.
    """

    if _cox_model == 'PCHazard':
        labtrans = PCHazard.label_transform(num_durations)
    elif _cox_model == 'LogisticHazard':
        labtrans = LogisticHazard.label_transform(num_durations)
    elif _cox_model == 'DeepHit':
        labtrans = DeepHitSingle.label_transform(num_durations)

    get_target = lambda df: (df['sur_duration'].values, df['survival'].values)
    if _cox_model in ['PCHazard', 'LogisticHazard', 'DeepHit']:
        y_train = labtrans.fit_transform(*get_target(df_train))
        y_tune = labtrans.transform(*get_target(df_tune))
        y_val = labtrans.transform(*get_target(df_val))
        out_features = labtrans.out_features
        duration_index = labtrans.cuts
    elif _cox_model == 'CoxPH':
        y_train = get_target(df_train)
        y_tune = get_target(df_tune)
        y_val = get_target(df_val)
        out_features = 1
        duration_index = labtrans.cuts
    
    print('y_train:', y_train)
    print('out_features:', out_features)
    print('duration_index:', duration_index)
    #print(labtrans.cuts[y_train[0]])
    # save duration index for train and evaluate steps
    np.save(os.path.join(pro_data_dir, 'duration_index.npy'), duration_index)
    
    # create dataset for data loader
    #--------------------------------
    datas = []
    for df in [df_train, df_tune, df_val]:
        imgs = df['img_dir'].to_list()
        labels = [(event, time) for event, time in zip(y_train[1], y_train[0])]
        print(labels[0:5])
        labels = np.array(labels, dtype=np.int64)
        #print(labels[0:5])
        data = [{'image': img, 'label': label} for img, label in zip(imgs, labels)]
        datas.append(data)
    train_data = datas[0]
    tune_data = datas[1]
    val_data = datas[2]
    print('train_data:', train_data[0:5])

    # Define transforms for image
    #--------------------------------
    train_transforms = Compose([
        LoadImaged(keys=['image']),
        AddChanneld(keys=['image']),
        ScaleIntensityd(keys=['image']),
        #Resized(keys=['img'], spatial_size=(96, 96, 96)),
        RandRotated(keys=['image'], prob=0.8, range_x=10.0),
        RandGaussianNoised(keys='image', prob=0.5),
        RandAffined(keys=['image'], prob=0.5, translate_range=10),
        EnsureTyped(keys=['image']),
        ToTensord(keys=['image'])
        ])

    tune_transforms = Compose([
        LoadImaged(keys=['image']),
        AddChanneld(keys=['image']),
        ScaleIntensityd(keys=['image']),
        #Resized(keys=['img'], spatial_size=(96, 96, 96)),
        EnsureTyped(keys=['image']),
        ToTensord(keys=['image']),
        ])
    
    val_transforms = Compose([
        LoadImaged(keys=['image']),
        AddChanneld(keys=['image']),
        ScaleIntensityd(keys=['image']),
        #Resized(keys=['img'], spatial_size=(96, 96, 96)),
        EnsureTyped(keys=['image']),
        ToTensord(keys=['image']),
        ])

    #post_pred = Compose([EnsureType(), Activations(softmax=True)])
    #post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

    # Define dataset, data loader
    check_ds = Dataset(data=train_data, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    print(check_data['image'].shape, check_data['label'])

    # create a training data loader
    train_ds = Dataset(data=train_data, transform=train_transforms)
    dl_train = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        #num_workers=4, 
        pin_memory=torch.cuda.is_available())

    # create a tuning data loader
    tune_ds = Dataset(data=tune_data, transform=tune_transforms)
    dl_tune = DataLoader(
        tune_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        #num_workers=4,
        pin_memory=torch.cuda.is_available())
    
    # create a validation data loader
    val_ds = Dataset(data=val_data, transform=val_transforms)
    dl_val = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        #num_workers=4, 
        pin_memory=torch.cuda.is_available())

    print('successfully created data loader!')

    return dl_train, dl_tune, dl_val






#original_transforms = Compose(
#    [
#        LoadImaged(keys=["image", "label"]),
#        AddChanneld(keys=["image", "label"]),
#        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
#        Orientationd(keys=["image", "label"], axcodes="RAS"),
#        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True,),
#        ToTensord(keys=["image", "label"]),
#    ]
#)
#
#generat_transforms = Compose(
#    [
#        LoadImaged(keys=["image", "label"]),
#        AddChanneld(keys=["image", "label"]),
#        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
#        Orientationd(keys=["image", "label"], axcodes="RAS"),
#        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True,),
#        RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10),
#        RandRotated(keys=['image', 'label'], prob=0.5, range_x=10.0),
#        RandGaussianNoised(keys='image', prob=0.5),
#        ToTensord(keys=["image", "label"]),
#    ]
#)

