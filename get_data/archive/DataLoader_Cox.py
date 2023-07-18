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



def DataLoader_Cox(proj_dir, batch_size=8, _cox_model='LogisticHazard', num_durations=10):

    """
    Create data augmentation and dataloder for image and lable inputs.
    
    @Args:
        proj_dir {path} -- project path for preprocessed data.
    @Keyword args:
        batch_size {int} -- batch size for data loading.
    @Return:
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


    # label transform
    #-------------------------------------------------
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
    
    print('y_train:\n', y_train)
    print('out_features:\n', out_features)
    print('duration_index:\n', duration_index)
    #print(labtrans.cuts[y_train[0]])
    
    # save duration index for train and evaluate steps
    dur_idx = os.path.join(pro_data_dir, 'duration_index.npy')
    if not os.path.exists(dur_idx):
        np.save(dur_idx, duration_index)
    
    # create dataset for data loader
    #--------------------------------
    datas = []
    for df in [df_train, df_tune, df_val]:
        imgs = df['img_dir'].to_list()
        times = y_train[0].tolist()
        events = y_train[1].tolist()
        #time = torch.from_numpy(y_train[0])
        #event = torch.from_numpy(y_train[1])
        #labels = (time, event)
        #print('labels:\n', labels)
        data = [{'image': img, 'label': (time, event)} for img, time, event in zip(imgs, times, events)]
        #print('data:\n', data)
        datas.append(data)
    train_data = datas[0]
    tune_data = datas[1]
    val_data = datas[2]
    #print('train_data:\n', train_data)

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

    # check data image and labels
    check_ds = Dataset(data=train_data, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    print('check data:\n', check_data)
    print('\ncheck image and lable shape:', check_data['image'].shape, check_data['label'])

    # create a training data loader
    ds_train = Dataset(data=train_data, transform=train_transforms)
    dl_train = DataLoader(
        ds_train, 
        batch_size=batch_size, 
        shuffle=True, 
        #num_workers=4, 
        pin_memory=torch.cuda.is_available()
        )
    # create a tuning data loader
    ds_tune = Dataset(data=tune_data, transform=tune_transforms)
    dl_tune = DataLoader(
        ds_tune, 
        batch_size=batch_size, 
        shuffle=False, 
        #num_workers=4,
        pin_memory=torch.cuda.is_available()
        )
    # create a validation data loader
    ds_val = Dataset(data=val_data, transform=val_transforms)
    dl_val = DataLoader(
        ds_val, 
        batch_size=batch_size, 
        #num_workers=4,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
        )

    print('\ntrain_ds:', ds_train)
    print('dl_train:', dl_train)
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

