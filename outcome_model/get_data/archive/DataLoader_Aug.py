


from monai.utils import first
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
    RandAffined,
    )

from monai.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import os
import torch
import nibabel as nib
from tqdm import tqd




def DataLoader_Augmentation(proj_dir, batch_size=8):

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

    datas = []
    for df in [df_train, df_tune, df_val]:
        imgs = df_train['img_dir'].to_list()
        labels = [(event, time) for event, label in zip(df['sur_duration'], df['survival'])]
        labels = np.array(labels, dtype=np.int64)
        data = [{'img': img, 'label': label} for img, label in zip(images, labels)]
        datas.append(data)
    train_data = datas[0]
    tune_data = datas[1]
    val_data = datas[2]

    # Define transforms for image
    train_transforms = Compose([
        LoadImaged(keys=['img']),
        AddChanneld(keys=['img']),
        ScaleIntensityd(keys=['img']),
        #Resized(keys=['img'], spatial_size=(96, 96, 96)),
        RandRotate90d(keys=['img'], prob=0.8, spatial_axes=[0, 2]),
        EnsureTyped(keys=['img']),
        ])

    tune_transforms = Compose([
        LoadImaged(keys=['img']),
        AddChanneld(keys=['img']),
        ScaleIntensityd(keys=['img']),
        #Resized(keys=['img'], spatial_size=(96, 96, 96)),
        EnsureTyped(keys=['img']),
        ])
    
    val_transforms = Compose([
        LoadImaged(keys=['img']),
        AddChanneld(keys=['img']),
        ScaleIntensityd(keys=['img']),
        #Resized(keys=['img'], spatial_size=(96, 96, 96)),
        EnsureTyped(keys=['img']),
        ])

    #post_pred = Compose([EnsureType(), Activations(softmax=True)])
    #post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

    # Define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    print(check_data['img'].shape, check_data['label'])

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_data, transform=train_transforms)
    train_DataLoader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available())

    # create a tuning data loader
    tune_ds = monai.data.Dataset(data=tune_data, transform=tune_transforms)
    tune_DataLoader = DataLoader(
        tune_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=torch.cuda.is_available())
    
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_data, transform=val_transforms)
    val_DataLoader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available())

    return train_DataLoader, tune_DataLoader, val_DataLoader 






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

