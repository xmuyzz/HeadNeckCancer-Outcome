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
import pandas as pd
from monai.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import os
import torch
import nibabel as nib
#from tqdm import tqd







def save_augmentation(proj_dir, aimlab_dir, number_runs=10):

    """
    Create and save data augmentation

    Arguments:
        proj_dir {path} -- project dir
        aimlab_dir {path} -- lab drive dir to store data

    Keyword arguments:
        number_runs {int} -- numbner of runs of augmentation

    Return:
        store nii images from data augmentation
    """

    torch.cuda.empty_cache()

    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    aug_data_dir = os.path.join(aimlab_dir, 'data/aug_data')
    if not os.path.exists(pro_data_dir): os.mkdir(pro_data_dir)
    if not os.path.exists(aug_data_dir): os.mkdir(aug_data_dir)

    ## load train and val dataset
    df_train_ = pd.read_csv(os.path.join(pro_data_dir, 'df_train0.csv'))
    df_train = df_train_.sample(frac=0.8, random_state=200)
    df_tune = df_train_.drop(df_train.index)
    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    print('df_train shape:', df_train.shape)
    print('df_tune shape:', df_tune.shape)
    print('df_val shape:', df_val.shape)
    #print(df_train)

    datas = []
    for df in [df_train, df_tune, df_val]:
        imgs = df_train['img_dir'].to_list()
        labels = [(event, time) for event, time in zip(df['sur_duration'], df['survival'])]
        labels = np.array(labels, dtype=np.int64)
        pat_ids = df_train['patid'].to_list()
        data = [{'image': img, 'label': label, 'ID': pat_id} for img, label, 
            pat_id in zip(imgs, labels, pat_ids)]
        datas.append(data)
    train_data = datas[0]
    tune_data = datas[1]
    val_data = datas[2]

    # define augmentaton transforms
    original_transforms = Compose([
        LoadImaged(keys=['image']),
        AddChanneld(keys=['image']),
        #Spacingd(keys=['image'], pixdim=(1.5, 1.5, 2.0), mode=('bilinear', 'nearest')),
        #Orientationd(keys=['image'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True,),
        ToTensord(keys=['image']),
        ])

    generat_transforms = Compose([
        LoadImaged(keys=['image']),
        AddChanneld(keys=['image']),
        #Spacingd(keys=['image'], pixdim=(1.5, 1.5, 2.0), mode=('bilinear', 'nearest')),
        #Orientationd(keys=['image'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True,),
        RandAffined(keys=['image'], prob=0.5, translate_range=10),
        RandRotated(keys=['image'], prob=0.5, range_x=10.0),
        RandGaussianNoised(keys='image', prob=0.5),
        ToTensord(keys=['image']),
        ])

    original_ds = Dataset(data=train_data, transform=original_transforms)
    original_loader = DataLoader(original_ds, batch_size=1)
    original_patient = first(original_loader)

    generat_ds = Dataset(data=train_data, transform=generat_transforms)
    generat_loader = DataLoader(generat_ds, batch_size=1)
    generat_patient = first(generat_loader)

    # save augmentaion images in nii
    for i in range(number_runs):
        print(i)
        name_folder = 'aug_data_' + str(i)
        output = os.path.join(aug_data_dir, name_folder)
        if not os.path.exists(output):
            os.mkdir(output)
        check_ds = Dataset(data=train_data, transform=generat_transforms)
        check_loader = DataLoader(check_ds, batch_size=1)
        check_data = first(check_loader)
        for index, patient in enumerate(check_loader):
            print(str(patient['ID'])[2:-2])
            print('label:', patient['label'])
            # Convert the torch tensors into numpy array
            arr = np.array(patient['image'].detach().cpu()[0, 0, :, :, :], dtype=np.float32)
            #arr = np.array(patient['image'].detach().cpu(), dtype=np.float32)
            print('arr.shape:', arr.shape)
            print('patient:', patient['image'].shape)
            # Convert the numpy array into nifti file
            img = nib.Nifti1Image(arr, np.eye(4))
            fn = str(patient['ID'])[2:-2] + '.nii.gz'
            nib.save(img, os.path.join(output, fn))
        print(f'step {i} done')
    print('saved all augmentation images!')




