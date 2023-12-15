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



def get_df(data_dir, surv_type, data_set, tumor_type, img_size, img_type, cox, num_durations):

    ## load train and val dataset
    #csv_dir = data_dir + '/data/' + img_type 
    csv_dir = data_dir + '/' + img_size + '_' + img_type + '/' + surv_type
    fn = data_set + '_img_label_' + tumor_type + '.csv'
    df = pd.read_csv(csv_dir + '/' + fn)
    print('\ndf shape:', df.shape)
    df = df.dropna(subset=[surv_type + '_event', surv_type + '_time'])
    df['time'], df['event'] = [df[surv_type + '_time'].values, df[surv_type + '_event'].values]

    return df


def dl_test(data_dir, surv_type, batch_size, cox, num_durations, data_set, img_size, img_type, tumor_type, in_channels):


    df = get_df(data_dir, surv_type, data_set, tumor_type, img_size, img_type, cox, num_durations)

    transforms = Compose([ScaleIntensity(minv=0.0, maxv=1.0), EnsureType(data_type='numpy')])

    ds = DatasetPred(df, transform=transforms, in_channels=in_channels)
    
    dl = DataLoader(dataset=ds, shuffle=False, batch_size=batch_size)

    #batch = next(iter(dl_tr))
    #print(batch.shapes())
    
    print('\nsuccessfully created data loaders!')

    return df, dl



