import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from PIL import Image
import torch
import torchtuples as tt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from pycox.models import PCHazard
from pycox.models import LogisticHazard
from pycox.utils import kaplan_meier


class DatasetBatch():

    def __init__(self, data, time, event):
        self.data = data
        self.time, self.event = tt.tuplefy(time, event).to_tensor()

    def __len__(self):
        return len(self.time)

    def __getitem__(self, index):
        if not hasattr(index, '__iter__'):
            index = [index]
        img = [self.dataset[i][0] for i in index]
        img = self.data[index]
        img = torch.stack(img)
        
        return tt.tuplefy(img, (self.time[index], self.event[index]))

def DataLoader(proj_dir):

    pro_data_dir = os.path.join(proj_dir, 'pro_data')

    ## load train and val dataset
    df_train = pd.read_csv(os.path.join(pro_data_dir, 'df_train0.csv'))
    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    print('df_train shape:', df_train.shape)
    print('df_val shape:', df_val.shape)
    #-------------------------------------------------
    ## get train and val label (events, duration time)
    #-------------------------------------------------
    num_durations = 10
    labtrans = PCHazard.label_transform(num_durations)
    #duration = df['sur_duration']
    #event = df['survival']
    get_target = lambda df: (df['sur_duration'].values, df['survival'].values)
    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))
    print('y_train shape:', y_train[0].shape)
    print('y_val shape:', y_val[0].shape)

    #----------
    # get data
    #----------
    imgss = [] 
    for dirs in [df_train['img_dir'], df_val['img_dir']]:
        imgs = []
        for dir_img in dirs:
            img = np.load(dir_img)
            imgs.append(img)
        imgss.append(imgs)
    x_train = imgss[0]
    x_val = imgss[1]
    print('x_train shape:', len(x_train))
    print('x_val shape:', len(x_val))

    #------------------
    # Batch data loader
    #------------------
    dataset_train = DatasetBatch(x_train, *y_train)
    dataset_val = DatasetBatch(x_val, *y_val)
    dl_train = tt.data.DataLoaderBatch(dataset_train, batch_size, shuffle=True)
    dl_test = tt.data.DataLoaderBatch(dataset_test, batch_size, shuffle=False)
    batch = next(iter(dl_train))
    print(batch.shapes())
    print(batch.dtypes())

    return dl_train, df_test

