
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn_pandas import DataFrameMapper
from PIL import Image
import gc
import torch
import torchtuples as tt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.models as models
from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from pycox.models import PCHazard
from pycox.models import LogisticHazard
from pycox.models import DeepHitSingle
from pycox.utils import kaplan_meier
#from models import ResNet
from models.cnn import cnn3d
from go_models.data_loader import DataLoader



def collate_fn(batch):
    
    """Stacks the entries of a nested tuple
    """
    return tt.tuplefy(batch).stack()


class dataset_pred():

    """load data for prediction
    """

    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        img = self.data[index]
        return img


class dataset1():
    
    """load img and labels for PCHazard model
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


class dataset0():

    """load img and labels for CoxPH model
    """

    def __init__(self, data, time, event):
        self.data = data
        self.time, self.event = tt.tuplefy(time, event).to_tensor()
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        if type(index) is not int:
            raise ValueError(f'Need `index` to be `int`. Got {type(index)}.')
        img = self.data[index]
        return img, (self.time[index], self.event[index])


def DataLoader(proj_dir, batch_size, _cox_model, num_durations):
    
    """
    create dataloder for image and lable inputs
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

    # load img arrays
    imgss = []
    for dirs in [df_train['img_dir'], df_tune['img_dir'], df_val['img_dir']]:
        imgs = []
        for dir_img in dirs:
            img = np.load(dir_img)
            imgs.append(img)
        imgss.append(imgs)
    # convert list of np array to torch tensor
    x_train = torch.from_numpy(np.array(imgss[0], dtype=np.float))
    x_tune = torch.from_numpy(np.array(imgss[1], dtype=np.float))
    x_val = torch.from_numpy(np.array(imgss[2], dtype=np.float))
    x_train = x_train.float()
    x_tune = x_tune.float()
    x_val = x_val.float()
    print('x_train:', x_train.shape)
    print('x_tune:', x_tune.shape)
    print('x_val:', x_val.shape)

    # get train and val label (events, duration time)
    #-------------------------------------------------
    """
    The survival methods require individual label transforms, so we have included 
    a proposed label_transform for each method. The LogisticHazard is a discrete-time method, 
    meaning it requires discretization of the event times to be applied to continuous-time data. 
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
        out_features = labtrans.out_features
        duration_index = labtrans.cuts
        print('y_train:', y_train)
    elif _cox_model == 'CoxPH':
        y_train = get_target(df_train)
        y_tune = get_target(df_tune)
        out_features = 1
        duration_index = labtrans.cuts
    
    print('out_features:', out_features)
    print('duration_index:', duration_index)
    #print(labtrans.cuts[y_train[0]])
    
    # save duration index for train and evaluate steps
    np.save(os.path.join(pro_data_dir, 'duration_index.npy'), duration_index)

    # train, tune, val dataset
    if _cox_model == 'CoxPH':
        dataset_train = dataset1(x_train, *y_train)
        dataset_tune = dataset1(x_tune, *y_tune)
        dataset_val = dataset_pred(x_val)
    elif _cox_model in ['PCHazard', 'LogisticHazard', 'DeepHit']:
        dataset_train = dataset0(x_train, *y_train)
        dataset_tune = dataset0(x_tune, *y_tune)
        dataset_val = dataset_pred(x_val)
    else:
        print('choose another cox model!')

    # batch data loader
    dl_train = torch.utils.data.DataLoader(
        dataset=dataset_train, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
        )
    dl_tune = torch.utils.data.DataLoader(
        dataset=dataset_tune, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
        )
    dl_val = torch.utils.data.DataLoader(
        dataset=dataset_val,
        batch_size=batch_size,
        shuffle=False
        )

    return dl_train, dl_tune, dl_val






