import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn_pandas import DataFrameMapper
from PIL import Image
import gc
import nibabel as nib
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
import monai
from monai.utils import first
#from models import ResNet
from models.cnn import cnn3d



def collate_fn(batch):
    
    """Stacks the entries of a nested tuple
    """
    return tt.tuplefy(batch).stack()


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


class dataset0(Dataset):

    """
    load img and labels for CoxPH model
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
        print('event size:', self.event.size(dim=0))
        return self.event.shape[0]
    
    def __getitem__(self, index):
        if type(index) is not int:
            raise ValueError(f'Need `index` to be `int`. Got {type(index)}.')
        img = nib.load(self.img_dir[index])
        arr = img.get_data()
        # choose image channel
        img = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        #if self.channel == 1:
        #    img = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        #elif self.channel == 3:
        #    img = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        # convert numpy array to torch tensor
        img = torch.from_numpy(img).float()
        # data and label transform
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, (self.time[index], self.event[index])


class dataset_pred(Dataset):

    """
    load img and labels for CoxPH model
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
        # choose image channel
        #if self.channel == 1:
        #    img = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        #elif self.channel == 3:
        #    img = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        img = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        # convert numpy array to torch tensor
        img = torch.from_numpy(img).float()
        # data and label transform
        if self.transform:
            img = self.transform(img)
        return img


def data_transform(proj_dir, batch_size, _cox_model, num_durations):
    
    """
    Create dataloder for image and lable inputs
    
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

    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(pro_data_dir): os.mkdir(pro_data_dir)

    ## load train and val dataset
    df_train_ = pd.read_csv(os.path.join(pro_data_dir, 'df_train0.csv')) 
    df_train = df_train_.sample(frac=0.8, random_state=200)
    #df_tune = df_train_.sample(frac=0.2, random_state=200)
    df_tune = df_train_.drop(df_train.index)
    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    print('df_train shape:', df_train.shape)
    print('df_tune shape:', df_tune.shape)
    print('df_val shape:', df_val.shape)

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

    dfs = []
    get_target = lambda df: (df['sur_duration'].to_numpy(), df['survival'].to_numpy())
    for df in [df_train, df_tune, df_val]:
        if _cox_model in ['PCHazard', 'LogisticHazard', 'DeepHit']:
            #labtrans = PCHazard.label_transform(num_durations)
            y = labtrans.fit_transform(*get_target(df))
            #y = labtrans.transform(*get_target(df))
            out_features = labtrans.out_features
            duration_index = labtrans.cuts
            print('y[0]:', y[0])
            print('y[1]:', y[1])
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
    print('out_features:', out_features)
    print('duration_index:', duration_index)
    #print(labtrans.cuts[y_train[0]])
    # save duration index for train and evaluate steps
    if not os.path.exists(os.path.join(pro_data_dir, 'duration_index.npy')):
        np.save(os.path.join(pro_data_dir, 'duration_index.npy'), duration_index)
     
    return df_train, df_tune, df_val


def data_loader2(proj_dir, batch_size, _cox_model, num_durations):

    df_train, df_tune, df_val = data_transform(
        proj_dir, 
        batch_size, 
        _cox_model, 
        num_durations)

    # train, tune, val dataset
    if _cox_model == 'CoxPH':
        ds_train = dataset1(x_train, *y_train)
        ds_tune = dataset1(x_tune, *y_tune)
        ds_val = dataset_pred(x_val)
    elif _cox_model in ['PCHazard', 'LogisticHazard', 'DeepHit']:
        ds_train = dataset0(df_train)
        ds_tune = dataset0(df_tune)
        ds_val = dataset_pred(df_val)
    else:
        print('choose another cox model!')
    
    # check data
    check_loader = DataLoader(ds_train, batch_size=1)
    check_data = first(check_loader)
    print('\ncheck image and lable shape:', check_data)

    # batch data loader
    dl_train = DataLoader(
        dataset=ds_train, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
        )
    dl_tune = DataLoader(
        dataset=ds_tune, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
        )
    dl_val = DataLoader(
        dataset=ds_val,
        batch_size=batch_size,
        shuffle=False
        )

    print('successfully created data loaders!')

    return dl_train, dl_tune, dl_val






