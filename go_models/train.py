import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn_pandas import DataFrameMapper
from PIL import Image
from time import localtime, strftime
import torch
import torchtuples as tt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import *
#import h5py
from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from pycox.models import PCHazard
from pycox.models import LogisticHazard
from pycox.models import DeepHitSingle
from pycox.utils import kaplan_meier





def train(proj_dir, out_dir, cox_model, epochs, verbose, 
          dl_train, dl_tune, dl_val):

    """
    train cox model

    Args:
        tumor_type {str} -- tumor + node or tumor;
        cox_model {model} -- tumor + node label dir CHUM cohort;
        dl_train {data loader} -- train data loader;

    Return:
        train/val loss, acc, trained model/weights;
    """


    output_dir = os.path.join(out_dir, 'output')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    if not os.path.exists(output_dir): os.mkdir(pro_data_dir)
    

    # train cox model
    #------------------
    callbacks = [tt.cb.EarlyStopping(patience=100)]
    model = cox_model
    log = model.fit_dataloader(
        dl_train,
        epochs=epochs,
        callbacks=None,
        verbose=verbose,
        val_dataloader=dl_tune
        )
    print('log: \n', log)
    # save model training curves
    plot = log.plot()
    fig = plot.get_figure()
    fn = 'log_' + str(epochs) + '_' + str(strftime('%Y_%m_%d_%H_%M_%S', localtime())) + '.png'
    fig.savefig(os.path.join(output_dir, fn))
    print('saved train acc and loss curves!')

    # save trained model
    #--------------------
    # save the whole network
    model.save_net(os.path.join(pro_data_dir, 'model.pt'))
    # only store weights
    model.save_model_weights(os.path.join(pro_data_dir, 'weights.pt'))
    
    print('saved trained model and weights!')

    
