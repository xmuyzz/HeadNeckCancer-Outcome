import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn_pandas import DataFrameMapper
from PIL import Image
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

#from models import ResNet
from go_models.data_loader import DataLoader
#from go_models.generate_model import generate_model
#from go_models.opt import opt
from models.cnn import cnn3d




def train(proj_dir, lab_drive_dir, cox_model, epochs, verbose, 
          dl_train, dl_tune, dl_val):

    """
    train cox model

    @params:
      tumor_type - required: tumor + node or tumor
      data_dir - required: tumor + node label dir CHUM cohort
      arr_dir - required: tumor + node label dir CHUS cohort
    """

    output_dir = os.path.join(lab_drive_dir, 'output')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    if not os.path.exists(output_dir): os.mkdir(pro_data_dir)
    

    # train cox model
    #------------------
    callbacks = [tt.cb.EarlyStopping(patience=100)]

    log = model.fit_dataloader(
        dl_train,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
        val_dataloader=dl_tune
        )
    print('log: \n', log)
    # save model training curves
    plot = log.plot()
    fig = plot.get_figure()
    fig.savefig(os.path.join(output_dir, 'log.png'))
    print('saved train acc and loss curves!')

    # save trained model
    #--------------------
    """save the whole network
    """
    model.save_net(os.path.join(pro_data_dir, 'model.pt'))
    
    """only store weights
    """
    model.save_model_weights(os.path.join(pro_data_dir, 'weights.pt'))
    
    print('saved trained model and weights!')

    
