import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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





def get_cox_model(proj_dir, cnn_model, _cox_model, lr):

    """
    get cox model

    Args:
        proj_dir {path} -- project folder;
        cnn_model {model} -- cnn model;
        _cox_model {str} -- cox model name;
        lr {float} -- learning rate;
    
    Returns:
        cox model;

    """


    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(pro_data_dir): os.mkdir(pro_data_dir)
 
    duration_index = np.load(os.path.join(pro_data_dir, 'duration_index.npy'))

    if _cox_model == 'PCHazard':
        """
        The Piecewise Constant Hazard (PC-Hazard) model assumes that 
        the continuous-time hazard function is constant in predefined intervals. 
        It is similar to the Piecewise Exponential Models and PEANN, 
        but with a softplus activation instead of the exponential function.
        """
        cox_model = PCHazard(
            net=cnn_model,
            optimizer=tt.optim.Adam(lr),
            duration_index=duration_index
            )
    elif _cox_model == 'LogisticHazard':
        """
        The Logistic-Hazard method parametrize the discrete hazards and 
        optimize the survival likelihood. It is also called Partial Logistic 
        Regression and Nnet-survival.
        """
        cox_model = LogisticHazard(
            net=cnn_model,
            optimizer=tt.optim.Adam(lr),
            duration_index=duration_index
            )
    elif _cox_model == 'DeepHitSingle':
        """
        DeepHit is a PMF method with a loss for improved ranking that can
        handle competing risks.
        """
        cox_model = DeepHitSingle(
            net=cnn_model, 
            optimizer=tt.optim.Adam(lr),
            alpha=0.2, 
            sigma=0.1, 
            duration_index=duration_index
            )
    elif _cox_model == 'CoxPH':
        """
        CoxPH is a Cox proportional hazards model also referred to as DeepSurv.
        """
        cox_model = CoxPH(
            net=cnn_model,
            optimizer=tt.optim.Adam(lr)
            )
    else:
        print('please select other cox models!')
    print('cox model:', _cox_model)

    return cox_model

