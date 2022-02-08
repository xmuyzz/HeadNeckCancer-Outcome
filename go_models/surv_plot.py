import os
import pandas as pd
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
from pycox.utils import kaplan_meier




def surv_plot(lab_drive_dir, x, y, fn):

    """
    plot survival curves
    """

    output_dir = os.path.join(lab_drive_dir, 'output')
    if not os.path.exists(output_dir): os.mkdir(output_dir)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.set_aspect('equal')
    plt.plot(x, y, linewidth=3)
    #plt.plot(surv.iloc[:, 0], surv.iloc[:, 137], linewidth=3, label='2')
    fig.suptitle('overall survival', fontweight='bold', fontsize=16)
    plt.ylabel('S(t | x)', fontweight='bold', fontsize=12)
    plt.xlabel('Time', fontweight='bold', fontsize=12)
    plt.xlim([0, 5000])
    plt.ylim([0, 1])
    ax.axhline(y=0, color='k', linewidth=2)
    ax.axhline(y=1, color='k', linewidth=2)
    ax.axvline(x=0, color='k', linewidth=2)
    ax.axvline(x=5000, color='k', linewidth=2)
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000], fontsize=12, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', prop={'size': 12, 'weight': 'bold'})
    plt.grid(True)
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    plt.savefig(os.path.join(output_dir, fn), format='png', dpi=600)
    #plt.show()
    plt.close()
    print('saved survival curves!')


