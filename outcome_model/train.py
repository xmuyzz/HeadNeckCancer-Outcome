import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from time import localtime, strftime
import matplotlib.pyplot as plt
import torch
import torchtuples as tt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import *
from pycox.datasets import metabric
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH, PCHazard, LogisticHazard, DeepHitSingle
from pycox.utils import kaplan_meier
from callbacks import concordance_callback, LRScheduler
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ReduceLROnPlateau
from logger import train_logger


def train(task_dir, surv_type, img_type, cnn_name, model_depth, cox, cnn_model, cox_model, epoch, batch_size, lr, 
          dl_tr, dl_va, dl_cb, dl_bl, df_va, target_c_index, target_loss):

    # train logger
    tr_log_path = train_logger(task_dir, surv_type, img_type, cnn_name, model_depth, cox, epoch, batch_size, lr, df_va)

    # callback: LR scheduler
    lambda1 = lambda epoch: 0.9 ** (epoch // 50)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr, weight_decay=0.002)
    scheduler_type='lambda'
    if scheduler_type == 'lambda':
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, 
            threshold=0.0001, threshold_mode='abs')
    lr_scheduler = LRScheduler(scheduler)

    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)

    # callback: metric monitor with c-index
    concordance = concordance_callback(
        task_dir=task_dir,
        tr_log_path=tr_log_path,
        dl_bl=dl_bl,
        dl_cb=dl_cb,
        df_va=df_va,
        cnn_name=cnn_name,
        cox=cox,
        model_depth=model_depth,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        target_c_index=target_c_index,
        target_loss=target_loss)

    # callback: early stopping with c-index
    early_stopping = tt.callbacks.EarlyStopping(
        get_score=concordance.get_last_score,
        minimize=False,
        patience=200,
        file_path=task_dir + '/models/cpt_weights.pt')
    
    # fit model
    print('start model training....')
    my_model = cox_model
    log = my_model.fit_dataloader(
        dl_tr,
        epochs=epoch,
        callbacks=[concordance, lr_scheduler],
        verbose=True,
        val_dataloader=dl_va)
    print('log: \n', log) 
    # save model training curves
    plot = log.plot()
    fig = plot.get_figure()
    fig.savefig(task_dir + '/final_loss.png')
    







