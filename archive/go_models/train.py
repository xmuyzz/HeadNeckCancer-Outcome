import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
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
from go_models.cindex_callback import Concordance




def train(proj_dir, cox_model, epochs, dl_train, dl_tune, 
          dl_val, cnn_name, lr, save_model='model'):

    """
    train cox model

    Args:
        tumor_type {str} -- tumor + node or tumor;
        cox_model {model} -- tumor + node label dir CHUM cohort;
        dl_train {data loader} -- train data loader;

    Return:
        train/val loss, acc, trained model/weights;
    """


    output_dir = os.path.join(proj_dir, 'output')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    if not os.path.exists(pro_data_dir): os.mkdir(pro_data_dir)
    

    # train cox model
    #------------------
    concordance = Concordance(
        x_test, 
        durations_test, 
        events_test
        )
    early_stopping = tt.callbacks.EarlyStopping(
        get_score=concordance.get_last_score,
        minimize=False
        )
    callbacks = [concordance, early_stopping]
    #callbacks = [tt.cb.EarlyStopping(patience=100)]
    model = cox_model
    log = model.fit_dataloader(
        dl_train,
        epochs=epochs,
        callbacks=callbacks,
        verbose=True,
        val_dataloader=dl_tune
        )
    print('log: \n', log)
    # save model training curves
    plot = log.plot()
    fig = plot.get_figure()
    log_fn = str(cnn_name) + '_' + str(epochs) + '_' + \
         str(lr) + '_' + 'log.png'
    fig.savefig(os.path.join(output_dir, log_fn))
    print('saved train acc and loss curves!')
    
    # concordance index
    #------------------
    # survival prediction
    surv = cox_model.predict_surv_df(dl_val)
    fn_surv = str(cnn_name) + '_' + str(epochs) + '_' + \
              str(lr) + '_' + 'surv.csv'
    surv.to_csv(os.path.join(pro_data_dir, fn_surv), index=False)
    # C-index
    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_val0.csv'))
    durations = df_val['death_time'].to_numpy()
    events = df_val['death_event'].to_numpy()
    ev = EvalSurv(
        surv=surv,
        durations=durations,
        events=events,
        censor_surv='km'
        )
    c_index = ev.concordance_td()
    print('concordance index:', round(c_index, 3))

    # save trained model
    #--------------------
    if save_model == 'model':
        # save the whole network
        model_fn = str(cnn_name) + '_' + str(epochs) + '_' + \
                   str(lr) + '_' + 'model.pt'
        model.save_net(os.path.join(pro_data_dir, model_fn))
    elif save_model == 'weights':
        # only store weights
        weights_fn = str(cnn_name) + '_' + str(epochs) + '_' + \
                     str(lr) + '_' + 'weights.pt'
        model.save_model_weights(os.path.join(pro_data_dir, weights_fn))
    
    print('saved trained model and weights!')

    # write txt file
    #------------------
    log_fn = 'train_logs.text'
    write_path = os.path.join(output_dir, log_fn)
    with open(write_path, 'a') as f:
        f.write('\n-------------------------------------------------------------------')
        f.write('\ncreated time: %s' % strftime('%Y-%m-%d %H:%M:%S', localtime()))
        f.write('\n-------------------------------------------------------------------')
        f.write('\nconcordance index: %s' % c_index)
        f.write('\ncnn model: %s' % cnn_name)
        f.write('\nepoch: %s' % epochs)
        f.write('\nlearning rate: %s' % lr)
        f.write('\n')
        f.close()
    print('successfully save train logs.')
