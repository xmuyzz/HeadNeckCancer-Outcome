import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from time import localtime, strftime
import torch
import torchtuples as tt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import *
#import h5py
from pycox.datasets import metabric
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH, PCHazard, LogisticHazard, DeepHitSingle
from pycox.utils import kaplan_meier
from callbacks import Concordance, LRScheduler
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ReduceLROnPlateau


def train(output_dir, pro_data_dir, log_dir, model_dir, cnn_model, cox_model, epochs, dl_train, 
          dl_tune, dl_val, dl_tune_cb, df_tune, cnn_name, model_depth, lr, save_model='model',
          scheduler_type='lambda'):
    
    # load model
    model = cox_model

    # choose callback functions
    #---------------------------
    # callback: LR scheduler
    lambda1 = lambda epoch: 0.9 ** (epoch // 50)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
    if scheduler_type == 'lambda':
        scheduler = LambdaLR(
            optimizer, 
            lr_lambda=[lambda1])
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.1, 
            patience=10, 
            threshold=0.0001, 
            threshold_mode='abs')
    lr_scheduler = LRScheduler(scheduler)

    # callback: metric monitor with c-index
    concordance = Concordance(
        save_dir=log_dir,
        run='os',
        df_tune=df_tune,
        dl_tune_cb=dl_tune_cb)

    # callback: early stopping with c-index
    saved_cpt = os.path.join(log_dir, 'cpt_weights.pt')
    early_stopping = tt.callbacks.EarlyStopping(
        get_score=concordance.get_last_score,
        minimize=False,
        patience=200,
        file_path=saved_cpt)
    
    callbacks = [concordance, early_stopping, lr_scheduler]
    #callbacks = [concordance, early_stopping]

    # fit model
    log = model.fit_dataloader(
        dl_train,
        epochs=epochs,
        callbacks=callbacks,
        verbose=True,
        val_dataloader=dl_tune)
    print('log: \n', log)
    # save model training curves
    plot = log.plot()
    fig = plot.get_figure()
    log_fn = str(cnn_name) + str(model_depth) + '_' + str(epochs) + '_' + \
         str(lr) + '_' + 'log.png'
    fig.savefig(os.path.join(output_dir, log_fn))
    print('saved train acc and loss curves!')
    
    # survival prediction
    surv = cox_model.predict_surv_df(dl_val)
    fn_surv = str(cnn_name) + str(model_depth) + '_' + str(epochs) + '_' + \
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
        censor_surv='km')
    c_index = ev.concordance_td()
    print('concordance index:', round(c_index, 3))

    # save trained model
    if save_model == 'model':
        # save the whole network
        model_fn = str(cnn_name) + str(model_depth) + '_' + str(epochs) + '_' + \
                   str(lr) + '_' + 'model.pt'
        model.save_net(os.path.join(model_dir, model_fn))
    elif save_model == 'weights':
        # only store weights
        weights_fn = str(cnn_name) + str(model_depth) + '_' + str(epochs) + '_' + \
                     str(lr) + '_' + 'weights.pt'
        model.save_model_weights(os.path.join(model_dir, weights_fn))
    print('saved trained model and weights!')

    # write txt files
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




if __name__ == '__main__':

    data_dir = '/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated'
    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    output_dir = os.path.join(proj_dir, 'output')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    if not os.path.exists(pro_data_dir): 
        os.makefirs(pro_data_dir)
    cnn_name = 'resnet'
    model_depth = 101  # [10, 18, 34, 50, 101, 152, 200]
    n_classes = 20
    in_channels = 1
    batch_size = 8
    epochs = 1
    lr = 0.001
    num_durations = 20
    _cox_model = 'LogisticHazard'
    cox_model = 'LogisticHazard'
    load_model = 'model'
    score_type = '3year_survival'   #'median'
    evaluate_only = False
    augmentation = True

    np.random.seed(1234)
    _ = torch.manual_seed(1234)

    if not augmentation:
        dl_train, dl_tune, dl_val = data_loader2(
            proj_dir=proj_dir,
            batch_size=batch_size,
            _cox_model=_cox_model,
            num_durations=num_durations)
    else:
        dl_train, dl_tune, dl_val, dl_test = data_loader_transform(
            proj_dir,
            batch_size=batch_size,
            _cox_model=_cox_model,
            num_durations=num_durations)
    for cnn_name in ['resnet18', 'resnet34', 'resnet50', 'resnet152', 'resnet200']:
        cnn_model = get_cnn_model(
            cnn_name=cnn_name,
            n_classes=n_classes,
            in_channels=in_channels)
        cox_model = get_cox_model(
            proj_dir=proj_dir,
            cnn_model=cnn_model,
            _cox_model=_cox_model,
            lr=lr)
        for epochs in [20]:
            for lr in [0.01, 0.0001, 0.00001, 0.1]:
                train(
                    output_dir=output_dir,
                    pro_data_dir=pro_data_dir,
                    cox_model=cox_model,
                    epochs=epochs,
                    dl_train=dl_train,
                    dl_tune=dl_tune,
                    dl_val=dl_val,
                    cnn_name=cnn_name,
                    lr=lr)
                evaluate(
                    proj_dir=proj_dir,
                    cox_model=cox_model,
                    load_model=load_model,
                    dl_val=dl_val,
                    score_type=score_type,
                    cnn_name=cnn_name,
                    epochs=epochs,
                    lr=lr)
