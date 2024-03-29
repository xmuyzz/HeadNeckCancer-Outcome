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
from callbacks import Concordance, LRScheduler
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ReduceLROnPlateau


def train(output_dir, pro_data_dir, log_dir, model_dir, cnn_model, cox_model, epochs, dl_tr, 
          dl_va, dl_cb, dl_bl, df_va, cnn_name, cox, model_depth, lr, target_c_index=0.8, 
          eval_model='best_model', scheduler_type='lambda', train_logs=True, 
          plot_c_indices=True, fit_model=True):
    
    # define callback functions
    #---------------------------
    # callback: LR scheduler
    lambda1 = lambda epoch: 0.9 ** (epoch // 50)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=lr)
    if scheduler_type == 'lambda':
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1])
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
    print(model_dir)
    concordance = Concordance(
        model_dir=model_dir,
        log_dir=log_dir,
        dl_bl=dl_bl,
        dl_cb=dl_cb,
        df_va=df_va,
        target_c_index=target_c_index,
        cnn_name=cnn_name,
        cox=cox,
        model_depth=model_depth,
        lr=lr)
    # callback: early stopping with c-index
    early_stopping = tt.callbacks.EarlyStopping(
        get_score=concordance.get_last_score,
        minimize=False,
        patience=200,
        file_path=log_dir + '/cpt_weights.pt')
    # combine call backs
    callbacks = [concordance, early_stopping, lr_scheduler]
    #callbacks = [lr_scheduler]
    
    # fit model
    #-----------
    print('start model training....')
    if fit_model:
        my_model = cox_model
        log = my_model.fit_dataloader(
            dl_tr,
            epochs=epochs,
            #callbacks=callbacks,
            callbacks=callbacks,
            verbose=True,
            val_dataloader=dl_va)
        print('log: \n', log) 
        # save model training curves
        plot = log.plot()
        fig = plot.get_figure()
        log_fn = cnn_name + str(model_depth) + '_' + 'loss.png'
        fig.savefig(log_dir + '/' + log_fn)
    
    # evalute model on val data
    #--------------------------
    fn = cnn_name + str(model_depth) + '_c_index.npy'
    c_indexs = np.load(log_dir + '/' + fn)
    print(c_indexs)
    if eval_model == 'best_model':
        c_index = np.amax(c_indexs)
        fn = cnn_name + str(model_depth) + '_' + str(c_index) + '_model.pt'
        print(fn)
        if not os.path.exists(model_dir +'/' + fn):
            print('all c-index lower than target!')
            fn = cnn_name + str(model_depth) + '_final_model.pt'
    elif eval_model == 'final_model':
        c_index = c_indexs[-1]
        fn = cnn_name + str(model_depth) + '_final_model.pt'
        print(fn)
    cox_model.load_net(model_dir + '/' + fn)
    if cox == 'CoxPH':
        print('compute baseline hazard for CoxPH!')
        _ = cox_model.compute_baseline_hazards()
    surv = cox_model.predict_surv_df(dl_va)
    fn_surv = cnn_name + str(model_depth) + str(c_index) + 'surv.csv'
    surv.to_csv(pro_data_dir + '/' + fn_surv, index=False)
    # val c-index
    df_val = pd.read_csv(pro_data_dir + '/df_pn_masked_val0.csv')
    durations = df_val['rfs_time'].to_numpy()
    events = df_val['rfs_event'].to_numpy()
    ev = EvalSurv(
        surv=surv,
        durations=durations,
        events=events,
        censor_surv='km')
    val_c_index = round(ev.concordance_td(), 3)
    print('val c-index:', val_c_index)

    # plot c-index
    #-------------
    if plot_c_indices:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        #ax.set_aspect('equal')
        #print('c_indexs:', c_indexs)
        plt.plot(c_indexs, color='red', linewidth=3, label='c-index')
        plt.xlim([0, epochs+1])
        plt.ylim([0, 1])
        #ax.axhline(y=0, color='k', linewidth=4)
        #ax.axhline(y=1.03, color='k', linewidth=4)
        #ax.axvline(x=-0.03, color='k', linewidth=4)
        #ax.axvline(x=1, color='k', linewidth=4)
        if epochs < 20:
            interval = 2
        elif epochs > 20 and epochs < 50:
            interval = 5
        elif epochs > 50:
            interval = 10
        x = np.arange(0, epochs+1, interval, dtype=int).tolist()
        plt.xticks(x, fontsize=12, fontweight='bold')
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12, fontweight='bold')
        plt.xlabel('Epochs', fontweight='bold', fontsize=16)
        plt.ylabel('C-index', fontweight='bold', fontsize=16)
        plt.legend(loc='lower right', prop={'size': 16, 'weight': 'bold'})
        plt.grid(True)
        plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
        fn = cnn_name + str(model_depth) + '_c_index.png'
        plt.savefig(log_dir + '/' + fn, format='png', dpi=600)
        plt.close()

    # write txt files
    #----------------
    if train_logs:
        log_fn = 'train_logs.text'
        write_path = log_dir + '/' + log_fn
        with open(write_path, 'a') as f:
            f.write('\n-------------------------------------------------------------------')
            f.write('\ncreated time: %s' % strftime('%Y-%m-%d %H:%M:%S', localtime()))
            f.write('\n-------------------------------------------------------------------')
            f.write('\nbest tuning c-index: %s' % np.amax(c_indexs))
            f.write('\nval c-index: %s' % val_c_index)
            f.write('\ncnn model: %s' % cnn_name)
            f.write('\nmodel depth: %s' % model_depth)
            f.write('\ncox model: %s' % cox)
            f.write('\nepoch: %s' % epochs)
            f.write('\nlearning rate: %s' % lr)
            f.write('\n')
            f.close()
        print('successfully save train logs.')




