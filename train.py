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



def train(output_dir, pro_data_dir, log_dir, model_dir, cnn_model, cox_model, epochs, dl_train, 
          dl_tune, dl_val, dl_tune_cb, df_tune, cnn_name, model_depth, lr, target_c_index,
          eval_model='best_model', scheduler_type='lambda', train_logs=True, plot_c_indices=True,
          fit_model=True):
    

    # choose callback functions
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
    concordance = Concordance(
        save_dir=model_dir,
        df_tune=df_tune,
        dl_tune_cb=dl_tune_cb,
        target_c_index=target_c_index,
        cnn_name=cnn_name,
        model_depth=model_depth,
        lr=lr)
    # callback: early stopping with c-index
    saved_cpt = os.path.join(log_dir, 'cpt_weights.pt')
    early_stopping = tt.callbacks.EarlyStopping(
        get_score=concordance.get_last_score,
        minimize=False,
        patience=200,
        file_path=saved_cpt)
    # combine call backs
    callbacks = [concordance, early_stopping, lr_scheduler]

    # fit model
    if fit_model:
        my_model = cox_model
        log = my_model.fit_dataloader(
            dl_train,
            epochs=epochs,
            callbacks=callbacks,
            verbose=True,
            val_dataloader=dl_tune)
        print('log: \n', log) 
        # save model training curves
        plot = log.plot()
        fig = plot.get_figure()
        log_fn = str(cnn_name) + str(model_depth) + '_' + 'loss.png'
        fig.savefig(os.path.join(output_dir, log_fn))
    
    # evalute model on val data
    fn = str(cnn_name) + str(model_depth) + '_c_indexs.npy'
    c_indexs = np.load(os.path.join(model_dir, fn))
    print(c_indexs)
    if eval_model == 'best_model':
        c_index = np.amax(c_indexs)
        fn = cnn_name + str(model_depth) + '_' + str(c_index) + '_final_model.pt'
    elif eval_model == 'final_model':
        c_index = c_indexs[-1]
        fn = cnn_name + str(model_depth) + '_' + str(c_index) + '_model.pt'
    cox_model.load_net(os.path.join(model_dir, fn))
    surv = cox_model.predict_surv_df(dl_val)
    fn_surv = cnn_name + str(model_depth) + str(c_index) + 'surv.csv'
    surv.to_csv(os.path.join(pro_data_dir, fn_surv), index=False)
    # val c-index
    df_val = pd.read_csv(os.path.join(pro_data_dir, 'df_pn_masked_val0.csv'))
    durations = df_val['death_time'].to_numpy()
    events = df_val['death_event'].to_numpy()
    ev = EvalSurv(
        surv=surv,
        durations=durations,
        events=events,
        censor_surv='km')
    val_c_index = round(ev.concordance_td(), 3)
    print('val c-index:', val_c_index)

    # plot c-index
    if plot_c_indices:
        x = np.arange(0, epochs+1, 1, dtype=int).tolist()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        #ax.set_aspect('equal')
        print('c_indexs:', c_indexs)
        print('x:', x)
        plt.plot(c_indexs, color='red', linewidth=3, label='c-index')
        plt.xlim([0, epochs+1])
        plt.ylim([0, 1])
        #ax.axhline(y=0, color='k', linewidth=4)
        #ax.axhline(y=1.03, color='k', linewidth=4)
        #ax.axvline(x=-0.03, color='k', linewidth=4)
        #ax.axvline(x=1, color='k', linewidth=4)
        plt.xticks(x, fontsize=16, fontweight='bold')
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
        plt.xlabel('Epochs', fontweight='bold', fontsize=16)
        plt.ylabel('C-index', fontweight='bold', fontsize=16)
        plt.legend(loc='lower right', prop={'size': 16, 'weight': 'bold'})
        plt.grid(True)
        plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
        fn = cnn_name + str(model_depth) + '_c_index.png'
        plt.savefig(os.path.join(output_dir, fn), format='png', dpi=600)
        plt.close()

    # write txt files
    if train_logs:
        log_fn = 'train_logs.text'
        write_path = os.path.join(output_dir, log_fn)
        with open(write_path, 'a') as f:
            f.write('\n-------------------------------------------------------------------')
            f.write('\ncreated time: %s' % strftime('%Y-%m-%d %H:%M:%S', localtime()))
            f.write('\n-------------------------------------------------------------------')
            f.write('\nbest tuning c-index: %s' % np.amax(c_indexs))
            f.write('\nval c-index: %s' % val_c_index)
            f.write('\ncnn model: %s' % cnn_name)
            f.write('\nmodel depth: %s' % model_depth)
            f.write('\nepoch: %s' % epochs)
            f.write('\nlearning rate: %s' % lr)
            f.write('\n')
            f.close()
        print('successfully save train logs.')


