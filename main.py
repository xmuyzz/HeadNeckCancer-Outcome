import os
import pandas as pd
import numpy as np
import torch
import random
from data_loader_transform import data_loader_transform
from train import train
from evaluate import evaluate
from get_cnn_model import get_cnn_model
from get_cox_model import get_cox_model
from opts import parse_opts
from get_data.data_loader2 import data_loader2


if __name__ == '__main__':

    opt = parse_opts()
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    if opt.proj_dir is not None:
        opt.output_dir = os.path.join(opt.proj_dir, opt.output)
        opt.pro_data_dir = os.path.join(opt.proj_dir, opt.pro_data)
        opt.log_dir = os.path.join(opt.proj_dir, opt.log)
        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)
        if not os.path.exists(opt.pro_data_dir):
            os.makefirs(opt.pro_data_dir)
        if not os.path.exists(opt.log_dir):
            os.makedirs(opt.log_dir)
    
    if opt.augmentation:
        dl_train, dl_tune, dl_val, dl_test, dl_tune_cb, df_tune = data_loader_transform(
            pro_data_dir=opt.pro_data_dir, 
            batch_size=opt.batch_size, 
            _cox_model=opt.cox_model_name, 
            num_durations=opt.num_durations)
    else:
        dl_train, dl_tune, dl_val = data_loader2(
            pro_data_dir=opt.pro_data_dir,
            batch_size=opt.batch_size,
            _cox_model=opt.cox_model_name,
            num_durations=opt.num_durations)
    
    cnns = ['resnet101']
    #cnns = ['resnet18', 'resnet34', 'resnet50', 'resnet152', 'resnet200']
    for cnn_name in cnns:   
        cnn_model = get_cnn_model(
            cnn_name=opt.cnn_name, 
            n_classes=opt.num_durations, 
            in_channels=opt.in_channels)
        cox_model = get_cox_model(
            pro_data_dir=opt.pro_data_dir,
            cnn_model=cnn_model,
            cox_model_name=opt.cox_model_name,
            lr=opt.lr)
        for epoch in [100]:
            #for lr in [0.01, 0.0001, 0.00001, 0.1]:
            for lr in [0.001]:
                train(
                    output_dir=opt.output_dir,
                    pro_data_dir=opt.pro_data_dir,
                    log_dir=opt.log_dir,
                    cox_model=cox_model,
                    epochs=epoch,
                    dl_train=dl_train,
                    dl_tune=dl_tune,
                    dl_val=dl_val,
                    dl_tune_cb=dl_tune_cb,
                    df_tune=df_tune,
                    cnn_name=opt.cnn_name,
                    lr=lr)

