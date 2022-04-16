import os
import pandas as pd
import numpy as np
import torch
import random
from data_loader_transform import data_loader_transform
from train import train
from test import test
from get_cnn_model import get_cnn_model
from get_cox_model import get_cox_model
from opts import parse_opts
from get_data.data_loader2 import data_loader2



def main(opt):

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    if opt.proj_dir is not None:
        opt.output_dir = os.path.join(opt.proj_dir, opt.output)
        opt.pro_data_dir = os.path.join(opt.proj_dir, opt.pro_data)
        opt.log_dir = os.path.join(opt.proj_dir, opt.log)
        opt.model_dir = os.path.join(opt.proj_dir, opt.model)
        opt.train_dir = os.path.join(opt.proj_dir, opt.train_folder)
        opt.val_dir = os.path.join(opt.proj_dir, opt.val_folder)
        opt.test_dir = os.path.join(opt.proj_dir, opt.test_folder)
        if not os.path.exists(opt.output_dir):
            os.makedirs(opt.output_dir)
        if not os.path.exists(opt.pro_data_dir):
            os.makefirs(opt.pro_data_dir)
        if not os.path.exists(opt.log_dir):
            os.makedirs(opt.log_dir)
        if not os.path.exists(opt.model_dir):
            os.makedirs(opt.model_dir)
        if not os.path.exists(opt.train_dir):
            os.makedirs(opt.train_dir)
        if not os.path.exists(opt.val_dir):
            os.makedirs(opt.val_dir)
        if not os.path.exists(opt.test_dir):
            os.makedirs(opt.test_dir)

    if opt.augmentation:
        dl_train, dl_tune, dl_val, dl_test, dl_tune_cb, df_tune, \
        dl_baseline = data_loader_transform(
            pro_data_dir=opt.pro_data_dir, 
            batch_size=opt.batch_size, 
            _cox_model=opt._cox_model, 
            num_durations=opt.num_durations,
            _outcome_model=opt._outcome_model,
            tumor_type=opt.tumor_type,
            input_data_type=opt.input_data_type,
            i_kfold=opt.i_kfold)
    else:
        dl_train, dl_tune, dl_val = data_loader2(
            pro_data_dir=opt.pro_data_dir,
            batch_size=opt.batch_size,
            _cox_model=opt.cox_model_name,
            num_durations=opt.num_durations)
   
    if opt.train:
        """
        CNN Models
        Implemented:
            cnn, MobileNetV2, MobileNet, ResNet, ResNetV2, WideResNet, 
            ShuffleNet, ShuffleNetV2, DenseNet, EfficientNet(b0-b9),
        Temperarily not working:
            SqueezeNet, ResNeXt, ResNeXtV2, C3D,  
        """

        cnns = ['DenseNet']
        model_depths = [121, 169, 201]
        for cnn_name in cnns:   
            for model_depth in model_depths:
                if cnn_name in ['resnet', 'ResNetV2', 'PreActResNet']:
                    assert model_depth in [10, 18, 34, 50, 152, 200]
                elif cnn_name in ['ResNeXt', 'ResNeXtV2', 'WideResNet']:
                    assert model_depth in [50, 101, 152, 200]
                elif cnn_name in ['DenseNet']:
                    assert model_depth in [121, 169, 201]
                elif cnn_name in ['MobileNet', 'MobileNetV2', 'ShuffleNet', 
                                   'ShuffleNetV2', 'EfficientNet']:
                    model_depth = 0
                if opt._cox_model == 'CoxPH':
                    n_classes = 1
                    print('n_classes:', n_classes)
                else:
                    n_classes = opt.num_durations
                cnn_model = get_cnn_model(
                    cnn_name=cnn_name,
                    model_depth=model_depth,
                    n_classes=n_classes, 
                    in_channels=opt.in_channels)
                cox_model = get_cox_model(
                    pro_data_dir=opt.pro_data_dir,
                    cnn_model=cnn_model,
                    _cox_model=opt._cox_model,
                    lr=opt.lr)
                for epoch in [100]:
                    for lr in [0.0001]:
                        train(
                            output_dir=opt.output_dir,
                            pro_data_dir=opt.pro_data_dir,
                            log_dir=opt.log_dir,
                            model_dir=opt.model_dir,
                            cnn_model=cnn_model,
                            model_depth=model_depth,
                            cox_model=cox_model,
                            epochs=epoch,
                            dl_train=dl_train,
                            dl_tune=dl_tune,
                            dl_val=dl_val,
                            dl_tune_cb=dl_tune_cb,
                            df_tune=df_tune,
                            dl_baseline=dl_baseline,
                            cnn_name=cnn_name,
                            _cox_model=opt._cox_model,
                            lr=lr,
                            target_c_index=0.75,
                            eval_model='best_model')
    if opt.test:
        test(
            run_type=run_typ,
            model_dir=model_dir,
            log_dir=log_dir,
            pro_data_dir=opt.pro_data_dir, 
            cox_model=cox_model, 
            dl_val=dl_val, 
            dl_test=dl_test,
            cnn_name=cnn_name, 
            model_depth=model_depth)


if __name__ == '__main__':

    opt = parse_opts()

    main(opt)


