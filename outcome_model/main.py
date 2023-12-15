import os
import pandas as pd
import numpy as np
import torch
import random
#import onnx
#from torchinfo import summary
#from onnx2torch import convert
from dl_train import dl_train
from dl_test import dl_test
from train import train
from test import test
from get_cnn_model import get_cnn_model
from get_cox_model import get_cox_model
from opts import parse_opts
import torch.cuda
import time
import warnings
import ast


def warn(*args, **kwargs):
    pass


def main(opt):

    warnings.warn = warn

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('\ntorch cuda is availbale:', torch.cuda.is_available())
    print('\ntorch cudas device count:', torch.cuda.device_count())
    print('\ntorch cuda current device:', torch.cuda.current_device())
    print('\ntorch version:', torch.__version__)

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)


    if opt.proj_dir is not None:
        print('creating directories for study...')
        task_dir = opt.proj_dir + '/task/' + opt.task + '_' + opt.surv_type + '_' + opt.img_size + '_' + \
                   opt.img_type + '_' + opt.tumor_type + '_' + opt.cox + '_' + opt.cnn_name + str(opt.model_depth) 
        model_dir = task_dir + '/models'
        metric_dir = task_dir + '/metrics'
        log_dir = task_dir + '/logs'
        for dir in [task_dir, model_dir, metric_dir, log_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)

    if opt.load_train_data:
        # train data loader
        print('\nloading data .......')
        dl_tr, dl_va, dl_cb, dl_bl, df_va = dl_train(
            data_dir=opt.data_dir, 
            metric_dir=metric_dir, 
            batch_size=opt.batch_size,
            cnn_name=opt.cnn_name, 
            cox=opt.cox, 
            num_durations=opt.num_durations, 
            surv_type=opt.surv_type, 
            img_size=opt.img_size, 
            img_type=opt.img_type, 
            tumor_type=opt.tumor_type, 
            rot_prob=opt.rot_prob, 
            gauss_prob=opt.gauss_prob, 
            flip_prob=opt.flip_prob, 
            in_channels=opt.in_channels)

    if opt.load_model:
        """
        CNN Models
        Implemented:
            cnn, MobileNetV2, MobileNet, ResNet, ResNetV2, WideResNet, 
            ShuffleNet, ShuffleNetV2, DenseNet, EfficientNet(b0-b9),
        Temperarily not working:
            SqueezeNet, ResNeXt, ResNeXtV2, C3D,  
        """
        #cnns = ['ResNetV2']
        #model_depths = [10, 18, 34, 50, 152, 200]
        print('\nloading model .......')
        if opt.cnn_name in ['resnet', 'ResNetV2', 'PreActResNet']:
            assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        elif opt.cnn_name in ['ResNeXt', 'ResNeXtV2', 'WideResNet']:
            assert opt.model_depth in [50, 101, 152, 200]
        elif opt.cnn_name in ['DenseNet']:
            assert opt.model_depth in [121, 169, 201]
        elif opt.cnn_name in ['MobileNet', 'MobileNetV2', 'ShuffleNet', 'ShuffleNetV2', 'EfficientNet']:
            model_depth = ''
        if opt.cox == 'CoxPH':
            n_classes = 1
            print('n_classes:', n_classes)
        else:
            n_classes = opt.num_durations

        cnn_model = get_cnn_model(cnn_name=opt.cnn_name, 
                                  model_depth=opt.model_depth, 
                                  n_classes=n_classes, 
                                  n_clinical=opt.n_clinical,
                                  in_channels=opt.in_channels)
        # use onnx for nnUNet_encoder
        #onnxPath = '/mnt/kannlab_rfa/Zezhong/HeadNeck/encoder_nnUNet/models/ONNX_MODEL/nnTransferModel.onnx'
        #onnx_model = onnx.load(onnxPath)
        #cnn_model = convert(onnx_model)
        cox_model = get_cox_model(task_dir=task_dir, 
                                  cnn_model=cnn_model, 
                                  cox=opt.cox, 
                                  lr=opt.lr)

    if opt.train:
        # load train data
        # dl_tr, dl_va, dl_cb, dl_bl, df_va = dl_train(data_dir=opt.data_dir, metric_dir=metric_dir, batch_size=opt.batch_size, 
        #     cox=opt.cox, num_durations=opt.num_durations, surv_type=opt.surv_type, img_size=opt.img_size, 
        #     img_type=opt.img_type, tumor_type=opt.tumor_type, rot_prob=opt.rot_prob, gauss_prob=opt.gauss_prob, flip_prob=opt.flip_prob, 
        #     in_channels=opt.in_channels)

        # train model
        print('\nmodel training start ......')
        train(task_dir=task_dir, 
              surv_type=opt.surv_type, 
              img_size=opt.img_size, 
              img_type=opt.img_type, 
              cnn_name=opt.cnn_name, 
              model_depth=opt.model_depth, 
              cox=opt.cox, 
              cnn_model=cnn_model, 
              cox_model=cox_model, 
              epoch=opt.epoch, 
              batch_size=opt.batch_size, 
              lr=opt.lr, 
              dl_tr=dl_tr, 
              dl_va=dl_va, 
              dl_cb=dl_cb, 
              dl_bl=dl_bl, 
              df_va=df_va, 
              target_c_index=opt.target_c_index, 
              target_loss=opt.target_loss, 
              gauss_prob=opt.gauss_prob, 
              rot_prob=opt.rot_prob, 
              flip_prob=opt.flip_prob)
    
    if opt.test:
        print('model testing start .......')
        if opt.cox == 'CoxPH':
            n_classes = 1
            print('n_classes:', n_classes)
        else:
            n_classes = opt.num_durations
        # test data loader
        for data_set in ['ts', 'tx_bwh', 'tx_maastro']:
            df, dl = dl_test(data_dir=opt.data_dir, 
                            surv_type=opt.surv_type, 
                            batch_size=opt.batch_size, 
                            cox=opt.cox, 
                            num_durations=opt.num_durations, 
                            data_set=data_set, 
                            img_size=opt.img_size, 
                            img_type=opt.img_type, 
                            tumor_type=opt.tumor_type, 
                            in_channels=opt.in_channels)
            # test model
            test(surv_type=opt.surv_type, 
                 data_set=data_set, 
                 task_dir=task_dir, 
                 eval_model=opt.eval_model, 
                 cox_model=cox_model, 
                 df=df, 
                 dl=dl,
                 cox=opt.cox)


if __name__ == '__main__':

    opt = parse_opts()
    main(opt)


