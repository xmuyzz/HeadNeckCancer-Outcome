import os
import pandas as pd
import numpy as np
import torch
import random
import onnx
from torchinfo import summary
from onnx2torch import convert
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

def warn(*args, **kwargs):
    pass


def main(opt):

    warnings.warn = warn

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    torch.cuda.is_available()
    print(torch.__version__)

    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    if opt.proj_dir is not None:
        task_dir = opt.proj_dir + '/task/' + opt.task + '_' + opt.surv_type + '_' + opt.img_type + '_' + \
            opt.tumor_type + '_' + opt.cox + '_' + opt.cnn_name + str(opt.model_depth) 
        model_dir = task_dir + '/models'
        metric_dir = task_dir + '/metrics'
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)
   
    if opt.load_data:
        # train data loader
        dl_tr, dl_va, dl_cb, dl_bl, df_va = dl_train(proj_dir=opt.proj_dir, metric_dir=metric_dir, batch_size=opt.batch_size, 
            cox=opt.cox, num_durations=opt.num_durations, surv_type=opt.surv_type, img_type=opt.img_type, 
            tumor_type=opt.tumor_type, rot_prob=opt.rot_prob, gau_prob=opt.gau_prob, flip_prob=opt.flip_prob, 
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

        #cnn_model = get_cnn_model(cnn_name=opt.cnn_name, model_depth=opt.model_depth, n_classes=n_classes, 
        #                         in_channels=opt.in_channels)
        onnxPath = '/mnt/kannlab_rfa/Zezhong/HeadNeck/encoder_nnUNet/models/ONNX_MODEL/nnTransferModel.onnx'
        onnx_model = onnx.load(onnxPath)
        cnn_model = convert(onnx_model)
        cox_model = get_cox_model(task_dir=task_dir, cnn_model=cnn_model, cox=opt.cox, lr=opt.lr)

    if opt.train:
        # train model
        train(task_dir=task_dir, surv_type=opt.surv_type, img_type=opt.img_type, cnn_name=opt.cnn_name, 
              model_depth=opt.model_depth, cox=opt.cox, cnn_model=cnn_model, cox_model=cox_model, epoch=opt.epoch, 
              batch_size=opt.batch_size, lr=opt.lr, dl_tr=dl_tr, dl_va=dl_va, dl_cb=dl_cb, dl_bl=dl_bl, df_va=df_va, 
              target_c_index=opt.target_c_index, target_loss=opt.target_loss)
    
    if opt.test:
        if opt.cox == 'CoxPH':
            n_classes = 1
            print('n_classes:', n_classes)
        else:
            n_classes = opt.num_durations
        # test data loader
        df, dl = dl_test(proj_dir=opt.proj_dir, surv_type=opt.surv_type, batch_size=opt.batch_size, cox=opt.cox, 
                         num_durations=opt.num_durations, data_set=opt.data_set, img_type=opt.img_type, tumor_type=opt.tumor_type,
                         in_channels=opt.in_channels)
        # test model
        test(surv_type=opt.surv_type, data_set=opt.data_set, task_dir=task_dir, eval_model=opt.eval_model, cox_model=cox_model, 
             df=df, dl=dl)

if __name__ == '__main__':

    opt = parse_opts()
    main(opt)
