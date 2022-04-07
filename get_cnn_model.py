import torch
from torch import nn
from models.cnn import cnn3d
#from models import resnet, resnet2p1d, pre_act_resnet, wide_resnet, resnext, densenet
from models import cnn, resnet



def get_cnn_model(cnn_name, model_depth, n_classes, in_channels):
 
    if cnn_name == 'cnn':
        """
        3D simple cnn model
        """
        print(cnn_name)
        model = cnn3d()

    elif cnn_name == 'resnet':
        """
        3D resnet with different depths
        model_depth = [10, 18, 34, 50, 101, 152, 200]
        """
        print(cnn_name[6:])
        model_depth = int(cnn_name[6:])
        model = resnet.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels,
            shortcut_type='B',
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            widen_factor=1.0
            )

    if torch.cuda.is_available():
        model.cuda()
    
    return model
