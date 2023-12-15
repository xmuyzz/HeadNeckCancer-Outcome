import torch
from torch import nn
from models.cnn import cnn3d
from models import (C3DNet, resnet, ResNetV2, ResNeXt, ResNeXtV2, WideResNet, PreActResNet,
        EfficientNet, DenseNet, Concat_DenseNet, ShuffleNet, ShuffleNetV2, SqueezeNet, MobileNet, MobileNetV2)
from models.SwinTransformerV2 import SwinTransformerV2

def get_cnn_model(cnn_name, model_depth, n_classes, n_clinical, in_channels, sample_size=96):
    """
    generate CNN models
    Args:
        cnn_name {str} -- cnn model names;
        model_depth {int} -- model depth number;
        n_classes {int} -- number of output classes;
        in_channels {int} -- image channels;
        sample_size {int} -- image size;
    Returns:
        cnn model for train;
    """
    if cnn_name == 'simple_cnn':
        """
        3D simple cnn model
        """
        print(cnn_name)
        model = cnn3d()
    
    elif cnn_name == 'C3D':
        """
        "Learning spatiotemporal features with 3d convolutional networks." 
        """
        model = C3DNet.get_model(
            sample_size=sample_size,
            sample_duration=16,
            num_classes=n_classes,
            in_channels=1)

    elif cnn_name == 'resnet':
        """
        3D resnet
        model_depth = [10, 18, 34, 50, 101, 152, 200]
        """
        model = resnet.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels,
            shortcut_type='B',
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            widen_factor=1.0)
    
    elif cnn_name == 'ResNetV2':
        """
        3D resnet
        model_depth = [10, 18, 34, 50, 101, 152, 200]
        """
        model = ResNetV2.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels,
            shortcut_type='B',
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            widen_factor=1.0)

    elif cnn_name == 'ResNeXt':
        """
        WideResNet
        model_depth = [50, 101, 152, 200]
        """
        model = ResNeXt.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            in_channels=in_channels,
            sample_size=sample_size,
            sample_duration=16)
    
    elif cnn_name == 'ResNeXtV2':
        """
        WideResNet
        model_depth = [50, 101, 152, 200]
        """
        model = ResNeXtV2.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels)

    elif cnn_name == 'PreActResNet':
        """
        WideResNet
        model_depth = [50, 101, 152, 200]
        """
        model = PreActResNet.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels)

    elif cnn_name == 'WideResNet':
        """
        WideResNet
        model_depth = [50, 101, 152, 200]
        """
        model = WideResNet.generate_model(
            model_depth=model_depth,
            n_classes=n_classes,
            n_input_channels=in_channels)

    elif cnn_name == 'DenseNet':
        """
        3D resnet
        model_depth = [121, 169, 201]
        """
        model = DenseNet.generate_model(
            model_depth=model_depth,
            num_classes=n_classes,
            n_input_channels=in_channels)

    elif cnn_name == 'DenseNet_Concat':
        """
        3D resnet
        model_depth = [121, 169, 201]
        """
        model = Concat_DenseNet.CustomDenseNet(
            model_depth=model_depth,
            n_classes=n_classes,
            num_clinical_features=n_clinical,
            in_channels=in_channels)

    elif cnn_name == 'SqueezeNet':
        """
        SqueezeNet
        "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and 
        <0.5MB model size"
        """
        model = SqueezeNet.get_model(
            version=1.0,
            sample_size=sample_size,
            sample_duration=16,
            num_classes=n_classes,
            in_channels=in_channels)
   
    elif cnn_name == 'ShuffleNetV2':
        """
        ShuffleNetV2
        "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
        """
        model = ShuffleNetV2.get_model(
            sample_size=sample_size,
            num_classes=n_classes,
            width_mult=1.,
            in_channels=in_channels)

    elif cnn_name == 'ShuffleNet':
        """
        ShuffleNet
        """
        model = ShuffleNet.get_model(
            groups=3,
            num_classes=n_classes,
            in_channels=in_channels)

    elif cnn_name == 'MobileNet':
        """
        MobileNet
        "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" 
        """
        model = MobileNet.get_model(
            sample_size=sample_size,
            num_classes=n_classes,
            in_channels=in_channels)

    elif cnn_name == 'MobileNetV2':
        """
        MobileNet
        "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
        """
        model = MobileNetV2.get_model(
            sample_size=sample_size,
            num_classes=n_classes,
            in_channels=in_channels)
    
    elif cnn_name == 'EfficientNet':
        """
        EfficientNet
        efficientnetb0, b1, ..., b9
        """
        model = EfficientNet.EfficientNet.from_name(
            'efficientnet-b4', 
            override_params={'num_classes': n_classes}, 
            in_channels=in_channels)
    
    elif cnn_name == 'SwinTransformerV2':
        """
        SwinTransformerV2
        """
        model = SwinTransformerV2(
            img_size=160, patch_size=4, in_chans=1, num_classes=10,
            embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
            window_size=7, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0])

    if torch.cuda.is_available():
        model.cuda()

    return model






