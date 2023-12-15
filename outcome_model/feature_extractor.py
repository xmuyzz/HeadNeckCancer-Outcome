import torch
import torchvision.models as models
import os
import pandas as pd
import numpy as np
import torch
import random
from get_cnn_model import get_cnn_model
from get_cox_model import get_cox_model
from opts import parse_opts
import torch.cuda



opt = parse_opts()

task_dir = opt.proj_dir + '/task/' + opt.task + '_' + opt.surv_type + '_' + opt.img_size + '_' + \
           opt.img_type + '_' + opt.tumor_type + '_' + opt.cox + '_' + opt.cnn_name + str(opt.model_depth) 

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

cox_model = get_cox_model(task_dir=task_dir, 
                            cnn_model=cnn_model, 
                            cox=opt.cox, 
                            lr=opt.lr)


cox_model.load_model_weights(task_dir + '/models/weights_best_cindex.pt')


# Load a pre-trained CNN model (e.g., ResNet50)
#model = models.resnet50(pretrained=True)

# Print the model architecture to identify the layer you want to extract
print(cox_model)
print(dir(cox_model))
print(cox_model.state_dict().keys())

# Extract the feature extractor (excluding the last fully connected layer)
last_fc_layer = cox_model.fc
feature_extractor = torch.nn.Sequential(*list(cox_model.children())[:-1])

# Set the model to evaluation mode
feature_extractor.eval()

# Example: Forward pass to extract features from an input image
input_image = torch.randn(1, 3, 224, 224)  # Adjust dimensions based on your model's input size
with torch.no_grad():
    features = feature_extractor(input_image)

# The 'features' variable now contains the output of the last convolutional layer
print(features)