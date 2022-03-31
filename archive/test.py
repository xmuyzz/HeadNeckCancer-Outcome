import numpy as np
import torch
#from get_data.data_loader2 import data_loader2
from get_data.data_loader_transform import data_loader_transform
#from get_data.DataLoader_Cox import DataLoader_Cox
from go_models.train import train
from go_models.evaluate2 import evaluate2
from go_models.get_cnn_model import get_cnn_model
from go_models.get_cox_model import get_cox_model


if __name__ == '__main__':

    data_dir = '/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated'
    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    cnn_name = 'resnet101'
    model_depth = 101  # [10, 18, 34, 50, 101, 152, 200]
    n_classes = 20
    in_channels = 1
    batch_size = 8
    epochs = 1
    lr = 0.001
    num_durations = 20
    _cox_model = 'LogisticHazard'
    load_model = 'model'
    score_type = '3year_survival'   #'median'
    model_fn = 'resnet101_20_0.0001_model.pt' 


    np.random.seed(1234)
    _ = torch.manual_seed(1234)

    dl_train, dl_tune, dl_val, dl_test = data_loader_transform(
        proj_dir, 
        batch_size=batch_size, 
        _cox_model=_cox_model, 
        num_durations=num_durations)
    
    cnn_model = get_cnn_model(
        cnn_name=cnn_name, 
        n_classes=n_classes, 
        in_channels=in_channels)
    
    cox_model = get_cox_model(
        proj_dir=proj_dir,
        cnn_model=cnn_model,
        _cox_model=_cox_model,
        lr=lr)

    evaluate2(
        proj_dir=proj_dir,
        cox_model=cox_model,
        load_model=load_model,
        model_fn=model_fn,
        data_loader=dl_val,)

