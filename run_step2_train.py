import numpy as np
import torch
#from get_data.data_loader2 import data_loader2
from get_data.data_loader_transform import data_loader_transform
#from get_data.DataLoader_Cox import DataLoader_Cox
from go_models.train import train
from go_models.evaluate import evaluate
from go_models.get_cnn_model import get_cnn_model
from go_models.get_cox_model import get_cox_model


if __name__ == '__main__':

    data_dir = '/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated'
    proj_dir = '/mnt/HDD_6TB/HN_Outcome'
    out_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    cnn_name = 'resnet'
    model_depth = 152  # [10, 18, 34, 50, 101, 152, 200]
    n_classes = 20
    in_channels = 1
    batch_size = 8
    epochs = 50
    lr = 0.00001
    num_durations = 20
    verbose = True
    _cox_model = 'LogisticHazard'
    cox_model = 'LogisticHazard'
    load_model = 'model'
    score_type = '3year_survival'   #'median'
    evaluate_only = True
    augmentation = True
    
    np.random.seed(1234)
    _ = torch.manual_seed(1234)

    if not augmentation:
        dl_train, dl_tune, dl_val = data_loader2(
            proj_dir=proj_dir,
            batch_size=batch_size,
            _cox_model=_cox_model,
            num_durations=num_durations
            )
    else:
        dl_train, dl_tune, dl_val = data_loader_transform(
            proj_dir, 
            batch_size=batch_size, 
            _cox_model=_cox_model, 
            num_durations=num_durations
            )
   
    cnn_model = get_cnn_model(
        cnn_name=cnn_name, 
        model_depth=model_depth, 
        n_classes=n_classes, 
        in_channels=in_channels
        )

    cox_model = get_cox_model(
        proj_dir=proj_dir,
        cnn_model=cnn_model,
        _cox_model=_cox_model,
        lr=lr
        )

    if not evaluate_only:
        train(
            proj_dir=proj_dir,
            out_dir=out_dir,
            cox_model=cox_model,
            epochs=epochs,
            verbose=verbose,
            dl_train=dl_train,
            dl_tune=dl_tune,
            dl_val=dl_val
            )
    
    evaluate(
        proj_dir=proj_dir,
        out_dir=out_dir,
        cox_model=cox_model,
        load_model=load_model,
        dl_val=dl_val,
        score_type=score_type,
        cnn_name=cnn_name,
        epochs=epochs
        )

